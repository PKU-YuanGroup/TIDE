"""Shared utilities for BD3LM distillation training."""

from typing import Dict, List, Optional, Tuple


def align_chat_tokens(
    messages: List[Dict[str, str]],
    tok_a,
    tok_b,
    roles: Optional[List[str]] = None,
    template_kwargs_a: Optional[dict] = None,
    template_kwargs_b: Optional[dict] = None,
) -> Tuple[List[int], List[int], List[int], List[int]]:
    """
    Align content tokens between two different chat templates.

    Both tokenizers must share the same base vocabulary (e.g., Qwen2/Qwen3 family).
    The function finds where each message's content text appears in each template's
    output, then maps character spans to token positions via offset_mapping.

    Args:
        messages: Chat messages [{"role": ..., "content": ...}, ...]
        tok_a: Tokenizer A (e.g., WeDLM / Qwen2)
        tok_b: Tokenizer B (e.g., Qwen3)
        roles: Only align these roles (e.g., ["assistant"]). None = all roles.
        template_kwargs_a: Extra kwargs for tok_a.apply_chat_template
        template_kwargs_b: Extra kwargs for tok_b.apply_chat_template

    Returns:
        ids_a: Full token sequence from tok_a
        ids_b: Full token sequence from tok_b
        align_a: Aligned positions in ids_a
        align_b: Aligned positions in ids_b

    Guarantee: ids_a[align_a[i]] == ids_b[align_b[i]] for all i
    """
    kw_a = {"tokenize": False, "add_generation_prompt": False}
    kw_b = {"tokenize": False, "add_generation_prompt": False}
    if template_kwargs_a:
        kw_a.update(template_kwargs_a)
    if template_kwargs_b:
        kw_b.update(template_kwargs_b)

    text_a = tok_a.apply_chat_template(messages, **kw_a)
    text_b = tok_b.apply_chat_template(messages, **kw_b)

    enc_a = tok_a(text_a, add_special_tokens=False, return_offsets_mapping=True)
    enc_b = tok_b(text_b, add_special_tokens=False, return_offsets_mapping=True)

    ids_a, offsets_a = enc_a["input_ids"], enc_a["offset_mapping"]
    ids_b, offsets_b = enc_b["input_ids"], enc_b["offset_mapping"]

    align_a, align_b = [], []
    search_a, search_b = 0, 0

    for msg in messages:
        if roles and msg["role"] not in roles:
            continue
        content = msg["content"]
        if not content:
            continue

        # Find content string in both formatted texts (sequential search)
        ca = text_a.find(content, search_a)
        cb = text_b.find(content, search_b)
        if ca == -1 or cb == -1:
            continue

        ca_end = ca + len(content)
        cb_end = cb + len(content)
        search_a = ca_end
        search_b = cb_end

        # Map char spans -> token indices via offset_mapping
        toks_a = [
            i for i, (s, e) in enumerate(offsets_a) if s >= ca and e <= ca_end and s < e
        ]
        toks_b = [
            i for i, (s, e) in enumerate(offsets_b) if s >= cb and e <= cb_end and s < e
        ]

        n = min(len(toks_a), len(toks_b))
        align_a.extend(toks_a[:n])
        align_b.extend(toks_b[:n])

    return ids_a, ids_b, align_a, align_b


def aligned_kl_sft_map_fn(
    row,
    *,
    student_tokenizer,
    teacher_tokenizer,
    max_length=None,
    mask_prompt_loss: bool = True,
    align_roles: Optional[List[str]] = None,
) -> dict:
    """
    Tokenize a chat row with both student and teacher tokenizers (different templates),
    then compute per-token alignment using character-span matching.

    Args:
        row: Dataset row with "messages" key.
        student_tokenizer: Student tokenizer (e.g., dllm's Qwen3 template).
        teacher_tokenizer: Teacher tokenizer (e.g., WeDLM's native template).
        max_length: Maximum student sequence length (truncates student, filters alignment).
        mask_prompt_loss: If True, mask prompt tokens in student labels with -100.
        align_roles: Roles to align (e.g., ["assistant"]). None = all roles.

    Returns:
        dict with input_ids, labels, teacher_input_ids, align_student, align_teacher,
        and optionally prompt_len.
    """
    teacher_ids, student_ids, align_teacher, align_student = align_chat_tokens(
        row["messages"],
        teacher_tokenizer,
        student_tokenizer,
        roles=align_roles,
    )

    labels = student_ids.copy()
    prompt_len = 0

    if mask_prompt_loss:
        prompt_tokens = student_tokenizer.apply_chat_template(
            row["messages"][:-1], tokenize=True, add_generation_prompt=True
        )
        prompt_len = min(len(prompt_tokens), len(labels))
        # Discard if prompt alone fills max_length (no room for response)
        if max_length is not None and prompt_len >= max_length:
            return {
                "input_ids": [],
                "labels": [],
                "teacher_input_ids": [],
                "align_student": [],
                "align_teacher": [],
                "prompt_len": prompt_len,
                "_discard": True,
            }
        labels[:prompt_len] = [-100] * prompt_len

    # Truncate student to max_length, filter alignment pairs beyond range
    if max_length is not None and len(student_ids) > max_length:
        student_ids = student_ids[:max_length]
        labels = labels[:max_length]
        # Filter alignment pairs where student position is beyond truncation
        filtered_align_s, filtered_align_t = [], []
        for s, t in zip(align_student, align_teacher):
            if s < max_length:
                filtered_align_s.append(s)
                filtered_align_t.append(t)
        align_student = filtered_align_s
        align_teacher = filtered_align_t

    # Always truncate teacher to the last aligned position — teacher tokens
    # beyond the last alignment point are never used for distillation loss
    if align_teacher:
        teacher_ids = teacher_ids[: max(align_teacher) + 1]

    result = {
        "input_ids": student_ids,
        "labels": labels,
        "teacher_input_ids": teacher_ids,
        "align_student": align_student,
        "align_teacher": align_teacher,
        "_discard": False,
    }
    if mask_prompt_loss:
        result["prompt_len"] = prompt_len

    return result


def _find_text_in_tokens_slow(
    text: str,
    tokens: list[int],
    tokenizer,
) -> tuple[int, int]:
    """
    Fallback O(n²) implementation using incremental decoding.

    Used when offset_mapping is unavailable or re-tokenization doesn't match.
    """
    full_decoded = tokenizer.decode(tokens, skip_special_tokens=False)

    char_start = full_decoded.find(text)
    if char_start < 0:
        return -1, -1
    char_end = char_start + len(text)

    start_idx = -1
    end_idx = -1

    for i in range(len(tokens)):
        cumulative = tokenizer.decode(tokens[: i + 1], skip_special_tokens=False)
        curr_len = len(cumulative)

        if start_idx < 0 and curr_len > char_start:
            start_idx = i

        if curr_len >= char_end:
            end_idx = i + 1
            break

    return start_idx, end_idx


def find_text_in_tokens(
    text: str,
    tokens: list[int],
    tokenizer,
) -> tuple[int, int]:
    """
    Find the token range in a token sequence that corresponds to a given text.

    Uses offset_mapping for O(n) complexity when available. Falls back to
    incremental decoding O(n²) if re-tokenization doesn't match original tokens.

    Args:
        text: The text to find in the decoded token sequence
        tokens: List of token IDs
        tokenizer: Tokenizer to use for decoding

    Returns:
        (start_idx, end_idx) as half-open range, or (-1, -1) if not found
    """
    # Step 1: Decode full sequence once
    full_decoded = tokenizer.decode(tokens, skip_special_tokens=False)

    # Step 2: Find text position in decoded string
    char_start = full_decoded.find(text)
    if char_start < 0:
        return -1, -1
    char_end = char_start + len(text)

    # Step 3: Try to get offset_mapping in one call (O(n))
    try:
        encoding = tokenizer(
            full_decoded,
            return_offsets_mapping=True,
            add_special_tokens=False,
        )
    except Exception:
        # Tokenizer doesn't support offset_mapping
        return _find_text_in_tokens_slow(text, tokens, tokenizer)

    # Check if re-tokenization matches original tokens
    if encoding["input_ids"] != tokens:
        # Re-tokenization differs, fall back to slow method
        return _find_text_in_tokens_slow(text, tokens, tokenizer)

    offset_mapping = encoding["offset_mapping"]

    # Step 4: Linear scan to find token boundaries (O(n))
    start_idx = -1
    end_idx = -1

    for i, (token_char_start, token_char_end) in enumerate(offset_mapping):
        # Skip tokens with no character span (e.g., some special tokens)
        if token_char_start == token_char_end:
            continue

        # Find first token that covers char_start
        if start_idx < 0 and token_char_start <= char_start < token_char_end:
            start_idx = i

        # Find last token that covers char_end
        if token_char_start < char_end <= token_char_end:
            end_idx = i + 1
            break

        # Handle case where char_end falls between tokens
        if start_idx >= 0 and token_char_start >= char_end:
            end_idx = i
            break

    return start_idx, end_idx


def find_content_ranges_by_text_match(
    messages: list[dict],
    teacher_tokens: list[int],
    student_tokens: list[int],
    teacher_tokenizer,
    student_tokenizer,
) -> list[tuple[int, int, int, int]]:
    """
    Find content ranges by matching original message text in both token sequences.

    This ensures teacher and student ranges correspond to exactly the same text,
    avoiding alignment mismatches caused by different chat templates.

    Args:
        messages: List of message dicts with "role" and "content" keys
        teacher_tokens: Token IDs from teacher tokenizer
        student_tokens: Token IDs from student tokenizer
        teacher_tokenizer: Teacher model's tokenizer
        student_tokenizer: Student model's tokenizer

    Returns:
        List of (t_start, t_end, s_start, s_end) tuples, one per message content.
        Each tuple guarantees that teacher_tokens[t_start:t_end] and
        student_tokens[s_start:s_end] decode to the same text.
    """
    ranges = []

    for msg in messages:
        content = msg.get("content", "")
        if not content or not content.strip():
            continue

        # Find content in teacher tokens
        t_start, t_end = find_text_in_tokens(content, teacher_tokens, teacher_tokenizer)

        # Find content in student tokens
        s_start, s_end = find_text_in_tokens(content, student_tokens, student_tokenizer)

        if t_start >= 0 and s_start >= 0:
            # Verify: decoded text should match
            t_text = teacher_tokenizer.decode(
                teacher_tokens[t_start:t_end], skip_special_tokens=False
            )
            s_text = student_tokenizer.decode(
                student_tokens[s_start:s_end], skip_special_tokens=False
            )
            if t_text.strip() == s_text.strip():
                ranges.append((t_start, t_end, s_start, s_end))

    return ranges


def distill_sft_map_fn(
    row,
    *,
    student_tokenizer,
    teacher_tokenizer,
    max_length=None,
    mask_prompt_loss: bool = True,
) -> dict:
    """
    Tokenize a chat row with both student and teacher tokenizers.

    Produces input_ids/labels for student and teacher_input_ids/teacher_labels
    for teacher. Pre-computes content ranges using text matching to ensure
    teacher and student ranges correspond to exactly the same text content.
    Actual alignment is deferred to the collator.
    """
    # Student tokenization (same as SFT pipeline)
    prompt_response_tokens = student_tokenizer.apply_chat_template(
        row["messages"], tokenize=True, add_generation_prompt=False
    )
    labels = prompt_response_tokens.copy()

    # Teacher tokenization (same as original LLaDA2 inference)
    teacher_tokens = teacher_tokenizer.apply_chat_template(
        row["messages"], tokenize=True, add_generation_prompt=False
    )

    teacher_labels = teacher_tokens.copy()
    if mask_prompt_loss:
        prompt_tokens = student_tokenizer.apply_chat_template(
            row["messages"][:-1], tokenize=True, add_generation_prompt=True
        )
        student_prompt_len = min(len(prompt_tokens), len(labels))
        # Discard if prompt alone fills max_length (no room for response)
        if max_length is not None and student_prompt_len >= max_length:
            return {
                "input_ids": [],
                "labels": [],
                "teacher_input_ids": [],
                "teacher_labels": [],
                "ranges": [],
                "prompt_len": student_prompt_len,
                "_discard": True,
            }
        labels[:student_prompt_len] = [-100] * student_prompt_len
        teacher_prompt_tokens = teacher_tokenizer.apply_chat_template(
            row["messages"][:-1], tokenize=True, add_generation_prompt=True
        )
        teacher_prompt_len = min(len(teacher_prompt_tokens), len(teacher_labels))
        teacher_labels[:teacher_prompt_len] = [-100] * teacher_prompt_len

    # Compute content ranges using text matching to ensure teacher and student
    # ranges correspond to exactly the same text content
    ranges = find_content_ranges_by_text_match(
        row["messages"],
        teacher_tokens,
        prompt_response_tokens,
        teacher_tokenizer,
        student_tokenizer,
    )

    # Simple student-only truncation (teacher cutoff is handled in collator)
    if max_length is not None and len(prompt_response_tokens) > max_length:
        prompt_response_tokens = prompt_response_tokens[:max_length]
        labels = labels[:max_length]

    result = {
        "input_ids": prompt_response_tokens,
        "labels": labels,
        "teacher_input_ids": teacher_tokens,
        "teacher_labels": teacher_labels,
        "ranges": ranges,  # New format: [(t_s, t_e, s_s, s_e), ...]
        "_discard": False,
    }
    if mask_prompt_loss:
        result["prompt_len"] = student_prompt_len

    return result
