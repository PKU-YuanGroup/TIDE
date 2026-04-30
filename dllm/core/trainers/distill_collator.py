"""
Distillation collator that wraps a base BD3LM collator to add:
- Padded teacher_input_ids, teacher_labels, and teacher_attention_mask
- Cross-tokenizer alignment matrices (student <-> teacher) via tokenkit

Uses text-matched content alignment: aligns message content ranges that are
guaranteed to correspond to the same text in both teacher and student
sequences, avoiding mismatches from different chat templates.
"""

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from dllm.utils.collators import CollatorWrapper


@dataclass
class DistillCollator(CollatorWrapper):
    """
    Collator wrapper for cross-tokenizer distillation.

    Extracts teacher_input_ids and teacher_labels from features before the
    base collator runs, then pads them and computes alignment matrices using
    pre-computed text-matched ranges in the after hook.
    """

    distill_mode: str = "alm"  # "alm" (cross-tokenizer) or "kl" (same-tokenizer)
    teacher_tokenizer: Any = None
    teacher_pad_token_id: int = 0
    byteify_teacher: Any = None
    byteify_student: Any = None
    max_teacher_length: int = 1024

    def __post_init__(self):
        self._teacher_input_ids_list: list = []
        self._teacher_labels_list: list = []
        self._ranges_list: list = []

    def before(self, features):
        """Extract teacher fields and pre-computed ranges before the base collator runs."""
        self._teacher_input_ids_list = []
        self._teacher_labels_list = []
        self._ranges_list = []
        if self.distill_mode not in ("alm", "alm_taid"):
            return features
        for ex in features:
            self._teacher_input_ids_list.append(ex.pop("teacher_input_ids"))
            self._teacher_labels_list.append(ex.pop("teacher_labels"))
            self._ranges_list.append(ex.pop("ranges", []))
        return features

    def after(self, outputs):
        """Pad teacher sequences and compute truncation-aware alignment matrices.

        Uses pre-computed text-matched ranges from the map function, and the
        actual student length (from attention_mask) to handle truncation.
        Each range tuple (t_start, t_end, s_start, s_end) is guaranteed to
        correspond to the same text content in both sequences.

        In KL mode (same-tokenizer), acts as a pass-through since no
        cross-tokenizer alignment is needed.
        """
        if self.distill_mode not in ("alm", "alm_taid"):
            return outputs

        from tokenkit.align import get_alignment_indices

        teacher_ids_list = self._teacher_input_ids_list
        teacher_labels_list = self._teacher_labels_list
        student_ids = outputs["input_ids"]  # [b, l_s]
        b, l_s = student_ids.shape
        device = student_ids.device

        # Actual student lengths (excluding padding)
        attention_mask = outputs.get("attention_mask")
        student_lengths = (
            attention_mask.sum(dim=1).tolist()
            if attention_mask is not None
            else [l_s] * b
        )

        student_ids_np = student_ids.cpu().numpy()

        all_sample_chunks = []
        teacher_cutoffs = []

        for i in range(b):
            t_ids = teacher_ids_list[i]
            actual_s_len = int(student_lengths[i])
            s_ids = student_ids_np[i]

            ranges = self._ranges_list[i]  # [(t_s, t_e, s_s, s_e), ...]

            all_chunks = []

            if ranges:
                t_tokens_full = self.byteify_teacher.convert_ids_to_tokens(t_ids)
                s_tokens_full = self.byteify_student.convert_ids_to_tokens(
                    s_ids[:actual_s_len].tolist()
                )

                # Process each text-matched range
                for t_start, t_end, s_start, s_end in ranges:
                    # Skip segments that are completely truncated
                    if s_start >= actual_s_len:
                        continue

                    effective_s_end = min(s_end, actual_s_len)
                    if effective_s_end <= s_start:
                        continue

                    t_segment_tokens = t_tokens_full[t_start:t_end]
                    s_segment_tokens = s_tokens_full[s_start:effective_s_end]

                    segment_align, _, _, _, _ = get_alignment_indices(
                        t_segment_tokens,
                        s_segment_tokens,
                        self.byteify_teacher,
                        self.byteify_student,
                        np.ones(len(t_segment_tokens), dtype=bool),
                        np.ones(len(s_segment_tokens), dtype=bool),
                    )

                    for si, ei, sj, ej in segment_align:
                        all_chunks.append(
                            (si + t_start, ei + t_start, sj + s_start, ej + s_start)
                        )

            # Teacher cutoff from alignment chunks
            teacher_cutoff = len(t_ids)
            for start_i, end_i, _, end_j in all_chunks:
                if end_j > actual_s_len:
                    teacher_cutoff = start_i
                    break
                teacher_cutoff = end_i

            teacher_cutoff = min(teacher_cutoff, self.max_teacher_length)
            all_sample_chunks.append(all_chunks)
            teacher_cutoffs.append(teacher_cutoff)

        for i in range(b):
            teacher_ids_list[i] = teacher_ids_list[i][: teacher_cutoffs[i]]
            teacher_labels_list[i] = teacher_labels_list[i][: teacher_cutoffs[i]]

        # --- Pad teacher_input_ids and teacher_labels ---
        max_l_t = max(len(ids) for ids in teacher_ids_list)
        padded_teacher_ids = []
        padded_teacher_labels = []
        teacher_attention_masks = []
        for idx, ids in enumerate(teacher_ids_list):
            pad_len = max_l_t - len(ids)
            padded_teacher_ids.append(ids + [self.teacher_pad_token_id] * pad_len)
            padded_teacher_labels.append(teacher_labels_list[idx] + [-100] * pad_len)
            teacher_attention_masks.append([1] * len(ids) + [0] * pad_len)

        teacher_input_ids = torch.tensor(
            padded_teacher_ids, dtype=torch.long, device=device
        )
        teacher_labels = torch.tensor(
            padded_teacher_labels, dtype=torch.long, device=device
        )
        teacher_attention_mask = torch.tensor(
            teacher_attention_masks, dtype=torch.long, device=device
        )

        # --- Build alignment matrices from pre-computed chunks ---
        max_n_chunks = max((len(chunks) for chunks in all_sample_chunks), default=0)

        all_align_student = []
        all_align_teacher = []

        for i in range(b):
            chunks = all_sample_chunks[i]
            n_chunks = len(chunks)

            align_s = np.zeros((l_s, n_chunks), dtype=bool)
            align_t = np.zeros((max_l_t, n_chunks), dtype=bool)
            for chunk_idx, (start_i, end_i, start_j, end_j) in enumerate(chunks):
                align_t[start_i:end_i, chunk_idx] = True
                align_s[start_j:end_j, chunk_idx] = True

            all_align_student.append(align_s)
            all_align_teacher.append(align_t)

        # Stack with padding on the chunks dimension
        padded_align_student = np.zeros((b, l_s, max_n_chunks), dtype=bool)
        padded_align_teacher = np.zeros((b, max_l_t, max_n_chunks), dtype=bool)
        for i in range(b):
            n_c = all_align_student[i].shape[1]
            padded_align_student[i, :, :n_c] = all_align_student[i]
            padded_align_teacher[i, :, :n_c] = all_align_teacher[i]

        outputs["teacher_input_ids"] = teacher_input_ids
        outputs["teacher_labels"] = teacher_labels
        outputs["teacher_attention_mask"] = teacher_attention_mask
        outputs["alignment_matrix_student"] = torch.tensor(
            padded_align_student, dtype=torch.bool, device=device
        )
        outputs["alignment_matrix_teacher"] = torch.tensor(
            padded_align_teacher, dtype=torch.bool, device=device
        )

        return outputs


# plt.imsave("teacher_tokens_to_chunks.png", padded_align_teacher[0])
# plt.imsave("student_tokens_to_chunks.png", padded_align_student[0])
# plt.imsave("teacher_tokens_to_chunks1.png", padded_align_teacher[1])
# plt.imsave("student_tokens_to_chunks1.png", padded_align_student[1])
