import torch


def sample_trim(tokenizer, seq_ids_list, input_ids_list) -> list[str]:
    """
    Return only the generated text, truncated at the first EOS **after** the prompt.

    Args:
        tokenizer: HF tokenizer with eos_token_id / pad_token_id.
        seq_ids: Full sequence token ids from the model (prompt + generation).
        input_ids: The prompt token ids that were fed into the model.

    Behavior:
        - Finds the first eos_token_id that occurs at or after len(input_ids).
        - Slices generation up to (but not including) that EOS.
        - Decodes only the generation span, skipping special/pad tokens.
    """
    # Make sure we can index these
    sequences = []
    for seq_ids, input_ids in zip(seq_ids_list, input_ids_list):
        full = list(seq_ids)
        prompt = list(input_ids)

        # Skip left padding tokens (necessary for dream)
        pad_id = getattr(tokenizer, "pad_token_id", None)
        if pad_id is not None:
            while full and full[0] == pad_id:
                full.pop(0)

        start = len(prompt)
        end = len(full)

        eos_id = getattr(tokenizer, "eos_token_id", None)
        eot_id = getattr(tokenizer, "eot_token_id", None)
        if eos_id is not None:
            for i in range(start, len(full)):
                if full[i] in (eos_id, eot_id):
                    end = i
                    break

        gen_ids = full[start:end]
        text = tokenizer.decode(gen_ids, skip_special_tokens=True)
        # in case there is no eos_id or eot_id, just strings
        eos = getattr(tokenizer, "eos_token", None)
        eot = getattr(tokenizer, "eot_token", None)
        if eos:
            text = text.split(eos)[0]
        if eot:
            text = text.split(eot)[0]
        # return text.strip()
        sequences.append(text)
    return sequences


def infill_trim(tokenizer, seq_ids_list, input_ids_list) -> list[str]:
    """
    Return only the generated text, truncated at the first EOS **after** the prompt.

    Args:
        tokenizer: HF tokenizer with eos_token_id / pad_token_id.
        seq_ids: Full sequence token ids from the model (prompt + generation).
        input_ids: The prompt token ids that were fed into the model.

    Behavior:
        - Finds the first eos_token_id that occurs at or after len(input_ids).
        - Slices generation up to (but not including) that EOS.
        - Decodes only the generation span, skipping special/pad tokens.
    """
    # Make sure we can index these
    sequences = []
    for seq_ids, input_ids in zip(seq_ids_list, input_ids_list):
        full = torch.tensor(seq_ids)
        prompt = torch.tensor(input_ids)

        # Skip left padding tokens (necessary for dream)
        pad_id = getattr(tokenizer, "pad_token_id", None)
        if pad_id is not None:
            while full.numel() and full[0].item() == pad_id:
                full = full[1:]

        masked_index = prompt == tokenizer.mask_token_id
        infill = full[masked_index]

        end = len(infill)
        eos_id = getattr(tokenizer, "eos_token_id", None)
        eot_id = getattr(tokenizer, "eot_token_id", None)

        if eos_id is not None:
            for i in range(len(infill)):
                if infill[i] in (eos_id, eot_id):
                    end = i
                    break

        gen_ids = infill[:end]
        text = tokenizer.decode(gen_ids, skip_special_tokens=True)
        # in case there is no eos_id or eot_id, just strings
        eos = getattr(tokenizer, "eos_token", None)
        eot = getattr(tokenizer, "eot_token", None)
        if eos:
            text = text.split(eos)[0]
        if eot:
            text = text.split(eot)[0]
        # return text.strip()
        sequences.append(text)
    return sequences
