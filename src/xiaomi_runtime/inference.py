"""Generation settings exactly matching the frozen B0 protocol."""

from typing import Any

MAX_NEW_TOKENS = 16384


def generate(model: Any, inputs: Any) -> tuple[Any, bool]:
    import torch

    with torch.inference_mode(), torch.autocast(model.device.type, dtype=torch.bfloat16):
        output = model.generate(
            **inputs, do_sample=False, repetition_penalty=1.0,
            max_new_tokens=MAX_NEW_TOKENS, use_cache=True,
            eos_token_id=model.config.eos_token_id, pad_token_id=model.config.pad_token_id,
        )[0, inputs["input_ids"].size(1):]
    eos = bool(output.numel() and output[-1].item() == model.config.eos_token_id)
    return output, eos
