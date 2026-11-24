"""Generation settings exactly matching the frozen B0 protocol."""

from typing import Any

from .budget import DeadlineStop, check_deadline

MAX_NEW_TOKENS = 16384

# Native token IDs are pinned by config.json; this is the effective decode protocol.
GENERATION_CONFIG = {
    "do_sample": False, "repetition_penalty": 1.0, "max_new_tokens": MAX_NEW_TOKENS,
    "use_cache": True, "eos_token_id": "model.config.eos_token_id",
    "pad_token_id": "model.config.pad_token_id", "dtype": "bfloat16",
    "attn_implementation": "sdpa", "skip_special_tokens": True,
    "clean_up_tokenization_spaces": False,
}


def generate(model: Any, inputs: Any, *, deadline: float | None = None) -> tuple[Any, bool]:
    import torch

    check_deadline(deadline)
    safety: dict[str, Any] = {}
    if deadline is not None:
        from transformers import StoppingCriteriaList

        safety["stopping_criteria"] = StoppingCriteriaList([DeadlineStop(deadline)])
    with torch.inference_mode(), torch.autocast(model.device.type, dtype=torch.bfloat16):
        output = model.generate(
            **inputs, do_sample=False, repetition_penalty=1.0,
            max_new_tokens=MAX_NEW_TOKENS, use_cache=True,
            eos_token_id=model.config.eos_token_id, pad_token_id=model.config.pad_token_id,
            **safety,
        )[0, inputs["input_ids"].size(1):]
    eos = bool(output.numel() and output[-1].item() == model.config.eos_token_id)
    return output, eos
