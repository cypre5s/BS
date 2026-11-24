#!/usr/bin/env python3
"""Exercise the installed official processor on CPU; never load model weights."""

import argparse
import inspect
import json
import sys
from dataclasses import asdict
from importlib.metadata import version
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from orchestration.storage import atomic_json, file_hash
from samr.config import IDENTITY
from samr.scales import ScaleLadder
from xiaomi_runtime.adapter import MODEL_FILES_SHA256
from xiaomi_runtime.inference import GENERATION_CONFIG
from xiaomi_runtime.processor import ProcessorLimits, encode_prompt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    import torch
    from PIL import Image
    from transformers import AutoProcessor
    from transformers.models.qwen2_vl import image_processing_qwen2_vl
    from transformers.models.qwen3_5 import modeling_qwen3_5
    from transformers.models.qwen3_vl import processing_qwen3_vl

    torch.set_num_threads(2)
    hashes = {name: file_hash(args.model / name) for name in MODEL_FILES_SHA256}
    assert hashes == MODEL_FILES_SHA256
    processor = AutoProcessor.from_pretrained(args.model, local_files_only=True)
    limits = ProcessorLimits.from_processor(processor)
    assert limits == ProcessorLimits()
    ladder = ScaleLadder(limits)
    image_token = processor.tokenizer.convert_tokens_to_ids(processor.image_token)
    rows = []
    nonimage_ids = None
    for size in ((245, 98), (777, 399), (256, 256), (80, 800), (321, 123), (512, 1024)):
        image = Image.new("RGB", size, "white")
        for scale in (1.0, 1.25, 1.5, 1.75, 2.0):
            plan = ladder.plan(size, scale)
            inputs = encode_prompt(processor, plan.prepare(image))
            grid = inputs["image_grid_thw"].tolist()
            ids = inputs["input_ids"][0].tolist()
            tokens = ids.count(image_token)
            width, height = plan.processed_size
            assert grid == [[1, height // 16, width // 16]]
            assert tokens == plan.estimated_visual_tokens == grid[0][1] * grid[0][2] // 4
            assert inputs["pixel_values"].shape[0] == tokens * 4
            assert "mm_token_type_ids" in inputs
            text_ids = [token for token in ids if token != image_token]
            if nonimage_ids is None:
                nonimage_ids = text_ids
            assert text_ids == nonimage_ids
            rows.append({**asdict(plan), "image_grid_thw": grid, "actual_visual_tokens": tokens,
                         "pixel_values_shape": list(inputs["pixel_values"].shape),
                         "multimodal_token_types_present": True})
    # Audit the native patchifier's grouping on a numbered synthetic image only.
    numbered = torch.arange(16).reshape(4, 4).repeat_interleave(16, 0).repeat_interleave(16, 1)
    numbered = numbered[None, None].expand(1, 3, -1, -1).float()
    patches, gh, gw = processor.image_processor.patchify(numbered, 16, 2, 2)
    patch_order = patches[0, :, 0].tolist()
    assert (gh, gw) == (4, 4)
    assert patch_order == [0, 1, 4, 5, 2, 3, 6, 7, 8, 9, 12, 13, 10, 11, 14, 15]
    after = {name: file_hash(args.model / name) for name in MODEL_FILES_SHA256}
    assert after == hashes
    sources = {}
    for module in (image_processing_qwen2_vl, processing_qwen3_vl, modeling_qwen3_5):
        path = Path(inspect.getfile(module))
        sources[module.__name__] = {"filename": path.name, "sha256": file_hash(path)}
    receipt = {"passed": True, "kind": "real_native_processor_CPU_not_model_inference",
               "transformers_version": version("transformers"), "torch_version": version("torch"),
               "processor_class": type(processor).__name__, "image_processor_class": type(processor.image_processor).__name__,
               "limits": asdict(limits), "sources": sources, "samples": rows, "sample_count": len(rows),
               "patch_order": patch_order, "prompt_tokens_equal_excluding_image_placeholders": True,
               "generation_config": GENERATION_CONFIG, "identity": IDENTITY,
               "hashes_before": hashes, "hashes_after": after, "cuda_initialized": torch.cuda.is_initialized(),
               "model_load_count": 0}
    assert not receipt["cuda_initialized"]
    atomic_json(args.output, receipt, immutable=True)
    print(json.dumps({key: receipt[key] for key in ("passed", "sample_count", "cuda_initialized", "model_load_count")}))


if __name__ == "__main__":
    main()
