"""One lazy, read-only Xiaomi model shared by all inference methods."""

import hashlib
import json
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from time import perf_counter
from typing import Any, Protocol

from PIL import Image

from .budget import BudgetExceeded, check_deadline
from .inference import generate
from .otsl import to_html
from .processor import TABLE_PROMPT, ProcessorLimits, encode_prompt, view_image

MODEL_SHA256 = "d4551e9f6127ac9e35cff7b4f7a72a90410e03e6c8676a36f0e2b7eaeb5d5348"
MODEL_REVISION = "bbd8cf8d21f256b3644b44cb860f3a7cecb93c30"
MODEL_FILES_SHA256 = {
    "model.safetensors": MODEL_SHA256,
    "config.json": "2eb95982069a24336fe66b3a0869ea5a711aba7566787d4a599654907ecc070c",
    "processor_config.json": "d89ef49ce9cd37fbf510158e13c1ef063d9286411c1ec9049932dbe0487143b1",
    "tokenizer_config.json": "792fa3f0cb88b111e54ef3134c873531008c4df471d108da17903426e308aa7b",
    "tokenizer.json": "06b9509352d2af50381ab2247e083b80d32d5c0aba91c272ca9ff729b6a0e523",
    "chat_template.jinja": "273d8e0e683b885071fb17e08d71e5f2a5ddfb5309756181681de4f5a1822d80",
}


def text_hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class Prediction:
    raw: str
    html: str
    table_id: str
    method: str
    view_type: str
    input_size: tuple[int, int] = (0, 0)
    processed_size: tuple[int, int] = (0, 0)
    image_grid_thw: list[list[int]] = field(default_factory=list)
    visual_tokens: int = 0
    output_tokens: int = 0
    latency: float = 0.0
    eos: bool = True
    truncated: bool = False
    prompt_hash: str = field(default_factory=lambda: text_hash(TABLE_PROMPT))
    model_hash: str = MODEL_SHA256
    output_hash: str = ""
    error: str | None = None


@dataclass(frozen=True)
class MethodResult:
    prediction: Prediction
    calls: list[Prediction]
    changed: bool
    reason: str
    diagnostics: dict[str, Any] = field(default_factory=dict)


class Runtime(Protocol):
    def infer(self, table_id: str, image: Image.Image, *, method: str,
              view_type: str, resolution: tuple[int, int] | None = None) -> Prediction: ...

    def count_tokens(self, raw: str) -> int: ...


def replay_baseline(runtime: Runtime, image: Image.Image, baseline: Prediction) -> dict:
    """Audit one native call against the frozen B0 output after the GPU gate opens."""
    actual = runtime.infer(baseline.table_id, image, method="B0_audit", view_type="global")
    fields = ("table_id", "raw", "html", "processed_size", "image_grid_thw", "visual_tokens", "output_tokens",
              "eos", "truncated", "prompt_hash", "model_hash")
    mismatches = [name for name in fields if getattr(actual, name) != getattr(baseline, name)]
    return {"passed": not mismatches, "mismatched_fields": mismatches,
            "table_id": baseline.table_id, "expected_output_hash": text_hash(baseline.raw),
            "actual_output_hash": text_hash(actual.raw), "latency": actual.latency,
            "visual_tokens": actual.visual_tokens, "output_tokens": actual.output_tokens,
            "purpose": "runtime compatibility audit; excluded from formal method-call averages"}


def cached_prediction(record: dict[str, Any], image: Image.Image) -> Prediction:
    """Load a pre-existing B0 record after its provenance has been checked by the caller."""
    raw = record["raw"]
    rendered = to_html(raw)
    if rendered != record["prediction_html"]:
        raise ValueError("Cached B0 HTML differs from the pinned upstream serializer")
    limits = ProcessorLimits()
    width, height = limits.native_size(image.size)
    grid = [[1, height // limits.patch_size, width // limits.patch_size]]
    tokens = height * width // limits.factor**2
    if tokens != record["visual_tokens"]:
        raise ValueError("Cached B0 token count differs from native dynamic resolution")
    eos = record["finish_reason"] == "stop"
    return Prediction(
        raw=raw, html=rendered, table_id=record["id"], method="B0", view_type="global",
        input_size=image.size, processed_size=(width, height), image_grid_thw=grid,
        visual_tokens=tokens, output_tokens=record["output_tokens"],
        latency=record["seconds"], eos=eos, truncated=not eos, output_hash=text_hash(raw),
    )


class XiaomiRuntime:
    """The permission callback must validate the orchestration gate before CUDA work."""

    def __init__(self, model_path: str | Path, *, permission_check: Callable[[], None],
                 device: str = "cuda", call_log: str | Path | None = None):
        self.model_path = Path(model_path)
        self.permission_check = permission_check
        self.device = device
        self.call_log = Path(call_log) if call_log is not None else None
        self.model: Any = None
        self.processor: Any = None
        self.load_count = 0
        self.call_count = 0

    def assert_weights_unchanged(self) -> dict[str, str]:
        digests = {name: file_hash(self.model_path / name) for name in MODEL_FILES_SHA256}
        if digests != MODEL_FILES_SHA256:
            raise ValueError("The local Xiaomi snapshot differs from the pinned B0 snapshot")
        return digests

    def _load(self) -> None:
        if self.model is not None:
            return
        self.permission_check()
        self.assert_weights_unchanged()
        import torch
        torch.set_num_threads(2)
        try:
            import fla.ops.gated_delta_rule  # noqa: F401
        except ImportError:
            pass
        from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration

        self.processor = AutoProcessor.from_pretrained(self.model_path, local_files_only=True)
        self.model = Qwen3_5ForConditionalGeneration.from_pretrained(
            self.model_path, local_files_only=True, dtype=torch.bfloat16,
            attn_implementation="sdpa",
        ).to(self.device)
        self.model.requires_grad_(False).eval()
        if self.model.config.model_type != "qwen3_5" or self.model.dtype != torch.bfloat16:
            raise ValueError("Expected the frozen BF16 Xiaomi Qwen3.5 model")
        if self.processor.tokenizer.eos_token_id != self.model.config.eos_token_id:
            raise ValueError("Model and tokenizer disagree about EOS")
        self.load_count += 1

    def infer(self, table_id: str, image: Image.Image, *, method: str,
              view_type: str, resolution: tuple[int, int] | None = None,
              max_visual_tokens: int | None = None, deadline: float | None = None) -> Prediction:
        self.permission_check()
        check_deadline(deadline)
        self._load()
        check_deadline(deadline)
        started = perf_counter()
        limits = ProcessorLimits.from_processor(self.processor)
        prepared = view_image(image, resolution, limits)
        inputs = encode_prompt(self.processor, prepared).to(self.model.device)
        grid = inputs["image_grid_thw"].tolist()
        if len(grid) != 1 or grid[0][0] != 1:
            raise ValueError("Expected a single static table image")
        _, height, width = grid[0]
        visual_tokens = int((inputs["input_ids"] == self.model.config.image_token_id).sum().item())
        if height % limits.merge_size or width % limits.merge_size:
            raise ValueError("Native processor produced an invalid merge grid")
        if visual_tokens != height * width // limits.merge_size**2:
            raise ValueError("Image placeholders differ from native grid token count")
        if visual_tokens > limits.max_visual_tokens:
            raise ValueError("Native image grid exceeds the pinned processor pixel budget")
        if max_visual_tokens is not None and visual_tokens > max_visual_tokens:
            raise BudgetExceeded("Actual native visual tokens exceed the remaining table budget")
        check_deadline(deadline)
        self.call_count += 1
        try:
            output, eos = (generate(self.model, inputs) if deadline is None
                           else generate(self.model, inputs, deadline=deadline))
            raw = self.processor.tokenizer.decode(
                output, skip_special_tokens=True, clean_up_tokenization_spaces=False
            ).strip()
            prediction = Prediction(
                raw=raw, html=to_html(raw), table_id=table_id, method=method, view_type=view_type,
                input_size=image.size, processed_size=(width * limits.patch_size, height * limits.patch_size),
                image_grid_thw=grid, visual_tokens=visual_tokens, output_tokens=int(output.numel()),
                latency=perf_counter() - started, eos=eos, truncated=not eos, output_hash=text_hash(raw),
                error="deadline_exceeded" if deadline is not None and perf_counter() >= deadline else None,
            )
        except BaseException as error:
            self._log_call({"table_id": table_id, "method": method, "view_type": view_type,
                            "status": "failed", "error_type": type(error).__name__,
                            "input_size": image.size, "processed_size": (width * limits.patch_size, height * limits.patch_size),
                            "image_grid_thw": grid, "visual_tokens": visual_tokens,
                            "latency": perf_counter() - started, "prompt_hash": text_hash(TABLE_PROMPT),
                            "model_hash": MODEL_SHA256, "output_hash": None, "output_tokens": None})
            raise
        metadata = asdict(prediction)
        metadata.pop("raw")
        metadata.pop("html")
        self._log_call({**metadata, "status": "completed"})
        return prediction

    def _log_call(self, metadata: dict) -> None:
        if self.call_log is not None:
            self.call_log.parent.mkdir(parents=True, exist_ok=True)
            with self.call_log.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(metadata, ensure_ascii=False) + "\n")

    def memory_stats(self) -> dict[str, int]:
        if self.model is None:
            return {"load_count": self.load_count, "allocated_bytes": 0, "peak_allocated_bytes": 0}
        import torch

        if self.model.device.type != "cuda":
            return {"load_count": self.load_count, "allocated_bytes": 0, "peak_allocated_bytes": 0}
        return {
            "load_count": self.load_count,
            "allocated_bytes": torch.cuda.memory_allocated(self.model.device),
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(self.model.device),
        }

    def count_tokens(self, raw: str) -> int:
        """Count a reconciled serialized output including the terminal EOS."""
        self.permission_check()
        self._load()
        return len(self.processor.tokenizer.encode(raw, add_special_tokens=False)) + 1
