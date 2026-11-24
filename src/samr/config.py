"""Pinned, typed SAMR settings with no machine paths or reference answers."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from math import isfinite
from pathlib import Path

import yaml

from orchestration.storage import object_hash
from xiaomi_runtime.adapter import MODEL_FILES_SHA256, MODEL_SHA256, text_hash
from xiaomi_runtime.inference import GENERATION_CONFIG
from xiaomi_runtime.processor import TABLE_PROMPT

IDENTITY = {
    "model_hash": MODEL_SHA256,
    "processor_config_hash": MODEL_FILES_SHA256["processor_config.json"],
    "prompt_hash": text_hash(TABLE_PROMPT),
    "generation_config_hash": object_hash(GENERATION_CONFIG),
}
CANDIDATE_SCALES = (1.0, 1.25, 1.5, 1.75, 2.0)


@dataclass(frozen=True)
class SAMRConfig:
    mode: str = "adaptive"
    scales: tuple[float, ...] = (1.0, 1.5, 1.75)
    max_scales_per_table: int = 3
    max_visual_tokens: int = 8192  # Cumulative, including the original global pass.
    max_latency_budget: float = 180.0
    routing_policy: str = "image_complexity_v1"
    fusion_policy: str = "global_anchor_blank_split_v1"
    resolution: str = "original_pixels_times_scale_then_native_processor"

    def __post_init__(self) -> None:
        if self.mode not in {"single", "fixed", "ladder", "adaptive"}:
            raise ValueError("Unknown SAMR mode")
        if (not self.scales or len(set(self.scales)) != len(self.scales)
                or any(type(scale) not in (int, float) for scale in self.scales)
                or any(scale not in CANDIDATE_SCALES for scale in self.scales)):
            raise ValueError("Expected unique supported candidate scales")
        if self.mode == "single" and len(self.scales) != 1:
            raise ValueError("Single scale mode needs exactly one scale")
        if self.mode != "single" and (self.scales[0] != 1.0 or tuple(sorted(self.scales)) != self.scales):
            raise ValueError("Multi-scale modes start with the original global scale and increase")
        if self.mode == "adaptive" and self.scales != (1.0, 1.5, 1.75):
            raise ValueError("The predetermined adaptive policy uses global, 1.5 and 1.75")
        if (type(self.max_scales_per_table) is not int or not 1 <= self.max_scales_per_table <= 5
                or type(self.max_visual_tokens) is not int or self.max_visual_tokens < 64):
            raise ValueError("Invalid scale or cumulative visual-token budget")
        if (type(self.max_latency_budget) not in (int, float)
                or not isfinite(self.max_latency_budget) or self.max_latency_budget <= 0):
            raise ValueError("Latency budget must be finite and positive")
        expected_policy = "image_complexity_v1" if self.mode == "adaptive" else "specified_scales"
        if self.routing_policy != expected_policy:
            raise ValueError("Routing policy does not match mode")
        if self.fusion_policy not in {"retain_global", "global_anchor_blank_split_v1"}:
            raise ValueError("Unknown fusion policy")
        if self.resolution != "original_pixels_times_scale_then_native_processor":
            raise ValueError("Unrecognized resolution contract")

    @classmethod
    def load(cls, path: Path) -> SAMRConfig:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise TypeError("Expected a YAML configuration mapping")
        for name, expected in IDENTITY.items():
            if data.pop(name, None) != expected:
                raise ValueError(f"Pinned identity mismatch: {name}")
        if "scales" in data:
            if not isinstance(data["scales"], list):
                raise ValueError("scales must be a list")
            data["scales"] = tuple(data["scales"])
        try:
            return cls(**data)
        except TypeError as error:
            raise ValueError("Unknown configuration field or invalid field type") from error

    def as_dict(self) -> dict:
        return {**IDENTITY, **asdict(self)}
