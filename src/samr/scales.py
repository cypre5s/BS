"""A1: isotropic PIL observations followed by the unchanged native processor."""

from __future__ import annotations

from dataclasses import dataclass
from math import sqrt

from PIL import Image

from xiaomi_runtime.adapter import Prediction
from xiaomi_runtime.processor import ProcessorLimits

from .config import CANDIDATE_SCALES


@dataclass(frozen=True)
class ScalePlan:
    requested_scale: float
    original_size: tuple[int, int]
    actual_size: tuple[int, int]
    processed_size: tuple[int, int]
    estimated_visual_tokens: int
    applied_scale: float

    def prepare(self, image: Image.Image) -> Image.Image:
        if image.size != self.original_size:
            raise ValueError("Scale plan belongs to another input size")
        if self.actual_size == image.size:
            return image
        return image.resize(self.actual_size, Image.Resampling.BICUBIC)

    def validate_prediction(self, prediction: Prediction, limits: ProcessorLimits) -> None:
        width, height = self.processed_size
        expected_grid = [[1, height // limits.patch_size, width // limits.patch_size]]
        if (prediction.processed_size != self.processed_size or prediction.image_grid_thw != expected_grid
                or prediction.visual_tokens != self.estimated_visual_tokens
                or prediction.input_size != self.actual_size):
            raise ValueError("Native processor metadata differs from the audited scale plan")


class ScaleLadder:
    def __init__(self, limits: ProcessorLimits | None = None):
        self.limits = limits or ProcessorLimits()

    def plan(self, original_size: tuple[int, int], scale: float) -> ScalePlan:
        if scale not in CANDIDATE_SCALES or min(original_size) <= 0:
            raise ValueError("Unsupported scale or image dimensions")
        width, height = original_size
        # The original global input is never resized here. Cap additional PIL allocations
        # isotropically; the official processor still performs all legal-grid resizing.
        applied = scale if scale == 1.0 else min(scale, sqrt(self.limits.max_pixels / (width * height)))
        actual = original_size if scale == 1.0 else (max(1, round(width * applied)),
                                                   max(1, round(height * applied)))
        processed = self.limits.native_size(actual)
        tokens = processed[0] * processed[1] // self.limits.factor**2
        return ScalePlan(scale, original_size, actual, processed, tokens, applied)
