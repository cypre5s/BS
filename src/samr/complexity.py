"""A3 image-only complexity; constants fixed from engineering ranges, not labels."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from math import log2

import numpy as np
from PIL import Image

WEIGHTS = {"aspect_ratio": 0.10, "original_area": 0.15, "edge_density": 0.20,
           "component_density": 0.20, "text_like_density": 0.20, "visual_token_count": 0.15}
LOW_THRESHOLD = 0.35
HIGH_THRESHOLD = 0.65


def _component_features(mask: np.ndarray) -> tuple[int, int]:
    """Four-connected components on a bounded thumbnail; return count and text-like area."""
    visited = np.zeros(mask.shape, dtype=bool)
    count = text_area = 0
    height, width = mask.shape
    for y, x in zip(*np.nonzero(mask)):
        if visited[y, x]:
            continue
        count += 1
        stack = [(int(y), int(x))]
        visited[y, x] = True
        min_x = max_x = int(x)
        min_y = max_y = int(y)
        area = 0
        while stack:
            row, column = stack.pop()
            area += 1
            min_x, max_x = min(min_x, column), max(max_x, column)
            min_y, max_y = min(min_y, row), max(max_y, row)
            for r, c in ((row - 1, column), (row + 1, column), (row, column - 1), (row, column + 1)):
                if 0 <= r < height and 0 <= c < width and mask[r, c] and not visited[r, c]:
                    visited[r, c] = True
                    stack.append((r, c))
        w, h = max_x - min_x + 1, max_y - min_y + 1
        if 3 <= area <= 4096 and 2 <= h <= 64 and 1 <= w <= 64 and 0.1 <= w / h <= 10:
            text_area += area
    return count, text_area


@dataclass(frozen=True)
class ComplexityFeatures:
    raw: dict[str, float]
    normalized: dict[str, float]
    weights: dict[str, float]
    score: float
    level: str
    feature_image_size: tuple[int, int]

    def as_dict(self) -> dict:
        return asdict(self)


class VisualComplexityEstimator:
    def estimate(self, image: Image.Image, *, visual_token_count: int) -> ComplexityFeatures:
        if type(visual_token_count) is not int or visual_token_count <= 0:
            raise ValueError("An actual positive native token count is required")
        thumbnail = image.convert("L")
        thumbnail.thumbnail((512, 512), Image.Resampling.BILINEAR)
        gray = np.asarray(thumbnail, dtype=np.int16)
        components, text_area = _component_features(gray < 160)
        gradients = [np.abs(np.diff(gray, axis=axis)).reshape(-1) for axis in (0, 1)]
        edges = np.concatenate(gradients)
        raw = {"aspect_ratio": max(image.size) / min(image.size),
               "original_area": float(image.width * image.height),
               "edge_density": float(np.mean(edges >= 32)) if len(edges) else 0.0,
               "component_density": components / gray.size, "text_like_density": text_area / gray.size,
               "visual_token_count": float(visual_token_count)}
        normalized = {"aspect_ratio": log2(raw["aspect_ratio"]) / 4,
                      "original_area": log2(max(1, raw["original_area"] / 65536)) / 8,
                      "edge_density": raw["edge_density"] / 0.20,
                      "component_density": raw["component_density"] / 0.01,
                      "text_like_density": raw["text_like_density"] / 0.20,
                      "visual_token_count": log2(max(1, visual_token_count / 64)) / 6}
        normalized = {name: max(0.0, min(1.0, value)) for name, value in normalized.items()}
        score = sum(WEIGHTS[name] * value for name, value in normalized.items())
        level = "LOW" if score < LOW_THRESHOLD else "MEDIUM" if score < HIGH_THRESHOLD else "HIGH"
        return ComplexityFeatures(raw, normalized, dict(WEIGHTS), score, level, thumbnail.size)
