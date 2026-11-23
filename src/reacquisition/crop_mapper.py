"""Exact affine coordinates for padded image crops and resized observations."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from math import ceil, floor, isfinite

Box = tuple[float, float, float, float]
Point = tuple[float, float]


@dataclass(frozen=True)
class CropTransform:
    original_size: tuple[int, int]
    requested_box: Box
    crop_box: tuple[int, int, int, int]
    resized_size: tuple[int, int]

    @classmethod
    def create(
        cls,
        original_size: tuple[int, int],
        box: Box,
        *,
        padding: int = 4,
        scale: float = 1.75,
    ) -> CropTransform:
        width, height = original_size
        left, top, right, bottom = box
        if min(width, height) < 1 or padding < 0 or not isfinite(scale) or scale <= 0:
            raise ValueError("Invalid crop dimensions, padding or scale")
        if not all(map(isfinite, box)) or right <= left or bottom <= top:
            raise ValueError("Crop box must have finite, positive dimensions")
        cropped = (
            max(0, floor(left) - padding),
            max(0, floor(top) - padding),
            min(width, ceil(right) + padding + 1),
            min(height, ceil(bottom) + padding + 1),
        )
        crop_width = cropped[2] - cropped[0]
        crop_height = cropped[3] - cropped[1]
        if min(crop_width, crop_height) <= 0:
            raise ValueError("Crop box lies outside the image")
        resized = (max(1, round(crop_width * scale)), max(1, round(crop_height * scale)))
        return cls(original_size, box, cropped, resized)

    @property
    def scale(self) -> Point:
        left, top, right, bottom = self.crop_box
        return self.resized_size[0] / (right - left), self.resized_size[1] / (bottom - top)

    def to_local(self, point: Point) -> Point:
        x_scale, y_scale = self.scale
        return (point[0] - self.crop_box[0]) * x_scale, (point[1] - self.crop_box[1]) * y_scale

    def to_global(self, point: Point) -> Point:
        x_scale, y_scale = self.scale
        return point[0] / x_scale + self.crop_box[0], point[1] / y_scale + self.crop_box[1]

    def box_to_local(self, box: Box) -> Box:
        return (*self.to_local(box[:2]), *self.to_local(box[2:]))

    def box_to_global(self, box: Box) -> Box:
        return (*self.to_global(box[:2]), *self.to_global(box[2:]))

    def as_dict(self) -> dict:
        return {**asdict(self), "scale": self.scale}


def intersection_over_union(first: Box, second: Box) -> float:
    width = max(0.0, min(first[2], second[2]) - max(first[0], second[0]))
    height = max(0.0, min(first[3], second[3]) - max(first[1], second[1]))
    intersection = width * height
    first_area = max(0.0, first[2] - first[0]) * max(0.0, first[3] - first[1])
    second_area = max(0.0, second[2] - second[0]) * max(0.0, second[3] - second[1])
    union = first_area + second_area - intersection
    return intersection / union if union else 0.0
