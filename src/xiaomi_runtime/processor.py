"""Native processor sizing with no feature or positional-token edits."""

from dataclasses import dataclass
from typing import Any

from PIL import Image

from ._vendor.smart_resize import smart_resize

TABLE_PROMPT = "Parse the table in the image into OTSL."


@dataclass(frozen=True)
class ProcessorLimits:
    patch_size: int = 16
    merge_size: int = 2
    min_pixels: int = 65536
    max_pixels: int = 16777216

    @property
    def factor(self) -> int:
        return self.patch_size * self.merge_size

    @property
    def max_visual_tokens(self) -> int:
        return self.max_pixels // self.factor**2

    def native_size(self, size: tuple[int, int], *, max_pixels: int | None = None) -> tuple[int, int]:
        width, height = size
        if width <= 0 or height <= 0:
            raise ValueError("Image dimensions must be positive")
        budget = self.max_pixels if max_pixels is None else min(max_pixels, self.max_pixels)
        if budget < self.min_pixels:
            raise ValueError("Pixel budget falls below the official minimum")
        height, width = smart_resize(
            height, width, factor=self.factor, min_pixels=self.min_pixels, max_pixels=budget
        )
        return width, height

    @classmethod
    def from_processor(cls, processor: Any) -> "ProcessorLimits":
        vision = processor.image_processor
        return cls(
            patch_size=vision.patch_size,
            merge_size=vision.merge_size,
            min_pixels=vision.size["shortest_edge"],
            max_pixels=vision.size["longest_edge"],
        )


def view_image(image: Image.Image, resolution: tuple[int, int] | None,
               limits: ProcessorLimits) -> Image.Image:
    """Leave the global PIL input intact and resize requested extra views only."""
    if resolution is None:
        return image
    legal_size = limits.native_size(resolution)
    return image.resize(legal_size, Image.Resampling.BICUBIC)


def encode_prompt(processor: Any, image: Image.Image) -> Any:
    messages = [{"role": "user", "content": [
        {"type": "text", "text": TABLE_PROMPT},
        {"type": "image", "image": image},
    ]}]
    return processor.apply_chat_template(
        messages, add_generation_prompt=True, enable_thinking=False,
        tokenize=True, return_dict=True, return_tensors="pt",
    )
