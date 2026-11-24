"""Synthetic fixtures only; no mock result is evidence of model accuracy."""

from time import perf_counter

import pytest
from PIL import Image, ImageDraw

from xiaomi_runtime.adapter import Prediction, text_hash
from xiaomi_runtime.budget import BudgetExceeded, DeadlineExceeded
from xiaomi_runtime.otsl import to_html
from xiaomi_runtime.processor import ProcessorLimits


class MockRuntime:
    def __init__(self, outputs=None):
        self.outputs = outputs or ["<fcel>A<fcel>B<nl>"]
        self.calls = []
        self.load_count = 0
        self.model = None
        self.weight_bytes = b"synthetic unchanged weights"

    def infer(self, table_id, image, *, method, view_type, max_visual_tokens, deadline):
        if perf_counter() >= deadline:
            raise DeadlineExceeded("synthetic deadline")
        width, height = ProcessorLimits().native_size(image.size)
        tokens = width * height // 1024
        if tokens > max_visual_tokens:
            raise BudgetExceeded("synthetic actual token guard")
        if self.model is None:
            self.load_count += 1
            self.model = object()
        raw = self.outputs[min(len(self.calls), len(self.outputs) - 1)]
        self.calls.append({"table_id": table_id, "image": image, "view_type": view_type,
                           "max_visual_tokens": max_visual_tokens, "deadline": deadline})
        if isinstance(raw, Exception):
            raise raw
        return Prediction(raw, to_html(raw), table_id, method, view_type,
                          image.size, (width, height), [[1, height // 16, width // 16]],
                          tokens, self.count_tokens(raw), 0.001, output_hash=text_hash(raw))

    def count_tokens(self, raw):
        return len(raw) + 1  # Deliberately synthetic, not the actual model tokenizer.


@pytest.fixture
def mock_runtime():
    return MockRuntime()


@pytest.fixture
def ruled_image():
    image = Image.new("RGB", (300, 200), "white")
    draw = ImageDraw.Draw(image)
    for x in (8, 150, 292):
        draw.line((x, 8, x, 192), fill="black", width=2)
    for y in (8, 100, 192):
        draw.line((8, y, 292, y), fill="black", width=2)
    draw.text((40, 140), "A", fill="black")
    draw.text((190, 140), "B", fill="black")
    return image
