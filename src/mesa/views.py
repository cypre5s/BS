"""Isotropic source-pixel views. Native resizing/patch order stay in Xiaomi."""
from dataclasses import asdict, dataclass
from math import ceil, floor, sqrt

import numpy as np
from PIL import Image

from samr.scales import ScaleLadder, ScalePlan

from .config import MESAConfig


@dataclass(frozen=True)
class VisualView:
    channel_id: str
    view_id: str
    source_image_id: str
    original_size: tuple[int, int]
    requested_box: tuple[int, int, int, int]
    crop_box: tuple[int, int, int, int]
    plan: ScalePlan
    overlap: float
    padding: int
    header_context: str

    @property
    def local(self):
        return self.channel_id not in ("global", "highres")

    def prepare(self, image):
        if image.size != self.original_size:
            raise ValueError("View belongs to another source")
        crop = image.crop(self.crop_box) if self.local else image
        return self.plan.prepare(crop)

    def to_original(self, point, *, processed=False):
        size = self.plan.processed_size if processed else self.plan.actual_size
        x0, y0, x1, y1 = self.crop_box
        return x0 + point[0] * (x1 - x0) / size[0], y0 + point[1] * (y1 - y0) / size[1]

    def from_original(self, point, *, processed=False):
        size = self.plan.processed_size if processed else self.plan.actual_size
        x0, y0, x1, y1 = self.crop_box
        return (point[0] - x0) * size[0] / (x1 - x0), (point[1] - y0) * size[1] / (y1 - y0)

    def as_dict(self):
        return asdict(self)


class VisualViewGenerator:
    def __init__(self, config: MESAConfig):
        self.config = config
        self.ladder = ScaleLadder()

    def _view(self, image, source, channel, index, box, scale, header="not_applicable"):
        w, h = image.size
        local = channel not in ("global", "highres")
        pad = self.config.padding if local else 0
        x0, y0, x1, y1 = box
        crop = max(0, x0 - pad), max(0, y0 - pad), min(w, x1 + pad), min(h, y1 + pad)
        cw, ch = crop[2] - crop[0], crop[3] - crop[1]
        # ScaleLadder caps PIL allocation isotropically before native quantization.
        plan = self.ladder.plan((cw, ch), scale)
        return VisualView(channel, f"{channel}_{index}", source, image.size, box, crop, plan,
                          self.config.overlap if local else 0.0, pad, header)

    def generate(self, image: Image.Image, source_image_id: str) -> list[VisualView]:
        w, h = image.size
        if min(w, h) < 2:
            raise ValueError("Input is too small")
        c, result = self.config, []
        result.append(self._view(image, source_image_id, "global", 0, (0, 0, w, h), 1.0))
        if "highres" in c.enabled:
            for scale in c.scales:
                result.append(self._view(image, source_image_id, "highres", scale, (0, 0, w, h), scale))
        if "horizontal" in c.enabled:
            # Move along x: full-height overlapping column strips retain the header.
            for i in range(c.strips):
                x0 = max(0, floor((i - c.overlap / 2) * w / c.strips))
                x1 = min(w, ceil((i + 1 + c.overlap / 2) * w / c.strips))
                result.append(self._view(image, source_image_id, "horizontal", i, (x0, 0, x1, h),
                                         c.detail_scale, "full_height_header_retained"))
        if "vertical" in c.enabled:
            # Move along y. Use contiguous prefixes, never paste disjoint headers
            # onto a crop (which would falsify a simple coordinate transform).
            header = ceil(h * c.header_fraction)
            for i in range(c.strips):
                y1 = min(h, max(header, ceil((i + 1 + c.overlap / 2) * h / c.strips)))
                result.append(self._view(image, source_image_id, "vertical", i, (0, 0, w, y1),
                                         c.detail_scale, "contiguous_header_prefix_and_neighbor_rows"))
        if "tiles" in c.enabled:
            thumb = image.convert("L")
            thumb.thumbnail((256, 256))
            gray = np.asarray(thumb, dtype=float)
            density = float((np.abs(np.diff(gray, axis=1)) > 32).mean())
            n = (3 if density >= c.density_threshold and w * h >= c.dense_image_area else 2)
            if c.tile_grid != "adaptive":
                n = int(c.tile_grid[0])
            # Adapt grid aspect while bounding either dimension by three.
            nx = min(3, max(2, round(n * sqrt(w / h)))) if c.tile_grid == "adaptive" else n
            ny = min(3, max(2, round(n * sqrt(h / w)))) if c.tile_grid == "adaptive" else n
            for j in range(ny):
                for i in range(nx):
                    box = (max(0, floor((i - c.overlap / 2) * w / nx)),
                           max(0, floor((j - c.overlap / 2) * h / ny)),
                           min(w, ceil((i + 1 + c.overlap / 2) * w / nx)),
                           min(h, ceil((j + 1 + c.overlap / 2) * h / ny)))
                    result.append(self._view(image, source_image_id, "tiles", f"{j}_{i}", box, c.detail_scale))
        return result
