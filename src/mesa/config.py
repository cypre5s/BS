"""Validated, reproducible channel and resource settings."""
from dataclasses import asdict, dataclass
from math import isfinite
from pathlib import Path

import yaml

from samr.config import IDENTITY

CHANNELS = ("global", "highres", "horizontal", "vertical", "tiles")


@dataclass(frozen=True)
class MESAConfig:
    channels: tuple[str, ...] = CHANNELS
    skip_channels: tuple[str, ...] = ()
    scales: tuple[float, ...] = (1.25, 1.5, 1.75)
    overlap: float = 0.2
    padding: int = 4
    detail_scale: float = 1.5
    strips: int = 2
    header_fraction: float = 0.15
    tile_grid: str = "adaptive"
    density_threshold: float = 0.15
    dense_image_area: int = 1000000
    fusion: bool = False
    max_views_per_table: int = 17
    max_visual_tokens_per_view: int = 8192
    max_total_visual_tokens: int = 32768
    max_latency_per_table: float = 240.0
    max_tiles_per_table: int = 9
    max_attempts_per_view: int = 2
    boundary_tolerance: float = 5.0
    vertical_segmentation_mode: str = "header_preserved"
    vertical_overlap_rows: int = 2
    header_context: bool = True
    max_vertical_views: int = 3
    grid_aware_crops: bool = True
    grid_min_confidence: float = 0.72
    grid_min_frame_support: float = 0.60
    grid_gap_pixels: int = 5
    grid_merge_distance: int = 4
    grid_min_line_fraction: float = 0.25
    grid_max_line_thickness: int = 8
    snap_max_distance: int = 64
    max_header_rows: int = 3
    header_max_fraction: float = 0.35
    seam_gap: int = 4

    def __post_init__(self):
        if self.vertical_segmentation_mode not in {"fixed_overlap", "grid_aligned", "header_preserved"}:
            raise ValueError("Unknown vertical segmentation mode")
        for key in ("vertical_overlap_rows", "grid_gap_pixels", "seam_gap", "grid_merge_distance"):
            if type(getattr(self, key)) is not int or getattr(self, key) < 0:
                raise ValueError("Invalid nonnegative integer: " + key)
        for key in ("max_vertical_views", "grid_max_line_thickness", "snap_max_distance", "max_header_rows"):
            if type(getattr(self, key)) is not int or getattr(self, key) < 1:
                raise ValueError("Invalid positive integer: " + key)
        if self.max_vertical_views > 3 or self.max_header_rows > 4:
            raise ValueError("Unbounded segmentation")
        if any(type(getattr(self, k)) is not bool for k in ("header_context", "grid_aware_crops")):
            raise ValueError("Context and grid flags must be booleans")
        for value in (self.grid_min_confidence, self.grid_min_line_fraction, self.header_max_fraction, self.grid_min_frame_support):
            if type(value) not in (int, float) or not isfinite(value) or not 0 < value <= 1:
                raise ValueError("Invalid grid/header confidence setting")
        if (not self.channels or len(set(self.channels)) != len(self.channels)
                or any(c not in CHANNELS for c in (*self.channels, *self.skip_channels))):
            raise ValueError("Unknown or repeated channel")
        if "global" in self.skip_channels:
            raise ValueError("Global anchor cannot be skipped")
        if not self.scales or any(s not in (1.25, 1.5, 1.75) for s in self.scales):
            raise ValueError("Unsupported high-resolution scale")
        if len(set(self.scales)) != len(self.scales):
            raise ValueError("Duplicate scales")
        for value in (self.overlap, self.header_fraction, self.density_threshold):
            if type(value) not in (int, float) or not isfinite(value) or not 0 <= value < 0.5:
                raise ValueError("Invalid overlap/header/density fraction")
        if self.overlap <= 0 or self.tile_grid not in ("2x2", "3x3", "adaptive"):
            raise ValueError("Tiles require positive overlap and a supported grid")
        if self.detail_scale not in (1.0, 1.25, 1.5, 1.75):
            raise ValueError("Invalid detail scale")
        for key in ("strips", "max_views_per_table", "max_visual_tokens_per_view", "max_total_visual_tokens",
                    "max_tiles_per_table", "max_attempts_per_view", "dense_image_area"):
            value = getattr(self, key)
            if type(value) is not int or value < 1:
                raise ValueError("Invalid positive integer: " + key)
        if self.strips > 4 or self.max_tiles_per_table > 9 or self.max_attempts_per_view > 3:
            raise ValueError("Unbounded crop/retry request")
        if type(self.padding) is not int or self.padding < 0:
            raise ValueError("Invalid padding")
        if type(self.fusion) is not bool:
            raise ValueError("fusion must be boolean")
        if not 64 <= self.max_visual_tokens_per_view <= 16384 or self.max_total_visual_tokens < 64:
            raise ValueError("Visual token bounds outside the native contract")
        for value in (self.max_latency_per_table, self.boundary_tolerance):
            if type(value) not in (int, float) or not isfinite(value) or value <= 0:
                raise ValueError("Invalid latency or boundary tolerance")

    @property
    def enabled(self):
        # A channel can run alone as an auxiliary experiment, but always with H0.
        return tuple(c for c in CHANNELS if c == "global" or c in self.channels and c not in self.skip_channels)

    def as_dict(self):
        return {**IDENTITY, **asdict(self)}

    @classmethod
    def load(cls, path: Path):
        data = yaml.safe_load(path.read_text())
        if not isinstance(data, dict):
            raise TypeError("Expected configuration mapping")
        for key, expected in IDENTITY.items():
            if data.pop(key, None) != expected:
                raise ValueError("Pinned runtime identity differs: " + key)
        for key in ("channels", "skip_channels", "scales"):
            if key in data:
                if not isinstance(data[key], list):
                    raise ValueError(key + " must be a list")
                data[key] = tuple(data[key])
        return cls(**data)
