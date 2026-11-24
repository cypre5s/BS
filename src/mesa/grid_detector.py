"""Image-only rule candidates, with explicit support rather than edge-as-truth."""
from dataclasses import asdict, dataclass

import numpy as np
from scipy import ndimage

from reacquisition.region_selector import GridGeometry
from render_verify.visual_checker import _longest_runs


@dataclass(frozen=True)
class Boundary:
    boundary_id: str
    coordinate: float
    orientation: str
    line_support: float
    intersection_support: float
    continuity: float
    confidence: float
    source_method: str = "adaptive_ink_directional_runs_nearby_rule_band_clustering"


@dataclass(frozen=True)
class GridDetection:
    horizontal: tuple[Boundary, ...]
    vertical: tuple[Boundary, ...]
    intersections: tuple[tuple[float, float], ...]
    threshold: float
    frame_support: float = 0.60

    def trusted(self, orientation):
        return tuple(b for b in (self.horizontal if orientation == "horizontal" else self.vertical)
                     if b.confidence >= self.threshold)

    @property
    def rows(self):
        return tuple(b.coordinate for b in self.trusted("horizontal"))

    @property
    def columns(self):
        return tuple(b.coordinate for b in self.trusted("vertical"))

    @property
    def reliable(self):
        # Stacked glyphs or short header strokes cannot establish an outer grid.
        horizontal = self.trusted("horizontal")
        vertical = self.trusted("vertical")
        return (len(horizontal) >= 3 and len(vertical) >= 2
                and horizontal[0].line_support >= self.frame_support
                and horizontal[-1].line_support >= self.frame_support
                and vertical[0].line_support >= self.frame_support
                and vertical[-1].line_support >= self.frame_support)

    def geometry(self, rows, columns):
        ok = self.reliable and len(self.rows) == rows + 1 and len(self.columns) == columns + 1
        return GridGeometry(self.rows, self.columns, ok,
                            "image_grid_matches_anchor" if ok else "image_grid_not_uniquely_matched")

    def as_dict(self):
        return {**asdict(self), "row_boundary_candidates": self.rows, "column_boundary_candidates": self.columns,
                "reliable": self.reliable}


class GridBoundaryDetector:
    def __init__(self, config):
        self.config = config

    def detect(self, image):
        cfg = self.config
        gray = np.asarray(image.convert("L"), dtype=np.float32)
        # Local background suppresses broad edge shadows; thresholding alone is
        # never sufficient: thinness, long strokes and intersections also matter.
        background = ndimage.maximum_filter(gray, size=21)
        dark = (gray < background - 35) & (gray < 210)
        gap = cfg.grid_gap_pixels + 1

        def candidates(mask):
            closed = ndimage.binary_closing(mask, structure=np.ones((1, gap), bool), border_value=1)
            strokes = ndimage.binary_opening(closed, structure=np.ones((1,max(8,int(mask.shape[1]*cfg.grid_min_line_fraction))),bool))
            runs = _longest_runs(closed)
            selected = np.flatnonzero(runs >= mask.shape[1] * cfg.grid_min_line_fraction)
            if not len(selected):
                return []
            groups = np.split(selected, np.flatnonzero(np.diff(selected) > 1) + 1)
            output = []
            for group in groups:
                if len(group) > cfg.grid_max_line_thickness:
                    continue
                at = int(group[np.argmax(runs[group])])
                stroke = np.any(strokes[max(0,at-1):at+2],axis=0)
                ink = np.any(mask[max(0, at-1):at+2], axis=0) & stroke
                pos = np.flatnonzero(stroke)
                if len(pos) < 2:
                    continue
                extent = pos[-1] - pos[0] + 1
                output.append((float(at), float(ink.mean()), min(1., float(runs[at] / extent)),
                               int(pos[0]), int(pos[-1]), len(group)))
            # Double rules are one separator band, not an invented tiny row.
            # The fixed pixel cap is explicit; no target row count is used here.
            clusters: list[list[tuple[float, float, float, int, int, int]]] = []
            for value in output:
                if clusters and value[0]-clusters[-1][0][0] <= cfg.grid_merge_distance:
                    clusters[-1].append(value)
                else:
                    clusters.append([value])
            return [(float(np.mean([v[0] for v in group])), max(v[1] for v in group),
                     float(np.mean([v[2] for v in group])), min(v[3] for v in group),
                     max(v[4] for v in group),sum(v[5] for v in group)) for group in clusters]

        horizontal, vertical = candidates(dark), candidates(dark.T)
        intersections = []

        def score(candidates_, cross, orientation):
            output = []
            typical_width = float(np.median([c[-1] for c in candidates_])) if candidates_ else 1.0
            for i, (at, support, continuity, start, stop, thickness) in enumerate(candidates_):
                hits, possible = 0, 0
                for other, _, _, cross_start, cross_stop, _ in cross:
                    if not (start - 2 <= other <= stop + 2 and cross_start - 2 <= at <= cross_stop + 2):
                        continue
                    x, y = (int(other), int(at)) if orientation == "horizontal" else (int(at), int(other))
                    possible += 1
                    exists = bool(dark[max(0,y-2):y+3, max(0,x-2):x+3].any())
                    hits += exists
                    if exists and orientation == "horizontal":
                        intersections.append((float(x), float(y)))
                junctions = hits / possible if possible >= 2 else 0.0
                confidence = (.35 * support + .35 * continuity + .30 * junctions) * min(1., 2*typical_width/thickness)
                output.append(Boundary(f"{orientation}_{i}", at, orientation, support, junctions,
                                       continuity, confidence))
            return tuple(output)
        h = score(horizontal, vertical, "horizontal")
        v = score(vertical, horizontal, "vertical")
        return GridDetection(h, v, tuple(intersections), cfg.grid_min_confidence, cfg.grid_min_frame_support)
