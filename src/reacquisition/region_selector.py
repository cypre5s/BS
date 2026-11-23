"""Map candidate rows to independently detected image rule boundaries."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import pairwise

import numpy as np
from PIL import Image

from xiaomi_runtime.otsl import TableStructure

from .crop_mapper import Box, CropTransform, intersection_over_union
from .uncertainty import CandidateRows


@dataclass(frozen=True)
class GridGeometry:
    horizontal: tuple[float, ...]
    vertical: tuple[float, ...]
    confident: bool
    reason: str


@dataclass(frozen=True)
class SelectedRegion:
    row_start: int
    row_stop: int
    column_start: int
    column_stop: int
    reasons: tuple[str, ...]
    transform: CropTransform

    @property
    def box(self) -> Box:
        return self.transform.requested_box


def _centres(indices: np.ndarray) -> tuple[float, ...]:
    if not indices.size:
        return ()
    groups = np.split(indices, np.flatnonzero(np.diff(indices) > 1) + 1)
    return tuple(float(group.mean()) for group in groups)


def _line_positions(ink: np.ndarray) -> tuple[float, ...]:
    """Require long contiguous strokes as well as broad line coverage."""
    length = ink.shape[1]
    required = max(8, int(np.ceil(length * 0.45)))
    if required > length:
        return ()
    prefix = np.pad(np.cumsum(ink, axis=1), ((0, 0), (1, 0)))
    longest_window = (prefix[:, required:] - prefix[:, :-required]).max(axis=1)
    candidates = (longest_window >= required) & (ink.mean(axis=1) >= 0.55)
    return _centres(np.flatnonzero(candidates))


def detect_grid(image: Image.Image, table: TableStructure) -> GridGeometry:
    """Only accept an observed grid that uniquely indexes every parsed row/column."""
    gray = np.asarray(image.convert("L"))
    low, high = float(gray.min()), float(np.quantile(gray, 0.99))
    if high - low < 32:
        return GridGeometry((), (), False, "insufficient_contrast")
    threshold = min(160.0, float(low + (high - low) * 0.5))
    ink = gray <= threshold
    if ink.mean() > 0.5:
        return GridGeometry((), (), False, "foreground_polarity_ambiguous")
    horizontal = _line_positions(ink)
    vertical = _line_positions(ink.T)
    counts_match = len(horizontal) == table.rows + 1 and len(vertical) == table.columns + 1
    minimum_gap = all(
        right - left >= 4
        for positions in (horizontal, vertical)
        for left, right in pairwise(positions)
    )
    if not counts_match or not minimum_gap:
        return GridGeometry(horizontal, vertical, False, "grid_does_not_uniquely_match_structure")
    return GridGeometry(horizontal, vertical, True, "observed_rule_grid_matches_structure")


def _close_over_spans(start: int, stop: int, table: TableStructure) -> tuple[int, int]:
    while True:
        intersecting = [cell for cell in table.cells if cell.row < stop and cell.row + cell.rowspan > start]
        expanded = (
            min([start, *(cell.row for cell in intersecting)]),
            max([stop, *(cell.row + cell.rowspan for cell in intersecting)]),
        )
        if expanded == (start, stop):
            return expanded
        start, stop = expanded


def select_regions(
    candidates: list[CandidateRows],
    table: TableStructure,
    geometry: GridGeometry,
    image_size: tuple[int, int],
    *,
    max_regions: int = 2,
    padding: int = 4,
    scale: float = 1.75,
    boundary_margin: int = 12,
) -> list[SelectedRegion]:
    if not geometry.confident:
        return []
    width, height = image_size
    candidates = list(candidates)
    if geometry.horizontal[0] <= boundary_margin:
        candidates.append(CandidateRows(0, min(2, table.rows), ("image_boundary",), 40))
    if height - geometry.horizontal[-1] <= boundary_margin:
        candidates.append(CandidateRows(max(0, table.rows - 2), table.rows, ("image_boundary",), 40))
    if geometry.vertical[0] <= boundary_margin or width - geometry.vertical[-1] <= boundary_margin:
        candidates.append(CandidateRows(0, min(2, table.rows), ("image_boundary",), 40))

    selected: list[SelectedRegion] = []
    for candidate in sorted(candidates, key=lambda row: (-row.priority, row.start, row.stop, row.reasons)):
        start, stop = _close_over_spans(candidate.start, candidate.stop, table)
        # A whole-table crop supplies no additional local evidence.
        if start == 0 and stop == table.rows:
            continue
        box = (geometry.vertical[0], geometry.horizontal[start], geometry.vertical[-1], geometry.horizontal[stop])
        if any(intersection_over_union(box, region.box) > 0 for region in selected):
            continue
        selected.append(SelectedRegion(
            start, stop, 0, table.columns, candidate.reasons,
            CropTransform.create(image_size, box, padding=padding, scale=scale),
        ))
        if len(selected) >= max_regions:
            break
    return selected
