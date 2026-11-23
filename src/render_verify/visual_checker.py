"""Compare line topology and blank regions, independent of rendered pixels."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from PIL import Image

from .html_renderer import RenderedTable
from .structural_checker import HTMLCell


@dataclass(frozen=True)
class GridEvidence:
    x_lines: tuple[float, ...]
    y_lines: tuple[float, ...]
    dark: np.ndarray
    threshold: float
    confidence: bool
    reason: str


@dataclass(frozen=True)
class Verification:
    confirmed_grid: bool
    blank_spans_to_split: tuple[tuple[int, int], ...]
    unsupported_boundaries: int
    source_grid: tuple[int, int]
    reason: str

    def as_dict(self) -> dict:
        return asdict(self)


def _longest_runs(mask: np.ndarray) -> np.ndarray:
    """Longest horizontal dark run for every scan line without CV dependencies."""
    longest = np.zeros(mask.shape[0], dtype=np.int32)
    current = np.zeros(mask.shape[0], dtype=np.int32)
    for column in range(mask.shape[1]):
        current = np.where(mask[:, column], current + 1, 0)
        longest = np.maximum(longest, current)
    return longest


def _centers(indices: np.ndarray) -> tuple[float, ...]:
    if len(indices) == 0:
        return ()
    groups = np.split(indices, np.flatnonzero(np.diff(indices) > 1) + 1)
    return tuple(float(group.mean()) for group in groups)


def detect_grid(image: Image.Image, *, minimum_line_fraction: float = 0.65) -> GridEvidence:
    gray = np.asarray(image.convert("L"), dtype=np.float32)
    background = float(np.percentile(gray, 90))
    threshold = min(190.0, background - 45.0)
    dark = gray < threshold
    y_lines = _centers(np.flatnonzero(_longest_runs(dark) >= gray.shape[1] * minimum_line_fraction))
    x_lines = _centers(np.flatnonzero(_longest_runs(dark.T) >= gray.shape[0] * minimum_line_fraction))
    confident = len(x_lines) >= 2 and len(y_lines) >= 2
    if confident:
        confident = (
            x_lines[-1] - x_lines[0] >= gray.shape[1] * 0.65
            and y_lines[-1] - y_lines[0] >= gray.shape[0] * 0.65
        )
    return GridEvidence(
        x_lines,
        y_lines,
        dark,
        threshold,
        confident,
        "framed_grid" if confident else "insufficient_line_evidence",
    )


def _line_present(
    evidence: GridEvidence, axis: str, position: float, begin: float, end: float, fraction: float
) -> bool:
    start, stop = round(begin), round(end) + 1
    at = round(position)
    if axis == "x":
        strip = evidence.dark[max(0, start) : stop, max(0, at - 2) : at + 3]
        coverage = np.any(strip, axis=1) if strip.size else np.zeros(0, dtype=bool)
    else:
        strip = evidence.dark[max(0, at - 2) : at + 3, max(0, start) : stop]
        coverage = np.any(strip, axis=0) if strip.size else np.zeros(0, dtype=bool)
    return bool(len(coverage) >= 5 and coverage.mean() >= fraction)


def _cell_blank(evidence: GridEvidence, cell: HTMLCell, maximum_ink_fraction: float) -> bool:
    x0 = round(evidence.x_lines[cell.column])
    x1 = round(evidence.x_lines[cell.column + cell.colspan]) + 1
    y0 = round(evidence.y_lines[cell.row])
    y1 = round(evidence.y_lines[cell.row + cell.rowspan]) + 1
    mask = evidence.dark[y0:y1, x0:x1].copy()
    excluded = np.zeros_like(mask, dtype=bool)
    for position in evidence.x_lines[cell.column : cell.column + cell.colspan + 1]:
        column = round(position) - x0
        excluded[:, max(0, column - 3) : column + 4] = True
    for position in evidence.y_lines[cell.row : cell.row + cell.rowspan + 1]:
        row = round(position) - y0
        excluded[max(0, row - 3) : row + 4, :] = True
    interior = mask[~excluded]
    return bool(interior.size >= 36 and interior.mean() <= maximum_ink_fraction)


def _missing_inside_span(evidence: GridEvidence, cell: HTMLCell, minimum_line_support: float) -> bool:
    x0, x1 = evidence.x_lines[cell.column], evidence.x_lines[cell.column + cell.colspan]
    y0, y1 = evidence.y_lines[cell.row], evidence.y_lines[cell.row + cell.rowspan]
    vertical = all(
        _line_present(evidence, "x", evidence.x_lines[column], y0, y1, minimum_line_support)
        for column in range(cell.column + 1, cell.column + cell.colspan)
    )
    horizontal = all(
        _line_present(evidence, "y", evidence.y_lines[row], x0, x1, minimum_line_support)
        for row in range(cell.row + 1, cell.row + cell.rowspan)
    )
    return vertical and horizontal


def verify(
    original: Image.Image,
    rendered: RenderedTable,
    *,
    minimum_line_fraction: float = 0.65,
    minimum_line_support: float = 0.95,
    maximum_blank_ink_fraction: float = 0.002,
) -> Verification:
    """Require whole-table axis alignment before proposing any blank-span repair."""
    evidence = detect_grid(original, minimum_line_fraction=minimum_line_fraction)
    structure = rendered.structure
    grid = (max(0, len(evidence.y_lines) - 1), max(0, len(evidence.x_lines) - 1))
    if not evidence.confidence or grid != (structure.rows, structure.columns):
        return Verification(False, (), 0, grid, "source_grid_is_ambiguous")
    # The renderer must visibly preserve every cell boundary recorded by its layout.
    rendered_evidence = detect_grid(rendered.image, minimum_line_fraction=minimum_line_fraction)
    unsupported = 0
    blank_spans = []
    for cell in structure.cells:
        if cell.rowspan == 1 and cell.colspan == 1:
            continue
        if not _missing_inside_span(evidence, cell, minimum_line_support):
            continue
        unsupported += (cell.rowspan - 1) * cell.colspan + (cell.colspan - 1) * cell.rowspan
        if cell.text.strip() or not _cell_blank(evidence, cell, maximum_blank_ink_fraction):
            continue
        blank_spans.append((cell.row, cell.column))
    # A rasterized table frame is essential evidence that the render succeeded.
    if not rendered_evidence.confidence:
        return Verification(False, (), unsupported, grid, "render_has_no_stable_frame")
    return Verification(
        True,
        tuple(blank_spans),
        unsupported,
        grid,
        "blank_merge_boundary_conflict" if blank_spans else "no_confirmed_blank_merge_error",
    )
