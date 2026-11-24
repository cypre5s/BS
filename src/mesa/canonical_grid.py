"""Official OTSL reconstruction and explicit, geometry-based local span mapping."""
from dataclasses import asdict, dataclass

from samr.canonical import CanonicalTableGrid, align_grids


@dataclass(frozen=True)
class Cell:
    row_start: int
    row_end: int
    col_start: int
    col_end: int
    rowspan: int
    colspan: int
    cell_type: str
    text: str
    source_channel: str
    source_view: str


@dataclass(frozen=True)
class CanonicalGrid:
    native: CanonicalTableGrid
    cells: tuple[Cell, ...]
    source_channel: str
    source_view: str

    @classmethod
    def from_otsl(cls, raw, channel, view):
        grid = CanonicalTableGrid.from_otsl(raw, 1.0)
        cells = tuple(Cell(c.row_index, c.row_index + c.rowspan, c.column_index, c.column_index + c.colspan,
                           c.rowspan, c.colspan, c.cell_type, c.text, channel, view) for c in grid.cells)
        return cls(grid, cells, channel, view)

    def as_dict(self):
        return {"rows": self.native.rows, "columns": self.native.columns, "status": self.native.status,
                "reason": self.native.reason, "cells": [asdict(c) for c in self.cells]}


def map_view(anchor, auxiliary, view, geometry, tolerance=5.0):
    if anchor.native.status != "RESOLVED" or auxiliary.native.status != "RESOLVED":
        return {"status": "UNRESOLVED", "reason": "invalid_canonical_grid"}
    if not view.local:
        aligned = align_grids(anchor.native, auxiliary.native)
        return {"status": aligned.status, "reason": aligned.reason, "offset": (0, 0),
                "shape": (anchor.native.rows, anchor.native.columns), "groups": aligned.groups}
    if not geometry.confident:
        return {"status": "UNRESOLVED", "reason": "no_unique_source_image_grid"}
    xs, ys = geometry.vertical, geometry.horizontal
    x0, y0, x1, y1 = view.crop_box
    def boundary(lines, value, lower):
        if lower and value <= lines[0] or not lower and value >= lines[-1]:
            return 0 if lower else len(lines) - 1
        hits = [i for i, p in enumerate(lines) if abs(p - value) <= tolerance]
        return hits[0] if len(hits) == 1 else None
    c0, c1 = boundary(xs, x0, True), boundary(xs, x1, False)
    r0, r1 = boundary(ys, y0, True), boundary(ys, y1, False)
    if None in (r0, r1, c0, c1) or r1 <= r0 or c1 <= c0:
        return {"status": "UNRESOLVED", "reason": "crop_cuts_cell_or_ambiguous_boundary"}
    if (auxiliary.native.rows, auxiliary.native.columns) != (r1 - r0, c1 - c0):
        return {"status": "UNRESOLVED", "reason": "local_dimensions_disagree_with_spatial_region"}
    for cell in anchor.cells:
        intersects = cell.row_start < r1 and cell.row_end > r0 and cell.col_start < c1 and cell.col_end > c0
        contained = r0 <= cell.row_start and cell.row_end <= r1 and c0 <= cell.col_start and cell.col_end <= c1
        if intersects and not contained:
            return {"status": "UNRESOLVED", "reason": "global_span_crosses_crop_boundary"}
    mapped = [{"local": (c.row_start, c.col_start), "global": (r0 + c.row_start, c0 + c.col_start),
               "rowspan": c.rowspan, "colspan": c.colspan} for c in auxiliary.cells]
    return {"status": "MAPPED", "reason": "observed_pixel_boundaries_and_complete_spans",
            "offset": (r0, c0), "shape": (r1 - r0, c1 - c0), "cells": mapped}
