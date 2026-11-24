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
        return {"status": aligned.status, "reason": aligned.reason,
                "row_indices": tuple(range(anchor.native.rows)),
                "column_indices": tuple(range(anchor.native.columns)), "groups": aligned.groups,
                "cells": [{"local": (c.row_start,c.col_start), "global": (c.row_start,c.col_start),
                           "rowspan": c.rowspan, "colspan": c.colspan} for c in auxiliary.cells]
                           if aligned.status == "MAPPED" else []}
    if view.mapping_quality != "HIGH":
        return {"status": "UNRESOLVED", "reason": "low_quality_proportional_crop_cannot_modify_global"}
    if not geometry.confident:
        return {"status": "UNRESOLVED", "reason": "no_unique_source_image_grid"}
    xs, ys = geometry.vertical, geometry.horizontal

    def boundary(lines, value, lower):
        if lower and value <= lines[0] or not lower and value >= lines[-1]:
            return 0 if lower else len(lines)-1
        hits = [i for i,p in enumerate(lines) if abs(p-value) <= tolerance]
        return hits[0] if len(hits)==1 else None

    if view.channel_id == "vertical" and view.row_context:
        rows, columns = view.row_context, view.column_context
        # Piece metadata is evidence only after revalidation against the current
        # independently detected source grid, not an unchecked cell-index offset.
        for piece in view.pieces:
            if not piece.global_rows:
                return {"status":"UNRESOLVED","reason":"piece_missing_row_mapping"}
            a,b=piece.global_rows[0],piece.global_rows[-1]+1
            box=piece.source_bbox
            if (a<0 or b>=len(ys) or (a>0 and abs(box[1]-ys[a])>tolerance)
                    or (b<len(ys)-1 and abs(box[3]-ys[b])>tolerance)
                    or box[0]>xs[0] or box[2]<xs[-1]):
                return {"status":"UNRESOLVED","reason":"piece_boundary_geometry_mismatch"}
    else:
        x0,y0,x1,y1=view.crop_box
        c0,c1=boundary(xs,x0,True),boundary(xs,x1,False)
        r0,r1=boundary(ys,y0,True),boundary(ys,y1,False)
        if None in (r0,r1,c0,c1) or r1<=r0 or c1<=c0:
            return {"status":"UNRESOLVED","reason":"crop_cuts_cell_or_ambiguous_boundary"}
        rows,columns=tuple(range(r0,r1)),tuple(range(c0,c1))
    if (len(set(rows))!=len(rows) or len(set(columns))!=len(columns)
            or tuple(sorted(rows))!=tuple(rows) or tuple(sorted(columns))!=tuple(columns)):
        return {"status":"UNRESOLVED","reason":"duplicate_or_reordered_rows"}
    if (auxiliary.native.rows,auxiliary.native.columns)!=(len(rows),len(columns)):
        return {"status":"UNRESOLVED","reason":"local_dimensions_disagree_with_spatial_region"}
    positions={(r,c) for r in rows for c in columns}
    for cell in anchor.native.cells:
        if cell.positions & positions and not cell.positions <= positions:
            return {"status":"UNRESOLVED","reason":"global_span_crosses_crop_boundary"}
    mapped=[]
    for cell in auxiliary.cells:
        rr=rows[cell.row_start:cell.row_end]
        cc=columns[cell.col_start:cell.col_end]
        if tuple(rr)!=tuple(range(rr[0],rr[0]+cell.rowspan)) or tuple(cc)!=tuple(range(cc[0],cc[0]+cell.colspan)):
            return {"status":"UNRESOLVED","reason":"local_span_crosses_header_data_seam"}
        mapped.append({"local":(cell.row_start,cell.col_start),"global":(rr[0],cc[0]),
                       "rowspan":cell.rowspan,"colspan":cell.colspan})
    return {"status":"MAPPED","reason":"image_boundaries_piecewise_rows_complete_spans",
            "row_indices":rows,"column_indices":columns,"cells":mapped,
            "positions":sorted(positions),"header_seam_checked":True}
