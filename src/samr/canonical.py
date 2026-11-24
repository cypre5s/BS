"""Strict, lossless coordinate views over Xiaomi's existing OTSL converter."""

from __future__ import annotations

from dataclasses import asdict, dataclass

from render_verify.structural_checker import inspect_html
from xiaomi_runtime.otsl import BOTH, CELL, EMPTY, LEFT, ROW_BREAK, TAG_RE, UP, parse_grid, parse_table


@dataclass(frozen=True)
class CanonicalCell:
    row_index: int
    column_index: int
    rowspan: int
    colspan: int
    cell_type: str
    text: str
    source_scale: float
    confidence: float  # Deterministic validation indicator; NOT a model probability.
    source_token_span: tuple[int, int] | None = None  # No fabricated tokenizer offsets.

    @property
    def origin(self) -> tuple[int, int]:
        return self.row_index, self.column_index

    @property
    def positions(self) -> frozenset[tuple[int, int]]:
        return frozenset((r, c) for r in range(self.row_index, self.row_index + self.rowspan)
                         for c in range(self.column_index, self.column_index + self.colspan))

    @property
    def topology(self) -> tuple[int, int, int, int]:
        return self.row_index, self.column_index, self.rowspan, self.colspan


@dataclass(frozen=True)
class CanonicalTableGrid:
    rows: int
    columns: int
    cells: tuple[CanonicalCell, ...]
    source_scale: float
    status: str
    reason: str
    otsl_valid: bool
    html_parse_status: str

    @classmethod
    def from_otsl(cls, raw: str, source_scale: float) -> CanonicalTableGrid:
        otsl_valid = False
        try:
            matches = list(TAG_RE.finditer(raw))
            if not matches or raw[:matches[0].start()].strip():
                raise ValueError("missing_tags_or_unexplained_prefix")
            widths: list[int] = []
            width = 0
            for index, match in enumerate(matches):
                tag = match.group()
                end = matches[index + 1].start() if index + 1 < len(matches) else len(raw)
                if tag != CELL and raw[match.end():end].strip():
                    raise ValueError("text_on_non_content_tag")
                if tag == ROW_BREAK:
                    if width == 0:
                        raise ValueError("empty_row_has_no_declared_columns")
                    widths.append(width)
                    width = 0
                else:
                    width += 1
            if width:
                widths.append(width)
            if not widths or len(set(widths)) != 1:
                raise ValueError("ragged_rows_need_upstream_repair")
            tags, texts = parse_grid(raw)
            official = parse_table(raw)
            if (official.rows, official.columns) != (len(widths), widths[0]):
                raise ValueError("upstream_repair_would_change_dimensions")
            cells = tuple(CanonicalCell(
                cell.row, cell.column, cell.rowspan, cell.colspan,
                tags[cell.row][cell.column][1:-1], texts[cell.row][cell.column], source_scale, 1.0,
            ) for cell in official.cells)
            owner: dict[tuple[int, int], CanonicalCell] = {}
            for cell in cells:
                for r, c in cell.positions:
                    if (r, c) in owner or r >= official.rows or c >= official.columns:
                        raise ValueError("overlapping_or_out_of_bounds_spans")
                    owner[r, c] = cell
                    expected = (tags[r][c] if (r, c) == cell.origin else
                                LEFT if r == cell.row_index else UP if c == cell.column_index else BOTH)
                    if tags[r][c] != expected:
                        raise ValueError("continuation_does_not_match_rectangular_span")
            if len(owner) != official.rows * official.columns:
                raise ValueError("unowned_continuation_or_grid_hole")
            otsl_valid = True
            html_grid = inspect_html(official.html)
            signature = tuple((c.row, c.column, c.rowspan, c.colspan) for c in html_grid.cells)
            if ((html_grid.rows, html_grid.columns) != (official.rows, official.columns)
                    or signature != tuple(c.topology for c in cells)):
                raise ValueError("official_html_topology_differs")
            return cls(official.rows, official.columns, cells, source_scale, "RESOLVED",
                       "official_grammar_and_html_verified", True, "valid")
        except (ValueError, TypeError, IndexError) as error:
            return cls(0, 0, (), source_scale, "UNRESOLVED", str(error), otsl_valid,
                       "invalid" if otsl_valid else "not_checked")

    def as_dict(self) -> dict:
        return asdict(self)

    def owners(self) -> dict[tuple[int, int], CanonicalCell]:
        return {position: cell for cell in self.cells for position in cell.positions}


@dataclass(frozen=True)
class Alignment:
    status: str
    reason: str
    groups: tuple[dict, ...] = ()


def align_grids(anchor: CanonicalTableGrid, auxiliary: CanonicalTableGrid) -> Alignment:
    """Intersect occupied lattice regions; never zip cells by ordinal index."""
    if anchor.status != "RESOLVED" or auxiliary.status != "RESOLVED":
        return Alignment("UNRESOLVED", "parse_or_ownership_ambiguous")
    if (anchor.rows, anchor.columns) != (auxiliary.rows, auxiliary.columns):
        return Alignment("UNRESOLVED", "different_global_dimensions")
    anchor_owner, helper_owner = anchor.owners(), auxiliary.owners()
    remaining = set(anchor_owner)
    groups = []
    while remaining:
        positions = {min(remaining)}
        while True:
            expanded = set(positions)
            for position in positions:
                expanded.update(anchor_owner[position].positions)
                expanded.update(helper_owner[position].positions)
            if expanded == positions:
                break
            positions = expanded
        before = tuple(sorted({anchor_owner[p].origin for p in positions}))
        after = tuple(sorted({helper_owner[p].origin for p in positions}))
        relation = ("same_region" if len(before) == len(after) == 1 else
                    "split" if len(before) == 1 else "merge" if len(after) == 1 else "UNRESOLVED")
        groups.append({"anchor_origins": before, "auxiliary_origins": after,
                       "relation": relation, "positions": sorted(positions)})
        remaining.difference_update(positions)
    return Alignment("MAPPED", "shared_lattice_regions_only_not_proof_of_semantic_correspondence", tuple(groups))


def split_blank_origins(raw: str, grid: CanonicalTableGrid, origins: set[tuple[int, int]]) -> str:
    """Edit only official continuation tags; preserve every content byte and blank slot."""
    positions: set[tuple[int, int]] = set()
    for cell in grid.cells:
        if cell.origin in origins:
            if cell.text or cell.cell_type != EMPTY[1:-1]:
                raise ValueError("Only declared empty origins may be split")
            positions.update(cell.positions - {cell.origin})
    replacements = []
    row = column = 0
    for match in TAG_RE.finditer(raw):
        if match.group() == ROW_BREAK:
            row += 1
            column = 0
            continue
        if (row, column) in positions:
            replacements.append((match.start(), match.end()))
        column += 1
    result = raw
    for start, end in reversed(replacements):
        result = result[:start] + EMPTY + result[end:]
    return result
