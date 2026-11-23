"""Thin accessors around Xiaomi's unchanged deployment OTSL converter."""

from dataclasses import dataclass

from ._vendor.otsl import (
    ALL_TAGS,
    BOTH,
    CELL,
    EMPTY,
    LEFT,
    ROW_BREAK,
    TAG_RE,
    UP,
    convert_otsl_to_html,
    pad_rows,
    parse_grid,
    resolve_cells,
)

__all__ = [
    "ALL_TAGS", "BOTH", "CELL", "EMPTY", "LEFT", "ROW_BREAK", "TAG_RE", "UP",
    "TableCell", "TableStructure", "pad_rows", "parse_grid", "parse_table",
    "resolve_cells", "to_html",
]


@dataclass(frozen=True)
class TableCell:
    row: int
    column: int
    rowspan: int
    colspan: int
    text: str


@dataclass(frozen=True)
class TableStructure:
    rows: int
    columns: int
    cells: tuple[TableCell, ...]
    html: str


def to_html(raw: str) -> str:
    return convert_otsl_to_html(raw)


def parse_table(raw: str) -> TableStructure:
    """Expose upstream grid coordinates, including its documented repairs."""
    if not isinstance(raw, str) or not TAG_RE.search(raw):
        raise ValueError("Expected Xiaomi OTSL cell tags")
    tags, texts = parse_grid(pad_rows(raw))
    cells = tuple(
        TableCell(cell.row, cell.column, cell.rowspan, cell.colspan, cell.text)
        for cell in resolve_cells(tags, texts)
    )
    if not cells:
        raise ValueError("The OTSL table has no origin cells")
    return TableStructure(len(tags), max(map(len, tags), default=0), cells, to_html(raw))
