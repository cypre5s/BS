"""Inspect HTML table topology without inventing OTSL parsing rules."""

from __future__ import annotations

from dataclasses import dataclass

from lxml import etree, html


@dataclass(frozen=True)
class HTMLCell:
    row: int
    column: int
    rowspan: int
    colspan: int
    text: str


@dataclass(frozen=True)
class HTMLStructure:
    rows: int
    columns: int
    cells: tuple[HTMLCell, ...]

    @property
    def signature(self) -> tuple:
        return (
            self.rows,
            self.columns,
            tuple((c.row, c.column, c.rowspan, c.colspan, c.text) for c in self.cells),
        )

    @property
    def content(self) -> tuple[str, ...]:
        return tuple(cell.text for cell in self.cells if cell.text.strip())


class StructureError(ValueError):
    """The input cannot be represented as one well-defined rectangular table."""


def inspect_html(markup: str) -> HTMLStructure:
    """Audit placement, spans, blanks and text in the official serializer output."""
    try:
        root = html.fragment_fromstring(markup, create_parent="div")
    except (etree.LxmlError, ValueError) as error:
        raise StructureError("HTML could not be parsed") from error
    tables = root.xpath(".//table")
    if len(tables) != 1:
        raise StructureError("Expected exactly one table")
    table = tables[0]
    rows = table.xpath("./tr|./thead/tr|./tbody/tr|./tfoot/tr")
    if not rows:
        raise StructureError("The table has no rows")
    occupied: set[tuple[int, int]] = set()
    cells = []
    width = 0
    for row_index, row in enumerate(rows):
        column = 0
        for element in row:
            if not isinstance(element.tag, str):
                continue
            if element.tag not in ("td", "th"):
                raise StructureError("Unexpected row element")
            while (row_index, column) in occupied:
                column += 1
            try:
                rowspan = int(element.get("rowspan", "1"))
                colspan = int(element.get("colspan", "1"))
            except ValueError as error:
                raise StructureError("Cell span is not an integer") from error
            if not 1 <= rowspan <= len(rows) - row_index or not 1 <= colspan <= 1000:
                raise StructureError("Cell span exceeds table bounds")
            positions = {
                (r, c) for r in range(row_index, row_index + rowspan) for c in range(column, column + colspan)
            }
            if occupied.intersection(positions):
                raise StructureError("Cell spans overlap")
            occupied.update(positions)
            cells.append(HTMLCell(row_index, column, rowspan, colspan, element.text_content()))
            column += colspan
            width = max(width, column)
    if width == 0 or len(occupied) != len(rows) * width:
        raise StructureError("Table rows do not cover a rectangular grid")
    return HTMLStructure(len(rows), width, tuple(cells))


def split_blank_spans(structure: HTMLStructure, positions: set[tuple[int, int]]) -> HTMLStructure:
    """Describe only the verified blank spans that a proposed repair may split."""
    cells = []
    for cell in structure.cells:
        if (cell.row, cell.column) not in positions:
            cells.append(cell)
            continue
        if cell.text.strip():
            raise StructureError("A text-bearing cell cannot be split automatically")
        cells.extend(
            HTMLCell(row, column, 1, 1, "")
            for row in range(cell.row, cell.row + cell.rowspan)
            for column in range(cell.column, cell.column + cell.colspan)
        )
    return HTMLStructure(
        structure.rows, structure.columns, tuple(sorted(cells, key=lambda c: (c.row, c.column)))
    )
