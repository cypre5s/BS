"""Deterministic candidates for closer observation of a parsed table."""

from __future__ import annotations

from dataclasses import dataclass

from xiaomi_runtime.otsl import TAG_RE, TableStructure


@dataclass(frozen=True)
class CandidateRows:
    start: int
    stop: int
    reasons: tuple[str, ...]
    priority: int


def raw_row_widths(raw: str) -> list[int]:
    """Read the row delimiters using the imported official token expression."""
    result = []
    width = 0
    for match in TAG_RE.finditer(raw):
        if match.group(0) == "<nl>":
            result.append(width)
            width = 0
        else:
            width += 1
    if width:
        result.append(width)
    return result


def candidate_rows(
    table: TableStructure,
    raw: str,
    *,
    long_text_chars: int = 80,
    dense_columns: int = 8,
) -> list[CandidateRows]:
    """Candidate flags express uncertainty, without asserting an error exists."""
    candidates = []
    top = [cell for cell in table.cells if cell.row < 2]
    if table.rows >= 2 and any(cell.colspan > 1 or cell.rowspan > 1 for cell in top):
        candidates.append(CandidateRows(0, min(3, table.rows), ("multi_level_header",), 90))

    if table.columns >= dense_columns:
        for start in range(0, table.rows, 2):
            candidates.append(CandidateRows(start, min(start + 2, table.rows), ("dense_cells",), 50))

    for cell in table.cells:
        if len(cell.text) >= long_text_chars:
            candidates.append(CandidateRows(cell.row, cell.row + cell.rowspan, ("long_text",), 80))

    merged = sorted({cell.row for cell in table.cells if cell.rowspan > 1 or cell.colspan > 1})
    for row in merged:
        if row + 1 in merged:
            candidates.append(CandidateRows(row, min(row + 2, table.rows), ("consecutive_merges",), 85))

    widths = raw_row_widths(raw)
    for row, width in enumerate(widths):
        if row < table.rows and width != table.columns:
            candidates.append(CandidateRows(row, row + 1, ("ragged_rows",), 100))
    return candidates
