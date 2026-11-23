"""Conservative content reconciliation after an explicit local-to-global mapping."""

from __future__ import annotations

import html
import re
from dataclasses import dataclass

from lxml import etree
from lxml import html as lxml_html

from xiaomi_runtime.otsl import TAG_RE, TableStructure, parse_table

from .region_selector import SelectedRegion
from .uncertainty import raw_row_widths


@dataclass(frozen=True)
class Reconciliation:
    raw: str
    accepted: bool
    reason: str
    mapping: tuple[dict, ...] = ()
    changes: tuple[dict, ...] = ()


def _plain(text: str) -> str:
    try:
        return lxml_html.fragment_fromstring(text, create_parent="div").text_content()
    except (ValueError, etree.ParserError):
        return html.unescape(text)


def _numbers(text: str) -> tuple[str, ...]:
    return tuple(re.findall(r"[+-]?\d[\d,.]*(?:[%‰])?", _plain(text)))


def _source_spans(raw: str) -> dict[tuple[int, int], tuple[int, int]]:
    """Locate existing content spans with the official token expression."""
    matches = list(TAG_RE.finditer(raw))
    positions = {}
    row = column = 0
    for index, match in enumerate(matches):
        tag = match.group(0)
        if tag == "<nl>":
            row += 1
            column = 0
            continue
        if tag == "<fcel>":
            end = matches[index + 1].start() if index + 1 < len(matches) else len(raw)
            positions[row, column] = match.end(), end
        column += 1
    return positions


def reconcile(
    global_raw: str,
    local_raw: str,
    region: SelectedRegion,
    *,
    minimum_anchor_fraction: float = 0.75,
    maximum_changed_fraction: float = 0.25,
    minimum_distinct_anchors: int = 2,
) -> Reconciliation:
    try:
        global_table = parse_table(global_raw)
        local_table = parse_table(local_raw)
    except (ValueError, TypeError) as error:
        return Reconciliation(global_raw, False, f"parse_failed_{type(error).__name__}")
    if any(width != global_table.columns for width in raw_row_widths(global_raw)):
        return Reconciliation(global_raw, False, "global_raw_requires_structural_repair")
    if any(width != local_table.columns for width in raw_row_widths(local_raw)):
        return Reconciliation(global_raw, False, "local_raw_requires_structural_repair")
    if not _valid_cell_layout(global_table) or not _valid_cell_layout(local_table):
        return Reconciliation(global_raw, False, "cell_ownership_is_ambiguous")
    shape = (region.row_stop - region.row_start, region.column_stop - region.column_start)
    if (local_table.rows, local_table.columns) != shape:
        return Reconciliation(global_raw, False, "local_shape_does_not_match_crop_mapping")

    selected = {}
    for cell in global_table.cells:
        touches = (cell.row < region.row_stop and cell.row + cell.rowspan > region.row_start
                   and cell.column < region.column_stop and cell.column + cell.colspan > region.column_start)
        if not touches:
            continue
        contained = (region.row_start <= cell.row and cell.row + cell.rowspan <= region.row_stop
                     and region.column_start <= cell.column
                     and cell.column + cell.colspan <= region.column_stop)
        if not contained:
            return Reconciliation(global_raw, False, "merged_cell_crosses_crop_boundary")
        selected[cell.row - region.row_start, cell.column - region.column_start] = cell
    local_cells = {(cell.row, cell.column): cell for cell in local_table.cells}
    if selected.keys() != local_cells.keys():
        return Reconciliation(global_raw, False, "local_origins_do_not_match_crop_mapping")
    if any((cell.rowspan, cell.colspan) != (local_cells[key].rowspan, local_cells[key].colspan)
           for key, cell in selected.items()):
        return Reconciliation(global_raw, False, "local_spans_do_not_match_crop_mapping")

    mapping = tuple({"local_row": key[0], "local_column": key[1],
                     "global_row": cell.row, "global_column": cell.column,
                     "rowspan": cell.rowspan, "colspan": cell.colspan}
                    for key, cell in sorted(selected.items()))
    changed = [(cell, local_cells[key]) for key, cell in selected.items()
               if cell.text != local_cells[key].text]
    if not changed:
        return Reconciliation(global_raw, False, "local_agrees_with_global", mapping)

    nonempty = [cell for cell in selected.values() if _plain(cell.text).strip()]
    anchors = [cell.text for key, cell in selected.items()
               if _plain(cell.text).strip() and cell.text == local_cells[key].text]
    if (len(set(anchors)) < minimum_distinct_anchors
            or len(anchors) / max(1, len(nonempty)) < minimum_anchor_fraction):
        return Reconciliation(global_raw, False, "insufficient_unchanged_cell_anchors", mapping)
    if len(changed) / max(1, len(selected)) > maximum_changed_fraction:
        return Reconciliation(global_raw, False, "too_many_proposed_cell_changes", mapping)
    if any(_numbers(before.text) != _numbers(after.text) for before, after in changed):
        return Reconciliation(global_raw, False, "numeric_change_requires_more_evidence", mapping)

    spans = _source_spans(global_raw)
    replacements = []
    changes = []
    for before, after in changed:
        position = before.row, before.column
        if position not in spans or not before.text.strip() or not after.text.strip():
            return Reconciliation(global_raw, False, "empty_cell_change_requires_more_evidence", mapping)
        replacements.append((*spans[position], after.text))
        changes.append({"row": before.row, "column": before.column,
                        "before": before.text, "after": after.text})
    result = global_raw
    for start, stop, text in sorted(replacements, reverse=True):
        result = result[:start] + text + result[stop:]
    verified = parse_table(result)
    if _shape_signature(verified) != _shape_signature(global_table):
        return Reconciliation(global_raw, False, "reconciled_structure_changed", mapping)
    return Reconciliation(result, True, "mapped_cells_with_unchanged_anchors", mapping, tuple(changes))


def _shape_signature(table: TableStructure) -> tuple:
    return (table.rows, table.columns,
            tuple((cell.row, cell.column, cell.rowspan, cell.colspan) for cell in table.cells))


def _valid_cell_layout(table: TableStructure) -> bool:
    occupied: set[tuple[int, int]] = set()
    for cell in table.cells:
        area = {(row, column) for row in range(cell.row, cell.row + cell.rowspan)
                for column in range(cell.column, cell.column + cell.colspan)}
        if occupied & area or any(row >= table.rows or column >= table.columns for row, column in area):
            return False
        occupied.update(area)
    return len(occupied) == table.rows * table.columns
