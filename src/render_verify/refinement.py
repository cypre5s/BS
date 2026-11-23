"""Verify renders and repair only blank merges contradicted by source lines."""

from __future__ import annotations

import hashlib
from dataclasses import replace
from time import perf_counter

from xiaomi_runtime.adapter import MethodResult, Prediction
from xiaomi_runtime.otsl import EMPTY, ROW_BREAK, pad_rows, parse_grid, to_html

from .html_renderer import HTMLRenderer, RenderError
from .structural_checker import StructureError, inspect_html, split_blank_spans
from .visual_checker import verify


def _split_verified_blanks(raw: str, coordinates: set[tuple[int, int]]) -> str:
    """Edit official parsed grid tags only at the verified blank-cell footprint."""
    structure = inspect_html(to_html(raw))
    tags, texts = parse_grid(pad_rows(raw))
    by_position = {(cell.row, cell.column): cell for cell in structure.cells}
    for position in coordinates:
        cell = by_position[position]
        if cell.text.strip():
            raise StructureError("Blank-span repair encountered cell text")
        for row in range(cell.row, cell.row + cell.rowspan):
            for column in range(cell.column, cell.column + cell.colspan):
                tags[row][column] = EMPTY
                texts[row][column] = ""
    return "".join(
        "".join(tag + text for tag, text in zip(row, content, strict=True)) + ROW_BREAK
        for row, content in zip(tags, texts, strict=True)
    )


class RenderGuidedVerification:
    """Method C uses the global prediction plus deterministic source/render geometry."""

    def __init__(self, runtime, config: dict | None = None):
        self.runtime = runtime
        self.config = config or {}
        self.renderer = HTMLRenderer(
            font_paths=self.config.get("font_paths"),
            font_size=int(self.config.get("font_size", 14)),
        )

    def run(self, table_id: str, image, baseline: Prediction | None = None) -> MethodResult:
        started = perf_counter()
        calls = []
        if baseline is None:
            baseline = self.runtime.infer(table_id, image, method="C", view_type="global")
            calls.append(baseline)
        if baseline.table_id != table_id:
            raise ValueError("Baseline table ID does not match the input")
        diagnostics: dict = {"verification": "pending", "refinement_attempted": False}
        try:
            original_html = to_html(baseline.raw)
            if original_html != baseline.html:
                raise RenderError("Baseline HTML differs from the shared official serializer")
            first_render = self.renderer.render(original_html)
            settings = {
                "minimum_line_fraction": float(self.config.get("minimum_line_fraction", 0.65)),
                "minimum_line_support": float(self.config.get("minimum_line_support", 0.95)),
                "maximum_blank_ink_fraction": float(self.config.get("maximum_blank_ink_fraction", 0.002)),
            }
            first = verify(image, first_render, **settings)
            diagnostics["first_verification"] = first.as_dict()
            diagnostics["render_size"] = list(first_render.image.size)
            if not self.config.get("refine_blank_spans", True) or not first.blank_spans_to_split:
                diagnostics["verification"] = first.reason
                diagnostics["seconds"] = perf_counter() - started
                return MethodResult(baseline, calls, False, first.reason, diagnostics)
            diagnostics["refinement_attempted"] = True
            positions = set(first.blank_spans_to_split)
            candidate_raw = _split_verified_blanks(baseline.raw, positions)
            candidate_html = to_html(candidate_raw)
            expected = split_blank_spans(first_render.structure, positions)
            candidate_structure = inspect_html(candidate_html)
            if candidate_structure.signature != expected.signature:
                raise StructureError("Repair altered cells outside the verified blank spans")
            if candidate_structure.content != first_render.structure.content:
                raise StructureError("Repair altered nonblank text")
            second_render = self.renderer.render(candidate_html)
            second = verify(image, second_render, **settings)
            diagnostics["second_verification"] = second.as_dict()
            accepted = (
                second.confirmed_grid
                and second.unsupported_boundaries < first.unsupported_boundaries
                and not second.blank_spans_to_split
            )
            diagnostics["seconds"] = perf_counter() - started
            if not accepted:
                diagnostics["verification"] = "repair_failed_second_verification"
                return MethodResult(baseline, calls, False, "repair_failed_second_verification", diagnostics)
            diagnostics["verification"] = "confirmed_blank_span_repair"
            candidate = replace(
                baseline,
                raw=candidate_raw,
                html=candidate_html,
                method="C",
                view_type="render_verified_blank_span_repair",
                output_tokens=self.runtime.count_tokens(candidate_raw),
                output_hash=hashlib.sha256(candidate_raw.encode("utf-8")).hexdigest(),
            )
            return MethodResult(candidate, calls, True, "confirmed_blank_span_repair", diagnostics)
        except (RenderError, StructureError, ValueError) as error:
            diagnostics.update(verification="unconfirmed", error=str(error), seconds=perf_counter() - started)
            return MethodResult(baseline, calls, False, "verification_unavailable", diagnostics)
