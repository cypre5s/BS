"""Separate structure/text/spatial evidence; never select a whole local output."""
import re
from dataclasses import replace
from difflib import SequenceMatcher

from reacquisition.region_selector import detect_grid
from samr.canonical import CanonicalCell, CanonicalTableGrid
from samr.fusion import fuse_global_anchor
from xiaomi_runtime.otsl import parse_table

from .canonical_grid import CanonicalGrid, map_view


def text_evidence(before, after):
    numbers = r"[+-]?\d[\d,.]*(?:[%‰])?"
    return {"equal": before == after, "numeric_equal": re.findall(numbers, before) == re.findall(numbers, after),
            "whitespace_only": before != after and "".join(before.split()) == "".join(after.split()),
            "character_difference": [op for op in SequenceMatcher(None, before, after, autojunk=False).get_opcodes()
                                     if op[0] != "equal"], "decision": "retain_global_text"}


class EvidenceAggregator:
    def aggregate(self, anchor_raw, observations, image, config):
        anchor = CanonicalGrid.from_otsl(anchor_raw, "global", "global_0")
        if anchor.native.status != "RESOLVED":
            return {"raw": anchor_raw, "changed": False, "reason": "unresolved_global", "evidence": []}
        geometry = detect_grid(image, parse_table(anchor_raw))
        helpers: list[CanonicalTableGrid] = []
        evidence, sources = [], []
        for view, prediction in observations:
            helper = CanonicalGrid.from_otsl(prediction.raw, view.channel_id, view.view_id)
            mapping = map_view(anchor, helper, view, geometry, config.boundary_tolerance)
            row = {"channel_id": view.channel_id, "view_id": view.view_id, "spatial": mapping,
                   "structural": {"global_shape": (anchor.native.rows, anchor.native.columns),
                                  "view_shape": (helper.native.rows, helper.native.columns),
                                  "cells": helper.as_dict()["cells"]}, "text": []}
            evidence.append(row)
            if mapping["status"] != "MAPPED":
                continue
            r0, c0 = mapping["offset"]
            h, w = mapping["shape"]
            projected = tuple(CanonicalCell(c.row_start + r0, c.col_start + c0, c.rowspan, c.colspan,
                                           c.cell_type, c.text, float(len(helpers) + 2), 1.0) for c in helper.cells)
            outside = tuple(c for c in anchor.native.cells
                            if not (r0 <= c.row_index < r0 + h and c0 <= c.column_index < c0 + w))
            combined = replace(helper.native, rows=anchor.native.rows, columns=anchor.native.columns,
                               cells=tuple(sorted(outside + projected, key=lambda c: c.origin)),
                               source_scale=float(len(helpers) + 2))
            by_origin = {c.origin: c for c in projected}
            row["structural"]["adjacency"] = [
                {"origin": c.origin, "right": (c.row_index, c.column_index + c.colspan),
                 "below": (c.row_index + c.rowspan, c.column_index)} for c in projected]
            for cell in anchor.native.cells:
                other = by_origin.get(cell.origin)
                if other:
                    row["text"].append({"origin": cell.origin, **text_evidence(cell.text, other.text)})
            helpers.append(combined)
            sources.append({"evidence_id": combined.source_scale, "channel": view.channel_id,
                            "view": view.view_id, "mapping": mapping})
        decision = fuse_global_anchor(anchor_raw, anchor.native, helpers, image)
        # Only image-verified, text-preserving local blank-span repairs from the
        # existing strict fusion primitive can alter H0. Numeric/text differences
        # are diagnostics, not replacement rules. The full local table is never used.
        return {"raw": decision.raw, "changed": decision.changed, "reason": decision.reason,
                "changes": decision.changes, "evidence": evidence, "sources": sources,
                "strict_fusion_audit": decision.as_dict(), "text_policy": "exact_global_text"}
