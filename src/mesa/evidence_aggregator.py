"""Separate structure/text/spatial evidence; never select a whole local output."""
import re
from dataclasses import replace
from difflib import SequenceMatcher

from samr.canonical import CanonicalCell, CanonicalTableGrid
from samr.fusion import fuse_global_anchor

from .canonical_grid import CanonicalGrid, map_view
from .grid_detector import GridBoundaryDetector


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
        detection = GridBoundaryDetector(config).detect(image)
        geometry = detection.geometry(anchor.native.rows, anchor.native.columns)
        helpers: list[CanonicalTableGrid] = []
        evidence, sources = [], []
        for view, prediction in observations:
            helper = CanonicalGrid.from_otsl(prediction.raw, view.channel_id, view.view_id)
            mapping = map_view(anchor, helper, view, geometry, config.boundary_tolerance)
            row = {"channel_id": view.channel_id, "view_id": view.view_id, "spatial": mapping,
                   "table_id": prediction.table_id, "channel": view.channel_id, "crop_id": view.view_id,
                   "original_bbox": view.requested_box, "snapped_bbox": view.crop_box,
                   "mapping_quality": view.mapping_quality, "mapped_global_cells": mapping.get("cells", []),
                   "accepted": False, "rejection_reason": mapping["reason"],
                   "structural": {"global_shape": (anchor.native.rows, anchor.native.columns),
                                  "view_shape": (helper.native.rows, helper.native.columns),
                                  "cells": helper.as_dict()["cells"]}, "text": []}
            evidence.append(row)
            if mapping["status"] != "MAPPED":
                continue
            rows, columns = mapping["row_indices"], mapping["column_indices"]
            positions = {(r,c) for r in rows for c in columns}
            projected = tuple(CanonicalCell(rows[c.row_start], columns[c.col_start], c.rowspan, c.colspan,
                                           c.cell_type, c.text, float(len(helpers) + 2), 1.0) for c in helper.cells)
            outside = tuple(c for c in anchor.native.cells
                            if not c.positions & positions)
            combined = replace(helper.native, rows=anchor.native.rows, columns=anchor.native.columns,
                               cells=tuple(sorted(outside + projected, key=lambda c: c.origin)),
                               source_scale=float(len(helpers) + 2))
            by_origin = {c.origin: c for c in projected}
            local_anchors = [c.text for c in anchor.native.cells if c.origin in by_origin and c.text
                             and c.topology == by_origin[c.origin].topology and c.text == by_origin[c.origin].text]
            row["local_exact_text_anchors"] = len(set(local_anchors))
            row["structural"]["adjacency"] = [
                {"origin": c.origin, "right": (c.row_index, c.column_index + c.colspan),
                 "below": (c.row_index + c.rowspan, c.column_index)} for c in projected]
            for cell in anchor.native.cells:
                other = by_origin.get(cell.origin)
                if other:
                    row["text"].append({"origin": cell.origin, **text_evidence(cell.text, other.text)})
            if view.local and len(set(local_anchors)) < 2:
                row["rejection_reason"] = "insufficient_actual_local_text_anchors"
                continue
            helpers.append(combined)
            sources.append({"evidence_id": combined.source_scale, "channel": view.channel_id,
                            "view": view.view_id, "mapping": mapping, "positions": sorted(positions)})
        # Shared pixels are corroboration checks, never extra votes. Disagreeing
        # local cell ownership/text at the same mapped position vetoes both helpers.
        rejected: set[float] = set()
        for i,left in enumerate(sources):
            if left["channel"] in ("global","highres"):
                continue
            for j,right in enumerate(sources[:i]):
                if right["channel"] in ("global","highres"):
                    continue
                overlap = set(map(tuple,left["positions"])) & set(map(tuple,right["positions"]))
                lo,ro = helpers[i].owners(),helpers[j].owners()
                conflicts = [p for p in overlap if (lo[p].topology,lo[p].text)!=(ro[p].topology,ro[p].text)]
                for row in evidence:
                    if row["view_id"] in (left["view"],right["view"]):
                        row.setdefault("overlap_checks",[]).append({"views":(left["view"],right["view"]),
                            "shared_positions":len(overlap),"conflicting_positions":sorted(conflicts)})
                if conflicts:
                    rejected.update((left["evidence_id"],right["evidence_id"]))
        helpers = [h for h in helpers if h.source_scale not in rejected]
        decision = fuse_global_anchor(anchor_raw, anchor.native, helpers, image)
        accepted_sources = {scale for change in decision.changes if change.get("accepted")
                            for scale in change.get("supporting_scales", [])} if decision.changed else set()
        for row in evidence:
            sources_for_view = [s["evidence_id"] for s in sources if s["view"] == row["view_id"]]
            row["accepted"] = any(s in accepted_sources for s in sources_for_view)
            if row["spatial"]["status"] == "MAPPED":
                if any(s in rejected for s in sources_for_view):
                    row["rejection_reason"] = "overlap_evidence_conflict"
                elif sources_for_view:
                    row["rejection_reason"] = None if row["accepted"] else decision.reason
        # Only image-verified, text-preserving local blank-span repairs from the
        # existing strict fusion primitive can alter H0. Numeric/text differences
        # are diagnostics, not replacement rules. The full local table is never used.
        return {"raw": decision.raw, "changed": decision.changed, "reason": decision.reason,
                "changes": decision.changes, "evidence": evidence, "sources": sources,
                "strict_fusion_audit": decision.as_dict(), "text_policy": "exact_global_text"}
