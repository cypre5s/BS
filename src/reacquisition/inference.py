"""Visual Evidence Reacquisition using one shared, read-only Xiaomi runtime."""

from __future__ import annotations

from dataclasses import asdict, replace
from hashlib import sha256
from typing import Any

from PIL import Image

from xiaomi_runtime.adapter import MethodResult, Prediction
from xiaomi_runtime.otsl import parse_table, to_html

from .reconciliation import reconcile
from .region_selector import detect_grid, select_regions
from .uncertainty import candidate_rows

DEFAULTS = {
    "max_regions": 2,
    "padding": 4,
    "scale": 1.75,
    "boundary_margin": 12,
    "long_text_chars": 80,
    "dense_columns": 8,
    "minimum_anchor_fraction": 0.75,
    "maximum_changed_fraction": 0.25,
    "minimum_distinct_anchors": 2,
}


class VisualEvidenceReacquisition:
    def __init__(self, runtime, config: dict | None = None):
        self.runtime = runtime
        self.config = {**DEFAULTS, **(config or {})}
        if self.config["max_regions"] < 1 or self.config["scale"] <= 1:
            raise ValueError("Reacquisition needs a positive region budget and an enlarged local view")

    def run(self, table_id: str, image: Image.Image, baseline: Prediction) -> MethodResult:
        if baseline.table_id != table_id:
            raise ValueError("Baseline prediction belongs to a different table")
        try:
            table = parse_table(baseline.raw)
        except (ValueError, TypeError) as error:
            return MethodResult(baseline, [], False, "baseline_parse_failed", {"error": str(error)})
        geometry = detect_grid(image, table)
        candidates = candidate_rows(table, baseline.raw, **{
            name: self.config[name] for name in ("long_text_chars", "dense_columns")
        })
        regions = select_regions(candidates, table, geometry, image.size, **{
            name: self.config[name] for name in ("max_regions", "padding", "scale", "boundary_margin")
        })
        diagnostics: dict[str, Any] = {
            "geometry": asdict(geometry), "candidates": [asdict(row) for row in candidates],
            "regions": [], "accepted_regions": 0,
        }
        if baseline.error or baseline.truncated or not baseline.eos:
            return MethodResult(baseline, [], False, "baseline_generation_incomplete", diagnostics)
        if not regions:
            return MethodResult(baseline, [], False, "no_unambiguous_local_region", diagnostics)

        calls = []
        current_raw = baseline.raw
        for index, region in enumerate(regions):
            crop = image.crop(region.transform.crop_box)
            local = self.runtime.infer(table_id, crop, method="B", view_type=f"local_{index}",
                                       resolution=region.transform.resized_size)
            calls.append(local)
            transform = replace(region.transform, resized_size=local.processed_size)
            entry = {"rows": [region.row_start, region.row_stop],
                     "columns": [region.column_start, region.column_stop],
                     "reasons": list(region.reasons), "transform": transform.as_dict()}
            diagnostics["regions"].append(entry)
            if local.table_id != table_id or local.prompt_hash != baseline.prompt_hash or local.model_hash != baseline.model_hash:
                entry["decision"] = "runtime_provenance_mismatch"
                continue
            if local.error or local.truncated or not local.eos:
                entry["decision"] = "local_generation_incomplete"
                continue
            result = reconcile(current_raw, local.raw, region, **{
                name: self.config[name] for name in (
                    "minimum_anchor_fraction", "maximum_changed_fraction", "minimum_distinct_anchors",
                )
            })
            entry.update(decision=result.reason, mapping=list(result.mapping), changes=list(result.changes))
            if result.accepted:
                current_raw = result.raw
                diagnostics["accepted_regions"] += 1

        if current_raw == baseline.raw:
            return MethodResult(baseline, calls, False, "local_evidence_kept_global", diagnostics)
        final = replace(baseline, method="B", view_type="reconciled", raw=current_raw,
                        html=to_html(current_raw), output_hash=sha256(current_raw.encode()).hexdigest(),
                        output_tokens=self.runtime.count_tokens(current_raw),
                        latency=sum(call.latency for call in calls))
        return MethodResult(final, calls, True, "mapped_local_content_accepted", diagnostics)
