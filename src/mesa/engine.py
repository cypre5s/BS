"""One resident runtime, immutable per-view cache and bounded retry accounting."""
from __future__ import annotations

import fcntl
from pathlib import Path
from time import perf_counter

from orchestration.storage import atomic_json, file_hash, object_hash, read_json
from samr.development import pixel_hash
from xiaomi_runtime.adapter import MODEL_SHA256, Prediction, text_hash
from xiaomi_runtime.otsl import to_html
from xiaomi_runtime.processor import TABLE_PROMPT, ProcessorLimits

from .budget import BudgetManager
from .canonical_grid import CanonicalGrid
from .config import MESAConfig
from .evidence_aggregator import EvidenceAggregator
from .views import VisualViewGenerator


class MESAInferenceEngine:
    def __init__(self, runtime, config: MESAConfig, output: Path):
        self.runtime, self.config, self.output = runtime, config, output
        self.generator = VisualViewGenerator(config)
        self.aggregator = EvidenceAggregator()

    def run(self, table_id, image, source_image_id=None):
        directory = self.output / text_hash(table_id)
        directory.mkdir(parents=True, exist_ok=True)
        with (directory / ".lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            return self._run(directory, table_id, image, source_image_id or table_id)

    def _run(self, directory, table_id, image, source_image_id):
        config = self.config
        source_hashes = {p.name: file_hash(p) for p in Path(__file__).parent.glob("*.py")}
        identity = {"table_id": table_id, "source_image_id": source_image_id, "pixels": pixel_hash(image),
                    "config": config.as_dict(), "implementation": source_hashes}
        identity_hash = object_hash(identity)
        manifest = directory / "manifest.json"
        if manifest.exists():
            if read_json(manifest)["identity_hash"] != identity_hash:
                raise ValueError("Existing experiment identity differs; use a new output directory")
        else:
            atomic_json(manifest, {"identity_hash": identity_hash, "identity": identity}, immutable=True)
        final = directory / "result.json"
        if final.exists():
            result = read_json(final)
            if result["identity_hash"] != identity_hash:
                raise ValueError("Final result identity differs")
            for rel, digest in result["cache_files"].items():
                if file_hash(directory / rel) != digest:
                    raise ValueError("Cached view changed")
            return result
        state_path = directory / "budget.json"
        state = read_json(state_path) if state_path.exists() else {"budget": BudgetManager().as_dict(), "in_flight": False}
        budget = BudgetManager(**state["budget"])
        if state["in_flight"]:
            # A killed call has unknown duration; reserve the whole remaining
            # latency instead of silently giving it another fresh budget.
            budget.elapsed = max(budget.elapsed, config.max_latency_per_table)
        observations, records, hashes = [], [], {}
        baseline = None
        for view in self.generator.generate(image, source_image_id):
            path = directory / "views" / (view.view_id + ".json")
            prediction = None
            if path.exists():
                saved = read_json(path)
                if saved["identity_hash"] != identity_hash or saved["view"] != read_jsonable(view.as_dict()):
                    raise ValueError("View cache identity mismatch")
                prediction = Prediction(**saved["prediction"])
                from dataclasses import replace
                prediction = replace(prediction, input_size=tuple(prediction.input_size),
                                     processed_size=tuple(prediction.processed_size))
                self._validate(prediction, view, table_id)
                records.append({"view_id": view.view_id, "channel_id": view.channel_id, "status": "cached"})
            else:
                for attempt in range(config.max_attempts_per_view):
                    reason = budget.rejection(view, config)
                    if reason:
                        records.append({"view_id": view.view_id, "channel_id": view.channel_id,
                                        "status": "skipped", "reason": reason})
                        break
                    budget.reserve(view, config)
                    atomic_json(state_path, {"budget": budget.as_dict(), "in_flight": True, "view_id": view.view_id})
                    started = perf_counter()
                    deadline = started + max(0, config.max_latency_per_table - budget.elapsed)
                    error = None
                    try:
                        prediction = self.runtime.infer(table_id, view.prepare(image), method="MESA",
                                                        view_type=view.view_id,
                                                        max_visual_tokens=view.plan.estimated_visual_tokens,
                                                        deadline=deadline)
                        self._validate(prediction, view, table_id)
                        if perf_counter() >= deadline or prediction.error or not prediction.eos or prediction.truncated:
                            raise ValueError("incomplete_or_late_prediction_discarded")
                    except Exception as failure:  # noqa: BLE001 - failed auxiliaries are audited and isolated
                        prediction, error = None, f"{type(failure).__name__}: {failure}"
                    finally:
                        budget.elapsed += perf_counter() - started
                        atomic_json(state_path, {"budget": budget.as_dict(), "in_flight": False})
                    event = {"view_id": view.view_id, "channel_id": view.channel_id,
                             "source_image_id": source_image_id, "table_id": table_id,
                             "status": "completed" if prediction else "failed", "error": error,
                             "attempt": budget.views, "elapsed": budget.elapsed}
                    records.append(event)
                    atomic_json(directory / "attempts" / f"{budget.views:04d}.json", event, immutable=True)
                    if prediction:
                        from dataclasses import asdict
                        grid = CanonicalGrid.from_otsl(prediction.raw, view.channel_id, view.view_id)
                        atomic_json(path, {"identity_hash": identity_hash, "view": view.as_dict(),
                                          "prediction": asdict(prediction), "canonical": grid.as_dict()}, immutable=True)
                        break
            if prediction:
                hashes[str(path.relative_to(directory))] = file_hash(path)
                if view.channel_id == "global":
                    baseline = prediction
                else:
                    observations.append((view, prediction))
            if view.channel_id == "global" and baseline is None:
                atomic_json(directory / "failure.json", {"reason": "global_unavailable_no_fabricated_B0",
                                                          "records": records, "budget": budget.as_dict()})
                raise RuntimeError("Global Original unavailable; no B0 fallback can be invented")
        if baseline is None:
            raise RuntimeError("Missing global anchor")
        fusion = {"raw": baseline.raw, "changed": False, "reason": "aggregation_disabled"}
        if config.fusion:
            if budget.elapsed >= config.max_latency_per_table:
                fusion["reason"] = "latency_exhausted_retain_global"
            else:
                started = perf_counter()
                try:
                    fusion = self.aggregator.aggregate(baseline.raw, observations, image, config)
                except Exception as failure:  # noqa: BLE001 - explicit aggregation fallback
                    fusion = {"raw": baseline.raw, "changed": False, "reason": "fusion_failed_retain_global",
                              "error": f"{type(failure).__name__}: {failure}"}
                budget.elapsed += perf_counter() - started
                if budget.elapsed > config.max_latency_per_table:
                    fusion.update(raw=baseline.raw, changed=False, reason="late_fusion_discarded")
        result = {"table_id": table_id, "identity_hash": identity_hash, "raw": fusion["raw"],
                  "html": to_html(fusion["raw"]), "H0": baseline.raw, "fusion": fusion,
                  "events": records, "budget": budget.as_dict(), "cache_files": hashes,
                  "model_load_count": getattr(self.runtime, "load_count", None),
                  "hard_wall_clock_guaranteed": False, "formal_evaluation": False}
        atomic_json(final, result, immutable=True)
        return result

    def _validate(self, prediction, view, table_id):
        view.plan.validate_prediction(prediction, ProcessorLimits())
        if (prediction.table_id != table_id or prediction.model_hash != MODEL_SHA256
                or prediction.prompt_hash != text_hash(TABLE_PROMPT)
                or prediction.output_hash != text_hash(prediction.raw) or prediction.html != to_html(prediction.raw)):
            raise ValueError("Runtime contract mismatch")


def read_jsonable(value):
    import json
    return json.loads(json.dumps(value))
