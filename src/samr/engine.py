"""One shared runtime, explicit budgets and immutable per-table SAMR receipts."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from pathlib import Path
from time import perf_counter
from typing import Protocol

from PIL import Image

from orchestration.storage import atomic_json
from xiaomi_runtime.adapter import Prediction, text_hash
from xiaomi_runtime.budget import BudgetExceeded
from xiaomi_runtime.otsl import to_html

from .canonical import CanonicalTableGrid
from .complexity import VisualComplexityEstimator
from .config import IDENTITY, SAMRConfig
from .fusion import FusionDecision, fuse_global_anchor
from .routing import BudgetLedger, RoutingDecision, route
from .scales import ScaleLadder, ScalePlan


class BudgetRuntime(Protocol):
    def infer(self, table_id: str, image: Image.Image, *, method: str, view_type: str,
              max_visual_tokens: int, deadline: float) -> Prediction: ...

    def count_tokens(self, raw: str) -> int: ...


@dataclass(frozen=True)
class SAMRResult:
    prediction: Prediction
    audit: dict


def _scale_record(plan: ScalePlan, prediction: Prediction, grid: CanonicalTableGrid) -> dict:
    return {"requested_scale": plan.requested_scale, "applied_scale": plan.applied_scale,
            "original_width": plan.original_size[0], "original_height": plan.original_size[1],
            "actual_width": plan.actual_size[0], "actual_height": plan.actual_size[1],
            "processed_width": prediction.processed_size[0], "processed_height": prediction.processed_size[1],
            "image_grid_thw": prediction.image_grid_thw, "visual_token_count": prediction.visual_tokens,
            "estimated_visual_tokens": plan.estimated_visual_tokens,
            "inference_latency": prediction.latency, "output_length": len(prediction.raw),
            "output_tokens": prediction.output_tokens, "otsl_validity": grid.otsl_valid,
            "html_parse_status": grid.html_parse_status, "canonical_grid": grid.as_dict(),
            "prediction": asdict(prediction), "raw_otsl": prediction.raw, "runtime_contract_valid": False}


class SAMRInferenceEngine:
    def __init__(self, runtime: BudgetRuntime, config: SAMRConfig | None = None):
        self.runtime = runtime
        self.config = config or SAMRConfig()
        self.ladder = ScaleLadder()
        self.estimator = VisualComplexityEstimator()

    def _infer(self, table_id: str, image: Image.Image, plan: ScalePlan, budget: BudgetLedger) -> Prediction:
        remaining = budget.remaining_tokens
        budget.reserve(plan)
        prediction = self.runtime.infer(table_id, plan.prepare(image), method="SAMR",
                                        view_type=f"scale_{plan.requested_scale:g}",
                                        max_visual_tokens=min(remaining, plan.estimated_visual_tokens),
                                        deadline=budget.deadline)
        return prediction

    def _validate(self, table_id: str, plan: ScalePlan, prediction: Prediction) -> None:
        plan.validate_prediction(prediction, self.ladder.limits)
        if (prediction.table_id != table_id or prediction.prompt_hash != IDENTITY["prompt_hash"]
                or prediction.model_hash != IDENTITY["model_hash"] or to_html(prediction.raw) != prediction.html):
            raise ValueError("Runtime identity or official HTML contract differs")

    def run(self, table_id: str, image: Image.Image, *, audit_directory: Path | None = None) -> SAMRResult:
        if not table_id:
            raise ValueError("table_id must be nonempty")
        receipt = (audit_directory / f"{text_hash(table_id)}.json") if audit_directory else None
        if receipt is not None and receipt.exists():
            raise FileExistsError("A SAMR table receipt already exists; use a new development output directory")
        budget = BudgetLedger(self.config)
        first = self.ladder.plan(image.size, self.config.scales[0])
        reason = budget.admit(first)
        if reason != "admitted":
            raise BudgetExceeded(f"Required first scale cannot fit: {reason}")
        # A first-call failure has no valid B0 to fall back to and must propagate honestly.
        baseline = self._infer(table_id, image, first, budget)
        self._validate(table_id, first, baseline)
        anchor = CanonicalTableGrid.from_otsl(baseline.raw, first.requested_scale)
        records = [_scale_record(first, baseline, anchor)]
        records[0]["runtime_contract_valid"] = True
        features = self.estimator.estimate(image, visual_token_count=baseline.visual_tokens)
        routing: RoutingDecision = route(self.config, features)
        routing.events.append({"scale": first.requested_scale, "decision": "executed_required_first_scale"})
        helpers = []
        anchor_usable = baseline.eos and not baseline.truncated and not baseline.error and anchor.status == "RESOLVED"
        for scale in routing.desired_scales[1:]:
            plan = self.ladder.plan(image.size, scale)
            reason = budget.admit(plan)
            if not anchor_usable and self.config.mode == "adaptive":
                reason = "anchor_unreliable_retain_B0"
            if reason != "admitted":
                routing.events.append({"scale": scale, "decision": "skipped", "reason": reason,
                                       "plan": asdict(plan)})
                continue
            try:
                prediction = self._infer(table_id, image, plan, budget)
                grid = CanonicalTableGrid.from_otsl(prediction.raw, scale)
                records.append(_scale_record(plan, prediction, grid))
                self._validate(table_id, plan, prediction)
                records[-1]["runtime_contract_valid"] = True
                usable = prediction.eos and not prediction.truncated and not prediction.error
                if usable:
                    helpers.append(grid)
                routing.events.append({"scale": scale, "decision": "executed", "usable": bool(usable),
                                       "reason": grid.reason if usable else "incomplete_or_timed_out_output"})
            except Exception as error:  # noqa: BLE001 - explicit auxiliary failure isolation, audited below
                # A failed auxiliary cannot replace the already saved native result.
                routing.events.append({"scale": scale, "decision": "failed", "reason": type(error).__name__,
                                       "detail": str(error), "plan": asdict(plan)})
        fusion = FusionDecision(baseline.raw, False, "single_scale_or_fusion_disabled")
        if not anchor_usable:
            fusion = FusionDecision(baseline.raw, False, "first_scale_unreliable_no_fabricated_repair")
        elif perf_counter() >= budget.deadline:
            fusion = FusionDecision(baseline.raw, False, "latency_budget_exhausted_retain_B0")
        elif self.config.mode != "single" and self.config.fusion_policy != "retain_global":
            try:
                fusion = fuse_global_anchor(baseline.raw, anchor, helpers, image)
            except Exception as error:  # noqa: BLE001 - contractual B0 fallback for any fusion failure
                fusion = FusionDecision(baseline.raw, False, f"fusion_exception_{type(error).__name__}_retain_B0")
        # Never accept a fusion that finished after the cooperative deadline.
        if fusion.changed and perf_counter() >= budget.deadline:
            fusion = FusionDecision(baseline.raw, False, "fusion_deadline_exceeded_retain_B0",
                                    fusion.evidence, fusion.changes, fusion.visual_evidence)
        prediction = baseline
        if fusion.changed:
            prediction = replace(baseline, raw=fusion.raw, html=to_html(fusion.raw), view_type="fused",
                                 output_tokens=self.runtime.count_tokens(fusion.raw), output_hash=text_hash(fusion.raw))
        audit = {"schema_version": 1, "table_id": table_id, "config": self.config.as_dict(),
                 "B0_raw_otsl": baseline.raw if first.requested_scale == 1.0 else None,
                 "first_scale_raw_otsl": baseline.raw, "scales": records,
                 "complexity": features.as_dict(), "routing_decision": asdict(routing),
                 "fusion_decision": fusion.as_dict(), "final_otsl": prediction.raw, "final_html": prediction.html,
                 "final_prediction": asdict(prediction), "budget": budget.as_dict(),
                 "inference_input_contract": "image_and_table_id_only_no_GT"}
        if receipt is not None:
            atomic_json(receipt, audit, immutable=True)
        return SAMRResult(prediction, audit)
