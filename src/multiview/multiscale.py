"""Select agreement across independent resolution views without labels."""

from typing import Any

from PIL import Image

from xiaomi_runtime.adapter import MethodResult, Prediction, Runtime
from xiaomi_runtime.otsl import parse_table
from xiaomi_runtime.processor import ProcessorLimits

from .resolution_policy import ResolutionPolicy


class AdaptiveMultiScaleInference:
    def __init__(self, runtime: Runtime, config: dict[str, Any] | None = None):
        self.runtime = runtime
        self.policy = ResolutionPolicy(**(config or {}))
        self.limits = ProcessorLimits()

    def run_view(self, table_id: str, image: Image.Image, view_type: str) -> Prediction:
        if view_type == "global":
            resolution = None
        elif view_type in ("medium", "high"):
            resolution = self.policy.views(image.size, self.limits)[view_type]
        else:
            raise ValueError(f"Unknown image view {view_type}")
        return self.runtime.infer(
            table_id, image, method="A", view_type=view_type, resolution=resolution
        )

    def run(self, table_id: str, image: Image.Image,
            baseline: Prediction | None = None) -> MethodResult:
        calls: list[Prediction] = []
        if baseline is None:
            baseline = self.run_view(table_id, image, "global")
            calls.append(baseline)
        if baseline.table_id != table_id:
            raise ValueError("Global prediction belongs to a different table")
        candidates = []
        views = self.policy.views(image.size, self.limits)
        native = self.limits.native_size(image.size)
        used_sizes = {native}
        skipped = []
        for name, size in views.items():
            if size in used_sizes:
                skipped.append(name)
                continue
            candidate = self.run_view(table_id, image, name)
            calls.append(candidate)
            candidates.append(candidate)
            used_sizes.add(size)
        usable = []
        for candidate in candidates:
            if candidate.truncated or candidate.error or not candidate.eos:
                continue
            try:
                parse_table(candidate.raw)
            except ValueError:
                continue
            usable.append(candidate)
        diagnostics = {"views": views, "skipped_duplicate_sizes": skipped,
                       "selection": "two distinct enhanced views must agree exactly after official serialization"}
        if len(usable) == 2 and usable[0].html == usable[1].html:
            selected = usable[-1]
            return MethodResult(selected, calls, selected.raw != baseline.raw, "enhanced_views_agree", diagnostics)
        return MethodResult(baseline, calls, False, "retain_global_on_disagreement", diagnostics)
