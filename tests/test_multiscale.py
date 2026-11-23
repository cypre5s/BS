from dataclasses import replace

import pytest
from PIL import Image

from multiview import AdaptiveMultiScaleInference
from multiview.resolution_policy import ResolutionPolicy
from xiaomi_runtime.adapter import Prediction
from xiaomi_runtime.otsl import to_html
from xiaomi_runtime.processor import ProcessorLimits


class MockRuntime:
    def __init__(self, predictions):
        self.predictions = predictions
        self.requests = []

    def infer(self, table_id, image, *, method, view_type, resolution=None):
        self.requests.append((table_id, image, method, view_type, resolution))
        return self.predictions[view_type]

    def count_tokens(self, raw):
        return len(raw) + 1


def prediction(raw, view="global", **kwargs):
    return Prediction(raw=raw, html=to_html(raw), table_id="fixture", method="A", view_type=view, **kwargs)


def test_two_distinct_enhanced_views_must_agree():
    baseline = prediction("<fcel>A<nl>")
    improved = prediction("<fcel>B<nl>", "medium")
    runtime = MockRuntime({"medium": improved, "high": replace(improved, view_type="high")})
    result = AdaptiveMultiScaleInference(runtime).run("fixture", Image.new("RGB", (256, 256)), baseline)
    assert result.changed
    assert result.prediction.raw == improved.raw
    assert len(result.calls) == 2
    assert [request[3] for request in runtime.requests] == ["medium", "high"]
    assert all(request[0] == "fixture" for request in runtime.requests)


@pytest.mark.parametrize("fault", ["disagreement", "truncated", "no_eos", "invalid"])
def test_ambiguous_candidate_retains_original(fault):
    baseline = prediction("<fcel>A<nl>")
    medium = prediction("<fcel>B<nl>", "medium")
    high = replace(medium, view_type="high")
    if fault == "disagreement":
        high = prediction("<fcel>C<nl>", "high")
    elif fault == "truncated":
        high = replace(high, truncated=True)
    elif fault == "no_eos":
        high = replace(high, eos=False)
    else:
        high = replace(high, raw="invalid")
    runtime = MockRuntime({"medium": medium, "high": high})
    result = AdaptiveMultiScaleInference(runtime).run("fixture", Image.new("RGB", (256, 256)), baseline)
    assert result.prediction is baseline
    assert not result.changed


def test_global_runs_first_without_cache():
    global_prediction = prediction("<fcel>A<nl>")
    runtime = MockRuntime({name: replace(global_prediction, view_type=name) for name in ("global", "medium", "high")})
    result = AdaptiveMultiScaleInference(runtime).run("fixture", Image.new("RGB", (256, 256)))
    assert [request[3] for request in runtime.requests] == ["global", "medium", "high"]
    assert runtime.requests[0][4] is None
    assert len(result.calls) == 3


def test_independent_view_entry_point_and_id_guard():
    runtime = MockRuntime({"high": prediction("<fcel>A<nl>", "high")})
    method = AdaptiveMultiScaleInference(runtime)
    method.run_view("fixture", Image.new("RGB", (256, 256)), "high")
    assert len(runtime.requests) == 1
    with pytest.raises(ValueError, match="different table"):
        method.run("wrong", Image.new("RGB", (256, 256)), prediction("<fcel>A<nl>"))


@pytest.mark.parametrize("size", [(256, 256), (777, 399), (2048, 512), (80, 800), (4096, 4096)])
def test_views_obey_official_grid_budget(size):
    limits = ProcessorLimits()
    policy = ResolutionPolicy()
    native = limits.native_size(size)
    for view in policy.views(size, limits).values():
        assert all(dimension % 32 == 0 for dimension in view)
        assert native[0] * native[1] <= view[0] * view[1] <= limits.max_pixels


def test_skip_duplicate_native_size_at_budget():
    baseline = prediction("<fcel>A<nl>")
    runtime = MockRuntime({})
    result = AdaptiveMultiScaleInference(runtime).run("fixture", Image.new("RGB", (4096, 4096)), baseline)
    assert not result.calls
    assert result.prediction is baseline
    assert result.diagnostics["skipped_duplicate_sizes"] == ["medium", "high"]
