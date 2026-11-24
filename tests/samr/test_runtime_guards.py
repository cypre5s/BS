from dataclasses import replace
from types import SimpleNamespace

import pytest
from PIL import Image

from samr.config import SAMRConfig
from samr.engine import SAMRInferenceEngine
from samr.routing import BudgetLedger
from xiaomi_runtime import adapter, inference
from xiaomi_runtime.adapter import XiaomiRuntime
from xiaomi_runtime.budget import BudgetExceeded, DeadlineExceeded, DeadlineStop, check_deadline


def test_decode_deadline_exact_boundary_and_expired_call(monkeypatch):
    now = [1.0]
    stop = DeadlineStop(2.0, lambda: now[0])
    assert not stop(None, None)
    now[0] = 2.0
    assert stop(None, None) and stop.expired
    monkeypatch.setattr("xiaomi_runtime.budget.perf_counter", lambda: 2.0)
    with pytest.raises(DeadlineExceeded):
        check_deadline(2.0)
    check_deadline(None)


def test_auxiliary_is_never_scheduled_after_global_uses_deadline(mock_runtime, monkeypatch):
    now = [0.0]
    config = replace(SAMRConfig(), mode="fixed", routing_policy="specified_scales", max_latency_budget=1.0)
    # Real runtime uses monotonic seconds; mock deliberately ignores deadline to simulate a long prefill.
    infer = mock_runtime.infer
    def slow(*args, **kwargs):
        kwargs["deadline"] = float("inf")
        result = infer(*args, **kwargs)
        now[0] = 2.0
        return result
    monkeypatch.setattr(mock_runtime, "infer", slow)
    monkeypatch.setattr("samr.engine.BudgetLedger", lambda config: BudgetLedger(config, lambda: now[0]))
    monkeypatch.setattr("samr.engine.perf_counter", lambda: now[0])
    result = SAMRInferenceEngine(mock_runtime, config).run("dev:slow", Image.new("RGB", (256, 256)))
    assert len(mock_runtime.calls) == 1
    assert result.audit["budget"]["deadline_overshoot_seconds"] == 1.0
    assert "latency_budget_exhausted" in result.audit["fusion_decision"]["reason"]
    assert all(event["reason"] == "latency_budget_exhausted" for event in result.audit["routing_decision"]["events"][1:])


def test_native_runtime_checks_actual_tokens_before_generate(tmp_path, monkeypatch):
    class Scalar:
        def item(self):
            return 64
    class Tokens:
        def __eq__(self, other):
            return self
        def sum(self):
            return Scalar()
    class Grid:
        def tolist(self):
            return [[1, 16, 16]]
    class Inputs(dict):
        def to(self, device):
            return self
    runtime = XiaomiRuntime(tmp_path, permission_check=lambda: None)
    runtime.model = SimpleNamespace(device="cpu", config=SimpleNamespace(image_token_id=123))
    runtime.processor = SimpleNamespace(image_processor=SimpleNamespace(
        patch_size=16, merge_size=2, size={"shortest_edge": 65536, "longest_edge": 16777216}))
    captured = []
    monkeypatch.setattr(adapter, "encode_prompt", lambda processor, image: Inputs(
        image_grid_thw=Grid(), input_ids=Tokens()))
    monkeypatch.setattr(adapter, "generate", lambda *args, **kwargs: captured.append(True))
    with pytest.raises(BudgetExceeded, match="Actual native"):
        runtime.infer("dev:budget", Image.new("RGB", (256, 256)), method="SAMR", view_type="global",
                      max_visual_tokens=63)
    assert not captured and runtime.call_count == 0


def test_generation_contract_is_pinned_and_runtime_is_not_forked():
    assert inference.GENERATION_CONFIG["do_sample"] is False
    assert inference.GENERATION_CONFIG["max_new_tokens"] == 16384
    assert inference.GENERATION_CONFIG["repetition_penalty"] == 1.0
    assert inference.GENERATION_CONFIG["clean_up_tokenization_spaces"] is False
    import inspect
    source = inspect.getsource(SAMRInferenceEngine)
    assert "from_pretrained" not in source and "pixel_values" not in source
