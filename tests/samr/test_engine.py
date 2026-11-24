import json
from dataclasses import replace
from pathlib import Path

import pytest
from PIL import Image

from samr.config import SAMRConfig
from samr.development import profile_rows
from samr.engine import SAMRInferenceEngine
from xiaomi_runtime.budget import BudgetExceeded


def fixed(scales=(1.0, 1.5, 1.75), **kwargs):
    return replace(SAMRConfig(), mode="fixed", scales=scales, routing_policy="specified_scales", **kwargs)


@pytest.mark.parametrize("name,count", [("scale_1_25", 1), ("scale_pair_global_medium", 2),
                                       ("fusion_global_anchor", 3), ("scale_all", 4)])
def test_modes_share_one_model_and_leave_mock_weights_unchanged(mock_runtime, name, count):
    before = mock_runtime.weight_bytes
    engine = SAMRInferenceEngine(mock_runtime, SAMRConfig.load(Path(f"configs/samr/{name}.yaml")))
    result = engine.run("dev:synthetic", Image.new("RGB", (512, 256), "white"))
    assert len(mock_runtime.calls) == len(result.audit["scales"]) == count
    assert mock_runtime.load_count == 1 and mock_runtime.weight_bytes == before
    assert len({call["deadline"] for call in mock_runtime.calls}) == 1
    assert result.prediction.raw == mock_runtime.outputs[0]


def test_default_route_is_global_only_for_simple_image(mock_runtime):
    SAMRInferenceEngine(mock_runtime).run("dev:simple", Image.new("RGB", (256, 256), "white"))
    assert len(mock_runtime.calls) == 1


def test_token_overbudget_preflight_and_extras_are_not_executed(mock_runtime):
    image = Image.new("RGB", (256, 256), "white")
    result = SAMRInferenceEngine(mock_runtime, fixed(max_visual_tokens=64)).run("dev:budget", image)
    assert len(mock_runtime.calls) == 1
    assert result.audit["budget"]["reserved_visual_tokens"] == 64
    assert all(event["reason"] == "visual_token_budget_exhausted" for event in result.audit["routing_decision"]["events"][1:])
    with pytest.raises(BudgetExceeded, match="Required first"):
        SAMRInferenceEngine(mock_runtime, fixed(max_visual_tokens=64)).run("dev:too_large", image.resize((512, 512)))
    assert len(mock_runtime.calls) == 1


@pytest.mark.parametrize("raw", ["bad OTSL", "<fcel><table><tr><td>nested</td></tr></table><nl>",
                                "<fcel>A<nl><fcel>B<nl>", "<fcel>longer content<fcel>B<nl>",
                                RuntimeError("synthetic CUDA failure")])
def test_invalid_or_conflicting_auxiliaries_retain_original(mock_runtime, raw):
    mock_runtime.outputs = ["<fcel>A<fcel>B<nl>", raw]
    result = SAMRInferenceEngine(mock_runtime, fixed()).run("dev:failure", Image.new("RGB", (400, 256), "white"))
    assert result.prediction.raw == mock_runtime.outputs[0]
    assert not result.audit["fusion_decision"]["changed"]


def test_fusion_exception_falls_back_and_logs_complete_evidence(mock_runtime, tmp_path, monkeypatch):
    def broken(*args):
        raise RuntimeError("synthetic fusion failure")
    monkeypatch.setattr("samr.engine.fuse_global_anchor", broken)
    result = SAMRInferenceEngine(mock_runtime, fixed()).run("dev:audit", Image.new("RGB", (400, 256), "white"),
                                                          audit_directory=tmp_path)
    receipt = json.loads(next(tmp_path.glob("*.json")).read_text())
    assert receipt["final_otsl"] == receipt["B0_raw_otsl"] == result.prediction.raw
    assert "fusion_exception" in receipt["fusion_decision"]["reason"]
    assert len(receipt["scales"]) == 3
    assert len(receipt["routing_decision"]["events"]) == 3
    for scale in receipt["scales"]:
        assert scale["raw_otsl"] and scale["canonical_grid"] and scale["image_grid_thw"]
        assert scale["visual_token_count"] == scale["estimated_visual_tokens"]
    with pytest.raises(FileExistsError):
        SAMRInferenceEngine(mock_runtime, fixed()).run("dev:audit", Image.new("RGB", (400, 256)), audit_directory=tmp_path)
    rows = profile_rows(receipt)
    assert all(row["teds"] is None and row["em"] is None and not row["scored_offline"] for row in rows)


def test_grid_metadata_drift_is_rejected_with_raw_preserved(mock_runtime, monkeypatch):
    infer = mock_runtime.infer
    def corrupt(*args, **kwargs):
        result = infer(*args, **kwargs)
        return result if len(mock_runtime.calls) == 1 else replace(result, image_grid_thw=[[1, 1, 1]])
    monkeypatch.setattr(mock_runtime, "infer", corrupt)
    result = SAMRInferenceEngine(mock_runtime, fixed()).run("dev:drift", Image.new("RGB", (400, 256), "white"))
    assert result.prediction.raw == mock_runtime.outputs[0]
    assert len(result.audit["scales"]) == 3
    assert result.audit["routing_decision"]["events"][1]["decision"] == "failed"


def test_positive_fusion_through_pipeline(mock_runtime, ruled_image):
    baseline = "<ecel><lcel><nl><fcel>A<fcel>B<nl>"
    split = "<ecel><ecel><nl><fcel>A<fcel>B<nl>"
    mock_runtime.outputs = [baseline, split, split]
    result = SAMRInferenceEngine(mock_runtime, fixed()).run("dev:blank_split", ruled_image)
    assert result.prediction.raw == split
    assert result.audit["B0_raw_otsl"] == baseline
    assert result.audit["fusion_decision"]["changed"]
