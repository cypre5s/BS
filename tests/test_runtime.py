import hashlib
from dataclasses import replace

import pytest
from PIL import Image

from xiaomi_runtime import adapter
from xiaomi_runtime.adapter import XiaomiRuntime, cached_prediction, replay_baseline
from xiaomi_runtime.processor import ProcessorLimits, encode_prompt, view_image


def test_gate_denial_precedes_any_model_load(tmp_path):
    events = []

    def denied():
        events.append("gate")
        raise PermissionError("B2/B3 still running")

    runtime = XiaomiRuntime(tmp_path, permission_check=denied)
    assert runtime.load_count == 0
    assert runtime.memory_stats()["allocated_bytes"] == 0
    with pytest.raises(PermissionError):
        runtime.infer("fixture", Image.new("RGB", (256, 256)), method="A", view_type="global")
    assert events == ["gate"]
    assert runtime.model is None


def test_snapshot_hash_guard_is_read_only(tmp_path, monkeypatch):
    path = tmp_path / "model.safetensors"
    path.write_bytes(b"synthetic immutable fixture")
    expected = hashlib.sha256(path.read_bytes()).hexdigest()
    monkeypatch.setattr(adapter, "MODEL_FILES_SHA256", {path.name: expected})
    runtime = XiaomiRuntime(tmp_path, permission_check=lambda: None)
    assert runtime.assert_weights_unchanged() == {path.name: expected}
    assert path.read_bytes() == b"synthetic immutable fixture"
    path.write_bytes(b"modified fixture")
    with pytest.raises(ValueError, match="differs"):
        runtime.assert_weights_unchanged()


def test_global_prompt_and_input_preserved():
    image = Image.new("RGB", (300, 200))
    assert view_image(image, None, ProcessorLimits()) is image

    class Processor:
        def apply_chat_template(self, messages, **kwargs):
            assert messages[0]["content"][0]["text"] == "Parse the table in the image into OTSL."
            assert messages[0]["content"][1]["image"] is image
            assert kwargs == {"add_generation_prompt": True, "enable_thinking": False,
                              "tokenize": True, "return_dict": True, "return_tensors": "pt"}
            return "encoded"

    assert encode_prompt(Processor(), image) == "encoded"


def test_cached_baseline_matches_parser_and_grid():
    source = {"id": "fixture", "raw": "<fcel>A<nl>",
              "prediction_html": "<table><tr><td>A</td></tr></table>",
              "visual_tokens": 64, "output_tokens": 5, "seconds": 1.0, "finish_reason": "stop"}
    baseline = cached_prediction(source, Image.new("RGB", (256, 256)))
    assert baseline.visual_tokens == 64
    assert baseline.image_grid_thw == [[1, 16, 16]]
    assert baseline.eos and not baseline.truncated
    assert baseline.model_hash == adapter.MODEL_SHA256
    with pytest.raises(ValueError, match="HTML"):
        cached_prediction({**source, "prediction_html": ""}, Image.new("RGB", (256, 256)))


def test_native_replay_detects_global_runtime_drift():
    image = Image.new("RGB", (256, 256))
    baseline = cached_prediction({"id": "fixture", "raw": "<fcel>A<nl>",
                                  "prediction_html": "<table><tr><td>A</td></tr></table>",
                                  "visual_tokens": 64, "output_tokens": 5,
                                  "seconds": 1.0, "finish_reason": "stop"}, image)

    class Runtime:
        result = baseline

        def infer(self, table_id, source, **kwargs):
            assert table_id == baseline.table_id and source is image
            assert kwargs == {"method": "B0_audit", "view_type": "global"}
            return self.result

    runtime = Runtime()
    assert replay_baseline(runtime, image, baseline)["passed"]
    runtime.result = replace(baseline, image_grid_thw=[[1, 8, 32]])
    result = replay_baseline(runtime, image, baseline)
    assert not result["passed"]
    assert result["mismatched_fields"] == ["image_grid_thw"]
