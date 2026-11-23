"""Synthetic integration of the public method APIs with one mock runtime."""

import ast
import json
import shutil
from copy import deepcopy
from dataclasses import asdict, replace
from hashlib import sha256
from pathlib import Path
from types import SimpleNamespace

from PIL import Image, ImageDraw

from orchestration.launcher import build_methods
from orchestration.settings import Settings
from xiaomi_runtime import adapter
from xiaomi_runtime.adapter import Prediction
from xiaomi_runtime.otsl import to_html
from xiaomi_runtime.processor import TABLE_PROMPT, ProcessorLimits

PROJECT = Path(__file__).resolve().parents[1]
ROWS = (
    ("Alpba", "North", "Label"),
    ("Gamma", "Delta", "Value"),
    ("Sigma", "Omega", "East"),
    ("West", "Name", "Group"),
    ("South", "Total", "Unit"),
)


def _otsl(rows):
    return "".join("".join("<fcel>" + text for text in row) + "<nl>" for row in rows)


def _image():
    image = Image.new("RGB", (301, 181), "white")
    draw = ImageDraw.Draw(image)
    horizontal = (5, 25, 65, 95, 125, 175)
    vertical = (5, 70, 190, 295)
    for y in horizontal:
        draw.line((5, y, 295, y), fill="black")
    for x in vertical:
        draw.line((x, 5, x, 175), fill="black")
    for row, values in enumerate(ROWS):
        for column, text in enumerate(values):
            draw.text((vertical[column] + 4, horizontal[row] + 3), text, fill="black")
    return image


def _prediction(raw, **overrides):
    return Prediction(raw=raw, html=to_html(raw), table_id="shared-synthetic-table",
                      method=overrides.pop("method", "B0"),
                      view_type=overrides.pop("view_type", "global"),
                      output_hash=sha256(raw.encode()).hexdigest(), **overrides)


class SharedMockRuntime:
    """The public runtime contract, with mock weights and no GPU dependencies."""

    def __init__(self):
        self.model = None
        self.load_count = 0
        self.calls = []
        self.fail_once = []

    @property
    def call_count(self):
        return len(self.calls)

    def infer(self, table_id, image, *, method, view_type, resolution=None):
        if self.model is None:
            self.model = object()
            self.load_count += 1
        self.calls.append({"instance": id(self), "model": id(self.model), "table_id": table_id,
                           "image": image, "method": method, "view_type": view_type,
                           "resolution": resolution})
        if (method, view_type) in self.fail_once:
            self.fail_once.remove((method, view_type))
            raise RuntimeError("synthetic interrupted generation")
        if view_type == "global":
            raw = _otsl(ROWS)
        elif method == "A":
            raw = _otsl(ROWS).replace("Alpba", "Alternate")
        elif view_type == "local_0":
            raw = _otsl(ROWS[:2]).replace("Alpba", "Alpha")
        else:
            raw = _otsl(ROWS[-2:])
        processed = resolution or ProcessorLimits().native_size(image.size)
        return replace(_prediction(raw, method=method, view_type=view_type), table_id=table_id,
                       input_size=image.size, processed_size=processed,
                       image_grid_thw=[[1, processed[1] // 16, processed[0] // 16]],
                       visual_tokens=processed[0] * processed[1] // 1024,
                       output_tokens=len(raw) + 1)

    def count_tokens(self, raw):
        return len(raw) + 1

    def memory_stats(self):
        return {"load_count": self.load_count, "allocated_bytes": 0, "mock": True}

    def assert_weights_unchanged(self):
        return {"mock_weights": "synthetic"}


def _settings(tmp_path):
    return Settings(project=PROJECT, source=tmp_path / "source", runtime=tmp_path / "runtime",
                    python=tmp_path / "python")


def test_three_methods_share_one_runtime_and_original_baseline(tmp_path):
    runtime = SharedMockRuntime()
    methods = build_methods(runtime, _settings(tmp_path))
    image = _image()
    pixels_before = image.tobytes()
    baseline = _prediction(_otsl(ROWS))
    baseline_before = deepcopy(asdict(baseline))
    assert tuple(methods) == ("A", "B", "C")
    assert all(method.runtime is runtime for method in methods.values())

    results = {name: method.run(baseline.table_id, image, baseline) for name, method in methods.items()}

    assert runtime.load_count == 1
    assert len({call["model"] for call in runtime.calls}) == 1
    assert len({call["instance"] for call in runtime.calls}) == 1
    assert [call["method"] for call in runtime.calls] == ["A", "A", "B", "B"]
    assert [len(results[name].calls) for name in methods] == [2, 2, 0]
    assert all(call["table_id"] == baseline.table_id for call in runtime.calls)
    assert results["A"].changed and "Alternate" in results["A"].prediction.raw
    assert results["B"].changed and "Alpha" in results["B"].prediction.raw
    assert "Alternate" not in results["B"].prediction.raw
    assert not results["C"].changed
    assert results["C"].prediction.raw == baseline.raw
    assert image.tobytes() == pixels_before
    assert asdict(baseline) == baseline_before
    assert all(result.prediction.model_hash == baseline.model_hash for result in results.values())
    assert all(result.prediction.prompt_hash == sha256(TABLE_PROMPT.encode()).hexdigest()
               for result in results.values())


def test_uncached_global_calls_use_original_image_and_default_resolution(tmp_path):
    runtime = SharedMockRuntime()
    methods = build_methods(runtime, _settings(tmp_path))
    image = _image()
    result = methods["A"].run("shared-synthetic-table", image)
    first = runtime.calls[0]
    assert first["image"] is image
    assert first["view_type"] == "global" and first["resolution"] is None
    assert len(result.calls) == 3
    assert runtime.load_count == 1

    methods["C"].run("shared-synthetic-table", image)
    final = runtime.calls[-1]
    assert final["image"] is image
    assert final["view_type"] == "global" and final["resolution"] is None
    assert runtime.load_count == 1


def test_method_packages_have_no_cross_method_imports():
    packages = {"multiview", "reacquisition", "render_verify"}
    for package in sorted(packages):
        forbidden = packages - {package}
        for source in (PROJECT / "src" / package).rglob("*.py"):
            parsed = ast.parse(source.read_text(), filename=source.name)
            for node in ast.walk(parsed):
                if isinstance(node, ast.Import):
                    imported = {alias.name.split(".")[0] for alias in node.names}
                elif isinstance(node, ast.ImportFrom) and node.level == 0:
                    imported = {(node.module or "").split(".")[0]}
                else:
                    continue
                assert not imported & forbidden, f"{source.name} crosses a method boundary"


def test_persistent_runner_uses_shared_runtime_and_writes_independent_results(tmp_path, monkeypatch):
    from evaluation import official_runner, omnidocbench
    from orchestration.launcher import run_methods

    shutil.copytree(PROJECT / "configs", tmp_path / "configs")
    (tmp_path / "experiments").mkdir()
    (tmp_path / "experiments" / "resources").symlink_to(
        PROJECT / "experiments" / "resources", target_is_directory=True)
    settings = replace(_settings(tmp_path), project=tmp_path)
    runtime = SharedMockRuntime()
    runtime.fail_once = [("A", "high")]
    image = _image()
    native = ProcessorLimits().native_size(image.size)
    raw = _otsl(ROWS)
    source = {"id": "shared-synthetic-table", "raw": raw, "prediction_html": to_html(raw),
              "finish_reason": "stop", "visual_tokens": native[0] * native[1] // 1024,
              "output_tokens": len(raw) + 1, "seconds": 0.01}
    table = SimpleNamespace(table_id=source["id"], page="synthetic-page", baseline=source)

    class MockCohort:
        def __init__(self, settings):
            self.identity = {"synthetic": True}
            self.tables = [table]

        def save_manifest(self):
            pass

        def image(self, selected):
            assert selected is table
            return image

    constructor_calls = []

    def make_runtime(*args, **kwargs):
        constructor_calls.append((args, kwargs))
        return runtime

    scoring_calls = []
    render_attempts = []

    def methods_with_transient_render_failure(runtime, settings):
        methods = build_methods(runtime, settings)
        original_render = methods["C"].renderer.render

        def render(markup):
            render_attempts.append(markup)
            if len(render_attempts) == 1:
                raise RuntimeError("synthetic transient render backend failure")
            return original_render(markup)

        methods["C"].renderer.render = render
        return methods

    from orchestration import launcher
    monkeypatch.setattr(launcher, "build_methods", methods_with_transient_render_failure)
    monkeypatch.setattr(omnidocbench, "Cohort", MockCohort)
    monkeypatch.setattr(adapter, "XiaomiRuntime", make_runtime)
    monkeypatch.setattr(official_runner, "score_method", lambda settings, method, ids: scoring_calls.append((method, ids)))
    state = {"commit": "synthetic-commit"}
    lease = SimpleNamespace(permission_check=lambda: None)
    run_methods(settings, state, lease, ("A", "B", "C"))

    assert len(constructor_calls) == runtime.load_count == 1
    assert state["status"] == "COMPLETED"
    assert state["progress"] == {"A": 1, "B": 1, "C": 1}
    assert scoring_calls == [(name, [source["id"]]) for name in ("A", "B", "C")]
    records = {}
    for name in ("A", "B", "C"):
        files = list((settings.output / f"method_{name.lower()}" / "predictions").glob("*.json"))
        assert len(files) == 1
        records[name] = json.loads(files[0].read_text())
        assert records[name]["table_id"] == source["id"]
        assert records[name]["status"] == "completed"
    assert [records[name]["calls"] for name in ("A", "B", "C")] == [5, 3, 1]
    assert [records[name]["successful_attempt_calls"] for name in ("A", "B", "C")] == [3, 3, 1]
    assert records["A"]["prior_failed_attempts"] == 1
    assert records["A"]["incremental_latency"] > records["A"]["successful_attempt_incremental_latency"]
    assert records["A"]["prediction"] == raw.replace("Alpba", "Alternate")
    assert records["B"]["prediction"] == raw.replace("Alpba", "Alpha")
    assert records["C"]["prediction"] == raw
    assert records["C"]["error"] is None
    assert len(list((settings.output / "method_c" / "attempts").glob("*.json"))) == 1
    assert len(render_attempts) == 2

    calls_before = len(runtime.calls)
    run_methods(settings, state, lease, ("A", "B", "C"))
    assert len(runtime.calls) == calls_before
    assert state["progress"] == {"A": 1, "B": 1, "C": 1}
