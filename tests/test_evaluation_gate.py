"""Gate decisions use real process identity and complete immutable receipts."""

import copy
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
from PIL import Image

from orchestration import evaluation_gate, state_monitor
from orchestration.evaluation_gate import EvaluationGate, ExclusiveGPU, validate_receipts
from orchestration.settings import Settings
from orchestration.storage import atomic_json, file_hash, object_hash
from xiaomi_runtime.adapter import XiaomiRuntime


@pytest.fixture
def settings(tmp_path):
    source = tmp_path / "source"
    project = tmp_path / "project"
    (source / "runs/xmod").mkdir(parents=True)
    project.mkdir()
    return Settings(project, source, tmp_path / "runtime", Path("python"), stability_seconds=30)


def ready_observation():
    return {"source_processes": [], "gpu": {"known": True, "processes": []},
            "groups": {name: {"evaluation_status": "COMPLETED"} for name in ("B2", "B3")},
            "final_results_exist": True, "final_report_exists": True, "output_snapshot": "initial"}


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def write_fixture(path, value):
    """Fixture generation is not the atomic-storage behavior under test."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


@pytest.fixture
def complete_receipts(settings):
    ids = [f"fixture-{index}" for index in range(665)]
    final: dict[str, Any] = {"evaluation": "table-recognition", "groups": {}}
    for group in ("B0", "B1", "B2", "B3"):
        stage = settings.historical / group
        identity = {"model": "fixture", "table_prompt": "fixture", "group": group}
        rows = [{"id": item, "finish_reason": "stop", "prediction_html": "<table/>",
                 "output_tokens": 5, "raw": "<ecel><nl>"} for item in ids]
        for index, row in enumerate(rows):
            write_fixture(stage / "pred_tables" / f"{index}.json", row)
        write_fixture(stage / "identity.json", identity)
        write_jsonl(stage / "table_pairs.jsonl", rows)
        write_jsonl(stage / "per_table.jsonl", [{"id": item} for item in ids])
        write_fixture(stage / "predictions.json", {"identity": identity,
                    "table_pairs": file_hash(stage / "table_pairs.jsonl")})
        write_fixture(stage / "official_table.json", {"diagnostics": {
            "sample_count": 665, "timeout_case_count": 0, "error_case_count": 0}})
        final["groups"][group] = {"tables": {"count": 665}}
    write_fixture(settings.historical / "results.json", final)
    report = settings.source / "reports/xmod/omnidocbench_report.md"
    report.parent.mkdir(parents=True)
    report.write_text("\n".join(f"| {name} | 665 | fixture |" for name in final["groups"]))
    return settings, ids


def write_completion_receipt(settings):
    receipt = {"B2_final_status": "COMPLETED", "B3_final_status": "COMPLETED",
               **validate_receipts(settings)}
    write_fixture(settings.reports / "b2_b3_completion.json", receipt)


def test_full_results_need_output_stability(settings):
    gate = EvaluationGate(settings)
    state = ready_observation()
    assert not gate.check(state, clock=100)["ready"]
    assert not gate.check(state, clock=129)["ready"]
    assert gate.check(state, clock=130)["ready"]
    state["output_snapshot"] = "still-writing"
    assert not gate.check(state, clock=140)["ready"]
    assert not gate.check(state, clock=169)["ready"]
    assert gate.check(state, clock=170)["ready"]


@pytest.mark.parametrize(("change", "reason"), [
    ({"source_processes": [{"pid": 123}]}, "historical evaluation"),
    ({"gpu": {"known": False, "processes": []}}, "unknown"),
    ({"gpu": {"known": True, "processes": [{"pid": 123}]}}, "GPU has"),
    ({"final_results_exist": False}, "missing"),
    ({"final_report_exists": False}, "missing"),
    ({"groups": {"B2": {"evaluation_status": "FAILED"}, "B3": {"evaluation_status": "COMPLETED"}}}, "FAILED"),
    ({"groups": {"B2": {"evaluation_status": "COMPLETED"}, "B3": {"evaluation_status": "INCOMPLETE"}}}, "INCOMPLETE"),
])
def test_finished_process_alone_never_opens_gate(settings, change, reason):
    state = ready_observation() | change
    gate = EvaluationGate(settings)
    gate.check(state, clock=0)
    result = gate.check(state, clock=60)
    assert not result["ready"]
    assert any(reason in message for message in result["reasons"])


def fake_proc(tmp_path, pid=47, state="S", ticks=1234):
    proc = tmp_path / "proc"
    folder = proc / str(pid)
    folder.mkdir(parents=True)
    (proc / "sys/kernel/random").mkdir(parents=True)
    (proc / "sys/kernel/random/boot_id").write_text("boot-one")
    fields = [state] + ["0"] * 49
    fields[19] = str(ticks)
    (folder / "stat").write_text(f"{pid} (python with ) parentheses) " + " ".join(fields))
    (folder / "cmdline").write_bytes(b"python\0scripts/xmod_evaluate.py\0")
    return proc


def test_linux_process_start_ticks_and_command(tmp_path):
    proc = fake_proc(tmp_path)
    identity = state_monitor.process_identity(47, proc)
    assert identity == {"pid": 47, "start_ticks": 1234, "boot_id": "boot-one",
                        "command": ["python", "scripts/xmod_evaluate.py"], "state": "S"}
    assert state_monitor.same_process(identity, identity)


@pytest.mark.parametrize(("field", "value"), [
    ("pid", 48), ("start_ticks", 5678), ("boot_id", "boot-two"), ("command", ["another-program"]),
])
def test_pid_reuse_reboot_or_changed_command_is_new_process(tmp_path, field, value):
    proc = fake_proc(tmp_path)
    original = state_monitor.process_identity(47, proc)
    assert not state_monitor.same_process(original, original | {field: value})
    assert not state_monitor.same_process(original, None)


def test_zombie_is_terminal(tmp_path):
    proc = fake_proc(tmp_path, state="Z")
    assert state_monitor.process_identity(47, proc) is None


def test_exact_665_coverage_and_receipts(complete_receipts):
    settings, ids = complete_receipts
    assert state_monitor.expected_ids(settings) == ids
    assert state_monitor.inspect_group(settings.historical / "B2", ids)["evaluation_status"] == "COMPLETED"
    assert validate_receipts(settings)["output_manifest_hash"]


def test_one_missing_prediction_and_duplicate_id_fail_coverage(complete_receipts):
    settings, ids = complete_receipts
    stage = settings.historical / "B2"
    (stage / "pred_tables/664.json").unlink()
    result = state_monitor.inspect_group(stage, ids)
    assert result["missing_samples"] == 1
    assert result["evaluation_status"] != "COMPLETED"
    atomic_json(stage / "pred_tables/duplicate.json", json.loads((stage / "pred_tables/0.json").read_text()))
    result = state_monitor.inspect_group(stage, ids)
    assert result["evaluation_status"] == "FAILED"
    with pytest.raises(ValueError):
        validate_receipts(settings)


def test_scorer_failure_prevents_completion(complete_receipts):
    settings, ids = complete_receipts
    stage = settings.historical / "B3"
    atomic_json(stage / "official_table.json", {"diagnostics": {
        "sample_count": 665, "timeout_case_count": 1, "error_case_count": 0}})
    assert state_monitor.inspect_group(stage, ids)["evaluation_status"] == "FAILED"


def test_receipt_hash_and_final_report_checked(complete_receipts):
    settings, _ = complete_receipts
    report = settings.source / "reports/xmod/omnidocbench_report.md"
    report.write_text("Pending evaluation")
    with pytest.raises(ValueError, match="placeholder"):
        validate_receipts(settings)
    report.write_text("\n".join(f"| {name} | 665 | fixture |" for name in ("B0", "B1", "B2", "B3")))
    pairs = settings.historical / "B2/table_pairs.jsonl"
    pairs.write_text(pairs.read_text() + "\n")
    with pytest.raises(ValueError, match="hash mismatch"):
        validate_receipts(settings)


def test_prediction_cache_tampering_detected(complete_receipts):
    settings, _ = complete_receipts
    path = settings.historical / "B3/pred_tables/2.json"
    content = json.loads(path.read_text())
    content["raw"] = "<fcel>changed<nl>"
    atomic_json(path, content)
    with pytest.raises(ValueError, match="disagrees"):
        validate_receipts(settings)


def test_completion_receipt_with_all_authoritative_sources(complete_receipts, monkeypatch):
    settings, _ = complete_receipts
    monkeypatch.setattr(evaluation_gate, "audit", lambda _: ready_observation())
    receipt = EvaluationGate(replace(settings, stability_seconds=0)).completion_receipt()
    assert receipt["completed_samples"] == {"B2": 665, "B3": 665}
    assert receipt["failed_samples"] == {"B2": 0, "B3": 0}
    assert receipt["evaluation_config_hash"]


def test_gpu_lease_rejects_second_owner_and_unknown_gpu(complete_receipts, monkeypatch):
    settings, _ = complete_receipts
    write_completion_receipt(settings)
    monkeypatch.setattr(state_monitor, "source_processes", lambda _: [])
    monkeypatch.setattr(state_monitor, "gpu_processes", lambda: {"known": False, "processes": []})
    with ExclusiveGPU(settings) as first:
        with pytest.raises(BlockingIOError), ExclusiveGPU(settings):
            pytest.fail("A second GPU lease was granted")
        with pytest.raises(RuntimeError, match="GPU process"):
            first.permission_check()


@pytest.mark.parametrize(("source", "gpu"), [
    ([{"pid": 123}], {"known": True, "processes": []}),
    ([], {"known": False, "processes": []}),
    ([], {"known": True, "processes": [{"pid": -1}]}),
])
def test_gate_denial_precedes_runtime_weight_access(complete_receipts, monkeypatch, source, gpu):
    settings, _ = complete_receipts
    write_completion_receipt(settings)
    monkeypatch.setattr(state_monitor, "source_processes", lambda _: source)
    monkeypatch.setattr(state_monitor, "gpu_processes", lambda: gpu)
    with ExclusiveGPU(settings) as lease:
        runtime = XiaomiRuntime(settings.project / "nonexistent-model", permission_check=lease.permission_check)
        monkeypatch.setattr(runtime, "assert_weights_unchanged", lambda: pytest.fail("Gate allowed model access"))
        with pytest.raises(RuntimeError):
            runtime.infer("fixture", Image.new("RGB", (32, 32)), method="A", view_type="medium")
        with pytest.raises(RuntimeError):
            runtime.count_tokens("<ecel><nl>")
        assert runtime.model is None
        assert runtime.load_count == 0
        assert runtime.call_count == 0


def test_failed_lease_acquisition_releases_file_lock(complete_receipts):
    settings, _ = complete_receipts
    rejected = ExclusiveGPU(settings)
    with pytest.raises(FileNotFoundError):
        rejected.__enter__()
    write_completion_receipt(settings)
    with ExclusiveGPU(settings) as accepted:
        assert accepted.file is not None and not accepted.file.closed


@pytest.mark.parametrize(("filename", "value"), [
    ("pred_tables/2.json", "{unfinished"),
    ("official_table.json", "{}"),
    ("per_table.jsonl", "{unfinished\n"),
])
def test_malformed_historical_output_is_failed_state(complete_receipts, filename, value):
    settings, ids = complete_receipts
    stage = settings.historical / "B2"
    (stage / filename).write_text(value)
    result = state_monitor.inspect_group(stage, ids)
    assert result["evaluation_status"] == "FAILED"
    assert result["errors"]


def test_real_live_gate_uses_processes_without_inference():
    config = Path(__file__).parents[1] / "configs/local.yaml"
    if not config.exists():
        pytest.skip("Local machine configuration is intentionally excluded from the repository")
    settings = Settings.load(config)
    observation = state_monitor.audit(settings)
    if not observation["source_processes"]:
        pytest.skip("Historical evaluation is already terminal on this machine")
    before = copy.deepcopy(observation["source_processes"])
    gate = EvaluationGate(settings)
    result = gate.check(observation, clock=0)
    assert not result["ready"]
    for identity in before:
        current = state_monitor.process_identity(identity["pid"])
        assert current is None or state_monitor.same_process(identity, current)
    atomic_json(settings.reports / "gate_cpu_test.json", {
        "observed_at": state_monitor.now(), "passed": True, "ready": result["ready"],
        "reasons": result["reasons"], "processes": before,
        "observation_hash": object_hash(observation), "gpu_inference_started": False,
    })
