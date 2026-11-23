"""Historical recovery is exercised with synthetic files and mocked native functions only."""
from __future__ import annotations

import fcntl
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from orchestration import recovery, state_monitor
from orchestration.settings import Settings
from orchestration.storage import atomic_json, file_hash, read_json, read_jsonl


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))


def reference(index):
    return {"id": f"page-{index}.png#box_1", "page": f"page-{index}.png", "annotation_id": "box_1",
            "html": "<table><tr><td>fixture</td></tr></table>", "polygon": [0, 0, 10, 0, 10, 10, 0, 10],
            "attributes": {}, "slices": ["simple"]}


def prediction(ref, **changes):
    return {"id": ref["id"], "raw": "<fcel>fixture<nl>", "prediction_html": ref["html"],
            "finish_reason": "stop", "seconds": 1.0, "visual_tokens": 20, "output_tokens": 10,
            "cross_attention_flops": 0, "peak_reserved_mib": 20, "peak_reserved_scope": "group", **changes}


def score(pair, value=0.5):
    structure = {"rows": 1, "cells": 1, "merged_cells": 0}
    return {key: pair[key] for key in ("id", "page", "slices", "seconds", "visual_tokens", "output_tokens",
                                     "finish_reason", "cross_attention_flops")} | {
        "teds": value, "teds_s": value, "exact": 0, "char_edits": 1, "char_count": 7,
        "pred_structure": structure, "gold_structure": structure,
    }


def official(pairs, rows, *, failures=()):
    indexed = {row["id"]: row for row in pairs}
    cases = [{"img_id": indexed[key]["page"], "gt_idx": indexed[key]["annotation_id"], "reason": "mock timeout"}
             for key in failures]
    return {"aggregates": {"TEDS": {"all": sum(row["teds"] for row in rows) / len(rows)},
                           "TEDS_structure_only": {"all": sum(row["teds_s"] for row in rows) / len(rows)}},
            "diagnostics": {"stage": "TEDS", "sample_count": len(pairs), "workers": 6,
                            "timeout_sec": 120.0, "timeout_case_count": len(cases), "timeout_cases": cases,
                            "error_case_count": 0, "error_cases": []}}


def populate(stage, refs):
    identity = {"mock": True, "group": stage.name}
    atomic_json(stage / "identity.json", identity)
    pairs = []
    for ref in refs:
        row = prediction(ref)
        atomic_json(stage / "pred_tables" / f"{Path(ref['page']).stem}_{ref['annotation_id']}.json", row)
        pairs.append(ref | row)
    write_jsonl(stage / "table_pairs.jsonl", pairs)
    atomic_json(stage / "predictions.json", {"identity": identity, "table_pairs": file_hash(stage / "table_pairs.jsonl"),
                                           "peak_reserved_mib": 20, "peak_reserved_scope": "group"})
    # Zero is an ordinary successful metric and must never trigger a retry.
    rows = [score(pair, index / max(1, len(pairs) - 1)) for index, pair in enumerate(pairs)]
    write_jsonl(stage / "per_table.jsonl", rows)
    atomic_json(stage / "official_table.json", official(pairs, rows))


@pytest.fixture
def history(tmp_path, monkeypatch):
    settings = Settings(tmp_path / "project", tmp_path / "source", tmp_path / "runtime", Path("mock-python"))
    refs = [reference(index) for index in range(3)]
    runs = settings.source / "runs/xmod"
    runs.mkdir(parents=True)
    (runs / ".run_all.lock").write_bytes(b"original lock contents\n")
    for group in recovery.GROUPS:
        atomic_json(runs / "full" / group / "summary.json", {"mock_summary": group})
        if group != "B0":
            (runs / "full" / group / "best.pt").write_bytes(f"synthetic {group} checkpoint".encode())
        populate(runs / recovery.BENCHMARK / group, refs)
    atomic_json(runs / recovery.BENCHMARK / "frozen_models/identity.json", {"mock_frozen": True})
    for name in ("scripts/xmod_evaluate.py", "scripts/xmod_run_all.py",
                 "src/xiaomi_ocr_xmod/evaluation.py", "src/xiaomi_ocr_xmod/reporting.py"):
        target = settings.source / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("# mock source; never executed\n")
    original_cohort = recovery._cohort
    monkeypatch.setattr(recovery, "_cohort", lambda path: original_cohort(path, count=len(refs)))
    return SimpleNamespace(settings=settings, refs=refs, runs=runs, work=runs / recovery.BENCHMARK)


def mark_failed_prediction(stage, ref):
    path = stage / "pred_tables" / f"{Path(ref['page']).stem}_{ref['annotation_id']}.json"
    atomic_json(path, prediction(ref, finish_reason="error", output_tokens=0, prediction_html=""))


def mark_failed_score(stage, table_id):
    path = stage / "official_table.json"
    saved = read_json(path)
    pair = next(row for row in read_jsonl(stage / "table_pairs.jsonl") if row["id"] == table_id)
    saved["diagnostics"].update(timeout_case_count=1, timeout_cases=[{
        "img_id": pair["page"], "gt_idx": pair["annotation_id"], "reason": "synthetic timeout"}])
    atomic_json(path, saved)


def mock_native(history, monkeypatch, *, fail_score_once=None):
    calls = {"predictions": [], "scores": [], "aggregate": [], "reports": []}
    source_runs = history.runs

    def native_predictions(group, summary, model, data, pages, refs, release, frozen, frozen_dir, output, **options):
        assert not output.is_relative_to(source_runs)
        assert options == {"evaluation": "table-recognition", "model_loading_lock": None}
        pairs, generated = [], []
        for ref in refs:
            path = output / "pred_tables" / f"{Path(ref['page']).stem}_{ref['annotation_id']}.json"
            if not path.exists():
                generated.append(ref["id"])
                atomic_json(path, prediction(ref, raw="<fcel>regenerated<nl>",
                                            prediction_html="<table><tr><td>regenerated</td></tr></table>"))
            pairs.append(ref | read_json(path))
        calls["predictions"].append((group, generated))
        write_jsonl(output / "table_pairs.jsonl", pairs)
        atomic_json(output / "predictions.json", {"identity": read_json(output / "identity.json"),
                                                 "table_pairs": file_hash(output / "table_pairs.jsonl"),
                                                 "peak_reserved_mib": 20})

    def native_score(group, output, data, release, **options):
        assert not output.is_relative_to(source_runs)
        assert options["launcher"] == "bwrap" and options["evaluation"] == "table-recognition"
        assert options["work"] == output.parent
        if (output / "official_table.json").exists() and (output / "per_table.jsonl").exists():
            return
        pairs = read_jsonl(output / "table_pairs.jsonl")
        calls["scores"].append((group, [row["id"] for row in pairs]))
        failures = [fail_score_once] if len(calls["scores"]) == 1 and fail_score_once else []
        rows = [score(pair, 0.0 if pair["id"] in failures else 0.75) for pair in pairs]
        write_jsonl(output / "per_table.jsonl", rows)
        atomic_json(output / "official_table.json", official(pairs, rows, failures=failures))

    def aggregate(work, full, execution, **options):
        assert not work.is_relative_to(source_runs)
        for group in recovery.GROUPS:
            rows = read_jsonl(work / group / "per_table.jsonl")
            result = read_json(work / group / "official_table.json")
            assert result["aggregates"]["TEDS"]["all"] == pytest.approx(sum(row["teds"] for row in rows) / len(rows))
        calls["aggregate"].append(work)
        atomic_json(work / "results.json", {"mock_completed": True})

    namespace = {"final_checkpoints": lambda runs: {group: {} for group in recovery.GROUPS},
                 "check_evaluator": lambda _path: None, "freeze_models": lambda *_args: {},
                 "load_pages": lambda _path: ([], {}), "table_references": lambda _pages: history.refs,
                 "group_predictions": native_predictions, "score_group": native_score,
                 "aggregate": aggregate, "write_jsonl": write_jsonl}
    monkeypatch.setattr(recovery.runpy, "run_path", lambda *_args, **_kwargs: namespace)

    def render(command, **options):
        assert "--refresh-reports" in command
        assert "--full-reference" in command and "--runs" in command
        target = Path(command[command.index("--reports") + 1])
        assert not target.is_relative_to(source_runs)
        target.mkdir(parents=True, exist_ok=True)
        (target / "omnidocbench_report.md").write_text("Mock completed report\n")
        (target / "full_training_report.md").write_text("Mock Full report\n")
        calls["reports"].append(command)

    monkeypatch.setattr(recovery.subprocess, "run", render)
    monkeypatch.setattr(recovery, "_validate_overlay", lambda work, report: {
        "final_results_hash": file_hash(work / "results.json"), "final_report_hash": file_hash(report)})
    return namespace, calls


def test_overlay_copies_all_caches_and_checkpoints_without_aliasing(history):
    manifest = recovery._source_manifest(history.settings.source)
    root = recovery._overlay(history.settings, manifest)
    copied = root / "runs/full/B2/best.pt"
    original = history.runs / "full/B2/best.pt"
    assert copied.read_bytes() == original.read_bytes()
    assert copied.stat().st_ino != original.stat().st_ino
    copied.write_bytes(b"isolated mutation")
    assert original.read_bytes() == b"synthetic B2 checkpoint"
    assert recovery._overlay(history.settings, manifest) == root
    assert recovery._source_manifest(history.settings.source) == manifest
    for group in recovery.GROUPS:
        assert len(list((root / "runs" / recovery.BENCHMARK / group / "pred_tables").glob("*.json"))) == 3


def test_overlay_rejects_symlinked_cached_files(history):
    (history.work / "unsafe.json").symlink_to(history.runs / "full/B2/best.pt")
    with pytest.raises(ValueError, match="Symlink"):
        recovery._source_manifest(history.settings.source)


def test_prediction_preparation_removes_only_failures_and_accepts_length(history):
    root = recovery._overlay(history.settings, recovery._source_manifest(history.settings.source))
    stage = root / "runs" / recovery.BENCHMARK / "B2"
    mark_failed_prediction(stage, history.refs[1])
    truncated = stage / "pred_tables/page-2_box_1.json"
    atomic_json(truncated, prediction(history.refs[2], finish_reason="length"))
    # Match this terminal prediction in the derived input so only the failed ID is dirty.
    pairs = read_jsonl(stage / "table_pairs.jsonl")
    pairs[2].update(read_json(truncated))
    write_jsonl(stage / "table_pairs.jsonl", pairs)
    receipt = read_json(stage / "predictions.json")
    atomic_json(stage / "predictions.json", receipt | {"table_pairs": file_hash(stage / "table_pairs.jsonl")})
    before = truncated.read_bytes()
    changed, preserved, rebuild = recovery._prepare_predictions(stage, history.refs)
    assert changed == {history.refs[1]["id"]} and rebuild
    assert truncated.read_bytes() == before
    assert len(preserved) == 2
    assert not (stage / "pred_tables/page-1_box_1.json").exists()
    assert not (stage / "predictions.json").exists()
    assert read_json(stage / "recovery_changed_predictions.json")["ids"] == sorted(changed)
    # An interruption after quarantine is resumable even with old scores still present.
    assert recovery._prepare_predictions(stage, history.refs)[0] == changed


def test_malformed_known_prediction_is_retried_but_wrong_identity_fails(history):
    stage = history.work / "B2"
    (stage / "pred_tables/page-1_box_1.json").write_text('{"unfinished":')
    _, _, failed = recovery._cache_records(stage, history.refs)
    assert failed == {history.refs[1]["id"]}
    atomic_json(stage / "pred_tables/page-1_box_1.json", prediction(history.refs[0]))
    with pytest.raises(ValueError, match="identity"):
        recovery._cache_records(stage, history.refs)


def test_scoring_selection_uses_diagnostics_not_low_scores(history):
    stage = history.work / "B2"
    mark_failed_score(stage, history.refs[1]["id"])
    selected, old, _ = recovery._score_selection(stage, read_jsonl(stage / "table_pairs.jsonl"), set())
    assert [row["id"] for row in selected] == [history.refs[1]["id"]]
    assert old[history.refs[0]["id"]]["teds"] == 0.0


@pytest.mark.parametrize("mutation", ["unknown_case", "count", "worker", "timeout"])
def test_ambiguous_failure_or_changed_scorer_configuration_is_rejected(history, mutation):
    stage = history.work / "B2"
    mark_failed_score(stage, history.refs[1]["id"])
    saved = read_json(stage / "official_table.json")
    if mutation == "unknown_case":
        saved["diagnostics"]["timeout_cases"][0]["gt_idx"] = "not-present"
    elif mutation == "count":
        saved["diagnostics"]["timeout_case_count"] = 2
    elif mutation == "worker":
        saved["diagnostics"]["workers"] = 1
    else:
        saved["diagnostics"]["timeout_sec"] = 600
    with pytest.raises(ValueError):
        recovery._failed_scores(saved, read_jsonl(stage / "table_pairs.jsonl"))


def test_missing_diagnostics_quarantines_uncommitted_scores_without_guessing_success(history):
    stage = history.work / "B2"
    (stage / "official_table.json").unlink()
    previous = (stage / "per_table.jsonl").read_bytes()
    selected, old, lines = recovery._score_selection(stage, read_jsonl(stage / "table_pairs.jsonl"), set())
    assert len(selected) == 3 and old == lines == {}
    assert not (stage / "per_table.jsonl").exists()
    assert next((stage / "recovery_discarded").glob("*/per_table.jsonl")).read_bytes() == previous


@pytest.mark.parametrize("broken", ["pairs", "receipt", "scores_without_diagnostics", "partial_diagnostics"])
def test_partial_historical_writes_recover_only_uncommitted_work(history, monkeypatch, broken):
    stage = history.work / "B2"
    if broken == "pairs":
        lines = (stage / "table_pairs.jsonl").read_bytes().splitlines(keepends=True)
        (stage / "table_pairs.jsonl").write_bytes(b"".join(lines[:-1]) + lines[-1][:50])
    elif broken == "receipt":
        (stage / "predictions.json").write_text('{"identity":')
    elif broken == "scores_without_diagnostics":
        (stage / "official_table.json").unlink()
        (stage / "per_table.jsonl").write_text('{"unfinished":')
    else:
        (stage / "official_table.json").write_text('{"diagnostics":')
    manifest = recovery._source_manifest(history.settings.source)
    overlay = recovery._overlay(history.settings, manifest)
    _, calls = mock_native(history, monkeypatch)
    result = recovery._execute_overlay(history.settings.source, overlay / "runs", overlay / "reports",
                                       history.settings.runtime)
    assert result["status"] == "COMPLETED"
    if broken in {"pairs", "receipt"}:
        assert calls["predictions"] == [("B2", [])]  # Rebuild receipts, never repeat successful inference.
    else:
        assert calls["predictions"] == []
    if broken == "pairs":
        assert calls["scores"] == [("B2", [history.refs[-1]["id"]])]
    elif broken == "receipt":
        assert calls["scores"] == []
    else:
        assert calls["scores"] == [("B2", [row["id"] for row in history.refs])]
    assert recovery._source_manifest(history.settings.source) == manifest


def test_recovery_executes_only_failed_predictions_and_scores_and_preserves_source(history, monkeypatch):
    mark_failed_prediction(history.work / "B2", history.refs[1])
    mark_failed_score(history.work / "B3", history.refs[2]["id"])
    manifest = recovery._source_manifest(history.settings.source)
    source_bytes = {path: path.read_bytes() for path in history.settings.source.rglob("*") if path.is_file()}
    original_rows = read_jsonl(history.work / "B2/per_table.jsonl")
    overlay = recovery._overlay(history.settings, manifest)
    _, calls = mock_native(history, monkeypatch)
    result = recovery._execute_overlay(history.settings.source, overlay / "runs", overlay / "reports",
                                       history.settings.runtime)
    assert result["status"] == "COMPLETED"
    assert calls["predictions"] == [("B2", [history.refs[1]["id"]])]
    assert calls["scores"] == [("B2", [history.refs[1]["id"]]), ("B3", [history.refs[2]["id"]])]
    assert len(calls["aggregate"]) == len(calls["reports"]) == 1
    assert (overlay / "reports/full_training_report.md").is_file()
    merged = read_jsonl(Path(result["historical"]) / "B2/per_table.jsonl")
    assert merged[0] == original_rows[0] and merged[2] == original_rows[2]
    for group in ("B0", "B1"):
        for name in ("per_table.jsonl", "official_table.json", "predictions.json", "table_pairs.jsonl"):
            assert (Path(result["historical"]) / group / name).read_bytes() == (history.work / group / name).read_bytes()
    assert all(path.read_bytes() == before for path, before in source_bytes.items())
    assert recovery._source_manifest(history.settings.source) == manifest
    assert not (history.settings.reports / "historical_recovery.json").exists()


def test_interrupted_subset_scoring_preserves_new_successes_and_retries_only_remaining_failure(history, monkeypatch):
    stage = history.work / "B2"
    for ref in history.refs[1:]:
        mark_failed_prediction(stage, ref)
    overlay = recovery._overlay(history.settings, recovery._source_manifest(history.settings.source))
    _, calls = mock_native(history, monkeypatch, fail_score_once=history.refs[2]["id"])
    args = (history.settings.source, overlay / "runs", overlay / "reports", history.settings.runtime)
    with pytest.raises(RuntimeError, match="still failed"):
        recovery._execute_overlay(*args)
    assert not (overlay / "completed.json").exists()
    recovered = overlay / "runs" / recovery.BENCHMARK / "B2"
    successful = read_jsonl(recovered / "per_table.jsonl")[1]
    assert successful["teds"] == 0.75
    assert read_json(recovered / "recovery_changed_predictions.json")["ids"] == []
    result = recovery._execute_overlay(*args)
    assert result["status"] == "COMPLETED"
    assert calls["predictions"] == [("B2", [row["id"] for row in history.refs[1:]])]
    assert calls["scores"] == [("B2", [row["id"] for row in history.refs[1:]]),
                                ("B2", [history.refs[2]["id"]])]
    assert read_jsonl(recovered / "per_table.jsonl")[1] == successful


def test_transaction_replay_after_score_file_replace_preserves_original_success_line_bytes(history, monkeypatch):
    overlay = recovery._overlay(history.settings, recovery._source_manifest(history.settings.source))
    stage = overlay / "runs" / recovery.BENCHMARK / "B2"
    mark_failed_score(stage, history.refs[1]["id"])
    # Preserve unusual but valid formatting byte-for-byte for successful rows.
    lines = (stage / "per_table.jsonl").read_bytes().splitlines(keepends=True)
    lines[0] = json.dumps(json.loads(lines[0]), separators=(",", ":")).encode() + b"\r\n"
    (stage / "per_table.jsonl").write_bytes(b"".join(lines))
    namespace, calls = mock_native(history, monkeypatch)
    original = recovery._atomic_bytes
    target = stage / "official_table.json"

    def interrupted(path, value):
        if path == target:
            raise RuntimeError("synthetic interruption between replacements")
        original(path, value)

    monkeypatch.setattr(recovery, "_atomic_bytes", interrupted)
    with pytest.raises(RuntimeError, match="between replacements"):
        recovery._repair_scores(namespace, stage, set(), data=Path("mock-data"), release={},
                                evaluator=Path("mock-evaluator"), runtime=Path("mock-runtime"))
    assert (stage / "recovery_merge_pending.json").exists()
    monkeypatch.setattr(recovery, "_atomic_bytes", original)
    recovery._resume_merge(stage)
    assert not (stage / "recovery_merge_pending.json").exists()
    result = recovery._repair_scores(namespace, stage, set(), data=Path("mock-data"), release={},
                                     evaluator=Path("mock-evaluator"), runtime=Path("mock-runtime"))
    assert result["rescored_ids"] == [] and len(calls["scores"]) == 1
    assert (stage / "per_table.jsonl").read_bytes().splitlines(keepends=True)[0] == lines[0]


def test_missing_score_files_are_generated_without_regenerating_predictions(history, monkeypatch):
    (history.work / "B3/per_table.jsonl").unlink()
    (history.work / "B3/official_table.json").unlink()
    overlay = recovery._overlay(history.settings, recovery._source_manifest(history.settings.source))
    _, calls = mock_native(history, monkeypatch)
    recovery._execute_overlay(history.settings.source, overlay / "runs", overlay / "reports", history.settings.runtime)
    assert calls["predictions"] == []
    assert calls["scores"] == [("B3", [row["id"] for row in history.refs])]


def test_report_only_recovery_does_not_infer_or_rescore(history, monkeypatch):
    overlay = recovery._overlay(history.settings, recovery._source_manifest(history.settings.source))
    _, calls = mock_native(history, monkeypatch)
    recovery._execute_overlay(history.settings.source, overlay / "runs", overlay / "reports", history.settings.runtime)
    assert calls["predictions"] == calls["scores"] == []
    assert len(calls["reports"]) == 1


def test_invalid_aggregate_is_recomputed_from_unchanged_per_table_scores(history, monkeypatch):
    stage = history.work / "B2"
    before = (stage / "per_table.jsonl").read_bytes()
    saved = read_json(stage / "official_table.json")
    saved["aggregates"]["TEDS"]["all"] = 0.99
    atomic_json(stage / "official_table.json", saved)
    namespace, calls = mock_native(history, monkeypatch)
    result = recovery._repair_scores(namespace, stage, set(), data=Path("mock"), release={},
                                     evaluator=Path("mock"), runtime=Path("mock"))
    assert result["recomputed_aggregates"] is True
    assert calls["scores"] == []
    assert (stage / "per_table.jsonl").read_bytes() == before
    assert read_json(stage / "official_table.json")["aggregates"]["TEDS"]["all"] == 0.5


@pytest.mark.parametrize("gpu", [{"known": False, "processes": []}, {"known": True, "processes": [{"pid": 1}]}])
def test_busy_or_unknown_gpu_never_prepares_or_launches_recovery(history, monkeypatch, gpu):
    monkeypatch.setattr(state_monitor, "source_processes", lambda _source: [])
    monkeypatch.setattr(state_monitor, "gpu_processes", lambda: gpu)
    monkeypatch.setattr(recovery.subprocess, "run", lambda *_args, **_kwargs: pytest.fail("must not launch"))
    with pytest.raises(RuntimeError, match="busy or unknown"):
        recovery.retry_historical(history.settings)
    assert not (history.settings.project / "experiments/historical_recovery").exists()


def test_live_historical_process_never_prepares_or_launches_recovery(history, monkeypatch):
    monkeypatch.setattr(state_monitor, "source_processes", lambda _source: [{"pid": 123}])
    monkeypatch.setattr(state_monitor, "gpu_processes", lambda: pytest.fail("must not inspect GPU"))
    assert recovery.retry_historical(history.settings) is None


def test_process_state_is_rechecked_under_original_lock(history, monkeypatch):
    observations = iter([[], [{"pid": 456}]])
    monkeypatch.setattr(state_monitor, "source_processes", lambda _source: next(observations))
    monkeypatch.setattr(state_monitor, "gpu_processes", lambda: {"known": True, "processes": []})
    lock = history.runs / ".run_all.lock"
    before = (lock.read_bytes(), lock.stat().st_mtime_ns)
    assert recovery.retry_historical(history.settings) is None
    assert (lock.read_bytes(), lock.stat().st_mtime_ns) == before
    assert not (history.settings.project / "experiments/historical_recovery").exists()


def test_original_lock_already_held_prevents_recovery(history, monkeypatch):
    monkeypatch.setattr(state_monitor, "source_processes", lambda _source: [])
    monkeypatch.setattr(state_monitor, "gpu_processes", lambda: {"known": True, "processes": []})
    with (history.runs / ".run_all.lock").open("rb") as held:
        fcntl.flock(held.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(BlockingIOError):
            recovery.retry_historical(history.settings)
    assert not (history.settings.project / "experiments/historical_recovery").exists()


def test_gpu_state_is_rechecked_after_original_lock(history, monkeypatch):
    monkeypatch.setattr(state_monitor, "source_processes", lambda _source: [])
    observations = iter([{"known": True, "processes": []}, {"known": True, "processes": [{"pid": 100}]}])
    monkeypatch.setattr(state_monitor, "gpu_processes", lambda: next(observations))
    with pytest.raises(RuntimeError, match="busy or unknown"):
        recovery.retry_historical(history.settings)
    assert not (history.settings.project / "experiments/historical_recovery").exists()


def test_failed_subprocess_does_not_publish_ready_pointer_and_keeps_overlay(history, monkeypatch):
    monkeypatch.setattr(state_monitor, "source_processes", lambda _source: [])
    monkeypatch.setattr(state_monitor, "gpu_processes", lambda: {"known": True, "processes": []})

    def fail(command, **options):
        assert "--execute-overlay" in command and "--runs" in command
        assert options["pass_fds"] and options["env"]["PYTHONDONTWRITEBYTECODE"] == "1"
        assert Path(command[command.index("--runs") + 1]) != history.runs
        raise RuntimeError("synthetic worker failure")

    monkeypatch.setattr(recovery.subprocess, "run", fail)
    manifest = recovery._source_manifest(history.settings.source)
    with pytest.raises(RuntimeError, match="worker failure"):
        recovery.retry_historical(history.settings)
    assert not (history.settings.reports / "historical_recovery.json").exists()
    assert recovery._source_manifest(history.settings.source) == manifest
    assert len(list((history.settings.project / "experiments/historical_recovery").glob("*/source_snapshot.json"))) == 1


def test_successful_retry_publishes_verified_overlay_once_and_reuses_completed_worker(history, monkeypatch):
    manifest = recovery._source_manifest(history.settings.source)
    overlay = recovery._overlay(history.settings, manifest)
    _, calls = mock_native(history, monkeypatch)
    completed = recovery._execute_overlay(history.settings.source, overlay / "runs", overlay / "reports",
                                          history.settings.runtime)
    monkeypatch.setattr(state_monitor, "source_processes", lambda _source: [])
    monkeypatch.setattr(state_monitor, "gpu_processes", lambda: {"known": True, "processes": []})
    monkeypatch.setattr(recovery.subprocess, "run", lambda *_args, **_kwargs: pytest.fail("completed worker repeated"))
    assert recovery.retry_historical(history.settings) == completed
    assert read_json(history.settings.reports / "historical_recovery.json") == completed
    assert history.settings.historical == Path(completed["historical"])
    assert history.settings.final_report == Path(completed["final_report_path"])
    assert len(calls["reports"]) == 1
    assert recovery._source_manifest(history.settings.source) == manifest


def test_mock_worker_launch_holds_original_lock_and_publishes_only_after_validation(history, monkeypatch):
    mark_failed_score(history.work / "B2", history.refs[1]["id"])
    mock_native(history, monkeypatch)
    render = recovery.subprocess.run
    monkeypatch.setattr(state_monitor, "source_processes", lambda _source: [])
    monkeypatch.setattr(state_monitor, "gpu_processes", lambda: {"known": True, "processes": []})
    invocations = []

    def launch(command, **options):
        if "--execute-overlay" not in command:
            return render(command, **options)
        invocations.append(command)
        assert not (history.settings.reports / "historical_recovery.json").exists()
        assert options["pass_fds"] and options["env"]["HF_HUB_OFFLINE"] == "1"
        assert options["env"]["PYTHONDONTWRITEBYTECODE"] == "1"
        with (history.runs / ".run_all.lock").open("rb") as other, pytest.raises(BlockingIOError):
            fcntl.flock(other.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        values = {name: Path(command[command.index(f"--{name}") + 1])
                  for name in ("source", "runs", "reports", "runtime")}
        recovery._execute_overlay(**values)

    monkeypatch.setattr(recovery.subprocess, "run", launch)
    result = recovery.retry_historical(history.settings)
    assert result["status"] == "COMPLETED" and len(invocations) == 1
    assert read_json(history.settings.reports / "historical_recovery.json") == result


def test_changed_successful_original_score_blocks_completion(history, monkeypatch):
    overlay = recovery._overlay(history.settings, recovery._source_manifest(history.settings.source))
    work = overlay / "runs" / recovery.BENCHMARK
    rows = read_jsonl(work / "B0/per_table.jsonl")
    rows[0]["teds"] = 0.3
    write_jsonl(work / "B0/per_table.jsonl", rows)
    with pytest.raises(ValueError, match="Successful historical score changed"):
        recovery._verify_original_successes(history.settings.source, work)


def test_no_direct_recovery_into_original_run_directory(history):
    with pytest.raises(ValueError, match="isolated overlay"):
        recovery._execute_overlay(history.settings.source, history.runs, history.settings.source / "reports",
                                  history.settings.runtime)


def test_recovery_cannot_write_reports_outside_overlay(history):
    overlay = recovery._overlay(history.settings, recovery._source_manifest(history.settings.source))
    with pytest.raises(ValueError, match="isolated overlay"):
        recovery._execute_overlay(history.settings.source, overlay / "runs", history.settings.source / "reports",
                                  history.settings.runtime)


def test_full_cohort_contract_rejects_missing_and_duplicate_ids(tmp_path):
    path = tmp_path / "table_pairs.jsonl"
    rows = [reference(index) for index in range(665)]
    write_jsonl(path, rows)
    assert len(recovery._cohort(path)) == 665
    write_jsonl(path, rows[:-1])
    with pytest.raises(ValueError, match="665 distinct"):
        recovery._cohort(path)
    write_jsonl(path, rows[:-1] + [rows[0]])
    with pytest.raises(ValueError, match="665 distinct"):
        recovery._cohort(path)
