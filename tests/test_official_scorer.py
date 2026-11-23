"""Pinned CPU scorer smoke test and failed-sample-only restart behavior."""

import json
import subprocess
from pathlib import Path

import pytest

from evaluation import official_runner
from orchestration.settings import Settings
from orchestration.state_monitor import now
from orchestration.storage import atomic_json, file_hash, read_json, read_jsonl


def test_pinned_official_runtime_scores_synthetic_html_without_dataset_reads(tmp_path):
    project = Path(__file__).resolve().parents[1]
    config = project / "configs/local.yaml"
    if not config.exists():
        pytest.skip("Pinned official CPU container is configured only on the evaluation machine")
    settings = Settings.load(config)
    result = subprocess.run(official_runner.command(settings, tmp_path) + ["--self-test"],
                            capture_output=True, text=True, timeout=60, check=True)
    assert '"synthetic_scorer_test": "passed"' in result.stdout
    diagnostics = read_json(tmp_path / "official_table.json")["diagnostics"]
    assert diagnostics["sample_count"] == 1
    assert diagnostics["error_case_count"] == diagnostics["timeout_case_count"] == 0
    atomic_json(settings.project / "reports/development/official_scorer_cpu_test.json", {
        "timestamp": now(), "passed": True, "synthetic_only": True,
        "diagnostics": diagnostics, "output_hash": file_hash(tmp_path / "per_table.jsonl"),
    })


def write_rows(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def test_only_failed_scorer_rows_are_retried_and_success_bytes_survive(tmp_path, monkeypatch):
    settings = Settings(tmp_path, tmp_path / "source", tmp_path / "runtime", Path("python"))
    directory = settings.output / "method_a"
    scoring = directory / "scoring"
    pairs = [{"id": f"synthetic#{index}", "page": "synthetic", "annotation_id": index,
              "slices": [], "seconds": 1.0, "visual_tokens": 64, "output_tokens": 8,
              "finish_reason": "stop", "cross_attention_flops": 0} for index in range(2)]
    write_rows(scoring / "table_pairs.jsonl", pairs)
    observed = []
    first_success = []
    monkeypatch.setattr(official_runner, "command", lambda _settings, work: [str(work)])

    def fake_scorer(command, **options):
        work = Path(command[0])
        selected = read_jsonl(work / "table_pairs.jsonl")
        observed.append([pair["id"] for pair in selected])
        fail = len(observed) == 1
        rows = [{**pair, "teds": 0.0 if fail and pair["annotation_id"] == 1 else 1.0,
                 "teds_s": 1.0, "exact": 1, "char_edits": 0, "char_count": 1,
                 "pred_structure": {}, "gold_structure": {}, "structure_em": 1} for pair in selected]
        write_rows(work / "per_table.jsonl", rows)
        if fail:
            first_success.append((work / "per_table.jsonl").read_bytes().splitlines(keepends=True)[0])
        atomic_json(work / "official_table.json", {
            "aggregates": {name: {"all": 1.0} for name in ("TEDS", "TEDS_structure_only")},
            "diagnostics": {"stage": "TEDS", "workers": 6, "timeout_sec": 120,
                            "sample_count": len(rows), "timeout_case_count": int(fail),
                            "timeout_cases": [{"img_id": "synthetic", "gt_idx": 1}] if fail else [],
                            "error_case_count": 0, "error_cases": []},
        })
        write_rows(work / "baseline.jsonl", [{"synthetic": True}])

    monkeypatch.setattr(official_runner.subprocess, "run", fake_scorer)
    official_runner.run_scores(settings, directory, pairs)
    assert observed == [["synthetic#0", "synthetic#1"], ["synthetic#1"]]
    assert (scoring / "per_table.jsonl").read_bytes().splitlines(keepends=True)[0] == first_success[0]
    assert read_json(scoring / "official_table.json")["diagnostics"]["timeout_case_count"] == 0
    official_runner.run_scores(settings, directory, pairs)
    assert len(observed) == 2
