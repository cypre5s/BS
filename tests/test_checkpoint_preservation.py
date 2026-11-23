"""Recovery retains historical bytes and immutable success records."""

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from orchestration.recovery import commit_prediction, completed_record, sample_name
from orchestration.settings import Settings
from orchestration.storage import atomic_json, file_hash, read_json


def record(table_id="fixture", identity="configuration-one"):
    return {"table_id": table_id, "identity_hash": identity, "status": "completed",
            "prediction": "<fcel>A<nl>", "html": "<table><tr><td>A</td></tr></table>"}


def test_success_survives_attempted_overwrite_and_resume(tmp_path):
    original = tmp_path / "historical/model.safetensors"
    original.parent.mkdir()
    original.write_bytes(b"synthetic frozen checkpoint")
    checkpoint_hash = file_hash(original)
    directory = tmp_path / "method_a"
    result = record()
    commit_prediction(directory, result)
    success = directory / "predictions" / sample_name(result["table_id"])
    saved = success.read_bytes()
    assert completed_record(directory, "fixture", "configuration-one") == result
    with pytest.raises(FileExistsError):
        commit_prediction(directory, result | {"prediction": "changed"})
    assert success.read_bytes() == saved
    assert file_hash(original) == checkpoint_hash


@pytest.mark.parametrize("changed", [
    {"identity_hash": "different"}, {"table_id": "another"}, {"status": "failed"},
])
def test_incompatible_resume_is_rejected(tmp_path, changed):
    path = tmp_path / "predictions" / sample_name("fixture")
    atomic_json(path, record() | changed)
    with pytest.raises(ValueError, match="provenance"):
        completed_record(tmp_path, "fixture", "configuration-one")


def test_failed_attempts_preserved_after_success(tmp_path):
    attempt = tmp_path / "attempts/fixture-attempt1.json"
    atomic_json(attempt, {"table_id": "fixture", "status": "failed", "error": "synthetic timeout"}, immutable=True)
    before = attempt.read_bytes()
    assert completed_record(tmp_path, "fixture", "configuration-one") is None
    commit_prediction(tmp_path, record())
    assert completed_record(tmp_path, "fixture", "configuration-one")["status"] == "completed"
    assert attempt.read_bytes() == before


def test_concurrent_immutable_commit_has_one_winner(tmp_path):
    def save(index):
        try:
            commit_prediction(tmp_path, record() | {"writer": index})
            return True
        except FileExistsError:
            return False

    with ThreadPoolExecutor(max_workers=4) as pool:
        assert sum(pool.map(save, range(4))) == 1
    assert read_json(tmp_path / "predictions" / sample_name("fixture"))["writer"] in range(4)
    assert not list(tmp_path.rglob("*.tmp"))


def test_invalid_json_write_preserves_previous_state(tmp_path):
    path = tmp_path / "state.json"
    atomic_json(path, {"status": "RUNNING_A"})
    before = path.read_bytes()
    with pytest.raises(ValueError):
        atomic_json(path, {"value": float("nan")})
    assert path.read_bytes() == before
    assert json.loads(path.read_text())["status"] == "RUNNING_A"
    assert not list(tmp_path.glob("*.tmp"))


def test_partial_scoring_resume_reuses_immutable_successes(tmp_path, monkeypatch):
    from evaluation import official_runner

    settings = Settings(tmp_path / "project", tmp_path / "source", tmp_path / "runtime", Path("python"))
    directory = settings.output / "method_a"
    scoring = directory / "scoring"
    scoring.mkdir(parents=True)
    ids = ["fixture-0", "fixture-1"]
    rows = []
    for index, table_id in enumerate(ids):
        row = {"id": table_id, "page": "fixture-page", "teds": 0.8, "teds_s": 1.0,
               "exact": 0, "strict_em": 0, "structure_em": 1, "char_count": 10,
               "char_edits": 2, "latency": 1.0, "calls": 1, "changed": False,
               "delta_teds": 0.0, "additional_calls": 0, "incremental_latency": 0.0,
               "status": "completed"}
        rows.append(row)
        commit_prediction(directory, record(table_id) | {
            "sample_filename": sample_name(table_id), "latency": 1.5, "incremental_latency": 0.5,
            "calls": 2, "additional_calls": 1, "changed": False, "output_tokens": 5,
            "finish_reason": "stop", "order": index,
        })
    references = settings.historical / "B0/table_pairs.jsonl"
    references.parent.mkdir(parents=True)
    payload = "".join(json.dumps(row) + "\n" for row in rows)
    references.write_text(payload)
    (scoring / "baseline.jsonl").write_text(payload)
    (scoring / "per_table.jsonl").write_text(payload)
    atomic_json(scoring / "official_table.json", {"diagnostics": {
        "sample_count": 2, "error_case_count": 0, "timeout_case_count": 0,
    }})
    checkpoint = settings.source / "synthetic.safetensors"
    checkpoint.write_bytes(b"frozen original model")
    checkpoint_hash = file_hash(checkpoint)
    monkeypatch.setattr(official_runner, "run_scores", lambda *_args, **_kwargs: None)
    real_commit = official_runner.atomic_json
    failure_target = directory / "samples" / sample_name(ids[1])

    def interrupted_commit(path, value, **options):
        if path == failure_target:
            raise RuntimeError("simulated scoring interruption")
        return real_commit(path, value, **options)

    monkeypatch.setattr(official_runner, "atomic_json", interrupted_commit)
    with pytest.raises(RuntimeError, match="simulated"):
        official_runner.score_method(settings, "A", ids)
    saved = directory / "samples" / sample_name(ids[0])
    original = saved.read_bytes()
    assert not (directory / "metrics.json").exists()
    monkeypatch.setattr(official_runner, "atomic_json", real_commit)
    result = official_runner.score_method(settings, "A", ids)
    assert result["metrics"]["count"] == 2
    assert saved.read_bytes() == original
    assert failure_target.exists()
    assert file_hash(checkpoint) == checkpoint_hash
