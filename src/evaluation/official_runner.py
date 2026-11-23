"""Run the untouched historical official scorer in its pinned CPU container."""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

from orchestration.settings import Settings
from orchestration.storage import atomic_json, file_hash, object_hash, read_json, read_jsonl

RUNTIME_DIGEST = "sha256:6116ad72172e763b5c43e963d5efebf2093f2362b975f58156ce4f6c9142e617"


def command(settings: Settings, work: Path) -> list[str]:
    release = read_json(settings.runtime / "release.json")
    if release["official_manifest"] != RUNTIME_DIGEST:
        raise ValueError("Official runtime digest differs")
    blobs = settings.runtime / "oci/blobs/sha256"
    manifest = read_json(blobs / RUNTIME_DIGEST.split(":")[1])
    image = read_json(blobs / manifest["config"]["digest"].split(":")[1])["config"]
    rootfs = settings.runtime / "bundle/rootfs"
    environment = dict(item.split("=", 1) for item in image["Env"])
    environment.update(PYTHONPATH="/workspace:/tmp/bs/src:/tmp/source/src:/tmp/source",
                       HOME="/tmp", LANG="C.UTF-8", OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
                       MKL_NUM_THREADS="1", HF_HUB_OFFLINE="1", HF_DATASETS_OFFLINE="1",
                       HF_HOME="/tmp/cache", MPLCONFIGDIR="/tmp/matplotlib", XDG_CACHE_HOME="/tmp/cache")
    result = ["bwrap", "--unshare-net", "--die-with-parent", "--clearenv", "--ro-bind", str(rootfs), "/",
              "--dev", "/dev", "--proc", "/proc", "--tmpfs", "/tmp"]
    for source, target, writable in ((settings.source / "data/omnidocbench/evaluator", "/workspace", False),
                                     (settings.source, "/tmp/source", False),
                                     (settings.project, "/tmp/bs", False), (work, "/tmp/results", True)):
        result.extend(("--bind" if writable else "--ro-bind", str(source), target))
    for key, value in environment.items():
        result.extend(("--setenv", key, value))
    historical = settings.historical
    mapped = (Path("/tmp/bs") / historical.relative_to(settings.project)
              if historical.is_relative_to(settings.project)
              else Path("/tmp/source") / historical.relative_to(settings.source))
    return result + ["--chdir", "/tmp/results", image["Entrypoint"][0], "/tmp/bs/scripts/score_predictions.py",
                     "--baseline-root", str(mapped / "B0")]


def run_scores(settings: Settings, directory: Path, pairs: list[dict]) -> None:
    """Commit subset scores atomically and retry only uncommitted/failed rows."""
    from orchestration.recovery import _merge_scores, _resume_merge, _score_selection

    scoring = directory / "scoring"
    _resume_merge(scoring)
    for _ in range(settings.max_attempts):
        selected, old, lines = _score_selection(scoring, pairs, set())
        if not selected:
            return
        attempts = scoring / "attempts" / object_hash(selected)[:24]
        attempts.mkdir(parents=True, exist_ok=True)
        previous = list(attempts.glob("attempt-*"))
        subset = attempts / f"attempt-{len(previous) + 1:04d}"
        subset.mkdir()
        payload = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in selected)
        (subset / "table_pairs.jsonl").write_text(payload)
        try:
            with (directory / "official.log").open("ab") as log:
                subprocess.run(command(settings, subset), check=True, stdout=log, stderr=subprocess.STDOUT)
        except subprocess.CalledProcessError:
            continue
        baseline = subset / "baseline.jsonl"
        saved_baseline = scoring / "baseline.jsonl"
        if saved_baseline.exists() and file_hash(saved_baseline) != file_hash(baseline):
            raise ValueError("Baseline scorer reference changed between retries")
        if not saved_baseline.exists():
            from orchestration.recovery import _atomic_bytes
            _atomic_bytes(saved_baseline, baseline.read_bytes())
        failures = _merge_scores(scoring, pairs, subset, selected, old, lines)
        if not failures:
            return
    raise RuntimeError("Official scorer still has failed or uncommitted samples after bounded retries")


def score_method(settings: Settings, method: str, ids: list[str]) -> dict:
    directory = settings.output / f"method_{method.lower()}"
    summary = directory / "metrics.json"
    if summary.exists():
        return read_json(summary)
    predictions = {row["table_id"]: row for row in
                   (read_json(path) for path in (directory / "predictions").glob("*.json"))}
    if set(predictions) != set(ids):
        raise ValueError("Scoring requires the same 665 table IDs for every method")
    references = read_jsonl(settings.historical / "B0/table_pairs.jsonl")
    scoring = directory / "scoring"
    scoring.mkdir(parents=True, exist_ok=True)
    # Ground truth is introduced only in this post-generation scoring subprocess.
    pairs = []
    for reference in references:
        record = predictions[reference["id"]]
        pairs.append({**reference, "raw": record["prediction"], "prediction_html": record["html"],
                      "seconds": record["latency"], "output_tokens": record["output_tokens"],
                      "finish_reason": record["finish_reason"], "cross_attention_flops": 0})
    payload = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in pairs)
    pairs_path = scoring / "table_pairs.jsonl"
    if pairs_path.exists() and pairs_path.read_text() != payload:
        raise ValueError("Existing scoring input differs")
    if not pairs_path.exists():
        from orchestration.recovery import _atomic_bytes
        _atomic_bytes(pairs_path, payload.encode())
    run_scores(settings, directory, pairs)
    official = read_json(scoring / "official_table.json")
    diagnostics = official["diagnostics"]
    if diagnostics["sample_count"] != len(ids) or diagnostics["error_case_count"] or diagnostics["timeout_case_count"]:
        raise RuntimeError("Official scorer had failures; see preserved official.log and retry")
    from .metrics import summarize
    from .paired_analysis import paired_comparison
    scores = read_jsonl(scoring / "per_table.jsonl")
    baseline = read_jsonl(scoring / "baseline.jsonl")
    baseline_index = {row["id"]: row for row in baseline}
    rows = []
    for score in scores:
        record = predictions[score["id"]]
        row = {**score, "strict_em": score["exact"], "latency": record["latency"],
               "incremental_latency": record["incremental_latency"], "calls": record["calls"],
               "additional_calls": record["additional_calls"], "changed": record["changed"],
               "failure": record.get("failure", False),
               "status": record["status"], "delta_teds": score["teds"] - baseline_index[score["id"]]["teds"]}
        rows.append(row)
        path = directory / "samples" / record["sample_filename"]
        scored = {**record, "metrics": row}
        if path.exists():
            if read_json(path) != scored:
                raise ValueError("Existing scored sample differs from resumed scoring")
        else:
            atomic_json(path, scored, immutable=True)
    aggregate = {"method": method, "metrics": summarize(rows), "baseline": summarize(baseline),
                 "paired": paired_comparison(baseline, rows), "official": official,
                 "scoring_input_hash": file_hash(pairs_path)}
    atomic_json(summary, aggregate, immutable=True)
    return aggregate
