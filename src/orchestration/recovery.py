"""Immutable success records and append-only attempts make restarts idempotent."""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import runpy
import shutil
import subprocess
import sys
from math import isclose, isfinite
from pathlib import Path
from tempfile import NamedTemporaryFile, mkdtemp
from time import perf_counter
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, cast

from .storage import atomic_json, file_hash, object_hash, read_json, read_jsonl

if TYPE_CHECKING:
    from .settings import Settings


def sample_name(table_id: str) -> str:
    return hashlib.sha256(table_id.encode()).hexdigest() + ".json"


def completed_record(directory: Path, table_id: str, identity_hash: str) -> dict | None:
    path = directory / "predictions" / sample_name(table_id)
    if not path.exists():
        return None
    record = read_json(path)
    if record["table_id"] != table_id or record["identity_hash"] != identity_hash or record["status"] != "completed":
        raise ValueError("Cached prediction has incompatible provenance")
    return record


def commit_prediction(directory: Path, record: dict) -> None:
    path = directory / "predictions" / sample_name(record["table_id"])
    atomic_json(path, record, immutable=True)


GROUPS = ("B0", "B1", "B2", "B3")
TABLE_COUNT = 665
BENCHMARK = Path("omnidocbench/table_recognition")


def _atomic_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with NamedTemporaryFile(dir=path.parent, delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _copy_isolated(source: Path, target: Path) -> None:
    """Reflink when supported, otherwise copy; never share a writable inode."""
    if source.is_symlink() or not source.is_file():
        raise ValueError(f"Historical cache must be a regular file: {source}")
    target.parent.mkdir(parents=True, exist_ok=True)
    with source.open("rb") as incoming, target.open("xb") as outgoing:
        try:
            fcntl.ioctl(outgoing.fileno(), 0x40049409, incoming.fileno())  # Linux FICLONE
        except OSError:
            outgoing.seek(0)
            outgoing.truncate()
            shutil.copyfileobj(incoming, outgoing, length=1024 * 1024)
    shutil.copystat(source, target)


def _source_manifest(source: Path) -> dict[str, str]:
    runs = source / "runs/xmod"
    files = [runs / "full" / group / "summary.json" for group in GROUPS]
    files += [runs / "full" / group / "best.pt" for group in GROUPS[1:]]
    cached = runs / BENCHMARK
    if not cached.is_dir():
        raise ValueError("Historical table-recognition cache is absent")
    for path in cached.rglob("*"):
        if path.is_symlink():
            raise ValueError(f"Symlink in historical cache is unsupported: {path}")
        if path.is_file():
            files.append(path)
    files += [source / name for name in (
        "scripts/xmod_evaluate.py", "scripts/xmod_run_all.py",
        "src/xiaomi_ocr_xmod/evaluation.py", "src/xiaomi_ocr_xmod/reporting.py",
    )]
    return {str(path.relative_to(source)): file_hash(path) for path in sorted(files)}


def _overlay(settings: Any, manifest: dict[str, str]) -> Path:
    parent = settings.project / "experiments/historical_recovery"
    parent.mkdir(parents=True, exist_ok=True)
    root = parent / object_hash(manifest)[:24]
    plan = {"schema": 1, "source": str(settings.source), "source_manifest": manifest,
            "source_manifest_hash": object_hash(manifest)}
    if root.exists():
        if read_json(root / "source_snapshot.json") != plan:
            raise ValueError("Recovery overlay belongs to a different source snapshot")
        return root
    temporary = Path(mkdtemp(prefix=".copying-", dir=parent))
    try:
        for relative, expected in manifest.items():
            original = settings.source / relative
            path = Path(relative)
            if not path.is_relative_to("runs/xmod"):
                continue
            copied = temporary / "runs" / path.relative_to("runs/xmod")
            _copy_isolated(original, copied)
            if file_hash(copied) != expected:
                raise ValueError(f"Historical file changed during copying: {relative}")
        atomic_json(temporary / "source_snapshot.json", plan)
        temporary.rename(root)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return root


def _cohort(path: Path, *, count: int = TABLE_COUNT) -> list[dict]:
    rows = read_jsonl(path)
    ids = [row["id"] for row in rows]
    if len(ids) != count or len(set(ids)) != count:
        raise ValueError(f"Historical B0 must contain exactly {count} distinct table IDs")
    return rows


def _prediction_ok(row: dict) -> bool:
    return (row.get("finish_reason") in {"stop", "length"}
            and isinstance(row.get("prediction_html"), str)
            and isinstance(row.get("output_tokens"), int) and row["output_tokens"] > 0
            and not row.get("error"))


def _cache_records(stage: Path, references: list[dict]) -> tuple[dict, dict, set[str]]:
    filenames = {f"{Path(row['page']).stem}_{row['annotation_id']}.json": row["id"] for row in references}
    if len(filenames) != len(references):
        raise ValueError("Historical prediction filenames are ambiguous")
    records, paths, failed = {}, {}, set()
    for path in sorted((stage / "pred_tables").glob("*.json")):
        if path.name not in filenames:
            raise ValueError(f"Unexpected historical prediction file: {path.name}")
        table_id = filenames[path.name]
        paths[table_id] = path
        try:
            row = read_json(path)
        except (ValueError, UnicodeDecodeError):
            failed.add(table_id)
            continue
        if not isinstance(row, dict) or "id" not in row:
            failed.add(table_id)
            continue
        if row["id"] != table_id:
            raise ValueError(f"Prediction identity does not match filename: {path.name}")
        if _prediction_ok(row):
            records[table_id] = row
        else:
            failed.add(table_id)
    return records, paths, failed


def _quarantine(stage: Path, path: Path) -> None:
    if not path.exists():
        return
    target = stage / "recovery_discarded" / file_hash(path) / path.relative_to(stage)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if file_hash(target) != file_hash(path):
            raise ValueError("Historical recovery quarantine collision")
        path.unlink()
    else:
        path.replace(target)


def _index(rows: list[dict], ids: set[str], label: str) -> dict[str, dict]:
    result = {row["id"]: row for row in rows}
    if len(result) != len(rows) or set(result) - ids:
        raise ValueError(f"Duplicate or unexpected IDs in {label}")
    return result


def _partial_pairs(path: Path, ids: set[str]) -> tuple[dict[str, dict], bool]:
    """Keep complete input records from an interrupted non-atomic historical write."""
    rows, damaged = [], False
    for line in path.read_bytes().splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except (ValueError, UnicodeDecodeError):
            damaged = True
            continue
        if not isinstance(row, dict) or "id" not in row:
            damaged = True
            continue
        rows.append(row)
    return _index(rows, ids, path.name), damaged


def _prepare_predictions(stage: Path, references: list[dict]) -> tuple[set[str], dict[str, str], bool]:
    """Remove only failed cache entries in the overlay, retaining success bytes."""
    records, paths, failed = _cache_records(stage, references)
    ids = {row["id"] for row in references}
    changed = ids - set(records)
    preserved = {str(path.relative_to(stage)): file_hash(path) for key, path in paths.items()
                 if key in records}
    receipt_path, pairs_path = stage / "predictions.json", stage / "table_pairs.jsonl"
    dirty_path = stage / "recovery_changed_predictions.json"
    needs_rebuild = bool(changed) or not receipt_path.is_file() or not pairs_path.is_file()
    if pairs_path.exists():
        pairs, damaged = _partial_pairs(pairs_path, ids)
        changed.update(ids - set(pairs))
        for table_id, row in records.items():
            if table_id in pairs and any(pairs[table_id].get(key) != value for key, value in row.items()):
                changed.add(table_id)
        needs_rebuild |= damaged or set(pairs) != ids or bool(changed)
    elif (stage / "per_table.jsonl").exists() and not dirty_path.exists():
        changed.update(ids)  # No committed inputs: existing scores cannot be certified as successes.
    if receipt_path.exists():
        try:
            receipt = read_json(receipt_path)
        except (ValueError, UnicodeDecodeError):
            needs_rebuild = True
        else:
            if not isinstance(receipt, dict) or receipt.get("identity") != read_json(stage / "identity.json"):
                raise ValueError("Historical prediction receipt has incompatible provenance")
            needs_rebuild |= not pairs_path.exists() or receipt.get("table_pairs") != file_hash(pairs_path)
    if dirty_path.exists():
        changed.update(read_json(dirty_path)["ids"])
    # Commit this before moving any failed file or derived receipt so an interrupted
    # preparation still knows which otherwise-valid score rows have stale inputs.
    atomic_json(dirty_path, {"ids": sorted(changed)})
    for table_id in failed:
        _quarantine(stage, paths[table_id])
    if needs_rebuild:
        _quarantine(stage, receipt_path)
        # The original predictor rewrites its derived input file after skipping cached successes.
        _quarantine(stage, pairs_path)
    return changed, preserved, needs_rebuild


def _score_lines(path: Path, ids: set[str]) -> tuple[dict[str, dict], dict[str, bytes]]:
    if not path.exists():
        return {}, {}
    rows, lines = [], {}
    for line in path.read_bytes().splitlines(keepends=True):
        if not line.strip():
            continue
        row = json.loads(line)
        rows.append(row)
        lines[row["id"]] = line
    return _index(rows, ids, path.name), lines


def _failed_scores(official: dict, pairs: list[dict]) -> set[str]:
    diagnostic = official["diagnostics"]
    if (diagnostic.get("sample_count") != len(pairs) or diagnostic.get("workers") != 6
            or diagnostic.get("timeout_sec") != 120 or diagnostic.get("stage") != "TEDS"):
        raise ValueError("Unsupported official scorer coverage or configuration")
    if set(official["aggregates"]) != {"TEDS", "TEDS_structure_only"} or any(
        set(value) != {"all"} for value in official["aggregates"].values()
    ):
        raise ValueError("Unsupported official aggregate grouping")
    ids = {row["id"] for row in pairs}
    failures = set()
    for kind in ("timeout", "error"):
        cases = diagnostic.get(f"{kind}_cases", [])
        if diagnostic.get(f"{kind}_case_count") != len(cases):
            raise ValueError("Official scorer failure count disagrees with its case list")
        for case in cases:
            annotation = case.get("gt_idx")
            if isinstance(annotation, list) and len(annotation) == 1:
                annotation = annotation[0]
            table_id = f"{case.get('img_id')}#{annotation}"
            if table_id not in ids:
                raise ValueError(f"Cannot identify official scorer failure: {case}")
            failures.add(table_id)
    return failures


def _score_ok(row: dict, pair: dict) -> bool:
    numerical = ("teds", "teds_s", "exact", "char_edits", "char_count", "seconds",
                 "visual_tokens", "output_tokens", "cross_attention_flops")
    if any(not isinstance(row.get(key), (int, float)) or not isfinite(row[key]) or row[key] < 0
           for key in numerical):
        return False
    if row["teds"] > 1 or row["teds_s"] > 1 or row["exact"] not in (0, 1):
        return False
    if not all(isinstance(row.get(key), dict) for key in ("pred_structure", "gold_structure")):
        return False
    return all(row.get(key) == pair.get(key) for key in (
        "page", "slices", "seconds", "visual_tokens", "output_tokens", "finish_reason", "cross_attention_flops"
    ))


def _score_selection(stage: Path, pairs: list[dict], changed: set[str]) -> tuple[list[dict], dict, dict]:
    ids = {row["id"] for row in pairs}
    old: dict[str, dict]
    lines: dict[str, bytes]
    official_path = stage / "official_table.json"
    official = None
    if official_path.exists():
        try:
            official = read_json(official_path)
        except (ValueError, UnicodeDecodeError):
            _quarantine(stage, official_path)
    if official is None:
        # score_tables writes per_table before official_table. Without the latter,
        # these rows are uncommitted, even if their numerical values look plausible.
        _quarantine(stage, stage / "per_table.jsonl")
        old, lines = {}, {}
        failed = set()
    else:
        old, lines = _score_lines(stage / "per_table.jsonl", ids)
        failed = _failed_scores(official, pairs)
    todo = changed | failed | (ids - set(old))
    todo.update(pair["id"] for pair in pairs if pair["id"] in old and not _score_ok(old[pair["id"]], pair))
    return [row for row in pairs if row["id"] in todo], old, lines


def _resume_merge(stage: Path) -> None:
    marker = stage / "recovery_merge_pending.json"
    if not marker.exists():
        return
    pending = read_json(marker)
    if pending["table_pairs"] != file_hash(stage / "table_pairs.jsonl"):
        raise ValueError("Prediction inputs changed during historical score merge")
    directory = stage / "recovery_merges" / pending["generation"]
    for name in ("per_table.jsonl", "official_table.json"):
        path = directory / name
        if file_hash(path) != pending["hashes"][name]:
            raise ValueError("Historical score merge staging was corrupted")
        _atomic_bytes(stage / name, path.read_bytes())
    dirty = stage / "recovery_changed_predictions.json"
    if dirty.exists():
        remaining = set(read_json(dirty)["ids"]) - set(pending["rescored_ids"])
        atomic_json(dirty, {"ids": sorted(remaining)})
    marker.unlink()


def _merge_scores(stage: Path, pairs: list[dict], subset: Path, selected: list[dict],
                  old: dict, old_lines: dict) -> set[str]:
    ids = {row["id"] for row in pairs}
    selected_ids = {row["id"] for row in selected}
    replacement, replacement_lines = _score_lines(subset / "per_table.jsonl", selected_ids)
    if set(replacement) != selected_ids:
        raise ValueError("Official retry did not produce every selected score")
    scored = read_json(subset / "official_table.json")
    failures = _failed_scores(scored, selected)
    if any(not _score_ok(replacement[row["id"]], row) for row in selected):
        raise ValueError("Official retry produced invalid per-table records")
    merged = old | replacement
    if set(merged) != ids:
        raise ValueError("Merged historical score coverage differs")
    lines = old_lines | replacement_lines
    payload = b"".join(line if line.endswith(b"\n") else line + b"\n"
                       for row in pairs for line in (lines[row["id"]],))
    official = {
        "aggregates": {name: {"all": sum(merged[row["id"]][key] for row in pairs) / len(pairs)}
                       for name, key in (("TEDS", "teds"), ("TEDS_structure_only", "teds_s"))},
        "diagnostics": scored["diagnostics"] | {"sample_count": len(pairs)},
    }
    generation = object_hash({"pairs": file_hash(stage / "table_pairs.jsonl"),
                              "scores": hashlib.sha256(payload).hexdigest(), "official": official})
    directory = stage / "recovery_merges" / generation
    _atomic_bytes(directory / "per_table.jsonl", payload)
    atomic_json(directory / "official_table.json", official)
    preserved = {table_id: object_hash(row) for table_id, row in old.items() if table_id not in selected_ids}
    receipt = {"table_pairs": file_hash(stage / "table_pairs.jsonl"), "generation": generation,
               "reused_ids": sorted(preserved), "reused_record_hashes": preserved,
               "rescored_ids": sorted(selected_ids), "remaining_failed_ids": sorted(failures),
               "subset": str(subset),
               "hashes": {name: file_hash(directory / name)
                          for name in ("per_table.jsonl", "official_table.json")}}
    atomic_json(directory / "receipt.json", receipt)
    atomic_json(stage / "recovery_merge_pending.json", receipt)
    _resume_merge(stage)
    atomic_json(subset / "merged.json", {"generation": generation, "remaining_failed_ids": sorted(failures)})
    return failures


def _repair_scores(namespace: dict, stage: Path, changed: set[str], *, data: Path,
                   release: dict, evaluator: Path, runtime: Path) -> dict:
    pairs = read_jsonl(stage / "table_pairs.jsonl")
    selected, old, old_lines = _score_selection(stage, pairs, changed)
    if not selected:
        official = read_json(stage / "official_table.json")
        corrected = {name: {"all": sum(old[row["id"]][key] for row in pairs) / len(pairs)}
                     for name, key in (("TEDS", "teds"), ("TEDS_structure_only", "teds_s"))}
        differs = any(not isinstance(official["aggregates"][name]["all"], (int, float)) or not isclose(
            official["aggregates"][name]["all"], value["all"], abs_tol=1e-12, rel_tol=0
        ) for name, value in corrected.items())
        if differs:
            atomic_json(stage / "official_table.json", official | {"aggregates": corrected})
        return {"reused_scores": len(old), "rescored_ids": [], "recomputed_aggregates": differs}
    identity = object_hash({"pairs": selected})[:24]
    attempts = stage / "recovery_scoring" / identity
    attempts.mkdir(parents=True, exist_ok=True)
    previous = sorted(attempts.glob("attempt-*"))
    if previous and not (previous[-1] / stage.name / "merged.json").exists():
        work = previous[-1]
    else:
        work = attempts / f"attempt-{len(previous) + 1:04d}"
    subset = work / stage.name
    subset.mkdir(parents=True, exist_ok=True)
    (work / "_cache").mkdir(exist_ok=True)
    namespace["write_jsonl"](subset / "table_pairs.jsonl", selected)
    namespace["score_group"](stage.name, subset, data, release, work=work, evaluator=evaluator,
                             runtime=runtime, launcher="bwrap", evaluation="table-recognition")
    failures = _merge_scores(stage, pairs, subset, selected, old, old_lines)
    if failures:
        raise RuntimeError(f"Official historical scorer still failed on {len(failures)} table(s)")
    return {"reused_scores": len(pairs) - len(selected), "rescored_ids": [row["id"] for row in selected]}


def _verify_preserved(stage: Path, preserved: dict[str, str]) -> None:
    for relative, checksum in preserved.items():
        if file_hash(stage / relative) != checksum:
            raise ValueError(f"Successful historical prediction changed: {stage.name}/{relative}")


def _verify_original_successes(source: Path, work: Path) -> dict:
    """Validate preservation against the original snapshot, including across restarts."""
    original = source / "runs/xmod" / BENCHMARK
    references = _cohort(original / "B0/table_pairs.jsonl")
    ids = {row["id"] for row in references}
    result = {}
    for group in GROUPS:
        before, after = original / group, work / group
        predictions, paths, _ = _cache_records(before, references)
        preserved = {str(paths[key].relative_to(before)): file_hash(paths[key]) for key in predictions}
        _verify_preserved(after, preserved)
        scores: dict[str, dict] = {}
        official = None
        if (before / "official_table.json").exists():
            try:
                official = read_json(before / "official_table.json")
            except (ValueError, UnicodeDecodeError):
                pass  # An incomplete diagnostic file never committed these scores.
        if official is not None:
            scores, _ = _score_lines(before / "per_table.jsonl", ids)
        current, _ = _score_lines(after / "per_table.jsonl", ids)
        successful_scores = {}
        if scores and official is not None and (before / "table_pairs.jsonl").exists():
            pairs, _ = _partial_pairs(before / "table_pairs.jsonl", ids)
            failed = _failed_scores(official, references)
            for pair in pairs.values():
                key = pair["id"]
                if (key not in scores or key in failed or key not in predictions or not _score_ok(scores[key], pair)
                        or any(pair.get(field) != value for field, value in predictions[key].items())):
                    continue
                if current.get(key) != scores[key]:
                    raise ValueError(f"Successful historical score changed: {group}/{key}")
                successful_scores[key] = object_hash(scores[key])
        result[group] = {"predictions": preserved, "scores": successful_scores}
    return result


def _validate_overlay(work: Path, report: Path) -> dict:
    from .evaluation_gate import validate_receipts
    from .state_monitor import inspect_group

    ids = [row["id"] for row in _cohort(work / "B0/table_pairs.jsonl")]
    for group in GROUPS:
        inspected = inspect_group(work / group, ids)
        if inspected["evaluation_status"] != "COMPLETED":
            raise ValueError(f"Recovered {group} is incomplete: {inspected['errors']}")
    # The shared validator uses only these two read-only Settings properties.
    view = cast("Settings", SimpleNamespace(historical=work, final_report=report))
    return validate_receipts(view)


def _execute_overlay(source: Path, runs: Path, reports: Path, runtime: Path) -> dict:
    """Execute the unchanged historical functions with every writable path in the overlay."""
    started = perf_counter()
    work, overlay = runs / BENCHMARK, runs.parent
    if (runs.resolve() == (source / "runs/xmod").resolve() or runs.name != "runs"
            or reports.resolve() != (overlay / "reports").resolve()
            or not (overlay / "source_snapshot.json").exists()):
        raise ValueError("Recovery requires a prepared isolated overlay")
    snapshot = read_json(overlay / "source_snapshot.json")
    if snapshot["source"] != str(source) or snapshot["source_manifest"] != _source_manifest(source):
        raise ValueError("Historical source changed before recovery")
    namespace = runpy.run_path(str(source / "scripts/xmod_evaluate.py"), run_name="historical_recovery_driver")
    full = namespace["final_checkpoints"](runs)
    model, data = source / "models/xiaomi-ocr-0", source / "data/omnidocbench/benchmark"
    evaluator = source / "data/omnidocbench/evaluator"
    namespace["check_evaluator"](evaluator)
    frozen_dir = work / "frozen_models"
    frozen = namespace["freeze_models"](model, runs, full, frozen_dir)
    pages, release = namespace["load_pages"](data)
    references = list(namespace["table_references"](pages))
    baseline = _cohort(work / "B0/table_pairs.jsonl")
    if [row["id"] for row in references] != [row["id"] for row in baseline]:
        raise ValueError("Pinned reference cohort differs from the historical B0 cohort")
    details = {}
    for group in GROUPS:
        stage = work / group
        _resume_merge(stage)
        changed, preserved, regenerate = _prepare_predictions(stage, references)
        # Persist dirty IDs before inference: if scoring fails or the process is stopped,
        # freshly generated predictions must never inherit an obsolete successful score.
        dirty_path = stage / "recovery_changed_predictions.json"
        if dirty_path.exists():
            changed.update(read_json(dirty_path)["ids"])
        atomic_json(dirty_path, {"ids": sorted(changed)})
        if regenerate:
            namespace["group_predictions"](group, full[group], model, data, pages, references,
                                           release, frozen, frozen_dir, stage,
                                           evaluation="table-recognition", model_loading_lock=None)
        _verify_preserved(stage, preserved)
        records, _, failed = _cache_records(stage, references)
        if failed or set(records) != {row["id"] for row in references}:
            raise RuntimeError(f"Historical {group} still has failed or missing predictions")
        details[group] = _repair_scores(namespace, stage, changed, data=data, release=release,
                                        evaluator=evaluator, runtime=runtime)
        atomic_json(dirty_path, {"ids": []})
        details[group]["preserved_prediction_hashes"] = preserved
    execution = {"gpu_workers": 1, "overlap_scoring": False, "runtime_launcher": "bwrap",
                 "invocation_wall_seconds": perf_counter() - started, "recovery_overlay": str(overlay)}
    namespace["aggregate"](work, full, execution, evaluation="table-recognition")
    # This branch only renders existing Full summaries and completed evaluation results.
    # It cannot run training, pilot, or inference.
    subprocess.run([sys.executable, str(source / "scripts/xmod_run_all.py"), "--runs", str(runs),
                    "--reports", str(reports), "--full-reference", str(runs / "full/B0/summary.json"),
                    "--evaluation", "table-recognition", "--refresh-reports"], cwd=source, check=True)
    final_report = reports / "omnidocbench_report.md"
    validation = _validate_overlay(work, final_report)
    preserved_original = _verify_original_successes(source, work)
    if snapshot["source_manifest"] != _source_manifest(source):
        raise ValueError("Historical source changed during recovery")
    result = {"status": "COMPLETED", "historical": str(work), "runs_path": str(runs),
              "overlay": str(overlay), "final_report_path": str(final_report), "groups": details,
              "preserved_original_hash": object_hash(preserved_original),
              "source_manifest_hash": snapshot["source_manifest_hash"], **validation}
    atomic_json(overlay / "completed.json", result)
    return result


def _idle(source: Path) -> bool:
    from .state_monitor import gpu_processes, source_processes

    if source_processes(source):
        return False
    gpu = gpu_processes()
    if not gpu["known"] or gpu["processes"]:
        raise RuntimeError("Cannot recover historical evaluation while GPU state is busy or unknown")
    return True


def retry_historical(settings: Any) -> dict | None:
    """Recover only after exclusion checks; publish a ready pointer only on verified success."""
    if not _idle(settings.source):
        return None
    # Open read-only: even the original lock file's contents and mtime stay untouched.
    with (settings.source / "runs/xmod/.run_all.lock").open("rb") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        if not _idle(settings.source):
            return None
        manifest = _source_manifest(settings.source)
        overlay = _overlay(settings, manifest)
        if _source_manifest(settings.source) != manifest:
            raise ValueError("Historical source changed while preparing recovery")
        runs, reports = overlay / "runs", overlay / "reports"
        settings.reports.mkdir(parents=True, exist_ok=True)
        command = [str(settings.python), "-u", "-m", "orchestration.recovery", "--execute-overlay",
                   "--source", str(settings.source), "--runs", str(runs), "--reports", str(reports),
                   "--runtime", str(settings.runtime)]
        environment = dict(os.environ, HF_HUB_OFFLINE="1", TOKENIZERS_PARALLELISM="false",
                           PYTHONDONTWRITEBYTECODE="1", PYTHONPATH=str(settings.project / "src"))
        if not (overlay / "completed.json").exists():
            with (settings.reports / "historical_recovery.log").open("ab") as log:
                subprocess.run(command, cwd=settings.source, check=True, stdout=log, stderr=subprocess.STDOUT,
                               env=environment, pass_fds=(lock.fileno(),))
        result = read_json(overlay / "completed.json")
        validation = _validate_overlay(runs / BENCHMARK, reports / "omnidocbench_report.md")
        preservation = object_hash(_verify_original_successes(settings.source, runs / BENCHMARK))
        if (result.get("status") != "COMPLETED" or result.get("historical") != str(runs / BENCHMARK)
                or result.get("final_report_path") != str(reports / "omnidocbench_report.md")
                or result.get("source_manifest_hash") != object_hash(manifest)
                or result.get("preserved_original_hash") != preservation
                or any(result.get(key) != value for key, value in validation.items())):
            raise ValueError("Recovery completion receipt does not match its validated outputs")
        if _source_manifest(settings.source) != manifest:
            raise ValueError("Historical source changed during recovery")
        atomic_json(settings.reports / "historical_recovery.json", result)
        return result


def _main() -> None:
    parser = argparse.ArgumentParser(description="Recover historical evaluation in an isolated output overlay")
    parser.add_argument("--execute-overlay", action="store_true", required=True)
    for name in ("source", "runs", "reports", "runtime"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    _execute_overlay(args.source.resolve(), args.runs.resolve(), args.reports.resolve(), args.runtime.resolve())


if __name__ == "__main__":
    _main()
