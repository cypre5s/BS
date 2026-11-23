"""Read actual Linux process identities and frozen evaluation receipts."""
from __future__ import annotations

import subprocess
from datetime import datetime, timezone
from pathlib import Path

from .settings import Settings
from .storage import object_hash, read_json, read_jsonl


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def process_identity(pid: int, proc: Path = Path("/proc")) -> dict | None:
    try:
        folder = proc / str(pid)
        stat = (folder / "stat").read_text().rsplit(")", 1)[1].split()
        if stat[0] == "Z":
            return None
        command = (folder / "cmdline").read_bytes().decode().split("\x00")
        return {"pid": pid, "start_ticks": int(stat[19]),
                "boot_id": (proc / "sys/kernel/random/boot_id").read_text().strip(),
                "command": [item for item in command if item], "state": stat[0]}
    except (FileNotFoundError, ProcessLookupError, PermissionError, ValueError):
        return None


def same_process(saved: dict, current: dict | None) -> bool:
    return current is not None and all(saved.get(key) == current.get(key)
                                      for key in ("pid", "start_ticks", "boot_id", "command"))


def source_processes(source: Path) -> list[dict]:
    found = []
    for folder in Path("/proc").iterdir():
        if not folder.name.isdigit():
            continue
        identity = process_identity(int(folder.name))
        if identity is None:
            continue
        arguments = identity["command"]
        matching = any(Path(arg).name in {"xmod_evaluate.py", "xmod_run_all.py", "xmod_train.py"}
                       for arg in arguments)
        if not matching:
            continue
        try:
            cwd = (folder / "cwd").resolve(strict=True)
        except (FileNotFoundError, PermissionError):
            continue
        if cwd == source or any(str(source) in arg for arg in arguments):
            found.append(identity)
    return found


def gpu_processes() -> dict:
    try:
        result = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,used_gpu_memory",
                                 "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=10,
                                check=False)
        if result.returncode:
            return {"known": False, "error": result.stderr.strip(), "processes": []}
        rows = []
        for line in result.stdout.splitlines():
            pid, memory = line.split(",", 1)
            rows.append({"pid": int(pid.strip()), "memory_mib": memory.strip(),
                         "identity": process_identity(int(pid.strip()))})
        return {"known": True, "processes": rows}
    except (OSError, ValueError, subprocess.TimeoutExpired) as error:
        return {"known": False, "error": str(error), "processes": []}


def expected_ids(settings: Settings) -> list[str]:
    rows = read_jsonl(settings.historical / "B0/table_pairs.jsonl")
    ids = [row["id"] for row in rows]
    if len(ids) != 665 or len(set(ids)) != 665:
        raise ValueError("B0 must contain exactly 665 distinct table IDs")
    return ids


def output_snapshot(root: Path) -> str:
    files = [(str(path.relative_to(root)), path.stat().st_size, path.stat().st_mtime_ns)
             for path in sorted(root.rglob("*")) if path.is_file()]
    return object_hash(files)


def inspect_group(root: Path, ids: list[str]) -> dict:
    predictions = list((root / "pred_tables").glob("*.json"))
    records = []
    errors = []
    malformed = 0
    for path in predictions:
        try:
            row = read_json(path)
            if not isinstance(row, dict) or not isinstance(row.get("id"), str):
                raise TypeError("prediction has no valid table ID")
            if not isinstance(row.get("output_tokens", 0), (int, float)):
                raise TypeError("prediction token count is invalid")
            records.append(row)
        except (OSError, ValueError, TypeError) as error:
            errors.append(f"{path.name}: invalid prediction ({type(error).__name__})")
            malformed += 1
    found = [row["id"] for row in records]
    successful = {row["id"] for row in records if row.get("finish_reason") in {"stop", "length"}
                  and isinstance(row.get("prediction_html"), str) and row.get("output_tokens", 0) > 0}
    required = ("predictions.json", "table_pairs.jsonl", "per_table.jsonl", "official_table.json")
    if len(found) != len(set(found)) or set(found) - set(ids):
        errors.append("duplicate or unexpected prediction IDs")
    diagnostics = None
    if (root / "official_table.json").is_file():
        try:
            diagnostics = read_json(root / "official_table.json")["diagnostics"]
            if diagnostics["sample_count"] != len(ids):
                errors.append("official scorer coverage mismatch")
            if diagnostics["timeout_case_count"] or diagnostics["error_case_count"]:
                errors.append("official scorer failed samples")
        except (OSError, ValueError, KeyError, TypeError):
            errors.append("official scorer diagnostics are invalid or incomplete")
    for filename in ("table_pairs.jsonl", "per_table.jsonl"):
        path = root / filename
        if path.exists():
            try:
                rows = read_jsonl(path)
                if len(rows) != len(ids) or {row["id"] for row in rows} != set(ids):
                    errors.append(f"{filename} coverage mismatch")
            except (OSError, ValueError, KeyError, TypeError):
                errors.append(f"{filename} is invalid or incomplete")
    latest = max((path.stat().st_mtime for path in predictions), default=None)
    completed = (successful == set(ids) and not errors and all((root / name).is_file() for name in required))
    return {"output_directory": str(root), "completed_samples": len(successful), "total_samples": len(ids),
            "failed_samples": len(set(found) - successful) + malformed, "missing_samples": len(set(ids) - successful),
            "truncated_samples": sum(row.get("finish_reason") == "length" for row in records),
            "last_prediction_timestamp": latest, "errors": errors, "diagnostics": diagnostics,
            "evaluation_status": "COMPLETED" if completed else "FAILED" if errors else "INCOMPLETE"}


def audit(settings: Settings) -> dict:
    ids = expected_ids(settings)
    processes = source_processes(settings.source)
    log = settings.source / "runs/xmod/run_all.log"
    groups = {group: inspect_group(settings.historical / group, ids) for group in ("B2", "B3")}
    evaluators = [item for item in processes if any(Path(arg).name == "xmod_evaluate.py"
                                                  for arg in item["command"])]
    candidates = [name for name in ("B2", "B3") if groups[name]["evaluation_status"] != "COMPLETED"]
    active = max(candidates, key=lambda name: groups[name]["last_prediction_timestamp"] or 0, default=None)
    for name, group in groups.items():
        group["process"] = evaluators[0] if evaluators and name == active else None
        group["process_attribution"] = "latest_prediction_activity_in_shared_evaluator" if group["process"] else None
        group["pid"] = group["process"]["pid"] if group["process"] else None
        if group["process"] and group["evaluation_status"] == "INCOMPLETE":
            group["evaluation_status"] = "RUNNING"
    report = settings.final_report
    return {"observed_at": now(), "groups": groups, "source_processes": processes,
            "shared_evaluator_processes": evaluators,
            "gpu": gpu_processes(), "last_log_timestamp": log.stat().st_mtime if log.exists() else None,
            "log_path": str(log), "final_results_exist": (settings.historical / "results.json").is_file(),
            "final_report_exists": report.is_file(), "final_report_path": str(report),
            "expected_table_ids_hash": object_hash(ids), "output_snapshot": output_snapshot(settings.historical)}
