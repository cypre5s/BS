"""Completion requires receipts, matching IDs, stable files, and departed processes."""
from __future__ import annotations

import fcntl
import os
import time
from typing import IO

from .settings import Settings
from .state_monitor import audit, expected_ids, inspect_group, now, output_snapshot
from .storage import atomic_json, file_hash, object_hash, read_json, read_jsonl


def validate_receipts(settings: Settings) -> dict:
    root = settings.historical
    ids = set(expected_ids(settings))
    final = read_json(root / "results.json")
    if final["evaluation"] != "table-recognition":
        raise ValueError("Historical evaluation task differs")
    snapshots = {}
    configs = {}
    for group in ("B0", "B1", "B2", "B3"):
        stage = root / group
        receipt = read_json(stage / "predictions.json")
        if receipt["identity"] != read_json(stage / "identity.json"):
            raise ValueError(f"{group} prediction identity mismatch")
        if receipt["table_pairs"] != file_hash(stage / "table_pairs.jsonl"):
            raise ValueError(f"{group} table-pair hash mismatch")
        inspected = inspect_group(stage, sorted(ids))
        if inspected["evaluation_status"] != "COMPLETED":
            raise ValueError(f"{group} outputs are incomplete or invalid: {inspected['errors']}")
        pairs = read_jsonl(stage / "table_pairs.jsonl")
        cached = {row["id"]: row for row in (read_json(path) for path in (stage / "pred_tables").glob("*.json"))}
        if len(pairs) != 665 or set(cached) != ids or {row["id"] for row in pairs} != ids:
            raise ValueError(f"{group} paired ID coverage mismatch")
        for row in pairs:
            for key, value in cached[row["id"]].items():
                if row.get(key) != value:
                    raise ValueError(f"{group} prediction disagrees with receipt")
        if final["groups"][group]["tables"]["count"] != 665:
            raise ValueError(f"{group} final result coverage mismatch")
        snapshots[group] = {path.name: file_hash(path) for path in sorted(stage.glob("*.json*"))}
        configs[group] = receipt["identity"]
    report = settings.final_report
    text = report.read_text()
    if not all(f"| {group} | 665 |" in text for group in ("B0", "B1", "B2", "B3")):
        raise ValueError("Final Markdown report is still a placeholder")
    return {"output_manifest_hash": object_hash(snapshots), "evaluation_config_hash": object_hash(configs),
            "final_results_hash": file_hash(root / "results.json"), "final_report_hash": file_hash(report)}


class EvaluationGate:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.snapshot: str | None = None
        self.stable_since: float | None = None

    def check(self, observation: dict | None = None, *, clock: float | None = None) -> dict:
        state = observation if observation is not None else audit(self.settings)
        current = time.monotonic() if clock is None else clock
        reasons = []
        if state["source_processes"]:
            reasons.append("historical evaluation process is still alive")
        if not state["gpu"]["known"]:
            reasons.append("GPU process state is unknown")
        elif state["gpu"]["processes"]:
            reasons.append("GPU has a running process")
        for name, group in state["groups"].items():
            if group["evaluation_status"] != "COMPLETED":
                reasons.append(f"{name} is {group['evaluation_status']}")
        if not state["final_results_exist"] or not state["final_report_exists"]:
            reasons.append("final result or report is missing")
        if state["output_snapshot"] != self.snapshot:
            self.snapshot = state["output_snapshot"]
            self.stable_since = current
        stable = self.stable_since is not None and current - self.stable_since >= self.settings.stability_seconds
        if not stable:
            reasons.append("output stability interval is pending")
        return {"ready": not reasons, "reasons": reasons, "observation": state}

    def completion_receipt(self) -> dict:
        state = self.check()
        if not state["ready"]:
            raise RuntimeError("; ".join(state["reasons"]))
        receipts = validate_receipts(self.settings)
        return {"completion_timestamp": now(), "B2_final_status": "COMPLETED", "B3_final_status": "COMPLETED",
                "completed_samples": {"B2": 665, "B3": 665}, "failed_samples": {"B2": 0, "B3": 0},
                "total_samples": 665, "snapshot": output_snapshot(self.settings.historical), **receipts}


class ExclusiveGPU:
    """Hold the original experiment lock through new inference, after its gate opens."""
    def __init__(self, settings: Settings):
        self.settings = settings
        self.file: IO[str] | None = None

    def __enter__(self):
        self.file = (self.settings.source / "runs/xmod/.run_all.lock").open("a")
        try:
            fcntl.flock(self.file, fcntl.LOCK_EX | fcntl.LOCK_NB)
            receipt = read_json(self.settings.reports / "b2_b3_completion.json")
            verified = validate_receipts(self.settings)
            if receipt.get("B2_final_status") != "COMPLETED" or receipt.get("B3_final_status") != "COMPLETED":
                raise RuntimeError("Historical completion receipt is invalid")
            if any(receipt.get(key) != value for key, value in verified.items()):
                raise RuntimeError("Historical completion hashes changed")
        except BaseException:
            self.file.close()
            raise
        return self

    def permission_check(self) -> None:
        from .state_monitor import gpu_processes, source_processes
        if self.file is None or self.file.closed:
            raise RuntimeError("GPU lease is absent")
        if source_processes(self.settings.source):
            raise RuntimeError("Historical evaluation is running")
        processes = gpu_processes()
        if not processes["known"] or any(row["pid"] != os.getpid() for row in processes["processes"]):
            raise RuntimeError("GPU process exclusion check failed")
        receipt = self.settings.reports / "b2_b3_completion.json"
        if not receipt.is_file():
            raise RuntimeError("Historical completion receipt is absent")

    def __exit__(self, *args):
        if self.file is not None:
            self.file.close()


def save_audit(settings: Settings) -> dict:
    result = audit(settings)
    atomic_json(settings.reports / "current_evaluation_state.json", result)
    return result
