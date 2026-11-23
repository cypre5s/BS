"""A detached supervisor gates and resumes the three independent methods."""
from __future__ import annotations

import fcntl
import os
import signal
import subprocess
import traceback
from dataclasses import asdict
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from platform import python_version
from time import perf_counter, sleep

import yaml

from .evaluation_gate import EvaluationGate, ExclusiveGPU, save_audit
from .recovery import commit_prediction, completed_record, sample_name
from .settings import Settings
from .state_monitor import now, process_identity, same_process
from .storage import atomic_json, file_hash, object_hash, read_json


def verify_publication(settings: Settings) -> str:
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=settings.project, text=True).strip()
    dirty = subprocess.check_output(["git", "status", "--porcelain"], cwd=settings.project, text=True).strip()
    if dirty:
        raise RuntimeError("Publish and validate source changes before running formal evaluation")
    remote = subprocess.check_output(["git", "ls-remote", "origin", "refs/heads/main"], cwd=settings.project,
                                     text=True, timeout=30).split()
    if not remote or remote[0] != commit:
        raise RuntimeError("Local source commit has not been pushed to origin/main")
    validation = read_json(settings.project / "reports/development/validation_receipt.json")
    if not validation.get("passed"):
        raise RuntimeError("CPU validation receipt is missing or failed")
    tracked = subprocess.check_output(["git", "ls-files"], cwd=settings.project, text=True).splitlines()
    if set(tracked) != set(validation["files"]):
        raise RuntimeError("Published source inventory differs from the CPU-validated inventory")
    for name, digest in validation["files"].items():
        if file_hash(settings.project / name) != digest:
            raise RuntimeError(f"Validated source changed: {name}")
    return commit


def state_write(settings: Settings, state: dict) -> None:
    state["updated_at"] = now()
    atomic_json(settings.output / "orchestrator_state.json", state)


def environment_identity() -> dict:
    result = {"python": python_version()}
    for name in ("torch", "transformers", "Pillow", "numpy", "weasyprint", "PyMuPDF", "fonttools"):
        try:
            result[name] = version(name)
        except PackageNotFoundError:
            result[name] = "unavailable"
    return result


def build_methods(runtime, settings: Settings) -> dict:
    from multiview.multiscale import AdaptiveMultiScaleInference
    from reacquisition import VisualEvidenceReacquisition
    from render_verify import RenderGuidedVerification
    config_files = {"A": "method_a_multiscale.yaml", "B": "method_b_reacquisition.yaml", "C": "method_c_render_verify.yaml"}
    configurations = {name: yaml.safe_load((settings.project / "configs" / filename).read_text())
                      for name, filename in config_files.items()}
    if "font_paths" in configurations["C"]:
        configurations["C"]["font_paths"] = [str((settings.project / path).resolve())
                                             for path in configurations["C"]["font_paths"]]
    return {"A": AdaptiveMultiScaleInference(runtime, configurations["A"]),
            "B": VisualEvidenceReacquisition(runtime, configurations["B"]),
            "C": RenderGuidedVerification(runtime, configurations["C"])}


def run_methods(settings: Settings, state: dict, lease: ExclusiveGPU, selected: tuple[str, ...]) -> None:
    from evaluation.official_runner import score_method
    from evaluation.omnidocbench import Cohort
    from xiaomi_runtime.adapter import XiaomiRuntime, cached_prediction, replay_baseline

    cohort = Cohort(settings)
    cohort.save_manifest()
    runtime = XiaomiRuntime(settings.source / "models/xiaomi-ocr-0", permission_check=lease.permission_check,
                           call_log=settings.output / "model_calls.jsonl")
    methods = build_methods(runtime, settings)
    identity = {"commit": state["commit"], "cohort_hash": object_hash(cohort.identity),
                "runtime_environment": environment_identity(),
                "renderer_fonts": [{"name": path.name, "sha256": file_hash(path)}
                                   for path in methods["C"].renderer.font_paths],
                "configs": {path.name: file_hash(path) for path in sorted((settings.project / "configs").glob("method_*.yaml"))}}
    identity_hash = object_hash(identity)
    identity_path = settings.output / "identity.json"
    if identity_path.exists():
        if read_json(identity_path) != identity:
            raise RuntimeError("Resume identity differs from the original experiment")
    else:
        atomic_json(identity_path, identity, immutable=True)
    state["progress"] = {name: sum(1 for table in cohort.tables if completed_record(
        settings.output / f"method_{name.lower()}", table.table_id, identity_hash)) for name in selected}
    state["methods"] = {name: {"log_path": str(settings.output / f"method_{name.lower()}/run.log"),
                               "status": "PENDING"} for name in selected}
    # Table-major scheduling executes A, then B, then C sequentially with one resident model.
    # Every method sees the same cached B0 result, never another method's output.
    for table in cohort.tables:
        pending = [name for name in selected if not completed_record(
            settings.output / f"method_{name.lower()}", table.table_id, identity_hash)]
        if not pending:
            continue
        image = cohort.image(table)
        baseline = cached_prediction(table.baseline, image)
        replay_path = settings.reports / "native_baseline_replay.json"
        if not replay_path.exists():
            replay = replay_baseline(runtime, image, baseline)
            atomic_json(replay_path, {**replay, "identity_hash": identity_hash, "timestamp": now()}, immutable=True)
        replay = read_json(replay_path)
        if replay.get("identity_hash") != identity_hash or not replay.get("passed"):
            raise RuntimeError("Native global inference differs from the frozen B0 compatibility receipt")
        for name in pending:
            directory = settings.output / f"method_{name.lower()}"
            directory.mkdir(parents=True, exist_ok=True)
            state.update(status=f"RUNNING_{name}", current_method=name)
            state["methods"][name]["status"] = "RUNNING"
            state_write(settings, state)
            error = None
            prior_attempts = [read_json(path) for path in (directory / "attempts").glob(
                f"{sample_name(table.table_id)[:-5]}-*.json")]
            if any(row.get("identity_hash") != identity_hash for row in prior_attempts):
                raise ValueError("Existing retry cost records have incompatible provenance")
            previous_calls = sum(row["additional_calls"] for row in prior_attempts)
            previous_seconds = sum(row["elapsed_seconds"] for row in prior_attempts)
            for attempt in range(settings.max_attempts):
                existing = completed_record(directory, table.table_id, identity_hash)
                if existing is not None:
                    state["progress"][name] = sum(1 for item in cohort.tables if completed_record(
                        directory, item.table_id, identity_hash))
                    state_write(settings, state)
                    error = None
                    break
                started = perf_counter()
                calls_before = runtime.call_count
                try:
                    result = methods[name].run(table.table_id, image, baseline)
                    if result.prediction.table_id != table.table_id:
                        raise ValueError("Method changed the table ID")
                    attempt_seconds = perf_counter() - started
                    incremental_seconds = previous_seconds + attempt_seconds
                    additional_calls = previous_calls + runtime.call_count - calls_before
                    record = {"table_id": table.table_id, "page": table.page, "method": name,
                              "prediction": result.prediction.raw, "html": result.prediction.html,
                              "metrics": None, "latency": baseline.latency + incremental_seconds,
                              "incremental_latency": incremental_seconds, "calls": 1 + additional_calls,
                              "additional_calls": additional_calls, "status": "completed",
                              "successful_attempt_calls": 1 + len(result.calls),
                              "successful_attempt_incremental_latency": attempt_seconds,
                              "prior_failed_attempts": len(prior_attempts) + attempt,
                              "error": result.diagnostics.get("error"),
                              "failure": bool(result.diagnostics.get("error")),
                              "failure_kind": "verification_unavailable" if result.diagnostics.get("error") else None,
                              "retryable": False,
                              "changed": result.prediction.raw != baseline.raw,
                              "reason": result.reason, "diagnostics": result.diagnostics,
                              "views": [asdict(call) for call in result.calls], "identity_hash": identity_hash,
                              "sample_filename": sample_name(table.table_id), "output_tokens": result.prediction.output_tokens,
                              "finish_reason": "length" if result.prediction.truncated else "stop",
                              "completed_at": now()}
                    commit_prediction(directory, record)
                    state["progress"][name] += 1
                    state["methods"][name].update(status="RUNNING", completed=state["progress"][name],
                                                  last_success=record["completed_at"])
                    with (directory / "run.log").open("a") as log:
                        log.write(f"{now()} completed={state['progress'][name]}/665 calls={len(result.calls)}\n")
                    state["runtime"] = runtime.memory_stats()
                    state_write(settings, state)
                    atomic_json(directory / "progress.json", {"identity_hash": identity_hash,
                                "completed": state["progress"][name], "total": 665, "updated_at": now()})
                    print(f"{now()} method={name} completed={state['progress'][name]}/665", flush=True)
                    error = None
                    break
                except BaseException as failure:
                    error = failure
                    failed_calls = runtime.call_count - calls_before
                    failed_seconds = perf_counter() - started
                    attempt_path = directory / "attempts" / f"{sample_name(table.table_id)[:-5]}-{os.getpid()}-{time_ns()}.json"
                    atomic_json(attempt_path, {"table_id": table.table_id, "attempt": attempt + 1,
                                             "identity_hash": identity_hash, "additional_calls": failed_calls,
                                             "elapsed_seconds": failed_seconds,
                                             "status": "failed", "error": str(failure), "time": now()}, immutable=True)
                    previous_calls += failed_calls
                    previous_seconds += failed_seconds
                    if not isinstance(failure, Exception):
                        raise
            if error is not None:
                raise RuntimeError(f"Method {name} failed after retries") from error
        if all(state["progress"][name] > 0 for name in selected):
            proof = settings.reports / "real_startup.json"
            if not proof.exists():
                atomic_json(proof, {"timestamp": now(), "process": process_identity(os.getpid()),
                                   "methods": state["methods"], "progress": state["progress"],
                                   "runtime": runtime.memory_stats(), "weights": runtime.assert_weights_unchanged(),
                                   "identity_hash": identity_hash}, immutable=True)
    runtime.assert_weights_unchanged()
    state["scoring"] = True
    state_write(settings, state)
    for name in selected:
        score_method(settings, name, [table.table_id for table in cohort.tables])
        state["methods"][name]["status"] = "COMPLETED"
        state_write(settings, state)
    state.update(status="COMPLETED", scoring=False, current_method=None)
    state_write(settings, state)


def time_ns() -> int:
    from time import time_ns as current
    return current()


def supervise(settings: Settings, selected: tuple[str, ...] = ("A", "B", "C")) -> None:
    settings.output.mkdir(parents=True, exist_ok=True)
    settings.reports.mkdir(parents=True, exist_ok=True)
    with (settings.output / ".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        state: dict = {"status": "WAITING_FOR_B2_B3", "process": process_identity(os.getpid()),
                 "log_path": str(settings.output / "orchestrator.log"), "current_method": None,
                 "progress": {}, "started_at": now(), "commit": verify_publication(settings)}
        state_write(settings, state)
        gate = EvaluationGate(settings)

        def interrupted(signum, _frame):
            raise SystemExit(128 + signum)

        previous_handler = signal.signal(signal.SIGTERM, interrupted)
        try:
            while True:
                observation = save_audit(settings)
                result = gate.check(observation)
                state["gate_reasons"] = result["reasons"]
                state_write(settings, state)
                if result["ready"]:
                    receipt = gate.completion_receipt()
                    atomic_json(settings.reports / "b2_b3_completion.json", receipt)
                    break
                if not observation["source_processes"] and observation["gpu"]["known"] and not observation["gpu"]["processes"]:
                    incomplete = any(row["evaluation_status"] != "COMPLETED" for row in observation["groups"].values())
                    missing_final = not observation["final_results_exist"] or not observation["final_report_exists"]
                    if incomplete or missing_final:
                        from .recovery import retry_historical
                        retry_historical(settings)
                sleep(settings.poll_seconds)
            state.update(status="READY", gate_reasons=[])
            state_write(settings, state)
            with ExclusiveGPU(settings) as lease:
                lease.permission_check()
                run_methods(settings, state, lease, selected)
        except BaseException as error:
            state.update(status="INTERRUPTED" if isinstance(error, (KeyboardInterrupt, SystemExit)) else "FAILED",
                         error=str(error), traceback=traceback.format_exc())
            state_write(settings, state)
            raise
        finally:
            signal.signal(signal.SIGTERM, previous_handler)


def detach(settings: Settings, config: Path) -> dict:
    settings.output.mkdir(parents=True, exist_ok=True)
    path = settings.output / "orchestrator_state.json"
    if path.exists():
        previous = read_json(path)
        identity = previous.get("process")
        if identity and same_process(identity, process_identity(identity["pid"])):
            return {"already_running": True, "process": identity}
    environment = dict(os.environ, PYTHONPATH=str(settings.project / "src"), HF_HUB_OFFLINE="1",
                       TOKENIZERS_PARALLELISM="false", PYTORCH_ALLOC_CONF="expandable_segments:True")
    with (settings.output / "orchestrator.log").open("ab", buffering=0) as log:
        process = subprocess.Popen([str(settings.python), "-u", str(settings.project / "scripts/wait_for_b2_b3.py"),
                                    "--config", str(config.resolve())], cwd=settings.project, env=environment,
                                   stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                                   start_new_session=True)
    identity = process_identity(process.pid)
    atomic_json(settings.output / "supervisor.json", {"process": identity, "log_path": str(settings.output / "orchestrator.log")})
    return {"process": identity}
