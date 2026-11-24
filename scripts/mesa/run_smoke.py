#!/usr/bin/env python3
"""Three to five non-benchmark development images; never a formal evaluation."""
import argparse
import fcntl
import json
import os
import subprocess
import sys
from pathlib import Path
from time import perf_counter

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from mesa.config import CHANNELS, MESAConfig
from mesa.engine import MESAInferenceEngine
from orchestration.storage import atomic_json, file_hash, read_json
from render_verify.structural_checker import inspect_html
from samr.development import load_development_inputs
from xiaomi_runtime.adapter import Prediction, XiaomiRuntime, replay_baseline
from xiaomi_runtime.otsl import parse_table


def gpu_idle():
    rows = subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"],
                                   text=True, timeout=10).splitlines()
    if any(row.strip() and row.strip() != str(os.getpid()) for row in rows):
        raise RuntimeError("GPU occupied by another compute process; do not start MESA")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--exclude-manifest", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/mesa/mesa_fusion.yaml"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpu-lock", type=Path, required=True)
    args = parser.parse_args()
    config = MESAConfig.load(args.config)
    inputs = load_development_inputs(args.manifest, args.exclude_manifest)
    if not 3 <= len(inputs) <= 5 or config.enabled != CHANNELS:
        parser.error("Real smoke requires 3-5 independent images and all five channels")
    args.output.mkdir(parents=True, exist_ok=True)
    args.gpu_lock.parent.mkdir(parents=True, exist_ok=True)
    with args.gpu_lock.open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        gpu_idle()
        runtime = XiaomiRuntime(args.model, permission_check=gpu_idle, call_log=args.output / "native_calls.jsonl")
        before = runtime.assert_weights_unchanged()
        setup = perf_counter()
        runtime._load()
        setup_seconds = perf_counter() - setup
        engine = MESAInferenceEngine(runtime, config, args.output / "tables")
        receipts, rows, parity = [], [], None
        try:
            for table_id, path, digest in inputs:
                if file_hash(path) != digest:
                    raise ValueError("Development image changed")
                with Image.open(path) as source:
                    image = source.convert("RGB")
                result = engine.run(table_id, image)
                receipts.append(result)
                from xiaomi_runtime.adapter import text_hash
                folder = args.output / "tables" / text_hash(table_id)
                for rel in result["cache_files"]:
                    record = read_json(folder / rel)
                    p, view = record["prediction"], record["view"]
                    valid, error = True, None
                    try:
                        parse_table(p["raw"])
                        inspect_html(p["html"])
                    except (ValueError, TypeError) as failure:
                        valid, error = False, str(failure)
                    rows.append({"table_id": table_id, "channel_id": view["channel_id"], "view_id": view["view_id"],
                                 "input_resolution": p["input_size"], "processed_resolution": p["processed_size"],
                                 "visual_token_count": p["visual_tokens"], "image_grid_thw": p["image_grid_thw"],
                                 "latency": p["latency"], "official_parse_success": valid, "parse_error": error,
                                 "canonical_status": record["canonical"]["status"], "status": "completed"})
                    if view["channel_id"] == "global" and parity is None:
                        from dataclasses import replace
                        original = Prediction(**p)
                        original = replace(original, input_size=tuple(original.input_size),
                                           processed_size=tuple(original.processed_size))
                        parity = replay_baseline(runtime, image, original)
                print(json.dumps({"table_id": table_id, "views": len(result["cache_files"]),
                                  "reason": result["fusion"]["reason"], "budget": result["budget"]}), flush=True)
        finally:
            after = runtime.assert_weights_unchanged()
            atomic_json(args.output / "runtime_receipt.json", {"before": before, "after": after,
                        "hashes_unchanged": before == after, "setup_seconds": setup_seconds,
                        "memory": runtime.memory_stats(), "model_load_count": runtime.load_count,
                        "calls": runtime.call_count, "frozen": not any(p.requires_grad for p in runtime.model.parameters())})
        channels_ok = all({row["channel_id"] for row in rows if row["table_id"] == table_id} == set(CHANNELS)
                          for table_id, _, _ in inputs)
        passed = (channels_ok and all(r["official_parse_success"] for r in rows) and before == after
                  and runtime.load_count == 1 and parity is not None and parity["passed"])
        summary = {"passed": passed, "images": len(inputs), "views": len(rows), "profile": rows,
                   "all_five_channels_per_image": channels_ok, "hashes_unchanged": before == after,
                   "model_load_count": runtime.load_count, "native_B0_parity": parity,
                   "config": config.as_dict(), "manifest_sha256": file_hash(args.manifest),
                   "exclusion_sha256": file_hash(args.exclude_manifest), "formal_665_evaluation": False,
                   "accuracy_claim": None}
        atomic_json(args.output / "summary.json", summary)
        print(json.dumps({key: value for key, value in summary.items() if key != "profile"}), flush=True)
        if not passed:
            raise SystemExit("MESA smoke did not pass; inspect the retained audit")


if __name__ == "__main__":
    main()
