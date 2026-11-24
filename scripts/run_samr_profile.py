#!/usr/bin/env python3
"""Small isolated development run, one resident original Xiaomi model, no training."""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from orchestration.storage import atomic_json, file_hash
from samr.config import SAMRConfig
from samr.development import load_development_inputs, profile_rows, write_profile
from samr.engine import SAMRInferenceEngine
from xiaomi_runtime.adapter import Prediction, XiaomiRuntime, replay_baseline
from xiaomi_runtime.budget import DeadlineExceeded
from xiaomi_runtime.processor import ProcessorLimits


def check_gpu_idle() -> None:
    rows = subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"], text=True, timeout=10
    ).splitlines()
    if any(row.strip() and row.strip() != str(os.getpid()) for row in rows):
        raise RuntimeError("GPU has another compute process; no smoke inference will be started")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--exclude-manifest", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/samr/samr_full.yaml"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--real-smoke", action="store_true", help="Require 3-5 images and all three scales")
    args = parser.parse_args()
    config = SAMRConfig.load(args.config)
    inputs = load_development_inputs(args.manifest, args.exclude_manifest)
    if args.real_smoke and (not 3 <= len(inputs) <= 5 or config.mode != "fixed"
                           or config.scales != (1.0, 1.5, 1.75)
                           or config.fusion_policy != "global_anchor_blank_split_v1"):
        parser.error("Real smoke requires 3-5 images, fixed scales [1.0, 1.5, 1.75] and global-anchor fusion")
    check_gpu_idle()
    args.output.mkdir(parents=True, exist_ok=False)
    runtime = XiaomiRuntime(args.model, permission_check=check_gpu_idle, call_log=args.output / "calls.jsonl")
    before = runtime.assert_weights_unchanged()
    # Model initialization is a recorded setup cost, outside per-table warm inference budgets.
    from time import perf_counter
    started = perf_counter()
    runtime._load()
    setup_seconds = perf_counter() - started
    engine = SAMRInferenceEngine(runtime, config)
    rows, receipts = [], []
    parity = None
    deadline_probe = None
    after = None
    try:
        for table_id, path, digest in inputs:
            if file_hash(path) != digest:
                raise ValueError("Development input changed after validation")
            with Image.open(path) as source:
                image = source.convert("RGB")
            result = engine.run(table_id, image, audit_directory=args.output / "audit")
            rows.extend(profile_rows(result.audit))
            receipts.append(result.audit)
            if args.real_smoke and parity is None:
                original = Prediction(**result.audit["scales"][0]["prediction"])
                parity = replay_baseline(runtime, image, original)
                atomic_json(args.output / "native_B0_parity.json", parity, immutable=True)
                probe_started = perf_counter()
                probe_deadline = probe_started + 0.25
                native = ProcessorLimits().native_size(image.size)
                try:
                    partial = runtime.infer(table_id, image, method="SAMR_budget_probe", view_type="global",
                                            max_visual_tokens=native[0] * native[1] // 1024,
                                            deadline=probe_deadline)
                    deadline_probe = {"stopped": partial.error == "deadline_exceeded", "eos": partial.eos,
                                      "output_tokens": partial.output_tokens, "truncated": partial.truncated,
                                      "partial_discarded": True, "budget_seconds": 0.25}
                except DeadlineExceeded:
                    deadline_probe = {"stopped": True, "phase": "before_generation",
                                      "partial_discarded": True, "budget_seconds": 0.25}
                elapsed = perf_counter() - probe_started
                deadline_probe.update(elapsed_seconds=elapsed, overshoot_seconds=max(0, elapsed - 0.25),
                                      hard_wall_clock_guaranteed=False)
                atomic_json(args.output / "deadline_probe.json", deadline_probe, immutable=True)
            print(json.dumps({"table_id": table_id, "scales": len(result.audit["scales"]),
                              "fusion_reason": result.audit["fusion_decision"]["reason"],
                              "elapsed": result.audit["budget"]["elapsed_seconds"]}), flush=True)
    finally:
        after = runtime.assert_weights_unchanged()
        atomic_json(args.output / "runtime_receipt.json", {"before": before, "after": after,
                    "hashes_unchanged": before == after, "memory": runtime.memory_stats(),
                    "call_count": runtime.call_count, "setup_seconds": setup_seconds,
                    "parameters_frozen": not any(p.requires_grad for p in runtime.model.parameters())}, immutable=True)
    write_profile(args.output / "profile.jsonl", rows)
    valid = all(row["runtime_contract_valid"] and row["parse_status"]["canonical"] == "RESOLVED"
                and row["parse_status"]["otsl"] and row["parse_status"]["html"] == "valid"
                and row["eos"] and not row["truncated"] and not row["error"] for row in rows)
    all_scales = all(len(receipt["scales"]) == 3 for receipt in receipts)
    passed = (valid and before == after and runtime.load_count == 1
              and (not args.real_smoke or (all_scales and parity is not None and parity["passed"]
                                           and deadline_probe is not None and deadline_probe["stopped"])))
    summary = {"kind": "real_gpu_smoke" if args.real_smoke else "development_profile", "passed": passed,
               "images": len(inputs), "inferences": len(rows), "all_three_scales": all_scales,
               "no_truncation_and_valid_parses": valid, "model_load_count": runtime.load_count,
               "hashes_unchanged": before == after, "manifest_hash": file_hash(args.manifest),
               "exclusion_manifest_hash": file_hash(args.exclude_manifest), "config": config.as_dict(),
               "native_B0_parity": parity, "deadline_probe": deadline_probe,
               "formal_665_evaluation": False, "accuracy_claim": None}
    atomic_json(args.output / "summary.json", summary, immutable=True)
    print(json.dumps(summary, indent=2))
    if args.real_smoke and not passed:
        raise SystemExit("Real GPU smoke failed; inspect immutable per-scale receipts")


if __name__ == "__main__":
    main()
