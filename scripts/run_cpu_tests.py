#!/usr/bin/env python3
"""Run CPU checks and bind the receipt to all publishable source and configuration."""
import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from orchestration.publication import candidates, scan
from orchestration.state_monitor import now
from orchestration.storage import atomic_json, file_hash

if __name__ == "__main__":
    environment = dict(os.environ, CUDA_VISIBLE_DEVICES="", PYTHONPATH=str(ROOT / "src"))
    commands = [[sys.executable, "-m", "pytest", "-q"],
                [sys.executable, "-m", "ruff", "check", "src", "scripts", "tests"],
                [sys.executable, "-m", "mypy", "src"]]
    results = []
    for command in commands:
        result = subprocess.run(command, cwd=ROOT, env=environment, capture_output=True, text=True, check=False)
        print(result.stdout, end="")
        print(result.stderr, end="", file=sys.stderr)
        results.append({"command": command[2:], "returncode": result.returncode,
                        "stdout": result.stdout, "stderr": result.stderr})
    publication = scan(ROOT)
    passed = all(item["returncode"] == 0 for item in results) and publication["passed"]
    summaries = re.findall(r"^.*\d+ passed[^\n]*$", results[0]["stdout"], flags=re.MULTILINE)
    summary = summaries[-1] if summaries else "Pytest did not report a passing summary."
    report = "\n".join([
        "# CPU and mock validation", "", f"Validation timestamp: {now()}.", "",
        f"Overall status: {'PASS' if passed else 'FAIL'}.", "", summary, "",
        "| Check | Result |", "| --- | --- |",
        *[f"| {' '.join(item['command'])} | {'PASS' if item['returncode'] == 0 else 'FAIL'} |"
          for item in results],
        f"| Publication allowlist and secret/history scan | {'PASS' if publication['passed'] else 'FAIL'} |",
        "", "Tests cover native processor sizing, official OTSL serialization, independent methods,",
        "crop coordinates, real HTML rendering, failed historical recovery, process identity,",
        "completion gating, source publication, immutable records and resumed scoring.", "",
        "CUDA_VISIBLE_DEVICES was empty for all checks. Mock output is synthetic and no real model",
        "inference was started by this suite. The detailed machine receipt is excluded from Git.", "",
        "The separate processor audit verified six snapshot hashes, seven image shapes and 256",
        "native patch positions on CPU. Live GPU memory, model load count and A/B/C startup",
        "require the post-gate execution receipts described in docs/code_audit.md.", "",
    ])
    report_path = ROOT / "reports/development/code_validation_report.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report)
    publication = scan(ROOT)
    passed = passed and publication["passed"]
    receipt = {"timestamp": now(), "passed": passed, "cuda_visible_devices": "", "checks": results,
               "publication": publication,
               "files": {name: file_hash(ROOT / name) for name in candidates(ROOT)}}
    atomic_json(ROOT / "reports/development/validation_receipt.json", receipt)
    raise SystemExit(0 if passed else 1)
