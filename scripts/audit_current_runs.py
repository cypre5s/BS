#!/usr/bin/env python3
"""Capture process identities and receipt coverage without running inference."""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from orchestration.evaluation_gate import save_audit
from orchestration.settings import Settings

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    state = save_audit(Settings.load(args.config))
    print(json.dumps({"observed_at": state["observed_at"], "groups": state["groups"]}, indent=2))
