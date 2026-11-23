#!/usr/bin/env python3
"""Scan only explicitly publishable source, with optional staged-content validation."""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from orchestration.publication import scan

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--staged", action="store_true")
    args = parser.parse_args()
    result = scan(ROOT, staged=args.staged)
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result["passed"] else 1)
