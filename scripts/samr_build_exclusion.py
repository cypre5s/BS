#!/usr/bin/env python3
"""Create a local exclusion manifest; do not publish it or the underlying images."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from samr.development import build_exclusion_manifest

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-pairs", type=Path, required=True)
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    build_exclusion_manifest(args.baseline_pairs, args.benchmark, args.output)
    print("Frozen 665-table exclusion manifest created; no inference performed")
