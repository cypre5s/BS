#!/usr/bin/env python3
"""Foreground durable worker; use launch_after_b2_b3.py to detach it."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from orchestration.launcher import supervise
from orchestration.settings import Settings

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--methods", nargs="+", choices=("A", "B", "C"), default=["A", "B", "C"])
    args = parser.parse_args()
    supervise(Settings.load(args.config), tuple(args.methods))
