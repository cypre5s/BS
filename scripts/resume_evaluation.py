#!/usr/bin/env python3
"""Recover after interruption; completed prediction files are reused."""
import runpy
from pathlib import Path

if __name__ == "__main__":
    runpy.run_path(str(Path(__file__).with_name("launch_after_b2_b3.py")), run_name="__main__")
