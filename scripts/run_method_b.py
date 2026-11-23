#!/usr/bin/env python3
"""Run method B alone with the same historical completion gate."""
import runpy
import sys
from pathlib import Path

if __name__ == "__main__":
    sys.argv.extend(["--methods", "B"])
    runpy.run_path(str(Path(__file__).with_name("wait_for_b2_b3.py")), run_name="__main__")
