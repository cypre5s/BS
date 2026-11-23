#!/usr/bin/env python3
"""Start a detached supervisor with an append-only log and process receipt."""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from orchestration.launcher import detach
from orchestration.settings import Settings

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(detach(Settings.load(args.config), args.config), indent=2))
