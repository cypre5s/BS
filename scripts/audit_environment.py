#!/usr/bin/env python3
"""CPU environment and process audit; loading model weights is unnecessary."""
import argparse
import json
import platform
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from orchestration.evaluation_gate import save_audit
from orchestration.settings import Settings

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    packages = {}
    for name in ("torch", "transformers", "Pillow", "numpy", "weasyprint", "PyMuPDF"):
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            packages[name] = None
    snapshot = save_audit(Settings.load(args.config))
    print(json.dumps({"python": sys.version, "system": platform.system(), "packages": packages,
                      "gpu": snapshot["gpu"], "source_processes": snapshot["source_processes"]}, indent=2))
