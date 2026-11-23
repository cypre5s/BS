"""Explicit machine paths belong in an excluded local configuration."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass(frozen=True)
class Settings:
    project: Path
    source: Path
    runtime: Path
    python: Path
    poll_seconds: float = 45
    stability_seconds: float = 30
    max_attempts: int = 3

    @property
    def historical(self) -> Path:
        from .storage import read_json
        receipt = self.reports / "historical_recovery.json"
        if receipt.exists():
            return Path(read_json(receipt)["historical"])
        return self.source / "runs/xmod/omnidocbench/table_recognition"

    @property
    def final_report(self) -> Path:
        from .storage import read_json
        receipt = self.reports / "historical_recovery.json"
        if receipt.exists():
            return Path(read_json(receipt)["final_report_path"])
        return self.source / "reports/xmod/omnidocbench_report.md"

    @property
    def output(self) -> Path:
        return self.project / "experiments/multiview"

    @property
    def reports(self) -> Path:
        return self.project / "reports/orchestration"

    @classmethod
    def load(cls, path: Path) -> Settings:
        data = yaml.safe_load(path.read_text())
        return cls(project=Path(data["project"]).expanduser().resolve(),
                   source=Path(data["source"]).expanduser().resolve(),
                   runtime=Path(data["runtime"]).expanduser().resolve(),
                   python=Path(data["python"]).expanduser().absolute(),
                   poll_seconds=float(data.get("poll_seconds", 45)),
                   stability_seconds=float(data.get("stability_seconds", 30)),
                   max_attempts=int(data.get("max_attempts", 3)))
