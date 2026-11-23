"""Expose images and cached B0 outputs to methods without reference answers."""
from __future__ import annotations

import subprocess
from dataclasses import dataclass
from math import ceil, floor
from pathlib import Path

from PIL import Image

from orchestration.settings import Settings
from orchestration.storage import atomic_json, file_hash, object_hash, read_json, read_jsonl

DATA_REVISION = "aa1ee96d106dbe53d0ae59474d75c6e6d9b53fec"
GT_SHA256 = "a45cd84b04ad8b793e775089640e6b681209abea33ead54c1828ddca35fae496"
EVALUATOR_COMMIT = "7ddb8df411ad9bd2fff139acafe0ed16aa1694fe"
TABLE_COUNT = 665


@dataclass(frozen=True)
class TableInput:
    table_id: str
    page: str
    polygon: tuple[float, ...]
    baseline: dict


class Cohort:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.data = settings.source / "data/omnidocbench/benchmark"
        self.release = read_json(self.data / "identity.json")
        if self.release["revision"] != DATA_REVISION or file_hash(self.data / "OmniDocBench.json") != GT_SHA256:
            raise ValueError("Dataset revision differs from the frozen cohort")
        evaluator = settings.source / "data/omnidocbench/evaluator"
        head = subprocess.check_output(["git", "-C", str(evaluator), "rev-parse", "HEAD"], text=True).strip()
        dirty = subprocess.check_output(["git", "-C", str(evaluator), "status", "--porcelain", "--untracked-files=no"], text=True)
        if head != EVALUATOR_COMMIT or dirty:
            raise ValueError("Official evaluator changed")
        records = read_jsonl(settings.historical / "B0/table_pairs.jsonl")
        # Reference HTML and metrics deliberately do not enter the inference input objects.
        self.tables = [TableInput(row["id"], row["page"], tuple(row["polygon"]),
                                 {key: row[key] for key in ("id", "raw", "prediction_html", "finish_reason",
                                                           "visual_tokens", "output_tokens", "seconds")})
                       for row in records]
        ids = [table.table_id for table in self.tables]
        if len(ids) != TABLE_COUNT or len(set(ids)) != TABLE_COUNT:
            raise ValueError("Expected 665 unique frozen B0 table IDs")
        baseline_identity = read_json(settings.historical / "B0/identity.json")
        from xiaomi_runtime.adapter import MODEL_SHA256, TABLE_PROMPT
        if (baseline_identity["model"] != MODEL_SHA256 or baseline_identity["table_prompt"] != TABLE_PROMPT
                or baseline_identity["max_new_tokens"] != 16384 or baseline_identity["do_sample"]
                or baseline_identity["dtype"] != "bfloat16"
                or baseline_identity["evaluator_commit"] != EVALUATOR_COMMIT):
            raise ValueError("B0 inference provenance is incompatible")
        self.identity = {"dataset_revision": DATA_REVISION, "ground_truth_hash": GT_SHA256,
                         "table_ids_hash": object_hash(ids), "table_ids": ids,
                         "baseline_pairs_hash": file_hash(settings.historical / "B0/table_pairs.jsonl"),
                         "baseline_identity": baseline_identity, "evaluator_commit": head}

    def save_manifest(self) -> None:
        path = self.settings.output / "cohort.json"
        if path.exists():
            if read_json(path) != self.identity:
                raise ValueError("Existing experiment cohort differs")
        else:
            atomic_json(path, self.identity, immutable=True)

    def image(self, table: TableInput) -> Image.Image:
        relative = Path("images") / table.page
        path = self.data / relative
        if file_hash(path) != self.release["files"][relative.as_posix()]:
            raise ValueError("Published page image hash differs")
        with Image.open(path) as source:
            image = source.convert("RGB")
        x, y = table.polygon[::2], table.polygon[1::2]
        box = (max(0, floor(min(x))), max(0, floor(min(y))),
               min(image.width, ceil(max(x))), min(image.height, ceil(max(y))))
        if box[0] >= box[2] or box[1] >= box[3]:
            raise ValueError("Invalid frozen table crop")
        return image.crop(box)
