#!/usr/bin/env python3
"""CPU scoring entry point inside the original OmniDocBench runtime."""
import argparse
import json
from pathlib import Path

from src.core.preprocess import normalized_table
from xiaomi_ocr_xmod.dataset import read_jsonl, write_jsonl
from xiaomi_ocr_xmod.evaluation import score_tables

from evaluation.metrics import structure_signature


def score(work):
    (work / "result").mkdir(exist_ok=True)
    pairs = read_jsonl(work / "table_pairs.jsonl")
    rows = score_tables(pairs, work)
    for row, pair in zip(rows, pairs):
        gold, pred = (normalized_table(pair[key], "html") for key in ("html", "prediction_html"))
        signature = structure_signature(gold)
        row["structure_em"] = int(signature is not None and signature == structure_signature(pred))
    write_jsonl(work / "per_table.jsonl", rows)
    return rows


def baseline_scores(source, work):
    base_pairs = read_jsonl(source / "table_pairs.jsonl")
    base_scores = {row["id"]: row for row in read_jsonl(source / "per_table.jsonl")}
    baseline = []
    for pair in base_pairs:
        row = base_scores[pair["id"]]
        gold, pred = (normalized_table(pair[key], "html") for key in ("html", "prediction_html"))
        signature = structure_signature(gold)
        baseline.append({**row, "strict_em": row["exact"],
                         "structure_em": int(signature is not None and signature == structure_signature(pred)),
                         "latency": pair["seconds"], "incremental_latency": 0, "calls": 1,
                         "additional_calls": 0, "changed": False, "delta_teds": 0, "status": "completed"})
    write_jsonl(work / "baseline.jsonl", baseline)
    return baseline


def self_test(work):
    table = '<table><tr><td rowspan="2">A</td><td></td></tr><tr><td>B</td></tr></table>'
    pair = {"id": "synthetic#0", "page": "synthetic", "annotation_id": 0,
            "html": table, "prediction_html": table, "attributes": {}, "slices": ["merged"],
            "seconds": 0.0, "visual_tokens": 64, "output_tokens": 12,
            "finish_reason": "stop", "cross_attention_flops": 0}
    write_jsonl(work / "table_pairs.jsonl", [pair])
    rows = score(work)
    assert len(rows) == 1 and rows[0]["teds"] == 1 and rows[0]["teds_s"] == 1
    assert rows[0]["exact"] == rows[0]["structure_em"] == 1 and rows[0]["char_edits"] == 0
    print(json.dumps({"synthetic_scorer_test": "passed", "real_dataset_read": False}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-root", type=Path, required=True)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    work = Path.cwd()
    if args.self_test:
        self_test(work)
    else:
        rows = score(work)
        baseline = baseline_scores(args.baseline_root, work)
        print(json.dumps({"scored": len(rows), "baseline": len(baseline)}))
