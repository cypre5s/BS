#!/usr/bin/env python3
"""Offline development scoring in the pinned official CPU environment, with no model."""

import argparse
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from evaluation.omnidocbench import EVALUATOR_COMMIT
from orchestration.storage import atomic_json, read_jsonl
from samr.development import write_profile
from samr.offline import score_profile


def official_scores(pairs: list[dict], output: Path) -> list[dict]:
    import Levenshtein
    from lxml import html
    from src.core.preprocess import normalized_table
    from src.metrics.cal_metric import call_TEDS

    normal, extra = [], []
    for index, row in enumerate(pairs):
        pred, gold = (normalized_table(row[key], "html") for key in ("prediction_html", "html"))
        texts = []
        for value in (pred, gold):
            document = html.fromstring(value) if value else None
            cells = document.xpath(".//td") if document is not None else []
            texts.append("\n".join(" ".join(cell.text_content().split()) for cell in cells))
        edits, characters = Levenshtein.distance(texts[1], texts[0]), len(texts[1])
        extra.append({"em": int(bool(gold) and pred == gold), "char_edits": edits, "char_count": characters,
                      "cer": edits / characters if characters else None})
        normal.append({"gt": gold, "pred": pred, "img_id": row["table_id"],
                       "gt_idx": index, "pred_idx": index, "gt_attribute": [{}]})
    (output / "result").mkdir(exist_ok=True)
    scorer = call_TEDS(normal, {"teds_workers": 2})
    samples, aggregates = scorer.evaluate(save_name=output.name)
    diagnostics = scorer.debug_info
    atomic_json(output / "official_receipt.json", {"evaluator_commit": EVALUATOR_COMMIT,
                "diagnostics": diagnostics, "aggregates": aggregates}, immutable=True)
    if diagnostics["error_case_count"] or diagnostics["timeout_case_count"]:
        raise RuntimeError("Official development scoring had failures; metrics were not accepted")
    for result, sample in zip(extra, samples):
        result.update(teds=sample["metric"]["TEDS"], teds_s=sample["metric"]["TEDS_structure_only"])
    return extra


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", type=Path)
    parser.add_argument("--references", type=Path)
    parser.add_argument("--evaluator", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    evaluator = args.evaluator.resolve()
    revision = subprocess.check_output(["git", "-C", str(evaluator), "rev-parse", "HEAD"], text=True).strip()
    dirty = subprocess.check_output(["git", "-C", str(evaluator), "status", "--porcelain", "--untracked-files=no"], text=True)
    if revision != EVALUATOR_COMMIT or dirty:
        raise ValueError("Official evaluator differs from the frozen scoring implementation")
    sys.path.insert(0, str(evaluator))
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    # The upstream scorer writes relative result files. Isolate those writes in this new directory.
    import os
    if args.self_test:
        markup = '<table><tr><td rowspan="2">A</td><td></td></tr><tr><td>B</td></tr></table>'
        rows = [{"table_id": "dev:synthetic", "scale": 1.0, "evaluation_split": "development",
                 "prediction_html": markup, "teds": None, "teds_s": None, "em": None, "cer": None}]
        refs = [{"table_id": "dev:synthetic", "html": markup}]
    else:
        if args.profile is None or args.references is None:
            parser.error("--profile and --references are required without --self-test")
        rows, refs = read_jsonl(args.profile), read_jsonl(args.references)
    os.chdir(output)
    result = score_profile(rows, refs, lambda pairs: official_scores(pairs, output))
    if args.self_test:
        assert result[0]["teds"] == result[0]["teds_s"] == result[0]["em"] == 1.0
        assert result[0]["cer"] == 0.0
    write_profile(output / "profile_scored.jsonl", result)
    print(json.dumps({"scored": len(result), "synthetic_self_test": args.self_test, "model_loaded": False}))


if __name__ == "__main__":
    main()
