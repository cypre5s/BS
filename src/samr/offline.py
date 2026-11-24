"""Optional post-inference scoring; this module is not imported by the engine."""

from collections.abc import Callable

from .development import MAX_DEV_IMAGES


def score_profile(rows: list[dict], references: list[dict], scorer: Callable[[list[dict]], list[dict]]) -> list[dict]:
    ids = {row["table_id"] for row in rows}
    if (not 1 <= len(ids) <= MAX_DEV_IMAGES or any(row["evaluation_split"] != "development" for row in rows)
            or any(not table_id.startswith("dev:") for table_id in ids)):
        raise ValueError("Offline profiling is restricted to isolated development samples")
    references_by_id = {row["table_id"]: row["html"] for row in references}
    if len(references_by_id) != len(references) or set(references_by_id) != ids:
        raise ValueError("Development references must match the profiled IDs exactly")
    keys = [(row["table_id"], row["scale"]) for row in rows]
    if len(set(keys)) != len(keys):
        raise ValueError("Repeated table/scale pair")
    pairs = [{"prediction_html": row["prediction_html"], "html": references_by_id[row["table_id"]],
              "table_id": row["table_id"], "scale": row["scale"]} for row in rows]
    scores = scorer(pairs)
    if len(scores) != len(rows):
        raise ValueError("Scorer returned an incomplete development result")
    result = []
    for row, score in zip(rows, scores):
        if not {"teds", "teds_s", "em", "cer"} <= score.keys():
            raise ValueError("Missing official metric fields")
        result.append({**row, **score, "scored_offline": True})
    return result
