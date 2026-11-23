"""Page bootstrap preserves the same paired table cohort for every method."""
from __future__ import annotations

import numpy as np


def paired_comparison(base: list[dict], candidate: list[dict], draws: int = 10000) -> dict:
    left = {row["id"]: row for row in base}
    right = {row["id"]: row for row in candidate}
    if not left or len(left) != len(base) or len(right) != len(candidate) or left.keys() != right.keys():
        raise ValueError("Paired evaluation requires identical unique table IDs")
    pages = sorted({row["page"] for row in base})
    fields = ("teds", "teds_s", "strict_em", "structure_em")
    values = []
    for page in pages:
        ids = [key for key in sorted(left) if left[key]["page"] == page]
        if any(right[key]["page"] != page or left[key]["char_count"] != right[key]["char_count"] for key in ids):
            raise ValueError("Paired references differ")
        values.append([sum(right[key][field] - left[key][field] for key in ids) for field in fields]
                      + [sum(right[key]["char_edits"] - left[key]["char_edits"] for key in ids),
                         len(ids), sum(left[key]["char_count"] for key in ids)])
    matrix = np.array(values, dtype=float)
    weights = np.random.default_rng(320).multinomial(len(pages), np.full(len(pages), 1 / len(pages)), draws)
    bootstrap = weights @ matrix
    point = matrix.sum(0)
    denominators = [5, 5, 5, 5, 6]
    valid = np.all(bootstrap[:, denominators] > 0, axis=1)
    if point[6] == 0:
        intervals = np.quantile(bootstrap[:, :4] / bootstrap[:, [5, 5, 5, 5]], [.025, .975], axis=0).T
        deltas = point[:4] / point[5]
        return {"pages": len(pages), "tables": len(base), "bootstrap_draws": draws,
                "delta": {**dict(zip(fields, deltas.tolist())), "cer": None},
                "ci95": {**dict(zip(fields, intervals.tolist())), "cer": None}}
    intervals = np.quantile(bootstrap[valid, :5] / bootstrap[valid][:, denominators], [.025, .975], axis=0).T
    deltas = point[:5] / point[denominators]
    return {"pages": len(pages), "tables": len(base), "bootstrap_draws": draws,
            "delta": dict(zip((*fields, "cer"), deltas.tolist())),
            "ci95": dict(zip((*fields, "cer"), intervals.tolist()))}
