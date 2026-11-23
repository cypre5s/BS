"""Additional reported metrics preserve the pinned official TEDS definitions."""
from __future__ import annotations

from lxml import html


def structure_signature(normalized: str):
    if not normalized:
        return None
    document = html.fromstring(normalized, parser=html.HTMLParser(remove_comments=True))
    tables = document.xpath("body/table")
    if not tables:
        return None

    def visit(node):
        if node.tag == "td":
            return ("td", int(node.get("rowspan", 1)), int(node.get("colspan", 1)))
        return (node.tag, tuple(visit(child) for child in node))

    return visit(tables[0])


def summarize(rows: list[dict]) -> dict:
    if not rows:
        raise ValueError("Cannot summarize an empty cohort")
    fields = ("teds", "teds_s", "strict_em", "structure_em", "latency", "calls")
    result = {key: sum(row[key] for row in rows) / len(rows) for key in fields}
    characters = sum(row["char_count"] for row in rows)
    result["cer"] = sum(row["char_edits"] for row in rows) / characters if characters else None
    result["count"] = len(rows)
    changed = [row for row in rows if row["changed"]]
    result["modified_count"] = len(changed)
    result["no_change_rate"] = 1 - len(changed) / len(rows)
    result["successful_refinement_rate"] = (sum(row["delta_teds"] > 0 for row in changed) / len(changed)
                                            if changed else None)
    result["harmful_modification_rate"] = (sum(row["delta_teds"] < 0 for row in changed) / len(changed)
                                          if changed else None)
    result["failure_rate"] = sum(row["status"] != "completed" or row.get("failure", False)
                                 for row in rows) / len(rows)
    result["additional_calls"] = sum(row["additional_calls"] for row in rows) / len(rows)
    result["incremental_latency"] = sum(row["incremental_latency"] for row in rows) / len(rows)
    return result
