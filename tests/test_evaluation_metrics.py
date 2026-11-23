"""Additional metrics and comparisons use identical reference cohorts."""

import pytest

from evaluation.metrics import structure_signature, summarize
from evaluation.paired_analysis import paired_comparison


def metric_row(index, **changes):
    return {"id": f"fixture-{index}", "page": f"page-{index // 2}",
            "teds": 0.8, "teds_s": 0.9, "strict_em": 0, "structure_em": 1,
            "latency": 2.0, "calls": 1, "char_count": 10, "char_edits": 2,
            "changed": False, "delta_teds": 0.0, "status": "completed",
            "additional_calls": 0, "incremental_latency": 0.0, **changes}


def test_structure_exact_match_ignores_text_and_preserves_spans():
    template = '<html><body><table><tr><td rowspan="2">{}</td><td></td></tr><tr><td>A</td></tr></table></body></html>'
    left = structure_signature(template.format("Original"))
    assert left == structure_signature(template.format("Changed"))
    assert left != structure_signature(template.format("Original").replace('rowspan="2"', 'rowspan="1"'))
    assert left != structure_signature(template.format("Original").replace("<td></td>", ""))
    assert structure_signature("") is None


def test_cer_is_character_weighted_and_rate_denominators_explicit():
    rows = [metric_row(0, char_count=10, char_edits=1, changed=True, delta_teds=0.1,
                       calls=3, additional_calls=2, incremental_latency=4),
            metric_row(1, char_count=90, char_edits=9, changed=True, delta_teds=-0.05),
            metric_row(2, char_count=0, char_edits=0)]
    summary = summarize(rows)
    assert summary["cer"] == pytest.approx(0.1)
    assert summary["successful_refinement_rate"] == 0.5
    assert summary["harmful_modification_rate"] == 0.5
    assert summary["no_change_rate"] == pytest.approx(1 / 3)
    assert summary["calls"] == pytest.approx(5 / 3)
    assert summary["additional_calls"] == pytest.approx(2 / 3)
    assert summary["failure_rate"] == 0


def test_no_modifications_produce_undefined_refinement_rates():
    summary = summarize([metric_row(0)])
    assert summary["successful_refinement_rate"] is None
    assert summary["harmful_modification_rate"] is None
    assert summary["no_change_rate"] == 1
    with pytest.raises(ValueError):
        summarize([])


def test_unavailable_verification_is_reported_even_with_safe_baseline_fallback():
    rows = [metric_row(0, failure=True), metric_row(1)]
    result = summarize(rows)
    assert result["failure_rate"] == 0.5
    assert result["no_change_rate"] == 1


@pytest.mark.parametrize("candidate", [
    [metric_row(0)],
    [metric_row(0), metric_row(0)],
    [metric_row(0), metric_row(7)],
    [metric_row(0), metric_row(1, page="wrong")],
    [metric_row(0), metric_row(1, char_count=11)],
])
def test_paired_comparison_rejects_unmatched_or_changed_references(candidate):
    with pytest.raises(ValueError):
        paired_comparison([metric_row(0), metric_row(1)], candidate, draws=100)


def test_page_bootstrap_counts_and_constant_delta():
    base = [metric_row(index) for index in range(6)]
    candidate = [row | {"teds": 0.9, "char_edits": 1} for row in base]
    result = paired_comparison(base, candidate, draws=250)
    assert result["pages"] == 3
    assert result["tables"] == 6
    assert result["delta"]["teds"] == pytest.approx(0.1)
    assert result["delta"]["cer"] == pytest.approx(-0.1)
    assert result["ci95"]["teds"] == pytest.approx([0.1, 0.1])
    assert result["ci95"]["cer"] == pytest.approx([-0.1, -0.1])


def test_paired_empty_cohort_rejected_and_zero_text_cer_is_undefined():
    with pytest.raises(ValueError, match="identical unique"):
        paired_comparison([], [], draws=10)
    base = [metric_row(index, char_count=0, char_edits=0) for index in range(4)]
    candidate = [row | {"teds": 0.9} for row in base]
    result = paired_comparison(base, candidate, draws=100)
    assert result["delta"]["cer"] is None
    assert result["ci95"]["cer"] is None
    assert result["delta"]["teds"] == pytest.approx(0.1)
