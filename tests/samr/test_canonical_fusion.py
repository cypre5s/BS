import pytest

from samr.canonical import CanonicalTableGrid, align_grids
from samr.fusion import fuse_global_anchor

ANCHOR = "<ecel><lcel><nl><fcel>A<fcel>B<nl>"
SPLIT = "<ecel><ecel><nl><fcel>A<fcel>B<nl>"


def grid(raw, scale=1.0):
    return CanonicalTableGrid.from_otsl(raw, scale)


def test_normal_cells_blanks_empty_declared_rows_and_exact_text():
    source = "<fcel>  1, 000.00  <ecel><nl><ecel><ecel><nl>"
    table = grid(source)
    assert table.status == "RESOLVED"
    assert (table.rows, table.columns) == (2, 2)
    assert len(table.cells) == 4
    assert table.cells[0].text == "  1, 000.00  "
    assert sum(cell.cell_type == "ecel" for cell in table.cells) == 3
    assert all(cell.source_token_span is None for cell in table.cells)


def test_combined_spans_and_multilevel_headers():
    table = grid("<fcel>H<lcel><fcel>J<nl><ucel><xcel><fcel>K<nl><fcel>A<fcel>B<fcel>C<nl>")
    assert table.status == "RESOLVED"
    assert table.cells[0].topology == (0, 0, 2, 2)
    assert table.cells[-1].origin == (2, 2)


def test_invalid_otsl_and_injected_html_are_unresolved():
    for raw in ("not OTSL", "<nl><fcel>A<nl>", "<fcel>A<fcel>B<nl><fcel>C<nl>",
                "<lcel><nl>", "<ecel>lost text<nl>", "<fcel>A<lcel><nl><ucel><fcel>B<nl>",
                "<fcel><table><tr><td>nested</td></tr></table><nl>"):
        assert grid(raw).status == "UNRESOLVED", raw


def test_unequal_dimensions_never_align_by_cell_index():
    first = grid("<fcel>A<fcel>B<nl>")
    second = grid("<fcel>A<nl><fcel>B<nl>", 1.5)
    aligned = align_grids(first, second)
    assert aligned.status == "UNRESOLVED"
    assert not aligned.groups


def test_different_cell_counts_and_spans_map_by_occupied_regions():
    aligned = align_grids(grid(ANCHOR), grid(SPLIT, 1.5))
    assert aligned.status == "MAPPED"
    split = aligned.groups[0]
    assert split["relation"] == "split"
    assert split["anchor_origins"] == ((0, 0),)
    assert split["auxiliary_origins"] == ((0, 0), (0, 1))
    assert aligned.groups[1]["anchor_origins"] == aligned.groups[1]["auxiliary_origins"] == ((1, 0),)


@pytest.mark.parametrize("before,after,area", [
    ("<ecel><fcel>A<nl><ucel><fcel>B<nl>", "<ecel><fcel>A<nl><ecel><fcel>B<nl>", 2),
    ("<ecel><lcel><nl><ucel><xcel><nl>", "<ecel><ecel><nl><ecel><ecel><nl>", 4),
])
def test_vertical_and_combined_spans_align_to_full_occupied_region(before, after, area):
    alignment = align_grids(grid(before), grid(after, 1.5))
    assert alignment.status == "MAPPED"
    assert alignment.groups[0]["relation"] == "split"
    assert len(alignment.groups[0]["positions"]) == area
    assert len(alignment.groups[0]["auxiliary_origins"]) == area


def test_real_structure_edit_requires_image_evidence_and_preserves_anchor(ruled_image):
    result = fuse_global_anchor(ANCHOR, grid(ANCHOR), [grid(SPLIT, 1.5)], ruled_image)
    assert result.changed and result.raw == SPLIT
    assert (grid(result.raw).rows, grid(result.raw).columns) == (2, 2)
    assert result.changes[0]["line_support"] and result.changes[0]["blank_support"]
    assert result.changes[0]["text_changed"] is False


def test_visual_evidence_absence_auxiliary_conflict_and_text_conflict_retain_B0(ruled_image):
    from PIL import Image
    image_free = fuse_global_anchor(ANCHOR, grid(ANCHOR), [grid(SPLIT, 1.5)], Image.new("RGB", (300, 200), "white"))
    assert image_free.raw == ANCHOR and not image_free.changed
    # Two votes do not override the contradictory third observation.
    conflict = fuse_global_anchor(ANCHOR, grid(ANCHOR), [grid(SPLIT, 1.25), grid(SPLIT, 1.5),
                                                       grid(ANCHOR, 1.75)], ruled_image)
    assert conflict.raw == ANCHOR
    text = SPLIT.replace("<fcel>A", "<fcel>A much longer text")
    text_conflict = fuse_global_anchor(ANCHOR, grid(ANCHOR), [grid(text, 1.5)], ruled_image)
    assert text_conflict.raw == ANCHOR
    assert text_conflict.evidence[0]["text_conflicts"][0]["decision"] == "retain_B0_text"


def test_ambiguous_text_splits_and_changed_row_counts_never_replace_anchor(ruled_image):
    baseline = "<fcel>A<lcel><nl><fcel>B<fcel>C<nl>"
    alternative = "<fcel>A<ecel><nl><fcel>B<fcel>C<nl>"
    assert fuse_global_anchor(baseline, grid(baseline), [grid(alternative, 1.5)], ruled_image).raw == baseline
    assert fuse_global_anchor(ANCHOR, grid(ANCHOR), [grid(SPLIT + "<ecel><ecel><nl>", 1.5)], ruled_image).raw == ANCHOR
