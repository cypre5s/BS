from dataclasses import replace
from hashlib import sha256

import pytest
from PIL import Image, ImageDraw

from reacquisition import VisualEvidenceReacquisition
from reacquisition.crop_mapper import CropTransform
from reacquisition.reconciliation import reconcile
from reacquisition.region_selector import SelectedRegion, detect_grid, select_regions
from reacquisition.uncertainty import CandidateRows, candidate_rows
from xiaomi_runtime.adapter import Prediction
from xiaomi_runtime.otsl import parse_table, to_html

ROWS = [
    ["Alpba", "North", "Label"],
    ["Gamma", "Delta", "Value"],
    ["Sigma", "Omega", "East"],
    ["West", "Name", "Group"],
    ["South", "Total", "Unit"],
]


def otsl(rows):
    return "".join("".join("<fcel>" + text for text in row) + "<nl>" for row in rows)


def table_image():
    image = Image.new("RGB", (301, 181), "white")
    draw = ImageDraw.Draw(image)
    horizontal = (5, 25, 65, 95, 125, 175)
    vertical = (5, 70, 190, 295)
    for y in horizontal:
        draw.line((5, y, 295, y), fill="black", width=1)
    for x in vertical:
        draw.line((x, 5, x, 175), fill="black", width=1)
    for row, values in enumerate(ROWS):
        for column, text in enumerate(values):
            draw.text((vertical[column] + 4, horizontal[row] + 3), text, fill="black")
    return image


def prediction(raw, method="B0", view_type="global", **kwargs):
    return Prediction(raw, to_html(raw), "mock-table", method, view_type,
                      input_size=(301, 181), processed_size=(320, 192),
                      output_hash=sha256(raw.encode()).hexdigest(), **kwargs)


class MockRuntime:
    def __init__(self, raw, *, truncated=False):
        self.raw = raw
        self.truncated = truncated
        self.calls = []

    def infer(self, table_id, image, *, method, view_type, resolution=None):
        self.calls.append((table_id, image.size, method, view_type, resolution))
        return replace(prediction(self.raw, method, view_type, truncated=self.truncated, eos=not self.truncated),
                       input_size=image.size, processed_size=resolution)

    def count_tokens(self, raw):
        return len(raw) + 1


def region(start=0, stop=2):
    return SelectedRegion(start, stop, 0, 3, ("test",),
                          CropTransform.create((301, 181), (5, 5, 295, 65)))


def test_detect_grid_uses_nonuniform_image_boundaries():
    geometry = detect_grid(table_image(), parse_table(otsl(ROWS)))
    assert geometry.confident
    assert geometry.horizontal == (5, 25, 65, 95, 125, 175)
    assert geometry.vertical == (5, 70, 190, 295)


def test_no_geometry_produces_no_model_calls():
    runtime = MockRuntime(otsl(ROWS[:2]))
    result = VisualEvidenceReacquisition(runtime).run(
        "mock-table", Image.new("RGB", (301, 181), "white"), prediction(otsl(ROWS)))
    assert result.prediction.raw == otsl(ROWS)
    assert not result.changed and result.calls == runtime.calls == []
    assert result.reason == "no_unambiguous_local_region"


def test_candidate_regions_cover_all_required_diagnostics():
    raw = ("<fcel>Header<lcel><fcel>A<fcel>B<nl>"
           "<fcel>Group<lcel><fcel>C<fcel>D<nl>"
           "<fcel>" + "long " * 30 + "<fcel>E<nl>")
    table = parse_table(raw)
    candidates = candidate_rows(table, raw, dense_columns=4)
    reasons = {reason for item in candidates for reason in item.reasons}
    assert {"multi_level_header", "dense_cells", "long_text", "consecutive_merges", "ragged_rows"} <= reasons
    basic = parse_table(otsl(ROWS))
    selected = select_regions([], basic, detect_grid(table_image(), basic), table_image().size)
    assert any("image_boundary" in item.reasons for item in selected)


def test_region_overlap_is_removed_deterministically():
    table = parse_table(otsl(ROWS))
    image = table_image()
    chosen = select_regions(
        [CandidateRows(0, 2, ("dense_cells",), 50), CandidateRows(1, 3, ("long_text",), 80)],
        table, detect_grid(image, table), image.size, max_regions=1)
    assert [(item.row_start, item.row_stop) for item in chosen] == [(1, 3)]


def test_reconcile_changes_only_one_mapped_cell():
    global_raw = otsl(ROWS)
    local = [list(row) for row in ROWS[:2]]
    local[0][0] = "Alpha"
    result = reconcile(global_raw, otsl(local), region())
    assert result.accepted
    assert result.raw == global_raw.replace("Alpba", "Alpha", 1)
    assert len(result.changes) == 1
    assert result.mapping[-1] == {"local_row": 1, "local_column": 2,
                                  "global_row": 1, "global_column": 2, "rowspan": 1, "colspan": 1}


def test_middle_crop_mapping_includes_row_offset():
    local = [list(row) for row in ROWS[2:4]]
    local[1][0] = "Westerly"
    result = reconcile(otsl(ROWS), otsl(local), region(2, 4))
    assert result.accepted
    assert result.raw == otsl(ROWS).replace("West", "Westerly")
    assert result.changes[0]["row"] == 3
    assert result.mapping[0]["global_row"] == 2


def test_crop_cannot_override_global_structure():
    local = [row[:2] for row in ROWS[:2]]
    result = reconcile(otsl(ROWS), otsl(local), region())
    assert not result.accepted
    assert result.raw == otsl(ROWS)
    assert result.reason == "local_shape_does_not_match_crop_mapping"


def test_numeric_change_is_held_for_more_evidence():
    rows = [list(row) for row in ROWS]
    rows[0][0] = "123.45"
    local = [list(row) for row in rows[:2]]
    local[0][0] = "128.45"
    result = reconcile(otsl(rows), otsl(local), region())
    assert not result.accepted
    assert result.reason == "numeric_change_requires_more_evidence"


def test_merge_crossing_crop_boundary_rejected():
    raw = ("<fcel>Header<fcel>X<fcel>Y<nl>"
           "<ucel><fcel>A<fcel>B<nl>"
           "<fcel>C<fcel>D<fcel>E<nl>")
    local = "<fcel>Header<fcel>A<fcel>B<nl><fcel>C<fcel>D<fcel>E<nl>"
    result = reconcile(raw, local, region(1, 3))
    assert result.reason == "merged_cell_crosses_crop_boundary"
    assert not result.accepted


def test_supported_merged_cells_keep_their_original_tokens():
    raw = ("<fcel>Header<lcel><fcel>Label<nl>"
           "<fcel>Alpba<fcel>North<fcel>South<nl>"
           "<fcel>Keep<fcel>Outside<fcel>Tail<nl>")
    local = ("<fcel>Header<lcel><fcel>Label<nl>"
             "<fcel>Alpha<fcel>North<fcel>South<nl>")
    result = reconcile(raw, local, region())
    assert result.accepted
    assert result.raw == raw.replace("Alpba", "Alpha")
    assert result.mapping[0]["colspan"] == 2


def test_ambiguous_overlapping_cell_ownership_rejected():
    raw = ("<fcel>Header<lcel><fcel>Label<nl>"
           "<ucel><fcel>Conflict<fcel>South<nl>"
           "<fcel>Keep<fcel>Outside<fcel>Tail<nl>")
    local = "<fcel>Header<lcel><fcel>Label<nl><ucel><fcel>Changed<fcel>South<nl>"
    result = reconcile(raw, local, region())
    assert not result.accepted
    assert result.reason == "cell_ownership_is_ambiguous"


def test_changed_cell_without_anchors_is_rejected():
    local = [["changed"] * 3, ["changed"] * 3]
    result = reconcile(otsl(ROWS), otsl(local), region())
    assert not result.accepted
    assert result.reason == "insufficient_unchanged_cell_anchors"


def test_method_calls_shared_runtime_and_records_transform():
    local = [list(row) for row in ROWS[:2]]
    local[0][0] = "Alpha"
    runtime = MockRuntime(otsl(local))
    result = VisualEvidenceReacquisition(runtime, {"max_regions": 1}).run(
        "mock-table", table_image(), prediction(otsl(ROWS)))
    assert result.changed
    assert result.prediction.raw == otsl(ROWS).replace("Alpba", "Alpha")
    assert len(result.calls) == len(runtime.calls) == 1
    assert result.diagnostics["accepted_regions"] == 1
    assert result.prediction.output_tokens == len(result.prediction.raw) + 1
    transform = result.diagnostics["regions"][0]["transform"]
    assert transform["original_size"] == (301, 181)
    assert transform["crop_box"] == (1, 1, 300, 70)
    assert transform["resized_size"] == runtime.calls[0][-1]


def test_incomplete_local_generation_falls_back():
    local = [list(row) for row in ROWS[:2]]
    local[0][0] = "Alpha"
    runtime = MockRuntime(otsl(local), truncated=True)
    result = VisualEvidenceReacquisition(runtime, {"max_regions": 1}).run(
        "mock-table", table_image(), prediction(otsl(ROWS)))
    assert not result.changed
    assert result.prediction.raw == otsl(ROWS)
    assert result.diagnostics["regions"][0]["decision"] == "local_generation_incomplete"


def test_mismatched_table_id_is_rejected():
    with pytest.raises(ValueError, match="different table"):
        VisualEvidenceReacquisition(MockRuntime(otsl(ROWS))).run(
            "different-id", table_image(), prediction(otsl(ROWS)))


def test_provenance_mismatch_preserves_global():
    class WrongModelRuntime(MockRuntime):
        def infer(self, table_id, image, *, method, view_type, resolution=None):
            return replace(super().infer(table_id, image, method=method, view_type=view_type,
                                         resolution=resolution), model_hash="different-model")

    local = [list(row) for row in ROWS[:2]]
    local[0][0] = "Alpha"
    result = VisualEvidenceReacquisition(WrongModelRuntime(otsl(local)), {"max_regions": 1}).run(
        "mock-table", table_image(), prediction(otsl(ROWS)))
    assert not result.changed
    assert result.prediction.raw == otsl(ROWS)
    assert result.diagnostics["regions"][0]["decision"] == "runtime_provenance_mismatch"
