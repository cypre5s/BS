"""CV verification and fail-closed repairs tested on synthetic ruled tables."""

import hashlib
from dataclasses import replace

import pytest
from PIL import Image, ImageDraw

from render_verify.html_renderer import HTMLRenderer
from render_verify.refinement import RenderGuidedVerification
from render_verify.structural_checker import StructureError, inspect_html
from render_verify.visual_checker import detect_grid, verify
from xiaomi_runtime.adapter import Prediction
from xiaomi_runtime.otsl import to_html


def grid_image(*, rows=3, columns=3, cell_width=100, cell_height=55, background="white", ink="black"):
    image = Image.new("RGB", (columns * cell_width + 1, rows * cell_height + 1), background)
    drawing = ImageDraw.Draw(image)
    for row in range(rows + 1):
        drawing.line((0, row * cell_height, image.width - 1, row * cell_height), fill=ink, width=2)
    for column in range(columns + 1):
        drawing.line((column * cell_width, 0, column * cell_width, image.height - 1), fill=ink, width=2)
    return image


def prediction(raw):
    return Prediction(
        raw=raw,
        html=to_html(raw),
        table_id="synthetic-table",
        method="B0",
        view_type="global",
        output_hash=hashlib.sha256(raw.encode()).hexdigest(),
    )


class MockRuntime:
    def __init__(self, value=None):
        self.value = value
        self.calls = []

    def infer(self, table_id, image, *, method, view_type, resolution=None):
        self.calls.append((table_id, method, view_type, resolution))
        return replace(self.value, method=method, view_type=view_type)

    def count_tokens(self, raw):
        return len(raw)


def test_blank_colspan_split_requires_original_boundaries():
    raw = "<ecel><lcel><ecel><nl><ecel><ecel><ecel><nl><ecel><ecel><ecel><nl>"
    method = RenderGuidedVerification(MockRuntime())
    result = method.run("synthetic-table", grid_image(), prediction(raw))
    assert result.changed
    assert result.reason == "confirmed_blank_span_repair"
    assert all(cell.colspan == 1 for cell in inspect_html(result.prediction.html).cells)
    assert result.diagnostics["second_verification"]["unsupported_boundaries"] == 0
    assert result.calls == []


def test_blank_rowspan_split_is_verified_again():
    raw = "<ecel><ecel><ecel><nl><ucel><ecel><ecel><nl><ecel><ecel><ecel><nl>"
    method = RenderGuidedVerification(MockRuntime())
    result = method.run("synthetic-table", grid_image(), prediction(raw))
    assert result.changed
    assert all(cell.rowspan == 1 for cell in inspect_html(result.prediction.html).cells)


def test_text_bearing_merged_cell_is_never_split():
    raw = "<fcel>123.45<lcel><ecel><nl><ecel><ecel><ecel><nl><ecel><ecel><ecel><nl>"
    result = RenderGuidedVerification(MockRuntime()).run("synthetic-table", grid_image(), prediction(raw))
    assert not result.changed
    assert result.prediction.raw == raw


def test_visible_ink_prevents_blank_cell_repair():
    raw = "<ecel><lcel><ecel><nl><ecel><ecel><ecel><nl><ecel><ecel><ecel><nl>"
    image = grid_image()
    ImageDraw.Draw(image).text((25, 20), "100.00", fill="black")
    result = RenderGuidedVerification(MockRuntime()).run("synthetic-table", image, prediction(raw))
    assert not result.changed
    assert result.prediction.raw == raw


def test_borderless_table_retains_baseline():
    raw = "<ecel><lcel><nl><ecel><ecel><nl>"
    result = RenderGuidedVerification(MockRuntime()).run(
        "synthetic-table", Image.new("RGB", (400, 200), "white"), prediction(raw)
    )
    assert not result.changed
    assert result.reason == "source_grid_is_ambiguous"


def test_inconsistent_axis_count_retains_baseline():
    raw = "<ecel><lcel><nl><ecel><ecel><nl>"
    result = RenderGuidedVerification(MockRuntime()).run("synthetic-table", grid_image(), prediction(raw))
    assert not result.changed


def test_render_style_does_not_change_line_topology():
    a = detect_grid(grid_image(background="white", ink="black"))
    b = detect_grid(grid_image(background=(230, 230, 230), ink=(80, 80, 80)))
    assert a.confidence and b.confidence
    assert a.x_lines == b.x_lines and a.y_lines == b.y_lines


def test_render_font_size_does_not_change_source_evidence():
    markup = "<table><tr><td>A</td><td>B</td><td>C</td></tr><tr><td>D</td><td>E</td><td>F</td></tr><tr><td>G</td><td>H</td><td>I</td></tr></table>"
    original = grid_image()
    small = verify(original, HTMLRenderer(font_size=11).render(markup))
    large = verify(original, HTMLRenderer(font_size=21).render(markup))
    assert small == large


def test_optional_global_call_uses_common_adapter():
    value = prediction("<ecel><ecel><ecel><nl>" * 3)
    runtime = MockRuntime(value)
    result = RenderGuidedVerification(runtime).run("synthetic-table", grid_image())
    assert len(runtime.calls) == 1 and len(result.calls) == 1
    assert runtime.calls[0][1:] == ("C", "global", None)


def test_modified_parser_output_fails_closed():
    value = replace(prediction("<fcel>A<nl>"), html="<table></table>")
    result = RenderGuidedVerification(MockRuntime()).run("synthetic-table", grid_image(), value)
    assert not result.changed
    assert result.reason == "verification_unavailable"


def test_bad_structure_rejected():
    with pytest.raises(StructureError):
        inspect_html('<table><tr><td rowspan="4">A</td></tr></table>')
    with pytest.raises(StructureError):
        inspect_html("<table><tr><td>A</td><td>B</td></tr><tr><td>C</td></tr></table>")


def test_a_legitimate_blank_merge_keeps_original_prediction():
    raw = "<ecel><lcel><ecel><nl><ecel><ecel><ecel><nl><ecel><ecel><ecel><nl>"
    image = grid_image()
    drawing = ImageDraw.Draw(image)
    drawing.rectangle((97, 3, 103, 51), fill="white")
    result = RenderGuidedVerification(MockRuntime()).run("synthetic-table", image, prediction(raw))
    assert not result.changed
    assert result.prediction.raw == raw


def test_unrelated_nonblank_cells_and_order_are_preserved():
    raw = "<ecel><lcel><fcel>Amount<nl><fcel>A<fcel>1.20<fcel>USD<nl><fcel>B<fcel>2.34<fcel>EUR<nl>"
    original = prediction(raw)
    result = RenderGuidedVerification(MockRuntime()).run("synthetic-table", grid_image(), original)
    assert result.changed
    assert inspect_html(original.html).content == inspect_html(result.prediction.html).content
    assert result.prediction.output_tokens == len(result.prediction.raw)
    assert result.prediction.model_hash == original.model_hash
    assert result.prediction.prompt_hash == original.prompt_hash
    assert result.prediction.output_hash == hashlib.sha256(result.prediction.raw.encode()).hexdigest()


def test_failed_second_verification_rolls_back(monkeypatch):
    import render_verify.refinement as module
    from render_verify.visual_checker import Verification

    actual_verify = module.verify
    calls = []

    def fail_second(image, rendered, **kwargs):
        calls.append(rendered)
        if len(calls) == 2:
            return Verification(False, (), 0, (3, 3), "mock_second_render_failure")
        return actual_verify(image, rendered, **kwargs)

    monkeypatch.setattr(module, "verify", fail_second)
    raw = "<ecel><lcel><ecel><nl><ecel><ecel><ecel><nl><ecel><ecel><ecel><nl>"
    result = RenderGuidedVerification(MockRuntime()).run("synthetic-table", grid_image(), prediction(raw))
    assert len(calls) == 2
    assert not result.changed
    assert result.prediction.raw == raw
