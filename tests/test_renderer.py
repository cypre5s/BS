"""Public HTML syntax fixtures, authored independently of any benchmark image."""

from pathlib import Path

import pytest

from render_verify.html_renderer import HTMLRenderer, RenderError
from render_verify.structural_checker import inspect_html

# These are executable HTML syntax examples, not sampled benchmark answers.
FORMATS = [
    ("<table><tr><td>A</td></tr></table>", 1, 1, ["A"]),
    ("<table><tr><td></td></tr></table>", 1, 1, [""]),
    ("<table><tr><td>A</td><td>B</td></tr></table>", 1, 2, ["A", "B"]),
    ("<table><tr><td>A</td></tr><tr><td>B</td></tr></table>", 2, 1, ["A", "B"]),
    (
        '<table><tr><td colspan="2">Title</td></tr><tr><td>A</td><td>B</td></tr></table>',
        2,
        2,
        ["Title", "A", "B"],
    ),
    ('<table><tr><td rowspan="2">A</td><td>B</td></tr><tr><td>C</td></tr></table>', 2, 2, ["A", "B", "C"]),
    (
        '<table><tr><td rowspan="2" colspan="2">A</td><td>B</td></tr><tr><td>C</td></tr></table>',
        2,
        3,
        ["A", "B", "C"],
    ),
    (
        "<table><thead><tr><th>Header</th></tr></thead><tbody><tr><td>Body</td></tr></tbody></table>",
        2,
        1,
        ["Header", "Body"],
    ),
    ("<table><tr><td>&amp; &lt; &gt;</td></tr></table>", 1, 1, ["& < >"]),
    ("<table><tr><td><b>A</b><sup>2</sup><sub>x</sub></td></tr></table>", 1, 1, ["A2x"]),
    ("<table><tr><td>First\nSecond</td></tr></table>", 1, 1, ["First\nSecond"]),
    ('<table><tr><td colspan="2"></td></tr><tr><td></td><td></td></tr></table>', 2, 2, ["", "", ""]),
]


@pytest.fixture(scope="module")
def renderer():
    return HTMLRenderer()


@pytest.mark.parametrize("markup,rows,columns,texts", FORMATS)
def test_public_html_formats_preserve_structure(renderer, markup, rows, columns, texts):
    rendered = renderer.render(markup)
    assert rendered.structure.rows == rows
    assert rendered.structure.columns == columns
    assert [cell.text for cell in rendered.structure.cells] == texts
    assert inspect_html(rendered.html).signature == inspect_html(markup).signature
    assert len(rendered.cell_boxes) == len(rendered.structure.cells)
    assert len(rendered.column_boundaries) == columns + 1
    assert len(rendered.row_boundaries) == rows + 1
    assert rendered.image.width > 0 and rendered.image.height > 0
    for cell, (left, top, right, bottom) in zip(rendered.structure.cells, rendered.cell_boxes, strict=True):
        assert left == rendered.column_boundaries[cell.column]
        assert right == rendered.column_boundaries[cell.column + cell.colspan]
        assert top == rendered.row_boundaries[cell.row]
        assert bottom == rendered.row_boundaries[cell.row + cell.rowspan]


def test_external_resources_are_rejected(renderer):
    with pytest.raises(RenderError, match="unsupported element"):
        renderer.render('<table><tr><td><img src="https://example.org/a.png"></td></tr></table>')


def test_missing_font_fails_closed():
    with pytest.raises(RenderError, match="does not exist"):
        HTMLRenderer(font_paths=["a-font-that-does-not-exist.ttf"])


def test_glyph_coverage_is_verified():
    renderer = HTMLRenderer()
    renderer._glyphs = {ord("A")}
    with pytest.raises(RenderError, match="does not cover"):
        renderer.render("<table><tr><td>B</td></tr></table>")


def test_cjk_font_if_provisioned():
    path = Path("experiments/resources/NotoSansCJKsc-Regular.otf")
    if not path.exists():
        pytest.skip("CJK font is an external runtime resource")
    rendered = HTMLRenderer(font_paths=[str(path)]).render(
        "<table><tr><td>中文表格</td><td>金额</td></tr></table>"
    )
    assert [cell.text for cell in rendered.structure.cells] == ["中文表格", "金额"]
