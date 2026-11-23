"""Public OTSL syntax fixtures, independent of benchmark data."""

from hashlib import sha256
from pathlib import Path

import pytest

from xiaomi_runtime.otsl import parse_table, to_html


@pytest.mark.parametrize(("raw", "rows", "columns", "html"), [
    ("<fcel>A<fcel>B<nl>", 1, 2, "<table><tr><td>A</td><td>B</td></tr></table>"),
    ("<ecel><fcel>B<nl>", 1, 2, "<table><tr><td></td><td>B</td></tr></table>"),
    ("<fcel>A<lcel><nl>", 1, 2, '<table><tr><td colspan="2">A</td></tr></table>'),
    ("<fcel>A<fcel>B<nl><ucel><fcel>C<nl>", 2, 2,
     '<table><tr><td rowspan="2">A</td><td>B</td></tr><tr><td>C</td></tr></table>'),
    ("<fcel>A<lcel><fcel>B<nl><ucel><xcel><fcel>C<nl>", 2, 3,
     '<table><tr><td rowspan="2" colspan="2">A</td><td>B</td></tr><tr><td>C</td></tr></table>'),
    ("<fcel><b>A</b><fcel>&amp;<nl>", 1, 2,
     "<table><tr><td><b>A</b></td><td>&amp;</td></tr></table>"),
])
def test_official_syntax(raw, rows, columns, html):
    parsed = parse_table(raw)
    assert parsed.rows == rows
    assert parsed.columns == columns
    assert parsed.html == html == to_html(raw)


def test_coordinates_survive_combined_span():
    parsed = parse_table("<fcel>A<lcel><fcel>B<nl><ucel><xcel><fcel>C<nl>")
    assert [(cell.row, cell.column, cell.rowspan, cell.colspan, cell.text) for cell in parsed.cells] == [
        (0, 0, 2, 2, "A"), (0, 2, 1, 1, "B"), (1, 2, 1, 1, "C"),
    ]


def test_converter_source_is_exact_upstream_copy():
    vendor = Path(__file__).parents[1] / "src/xiaomi_runtime/_vendor/otsl.py"
    assert sha256(vendor.read_bytes()).hexdigest() == "db46f5a05d9742a7bcfd3931428bdd9e8de845892b66fa4cf09bd21f83cf2b14"


@pytest.mark.parametrize("raw", ["", "plain text", "<ucel><nl>"])
def test_missing_origin_cells_rejected(raw):
    with pytest.raises(ValueError):
        parse_table(raw)
