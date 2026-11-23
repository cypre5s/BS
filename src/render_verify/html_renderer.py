"""Render official HTML with a real HTML/CSS engine and inspect its geometry."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from lxml import etree, html
from PIL import Image

from .structural_checker import HTMLStructure, inspect_html


class RenderError(ValueError):
    """Rendering is incomplete or cannot preserve the table's text and topology."""


@dataclass
class RenderedTable:
    image: Image.Image
    structure: HTMLStructure
    column_boundaries: tuple[float, ...]
    row_boundaries: tuple[float, ...]
    cell_boxes: tuple[tuple[float, float, float, float], ...]
    html: str


def _font_paths(configured: list[str] | None) -> list[Path]:
    paths = [Path(value).expanduser() for value in (configured or [])]
    paths.extend(
        Path(value) for value in os.environ.get("BS_RENDER_FONT_PATHS", "").split(os.pathsep) if value
    )
    default = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
    if default.is_file():
        paths.append(default)
    return list(dict.fromkeys(paths))


def _safe_table(markup: str) -> str:
    root = html.fragment_fromstring(markup, create_parent="div")
    table = root.find(".//table")
    if table is None:
        raise RenderError("HTML contains no table")
    allowed = {
        "table",
        "thead",
        "tbody",
        "tfoot",
        "tr",
        "td",
        "th",
        "b",
        "strong",
        "i",
        "em",
        "sup",
        "sub",
        "br",
        "span",
    }
    for element in table.iter():
        if not isinstance(element.tag, str) or element.tag not in allowed:
            raise RenderError("HTML contains an unsupported element")
        attributes = {key: value for key, value in element.attrib.items() if key in ("rowspan", "colspan")}
        element.attrib.clear()
        element.attrib.update(attributes)
    for index, cell in enumerate(table.xpath(".//td|.//th")):
        cell.set("data-cell-index", str(index))
    return etree.tostring(table, encoding="unicode", method="html")


class HTMLRenderer:
    """CPU renderer with bounded pages and explicit glyph coverage checks."""

    def __init__(
        self, *, font_paths: list[str] | None = None, font_size: int = 14, max_dimension: int = 32768
    ):
        self.font_paths = _font_paths(font_paths)
        self.font_size = font_size
        self.max_dimension = max_dimension
        self._font_css, self._glyphs = self._load_fonts()

    def _load_fonts(self) -> tuple[str, set[int]]:
        from fontTools.ttLib import TTFont

        rules = []
        glyphs: set[int] = set()
        for index, path in enumerate(self.font_paths):
            if not path.is_file():
                raise RenderError("Configured renderer font does not exist")
            with TTFont(path, fontNumber=0, lazy=True) as font:
                glyphs.update((font.getBestCmap() or {}).keys())
            rules.append(
                f"@font-face {{font-family: TableFont{index}; src: url('{path.resolve().as_uri()}');}}"
            )
        if not rules:
            raise RenderError("Configure a font with verified glyph coverage")
        return "\n".join(rules), glyphs

    def render(self, markup: str) -> RenderedTable:
        import pymupdf
        from weasyprint import HTML

        structure = inspect_html(markup)
        required = {ord(char) for cell in structure.cells for char in cell.text if not char.isspace()}
        if required - self._glyphs:
            raise RenderError("A renderer font does not cover all table characters")
        safe = _safe_table(markup)
        if inspect_html(safe).signature != structure.signature:
            raise RenderError("HTML sanitization altered the table")
        width = min(4096, max(320, structure.columns * 120))
        families = ", ".join(f"TableFont{index}" for index in range(len(self.font_paths)))
        css = (
            self._font_css
            + f"""
            @page {{size: {width + 8}px {self.max_dimension}px; margin: 4px;}}
            html, body {{margin: 0; padding: 0;}}
            table {{width: {width}px; table-layout: fixed; border-collapse: collapse;}}
            td, th {{border: 1px solid black; padding: 6px; white-space: pre-wrap;
                     overflow-wrap: anywhere; font-family: {families};
                     font-size: {self.font_size}px; line-height: 1.3; font-weight: normal;}}
            tr {{height: 32px;}}
        """
        )
        document = HTML(string=f"<style>{css}</style>{safe}", url_fetcher=self._fetch_font).render()
        if len(document.pages) != 1:
            raise RenderError("The table exceeds a single bounded render page")
        boxes = list(document.pages[0]._page_box.descendants())
        tables = [box for box in boxes if type(box).__name__ == "TableBox"]
        row_boxes = [box for box in boxes if type(box).__name__ == "TableRowBox"]
        cell_boxes = [box for box in boxes if type(box).__name__ == "TableCellBox"]
        if len(tables) != 1 or len(row_boxes) != structure.rows or len(cell_boxes) != len(structure.cells):
            raise RenderError("Rendered rows or cells differ from the HTML")
        table = tables[0]
        if len(table.column_widths) != structure.columns:
            raise RenderError("Rendered column count differs from the HTML")
        for source, box in zip(structure.cells, cell_boxes, strict=True):
            if (box.grid_x, box.rowspan, box.colspan) != (source.column, source.rowspan, source.colspan):
                raise RenderError("The HTML engine altered a cell span")
        left, top = float(table.position_x), float(table.position_y)
        right, bottom = left + table.border_width(), top + table.border_height()
        if right > document.pages[0].width or bottom > document.pages[0].height:
            raise RenderError("The table extends beyond the render page")
        clip = pymupdf.Rect(max(0, left - 2), max(0, top - 2), right + 2, bottom + 2)
        pdf = document.write_pdf()
        with pymupdf.open(stream=pdf, filetype="pdf") as pages:
            # CSS pixels are 96 DPI. PDF points are 72 DPI.
            pdf_clip = pymupdf.Rect(clip.x0 * 0.75, clip.y0 * 0.75, clip.x1 * 0.75, clip.y1 * 0.75)
            pixmap = pages[0].get_pixmap(matrix=pymupdf.Matrix(4 / 3, 4 / 3), clip=pdf_clip, alpha=False)
            image = Image.frombytes("RGB", (pixmap.width, pixmap.height), pixmap.samples)
        columns = tuple(float(position - clip.x0) for position in table.column_positions)
        columns += (float(table.column_positions[-1] + table.column_widths[-1] - clip.x0),)
        rows = tuple(float(box.position_y - clip.y0) for box in row_boxes)
        rows += (float(row_boxes[-1].position_y + row_boxes[-1].height - clip.y0),)
        bounds = tuple(
            (
                columns[cell.column],
                rows[cell.row],
                columns[cell.column + cell.colspan],
                rows[cell.row + cell.rowspan],
            )
            for cell in structure.cells
        )
        return RenderedTable(image, structure, columns, rows, bounds, safe)

    def _fetch_font(self, url: str, *args, **kwargs):
        from weasyprint import default_url_fetcher

        if url not in {path.resolve().as_uri() for path in self.font_paths}:
            raise RenderError("External resources are disabled in table rendering")
        return default_url_fetcher(url, *args, **kwargs)
