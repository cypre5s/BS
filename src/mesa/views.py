"""Isotropic source-pixel views. Native resizing/patch order stay in Xiaomi."""
from dataclasses import asdict, dataclass
from math import ceil, floor, sqrt

import numpy as np
from PIL import Image

from samr.scales import ScaleLadder, ScalePlan

from .config import MESAConfig
from .crop_mapping import (
    CropPiece,
    UnresolvedMapping,
    piece_from_original,
    piece_to_original,
    snap_box,
    transform_record,
)
from .grid_detector import GridBoundaryDetector


@dataclass(frozen=True)
class VisualView:
    channel_id: str
    view_id: str
    source_image_id: str
    original_size: tuple[int, int]
    requested_box: tuple[int, int, int, int]
    crop_box: tuple[int, int, int, int]
    plan: ScalePlan
    overlap: float
    padding: int
    header_context: str
    pieces: tuple[CropPiece, ...] = ()
    mapping_quality: str = "LOW"
    grid_boundary_ids: tuple[str, ...] = ()
    row_context: tuple[int, ...] = ()
    column_context: tuple[int, ...] = ()
    grid_evidence: dict | None = None

    @property
    def local(self):
        return self.channel_id not in ("global", "highres")

    def prepare(self, image):
        if image.size != self.original_size:
            raise ValueError("View belongs to another source")
        if self.pieces:
            crop = Image.new("RGB", self.plan.original_size, "white")
            for piece in self.pieces:
                crop.paste(image.crop(piece.source_bbox), piece.canvas_bbox[:2])
        else:
            crop = image.crop(self.crop_box) if self.local else image
        return self.plan.prepare(crop)

    def to_original(self, point, *, processed=False, piece_index=None):
        size = self.plan.processed_size if processed else self.plan.actual_size
        if self.pieces:
            if piece_index is None:
                x, y = point[0]*self.plan.original_size[0]/size[0], point[1]*self.plan.original_size[1]/size[1]
                matches = [i for i,p in enumerate(self.pieces) if p.canvas_bbox[0] <= x <= p.canvas_bbox[2]
                           and p.canvas_bbox[1] <= y <= p.canvas_bbox[3]]
                if len(self.pieces) == 1:
                    matches = [0]
                if len(matches) != 1:
                    raise UnresolvedMapping("Point in padding/seam or ambiguous piece; specify piece_index")
                piece_index = matches[0]
            return piece_to_original(point, self.pieces[piece_index], self.plan.original_size, size)
        x0, y0, x1, y1 = self.crop_box
        return x0 + point[0] * (x1 - x0) / size[0], y0 + point[1] * (y1 - y0) / size[1]

    def from_original(self, point, *, processed=False, piece_index=None):
        size = self.plan.processed_size if processed else self.plan.actual_size
        if self.pieces:
            if piece_index is None:
                matches = [i for i,p in enumerate(self.pieces) if p.source_bbox[0] <= point[0] <= p.source_bbox[2]
                           and p.source_bbox[1] <= point[1] <= p.source_bbox[3]]
                if len(self.pieces) == 1:
                    matches = [0]
                if len(matches) != 1:
                    raise UnresolvedMapping("Original point not uniquely contained in a crop piece")
                piece_index = matches[0]
            return piece_from_original(point, self.pieces[piece_index], self.plan.original_size, size)
        x0, y0, x1, y1 = self.crop_box
        return (point[0] - x0) * size[0] / (x1 - x0), (point[1] - y0) * size[1] / (y1 - y0)

    def as_dict(self):
        return {**asdict(self), **transform_record(self)}


class VisualViewGenerator:
    def __init__(self, config: MESAConfig):
        self.config = config
        self.ladder = ScaleLadder()

    def _view(self, image, source, channel, index, box, scale, header="not_applicable"):
        local = channel not in ("global", "highres")
        pad = self.config.padding if local else 0
        crop, ids, quality = box, (), "HIGH" if not local else "LOW"
        if local and self.config.grid_aware_crops:
            crop, ids, quality = snap_box(box, image.size, self.detection, self.config.snap_max_distance,
                                          axes=(True, channel != "horizontal"))
        cw, ch = crop[2] - crop[0], crop[3] - crop[1]
        # ScaleLadder caps PIL allocation isotropically before native quantization.
        plan = self.ladder.plan((cw+2*pad, ch+2*pad), scale)
        pieces = (CropPiece(crop, (pad,pad,pad+cw,pad+ch), "region"),) if local else ()
        return VisualView(channel, f"{channel}_{index}", source, image.size, box, crop, plan,
                          self.config.overlap if local else 0.0, pad, header, pieces, quality, ids,
                          grid_evidence=self.detection.as_dict())

    def _vertical(self, image, source):
        c, d = self.config, self.detection
        w, h = image.size
        ys = d.rows
        use_grid = c.grid_aware_crops and d.reliable and c.vertical_segmentation_mode != "fixed_overlap"
        header_rows = 0
        if use_grid and c.header_context:
            ends = [i for i,b in enumerate(d.trusted("horizontal")) if 0 < i <= c.max_header_rows
                    and b.coordinate - ys[0] <= (ys[-1]-ys[0])*c.header_max_fraction and b.line_support >= .85]
            if not ends:
                use_grid = False
            else:
                header_rows = ends[0]
        if use_grid:
            total = len(ys)-1
            count = min(c.max_vertical_views, max(1, ceil((total-header_rows)/(c.vertical_overlap_rows+2))))
            cuts = np.linspace(header_rows, total, count+1).round().astype(int)
            ranges = [(max(header_rows, int(cuts[i])-c.vertical_overlap_rows//2),
                       min(total, int(cuts[i+1])+ceil(c.vertical_overlap_rows/2))) for i in range(count)]
            header_end = ceil(ys[header_rows]) + 1 if header_rows else 0
            boxes = [(0, floor(ys[a]) if a > header_rows else header_end, w,
                      ceil(ys[b]) if b < total else h) for a,b in ranges]
        else:
            header_end = ceil(h*c.header_fraction) if c.header_context else 0
            count = min(c.max_vertical_views, max(1,h//64))
            span = (h-header_end)/count
            boxes = [(0,max(header_end,floor(header_end+(i-c.overlap/2)*span)),w,
                      min(h,ceil(header_end+(i+1+c.overlap/2)*span))) for i in range(count)]
            ranges = [(0,0)]*count
        views, seen = [], set()
        for i,(box,(a,b)) in enumerate(zip(boxes,ranges)):
            if box[3] <= box[1]:
                continue
            row_ids = tuple(range(header_rows)) + tuple(range(a,b)) if use_grid else ()
            if i == 0:
                regions = [((0,0,w,box[3]), "header_and_data", tuple(range(b)) if use_grid else ())]
            else:
                regions = ([((0,0,w,header_end), "header", tuple(range(header_rows)))] if header_end else [])
                regions += [(box,"data",tuple(range(a,b)) if use_grid else ())]
            key = tuple(r[0] for r in regions)
            if key in seen:
                continue
            seen.add(key)
            pad, cursor, pieces = c.padding, c.padding, []
            for bbox, role, rows in regions:
                ph = bbox[3]-bbox[1]
                pieces.append(CropPiece(bbox,(pad,cursor,pad+w,cursor+ph),role,rows))
                cursor += ph+c.seam_gap
            canvas = (w+2*pad,cursor-c.seam_gap+pad)
            plan = self.ladder.plan(canvas,c.detail_scale)
            ids = tuple(v.boundary_id for v in (*d.trusted("horizontal"),*d.trusted("vertical"))) if use_grid else ()
            views.append(VisualView("vertical",f"vertical_{i}",source,image.size,box,box,plan,c.overlap,pad,
                "image_header_candidate_preserved" if header_end else "disabled",tuple(pieces),
                "HIGH" if use_grid else "LOW",ids,row_ids,
                tuple(range(len(d.columns)-1)) if use_grid else (),d.as_dict()))
        return views

    def generate(self, image: Image.Image, source_image_id: str) -> list[VisualView]:
        w, h = image.size
        if min(w, h) < 2:
            raise ValueError("Input is too small")
        c, result = self.config, []
        self.detection = GridBoundaryDetector(c).detect(image)
        result.append(self._view(image, source_image_id, "global", 0, (0, 0, w, h), 1.0))
        if "highres" in c.enabled:
            for scale in c.scales:
                result.append(self._view(image, source_image_id, "highres", scale, (0, 0, w, h), scale))
        if "horizontal" in c.enabled:
            # Move along x: full-height overlapping column strips retain the header.
            for i in range(c.strips):
                x0 = max(0, floor((i - c.overlap / 2) * w / c.strips))
                x1 = min(w, ceil((i + 1 + c.overlap / 2) * w / c.strips))
                result.append(self._view(image, source_image_id, "horizontal", i, (x0, 0, x1, h),
                                         c.detail_scale, "full_height_header_retained"))
        if "vertical" in c.enabled:
            result.extend(self._vertical(image,source_image_id))
        if "tiles" in c.enabled:
            thumb = image.convert("L")
            thumb.thumbnail((256, 256))
            gray = np.asarray(thumb, dtype=float)
            density = float((np.abs(np.diff(gray, axis=1)) > 32).mean())
            n = (3 if density >= c.density_threshold and w * h >= c.dense_image_area else 2)
            if c.tile_grid != "adaptive":
                n = int(c.tile_grid[0])
            # Adapt grid aspect while bounding either dimension by three.
            nx = min(3, max(2, round(n * sqrt(w / h)))) if c.tile_grid == "adaptive" else n
            ny = min(3, max(2, round(n * sqrt(h / w)))) if c.tile_grid == "adaptive" else n
            for j in range(ny):
                for i in range(nx):
                    box = (max(0, floor((i - c.overlap / 2) * w / nx)),
                           max(0, floor((j - c.overlap / 2) * h / ny)),
                           min(w, ceil((i + 1 + c.overlap / 2) * w / nx)),
                           min(h, ceil((j + 1 + c.overlap / 2) * h / ny)))
                    result.append(self._view(image, source_image_id, "tiles", f"{j}_{i}", box, c.detail_scale))
        # Remove exact source-region duplicates within a channel after outward snapping.
        seen, unique = set(), []
        for view in result:
            key = (view.channel_id, tuple(p.source_bbox for p in view.pieces), view.plan.actual_size)
            if key in seen:
                continue
            seen.add(key)
            unique.append(view)
        return unique
