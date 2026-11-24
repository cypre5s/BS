"""Outward-only snapping and piecewise coordinates for real header pixels."""
from dataclasses import asdict, dataclass
from math import ceil, floor


class UnresolvedMapping(ValueError):
    pass


@dataclass(frozen=True)
class CropPiece:
    source_bbox: tuple[int, int, int, int]
    canvas_bbox: tuple[int, int, int, int]
    role: str
    global_rows: tuple[int, ...] = ()


def snap_box(box, size, detection, distance, axes=(True, True)):
    """Expand to nearby trustworthy boundaries, never remove target pixels."""
    w, h = size
    original = tuple(box)
    values, ids, resolved = list(box), [], True
    for axis, boundaries, limit in ((0, detection.trusted("vertical"), w),
                                    (1, detection.trusted("horizontal"), h)):
        if not axes[axis]:
            continue
        for edge in (axis, axis + 2):
            value = box[edge]
            lower = edge < 2
            if value == (0 if lower else limit):
                if boundaries:
                    ids.append(boundaries[0 if lower else -1].boundary_id)
                else:
                    resolved = False
                continue
            candidates = [b for b in boundaries if (b.coordinate <= value if lower else b.coordinate >= value)
                          and abs(b.coordinate - value) <= distance]
            if not candidates:
                resolved = False
                continue
            chosen = min(candidates, key=lambda b: abs(b.coordinate - value))
            values[edge] = floor(chosen.coordinate) if lower else ceil(chosen.coordinate)
            ids.append(chosen.boundary_id)
    if not resolved:
        return original, (), "LOW"
    snapped = tuple(values)
    assert snapped[0] <= box[0] and snapped[1] <= box[1] and snapped[2] >= box[2] and snapped[3] >= box[3]
    return snapped, tuple(ids), "HIGH" if detection.reliable else "LOW"


def piece_to_original(point, piece, canvas_size, resized_size):
    x, y = point[0] * canvas_size[0] / resized_size[0], point[1] * canvas_size[1] / resized_size[1]
    a, b, _, _ = piece.canvas_bbox
    sx, sy, _, _ = piece.source_bbox
    return x - a + sx, y - b + sy


def piece_from_original(point, piece, canvas_size, resized_size):
    a, b, _, _ = piece.canvas_bbox
    sx, sy, _, _ = piece.source_bbox
    return (point[0] - sx + a) * resized_size[0] / canvas_size[0], (point[1] - sy + b) * resized_size[1] / canvas_size[1]


def transform_record(view):
    return {"original_bbox": view.requested_box, "snapped_bbox": view.crop_box, "padding": view.padding,
            "resize_ratio_x": view.plan.actual_size[0] / view.plan.original_size[0],
            "resize_ratio_y": view.plan.actual_size[1] / view.plan.original_size[1],
            "processed_width": view.plan.processed_size[0], "processed_height": view.plan.processed_size[1],
            "source_table_id": view.source_image_id, "grid_boundary_ids": view.grid_boundary_ids,
            "pieces": [asdict(p) for p in view.pieces], "mapping_quality": view.mapping_quality,
            "row_context": view.row_context, "column_context": view.column_context,
            "tile_id": view.view_id if view.channel_id == "tiles" else None,
            "padding_is_synthetic_white_not_source_content": True}
