"""A2: coordinate alignment with image-supported, text-preserving blank repairs."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

from PIL import Image

from render_verify.structural_checker import HTMLCell
from render_verify.visual_checker import _cell_blank, _missing_inside_span, detect_grid

from .canonical import CanonicalCell, CanonicalTableGrid, align_grids, split_blank_origins


@dataclass(frozen=True)
class FusionDecision:
    raw: str
    changed: bool
    reason: str
    evidence: tuple[dict, ...] = ()
    changes: tuple[dict, ...] = ()
    visual_evidence: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return asdict(self)


def _proposal(cell: CanonicalCell, helper: CanonicalTableGrid) -> bool:
    inside = [candidate for candidate in helper.cells if candidate.positions & cell.positions]
    return (len(inside) == len(cell.positions)
            and all(candidate.positions <= cell.positions and candidate.rowspan == candidate.colspan == 1
                    and candidate.cell_type == "ecel" and candidate.text == "" for candidate in inside))


def _context_anchors(anchor: CanonicalTableGrid, helper: CanonicalTableGrid) -> bool:
    by_origin = {cell.origin: cell for cell in helper.cells}
    nonempty = [cell for cell in anchor.cells if cell.text]
    # Unique exact text anchors pin the shared lattice; never normalize amounts or spaces.
    if len({cell.text for cell in nonempty}) < 2:
        return False
    return all(cell.origin in by_origin and by_origin[cell.origin].topology == cell.topology
               and by_origin[cell.origin].text == cell.text for cell in nonempty)


def fuse_global_anchor(
    anchor_raw: str,
    anchor: CanonicalTableGrid,
    helpers: list[CanonicalTableGrid],
    image: Image.Image,
) -> FusionDecision:
    if anchor.status != "RESOLVED":
        return FusionDecision(anchor_raw, False, "anchor_unresolved")
    evidence = []
    mapped = []
    for helper in helpers:
        alignment = align_grids(anchor, helper)
        same = {cell.topology: cell for cell in helper.cells}
        text_matches = sum(cell.topology in same and same[cell.topology].text == cell.text
                           for cell in anchor.cells)
        conflicts = [{"origin": cell.origin, "anchor_text": cell.text, "auxiliary_text": same[cell.topology].text,
                      "decision": "retain_B0_text"} for cell in anchor.cells
                     if cell.topology in same and same[cell.topology].text != cell.text]
        components = {"otsl_validity": int(helper.otsl_valid),
                      "html_validity": int(helper.html_parse_status == "valid"),
                      "grid_consistency": int(alignment.status == "MAPPED"),
                      "span_consistency": sum(cell.topology in same for cell in anchor.cells) / len(anchor.cells),
                      "text_agreement": text_matches / len(anchor.cells),
                      "cross_scale_agreement": int(tuple((c.topology, c.text) for c in anchor.cells)
                                                   == tuple((c.topology, c.text) for c in helper.cells))}
        evidence.append({"source_scale": helper.source_scale, "alignment": asdict(alignment),
                         "components": components, "validation_score": sum(components.values()) / 6,
                         "score_is_probability": False, "text_conflicts": conflicts,
                         "context_anchors": _context_anchors(anchor, helper) if alignment.status == "MAPPED" else False})
        if alignment.status == "MAPPED":
            mapped.append(helper)
    if not mapped:
        return FusionDecision(anchor_raw, False, "no_unambiguous_auxiliary_grid", tuple(evidence))
    candidates = [cell for cell in anchor.cells
                  if cell.cell_type == "ecel" and not cell.text and len(cell.positions) > 1
                  and any(_proposal(cell, helper) and _context_anchors(anchor, helper) for helper in mapped)]
    if not candidates:
        return FusionDecision(anchor_raw, False, "no_supported_structure_repair_preserve_text", tuple(evidence))

    # Reuse the existing image geometry primitives, without invoking Method C or its renderer.
    visual = detect_grid(image)
    shape = (len(visual.y_lines) - 1, len(visual.x_lines) - 1)
    visual_log = {"reason": visual.reason, "x_lines": visual.x_lines, "y_lines": visual.y_lines,
                  "detected_shape": shape, "minimum_line_support": 0.95, "maximum_blank_ink_fraction": 0.002}
    if not visual.confidence or shape != (anchor.rows, anchor.columns):
        return FusionDecision(anchor_raw, False, "image_lattice_unresolved", tuple(evidence), (), visual_log)
    accepted = set()
    changes = []
    for cell in candidates:
        contextual = [helper for helper in mapped if _context_anchors(anchor, helper)]
        # Agreement is a veto gate, not majority voting or a whole-HTML selection score.
        if not all(_proposal(cell, helper) for helper in contextual):
            changes.append({"origin": cell.origin, "accepted": False, "reason": "auxiliary_structure_conflict"})
            continue
        geometry = HTMLCell(*cell.topology, cell.text)
        line_support = _missing_inside_span(visual, geometry, 0.95)
        blank_support = _cell_blank(visual, geometry, 0.002)
        passed = line_support and blank_support
        changes.append({"origin": cell.origin, "accepted": passed,
                        "before_span": (cell.rowspan, cell.colspan), "after_origins": sorted(cell.positions),
                        "reason": "mapped_blank_split_with_visible_boundaries" if passed else "image_evidence_insufficient",
                        "line_support": line_support, "blank_support": blank_support,
                        "supporting_scales": [helper.source_scale for helper in contextual],
                        "text_changed": False})
        if passed:
            accepted.add(cell.origin)
    if not accepted:
        return FusionDecision(anchor_raw, False, "no_image_verified_repair", tuple(evidence), tuple(changes), visual_log)
    raw = split_blank_origins(anchor_raw, anchor, accepted)
    verified = CanonicalTableGrid.from_otsl(raw, 1.0)
    before_text = tuple((cell.origin, cell.text) for cell in anchor.cells if cell.text)
    after_text = tuple((cell.origin, cell.text) for cell in verified.cells if cell.text)
    if (verified.status != "RESOLVED" or (verified.rows, verified.columns) != (anchor.rows, anchor.columns)
            or before_text != after_text):
        return FusionDecision(anchor_raw, False, "fusion_roundtrip_failed_retain_B0", tuple(evidence), tuple(changes), visual_log)
    return FusionDecision(raw, raw != anchor_raw, "image_verified_global_anchored_blank_split",
                          tuple(evidence), tuple(changes), visual_log)
