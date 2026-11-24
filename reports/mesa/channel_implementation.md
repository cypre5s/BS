# Channel implementation

All view plans are implemented in `src/mesa/views.py`; all native processing remains in `src/xiaomi_runtime/adapter.py` and `processor.py`.

| Channel | Definition | Source coordinates and limitations |
|---|---|---|
| Global Original | Original PIL object, scale 1.0 | No added resize; identical runtime configuration to B0 |
| Global High Resolution | 1.25, 1.5 and 1.75 isotropic candidates | PIL allocation capped by native maximum pixels; actual native grid audited |
| Horizontal Detail | Move along x, overlapping left/right column strips | Full height retains header; clipped padding; no pasted context |
| Vertical Detail | Move along y, contiguous prefixes retaining header and adjacent rows | Later prefixes may approach the whole table; more redundant but no fabricated coordinate continuity |
| Overlapping Tiles | 2×2, 3×3, or density/area/aspect-adaptive 2–3 rows and columns | Positive overlap; full coverage; maximum nine tile proposals with independent budget admission |

Coordinates use half-open pixel boxes. `requested_box` is the intended observation; `crop_box` includes clipped padding. `to_original`/`from_original` handle both actual scaled pixels and native processed pixels. Integer resize rounding is at most half a pixel per axis before native quantization. No arbitrary independent width/height target is supplied to the runtime. The native processor may round dimensions to its patch/merge lattice; this unavoidable quantization is recorded rather than described as mathematically exact aspect preservation.

E1–E4 each contain the global anchor and one auxiliary channel; E5 runs all channels without fusion; E6 adds evidence aggregation. The baseline and six experiment definitions are the seven YAML files in `configs/mesa/`. A specified standalone auxiliary channel still acquires H0 first, and `skip_channels` never permits skipping the global anchor.
