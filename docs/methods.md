# Methods

Method A uses native global processing and two image-only resolution variations. Extra views use the official processor's legal sizing and positional metadata. Their default merged-token budgets are 2,048 and 4,096, with 1.5× and 2× target linear scales. A result is accepted only when two distinct enhanced views terminate normally and agree in official HTML. Independent view execution is available through `run_view`.

Method B derives deterministic observation candidates from multilevel headers, density, text length, merged cells, row-width variation and boundary position. Image-detected ruling lines must establish a unique row/column mapping. It records padding, crop coordinates and scale transformations. Reconciliation preserves complete cell spans, requires unchanged anchors, limits proposed edits and refuses unsupported numeric changes. Borderless or ambiguous images retain the global result. See `reacquisition.md` for the exact acceptance conditions.

Method C uses an actual WeasyPrint HTML render rasterized with PyMuPDF. It checks cell count, order, spans, blank cells, content and font coverage. Source and rendered line geometry support verification. Its candidate repair splits an empty merged cell only when source boundaries and blank interiors provide evidence. Text-bearing cells remain part of the preservation check, and every repair requires another render. See `render_verification.md`.

All thresholds are predeclared in YAML and exercised with synthetic fixtures. These methods have conservative fallback behavior. No-change, successful-refinement and harmful-modification rates show how often their acceptance rules help.
