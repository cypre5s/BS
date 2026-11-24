# Canonical grid and mapping

`src/mesa/canonical_grid.py` exposes half-open row/column starts and ends, row/column spans, cell type, exact OTSL text, source channel and source view. It delegates grammar parsing to Xiaomi through `samr.canonical.CanonicalTableGrid`.

The six OTSL tags, declared blank cells, merged cells and fully spanned rows retain their ownership semantics. Ragged rows requiring invented padding, orphan continuations, overlapping spans and ambiguous bare empty rows are UNRESOLVED. Nested tables are unsupported by the one-table strict wrapper and are rejected rather than flattened. These restrictions preserve the existing parser contract.

Global views can be compared on the shared lattice only when dimensions agree. Local views require a unique image-rule grid matching the anchor; crop boundaries must correspond to observed lines within the configured tolerance, local dimensions must equal the mapped region, and no global merged cell may cross its boundary. Local cell offsets and spans are then transformed explicitly. Cell ordinal index is never used for correspondence. A complete local table is not interpreted as a complete global table.

`tests/mesa/test_mesa.py` verifies a two-row/two-column merged cell, blank rows, invalid continuation fallback, different local/global dimensions, affine coordinate round trips, crop cuts and merged spans crossing a crop boundary. Borderless tables and ambiguous geometry retain H0.
