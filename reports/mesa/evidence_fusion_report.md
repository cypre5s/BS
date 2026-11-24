# Evidence aggregation

`EvidenceAggregator` separates three audit streams:

1. Structural: global/local dimensions, mapped positions, spans and adjacency.
2. Text: exact equality, numeric sequences, whitespace-only differences and character edits.
3. Spatial: original pixel crop, observed rule boundaries, complete-span containment and two-dimensional offsets.

Global Original is the default anchor. Numeric or longer text never receives automatic preference; text remains unchanged. Local helpers are projected into explicitly mapped regions for structural comparison, keeping all unrelated global cells intact. Source channel/view identifiers accompany every evidence source. Overlapping views do not become votes and are not counted as independent proof.

The first release uses the existing `fuse_global_anchor` acceptance primitive: exact context anchors, supported blank-cell span splitting, auxiliary conflict veto, visible line support and blank interior evidence, followed by canonical/HTML round-trip verification. All accepted and rejected modification reasons and source-view mappings are retained. Local shape changes, text conflicts, incomplete spans, invalid parsers or unavailable geometry trigger fallback.

This policy is deliberately narrow. It implements executable evidence aggregation and demonstrates a supported structural modification on a synthetic unit case; it does not establish useful corrections on real tables. Arbitrary text replacement, borderless-table structure repair and general split/merge reasoning remain unproven. A smoke pass is neither an accuracy comparison nor evidence of TEDS improvement.
