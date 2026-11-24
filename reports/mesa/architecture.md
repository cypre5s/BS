# MESA architecture

`VisualViewGenerator` produces source-pixel crop and isotropic scale plans. `MESAInferenceEngine` sends each admitted PIL observation through the same `XiaomiRuntime.infer` instance, which applies the native processor and original prompt/generation configuration. Each returned native grid, visual-token count, dimensions and identity hash is checked against its plan.

`CanonicalGrid` reconstructs two-dimensional ownership through the pinned official OTSL parser and the strict SAMR validation wrapper. `map_view` associates local cells with observed original-image rule boundaries, complete spans and explicit row/column offsets. Dimensions alone never establish local-to-global correspondence. `EvidenceAggregator` records independent structure, text and spatial diagnostics; the final modification primitive only permits image-supported blank-span splits preserving global text.

The result directory is keyed by table ID. A manifest binds pixel hash, source ID, runtime identity, full configuration and MESA implementation hashes. Per-view predictions are immutable. File locks prevent concurrent processing of the same table. Completed final receipts bind hashes of their per-view cache files. Successful views are reusable without model calls; incomplete attempts reserve budget before execution. A crash during an in-flight call conservatively exhausts remaining latency instead of giving unknown work a free retry.

Physical attempts, including failures, consume view and visual-token budgets. Cached successful predictions do not consume another call. Global Original has first priority, then high-resolution, horizontal, vertical and tile observations. If the required global view cannot fit or fails, execution explicitly fails rather than inventing B0. Auxiliary failures, skipped budgets and fusion exceptions retain H0 and are recorded.

Latency stops are cooperative. A GPU forward/kernel is not preemptible and may overshoot. Late auxiliary output or late fusion is discarded. Loading/weight hashing are reported setup costs outside the warm inference budget. This is a bounded development runner, not a formal benchmark service.
