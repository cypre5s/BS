# A3 complexity and budget contract

The constants below were specified from engineering ranges before development
inference. They were not selected using OmniDocBench labels, results, or development
accuracy scores. No router is trained. The estimator signature accepts only an image
and the actual native global visual-token count.

Features use a grayscale image no larger than 512×512. Dark pixels are below 160;
edges are adjacent grayscale differences of at least 32. Four-connected components
are counted. A component is text-like when its area is 3–4096 pixels, height 2–64,
width 1–64 and width/height 0.1–10. These are heuristic cost features, not OCR or
reference cell detection.

| Feature | Normalization before clamping to [0,1] | Weight |
| --- | --- | ---: |
| Larger/smaller aspect ratio | `log2(ratio) / 4` | 0.10 |
| Original pixel area | `log2(max(1, area/65536)) / 8` | 0.15 |
| Edge density | `density / 0.20` | 0.20 |
| Connected-component density | `components / thumbnail_pixels / 0.01` | 0.20 |
| Text-like pixel density | `text_like_area / thumbnail_pixels / 0.20` | 0.20 |
| Actual global visual tokens | `log2(max(1, tokens/64)) / 6` | 0.15 |

The weighted sum is recorded with every raw and normalized feature. Below 0.35 is
LOW (global only); [0.35,0.65) is MEDIUM (global + 1.5); at least 0.65 is HIGH
(global + 1.5 + 1.75). Low complexity is not a claim of high recognition accuracy.
Fixed/ladder profiles bypass this choice, but retain the same resource guards.

Default adaptive limits are three scales, 8,192 cumulative visual tokens and a
180-second cooperative processing deadline per table. The original global pass has
priority. An impossible global token budget is rejected before model work. Duplicate
native sizes, excessive scale counts and excessive token reservations are skipped with
reasons. Failed/aborted calls consume their reservation conservatively. Actual native
placeholder counts are checked before generation, so processor drift cannot consume
more tokens than reserved.

Latency has an explicit limitation: this is not a hard real-time CUDA service.
Admission checks run before calls; the same absolute deadline is passed to all scales;
native generation checks it between decode steps. No later scale is started after
expiry. A late/truncated auxiliary or late fusion is discarded. Native prefill, one
decode step, image processing and CPU fusion cannot be interrupted mid-operation, so
elapsed time may exceed the deadline. The audit records that overshoot. Model setup
is separately timed by the runner, and receipt-file serialization is outside the
reported per-table processing interval.

Q14 therefore has two answers: **strict cumulative token/scale admission passes**;
**a strict end-to-end wall-clock upper bound is not guaranteed**. Cooperative stopping
and overshoot reporting are tested, including an actual GPU deadline probe. This is a
declared platform limitation, not an unreported hard-budget pass. Process termination
would destroy the resident model and is not silently substituted for native inference.

Tests: `test_scales_routing.py::test_budget_boundaries_tokens_count_duplicates_and_time`,
`test_engine.py::test_token_overbudget_preflight_and_extras_are_not_executed`, and all
of `test_runtime_guards.py`. Real measurements appear in `test_report.md`.
