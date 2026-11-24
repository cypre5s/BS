# MESA validation evidence

This report covers implementation and development smoke validation only. No formal 665-table evaluation or accuracy comparison is performed.

## CPU and static validation

The final full test run completed with **292 passed, 1 skipped**. The skip is the existing historical live-state test, whose original prerequisite no longer applies. The original tests remain unchanged. MESA contributes 29 tests; the targeted MESA/publication suite reports 47 passed. Ruff, Mypy over MESA plus its CLI, CPU imports and all seven experiment configurations pass.

Local machine evidence is saved as `experiments/mesa/final_cpu_tests.xml`, `final_cpu_tests.log`, `targeted_tests.xml` and the publication/preservation JSON receipts. Generated test artifacts are intentionally not committed.

## Requirement-to-evidence map

| Question | Code | Executable evidence and limits |
|---|---|---|
| Q1 Native processor for all five channels? | `mesa/engine.py:_validate`, `xiaomi_runtime/adapter.py:infer` | Every channel invokes the same runtime; native grid/placeholders are validated. Mock channel matrix plus GPU profile. |
| Q2 Actual visual tokens? | Per-view `prediction.visual_tokens`, smoke `profile` | Actual counts, not raw pixel estimates, are recorded and compared with plans; GPU totals are appended below. |
| Q3 Aspect ratio? | `mesa/views.py:VisualView.prepare`, `samr/scales.py:ScaleLadder.plan` | `test_views_roundtrip_overlap_and_native_metadata` checks isotropic rounding within half a pixel. Native grid rounding remains explicit. |
| Q4 Crop coordinates? | `VisualView.to_original/from_original` | Tests check corners and interior points in both prepared and processed coordinate systems. |
| Q5 Tile overlap? | `VisualViewGenerator.generate` | 2×2/3×3 counts and actual overlap tested; overlap is spatial evidence, not repeated votes. |
| Q6 Different row/column counts? | `canonical_grid.py:map_view` | Local counts must equal the mapped subregion; a 2×1 local table maps into a 2×2 global table. Unresolved boundaries fall back. |
| Q7 Spans preserved? | Official parser, `CanonicalGrid`, `map_view` | 2×2 merged-cell reconstruction and crop-crossing-span rejection tests. |
| Q8 Separate text/structure? | `EvidenceAggregator`, `text_evidence` | Numeric changes, whitespace-only edits and longer text are reported but never preferred automatically. |
| Q9 Wrong B0 overwrite? | Immutable H0; `fuse_global_anchor`; `engine.py` fallback | Synthetic supported blank split is accepted; conflict and isolated-tile tests retain H0. Residual geometry error is possible; smoke is not a safety/accuracy guarantee. Historical B0 bytes are unchanged. |
| Q10 One model? | One injected `XiaomiRuntime` per engine/run | Six mock channel combinations verify `load_count == 1`; real smoke records the same check. |
| Q11 Weights unchanged? | `XiaomiRuntime.assert_weights_unchanged` | Real smoke hashes all six model/processor/tokenizer files before and after; parameters stay frozen. |
| Q12 Budget/time risk? | `BudgetManager`, `engine.py` deadline/late-output gates | All budget types, retries, incomplete auxiliary output and unknown in-flight cost tested. Native GPU operations may overshoot cooperative time limits. |

## GPU smoke

The real RTX 4070 smoke **passed** on three independent PubTabNet validation images. All five channels ran on each image, yielding **42 successful views** (14 per image), plus one native B0 compatibility replay: **43 physical calls**, one resident BF16 model load, no failed/skipped calls and no truncation. All 42 official OTSL/HTML parses and strict canonical grids resolved successfully. B0 replay matched exactly. All six pinned model/processor/tokenizer file hashes remained unchanged; parameters remained frozen.

| Channel | Views | Actual visual tokens: min–max | Mean native latency (s) |
|---|---:|---:|---:|
| global | 3 | 72–128 | 22.704 |
| highres | 9 | 105–392 | 21.898 |
| horizontal | 6 | 80–156 | 13.329 |
| vertical | 6 | 95–288 | 16.712 |
| tiles | 18 | 66–80 | 5.754 |

Peak allocated CUDA memory: 1.692 GiB. Model setup: 7.474 s. Per-table charged inference/fusion costs: 180.540 s, 237.335 s, 132.492 s.

Warm latency accounting charges actual inference, crop preparation inside the call, retries and fusion; filesystem bookkeeping and view-plan construction are outside this compute budget. It is not an end-to-end wall-clock SLA. Nonpreemptible GPU calls may overshoot and late results are discarded.

All three final outputs retained H0. This is a successful implementation smoke test, **not evidence of improved recognition accuracy**. Full per-view resolutions, native grids, token counts, coordinate transforms and hash receipts are stored locally under `experiments/mesa/real_gpu_v1/` and excluded from Git.
