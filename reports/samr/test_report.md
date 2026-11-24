# SAMR correctness and smoke report

Validation date: 2026-11-24. The final full CPU suite passed **263 tests**, with one
state-dependent historical test skipped. All **73 new SAMR tests passed**. The skipped
test asks whether the old evaluation is still active; the actual evaluation is already
terminal. It is not a SAMR correctness failure. The existing official CPU scorer test
also ran successfully. One WeasyPrint dependency deprecation warning is unrelated to
SAMR inference. Ruff passed; Mypy passed across 43 source files; all SAMR modules
imported in the CPU environment without importing Torch. Twelve pinned configurations
passed validation. The native processor audit passed 30 cases with zero model loads.

Evidence is under `experiments/samr/` locally and deliberately excluded from Git:
`final_full_results.xml`, `final_full_tests.log`, `static_validation.json`,
`processor_audit.json`, `real_gpu_v2/`, `official_self_test/scored/` and
`history_preservation.json`. The static receipt binds 41 SAMR/runtime/config/test/script
files to their hashes. The final suite includes the stricter configuration type checks.

## Real GPU interface results

The final smoke run used three existing PubTabNet validation images selected only by
predeclared image-size limits, without looking at reference HTML or accuracy. All were
checked against the exact 665-table exclusion manifest. Each ran 1.0, 1.5 and 1.75 scales.
All nine normal outputs reached EOS, were untruncated, had valid OTSL and official HTML,
and passed canonical/native metadata checks. The fixed profile intentionally ran all
three scales, even though A3's recorded features classified these samples MEDIUM.

| Dev sample | Original pixels | Tokens at 1.0 / 1.5 / 1.75 | Output tokens at each scale | Pipeline seconds | Fusion |
| --- | --- | --- | --- | ---: | --- |
| 1 | 503×259 | 128 / 288 / 392 | 790 / 790 / 790 | 65.91 | Retained B0; no supported structural repair |
| 2 | 488×198 | 90 / 207 / 297 | 625 / 573 / 573 | 48.29 | Retained B0; auxiliary dimensions differed |
| 3 | 396×177 | 72 / 152 / 220 | 1016 / 986 / 986 | 85.28 | Retained B0; no supported structural repair |

These are interface/cost observations, not accuracy scores. No real reference metrics
were computed for selection or tuning. Positive fusion is exercised by explicitly
synthetic geometry fixtures; no real development output was modified in this smoke.

One resident original BF16 model served the final run: nine scale calls, one native B0
parity call and one deadline probe, **11 calls in total**. Model loading took 5.24 seconds
outside the warm table budgets. Peak allocated CUDA memory was 1,816,297,984 bytes
(about 1.69 GiB). Parameters remained frozen. All six pinned snapshot hashes matched
before and after; weight SHA-256 remained
`d4551e9f6127ac9e35cff7b4f7a72a90410e03e6c8676a36f0e2b7eaeb5d5348`.

The native B0 parity call matched raw OTSL, official HTML, processed size, grid, visual
and output counts, EOS/truncation and prompt/model identity, with no mismatched fields.
The 0.250-second deadline probe stopped after seven output tokens in 0.257704 seconds.
Its intentionally truncated partial output was discarded, separate from the nine
normal outputs. The observed 7.704-millisecond overshoot confirms the documented
nonpreemptible-operation limitation. No hard wall-clock guarantee is asserted.

An earlier three-image smoke also passed nine calls before the final reservation guard
and diagnostic probes were added. Its immutable `real_gpu_v1/` evidence remains local;
the results above refer to the final `real_gpu_v2/` run.

## Q1–Q16 evidence

Test filenames below are under `tests/samr/` unless specified otherwise. Native/GPU
scripts are executable tests as well as diagnostic tools. Mock tests are never
substituted for the separate real processor/GPU evidence.

| Question | Test code | Result and evidence | Verdict |
| --- | --- | --- | --- |
| Q1 Official processor at every scale? | `scripts/audit_samr_processor.py`; `test_runtime_guards.py::test_native_runtime_checks_actual_tokens_before_generate`; existing `tests/test_runtime.py::test_global_prompt_and_input_preserved` | 30 native CPU cases and all nine GPU scale calls passed through the shared processor; `processor_audit.json`, `real_gpu_v2/audit/*.json` | PASS |
| Q2 Actual visual counts recorded? | `test_engine.py::test_fusion_exception_falls_back_and_logs_complete_evidence`; native processor audit | Placeholder counts equal native grid/4; real counts 72–392, recorded separately from estimates | PASS |
| Q3 Aspect ratio / cropping? | `test_scales_routing.py::test_aspect_ratio_crop_and_native_grid` (25 cases), `test_native_limits_extreme_aspect_and_duplicate_sizes` | One isotropic preparation factor, ≤0.5-pixel axis rounding, both colored corner markers preserved; official grid quantization reported, not hidden | PASS within native quantization |
| Q4 Grid consistency? | `test_engine.py::test_grid_metadata_drift_is_rejected_with_raw_preserved`; native processor audit | Deliberate drift rejected; all nine real grids/processed sizes/token counts verified | PASS |
| Q5 Unified OTSL parsing? | `test_canonical_fusion.py::test_normal_cells_blanks_empty_declared_rows_and_exact_text`, `test_invalid_otsl_and_injected_html_are_unresolved` | Official parser and one canonical validator used across scales; all nine real outputs resolved | PASS |
| Q6 Different cell counts incorrectly indexed? | `test_unequal_dimensions_never_align_by_cell_index`, `test_different_cell_counts_and_spans_map_by_occupied_regions` | Different dimensions unresolved; split groups use occupied-coordinate unions; real sample 2 retained B0 | PASS |
| Q7 Row/column spans mapped? | `test_combined_spans_and_multilevel_headers`, `test_vertical_and_combined_spans_align_to_full_occupied_region` | Rowspan, colspan and combined spans retain origin/coverage; contradictory ownership rejected | PASS |
| Q8 Blank cells retained? | `test_normal_cells_blanks_empty_declared_rows_and_exact_text`, `test_real_structure_edit_requires_image_evidence_and_preserves_anchor` | Declared blanks/blank rows retained; blank split preserves full global lattice and original content | PASS |
| Q9 Text conflict overwrites B0? | `test_visual_evidence_absence_auxiliary_conflict_and_text_conflict_retain_B0` | Longer conflicting text and majority disagreement cannot override B0; exact payload retained | PASS |
| Q10 Fusion failure fallback? | `test_engine.py::test_fusion_exception_falls_back_and_logs_complete_evidence`, `test_invalid_or_conflicting_auxiliaries_retain_original` | Exceptions, invalid OTSL/HTML, differing dimensions and failed auxiliary inference retain B0 | PASS |
| Q11 A3 consumes GT? | `test_scales_routing.py::test_complexity_has_only_image_and_actual_native_token_features`, `test_development_profile.py::test_exclusion_blocks_renamed_reencoded_cohort_images_and_gt_fields` | GT argument/input fields rejected; renamed/re-encoded excluded pixels blocked; offline scoring has a separate interface | PASS |
| Q12 One model load? | `test_engine.py::test_modes_share_one_model_and_leave_mock_weights_unchanged`; `scripts/run_samr_profile.py --real-smoke` | Mock and actual final GPU run each report one load; 11 real calls share the instance | PASS |
| Q13 Weight hashes unchanged? | Existing `tests/test_runtime.py::test_snapshot_hash_guard_is_read_only`; processor/GPU audit scripts | Six before/after hashes equal; actual model parameters frozen; `real_gpu_v2/runtime_receipt.json` | PASS |
| Q14 Strict budgets? | `test_budget_boundaries_tokens_count_duplicates_and_time`, `test_token_overbudget_preflight_and_extras_are_not_executed`, all `test_runtime_guards.py`; real deadline probe | Token/scale reservations and actual pre-generation guards pass. No new calls admitted after deadline. GPU probe records a 7.704 ms overshoot | Token/scale PASS; cooperative stop PASS; hard wall-clock bound NOT guaranteed |
| Q15 Same prompt / generation? | `test_runtime_guards.py::test_generation_contract_is_pinned_and_runtime_is_not_forked`; processor audit; native replay | CPU non-image prompt IDs equal, pinned generation hash identical, native B0 replay zero mismatches. Deadline safety guard is separately recorded | PASS |
| Q16 Complete decisions/audit? | `test_engine.py::test_fusion_exception_falls_back_and_logs_complete_evidence`; `test_positive_fusion_through_pipeline` | Immutable per-table original/all-scale/final OTSL, canonical grids, cost metadata, routing events and fusion reasons; repeat write refused | PASS |

The acceptance scope is code correctness and smoke validation. Q14's negative hard
wall-clock answer is explicit; it is not reported as a strict-latency pass. A hard
real-time service would require a different execution/isolation contract.

## Other integrity checks

`scripts/score_samr_profile.py --self-test` passed with the unchanged pinned official
CPU evaluator. Its synthetic identical-table TEDS, TEDS-S and EM were 1, CER 0, with
zero scorer errors/timeouts. This verifies metric wiring only, not real-image accuracy.

All **6,720 historical artifacts** matched the initial preservation snapshot after
smoke/testing. Original Method A/B/C files and the vendor converter are unchanged.
No formal SAMR OmniDocBench job was started. Publication scans and commit details
are recorded separately in `git_release_report.md`.

| Local evidence | SHA-256 |
| --- | --- |
| `processor_audit.json` | `a5c8ea80aaef2a1a7a15be6ad7bc1b063b0c67eb95a360afc2aae18cd79eefc7` |
| `real_gpu_v2/summary.json` | `349833cee61dcdb03b060842fa0ef09966dcf6050095376c11d155a40ee0b072` |
| `real_gpu_v2/runtime_receipt.json` | `97f9d25bdaef32376fc1a6a8a75cf8d09953f755be9cba969b753640e2e1e462` |
| `real_gpu_v2/native_B0_parity.json` | `1cf72af702124442c48a81d733864549a8aaea4f393ec567e5d88ec2deeebf82` |
| `real_gpu_v2/deadline_probe.json` | `84b721a5e3878e97c5db6a2dc1e97e910aa9eded51c446c5960d35956c189352` |
| `official_self_test/scored/official_receipt.json` | `4235cd5c4a7ce551e9db1022f6e448fe2206d8159e473e245be8bfd41093e806` |
| `history_preservation.json` | `a66618253afe10c959409c075671e05862b97050143e3bf146e26b852500fc77` |
