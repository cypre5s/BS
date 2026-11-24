# SAMR engineering delivery

A1, A2 and A3 are implemented with `SAMRInferenceEngine`, twelve pinned configurations
and a development profile/scoring workflow. All scales reuse the original Xiaomi
processor, prompt, generation protocol and one resident BF16 model. There is no training,
LoRA, weight update or reference-answer routing.

A1 supports single, fixed and ladder profiles with actual dimensions/grids/token counts.
A2 creates canonical grids and maps occupied regions, preserving B0's global dimensions
and all text; its implemented repair is an image-verified split of declared blank spans.
A3 records six image/token features and applies predetermined LOW/MEDIUM/HIGH routing
under cumulative token/scale limits and a cooperative deadline.

Validation: 263 full-suite tests passed, one inapplicable historical-live-state test
skipped; all 73 SAMR tests passed; Ruff, Mypy and CPU imports passed. The actual native
processor passed 30 cases. The final real GPU smoke passed three development images
at three scales, plus native B0 parity and a discarded deadline probe. Model load count
was one, all six snapshot hashes were unchanged, and the nine normal outputs were
untruncated and parsed successfully. The original 6,720 historical artifacts were
unchanged. Q1–Q16 are answered with code and local evidence in `test_report.md`.

The latency limitation is explicit: cumulative token/scale admission is strict;
native GPU computation cannot guarantee a hard wall-clock deadline. The 250 ms real
probe stopped in 257.704 ms, and its partial output was discarded. The reports do not
mislabel this as a strict wall-time guarantee.

All three real development samples retained B0. That is successful conservative
execution, not an accuracy gain. No formal full SAMR OmniDocBench evaluation was run.
Historical A/B/C full evaluations remain separate and intact. No development reference
scores were used to adjust thresholds.

Delivery documents: `existing_project_audit.md`, `xiaomi_processor_audit.md`,
`architecture_report.md`, `complexity_routing_report.md`, `fusion_logic_report.md`,
`development_profile_schema.md`, `test_report.md` and `git_release_report.md`.
Reproduction commands are in `docs/smoke_test.md`; README and architecture/method
documentation distinguish SAMR from historical experiments.

Publication is source/configuration/sanitized Markdown only. Data, model files, images,
raw predictions, local receipts, caches and secrets remain excluded. The actual pushed
commit and clean-tree verification are recorded in `git_release_report.md`.
