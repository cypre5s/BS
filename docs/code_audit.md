# Code audit evidence

| Question | Evidence and scope |
| --- | --- |
| Q1, processor and token flow | `runtime_audit.md` identifies native functions. The CPU audit uses seven synthetic shapes and 256 numbered patches. Its local JSON records metadata, hashes and CUDA-uninitialized status. |
| Q2, weight preservation | Six fixed snapshot hashes are checked before model load and after runs. CPU audit hashes matched before and after. `test_checkpoint_preservation.py` tests immutable records. Post-gate GPU evidence is recorded in `real_startup.json`. |
| Q3, crop mapping | `test_crop_mapping.py` covers six forward/inverse coordinate cases including padding, clipping and scaling. Reconciliation tests verify cell anchoring and refusal at crossing spans. |
| Q4, official OTSL | The unchanged deployment converter is vendored with SHA-256, license and provenance. `test_otsl.py` covers ordinary cells, blanks, rowspan, colspan and combined spans. |
| Q5, renderer | `test_renderer.py` renders twelve public HTML syntax fixtures and checks dimensions, spans, cell order and text. Further tests cover CJK glyphs and blank-span corrections. |
| Q6, independent methods | `test_pipeline.py` audits imports and verifies that three methods receive the original shared B0 input. |
| Q7, leakage control | Method interfaces exclude reference HTML and metrics. Synthetic fixtures determine behavior before code publication. The protocol acknowledges previously seen historical results. Actual scoring runs after generation. |
| Q8, one model | A sequential A/B/C mock test asserts one model load. The post-gate startup receipt records actual model load count and CUDA memory. Actual GPU evidence remains pending until that receipt exists. |
| Q9, publication | `scan_publication.py` checks the source allowlist, staged content, credential patterns, large files and complete Git history. `test_secret_filter.py` includes a deleted sensitive-history fixture. The upload receipt records exact committed paths. |
| Q10, completion gate | `test_evaluation_gate.py` covers running processes, reused PID, unknown GPU, missing outputs, incomplete reports and stability. `gate_cpu_test.json` records a real blocked check while B2/B3 run. Post-gate completion is proved by a separate receipt. |
| Q11, Agent-independent execution | `detach` creates a new POSIX session with redirected input/output, a supervisor PID receipt and an exclusive worker lock. Actual PID/start-time checks are required after launch. |
| Q12, restart recovery | Atomic create-if-absent success records and independent failure attempts make recovery idempotent. `resume_evaluation.py` reconstructs progress and skips compatible successful records. Historical recovery preserves original files in an isolated copy. `test_historical_recovery.py` and `test_official_scorer.py` cover partial writes and failed-row-only retries. |

The local final completion report includes actual process and progress evidence. CPU/mock checks establish correctness of those paths. Live GPU operation is claimed only after a validated completion gate and real sample outputs.
