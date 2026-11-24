# Existing project audit — SAMR

Audit date: 2026-11-24. Repository: <https://github.com/cypre5s/BS>.

The starting working tree was clean. `main` and `origin/main` both pointed to
`5d6eab155cf4c45622ab84c05c984d4ad6861773`; fetching the remote confirmed that history.
This is the existing repository, not a replacement or a new root history.

| Existing module | Reuse in SAMR | Preservation |
| --- | --- | --- |
| `xiaomi_runtime.adapter.XiaomiRuntime` | One lazy BF16 instance, permission callback, six snapshot hashes, native inference metadata | Optional token/deadline guards only; original callers retain their defaults |
| `xiaomi_runtime.processor` | Original prompt/template, `ProcessorLimits`, exact native sizing helper | Unchanged |
| `xiaomi_runtime.otsl` and vendored deployment converter | Official grammar, grid resolution and HTML conversion | Unchanged |
| `multiview.AdaptiveMultiScaleInference` | Reference interface and original Method A | Unchanged; SAMR has its own entry point |
| `reacquisition` | Existing Method B remains available | Unchanged |
| `render_verify.structural_checker` / `visual_checker` | HTML topology and image ruling-line/blank-region primitives | Unchanged; SAMR does not invoke Method C's rendering pipeline |
| `evaluation` | Pinned evaluator identity and existing metric definitions | Unchanged; development scoring runs separately |
| `orchestration.storage` | Atomic immutable receipts, file/object hashes | Unchanged |
| `orchestration.publication` | Secret, binary, size, path and history checks | Add only nested SAMR YAML and Markdown report allowances |

New code lives in `src/samr`: typed configurations, A1 scale plans, canonical grids,
A2 fusion, A3 visual complexity and resource routing, one engine, development input
isolation and offline scoring. New scripts expose processor auditing, development
profiling, exclusion-manifest creation and optional offline scoring. Twelve configurations
record model, processor, prompt and generation identities.

Potential conflicts were resolved explicitly. Old Method A's scale factors are relative
to its native input; SAMR factors are relative to original image pixels and are recorded
as such. The official converter repairs some malformed OTSL; SAMR retains its converter
but refuses ambiguous grids for fusion. OTSL has no semantic header-type token, so the
canonical representation records `fcel`/`ecel` rather than guessing `th` labels. Strict EM
retains the established normalized-HTML equality definition in offline scoring.

The local preservation snapshot covers 6,720 historical JSON, JSONL, CSV and Markdown
artifacts from B0–B3 and A/B/C. Its initial file SHA-256 is
`a1bd123914cd574ff1251ca0b177b317f3e4eea5ea2881272aef48219eeed943`.
The snapshot and predictions remain excluded from Git. Final preservation verification
is recorded in `test_report.md`; no historical run is used to tune SAMR thresholds.

The GPU was idle before development smoke inference. The new runner checks for other
compute processes before model loading and each call. It neither kills nor preempts
existing jobs. Only independent development images are passed to this runner.
