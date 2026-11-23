# CPU and mock validation

Validation timestamp: 2026-11-23T12:48:42+00:00.

Overall status: PASS.

191 passed, 1 warning in 20.86s

| Check | Result |
| --- | --- |
| pytest -q | PASS |
| ruff check src scripts tests | PASS |
| mypy src | PASS |
| Publication allowlist and secret/history scan | PASS |

Tests cover native processor sizing, official OTSL serialization, independent methods,
crop coordinates, real HTML rendering, failed historical recovery, process identity,
completion gating, source publication, immutable records and resumed scoring.

CUDA_VISIBLE_DEVICES was empty for all checks. Mock output is synthetic and no real model
inference was started by this suite. The detailed machine receipt is excluded from Git.

The separate processor audit verified six snapshot hashes, seven image shapes and 256
native patch positions on CPU. Live GPU memory, model load count and A/B/C startup
require the post-gate execution receipts described in docs/code_audit.md.
