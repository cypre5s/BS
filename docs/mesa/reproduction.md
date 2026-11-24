# MESA development reproduction

For the current grid-aware implementation, see [MESA v2 reproduction](v2_reproduction.md) and [v2 validation](../../reports/mesa_v2/smoke_test_report.md). Earlier v1 experiment directories are historical evidence and must not be reused for v2 outputs.

Run from the BS repository using the existing CPU development environment:

```bash
.venv/bin/python -m pytest tests/mesa -q
.venv/bin/ruff check src/mesa tests/mesa scripts/mesa
.venv/bin/mypy src/mesa scripts/mesa
```

The GPU interpreter must contain the existing pinned Xiaomi runtime dependencies. Supply the local original model snapshot, development manifest, frozen test-set exclusion manifest and shared project GPU lock. The manifest must contain 3–5 `dev:` image IDs with image paths and SHA-256 values; annotations and labels are not accepted.

```bash
../.venv/bin/python -u scripts/mesa/run_smoke.py \
  --model ../models/xiaomi-ocr-0 \
  --manifest experiments/samr/development_manifest.json \
  --exclude-manifest experiments/samr/exclusion.json \
  --config configs/mesa/mesa_fusion.yaml \
  --output experiments/mesa_v2/development_run \
  --gpu-lock ../runs/xmod/.run_all.lock
```

The command refuses another GPU compute process and acquires the shared advisory lock. All five channels share one BF16 model. Runtime metadata and predictions remain under the ignored `experiments/` tree. It never invokes the official 665-table evaluator. Reusing an identical output directory reuses completed table/view caches; changed pixels, configuration or implementation require a new directory. In-flight unknown crash cost is conservatively charged against the remaining latency budget.

For CPU mock flows and isolated auxiliary channels, use `MESAInferenceEngine(runtime, config, output)` with a tested runtime implementing `infer`. Selecting one auxiliary channel implicitly retains the required global original view. Do not use formal benchmark reference answers to select channels or fusion decisions.
