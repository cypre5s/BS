# MESA v2 reproduction

Run from the v2 checkout using the existing CPU development Python (with NumPy, Pillow, SciPy, pytest, Ruff and Mypy):

```bash
python -m pytest tests/mesa -q
python -m ruff check src/mesa scripts/mesa tests/mesa
python -m mypy src/mesa scripts/mesa
python scripts/mesa/audit_geometry.py --output experiments/mesa_v2/synthetic_diagnostics.json
```

The unchanged native Xiaomi GPU environment is required for the real smoke:

```bash
python -u scripts/mesa/run_smoke.py \
  --model /path/to/original/xiaomi-ocr-0 \
  --manifest /path/to/development_manifest.json \
  --exclude-manifest /path/to/frozen_exclusion.json \
  --config configs/mesa/mesa_fusion.yaml \
  --output experiments/mesa_v2/real_gpu_v3 \
  --gpu-lock /path/to/shared/.run_all.lock
```

Use a new output directory for v2; its configuration and source hashes intentionally differ from v1. Three to five independent development images must pass the existing formal-test exclusion check. Another GPU process prevents startup. All observations share one original frozen BF16 model. The seven YAML definitions retain all five channels and expose the new detector, snapping and vertical segmentation parameters. The view cap is 18 to accommodate one extra vertical segment; pixel/token and latency caps remain bounded.

The recorded final run uses `real_gpu_v3`; use another unused output path when reproducing it. Earlier development and interrupted runs are retained independently. Completed view caches are immutable and identity-bound. Resuming an interrupted run preserves its spent budgets, so a new complete validation run must use a new directory instead of resetting an old budget or overwriting its evidence.

The runner records per-table image-only grid diagnostics, snap and mapping availability, crop/overlap coverage and vertical diversity. Canonical mapping accuracy on unlabelled real images is null rather than inferred from internal consistency. No formal 665-table run or TEDS calculation is invoked.

Publication uses explicit allowed MESA paths and force-adds only the six Markdown files in `reports/mesa_v2/`, which the historical root ignore policy excludes. `scripts/mesa/scan_release.py` scans staged blobs and all reachable history for secrets, binary/large files and paths outside the requested v2 scope. No root publication rules or old algorithms need to be changed.
