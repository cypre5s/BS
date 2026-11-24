# SAMR smoke and development usage

SAMR does not train or modify model weights. Install the source and CPU checks with
`python -m pip install -e '.[dev]'`. Use the existing compatible Xiaomi inference
environment for GPU commands (the audited environment uses Transformers 5.15.1 and
Torch 2.13.0 with BF16/SDPA). Model and dataset locations are supplied by the user;
none are bundled or downloaded by these commands.

CPU and static checks:

```bash
python -m pytest tests/samr -q
python -m pytest -q
python -m ruff check src scripts tests
python -m mypy src
python scripts/audit_samr_processor.py --model /path/to/xiaomi-snapshot --output experiments/samr/processor_audit.json
```

The processor audit invokes the real native processor on CPU and loads zero models.
The unit/runtime fixtures are synthetic tests, not reference labels for real images.

Create the local benchmark exclusion receipt using the existing historical crop
provenance. This is an identity check, not formal evaluation or a routing input:

```bash
python scripts/samr_build_exclusion.py --baseline-pairs /path/to/B0/table_pairs.jsonl --benchmark /path/to/omnidocbench-release --output experiments/samr/exclusion.json
```

Prepare a small development manifest as described in
[the profile schema](../reports/samr/development_profile_schema.md). The source must be
independent of OmniDocBench, with 1–32 images. For real smoke choose 3–5 images whose
native resolutions remain distinct at 1.0, 1.5 and 1.75. Then run:

```bash
python scripts/run_samr_profile.py --model /path/to/xiaomi-snapshot --manifest /path/to/development.json --exclude-manifest experiments/samr/exclusion.json --config configs/samr/fusion_global_anchor.yaml --output experiments/samr/real_smoke --real-smoke
```

This runs nine scale calls for three images, one extra native B0 parity call and a
short cooperative deadline probe. The probe's partial output is discarded and is not
one of the nine normal scale outputs. A single model serves all calls. The runner
refuses to start if another GPU compute process is present. It requires a new output
directory, performs before/after file-hash checks and records setup time separately.
Inspect `summary.json`, `runtime_receipt.json`, `native_B0_parity.json`,
`deadline_probe.json`, `profile.jsonl` and `audit/*.json` locally. They remain ignored
by Git. A hard wall-clock deadline is not guaranteed by native GPU decoding.

For ordinary development use the same command without `--real-smoke` and select a
configuration: `baseline.yaml`, one of the four single-scale files, the medium/high
pairs, `scale_all.yaml`/`scale_ladder.yaml`, `fusion_global_anchor.yaml`,
`routing_low_medium_high.yaml` or `samr_full.yaml`. The latter is adaptive; it does not
automatically execute the whole ladder. Optional 2.0 profiling requires an explicit
local YAML configuration using the same pinned identities and sufficient budgets.

After inference, optional development-only scoring runs in the existing pinned official
CPU environment with evaluator dependencies available:

```bash
python scripts/score_samr_profile.py --evaluator /path/to/pinned-official-evaluator --profile /path/to/profile.jsonl --references /path/to/development_references.jsonl --output /path/to/new-scoring-directory
python scripts/score_samr_profile.py --evaluator /path/to/pinned-official-evaluator --output /path/to/new-synthetic-check --self-test
```

The scorer never loads a model and never feeds labels into A3. The synthetic self-test
checks official metric wiring; its score is not a real-image accuracy result. SAMR has
not run a formal full 665-table evaluation in this release. The historical A/B/C
experiment is separate and remains intact.
