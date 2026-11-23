# Reproduction

Use a clean clone of the repository and an existing pinned Xiaomi inference environment. The original experiment must expose the same B0–B3 receipts, full-stage summaries, selected adapter snapshots, dataset and official CPU runtime. CPU development can use a separate environment created from the development requirements.

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements-dev.txt
cp configs/local.example.yaml configs/local.yaml
```

Set `project`, `source`, `runtime` and `python` in `configs/local.yaml`. `python` is the interpreter inside the verified Xiaomi inference environment. The exact environment used for the original experiment was Python 3.12, Torch 2.13.0 with CUDA 13.0, Transformers 5.15.1, torchvision 0.28.0 and fla-core 0.6.0. WeasyPrint 70.0 and PyMuPDF are needed for C. Keep any existing evaluation environment active while preparing the independent CPU environment.

Download the open Noto CJK font into the excluded local resources directory. The source is the Noto CJK project under the SIL Open Font License. The default C configuration points at this local file.

```bash
mkdir -p experiments/resources
curl -fL -o experiments/resources/NotoSansCJKsc-Regular.otf \
  https://raw.githubusercontent.com/notofonts/noto-cjk/main/Sans/OTF/SimplifiedChinese/NotoSansCJKsc-Regular.otf
.venv/bin/python scripts/run_cpu_tests.py
python scripts/audit_environment.py --config configs/local.yaml
python scripts/audit_current_runs.py --config configs/local.yaml
```

The completion gate requires a successful local validation receipt and a clean code commit already published to `origin/main`. It verifies the remote commit before establishing the detached worker. The source scanner uses a publication allowlist, examines staged content and scans every reachable historical blob.

```bash
python scripts/scan_publication.py
python scripts/launch_after_b2_b3.py --config configs/local.yaml
```

`experiments/multiview/orchestrator_state.json` contains the live state, process identity, method and counts. The append-only supervisor log is `experiments/multiview/orchestrator.log`. Each method has a `run.log` and `progress.json`. A process ID is considered live only when boot ID, start ticks and command still match.

After an Agent exits, the detached worker continues. After a machine restart or recoverable failure, run the following command from the repository. It checks the previous process identity, preserves completed predictions, and retries missing or failed work under the frozen configuration.

```bash
python scripts/resume_evaluation.py --config configs/local.yaml
```

For a separately selected method, `scripts/run_method_a.py`, `scripts/run_method_b.py` and `scripts/run_method_c.py` accept the same `--config` argument and use the same completion gate. They are foreground commands and are intended for explicit recovery/debugging. The normal durable entry point is `launch_after_b2_b3.py`.

Failure attempts contain local trace information and are excluded from Git. Correct operational failures before resuming. Scientific method changes require a fresh experiment identity and output directory, so already produced comparisons stay attributable to their original code and configuration.
