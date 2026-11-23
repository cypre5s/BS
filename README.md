# Xiaomi-OCR-0 Training-Free Inference

This project evaluates three independent inference methods on the frozen 665-table OmniDocBench cohort used by an existing Xiaomi-OCR-0 experiment. One original BF16 model serves all additional inference calls.

| Method | Evidence used | Acceptance rule |
| --- | --- | --- |
| A, Adaptive Multi-Scale | Original, medium and high resolution images | Two distinct enhanced resolutions must agree after the official serializer. |
| B, Visual Evidence Reacquisition | Global structure and image-detected row and column boundaries | A local crop must map back to complete global cells with matching spans and sufficient unchanged anchors. |
| C, Render-Guided Verification | An HTML render and source-image line geometry | Only verified blank merges can be split, and the candidate must pass a second render verification. |

Every method can retain the original B0 prediction. Results and model-call metadata make this decision auditable. Method C ordinarily uses CPU rendering and geometry with the cached global prediction. Its real execution is demonstrated by persisted verification diagnostics.

The worker waits for B2 and B3 to finish. The gate checks Linux process identity, all 665 table IDs, prediction receipts, official scoring diagnostics, the final report, file stability and GPU availability. An exclusive lock prevents another project worker from entering the historical run while the new evaluation is active.

The worker visits table IDs in the frozen order. For each table it executes A, then B, then C sequentially. This schedule shares one resident model while giving every method the same original input and cached B0 prediction. Each method has its own immutable records, attempts, progress file and log.

Start with `docs/reproduction.md`. The inference environment and existing benchmark assets are supplied through an excluded machine configuration. The repository contains source, tests, configurations and documentation. Generated experiments and machine receipts are stored locally.

```bash
python scripts/run_cpu_tests.py
python scripts/audit_current_runs.py --config configs/local.yaml
python scripts/launch_after_b2_b3.py --config configs/local.yaml
python scripts/resume_evaluation.py --config configs/local.yaml
```

The scientific protocol is described in `docs/evaluation_protocol.md`. `docs/code_audit.md` maps Q1–Q12 to executable evidence. The methods are hypotheses pending paired evaluation.
