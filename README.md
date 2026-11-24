# Xiaomi-OCR-0 Training-Free Inference

This project provides training-free table inference with one frozen original Xiaomi-OCR-0 BF16 model. The existing A/B/C methods and their historical 665-table experiment remain available. SAMR adds a separate structure-aware multi-resolution upgrade to Method A; SAMR has not undergone a formal full OmniDocBench evaluation.

SAMR combines A1 full-image scale profiling, A2 global-anchored canonical-grid fusion, and A3 image-complexity resource routing. It does not train, modify weights, consume ground-truth routing features or select longer outputs. The default adaptive path runs global only for LOW complexity, global + 1.5 for MEDIUM, and global + 1.5 + 1.75 for HIGH. Uncertain evidence retains B0. The current repair policy is limited to image-supported splits of blank merged cells.

See [SAMR smoke and development commands](docs/smoke_test.md), [architecture](reports/samr/architecture_report.md), [Q1–Q16 evidence](reports/samr/test_report.md), and [budget semantics](reports/samr/complexity_routing_report.md). Visual-token budgets are enforced before generation; latency uses cooperative decoding stops and can overshoot during a nonpreemptible GPU operation. Smoke success is not evidence of improved accuracy.

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

The historical scientific protocol is described in `docs/evaluation_protocol.md`. `docs/code_audit.md` maps its Q1–Q12 to executable evidence. Historical A/B/C evaluations and the new SAMR smoke validation are separate records; SAMR accuracy gains remain unestablished.
