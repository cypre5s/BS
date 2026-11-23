# Architecture

`xiaomi_runtime` owns the original BF16 model, processor, tokenizer, generation settings and official OTSL serializer. Loading is lazy and requires a permission callback from the completion gate. The same runtime instance is injected into all methods. The CPU test environment can import the interface without loading Torch or opening CUDA.

`multiview`, `reacquisition` and `render_verify` implement independent methods through `run(table_id, image, baseline)`. Methods receive original image pixels and cached B0 output. Ground-truth HTML and official metric records enter the separate scoring subprocess after generation is complete. The methods return the chosen prediction, their additional model calls, a decision reason and diagnostics.

`evaluation.omnidocbench.Cohort` verifies the pinned dataset and evaluator, checks B0 provenance, and exposes only the image crop and original prediction to methods. Images are checked against their published hashes when read. The global crop uses the exact clamped polygon bounding box from the existing experiment.

`orchestration` snapshots the live process state, checks completion, holds an exclusive GPU lease and maintains detached execution. Successful sample records use atomic create-if-absent writes. Failure attempts are separate immutable records. A restart validates the experiment identity and skips successful records.

The frozen experiment identity includes the published source commit, method configuration hashes, renderer font hashes and inference-package versions. A single post-gate native global call must reproduce the cached B0 prediction, grid and provenance before enhanced inference proceeds. This compatibility audit has its own receipt and is excluded from formal method-call averages.

If the original evaluation ends with incomplete artifacts, recovery first acquires its original run lock and rechecks process and GPU state. It creates an isolated recovery copy with separate inodes, preserves all verified successful predictions and score lines, and invokes the unchanged original code for missing predictions and failed scoring rows. The original experiment directory remains unchanged. A completed recovery receipt directs the gate to the validated copy.

The supervisor remains alive independently of its launching shell through a new POSIX session and redirected standard streams. Its own PID, boot ID, start ticks and command line are recorded. Machine restart ends processes, and the documented resume command reconstructs progress from disk.

One table is processed through A, B and C in order before the next table. Each method uses its original B0 input. Model calls and elapsed time are recorded separately from shared baseline cost. Scoring runs after all selected methods have generated the identical cohort.
