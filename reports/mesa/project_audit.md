# MESA project audit

The implementation starts from BS commit `bb596b4f7ebcfe24e3646eeb098f76df9dd053e1`.
The original `xiaomi_runtime`, `multiview`, `reacquisition`, `render_verify`, `samr` and `evaluation` implementations remain intact. Existing tests are retained. MESA resides in independent source, configurations, tests, scripts, documentation and reports.

The existing runtime already supplies lazy single-model loading, frozen BF16 parameters, the pinned prompt, native processor/token-grid checks, bounded visual tokens and cooperative decode deadlines. MESA reuses these interfaces without editing token tensors or `image_grid_thw`.

The existing strict SAMR canonical parser and image-verified blank-span repair are reused as conservative primitives. MESA adds channel/view provenance, pixel-to-lattice mapping and separate text diagnostics. It does not depend on the paused A2++ draft. The A2++ snapshot and working files remain local and are excluded from this release.

Before work, byte hashes of tracked code, paused A2++ files and historical SAMR/A/B/C experiment artifacts were saved locally in `experiments/mesa/preservation_before.json`. Publication changes are limited to MESA and its README/allowlist integration. Dataset, predictions, image artifacts, machine paths and weight files are excluded from commits.

The official OTSL implementation is the existing pinned [Xiaomi converter](https://github.com/xiaomi-research/Xiaomi-OCR-0/blob/4abc240285b69a2965cf64762101ddb4ab35f32d/postprocess/otsl.py). Its local vendor SHA-256 remains `db46f5a05d9742a7bcfd3931428bdd9e8de845892b66fa4cf09bd21f83cf2b14`.
