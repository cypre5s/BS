# MESA v2 validation

This release validates crop/mapping infrastructure and development inference. It does not run the formal 665-table evaluation, train a model or claim improved TEDS.

## CPU and static checks

Full suite: **375 passed, 2 skipped**. MESA cases within that suite: **113 passed**. The two skips require a local evaluation-machine configuration and the pinned official scoring container; neither is a MESA test. The first isolated-worktree regression run exposed missing legacy renderer fonts. Linking the existing ignored font resources resolved those environment failures without modifying old tests or algorithms. Ruff and Mypy (13 MESA/script source files) pass. All seven YAML configurations are loaded by the test suite. Final CPU evidence is `experiments/mesa_v2/final_full_v3.xml`; final independent geometry evidence is `synthetic_diagnostics_final.json` in that directory.

Coverage includes eight vertical table families, nine rule/noise/shadow cases, 36 crop/scale/padding coordinate cases, 24 OTSL structures, right-hand partial columns, all three segmentation modes, seam-crossing spans, low-quality fallback, cache recovery and five-channel mock execution. Synthetic images are unit inputs only and are not saved as training data.

Independent geometric diagnostics checked 3,168 cell mappings, all correct, across 36 fixtures. Outward snapping reduced 144 known cut boundaries to zero without losing the target area. These measurements apply to the synthetic fixtures only.

## Evidence for Q1–Q12

| Question | Implementation and evidence |
|---|---|
| Q1 Different vertical regions? | `views.py:_vertical`; eight-family diversity/coverage test and per-image data-region IoU diagnostics. |
| Q2 Header context? | Original pixel `CropPiece` header bands; multi-level fixture retains both leading rows. Semantic header extent can be uncertain; fallback preserves the leading image band and is LOW quality. |
| Q3 Fewer cell cuts? | `crop_mapping.snap_box`; independent known-geometry boundary count 144→0, outward containment verified. Real improvement is not inferred from this. |
| Q4 Borderless behavior? | Detector marks unreliable geometry, proportional segments remain available, `map_view` rejects LOW-quality structural modifications. |
| Q5 Coordinates? | Piecewise forward/inverse transforms; 36 cases test four corners, interior, padding and resized/native coordinates at 1e-9 tolerance. Seam ambiguity is rejected. |
| Q6 Partial OTSL mapping? | Explicit original row/column coordinates and complete-span checks; 24 cases plus nonzero right-column offsets. Uncertain mappings are UNRESOLVED. |
| Q7 Spans? | Official parser, global-boundary span checks and local header/data seam span veto. |
| Q8 Original Xiaomi compatibility? | Existing BF16 runtime, unchanged native prompt/processor/generation path; five-channel mocks and real smoke receipts. |
| Q9 Weight identity? | All six pinned model/processor/tokenizer files are hashed before/after the real run. |
| Q10 CPU/Mock? | 375 full-suite passes, including 113 MESA cases, and static/config checks above. |
| Q11 Real GPU? | Passed: three images, 43 views, 44 native calls including one independent B0 replay; see measured evidence below. |
| Q12 Leakage? | Explicit requested-path staging, source-only size/binary/secret checks and full Git-history scan; artifacts remain ignored. See release report. |

## Diagnostic definitions

Grid Detection Coverage is the fraction of development images with a credible image-only rule grid. Grid Snap Rate counts actually expanded horizontal/tile boxes over proposed horizontal/tile boxes; vertical grid-selected row segments are reported through their mapping quality instead. Crop Mapping Success/UNRESOLVED rates use completed local views, with every failure retained in the denominator. Vertical View Diversity is one minus mean pairwise IoU of data boxes, excluding the repeated header context. Duplicate View Rate checks identical piece footprints plus prepared resolution within a channel. Crop Coverage and Overlap Coverage are source-image area covered at least once/twice by each channel's proposed views. Planned coverage is distinguished from successful-inference coverage when views are skipped.

Canonical Grid Mapping Accuracy is measured on the independently known synthetic fixtures. It is **unavailable on unlabelled real development images** and saved as null with an explicit reason. Mapping availability and coordinate consistency do not establish semantic correctness. None of these diagnostics replaces formal TEDS.

## Real GPU measured results

Final run: `experiments/mesa_v2/real_gpu_v3`, using three frozen PubTabNet validation images on the local RTX 4070. Selection used image size and image-only rule evidence, with no labels, predictions or scores used for selection. The existing formal-test exclusion check passed. The earlier `real_gpu_v1` development run exposed the double-rule issue; `real_gpu_v2` was interrupted at the user's pause request. Both are retained. The final run used a fresh directory, all final source hashes, the same three-image manifest and unchanged budgets.

**Passed:** 43/43 views completed across all five channels for each image, with no failed/skipped calls. All 43 predictions pass official OTSL/HTML parsing and canonical construction. One independent native B0 replay also matches exactly, giving 44 physical model calls. The original BF16 model is loaded once and stays frozen. All six pinned model/configuration/processor/tokenizer files have identical before/after hashes. Peak PyTorch allocated GPU memory was 1.692 GiB; the sum of the 43 recorded inference latencies was 315.99 seconds, excluding setup and the independent B0 replay.

| Development image | Views | Reliable rule grid / matches anchor | Expanded H/tile crops | Local mapping | UNRESOLVED | Vertical views / diversity |
|---|---:|---|---:|---:|---:|---|
| PMC2374446_005_00.png | 15 | No / No | 0/8 | 0/11 (0%) | 11/11 (100%) | 3 / 93.51% |
| PMC2570547_001_00.png | 15 | Yes / Yes | 3/8 | 6/11 (54.55%) | 5/11 (45.45%) | 3 / 85.66% |
| PMC3267163_005_00.png | 13 | Yes / No | 2/8 | 0/9 (0%) | 9/9 (100%) | 1 / not applicable |

Overall Grid Detection Coverage is **2/3 (66.67%)**; actual expanded-crop rate is **5/24 (20.83%)**. Local Crop Mapping Success is **6/31 (19.35%)**, and UNRESOLVED is **25/31 (80.65%)**. Duplicate View Rate is zero for every image/channel. Vertical diversity is undefined for the last image because its five visible physical row bands produce only one vertical region; the implementation does not force three redundant views.

| Image | Crop coverage H / V / tiles | Overlap coverage H / V / tiles |
|---|---|---|
| PMC2374446_005_00.png | 100% / 100% / 100% | 10.14% / 26.64% / 22.89% |
| PMC2570547_001_00.png | 100% / 100% / 100% | 17.21% / 39.59% / 26.71% |
| PMC3267163_005_00.png | 100% / 100% / 100% | 10.29% / 0% / 43.50% |

No planned views were skipped, so proposed-view coverage equals completed-view coverage in this final run. Original image pixels, rather than synthetic padding or resized canvas area, define these percentages.

On the second image, clustering the double header rule yields the image-derived 13-by-3 grid matching the anchor. All three header-preserved vertical views, one horizontal view and two tiles have spatially resolved mappings. Five of these six local helpers are subsequently vetoed because overlapping OCR text/topology disagrees; the remaining helper provides no supported blank-span repair. The first image lacks a trustworthy rule grid. On the third, five visible physical row bands do not uniquely match the anchor's 12 logical rows, so the mapper does not invent missing separators.

**All three fused outputs equal their original B0 outputs.** Spatial mapping availability improved for this particular ruled development image, but no recognition-quality gain is established. Real Canonical Grid Mapping Accuracy remains null because this run has no independent cell-coordinate labels. The independent synthetic result is 3,168/3,168 correct coordinates. No formal TEDS, model training, weight change or 665-table evaluation was performed.

Local evidence hashes (the files themselves are excluded from Git):

- Development manifest SHA-256: `70104c3de2e125605df621a0a6ffdf8dcfbbba573e298ed075d3fe5fba516971`.
- Frozen exclusion manifest SHA-256: `9939cc1c9da35f278355d1d9cad9f345092a4c2af8075d75c3ed6d16a03527e8`.
- Original model weight SHA-256, before and after: `d4551e9f6127ac9e35cff7b4f7a72a90410e03e6c8676a36f0e2b7eaeb5d5348`.
- Final summary SHA-256: `1177076fa8c75d1424afc7f255f1763650fdbdb9648d8e56e8f0e39b473411ed`.
- Runtime receipt SHA-256: `3227bcb186e5d479c8852e71ab983c47bb23e4933427f6497a60adab5b5d5351`.
