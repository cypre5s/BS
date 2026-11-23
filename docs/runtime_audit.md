# Xiaomi runtime and image audit

The adapter uses the exact B0 model snapshot identified by revision `bbd8cf8d21f256b3644b44cb860f3a7cecb93c30`. The BF16 weight SHA-256 is `d4551e9f6127ac9e35cff7b4f7a72a90410e03e6c8676a36f0e2b7eaeb5d5348`. Configuration, tokenizer, template and processor files have separate fixed hashes in `xiaomi_runtime.adapter.MODEL_FILES_SHA256`. Loading checks every file. The runtime keeps one frozen model instance and exposes `load_count`, CUDA memory measurements and an after-run hash check.

The CPU audit loaded the native processor and processed seven synthetic images. It initialized zero models and no CUDA context. The tested input sizes were 256×256, 777×399, 2048×512, 80×800, 512×2048, 321×123 and 65×96. Every native grid and image-placeholder count matched the audited sizing helper. A numbered-patch image also verified the native 2×2 grouping order across all 256 patch positions. All six pinned snapshot hashes matched before and after the audit. Local detailed evidence is generated under `reports/architecture/processor_cpu_audit.json` and excluded from source publication.

The installed Transformers 5.15.1 files audited were `models/qwen2_vl/image_processing_qwen2_vl.py`, `models/qwen3_vl/processing_qwen3_vl.py`, `models/qwen3_5/modeling_qwen3_5.py` and `vision_utils.py`.

| Stage | Function | Observed behavior |
| --- | --- | --- |
| Dynamic sizing | `smart_resize`, `Qwen2VLImageProcessor.resize` | Rounds dimensions to multiples of patch size × merge size and preserves aspect ratio within pixel limits. |
| Patch serialization | `Qwen2VLImageProcessor.patchify` | Groups each 2×2 patch neighborhood before flattening. A static image repeats its temporal patch internally. |
| Image metadata | `Qwen2VLImageProcessor._preprocess` | Emits `image_grid_thw = [1, H/16, W/16]` for each static image. |
| Placeholder expansion | `Qwen3VLProcessor.replace_image_token` | Inserts `prod(image_grid_thw) / 4` visual placeholders. |
| Vision position encoding | `Qwen3_5VisionModel.forward` | Computes interpolation indices and vision positions through native vision utilities, then applies learned positions and rotary embeddings. |
| Model insertion | `Qwen3_5Model.get_image_features`, `forward` | Splits merger output by native grid counts and inserts embeddings into the matching image placeholders. |
| Language positions | `compute_3d_position_ids`, `get_rope_index` | Uses native grid metadata and multimodal token types for M-RoPE. |

The pinned processor uses 16-pixel patches, merge size 2 and temporal patch size 2. Its minimum pixel budget is 65,536 and maximum is 16,777,216. These limits give at most 16,384 merged visual tokens. The global view sends the original image directly through the same native chat template and processor as B0. Extra views resize the PIL image and then pass through that same processor. The adapter verifies grid divisibility and equality between expected visual tokens and image placeholders.

Method A requests 1.5× and 2× linear scales relative to the native processed image. Their additional-view budgets are 2,048 and 4,096 visual tokens. A view is skipped when its legal size duplicates the global or another requested view. Already large global inputs retain the original processing path. Selection accepts an enhanced result only when two distinct enhanced resolutions produce identical official HTML and both stop at EOS with a valid OTSL table. Otherwise it retains the global result. Selection consumes images and predictions only.

The prompt is `Parse the table in the image into OTSL.`. The adapter uses greedy decoding, repetition penalty 1.0, 16,384 maximum generated tokens, native EOS and padding IDs, BF16 autocast, SDPA and the original tokenizer cleanup settings. Calls record the table ID, method, view, input and processed dimensions, grid metadata, token counts, timing and hashes.

The permission callback runs before any model load or inference. CPU tests exercise a denied gate and prove that it leaves `model` unset and `load_count` at zero. Real GPU parity and memory results are recorded only after the orchestration gate allows inference. Mock tests establish interface and selection behavior. They provide no real-model accuracy claim.

`xiaomi_runtime.otsl` delegates to the unchanged Xiaomi deployment converter. Its source hash is verified by a test. Normal cells, blank cells, horizontal and vertical spans, combined spans and inline formatting have explicit expected HTML fixtures. The CLI skill converter was considered, but its HTML escaping and continuation implementation differ from the B0 deployment path. The selected upstream converter preserves the original experiment's serializer.

[Pinned Xiaomi converter](https://github.com/xiaomi-research/Xiaomi-OCR-0/blob/4abc240285b69a2965cf64762101ddb4ab35f32d/postprocess/otsl.py)
