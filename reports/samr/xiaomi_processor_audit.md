# Xiaomi processor audit

This audit inspected the installed implementation and exercised the actual processor.
The [Xiaomi repository](https://github.com/xiaomi-research/Xiaomi-OCR-0) and
[technical report](https://arxiv.org/abs/2609.36136) provide project context; they are
not substitutes for the installed source and measured tensor metadata below.

The installed versions are Transformers 5.15.1 and Torch 2.13.0. The loaded classes are
`Qwen3VLProcessor` and `Qwen2VLImageProcessor`. The original model revision is
`bbd8cf8d21f256b3644b44cb860f3a7cecb93c30`. The six immutable file identities are defined
in `xiaomi_runtime.adapter.MODEL_FILES_SHA256` and checked before loading and after smoke.

| Audit item | Observed implementation and SAMR contract |
| --- | --- |
| Image processor | Official `AutoProcessor.from_pretrained`, local pinned snapshot; no replacement processor |
| Dynamic resolution | Installed `smart_resize` and `Qwen2VLImageProcessor.resize`; existing helper predicts sizes for admission only |
| Resize | Native bicubic resize; factor is patch size × spatial merge size = 32 |
| Crop | `_preprocess` performs resizing, normalization and patchification; no crop in this static-image path. SAMR preserves the full image |
| Patch size | 16 pixels; temporal patch size 2 for the internally repeated static frame |
| Maximum pixels | 16,777,216, an area limit rather than a fixed maximum side length |
| Minimum pixels | 65,536 target area; native upward rounding can produce an area slightly larger than this minimum |
| Grid metadata | Full native batch features, including `mm_token_type_ids`, enter the model unchanged |
| `image_grid_thw` | Static image `[1, processed_height / 16, processed_width / 16]`; spatial axes must be divisible by merge size 2 |
| Visual token count | Actual image-placeholder count must equal `prod(image_grid_thw) / 4`; maximum 16,384 per native call |
| Token order / positions | Native patchifier groups 2×2 patches, then native merger/placeholder insertion and `get_rope_index` / `compute_3d_position_ids` supply positions; SAMR never concatenates or reorders tensors |
| Generation | Original greedy decoding, repetition penalty 1.0, maximum 16,384 new tokens, cache enabled, native EOS/PAD, BF16, SDPA; decode skips special tokens and disables whitespace cleanup |

The processor accepts aspect ratios up to 200; its installed check rejects values above
200. Legal processed widths/heights are multiples of 32 within the native area rules.
SAMR does not claim every arbitrary requested pixel size is a native processed size.

Four sizes are distinguished: original pixels; requested multiplier; actual prepared PIL
dimensions; actual native processed dimensions. The global 1.0 image object is unchanged.
Additional images use one isotropic factor and nearest-integer pixel dimensions (at most
0.5 pixel rounding per axis), with an isotropic allocation cap for very large inputs.
The official processor then performs its own grid rounding. Exact aspect-ratio equality
after native grid rounding is neither achievable nor claimed. There is no additional
anisotropic stretch or crop in SAMR.

Measured example for an original 777×399 image:

| Requested factor | Prepared pixels | Native processed pixels | Native grid | Actual visual tokens |
| --- | --- | --- | --- | ---: |
| 1.0 | 777×399 | 768×384 | `[1,24,48]` | 288 |
| 1.25 | 971×499 | 960×512 | `[1,32,60]` | 480 |
| 1.5 | 1166×598 | 1152×608 | `[1,38,72]` | 684 |
| 1.75 | 1360×698 | 1344×704 | `[1,44,84]` | 924 |
| 2.0 | 1554×798 | 1568×800 | `[1,50,98]` | 1225 |

`scripts/audit_samr_processor.py` passed 30 actual CPU processor cases (six original
sizes × five factors). Every token/grid count matched, non-image prompt token IDs were
identical across scales, and multimodal position metadata was present. A numbered
4×4 patch fixture verified native grouping order
`0,1,4,5,2,3,6,7,8,9,12,13,10,11,14,15`. This test invokes the native patchifier on a
synthetic image; it does not supply constructed vision tensors to a model. No weights
were loaded and no CUDA context was initialized by this audit.

| Installed source | SHA-256 |
| --- | --- |
| `image_processing_qwen2_vl.py` | `4e1da45f9e7e157ca08aa88243fcc1495ea6a333e40e0c5035ef17ef2747a97d` |
| `processing_qwen3_vl.py` | `8ce525e56e0ddade5a665a70f0727b5576f6f51610d068ddfaf4cc60333415f0` |
| `modeling_qwen3_5.py` | `67cf849081143a998f0e189c551e4b5f532365a055805570747385e400372abb` |

Local evidence: `experiments/samr/processor_audit.json`. Real inference and native B0
parity evidence are summarized in `test_report.md`. Raw receipts are not published.
The video-processor import emits two upstream documentation warnings about frame
arguments; the static-image audit and inference complete successfully.
