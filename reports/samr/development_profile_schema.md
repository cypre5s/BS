# Development profile schema and isolation

`scripts/run_samr_profile.py` accepts one explicit development manifest and a pinned
665-table exclusion manifest. It refuses more than 32 images and an existing output
directory. This entry point does not launch the historical benchmark orchestrator.

Example local manifest (the user supplies real files and their hashes):

```json
{
  "split": "development",
  "source": "Independent public development dataset and split",
  "images": [
    {"table_id": "dev:example-1", "image": "images/example-1.png", "sha256": "REPLACE_WITH_SHA256"}
  ]
}
```

Only those keys are accepted. Relative image paths resolve against the manifest.
GT rows, columns, boxes, spans or HTML are not accepted in this input schema. Images
are checked for file and decoded-pixel equality with excluded pages and exact table
crops, including renamed/re-encoded copies. The exclusion payload is pinned by SHA-256
`53f6baba71c05f07999df1d51e370f4f3bf613f5f42d073d8febf7cf56d513ef`.
This protects exact sample identity; it does not establish absence of all near-duplicate
or transformed images in arbitrarily supplied datasets. Dataset provenance remains
explicit in the local manifest.

`profile.jsonl` has one row per executed scale:

| Field | Meaning |
| --- | --- |
| `table_id`, `scale`, `requested_scale` | Development identity and multiplier relative to original pixels |
| `actual_resolution` | Actual prepared PIL width/height |
| `processed_resolution`, `image_grid_thw` | Recorded native processor dimensions and grid |
| `visual_token_count` | Actual native placeholder count, not merely a scale estimate |
| `latency` | Native inference seconds, excluding shared model loading |
| `teds`, `teds_s`, `em`, `cer` | Initially `null`; optional separate offline scoring only |
| `parse_status` | OTSL validity, HTML status and canonical resolution status |
| `runtime_contract_valid` | Native size/grid/token/prompt/model/serializer checks passed |
| `prediction_html` | Local prediction needed for later offline scoring; never published |
| `eos`, `truncated`, `error` | Completion status; partial output cannot silently pass smoke |
| `evaluation_split`, `scored_offline` | Must be development; initially false |

Per-sample audit JSON includes every raw OTSL, canonical grid, route, fusion decision
and final OTSL/HTML. Skipped/failed calls and their reasons appear in routing events;
no fabricated prediction row is emitted for a call that returned no output. Every
receipt is immutable and excluded from source publication.

Optional reference JSONL has only development `table_id` and `html` entries, with IDs
matching the profile exactly. Run `scripts/score_samr_profile.py` afterwards in the
pinned official evaluator's CPU environment. It verifies evaluator commit
`7ddb8df411ad9bd2fff139acafe0ed16aa1694fe`, uses its unchanged normalization and
`call_TEDS`, and writes a new `profile_scored.jsonl` plus scoring diagnostics. The
inference engine does not import the scoring module or receive references.

TEDS/TEDS-S use the pinned official definitions. EM is the historical strict equality
of the two complete officially normalized HTML strings, not cell-level equality or
unprocessed OTSL equality. CER uses the historical cell-text extraction and edit
distance; per-sample rows include character edits/counts so aggregate CER is a ratio
of sums. A zero reference-character count gives `null`, not a fabricated zero error.
No development accuracy scores were used to select this release's thresholds.
