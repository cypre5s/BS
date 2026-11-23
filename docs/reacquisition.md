# Visual Evidence Reacquisition

Method B takes the shared runtime and the original B0 prediction. Deterministic rules nominate rows containing merged headers, many columns, long content, consecutive merges, or ragged raw OTSL rows. Image rules also nominate rows close to the table image boundary. A nomination expresses a reason to inspect a region.

The image mapper detects long horizontal and vertical strokes on the CPU. It accepts a grid only when the observed boundaries uniquely match every row and column in the official parsed structure. Boundary positions come from the image and can have unequal spacing. Tables with insufficient rules, ambiguous counts, low contrast, or an unsupported polarity retain B0.

Each crop stores its original table coordinates, requested box, clipped integer box, padding, actual processor output dimensions, and separate horizontal and vertical resize ratios. Forward and inverse affine transforms preserve those coordinates. Region selection expands over merged cells and removes overlapping candidates. Crops covering the entire original table are skipped.

The same Xiaomi runtime reads each local crop with the original table prompt. A local prediction must end at EOS and match the mapped row count, column count, cell origins, and spans. It must retain at least two distinct nonempty cell anchors and 75% of nonempty anchors. At most 25% of mapped cells may change. Numeric changes and changes between blank and nonblank cells are held for additional evidence. These conservative defaults were chosen from the mapping design and synthetic tests before new benchmark results are scored.

An accepted edit replaces only the original content interval of each explicitly mapped global cell. The intervals use Xiaomi's official token expression. The implementation preserves all original span and row tokens and all content outside the accepted region. The candidate is parsed again and its complete structural signature must match the original. Unsupported or inconsistent evidence returns the original B0 OTSL.

The resulting record includes local calls, crop transforms, row and column mappings, accepted cell edits, and rejection reasons. These artifacts are private runtime output. The source tests use generated shapes and invented text strings solely as unit-test fixtures.

`test_crop_mapping.py` covers six distinct crop geometries, clipped padding, processor dimension rounding, coordinate round trips, and overlap. `test_reacquisition.py` covers uneven image grids, all candidate categories, mapped edits, unchanged outside cells, ambiguous geometry, incomplete generation, mismatched shape, missing anchors, numeric refusal, and merged cells crossing a crop boundary.

Method B imports `xiaomi_runtime` and its own package. It shares the pinned official OTSL converter with all methods. It has no dependency on method A or C.
