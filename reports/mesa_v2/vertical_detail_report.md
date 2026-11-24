# Header-preserved vertical segmentation

`VisualViewGenerator._vertical` replaces the original repeated top prefixes with data-row partitions. Defaults: `vertical_segmentation_mode: header_preserved`, `grid_aware_crops: true`, two overlap rows, header context enabled, at most three vertical views. Short tables yield fewer views. `fixed_overlap` and `grid_aligned` modes remain configurable.

Trusted horizontal boundaries select whole data rows. A leading image header candidate ends at the first strongly supported full-width divider within the configured header band, including preceding partial dividers in multi-level layouts. It is an image-derived context candidate, not a semantic header oracle. Complex headers without distinguishable visual separators remain uncertain.

The first view retains the original header and early data in one contiguous region. Later views copy the original header pixels and a different middle/late data region into explicitly separate canvas pieces. The seam is synthetic whitespace, not a fictitious stretch of original image. Each piece records its original and canvas box plus original row IDs. Padding is white canvas context rather than a sliver of an adjacent cell. Isotropic resizing occurs after assembly, followed by the unchanged native processor.

When the image cannot establish a credible rule grid/header candidate, ordinary overlapping proportional data segments are used with a copied leading-image context band. These views are LOW quality and cannot modify global structure. No semantic header correctness is claimed for that fallback band. All original image pixels are covered by the union of vertical pieces; copied headers overlap intentionally and do not count as independent votes.

Unit evidence: eight families cover ordinary long tables, multi-level headers, very few/many rows, irregular heights, borderless/partial rules and a title above the table. Tests assert different data regions, no duplicate whole-table/prefix pair when multiple views exist, header pixels present, and complete union coverage. A single whole-table observation for a very short table is permitted; three redundant views are never forced.
