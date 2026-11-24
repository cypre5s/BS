# A2 canonical alignment and fusion policy

The unchanged Xiaomi deployment OTSL converter remains authoritative for parsing and
HTML. SAMR validates its output instead of introducing a new grammar. `fcel` and `ecel`
are origins; `lcel`, `ucel`, `xcel` continue left, up or both; `nl` terminates a row.
Each canonical cell stores row/column, row/column spans, original tag type, exact text
payload, source scale and a deterministic validation confidence. The confidence is
not a model probability. Model-token spans are `null`: character offsets are not
misrepresented as tokenizer offsets.

Normal cells, declared blanks, fully blank rows, multilevel header topology, rowspans,
colspans and combined spans are supported. OTSL carries no semantic header labels.
A bare empty row has no declared width, so it is unresolved rather than silently
deleted. Ragged rows, orphan continuations, intersecting ownership, inconsistent
rectangle continuations or HTML topology changes are unresolved. Official fallback
serialization remains available even when strict fusion validation rejects a grid.

Alignment compares occupied coordinates, not cell-list indices. Equal global lattice
dimensions allow bipartite overlap groups: same region, split, merge or unresolved
many-to-many. This is only a structural correspondence proposal. Different dimensions
have no mapping and cannot alter B0's global row/column count.

All six evidence components are recorded: OTSL validity, HTML validity, grid consistency,
span consistency, text agreement and exact cross-scale agreement. Their arithmetic
mean is a traceable validation score, not an acceptance ranking. Neither a longer
HTML string nor a higher score selects a complete candidate table.

The implemented repair is deliberately narrow:

1. The B0 origin must explicitly be `ecel`, have no text, and span multiple positions.
2. A helper must split exactly that occupied region into unit blank cells.
3. All nonempty B0 cells must retain exact text, coordinates and spans in the helper;
   at least two distinct nonempty text anchors are required.
4. A framed image grid must independently establish the same global row/column
   lattice. Every proposed internal boundary needs at least 95% line support.
   Blank interiors allow at most 0.2% dark pixels after ruling-line exclusion.
5. Other context-compatible helpers may veto the repair; two votes cannot defeat a
   contradictory helper. Agreement alone never proves the structure. The image
   evidence and ownership constraints decide eligibility.
6. Only original continuation tags change to `ecel`. Content bytes, original blank
   origins, global dimensions and unrelated cells remain unchanged. The fused OTSL
   is parsed and officially serialized again before acceptance.

The geometry primitives are reused from `render_verify.visual_checker`; the old
Method C implementation is untouched and no rendering stage is run by SAMR. The
fixed geometric thresholds are inherited engineering checks, not tuned on the new
test cohort. General text-bearing span changes, merges, row insertion/deletion and
uncertain borderless-table repairs retain B0.

Text is compared independently from structure. Conflicts always preserve B0; longer
text, numeric normalization and whitespace cleanup are never replacement rules. Exact
raw strings are preserved when no repair is accepted. Offline strict EM continues to
use the established official-normalized-HTML equality definition.

Synthetic positive tests demonstrate a real structural edit with visible boundaries.
Negative tests cover unequal cell counts/dimensions, ambiguous spans, no image evidence,
majority disagreement, longer conflicting text and exceptions. Actual development
fallbacks are reported honestly, without relabeling unchanged outputs as improvements.
