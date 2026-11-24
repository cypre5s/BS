# SAMR Git release record

Repository: <https://github.com/cypre5s/BS>, existing `main` branch.
Release date: 2026-11-24. No force push, history replacement or old-method deletion
was performed.

The implementation, tests, configurations and technical reports were committed and
successfully pushed as
[`b510cbc7ba4ae1f1470ad1e2e78ff0944ee8fd2f`](https://github.com/cypre5s/BS/commit/b510cbc7ba4ae1f1470ad1e2e78ff0944ee8fd2f),
“Add structure-aware multi-resolution inference and audited development smoke”.
The existing parent is `5d6eab155cf4c45622ab84c05c984d4ad6861773`.
An explicit `git ls-remote origin refs/heads/main` matched the implementation SHA
after the ordinary push. `git status --porcelain` was empty at that verification.
This release-record document is a subsequent documentation commit; it does not
change the tested implementation. The final delivery receipt separately records its
own HEAD/remote equality and clean working tree, avoiding a self-referential commit hash.

The implementation commit changes 49 allowed files, with 2,481 added and six removed
lines. Existing Method A/B/C and the vendor converter have no diff. Shared runtime
changes are optional budget guards; historical default behavior is retained and the
real native B0 replay passed. The final documentation commit only records release
provenance and links the implementation SHA.

| Publication check | Observed result |
| --- | --- |
| Initial branch/remote/history | Clean existing main, fetched remote matched the parent SHA |
| File whitelist | Only source, tests, scripts, YAML configuration and sanitized Markdown staged |
| Working-source secret scan | PASS |
| Staged-content secret/path/binary scan | PASS |
| Large-file scan | Largest implementation staged file 11,057 bytes; all below the 1 MiB source limit |
| Historical Git objects | 81 original blobs checked before commit; 130 distinct path/blob pairs checked after implementation push; PASS |
| `.gitignore` probes | Predictions/JSONL, image data, weights, nested SAMR local YAML, environment files and SSH key filenames ignored |
| Historical experiment preservation | 6,720 files matched their pre-change hashes |
| Test/source receipt | CPU/static receipt binds 41 relevant code/config/test/script files to SHA-256 values |
| GitHub push | Ordinary `git push origin main` succeeded and remote SHA matched |

Only publication allowlist candidates were passed to `git add`. The precommit check
caught a missing nested-local-config ignore rule; `configs/samr/local*.yaml` was added
and every ignore probe passed before any commit. No excluded artifact was staged.

Local evidence remains excluded under `experiments/samr/`:
`publication_precommit.json`, `publication_after_code_push.json`,
`history_preservation.json`, `static_validation.json` and the final delivery receipt.
Data, images, actual predictions, raw smoke receipts, weights, caches, credentials and
personal machine paths were not published. No formal SAMR 665-table job was launched.
