# MESA v2 publication

MESA v2 is based on e743c68dff189258558e9872ebee3742c517bc09 and was developed on `mesa-v2` in the isolated `BS-mesa-v2` worktree. Its destination is the existing [cypre5s/BS main branch](https://github.com/cypre5s/BS/tree/main). The original main worktree is fast-forwarded to the same release; existing history is retained and no force push is used.

## A2++ cleanup

On resuming this goal, the user explicitly requested clearing A2++. Its five source modules and one configuration were untracked drafts, not a main-branch implementation. Their bytes were verified against the existing pause archive before removing the working directories. The two A2++ ignore exceptions were removed, restoring `.gitignore` byte-for-byte to existing main. Thus the cleanup creates no tracked deletion or root-policy change to publish. Historical A2++ experiment records and the pause archive remain local for recovery. MESA has no A2++ import or launch dependency. Old release reports remain historical records rather than being rewritten.

## Scope and checks

Only `src/mesa/`, `tests/mesa/`, `configs/mesa/`, `scripts/mesa/`, `docs/mesa/` and the six Markdown reports under `reports/mesa_v2/` are staged. The v2 scanner accepts that report directory while retaining the existing content scanner's secret-pattern, binary, personal-path and one-MiB limits. No root publication rules or legacy Runtime/SAMR/A/B/C implementation are modified.

The staged diff and tracked-file inventory are reviewed together with `git diff --check`. Secret, large-file, data-path and complete reachable-history scans run before push and again on the release commit. The preflight scanned 160 historical path/blob pairs with zero findings. Final publication evidence is stored locally in `experiments/mesa_v2/`; those receipts, real predictions, images, datasets, PDFs, model/tokenizer assets, checkpoints, credentials and caches are excluded from Git.

Preservation checks verified 4,843 files from the v2 baseline manifest, with the six explicitly authorized A2++ removals accounted for separately. A further 8,107 historical artifact entries from the older preservation manifest also match their recorded hashes (these manifests may overlap). The unchanged A2++ archive provides recovery for the removed drafts. Native model/processor/tokenizer hash preservation is covered by the GPU smoke receipt and validation report.

## Release verification

Commit and remote verification are recorded after the final validated upload.
