# MESA v2 before audit

Base: e743c68dff189258558e9872ebee3742c517bc09; implementation lives in f3662d02f6d8e7cd8457f367bf8de12bb95d9fcf. Original main tracks origin/main. The original worktree contains paused A2++ changes (.gitignore, src/a2pp, configs/a2pp). Work proceeds in an isolated mesa-v2 worktree; those files and all historical outputs are preserved.

The original Vertical Detail uses y=0 for every crop: a top prefix then the whole table for two strips. Horizontal and tile boundaries are proportional; padding is applied before inference. Mapping later requires a separately detected rule grid, near-boundary crops and matching local dimensions. No crop_mapping.py previously existed. Fusion delegates to strict image-supported blank-span splitting and preserves all text; v2 must not relax that policy.

Required git status, branch -vv, log -5 and remote inspection confirmed the above base and https://github.com/cypre5s/BS.git. Publication remains limited to the requested MESA paths and reports/mesa_v2. The new worktree will push to existing remote main without rewriting history. Historical bytes were fingerprinted before edits in the local preservation manifest.

After the user resumed this goal, they explicitly requested removing A2++ and merging MESA v2 into main. The six untracked A2++ source/configuration files were checked against their existing pause archive, then removed from the working source tree. The two A2++ ignore exceptions were removed, restoring `.gitignore` exactly to existing main. Historical experiment outputs and the archive remain local; no A2++ implementation was tracked or pushed. The original main worktree is clean and can be fast-forwarded to the verified MESA v2 release.
