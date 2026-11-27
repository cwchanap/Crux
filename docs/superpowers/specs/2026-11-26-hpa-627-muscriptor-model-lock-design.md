# HPA-627 — Freeze the canonical MuScriptor model lock

Date: 2026-11-26 · Status: proposed; authenticated execution blocked until proven available

[Linear HPA-627](https://linear.app/cwchanap/issue/HPA-627/commit-frozen-muscriptor-model-lock-runtimemuscriptormodeljson) · [Original P1 on PR #24](https://github.com/cwchanap/Crux/pull/24#issuecomment-5318296815)

## Problem and why now

Crux's remaining non-epic open issue is HPA-627. HPA-319 is the parent benchmark requirements epic, HPA-395 is marked Done for its implementation, and the later HPA-856 and HPA-828 are Done. There are no other open Crux PRs or independent ready backlog tasks at planning time. HPA-627 is not unblocked merely because a plan exists.

Main still has `runtime/muscriptor/smoke.json` but the required `runtime/muscriptor/model.json` returns 404. `src/benchmark/backends/muscriptor.py` defaults to this path and `muscriptor_corpus_run.py` verifies the lock and checkpoint before inference. The HPA-395 implementation plan Task 1 Step 8 required an authenticated freeze. The original review recorded a gated-download HTTP 401, public metadata revision `f32236969308476e01fd3aae67357de5feb05a2d`, and weight license CC BY-NC 4.0; this is historical evidence, not proof of current authentication or checkpoint contents. HPA-395 is therefore operationally incomplete despite its tracker status.

## Decision

Complete the existing freeze, not a new integration. Keep MuScriptor v0.3.0 (MIT code, upstream source commit `d73147e75e5b9b0c0a79ebe154587db4fd603e0c`) and the previously selected medium checkpoint, `MuScriptor/muscriptor-medium`, at the historical pre-score revision above. No authenticated bytes were ever locked, so this is the first real freeze of that intended revision, not a reproduction of an already measured benchmark.

Use one supported, affordable execution host (prefer an available GPU/Colab if medium is impractical on CPU); resolve device and dtype before inference and preserve the exact values in the lock. No hardware-specific changes or additional platforms are required. Medium-to-small substitution is not automatic; if the selected revision or hardware is unavailable, document the blocker and stop, rather than silently selecting a different revision or variant.

Use the existing `scripts/freeze_muscriptor_model.py`: it verifies the installed package, resolves HF metadata, downloads the two files by exact revision, hashes both, derives the model ID, and publishes canonical JSON. It currently resolves a moving repo HEAD. If HEAD still equals the intended revision, no implementation code is needed; freeze in scratch and confirm the generated revision before publication. If HEAD differs, add only the optional exact `--revision` selector to that script and focused offline tests, requesting metadata for that revision and checking its returned SHA before downloading either file. No new model-lock schema, loader, fetch service, CLI subcommand, or compatibility layer.

## Data flow and ownership

1. Authorized Hugging Face session with accepted model terms -> official revision metadata and actual `model.safetensors` / `config.json` downloads.
2. Existing freeze script -> canonical `crux.muscriptor-model/v1` JSON; actual byte lengths, SHA-256 values, package/revision/licenses, resolved device/dtype, model ID, existing deterministic inference settings.
3. Existing `load_muscriptor_model_lock()`, `model_lock_payload()`, `verify_muscriptor_checkpoint()` -> independently validate serialized identity and both local files.
4. Commit only the generated lock (plus these planning docs and any necessary tiny revision-selection fix/tests). Restore exact checkpoint bytes into ignored `artifacts/muscriptor/checkpoint/` from the committed repo/revision/filenames, never re-freeze against HEAD for a consumer checkout.
5. Existing `create_backend()` -> real verified native drum-only inference; existing corpus acceptance test -> persisted report/scorer integration regression. The runtime descriptor and run artifacts continue to carry existing lock/inference identity.

No weight/config bytes, tokens, caches, source audio, predictions, or runtime artifacts are committed. The lock remains the single canonical identity. CI remains offline; gated network access is an explicit operational validation, not a CI dependency. The smoke checks actual model loading and native event validity, not accuracy or model selection.

## Operational gate and scope

A legitimate account must already have accepted the gated model terms and provide authenticated read access in the execution environment (e.g. HF login/secret injection; no credentials in GitHub/Linear/chat or logs). Test actual authenticated download, not merely public metadata. If authorization fails or access is not available, leave this PR draft and HPA-627 open with a bounded error summary. Do not fabricate hashes, commit a placeholder lock, claim actual smoke evidence, or mark HPA-395/HPA-319 operationally complete. No license acceptance on behalf of the account owner.

No full-corpus rerun, broad/reviewed scoring or paired comparison publication, new MLflow dashboard, model training, separator changes, OaF runtime changes, product UI or infrastructure. Those are existing downstream evidence obligations of HPA-395/HPA-319, not additional work to bundle into this lock ticket.

## Acceptance contract

- Actual downloaded medium revision equals the selected historical revision; strict canonical loader and independent verification pass against both downloaded files; model ID and lock SHA are recorded.
- Fresh checkout restores the two exact files by committed revision without running the freeze again; the same lock bytes and identity verify. Edited/missing bytes fail closed through existing verifier tests.
- Agent runs real end-to-end lock -> restore -> backend creation -> bounded native drum transcription on the selected host through available execution MCP, records the exact commands, exit status, device/dtype and validated event summary. Empty valid output on a synthetic smoke is not an accuracy claim; no fake model may satisfy the real gate.
- Existing offline model/backend and persisted corpus acceptance tests pass; normal CI passes. No UI changes or visual acceptance apply.
- Final PR diff contains only the small lock, planning/evidence docs, and the revision selector/test if needed. No weight data or secrets tracked.
- Original PR #24 finding is updated with actual lock link and evidence (or a clear continued block, not a success claim). Keep the same draft PR for this ticket through completion.

See the [implementation plan](../plans/2026-11-26-hpa-627-muscriptor-model-lock.md) for the bounded commands and stop conditions.
