# HPA-627 — Freeze the canonical MuScriptor model lock

Date: 2026-11-26 · Status: proposed; authenticated execution blocked until verified

[Linear HPA-627](https://linear.app/cwchanap/issue/HPA-627/commit-frozen-muscriptor-model-lock-runtimemuscriptormodeljson) · [Original P1 on PR #24](https://github.com/cwchanap/Crux/pull/24#issuecomment-5318296815) · [Design review](https://github.com/cwchanap/Crux/pull/38#pullrequestreview-6084406646)

## Problem

HPA-627 is the remaining concrete Crux backlog task; HPA-319 is the parent epic. Main lacks `runtime/muscriptor/model.json`. The existing backend and corpus runner load and verify this lock and checkpoint before inference. HPA-395's Task 1 Step 8 required an authenticated freeze, but the original PR recorded HTTP 401 on gated downloads. HPA-395 is code-complete but not operationally complete.

## Freeze identity and execution host

Use MuScriptor v0.3.0 (MIT code; source commit `d73147e75e5b9b0c0a79ebe154587db4fd603e0c`), medium checkpoint `MuScriptor/muscriptor-medium`, revision `f32236969308476e01fd3aae67357de5feb05a2d`, weight license CC BY-NC 4.0. The review reports public HEAD still equals that revision; recheck immediately before the authenticated freeze. If HEAD differs or the exact intended bytes cannot be obtained, stop and reassess rather than add a speculative revision selector or silently substitute a checkpoint.

**Proposed HPA-395 corpus host:** Linux x86-64 Colab GPU, preferably a T4-class GPU, Python 3.12 with the existing locked MuScriptor extra, explicit `cuda` / `float32` (the existing CUDA auto-dtype default). This is the selected cost-conscious target, not a claim of verified GPU availability. Verify the actual CUDA runtime, package, memory and disk before publishing anything. The lock's device/dtype are enforced by the backend and enter the run identity; do not freeze against incidental Mac MPS or CPU hardware. The freeze itself may run on another host with the explicit target values. The real smoke must run on the intended corpus host; record GPU/runtime, elapsed transcription time and RTF. This is a feasibility signal, not a full-corpus runtime projection. If the target is unavailable or medium is impractical, stop with evidence; HPA-395 permits a separate pre-score feasibility decision, never an automatic variant or dtype switch.

The existing freeze script already verifies package/license/metadata, downloads both files by exact revision, hashes actual bytes, publishes canonical JSON, reloads and verifies it. Stage the result under ignored artifacts, check the selected revision/variant/device/dtype and record the lock SHA before publication. No redundant verifier implementation or new lock fields.

## Reusable consumer preparation

Extend the existing `crux benchmark prepare-backend` command rather than add another CLI or ask every fresh Colab host to copy a heredoc.

- In `src/benchmark/checkpoint_acquisition.py`, extract the existing HF byte download into one MuScriptor-specific helper, with lazy optional HF import and sanitized acquisition failure. Freeze uses that helper and existing immutable publication to obtain bytes *before* the lock exists. A locked consumer cannot be the freeze entrypoint because checkpoint hashes do not yet exist.
- Add `prepare_muscriptor_checkpoint(lock, checkpoint_dir, *, download)`: verify-only mode calls the existing `verify_muscriptor_checkpoint` without networking; download mode restores missing locked files using the shared helper with the lock's exact repo/revision/filenames, checks expected length and SHA-256 before immutable publication, and verifies the pair. Already-valid cached files need no network; existing mismatched bytes fail without overwrite. A partial legitimate download may be resumed; no new generic cache framework or streaming overhaul. Reuse the existing error family and verifier rather than implementing a parallel lock parser.
- `prepare-backend --backend muscriptor [--download]` loads the canonical lock and defaults to the same paths as MuScriptor corpus/backend (`runtime/muscriptor/model.json`, `artifacts/muscriptor/checkpoint`, honoring existing CRUX_MUSCRIPTOR_MODEL_LOCK and CRUX_MUSCRIPTOR_CHECKPOINT_DIR). Explicit `--cache-root` names the MuScriptor checkpoint directory; when omitted, select the backend-specific default. Reject `--archive` for MuScriptor. Preserve OaF's existing default cache root and behavior, existing JSON summary shape/status/exit conventions; no registry expansion or schema change.
- Add offline acquisition tests and focused CLI dispatch/option tests. Commit a direct canonical-lock test in the same change as the real lock; do not add a skip that would mask a missing lock.

## Operational and acceptance gate

The reviewer reports HF auto-gating: the account owner accepts the terms on the model page and authenticates on the execution host using a read credential (`hf auth login` or securely injected token). Verify actual download rather than assuming that public metadata or login alone grants access. No credentials or gated weight/config bytes are committed. CI stays offline. No estimate or claim of successful authentication until measured.

The fresh-checkout acceptance exercises the actual consumer CLI, then `create_backend()` against the repository's existing `tests/fixtures/oaf_tf1_smoke/canonical.wav`, using `load_direct_audio`. Require nonempty native events, correct descriptor and input identity, and actual completion on the locked device/dtype. The existing adapter checks native events; do not duplicate it. The OaF oracle demonstrates drum activity on this fixture, but MuScriptor's response is not known until execution. Zero hits fail this event-path acceptance, not proof of model accuracy or grounds to retune. Record wall time/RTF, not F1. HPA-395's five-case, repeatability and broad-scoring obligations remain downstream.

No full corpus rerun, new scorer, training, separation, MLflow, UI, new infrastructure or second PR. No change to the backend, inference schema, model dependency constraints or OaF runtime. If any real gate fails, preserve a sanitized diagnosis, leave the PR draft and HPA-627 open; no placeholder lock or claim of benchmark completion.

## Acceptance contract

- Authenticated exact-revision medium bytes are frozen using explicit target device/dtype; the actual generated canonical lock is committed, with no checkpoint/config/credentials tracked.
- Fresh worktree restores exact bytes with the existing CLI's MuScriptor mode and verifies them without re-freezing; verify-only does not network, conflicts fail closed.
- Offline committed-lock test enforces canonical parsing, medium/revision and cuda/float32; mocked acquisition and CLI tests cover download, reuse, invalid bytes, verify-only and option dispatch. OaF regression suite remains green.
- Agent executes real restore and nonempty native inference on the selected corpus host via available execution MCP; records the exact commands, exit, model/lock and input identities, hardware, time and RTF. Existing persisted corpus acceptance and CI pass; verify CI evidence via GitHub MCP. No human acceptance or visual acceptance gate (backend only).
- Final diff limited to lock, planning/evidence, small shared acquisition/CLI/freeze changes and corresponding tests. Original PR #24 finding and Linear get evidence or continued-blocker links. Same PR through completion.

See [implementation plan](../plans/2026-11-26-hpa-627-muscriptor-model-lock.md).
