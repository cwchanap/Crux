# HPA-627 — implementation and acceptance plan

> Keep this ticket in the same draft PR. Follow the [design](../specs/2026-11-26-hpa-627-muscriptor-model-lock-design.md) and [Linear HPA-627](https://linear.app/cwchanap/issue/HPA-627/commit-frozen-muscriptor-model-lock-runtimemuscriptormodeljson).

**Files:** `src/benchmark/checkpoint_acquisition.py`, `scripts/freeze_muscriptor_model.py`, `src/cli/benchmark.py`, `src/benchmark/muscriptor_model.py`, the MuScriptor backend/runner (path resolution only), existing acquisition/CLI/model tests, generated `runtime/muscriptor/model.json`, these docs. No new dependencies, model schema, inference behavior, or CI infrastructure.

## Task 0 — Offline acquisition and CLI implementation (unblocked)

- [x] Before editing the CLI, characterize the current OaF `prepare-backend` behavior with focused `CliRunner` tests in `tests/test_cli_benchmark.py` (mock checkpoint acquisition, not download): default and explicit cache roots, download/archive mutual exclusion (Click usage exit 2), unknown backend (exit 2 + integrity_failed JSON), ready summary and existing lock/acquisition error statuses. These are new baseline tests, not assumed existing coverage.
- [x] Extract the HF byte-fetch portion of `_download_immutable` into a small MuScriptor-specific helper in `checkpoint_acquisition.py` (repo ID, revision, filename). Lazily import HF; normalize failures to `CheckpointAcquisitionError`, with bounded HTTP status (401/403 etc., when available) and chained cause, not raw credential-bearing exception text. Freeze continues to publish immutable bytes and preserve its error/exit contract; it cannot call a lock-based consumer before the hashes exist. Do not change its metadata checks, hash/model ID derivation or canonical publication.
- [x] Add `prepare_muscriptor_checkpoint(lock, checkpoint_dir, *, download) -> Path`, returning the verified **checkpoint directory**. Verify-only uses the existing verifier and returns the directory, without HF import/network. In download mode, first inspect the directory (reject symlink or existing non-directory) and both expected files, including dangling symlinks: treat only actual absence as missing; require each present regular file to match the locked length/hash. Finish the entire preflight before the first HF call. Restore only missing files from the exact repo/revision/filenames; validate locked length/SHA-256 before immutable publication; finally call the existing pair verifier and return the directory. Do not overwrite mismatches, and reuse valid partial downloads. Translate `MuscriptorModelLockError` from verification and all relevant `OSError` (including `ArtifactPublicationError`) from file/directory/publication work into `CheckpointAcquisitionError`; retain useful cause. No duplicated lock parser or generic cache manager.
- [x] Centralize the two default paths and small path resolvers in `muscriptor_model.py`, and reuse them in `backends/muscriptor.py`, `muscriptor_corpus_run.py` and the CLI. Preserve the runner's repo-anchored default lock; checkpoint directory remains CWD-relative. Both env vars treat unset and empty strings as defaults; nonempty values are used as supplied. Explicit backend arguments and CLI cache root override env/default. Reuse existing constants/seams or update callers/tests together; no third path implementation.
- [x] Extend `prepare-backend`: dispatch MuScriptor only after loading its strict lock; separate loader exceptions (integrity_failed / exit 2) from normalized acquisition exceptions (acquisition_failed / exit 1). Ready summary emits the directory (exit 0). Resolve omitted `--cache-root` using the backend-specific default/env; explicit MuScriptor root names the checkpoint directory itself. Reject `--archive` in MuScriptor mode as a usage error. Keep existing OaF path, option behavior and JSON shape. No eager HF import.
- [x] Extend `tests/benchmark/test_checkpoint_acquisition.py` with small locked fixture bytes and mocked HF module/downloads: exact repo/revision/filenames, directory return, repeat/verify-only with HF import forbidden, invalid downloaded bytes not published, existing mismatched config or weight and symlink destination rejected before any download, missing/corrupt verify-only and publication conflict normalized to acquisition errors. Exercise freeze helper wiring with mocked HF metadata/downloads, including distinct safe 401/403 diagnostics and cause. Reuse existing fixtures; no actual tokens or network.
- [x] Extend the existing CLI harness with focused tests for MuScriptor default/explicit/env paths, verify-only vs download, archive rejection, load vs acquisition failure classification and directory JSON summary. Cover shared resolver defaults from outside the repo, explicit relative/absolute values, empty env fallback and explicit overrides. Verify baseline OaF tests remain green. Run the acquisition, backend, corpus and relevant CLI suites on an ordinary CPU host with the base environment, no MuScriptor extra or GPU required. Run offline lint for scoped changes. Commit the tested offline implementation to this same draft before provisioning Colab; the non-skipping canonical-lock test comes only with the generated lock in Task 2.

## Task 1 — Establish corpus host and authenticated access

- [x] Confirm main has no canonical model lock; otherwise inspect it rather than replace it. Confirm intended medium revision `f32236969308476e01fd3aae67357de5feb05a2d` remains HF HEAD with the two expected files and CC BY-NC 4.0 metadata. Public metadata is not download evidence. If HEAD moves, stop and reassess; no new revision flag.
- [ ] Only after Task 0 passes, provision the proposed Linux x86-64 Colab GPU (prefer T4-class), Python 3.12, explicit `cuda`/`float32`. Use one absolute GPU project environment outside both worktrees (set `UV_PROJECT_ENVIRONMENT` to a session-local absolute path), then `uv sync --locked --extra muscriptor`; confirm installed package, actual CUDA usability, VRAM, RAM and disk. The locked Linux Torch wheel/dependencies exist but GPU operation is not established. Do not publish a lock while the runtime is unverified. No MPS/CPU fallback or dependency upgrade.
- [ ] Account owner accepts terms at [MuScriptor/muscriptor-medium](https://huggingface.co/MuScriptor/muscriptor-medium); authenticate with a read credential through `hf auth login` or securely injected secret. Use supported HF tooling for the locked environment. Never log tokens or commit secrets. The reviewer reports automatic access after acceptance, but actual authenticated download must succeed. If permission, host or package prerequisites fail, record blocker and keep the draft open; Task 0's offline work remains valid.

## Task 2 — Authenticated freeze and commit the canonical lock

- [ ] From repo root, in the verified target environment, create an empty ignored staging area. The freeze may run on another host only after the intended corpus host is established; explicit device/dtype are required.

```bash
mkdir -p artifacts
STAGE="$(mktemp -d "$PWD/artifacts/muscriptor-freeze.XXXXXX")"
uv run --no-sync python scripts/freeze_muscriptor_model.py \
  --variant medium --device cuda --dtype float32 \
  --checkpoint-dir "$STAGE/checkpoint" \
  --output "$STAGE/model.json"
```

- [ ] Require successful exit, then load the staged lock through the existing loader; assert medium, exact historical revision, cuda/float32; record model ID, lock SHA-256 and actual locked file hashes/lengths. The script already verifies actual bytes, model ID and canonical serialization; don't duplicate that work. On mismatch, failed download or host failure, stop without publishing.
- [ ] Copy exact validated staged lock bytes to absent `runtime/muscriptor/model.json`. Add a direct non-skipping test in `tests/benchmark/test_muscriptor_model.py` that loads the repo's actual lock and asserts medium, exact revision, cuda and float32. The strict loader checks canonical form and all frozen settings. Add the test and generated lock in the same commit. Do not commit weights, config, source audio or tokens. Only publish the lock when the host/access gate succeeded; leave draft if later real smoke fails.

## Task 3 — Fresh-checkout real smoke and feasibility

- [ ] On the selected corpus host, create a fresh worktree at the lock commit, with an empty checkpoint destination and no lock/checkpoint overrides. Reuse the absolute `UV_PROJECT_ENVIRONMENT` from Task 1 (no second multi-GB venv). From the fresh worktree root run `uv sync --locked --extra muscriptor` against that same environment, ensuring the editable project is rebound to the new checkout; verify `src.cli.benchmark.__file__` resolves inside the fresh worktree, reinstall only the editable project if needed. Do not run the old checkout concurrently in the shared environment. Keep authenticated HF session/cache available without copying a prior prepared checkpoint directory. With the environment synced once for this worktree, run:

```bash
uv run --no-sync crux benchmark prepare-backend \
  --backend muscriptor --download
uv run --no-sync crux benchmark prepare-backend --backend muscriptor
```

Both commands must report ready at the default MuScriptor checkpoint path; second is local verification, no network. This is the independent restore/verification of actual bytes against the committed lock, not another freeze. Record the lock SHA and compare to Task 2.

- [ ] On the same target GPU, exercise the real existing backend using the existing canonical drum-active WAV (three snare hits in OaF oracle, not an expected MuScriptor count):

```python
from pathlib import Path
from time import perf_counter

from src.benchmark.backends.muscriptor import create_backend, descriptor_for_lock
from src.benchmark.input_view import load_direct_audio
from src.benchmark.muscriptor_model import load_muscriptor_model_lock

lock = load_muscriptor_model_lock(Path("runtime/muscriptor/model.json"))
audio = load_direct_audio(
    Path("tests/fixtures/oaf_tf1_smoke/canonical.wav"),
    source_audio_id="muscriptor-lock-smoke",
    input_view_id="crux.oaf-full-mix-mono44k1-pcm16/v1",
    max_input_audio_frames=None,
)
backend = create_backend()
try:
    start = perf_counter()
    prediction = backend.transcribe(audio)
    elapsed = perf_counter() - start
    assert prediction.descriptor == descriptor_for_lock(lock)
    assert prediction.audio == audio
    assert prediction.events, "smoke emitted no native drum events"
    duration = audio.audio_frame_count / audio.sample_rate
    print({"model_id": lock.model_id, "audio_sha256": audio.input_audio_sha256,
           "events": len(prediction.events), "elapsed_sec": elapsed,
           "rtf": elapsed / duration})
finally:
    backend.close()
```

Run with `uv run --no-sync python` on the real target; all GPU-stage commands use the synced optional environment without dropping the extra. The adapter already rejects invalid drum events and preserves null velocity. Do not reproduce its per-event validators. Record target GPU/VRAM, Torch/CUDA versions, device/dtype, exact commit, exit status, elapsed time, duration/RTF and event count. Time the inference only (model initialization separate); this short smoke is not a corpus projection. Zero events, loading errors, unavailable GPU or infeasible medium runtime fail the gate; report evidence and stop, not model retuning. Any variant/device/dtype change requires a documented pre-score reassessment of this ticket. Full five-song smoke/repeatability and scored corpus remain HPA-395 obligations.

## Task 4 — Offline regression, ready-for-review CI, and evidence

- [ ] While the PR is draft, after real GPU acceptance, run the acquisition, model, backend, corpus and CLI regression and lint on an ordinary CPU checkout at the final implementation head using the base environment (do not resync the active GPU environment to base dependencies):

```bash
uv run --locked pytest \
  tests/benchmark/test_checkpoint_acquisition.py \
  tests/benchmark/test_muscriptor_model.py \
  tests/benchmark/test_muscriptor_backend.py \
  tests/benchmark/test_muscriptor_corpus_run.py \
  tests/benchmark/test_muscriptor_corpus_run_acceptance.py -q
uv run --locked pytest tests/test_cli_benchmark.py -k 'muscriptor or prepare' -q
```

Run added focused tests by exact path, and local `uv run ruff check .`, `uv run ruff format --check src tests`, `uv run pylint --errors-only src`. Run the offline suite required by the existing CI contract or equivalent local command before marking ready. Ensure no gated download or optional HF import is required. Verify the committed-lock test fails for missing/malformed/wrong identity (not skipped). Confirm final diff limited to scoped files.

- [ ] End-to-end acceptance by the agent through available execution MCP: Task 2 actual freeze and Task 3 real CLI restore, verify-only and nonempty inference on selected host. GitHub MCP verifies committed lock and exact PR head; CI verification follows readiness in the next step. Local `gh` may assist with execution but does not replace MCP acceptance. Record command, commit, lock SHA, checkpoint/config hashes/lengths, GPU/versions, output/exit and RTF here, excluding secrets/artifact bodies. If runtime or credentials unavailable, retain blocker and unchecked gate. No human acceptance or UI/screenshot criterion applies.
- [ ] Commit the real acceptance evidence before readiness. The current `.github/workflows/ci.yml` skips draft PRs and listens for `ready_for_review`. Once all local offline/lint and real GPU gates have passed, mark this same PR ready for review, then verify actual successful lint, test and IDM-runtime jobs for the exact PR head via GitHub MCP. A skipped draft job is not acceptance. If CI fails, fix in this PR and rerun; no merge or ticket closure until green. If an operational gate is blocked, stay draft and record it instead.
- [ ] Update original [PR #24 finding](https://github.com/cwchanap/Crux/pull/24#issuecomment-5318296815) and Linear with actual evidence or continued blocker. Close HPA-627 only after acceptance and merge; do not close HPA-319 or claim HPA-395 broad scoring is complete. One ticket, one PR.

## Execution record

2026-11-26 planning: inspected existing code and tests; lock absent. Round-one and round-two design reviews incorporated (offline-first, normalized error boundary, shared paths, one GPU environment, directory return and CI readiness ordering). The round-two review's Torch CUDA dependency and draft CI gate claims were verified against main. Proposed CUDA/float32 Colab target is not yet provisioned or verified. The review reports unchanged HF HEAD and auto-gating; independent public API retrieval in this review was unavailable and the authorized download remains untested. No code implementation, model freeze, smoke or test execution claimed.

**2026-11-28 offline implementation.** The existing CLI now restores the two files from the locked revision with `--backend muscriptor --download` and verifies local files with `--backend muscriptor`. Restore checks both destinations before downloading and uses the immutable publisher with a directory descriptor. The freeze script shares the download helper. The CLI, backend and corpus runner share the checkout-anchored lock path and working-directory-relative checkpoint default.

The 330 focused acquisition, model, backend, corpus and CLI tests passed in the locked Python 3.12 environment. The installed CLI verified a fixture checkpoint from outside the checkout. All 32 IDM runtime tests passed on Python 3.11. Ruff lint and formatting and Pylint errors-only checks passed.

**2026-11-28 public preflight.** Main at `65ec2e1` still lacks `runtime/muscriptor/model.json`. The public API reported revision `f32236969308476e01fd3aae67357de5feb05a2d`, automatic gating, license `cc-by-nc-4.0`, and both required filenames. Recheck HEAD on the execution host immediately before the freeze. [Hugging Face model metadata](https://huggingface.co/api/models/MuScriptor/muscriptor-medium)

The current Linux ARM64 runner has no CUDA device or configured Hugging Face credential. The authenticated freeze and real inference still need access to the selected Linux x86-64 CUDA host with gated model access.
