# HPA-627 — implementation and acceptance plan

> Keep this ticket in the same draft PR. Follow the [design](../specs/2026-11-26-hpa-627-muscriptor-model-lock-design.md) and [Linear HPA-627](https://linear.app/cwchanap/issue/HPA-627/commit-frozen-muscriptor-model-lock-runtimemuscriptormodeljson).

**Files:** `src/benchmark/checkpoint_acquisition.py`, `scripts/freeze_muscriptor_model.py`, `src/cli/benchmark.py`, existing acquisition/CLI/model tests, generated `runtime/muscriptor/model.json`, these docs. No new dependencies, model schema, adapter, runner, or CI infrastructure.

## Task 0 — Establish the actual corpus host and authenticated access

- [ ] Confirm main has no canonical model lock; otherwise inspect the existing lock instead of replacing it. Confirm the intended medium revision `f32236969308476e01fd3aae67357de5feb05a2d` is still HF HEAD with the two expected files and CC BY-NC 4.0 metadata. Public metadata is not download evidence. If HEAD changes, stop and reassess; no new revision flag in this ticket.
- [ ] Provision the proposed HPA-395 corpus host: Linux x86-64 Colab GPU (prefer T4-class), Python 3.12, explicit `cuda`/`float32`. Run `uv sync --locked --extra muscriptor`; confirm installed package and CUDA usability, device memory, RAM and disk. Do not publish a lock while the target runtime is unverified. No MPS/CPU fallback or dependency upgrade.
- [ ] Account owner accepts the terms at [MuScriptor/muscriptor-medium](https://huggingface.co/MuScriptor/muscriptor-medium); authenticate via `hf auth login` with a read credential or an injected secret on the execution host. Use the CLI installed with the locked environment if available, otherwise supported HF authentication tooling. Never log tokens or commit secrets. The reviewer reports automatic access after acceptance, but verify actual authenticated file download before claiming the gate passed. If permission, host or package prerequisites fail, record the blocker; no production lock.

## Task 1 — Reuse acquisition in freeze and consumer paths

- [ ] Extract the HF fetch portion of `_download_immutable` into a small MuScriptor-specific byte-download helper in `checkpoint_acquisition.py` (repo ID, revision, filename). Lazily import HF and surface safe `CheckpointAcquisitionError` messages; keep the freeze script's immutable publication and error/exit contract. Do not change the freeze's metadata checks, hash/model ID derivation or canonical publication.
- [ ] Add `prepare_muscriptor_checkpoint(lock, checkpoint_dir, *, download)` returning the verified weight path from `verify_muscriptor_checkpoint`. Verify-only calls the existing verifier without network. Download mode checks existing files; download only missing files from the lock's exact repo/revision/filenames. Check locked length and SHA-256 before using `publish_immutable_file`; reject conflicts without replacing bytes, and finally verify both files through the existing verifier. Existing valid files are reusable, including after a partial download. Use the existing regular-file reader/immutable publisher, not a generic cache manager. Keep HF optional; OaF must not import it.
- [ ] Extend existing `prepare-backend` in `src/cli/benchmark.py`: dispatch MuScriptor to the lock reader and preparation function, use `CRUX_MUSCRIPTOR_MODEL_LOCK` and `CRUX_MUSCRIPTOR_CHECKPOINT_DIR` defaults consistent with the current runner/backend, and reject `--archive` in MuScriptor mode. Resolve omitted `--cache-root` to the backend-specific default (OaF unchanged); an explicit MuScriptor root is the checkpoint directory itself. Keep existing JSON summary fields and error conventions: invalid selection/lock integrity -> exit 2; acquisition/verification failure -> exit 1; success -> ready/exit 0. Preserve the existing OaF path and option behavior.
- [ ] Extend `tests/benchmark/test_checkpoint_acquisition.py` with small locked fixture bytes and mocked HF downloads: both calls use exact repo/revision/filenames, bytes are restored and verified, repeat/verify-only makes no network call, invalid downloaded bytes are not published, existing mismatches are rejected, missing verify-only fails. Use existing test fixtures/helpers where available. Exercise freeze download-helper wiring using mocked metadata/downloads (no actual token); retain no-network OaF tests.
- [ ] Add focused Click tests in the existing CLI test harness: default and explicit MuScriptor path including environment, verify-only vs download, unsupported archive, lock/load errors and acquisition errors; existing OaF defaults/summary and unsupported backend behavior remain. Run the acquisition and relevant CLI tests before authenticated work. No need to write additional smoke infrastructure.

## Task 2 — Authenticated freeze and commit the canonical lock

- [ ] From repo root, in the verified target environment, create an empty ignored staging area. The freeze may run on another host only after the intended corpus host is established; explicit device/dtype are required.

```bash
mkdir -p artifacts
STAGE="$(mktemp -d "$PWD/artifacts/muscriptor-freeze.XXXXXX")"
uv run --locked --extra muscriptor python scripts/freeze_muscriptor_model.py \
  --variant medium --device cuda --dtype float32 \
  --checkpoint-dir "$STAGE/checkpoint" \
  --output "$STAGE/model.json"
```

- [ ] Require successful exit, then load the staged lock through the existing loader; assert medium, exact historical revision, cuda/float32; record model ID, lock SHA-256 and actual locked file hashes/lengths. The script already verifies actual bytes, model ID and canonical serialization; don't duplicate that work. On mismatch, failed download or host failure, stop without publishing.
- [ ] Copy exact validated staged lock bytes to absent `runtime/muscriptor/model.json`. Add a direct non-skipping test in `tests/benchmark/test_muscriptor_model.py` that loads the repo's actual lock and asserts medium, exact revision, cuda and float32. The strict loader checks canonical form and all frozen settings. Add the test and generated lock in the same commit. Do not commit weights, config, source audio or tokens. Only publish the lock when the host/access gate succeeded; leave draft if later real smoke fails.

## Task 3 — Fresh-checkout real smoke and feasibility

- [ ] On the selected corpus host, create a fresh worktree at the lock commit, with an empty checkpoint destination and no lock/checkpoint overrides. Install the locked MuScriptor extra; keep authenticated HF session/cache available without copying a prior prepared checkpoint directory. From the worktree root run:

```bash
uv run --locked --extra muscriptor crux benchmark prepare-backend \
  --backend muscriptor --download
uv run --locked crux benchmark prepare-backend --backend muscriptor
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

Run with `uv run --locked --extra muscriptor python` on the real target. The adapter already rejects invalid drum events and preserves null velocity. Do not reproduce its per-event validators. Record target GPU/VRAM, Torch/CUDA versions, device/dtype, exact commit, exit status, elapsed time, duration/RTF and event count. Time the inference only (model initialization separate); this short smoke is not a corpus projection. Zero events, loading errors, unavailable GPU or infeasible medium runtime fail the gate; report evidence and stop, not model retuning. Any variant/device/dtype change requires a documented pre-score reassessment of this ticket. Full five-song smoke/repeatability and scored corpus remain HPA-395 obligations.

## Task 4 — Offline regression and acceptance evidence

- [ ] Execute acquisition, model, backend, corpus and CLI regression:

```bash
uv run --locked pytest \
  tests/benchmark/test_checkpoint_acquisition.py \
  tests/benchmark/test_muscriptor_model.py \
  tests/benchmark/test_muscriptor_backend.py \
  tests/benchmark/test_muscriptor_corpus_run.py \
  tests/benchmark/test_muscriptor_corpus_run_acceptance.py -q
uv run --locked pytest tests/test_cli_benchmark.py -k 'muscriptor or prepare' -q
```

Run any other existing CLI tests covering prepare-backend and added focused tests by their exact path. Normal CI runs full offline suite and lint; ensure no gated download or optional HF import is required in ordinary CI. Verify the committed-lock test fails for missing/malformed/wrong identity (not skipped). Confirm final diff is limited to scoped files.

- [ ] End-to-end acceptance by the agent through available execution MCP: Task 2 actual freeze and Task 3 real CLI restore, verify-only and nonempty inference on selected host. GitHub MCP verifies committed lock, PR head and CI job evidence. Local `gh` may assist with execution but does not replace MCP acceptance. Record command, commit, lock SHA, checkpoint/config hashes/lengths, GPU/versions, output/exit and RTF here, excluding secrets/artifact bodies. If runtime or credentials unavailable, retain blocker and unchecked gate. No human acceptance or UI/screenshot criterion applies.
- [ ] Update original [PR #24 finding](https://github.com/cwchanap/Crux/pull/24#issuecomment-5318296815) and Linear with actual evidence or continued blocker. Keep this PR draft until all acceptance gates pass. Do not close HPA-319 or claim HPA-395 broad scoring is complete. One ticket, one PR.

## Execution record

2026-11-26 planning: inspected existing code and tests; lock absent. Round-one design review incorporated. Proposed CUDA/float32 Colab target is not yet provisioned or verified. The review reports unchanged HF HEAD and auto-gating; independent public API retrieval in this review was unavailable and the authorized download remains untested. No code implementation, model freeze, smoke or test execution claimed.
