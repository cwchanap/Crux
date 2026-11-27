# HPA-627 — implementation and acceptance plan

> Execute in the same draft PR. This is an operational freeze with a small Git diff, not a new benchmark subsystem.

[Design specification](../specs/2026-11-26-hpa-627-muscriptor-model-lock-design.md) · [Linear](https://linear.app/cwchanap/issue/HPA-627/commit-frozen-muscriptor-model-lock-runtimemuscriptormodeljson)

**Expected implementation files:** `runtime/muscriptor/model.json` (generated, not hand-authored); this plan's execution record. Only if HF HEAD differs from the chosen revision: `scripts/freeze_muscriptor_model.py` and a focused freeze-script test. Existing `pyproject.toml`, `uv.lock`, strict loader, backend, runner and smoke manifest otherwise stay unchanged.

## Task 0 — Establish access and pre-score identity

- [ ] On the PR branch from current main, verify `runtime/muscriptor/model.json` is absent, `runtime/muscriptor/smoke.json` is present, and `artifacts/` is ignored. If a canonical lock has since landed, inspect it instead of overwriting or making a second lock.
- [ ] Use the existing Python 3.12 environment and optional dependency: `uv sync --locked --extra muscriptor`. Confirm host device/dtype and available memory/storage before downloading; use one affordable execution host. Do not change the root dependency constraints just to work around an unmeasured failure.
- [ ] In an authorized runtime, confirm access to [MuScriptor/muscriptor-medium](https://huggingface.co/MuScriptor/muscriptor-medium). Account terms/access must already be accepted; use a read credential provided through the execution environment, not through committed files or command output. Public model metadata alone is insufficient.
- [ ] Compare metadata for repo HEAD and revision `f32236969308476e01fd3aae67357de5feb05a2d`. Confirm the official repo, exact revision, two expected files and CC BY-NC 4.0 metadata. Confirm the package is `muscriptor==0.3.0` and MIT. If exact historical revision cannot be verified/downloaded, stop and record it. No moving-HEAD substitution and no score-informed selection.

**Conditional minimal change only if HEAD moved:** extend the existing freeze script with optional `--revision <40-lowercase-hex>` (default existing resolution). Feed that revision into `HfApi().model_info(repo_id, revision=...)`; require returned SHA equality, retain existing file/license checks, and pass the exact revision to both existing download calls. Focused offline tests mock HF metadata/download calls and check exact revision propagation, invalid or mismatched revision rejection before downloads, and existing no-argument behavior. No second script, schema change or API beyond this selector. If HEAD remains the intended revision, skip this entirely.

## Task 1 — Authenticate and freeze once

- [ ] Run existing offline tests first:

```bash
uv run --locked pytest tests/benchmark/test_muscriptor_model.py \
  tests/benchmark/test_muscriptor_backend.py -q
```

- [ ] Create a new empty ignored staging directory (never reuse partial files from another attempt). From repo root:

```bash
STAGE="$(mktemp -d "$PWD/artifacts/muscriptor-freeze.XXXXXX")"
uv run --locked --extra muscriptor python scripts/freeze_muscriptor_model.py \
  --variant medium --device auto --dtype auto \
  --checkpoint-dir "$STAGE/checkpoint" \
  --output "$STAGE/model.json"
```

Create the ignored `artifacts/` parent before `mktemp`. If Task 0 required the selector, append `--revision f32236969308476e01fd3aae67357de5feb05a2d`. The existing auto choices must resolve to the selected host; explicit device/dtype are acceptable if chosen and recorded before real inference. No CPU/GPU or variant change after result inspection.

- [ ] Validate the actual output rather than trusting script exit alone. Use existing `load_muscriptor_model_lock`, `derive_muscriptor_model_id`, `model_lock_payload`, `canonical_json_bytes`, and `verify_muscriptor_checkpoint` to confirm selected revision, medium, actual weight/config byte hashes and lengths, canonical serialization including final newline, derived ID, MIT/CC BY-NC, drums-only/no-sampling settings, and actual device/dtype. Record lock file SHA-256. The two checkpoint downloads must be authenticated and verified; no placeholder or hash copied from public metadata.
- [ ] Only after validation, publish the exact staged lock bytes as `runtime/muscriptor/model.json` (destination must be absent); leave checkpoint bytes under ignored artifacts. Stage only the lock and intended docs/test changes. Verify the lock is not ignored and no checkpoint/config blobs or credentials are staged. Commit to this same PR branch.

**Stop:** no lock publication on gated-download failure, incomplete bytes, unavailable selected revision, environment failure or identity mismatch. Preserve only sanitized diagnosis in the PR/ticket; a failed run is not acceptance.

## Task 2 — Reproduce from a fresh checkout and exercise the real backend

- [ ] Create a fresh detached worktree at the commit containing the generated lock. Use the locked Python environment with MuScriptor extra. Restore the two files using the committed identity, not another freeze (run inside that checkout):

```bash
uv run --locked --extra muscriptor python - <<'PY'
from pathlib import Path
from shutil import copyfile
from huggingface_hub import hf_hub_download
from src.benchmark.muscriptor_model import (
    load_muscriptor_model_lock, verify_muscriptor_checkpoint,
)
lock = load_muscriptor_model_lock(Path("runtime/muscriptor/model.json"))
target = Path("artifacts/muscriptor/checkpoint")
target.mkdir(parents=True, exist_ok=True)
for name in (lock.checkpoint_filename, lock.checkpoint_config_filename):
    cached = hf_hub_download(
        repo_id=lock.checkpoint_repo_id,
        revision=lock.checkpoint_revision,
        filename=name,
    )
    copyfile(cached, target / name)
verify_muscriptor_checkpoint(lock, target)
print(lock.model_id)
PY
```

Use a clean checkout without any old checkpoint destination; HF's verified cache may avoid a redundant large download. Compare committed lock SHA to Task 1; verify both actual restored files. Ensure no environment variable overrides the canonical model lock or checkpoint directory during the smoke.

- [ ] Exercise the actual upstream package through the real `create_backend()` default path with one deterministic short 44.1 kHz mono PCM16 WAV (e.g. six seconds with brief periodic clicks, created under ignored artifacts). Use no scored chart data. Construct `CanonicalAudio` from its true byte SHA-256, byte count, frame count, sample rate/channel/sample width, source ID and existing full-mix input-view ID `crux.oaf-full-mix-mono44k1-pcm16/v1`. Run `backend.transcribe(audio)` in a try/finally that closes the backend. Assert descriptor model ID equals the committed lock, returned prediction refers to the same audio, every emitted event is drums-only with finite nonnegative onset, integer MIDI pitch in 0–127, and no fabricated velocity. Report event count (including zero if the input produces no valid hits), duration and completion, not a score. Use the exact locked device/dtype; no mock, heuristic, MIDI-only path or alternative model. This verifies actual authenticated lock -> restored bytes -> adapter inference, and does not claim full HPA-395 production evidence.
- [ ] If the actual loader or inference fails, stop and capture the exact sanitized failure. Do not change the model or schema to make a smoke pass. Any nontrivial integration repair is a separately scoped follow-up; do not claim completion.

## Task 3 — Regression and final acceptance evidence

- [ ] Run the existing strict model/backend and persisted corpus E2E acceptance on the exact branch:

```bash
uv run --locked pytest \
  tests/benchmark/test_muscriptor_model.py \
  tests/benchmark/test_muscriptor_backend.py \
  tests/benchmark/test_muscriptor_corpus_run.py \
  tests/benchmark/test_muscriptor_corpus_run_acceptance.py -q
uv run --locked pytest tests/test_cli_benchmark.py -k muscriptor -q
```

If the conditional selector was implemented, run its focused tests as well. The corpus test reconstructs HPA-325 reports from persisted predictions; it complements, but does not replace, the real smoke. Use normal GitHub CI for the complete offline suite and lint. Verify the exact PR head and CI job results via GitHub MCP.

- [ ] Agentic acceptance: use available execution MCP for Task 1 and fresh-checkout Task 2, and GitHub MCP for committed lock diff/CI verification. Retain a compact Markdown evidence record in this plan with revision, lock SHA, actual checkpoint/config SHA and lengths, resolved device/dtype, exact runtime/commands, exits and native smoke summary. Do not include credentials, checkpoint bodies or raw artifact dumps. If execution MCP/authenticated runtime is unavailable, record the blocker and leave acceptance unchecked. No human acceptance gate; no visual/UI acceptance because this is backend-only.
- [ ] Confirm final PR diff contains only the generated canonical lock, plan/spec/evidence and conditional selector/test if needed. Confirm `git status --short` / `git ls-files` show no tracked model artifacts or credentials. Update the original [PR #24 finding](https://github.com/cwchanap/Crux/pull/24#issuecomment-5318296815) and Linear with actual lock and evidence references or continued operational blocker. Keep HPA-627 open until accepted; do not automatically close the HPA-319 epic or claim full corpus comparison. No separate PR for the lock.

## Execution record

Planning inspection (2026-11-26): main has the existing freeze script, optional package, strict loader, backend and corpus acceptance tests; the canonical lock is absent. The recorded gated-access failure from PR #24 has not been retested with authorized credentials in this session. No freeze, model execution or tests have been run as part of planning. All implementation acceptance items remain unchecked.
