# HPA-396 final whole-branch fix report

Base: `7195f83ff942a578d1d169524d5b1f55d9460b19`

This round fixes the three Important findings from the final review without
editing the progress ledger.

## TDD evidence

Observed RED before the implementation:

```text
uv run pytest tests/benchmark/test_idm_pilot_run_acceptance.py \
  -k 'private_verified_input or output_namespace' -q
4 failed, 1 passed, 15 deselected

uv run pytest tests/benchmark/test_idm_backend.py \
  -k 'wrong_ready_identity or non_kiss_runtime or ready_reports' -q
7 failures
```

The first command observed the worker receiving replacement bytes from the
retained pathname and the namespace cases reaching the old write path. The
second observed missing host validation for model name, Python version, device,
and dtype, and the worker ready payload lacked those effective facts.

Focused GREEN:

```text
uv run pytest tests/benchmark/test_idm_pilot_run_acceptance.py \
  -k 'private_verified_input or output_namespace' -q
5 passed, 15 deselected

uv run pytest tests/benchmark/test_idm_backend.py -q
37 passed
```

The complete HPA-396 regression set passed:

```text
uv run pytest \
  tests/benchmark/test_idm_model.py \
  tests/benchmark/test_idm_backend.py \
  tests/benchmark/test_idm_pilot_run.py \
  tests/benchmark/test_idm_pilot_run_acceptance.py \
  tests/benchmark/test_idm_comparison.py \
  tests/benchmark/test_backend_identity.py \
  tests/benchmark/test_cohort_scoring.py \
  tests/benchmark/test_mapping.py \
  tests/benchmark/test_prediction_artifact.py \
  tests/benchmark/test_prediction_artifact_coverage.py \
  tests/runtime/test_idm_wheel_builder.py \
  tests/test_cli_benchmark.py \
  tests/test_cli_benchmark_coverage.py -q
357 passed
```

## Implementation

### Verified retained audio handoff

`_prepare_handoff_row()` now retains the exact no-follow-verified WAV bytes.
Before IDM inference, those bytes are published once into a run-owned
`runs/<run-id>/inputs/<simfile-id>/verified.wav` regular file through a held
`O_NOFOLLOW|O_DIRECTORY` descriptor. The write is durable, immutable, and
private (`0600`); the bytes are re-read and checked before use. `CanonicalAudio`
is replaced with that staged path and the backend input root is the run-owned
input root, while source/input hashes remain the verified handoff identity.
Existing identical staged bytes are reused; conflicting bytes fail closed.
The acceptance regression replaces the retained source after verification and
proves the backend reads the staged verified bytes instead.

### Primary output namespace

The primary runner now validates the output root, `runs`, exact run directory,
`run.json`, `predictions`, run-owned inputs, reports, and both report roots by
`lstat()` without following preseeded symlinks. It rejects non-directories in
intermediate components, non-regular output leaves, and resolved paths outside
the output root. Resume behavior remains identity-based: an existing exact
snapshot is validated, while a changed identity may create its own new run
directory as before. Prediction targets are checked before inference and again
before immutable publication; report roots and all six report leaves are
checked before report generation. Durable directory creation is used at the
run, input, prediction, and report boundaries.

Parameterized acceptance coverage proves `runs`, exact run directory,
`predictions`, and `reports` symlinks cannot redirect writes to an outside
sentinel and fail before the backend is invoked.

### Worker readiness attestation

The isolated IDM worker now reports the actual loaded model name and class
ordering, encoder sample/frame rates, exact interpreter version, and effective
parameter device/dtype. Missing tensor facts, mixed devices/dtypes, and any
requested/effective device mismatch fail closed. The host validates all of
these fields against `IdmModelLock` and rejects any KISS lock/config that is not
CPU/float32 before starting a worker. The `-I -S` startup and import defenses
remain unchanged.

## Files changed

- `runtime/idm/worker.py`
- `src/benchmark/backends/idm.py`
- `src/benchmark/idm_pilot_run.py`
- `tests/benchmark/test_idm_backend.py`
- `tests/benchmark/test_idm_pilot_run_acceptance.py`
- `.superpowers/sdd/2026-08-20-hpa-396-idm-stem-pilot/final-fix-report.md`

## Final verification

```text
uv run pytest -q
3132 passed in 256.37s (0:04:16)

uv run ruff check .
All checks passed!

uv run ruff format --check src tests
170 files already formatted

uv run pylint --errors-only --disable=E1120,E0401 --jobs=1 src
exit 0; no output

git diff --check
exit 0; no output
```

The real pinned runtime/WAV probe also passed after the final readiness
hardening:

```text
ready: backend_id=idm-44-train-kits-v1
model_id=idm-44-train-kits-456656868538-5856a9bee7c6
model_name=idm-44-train-kits
python_version=3.11.12
train_classes=[CY_CR,CY_RD,HH_CHH,HH_OHH,KD,SD,TT_HFT,TT_HMT,TT_LMT]
sample_rate_hz=44100 activation_rate_hz=172.265625 device=cpu dtype=float32
events=78
```

## Self-review and remaining limitation

The change is limited to the three reviewed integrity boundaries. No generic
namespace framework, new dependency, full-mix redesign, production registry,
or progress-ledger edit was introduced. The real probe uses the pinned demo
WAV only; the checkout still has no production HPA-328 immutable handoff and
no `runtime/idm/smoke.json`. Therefore production five-song membership,
full-corpus stem inference, scored comparison, and operational smoke remain
blocked until that upstream handoff and attested production evidence exist.
