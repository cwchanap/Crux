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

## Fix Round 2 — final re-review integrity fixes

This round closes the two remaining Important findings from the final
re-review without changing the full-mix path or the progress ledger.

### TDD RED

The new publication-seam tests were run before their descriptor-bound helpers
existed. The initial focused run observed three failures for the missing
primary namespace/prediction/report descriptor seams. The worker identity
regressions were likewise added at the request boundary before the worker and
backend changes were completed; they exercised the old pathname-based audio
decode and missing request identity fields.

### GREEN and verification

The focused IDM backend, worker-process, and primary acceptance suites passed:

```text
.venv/bin/pytest -q tests/benchmark/test_idm_backend.py \
  tests/benchmark/test_worker_process.py \
  tests/benchmark/test_idm_pilot_run_acceptance.py
80 passed in 19.75s

.venv/bin/pytest -q tests/benchmark/test_idm_pilot_run_acceptance.py
23 passed in 17.14s
```

The complete HPA-396 matrix, including the new audio identity and publication
race cases, passed `367` tests in `21.93s`. The full repository suite passed
`3142` tests in `292.03s (0:04:52)`:

```text
.venv/bin/pytest -q <HPA-396 test matrix>
367 passed in 21.93s

.venv/bin/pytest -q
3142 passed in 292.03s (0:04:52)
```

Static verification also passed:

```text
.venv/bin/ruff check .
All checks passed!

.venv/bin/ruff format --check src tests
170 files already formatted

.venv/bin/pylint --errors-only --disable=E1120,E0401 --jobs=1 src
exit 0; no output

git diff --check
exit 0; no output
```

The pinned real IDM runtime/WAV probe still passed after the descriptor
changes, with `python_version=3.11.12`, the locked model/classes,
`sample_rate_hz=44100`, `activation_rate_hz=172.265625`, `device=cpu`,
`dtype=float32`, and `events=78`.

### Files changed

- `runtime/idm/worker.py`
- `src/benchmark/backends/idm.py`
- `src/benchmark/idm_pilot_run.py`
- `src/benchmark/worker_process.py`
- `tests/benchmark/test_idm_backend.py`
- `tests/benchmark/test_idm_pilot_run_acceptance.py`
- this report

### Self-review

The IDM request now carries the retained CanonicalAudio byte length and
SHA-256. The isolated worker opens the staged leaf once through held
`O_NOFOLLOW` ancestor descriptors, requires a regular file, reads and hashes
the bytes from that descriptor, and decodes the verified bytes from memory;
it never reopens or decodes the pathname. Leaf, ancestor, symlink,
non-regular, length, digest, and protocol mismatch cases fail closed before a
prediction can be labeled with the retained identity.

The primary stem runner holds no-follow descriptors for the output root,
`runs`, exact run, input, predictions, dynamic prediction parent, reports, and
owned report roots. Checkpoints, retained-input materialization, prediction
publication, and final report leaves use those held descriptors with private
temporary files, no-follow opens, atomic replacement, and fsync. Report bytes
are still produced by the existing scoring/report helpers. Resume identity and
immutable prediction conflict behavior remain intact. The publication-seam
tests swap the run, dynamic prediction, and report directories after
validation and assert that no outside sentinel is changed.

No generic filesystem framework, dependency, full-mix redesign, or speculative
scope was introduced.

### Remaining production operational block

The checkout still has no production HPA-328 immutable handoff and no
`runtime/idm/smoke.json`. The pinned demo probe is therefore the available
production-like evidence; production five-song membership, full-corpus stem
inference, scored comparison, and the operational smoke remain blocked until
that upstream handoff and attested production evidence exist.
