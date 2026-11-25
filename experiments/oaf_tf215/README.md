# OaF TensorFlow 2.15 experiment (Python 3.11)

Bounded compatibility experiment for HPA-856. It asks one question: can the
existing OaF drum checkpoint install, execute its graph, restore and transcribe
under Python 3.11 / TensorFlow CPU 2.15.1, and how does that output compare with
the frozen TF1 oracle?

The probes use a scratch runtime built from the frozen TF1 source. The corpus
benchmark opts in through `crux benchmark run-oaf-corpus --backend oaf-tf215`
and binds prediction paths to the candidate descriptor and runtime identity.
Graph checks cover the exercised TensorFlow compatibility sessions. The
full-model smoke comparison checks exact parity against the frozen TF1 oracle.

## Files

| Path | Purpose |
| --- | --- |
| `requirements.in` | Candidate pins (TF CPU 2.15.1, Estimator/Keras 2.15, tf-slim, and the frozen numerical/audio/MIDI stack). |
| `requirements.lock` | Hashed lock resolved for the frozen runtime's target: Python 3.11, `x86_64-manylinux_2_17`. |
| `graph_check.py` | Executes real Slim Conv2D + batch-normalization and legacy LSTM sessions and records stage-specific diagnostics. |
| `build_runtime.py` | Creates one scratch `runtime/` + `upstream/` from the frozen TF1 source, applies the instrumentation patch first and then `patches/tf215-compat.patch`. |
| `patches/tf215-compat.patch` | The single reviewed TF 2.15 compatibility patch (drum `tf_slim`, HParams/RNN, Estimator, unused package-initializer imports). |
| `compare.py` | Python 3.12 driver: verifies the checkpoint and fixture, then runs two requests through one loaded Python 3.11 scratch worker plus one through a fresh worker and compares the exact native events with the frozen binary64 smoke oracle. |
| `selected_compare.py` | Seals archived or fresh TF1 inputs for the five fixed songs and measures TF2.15 drift with a separate runtime identity. |
| [`src/benchmark/oaf_tf215_runtime.py`](../../src/benchmark/oaf_tf215_runtime.py) | Fingerprints source files and inventories the isolated worker environment. |
| `check_workflow_result_fallback.py` | Stdlib-only local check: the workflow's always-on result step writes a failed sync/check/build/graph/environment record when no graph output exists. |
| `README.md` | This file. Reproduction commands and evidence scope. |
| `RESULTS.md` | Recorded CI, graph, restore, inference and manual-job evidence plus the decision checkpoint; not a parity claim. |

`tests/runtime/test_oaf_tf215_builder.py` (stdlib only) and
`tests/benchmark/test_oaf_tf215_compare.py` / `test_oaf_tf215_workflow.py`
(project Python 3.12) cover the builder and the driver/workflow behavior.

## Target platform

The first execution host is GitHub Actions `ubuntu-24.04`, Linux **x86-64**,
which matches the frozen TF1 Docker image's `linux/amd64` target. Do not resolve
or run this lock on macOS/ARM64 and call it evidence: `tensorflow-cpu==2.15.1`
has no macOS ARM64 wheel, and the lock is compiled explicitly for
`x86_64-manylinux_2_17`.

## Five-song comparison

The selected comparison uses the TF1 reports pinned in the
[design](../../docs/superpowers/specs/2026-11-17-hpa-856-tf215-identity-comparison-design.md)
and the original archived predictions for simfiles 389, 391, 27, 241 and 207.
The [results](RESULTS.md#five-song-continuation) track access to that archive.

Export the three pinned reports from Git to `$TF1_REPORTS`. Stage a byte-for-byte
copy of the matching TF1 `run.json` at `$TF1_BUNDLE/run.json`, with its original
`predictions/…` paths underneath the same directory. In the original corpus
output the run lives under `runs/<run-id>/`, while its prediction paths are
relative to the corpus output root. Keep the original manifest layout for the
reference and timing manifests and their native event bodies. Preparation also
requires the verified source audio cache.

Run preparation once with those inputs. Use its printed digest for each
execution. Copy the sealed selection, canonical WAVs and archived prediction
bundle to each fresh Linux x86-64 Colab CPU VM through an authorized data source.
On Colab, give `prepare`, `prepare-fresh`, and `prepare-panel` a VM-local
`--input-root` under `/content`. Preparation publishes the complete WAV
directory with an exclusive atomic rename. Copy the completed artifacts to
Drive after preparation.

```bash
EXPERIMENT=artifacts/experiments/oaf-tf215
SELECTION_SHA256=$(uv run experiments/oaf_tf215/selected_compare.py prepare \
  --baseline-evidence-dir "$TF1_REPORTS" --baseline-run "$TF1_BUNDLE/run.json" \
  --reference-manifest "$REFERENCE_MANIFEST" --timing-manifest "$TIMING_MANIFEST" \
  --cache-dir "$CORPUS_CACHE" --input-root "$EXPERIMENT/selected-inputs" \
  --output "$EXPERIMENT/selection.json")
```

On each VM, provision the locked worker, build the scratch runtime, run the
graph checks and populate the public checkpoint cache using steps 1–4 below.

```bash
uv run experiments/oaf_tf215/selected_compare.py run \
  --selection-path "$EXPERIMENT/selection.json" --selection-sha256 "$SELECTION_SHA256" \
  --baseline-run "$TF1_BUNDLE/run.json" --input-root "$EXPERIMENT/selected-inputs" \
  --python311 "$PY311" --runtime-root "$SCRATCH/runtime" \
  --checkpoint-cache artifacts/benchmark/model-cache --output-root "$EXPERIMENT/selected"
```

Each invocation prints its status and output directory. A completed run
contains the sealed selection, the binary64 smoke result, all 15 raw song
attempts, and the comparison report. The report covers strict 1 ms agreement,
possible shifts within one emitted frame (about 11.61 ms), and remaining
additions and removals. Selected-song metrics use the six-decimal precision of
the archived TF1 artifacts. Exit 0 confirms a completed repeatable
measurement, exit 2 indicates blocked access, and exit 1 indicates a failed
or non-repeatable measurement. Early failures write `comparison.json` under
`selected/preflight/<execution-id>/` when a runtime identity is unavailable.

### Fresh September baseline

The fresh path uses the same five song IDs and the pinned September reference
and timing manifests. Run the frozen TF1 corpus driver with those manifests,
the verified source cache and one `--include-simfile-id` flag for each song.
The stock CLI requires the built `crux-oaf-tf1:local` image and a working
Docker runtime. The Colab experiment uses the same corpus driver with the
pinned OCI rootfs launcher recorded in [RESULTS.md](RESULTS.md#fresh-september-tf1-baseline).
The runner preflights all 180 eligible timing artifacts. A complete run
produces `runs/oaf-9dce500b62a3e24c/run.json` under its output root.

```bash
SELECTION_SHA256=$(uv run experiments/oaf_tf215/selected_compare.py prepare-fresh \
  --baseline-run "$TF1_OUTPUT/runs/oaf-9dce500b62a3e24c/run.json" \
  --reference-manifest "$REFERENCE_MANIFEST" --timing-manifest "$TIMING_MANIFEST" \
  --cache-dir "$CORPUS_CACHE" --input-root "$EXPERIMENT/selected-inputs" \
  --output "$EXPERIMENT/selection.json")
```

Use the same `run` command above with the fresh `run.json` path and seal. The
fresh seal carries the October ranking as selection provenance and derives its
prediction counts from the fresh TF1 artifacts. For the second VM, retain the
same TF1 run and predictions, and check the rematerialized WAVs against the
original selection digest.

### Twenty-song panel

The [panel protocol](PANEL.md) fixes the membership, exact-input policy and
evaluation gate. To evaluate it, use `measure-panel` on the x86-64 Colab CPU VM
with the locked TF2.15 worker and verified checkpoint cache. Set
`TF1_ROOTFS_LAUNCHER` to the `tf1_rootfs.py` from the fresh September evidence.
The command checks its SHA-256
(`c503911ca6eab92b93fff71f76c4e397209ed3cf43105d306ee28c5591d1553c`)
and installs a fresh TF1 rootfs from that launcher.

```bash
uv run experiments/oaf_tf215/selected_compare.py measure-panel \
  --tf1-rootfs-launcher "$TF1_ROOTFS_LAUNCHER" \
  --reference-manifest "$REFERENCE_MANIFEST" --timing-manifest "$TIMING_MANIFEST" \
  --cache-dir "$CORPUS_CACHE" --python311 "$PY311" --runtime-root "$SCRATCH/runtime" \
  --checkpoint-cache artifacts/benchmark/model-cache --output-root /content/oaf-panel-results
```

Keep `--output-root` on VM-local storage so the paired run can stage and
publish the WAVs there. Each invocation creates a new directory with the
frozen smoke prediction, TF1 run, sealed selection, and candidate comparison.
The same verified TF1 worker launcher runs the smoke and baseline. The seal
binds the VM boot ID to the exact baseline run and smoke digests. The gate
checks that the candidate boot matches the TF1 measurement boot.

For an existing TF1 baseline, use the fresh September preparation command
with `prepare-panel` and the panel's `runs/oaf-ec9176f5c314b5b6/run.json`.
Run `run-panel` with the same options as `run` above plus the same reference
and timing manifest paths. This compares a sealed baseline, with
`panel_gate_passed` set to `false`.

Both commands report raw 50 ms F1, native drift, warm and startup-inclusive
RTF, and exact repeats for the two stress anchors. Gate approval requires
an exact TF1 smoke match, same-VM measurements, successful repeats, and
candidate micro-F1 at least as high as TF1.

## Reproduce the smoke comparison on x86-64 Linux

Run all commands from the repository root. On Colab, keep the notebook kernel
and the Crux Python 3.12 project environment unchanged: the Python 3.11 venv is
only the scratch worker/TensorFlow environment, and `uv run` is the existing
3.12 host driver. The workflow's manual `model` job runs this sequence; the
native Linux x86-64 confirmation run used the same scripts and lock with
Colab paths (`/content/oaf-tf215-*`).

### 1. Provision the Python 3.11 environment

```bash
uv python install 3.11
uv venv --python 3.11 /tmp/oaf-tf215-venv
export PY311=/tmp/oaf-tf215-venv/bin/python
uv pip sync --python "$PY311" --require-hashes \
  experiments/oaf_tf215/requirements.lock
uv pip check --python "$PY311"
```

### 2. Build the scratch runtime

```bash
SCRATCH=/tmp/oaf-tf215-scratch
"$PY311" experiments/oaf_tf215/build_runtime.py --output "$SCRATCH"
# -> /tmp/oaf-tf215-scratch/runtime
```

### 3. Execute the graph sessions

```bash
mkdir -p /tmp/numba-cache
CUDA_VISIBLE_DEVICES=-1 PYTHONHASHSEED=0 TF_ENABLE_ONEDNN_OPTS=0 \
NUMBA_CACHE_DIR=/tmp/numba-cache \
  "$PY311" experiments/oaf_tf215/graph_check.py \
  --output artifacts/experiments/oaf-tf215/graph-check.json
```

`PYTHONHASHSEED` is only read when the interpreter starts, so it must be set on
the process that launches `graph_check.py`; setting it inside the script has no
effect. `graph_check.py` validates all four values before importing TensorFlow
and fails the `environment` stage when they do not match.

### 4. Prepare the public checkpoint (network, no secrets)

```bash
uv run crux benchmark prepare-backend --backend oaf --download
# -> artifacts/benchmark/model-cache/sha256/<archive sha256>
```

This is the command the existing `oaf-smoke.yml` job uses. `compare.py` only
verifies the populated cache and never downloads; a missing or unavailable
cache is recorded as a `checkpoint` stage and blocked, not as an inference
result.

### 5. Run the full-model comparison

```bash
uv run experiments/oaf_tf215/compare.py \
  --python311 "$PY311" \
  --runtime-root "$SCRATCH/runtime" \
  --checkpoint-cache "$PWD/artifacts/benchmark/model-cache" \
  --fixture "$PWD/tests/fixtures/oaf_tf1_smoke/canonical.wav" \
  --oracle "$PWD/tests/fixtures/oaf_tf1_smoke/smoke-oracle.json" \
  --output artifacts/experiments/oaf-tf215/compare.json
```

The driver writes `compare.json` **before** it returns, then exits `0` only for
`smoke_exact`. Every other label exits `1`, including `oracle_divergent`, which
is repeatable but different from the frozen oracle. Preserve the record on a
divergent run instead of discarding it:

```bash
rc=0
uv run experiments/oaf_tf215/compare.py ... --output artifacts/experiments/oaf-tf215/compare.json \
  || rc=$?
echo "compare exit: $rc"
```

The recorded divergence from this candidate is quantified in `RESULTS.md`; a
nonzero exit is evidence, not a flaky failure to retry away.

## Regenerate the lock

Only after a successful solve, and only for the Linux x86-64 target:

```bash
uv pip compile --python-version 3.11 \
  --python-platform x86_64-manylinux_2_17 --generate-hashes \
  experiments/oaf_tf215/requirements.in \
  -o experiments/oaf_tf215/requirements.lock
```

Commit a candidate lock only when that command resolves. A solver failure is
recorded as the `solve` stage in the PR notes/logs, never committed as a partial
lock.

## Build the scratch runtime

`build_runtime.py` reads the frozen `runtime/oaf_tf1` tree and writes one new
scratch directory; it refuses an existing destination or a destination symlink
(including a dangling one) and returns the scratch `runtime/` path. It is safe
to build under a Git checkout: the compatibility `git apply` calls remove
inherited repository-selection variables (`GIT_DIR`, `GIT_WORK_TREE`, etc.) and
set `GIT_CEILING_DIRECTORIES` to the scratch root's parent, so patch paths are
never rebased against an enclosing repository and silently skipped.

```bash
SCRATCH=/tmp/oaf-tf215-scratch
"$PY311" experiments/oaf_tf215/build_runtime.py --output "$SCRATCH"
# -> /tmp/oaf-tf215-scratch/runtime
```

The build order is fixed and enforced:

1. copy `runtime/oaf_tf1/{model.py,model.json,worker.py}` into
   `runtime/` and the pristine vendored source into `upstream/`
   (`upstream/magenta/**` plus `upstream/LICENSE`);
2. `runtime/vendor/` **must not exist**; the existing
   `apply_instrumentation_patch.py` applies the reviewed emitted-frame patch
   to create it from `upstream/`;
3. `git apply --check` validates the compatibility patch preimages against the
   **instrumented** tree, then `git apply` applies the single
   `patches/tf215-compat.patch`.

The compatibility patch is the only source adaptation, and it is limited to the
drum path:

- `tensorflow.contrib.slim` / `tensorflow.contrib.layers` → `tf_slim` for the
  drum `model_tpu.py`, `estimator_spec_util.py` and `common/tf_utils.py`;
- `HParams` and `stack_bidirectional_dynamic_rnn` → the upstream forks pinned
  below;
- `tensorflow.contrib.tpu` / `cluster_resolver` → `tf.compat.v1.estimator.tpu`
  and `tf.distribute.cluster_resolver`, and `wait_for_new_checkpoint` →
  `tf_slim.evaluation`;
- the two Task 1 tf-slim controls are carried into the scratch model: the
  `framework_ops.Tensor` alias is restored in `runtime/model.py` and the drum
  graph calls `slim.batch_norm` with `fused=False,
  updates_collections=None`; `runtime/model.py` also calls
  `tf.compat.v1.disable_v2_behavior()` before any graph is built;
- unused package-initializer imports are removed so the environment does not
  install `tensorflow-probability`, `bokeh`, `IPython` or `pandas`:
  `magenta/__init__.py` exports only the version, `magenta/common/__init__.py`
  drops `.nade`, and `magenta/music/__init__.py` drops `notebook_utils`;
- `configs.py` drops the unused piano `model` import and the
  `CONFIG_MAP['onsets_frames']` entry; `CONFIG_MAP['drums']`,
  `spec_hop_length=512`, `hop_length=441` and the drum hparams stay unchanged.

`runtime/model.py` differs from the frozen file only by the two TF-compat
controls (alias plus `disable_v2_behavior()`) inside `_initialize_tf_compat()`,
the single call to that helper at the start of `load()`, and the two scratch
path constants (`MODEL_SCRATCH_DIRECTORY`,
`DEFAULT_UNINSTRUMENTED_SEQUENCE_LIB`). Coverage, stochastic-op and
preprocessing implementations are otherwise unchanged; the stdlib test removes
exactly those additions and compares the remaining ASTs.

`requirements.in` carries the frozen runtime's pins that the real drum import
needs and the Task 1 graph probe did not: `attrs==21.4.0`
(performance_lib), `mir_eval==0.8.2` (metrics), `sox==1.4.1`
(audio_transform) and `setuptools==68.0.0`. The 84.x setuptools that resolved
transitively no longer ships `pkg_resources`, which librosa 0.8.1 imports at
module load; the emulated Python 3.11 drum import failed with
`ModuleNotFoundError: No module named 'pkg_resources'` before the pin. The SoX
**binary** is not required for the import path or inference
(`transform_audio=False`); pysox prints a "SoX could not be found" warning at
import and `run_pipeline` would need the CLI only if audio augmentation were
enabled.

### Attribution

The copied helpers are selective source reuse under the upstream Apache-2.0
license, pinned to Magenta
`c15687ebd1c1cc12658cd468c737d3c24c02b918`:

| Scratch file | Upstream origin |
| --- | --- |
| `vendor/magenta/contrib/training.py` | `magenta/contrib/training.py` (verbatim; HParams fork) |
| `vendor/magenta/contrib/rnn.py` | `magenta/contrib/rnn.py` (`_hasattr`, `_is_sequence`, `stack_bidirectional_dynamic_rnn` only) |
| `vendor/magenta/contrib/__init__.py` | `magenta/contrib/__init__.py` |

The modified vendored files keep their original Apache-2.0 headers and the
upstream LICENSE is copied into the scratch `upstream/`. The patch is a
reviewed adaptation, not a claim that this tree is upstream Magenta.

## Graph check stages and exit status

`graph_check.py --output PATH` writes a compact JSON result on every path
(including failure) and prints it to stdout:

| Stage | What must actually happen |
| --- | --- |
| `environment` | All four process variables match, `PYTHONHASHSEED` was applied at interpreter startup, and `NUMBA_CACHE_DIR` is writable. |
| `import` | `tensorflow` imports, `tf.compat.v1.disable_v2_behavior()` is called, `tf_slim` imports; versions are recorded. |
| `slim_batchnorm` | A `tf_slim.conv2d` + `slim.batch_norm` graph runs in a `tf.compat.v1.Session` with its moving statistics actually updated; output and statistics are finite. |
| `legacy_lstm` | A `tf.compat.v1.nn.rnn_cell.LSTMCell` graph runs through `dynamic_rnn`; output and final state are finite. |

Exit 0 only when all four stages pass. Exit 1 on the first failure, with
`failed_stage` naming it and the error message recorded for that stage. Imports
alone, a constructed-but-unexecuted graph, or a skipped stage is a failure. No
checkpoint is downloaded or required by this check.

### TF 2.15 / tf-slim 1.1 compatibility controls

Executing `slim.batch_norm` under TensorFlow 2.15.1 needs two explicit controls;
`graph_check.py` records both in its import/slim stage details, and the scratch
runtime carries them in its single compatibility patch:

1. **`framework_ops.Tensor` alias.** TF 2.14 moved the class to
   `tensorflow.python.framework.tensor.Tensor` and removed the old
   `tensorflow.python.framework.ops.Tensor` name, which
   `tf_slim.layers.utils.constant_value` still reads. `graph_check.py` re-adds
   the alias before importing `tf_slim` (recorded as
   `tf_slim_tensor_alias_restored`); the scratch `runtime/model.py` re-adds it
   in `_initialize_tf_compat()` before it imports the vendored model.
2. **`fused=False, updates_collections=None`.** tf-slim 1.1's default
   `batch_norm` takes its Keras legacy-layer branch, which on TF 2.15 lazily
   imports `tf_keras.legacy_tf_layers`; there is no installable `tf_keras`
   package that provides that path, so no pin is added. These two arguments
   select tf-slim's own TF1 normalization implementation: the same
   `moving_mean`/`moving_variance` variables and the same inference math, with
   the moving-average update executed in the forward pass. The scratch
   `model_tpu.py` passes them through the `slim.batch_norm` arg scope.

Demonstrated failures from the locked environment (Python 3.11 / TF CPU 2.15.1 /
tf-slim 1.1.0), on Linux x86-64, recorded before the controls were added:

| Call | Result |
| --- | --- |
| `slim.batch_norm(..., is_training=True)` | `ModuleNotFoundError: No module named 'tf_keras'`. |
| the same call with `tf-keras==2.15.1` installed | `ModuleNotFoundError: No module named 'tf_keras.legacy_tf_layers'` — the published wheel ships `tf_keras/src/legacy_tf_layers`, not the top-level path TF 2.15 asks for. |
| the same call with `updates_collections=None` only | `AttributeError: module 'tensorflow.python.framework.ops' has no attribute 'Tensor'`. |

A `tf_keras` pin is therefore **not** part of `requirements.in`; the task brief
allows one only on a demonstrated fix, and none exists for this wheel set.


## CI

`.github/workflows/oaf-tf215-probe.yml` has a routine pull-request job and a
manual full-model job.

### PR graph job (checkpoint-free, runs on draft PRs)

The `graph` job runs on path-filtered pull requests
(`experiments/oaf_tf215/**`, the three scoped test modules and the workflow
itself), including draft PRs, unlike the root `ci.yml` which skips drafts. On
`ubuntu-24.04` it installs Python 3.11 with `uv python install 3.11`, creates
the venv with `uv venv --python 3.11`, installs the hash lock with
`uv pip sync --require-hashes` and runs `uv pip check`. The fresh runner
install *is* the clean-reinstall proof; do not repeat it in a second manual
venv.

The next step (`build`) runs the stdlib-only builder tests with a `pytest`
install added after the lock checks, executes the real `build_runtime.py`, and
then imports the generated drum configuration and the scratch worker in two
separate Python 3.11 interpreters. The drum import asserts
`spec_hop_length=512`, `hop_length=441`, `sample_rate=44100` and that the piano
config is gone; the worker import asserts the frozen backend ID. A failure at
any of those points is a `build` stage failure. No checkpoint is downloaded by
any of these steps.

The `driver` step then runs `tests/benchmark/test_oaf_tf215_compare.py` and
`tests/benchmark/test_oaf_tf215_workflow.py` in the Python 3.12 project
environment (fake worker only).

After the driver stage, `graph_check.py` runs with the four environment
controls. The compact `artifacts/experiments/oaf-tf215/graph-check.json` result
and job summary are kept on success and failure. Installation happens in two
named steps (`sync`, then `check`); graph failures name their own stage. If a
step fails before `graph_check.py` writes its record (setup, `sync`, `check`,
`build`, `driver`, or the graph process being killed), the always-running
result step writes a minimal failed JSON naming `environment`, `sync`, `check`,
`build`, `driver` or `graph` plus all step outcomes, so the artifact and
summary are never absent. Existing stage records are never overwritten, and
the passed record is never replaced.

### Manual full-model job (`workflow_dispatch`)

The `model` job is gated by
`if: github.event_name == 'workflow_dispatch'`, so a pull request never
downloads the checkpoint. It uses the same hash lock and `build_runtime.py`
builder, prepares the public checkpoint with the identical
`uv run crux benchmark prepare-backend --backend oaf --download` command from
`oaf-smoke.yml` (no secrets), and runs the Python 3.12 `compare.py` driver
against the Python 3.11 scratch worker.

The compare step captures the driver's exit status with
`... | tee compare.log || rc=$?`, records `COMPARE_EXIT` and then re-exits with
that status. `compare.py` writes `compare.json` before returning, so a
divergent run keeps its full record and log while the job stays red; the
always-on summary and upload steps run afterwards. `oracle_divergent` is never
turned into a pass. The model job uploads `oaf-tf215-compare` (`compare.json`,
`compare.log`); the PR graph job uploads `oaf-tf215-graph-check`.

### Dispatch and executed model job

An earlier preflight predicted that `workflow_dispatch` would be unavailable
before merge because the workflow file is not on `main`. That prediction was
wrong: the dispatch

```bash
gh workflow run oaf-tf215-probe.yml \
  --ref docs/hpa-856-python311-tf215-prototype
```

was accepted and created run `43963725603` at `c3ffbef`. The graph job passed.
The model job intentionally failed at the comparison step after writing
`compare.json` with `repeatable=true`, `oracle_exact=false` and
`status=oracle_divergent`, and after setting `COMPARE_EXIT=1`; the summary and
upload steps then succeeded and attached `oaf-tf215-compare` and
`oaf-tf215-graph-check`. The red job is the parity gate, not a pass. The same
command sequence was also executed independently on a fresh native Linux x86-64
Colab CPU VM; `RESULTS.md` records both hosts, interpreter builds, hashes and
the actual graph/restore/inference output.

`check_workflow_result_fallback.py` is the focused local check for the graph
job's fallback: it extracts the step's shell block from the workflow and
exercises the sync, check, build, driver, graph, setup, existing-failure and
passed paths. `tests/benchmark/test_oaf_tf215_workflow.py` checks both jobs:
the graph job stays checkpoint-free, the model job uses the same public
checkpoint command as the smoke workflow with no secrets, and a divergent
driver exit still leaves and uploads its record.

## What this does and does not prove

Proves on Linux x86-64 (GitHub Actions PR and manual model jobs, and a fresh
native Colab CPU VM, recorded with commit and host in `RESULTS.md`): the
candidate hash lock resolves and installs, TensorFlow 2.15 executes the
TF1-compatibility Slim batch-norm and legacy LSTM graphs under the configured
controls, the generated drum configuration and scratch worker import under
Python 3.11 without the unused notebook/NADE packages, the public checkpoint
cache verifies, the frozen 130/78/52 partition checks run before the worker
reports ready with 78 restored inference tensors, and three inferences (two
same-worker, one fresh-worker) are repeatable within each host.

Does not prove: equality with the frozen TF1 oracle. The recorded runs on both
hosts are repeatable but `oracle_divergent` (see `RESULTS.md` for the exact
event deltas), so no TF1 parity, cache/identity reuse or production equivalence
is claimed. The two hosts are not bit-identical to each other (event 0
confidence differs), so cross-host equality is not claimed either. It also does
not establish behaviour on TF1, a broad corpus, or any downstream production
path. The builder itself is verified locally by the stdlib tests; the
TensorFlow imports of the scratch tree are proven only on Linux x86-64 (GitHub
or native Colab), not by a local aarch64 interpreter.

## Opt-in corpus benchmark

Run from the repository checkout on Linux x86-64. Set `WORKER` and `SCRATCH`
to fresh directories, and `CHECKPOINT_CACHE` to a local checkpoint cache.
Prepare the locked project and worker environments.

```bash
uv sync --locked --python 3.12 --no-dev
uv venv --python 3.11 "$WORKER"
uv pip sync --python "$WORKER/bin/python" --require-hashes experiments/oaf_tf215/requirements.lock
uv pip check --python "$WORKER/bin/python"
"$WORKER/bin/python" experiments/oaf_tf215/build_runtime.py --output "$SCRATCH"
mkdir -p /tmp/numba-cache
uv run --no-sync crux benchmark prepare-backend --backend oaf --download --cache-root "$CHECKPOINT_CACHE"
```

Set `REF` and `TIMING` to the September manifests and `CACHE` to the indexed
audio cache. The timing root must include all 180 eligible event bodies. The
verified digests and two-song scope are in the [implementation plan](../../docs/superpowers/plans/2026-11-25-hpa-856-opt-in-corpus-backend.md#task-4--prove-the-normal-cli-on-colab-and-record-results).

```bash
run_pilot=(uv run --no-sync crux benchmark run-oaf-corpus
  --backend oaf-tf215 --python311 "$WORKER/bin/python"
  --runtime-root "$SCRATCH/runtime" --checkpoint-cache "$CHECKPOINT_CACHE"
  --manifest "$REF" --timing-manifest "$TIMING" --cache-dir "$CACHE"
  --output-dir "$OUT" --include-simfile-id 200 --include-simfile-id 209)
"${run_pilot[@]}"
```

Start with a fresh output directory. Save the first `run.json`, reports and
CLI summary, then append `--resume` to the same command. Keep the worker assets
fixed throughout the two invocations. The run snapshot records the preflight
runtime identity and the inference configuration includes its SHA-256.
[Pilot results](RESULTS.md#opt-in-corpus-pilot)
