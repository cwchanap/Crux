# OaF TensorFlow 2.15 experiment (Python 3.11)

Bounded compatibility experiment for HPA-856. It asks one question: can the
existing OaF drum checkpoint install, execute its graph, restore and transcribe
under Python 3.11 / TensorFlow CPU 2.15.1, and how does that output compare with
the frozen TF1 oracle?

**This is an experiment, not production integration.** `runtime/oaf_tf1/`,
`src/`, the root `pyproject.toml`/`uv.lock`, the frozen checkpoint, its
instrumentation patch and the canonical smoke oracle stay untouched. Nothing
here is wired into Crux's CLI, backend factory or caches, and a successful graph
check does not by itself establish model parity.

## Files

| Path | Purpose |
| --- | --- |
| `requirements.in` | Candidate pins (TF CPU 2.15.1, Estimator/Keras 2.15, tf-slim, and the frozen numerical/audio/MIDI stack). |
| `requirements.lock` | Hashed lock resolved for the frozen runtime's target: Python 3.11, `x86_64-manylinux_2_17`. |
| `graph_check.py` | Executes real Slim Conv2D + batch-normalization and legacy LSTM sessions and records stage-specific diagnostics. |
| `check_workflow_result_fallback.py` | Stdlib-only local check: the workflow's always-on result step writes a failed sync/check/graph/environment record when no graph output exists. |
| `README.md` | This file. Reproduction commands and evidence scope. |

`build_runtime.py`, `compare.py`, the compatibility patch and `RESULTS.md` land
in later tasks of the same plan and PR.

## Target platform

The first execution host is GitHub Actions `ubuntu-24.04`, Linux **x86-64**,
which matches the frozen TF1 Docker image's `linux/amd64` target. Do not resolve
or run this lock on macOS/ARM64 and call it evidence: `tensorflow-cpu==2.15.1`
has no macOS ARM64 wheel, and the lock is compiled explicitly for
`x86_64-manylinux_2_17`.

## Reproduce locally / on Colab (x86-64 Linux only)

```bash
uv python install 3.11
uv venv --python 3.11 /tmp/oaf-tf215-venv
uv pip sync --python /tmp/oaf-tf215-venv/bin/python --require-hashes \
  experiments/oaf_tf215/requirements.lock
uv pip check --python /tmp/oaf-tf215-venv/bin/python

mkdir -p /tmp/numba-cache
CUDA_VISIBLE_DEVICES=-1 PYTHONHASHSEED=0 TF_ENABLE_ONEDNN_OPTS=0 \
NUMBA_CACHE_DIR=/tmp/numba-cache \
  /tmp/oaf-tf215-venv/bin/python experiments/oaf_tf215/graph_check.py \
  --output artifacts/experiments/oaf-tf215/graph-check.json
```

`PYTHONHASHSEED` is only read when the interpreter starts, so it must be set on
the process that launches `graph_check.py`; setting it inside the script has no
effect. `graph_check.py` validates all four values before importing TensorFlow
and fails the `environment` stage when they do not match.

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
both are local to this probe and recorded in the import/slim stage details:

1. **`framework_ops.Tensor` alias.** TF 2.14 moved the class to
   `tensorflow.python.framework.tensor.Tensor` and removed the old
   `tensorflow.python.framework.ops.Tensor` name, which
   `tf_slim.layers.utils.constant_value` still reads. `graph_check.py` re-adds
   the alias before importing `tf_slim` (recorded as
   `tf_slim_tensor_alias_restored`). The scratch runtime needs the same line in
   its compatibility patch.
2. **`fused=False, updates_collections=None`.** tf-slim 1.1's default
   `batch_norm` takes its Keras legacy-layer branch, which on TF 2.15 lazily
   imports `tf_keras.legacy_tf_layers`; there is no installable `tf_keras`
   package that provides that path, so no pin is added. These two arguments
   select tf-slim's own TF1 normalization implementation: the same
   `moving_mean`/`moving_variance` variables and the same inference math, with
   the moving-average update executed in the forward pass.

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

`.github/workflows/oaf-tf215-probe.yml` runs on path-filtered pull requests
(`experiments/oaf_tf215/**` and the workflow itself), including draft PRs,
unlike the root `ci.yml` which skips drafts. On `ubuntu-24.04` it installs
Python 3.11 with `uv python install 3.11`, creates the venv with
`uv venv --python 3.11`, installs the hash lock with
`uv pip sync --require-hashes`, runs `uv pip check`, and launches
`graph_check.py` with the four environment controls. The fresh runner install
*is* the clean-reinstall proof; do not repeat it in a second manual venv.

The compact `artifacts/experiments/oaf-tf215/graph-check.json` result and job
summary are kept on success and failure. Installation happens in two named
steps (`sync`, then `check`); graph failures name their own stage. If a step
fails before `graph_check.py` writes its record (setup, `sync`, `check`, or the
graph process being killed), the always-running result step writes a minimal
failed JSON naming `environment`, `sync`, `check` or `graph` plus all step
outcomes, so the artifact and summary are never absent. Existing stage records
are never overwritten, and the passed record is never replaced. The routine PR
job is checkpoint-free. A manual full-model `workflow_dispatch` job is added by
a later task, not here.

`check_workflow_result_fallback.py` is the focused local check for that
fallback: it extracts the step's shell block from the workflow and exercises
the sync, check, graph, setup, existing-failure and passed paths.

## What this does and does not prove

Proves: the candidate hash lock resolves and installs on a fresh Linux x86-64
runner, and TensorFlow 2.15 can execute a TF1-compatibility Slim batch-norm
graph and a legacy LSTM graph under the configured controls.

Does not prove: checkpoint restore, 130/78/52 tensor coverage, inference,
repeatability, or equality with the frozen TF1 oracle. Those are separate,
later gate results.
