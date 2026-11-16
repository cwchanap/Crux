# HPA-856 — Python 3.11 / TensorFlow 2.15 Prototype Implementation Plan

> **For agentic workers:** Use `superpowers:executing-plans` (or subagent-driven development when available) to execute these tasks in order. Steps use checkbox syntax for tracking. This is a bounded experiment, not authorization for production integration.

**Goal:** Reproduce installation, graph execution, checkpoint restoration and native-event comparison for the existing OaF drum model under Python 3.11 / TF CPU 2.15.1.
**Architecture:** One isolated experiment directory builds a patched scratch runtime from the frozen TF1 source and selectively reused upstream compatibility code. A standalone probe emits diagnostic evidence only. Keep production runners, locks, schemas and goldens unchanged until the evidence-based integration gate.
**Tech stack:** Python 3.11, TensorFlow CPU 2.15.1, TF1 compatibility graph mode, Estimator 2.15, tf-slim, existing NumPy/Librosa frontend, pytest.
**Spec:** [2026-11-11-hpa-856-colab-native-oaf-design.md](../specs/2026-11-11-hpa-856-colab-native-oaf-design.md).
**Delivery:** HPA-856 / [PR #35](https://github.com/cwchanap/Crux/pull/35), **one ticket / one PR**. Multiple commits are fine; no second implementation PR. Replace the previous 3.7 retry plan with this sequence.
**Status:** Planning only. Source research and stdlib comparator checks exist; none of the TF-dependent gates below has executed successfully.

## Global constraints

- Target Python **3.11**, `tensorflow-cpu==2.15.1`; record the exact interpreter build after successful provisioning. Do not change notebook Python or Crux's Python 3.12 project environment.
- Preserve `runtime/oaf_tf1/`, `src/`, original checkpoint bytes, emitted-frame instrumentation and oracle; perform adaptations only in an experimental scratch copy.
- Source compatibility reference: Magenta `c15687ebd1c1cc12658cd468c737d3c24c02b918`. Selective source reuse with licenses, no wholesale Magenta install or neural-network rewrite.
- Set `CUDA_VISIBLE_DEVICES=-1`, `PYTHONHASHSEED=0`, `TF_ENABLE_ONEDNN_OPTS=0` before TF import; enable TF1 compatibility before constructing objects. Numerical equality remains a measured result.
- Separate experiment artifacts from all production prediction/run/report caches. No new CLI flags, `CRUX_OAF_NATIVE_PYTHON` wiring, shared-helper refactor, checkpoint conversion, R2 corpus run, separator install, MLflow change, or notebook UI in the immediate scope.

## Review focus

1. Package install/import can succeed while Slim normalization or legacy RNN graph execution fails: Task 1 executes both graphs.
2. Compatibility patches can change graph scopes or preprocessing despite a loadable checkpoint: Tasks 2–3 preserve the frontend and validate required tensor names/shapes/dtypes, not merely counts.
3. Exact time/confidence bits can differ while note counts/F1 match: Task 4 compares the entire native event sequence and repeat runs.
4. A partial probe can look like a pass or write into a canonical cache: Tasks 1 and 4 distinguish unexecuted stages and reject canonical output destinations.
5. A successful first environment can rely on ambient packages/files: Task 5 repeats from the lock and a fresh bundle, and verifies no tracked TF1 source changed.

## Files and ownership

| Planned file | Responsibility |
| --- | --- |
| `experiments/oaf_tf215/requirements.in` | Documented starting constraints and justified compatibility dependencies. |
| `experiments/oaf_tf215/requirements.lock` | Generated exact, hashed dependency solution for the isolated target; create after a real successful solve. |
| `experiments/oaf_tf215/build_runtime.py` | Fresh scratch layout and ordered instrumentation/compatibility patch application. |
| `experiments/oaf_tf215/patches/tf215-compat.patch` | Selected upstream adaptations and required support-file additions, with original licensing. |
| `experiments/oaf_tf215/probe.py` | Version/graph/input/restore/inference/repeatability/oracle stages and diagnostic JSON. |
| `experiments/oaf_tf215/README.md`, `RESULTS.md` | Reproduction commands/source attribution and measured outcome with remaining blockers. |
| `tests/runtime/test_oaf_tf215_probe.py` | Small fixture-based builder, input-validation and result-classification regressions. |
| `.github/workflows/oaf-tf215-probe.yml` | Path-filtered Python 3.11 environment/graph CI; no checkpoint, R2 or hosted-service credentials. |

Reuse the conversation investigation's `probe_tf215.py` and candidate requirements as starting material, not finished code or a lock. Commit the adapted source in Task 1; later execution must not depend on an unavailable ZIP attachment. Avoid a second framework or generic experiment runner.

## Task 1 — Establish the isolated environment and executable graph gate

**Files:** create `requirements.in`, `probe.py`, `README.md`, and focused tests; generate `requirements.lock` only after validation.
**Interface:** `probe.py --output PATH` runs environment-only checks; emits `environment_only_passed` only after actual graph execution, with `inference_completed`, `repeatable`, and `oracle_exact` null. Return nonzero with the failed stage on a genuine error.

- [ ] Bring the diagnostic probe into the experiment directory. Add a focused test proving an environment-only success does not set a model-parity result, and a forced Slim failure records its stage and nonzero exit. Run the targeted test and confirm the missing behavior fails first; then implement only that behavior.
- [ ] Seed `requirements.in` with TF CPU 2.15.1, Estimator 2.15.0, Keras 2.15.0, tf-slim 1.1.0, protobuf 3.20.3, NumPy 1.23.5, Librosa 0.8.1, Numba 0.57.1, llvmlite 0.40.1, SciPy 1.10.1 and the supplied audio/MIDI dependencies. Label it unvalidated until solved. Resolve it in a dedicated Python 3.11 environment; run `uv pip check --python "$PY311"`.
- [ ] Execute version validation, `disable_v2_behavior()`, Estimator availability, a real Slim Conv2D + batch-normalization session and a legacy LSTM session. Preserve stage-specific exceptions. Diagnose any `tf_keras.legacy_tf_layers` failure before adding a dependency; pin/document only a demonstrated fix. Do not add a fake global `tensorflow.contrib` module.
- [ ] Generate the exact hash lock with `uv pip compile --python "$PY311" --generate-hashes experiments/oaf_tf215/requirements.in -o experiments/oaf_tf215/requirements.lock`; install into a second clean environment with `uv pip sync --python "$CLEAN_PY311" --require-hashes experiments/oaf_tf215/requirements.lock` and repeat the graph probe. If dependency resolution or graph execution fails, capture the blocker instead of labelling the lock verified.
- [ ] Commit the probe, validated lock (when available), tests and reproduction notes on this PR. No model checkpoint needed at this gate.

**Verify:** `PYTHONHASHSEED=0 "$CLEAN_PY311" experiments/oaf_tf215/probe.py --output "$PWD/artifacts/experiments/oaf-tf215/environment.json"` exits 0 and records both executed graph checks; `uv pip check --python "$CLEAN_PY311"` passes. A no-network host remains blocked, not incompatible.

## Task 2 — Build the smallest adapted drum runtime in scratch

**Files:** `build_runtime.py`, `patches/tf215-compat.patch`, tests and README attribution.
**Interface:** `build_runtime(repo_root: Path, output_root: Path) -> Path`; CLI `--repo-root PATH --output-root PATH`; returns `<output_root>/runtime`. Refuse an existing destination; callers choose a fresh scratch root.

- [ ] Add a real-builder test using the retained vendor source. It must prove the original tree/model lock are unchanged, the scratch `upstream` sequence converter remains pristine, `runtime/vendor` is absent when instrumentation starts, and the completed bundle has patched decoding plus the intended compatibility imports. Confirm the test fails before the builder exists.
- [ ] Copy retained model/worker/config and pristine vendor into scratch. Create `runtime/` and `upstream/`; call `apply_reviewed_patch()` with the **uncreated** `runtime/vendor` destination. Keep the original instrumentation patch and its preimage checks unchanged.
- [ ] Create one compatibility patch against this generated layout. Selectively adapt `tf_slim`, HParams/RNN support and `compat.v1.estimator.tpu` using the pinned upstream revision. Include needed upstream support files/license notices in the patch rather than fetching moving source at probe runtime. Record source paths/revision in README.
- [ ] Restrict config imports to what drums inference needs, avoiding an unnecessary piano/CuDNN port. Preserve `configs.CONFIG_MAP['drums']`, all effective hparams, variable scopes, `spec_hop_length` and the existing Librosa frontend. In the copied model only, make the pristine-converter path relative to its bundle and initialize TF1 compatibility before model imports. Leave weights/thresholds/native decoding untouched.
- [ ] Run `git apply --check` before applying the compatibility patch, then import the generated drum configuration under the validated target interpreter. A patch/import failure stops here and produces a diagnostic; do not modify the frozen source to make the patch apply. Rerun builder tests and commit.

**Verify:** target-interpreter import succeeds, real builder test passes, and both `git diff --exit-code f5e0d182 HEAD -- runtime/oaf_tf1 src` and the working-tree diff for those paths are empty. This is not yet checkpoint evidence.

## Task 3 — Verify original inputs and restore every required tensor

**Files:** extend `probe.py` and tests; update README commands.
**Interface:** model mode requires all of `--runtime-root PATH --checkpoint-dir PATH --fixture PATH --oracle PATH --output PATH`. Explicit absolute inputs; no production smoke-command or cache-selector changes.

- [ ] Add tests rejecting a wrong fixture digest and a modified/missing checkpoint component **before** model construction. Use tiny synthetic files/mocks; do not commit checkpoint bytes. Add a test that a mismatched required tensor name/shape/dtype cannot be accepted merely because a reported count is 78.
- [ ] Obtain the existing released checkpoint through the existing acquisition procedure or a previously verified cache. Read component hashes from the frozen `runtime/oaf_tf1/model.json`; never redownload/rewrite weights silently. Validate the canonical WAV against the oracle input SHA.
- [ ] Build the actual drum graph and call the copied `OafModel.load()` with the original configuration. Reuse its existing coverage logic: 130 checkpoint tensors, 78 required inference tensors and 52 classified non-inference tensors, with names/shapes/dtypes and no uninitialized required tensor. Preserve stochastic-op checks. Emit coverage details, not just a success string.
- [ ] Record distinct graph-build, input-integrity and checkpoint-restore stages. If variable names/shapes differ, inspect the selected adaptation; do not rename weights, initialize missing tensors, or lower the coverage requirement. Rerun focused tests and commit the probe changes.

**Verify:** a real model-mode run reaches `checkpoint_restore=passed` with complete coverage. A synthetic corrupt input fails before restore. No R2/full-song dataset or separator is needed.

## Task 4 — Measure repeatability and exact native-event parity

**Files:** extend `probe.py` and focused tests.
**Interfaces:** retain the starting `native_payload(event)` / `compare_events(events, expected)` helpers, matching `src/benchmark/oaf_smoke_oracle.py` exactly for every retained field. Model outcomes are `failed`, `non_repeatable`, `runs_divergent`, or `smoke_exact`; environment-only mode stays separate.

- [ ] Pin comparator behavior with tests: exact match passes; one-ULP confidence change, event removal, reordered distinct events, changed timestamp/class/group/pitch/velocity fail. Keep null confidence distinct from numeric zero. Match the production binary64 representation rather than rounded decimal/F1 comparison.
- [ ] Transcribe the canonical WAV twice using one loaded model, then repeat once in a **fresh interpreter process** using the same lock/bundle/inputs. Preserve the complete native event lists. Do not change model math or thresholds between runs.
- [ ] Compare candidate-to-candidate equality first, then each candidate to the original oracle. Store `inference_completed`, `repeatable`, `oracle_exact`, actual/expected counts, complete output digests and at most the first 20 field differences in summary JSON; retain full events separately. Classify non-repeatability independently from deterministic TF1 differences.
- [ ] Use a separate experiment-output root and a simple guard against writes to fixture/oracle, checkpoint, frozen runtime or production benchmark-cache locations. Do not publish a production prediction/report. Record the old backend ID only as `baseline_backend_id`, with an explicit experimental candidate label and dependency/source hashes.
- [ ] Add a forced divergent-result test showing diagnostics are saved and the process exits nonzero. Exit 0 for model mode requires repeatable exact smoke, not merely inference completion. Record repeatable divergence as useful experimental evidence without calling it parity. Run focused tests and commit.

**Verify:** every run produces a stage/result record; repeatability and exactness can be independently audited. No oracle, model lock, default backend or old prediction cache changes.

## Task 5 — Reproduce in CI and on the target Colab interpreter

**Files:** `.github/workflows/oaf-tf215-probe.yml`, README and RESULTS; tests from prior tasks.

- [ ] Add a path-filtered PR CI job for this experiment/workflow/tests, with Python 3.11, the candidate lock, dependency cache and ordinary bounded job timeout. Run focused fixture tests plus the real environment-only graph probe; upload the diagnostic on success/failure. Do not install TF2.15 into the root Python 3.12 job. No R2, checkpoint download or DagsHub secrets in automatic CI; do not claim a skipped draft job passed.
- [ ] In an authenticated Colab CPU runtime, create a dedicated Python 3.11 environment from the lock and run the **same** builder/probe CLI through that interpreter, leaving the notebook kernel alone. Record the exact commit, interpreter/package versions, OS/architecture, source/lock hashes, checkpoint/fixture hashes and numerical environment flags. Use README shell cells, not a new custom notebook/launcher UI.
- [ ] Repeat environment + restore + three-inference comparison from a clean bundle. Save small diagnostic JSON/full-event evidence privately and a concise redacted result summary in `RESULTS.md`. Missing authentication/network/checkpoint is explicitly blocked, not a fabricated graph or compatibility result.
- [ ] Run `uv run pytest tests/runtime/test_oaf_tf215_probe.py -q`, the existing regression suite, `uv run ruff check .`, `uv run ruff format --check src tests`, and `uv run pylint --errors-only src`. Target TF-specific checks execute under `$PY311`, not `uv run` in the root project. Verify the frozen source/weights/oracle are unchanged and no large binaries or credentials enter the PR.

**Acceptance:** CI executes the graph checks, and actual target-Colab evidence separately reaches restore/inference with an honest exact/divergent/non-repeatable classification. There is no separate Linux/Android desktop acceptance; no iOS/macOS/Windows application or UI mockup is changed by this phase.

## Task 6 — Record the outcome and define the continuation on the same PR

**Files:** `experiments/oaf_tf215/RESULTS.md` and these existing spec/plan files; update the PR/Linear discussion.

- [ ] Summarize exact commands, environment, each completed/failed stage, coverage, repeatability and per-field differences. Distinguish “prototype produced an answer” from “migrated runtime is accepted.” Keep unsuccessful attempts concise and reproducible.
- [ ] Apply the spec's decision table: unresolved environment/restore/non-repeatability -> stop; repeatable divergence -> describe a separately versioned runtime and quantified deltas; exact smoke -> selected-song diagnostics before broader equivalence claims. Do not auto-promote either successful path to the default backend.
- [ ] Before any production/corpus continuation, revise **this same PR's** integration contract using the measured outcome: candidate identity/config/hash inputs, separate cache/output roots, one runtime selection seam and mapping/scorer reuse. A framework/source change must remain visible in provenance even when one oracle matches. Do not implement two possible production adapters in advance.
- [ ] Preserve the original HPA-856 follow-on scope as gated work: reference identity, broad corpus and per-song/per-class comparison, existing reviewed inputs, same-machine separation, fixed v2 separator companions/pilot commit, Drive resume and existing MLflow publisher. HPA-856 remains open until that scope is delivered or explicitly changed; no second PR is planned.

**Stop boundary:** planning this prototype does not authorize model-training/quality-tolerance changes or completion claims for the full Colab pipeline. The next executable action is Task 1's isolated Python 3.11 dependency solve and real graph probe, not another Python 3.7 retry.
