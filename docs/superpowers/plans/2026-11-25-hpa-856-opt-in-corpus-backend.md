# HPA-856 Opt-in TF2.15 Corpus Backend Implementation Plan

> **For agentic workers** Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task by task. Track steps with checkboxes.

**Goal** Run and resume the existing OaF corpus benchmark with a separately identified Docker-free TF2.15 worker.

**Architecture** Register the measured candidate with the existing prediction and scoring contracts. Reuse the OaF adapter and corpus loop, with an explicit native launch path and a runtime-bound candidate inference configuration. Verify both execution and resume through the normal CLI on Colab.

**Tech stack** Python 3.12 host, isolated Python 3.11 with `tensorflow-cpu==2.15.1`, Click, pytest, canonical JSON and the installed Colab CLI.

**Spec** [Opt-in corpus backend design](../specs/2026-11-25-hpa-856-opt-in-corpus-backend-design.md)

## Global constraints

- Deliver the milestone in existing [draft PR #36](https://github.com/cwchanap/Crux/pull/36). The local twenty-song evidence base is `fec40eb`. Preserve unrelated worktree changes.
- The default selection is `oaf`. Preserve its Docker command, model, smoke oracle, descriptor, map, inference-config v1 bytes, prediction paths, and historical resume behavior.
- Candidate backend ID is `magenta-egmd-tf215-94529798-8hit-experimental-v1`. Reuse the original checkpoint, frozen worker source and 8-hit classification semantics. The candidate publishes its own descriptor and map.
- Provision Linux x86-64 CPU with the existing Python 3.11 hash lock and scratch builder. Use the separate Python 3.12 project environment. Keep TensorFlow imports inside the worker.
- Candidate lookup uses the full runtime-bound config digest. Both backends use the shared resume path in a runner-owned, single-writer output root. The pilot holds the worker environment, scratch tree and checkpoint fixed during each invocation.
- The app backend, broad corpus, reviewed-subset execution, separation, cross-VM Drive resume and MLflow publication are later milestones.
- Use the installed Colab CLI to start a separate Linux x86-64 CPU test session before Task 1. Transfer each task's source snapshot, including file removals, and record the tested commit or overlay digest. Prepare the locked Python 3.12 project environment with `uv sync --locked --extra mlflow --extra r2`, then run `PRELOAD_MODEL=0 uv run --no-sync pytest -q -o addopts=`. Run targeted tests for the red/green cycle and the full suite before each task commit and before building the separate pilot bundle. Retain the suite logs and stop the test session once the pre-bundle gate is complete.

## Review focus

1. A frozen TF1 wire-ready ID must never cause TF2.15 artifacts to be labeled TF1, and a native descriptor cannot enter the Docker launch path. Tasks 1 and 2 pin both boundaries.
2. Empty candidate predictions still have a map identity. Rounded event-key collisions fail the affected row without losing the diagnostic. Tasks 1 and 3 exercise both.
3. A changed runtime at preflight selects a different run and prediction path. A changed artifact or canonical WAV must fail the existing resume checks. Task 3 covers these cases.
4. Probe and launch must use the same venv executable and captured environment. Task 2 exercises the symlink case and the stdlib-only child probe.
5. A candidate parent must fail frozen separation preflight. Task 1 covers the guard, and Task 4 verifies that ordinary readers accept the corpus artifacts.

## File structure

| Files | Responsibility |
| --- | --- |
| `src/benchmark/{backend_identity,taxonomy,prediction_artifact,cohort_scoring}.py` | Candidate identity, map, canonical event validation and zero-hit scoring. |
| `src/benchmark/separation_pilot.py` | Explicit frozen parent validation at the existing separation handoff. |
| Create `src/benchmark/oaf_tf215_runtime.py` | Shared probe, runtime fingerprint, checkpoint verification and native worker environment. |
| `src/benchmark/backends/oaf.py` | Select the subprocess launch command while reusing the adapter lifecycle. |
| Move `experiments/oaf_tf215/runtime_identity.py` and update `experiments/oaf_tf215/{compare,selected_compare,panel_metrics}.py` | Point experiment callers and source inventories at the shared module. |
| `.github/workflows/oaf-tf215-probe.yml` | Keep the existing probe workflow triggered by changes to the moved module. |
| `src/benchmark/oaf_corpus_run.py`, `src/cli/benchmark.py` | Opt-in selection, runtime-bound run identity, shared resume, reports and CLI diagnostics. |
| `tests/benchmark/`, `tests/test_cli_benchmark.py` | Focused contract and regression tests named below. |
| `experiments/oaf_tf215/{README,RESULTS}.md` | Reproduction instructions and actual pilot evidence. |

### Task 1 — Register candidate artifacts and protect the separation handoff

**Interfaces** Export `OAF_TF215_BACKEND_ID` from `backend_identity.py` and `OAF_TF215_PREDICTION_MAP_ID = "crux.prediction-map/oaf-egmd-tf215-8hit-v1"` and `OAF_TF215_PREDICTION_MAP: PredictionMap` from `taxonomy.py`. Callers use the existing `map_oaf_prediction(native, prediction_map=OAF_TF215_PREDICTION_MAP)`, `render_prediction_artifact` and `read_prediction_artifact`.

- [x] In `tests/benchmark/test_oaf_tf215_runtime_identity.py::test_candidate_changes_only_backend_identity`, assert candidate descriptor SHA-256 `938cbcccff5adda03f11344596e21534e178c98929098bf8327feb1e9ed4c890` and expect acceptance by the known-backend validator. Retain unknown-family coverage in `tests/benchmark/test_backend_identity.py`.
- [x] Use the existing OaF fixture in `tests/benchmark/test_prediction_artifact_coverage.py` to map and round-trip candidate MIDI 38, model output bin 17 and group `snare`, and assert the candidate descriptor and map IDs survive. Reject a bad bin/note pair and verify the existing `duplicate_native_event` error when two events normalize to the same key. Run the existing TF1 golden assertions.
- [x] Add `test_zero_hit_tf215_artifact_uses_candidate_map` to `tests/benchmark/test_cohort_scoring.py`. An empty successful artifact with one mapped reference event must receive the candidate map and score TP/FP/FN of 0/0/1. Add `test_separation_rejects_tf215_parent_descriptor` to `tests/benchmark/test_separation_pilot.py`. Call `_validate_parent_identity` with a candidate descriptor and correct descriptor hash. Assert that the TF1 parent check rejects it before inference-config validation.
- [x] Run the focused tests below and confirm failures are limited to missing candidate support.
- [x] Register the candidate in `backend_identity._DESCRIPTOR_POLICIES` with the OaF v2 fields and same pinned model identity. Create the distinct `PredictionMap` in `taxonomy.py` with the existing read-only class mapping. In `prediction_artifact._normalize_event` (the `OAF_BACKEND_ID` branch), apply the existing OaF native rules to both IDs. Keep map identity checks at mapping, resume and cohort scoring. In `cohort_scoring.ZERO_HIT_PREDICTION_MAPS`, add the candidate entry. In `separation_pilot._validate_parent_identity`, require the normalized parent descriptor's ID to be `OAF_BACKEND_ID` before config processing or runtime setup. Require the frozen TF1 inference schema in its existing config validation.
- [x] Run the focused pytest modules `test_oaf_tf215_runtime_identity.py`, `test_backend_identity.py`, `test_mapping.py`, `test_prediction_artifact.py`, `test_prediction_artifact_coverage.py`, `test_cohort_scoring.py`, and `test_separation_pilot.py` under `tests/benchmark/`. Expect candidate round-trips, early separation rejection, and the existing frozen tests to pass.
- [x] Run the full-suite gate from Global constraints. This includes the prediction-artifact coverage, separation handoff, reviewed-subset, IDM pilot and cross-comparison tests.
- [x] Commit the artifact, map and separation changes with their tests as `feat(benchmark): recognize TF2.15 OaF artifacts`.

### Task 2 — Share runtime verification and add the native launcher

**Interfaces** `oaf_tf215_runtime.py` produces:

```python
@dataclass(frozen=True)
class OafTf215Runtime:
    python311: Path
    runtime_root: Path
    checkpoint_dir: Path
    worker_environment: Mapping[str, str]
    descriptor: BackendDescriptor
    identity: dict[str, JsonValue]
    sha256: str

def prepare_tf215_runtime(
    repo_root: Path,
    *,
    python311: Path,
    runtime_root: Path,
    checkpoint_cache: Path,
) -> OafTf215Runtime: ...
```

`backends/oaf.py` produces `create_tf215_backend(runtime: OafTf215Runtime, *, input_root: Path, process_factory: ProcessFactory = WorkerProcess.start, timeout_seconds: float = 30.0, close_timeout_seconds: float = 30.0) -> OafBackend`. It exposes the candidate descriptor and verifies the existing TF1 wire ready ID.

- [x] Extend `tests/benchmark/test_oaf_tf215_runtime_identity.py` to exercise the shared module. Assert that a blocked, malformed or timed-out probe, incorrect TF version, missing or symlinked `worker.py`, or changed scratch model config prevents preparation. Change a scratch worker file or a CPU field and assert the preflight digest changes. Edit a host CLI file and assert the worker-only digest is stable. Assert that the probe and launcher retain the absolute **venv interpreter path**, even when it is a symlink, and use the same captured worker environment. Keep the stdlib-only direct `--probe` subprocess test.
- [x] Add native-launch cases to `tests/benchmark/test_task_d_contract.py`. With a fake process factory assert this exact command and the resolved input-root working directory:

  ```text
  [python311, "-s", runtime_root / "worker.py", "--checkpoint-dir", checkpoint_dir]
  ```

  Assert two requests use one process, both return the candidate descriptor, and the process closes once. Check that a wrong wire backend ID or restore count of `77`, `78.0` or `True` closes the process before requests. Pin the integer validation for both launch modes. Assert the existing Docker path rejects a supplied candidate descriptor. Retain the existing Docker argv and timeout compatibility tests.
- [x] Run the four test modules below and expect missing shared-runtime/native interfaces before implementation.
- [x] Move the existing probe and identity functions to `src/benchmark/oaf_tf215_runtime.py`. Preserve stdlib-only direct execution with `--probe` and lazy host imports. Export `tf215_worker_environment() -> dict[str, str]` using the current sanitizer and overrides. Point `compare.py`, `selected_compare.py`, `panel_metrics.py` and the three tests importing the old module (`test_oaf_tf215_runtime_identity.py`, `test_oaf_tf215_selected_compare.py`, `test_oaf_tf215_panel_metrics.py`) at the shared module. Remove the old path from `selected_compare.EXPERIMENT_MODULES`, since the existing benchmark source glob includes the new module. Update the README file table and add the new module to the existing workflow's `pull_request.paths` filter.
- [x] In `prepare_tf215_runtime`, retain the absolute venv path without resolving its interpreter symlink. Require a regular `worker.py` and preserve the scratch inventory's symlink rejection. Capture the environment, probe once and use `prepare_oaf_checkpoint(config, checkpoint_cache, download=False)`. Preserve reasons from a valid blocked probe and surface malformed probe records and timeouts as bounded preflight failures. Return absolute scratch and checkpoint paths so launch works from `input_root`. Build the corpus fingerprint with `build_runtime_identity(..., driver_files=())` and `sha256_hex(canonical_json_bytes(identity))`. Allow the empty list and preserve explicit driver hashing for existing experiment callers. The corpus fingerprint contains the following inputs.

  - Interpreter version and resolved binary digest, installed distribution names and versions, host platform/libc and CPU model, flags, count and affinity, and the captured environment allowlist.
  - `experiments/oaf_tf215/requirements.lock`, `runtime/oaf_tf1/model.json`, `runtime/oaf_tf1/patches/capture-emitted-frame.patch`, `experiments/oaf_tf215/patches/tf215-compat.patch` and the configured checkpoint archive SHA-256.
  - The pristine `upstream/magenta/music/sequences_lib.py` under the scratch build root and the existing inventory of `$SCRATCH/runtime`, including `worker.py`, `model.json` and vendor sources. Keep the existing cache exclusions. Use the adapter, map and canonicalization revisions for semantic host changes. Record the source revision with the pilot provenance.

- [x] In `backends/oaf.py`, add a native mode to command construction and make `_validate_descriptor` bind the expected descriptor ID to the selected launch mode. Preserve the existing TF1 check on the Docker path. Continue through the current relative input validator, response decoder and cleanup. Use `WorkerProcess.start` with `cwd=input_root`, the captured environment and the passed timeouts. In shared `_validate_ready`, require the existing TF1 wire ID and an integer restore count of exactly `78`, using the experiment's integer check.
- [x] Run `uv run pytest -q -o addopts= tests/benchmark/test_oaf_tf215_runtime_identity.py tests/benchmark/test_oaf_tf215_compare.py tests/benchmark/test_oaf_tf215_selected_compare.py tests/benchmark/test_task_d_contract.py`. Expect legacy and native tests to pass.
- [x] Run the full-suite gate from Global constraints.
- [x] Commit the shared runtime and native launcher with their tests as `feat(benchmark): launch the attested TF2.15 worker`.

### Task 3 — Wire the corpus command and shared resume

**Interfaces** Extend `OafCorpusRunRequest` with defaulted `backend: Literal["oaf", "oaf-tf215"] = "oaf"`, `python311: Path | None = None`, `runtime_root: Path | None = None` and `checkpoint_cache: Path | None = None`. `run_oaf_corpus(request, ...)` keeps its existing test seams. Add optional `failure_code` and `failure_detail` to its outcome with null defaults. Extend `build_inference_config(config, descriptor, model_lock_sha256, *, input_view_id=OAF_FULL_MIX_INPUT_VIEW_ID, runtime_sha256: str | None = None) -> dict[str, str]`. Candidate schema is `crux.oaf-tf215-inference-config/v1`, containing the existing fields plus its mandatory `runtime_sha256` and candidate map ID. The existing snapshot envelope remains `crux.oaf-corpus-run/v1`.

- [x] Extend `tests/benchmark/test_oaf_corpus_run.py` with exact-key and hash tests for both inference schemes. Change only one candidate runtime digest and assert that both `build_run_id` and `prediction_path` change. Retain the existing TF1 v1 asserts and reject a runtime field in v1 or a missing runtime field in the candidate schema.
- [x] In `tests/benchmark/test_oaf_corpus_run_acceptance.py`, use the established fake-manifest/materializer fixture for two candidate songs. Expect two inferred rows and ordinary reports under the candidate descriptor and map. On identical resume expect the same run ID, both dispositions `resumed`, the same prediction hashes and regenerated reports. Use a factory that fails the test if called on resume. Change persisted artifact bytes or the materialized WAV and assert the existing resume validation rejects reuse. Test that a candidate rounded-key collision already yields `prediction_publish_failed` with `duplicate_native_event` detail and the next independent row still runs.
- [x] In `tests/test_cli_benchmark.py`, assert candidate dispatch and usage errors for incomplete or incompatible flags. The candidate requires `--python311` and `--runtime-root`, and its `--checkpoint-cache` defaults to `artifacts/benchmark/model-cache`. Enforce the same backend and path contract and candidate cache default on direct `OafCorpusRunRequest` callers. Exercise shared preflight diagnostics using one TF1 and one TF2.15 failure.
- [x] Run the three modules below and confirm they fail on absent candidate support before implementing.
- [x] In `run_oaf_corpus`, finish the existing lineage and all-eligible timing preflight, then select the descriptor, prediction map, inference payload and lazy backend factory. Prepare the candidate runtime once per invocation, including resume, using the checkout root derived from `_model_lock_path()`. Keep the existing injected factory seam usable in both modes. Store the candidate identity dict in the run header as `runtime_identity`. Derive config SHA, run ID and prediction paths with the existing helpers.
- [x] Update the following sites in `src/benchmark/oaf_corpus_run.py`.

  | Current site | Required change |
  | --- | --- |
  | `build_inference_config`, TF1 map ID at line 281 | Select the map and schema by descriptor backend ID. Require `runtime_sha256` for TF2.15 and reject it for TF1. |
  | `inference_config_sha256`, exact keys and schema at lines 289–297 | Validate the exact field set for each schema and require a valid SHA-256 for the candidate runtime. Preserve the TF1 hash. |
  | Resume `prediction_artifact_matches_audio`, TF1 map at line 1399 | Pass the selected `prediction_map.map_id`. Keep prior row checks when a row is available, canonical WAV checks, and normal recovery of a valid artifact published before the snapshot. A missing path triggers inference. A present invalid artifact fails its row and stays untouched. |
  | Inference `map_oaf_prediction(native)` at line 1462 | Pass the selected `prediction_map` so emitted records and reports carry the selected map. |

- [x] Add optional `failure_code` and bounded `failure_detail` parameters to the shared `_fatal_outcome` at line 1011. Reuse `_bounded_error` for caught exceptions. Include useful preflight reasons for both modes and route the command's request-construction failure through the same outcome. Add the four CLI options with `click.Choice(("oaf", "oaf-tf215"))`, default `oaf`, nullable candidate paths and validation before dispatch. Emit populated failure fields for either backend.
- [x] Run `uv run pytest -q -o addopts= tests/benchmark/test_oaf_corpus_run.py tests/benchmark/test_oaf_corpus_run_acceptance.py tests/test_cli_benchmark.py`. Expect every candidate case and the existing TF1 recovery tests to pass.
- [x] Run the full-suite gate from Global constraints, including the existing corpus-run branches and coverage tests.
- [x] Commit the corpus runner and CLI with their tests as `feat(benchmark): opt into TF2.15 corpus runs`.

### Task 4 — Prove the normal CLI on Colab and record results

**Inputs** Use the existing verified `artifacts/experiments/oaf-tf215/fresh-september/` reference-set and reference-timing roots, including **all 180** eligible event bodies. Read source bytes for IDs **200** and **209** from `artifacts/experiments/oaf-tf215/twenty-song-panel/audio-cache/` with the cache index. Use a fresh output root for the pilot, since the prediction cache key does not include the run scope. Adapt the prior tooling under `twenty-song-panel/tools/` for the tested implementation revision, two manifests, 180 timing bodies, cache index and two verified audio blobs. Keep segmented upload, streamed heartbeats, archive verification and shutdown on exit. The pilot uses the normal checkout entry point and native worker.

- [x] Run `uv run ruff check` and `uv run ruff format --check` on the changed Python files. Run the full-suite gate from Global constraints again **before building the bundle**. Record the exact source revision or overlay digests and test result.
- [x] Start one fresh CPU VM using the authenticated Colab CLI. Provision a clean project environment at Python 3.12 and a separate worker environment with the existing lock. Preserve setup logs:

  ```bash
  uv sync --locked --no-dev
  uv venv --python 3.11 "$WORKER"
  uv pip sync --python "$WORKER/bin/python" --require-hashes experiments/oaf_tf215/requirements.lock
  uv pip check --python "$WORKER/bin/python"
  mkdir -p /tmp/numba-cache
  "$WORKER/bin/python" experiments/oaf_tf215/build_runtime.py --output "$SCRATCH"
  uv run --no-sync crux benchmark prepare-backend --backend oaf --download --cache-root "$CHECKPOINT_CACHE"
  ```

- [x] Execute the normal command in that VM. Set `REF` and `TIMING` to the files under `reference-set/manifests/` and `reference-timing/manifests/` with the respective digest filenames below and the `.jsonl` extension. Set `CACHE` to the verified panel audio cache.

  ```bash
  run_pilot=(uv run --no-sync crux benchmark run-oaf-corpus
    --backend oaf-tf215 --python311 "$WORKER/bin/python"
    --runtime-root "$SCRATCH/runtime" --checkpoint-cache "$CHECKPOINT_CACHE"
    --manifest "$REF" --timing-manifest "$TIMING" --cache-dir "$CACHE"
    --output-dir "$OUT" --include-simfile-id 200 --include-simfile-id 209)
  "${run_pilot[@]}"
  ```

  Verify manifest SHA-256 values `4f5d5e859650cb78a9f78697e8f19ab1ac3e7ea3c86dbea0e1f00c876357c7a3` (reference) and `372979d42fff94438fe8d844fca2770d430a3c0898a306a4bc46bd8b2887e38f` (timing). Expect exit `0`, two inferred rows and the candidate descriptor and map. Verify that the run header's runtime identity hashes to the runtime digest in the inference configuration. Read both predictions through `read_prediction_artifact` and verify their row bindings:

  | Simfile | Canonical WAV SHA-256 | Native events | Raw 50 ms TP / FP / FN | F1 |
  | ---: | --- | ---: | --- | ---: |
  | 200 | `29d4dedbad0360dce060ec3ecdc5ee09a03bc87b1e5bb197cc9d26cc8dc92977` | 283 | 210 / 73 / 563 | 0.397727 |
  | 209 | `78fe085e51b407eed57ff592812f796cf1f8311d5f0c200b970dcfbf3bb91841` | 296 | 72 / 224 / 1,129 | 0.096192 |

- [x] Save the first `run.json` byte-for-byte and move its reports to the first-run evidence directory. Invoke `"${run_pilot[@]}" --resume` against the same root and runtime. Expect the same run ID, both rows `resumed`, the same prediction paths, identities and hashes, and regenerated reports. Preserve both report sets, both snapshots, and both CLI summaries and exit statuses. Compare the scientific report fields. The fake-factory acceptance test checks that all-resumed runs bypass model worker construction.
- [x] After both measurements, run `uv sync --locked --extra mlflow --extra r2` and `PRELOAD_MODEL=0 uv run --no-sync pytest -q -o addopts=` on the supported Colab host. Capture the test count, result and duration.
- [x] Retrieve the reports, prediction artifacts, runtime evidence, setup logs and command summaries to `artifacts/experiments/oaf-tf215/opt-in-corpus/`. Verify the downloaded archive and terminate the VM.
- [x] Update the experiment README and RESULTS with the actual pilot commands, runtime fingerprint, dispositions, report metrics and test outcome. Record any failed stage or unavailable data with its specific reason.
- [x] Commit the evidence narrative as `docs(experiments): record the opt-in OaF corpus pilot`.

The pilot passed on implementation revision `1648da2`. [Measured results](../../../experiments/oaf_tf215/RESULTS.md#opt-in-corpus-pilot)
