# HPA-856 — TF2.15 Identity and Bounded Comparison Implementation Plan

> **For agentic workers:** Use `superpowers:executing-plans` to implement this plan task by task. Steps use checkbox syntax. Continue in draft PR #36; this plan covers experiment tooling and measurements.

**Goal:** Give TF2.15 a distinct experimental identity and quantify its drift
against five fixed archived TF1 full-mix predictions.
**Architecture:** Keep a small identity module and a selected-input driver
under `experiments/oaf_tf215/`. Reuse the prototype worker and existing typed
artifact readers, materializer, binary64 event representation and matcher.
Prepare and seal input bindings before executing the candidate.
**Tech stack:** Host Python 3.12, isolated Python 3.11 / TensorFlow CPU 2.15.1,
existing canonical JSON, prediction-v2 readers, pytest and fake workers.
**Spec:** [Identity and comparison design](../specs/2026-11-17-hpa-856-tf215-identity-comparison-design.md).
**Status:** Experiment tooling implemented. Real measurements are blocked on
the matching archived TF1 inputs. See [recorded results](../../../experiments/oaf_tf215/RESULTS.md#five-song-continuation).

## Global constraints

- Continue HPA-856 in draft PR #36. Preserve `runtime/oaf_tf1/`, `src/`, root dependencies and frozen fixtures.
- Candidate backend ID: `magenta-egmd-tf215-94529798-8hit-experimental-v1`. Use the existing descriptor-v2 key set and hash helper locally; do not register the candidate in production readers.
- Target Linux x86-64 CPU; preserve the prototype lock, compatibility patch and four worker controls. Keep notebook/kernel and host project environments separate from Python 3.11.
- Fixed simfile order: `(389, 391, 27, 241, 207)`. Require identical archived canonical input bytes. No replacement song or prediction-cache reuse.
- Use a fresh UUID per execution and partition artifacts by descriptor hash, runtime hash and execution ID under `artifacts/experiments/oaf-tf215/selected/`.
- The smoke oracle remains an exact binary64 gate. Selected-song drift uses archived six-decimal precision: strict 0.001-second matching, then a separate strict-unmatched diagnostic within `512 / 44100 + 0.000001` seconds, both at zero offset.
- No production adapter/CLI/cache, broad corpus, reviewed subset, separators, Drive or MLflow in this phase.

## Review focus

1. TF1 wire metadata accidentally becomes candidate identity: Task 1 tests the distinct descriptor and Task 3 retains `ready.backend_id` only as protocol metadata.
2. Identical lock hides host/source differences: Task 1 fingerprints CPU count/affinity, the explicit environment allowlist and host benchmark modules; Task 3 uses fresh execution IDs without cache reads.
3. Matching song ID hides different input or archived predictions: Task 2 validates typed bindings, full digests and all five inputs before inference.
4. One extra event, a frame shift or a rounding collision obscures drift: Task 3 tests separate strict/frame-shift matching, unmatched accounting, event adapters, multiplicity, empty outputs and missing confidence.
5. Blocked, partial or non-repeatable runs look complete: Tasks 2–3 retain failed stages and five row outcomes, and test the distinction between measurement completion and parity.

## Files and interfaces

| File | Responsibility |
| --- | --- |
| Create `experiments/oaf_tf215/runtime_identity.py` | Probe the isolated interpreter and build the experiment-local descriptor/runtime fingerprint. |
| Create `experiments/oaf_tf215/selected_compare.py` | Expose the prepare/run commands and execute the bounded comparison. |
| Create `experiments/oaf_tf215/selected_inputs.py` and `selected_metrics.py` | Verify seals and inputs, and calculate the comparison metrics. |
| Create `tests/benchmark/test_oaf_tf215_runtime_identity.py` | Host-side identity tests using fake probe data and temporary trees. |
| Create `tests/benchmark/test_oaf_tf215_selected_{compare,inputs,metrics}.py` | Selection, binding, measurement and fake-worker execution tests. |
| Modify `.github/workflows/oaf-tf215-probe.yml` | Include both test modules in paths and the existing host-test step. |
| Modify `tests/benchmark/test_oaf_tf215_workflow.py` | Verify checkpoint-free PR execution includes the new modules. |
| Modify `experiments/oaf_tf215/README.md`, `RESULTS.md` | Document executed commands, provenance and actual new outcomes. |

### Task 1: Build experiment-local identity

**Interfaces:** `runtime_identity.py` produces
`build_candidate_descriptor(config: OafModelConfig) -> BackendDescriptor`,
`collect_probe(python311: Path, *, worker_environment: Mapping[str, str]) -> dict[str, object]`, and
`build_runtime_identity(repo_root: Path, runtime_root: Path, probe: Mapping[str, object], *, driver_files: Sequence[Path]) -> dict[str, object]`.
The returned runtime object has exactly the field groups specified by the
design. Hash it with `sha256_hex(canonical_json_bytes(...))`; execution ID and
provenance remain outside. The caller passes the same `_worker_environment()`
mapping used for workers. `collect_probe()` launches
`[python311, "-s", runtime_identity.py, "--probe"]` and preserves the JSON
diagnostic even for blocked exit 2. A blocked probe cannot build an identity.
The required `driver_files` contains repository-relative paths; Task 1 tests
temporary file lists before Task 2 creates the selected driver. Task 3 supplies
the complete live source set specified by the design.

- [ ] Write tests in `test_oaf_tf215_runtime_identity.py`: derive the baseline through `_expected_oaf_descriptor(config)` and change only `backend_id` through `build_descriptor()`. Assert descriptor SHA changes while model/output semantics and ordinary TF1 validation remain unchanged.
- [ ] Add temporary-tree tests: changed lock, scratch source, supplied host module, interpreter hash, installed version, CPU count/affinity or any allowlisted environment setting changes the runtime fingerprint; moving identical trees or changing Git/provenance paths does not. Preserve unset versus empty environment values. Reject an empty file list, missing supplied files and source symlinks. Complete live-list assembly is tested in Task 3.
- [ ] Add probe fixtures for supported x86, this host's aarch64 `Features`/implementer/part layout, unavailable affinity and missing CPU data. Expect one JSON `status=blocked` diagnostic and exit 2 for unsupported/unavailable hosts, without an identity or traceback. In a clean subprocess, block imports of `src`, `runtime` and third-party packages and assert `--probe` still emits its diagnostic. Assert probe launch uses `-s` and the exact worker environment.
- [ ] Run `uv run pytest -q tests/benchmark/test_oaf_tf215_runtime_identity.py`; expect an import/missing-interface failure before implementation.
- [ ] Implement the three interfaces. Keep top-level imports standard-library only; use postponed annotations, `TYPE_CHECKING` imports and lazy project imports in host-only descriptor functions. Reuse `_expected_oaf_descriptor(config)` rather than reconstructing the baseline payload. Implement the spec's structured probe contract, CPU count/affinity and full explicit environment allowlist. Hash actual supplied source files and record Git HEAD/dirty state separately.
- [ ] Run the test module again; expect all tests to pass without TensorFlow or a model worker. On this aarch64 development host, invoke `--probe` and verify structured blocked-platform JSON with exit 2. An x86/Python 3.11 probe with all required data exits 0. Neither is a model-compatibility check.
- [ ] Commit the implementation and tests as `feat(experiments): identify TF2.15 candidate executions`.

### Task 2: Prepare and seal the five input bindings

**Interfaces:** `selected_compare.py` produces
`prepare_selection(*, baseline_evidence_dir: Path, baseline_run: Path, reference_manifest: Path, timing_manifest: Path, cache_dir: Path, input_root: Path) -> dict[str, object]`
and `selection_sha256(selection: Mapping[str, object]) -> str`.
The selection contains exactly the identities and five row bindings described
by `crux.oaf-tf215-selection/v1` in the spec. CLI subcommand `prepare` exposes
the six input paths and `--output PATH`. Require a new experimental
input directory; write no selection artifact until all five bindings validate.

- [ ] Write synthetic report/run/artifact tests for the fixed rank rule, Decimal score ordering and integer-ID tie breaks. A fixture may exercise the general rank selector with fewer rows; the real preparation path requires the pinned 182-success evidence and exact five IDs.
- [ ] Add tests for missing prediction/source bodies, changed report digest, wrong cohort/reference/timing identity, duplicate score IDs, changed baseline descriptor/model lock/inference config, artifact/run-row mismatch and canonical WAV hash mismatch. Include a valid prediction body from another registered backend, a body with the wrong TF1 descriptor digest, and changed event/class counts while other run-row fields agree. Missing access is blocked; integrity mismatches fail. Assert no worker starts and no alternate song is selected.
- [ ] Run `uv run pytest -q tests/benchmark/test_oaf_tf215_selected_compare.py -k 'selection or prepare'`; expect missing-interface failure.
- [ ] Implement preparation using `parse_oaf_corpus_run()`, `load_reference_set_manifest()`, the timing-manifest loader/preflight, `read_prediction_artifact()`, `prediction_artifact_matches_run_row()`, `resolve_source_audio()` and `materialize_full_mix_audio()`. Use `_expected_oaf_descriptor(config)` in the spec's full baseline-lineage checks: descriptor payload/digest, model/checkpoint/inference identity, rederived run ID, complete population, shared reference/timing identity, and each body's descriptor/map/event-count/class-count agreement with the pinned reports. The named readers do not enforce that chain by themselves. Require the known OaF full-mix view, frozen frame limit and exact archived input hash.
- [ ] Seal the verified descriptor/model/inference/reference/timing identities, per-song event/class counts, relative WAV/prediction paths and every required digest. Prediction paths resolve beneath `baseline_run.parent`; WAV paths resolve beneath `input_root`. Reject absolute paths, traversal and symlinks. Hash canonical JSON without a trailing newline; print that digest for the run command's required `--selection-sha256` flag, outside the hashed document. Test portable root relocation with unchanged digests and rejection of escaping paths.
- [ ] Rerun the targeted tests; expect all to pass. Recompute membership from Git commit `599033663ca3bb495d8faaac13180206eaafadb4` and confirm IDs/ranks/F1/counts against the spec's table without running inference.
- [ ] Commit as `feat(experiments): seal the five-song TF2.15 comparison`.

### Task 3: Measure drift and preserve failures

**Interfaces:** `selected_compare.py` produces
`baseline_native_events(artifact: PredictionArtifact) -> tuple[NativeEvent, ...]`,
`candidate_native_events(events: tuple[SmokeOracleEvent, ...]) -> tuple[NativeEvent, ...]`,
`compare_events(baseline: tuple[NativeEvent, ...], candidate: tuple[NativeEvent, ...]) -> dict[str, object]`
and
`run_selected_comparison(*, selection_path: Path, expected_selection_sha256: str, baseline_run: Path, python311: Path, runtime_root: Path, checkpoint_cache: Path, input_root: Path, output_root: Path, process_factory: Callable[..., Any] | None = None) -> dict[str, object]`.
CLI subcommand `run` exposes those **seven paths**, plus required scalar
`--selection-sha256`. Resolve `fixture` and `oracle_path` from the repository
root to `tests/fixtures/oaf_tf1_smoke/canonical.wav` and
`tests/fixtures/oaf_tf1_smoke/smoke-oracle.json`; pass them explicitly to
`compare.run_comparison()`. They are fixed inputs, not additional run flags.
The result uses
`crux.oaf-tf215-selected-comparison/v1`, with the identity envelope,
unmodified smoke result, five row outcomes and the precision/matching policy.

- [ ] Test event adapters: unwrap parsed mapped events into the existing float-valued `NativeEvent`s; decode candidate binary64 hex with `struct.unpack()` and `_decode_native_event()`. Assert `_event_from_native()` restores the exact candidate arrays, including null confidence. `compare_events()` receives only native tuples.
- [ ] Write metric tests with an inserted event, different native classes, empty outputs, equal-time hits, quantization-induced tie ordering, missing confidence and changed note/bin/group/velocity. Record duplicate rounded keys while retaining multiplicity and original indexes; a collision is a diagnostic, not a failed row. Assert six-decimal precision, signed deltas, max/median/p95 statistics, uncapped full counts and at most eight examples.
- [ ] Test a one-frame shift at `512 / 44100` seconds: it remains strict-unmatched at 1 ms, then becomes a separate frame-shift candidate. Test both signs, six-decimal rounding at the one-frame boundary, a two-frame shift, different classes, ambiguous residual pairs and no double use of a strict match. Reuse `score_events()` on residuals at `512 / 44100 + 0.000001` seconds with zero offset. Assert separate strict/shift metrics and exact count reconciliation.
- [ ] Add fake-worker tests for per-song two-same-worker plus one-fresh-worker boundaries, exact native repeatability, 78-tensor ready checks, cleanup and existing 3600-second corpus request/close timeout policies. Reuse `WorkerProcess`, environment sanitization, ready validation and event decoding from the prototype; avoid a parallel worker protocol or refactoring shared production code.
- [ ] Test execution preflight against the prepared seal: reject a changed selection digest, run digest, body descriptor/count/digest or WAV digest/frame count before the first worker. Verify it rechecks pinned membership and sealed lineage identities without reopening report/reference/timing manifests or rematerializing source audio. Test both relative-path roots, root relocation and path escape rejection. Include the six experiment scripts, every benchmark module, the frozen host model and the consumed pristine converter in the runtime identity. A fresh execution never reads prior experimental predictions. Changed source/probe data after measurement fails the record. Keep explicit failed and unattempted row outcomes.
- [ ] Test the smoke prerequisite through `compare.run_comparison()`, asserting the fixed repository fixture/oracle arguments: accept `smoke_exact` and repeatable `oracle_divergent`; block or fail for other outcomes. Preserve its binary64 comparator and CLI exit behavior. Test selected exit 0 for complete repeatable measurements with drift/collisions and nonzero for blocked, failed, partial or non-repeatable execution. A blocked probe prevents any model start.
- [ ] Run `uv run pytest -q tests/benchmark/test_oaf_tf215_selected_compare.py`; expect the new behavior tests to fail before implementation.
- [ ] Implement the interfaces and CLI. Supply Task 1 with the live source set from the design. Reuse Task 2's sealed bindings and the spec's explicit event adapters. Keep original binary64 arrays for repeatability and use `quantize_six()` for archived-precision views, preserving rounded collisions. Match first at 1 ms and then only strict residuals within one emitted frame plus the rounding allowance. Retain indexes, metadata, distinct metric buckets, missing-confidence counts and every failed row.
- [ ] Write the four artifacts specified by the design under the fresh execution directory. The raw event file retains every completed attempt; a complete run has 15 song attempts. Never publish them as canonical prediction-v2 bodies. Recollect the interpreter/host probe and recompute the runtime fingerprint before marking measurement complete.
- [ ] Add both new test modules to the existing workflow's path filters and host-test command, updating its workflow tests. Keep routine PR CI checkpoint-free and add no corpus credentials or selected-song model job.
- [ ] Run `uv run pytest -q tests/benchmark/test_oaf_tf215_runtime_identity.py tests/benchmark/test_oaf_tf215_selected_compare.py tests/benchmark/test_oaf_tf215_compare.py tests/benchmark/test_oaf_tf215_workflow.py tests/runtime/test_oaf_tf215_builder.py`. Expect all to pass; run `uv run ruff check .`, `uv run ruff format --check src tests` and `uv run pylint --errors-only src`.
- [ ] Commit as `feat(experiments): measure bounded TF2.15 prediction drift`.

### Task 4: Execute the bounded comparison and record the decision

**Consumes:** The Task 1–3 CLI, the existing scratch-builder/lock commands in
the prototype README, and locally supplied verified baseline/source artifacts.
**Produces:** Actual preparation/model records and a RESULTS update, with
commands, host versions, identities, failures and per-song measurements.

- [ ] Locate the original `run.json`, its five prediction bodies, matching manifests and source cache through available artifact sources. The three archived Git reports are sufficient for membership, not for inference/input equivalence. If required bodies cannot be obtained, record the specific missing inputs as blocked; never synthesize digests or claim a prepared selection.
- [ ] Use a fresh Linux x86-64 Colab CPU VM and the README's same lock sync/check, scratch builder and graph checks. Keep the notebook kernel unchanged. Run `selected_compare.py prepare` with the original archived artifacts and a new experimental input root. Seal all five IDs before executing TF2.15.
- [ ] Run `selected_compare.py run` with the resulting selection and isolated worker interpreter. Retain `selection.json`, `smoke.json`, `native-events.jsonl`, `comparison.json`, the command log and lock/install-check evidence. Verify each worker's restore count, each song's 2+1 repeatability and the fingerprint-before/after check. Report every row, including any failed measurement.
- [ ] Repeat on a second fresh Linux x86-64 Colab CPU VM with the identical sealed selection and verified input/prediction bodies. Supply the printed selection digest, revalidate the seal on each VM and use new runtime records/execution IDs. Compare both VMs explicitly; label this as cross-VM evidence on one provider. Missing data remains blocked.
- [ ] Record the GitHub selected-song repeat as expected blocked: this phase provides no corpus credentials or artifact-delivery mechanism. Use GitHub optionally only if an authorized input-delivery path already exists; add no provisioning project or corpus credentials to CI. No arbitrary shell failure is swallowed to get a green job.
- [ ] Update README and RESULTS with actual executed evidence, precision limits, all five song outcomes and cross-host differences. Do not overwrite or relabel the prior `c3ffbef` smoke evidence. A successful measurement is not TF1 parity or full HPA-856 completion.
- [ ] Run the full suite with the established MLflow/R2 extras on a supported host and existing lint/format checks for implementation changes; record exact commands, revision, results and any host limitations.
- [ ] Commit the evidence narrative as `docs(experiments): record the bounded TF2.15 comparison`. Keep large inputs/model outputs in the experiment artifact area and keep PR #36 draft for the resulting integration/broader-measurement decision.

## Handoff boundary

This continuation makes the next implementation and evidence gates concrete.
Tasks 1–3 can be implemented and verified with local fakes before data access.
Task 4 needs the archived input/prediction bodies, which are absent from this
checkout. There is no scientific acceptance threshold to infer from a zero
measurement exit status; the final evidence informs the next scoped decision.
