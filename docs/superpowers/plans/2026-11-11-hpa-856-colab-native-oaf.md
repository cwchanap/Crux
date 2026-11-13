# HPA-856 — Docker-free OaF Benchmark on Colab: Implementation Plan

**Design:** [2026-11-11-hpa-856-colab-native-oaf-design.md](../specs/2026-11-11-hpa-856-colab-native-oaf-design.md)  
**Linear:** [HPA-856](https://linear.app/cwchanap/issue/HPA-856/run-the-oaf-benchmark-pipeline-on-colab-without-docker)  
**Related:** [HPA-86 / GitHub PR #34](https://github.com/cwchanap/Crux/pull/34) (**merged into main**; MLflow publication), HPA-328 (existing separation pipeline)

**Delivery:** This draft PR initially carries **only design + plan**. Implement, test, and append Colab evidence on **this same PR** after plan review. One ticket, one PR. Keep the current Docker path as the default. **Baseline review (2026-11-13):** HPA-86 is merged at `main` commit `f5e0d182`; no publication dependency remains.

## Global constraints

- No TF1 upgrade, scoring changes, reference-rule changes, report schema changes, new generic orchestration framework, or new MLflow code.
- Reuse `WorkerProcess.start()`, `create_backend()`, `run_oaf_corpus(..., backend_factory=...)`, `run_oaf_separation_pilot(..., backend_factory=...)`, v2 separator runtime freezer/attestation, and HPA-86 publication command.
- Never commit credentials, audio, checkpoints, model archives, separator stems, or generated lock environments. Generated runtime locks belong in the private Drive workspace.
- CI remains offline and fixture/fake based. The real Colab benchmark is an explicit manual acceptance gate, **not** claimed to have been executed by this planning PR.
- Stop at the first genuine environment/identity blocker; document it on this PR and HPA-856 instead of weakening the frozen provenance contracts.

## Task 0 — Verify Colab legacy-runtime feasibility **before production code**

The frozen OaF image uses Python 3.7 and TF1.15.5, whereas the Crux CLI needs Python 3.12. Prove that the native OaF worker can run on a real current Colab VM before investing in notebook orchestration or the full R2 run.

- [ ] On Colab, record date, VM image/Python version, architecture, glibc version, CPU, available disk, and pinned Crux commit. Confirm Python 3.7.17 can be provisioned with micromamba/conda on the VM, without replacing notebook Python.
- [ ] Install `libsndfile1`; use `runtime/oaf_tf1/requirements-build.lock` then `requirements.lock` with their existing hash/install constraints, in a dedicated Python 3.7 environment. Do **not** silently regenerate either lock.
- [ ] Mirror the Docker runtime layout in a disposable host directory: pristine vendor under `upstream/magenta`, checked-in worker/model/lock under `runtime`, and patched vendor produced by `apply_instrumentation_patch.py` with the committed patch. This can be done with temporary notebook commands before implementing the adapter.
- [ ] Acquire and hash-verify the existing OaF checkpoint through `prepare-backend` / checked-in model descriptor; launch the worker directly with its Python 3.7 interpreter, input-root working directory, CPU-only env, and smoke fixture. Confirm the ready response restores **78 tensors** and returned native events match the checked-in smoke oracle. Compare against existing Docker smoke evidence, rather than comparing only whether imports succeed.
- [ ] Exercise a tiny Drive-backed write/restore (including overwrite/rename assumptions needed by the existing artifact publisher) before committing to storing the entire corpus there. Record a failed atomic/durability assumption as a blocker.
- [ ] **STOP condition:** if Python 3.7/TF1 wheels cannot load or the smoke oracle differs, do not implement new scoring or switch to a different TF/OaF model. Capture actionable diagnostics and re-evaluate HPA-856's runtime approach **within the same PR**.

**Evidence:** non-secret Colab environment metadata, redacted install/smoke logs, smoke identity and oracle comparison. **Status at planning time: not executed.**

## Task 1 — Add a native launch option to the existing OaF backend

**Modify:** `src/benchmark/backends/oaf.py`, `runtime/oaf_tf1/model.py`  
**Tests:** `tests/benchmark/test_oaf_backend.py` (if present, otherwise focused existing backend tests); `tests/benchmark/test_worker_process.py`; `tests/runtime/oaf_tf1/`.

- [ ] Introduce a small native launcher builder alongside unchanged `build_docker_command()`: given `native_python`, `native_runtime_root`, `checkpoint_dir`, and `input_root`, produce the fixed direct-worker argv plus `cwd=input_root` and CPU-only subprocess environment. Prefer straightforward optional constructor arguments; no pluggable launcher hierarchy.
- [ ] In `OafBackend._ensure_process()`, branch once by the paired native configuration. Preserve the single worker protocol, timeout handling, ready validation, descriptor construction, native-event decode, close behavior, and injected process factory.
- [ ] Reject an incomplete native option pair and missing/wrong native-bundle layout before worker startup. The runtime's `model.json` must match the checked-in lock; descriptor identity always originates from the checked-in `runtime/oaf_tf1/model.json`. Keep absolute paths out of persisted model/run identity.
- [ ] Make the pristine Magenta `sequences_lib.py` source path relative to the worker/model file location so both Docker's `/opt/crux` layout and the mirrored native layout resolve the **same** uninstrumented source. Do not modify Magenta, the patch, or checkpoint descriptor.
- [ ] Test precise default Docker argv, precise native argv/cwd/env, paired options, missing vendor/upstream/lock, model-lock mismatch, worker-ready failure, and worker close. Use fake process factories; no TF1 install required in CI.
- [ ] Ensure any backend startup/worker failure remains a bounded existing `OafBackendError`, and no failed native startup produces a fake successful prediction.

**Check:** focused OaF backend, worker process, and runtime tests; Docker launch unit tests unchanged.

## Task 2 — Wire native selection to exactly the required benchmark commands

**Modify:** `src/cli/benchmark.py`  
**Tests:** `tests/test_cli_benchmark.py`, `tests/benchmark/test_oaf_corpus_run.py`, `tests/benchmark/test_separation_pilot.py`.

- [ ] Add the paired `--oaf-native-python PATH` and `--oaf-native-root PATH` flags to `smoke-backend`, `run-oaf-corpus`, and `run-oaf-separation-pilot`. No flag defaults to existing Docker behavior; both flags select native; exactly one flag is a clear argument error.
- [ ] Bind the selected native launcher through the **existing** backend-factory injection for corpus and separation. Do not change `OafCorpusRunRequest`, `OafSeparationPilotRequest`, run identifiers, scope selection, or persisted snapshots.
- [ ] Preserve the parent-run checkpoint/descriptor binding in `separation_pilot._bind_oaf_backend_factory` (its override must still win over any notebook configuration); preserve `--resume` semantics.
- [ ] Test both commands forward native configuration to the injected backend, default calls remain Docker-only, invalid option pairs fail, and frozen parent binding is not bypassed.

**Check:** CLI, corpus-run, and pilot fixture suites. No real R2 or separation execution in CI.

## Task 3 — Add one Colab notebook with explicit environment and identity gates

**Create:** `notebooks/oaf_benchmark_colab.ipynb`  
**Create:** `docs/benchmark/colab.md`  
**Modify:** `README.md` to add one Colab benchmark link **alongside the existing MLflow publishing link**.

- [ ] Provide a configuration cell for **pinned** repository commit, Drive-backed workspace, absolute paths, CPU/GPU availability, and R2/MLflow credential lookup from Colab Secrets. Don't print tokens.
- [ ] Create Crux Python 3.12 with `uv sync --extra r2` for benchmark work; enable the **already merged** `mlflow` extra only for publication with `uv sync --extra r2 --extra mlflow`. Create frozen OaF Python 3.7, separate Spleeter 2.4.2, and separate HTDemucs 4.1.0 environments. Use shell calls to existing tools, not a second Python orchestration package. OaF stays CPU; GPU is optional for HTDemucs.
- [ ] Materialize the small patched/native OaF bundle by calling the **existing** instrumentation script. Reuse the exact original `model.json` and checkpoint hash. Print non-secret checkpoint/model/commit identities before executing inference.
- [ ] Initialize stable Drive paths for manifests, reference events, corpus cache, OaF/pilot run roots, and reports. Keep checkpoint/environment directories at stable target paths on `/content` and archive/restore them to/from Drive where the probe requires identical environment bytes/locations.
- [ ] On every notebook rerun, require the prior pinned commit, manifest/model-lock hashes, separator lock/environment manifests, and interpreter+model-root attestation to match. Do not automatically re-freeze a changed runtime during `--resume`.
- [ ] Keep cells resumable and independent; no general DAG, server, scheduler, CI benchmark job, or hidden background step.
- [ ] Provide an explicit, actionable stop message at missing Colab Secret, runtime incompatibility, incomplete manual review, mismatched evidence, or missing/invalid HTTPS MLflow configuration. The HPA-86 command is already on `main`; do not create a replacement publisher.

**Check:** validate notebook JSON schema and inspect/smoke each cell on a real Colab VM. Do not rely on local Python 3.12 to execute cells intended for legacy interpreters.

## Task 4 — Rerun authoritative full mix and complete the human-reviewed subset

**Notebook cells only** (reuse existing CLI and published manifests).

- [ ] Sync R2 with `crux benchmark sync-r2-corpus`. Build authoritative selected-chart, reference-timing, and reference-set manifests with their existing commands. Persist **source files and exact manifest identities**, not merely the latest `run.json`.
- [ ] Compare regenerated identities and the deterministic candidate slate with available prior benchmark evidence. If identities differ, diagnose source/clock/chart revision **before** inference; a changed 30-song slate requires new human review.
- [ ] Run `crux benchmark run-oaf-corpus --manifest ... --timing-manifest ... --cache-dir ... --output-dir ... --resume --oaf-native-python ... --oaf-native-root ...`.
- [ ] Record runtime, count, report identity, and the exact event-micro F1 value. The acceptance baseline is **182 successful songs, 50 ms raw event-micro F1 = 0.234846**; do not round a mismatch into success or accept an incomplete population.
- [ ] Run `prepare-reviewed-subset` to produce the candidate CSV. **Stop notebook execution** and request genuine listening review. The reviewer completes the CSV manually and returns it via Drive.
- [ ] Resume with the completed CSV, `finalize-reviewed-subset`, and `score-oaf-reviewed-subset`. Verify accepted subset identity and reviewed full-mix report. No automatic review answers.

**Check:** full mix and accepted 30-song review evidence are auditable from persisted manifests, CSV and canonical reports.

## Mainline integration checkpoint (2026-11-13)

- [x] Reviewed the new `main` baseline `f5e0d182` after merged PR #34: it adds the optional `mlflow` extra, `publish-mlflow-cohort` CLI, `docs/benchmark/mlflow.md`, `notebooks/mlflow_publish_colab.ipynb`, strict publication tests, and README link.
- [x] HPA-86's merge dependency is resolved; HPA-856 now only needs to call the existing CLI.
- [ ] In implementation, preserve the existing HPA-86 code in `src/cli/benchmark.py`, `tests/test_cli_benchmark.py`, and `README.md`. Do not duplicate the publishing notebook or exporter.

## Task 5 — Run the separation pilot on the same VM and publish to MLflow

**Notebook cells only**, after Task 4. HPA-86's publisher is already on `main`; no separate PR merge gate remains.

- [ ] Install/restore and freeze Spleeter 2.4.2 + HTDemucs 4.1.0 from the existing standalone-Python process. Use the current `freeze-separator-runtime` CLI and fixed model-root policy. Validate v2 lock + `environment.json` + model-root probe before the pilot begins.
- [ ] Run `crux benchmark run-oaf-separation-pilot --manifest ... --timing-manifest ... --subset-manifest ... --oaf-run ... --cache-dir ... --output-dir ... --spleeter-python ... --demucs-python ... --spleeter-model-root ... --demucs-model-root ... --resume --oaf-native-python ... --oaf-native-root ...`. Keep stem and derived OaF execution on the **same** Colab VM; do not port unverified Mac stems.
- [ ] Inspect full-mix control, Spleeter and HTDemucs reports, recorded failures/RTF, and the paired comparison. Finalize using `finalize-oaf-separation-pilot` with a **human-provided** decision/rationale.
- [ ] Call the **merged** `crux benchmark publish-mlflow-cohort --reports ... --scope ... --experiment crux-benchmark` separately for the **reviewed** full-mix report (`scope=reviewed`) and each Spleeter/HTDemucs pilot report (`scope=pilot`). Do not add or modify any MLflow publisher; reuse the existing optional extra, HTTPS-only tracking URI, Colab Secret credentials, and bounded CLI results.
- [ ] Verify all three are visible in the same `crux-benchmark` experiment with distinct `crux.input_view_id` and expected shared reference identities. Publish **only after** canonical report generation has finished (HPA-86's five-file snapshot is not atomic against concurrent writes). Re-publish each stable report, assert the same `run_id` and `created=false`, and confirm only the five allowlisted report artifacts were uploaded and read back byte-for-byte. Never upload audio, stems, predictions, or checkpoints.
- [ ] Keep canonical Crux reports complete and usable even if MLflow cannot be reached.

**Check:** canonical pilot run/comparison/handoff, both derived reports, and three MLflow run IDs with idempotency evidence (non-secret only).

## Task 6 — Final regression and PR closeout

- [ ] Run focused offline tests covering native command selection, Docker parity, worker errors, CLI flags, factory forwarding, run identity, and separation parent binding.
- [ ] Run repository gates: `uv run pytest`, `uv run ruff check .`, `uv run ruff format --check src tests`, and `uv run pylint --errors-only src`. Inspect failures and do not change unrelated code merely to increase a lint score.
- [ ] Record one real Colab smoke pass and a full end-to-end evidence run. In particular, verify a VM reset can `--resume` using stable Drive data and the **same** runtime/environment identities. An untested VM-reset claim remains unchecked.
- [ ] Review the PR diff for only scoped production changes, unit tests, notebook and docs; no credential or large artifact accidentally committed.
- [ ] Update the PR description and HPA-856 with verified evidence/remaining blockers. Mark this **same** PR ready for review only after acceptance; HPA-86 is already merged. Do not create an implementation PR.

## Expected scoped implementation files

| Area | Primary files |
| --- | --- |
| Native OaF command and source resolution | `src/benchmark/backends/oaf.py`, `runtime/oaf_tf1/model.py` |
| Existing CLI entry points | `src/cli/benchmark.py` |
| Native/backend/CLI/runner test coverage | Existing `tests/benchmark/test_oaf_*.py`, `test_separation_pilot.py`, `test_worker_process.py`, `tests/test_cli_benchmark.py`, and `tests/runtime/oaf_tf1/` |
| Colab operation and documentation | `notebooks/oaf_benchmark_colab.ipynb`, `docs/benchmark/colab.md`, `README.md` |
| Planning documentation | This plan and the linked design spec |

The **already-merged** MLflow exporter, its optional-dependency configuration, HPA-86's publication-only notebook and its tests, scoring/report publication modules, separator attestation policy, benchmark manifest schemas, and Dockerfile remain outside implementation scope. Extend `src/cli/benchmark.py` and `tests/test_cli_benchmark.py` without replacing the existing MLflow command/tests.
