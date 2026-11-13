# HPA-856 — Docker-free OaF Benchmark on Colab: Design

**Issue:** [HPA-856](https://linear.app/cwchanap/issue/HPA-856/run-the-oaf-benchmark-pipeline-on-colab-without-docker)  
**Status:** Planning only; no Colab runtime verification has been performed.  
**Reviewed baseline:** `main` at `f5e0d182` on 2026-11-13; HPA-86 / PR #34 is merged.  
**Delivery:** One ticket, one PR (this draft holds the plan first, then implementation and evidence).

## Goal and first use

Run the existing authoritative R2 → reference manifests → OaF full-mix → human-reviewed subset → Spleeter/HTDemucs separation pilot → canonical reports pipeline inside a Google Colab VM, with no Docker or Mac/Rosetta dependency. Persist its benchmark state in Google Drive so `--resume` can continue after a VM reset. Publish the three reviewed/pilot views using the **already-merged** HPA-86 MLflow command from [PR #34](https://github.com/cwchanap/Crux/pull/34).

Do not broaden this ticket into generic remote execution, new scoring, fine-tuning, or another experiment tracker.

## Current repository seams

- `src/benchmark/backends/oaf.py` owns the Docker command and the persistent `WorkerProcess`; `src/benchmark/backend_registry.py` still selects the same `oaf` backend.
- `src/benchmark/worker_process.py::WorkerProcess.start` already supports `cwd` and `env`. The OaF JSON-line worker accepts **relative** audio paths and `--checkpoint-dir`; a native worker therefore needs its working directory set to the canonical input root.
- `runtime/oaf_tf1/Dockerfile` installs hashed `requirements-build.lock` then `requirements.lock`, copies a separately patched Magenta vendor tree, retains a pristine upstream copy, and runs `runtime/worker.py` under Python 3.7. The model's pristine-sequence-module path is currently hard-coded to `/opt/crux/upstream`.
- `run_oaf_corpus(request, backend_factory=...)` and `run_oaf_separation_pilot(request, backend_factory=...)` already accept backend factories. Use those seams; do **not** add machine-specific runtime paths to persisted run headers, cohort identities, or public scoring schemas.
- `src/benchmark/separators.py` already implements the required v2 lock plus environment-manifest and model-root probes for isolated Spleeter and HTDemucs runtimes. Keep this policy unchanged.
- Merged HPA-86 provides `publish-mlflow-cohort --reports ... --scope ... [--experiment crux-benchmark]`, `mlflow-skinny` in the optional `mlflow` extra, `docs/benchmark/mlflow.md`, and `notebooks/mlflow_publish_colab.ipynb` (publication-only). Reuse the command and existing credential setup; do **not** copy or fork the exporter or duplicate the publication-only notebook.

## Runtime selection (minimum new production surface)

The Docker implementation remains the default. Add an optional **paired** `--oaf-native-python PATH` and `--oaf-native-root PATH` to `smoke-backend`, `run-oaf-corpus`, and `run-oaf-separation-pilot`. With neither option, use today's Docker command unchanged. With both, bind `create_backend(native_python=..., native_runtime_root=...)` through the existing caller's `backend_factory` argument. Exactly one of the pair is a CLI error.

In `OafBackend`, select just one of two launch commands, keeping a single descriptor, worker protocol, ready validation, model lock, checkpoint binding, and event decoder:

- Docker: unchanged `build_docker_command(...)` and unchanged subprocess flags.
- Native: `[<python37>, "-s", <native-root>/runtime/worker.py, "--checkpoint-dir", <verified-checkpoint-dir>]`, launched with `cwd=<canonical-input-root>` and the Docker-equivalent `CUDA_VISIBLE_DEVICES=-1`, `PYTHONHASHSEED=0`, and `NUMBA_CACHE_DIR` settings. Use the existing subprocess factory's optional `cwd`/`env` support; do not use `shell=True`.
- Native preflight must reject missing interpreter, runtime files, patched vendor, upstream sequence source, or model-lock mismatch **before** starting the worker. Verify the ready response as today (including 78 restored tensors). Keep the source of the `BackendDescriptor` at checked-in `runtime/oaf_tf1/model.json`.

**Native bundle layout:** Reproduce only the relevant Docker file layout under an ephemeral local `<native-root>`:

```text
<native-root>/
  runtime/{worker.py,model.py,model.json,vendor/magenta/...}
  upstream/magenta/...
```

The notebook copies the checked-in worker/model/descriptor and pristine Magenta tree, then runs **the existing** `apply_instrumentation_patch.py` with its committed patch and hash checks to produce `runtime/vendor`. Change `DEFAULT_UNINSTRUMENTED_SEQUENCE_LIB` in `runtime/oaf_tf1/model.py` to be relative to the runtime file's parent (resolving to the same `/opt/crux/upstream/...` in Docker). This avoids a new environment-variable protocol or a second OaF implementation.

Do not commit generated runtime trees, checkpoints, extracted stems, environment archives, or secret values. Do not change `runtime/oaf_tf1/model.json`, the instrumentation patch, the pinned requirements, or Docker behavior.

## Colab environments and persistence

**Go/no-go first.** Colab's 2026.10 image uses Python 3.13; Crux's project requires `==3.12.*`, while the frozen TensorFlow 1.15.5 wheel is CPython 3.7 Linux x86-64. Bootstrap four isolated host environments rather than downgrade Colab's notebook interpreter:

1. Crux CLI: managed Python 3.12 with `uv sync --extra r2` during benchmark work; add the already-available optional `--extra mlflow` to this same environment only when publishing (`uv sync --extra r2 --extra mlflow`).
2. OaF: a separate Python 3.7.17 environment (micromamba/conda is acceptable) with both existing hashed requirements lockfiles, the required system `libsndfile1`, and CPU-only execution.
3. Spleeter 2.4.2 and HTDemucs 4.1.0: two separate local stand-alone Python environments, the same verified model-root layouts, and the existing freeze+attestation CLI. Do not install them into the Crux or OaF interpreter.

Google's [Colab runtime list](https://research.google.com/colaboratory/runtime-version-faq.html) and [TensorFlow 1.15.5 wheel list](https://pypi.org/project/tensorflow/1.15.5/) establish the **need** for this separation; they do **not** prove legacy wheels and libraries execute on today's Colab VM. Before any full-corpus work, install OaF and execute the repository's actual smoke fixture and oracle on the target Colab VM. Failure is a hard stop and evidence to document, not permission to rewrite the model.

Use **stable absolute Drive paths** for R2 cache, source/reference/timing/subset manifests, OaF and pilot run roots, checkpoints, and canonical reports. Keep the checkout at a pinned commit for a run's full lifetime; do not silently clone a moving `main` on resume. Stage bulky interpreter/model files under stable absolute `/content` paths for native execution; after first successful separator attestation, archive the exact interpreter environments and model roots to Drive and restore them to the **same** paths on a new VM. Preserve lock and `environment.json` bytes and re-attest before resuming; if reconstruction changes their identity, stop rather than misattribute old stems. Do not introduce a separate artifact database, new resume schema, or a background service.

Drive FUSE behavior, quotas, and atomic-publication assumptions require one small real-VM pilot before expensive work. Mounting Drive requires interactive permission; never persist R2 keys or MLflow tokens in Drive/notebook output. Read them from Colab Secrets into process environment. Colab sessions are not guaranteed to last for the corpus run ([official FAQ](https://research.google.com/colaboratory/faq.html)), so `--resume` is mandatory for long stages.

## Notebook flow (explicit cells, not a new orchestrator)

Add `notebooks/oaf_benchmark_colab.ipynb`, alongside the HPA-86 publication-only notebook:

1. **Configure:** mount Drive, read Colab Secrets, pin Crux commit, declare one workspace root and predictable absolute paths; print non-secret environment versions/available disk space.
2. **Bootstrap & gate:** prepare separate Python environments, generate the instrumented native bundle, acquire/verify the existing OaF checkpoint, and run the frozen smoke oracle. Stop on any failure.
3. **Reference inputs:** `sync-r2-corpus`, `select-reference-charts`, `build-reference-timing`, `build-reference-set`. Preserve their published manifests, not just a textual run log; compare identity against available Mac evidence before running OaF.
4. **OaF broad:** `run-oaf-corpus --resume` with the native option pair. Assert the complete broad report has **182 successes** and **50 ms raw event-micro F1 = 0.234846**; do not accept an incomplete or changed-population result as parity.
5. **Human checkpoint:** `prepare-reviewed-subset`, display/download its CSV, stop execution, let a human listen and fill the decisions, then read the completed Drive CSV and `finalize-reviewed-subset`. Compare the 30-song candidate slate with previously reviewed evidence if available; changed slate requires a new explicit review, not an automatic accept. Score the broad run over the accepted subset using `score-oaf-reviewed-subset`.
6. **Separated views:** prepare/attest Spleeter and HTDemucs, run `run-oaf-separation-pilot --resume`, compare the canonical full-mix/Spleeter/HTDemucs reports, then `finalize-oaf-separation-pilot` with a **human-provided** decision/rationale. Native stems and their OaF inference stay on the same Colab VM.
7. **Publish:** call the merged `crux benchmark publish-mlflow-cohort --reports PATH --scope reviewed|pilot --experiment crux-benchmark` separately on completed **reviewed full-mix** (`reviewed`), **Spleeter** (`pilot`), and **HTDemucs** (`pilot`) report directories. Use the optional MLflow extra, a verified HTTPS tracking URI, and credentials from Colab Secrets. Do not disable TLS verification or allow redirects. Do not run publication while any report directory is being regenerated: HPA-86 snapshots the five canonical report files but explicitly does not promise an atomic snapshot across concurrent writes. Check that the three runs carry distinct `crux.input_view_id` values in the same experiment and that each repeated publish returns the **same run ID and `created=false`**. MLflow failure must not undo canonical reports.

The notebook's human checkpoint is deliberate. Do not auto-fill review CSVs or make an automated choice for final pilot disposition.

## Merged MLflow operational contract

The HPA-86 implementation on `main` explicitly requires an HTTPS tracking URI, rejects insecure TLS and redirects, validates the artifact destination, uploads only five canonical report files, then downloads them to verify exact bytes. Its `created`/`run_id` publication result is the acceptance evidence. A matching `RUNNING`/`SCHEDULED` publication may be resumed by the existing publisher; do not add a HPA-856 retry layer.

Reuse the Colab Secrets pattern in `notebooks/mlflow_publish_colab.ipynb`, but call the existing CLI from this notebook only **after** all three report directories are stable. Never make a Colab session's publication success a prerequisite for canonical report completion. Follow `docs/benchmark/mlflow.md` if provider behavior changes.

## Acceptance and failure policy

- Native OaF smoke returns the same checked-in oracle events and ready identity as Docker; actual Colab broad metrics and population exactly match the ticket's baseline.
- Running without native options still generates precisely the existing Docker launch command and preserves its tests.
- An incomplete/malformed native option pair or native runtime fails before inferred work and does not fabricate a successful run.
- The subset is human reviewed; separation runtime attestation passes **before** the pilot's mutable outputs; the pilot completes and is finalized with both derived reports.
- Across a fresh VM reset, a resumed stage uses the same pinned commit, stable paths, manifest/model/runtime identities, and already published per-song artifacts; no silent re-freeze or re-scoring under changed locks.
- The three canonical report views are visible side by side through the **merged** HPA-86 publisher; the only uploaded files are `summary.json`, `summary.md`, `items.csv`, `per_song.csv`, and `per_class.csv`. Repeated publication is idempotent and reports remain canonical even if the MLflow server fails.
- CI covers CLI selection, command/cwd/env construction, Docker default parity, missing files/invalid paths, and backend-factory forwarding with fakes. One manual Colab smoke and end-to-end evidence run is required; CI does not run a full corpus or contact R2/DagsHub.

## Boundaries, dependency, and risks

- No fine-tuning, model changes, generic runner abstraction, GPU OaF, Docker removal, TF1-to-TF2 port, network sandbox, model-lock format change, new scoring/report schema, automated human review, or MLflow duplication.
- HPA-86 / PR #34 is **already merged into `main`**, so publication is no longer blocked on another PR. Build HPA-856 on the merged CLI and tests, particularly `src/cli/benchmark.py` and `tests/test_cli_benchmark.py`, which HPA-86 extended. Keep both tickets' behavior on one updated HPA-856 PR branch; do not copy or reimplement MLflow.
- The largest risk is legacy Python 3.7 / TF1.15.5 native compatibility with modern Colab. Prove it before building the rest. The next risks are Drive filesystem semantics/quotas and separator-environment identity across VM resets. Fail explicitly and record evidence if a gate cannot be met; do not weaken source/lock verification to make the notebook appear to succeed.
