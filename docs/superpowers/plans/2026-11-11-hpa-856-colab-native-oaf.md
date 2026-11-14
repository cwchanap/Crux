# HPA-856 — Docker-free OaF Benchmark on Colab: Implementation Plan

**Design:** [2026-11-11-hpa-856-colab-native-oaf-design.md](../specs/2026-11-11-hpa-856-colab-native-oaf-design.md)  
**Linear:** [HPA-856](https://linear.app/cwchanap/issue/HPA-856/run-the-oaf-benchmark-pipeline-on-colab-without-docker)  
**PR:** [#35](https://github.com/cwchanap/Crux/pull/35). **One ticket, one PR**: keep design, implementation, tests, and evidence here.  
**Reviewed baseline:** main `f5e0d182`; HPA-86/PR #34 is merged.  
**Status:** Planning only; real Colab Task 0 and production coding have **not** been executed.

## Scope decisions

- Docker remains default, with a **single** `CRUX_OAF_NATIVE_PYTHON` opt-in read in `create_backend()`. **No new CLI options, request fields, runner factories, or schema changes.**
- One backend-owned native temporary bundle, generated using existing `apply_reviewed_patch()`, not duplicated in a Colab cell. Reuse worker JSON-line protocol and common worker process utilities.
- Keep two reproducibility pins: `CORPUS_COMMIT` (after code stabilized) and `PILOT_COMMIT` (after the four v2 separator lock/companion files have been committed). The pilot's deterministic run ID includes its `crux_commit`; never advance the pilot checkout after starting it.
- One Colab notebook generates benchmark evidence; modify the already-merged publication-only notebook slightly to consume the results. No MLflow integration rewrite.
- Keep human review and final separation decision manual; no score-based model selection or trial-and-error tuning.

## Task 0 — Real-Colab legacy OaF compatibility gate (must precede coding)

**Evidence only:** Colab console/redacted logs, VM version, exact checked-out commit, interpreter/wheel versions and smoke output. Do **not** treat a CI fake as Colab feasibility.

- [ ] On an actual Colab VM record architecture, OS/glibc, notebook Python, CPU, available disk, and pinned repo revision. Provision a separate Python 3.7.17 via micromamba/conda without changing notebook Python.
- [ ] Install system `libsndfile1`. Install `runtime/oaf_tf1/requirements-build.lock` followed by `requirements.lock` with their existing required hashes and binary rules. No lock regeneration or upgraded TF.
- [ ] As a **temporary manual probe**, mirror `/opt/crux/runtime`, `/opt/crux/upstream/magenta`, and patched `runtime/vendor` from the exact committed files, using the existing instrumentation script. This is not the production bundle builder and does not go into the notebook.
- [ ] Acquire/hash-verify the canonical OaF checkpoint using the existing `prepare-backend` path. Launch the native Python 3.7 worker and assert the ready identity (78 restored tensors) plus **exact** native-event equality to `tests/fixtures/oaf_tf1_smoke/smoke-oracle.json` using `src/benchmark/oaf_smoke_oracle.py`. Compare against the known Docker fixture/oracle, not import-only startup.
- [ ] Prove a small Drive-backed write/restore on this VM is compatible with the report publisher's durability assumptions. Verify it persists after remount where possible.
- [ ] **STOP if blocked:** record the actual Python/TF/patch/Drive failure in this same PR and HPA-856. Do not silently switch models or weaken parity.

**Go/no-go:** the frozen worker and checkpoint operate and pass the exact smoke oracle on Colab.

## Task 1 — One native launcher and a tested bundle materializer

**Modify:** `src/benchmark/backends/oaf.py`, `runtime/oaf_tf1/model.py`, `src/benchmark/worker_process.py`, `src/benchmark/backends/idm.py`.  
**Tests:** `tests/benchmark/test_task_d_contract.py`, `tests/benchmark/test_worker_process.py`, `tests/benchmark/test_idm_backend.py`, `tests/runtime/oaf_tf1/test_model.py`. A focused materializer test can live in the existing OaF contract test module.

- [ ] Move IDM's `_isolated_worker_environment()` and `_factory_accepts_keyword()` into the shared worker-process module; have IDM import them without changing observable behavior. Reuse OaF's existing `_build_factory_kwargs()` for timeout keywords.
- [ ] Add `native_python: Path | None` as one optional `OafBackend` constructor field; `create_backend()` reads `CRUX_OAF_NATIVE_PYTHON` when not explicitly supplied. Unset -> existing `build_docker_command` and original process kwargs. Invalid native setting -> an actionable backend startup failure, never Docker fallback.
- [ ] Add one small function that materializes `<tmp>/runtime/{worker.py,model.py,model.json,vendor/magenta/...}` and `<tmp>/upstream/magenta/...` from checked-in files. Create **`<tmp>/runtime` before calling** `apply_reviewed_patch(source_root=<tmp>/upstream, destination_root=<tmp>/runtime/vendor, patch_path=...)`, but **do not create `<tmp>/runtime/vendor`**: the existing patch helper rejects an existing destination and requires its parent to be a regular directory. The backend owns a `TemporaryDirectory` for one worker lifecycle; release it in `close()` and on startup failure. Do not add a bundle path flag, manifest format, or redundant post-build file inventory.
- [ ] Adjust `DEFAULT_UNINSTRUMENTED_SEQUENCE_LIB` to resolve relative to `model.__file__`: Docker continues to resolve `/opt/crux/upstream/magenta/music/sequences_lib.py`; the native bundle resolves its sibling `upstream`. Update `tests/runtime/oaf_tf1/test_model.py:118` to assert the relative **path formula** rather than the old literal `/opt/crux` path (the unbundled repo has no `runtime/upstream` tree).
- [ ] Before launch, resolve checkpoint and input root to **absolute** paths. Native argv is `[native_python, "-s", "<bundle>/runtime/worker.py", "--checkpoint-dir", "<absolute checkpoint>"]` with `cwd=<absolute input root>`. Strip inherited `PYTHONPATH`, `PYTHONHOME`, and other Python discovery variables using the reused helper; set Dockerfile CPU/env values (`CUDA_VISIBLE_DEVICES=-1`, `PYTHONHASHSEED=0`, `NUMBA_CACHE_DIR=/tmp/numba-cache`). No `-I` or `-E`.
- [ ] Forward `cwd`/`env` only when the injected factory supports those keywords, preserving existing fake factories. Keep the worker ready check, mapping, timeouts, descriptor, and error behavior. No Docker command changes.
- [ ] Add tests asserting unchanged exact Docker argv, native argv/cwd/env, **absolute checkpoint** despite relative cache configuration, controlled factory keyword forwarding, missing native interpreter/patch error, and cleanup after close/failure. Add one real **CI** test that calls the bundle materializer on the committed source/patch, asserts `<tmp>/runtime` exists and `<tmp>/runtime/vendor` is **absent before** applying the patch, then checks the patched/pristine layout and cleanup. Run the existing IDM tests after moving helpers.

**Acceptance:** unit tests pass, Docker default unchanged, one native worker can be started without exposing notebook import paths.

## Task 2 — Select native mode from the existing OaF environment

**No CLI code changes** in `src/cli/benchmark.py`. No new CLI-specific tests.

- [ ] Set `CRUX_OAF_NATIVE_PYTHON=/absolute/path/to/python3.7` in Colab before invoking the existing commands. Use a stable absolute Drive-backed `CRUX_OAF_CHECKPOINT_CACHE` and benchmark artifact roots; clear native env var for Docker runs. **Exception:** the active `smoke-backend` currently overrides `checkpoint_dir` with the repository-relative `artifacts/benchmark/model-cache` path. In Colab, symlink **only** that default model-cache path to the same absolute Drive cache, then use the existing smoke command unchanged. Do not broaden this ticket by changing CLI cache selection.
- [ ] Confirm `CRUX_OAF_NATIVE_PYTHON` reaches `smoke-backend`, `run-oaf-corpus`, and `run-oaf-separation-pilot` through the current factory; for smoke, first verify that the default checkpoint path resolves through the Drive symlink to the exact same cache. The pilot's existing `_bind_oaf_backend_factory` still overrides checkpoint and descriptor from its parent run. **Do not test `verify-backend` or `transcribe-one` as native flows**: both are removed stubs that unconditionally raise `ClickException`.
- [ ] Confirm runtime mode is **not** serialized in OaF/pilot identity, and invalid native configuration fails rather than silently using Docker.

**Acceptance:** active smoke/corpus/pilot command lines remain unchanged, select native mode through the single opt-in environment variable, and smoke resolves its existing hard-coded checkpoint path to the Drive cache. Removed-command stubs are excluded.

## Task 3 — Build one Colab benchmark notebook and concise runbook

**Create:** `notebooks/oaf_benchmark_colab.ipynb`, `docs/benchmark/colab.md`.  
**Modify:** `README.md` (add link beside existing HPA-86 MLflow docs).

- [ ] Notebook setup: mount Drive, read R2 Secrets, pin `CORPUS_COMMIT`, check `git rev-parse HEAD`, establish stable absolute Drive-backed cache/run/manifests and env vars, print non-secret environment/commit evidence. Link `artifacts/benchmark/model-cache` to the Drive checkpoint cache **without** replacing an existing populated local directory; ensure both the smoke command's resolved path and `CRUX_OAF_CHECKPOINT_CACHE` point at that same root. Do not clone moving `main` on resume.
- [ ] Install four **separate** environments: Crux Python 3.12 with `uv sync --extra r2`, frozen Python 3.7 OaF, Spleeter 2.4.2, HTDemucs 4.1.0. OaF CPU; HTDemucs GPU optional.
- [ ] **Do not** build the OaF bundle in the notebook: the production backend does that on first worker launch. The notebook has no duplicate file-layout logic.
- [ ] Resumable manual cells for R2 sync, chart selection, timing, reference set, full mix, reviewed-subset preparation/human pause, separator freeze/commit/repin, pilot run/human finalize, and three Drive report-folder handoff. Do **not** call MLflow here.
- [ ] Save source and reference artifacts exactly. For separator venvs/model roots, archive them privately on Drive, restore at the **same absolute paths** on a later VM, and let current CLI v2 attestation verify their bytes; do not add an independent notebook hash gate.
- [ ] Report the current commit pin on every run. Upon switching from corpus to pilot, require recorded `PILOT_COMMIT`. No hidden scheduling or background jobs.

**Acceptance:** notebook JSON validates; each manual cell is readable/restartable and the CLI does the authoritative identity checking.

## Task 4 — Exact reference identity + broad full-mix parity

**Comparison source:** `hpa86-evidence-broad` at commit `599033663ca3bb495d8faaac13180206eaafadb4`, `evidence/oaf-broad/` (fetch that **exact** commit, not moving branch contents).

- [ ] Run `sync-r2-corpus`, `select-reference-charts`, `build-reference-timing`, and `build-reference-set` using the persisted Drive workspace.
- [ ] **Before** OaF inference, assert the regenerated `reference_manifest_sha256` is exactly `96c9971b2c9113e72cb293862badcc88d9146e689ea329f45576d7fb7349f041` and `reference_timing_version` is exactly `sha256:10868dfb1f87490692d28f6f115dfa164f20be69247115be52a76ef59c954fcf`. If a source identity changed, stop and diagnose the R2/reference input (do not hide drift by accepting approximate metrics).
- [ ] Run the unchanged `run-oaf-corpus ... --resume` command with `CRUX_OAF_NATIVE_PYTHON` set. Assert `cohort_id=oaf-2d67684ad190163e`, 320 total / **182 success** / 138 quarantined, and raw 50 ms event-micro F1 **0.234846**.
- [ ] Compare **exact bytes / SHA-256** for **all six** files in `evidence/oaf-broad/` to the newly scored report directory: `summary.json`, `summary.md`, `items.csv`, `per_song.csv`, `per_class.csv`, `event_diagnostics.jsonl`. No ignored rows, timestamps, tolerances, or aggregate-only match. Publish an evidence table naming every file and whether it matches.
- [ ] Run `prepare-reviewed-subset`, **stop** for real human review, import the completed CSV from Drive, then `finalize-reviewed-subset` and `score-oaf-reviewed-subset`. Compare the 30-song slate with the existing review evidence if available; never auto-approve a modified slate.

**Acceptance:** exact reference lineage and six-file full-mix parity are proven before moving to the separator pilot. Human review remains explicit.

## Task 5 — Freeze, commit, pin, execute, and finalize the separator pilot

**Commit new source-controlled evidence (only when real Colab environments exist):**
- `runtime/separators/spleeter/model.json`
- `runtime/separators/spleeter/environment.json`
- `runtime/separators/htdemucs/model.json`
- `runtime/separators/htdemucs/environment.json`

- [ ] Install/restore both existing fixed separator environments/model roots. Use `freeze-separator-runtime` and the current v2 probe/policy; **do not** modify `src/benchmark/separators.py` or generated output schemas.
- [ ] Commit the four lock+companion files to **this same PR**, before running any pilot scoring; do not commit venvs, checkpoint bytes, model roots, stems, or tokens. Record and push that commit as `PILOT_COMMIT`; the corpus stage used `CORPUS_COMMIT`.
- [ ] Checkout the exact pilot commit and **keep it pinned across all pilot resumes**. The pilot run identity includes `crux_commit` plus both separator lock hashes; advancing the checkout during a partial pilot produces another run, not a continuation. The canonical parent OaF run remains on Drive and still binds the same model/reference/timing identity.
- [ ] Run unchanged `run-oaf-separation-pilot --manifest ... --timing-manifest ... --subset-manifest ... --oaf-run ... --cache-dir ... --output-dir ... --spleeter-python ... --demucs-python ... --spleeter-model-root ... --demucs-model-root ... --resume`. All required paths absolute; the existing CLI preflights v2 companion, interpreter distributions and model-root hashes.
- [ ] On a new Colab VM, restore the two Python environments and model roots at their same absolute locations; checkout the exact `PILOT_COMMIT`; rerun `--resume` and verify the *same* pilot run ID and preserved predictions/stems. If identity fails, stop rather than re-freeze or start a different run.
- [ ] Inspect full-mix control, Spleeter/HTDemucs derived reports and paired comparison. Finalize using a **human-provided** `finalize-oaf-separation-pilot --decision ... --rationale ...`.

**Acceptance:** both views finish, provenance locked before scoring, same-run resume verified, and the human-reviewed pilot is finalized.

## Task 6 — Reuse existing publication notebook; regression and closeout

**Modify:** `notebooks/mlflow_publish_colab.ipynb` only (two small behaviors). **No changes** to `src/benchmark/mlflow_export.py` or its tests unless a new defect is actually demonstrated.

- [ ] Allow `REPORTS_ZIP` to be unset and `REPORTS_ROOT` to point **directly** to the three stable Drive report folders. The unzip/delete command must run **only** for the existing ZIP path and **never** remove or alter a direct Drive report directory. Keep existing ZIP behavior for HPA-86.
- [ ] Set `REPORTS` to the reviewed full-mix / Spleeter pilot / HTDemucs pilot folder-scope mapping; repeat publication for **every** entry (not only the first), asserting each existing `run_id` is reused and `created=false`. Rely on HPA-86 for HTTPS/auth safety, five-file snapshot/upload/checksum validation, and error handling. No duplicate notebook-side API checks.
- [ ] Ensure all report generation is complete **before** publication; HPA-86's snapshot is not atomic against concurrent writers. Evidence must show three distinct `crux.input_view_id` values under one `crux-benchmark` experiment and no raw audio/stems/predictions/checkpoints uploaded.
- [ ] Run offline CI gates: `uv run pytest`, `uv run ruff check .`, `uv run ruff format --check src tests`, `uv run pylint --errors-only src`, plus notebook JSON validation. Regress the unchanged macOS Docker behavior and shared IDM helper behavior.
- [ ] Verify this **same PR** contains scoped code, tests, both notebooks/docs and the four frozen separator companion files; no credentials or large binaries. Record real Colab Task 0, exact report diff, pilot resume, and MLflow run IDs. Mark ready for review only when acceptance is met.

## Scoped implementation surface

| Area | Paths |
| --- | --- |
| Native worker/bundle, environment helper reuse | `src/benchmark/backends/oaf.py`, `src/benchmark/worker_process.py`, `src/benchmark/backends/idm.py`, `runtime/oaf_tf1/model.py` |
| Tests | `tests/benchmark/test_task_d_contract.py`, `tests/benchmark/test_worker_process.py`, `tests/benchmark/test_idm_backend.py`, `tests/runtime/oaf_tf1/test_model.py` |
| Colab run | `notebooks/oaf_benchmark_colab.ipynb`, `docs/benchmark/colab.md`, `README.md` |
| Existing publication handoff | `notebooks/mlflow_publish_colab.ipynb` |
| Frozen runtime identities produced at Task 5 | `runtime/separators/{spleeter,htdemucs}/{model.json,environment.json}` |
| Planning (already in PR) | This plan and the linked design |

No new OaF CLI flags, CLI tests, scoring changes, report schemas, separator attestation policy, MLflow exporter or second PR.
