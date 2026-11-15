# HPA-856 — Docker-free OaF Benchmark on Colab: Design

**Linear:** [HPA-856](https://linear.app/cwchanap/issue/HPA-856/run-the-oaf-benchmark-pipeline-on-colab-without-docker)  
**PR:** [#35](https://github.com/cwchanap/Crux/pull/35), one ticket / one PR; this revision resolves the 2026-11-13 design review.  
**Baseline reviewed:** main `f5e0d182`, including merged HPA-86 / PR #34.  
**Status (2026-11-15):** A real Colab CPU VM ran, but Task 0 stopped during Python **3.7.17 provisioning**, before TensorFlow, worker, oracle, or Drive checks. A single Python 3.7.16 provisioning retry is planned; **no native runtime parity is established**.

## Goal and constraints

Run the same R2 → authoritative reference → OaF full mix → human-reviewed subset → Spleeter + HTDemucs pilot pipeline on Colab **without Docker**. Preserve source/model/scoring provenance, canonical reports, and resumable evidence across Colab VM resets. Keep Docker the default on existing hosts. Publish the resulting three reviewed/pilot report views through **the already-merged** HPA-86 publisher, not through another MLflow implementation.

No TF2 port, new transcription model, new scoring/report schema, generic runner/plugin architecture, GPU OaF, automatic listening review, or native Docker-equivalent network/read-only sandbox.

## Current seams and reuse

- `src/benchmark/backends/oaf.py` has one persistent `OafBackend` with `_ensure_process()`, `build_docker_command()`, `_validate_ready()`, and the checked-in `runtime/oaf_tf1/model.json` descriptor.
- `src/benchmark/worker_process.py::WorkerProcess.start()` already accepts `cwd` and `env`. `runtime/oaf_tf1/worker.py` accepts **relative** audio paths and `--checkpoint-dir`.
- `src/benchmark/backends/idm.py` already has `_isolated_worker_environment()` and `_factory_accepts_keyword()` helpers. Move these small shared functions into `worker_process.py`; import them from IDM and OaF rather than duplicating implementations. Preserve IDM behavior and OaF's existing `_build_factory_kwargs` timeout forwarding.
- `runtime/oaf_tf1/apply_instrumentation_patch.py::apply_reviewed_patch()` already validates the exact patch and source preimages, then writes the patched source tree atomically. Reuse it for native OaF, **not** a notebook-authored copy/patch implementation.
- `create_backend()` already consumes `CRUX_OAF_CHECKPOINT_CACHE` / `CRUX_OAF_INPUT_ROOT`. The corpus and separation runners use `create_backend` by default; the separation pilot's `_bind_oaf_backend_factory` overrides checkpoint/descriptor from the frozen parent run. **Exception:** `smoke-backend` explicitly passes `Path("artifacts/benchmark/model-cache").resolve()/sha256/<checkpoint-sha>` and does not honor `CRUX_OAF_CHECKPOINT_CACHE`. Keep the CLI unchanged: the Colab setup mounts the Drive cache through a symlink at that default location, so both smoke and corpus use the same checkpoint bytes.
- `src/benchmark/separation_pilot.py::SEPARATOR_LOCK_PATHS` is fixed at `<repo>/runtime/separators/{spleeter,htdemucs}/model.json`. Its v2 companion `environment.json` is loaded from the same folder. Existing `attest_separator_runtime()` checks interpreter, installed distributions, and model roots on **every pilot call**.
- Merged HPA-86 already supplies `publish-mlflow-cohort`, optional `mlflow` extra, `docs/benchmark/mlflow.md`, and `notebooks/mlflow_publish_colab.ipynb`.

## Native launch: one environment selector, one backend branch

Use **one** opt-in setting: `CRUX_OAF_NATIVE_PYTHON`, read in `create_backend()` and represented internally as `native_python: Path | None`. **Unset = unchanged Docker.** Set = direct host Python 3.7 worker. Invalid/missing interpreter must fail clearly rather than falling back to Docker.

No `--oaf-native-*` CLI flags, CLI plumbing, request fields, generic launcher hierarchy, second bundle-root setting, or new run-identity field. The active entry points are `smoke-backend`, `run-oaf-corpus`, and `run-oaf-separation-pilot`. `verify-backend` and `transcribe-one` remain removed-command stubs that raise `ClickException`; they are **not** part of native-mode acceptance.

On the first `OafBackend._ensure_process()` in native mode:

1. Build a temporary bundle **owned by this backend** using checked-in `worker.py`, `model.py`, `model.json`, and the pristine vendored Magenta tree. Create `<temp>/runtime` and `<temp>/upstream/magenta`, copy the checked-in files and pristine source, then invoke `apply_reviewed_patch(source_root=<temp>/upstream, destination_root=<temp>/runtime/vendor, patch_path=<checked-in patch>)`. **Leave `<temp>/runtime/vendor` uncreated**: the existing helper requires its parent to exist as a real directory, but refuses an already-existing destination. Only one implementation of bundle construction and patching, exercised in CI. No stale bundle survives `close()`.
2. Set `runtime/oaf_tf1/model.py::DEFAULT_UNINSTRUMENTED_SEQUENCE_LIB` relative to its own `__file__`: in Docker this still resolves to `/opt/crux/upstream/magenta/music/sequences_lib.py`; in the bundle it resolves to `<temp>/upstream/magenta/music/sequences_lib.py`. Preserve the pristine-vs-patched parity check.
3. Resolve **both** `checkpoint_dir` and `input_root` to absolute paths *before* launching. This matters because the pilot's checkpoint binding and the default `CRUX_OAF_CHECKPOINT_CACHE` may be relative, while the native subprocess must run with `cwd=<absolute input_root>`.
4. Launch `[<native_python>, "-s", "<temp>/runtime/worker.py", "--checkpoint-dir", "<absolute checkpoint_dir>"]`. Use the shared isolated environment helper to strip Python import-discovery variables, then overlay `CUDA_VISIBLE_DEVICES=-1`, `PYTHONHASHSEED=0`, and `NUMBA_CACHE_DIR=/tmp/numba-cache` from the Dockerfile. Do **not** rely on `-I` (breaks direct worker import) or `-E` (ignores `PYTHONHASHSEED`).
5. Keep the existing `_validate_ready()` (78 restored tensors), descriptor, request protocol, event decode, timeout handling, and close semantics. Pass `cwd`/`env` only when the injected process factory declares those keywords or accepts `**kwargs`; real `WorkerProcess.start()` supports both. Default Docker argv and process kwargs stay byte-for-byte unchanged.

The materializer's error path cleans up the temporary directory; `close()` cleans it on normal shutdown. Do not add separate notebook bundle-building, duplicate preflight of generated source trees, or a second persistent native runtime artifact. The committed patch and backend/ready contracts remain the integrity gates.

## Full-mix parity: use the recorded evidence, not two summary numbers

Canonical comparator: [`hpa86-evidence-broad` commit `599033663ca3bb495d8faaac13180206eaafadb4`](https://github.com/cwchanap/Crux/tree/599033663ca3bb495d8faaac13180206eaafadb4/evidence/oaf-broad), `evidence/oaf-broad/`. It contains **six** report files: `summary.json`, `summary.md`, `items.csv`, `per_song.csv`, `per_class.csv`, and `event_diagnostics.jsonl`.

Before running inference, after building the authoritative reference set, assert:
- `reference_manifest_sha256 = 96c9971b2c9113e72cb293862badcc88d9146e689ea329f45576d7fb7349f041`
- `reference_timing_version = sha256:10868dfb1f87490692d28f6f115dfa164f20be69247115be52a76ef59c954fcf`

After the corpus run, require the same `cohort_id = oaf-2d67684ad190163e`, population of **320 total / 182 success / 138 quarantine**, and **50 ms raw event-micro F1 = 0.234846**. Then compare **exact bytes** for **all six canonical report files**, including every per-song/per-class row and diagnostics. No tolerance-based or aggregate-only pass; a mismatch must be diagnosed before claiming parity. If the authoritative R2 inputs or identities have changed, stop and record the divergence rather than silently accept a different cohort.

The checked-in `tests/fixtures/oaf_tf1_smoke/smoke-oracle.json` and `src/benchmark/oaf_smoke_oracle.py` provide the earlier, cheaper **exact-event** native smoke gate.

## Task 0 evidence and narrow provisioning alternative (2026-11-15)

An authenticated Colab CPU VM was reached and measured at `5fc05db4`: Linux x86-64, glibc 2.39, notebook Python 3.13.18, two CPUs, 242381135872 bytes free, and Ubuntu's `libsndfile1` 1.2.2. The prescribed `micromamba ... python=3.7.17 pip` solve failed on conda-forge; a query against Anaconda `pkgs/main` also found **no Python 3.7.17**. This is a **package availability failure**, not a failed import, failed TensorFlow wheel, or failed OaF oracle.

Keep the Docker baseline at Python 3.7.17. For the Colab **host-only** worker, permit exactly one bounded alternative:

- Try **Python 3.7.16**, pinned to the available [Anaconda `pkgs/main` Linux x86-64 package](https://anaconda.org/anaconda/python/files/manage?version=3.7.16), using an explicitly selected channel and the same locked pip dependency files. This package is **listed**, not yet proven installable on the target VM.
- The [TensorFlow 1.15.5 PyPI Linux wheel](https://pypi.org/project/tensorflow/1.15.5/) uses `cp37-cp37m` rather than a particular CPython 3.7 patch release; **ABI compatibility makes a test reasonable but does not prove runtime or bit-exact inference compatibility**. Preserve the checked-in TensorFlow, NumPy and other hashes, Magenta patch, checkpoint and model descriptor; record the exact host interpreter version in evidence.
- Start at the environment solve and frozen dependency installation. If they work, run the same **exact smoke oracle** and subsequently the six-file broad report comparison before allowing shared benchmark identity. If they fail, stop and capture precise diagnostics; do not auto-try unbounded channels, compile an obsolete Python, port TF1 to TF2, modify the model identity, or implement a rootless-container alternative without revising this **same PR's** architecture.
- Real Colab Drive storage/durability and all OaF numerical/parity checks remain **unexecuted** until the interpreter is working. Task 1 production coding stays gated.

## Colab workflow: four isolated interpreters and two pins

Create `notebooks/oaf_benchmark_colab.ipynb` for benchmark **generation only**; use clear manual cells, no workflow engine.

- **Task 0 gate (before production coding):** the first real Colab attempt could not resolve Python 3.7.17. On the same kind of VM, try the **single pinned 3.7.16 host-only alternative** above, then install the frozen TF1.15.5 hashed requirements plus `libsndfile1`, temporarily mirror the Docker layout, and run exact smoke-oracle parity. Verify Drive-backed write/restore. If interpreter provisioning, frozen imports, smoke, or Drive fail, stop and record diagnostics. The unchanged Docker image remains Python 3.7.17; no runtime equivalence is assumed before proof.
- **Environment setup (only after the Task 0 gate passes):** separate Crux Python 3.12 (`uv sync --extra r2`), OaF exact pinned Python **3.7.16 if successfully validated**, Spleeter 2.4.2, HTDemucs 4.1.0. OaF uses CPU. Optional GPU only for HTDemucs. Do not downgrade the notebook interpreter. Credentials come from Colab Secrets.
- **Corpus stage:** pin a concrete `CORPUS_COMMIT` after the code changes are stable, record it in the notebook/workspace, check `git rev-parse HEAD`, and mount Drive at the same path on every VM. Place the checkpoint cache on Drive, and create the **single** symlink `artifacts/benchmark/model-cache -> <absolute Drive checkpoint-cache root>` inside the checkout. Set `CRUX_OAF_CHECKPOINT_CACHE` to that same Drive root and `CRUX_OAF_NATIVE_PYTHON` to the frozen Python 3.7 interpreter; retain stable absolute paths for the remaining artifacts. This accommodates `smoke-backend`'s explicit default checkpoint path without changing the CLI. Run R2 sync and reference commands, fail early on reference drift, then run `run-oaf-corpus --resume` and exact report parity.
- **Human review stage:** produce `review-candidates.csv`; **stop** for the 30-song listening review, accept only human-filled decisions, `finalize-reviewed-subset`, and score the reviewed full-mix cohort.
- **Separator/pilot stage:** build isolated separator environments and fixed model roots. Freeze both environments **once**, then commit **four small provenance files** to **this same HPA-856 PR**: `runtime/separators/{spleeter,htdemucs}/{model.json,environment.json}`. These are versioned **identities**, not venv archives/weights or stem data. Record the resulting `PILOT_COMMIT` and use that exact checkout for the pilot and all resumes. The corpus run's earlier commit remains recorded; the pilot includes its own `crux_commit` in the deterministic run ID. Never advance the pilot checkout partway through a partial run.
- **Pilot execution:** archive the actual separator interpreters/venvs and policy-owned model roots to private Drive storage and restore to the **same absolute paths** after VM reset. The existing v2 lock loader, runtime attestation, and run parsers perform identity verification. Notebook-side check: pinned commit and stable paths; no duplicate hand-rolled hash validator. Run `run-oaf-separation-pilot --resume`, inspect both derived views and comparison, and require a human `finalize-oaf-separation-pilot` decision/rationale.

`--resume` is identity-bound: the corpus checks `expected_run_id`, and the pilot run ID includes its separator lock hashes and **the pilot commit**. Re-freezing locks or changing the checkout creates a **different** identity rather than resuming the old run. Restore and pin; never accept a second run as resumed evidence.

## MLflow: hand off to the existing publication notebook

The benchmark notebook **ends when three canonical report directories have been placed on Drive**, ready for publication. It does not install MLflow or call the publisher.

Make only two scoped changes to `notebooks/mlflow_publish_colab.ipynb`:
1. When `REPORTS_ZIP` is unset, read a configured Drive `REPORTS_ROOT` directly; do **not** delete or unzip into that canonical directory. Keep existing ZIP support for HPA-86 users.
2. Republish **every** `REPORTS` entry and assert same `run_id` / `created=false`, rather than only checking the first. Configure the three directories/scopes for reviewed full mix, Spleeter pilot, and HTDemucs pilot.

The merged CLI owns HTTPS/TLS/redirect checks, staged five-file snapshot, artifact allowlist, byte-for-byte upload verification, and publication idempotency. The notebook must not redo those checks or publish concurrently with report regeneration. Do not upload raw audio, stems, predictions, model checkpoints, or credentials.

## Acceptance (real target + CI)

- [ ] Task 0 on the target Colab VM: after the unsuccessful 3.7.17 package solve, prove the pinned 3.7.16 candidate installs the **unchanged** TF1.15.5 runtime and matches the full checked-in native-event oracle exactly. Prove Drive durability. A failed provisioning/import/oracle gate is documented, not bypassed.
- [ ] The normal macOS/CI Docker path remains unchanged. Native opt-in works via one environment setting, builds/cleans its own bundle, resolves absolute paths, and isolates Python import environment variables. The Colab smoke command reads the **Drive-backed checkpoint** via its existing `artifacts/benchmark/model-cache` path; `verify-backend` and `transcribe-one` remain out of scope.
- [ ] Colab reference manifest/timing identities match the immutable baseline **before** inference; all six full-mix canonical report files match the baseline **exactly** afterward.
- [ ] The 30-song review and final pilot disposition remain human; both frozen separator lock+environment pairs are versioned in this PR **before** pilot scoring. Pilot resumes after a VM reset using the same pinned commit and attested restored runtimes.
- [ ] The existing MLflow notebook publishes three distinct report views, repeated publication reuses the corresponding runs, and no raw model/audio/stem artifacts are uploaded.
- [ ] CI verifies Docker default, native argv/cwd/env, temporary bundle content and cleanup, shared IDM helper compatibility, and the model source-path test; no R2, Colab, or DagsHub calls in CI.

**Implementation stays in this same PR.** Do not change benchmark scoring, report schemas, separator attestation policy, or HPA-86 publisher.
