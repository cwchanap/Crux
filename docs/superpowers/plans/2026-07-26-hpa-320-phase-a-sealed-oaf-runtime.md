# HPA-320 Phase A Sealed OaF Runtime Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use
> superpowers:subagent-driven-development (recommended) or
> superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox
> (`- [ ]`) syntax for tracking.

**Goal:** Reproduce the released Magenta E-GMD Onsets and Frames Drums model in an
isolated TensorFlow 1.15.5 `linux/amd64` runtime, prove 78/78 inference-tensor
restoration, seal every inference-relevant byte, and expose `prepare-backend` and
`verify-backend` without fallback.

**Architecture:** The Python 3.12 host loads strict locks, validates native-host
evidence, launches one hardened persistent container, drains stderr concurrently,
and speaks one-line JSON over stdin/stdout. The Python 3.7 container vendors the
frozen Magenta source, constructs the exact `drums` PREDICT graph, instruments only
the emitted-frame metadata path, restores the official checkpoint, and returns native
events. A disposable native-`amd64` calibration pass creates the oracle, exact
resource profile, evidence record, and final locks; no guessed values flow past that
gate.

**Tech Stack:** Python 3.12 host, Python 3.7.17 container,
TensorFlow 1.15.5 CPU, Docker/OCI `linux/amd64`, Click 8, Pytest, SHA-256,
standard-library subprocess/threading/zipfile/wave/json.

## Global Constraints

- Prerequisite: complete Phase B Tasks 1-6 so `TranscriptionBackend`,
  `BackendDescriptor`, `BackendVerification`, reports, publication, registry, and
  prediction artifacts already exist.
- Backend ID: `magenta-egmd-tf1-94529798-8hit-v1`.
- Model ID: `magenta-egmd-ckpt-569400-v1`.
- Architecture ID: `magenta-oaf-model-tpu-drums-v1`.
- Native output space: `magenta-oaf-midi88-a0-v1`.
- Training map: `magenta-egmd-data-8hit-94529798-v1`.
- Native metadata schema: `magenta-oaf-native-metadata-v1`.
- Magenta source commit:
  `94529798dfbbb14c27ddfd76f23027dc8e2ce185`.
- Checkpoint archive SHA-256:
  `09765ae0ff19c7d769a3c20e158eba3b9cd279429b02e498b1e911d16f82e2c0`.
- Index SHA-256:
  `475ca21993e102d52e2e6c7066b598b9b9e9a097bfb2fa6b770d40eedaa6139a`.
- Data SHA-256:
  `6312cd6f8e29ee6f21f4a763822d554f46393e0e336812ba4a8cb324d2e8b0b5`.
- Meta SHA-256:
  `e36105b27c17f2367292682df825b8ad16eb887ccad465bbb5ddf955ef75d422`.
- Observed legacy HDF5 audit SHA-256:
  `d36ced8b2ee241bc37ad6fbb918ba38e95d666350dd4888bca59a1243bf4d10e`.
- Runtime: `python:3.7.17-slim-bullseye` on `linux/amd64`, base manifest
  `sha256:ea8897698c0955ba96144bd2b7310ef7884ccce4db7a1f97ffc21fb8b89d1673`,
  `tensorflow==1.15.5`, wheel SHA-256
  `29831dda98d668067de75403b2fca0d06a2f026ef6f217fa2ca873c20b4ee4d3`.
- Official execution is CPU-only, native `linux/amd64`, non-root,
  `--network=none`, read-only root/mounts, capabilities dropped,
  `no-new-privileges`, and exact CPU/memory/PID/tmpfs/deadline bounds.
- The launcher, not in-process Python code, supplies `PYTHONHASHSEED=0` before
  CPython starts. The runner asserts the complete environment before TensorFlow or
  NumPy imports.
- The fresh environment is exactly `CUDA_VISIBLE_DEVICES=-1`,
  `OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1`,
  `PYTHONHASHSEED=0`, `TF_NUM_INTRAOP_THREADS=1`, and
  `TF_NUM_INTEROP_THREADS=1`; missing, extra, or changed variables fail startup.
- After environment validation, the runner calls `random.seed(0)`,
  `numpy.random.seed(0)`, and `tf.set_random_seed(0)` before graph construction.
- `ConfigProto` sets inter-op and intra-op threads to `1`; matching
  `TF_NUM_INTEROP_THREADS=1` and `TF_NUM_INTRAOP_THREADS=1` remain locked fallback
  controls.
- Input uses `spec_hop_length=512`; the separate `hop_length=441` hparam is recorded
  but never substituted for feature extraction.
- `drum_prediction_map=''`; inference emits all 88 MIDI-relative bins.
- No partial restore, random initialization, mutable checkpoint lookup, network
  inference, alternate URL, HDF5 substitution, hidden chunking, or heuristic
  fallback.
- No training, fine-tuning, weight modification, architecture replacement, or
  result-driven threshold tuning.
- Raw NaN velocity is a named identity-bound adapter failure, even though upstream
  converts that case to zero.
- Final lock, resource, deadline, UID/GID, archive-size, package, smoke, and
  `max_input_audio_frames` values come only from Task 8's accepted native evidence.
  Sentinel values and prose defaults are forbidden.
- Python lines stay within 100 characters.

## Execution Order

Tasks 1-7 are implementable on the current host with fake runners, injected lock
fixtures, and emulated diagnostics. Task 8 requires an accepted native
`linux/amd64` worker and is the only point allowed to publish final locks, the smoke
oracle, seal evidence, or a verified real-checkpoint result. Stop at Task 8 if that
worker or its accepted attestation is unavailable; do not fill exact fields by
guessing.

---

## File Map

### New host source

- `src/benchmark/backend_lock.py` — strict backend/runtime/seal/audit lock loading
  and cross-identity validation.
- `src/benchmark/backend_prepare.py` — explicit offline-cache verification and
  opt-in archive acquisition.
- `src/benchmark/backend_process.py` — native-host evidence, hardened container
  command, persistent protocol process, deadlines, and bounded stderr drain.
- `src/benchmark/backends/oaf_tf1.py` — OaF adapter, preflight, handshake, smoke,
  request mapping, and backend-fatal classification.

### New runtime and tooling

- `runtime/oaf_tf1/Dockerfile`
- `runtime/oaf_tf1/entrypoint.py`
- `runtime/oaf_tf1/protocol.py`
- `runtime/oaf_tf1/oaf_backend.py`
- `runtime/oaf_tf1/requirements.in`
- `runtime/oaf_tf1/requirements.lock`
- `runtime/oaf_tf1/requirements-test.in`
- `runtime/oaf_tf1/requirements-test.lock`
- `runtime/oaf_tf1/source-manifest.json`
- `runtime/oaf_tf1/runner-source-manifest.json`
- `runtime/oaf_tf1/host-adapter-source-manifest.json`
- `runtime/oaf_tf1/patches/capture-emitted-frame.patch`
- `runtime/oaf_tf1/vendor/magenta/` — exact frozen source subset.
- `tools/hpa320/audit_legacy_tf2_conversion.py`
- `tools/hpa320/vendor_magenta.py`
- `tools/hpa320/resolve_oaf_runtime.py`
- `tools/hpa320/generate_smoke_fixture.py`
- `tools/hpa320/seal_oaf_backend.py`

### Generated, reviewed configuration/evidence

- `docs/superpowers/evidence/hpa-320/legacy-tf2-conversion-coverage.json`
- `docs/superpowers/evidence/hpa-320/oaf-tensor-coverage.json`
- `docs/superpowers/evidence/hpa-320/oaf-security-scan.json`
- `docs/superpowers/evidence/hpa-320/oaf-oci-layout-manifest.json`
- `config/benchmark/backends/magenta-egmd-tf1-94529798-8hit-v1.backend-lock.json`
- `config/benchmark/backends/magenta-egmd-tf1-94529798-8hit-v1.runtime-lock.json`
- `config/benchmark/backends/magenta-egmd-tf1-94529798-8hit-v1.seal-evidence.json`
- `tests/fixtures/oaf_tf1_smoke/generator-parameters.json`
- `tests/fixtures/oaf_tf1_smoke/canonical.wav`
- `tests/fixtures/oaf_tf1_smoke/smoke-oracle.json`

### New tests

- `tests/benchmark/test_backend_lock.py`
- `tests/benchmark/test_backend_prepare.py`
- `tests/benchmark/test_backend_process.py`
- `tests/benchmark/test_oaf_backend.py`
- `tests/benchmark/test_oaf_real_checkpoint.py`
- `tests/benchmark/test_oaf_seal.py`
- `tests/benchmark/test_legacy_conversion_audit.py`
- `tests/benchmark/test_oaf_smoke_generator.py`
- `tests/fixtures/fake_oaf_runner.py`
- `runtime/oaf_tf1/tests/test_protocol.py`
- `runtime/oaf_tf1/tests/test_instrumentation.py`
- `runtime/oaf_tf1/tests/test_tensor_coverage.py`

### Modified files

- `src/benchmark/backend_registry.py` — lazy OaF factory becomes available.
- `src/cli/benchmark.py` — add `prepare-backend` and `verify-backend`.
- `tests/test_cli_benchmark.py` — setup/verification summaries and exit behavior.
- `docs/drumery-dtx-midi-benchmarking-reference.md` — operator and evidence contract.
- `.github/workflows/oaf-backend-integration.yml` — dedicated native integration.

### Cross-task interfaces

- Locking: `LoadedBackendLock`, `LoadedRuntimeLock`, `LoadedSealEvidence`,
  `LoadedConversionAudit`, `load_backend_lock`, `load_runtime_lock`,
  `load_seal_evidence`, `load_conversion_audit`, `validate_oaf_lock_set`.
- Preparation: `PrepareBackendRequest`, `PrepareBackendOutcome`,
  `prepare_oaf_backend`.
- Process: `NativeHostEvidence`, `RunnerLaunchProfile`, `RunnerHandshake`,
  `RunnerResponse`, `RunnerProcess`, `build_docker_command`.
- OaF: `OafBackendConfig`, `OafTf1Backend`, `create_backend`.
- Runtime protocol: `PROTOCOL_SCHEMA`, `load_and_verify_locks`,
  `build_and_restore_model`, `transcribe_canonical_wav`, `main`.

---

### Task 1: Implement strict OaF locks and evidence cross-checking

**Files:**

- Create: `src/benchmark/backend_lock.py`
- Create: `tests/benchmark/test_backend_lock.py`

**Interfaces:**

- Consumes: Phase B strict JSON/SHA helpers.
- Produces: all lock/evidence records and loaders listed above.

- [ ] **Step 1: Write failing strict-schema and cross-hash tests**

```python
def test_backend_lock_rejects_missing_audio_frame_bound(tmp_path: Path) -> None:
    payload = valid_backend_lock_payload()
    del payload["max_input_audio_frames"]
    path = write_json(tmp_path / "backend.json", payload)

    with pytest.raises(BackendLockError, match="backend lock fields"):
        load_backend_lock(path)


def test_lock_set_rejects_runtime_hash_mismatch(tmp_path: Path) -> None:
    backend = load_backend_lock(write_backend_lock(tmp_path, runtime_hash="a" * 64))
    runtime = load_runtime_lock(write_runtime_lock(tmp_path))
    seal = load_seal_evidence(write_seal_evidence(tmp_path))
    audit = load_conversion_audit(write_conversion_audit(tmp_path))

    with pytest.raises(BackendLockError, match="runtime lock SHA-256 mismatch"):
        validate_oaf_lock_set(backend, runtime, seal, audit)
```

Add tests for duplicate/unknown keys, wrong schema, uppercase hashes, archive/component
size/hash mismatch, duplicate checkpoint names, tensor inventory overlap, counts
other than 130/78/52, missing exact hparams, nonempty prediction map, incorrect
88-bin arithmetic, invalid 8-hit groups, metadata schema drift, zero/sentinel
resource values, final-lock hashes appearing inside seal evidence, image/platform
mismatch, source/runtime manifest mismatch, smoke/audit evidence mismatch, and
descriptor reproduction.

- [ ] **Step 2: Run tests and verify the missing-module failure**

```bash
rtk uv run pytest -q tests/benchmark/test_backend_lock.py
```

Expected: collection FAIL.

- [ ] **Step 3: Implement strict loaded records**

```python
@dataclass(frozen=True)
class LoadedBackendLock:
    path: Path
    payload: Mapping[str, JsonValue]
    sha256: str
    descriptor: BackendDescriptor
    max_input_audio_frames: int


@dataclass(frozen=True)
class LoadedRuntimeLock:
    path: Path
    payload: Mapping[str, JsonValue]
    sha256: str


@dataclass(frozen=True)
class LoadedSealEvidence:
    path: Path
    payload: Mapping[str, JsonValue]
    sha256: str


@dataclass(frozen=True)
class LoadedConversionAudit:
    path: Path
    payload: Mapping[str, JsonValue]
    sha256: str
```

Each loader reads bytes once with a no-follow regular-file descriptor, strict-parses
canonical JSON with one final newline, checks its exact schema key set, validates
types/ranges/hashes, and stores the content SHA-256. `validate_oaf_lock_set` compares
every cross-reference and independently rebuilds the descriptor and
model-artifact-set identity.

- [ ] **Step 4: Encode the exact invariant tables**

Store immutable module constants for:

- the three component names and fixed hashes;
- 88 rows `bin 0..87 -> MIDI 21..108 -> midi_<pitch>`;
- the eight upstream groups and complete member pitches from the design;
- required environment names/values;
- required hparams including `hop_length=441`, `spec_hop_length=512`,
  `drum_data_map="8-hit"`, and `drum_prediction_map=""`;
- exact OaF descriptor keys;
- exact backend/runtime/seal/audit top-level key sets.

Do not provide defaults for seal-generated values. Their validators require positive
integers or exact strings from the checked-in files.

- [ ] **Step 5: Run lock tests and style checks**

```bash
rtk uv run pytest -q tests/benchmark/test_backend_lock.py
rtk uv run ruff check src/benchmark/backend_lock.py tests/benchmark/test_backend_lock.py
rtk uv run black --check src/benchmark/backend_lock.py tests/benchmark/test_backend_lock.py
```

Expected: PASS.

- [ ] **Step 6: Commit strict lock validation**

```bash
rtk git add src/benchmark/backend_lock.py tests/benchmark/test_backend_lock.py
rtk git commit -m "feat: validate frozen OaF backend locks"
```

---

### Task 2: Generate reviewable zero-of-78 legacy conversion evidence

**Files:**

- Create: `tools/hpa320/audit_legacy_tf2_conversion.py`
- Create: `tests/benchmark/test_legacy_conversion_audit.py`
- Generate in Task 8:
  `docs/superpowers/evidence/hpa-320/legacy-tf2-conversion-coverage.json`

**Interfaces:**

- Consumes: explicit checkpoint prefix, HDF5 path, converter source, TF2 model
  source, and required OaF inventory.
- Produces: canonical `crux.legacy-tf2-conversion-coverage/v1` evidence.

- [ ] **Step 1: Write failing pure matching-audit tests**

```python
def test_audit_reports_zero_restored_required_tensors() -> None:
    result = audit_conversion_coverage(
        checkpoint_variables=(
            VariableSpec("onsets/conv0/weights", (3, 3, 1, 16), "float32"),
        ),
        required_variables=(
            VariableSpec("onsets/conv0/weights", (3, 3, 1, 16), "float32"),
        ),
        keras_weights=(
            VariableSpec("conv_0/kernel", (3, 3, 229, 32), "float32"),
        ),
    )

    assert result.restored_required == ()
    assert result.restored_required_count == 0
    assert result.unmatched_required == ("onsets/conv0/weights",)
```

Add cases proving that loose `"conv"` substring candidates do not count without an
actual assignment, dense transpose matching is recorded separately, dtype/shape
mismatches fail, and output ordering is lexical and deterministic.

- [ ] **Step 2: Run tests and verify the missing tool**

```bash
rtk uv run pytest -q tests/benchmark/test_legacy_conversion_audit.py
```

Expected: collection FAIL.

- [ ] **Step 3: Implement the pure audit and CLI**

Define:

```python
@dataclass(frozen=True)
class VariableSpec:
    name: str
    shape: tuple[int, ...]
    dtype: str


@dataclass(frozen=True)
class ConversionCoverage:
    restored_required: tuple[str, ...]
    unmatched_required: tuple[str, ...]
    candidate_matches: tuple[Mapping[str, JsonValue], ...]

    @property
    def restored_required_count(self) -> int:
        return len(self.restored_required)
```

The CLI requires:

```text
--checkpoint-prefix PATH, or both --candidate PATH and --model-cache-root PATH
--hdf5 PATH
--expected-hdf5-sha256 HEX
--required-inventory PATH
--converter-source src/cli/convert.py
--tf2-model-source src/app/tf2_magenta_model.py
--output docs/superpowers/evidence/hpa-320/legacy-tf2-conversion-coverage.json
```

It enumerates real checkpoint name/shape/dtype with TensorFlow's checkpoint reader,
enumerates HDF5 datasets, hashes every input, executes the converter's exact matching
rules as an audit without mutating weights, writes the strict canonical evidence
schema, and exits nonzero unless the HDF5 hash, required-inventory hash, and expected
zero count match. The candidate form strict-loads the calibrated candidate manifest,
uses its model-artifact-set SHA-256 to resolve the content-addressed cache directory,
and rejects a path or hash disagreement.

- [ ] **Step 4: Run pure tests**

```bash
rtk uv run pytest -q tests/benchmark/test_legacy_conversion_audit.py
rtk uv run ruff check \
  tools/hpa320/audit_legacy_tf2_conversion.py \
  tests/benchmark/test_legacy_conversion_audit.py
```

Expected: PASS. Do not generate the final evidence file until Task 8 supplies the
reviewed checkpoint, HDF5, and required inventory.

- [ ] **Step 5: Commit the audit generator**

```bash
rtk git add \
  tools/hpa320/audit_legacy_tf2_conversion.py \
  tests/benchmark/test_legacy_conversion_audit.py
rtk git commit -m "feat: audit legacy checkpoint conversion coverage"
```

---

### Task 3: Implement explicit checkpoint preparation and cache verification

**Files:**

- Create: `src/benchmark/backend_prepare.py`
- Create: `tests/benchmark/test_backend_prepare.py`
- Modify: `src/cli/benchmark.py`
- Modify: `tests/test_cli_benchmark.py`

**Interfaces:**

- Consumes: `LoadedBackendLock`, explicit `--download` or `--archive`.
- Produces: `PrepareBackendRequest`, `PrepareBackendOutcome`,
  `prepare_oaf_backend`.

- [ ] **Step 1: Write failing acquisition/cache tests**

```python
def test_archive_installs_only_after_every_hash_matches(tmp_path: Path) -> None:
    request = PrepareBackendRequest(
        backend_id=OFFICIAL_BACKEND_ID,
        cache_root=tmp_path / "cache",
        archive_path=write_valid_checkpoint_archive(tmp_path),
        download=False,
    )

    outcome = prepare_oaf_backend(request, backend_lock=fake_lock())

    assert outcome.status == "ready"
    assert outcome.exit_code == 0
    assert outcome.model_cache_path.name == fake_lock().model_artifact_set_sha256
```

Add tests for mutual exclusion, verify-only missing cache, exact archive byte length
and hash, path traversal, symlink entries, duplicate members, extra/missing
components, component size/hash mismatch, interrupted extraction, existing valid
cache, conflicting cache, cleanup failure, atomic publication, URL allowlist,
download failure as exit `1`, integrity failure as exit `2`, and unchanged prior
cache.

- [ ] **Step 2: Run tests and verify failure**

```bash
rtk uv run pytest -q \
  tests/benchmark/test_backend_prepare.py \
  tests/test_cli_benchmark.py -k prepare_backend
```

Expected: FAIL because the command/module is absent.

- [ ] **Step 3: Implement preparation transaction**

Define:

```python
@dataclass(frozen=True)
class PrepareBackendRequest:
    backend_id: str
    cache_root: Path
    archive_path: Path | None
    download: bool


@dataclass(frozen=True)
class PrepareBackendOutcome:
    status: Literal["ready", "acquisition_failed", "integrity_failed"]
    exit_code: Literal[0, 1, 2]
    model_cache_path: Path | None
```

Stage under a fresh directory beside the final cache, read ZIP entries without
extract-all, require exactly the three locked basenames, stream/hash each component,
fsync each staged file and directory, then publish the directory at:

```text
artifacts/benchmark/model-cache/sha256/<model-artifact-set-sha256>/
```

The download form uses only the informational lock URL and verifies archive size/hash
before ZIP parsing. Verify-only mode performs no network operation.

- [ ] **Step 4: Register setup-only CLI summary**

Add `prepare-backend --backend --download/--archive --cache-root`. After Click
parsing, print one sorted JSON line with exactly `status`, `exit_code`,
`report_path: null`, and `report_sha256: null`. Use `ctx.exit` for nonzero outcomes.
Never write a verification/execution report.

- [ ] **Step 5: Run focused tests and commit**

```bash
rtk uv run pytest -q \
  tests/benchmark/test_backend_prepare.py \
  tests/test_cli_benchmark.py -k prepare_backend
rtk uv run ruff check \
  src/benchmark/backend_prepare.py \
  src/cli/benchmark.py \
  tests/benchmark/test_backend_prepare.py \
  tests/test_cli_benchmark.py
rtk git add \
  src/benchmark/backend_prepare.py \
  src/cli/benchmark.py \
  tests/benchmark/test_backend_prepare.py \
  tests/test_cli_benchmark.py
rtk git commit -m "feat: prepare frozen OaF checkpoint cache"
```

---

### Task 4: Build hardened persistent process control with a fake runner

**Files:**

- Create: `src/benchmark/backend_process.py`
- Create: `tests/benchmark/test_backend_process.py`
- Create: `tests/fixtures/fake_oaf_runner.py`

**Interfaces:**

- Consumes: exact runtime launch profile and mounted locks.
- Produces: native-host evidence records, Docker command builder, and
  `RunnerProcess`.

- [ ] **Step 1: Write failing launch and stderr stress tests**

```python
def test_docker_command_contains_every_hardening_control(tmp_path: Path) -> None:
    command = build_docker_command(fake_launch_profile(tmp_path))

    assert command[:4] == ["docker", "run", "--rm", "-i"]
    assert "--network=none" in command
    assert "--read-only" in command
    assert "--cap-drop=ALL" in command
    assert "--security-opt=no-new-privileges" in command
    assert "--platform=linux/amd64" in command


def test_stderr_over_pipe_and_ring_capacity_does_not_deadlock(tmp_path: Path) -> None:
    runner = RunnerProcess.start(
        fake_process_profile(tmp_path, stderr_ring_buffer_bytes=4096),
        popen_factory=fake_runner_popen_factory(stderr_bytes=1024 * 1024),
    )

    response = runner.request({"type": "ping"}, deadline_seconds=2)

    assert response.payload == {"type": "pong"}
    assert runner.stderr_truncated is True
    assert runner.stderr_retained_bytes <= 4096
```

Add tests for environment before exec, missing/extra environment variables,
`PYTHONHASHSEED`, accepted GitHub/orchestrator/local-host evidence, bare local host
diagnostic-only, emulation opt-in, read-only mounts, numeric UID/GID, tmpfs/CPU/memory
/PID arguments, startup timeout, request timeout, malformed stdout, stray stdout,
wrong request ID, process death, concurrent request rejection, stderr redaction,
oversized/unterminated lines, and close/kill behavior.

The redaction cases cover credentials/secrets, exact environment values, absolute
host/container paths, URL userinfo/query strings, audio bytes, decoded samples, and
tracebacks. Allowed diagnostics cover stable codes, tensor names/counts, bounded
timings, and source-relative sanitized frames.

- [ ] **Step 2: Run tests and verify missing module**

```bash
rtk uv run pytest -q tests/benchmark/test_backend_process.py
```

Expected: collection FAIL.

- [ ] **Step 3: Implement exact launch records and evidence policy**

```python
@dataclass(frozen=True)
class NativeHostEvidence:
    kind: Literal["github_hosted", "orchestrator_signed", "approved_local"]
    payload: Mapping[str, JsonValue]
    sha256: str
    official_execution_allowed: bool


@dataclass(frozen=True)
class RunnerLaunchProfile:
    image_manifest_digest: str
    backend_lock_path: Path
    runtime_lock_path: Path
    model_cache_path: Path
    input_root: Path
    environment: Mapping[str, str]
    uid: int
    gid: int
    cpu_limit: str
    memory_bytes: int
    pid_limit: int
    tmp_bytes: int
    shm_bytes: int
    startup_deadline_seconds: int
    request_deadline_seconds: int
    stdout_max_line_bytes: int
    stderr_read_chunk_bytes: int
    stderr_max_line_bytes: int
    stderr_ring_buffer_bytes: int
```

Validate all values as exact positive lock values. `build_docker_command` uses
absolute mounted paths, no shell, and a fresh environment list. Native evidence must
match one of the three accepted forms; in-container `uname` is never sufficient.

- [ ] **Step 4: Implement concurrent process I/O**

`RunnerProcess.start` launches with `subprocess.Popen` list arguments, binary pipes,
and no inherited environment. Start a daemon stderr reader before reading the
handshake or writing a request. The reader consumes fixed chunks continuously,
sanitizes bounded logical lines, retains only the newest ring bytes, and records total
bytes/truncation. A lock enforces one in-flight request. Stdout is read in bounded
chunks, rejects a physical line before exceeding the exact seal-required
`stdout_max_line_bytes`, accepts exactly one newline-terminated UTF-8 JSON object per
line, and matches request IDs. Timeout/process/protocol failures kill the process and
raise stable backend-fatal codes.

- [ ] **Step 5: Run process tests and commit**

```bash
rtk uv run pytest -q tests/benchmark/test_backend_process.py
rtk uv run ruff check \
  src/benchmark/backend_process.py \
  tests/benchmark/test_backend_process.py \
  tests/fixtures/fake_oaf_runner.py
rtk git add \
  src/benchmark/backend_process.py \
  tests/benchmark/test_backend_process.py \
  tests/fixtures/fake_oaf_runner.py
rtk git commit -m "feat: control isolated transcription runner"
```

---

### Task 5: Vendor frozen source, resolve runtime bytes, and generate smoke audio

**Files:**

- Create: `tools/hpa320/vendor_magenta.py`
- Create: `tools/hpa320/resolve_oaf_runtime.py`
- Create: `tools/hpa320/oaf_system_packages.py`
- Create: `tools/hpa320/generate_smoke_fixture.py`
- Create: `runtime/oaf_tf1/requirements.in`
- Create: `runtime/oaf_tf1/requirements-test.in`
- Create: `runtime/oaf_tf1/requirements-build.in`
- Generate: `runtime/oaf_tf1/vendor/magenta/`
- Generate: `runtime/oaf_tf1/source-manifest.json`
- Generate: `runtime/oaf_tf1/requirements.lock`
- Generate: `runtime/oaf_tf1/requirements-test.lock`
- Generate: `runtime/oaf_tf1/requirements-build.lock`
- Generate: `runtime/oaf_tf1/distribution-build-manifest.json`
- Create: `runtime/oaf_tf1/Dockerfile`
- Create: `runtime/oaf_tf1/tests/.gitkeep`
- Create: `tests/benchmark/test_oaf_smoke_generator.py`
- Generate: `tests/fixtures/oaf_tf1_smoke/generator-parameters.json`
- Generate: `tests/fixtures/oaf_tf1_smoke/canonical.wav`

**Interfaces:**

- Consumes: a local Magenta checkout at the exact commit and the pinned base
  manifest.
- Produces: vendored/hash-manifested source, exact distribution lock, deterministic
  smoke WAV, and provisional image definition.

- [ ] **Step 1: Write failing vendor/runtime/smoke tests**

Test that vendor generation rejects the wrong commit, symlinks, unlisted paths, and
hash drift; runtime resolution rejects non-cp37/non-x86_64 wheels, unapproved sdists,
and the wrong TensorFlow wheel; the explicit `pretty_midi==0.2.10` sdist exception
builds byte-identical pure-Python wheels twice under the pinned offline toolchain;
smoke generation produces RIFF PCM16 mono 44.1-kHz bytes identically twice and
matches the checked-in parameter/WAV hash.

```python
def test_smoke_generator_is_byte_deterministic(tmp_path: Path) -> None:
    first = generate_smoke_wav(FIXED_PARAMETERS)
    second = generate_smoke_wav(FIXED_PARAMETERS)

    assert first == second
    assert parse_canonical_wav(first, max_input_audio_frames=44100).audio_frame_count == 44100
```

Run:

```bash
rtk uv run pytest -q tests/benchmark/test_oaf_smoke_generator.py
```

Expected: collection FAIL because the generator modules do not exist.

- [ ] **Step 2: Implement deterministic smoke synthesis**

Use exactly 44,100 PCM sample frames. Parameters define kick at sample `4410`, snare
at `17640`, and cymbal-like transient at `30870`. Use integer-only envelopes, an
explicit 32-bit linear-congruential noise state seeded with `0`, saturating int16
addition, and `struct.pack("<h", sample)`; do not use NumPy, `math.sin`, random module
defaults, resampling, or codecs. Write canonical parameter JSON and WAV through the
Phase B publication helper.

- [ ] **Step 3: Vendor the exact source commit**

Create a fresh local checkout:

```bash
hpa320_magenta_dir="$(rtk mktemp -d /private/tmp/hpa320-magenta.XXXXXX)"
rtk git clone --filter=blob:none https://github.com/magenta/magenta.git \
  "$hpa320_magenta_dir"
rtk git -C "$hpa320_magenta_dir" checkout \
  94529798dfbbb14c27ddfd76f23027dc8e2ce185
```

`vendor_magenta.py` verifies `git rev-parse HEAD`, copies the complete
`magenta/models/onsets_frames_transcription/` package plus imported
`magenta/music/sequences_lib.py` and required protobuf/package modules, rejects
symlinks, and emits canonical manifest rows `{path, sha256, license}` sorted by UTF-8
path bytes. Review the import closure from the frozen entrypoint before accepting the
manifest.

Run:

```bash
rtk uv run python tools/hpa320/vendor_magenta.py \
  --source "$hpa320_magenta_dir" \
  --destination runtime/oaf_tf1/vendor \
  --manifest runtime/oaf_tf1/source-manifest.json
```

- [ ] **Step 4: Resolve exact Python distributions**

`requirements.in` contains the explicit TensorFlow pin and direct packages imported
by the vendored closure. `requirements-test.in` contains only `pytest==7.4.4`;
its transitive test dependencies remain separate from the final runtime.
`requirements-build.in` contains only the exact CPython-3.7-compatible build
toolchain used for explicitly allowlisted sdist exceptions; those tools never enter
the final runtime.
`resolve_oaf_runtime.py` runs package resolution for CPython 3.7
`manylinux2010_x86_64`, downloads every published wheel, requires the fixed
TensorFlow wheel hash, records name/version/filename/SHA-256 for every transitive
distribution, and renders hash-required runtime and test locks. If the exact resolver
proves that a required version has no compatible published wheel, it may add that
package/version to an explicit closure-derived sdist allowlist only after proving the
sdist builds a pure-Python `py3-none-any` wheel. For every allowlisted entry, verify
the exact source-archive hash, build it twice with network disabled under the
hash-locked build toolchain and canonical environment/recipe, require byte-identical
wheels, and record the complete source/toolchain/recipe/result identity in
`distribution-build-manifest.json`. That manifest also records every file published
for each allowlisted PyPI release and derives `required_by` from the selected direct
requirements and wheel `Requires-Dist` graph. The resolver provides an exact
`--materialize-wheelhouse` workflow which reconstructs ignored runtime/test
wheelhouses from checked locks, uses an explicit offline cache for locally built
allowlisted wheels, and rejects extras, symlinks, or hash drift. The initial proven entries are
`pretty_midi==0.2.10` and TensorFlow dependency `gast==0.2.2`. A native-extension
sdist or a package with a compatible published wheel is not eligible. Task 8 must
reproduce every allowlisted wheel natively before sealing. The Dockerfile installs
the runtime lock into its final stage with
`--require-hashes --no-index`; a separate `test` stage installs the test lock and
copies `runtime/oaf_tf1/tests`. Build-tool distributions do not enter either final
runtime or test dependency closure. Both stages copy vendored/runtime sources and run
as the seal-selected numeric UID/GID. Final lock files remain outside the image.
The final-only image stage consumes an ignored
`crux.oaf-system-package-bundle/v2` bundle containing a canonical manifest, exact
`InRelease`, selected signed `Packages`/`Packages.xz` indexes at their Release paths,
every local `.deb`, and the complete three-column expected `dpkg` inventory. The
only trust anchor is `/usr/share/keyrings/debian-archive-keyring.gpg` already inside
the digest-pinned base image; a bundle-supplied keyring is forbidden. It verifies
that base keyring's expected hash, the archive signature and exact single reviewed
`VALIDSIG` fingerprint, codename, `amd64` architecture, each signed index
path/size/hash, bounded decompression and strict stanza parsing, and exact local
package/inventory binding before installation. It installs only the authenticated
local closure and requires exact final inventory equality. Task 8 owns materializing
the native bundle and final values.

- [ ] **Step 5: Run generators twice and inspect diffs**

```bash
rtk uv run pytest -q tests/benchmark/test_oaf_smoke_generator.py
rtk uv run python tools/hpa320/generate_smoke_fixture.py \
  --parameters tests/fixtures/oaf_tf1_smoke/generator-parameters.json \
  --output tests/fixtures/oaf_tf1_smoke/canonical.wav
rtk git diff --exit-code \
  tests/fixtures/oaf_tf1_smoke/generator-parameters.json \
  tests/fixtures/oaf_tf1_smoke/canonical.wav \
  runtime/oaf_tf1/source-manifest.json \
  runtime/oaf_tf1/requirements.lock \
  runtime/oaf_tf1/requirements-test.lock
```

Expected: the second generation changes no bytes.

- [ ] **Step 6: Commit vendoring/runtime inputs**

```bash
rtk git add \
  tools/hpa320/vendor_magenta.py \
  tools/hpa320/resolve_oaf_runtime.py \
  tools/hpa320/oaf_system_packages.py \
  tools/hpa320/generate_smoke_fixture.py \
  runtime/oaf_tf1/Dockerfile \
  runtime/oaf_tf1/requirements.in \
  runtime/oaf_tf1/requirements.lock \
  runtime/oaf_tf1/requirements-build.in \
  runtime/oaf_tf1/requirements-build.lock \
  runtime/oaf_tf1/distribution-build-manifest.json \
  runtime/oaf_tf1/requirements-test.in \
  runtime/oaf_tf1/requirements-test.lock \
  runtime/oaf_tf1/source-manifest.json \
  runtime/oaf_tf1/tests/.gitkeep \
  runtime/oaf_tf1/vendor/magenta \
  tests/benchmark/test_oaf_smoke_generator.py \
  tests/fixtures/oaf_tf1_smoke/generator-parameters.json \
  tests/fixtures/oaf_tf1_smoke/canonical.wav
rtk git commit -m "build: pin frozen OaF runtime inputs"
```

---

### Task 6: Implement the TensorFlow 1 runner and frame instrumentation

**Files:**

- Create: `runtime/oaf_tf1/entrypoint.py`
- Create: `runtime/oaf_tf1/protocol.py`
- Create: `runtime/oaf_tf1/oaf_backend.py`
- Create: `runtime/oaf_tf1/apply_instrumentation_patch.py`
- Create: `runtime/oaf_tf1/patches/capture-emitted-frame.patch`
- Generate: `runtime/oaf_tf1/runner-source-manifest.json`
- Create: `tools/hpa320/generate_runner_source_manifest.py`
- Modify: `runtime/oaf_tf1/Dockerfile`
- Create: `runtime/oaf_tf1/tests/test_protocol.py`
- Create: `runtime/oaf_tf1/tests/test_instrumentation.py`
- Create: `runtime/oaf_tf1/tests/test_tensor_coverage.py`

**Interfaces:**

- Consumes: mounted locks, model cache, canonical WAV, vendored Magenta source.
- Produces: handshake and transcribe protocol for `RunnerProcess`.

- [ ] **Step 1: Write runner protocol tests before TensorFlow integration**

Test environment assertion before numeric imports, lock duplicate/unknown-key
rejection, mounted-lock hash output, stdout protocol purity, sanitized stderr,
request/path/hash validation, request ID echo, exact error codes, and one request at
a time.

```python
def test_entrypoint_rejects_python_hash_seed_before_runner_import(monkeypatch) -> None:
    monkeypatch.setenv("PYTHONHASHSEED", "unexpected")

    with pytest.raises(SystemExit) as error:
        validate_process_environment(EXPECTED_ENVIRONMENT)

    assert error.value.code == 2
```

Run:

```bash
rtk uv run pytest -q \
  runtime/oaf_tf1/tests/test_protocol.py \
  runtime/oaf_tf1/tests/test_instrumentation.py
```

Expected: collection FAIL because the runner and patch modules do not exist.

- [ ] **Step 2: Apply and test the instrumentation patch**

Keep the checked-in vendored upstream tree byte-identical to its Task 5 manifest.
Patch only an image-build copy of the vendored call path around
`pianoroll_to_note_sequence` so it returns the unmodified `NoteSequence` plus ordered
metadata `(start_frame, pitch, raw_velocity)` from the same note-emission loop.
`infer_util` pairs confidence from `onset_probs[start_frame, pitch]`. It never
reconstructs a frame from `start_time`.

The pure-Python patch applier verifies the exact upstream preimage and fixed patch
identity before changing the image-build copy. The runner-source manifest covers the
applier, patch, runner modules, and Dockerfile; the upstream source manifest remains
unchanged.

The parity test feeds fixed onset/frame/offset/velocity arrays into patched and
unmodified conversion, serializes both `NoteSequence` values, and requires byte
identity. It separately asserts captured frame/pitch/velocity and confidence.

Add focused boundaries proving:

```text
frames_per_second = 44100 / 512
frame_length_seconds = 1.0 / frames_per_second
time_sec_raw = frame_index * frame_length_seconds
```

is evaluated in that binary64 order, not as `frame_index * 512 / 44100`. For finite
raw velocity, test values below zero, zero, fractional, one, and above one through
clamp, multiply by `127`, add `0`, and `int` truncation. Reject NaN and both
infinities before clamping with the named adapter error.

- [ ] **Step 3: Implement exact graph construction and invocation**

`build_and_restore_model` must:

1. load `configs.CONFIG_MAP["drums"]` with no runtime hparam overrides;
2. set `batch_size=1` and `truncated_length_secs=0`;
3. construct the exact PREDICT graph and `data.provide_batch` call;
4. set explicit `ConfigProto` inter/intra threads to `1`;
5. enumerate graph/checkpoint variables;
6. require exact name/shape/dtype for all 78 required entries;
7. classify all 52 allowed non-inference entries;
8. restore and prove no required variable is uninitialized;
9. prove no active stochastic dropout in PREDICT.

`transcribe_canonical_wav` uses exact `create_example`/`process_record` arguments:
`sample_rate=44100`, empty `NoteSequence`, UTF-8 filename ID, `min_length=0`,
`max_length=-1`, `allow_empty_notesequence=True`,
`load_audio_with_librosa=False`, one example, preprocessed non-training
`provide_batch`, explicit checkpoint prefix, `yield_single_examples=False`, and one
prediction batch.

- [ ] **Step 4: Implement handshake, smoke, and requests**

At startup, strict-load/hash mounted locks, build/restore the model, run the checked
smoke fixture, and emit `ready` only after exact oracle comparison. The handshake
includes descriptor, descriptor hash, runtime facts, 78/78/52 counts/hashes, and
smoke hashes/status. Transcribe responses emit raw native events with selected frame,
bin/pitch/class, exact group metadata, raw confidence, and velocity for host
quantization.

- [ ] **Step 5: Run runtime tests inside the provisional image**

```bash
rtk uv run python tools/hpa320/resolve_oaf_runtime.py \
  --materialize-wheelhouse \
  --lock runtime/oaf_tf1/requirements.lock \
  --wheelhouse runtime/oaf_tf1/wheelhouse/runtime \
  --offline-cache /workspace/hpa320/oaf-wheel-cache
rtk uv run python tools/hpa320/resolve_oaf_runtime.py \
  --materialize-wheelhouse \
  --lock runtime/oaf_tf1/requirements-test.lock \
  --wheelhouse runtime/oaf_tf1/wheelhouse/test \
  --offline-cache /workspace/hpa320/oaf-wheel-cache
rtk docker buildx build \
  --platform linux/amd64 \
  --target test \
  --load \
  --tag crux-oaf-tf1:hpa320-provisional \
  --build-arg RUNTIME_UID=10001 \
  --build-arg RUNTIME_GID=10001 \
  --file runtime/oaf_tf1/Dockerfile \
  .
rtk docker run --rm \
  --platform linux/amd64 \
  --entrypoint /bin/sh \
  crux-oaf-tf1:hpa320-provisional \
  -ec 'test "$(id -u)" = 10001; test "$(id -g)" = 10001; \
    test -r /opt/crux/runtime/tests; \
    /opt/crux/venv/bin/python -m pytest -q /opt/crux/runtime/tests'
```

Expected on emulation: tests may establish diagnostic behavior but cannot produce a
seal. All pure protocol/instrumentation tests must pass.
The numeric IDs above are explicit diagnostic provisional values only; they are not
eligible for the final lock or seal.

- [ ] **Step 6: Commit the provisional runner**

```bash
rtk git add \
  runtime/oaf_tf1/entrypoint.py \
  runtime/oaf_tf1/protocol.py \
  runtime/oaf_tf1/oaf_backend.py \
  runtime/oaf_tf1/apply_instrumentation_patch.py \
  runtime/oaf_tf1/patches/capture-emitted-frame.patch \
  runtime/oaf_tf1/runner-source-manifest.json \
  runtime/oaf_tf1/Dockerfile \
  tools/hpa320/generate_runner_source_manifest.py \
  runtime/oaf_tf1/tests/test_protocol.py \
  runtime/oaf_tf1/tests/test_instrumentation.py \
  runtime/oaf_tf1/tests/test_tensor_coverage.py
rtk git commit -m "feat: implement frozen OaF runner"
```

---

### Task 7: Integrate the OaF host adapter and `verify-backend`

**Files:**

- Create: `src/benchmark/backends/oaf_tf1.py`
- Create: `tests/benchmark/test_oaf_backend.py`
- Modify: `src/benchmark/backend_registry.py`
- Modify: `src/cli/benchmark.py`
- Modify: `tests/test_cli_benchmark.py`

**Interfaces:**

- Consumes: Phase B protocol/reports, strict locks, process controller.
- Produces: `OafTf1Backend`, lazy `create_backend`, and CLI verification.

- [ ] **Step 1: Write failing fake-runner adapter tests**

```python
def test_verify_accepts_only_matching_handshake_and_smoke(tmp_path: Path) -> None:
    backend = OafTf1Backend(
        fake_oaf_config(tmp_path),
        process_factory=fake_runner_factory(handshake=valid_handshake()),
    )

    verification = backend.verify()

    assert verification.status == "verified"
    assert verification.max_input_audio_frames == fake_lock().max_input_audio_frames


def test_transcribe_never_falls_back_after_runner_failure(tmp_path: Path) -> None:
    backend = OafTf1Backend(
        fake_oaf_config(tmp_path),
        process_factory=fake_runner_factory(request_error="runner_process_died"),
    )

    with pytest.raises(OafBackendFatal, match="runner_process_died"):
        backend.transcribe(fake_canonical_audio(tmp_path))
```

Add mismatch cases for every descriptor/lock/runtime/model/map/schema field,
78/78/52 counts, smoke bytes, input hash, path escape, malformed event, request
timeout, process death, OOM, emulated environment, dirty inference-relevant source,
unrelated dirty documentation, execution-profile attestation fields, and idempotent
close. Add report-publication failure with sanitized standard error, no summary, and
exit `2`.

Run:

```bash
rtk uv run pytest -q \
  tests/benchmark/test_oaf_backend.py \
  tests/test_cli_benchmark.py -k verify_backend
```

Expected: collection FAIL because the OaF adapter does not exist and the command is
not registered.

- [ ] **Step 2: Implement adapter phase ordering**

```python
@dataclass(frozen=True)
class OafBackendConfig:
    backend_lock_path: Path
    runtime_lock_path: Path
    seal_evidence_path: Path
    conversion_audit_path: Path
    host_adapter_source_manifest_path: Path
    model_cache_root: Path
    input_root: Path
    native_host_evidence: NativeHostEvidence
    allow_emulated_diagnostics: bool
    strict_checkout: bool
```

`verify` loads/cross-validates all locks, verifies the cached model and image digest,
builds the exact launch profile, starts one process, compares every handshake field,
publishes a Phase B execution attestation from the union of the locked host/runner
source manifests and exact launch conditions, and returns the report-facing
verification record. Strict checkout mode rejects an inference-relevant change
before launch; non-strict mode records the canonical changed-file manifest. An
unrelated dirty path affects `checkout_dirty` but does not enter that manifest.
`transcribe` requires the same live verified process, validates the response
identities, converts raw runner events to `NativePrediction`, and treats nonfinite
velocity as an item error without fallback. `close` terminates once.

`create_backend` resolves checked-in lock paths and cache/input roots from explicit
environment-free defaults under the repository; tests inject config directly.

- [ ] **Step 3: Add `verify-backend` CLI**

Options:

```text
--backend TEXT             default magenta-egmd-tf1-94529798-8hit-v1
--reports-root DIRECTORY   default artifacts/benchmark/backends
--allow-emulated-diagnostics
```

After Click parsing, publish exactly one
`crux.backend-verification-report/v1`, write one four-field canonical summary, and
map verified `0`, environment unsupported `1`, backend-fatal `2`. Click usage exit
`2` occurs before a report/summary.
If Phase B report publication raises, emit `report_publication_failed` only on
sanitized standard error, emit no summary, and exit `2`.

- [ ] **Step 4: Run fake-runner and CLI tests**

```bash
rtk uv run pytest -q \
  tests/benchmark/test_backend_lock.py \
  tests/benchmark/test_backend_process.py \
  tests/benchmark/test_oaf_backend.py \
  tests/test_cli_benchmark.py -k 'verify_backend or prepare_backend'
```

Expected: PASS without Docker or TensorFlow 1.

- [ ] **Step 5: Commit OaF host integration**

```bash
rtk git add \
  src/benchmark/backends/oaf_tf1.py \
  src/benchmark/backend_registry.py \
  src/cli/benchmark.py \
  tests/benchmark/test_oaf_backend.py \
  tests/test_cli_benchmark.py
rtk git commit -m "feat: verify frozen OaF backend"
```

---

### Task 8: Run the native-amd64 calibration and seal gate

**Files:**

- Create: `tools/hpa320/seal_oaf_backend.py`
- Create: `tests/benchmark/test_oaf_seal.py`
- Create: `tests/benchmark/test_oaf_real_checkpoint.py`
- Generate: final backend/runtime locks, seal evidence, conversion audit, smoke
  oracle, tensor coverage report, OCI archive evidence.
- Generate, ignored: `runtime/oaf_tf1/wheelhouse/runtime/`
- Generate, ignored: `runtime/oaf_tf1/wheelhouse/test/`
- Generate, ignored: `runtime/oaf_tf1/system-packages/`
- Generate, ignored: `/workspace/hpa320/hpa320-system-package-build-args.env`
- Modify after review: exact generated files only.

**Interfaces:**

- Consumes: Tasks 1-7 plus an accepted native-host evidence source.
- Produces: the only accepted exact values and final immutable OaF identity.

- [ ] **Step 1: Implement the seal orchestrator and verify accepted host evidence**

First write `test_oaf_seal.py` with fake candidate directories covering strict
host-evidence parsing, missing/sentinel calibration values, count-only tensor
evidence, non-native/emulated rejection, OCI hash mismatch, HDF5 audit mismatch,
lock-cycle rejection, atomic output publication, and deterministic reruns.

Run:

```bash
rtk uv run pytest -q tests/benchmark/test_oaf_seal.py
```

Expected: collection FAIL because `seal_oaf_backend.py` does not exist.

Implement exact `validate-host`, `calibrate`, and `seal` subcommands. The first is
read-only. `calibrate` writes only a candidate directory after all measured evidence
passes. `seal` strict-reads that complete candidate and the reviewed conversion
audit, derives manifests/locks/oracle without final-lock self-references, validates
them in memory, and atomically publishes each immutable file. It publishes
evidence/oracle/manifests first, the runtime lock second, and the backend lock last.
A failure may leave an unreferenced new immutable dependency, but never overwrites a
prior file or publishes a backend lock that references an incomplete set. Run the
focused fake-evidence suite until it passes.

On the native worker, record exactly one accepted form:

- GitHub-hosted Linux X64 job API record plus `RUNNER_OS=Linux`,
  `RUNNER_ARCH=X64`, workflow commit, job ID, and immutable run URL;
- approved orchestrator-signed physical-worker attestation; or
- approved signed local seal-host record combining host `linux/x86_64` and container
  daemon/worker metadata.

Run the evidence validator before any seal output:

```bash
rtk uv run python tools/hpa320/seal_oaf_backend.py \
  validate-host \
  --evidence /workspace/hpa320/native-host-evidence.json
```

Expected: exit `0` and `official_execution_allowed: true`. Otherwise stop this task.

- [ ] **Step 2: Prepare exact checkpoint bytes**

```bash
rtk uv run crux benchmark prepare-backend \
  --backend magenta-egmd-tf1-94529798-8hit-v1 \
  --download \
  --cache-root artifacts/benchmark/model-cache
```

Expected: canonical summary with `status: "ready"` and exit `0`.

- [ ] **Step 3: Build and preserve the provisional image**

```bash
rtk uv run python tools/hpa320/resolve_oaf_runtime.py \
  --materialize-wheelhouse \
  --lock runtime/oaf_tf1/requirements.lock \
  --wheelhouse runtime/oaf_tf1/wheelhouse/runtime \
  --offline-cache /workspace/hpa320/oaf-wheel-cache
rtk uv run python tools/hpa320/resolve_oaf_runtime.py \
  --materialize-wheelhouse \
  --lock runtime/oaf_tf1/requirements-test.lock \
  --wheelhouse runtime/oaf_tf1/wheelhouse/test \
  --offline-cache /workspace/hpa320/oaf-wheel-cache
rtk uv run python tools/hpa320/seal_oaf_backend.py \
  materialize-system-packages \
  --host-evidence /workspace/hpa320/native-host-evidence.json \
  --bundle runtime/oaf_tf1/system-packages \
  --build-args-output /workspace/hpa320/hpa320-system-package-build-args.env
. /workspace/hpa320/hpa320-system-package-build-args.env
rtk docker buildx build \
  --platform linux/amd64 \
  --load \
  --tag crux-oaf-tf1:hpa320-seal \
  --build-arg RUNTIME_UID="${HPA320_RUNTIME_UID:?}" \
  --build-arg RUNTIME_GID="${HPA320_RUNTIME_GID:?}" \
  --build-arg DEBIAN_SNAPSHOT_URL="${HPA320_DEBIAN_SNAPSHOT_URL:?}" \
  --build-arg DEBIAN_INRELEASE_SHA256="${HPA320_DEBIAN_INRELEASE_SHA256:?}" \
  --build-arg DEBIAN_ARCHIVE_KEYRING_SHA256="${HPA320_DEBIAN_ARCHIVE_KEYRING_SHA256:?}" \
  --build-arg DEBIAN_SIGNING_FINGERPRINT="${HPA320_DEBIAN_SIGNING_FINGERPRINT:?}" \
  --build-arg DEBIAN_CODENAME="${HPA320_DEBIAN_CODENAME:?}" \
  --build-arg DEBIAN_ARCHITECTURE="${HPA320_DEBIAN_ARCHITECTURE:?}" \
  --build-arg SYSTEM_PACKAGE_MANIFEST_SHA256="${HPA320_SYSTEM_PACKAGE_MANIFEST_SHA256:?}" \
  --build-arg SYSTEM_PACKAGE_INVENTORY_SHA256="${HPA320_SYSTEM_PACKAGE_INVENTORY_SHA256:?}" \
  --file runtime/oaf_tf1/Dockerfile \
  .
rtk docker image inspect crux-oaf-tf1:hpa320-seal
```

Prove the base manifest digest, installed wheel/system-package bytes, Python/TF ABI,
numeric UID/GID, and runner/source manifests. Export the complete image as an OCI
layout through `seal_oaf_backend.py`; record manifest/config/layer and archive hashes.
The materializer must obtain the Debian snapshot `InRelease`, selected exact signed
package indexes, complete authenticated local `.deb` closure, reviewed
fingerprint/codename/architecture, digest-pinned base-keyring hash, and expected full
three-column `dpkg` inventory on the accepted native worker. It must never copy a
keyring into the bundle; it writes no final lock and emits no provisional identity.

- [ ] **Step 4: Calibrate exact resource and input bounds**

Run the complete smoke and coverage flow while measuring peak RSS, CPU, tmpfs,
`/dev/shm`, PID count, startup, and request duration. Select reviewed positive
headroom values and a positive `max_input_audio_frames`; rerun at bound-minus-one,
exact bound, and over bound. In-bound OOM/process failure is fatal and invalidates
the candidate profile.

The seal command writes a candidate evidence directory:

```bash
rtk uv run python tools/hpa320/seal_oaf_backend.py \
  calibrate \
  --image crux-oaf-tf1:hpa320-seal \
  --host-evidence /workspace/hpa320/native-host-evidence.json \
  --model-cache artifacts/benchmark/model-cache \
  --output artifacts/benchmark/backends/hpa320-seal-candidate
```

- [ ] **Step 5: Generate and review tensor, smoke, security, and conversion evidence**

Require:

- exactly 130 checkpoint entries;
- exactly 78 required graph variables restored by name/shape/dtype;
- exactly 52 individually classified non-inference entries;
- no uninitialized required variable;
- no active PREDICT dropout;
- patched/unmodified `NoteSequence` byte parity;
- nonempty exact smoke output;
- legacy conversion audit restored-required count `0`;
- security scan/advisory snapshot;
- exact package and OCI byte inventories.

Review every generated JSON file and hash before sealing. Do not accept count-only
reports.

Generate the conversion evidence from the explicitly supplied observed HDF5 file:

```bash
rtk uv run python tools/hpa320/audit_legacy_tf2_conversion.py \
  --candidate artifacts/benchmark/backends/hpa320-seal-candidate \
  --model-cache-root artifacts/benchmark/model-cache \
  --hdf5 /workspace/hpa320/tf2_model.weights.h5 \
  --expected-hdf5-sha256 \
    d36ced8b2ee241bc37ad6fbb918ba38e95d666350dd4888bca59a1243bf4d10e \
  --required-inventory \
    artifacts/benchmark/backends/hpa320-seal-candidate/required-inventory.json \
  --converter-source src/cli/convert.py \
  --tf2-model-source src/app/tf2_magenta_model.py \
  --output \
    docs/superpowers/evidence/hpa-320/legacy-tf2-conversion-coverage.json
```

Expected: the supplied HDF5 hash matches and restored-required count is exactly `0`.
If the observed file is unavailable, stop the seal; do not substitute another HDF5
artifact.

- [ ] **Step 6: Publish final evidence, oracle, and locks**

```bash
rtk uv run python tools/hpa320/seal_oaf_backend.py \
  seal \
  --candidate artifacts/benchmark/backends/hpa320-seal-candidate \
  --backend-lock \
    config/benchmark/backends/magenta-egmd-tf1-94529798-8hit-v1.backend-lock.json \
  --runtime-lock \
    config/benchmark/backends/magenta-egmd-tf1-94529798-8hit-v1.runtime-lock.json \
  --seal-evidence \
    config/benchmark/backends/magenta-egmd-tf1-94529798-8hit-v1.seal-evidence.json \
  --conversion-audit \
    docs/superpowers/evidence/hpa-320/legacy-tf2-conversion-coverage.json \
  --host-adapter-source-manifest \
    runtime/oaf_tf1/host-adapter-source-manifest.json \
  --tensor-coverage \
    docs/superpowers/evidence/hpa-320/oaf-tensor-coverage.json \
  --security-scan \
    docs/superpowers/evidence/hpa-320/oaf-security-scan.json \
  --oci-layout-manifest \
    docs/superpowers/evidence/hpa-320/oaf-oci-layout-manifest.json \
  --smoke-oracle tests/fixtures/oaf_tf1_smoke/smoke-oracle.json
```

The seal evidence excludes final lock hashes. Final locks reference the accepted
evidence hash and already-built image manifest digest. The generated host-adapter
manifest enumerates `backend_lock.py`, `backend_process.py`, `backends/oaf_tf1.py`,
`backend_identity.py`, `backends/base.py`, `input_view.py`,
`prediction_artifact.py`, `backend_attestation.py`, and
`backend_publication.py`, with reviewed `covered_roots`; it excludes CLI/report-only
files.

- [ ] **Step 7: Rerun final verification twice in-process and once fresh**

Implement `test_oaf_real_checkpoint.py` as a native-only test guarded by an explicit
environment flag and accepted host evidence. It performs two requests on one
verified process and one request on a fresh process, strict-reads every artifact,
generates expected bytes only from the oracle plus final locks/descriptor, and
requires exact byte equality.

```bash
rtk uv run crux benchmark verify-backend \
  --backend magenta-egmd-tf1-94529798-8hit-v1
rtk uv run pytest -q tests/benchmark/test_oaf_real_checkpoint.py
```

The integration test performs two requests on one verified process and one on a fresh
process. All three native prediction artifacts must be byte-identical and exactly
match the generated oracle after six-decimal quantization, with no numeric tolerance.

- [ ] **Step 8: Commit only reviewed exact seal outputs**

```bash
rtk git add \
  tools/hpa320/seal_oaf_backend.py \
  tests/benchmark/test_oaf_seal.py \
  config/benchmark/backends/magenta-egmd-tf1-94529798-8hit-v1.backend-lock.json \
  config/benchmark/backends/magenta-egmd-tf1-94529798-8hit-v1.runtime-lock.json \
  config/benchmark/backends/magenta-egmd-tf1-94529798-8hit-v1.seal-evidence.json \
  docs/superpowers/evidence/hpa-320/legacy-tf2-conversion-coverage.json \
  docs/superpowers/evidence/hpa-320/oaf-tensor-coverage.json \
  docs/superpowers/evidence/hpa-320/oaf-security-scan.json \
  docs/superpowers/evidence/hpa-320/oaf-oci-layout-manifest.json \
  tests/benchmark/test_oaf_real_checkpoint.py \
  tests/fixtures/oaf_tf1_smoke/smoke-oracle.json \
  runtime/oaf_tf1/requirements.lock \
  runtime/oaf_tf1/source-manifest.json \
  runtime/oaf_tf1/runner-source-manifest.json \
  runtime/oaf_tf1/host-adapter-source-manifest.json
rtk git commit -m "build: seal frozen OaF backend identity"
```

---

### Task 9: Add native CI, operator documentation, and final acceptance

**Files:**

- Create: `.github/workflows/oaf-backend-integration.yml`
- Test: `tests/benchmark/test_oaf_real_checkpoint.py`
- Modify: `docs/drumery-dtx-midi-benchmarking-reference.md`
- Modify: `tests/test_cli_benchmark.py`

**Interfaces:**

- Consumes: final sealed runtime/adapter.
- Produces: required native check and auditable artifacts.

- [ ] **Step 1: Add dedicated workflow**

Trigger on runtime/lock/vendor/smoke/schema changes, schedule, and manual dispatch.
Use a GitHub-hosted Linux X64 runner, verify its job record and
`RUNNER_ARCH/RUNNER_OS`, download the locked archive explicitly, verify exact size
and SHA before extraction, build/pull the immutable image, run `verify-backend`, run
the real-checkpoint suite, and upload verification, tensor coverage, smoke, security,
and seal artifacts. Acquisition failure fails the job; emulation cannot satisfy it.

- [ ] **Step 2: Document preparation, verification, and diagnostic execution**

Add exact commands, model-cache paths, official-vs-emulated behavior, report paths,
exit meanings, network boundaries, known EOL/CVE posture, preserved OCI bytes, and
the rule that no failure invokes another backend.

- [ ] **Step 3: Run host and native quality gates**

Host:

```bash
rtk uv run pytest -q
rtk uv run ruff check src tests tools
rtk uv run black --check src tests tools
rtk uv run pylint src/app src/cli src/benchmark
rtk git diff --check
```

Native:

```bash
rtk uv run pytest -q tests/benchmark/test_oaf_real_checkpoint.py
rtk uv run crux benchmark verify-backend \
  --backend magenta-egmd-tf1-94529798-8hit-v1
```

Expected: every command exits `0`; the native verification report is `verified`,
78/78/52 coverage is exact, and smoke artifacts are byte-identical.

- [ ] **Step 4: Verify scope and fallback absence**

```bash
rtk rg -n 'fallback|DrumTranscriber|tf2_model.weights.h5' \
  src/benchmark/backends/oaf_tf1.py \
  src/benchmark/backend_process.py \
  runtime/oaf_tf1
rtk git diff --name-only
```

Inspect every match. Expected: no executable HDF5/`DrumTranscriber` import or
alternate-backend branch appears in the OaF implementation; occurrences of the word
`fallback` are limited to explicit rejection, environment controls, or tests. Changed
files stay inside the Phase A map plus common CLI/docs.

- [ ] **Step 5: Commit CI and acceptance coverage**

```bash
rtk git add \
  .github/workflows/oaf-backend-integration.yml \
  tests/benchmark/test_oaf_real_checkpoint.py \
  tests/test_cli_benchmark.py \
  docs/drumery-dtx-midi-benchmarking-reference.md
rtk git commit -m "ci: verify frozen OaF backend natively"
```

Phase A is complete only after Task 8's accepted seal and this native check pass.
