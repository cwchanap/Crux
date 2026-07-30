# HPA-320 Phase C Compatibility Adapters Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use
> superpowers:subagent-driven-development (recommended) or
> superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox
> (`- [ ]`) syntax for tracking.

**Goal:** Preserve the current heuristic and TensorFlow 2/HDF5 transcription paths
only through explicit compatibility IDs, add deterministic MIDI derivation from
native prediction events, and make `transcribe-and-score` publish a typed legacy
wrapper without ever claiming frozen OaF identity.

**Architecture:** Run `heuristic-onset-v1` in a content-addressed virtual environment
whose interpreter, package bytes, adapter source, and resolved algorithm parameters
are locked. Adapt its events to the common Phase B prediction envelope. Keep
`legacy-tf2-h5-v0` outside `TranscriptionBackend`: a narrow wrapper invokes only the
TF2 model path and treats unavailable or corrupt weights as failure instead of
entering `DrumTranscriber`'s heuristic fallback. The legacy combined scorer accepts
one explicit backend ID, creates MIDI through the selected compatibility path, calls
the existing mapping/scoring pipeline unchanged, and wraps the existing score report
in `crux.legacy-score-report/v1`.

**Tech Stack:** Python 3.12, isolated `uv` virtual environments, Librosa 0.11,
NumPy, SciPy, Numba, SoundFile, a distribution-locked MIDI writer, TensorFlow 2
compatibility code, Click 8, Pytest, and the Phase B canonical artifact/report
helpers.

## Global Constraints

- Phase B is a prerequisite. Use its exact `BackendDescriptor`, `CanonicalAudio`,
  `NativeEvent`, `NativePrediction`, `BackendVerification`, prediction-artifact,
  publication, registry, and report interfaces.
- `heuristic-onset-v1` is selected only by its complete ID. Model failure never
  selects it.
- The heuristic descriptor schema is exactly
  `crux.heuristic-backend-descriptor/v1`; the parameter-lock schema is exactly
  `crux.heuristic-parameter-lock/v1`.
- The heuristic model, architecture, output-space, and metadata IDs are exactly
  `crux-heuristic-onset-nonmodel-v1`,
  `librosa-onset-centroid-zcr-v1`, `crux-heuristic-midi7-v1`, and
  `crux-empty-native-metadata-v1`.
- A heuristic descriptor has a non-null parameter-lock SHA-256 and null backend
  lock, runtime lock, model-artifact set, upstream commit, and training-data map.
- Heuristic input is canonical mono 16-bit 44.1 kHz PCM WAV. Decoding, mixing, and
  resampling remain upstream preparation responsibilities.
- Heuristic events use `midi_<pitch>` class IDs, null output bin, null confidence,
  an empty native-metadata object, and the exact pitch/velocity pairs
  `36:80`, `38:70`, `42:60`, `45:65`, `47:65`, `49:65`, and `50:65`.
- `legacy-tf2-h5-v0` remains combined-score-only, has no frozen descriptor or lock,
  and is always reported as `backend_validation_status: "unvalidated"`.
- `transcribe-and-score` has no default backend. Its executable selections are
  `legacy-tf2-h5-v0` and `heuristic-onset-v1`.
- The command recognizes `magenta-egmd-tf1-94529798-8hit-v1` only to publish
  `canonical_mapping_required` with exit `1`. It does not verify, start, or run OaF
  and does not call mapping or scoring.
- The existing `run_score_midi` mapping, alignment, metric, and summary semantics are
  unchanged.
- Native JSONL is authoritative. MIDI is a deterministic derivative and is never
  parsed to reconstruct native events.
- Optional MIDI is fixed at 220 PPQ, 120.0 quarter notes per minute at tick zero,
  zero-based channel 9, program 0, and 0.05-second note duration.
- A requested MIDI failure retains the valid JSONL, reports
  `midi_derivation_failed`, marks the item incomplete, and returns exit `1`.
- The existing FastAPI `DrumTranscriber.transcribe` path remains unchanged. This
  phase removes implicit fallback only from benchmark compatibility orchestration.
- Standard output is one canonical summary after successful Click parsing. Human
  progress and sanitized diagnostics use standard error.
- Python lines stay within the repository's 100-character formatting limit.

## Execution Order

Complete Phase B first. Tasks 1 and 2 create the independently verifiable heuristic
backend. Task 3 completes the Phase B optional-MIDI seam. Tasks 4 and 5 isolate the
legacy TF2 path and combined-score wrapper. Task 6 performs the intentional CLI
migration characterized in Phase B Task 1. Phase A Task 7 must supply the generic
`verify-backend` command before Phase C Task 7 runs its CLI acceptance checks, but
the native `amd64` Phase A seal is not required.

---

## File Map

### New runtime and lock files

- `runtime/heuristic_onset/pyproject.toml` — isolated environment project with exact
  direct dependencies and Python requirement.
- `runtime/heuristic_onset/uv.lock` — complete resolution for supported platforms.
- `runtime/heuristic_onset/requirements.lock` — selected distribution filenames and
  hashes for the supported execution platform.
- `runtime/heuristic_onset/runner.py` — one-line JSON request/response heuristic
  process with no Crux runtime imports.
- `runtime/heuristic_onset/algorithm.py` — exact onset, classification, and event
  rendering implementation.
- `runtime/heuristic_onset/source-manifest.json` — enumerated adapter files plus
  `covered_roots` for execution-attestation dirty checks.
- `config/benchmark/backends/prediction-midi-v1.writer-lock.json` — exact MIDI
  writer distribution filename, version, SHA-256, and rendering settings.
- `runtime/prediction_midi/wheels/` — checked-in pure-Python Mido and transitive
  Packaging wheels selected by the writer lock.
- `runtime/prediction_midi/distribution-manifest.json` — canonical manifest for the
  checked-in writer distributions.
- `config/benchmark/backends/heuristic-onset-v1.parameters.json` — reviewed
  parameter lock generated from source, interpreter, and distribution identities.

### New host source files

- `src/benchmark/heuristic_lock.py` — strict parameter-lock parsing and descriptor
  construction.
- `src/benchmark/heuristic_process.py` — content-addressed environment validation,
  subprocess protocol, deadlines, and sanitized failure mapping.
- `src/benchmark/backends/heuristic.py` — Phase B `TranscriptionBackend` adapter.
- `src/benchmark/prediction_midi.py` — deterministic MIDI plus strict source-hash
  sidecar publication.
- `src/benchmark/backends/legacy_tf2.py` — no-fallback TF2/HDF5 compatibility
  adapter that returns MIDI bytes.
- `src/benchmark/legacy_score.py` — explicit backend dispatch, existing scorer
  invocation, and typed legacy-score wrapper.

### New tools

- `tools/seal_heuristic_backend.py` — resolve exact algorithm/package/interpreter
  identity and write the reviewed parameter lock.
- `tools/build_heuristic_environment.py` — install and verify the content-addressed
  virtual environment using only lock-selected distributions.
- `tools/seal_prediction_midi_writer.py` — resolve and hash the exact MIDI writer
  distribution used for golden and production rendering.

### New tests

- `tests/benchmark/test_heuristic_lock.py`
- `tests/benchmark/test_heuristic_process.py`
- `tests/benchmark/test_heuristic_backend.py`
- `tests/benchmark/test_heuristic_real_environment.py`
- `tests/benchmark/test_prediction_midi.py`
- `tests/benchmark/test_legacy_tf2_backend.py`
- `tests/benchmark/test_legacy_score.py`
- `tests/fixtures/heuristic_onset/canonical.wav`
- `tests/fixtures/heuristic_onset/expected-events.json`
- `tests/fixtures/prediction_midi/expected.mid`
- `tests/fixtures/prediction_midi/expected.sidecar.json`

### Modified files

- `src/benchmark/backend_registry.py` — activate the lazy heuristic factory while
  continuing to reject legacy TF2 as a `TranscriptionBackend`.
- `src/benchmark/runner.py:115-212` — keep the scoring mechanics but remove implicit
  benchmark construction of `DrumTranscriber`.
- `src/cli/benchmark.py:273-300` — require `--backend`, call `run_legacy_score`, and
  emit the canonical wrapper summary.
- `tests/benchmark/test_runner.py:49-147` — update the characterized boundary to
  require an explicit MIDI-producing callback.
- `tests/test_cli_benchmark.py:325-355` — replace the pre-migration expectation with
  explicit backend and typed-report tests.
- `docs/drumery-dtx-midi-benchmarking-reference.md:363-680` — document compatibility
  IDs, status labels, native JSONL, MIDI derivation, and migration commands.

### Cross-task interfaces

- Heuristic lock: `LockedDistribution`, `HeuristicParameterLock`,
  `load_heuristic_parameter_lock`, `build_heuristic_descriptor`.
- Heuristic process: `HeuristicProcessConfig`, `HeuristicProcess`,
  `ensure_heuristic_environment`.
- Heuristic adapter: `HeuristicBackend`, `create_backend`.
- MIDI: `MidiDerivativeConfig`, Phase B `MidiDerivative`,
  `publish_prediction_midi`.
- Legacy TF2: `LegacyTf2Error`, `LegacyTf2Backend`,
  `create_legacy_tf2_backend`.
- Legacy scorer: `LegacyScoreRequest`, `LegacyScoreOutcome`,
  `run_legacy_score`.

---

### Task 1: Build strict lock tooling for the isolated heuristic environment

**Files:**

- Create: `runtime/heuristic_onset/pyproject.toml`
- Create: `runtime/heuristic_onset/uv.lock`
- Create: `runtime/heuristic_onset/requirements.lock`
- Create: `src/benchmark/heuristic_lock.py`
- Create: `tools/seal_heuristic_backend.py`
- Create: `tools/build_heuristic_environment.py`
- Create: `tests/benchmark/test_heuristic_lock.py`

**Interfaces:**

- Consumes: Phase B strict JSON/hash helpers and the checked-in heuristic runtime
  project.
- Produces: `HeuristicParameterLock`, strict descriptor construction, and tooling
  that can seal and build an environment after Task 2 supplies final adapter source.

- [ ] **Step 1: Write failing strict-lock and descriptor tests**

```python
def test_parameter_lock_rejects_unknown_and_duplicate_fields(tmp_path: Path) -> None:
    path = tmp_path / "parameters.json"
    path.write_text(parameter_lock_text(extra='"unexpected":1,'), encoding="utf-8")

    with pytest.raises(StrictJsonError, match="unknown field"):
        load_heuristic_parameter_lock(path)

    path.write_text(parameter_lock_text(duplicate_backend_id=True), encoding="utf-8")
    with pytest.raises(StrictJsonError, match="duplicate key"):
        load_heuristic_parameter_lock(path)


def test_descriptor_contains_only_heuristic_identity_fields(
    valid_parameter_lock: HeuristicParameterLock,
) -> None:
    descriptor = build_heuristic_descriptor(valid_parameter_lock)

    assert descriptor.to_dict() == {
        "descriptor_schema": "crux.heuristic-backend-descriptor/v1",
        "backend_id": "heuristic-onset-v1",
        "prediction_schema": "crux.drum-prediction-events/v1",
        "adapter_source_manifest_sha256": valid_parameter_lock.adapter_source_manifest_sha256,
        "parameter_lock_sha256": valid_parameter_lock.sha256,
        "model_id": "crux-heuristic-onset-nonmodel-v1",
        "architecture_id": "librosa-onset-centroid-zcr-v1",
        "native_output_space_id": "crux-heuristic-midi7-v1",
        "native_metadata_schema_id": "crux-empty-native-metadata-v1",
    }
```

Add tests requiring exact lowercase SHA-256 values, one distribution record per
normalized package name, lexicographic distribution ordering, unique filenames,
positive interpreter ABI fields, and exact resolved algorithm keys. Reject a lock
that omits any transitive distribution or changes a threshold, call option, branch
order, pitch, velocity, sample rate, hop length, or classification window.

Run:

```bash
rtk uv run pytest -q tests/benchmark/test_heuristic_lock.py
```

Expected: collection FAIL because `src.benchmark.heuristic_lock` does not exist.

- [ ] **Step 2: Define the exact lock records**

In `src/benchmark/heuristic_lock.py`, define immutable records:

```python
@dataclass(frozen=True)
class LockedDistribution:
    name: str
    version: str
    filename: str
    sha256: str


@dataclass(frozen=True)
class HeuristicParameterLock:
    schema: Literal["crux.heuristic-parameter-lock/v1"]
    backend_id: Literal["heuristic-onset-v1"]
    prediction_schema: Literal["crux.drum-prediction-events/v1"]
    python_implementation: str
    python_version: str
    python_abi: str
    platform_tag: str
    adapter_source_manifest_sha256: str
    input_contract: Mapping[str, JsonValue]
    algorithm: Mapping[str, JsonValue]
    distributions: tuple[LockedDistribution, ...]
    sha256: str
```

The `input_contract` object allows exactly `container`, `codec`, `channels`,
`sample_width_bytes`, and `sample_rate`; their values are `RIFF/WAVE`, `PCM`, `1`,
`2`, and `44100`.

The `algorithm` object allows exactly:

```text
sample_rate, hop_length, onset_strength_parameters, onset_detect_parameters,
frame_to_time_parameters, classification_window_hops,
spectral_centroid_thresholds_hz, zero_crossing_rate_thresholds,
classification_branch_order, pitch_velocity_pairs, native_class_format
```

Resolve all Librosa defaults consumed by `onset_strength`,
`onset_detect`, `frames_to_time`, `spectral_centroid`, and
`zero_crossing_rate` into those nested objects. Do not omit a value because it
currently equals a library default.

- [ ] **Step 3: Create and resolve the isolated runtime project**

`runtime/heuristic_onset/pyproject.toml` has no Crux dependency and requires Python
`==3.12.*`. Its direct dependencies are:

```toml
dependencies = [
  "librosa==0.11.0",
  "soundfile==0.13.1",
]
```

Generate the complete independent resolution:

```bash
rtk uv lock --project runtime/heuristic_onset
rtk uv export \
  --project runtime/heuristic_onset \
  --frozen \
  --format requirements.txt \
  --no-emit-project \
  --output-file runtime/heuristic_onset/requirements.lock
```

`tools/seal_heuristic_backend.py` must resolve an exact Python 3.12 patch interpreter,
download the lock-selected artifacts into the requested wheel cache, create a
disposable candidate environment, inspect every installed distribution with
`importlib.metadata`, hash each exact artifact, verify that the transitive closure
equals the environment, hash the adapter source manifest, and write the strict
parameter lock canonically. It fails if an installed package is not locked or a
locked package is absent.

- [ ] **Step 4: Build by content address and verify package bytes**

`tools/build_heuristic_environment.py` accepts:

```text
--parameter-lock, --requirements-lock, --environment-root, --wheel-cache, --download
```

It builds under
`<environment-root>/<parameter-lock-sha256>/`, installs with
`--require-hashes --no-deps`, verifies the interpreter identity and every installed
distribution against the parameter lock, writes a canonical
`environment-manifest.json`, fsyncs it, and atomically publishes the completed
directory. It refuses a symlink, a mismatched pre-existing directory, an unlocked
distribution, or an unavailable wheel. It never falls back to Crux's active
environment. Without explicit `--download`, it performs no network access. With
`--download`, it acquires only lock-enumerated artifacts, verifies every byte before
installation, and leaves an existing valid environment unchanged on acquisition
failure.

- [ ] **Step 5: Test sealing and building with synthetic adapter source**

Add tool tests using two small source files and a local wheel fixture. Seal twice and
require byte-identical parameter-lock bytes. Build twice into the same environment
root and require the second call to verify and reuse the exact content-addressed
directory. Mutate one source byte, wheel byte, installed distribution, interpreter
field, or algorithm value and require a typed mismatch rather than reuse. Prove
network access occurs only with `--download`, only for lock-enumerated URLs, and that
an acquisition/hash failure leaves an existing environment unchanged.

- [ ] **Step 6: Run focused tests and style checks**

```bash
rtk uv run pytest -q tests/benchmark/test_heuristic_lock.py
rtk uv run ruff check \
  src/benchmark/heuristic_lock.py \
  tools/seal_heuristic_backend.py \
  tools/build_heuristic_environment.py \
  tests/benchmark/test_heuristic_lock.py
rtk uv run black --check \
  src/benchmark/heuristic_lock.py \
  tools/seal_heuristic_backend.py \
  tools/build_heuristic_environment.py \
  tests/benchmark/test_heuristic_lock.py
```

Expected: PASS.

- [ ] **Step 7: Commit the environment identity**

```bash
rtk git add \
  runtime/heuristic_onset/pyproject.toml \
  runtime/heuristic_onset/uv.lock \
  runtime/heuristic_onset/requirements.lock \
  src/benchmark/heuristic_lock.py \
  tools/seal_heuristic_backend.py \
  tools/build_heuristic_environment.py \
  tests/benchmark/test_heuristic_lock.py
rtk git commit -m "feat: define heuristic environment lock"
```

---

### Task 2: Implement the explicit heuristic backend

**Files:**

- Create: `runtime/heuristic_onset/algorithm.py`
- Create: `runtime/heuristic_onset/runner.py`
- Generate: `config/benchmark/backends/heuristic-onset-v1.parameters.json`
- Generate: `runtime/heuristic_onset/source-manifest.json`
- Create: `src/benchmark/heuristic_process.py`
- Create: `src/benchmark/backends/heuristic.py`
- Create: `tests/benchmark/test_heuristic_process.py`
- Create: `tests/benchmark/test_heuristic_backend.py`
- Create: `tests/benchmark/test_heuristic_real_environment.py`
- Create: `tests/fixtures/heuristic_onset/canonical.wav`
- Create: `tests/fixtures/heuristic_onset/expected-events.json`
- Modify: `src/benchmark/backend_registry.py`
- Modify: `tests/benchmark/test_backend_registry.py`

**Interfaces:**

- Consumes: Task 1 lock tooling and the Phase B backend protocol.
- Produces: verified `heuristic-onset-v1` native predictions without importing
  Librosa into the Crux process.

- [ ] **Step 1: Write failing algorithm and isolation tests**

Use a small procedurally generated canonical WAV with fixed integer PCM samples.
Check in the resulting WAV and expected event object only after inspecting both.
Put the real-environment fixture test in
`test_heuristic_real_environment.py`, guarded by
`CRUX_RUN_HEURISTIC_INTEGRATION=1`; unit tests use a fake protocol process.

```python
def test_algorithm_matches_locked_fixture(
    heuristic_runner: HeuristicProcess,
    heuristic_audio: CanonicalAudio,
) -> None:
    events = heuristic_runner.transcribe(heuristic_audio)

    assert [event.to_dict() for event in events] == load_expected_events()


def test_backend_does_not_import_numeric_stack_in_host_process(
    heuristic_backend: HeuristicBackend,
) -> None:
    forbidden = {"librosa", "numba", "scipy", "soundfile"}

    assert forbidden.isdisjoint(sys.modules)
    assert heuristic_backend.verify().status == "verified"
    assert forbidden.isdisjoint(sys.modules)


def test_missing_environment_is_failure_not_fallback(
    checked_in_parameter_lock: Path,
    tmp_path: Path,
) -> None:
    backend = create_backend(
        parameter_lock_path=checked_in_parameter_lock,
        environment_root=tmp_path / "missing",
    )

    result = backend.verify()

    assert result.status == "failed"
    assert result.errors[0].code == "heuristic_environment_unavailable"
```

Add process tests for malformed JSON, duplicate keys, unexpected response IDs,
nonfinite results, timeouts, premature exit, stderr redaction, locked-interpreter
mismatch, distribution-byte mismatch, strict dirty-checkout rejection, non-strict
changed-file attestation, unrelated dirty documentation, and clean close.

Run:

```bash
rtk uv run pytest -q \
  tests/benchmark/test_heuristic_process.py \
  tests/benchmark/test_heuristic_backend.py \
  tests/benchmark/test_heuristic_real_environment.py
```

Expected: collection FAIL because the process and adapter modules do not exist.

- [ ] **Step 2: Port the current algorithm without semantic cleanup**

`runtime/heuristic_onset/algorithm.py` must implement exactly:

1. Read canonical PCM samples without resampling.
2. Call `librosa.onset.onset_strength` with locked `sr=44100` and
   `hop_length=512` plus every resolved default.
3. Call `librosa.onset.onset_detect` with the locked envelope, `sr=44100`,
   `hop_length=512`, `backtrack=True`, and every resolved default.
4. Call `librosa.frames_to_time` with the locked sample rate, hop, and resolved
   arguments.
5. For each paired time/frame, classify the two-hop audio slice using the locked
   `spectral_centroid` and `zero_crossing_rate` calls.
6. Apply branches in this exact order:
   `<150`; `<350 and zcr>0.1`; `>3000 and zcr>0.2`; `>2000`; `<1000` with
   `<500`, `<750`, and otherwise sub-branches; otherwise emit no event.
7. Emit fixed pitch/velocity pairs `36:80`, `38:70`, `42:60`, `49:65`,
   `45:65`, `47:65`, and `50:65` from those branches.

Do not change strict inequalities, add normalization, estimate velocity, merge
events, or add a default class. The host later sorts and quantizes through the common
artifact layer.

- [ ] **Step 3: Implement the isolated runner protocol**

`runtime/heuristic_onset/runner.py` accepts newline-delimited strict JSON with exact
operations `handshake`, `transcribe`, and `shutdown`. The handshake returns the
parameter-lock hash, interpreter identity, complete distribution manifest hash,
algorithm source hash, and supported input contract. A transcription response
contains only request ID, verified audio SHA-256, and raw events.

The runner resolves the canonical WAV path beneath its configured read-only input
root, re-hashes the file, validates mono/16-bit/44.1 kHz PCM, and rejects nonempty
trailing data or malformed RIFF chunks. Standard output contains protocol objects
only. Sanitized diagnostics use standard error.

- [ ] **Step 4: Implement host process and backend records**

Define:

```python
@dataclass(frozen=True)
class HeuristicProcessConfig:
    parameter_lock_path: Path
    source_manifest_path: Path
    environment_path: Path
    repository_root: Path
    strict_checkout: bool
    startup_timeout_seconds: float
    request_timeout_seconds: float
    stderr_byte_limit: int


class HeuristicBackend:
    def descriptor(self) -> BackendDescriptor:
        return self._descriptor

    def verify(self) -> BackendVerification:
        return self._process.verify()

    def transcribe(self, audio: CanonicalAudio) -> NativePrediction:
        return self._process.transcribe(audio)

    def close(self) -> None:
        self._process.close()
```

The adapter converts each runner event to:

```python
NativeEvent(
    time_sec=event.time_sec,
    native_class_id=f"midi_{event.pitch}",
    model_output_bin=None,
    native_midi_note=event.pitch,
    native_metadata={},
    confidence=None,
    velocity_midi=event.velocity,
)
```

`verify()` validates the parameter lock, descriptor, environment, handshake, and
canonical input contract. Its `BackendVerification.max_input_audio_frames` is
null because the heuristic parameter lock defines no duration cap. It returns zero
tensor counts and `not_applicable` smoke and
tensor-coverage status through the Phase B verification report. It publishes a Phase
B execution attestation from the checked-in source manifest: container resource
fields are null, while startup/request deadlines are exact positive adapter
configuration. Strict mode rejects an inference-relevant dirty file before process
launch. It never creates or imports `DrumTranscriber`.

- [ ] **Step 5: Activate only the heuristic registry factory**

Update `default_backend_registry()` so `HEURISTIC_BACKEND_ID` lazily imports
`src.benchmark.backends.heuristic.create_backend`. Retain the Phase B behavior that
rejects `LEGACY_TF2_BACKEND_ID` as unavailable through `TranscriptionBackend`.

Add:

```python
def test_registry_never_substitutes_heuristic_for_failed_oaf(
    registry: BackendRegistry,
) -> None:
    registry.replace_for_test(OFFICIAL_BACKEND_ID, failing_oaf_factory)
    registry.replace_for_test(HEURISTIC_BACKEND_ID, recording_heuristic_factory)

    with pytest.raises(BackendUnavailable, match="official unavailable"):
        registry.create(OFFICIAL_BACKEND_ID)

    assert recording_heuristic_factory.calls == 0
```

- [ ] **Step 6: Seal and build the final content-addressed environment**

After every adapter source file above is implemented, generate the final parameter
lock:

```bash
rtk uv run python tools/seal_heuristic_backend.py \
  --python 3.12 \
  --project runtime/heuristic_onset \
  --requirements-lock runtime/heuristic_onset/requirements.lock \
  --wheel-cache artifacts/benchmark/heuristic-wheel-cache \
  --adapter-source runtime/heuristic_onset/algorithm.py \
  --adapter-source runtime/heuristic_onset/runner.py \
  --adapter-source src/benchmark/heuristic_lock.py \
  --adapter-source src/benchmark/heuristic_process.py \
  --adapter-source src/benchmark/backends/heuristic.py \
  --source-manifest runtime/heuristic_onset/source-manifest.json \
  --output config/benchmark/backends/heuristic-onset-v1.parameters.json
rtk uv run python tools/build_heuristic_environment.py \
  --parameter-lock \
    config/benchmark/backends/heuristic-onset-v1.parameters.json \
  --requirements-lock runtime/heuristic_onset/requirements.lock \
  --environment-root artifacts/benchmark/heuristic-environments \
  --wheel-cache artifacts/benchmark/heuristic-wheel-cache \
  --download
```

Review the exact Python patch/ABI/platform, every distribution filename/hash, every
resolved Librosa call argument, the five-file adapter manifest and its
`covered_roots`, and the exact canonical-WAV contract. Re-run both commands. The
lock and source manifest must remain byte-identical and the builder must verify/reuse
the same content-addressed directory.

- [ ] **Step 7: Run focused tests and deterministic replay**

```bash
rtk uv run pytest -q \
  tests/benchmark/test_heuristic_process.py \
  tests/benchmark/test_heuristic_backend.py \
  tests/benchmark/test_backend_registry.py \
  tests/benchmark/test_transcription.py
rtk env CRUX_RUN_HEURISTIC_INTEGRATION=1 uv run pytest -q \
  tests/benchmark/test_heuristic_real_environment.py
```

Run the byte-identity test a second time with the same command and require the same
prediction SHA-256 in both test logs.

- [ ] **Step 8: Commit the heuristic adapter**

```bash
rtk git add \
  runtime/heuristic_onset/algorithm.py \
  runtime/heuristic_onset/runner.py \
  runtime/heuristic_onset/source-manifest.json \
  config/benchmark/backends/heuristic-onset-v1.parameters.json \
  src/benchmark/heuristic_process.py \
  src/benchmark/backends/heuristic.py \
  src/benchmark/backend_registry.py \
  tests/benchmark/test_heuristic_process.py \
  tests/benchmark/test_heuristic_backend.py \
  tests/benchmark/test_heuristic_real_environment.py \
  tests/benchmark/test_backend_registry.py \
  tests/fixtures/heuristic_onset/canonical.wav \
  tests/fixtures/heuristic_onset/expected-events.json
rtk git commit -m "feat: add explicit heuristic backend"
```

---

### Task 3: Add deterministic MIDI derivation from native events

**Files:**

- Create: `src/benchmark/prediction_midi.py`
- Create: `tools/seal_prediction_midi_writer.py`
- Generate: `config/benchmark/backends/prediction-midi-v1.writer-lock.json`
- Generate: `runtime/prediction_midi/wheels/`
- Generate: `runtime/prediction_midi/distribution-manifest.json`
- Create: `tests/benchmark/test_prediction_midi.py`
- Create: `tests/fixtures/prediction_midi/expected.mid`
- Create: `tests/fixtures/prediction_midi/expected.sidecar.json`
- Modify: `src/benchmark/transcription.py`
- Modify: `tests/benchmark/test_transcription.py`

**Interfaces:**

- Consumes: a strict-read Phase B `PredictionArtifact`.
- Produces: `MidiDerivative` and a sidecar bound to the complete prediction
  SHA-256.

- [ ] **Step 1: Write failing golden and failure-retention tests**

```python
def test_midi_derivative_matches_golden(
    prediction_artifact: PredictionArtifact,
    tmp_path: Path,
) -> None:
    result = publish_prediction_midi(
        prediction_artifact,
        tmp_path / "prediction.mid",
    )

    assert result.midi.path.read_bytes() == fixture_bytes("expected.mid")
    assert result.sidecar.path.read_bytes() == fixture_bytes("expected.sidecar.json")


def test_requested_midi_failure_keeps_prediction(
    transcribe_request: TranscribeOneRequest,
    fake_backend: FakeBackend,
    tmp_path: Path,
) -> None:
    outcome = run_transcribe_one(
        replace(transcribe_request, midi_output_path=tmp_path / "prediction.mid"),
        registry=registry_for(fake_backend),
        midi_writer=raising_midi_writer,
    )

    assert outcome.exit_code == 1
    report = read_execution_report(outcome.report_artifact.path)
    assert report.status == "partial"
    assert report.items[0].status == "incomplete"
    assert report.items[0].prediction is not None
    assert report.items[0].midi is None
    assert report.items[0].errors[0].code == "midi_derivation_failed"
```

The golden prediction includes simultaneous events with distinct pitches, a
fractional tick boundary, minimum and maximum velocities, and zero events after the
last note. Add validation tests for null/out-of-range MIDI pitch or velocity and a
sidecar prediction-hash mismatch. Add failures for unknown writer-lock fields,
distribution-manifest drift, missing wheel bytes, installed Mido file drift, and
writer-source hash drift.

Run:

```bash
rtk uv run pytest -q \
  tests/benchmark/test_prediction_midi.py \
  tests/benchmark/test_transcription.py
```

Expected: collection FAIL because `src.benchmark.prediction_midi` does not exist.

- [ ] **Step 2: Define the frozen MIDI contract**

```python
@dataclass(frozen=True)
class MidiDerivativeConfig:
    ticks_per_quarter: Literal[220] = 220
    tempo_qpm: Decimal = Decimal("120.0")
    channel: Literal[9] = 9
    program: Literal[0] = 0
    note_duration_seconds: Decimal = Decimal("0.05")


```

Use the Phase B `MidiDerivative` record for the result.

The writer-lock schema is `crux.prediction-midi-writer-lock/v1`. It records the exact
Python implementation/version/ABI, `prediction_midi.py` source SHA-256, rendering
settings, and the complete Mido/Packaging distribution records with name, version,
filename, and SHA-256.

The sidecar schema is `crux.prediction-midi-derivative/v1` and has exactly:

```text
schema, prediction_sha256, midi_sha256, writer_lock_sha256, writer_distribution,
ticks_per_quarter, tempo_qpm, channel, program, note_duration_seconds
```

`writer_distribution` is an object with exactly `name`, `version`, `filename`, and
`sha256` for `mido==1.3.3`; its wheel is
`mido-1.3.3-py3-none-any.whl` with SHA-256
`01033c9b10b049e4436fca2762194ca839b09a4334091dd3c34e7f4ae674fd8a`.
The writer verifies the checked-in wheel, installed file set, source, interpreter,
and complete lock before importing Mido.

- [ ] **Step 3: Generate and review the writer lock and distributions**

```bash
rtk uv run python tools/seal_prediction_midi_writer.py \
  --project-lock uv.lock \
  --python 3.12 \
  --writer-source src/benchmark/prediction_midi.py \
  --distribution mido==1.3.3 \
  --wheel-output runtime/prediction_midi/wheels \
  --distribution-manifest runtime/prediction_midi/distribution-manifest.json \
  --output config/benchmark/backends/prediction-midi-v1.writer-lock.json
```

Require the known Mido wheel hash above, a complete transitive Packaging record,
`packaging-23.2-py3-none-any.whl` with SHA-256
`8c491190033a9af7e1d931d0b5dacc2ef47509b34dd0de67ed209b5203fc88c7`,
pure-Python compatible tags, canonical manifest bytes, and no extra wheel. Run the
command twice and require no diff in the lock, manifest, or wheels.

- [ ] **Step 4: Render MIDI from sorted native records**

Strict-read the complete prediction artifact first. Strict-load and verify the
writer lock and distributions before importing Mido. Reject any event without
`native_midi_note` or `velocity_midi`. Convert the already quantized decimal seconds
to ticks using exact decimal arithmetic and round half to even. Emit tempo at tick
zero, deterministic note-on/note-off ordering, zero-based channel 9, and no
wall-clock or path metadata.

Write MIDI and sidecar to temporary files in their destination directories, fsync,
strict-read the sidecar, verify its source and MIDI hashes, atomically rename both,
and fsync both parent directories. If either publication fails, remove only
unpublished temporary files; never remove the native JSONL.

- [ ] **Step 5: Wire the default Phase B MIDI callback**

In `run_transcribe_one`, use `publish_prediction_midi` when the request includes
`midi_output_path` and no test callback was injected. Map writer validation,
serialization, and publication failures to item status `incomplete`, execution
status `partial`, stable error `midi_derivation_failed`, and exit `1`.

- [ ] **Step 6: Run focused tests and inspect golden bytes**

```bash
rtk uv run pytest -q \
  tests/benchmark/test_prediction_midi.py \
  tests/benchmark/test_transcription.py
rtk xxd -g 1 tests/fixtures/prediction_midi/expected.mid
rtk sed -n '1,120p' tests/fixtures/prediction_midi/expected.sidecar.json
```

Expected: PASS, with no timestamp, source path, or nondeterministic metadata in the
golden artifacts.

- [ ] **Step 7: Commit MIDI derivation**

```bash
rtk git add \
  src/benchmark/prediction_midi.py \
  src/benchmark/transcription.py \
  tools/seal_prediction_midi_writer.py \
  config/benchmark/backends/prediction-midi-v1.writer-lock.json \
  runtime/prediction_midi/distribution-manifest.json \
  runtime/prediction_midi/wheels/mido-1.3.3-py3-none-any.whl \
  runtime/prediction_midi/wheels/packaging-23.2-py3-none-any.whl \
  tests/benchmark/test_prediction_midi.py \
  tests/benchmark/test_transcription.py \
  tests/fixtures/prediction_midi/expected.mid \
  tests/fixtures/prediction_midi/expected.sidecar.json
rtk git commit -m "feat: derive deterministic MIDI from native events"
```

---

### Task 4: Isolate the legacy TF2/HDF5 compatibility path

**Files:**

- Create: `src/benchmark/backends/legacy_tf2.py`
- Create: `tests/benchmark/test_legacy_tf2_backend.py`

**Interfaces:**

- Consumes: existing `DrumTranscriber` TF2 feature/inference/MIDI helpers.
- Produces: explicit MIDI bytes or a stable compatibility failure; never a frozen
  prediction descriptor.

- [ ] **Step 1: Write failing no-fallback tests**

```python
def test_missing_legacy_model_fails_without_heuristic(
    monkeypatch,
    tmp_path: Path,
) -> None:
    detect = Mock(side_effect=AssertionError("heuristic must not run"))
    monkeypatch.setattr(DrumTranscriber, "_detect_onsets_from_audio", detect)
    backend = create_legacy_tf2_backend(model_path=tmp_path / "missing.weights.h5")

    with pytest.raises(LegacyTf2Error, match="legacy_model_unavailable"):
        backend.transcribe_midi(canonical_wav_path())

    detect.assert_not_called()


def test_tf2_inference_failure_is_not_retried_as_heuristic(monkeypatch) -> None:
    transcriber = fake_built_transcriber()
    monkeypatch.setattr(
        transcriber,
        "_run_tf2_model_inference",
        Mock(side_effect=ValueError("bad tensor")),
    )
    detect = Mock(side_effect=AssertionError("heuristic must not run"))
    monkeypatch.setattr(transcriber, "_detect_onsets_from_audio", detect)

    with pytest.raises(LegacyTf2Error, match="legacy_inference_failed"):
        LegacyTf2Backend(transcriber).transcribe_midi(canonical_wav_path())

    detect.assert_not_called()
```

Add tests for corrupt HDF5 weights, missing TensorFlow, decode failure, successful
MIDI bytes, missing default weights without a network request, and an explicit proof
that `_download_model` is never called. Assert the module exposes no `descriptor()`,
`verify()`, or Phase B `NativePrediction`.

Run:

```bash
rtk uv run pytest -q tests/benchmark/test_legacy_tf2_backend.py
```

Expected: collection FAIL because `src.benchmark.backends.legacy_tf2` does not
exist.

- [ ] **Step 2: Implement the narrow compatibility adapter**

Define:

```python
class LegacyTf2Error(RuntimeError):
    def __init__(self, code: str, sanitized_message: str) -> None:
        super().__init__(f"{code}: {sanitized_message}")
        self.code = code
        self.sanitized_message = sanitized_message


class LegacyTf2Backend:
    backend_id = "legacy-tf2-h5-v0"

    def transcribe_midi(self, audio_path: Path) -> bytes:
        audio = self._load_audio(audio_path)
        model = self._require_model()
        drum_events = self._run_model(model, audio)
        return self._render_midi(drum_events)
```

Construct `DrumTranscriber` with the explicit HDF5 path and call only its TF2 model,
audio feature/inference, and MIDI helpers. Do not call
`DrumTranscriber.transcribe`, because that public method is permitted to enter the
service fallback. If `_build_model()` returns null or any requested TF2 stage fails,
raise a stable `LegacyTf2Error`.

When no path is injected, resolve only
`DrumTranscriber.TF2_WEIGHTS_RELATIVE_PATH` through
`DrumTranscriber._resolve_existing_path`. Require the resolved regular filename to
end in `.weights.h5`; if it is absent, fail without constructing a transcriber.
Never call `_download_model`, scan for a TF1 checkpoint, or make a network request.

Sanitize exception text before it leaves this module. Do not expose an absolute path,
environment value, model URL, audio bytes, or traceback.

- [ ] **Step 3: Prove compatibility labeling and registry exclusion**

Add:

```python
def test_legacy_tf2_is_not_a_transcription_backend() -> None:
    backend = create_legacy_tf2_backend(model_path=Path("model.weights.h5"))

    assert not hasattr(backend, "descriptor")
    assert not hasattr(backend, "verify")
    assert not hasattr(backend, "transcribe")
    with pytest.raises(BackendUnavailable):
        default_backend_registry().create(LEGACY_TF2_BACKEND_ID)
```

The contract under test is that the compatibility object cannot enter
frozen-backend orchestration.

- [ ] **Step 4: Run focused tests and fallback regression**

```bash
rtk uv run pytest -q \
  tests/benchmark/test_legacy_tf2_backend.py \
  tests/benchmark/test_backend_registry.py \
  tests/test_transcriber_fallback.py
rtk uv run ruff check \
  src/benchmark/backends/legacy_tf2.py \
  tests/benchmark/test_legacy_tf2_backend.py
```

Expected: PASS. The FastAPI fallback characterization remains green while benchmark
compatibility tests prove it is unreachable.

- [ ] **Step 5: Commit the explicit legacy adapter**

```bash
rtk git add \
  src/benchmark/backends/legacy_tf2.py \
  tests/benchmark/test_legacy_tf2_backend.py \
  tests/benchmark/test_backend_registry.py
rtk git commit -m "feat: isolate legacy TF2 transcription"
```

---

### Task 5: Add typed legacy combined-score orchestration

**Files:**

- Create: `src/benchmark/legacy_score.py`
- Create: `tests/benchmark/test_legacy_score.py`
- Modify: `src/benchmark/runner.py:115-212`
- Modify: `tests/benchmark/test_runner.py:49-147`

**Interfaces:**

- Consumes: Task 2 heuristic backend, Task 3 MIDI derivative, Task 4 legacy TF2
  adapter, existing `run_score_midi`, and Phase B legacy report publication.
- Produces: `LegacyScoreOutcome` for one explicit compatibility backend.

- [ ] **Step 1: Write failing dispatch, refusal, and wrapper tests**

```python
def test_oaf_refusal_runs_neither_backend_nor_scorer(
    request: LegacyScoreRequest,
    spies: LegacyScoreSpies,
) -> None:
    outcome = run_legacy_score(
        replace(request, backend_id=OFFICIAL_BACKEND_ID),
        dependencies=spies.dependencies,
    )

    report = read_legacy_score_report(outcome.report_artifact.path)
    assert outcome.exit_code == 1
    assert report.status == "canonical_mapping_required"
    assert report.backend_validation_status == "not_checked"
    assert report.score_report is None
    assert spies.oaf.calls == 0
    assert spies.heuristic.calls == 0
    assert spies.legacy_tf2.calls == 0
    assert spies.score.calls == 0


@pytest.mark.parametrize(
    ("backend_id", "validation_status"),
    [
        ("legacy-tf2-h5-v0", "unvalidated"),
        ("heuristic-onset-v1", "parameter_locked"),
    ],
)
def test_success_wraps_existing_score_report(
    request: LegacyScoreRequest,
    backend_id: str,
    validation_status: str,
    successful_dependencies: LegacyScoreDependencies,
) -> None:
    outcome = run_legacy_score(
        replace(request, backend_id=backend_id),
        dependencies=successful_dependencies,
    )

    report = read_legacy_score_report(outcome.report_artifact.path)
    assert outcome.exit_code == 0
    assert report.workflow_mode == "legacy_compatibility"
    assert report.backend_id == backend_id
    assert report.backend_validation_status == validation_status
    assert report.status == "complete"
    assert report.score_report.sha256 == sha256_file(report.score_report.path)
```

Add tests for missing audio, one or more transcription failures, no matched charts,
heuristic verification failure, score failure, report publication failure, stable
error ordering, repository-relative score-report paths, and exact summary schema.

Run:

```bash
rtk uv run pytest -q \
  tests/benchmark/test_legacy_score.py \
  tests/benchmark/test_runner.py
```

Expected: collection FAIL because `src.benchmark.legacy_score` does not exist.

- [ ] **Step 2: Make the low-level runner callback-only**

Change `run_transcribe_and_score` to require
`transcribe: Callable[[Path], bytes]`. Remove `_create_shared_transcriber` and
`_make_transcribe_fn` from `src/benchmark/runner.py`. Keep all existing chart
selection, stale compatibility-output cleanup, MIDI writing, matched-chart copying,
warnings, and `run_score_midi(..., align=True)` behavior unchanged.

Update its tests so every call passes a fake MIDI callback. The Phase B
characterization test now asserts that omitting `transcribe` is a Python argument
error rather than implicit `DrumTranscriber` construction.

- [ ] **Step 3: Define legacy request and outcome records**

```python
@dataclass(frozen=True)
class LegacyScoreRequest:
    backend_id: str
    charts_dir: Path
    audio_dir: Path
    output_dir: Path
    tolerance_ms: tuple[int, ...]
    reports_root: Path


@dataclass(frozen=True)
class LegacyScoreOutcome:
    status: Literal["complete", "failed", "canonical_mapping_required"]
    exit_code: Literal[0, 1]
    report_artifact: PublishedArtifact
```

`run_legacy_score` must:

1. Refuse OaF before resolving any backend or opening score output.
2. For `legacy-tf2-h5-v0`, create one shared `LegacyTf2Backend` and pass its
   `transcribe_midi` method to `run_transcribe_and_score`.
3. For `heuristic-onset-v1`, verify one `HeuristicBackend`, canonical-validate each
   complete WAV, use the chart stem as `source_audio_id`,
   `legacy-compatibility-canonical-wav-v1` as `input_view_id`, and the same re-hashed
   WAV SHA-256 as source/input identity, produce and strict-read a native artifact,
   derive MIDI from its sorted events, and return the MIDI bytes to the existing
   scorer callback.
4. Reject every other ID before creating a backend.
5. After `run_transcribe_and_score` succeeds, require
   `<output_dir>/summary.json`, hash those exact existing report bytes, and publish
   exactly one `crux.legacy-score-report/v1`.
6. Close a constructed backend in `finally`.

If wrapper publication fails, propagate the Phase B
`OperationalReportPublicationError`; do not return a `LegacyScoreOutcome` with a
missing artifact or imply that the existing score report is the typed wrapper.

Temporary per-chart native and MIDI artifacts used by the combined compatibility
workflow live beneath `<output_dir>/_legacy_native/<backend-id>/`. They do not claim
official benchmark eligibility and are not referenced by the legacy wrapper; the
wrapper references only the existing combined-score report. A failed attempt must
not delete previously published immutable Phase B native predictions.

- [ ] **Step 4: Preserve exact legacy wrapper semantics**

Populate exactly:

```text
schema, report_type, run_id, started_at, finished_at, workflow_mode, backend_id,
backend_validation_status, status, exit_code, score_report, errors
```

Use `report_type: "legacy_score"` and
`workflow_mode: "legacy_compatibility"`. Validation status is `unvalidated`,
`parameter_locked`, or `not_checked` according to the selected ID. A failure always
has exit `1`, a null score report unless the existing report completed and was
strictly hashed, and sorted sanitized errors.

- [ ] **Step 5: Run focused tests and existing scoring regressions**

```bash
rtk uv run pytest -q \
  tests/benchmark/test_legacy_score.py \
  tests/benchmark/test_runner.py \
  tests/benchmark/test_reports.py \
  tests/benchmark/test_scoring.py
```

Expected: PASS.

- [ ] **Step 6: Commit legacy-score orchestration**

```bash
rtk git add \
  src/benchmark/legacy_score.py \
  src/benchmark/runner.py \
  tests/benchmark/test_legacy_score.py \
  tests/benchmark/test_runner.py
rtk git commit -m "feat: type legacy transcription scoring"
```

---

### Task 6: Require explicit backend selection in the CLI

**Files:**

- Modify: `src/cli/benchmark.py:273-300`
- Modify: `tests/test_cli_benchmark.py:325-355`

**Interfaces:**

- Consumes: `LegacyScoreRequest` and `run_legacy_score`.
- Produces: the intentional command-line migration and one canonical summary line.

- [ ] **Step 1: Replace characterization with failing migration tests**

```python
def test_transcribe_and_score_requires_backend() -> None:
    result = CliRunner().invoke(
        main,
        [
            "benchmark",
            "transcribe-and-score",
            "--charts-dir",
            "charts",
            "--audio-dir",
            "audio",
        ],
    )

    assert result.exit_code == 2
    assert "Missing option '--backend'" in result.output
    assert '"report_path"' not in result.output


def test_oaf_selection_emits_typed_refusal(monkeypatch, tmp_path: Path) -> None:
    refusal = published_legacy_outcome(
        tmp_path,
        status="canonical_mapping_required",
        exit_code=1,
    )
    run = Mock(return_value=refusal)
    monkeypatch.setattr("src.benchmark.legacy_score.run_legacy_score", run)

    result = invoke_transcribe_and_score(
        tmp_path,
        backend="magenta-egmd-tf1-94529798-8hit-v1",
    )

    assert result.exit_code == 1
    assert json.loads(result.stdout) == {
        "exit_code": 1,
        "report_path": refusal.report_artifact.path.as_posix(),
        "report_sha256": refusal.report_artifact.sha256,
        "status": "canonical_mapping_required",
    }
```

Add parameterized tests for both executable compatibility IDs, unknown-ID Click
rejection, one-line standard output, progress on standard error, wrapper publication,
unvalidated legacy TF2 labeling, parameter-locked heuristic labeling, and exit `1`
on transcription/scoring failure.

Run:

```bash
rtk uv run pytest -q \
  tests/test_cli_benchmark.py::test_transcribe_and_score_requires_backend \
  tests/test_cli_benchmark.py::test_oaf_selection_emits_typed_refusal
```

Expected: FAIL because the current command has no `--backend` option and still invokes
the implicit transcriber path.

- [ ] **Step 2: Add the required exact-ID option**

Add:

```python
@click.option(
    "--backend",
    type=click.Choice(
        [
            "legacy-tf2-h5-v0",
            "heuristic-onset-v1",
            "magenta-egmd-tf1-94529798-8hit-v1",
        ],
        case_sensitive=True,
    ),
    required=True,
)
```

Do not set a default and do not accept short aliases such as `legacy`, `heuristic`,
or `oaf`.

- [ ] **Step 3: Wire orchestration and canonical summary output**

Resolve output exactly as today, construct `LegacyScoreRequest`, call
`run_legacy_score`, and set `reports_root` to
`artifacts/benchmark/backends/<backend-id>` through the Phase B operational-root
helper. Emit the Phase B canonical summary containing only `status`, `exit_code`,
`report_path`, and `report_sha256`. Raise
`click.exceptions.Exit(outcome.exit_code)` after writing the summary when the exit
code is nonzero.

Once Click parsing succeeds, do not use `ClickException` for domain failures because
it would replace the machine-readable summary with human prose.
For `OperationalReportPublicationError`, emit no summary, write
`report_publication_failed` and a sanitized message to standard error, and exit `2`.

- [ ] **Step 4: Run CLI tests and inspect help**

```bash
rtk uv run pytest -q \
  tests/test_cli_benchmark.py \
  tests/benchmark/test_legacy_score.py
rtk uv run crux benchmark transcribe-and-score --help
```

Expected: PASS. Help lists all three exact IDs and shows `--backend` as required.

- [ ] **Step 5: Commit the CLI migration**

```bash
rtk git add src/cli/benchmark.py tests/test_cli_benchmark.py
rtk git commit -m "feat: require benchmark transcription backend"
```

---

### Task 7: Document and verify the compatibility boundary

**Files:**

- Modify: `docs/drumery-dtx-midi-benchmarking-reference.md:363-680`
- Test: all files created or modified by this plan.

**Interfaces:**

- Consumes: completed Tasks 1-6.
- Produces: user migration guidance and the Phase C acceptance evidence.

- [ ] **Step 1: Document exact commands and eligibility**

Add examples for:

```bash
rtk uv run python tools/build_heuristic_environment.py \
  --parameter-lock \
    config/benchmark/backends/heuristic-onset-v1.parameters.json \
  --requirements-lock runtime/heuristic_onset/requirements.lock \
  --environment-root artifacts/benchmark/heuristic-environments \
  --wheel-cache artifacts/benchmark/heuristic-wheel-cache \
  --download
rtk uv run crux benchmark verify-backend --backend heuristic-onset-v1
rtk uv run crux benchmark transcribe-one \
  --backend heuristic-onset-v1 \
  --audio artifacts/benchmark/input.wav \
  --source-audio-id song-001 \
  --input-view-id canonical-wav-v1 \
  --output artifacts/benchmark/predictions/song-001.jsonl
rtk uv run crux benchmark transcribe-and-score \
  --backend legacy-tf2-h5-v0 \
  --charts-dir artifacts/benchmark/charts \
  --audio-dir artifacts/benchmark/audio \
  --output-dir artifacts/benchmark/legacy-score
```

State explicitly that:

- heuristic output is parameter-locked native evidence, not OaF output;
- legacy TF2/HDF5 is unvalidated and prohibited from official comparison;
- OaF cannot use the combined scorer before HPA-324/HPA-325;
- old invocations without `--backend` now fail with Click exit `2`; and
- requested MIDI failure leaves native JSONL intact and returns exit `1`.

- [ ] **Step 2: Run the Phase C focused acceptance suite**

```bash
rtk uv run pytest -q \
  tests/benchmark/test_heuristic_lock.py \
  tests/benchmark/test_heuristic_process.py \
  tests/benchmark/test_heuristic_backend.py \
  tests/benchmark/test_prediction_midi.py \
  tests/benchmark/test_legacy_tf2_backend.py \
  tests/benchmark/test_legacy_score.py \
  tests/benchmark/test_backend_registry.py \
  tests/benchmark/test_transcription.py \
  tests/benchmark/test_runner.py \
  tests/test_cli_benchmark.py \
  tests/test_transcriber_fallback.py
```

Expected: PASS.

Run the real isolated-environment replay separately:

```bash
rtk env CRUX_RUN_HEURISTIC_INTEGRATION=1 uv run pytest -q \
  tests/benchmark/test_heuristic_real_environment.py
```

Expected: PASS after the explicit environment-preparation command in Step 1.

- [ ] **Step 3: Run the repository quality gates**

```bash
rtk uv run pytest
rtk uv run ruff check src tests tools
rtk uv run black --check src tests tools
rtk uv run pylint src/app src/cli
```

Expected: PASS. If a pre-existing unrelated failure occurs, record its exact command,
test name, and error separately; do not weaken HPA-320 checks.

- [ ] **Step 4: Prove there is no benchmark fallback or identity confusion**

```bash
rtk rg -n \
  "_detect_onsets_from_audio|DrumTranscriber\\.transcribe|legacy-tf2-h5-v0" \
  src/benchmark src/cli tests/benchmark tests/test_cli_benchmark.py
rtk rg -n \
  "magenta-egmd-tf1-94529798-8hit-v1|heuristic-onset-v1" \
  src/benchmark/backends src/benchmark/legacy_score.py
```

Inspect every match. Required result:

- no benchmark module calls `_detect_onsets_from_audio`;
- no benchmark module calls `DrumTranscriber.transcribe`;
- legacy TF2 appears only in compatibility dispatch/reporting;
- heuristic native artifacts contain only the heuristic descriptor; and
- OaF legacy-score refusal occurs before any backend factory or scorer call.

- [ ] **Step 5: Commit documentation and acceptance**

```bash
rtk git add docs/drumery-dtx-midi-benchmarking-reference.md
rtk git commit -m "docs: explain transcription compatibility backends"
```

- [ ] **Step 6: Record final verification evidence**

Capture:

```bash
rtk git status --short
rtk git log --oneline --decorate -12
rtk git diff origin/main...HEAD --check
```

Expected: clean worktree, the Phase C commits in task order, and no whitespace
errors.
