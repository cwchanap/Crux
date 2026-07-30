# HPA-320 Phase C Compatibility Closure Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use
> superpowers:subagent-driven-development (recommended) or
> superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox
> (`- [ ]`) syntax for tracking.

**Goal:** Finish the explicit compatibility backends, deterministic MIDI derivative,
and typed legacy-scoring boundary required by the final HPA-320 design without
allowing either compatibility path to impersonate the frozen OaF backend.

**Architecture:** Preserve the existing common Phase B contracts and replace only
the remaining implicit benchmark transcription seam. Run `heuristic-onset-v1`
through a content-addressed native-`linux/amd64` environment whose interpreter,
distribution bytes, adapter source, and resolved algorithm values are locked. Keep
`legacy-tf2-h5-v0` outside `TranscriptionBackend` and call only its direct TF2/HDF5
path. Both compatibility choices feed the unchanged combined scorer through an
explicit MIDI callback and publish a strict `crux.legacy-score-report/v1` wrapper.

**Tech Stack:** Python 3.12 host, isolated `uv` environments, Librosa 0.11,
NumPy, SciPy, Numba, SoundFile, Mido 1.3.3, TensorFlow 2 compatibility code,
Click 8, Pytest, canonical JSON/JSONL, SHA-256, Ruff, Black, and Pylint.

## Global Constraints

- Prerequisite: complete
  `docs/superpowers/plans/2026-07-28-hpa-320-phase-b-contract-closure.md`.
- Phase A may proceed independently, but the final Task 7 acceptance uses its
  generic `verify-backend` and `transcribe-one` commands.
- Normative design:
  `docs/superpowers/specs/2026-07-26-hpa-320-freeze-oaf-drums-backend-design.md`
  at or after commit `d2ca20c`.
- This is a delta plan. The historical Phase C plan was written before the final
  schema appendix and before Phase B landed; use this plan for implementation.
- `heuristic-onset-v1` is selected only by its complete ID. A missing, corrupt,
  incompatible, or slow model never selects it.
- Its descriptor schema is `crux.heuristic-backend-descriptor/v1`; its parameter
  lock is `crux.heuristic-parameter-lock/v1`.
- Its identities are exactly `crux-heuristic-onset-nonmodel-v1`,
  `librosa-onset-centroid-zcr-v1`, `crux-heuristic-midi7-v1`, and
  `crux-empty-native-metadata-v1`.
- The parameter-lock root has exactly
  `adapter_source_manifest_sha256`, `algorithm`, `backend_id`, `distributions`,
  `platform`, `python_abi`, `python_implementation`, `python_version`, `schema`,
  `serialization`, and `wheel_tags`. Do not restore stale `prediction_schema`,
  `input_contract`, or `platform_tag` fields from the historical plan.
- Official heuristic verification and publication are native `linux/amd64` only.
  Other platforms return `environment_unsupported` and publish no official
  prediction under this ID.
- Heuristic input is canonical mono 16-bit 44.1 kHz PCM WAV. Preparation remains
  outside the adapter.
- Heuristic events use unpadded `midi_<pitch>`, null output bin, null confidence,
  exact empty metadata, and the pitch/velocity pairs `36:80`, `38:70`, `42:60`,
  `45:65`, `47:65`, `49:65`, and `50:65`.
- `legacy-tf2-h5-v0` is combined-score-only, has no frozen descriptor or lock, and
  is always labeled `backend_validation_status: "unvalidated"`.
- `transcribe-and-score` has no default. Its executable choices are
  `legacy-tf2-h5-v0` and `heuristic-onset-v1`.
- The command recognizes the OaF ID only to publish
  `canonical_mapping_required`, exit `1`. It invokes no OaF factory, mapping, or
  scoring code.
- The existing mapping, alignment, metric, per-chart report, and `summary.json`
  semantics remain unchanged.
- Native JSONL is authoritative. MIDI is never parsed to reconstruct native events.
- MIDI uses 220 PPQ, 120.0 QPM at tick zero, zero-based channel 9, program 0, and
  0.05-second notes.
- A requested MIDI failure retains valid JSONL, reports
  `midi_derivation_failed`, marks the item incomplete, and exits `1`.
- The FastAPI `DrumTranscriber.transcribe` fallback behavior is outside HPA-320 and
  remains unchanged. No benchmark module may call it after Task 5.
- Every post-Click outcome publishes one canonical summary containing exactly
  `status`, `exit_code`, `report_path`, and `report_sha256`.
- Python lines remain within 100 characters.

## Execution Order

Tasks 1 and 2 freeze and implement the heuristic. Task 3 completes the optional MIDI
seam. Tasks 4 and 5 isolate the legacy TF2 path and add the typed compatibility
wrapper. Task 6 performs the intentional CLI migration. Task 7 appends Phase C
goldens and runs the cross-phase acceptance suite.

---

## File Map

### New runtime and lock files

- `runtime/heuristic_onset/pyproject.toml`
- `runtime/heuristic_onset/uv.lock`
- `runtime/heuristic_onset/requirements.lock`
- `runtime/heuristic_onset/algorithm.py`
- `runtime/heuristic_onset/runner.py`
- `runtime/heuristic_onset/source-manifest.json`
- `config/benchmark/backends/heuristic-onset-v1.parameters.json`
- `config/benchmark/backends/prediction-midi-v1.writer-lock.json`
- `runtime/prediction_midi/distribution-manifest.json`
- `runtime/prediction_midi/wheels/mido-1.3.3-py3-none-any.whl`
- `runtime/prediction_midi/wheels/packaging-23.2-py3-none-any.whl`

### New source and tools

- `src/benchmark/heuristic_lock.py`
- `src/benchmark/heuristic_process.py`
- `src/benchmark/backends/heuristic.py`
- `src/benchmark/prediction_midi.py`
- `src/benchmark/backends/legacy_tf2.py`
- `src/benchmark/legacy_score.py`
- `tools/seal_heuristic_backend.py`
- `tools/build_heuristic_environment.py`
- `tools/seal_prediction_midi_writer.py`

### New tests and fixtures

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

- `src/benchmark/backend_registry.py`
- `src/benchmark/transcription.py`
- `src/benchmark/runner.py`
- `src/cli/benchmark.py`
- `tests/benchmark/test_backend_registry.py`
- `tests/benchmark/test_transcription.py`
- `tests/benchmark/test_runner.py`
- `tests/test_cli_benchmark.py`
- `tests/benchmark/schema_goldens/manifest.json`
- `docs/drumery-dtx-midi-benchmarking-reference.md`

### Cross-task interfaces

- Heuristic lock:
  `LockedDistribution`, `HeuristicParameterLock`,
  `load_heuristic_parameter_lock`,
  `load_heuristic_parameter_lock_bytes`, `build_heuristic_descriptor`.
- Heuristic process:
  `HeuristicProcessConfig`, `HeuristicProcess`,
  `ensure_heuristic_environment`.
- Heuristic backend: `HeuristicBackend`, `create_backend`.
- MIDI:
  `MidiDerivativeConfig`, existing `MidiDerivative`,
  `publish_prediction_midi`.
- Legacy TF2:
  `LegacyTf2Error`, `LegacyTf2Backend`,
  `create_legacy_tf2_backend`.
- Legacy score:
  `LegacyScoreRequest`, `LegacyScoreOutcome`, `LegacyScoreDependencies`,
  `run_legacy_score`.

---

### Task 1: Freeze the heuristic parameter and environment contract

**Files:**

- Create: `runtime/heuristic_onset/pyproject.toml`
- Create: `runtime/heuristic_onset/uv.lock`
- Create: `runtime/heuristic_onset/requirements.lock`
- Create: `src/benchmark/heuristic_lock.py`
- Create: `tools/seal_heuristic_backend.py`
- Create: `tools/build_heuristic_environment.py`
- Create: `tests/benchmark/test_heuristic_lock.py`

**Interfaces:**

- Consumes: Phase B strict JSON, hash, descriptor, and atomic-publication helpers.
- Produces:

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
    platform: Literal["linux/amd64"]
    python_implementation: str
    python_version: str
    python_abi: str
    wheel_tags: tuple[str, ...]
    adapter_source_manifest_sha256: str
    algorithm: Mapping[str, JsonValue]
    distributions: tuple[LockedDistribution, ...]
    serialization: Mapping[str, JsonValue]
    sha256: str
```

- [ ] **Step 1: Write failing strict-lock tests**

Add tests that accept the exact root keys and reject any missing, extra, duplicate,
renamed, or incorrectly typed key. Include explicit regressions proving
`prediction_schema`, `input_contract`, and `platform_tag` are rejected.

```python
def test_parameter_lock_uses_final_root_schema(valid_lock_bytes: bytes) -> None:
    lock = load_heuristic_parameter_lock_bytes(valid_lock_bytes)

    assert lock.backend_id == "heuristic-onset-v1"
    assert lock.platform == "linux/amd64"
    assert set(strict_json_loads(valid_lock_bytes)) == {
        "adapter_source_manifest_sha256",
        "algorithm",
        "backend_id",
        "distributions",
        "platform",
        "python_abi",
        "python_implementation",
        "python_version",
        "schema",
        "serialization",
        "wheel_tags",
    }
```

Require lowercase SHA-256, normalized unique package names, unique filenames,
lexicographic distribution ordering, a nonempty compatible wheel-tag set, and exact
interpreter fields. Reject version-only package agreement.

- [ ] **Step 2: Run the focused test and confirm the module is absent**

```bash
rtk uv run pytest -q tests/benchmark/test_heuristic_lock.py
```

Expected: collection fails because `src.benchmark.heuristic_lock` does not exist.

- [ ] **Step 3: Implement the final algorithm schema**

`algorithm` has exactly:

```text
classifier_branches, fixed_native_midi_notes, fixed_velocity_midi,
hop_length, librosa_calls, sample_rate
```

Each Librosa call row has exactly `arguments` and `call`; each argument row has
exactly `name` and `value`; each classifier row has exactly `condition`,
`native_midi_note`, and `velocity_midi`. Freeze all consumed defaults as argument
rows, including `backtrack=true`, frame-to-time conversion, the two-hop
classification window, spectral-centroid thresholds `150`, `350`, `500`, `750`,
`1000`, `2000`, and `3000`, and zero-crossing thresholds `0.1` and `0.2`.

The two fixed arrays are exactly:

```python
fixed_native_midi_notes = [36, 38, 42, 45, 47, 49, 50]
fixed_velocity_midi = [80, 70, 60, 65, 65, 65, 65]
```

`sample_rate` is `44100`; `hop_length` is `512`. `serialization` uses the shared
normative shape and has exactly `encoding`, `final_newline`, `key_order`, and
`whitespace`. Numeric validation and event ordering remain in the common prediction
contract and the locked algorithm/protocol tests; do not add them as open
serialization keys.

- [ ] **Step 4: Create the isolated resolution**

The project has no Crux dependency, uses `requires-python = "==3.12.*"`, and pins:

```toml
dependencies = [
  "librosa==0.11.0",
  "soundfile==0.13.1",
]
```

The sealer records the exact selected Python patch and ABI. Generate the full
independent resolution:

```bash
rtk uv lock --project runtime/heuristic_onset
rtk uv export \
  --project runtime/heuristic_onset \
  --frozen \
  --format requirements.txt \
  --no-emit-project \
  --output-file runtime/heuristic_onset/requirements.lock
```

The exported lock must contain hashes for Librosa, NumPy, SciPy, Numba, SoundFile,
and every transitive numeric/audio distribution selected for native
`linux/amd64`.

- [ ] **Step 5: Implement deterministic seal and build tools**

`seal_heuristic_backend.py` creates a disposable native candidate environment,
inspects every installed distribution with `importlib.metadata`, identifies the
exact installed artifact filename, hashes its bytes, verifies that installed and
resolved transitive sets are identical, hashes the final adapter source manifest,
and writes canonical lock bytes.

`build_heuristic_environment.py` accepts exactly:

```text
--parameter-lock, --requirements-lock, --environment-root, --wheel-cache, --download
```

It publishes to `<environment-root>/<parameter-lock-sha256>/`, installs with
`--require-hashes --no-deps`, verifies interpreter and every distribution byte, and
atomically writes `environment-manifest.json`. Without `--download` it performs no
network access. It rejects symlinks, unlocked packages, missing packages, altered
wheel bytes, and mismatched pre-existing environments.

- [ ] **Step 6: Test reproducible sealing and building**

Use local synthetic wheels and adapter files. Seal twice and require byte-identical
lock output. Build twice and require reuse of the same content-addressed directory.
Mutate one source byte, distribution byte, interpreter field, wheel tag, algorithm
value, or serialization value and require a typed mismatch. Prove a failed
acquisition cannot alter a previously valid environment.

- [ ] **Step 7: Run checks and commit**

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
rtk git add \
  runtime/heuristic_onset/pyproject.toml \
  runtime/heuristic_onset/uv.lock \
  runtime/heuristic_onset/requirements.lock \
  src/benchmark/heuristic_lock.py \
  tools/seal_heuristic_backend.py \
  tools/build_heuristic_environment.py \
  tests/benchmark/test_heuristic_lock.py
rtk git commit -m "feat: freeze heuristic environment contract"
```

---

### Task 2: Implement the isolated heuristic backend

**Files:**

- Create: `runtime/heuristic_onset/algorithm.py`
- Create: `runtime/heuristic_onset/runner.py`
- Create: `runtime/heuristic_onset/source-manifest.json`
- Create: `config/benchmark/backends/heuristic-onset-v1.parameters.json`
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

- Consumes: Task 1 lock and Phase B `TranscriptionBackend`.
- Produces verified heuristic native predictions without importing the numeric stack
  into the Crux host process.

- [ ] **Step 1: Write failing algorithm, protocol, and isolation tests**

Check in a procedurally generated canonical WAV only after reviewing its integer PCM
samples and expected events.

```python
def test_backend_emits_exact_heuristic_envelope(
    backend: HeuristicBackend,
    canonical_audio: CanonicalAudio,
) -> None:
    prediction = backend.transcribe(canonical_audio)

    assert all(event.model_output_bin is None for event in prediction.events)
    assert all(event.confidence is None for event in prediction.events)
    assert all(event.native_metadata == {} for event in prediction.events)
    assert {
        (event.native_midi_note, event.velocity_midi)
        for event in prediction.events
    }.issubset({(36, 80), (38, 70), (42, 60), (45, 65), (47, 65), (49, 65), (50, 65)})
```

Add tests for strict canonical WAV validation; request/response ID mismatch;
duplicate/unknown JSON keys; nonfinite times; unknown pitch/velocity pairs;
malformed lines; timeout; premature exit; bounded sanitized stderr; interpreter,
distribution, and manifest mismatch; non-native platform; clean close; strict dirty
checkout; and non-strict changed-file attestation.

Assert `librosa`, `numba`, `scipy`, and `soundfile` never enter host
`sys.modules`. Guard the real environment replay with
`CRUX_RUN_HEURISTIC_INTEGRATION=1`.

- [ ] **Step 2: Run tests and confirm the implementation is absent**

```bash
rtk uv run pytest -q \
  tests/benchmark/test_heuristic_process.py \
  tests/benchmark/test_heuristic_backend.py \
  tests/benchmark/test_heuristic_real_environment.py
```

Expected: collection fails because the process and backend modules do not exist.

- [ ] **Step 3: Port the current heuristic semantics into the isolated runtime**

`algorithm.py` reads canonical PCM without decoding or resampling, calls the
lock-enumerated Librosa functions with every argument explicit, classifies the exact
two-hop window in locked branch order, and returns only time, pitch, and velocity.
Do not refactor thresholds or infer defaults from installed Librosa.

`runner.py` is a one-request-at-a-time JSON-lines process with no Crux import. Its
startup handshake exact-matches the parameter-lock hash, source-manifest hash,
interpreter, platform, wheel tags, and complete distribution set. It re-hashes the
WAV, validates RIFF/WAVE PCM mono/16-bit/44.1 kHz and rejects nonempty trailing
bytes. Standard output contains protocol objects only.

- [ ] **Step 4: Implement the host process and adapter**

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
    def descriptor(self) -> BackendDescriptor: ...
    def verify(self) -> BackendVerification: ...
    def transcribe(self, audio: CanonicalAudio) -> NativePrediction: ...
    def close(self) -> None: ...
```

The descriptor contains exactly:

```text
descriptor_schema, backend_id, prediction_schema,
adapter_source_manifest_sha256, parameter_lock_sha256, model_id,
architecture_id, native_output_space_id, native_metadata_schema_id
```

It does not fabricate backend-lock, runtime-lock, model-artifact, upstream-source,
checkpoint, or training-map identities. Verification returns non-null descriptor and
parameter-lock hashes; null OaF-only identities; zero tensor counts;
`not_applicable` coverage/smoke; and `max_input_audio_frames=None`.

Each event is:

```python
NativeEvent(
    time_sec=runner_event.time_sec,
    native_class_id=f"midi_{runner_event.native_midi_note}",
    model_output_bin=None,
    native_midi_note=runner_event.native_midi_note,
    native_metadata={},
    confidence=None,
    velocity_midi=runner_event.velocity_midi,
)
```

- [ ] **Step 5: Activate only the heuristic factory**

Retain the existing lazy registry entry and make it succeed only when the checked-in
lock, source manifest, and content-addressed environment agree. Keep
`legacy-tf2-h5-v0` rejected by `BackendRegistry.create`.

Add a regression that replaces the OaF factory with a failure and the heuristic
factory with a spy, selects OaF, and proves the heuristic spy remains unused.

- [ ] **Step 6: Seal and review the real native environment**

After every adapter source is final:

```bash
rtk uv run python tools/seal_heuristic_backend.py \
  --python 3.12 \
  --project runtime/heuristic_onset \
  --requirements-lock runtime/heuristic_onset/requirements.lock \
  --wheel-cache artifacts/benchmark/heuristic-wheel-cache \
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

Review the exact Python patch/ABI, `linux/amd64`, wheel tags, every distribution
filename/hash, every Librosa call argument, classifier branch order, source-manifest
file, and `covered_roots`. Run both commands again and require no checked-in diff.

- [ ] **Step 7: Run deterministic replay and commit**

```bash
rtk uv run pytest -q \
  tests/benchmark/test_heuristic_process.py \
  tests/benchmark/test_heuristic_backend.py \
  tests/benchmark/test_backend_registry.py \
  tests/benchmark/test_transcription.py
rtk env CRUX_RUN_HEURISTIC_INTEGRATION=1 uv run pytest -q \
  tests/benchmark/test_heuristic_real_environment.py
rtk git add \
  runtime/heuristic_onset \
  config/benchmark/backends/heuristic-onset-v1.parameters.json \
  src/benchmark/heuristic_process.py \
  src/benchmark/backends/heuristic.py \
  src/benchmark/backend_registry.py \
  tests/benchmark/test_heuristic_process.py \
  tests/benchmark/test_heuristic_backend.py \
  tests/benchmark/test_heuristic_real_environment.py \
  tests/benchmark/test_backend_registry.py \
  tests/fixtures/heuristic_onset
rtk git commit -m "feat: add explicit heuristic backend"
```

---

### Task 3: Publish deterministic MIDI derivatives

**Files:**

- Create: `src/benchmark/prediction_midi.py`
- Create: `tools/seal_prediction_midi_writer.py`
- Create: `config/benchmark/backends/prediction-midi-v1.writer-lock.json`
- Create: `runtime/prediction_midi/distribution-manifest.json`
- Create: `runtime/prediction_midi/wheels/mido-1.3.3-py3-none-any.whl`
- Create: `runtime/prediction_midi/wheels/packaging-23.2-py3-none-any.whl`
- Create: `tests/benchmark/test_prediction_midi.py`
- Create: `tests/fixtures/prediction_midi/expected.mid`
- Create: `tests/fixtures/prediction_midi/expected.sidecar.json`
- Modify: `src/benchmark/transcription.py`
- Modify: `tests/benchmark/test_transcription.py`

**Interfaces:**

- Consumes: a strict-read `PredictionArtifact`.
- Produces: existing `MidiDerivative` plus a source-bound strict sidecar.

- [ ] **Step 1: Write failing golden and retention tests**

The prediction fixture contains simultaneous events, a half-tick boundary, minimum
and maximum valid velocities, and a final silence interval.

```python
def test_midi_failure_keeps_authoritative_prediction(
    request_with_midi: TranscribeOneRequest,
    fake_backend: FakeBackend,
) -> None:
    outcome = run_transcribe_one(
        request_with_midi,
        registry=registry_for(fake_backend),
        midi_writer=raising_midi_writer,
    )

    report = read_execution_report(outcome.report_artifact.path)
    assert outcome.exit_code == 1
    assert report.status == "partial"
    assert report.items[0].status == "incomplete"
    assert report.items[0].prediction is not None
    assert report.items[0].midi is None
    assert report.items[0].errors[0].code == "midi_derivation_failed"
```

Add rejection tests for null/out-of-range MIDI pitch or velocity, unknown writer-lock
fields, source-hash mismatch, distribution-manifest drift, wheel drift,
installed-file drift, and writer-source drift.

- [ ] **Step 2: Run tests and confirm the writer is absent**

```bash
rtk uv run pytest -q \
  tests/benchmark/test_prediction_midi.py \
  tests/benchmark/test_transcription.py
```

Expected: collection fails because `src.benchmark.prediction_midi` does not exist.

- [ ] **Step 3: Implement the frozen writer contract**

```python
@dataclass(frozen=True)
class MidiDerivativeConfig:
    ticks_per_quarter: Literal[220] = 220
    tempo_qpm: Decimal = Decimal("120.0")
    channel: Literal[9] = 9
    program: Literal[0] = 0
    note_duration_seconds: Decimal = Decimal("0.05")
```

The writer lock records the exact interpreter, writer-source SHA-256, rendering
settings, and complete Mido/Packaging distribution identities. The sidecar has
exactly:

```text
schema, prediction_sha256, midi_sha256, writer_lock_sha256,
writer_distribution, ticks_per_quarter, tempo_qpm, channel, program,
note_duration_seconds
```

Require Mido 1.3.3 wheel SHA-256
`01033c9b10b049e4436fca2762194ca839b09a4334091dd3c34e7f4ae674fd8a`
and Packaging 23.2 wheel SHA-256
`8c491190033a9af7e1d931d0b5dacc2ef47509b34dd0de67ed209b5203fc88c7`.

- [ ] **Step 4: Seal distributions and render exact bytes**

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

Strict-read the complete prediction before rendering. Convert quantized decimal
seconds to ticks using exact `Decimal` arithmetic and round half to even. Emit one
tempo event at tick zero, deterministic simultaneous note ordering, channel 9, and
no timestamp or source path metadata. Publish MIDI and sidecar through validated,
fsynced temporary files and atomic renames.

- [ ] **Step 5: Wire the Phase B callback**

When `TranscribeOneRequest.midi_output_path` is non-null and no test callback is
injected, call `publish_prediction_midi`. Map every validation, serialization, or
publication failure to `midi_derivation_failed`, item `incomplete`, execution
`partial`, and exit `1`. Delete only unpublished temporary files.

- [ ] **Step 6: Run checks, inspect bytes, and commit**

```bash
rtk uv run pytest -q \
  tests/benchmark/test_prediction_midi.py \
  tests/benchmark/test_transcription.py
rtk xxd -g 1 tests/fixtures/prediction_midi/expected.mid
rtk sed -n '1,120p' tests/fixtures/prediction_midi/expected.sidecar.json
rtk git add \
  src/benchmark/prediction_midi.py \
  src/benchmark/transcription.py \
  tools/seal_prediction_midi_writer.py \
  config/benchmark/backends/prediction-midi-v1.writer-lock.json \
  runtime/prediction_midi \
  tests/benchmark/test_prediction_midi.py \
  tests/benchmark/test_transcription.py \
  tests/fixtures/prediction_midi
rtk git commit -m "feat: derive deterministic prediction MIDI"
```

---

### Task 4: Isolate the legacy TF2/HDF5 compatibility path

**Files:**

- Create: `src/benchmark/backends/legacy_tf2.py`
- Create: `tests/benchmark/test_legacy_tf2_backend.py`

**Interfaces:**

- Consumes: only the existing TF2 feature, inference, and MIDI helper seam.
- Produces explicit MIDI bytes or a stable compatibility error; never a frozen
  descriptor or `NativePrediction`.

- [ ] **Step 1: Write failing no-fallback tests**

```python
def test_missing_legacy_model_does_not_run_heuristic(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    detect = Mock(side_effect=AssertionError("heuristic must not run"))
    monkeypatch.setattr(DrumTranscriber, "_detect_onsets_from_audio", detect)
    backend = create_legacy_tf2_backend(model_path=tmp_path / "missing.weights.h5")

    with pytest.raises(LegacyTf2Error, match="legacy_model_unavailable"):
        backend.transcribe_midi(canonical_wav_path())

    detect.assert_not_called()
```

Add tests for corrupt HDF5, missing TensorFlow, decode failure, successful MIDI,
missing default weights without download, direct inference failure, absence of
`descriptor`, `verify`, and `transcribe`, and continued registry rejection.

- [ ] **Step 2: Run the test and confirm the module is absent**

```bash
rtk uv run pytest -q tests/benchmark/test_legacy_tf2_backend.py
```

Expected: collection fails because `src.benchmark.backends.legacy_tf2` does not
exist.

- [ ] **Step 3: Implement the narrow adapter**

```python
class LegacyTf2Error(RuntimeError):
    def __init__(self, code: str, sanitized_message: str) -> None:
        self.code = code
        self.sanitized_message = sanitized_message
        super().__init__(code)


class LegacyTf2Backend:
    backend_id = "legacy-tf2-h5-v0"

    def transcribe_midi(self, audio_path: Path) -> bytes: ...
```

Resolve only the configured HDF5 path or the existing
`TF2_WEIGHTS_RELATIVE_PATH`, require a regular `.weights.h5` file, construct one
shared transcriber, and invoke only direct TF2 audio/feature/model/MIDI helpers.
Never call `DrumTranscriber.transcribe`, `_detect_onsets_from_audio`,
`_download_model`, a TF1 checkpoint, or a network client. Sanitize all exception
text before it leaves the module.

- [ ] **Step 4: Run focused regressions and commit**

```bash
rtk uv run pytest -q \
  tests/benchmark/test_legacy_tf2_backend.py \
  tests/benchmark/test_backend_registry.py \
  tests/test_transcriber_fallback.py
rtk uv run ruff check \
  src/benchmark/backends/legacy_tf2.py \
  tests/benchmark/test_legacy_tf2_backend.py
rtk git add \
  src/benchmark/backends/legacy_tf2.py \
  tests/benchmark/test_legacy_tf2_backend.py
rtk git commit -m "feat: isolate legacy TF2 transcription"
```

---

### Task 5: Publish typed legacy combined-score outcomes

**Files:**

- Create: `src/benchmark/legacy_score.py`
- Create: `tests/benchmark/test_legacy_score.py`
- Modify: `src/benchmark/runner.py`
- Modify: `tests/benchmark/test_runner.py`

**Interfaces:**

- Consumes: Tasks 2-4, existing `run_score_midi`, and the Phase B legacy report
  normalizer/publisher.
- Produces one immutable `crux.legacy-score-report/v1`.

- [ ] **Step 1: Write failing dispatch, refusal, and wrapper tests**

```python
def test_oaf_refusal_calls_nothing(
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
    assert spies.total_backend_calls == 0
    assert spies.score.calls == 0
```

Parameterize successful `legacy-tf2-h5-v0`/`unvalidated` and
`heuristic-onset-v1`/`parameter_locked` outcomes. Add cases for unknown IDs, missing
audio, no matched charts, transcription failure, heuristic verification failure,
score failure, report-publication failure, exact score-report hash, relative paths,
stable error ordering, and close-on-success/failure.

- [ ] **Step 2: Run tests and confirm orchestration is absent**

```bash
rtk uv run pytest -q \
  tests/benchmark/test_legacy_score.py \
  tests/benchmark/test_runner.py
```

Expected: collection fails because `src.benchmark.legacy_score` does not exist.

- [ ] **Step 3: Make the low-level runner callback-only**

Change:

```python
def run_transcribe_and_score(
    charts_dir: Path,
    audio_dir: Path,
    output_dir: Path,
    tolerance_ms: list[int],
    transcribe: Callable[[Path], bytes],
) -> list[ChartReport]:
    ...
```

Remove `_create_shared_transcriber` and `_make_transcribe_fn`. Keep chart selection,
stale compatibility-output cleanup, MIDI writing, matched-chart copying, warnings,
and `run_score_midi(..., align=True)` unchanged. Every test passes a fake callback;
omitting it is a Python argument error.

- [ ] **Step 4: Implement exact compatibility dispatch**

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

Dispatch in this order:

1. Recognized OaF publishes `canonical_mapping_required` before any factory, audio,
   output, mapping, or scoring access.
2. Legacy TF2 creates one shared `LegacyTf2Backend` and passes
   `transcribe_midi`.
3. Heuristic verifies one `HeuristicBackend`; strict-validates each canonical WAV;
   publishes and strict-reads its native artifact beneath
   `<output-dir>/_legacy_native/heuristic-onset-v1/`; derives MIDI; and returns
   those bytes to the scorer callback.
4. Every other ID is rejected before backend creation.
5. A successful scorer requires and hashes the exact existing
   `<output-dir>/summary.json`.
6. Every constructed backend closes in `finally`.

The wrapper has exactly:

```text
schema, report_type, run_id, started_at, finished_at, workflow_mode, backend_id,
backend_validation_status, status, exit_code, score_report, errors
```

Use `report_type: "legacy_score"` and
`workflow_mode: "legacy_compatibility"`. A wrapper publication error is backend-fatal
and cannot return an outcome without a report artifact.

- [ ] **Step 5: Run scoring regressions and commit**

```bash
rtk uv run pytest -q \
  tests/benchmark/test_legacy_score.py \
  tests/benchmark/test_runner.py \
  tests/benchmark/test_reports.py \
  tests/benchmark/test_scoring.py
rtk git add \
  src/benchmark/legacy_score.py \
  src/benchmark/runner.py \
  tests/benchmark/test_legacy_score.py \
  tests/benchmark/test_runner.py
rtk git commit -m "feat: type legacy transcription scoring"
```

---

### Task 6: Require explicit CLI backend selection

**Files:**

- Modify: `src/cli/benchmark.py`
- Modify: `tests/test_cli_benchmark.py`

**Interfaces:**

- Consumes: `LegacyScoreRequest` and `run_legacy_score`.
- Produces the intentional CLI migration and one canonical summary line.

- [ ] **Step 1: Write failing CLI migration tests**

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
```

Add exact-ID cases for both executable choices, OaF typed refusal, unknown-ID Click
rejection, no aliases, one stdout line, sanitized stderr, report publication,
validation-status labels, and exit `1` on compatibility failure.

- [ ] **Step 2: Run tests and observe current implicit behavior**

```bash
rtk uv run pytest -q \
  tests/test_cli_benchmark.py \
  -k 'transcribe_and_score'
```

Expected: FAIL because the command currently has no `--backend`, constructs
`DrumTranscriber` through the runner, and prints prose.

- [ ] **Step 3: Add the required exact-ID option and typed orchestration**

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

Construct `LegacyScoreRequest`, place reports under
`artifacts/benchmark/backends/<backend-id>/`, and emit only canonical
`status`, `exit_code`, `report_path`, and `report_sha256`. Once Click parsing
succeeds, domain failures must not become `ClickException` prose.

If report publication itself fails, emit no summary, write a sanitized
`report_publication_failed` diagnostic to stderr, and exit `2`.

- [ ] **Step 4: Run CLI checks and commit**

```bash
rtk uv run pytest -q \
  tests/test_cli_benchmark.py \
  tests/benchmark/test_legacy_score.py
rtk uv run crux benchmark transcribe-and-score --help
rtk git add src/cli/benchmark.py tests/test_cli_benchmark.py
rtk git commit -m "feat: require legacy score backend selection"
```

Expected: help lists all three exact IDs and marks `--backend` required.

---

### Task 7: Lock the Phase C schema and verify the compatibility boundary

**Files:**

- Create:
  `tests/benchmark/schema_goldens/crux.heuristic-parameter-lock-v1.json`
- Modify: `tests/benchmark/schema_goldens/manifest.json`
- Modify: `tests/benchmark/test_schema_goldens.py`
- Modify: `docs/drumery-dtx-midi-benchmarking-reference.md`

**Interfaces:**

- Consumes: completed Tasks 1-6 and the Phase B schema-golden harness.
- Produces executable Phase C schema drift detection, migration guidance, and final
  compatibility acceptance evidence.

- [ ] **Step 1: Append the exact Phase C schema golden**

Register the normative heuristic parameter-lock schema with every production
validator that consumes it. Reuse the Phase B heuristic descriptor and legacy-score
goldens. Mutate one key at a time, add an unknown field, duplicate a raw JSON key,
and change every enum/identity field; require every registered validator to reject
the mutation.

The MIDI writer lock and derivative sidecar are strict implementation artifacts, not
additional schemas in the design's normative appendix. Their canonical JSON fixtures
and mutation coverage stay in `test_prediction_midi.py`; do not register invented
schema IDs in the common manifest.

Run:

```bash
rtk uv run pytest -q tests/benchmark/test_schema_goldens.py
```

Expected: PASS only after all Phase C validators accept the same canonical bytes.

- [ ] **Step 2: Document exact compatibility commands and eligibility**

Document environment preparation, explicit heuristic verification/transcription,
explicit legacy scoring, and OaF refusal. State:

- heuristic output is parameter-locked native evidence, not OaF output;
- legacy TF2/HDF5 is unvalidated and prohibited from official comparison;
- OaF cannot enter the combined scorer before HPA-324/HPA-325;
- old invocations without `--backend` now fail with Click exit `2`; and
- requested MIDI failure leaves native JSONL intact and returns exit `1`.

- [ ] **Step 3: Run the focused Phase C suite**

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
  tests/benchmark/test_schema_goldens.py \
  tests/test_cli_benchmark.py \
  tests/test_transcriber_fallback.py
rtk env CRUX_RUN_HEURISTIC_INTEGRATION=1 uv run pytest -q \
  tests/benchmark/test_heuristic_real_environment.py
```

Expected: PASS, including byte-identical native artifact replay in the isolated
native environment.

- [ ] **Step 4: Prove fallback and identity separation**

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
- heuristic artifacts contain only the heuristic descriptor;
- OaF refusal occurs before any backend factory or scorer call; and
- no compatibility failure selects another backend.

- [ ] **Step 5: Run repository gates**

```bash
rtk uv run pytest
rtk uv run ruff check src tests tools
rtk uv run black --check src tests tools
rtk uv run pylint src/app src/cli
```

Expected: PASS. Record any unrelated pre-existing failure without weakening an
HPA-320 assertion.

- [ ] **Step 6: Commit documentation and schema acceptance**

```bash
rtk git add \
  tests/benchmark/schema_goldens \
  tests/benchmark/test_schema_goldens.py \
  docs/drumery-dtx-midi-benchmarking-reference.md
rtk git commit -m "docs: lock transcription compatibility contracts"
```

- [ ] **Step 7: Record final evidence**

```bash
rtk git status --short
rtk git log --oneline --decorate -14
rtk git diff origin/main...HEAD --check
```

Expected: clean worktree, Phase C commits in task order, and no whitespace errors.
