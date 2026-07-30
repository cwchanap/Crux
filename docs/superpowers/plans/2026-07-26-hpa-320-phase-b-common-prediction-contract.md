# HPA-320 Phase B Common Prediction Contract Implementation Plan

> **Historical snapshot:** This plan captures the design intent at the time of writing. Checkbox states and version references reflect the original plan, not the current repository state.
> **For agentic workers:** REQUIRED SUB-SKILL: Use
> superpowers:subagent-driven-development (recommended) or
> superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox
> (`- [ ]`) syntax for tracking.

**Goal:** Build the backend-agnostic audio provenance, transcription protocol,
prediction JSONL, operational reports, registry, and `transcribe-one` workflow that
the frozen OaF and heuristic adapters can both implement.

**Architecture:** Keep TensorFlow and Librosa outside the common layer. The common
layer owns immutable records, strict canonical JSON, canonical-WAV validation,
content-addressed item identity, atomic publication, and orchestration through a
small `TranscriptionBackend` protocol. Tests use an in-process fake backend so this
phase lands and passes without TensorFlow 1, Docker, model weights, or a native
`amd64` host.

**Tech Stack:** Python 3.12, standard-library `dataclasses`, `decimal`, `hashlib`,
`json`, `struct`, `wave`, `pathlib`, Click 8, Pytest, existing Crux durability
helpers.

## Global Constraints

- The public artifact schema is exactly `crux.drum-prediction-events/v1`.
- The official frozen backend ID is
  `magenta-egmd-tf1-94529798-8hit-v1`; it is the default only for
  `verify-backend`, `transcribe-one`, and later HPA-326 inference.
- Canonical input is RIFF/WAVE PCM, mono, signed 16-bit little-endian, exactly
  44,100 Hz, with a positive audio sample-frame count.
- Direct mode requires `--source-audio-id` and `--input-view-id`; derived mode
  requires `--input-view-manifest`; the modes are mutually exclusive.
- Caller-supplied source hashes are forbidden. Direct mode hashes `--audio` as both
  source and canonical input; derived mode re-hashes both manifest artifacts.
- Persisted float values use
  `Decimal.from_float(value).quantize(Decimal("0.000001"), ROUND_HALF_EVEN)`.
- Quantize before sorting. A complete six-field sort-key tie fails with
  `duplicate_native_event`; no emission-order tiebreak or deduplication is allowed.
- Native artifacts always emit `mapping_status: "not_applied"`,
  `prediction_map_version: null`, and `canonical_class: null`.
- MIDI is never parsed to reconstruct native events. Phase C adds the optional
  derivative writer through the interface defined here.
- Verification, execution, and legacy-score reports are strict, canonical,
  atomically published schemas. Mutable `latest-*.json` files are convenience
  copies, never identity inputs.
- Standard output is one canonical JSON summary after Click parsing. Human progress
  and sanitized diagnostics go to standard error.
- The FastAPI `DrumTranscriber` path, canonical taxonomy, scoring semantics, and
  corpus orchestration remain unchanged.
- Python lines stay within the repository's 100-character formatting limit.

## Execution Order

This is the first executable HPA-320 phase. Complete Tasks 1-6 before Phase A starts
consuming the common interfaces. Task 7 can land with lazy backend factories: until
Phase A and Phase C provide their modules, selecting those factories returns the
typed `backend_unavailable` outcome instead of importing TensorFlow or falling back.

---

## File Map

### New source files

- `src/benchmark/backend_identity.py` — strict duplicate-key JSON loading,
  canonical JSON rendering, SHA-256 validation, six-decimal quantization, and
  descriptor identity.
- `src/benchmark/backends/__init__.py` — public backend package exports.
- `src/benchmark/backends/base.py` — immutable common records,
  `TranscriptionBackend`, native metadata policies, and stable error vocabulary.
- `src/benchmark/input_view.py` — strict canonical-WAV parsing plus direct and
  manifest-derived provenance.
- `src/benchmark/backend_publication.py` — no-follow immutable publication, atomic
  replacement, directory durability, and published-artifact records.
- `src/benchmark/backend_attestation.py` — source-manifest dirty scope,
  canonical changed-file manifests, execution conditions, and per-run Git
  attestation.
- `src/benchmark/prediction_artifact.py` — strict JSONL rendering, reading,
  validation, sorting, terminal hash, and atomic prediction publication.
- `src/benchmark/scorer_input.py` — backend-independent scorer-facing gate that
  requires a separately mapped canonical artifact.
- `src/benchmark/backend_reports.py` — verification, execution, and legacy-score
  report records, content-addressed item IDs, report publication, and latest copies.
- `src/benchmark/backend_registry.py` — explicit backend IDs and lazy factories.
- `src/benchmark/transcription.py` — `transcribe-one` orchestration and outcome
  mapping.

### New tests

- `tests/benchmark/test_backend_identity.py`
- `tests/benchmark/test_backend_types.py`
- `tests/benchmark/test_input_view.py`
- `tests/benchmark/test_backend_publication.py`
- `tests/benchmark/test_backend_attestation.py`
- `tests/benchmark/test_prediction_artifact.py`
- `tests/benchmark/test_scorer_input.py`
- `tests/benchmark/test_backend_reports.py`
- `tests/benchmark/test_backend_registry.py`
- `tests/benchmark/test_transcription.py`

### Modified files

- `src/cli/benchmark.py:1-300` — register `transcribe-one`, direct/derived options,
  output summary, and explicit exit mapping.
- `tests/test_cli_benchmark.py:325-355` — preserve current behavior first, then add
  `transcribe-one` command tests.
- `tests/benchmark/test_runner.py:49-147` — characterize the current implicit legacy
  workflow before Phase C rewires it.
- `docs/drumery-dtx-midi-benchmarking-reference.md:363-680` — document native
  prediction artifacts and `transcribe-one`.

### Cross-task interfaces

Later tasks and the Phase A/C plans must use these exact public names:

- Identity: `JsonValue`, `StrictJsonError`, `canonical_json_bytes`,
  `strict_json_loads`, `sha256_hex`, `require_sha256`, `quantize_six`,
  `build_descriptor`.
- Backend records: `BackendError`, `PublishedArtifact`, `BackendDescriptor`,
  `MidiDerivative`, `CanonicalAudio`, `NativeEvent`, `NativePrediction`,
  `TensorCoverageCheck`, `SmokeCheck`, `BackendVerification`,
  `TranscriptionBackend`.
- Input: `InputViewManifest`, `parse_canonical_wav`, `load_direct_audio`,
  `load_derived_audio`.
- Prediction artifact: `PredictionArtifact`, `render_prediction_artifact`,
  `read_prediction_artifact`, `publish_prediction_artifact`.
- Scorer input: `CanonicalMappingRequired`, `read_scorer_events`.
- Reports: `ExecutionItem`, `VerificationReport`, `ExecutionReport`,
  `LegacyScoreReport`, `OperationalReportPublicationError`, `derive_item_id`,
  `publish_operational_report`.
- Attestation: `ChangedFile`, `ExecutionConditions`, `ExecutionAttestation`,
  `build_changed_file_manifest`, `publish_execution_attestation`.
- Registry: `OFFICIAL_BACKEND_ID`, `HEURISTIC_BACKEND_ID`,
  `LEGACY_TF2_BACKEND_ID`, `BackendRegistry`, `default_backend_registry`.
- Orchestration: `TranscribeOneRequest`, `TranscribeOneOutcome`,
  `run_transcribe_one`.

---

### Task 1: Characterize the current combined transcription workflow

**Files:**

- Modify: `tests/test_cli_benchmark.py:325-355`
- Modify: `tests/benchmark/test_runner.py:49-147`
- Test: `tests/test_transcriber_fallback.py`

**Interfaces:**

- Consumes: current `transcribe-and-score`, `run_transcribe_and_score`, and
  `DrumTranscriber` behavior.
- Produces: regression evidence that Phase C must intentionally update rather than
  silently reinterpret.

- [ ] **Step 1: Add explicit characterization tests**

Add these tests without changing production code:

```python
def test_transcribe_and_score_help_has_no_backend_option() -> None:
    result = CliRunner().invoke(main, ["benchmark", "transcribe-and-score", "--help"])

    assert result.exit_code == 0
    assert "--backend" not in result.output


def test_current_default_transcribe_constructs_drum_transcriber_once(
    tmp_path: Path,
    monkeypatch,
) -> None:
    charts = tmp_path / "charts"
    audio = tmp_path / "audio"
    output = tmp_path / "output"
    charts.mkdir()
    audio.mkdir()
    for chart_id in ("a", "b"):
        (charts / f"{chart_id}.dtx").write_text(
            "#BPM: 120\n#00013: 0100\n",
            encoding="utf-8",
        )
        (audio / f"{chart_id}.wav").write_bytes(b"characterization")

    calls: list[Path] = []

    def transcribe(path: Path) -> bytes:
        calls.append(path)
        return write_prediction_bytes()

    run_transcribe_and_score(charts, audio, output, [50], transcribe=transcribe)

    assert calls == [audio / "a.wav", audio / "b.wav"]
```

Retain the existing fallback tests proving `_build_model()` can lead to
`_detect_onsets_from_audio`; do not change the FastAPI behavior.

- [ ] **Step 2: Run the characterization tests**

Run:

```bash
rtk uv run pytest -q \
  tests/test_cli_benchmark.py::test_transcribe_and_score_help_has_no_backend_option \
  tests/benchmark/test_runner.py::test_current_default_transcribe_constructs_drum_transcriber_once \
  tests/test_transcriber_fallback.py
```

Expected: PASS. These are characterization tests, so a red failure is not required.

- [ ] **Step 3: Commit the characterization boundary**

```bash
rtk git add tests/test_cli_benchmark.py tests/benchmark/test_runner.py
rtk git commit -m "test: characterize legacy transcription workflow"
```

---

### Task 2: Add strict canonical identity and backend domain records

**Files:**

- Create: `src/benchmark/backend_identity.py`
- Create: `src/benchmark/backends/__init__.py`
- Create: `src/benchmark/backends/base.py`
- Create: `tests/benchmark/test_backend_identity.py`
- Create: `tests/benchmark/test_backend_types.py`

**Interfaces:**

- Consumes: schema names and descriptor rules from the design.
- Produces: all identity and immutable record types listed in the cross-task
  interface.

- [ ] **Step 1: Write failing canonical JSON and descriptor tests**

```python
from decimal import Decimal

import pytest

from src.benchmark.backend_identity import (
    StrictJsonError,
    canonical_json_bytes,
    quantize_six,
    strict_json_loads,
)


def test_canonical_json_sorts_keys_and_preserves_decimal_number() -> None:
    payload = {"z": Decimal("1.250000"), "a": "音楽"}

    assert canonical_json_bytes(payload, trailing_newline=True) == (
        b'{"a":"\xe9\x9f\xb3\xe6\xa5\xbd","z":1.25}\n'
    )


def test_strict_json_rejects_duplicate_keys() -> None:
    with pytest.raises(StrictJsonError, match="duplicate key"):
        strict_json_loads(b'{"schema":"a","schema":"b"}')


def test_quantize_six_uses_binary64_then_half_even() -> None:
    assert quantize_six(0.1234565) == Decimal.from_float(0.1234565).quantize(
        Decimal("0.000001")
    )
```

Add tests rejecting floats passed directly to `canonical_json_bytes`, NaN/infinity,
surrogate strings, non-string object keys, unsupported containers, uppercase hashes,
and descriptor fields that are not strings. Assert exact descriptor bytes/hash and
that timestamps, request IDs, paths, dirty state, and a trailing newline cannot enter
or alter the descriptor payload.

- [ ] **Step 2: Run the new tests and verify the missing-module failure**

```bash
rtk uv run pytest -q \
  tests/benchmark/test_backend_identity.py \
  tests/benchmark/test_backend_types.py
```

Expected: collection FAIL because the two modules do not exist.

- [ ] **Step 3: Implement canonical JSON and descriptor identity**

Define these exact aliases and functions:

```python
from __future__ import annotations

import hashlib
import json
import math
from decimal import Decimal, ROUND_HALF_EVEN
from typing import TypeAlias

JsonScalar: TypeAlias = None | bool | int | str | Decimal
JsonValue: TypeAlias = JsonScalar | list["JsonValue"] | dict[str, "JsonValue"]
SIX_PLACES = Decimal("0.000001")


class StrictJsonError(ValueError):
    pass


def quantize_six(value: float) -> Decimal:
    if not math.isfinite(value):
        raise StrictJsonError("nonfinite binary float")
    quantized = Decimal.from_float(value).quantize(SIX_PLACES, rounding=ROUND_HALF_EVEN)
    return Decimal(0) if quantized.is_zero() else quantized


def sha256_hex(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def require_sha256(value: str, field: str) -> str:
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise StrictJsonError(f"{field} must be lowercase SHA-256")
    return value
```

Implement `canonical_json_bytes` with a recursive renderer that accepts only the
`JsonValue` union, uses `json.dumps` only for strings, sorts object keys
lexicographically, renders `Decimal` with fixed-point notation and stripped trailing
zeros, and adds a newline only when requested. Implement `strict_json_loads` with
`object_pairs_hook` duplicate detection, `parse_float=Decimal`, rejection of
`parse_constant`, strict UTF-8 decoding, and a canonical re-render check when the
caller requests canonical bytes.

`build_descriptor(payload, allowed_keys, schema)` must require the exact key set,
require every value to be a string, require `descriptor_schema == schema`, render
without a newline, and return:

```python
@dataclass(frozen=True)
class BackendDescriptor:
    payload: Mapping[str, str]
    sha256: str
```

- [ ] **Step 4: Define common backend records and protocol**

In `src/benchmark/backends/base.py`, add:

```python
@dataclass(frozen=True)
class BackendError:
    code: str
    message: str


@dataclass(frozen=True)
class PublishedArtifact:
    role: str
    path: Path
    sha256: str


@dataclass(frozen=True)
class MidiDerivative:
    midi: PublishedArtifact
    sidecar: PublishedArtifact


@dataclass(frozen=True)
class CanonicalAudio:
    path: Path
    source_audio_id: str
    source_audio_sha256: str
    input_view_id: str
    input_audio_sha256: str
    byte_length: int
    sample_rate: int
    channel_count: int
    sample_width_bytes: int
    audio_frame_count: int


@dataclass(frozen=True)
class NativeEvent:
    time_sec: float
    native_class_id: str
    model_output_bin: int | None
    native_midi_note: int | None
    native_metadata: Mapping[str, str | None]
    confidence: float | None
    velocity_midi: int | None


@dataclass(frozen=True)
class NativePrediction:
    audio: CanonicalAudio
    descriptor: BackendDescriptor
    events: tuple[NativeEvent, ...]
    backend_lock_sha256: str | None
    runtime_lock_sha256: str | None
    parameter_lock_sha256: str | None
    model_artifact_set_sha256: str | None
    upstream_source_commit: str | None
    training_data_map_id: str | None


@dataclass(frozen=True)
class TensorCoverageCheck:
    status: Literal["passed", "failed", "not_run", "not_applicable"]
    required_count: int
    restored_count: int
    non_inference_count: int
    required_inventory_sha256: str | None
    non_inference_inventory_sha256: str | None
    report: PublishedArtifact | None


@dataclass(frozen=True)
class SmokeCheck:
    status: Literal["passed", "failed", "not_run", "not_applicable"]
    audio_sha256: str | None
    oracle_sha256: str | None
    prediction: PublishedArtifact | None


@dataclass(frozen=True)
class BackendVerification:
    status: Literal["verified", "failed", "environment_unsupported"]
    descriptor: BackendDescriptor | None
    max_input_audio_frames: int | None
    backend_lock_sha256: str | None
    runtime_lock_sha256: str | None
    parameter_lock_sha256: str | None
    seal_evidence_sha256: str | None
    execution_attestation: PublishedArtifact | None
    tensor_coverage: TensorCoverageCheck
    smoke: SmokeCheck
    errors: tuple[BackendError, ...]


class TranscriptionBackend(Protocol):
    def descriptor(self) -> BackendDescriptor:
        ...

    def verify(self) -> BackendVerification:
        ...

    def transcribe(self, audio: CanonicalAudio) -> NativePrediction:
        ...

    def close(self) -> None:
        ...
```

Export these names from `src/benchmark/backends/__init__.py`. Keep request IDs,
timestamps, paths, host facts, and verification status outside descriptor payloads.

- [ ] **Step 5: Run focused tests and style checks**

```bash
rtk uv run pytest -q \
  tests/benchmark/test_backend_identity.py \
  tests/benchmark/test_backend_types.py
rtk uv run ruff check \
  src/benchmark/backend_identity.py \
  src/benchmark/backends \
  tests/benchmark/test_backend_identity.py \
  tests/benchmark/test_backend_types.py
rtk uv run black --check \
  src/benchmark/backend_identity.py \
  src/benchmark/backends \
  tests/benchmark/test_backend_identity.py \
  tests/benchmark/test_backend_types.py
```

Expected: all commands exit `0`.

- [ ] **Step 6: Commit the common identity layer**

```bash
rtk git add \
  src/benchmark/backend_identity.py \
  src/benchmark/backends/__init__.py \
  src/benchmark/backends/base.py \
  tests/benchmark/test_backend_identity.py \
  tests/benchmark/test_backend_types.py
rtk git commit -m "feat: add transcription backend identity primitives"
```

---

### Task 3: Validate canonical WAV and provenance modes

**Files:**

- Create: `src/benchmark/input_view.py`
- Create: `tests/benchmark/test_input_view.py`

**Interfaces:**

- Consumes: `CanonicalAudio`, `strict_json_loads`, `require_sha256`,
  `sha256_hex`.
- Produces: `InputViewManifest`, `parse_canonical_wav`,
  `load_direct_audio`, `load_derived_audio`.

- [ ] **Step 1: Write failing WAV and provenance tests**

```python
def test_direct_audio_hashes_one_file_as_source_and_input(tmp_path: Path) -> None:
    audio_path = tmp_path / "input.wav"
    audio_path.write_bytes(canonical_wav_bytes(sample_frames=441))

    audio = load_direct_audio(
        audio_path,
        source_audio_id="song-42-source-v1",
        input_view_id="full-mix-canonical-wav-v1",
        max_input_audio_frames=441,
    )

    assert audio.source_audio_sha256 == audio.input_audio_sha256
    assert audio.audio_frame_count == 441
    assert audio.sample_rate == 44100


def test_derived_audio_rehashes_source_and_input(tmp_path: Path) -> None:
    source = tmp_path / "source.flac"
    canonical = tmp_path / "canonical.wav"
    source.write_bytes(b"source bytes")
    canonical.write_bytes(canonical_wav_bytes(sample_frames=882))
    manifest = write_input_view_manifest(tmp_path, source, canonical)

    audio = load_derived_audio(
        canonical,
        manifest,
        max_input_audio_frames=882,
    )

    assert audio.source_audio_sha256 == sha256(source.read_bytes()).hexdigest()
    assert audio.input_audio_sha256 == sha256(canonical.read_bytes()).hexdigest()
```

Add cases for stereo, 24-bit, non-44.1-kHz, float WAV, zero frames, odd data length,
extra/trailing chunks, RIFF-size mismatch, over-bound audio, absolute manifest paths,
`..`, symlink/path escape, duplicate/unknown manifest keys, source hash mismatch,
input hash mismatch, and `--audio` not matching the canonical manifest path. Cover
`max_input_audio_frames - 1`, the exact boundary, boundary plus one, and a null
adapter bound explicitly.

- [ ] **Step 2: Run the tests and verify the missing-module failure**

```bash
rtk uv run pytest -q tests/benchmark/test_input_view.py
```

Expected: collection FAIL because `src.benchmark.input_view` does not exist.

- [ ] **Step 3: Implement exact input records and strict WAV parsing**

Use this manifest shape:

```python
@dataclass(frozen=True)
class InputViewManifest:
    schema: Literal["crux.input-view-manifest/v1"]
    source_audio_id: str
    source_audio_sha256: str
    source_path: str
    input_view_id: str
    input_audio_sha256: str
    input_audio_path: str
```

`parse_canonical_wav(content, max_input_audio_frames)` must parse RIFF bytes directly
with `struct`, allow exactly one 16-byte PCM `fmt ` chunk followed by exactly one
`data` chunk, require format `1`, channels `1`, sample rate `44100`, byte rate
`88200`, block alignment `2`, bits per sample `16`, positive even data length, exact
RIFF size, and no trailing bytes. `max_input_audio_frames` is `int | None`; when it
is non-null, require
`audio_frame_count = data_length // 2 <= max_input_audio_frames`. The official OaF
backend always supplies its positive locked bound; an adapter without a locked bound
passes null and still receives every other strict WAV check.

`load_direct_audio` reads once, validates, hashes exact bytes, rejects empty IDs, and
sets both hashes from the same content. `load_derived_audio` strict-parses the seven
manifest fields, resolves POSIX relative paths below the manifest directory without
following symlinks outside it, independently re-hashes both files, requires
`audio_path` to bind to `input_audio_path`, then validates the canonical WAV.

- [ ] **Step 4: Run input tests and format checks**

```bash
rtk uv run pytest -q tests/benchmark/test_input_view.py
rtk uv run ruff check src/benchmark/input_view.py tests/benchmark/test_input_view.py
rtk uv run black --check src/benchmark/input_view.py tests/benchmark/test_input_view.py
```

Expected: all commands exit `0`.

- [ ] **Step 5: Commit canonical audio validation**

```bash
rtk git add src/benchmark/input_view.py tests/benchmark/test_input_view.py
rtk git commit -m "feat: validate transcription input provenance"
```

---

### Task 4: Render and read the canonical prediction JSONL

**Files:**

- Create: `src/benchmark/prediction_artifact.py`
- Create: `src/benchmark/scorer_input.py`
- Create: `tests/benchmark/test_prediction_artifact.py`
- Create: `tests/benchmark/test_scorer_input.py`

**Interfaces:**

- Consumes: `NativePrediction`, canonical JSON, metadata schema IDs from descriptor
  payloads.
- Produces: `PredictionArtifact`, `render_prediction_artifact`,
  `read_prediction_artifact`, `publish_prediction_artifact`.

- [ ] **Step 1: Write failing canonical byte and validation tests**

```python
def test_prediction_quantizes_then_sorts_and_assigns_indexes() -> None:
    prediction = make_prediction(
        events=(
            make_event(time_sec=0.5000004, native_class_id="midi_38", model_output_bin=17),
            make_event(time_sec=0.4999996, native_class_id="midi_36", model_output_bin=15),
        )
    )

    content = render_prediction_artifact(prediction)
    records = [strict_json_loads(line) for line in content.splitlines()]

    assert records[1]["event_index"] == 0
    assert records[1]["native_class_id"] == "midi_36"
    assert records[1]["time_sec"] == Decimal("0.5")
    assert records[-1]["event_count"] == 2


def test_prediction_rejects_full_sort_key_tie() -> None:
    event = make_event(time_sec=0.5, native_class_id="midi_36", model_output_bin=15)

    with pytest.raises(PredictionArtifactError, match="duplicate_native_event"):
        render_prediction_artifact(make_prediction(events=(event, event)))
```

Add tests for header key/nullability combinations, OaF bin/pitch/class agreement,
confidence/velocity range, exact OaF metadata key and eight allowed group IDs,
empty heuristic metadata, unknown metadata keys, nonfinite/negative time, zero-event
artifacts, mapping fields, terminal prefix hash, complete-file SHA-256, final
newlines, unknown/duplicate keys, reordered fields, exponent notation, negative
zero, round-trip byte identity, and a generic reader that does not import an OaF
module.

In `test_scorer_input.py`, assert a valid native artifact raises
`CanonicalMappingRequired("canonical_mapping_required")`, no MIDI/class guess occurs,
and importing/using the gate does not import an OaF adapter or mapping module.

- [ ] **Step 2: Run the tests and verify the missing-module failure**

```bash
rtk uv run pytest -q \
  tests/benchmark/test_prediction_artifact.py \
  tests/benchmark/test_scorer_input.py
```

Expected: collection FAIL because the artifact and scorer-input modules do not exist.

- [ ] **Step 3: Implement strict policies and deterministic rendering**

Define:

```python
OAF_GROUP_IDS = frozenset(
    {"kick", "snare", "toms", "hihat", "ride", "ride_bell", "crash", "sticks"}
)

NATIVE_METADATA_SCHEMAS = {
    "magenta-oaf-native-metadata-v1": {
        "upstream_8hit_group_id": OAF_GROUP_IDS | {None},
    },
    "crux-empty-native-metadata-v1": {},
}


@dataclass(frozen=True)
class PredictionArtifact:
    prediction: NativePrediction
    event_count: int
    prefix_sha256: str
    artifact_sha256: str
    content: bytes
```

Build the header with exactly the fields in the design, validate backend-specific
nullability from descriptor IDs, quantize `time_sec` and confidence before sorting,
map missing numeric values to `-1` only inside the sort key, reject a full key tie,
then assign `event_index`. Render one canonical line per record. The terminal hashes
the exact header/event bytes including their newlines; the complete artifact hash
includes the terminal line.

The reader must strict-parse each physical line, require exactly one header first and
one terminal last, reject blank or trailing records, re-render every record
canonically, validate indexes/count/hash, and return the same `PredictionArtifact`.

`read_scorer_events` calls only the common strict reader. It raises
`CanonicalMappingRequired("canonical_mapping_required")` whenever
`artifact_role == "native"`, `mapping_status == "not_applied"`, or any canonical
class is null. It never imports backend-specific code, derives a class from MIDI, or
changes HPA-325 scoring semantics.

- [ ] **Step 4: Run artifact tests and deterministic replay**

```bash
rtk uv run pytest -q \
  tests/benchmark/test_prediction_artifact.py \
  tests/benchmark/test_scorer_input.py
rtk uv run pytest -q \
  tests/benchmark/test_prediction_artifact.py \
  -k prediction_quantizes_then_sorts_and_assigns_indexes
rtk uv run pytest -q \
  tests/benchmark/test_prediction_artifact.py \
  -k prediction_quantizes_then_sorts_and_assigns_indexes
```

Both executions must pass and produce the same asserted bytes.

- [ ] **Step 5: Commit the common prediction artifact**

```bash
rtk git add \
  src/benchmark/prediction_artifact.py \
  src/benchmark/scorer_input.py \
  tests/benchmark/test_prediction_artifact.py \
  tests/benchmark/test_scorer_input.py
rtk git commit -m "feat: add canonical drum prediction artifacts"
```

---

### Task 5: Add durable publication and operational report schemas

**Files:**

- Create: `src/benchmark/backend_publication.py`
- Create: `src/benchmark/backend_attestation.py`
- Create: `src/benchmark/backend_reports.py`
- Create: `tests/benchmark/test_backend_publication.py`
- Create: `tests/benchmark/test_backend_attestation.py`
- Create: `tests/benchmark/test_backend_reports.py`
- Modify: `src/benchmark/prediction_artifact.py`

**Interfaces:**

- Consumes: `ensure_durable_directory`, `fsync_directory`, canonical JSON,
  `PublishedArtifact`.
- Produces: no-follow publication helpers, changed-file/execution attestations,
  report records, `derive_item_id`, and `publish_operational_report`.

- [ ] **Step 1: Write failing durability and report tests**

```python
def test_derive_item_id_excludes_backend_and_run_identity() -> None:
    first = derive_item_id(
        source_audio_id="source",
        source_audio_sha256="a" * 64,
        input_view_id="view",
        input_audio_sha256="b" * 64,
    )
    second = derive_item_id(
        source_audio_id="source",
        source_audio_sha256="a" * 64,
        input_view_id="view",
        input_audio_sha256="b" * 64,
    )

    assert first == second
    assert first.startswith("sha256:")


def test_report_publication_updates_latest_after_immutable_report(tmp_path: Path) -> None:
    report = make_execution_report(status="complete", exit_code=0)

    published = publish_operational_report(
        tmp_path,
        backend_id="fake-backend-v1",
        report=report,
        now=FIXED_UTC,
        run_id=UUID("12345678-1234-4678-9234-567812345678"),
    )

    latest = tmp_path / "fake-backend-v1" / "latest-execution.json"
    assert published.path.exists()
    assert latest.read_bytes() == published.path.read_bytes()
```

Add tests for verification/execution/legacy exact key sets and status enums, sorted
errors/artifacts, nullable pre-establishment fields, OaF verified-field
requirements, heuristic non-applicable fields, request-order items, MIDI-incomplete
items, timestamp/run-ID formatting, collision-proof filenames, existing immutable
content mismatch, symlink destinations, write/fsync/replace failures, latest pointer
rollback, and preservation of a previously valid prediction on report failure.

Add attestation tests for exact Git commit, unrelated dirty files, modified/deleted
enumerated files, untracked files under `covered_roots`, rename as
deleted-plus-untracked, NFC/UTF-8/path validation, symlink rejection, canonical
changed-file ordering/hash, strict-mode rejection, and exact execution conditions.

- [ ] **Step 2: Run tests and verify missing modules**

```bash
rtk uv run pytest -q \
  tests/benchmark/test_backend_publication.py \
  tests/benchmark/test_backend_attestation.py \
  tests/benchmark/test_backend_reports.py
```

Expected: collection FAIL for missing modules.

- [ ] **Step 3: Implement no-follow durable publication**

Expose:

```text
publish_immutable_bytes(
    path: Path, content: bytes, expected_sha256: str, *, role: str
) -> PublishedArtifact
atomic_replace_bytes(path: Path, content: bytes) -> None
```

Use `ensure_durable_directory`, `O_NOFOLLOW`, regular-file descriptor checks,
temporary files created with `"xb"`, file `flush`/`fsync`, hard-link publication for
immutable destinations, path/descriptor inode binding, parent-directory `fsync`,
and cleanup failure propagation. Do not call or alter private functions in
`corpus_manifest.py`; preserve HPA-321 behavior.

Update `publish_prediction_artifact` to render, publish immutably to the caller's
required destination, read it back through the strict reader, and return a
`PublishedArtifact`.

- [ ] **Step 4: Implement changed-file scope and execution attestations**

Define:

```python
@dataclass(frozen=True)
class ChangedFile:
    path: str
    status: Literal["modified", "deleted", "untracked"]
    sha256: str | None


@dataclass(frozen=True)
class ExecutionConditions:
    cpu_limit: str | None
    memory_bytes: int | None
    pid_limit: int | None
    tmp_bytes: int | None
    shm_bytes: int | None
    startup_deadline_seconds: int
    request_deadline_seconds: int


@dataclass(frozen=True)
class ExecutionAttestation:
    schema: Literal["crux.backend-execution-attestation/v1"]
    backend_id: str
    descriptor_sha256: str
    git_commit: str
    checkout_dirty: bool
    strict_mode: bool
    changed_files_manifest: PublishedArtifact | None
    conditions: ExecutionConditions
```

Strict-load every referenced source manifest and take the union of its enumerated
paths and `covered_roots`. Query Git with list arguments and parse NUL-delimited
status output. Include modified/deleted enumerated files and untracked files beneath
covered roots; ignore unrelated dirty paths only in the changed-file manifest.
Require regular NFC UTF-8 paths, represent renames as deletion plus untracked, hash
current bytes for modified/untracked entries, sort by UTF-8 path bytes, render the
exact three-key array with one newline, and publish it immutably. Strict mode rejects
any nonempty inference-relevant change set. `checkout_dirty` reflects the whole Git
checkout, including unrelated paths.

The canonical attestation contains exactly:

```text
schema, backend_id, descriptor_sha256, git_commit, checkout_dirty, strict_mode,
changed_files_manifest, cpu_limit, memory_bytes, pid_limit, tmp_bytes, shm_bytes,
startup_deadline_seconds, request_deadline_seconds
```

Container resource fields are required positive OaF values and may be null for a
non-container heuristic execution. Both deadline fields are always positive.
Publish a nonempty changed-file array at
`<backend-root>/attestations/changed-files/sha256/<manifest-sha256>.json`; use null
for a clean inference-relevant scope. Publish the canonical attestation at
`<backend-root>/attestations/YYYYMMDDTHHMMSSffffffZ-<lowercase-uuidv4>.json`.
Both paths use the no-follow immutable publication helper and are serialized into
reports as repository-relative POSIX paths.

- [ ] **Step 5: Implement report records and canonical publication**

Define the exact top-level records:

```python
@dataclass(frozen=True)
class ExecutionItem:
    item_id: str
    source_audio_id: str
    source_audio_sha256: str
    input_view_id: str
    input_audio_sha256: str
    status: Literal["complete", "incomplete", "failed"]
    prediction: PublishedArtifact | None
    midi: PublishedArtifact | None
    errors: tuple[BackendError, ...]


@dataclass(frozen=True)
class VerificationReport:
    payload: Mapping[str, JsonValue]


@dataclass(frozen=True)
class ExecutionReport:
    payload: Mapping[str, JsonValue]


@dataclass(frozen=True)
class LegacyScoreReport:
    payload: Mapping[str, JsonValue]
```

`derive_item_id` renders the exact
`crux.backend-execution-item-id/v1` five-field payload without a newline and prefixes
the digest with `sha256:`. `publish_operational_report` selects
`latest-verification.json`, `latest-execution.json`, or
`latest-legacy-score.json`, publishes
`reports/YYYYMMDDTHHMMSSffffffZ-<lowercase-uuidv4>.json`, then atomically copies its
exact canonical bytes to the latest path. Any failure before the immutable report is
durable raises `OperationalReportPublicationError`; it never returns a partial
`PublishedArtifact`.

- [ ] **Step 6: Run report, publication, attestation, and HPA-321 regression tests**

```bash
rtk uv run pytest -q \
  tests/benchmark/test_backend_publication.py \
  tests/benchmark/test_backend_attestation.py \
  tests/benchmark/test_backend_reports.py \
  tests/benchmark/test_prediction_artifact.py \
  tests/benchmark/test_corpus_manifest.py \
  tests/benchmark/test_durability.py
```

Expected: PASS with HPA-321 publication semantics unchanged.

- [ ] **Step 7: Commit durable prediction/report publication**

```bash
rtk git add \
  src/benchmark/backend_publication.py \
  src/benchmark/backend_attestation.py \
  src/benchmark/backend_reports.py \
  src/benchmark/prediction_artifact.py \
  tests/benchmark/test_backend_publication.py \
  tests/benchmark/test_backend_attestation.py \
  tests/benchmark/test_backend_reports.py
rtk git commit -m "feat: publish backend artifacts and reports durably"
```

---

### Task 6: Add the explicit registry and `transcribe-one` orchestration

**Files:**

- Create: `src/benchmark/backend_registry.py`
- Create: `src/benchmark/transcription.py`
- Create: `tests/benchmark/test_backend_registry.py`
- Create: `tests/benchmark/test_transcription.py`

**Interfaces:**

- Consumes: common backend records, input loading, artifact/report publication.
- Produces: lazy explicit backend selection and `run_transcribe_one`.

- [ ] **Step 1: Write failing registry and orchestration tests**

```python
def test_registry_uses_official_backend_only_when_backend_is_omitted() -> None:
    registry = BackendRegistry(
        default_backend_id=OFFICIAL_BACKEND_ID,
        factories={OFFICIAL_BACKEND_ID: FakeBackend},
    )

    backend = registry.create(None)

    assert backend.descriptor().payload["backend_id"] == OFFICIAL_BACKEND_ID


def test_transcribe_one_publishes_prediction_then_complete_report(tmp_path: Path) -> None:
    request = direct_request(tmp_path)
    registry = BackendRegistry(
        default_backend_id=OFFICIAL_BACKEND_ID,
        factories={OFFICIAL_BACKEND_ID: FakeBackend},
    )

    outcome = run_transcribe_one(
        request,
        registry=registry,
        now=FIXED_UTC,
        run_id=FIXED_UUID,
    )

    assert outcome.exit_code == 0
    assert outcome.status == "complete"
    assert request.output_path.exists()
    assert outcome.report_artifact.path.exists()
```

Add tests for explicit heuristic selection, unknown backend, lazy missing-module
mapping to `backend_unavailable`, preflight before output creation, direct/derived
mode exclusivity, source/input hash failures, item inference failure, backend-fatal
verification/process error, artifact round-trip failure, prior prediction
preservation, optional MIDI requested without a writer, and guaranteed `close()` in
success/failure paths.

- [ ] **Step 2: Run tests and verify missing modules**

```bash
rtk uv run pytest -q \
  tests/benchmark/test_backend_registry.py \
  tests/benchmark/test_transcription.py
```

Expected: collection FAIL.

- [ ] **Step 3: Implement lazy explicit backend factories**

Define constants and registry:

```python
OFFICIAL_BACKEND_ID = "magenta-egmd-tf1-94529798-8hit-v1"
HEURISTIC_BACKEND_ID = "heuristic-onset-v1"
LEGACY_TF2_BACKEND_ID = "legacy-tf2-h5-v0"


class BackendUnavailable(RuntimeError):
    pass


@dataclass(frozen=True)
class BackendRegistry:
    default_backend_id: str
    factories: Mapping[str, Callable[[], TranscriptionBackend]]

    def create(self, backend_id: str | None) -> TranscriptionBackend:
        selected = self.default_backend_id if backend_id is None else backend_id
        factory = self.factories.get(selected)
        if factory is None:
            raise BackendUnavailable("unknown backend")
        return factory()
```

`default_backend_registry` registers three functions whose module imports occur
inside the function body:

- OaF imports `src.benchmark.backends.oaf_tf1.create_backend`.
- Heuristic imports `src.benchmark.backends.heuristic.create_backend`.
- Legacy TF2 raises `BackendUnavailable` because it is not a
  `TranscriptionBackend`; Phase C routes it only through the legacy scorer.

Catch `ImportError` and missing checked-in lock errors as
`BackendUnavailable("backend implementation is unavailable")`. Never select a
different factory.

- [ ] **Step 4: Implement `run_transcribe_one` phase ordering**

Define:

```python
@dataclass(frozen=True)
class TranscribeOneRequest:
    backend_id: str | None
    audio_path: Path
    output_path: Path
    source_audio_id: str | None
    input_view_id: str | None
    input_view_manifest: Path | None
    midi_output_path: Path | None
    reports_root: Path


@dataclass(frozen=True)
class TranscribeOneOutcome:
    status: Literal["complete", "partial", "failed", "environment_unsupported"]
    exit_code: Literal[0, 1, 2]
    report_artifact: PublishedArtifact

```

The exact callable signature is:

```text
run_transcribe_one(
    request: TranscribeOneRequest,
    *,
    registry: BackendRegistry,
    now: datetime | None = None,
    run_id: UUID | None = None,
    midi_writer: Callable[[PredictionArtifact, Path], MidiDerivative] | None = None,
) -> TranscribeOneOutcome
```

The implementation order is exact:

1. Resolve explicit/default backend without opening output files.
2. Call `verify`; map `environment_unsupported` to exit `1` and verification failure
   to exit `2`.
3. If `verification.backend_lock_sha256` is non-null, require a positive
   `verification.max_input_audio_frames`; otherwise allow null. Then load and re-hash
   direct or derived canonical audio against the supplied bound when present.
4. Call `transcribe`.
5. Render and strict-read prediction bytes.
6. Publish prediction atomically.
7. If MIDI was requested and no writer is injected, retain JSONL, mark the item
   incomplete with `midi_derivation_failed`, and return exit `1`. If the writer
   succeeds, report `MidiDerivative.midi`; the adjacent sidecar remains independently
   hash-bound to the prediction.
8. Publish one execution report and return its status.
9. Close the backend in `finally`.

No failure deletes a previously published immutable prediction.
If the execution report itself cannot be published, propagate
`OperationalReportPublicationError`; there is no valid `TranscribeOneOutcome`.

- [ ] **Step 5: Run orchestration tests and focused regressions**

```bash
rtk uv run pytest -q \
  tests/benchmark/test_backend_registry.py \
  tests/benchmark/test_transcription.py \
  tests/benchmark/test_prediction_artifact.py \
  tests/benchmark/test_backend_reports.py
```

Expected: PASS.

- [ ] **Step 6: Commit registry and orchestration**

```bash
rtk git add \
  src/benchmark/backend_registry.py \
  src/benchmark/transcription.py \
  tests/benchmark/test_backend_registry.py \
  tests/benchmark/test_transcription.py
rtk git commit -m "feat: orchestrate explicit transcription backends"
```

---

### Task 7: Expose `transcribe-one` and document the common contract

**Files:**

- Modify: `src/cli/benchmark.py:1-300`
- Modify: `tests/test_cli_benchmark.py`
- Modify: `docs/drumery-dtx-midi-benchmarking-reference.md:363-680`

**Interfaces:**

- Consumes: `TranscribeOneRequest`, `run_transcribe_one`,
  `default_backend_registry`.
- Produces: final Phase B CLI surface and operator documentation.

- [ ] **Step 1: Write failing Click tests**

```python
def test_transcribe_one_direct_mode_emits_canonical_summary(
    tmp_path: Path,
    monkeypatch,
) -> None:
    report = PublishedArtifact("execution_report", tmp_path / "report.json", "a" * 64)
    monkeypatch.setattr(
        "src.benchmark.transcription.run_transcribe_one",
        lambda request, registry: TranscribeOneOutcome("complete", 0, report),
    )

    result = CliRunner().invoke(
        main,
        [
            "benchmark",
            "transcribe-one",
            "--audio",
            str(tmp_path / "audio.wav"),
            "--source-audio-id",
            "source-v1",
            "--input-view-id",
            "view-v1",
            "--output",
            str(tmp_path / "prediction.jsonl"),
        ],
    )

    assert result.exit_code == 0
    assert json.loads(result.output) == {
        "exit_code": 0,
        "report_path": str(report.path),
        "report_sha256": "a" * 64,
        "status": "complete",
    }
```

Add cases for derived mode, mixed direct/derived arguments as Click exit `2`, missing
direct IDs, default official ID, explicit heuristic ID, unknown ID producing a typed
post-parse report, exit `1`, exit `2` with summary, and no human prose on stdout.

- [ ] **Step 2: Run CLI tests and verify the command is absent**

```bash
rtk uv run pytest -q tests/test_cli_benchmark.py -k transcribe_one
```

Expected: FAIL because `transcribe-one` is not registered.

- [ ] **Step 3: Register the command**

Add Click options:

```text
--backend TEXT
--audio FILE                         required
--source-audio-id TEXT
--input-view-id TEXT
--input-view-manifest FILE
--output FILE                        required
--midi-output FILE
--reports-root DIRECTORY             default artifacts/benchmark/backends
```

Use a custom validation function so exactly one provenance mode is accepted. Build
the request, call `run_transcribe_one`, and write
`canonical_json_bytes(summary, trailing_newline=True)` directly to standard output.
The summary has only `status`, `exit_code`, `report_path`, and `report_sha256`. Then
call `ctx.exit` for nonzero outcomes. Import backend modules lazily inside the
command so `crux benchmark --help` does not import TensorFlow or Librosa.

Catch `OperationalReportPublicationError` only at the CLI boundary, write the stable
code `report_publication_failed` plus a sanitized message to standard error, emit no
summary, and exit `2`. This is distinguishable from a Click usage error by standard
error; neither case falsely claims a published report.

- [ ] **Step 4: Update the benchmark reference**

Document:

- the common JSONL record order and canonical mapping state;
- direct and manifest-derived commands;
- default/explicit backend behavior;
- report paths under `artifacts/benchmark/backends/<backend-id>/`;
- exit `0/1/2` and machine-summary handling;
- that OaF implementation and heuristic/MIDI implementation land in Phases A/C;
- that native JSONL, not MIDI, is the authoritative prediction.

- [ ] **Step 5: Run CLI, import-boundary, and documentation checks**

```bash
rtk uv run pytest -q \
  tests/test_cli_benchmark.py \
  tests/benchmark/test_transcription.py
rtk uv run python -c "from src.cli.main import main; print(sorted(main.commands))"
rtk rg -n 'transcribe-one|crux.drum-prediction-events/v1|mapping_status' \
  docs/drumery-dtx-midi-benchmarking-reference.md
```

Expected: tests pass, the import command prints `benchmark` and `convert` without
TensorFlow 1 imports, and the documentation search returns the new contract.

- [ ] **Step 6: Commit CLI and documentation**

```bash
rtk git add \
  src/cli/benchmark.py \
  tests/test_cli_benchmark.py \
  docs/drumery-dtx-midi-benchmarking-reference.md
rtk git commit -m "feat: expose native transcription artifacts"
```

---

### Task 8: Prove Phase B acceptance and hand interfaces to Phases A/C

**Files:**

- Modify: `tests/benchmark/test_transcription.py`
- Modify: `tests/benchmark/test_prediction_artifact.py`
- Create: `tests/benchmark/test_backend_contract_acceptance.py`

**Interfaces:**

- Consumes: every Phase B public interface.
- Produces: one cross-module fake-backend acceptance test and a stable handoff.

- [ ] **Step 1: Add an end-to-end deterministic fake-backend test**

```python
def test_fake_backend_repeats_byte_identical_prediction(tmp_path: Path) -> None:
    first = run_fake_transcription(tmp_path / "first", run_id=FIRST_UUID)
    second = run_fake_transcription(tmp_path / "second", run_id=SECOND_UUID)

    assert first.prediction_bytes == second.prediction_bytes
    assert first.item_id == second.item_id
    assert first.report_bytes != second.report_bytes
    assert first.report.descriptor_sha256 == second.report.descriptor_sha256
```

The fixture must include simultaneous distinct events, nullable heuristic fields, one
OaF metadata example, Unicode source IDs, direct and derived provenance, and a
round-trip through the strict reader. Assert report variability is limited to
run/timestamp/path fields.

- [ ] **Step 2: Run the focused Phase B suite**

```bash
rtk uv run pytest -q \
  tests/benchmark/test_backend_identity.py \
  tests/benchmark/test_backend_types.py \
  tests/benchmark/test_input_view.py \
  tests/benchmark/test_backend_publication.py \
  tests/benchmark/test_backend_attestation.py \
  tests/benchmark/test_prediction_artifact.py \
  tests/benchmark/test_scorer_input.py \
  tests/benchmark/test_backend_reports.py \
  tests/benchmark/test_backend_registry.py \
  tests/benchmark/test_transcription.py \
  tests/benchmark/test_backend_contract_acceptance.py \
  tests/test_cli_benchmark.py
```

Expected: PASS.

- [ ] **Step 3: Run repository quality gates**

```bash
rtk uv run pytest -q
rtk uv run ruff check src tests
rtk uv run black --check src tests
rtk uv run pylint src/app src/cli src/benchmark
rtk git diff --check
```

Expected: every command exits `0`.

- [ ] **Step 4: Verify import and scope boundaries**

```bash
rtk git diff --name-only
rtk rg -n 'tensorflow|librosa|DrumTranscriber' \
  src/benchmark/backend_identity.py \
  src/benchmark/backends/base.py \
  src/benchmark/input_view.py \
  src/benchmark/prediction_artifact.py \
  src/benchmark/scorer_input.py \
  src/benchmark/backend_attestation.py \
  src/benchmark/backend_reports.py \
  src/benchmark/backend_registry.py \
  src/benchmark/transcription.py
```

Expected: the changed files match the Phase B map, and the dependency search returns
only lazy factory import strings in `backend_registry.py`.

- [ ] **Step 5: Commit acceptance coverage**

```bash
rtk git add tests/benchmark/test_backend_contract_acceptance.py
rtk git commit -m "test: prove backend contract repeatability"
```

Phase B is complete when this task passes. Phase A may then implement the OaF lock,
runner, and factory without changing these common signatures; Phase C may implement
the heuristic, MIDI writer, and legacy wrapper through the same contracts.
