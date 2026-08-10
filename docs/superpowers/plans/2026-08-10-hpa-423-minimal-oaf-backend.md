# HPA-423 Minimal Reusable OaF Backend Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the unusable sealed OaF path with one reusable persistent OaF backend that verifies the released checkpoint, persists HPA-324-compatible prediction events, and passes one real native smoke inference.

**Architecture:** Keep the existing canonical-audio/native-event domain model and extract the working TensorFlow inference code into `runtime/oaf_tf1/model.py`. A small line-JSON worker loads that model once; a thin host adapter owns the persistent process and returns `NativePrediction`; an OaF-specific mapper consumes HPA-324 taxonomy data and writes prediction artifact v2 with common classes for scoring. Remove seal/attestation/report/heuristic machinery instead of adapting it.

**Tech Stack:** Python 3.12 host tooling/tests, existing pinned Python 3.7/TensorFlow 1 OaF container dependencies, Click CLI, Docker, pytest, Ruff, Pylint, GitHub Actions `ubuntu-24.04`.

## Global Constraints

- Preserve `OAF_BACKEND_ID = "magenta-egmd-tf1-94529798-8hit-v1"`.
- Preserve upstream Magenta revision `94529798dfbbb14c27ddfd76f23027dc8e2ce185`.
- Preserve checkpoint archive SHA-256 `09765ae0ff19c7d769a3c20e158eba3b9cd279429b02e498b1e911d16f82e2c0`.
- Preserve checkpoint component SHA-256 values `6312cd6f8e29ee6f21f4a763822d554f46393e0e336812ba4a8cb324d2e8b0b5`, `475ca21993e102d52e2e6c7066b598b9b9e9a097bfb2fa6b770d40eedaa6139a`, and `e36105b27c17f2367292682df825b8ad16eb887ccad465bbb5ddf955ef75d422`.
- Require all 78 inference tensors to restore before the worker reports ready.
- Preserve `CanonicalAudio.source_audio_id`, `source_audio_sha256`, `input_view_id`, and `input_audio_sha256`.
- Preserve OaF `native_class_id="midi_<note>"` and `native_metadata["upstream_8hit_group_id"]` identity.
- HPA-324 owns taxonomy/map data. Do not duplicate a temporary taxonomy or OaF class table inside HPA-423.
- Prediction persistence is a breaking replacement to `crux.drum-prediction-events/v2`; do not add a v1 compatibility reader.
- One persistent worker, one request at a time. No worker pool, queue, retry framework, or distributed execution.
- Do not add OIDC, Sigstore, host attestation, runtime seals, deterministic OCI publication, or security-only policy code.
- Keep TensorFlow imports lazy/outside host module import paths.
- Do not change the pinned TensorFlow/OaF dependency set unless extraction proves a dependency is unused and removing it is independently safe.
- Use repository validation from `CLAUDE.md`: `uv run pytest`, `uv run ruff check src tests`, `uv run ruff format --check src tests`, and error-focused Pylint on changed host modules.

---

## Target Active File Structure

```text
src/benchmark/
  artifact_io.py
  backend_identity.py
  backend_process.py
  backend_registry.py
  checkpoint_acquisition.py
  prediction_mapping.py
  prediction_artifact.py
  scorer_input.py
  backends/
    base.py
    oaf.py
runtime/oaf_tf1/
  model.json
  model.py
  worker.py
  Dockerfile
  requirements.lock
  wheelhouse/
  vendor/
  patches/capture-emitted-frame.patch
.github/workflows/
  oaf-smoke.yml
```

`src/benchmark/taxonomy.py` is supplied by HPA-324 and is consumed here; HPA-423 does not own its policy data.

---

### Task 1: Remove seal state from the shared backend contract and registry

**Files:**
- Modify: `src/benchmark/backends/base.py`
- Modify: `src/benchmark/backends/__init__.py`
- Modify: `src/benchmark/backend_identity.py`
- Modify: `src/benchmark/backend_registry.py`
- Modify: `tests/benchmark/test_backend_contract_acceptance.py`
- Modify: `tests/benchmark/test_backend_registry.py`
- Modify: `tests/benchmark/test_backend_identity.py`
- Modify: `tests/benchmark/test_schema_goldens.py`
- Delete: `tests/benchmark/schema_goldens/crux.heuristic-backend-descriptor-v1.json`
- Delete: `tests/benchmark/schema_goldens/crux.transcription-backend-descriptor-v1.json`

**Interfaces:**
- Produces `CanonicalAudio`, `NativeEvent`, `NativePrediction(audio, descriptor, events)`, and `TranscriptionBackend.descriptor/transcribe/close`.
- Produces compact `BackendDescriptor` v2 for OaF identity.
- Produces `default_backend_registry()` with one key, `"oaf"`.

- [ ] **Step 1: Write failing contract tests**

Add exact assertions:

```python
import src.benchmark.backends as backends


def test_native_prediction_has_only_transcription_domain_fields():
    assert set(backends.NativePrediction.__dataclass_fields__) == {
        "audio",
        "descriptor",
        "events",
    }


def test_canonical_audio_retains_source_and_input_identity():
    fields = set(backends.CanonicalAudio.__dataclass_fields__)
    assert {
        "source_audio_id",
        "source_audio_sha256",
        "input_view_id",
        "input_audio_sha256",
    } <= fields


def test_verification_types_are_not_public_backend_contract():
    assert not hasattr(backends, "BackendVerification")
    assert not hasattr(backends, "TensorCoverageCheck")
    assert not hasattr(backends, "SmokeCheck")
```

Use `typing.get_type_hints(TranscriptionBackend)` plus direct attribute checks to require `descriptor`, `transcribe`, and `close`, and explicitly assert `verify` is absent.

- [ ] **Step 2: Write failing registry tests**

```python
import pytest

from src.benchmark.backend_registry import BackendUnavailable, default_backend_registry


def test_default_registry_contains_only_oaf():
    registry = default_backend_registry()
    assert registry.default_backend_id == "oaf"
    assert set(registry.factories) == {"oaf"}


def test_removed_heuristic_is_unknown():
    registry = default_backend_registry()
    with pytest.raises(BackendUnavailable, match="unknown backend"):
        registry.create("heuristic-onset-v1")
```

Also assert the module no longer exposes `SealState`, `BackendNotSealed`, `BackendIntegrityUnavailable`, `HEURISTIC_BACKEND_ID`, or `LEGACY_TF2_BACKEND_ID`.

- [ ] **Step 3: Write failing compact-descriptor tests**

Require exact OaF descriptor keys:

```python
EXPECTED_OAF_DESCRIPTOR_KEYS = {
    "architecture_id",
    "backend_id",
    "descriptor_schema",
    "model_id",
    "native_metadata_schema_id",
    "native_output_space_id",
    "prediction_schema",
    "training_data_map_id",
    "upstream_source_commit",
}


def test_oaf_descriptor_v2_has_only_model_identity():
    assert OAF_DESCRIPTOR_SCHEMA == "crux.transcription-backend-descriptor/v2"
    assert OAF_DESCRIPTOR_KEYS == EXPECTED_OAF_DESCRIPTOR_KEYS
```

Assert the expected descriptor constants include `prediction_schema="crux.drum-prediction-events/v2"` and do not contain lock hashes, runtime image digest, or protocol schema.

- [ ] **Step 4: Run focused tests and confirm the old contract fails them**

```bash
uv run pytest \
  tests/benchmark/test_backend_contract_acceptance.py \
  tests/benchmark/test_backend_registry.py \
  tests/benchmark/test_backend_identity.py -q
```

Expected: failures expose verification fields, `seal_state`, heuristic registration, and v1 descriptor shape.

- [ ] **Step 5: Reduce `backends/base.py` and exports**

Use this exact domain shape:

```python
@dataclass(frozen=True)
class NativePrediction:
    audio: CanonicalAudio
    descriptor: BackendDescriptor
    events: tuple[NativeEvent, ...]


class TranscriptionBackend(Protocol):
    def descriptor(self) -> BackendDescriptor:
        raise NotImplementedError

    def transcribe(self, audio: CanonicalAudio) -> NativePrediction:
        raise NotImplementedError

    def close(self) -> None:
        raise NotImplementedError
```

Keep existing `CanonicalAudio` and `NativeEvent` fields unchanged. Remove verification, smoke, tensor, lock, and publication result types from the backend contract and `backends/__init__.py`.

- [ ] **Step 6: Replace backend descriptor v1 with compact v2**

Keep shared canonical JSON/hash utilities. Define `OAF_DESCRIPTOR_SCHEMA`, `OAF_DESCRIPTOR_KEYS`, the frozen OaF identity values, and:

```python
def build_oaf_descriptor(payload: Mapping[str, str]) -> BackendDescriptor:
    return build_descriptor(payload, OAF_DESCRIPTOR_KEYS, OAF_DESCRIPTOR_SCHEMA)
```

Delete all heuristic descriptor constants and all OaF lock/image/protocol descriptor fields. Validation still checks the exact backend/model/source identities and the 40-character upstream commit.

- [ ] **Step 7: Replace `backend_registry.py` with a one-entry static registry**

```python
class BackendUnavailable(ValueError):
    pass


@dataclass(frozen=True)
class BackendRegistry:
    default_backend_id: str
    factories: Mapping[str, Callable[[], TranscriptionBackend]]

    def create(self, backend_id: str | None) -> TranscriptionBackend:
        selected = self.default_backend_id if backend_id is None else backend_id
        factory = self.factories.get(selected)
        if factory is None:
            raise BackendUnavailable(f"unknown backend: {selected}")
        return factory()
```

`default_backend_registry()` returns only `{"oaf": _create_oaf_backend}`. `_create_oaf_backend` keeps a lazy import of `src.benchmark.backends.oaf.create_backend` so Task 1 can land before Task 6.

- [ ] **Step 8: Remove old descriptor goldens**

Delete both v1 descriptor goldens and remove their entries from `test_schema_goldens.py`. Do not replace them with a new exact JSON golden; `test_backend_identity.py` is the active descriptor contract test.

- [ ] **Step 9: Run focused tests**

```bash
uv run pytest \
  tests/benchmark/test_backend_contract_acceptance.py \
  tests/benchmark/test_backend_registry.py \
  tests/benchmark/test_backend_identity.py \
  tests/benchmark/test_schema_goldens.py -q
```

Expected: PASS.

- [ ] **Step 10: Commit**

```bash
git add src/benchmark/backends src/benchmark/backend_identity.py \
  src/benchmark/backend_registry.py tests/benchmark
git commit -m "refactor: remove OaF seal lifecycle from backend contract"
```

---

### Task 2: Extract generic artifact I/O and make `model.json` drive checkpoint preparation

**Files:**
- Create: `src/benchmark/artifact_io.py`
- Create: `runtime/oaf_tf1/model.json`
- Create: `runtime/oaf_tf1/model.py` initially with dependency-light config types/loader only
- Modify: `src/benchmark/input_view.py`
- Rewrite: `src/benchmark/checkpoint_acquisition.py`
- Modify: `src/benchmark/prediction_artifact.py` imports only
- Modify: `src/cli/benchmark.py` `prepare-backend` command
- Add: `tests/benchmark/test_artifact_io.py`
- Rewrite: `tests/benchmark/test_checkpoint_acquisition.py`
- Modify: `tests/test_cli_benchmark.py`
- Delete when imports are migrated: `src/benchmark/backend_prepare.py`
- Delete when imports are migrated: `src/benchmark/backend_publication.py`

**Interfaces:**
- `read_regular_file_no_follow(path: Path) -> bytes`
- `publish_immutable_file(path: Path, content: bytes) -> PublishedArtifact`
- `load_model_config(path: Path = Path(__file__).with_name("model.json")) -> OafModelConfig`
- `prepare_oaf_checkpoint(config: OafModelConfig, cache_root: Path, *, download: bool, archive_path: Path | None = None) -> Path`

- [ ] **Step 1: Write failing artifact I/O tests**

Add concrete tests equivalent to:

```python
def test_read_regular_file_no_follow_rejects_symlink(tmp_path):
    target = tmp_path / "target.bin"
    target.write_bytes(b"abc")
    link = tmp_path / "link.bin"
    link.symlink_to(target)
    with pytest.raises(OSError):
        read_regular_file_no_follow(link)


def test_publish_immutable_file_is_idempotent_for_same_bytes(tmp_path):
    path = tmp_path / "nested" / "artifact.json"
    first = publish_immutable_file(path, b"{\"ok\":true}\n")
    second = publish_immutable_file(path, b"{\"ok\":true}\n")
    assert first == second
    assert path.read_bytes() == b"{\"ok\":true}\n"


def test_publish_immutable_file_rejects_changed_existing_bytes(tmp_path):
    path = tmp_path / "artifact.json"
    publish_immutable_file(path, b"one")
    with pytest.raises(ArtifactPublicationError):
        publish_immutable_file(path, b"two")
```

Also assert returned SHA-256 equals `hashlib.sha256(content).hexdigest()`.

- [ ] **Step 2: Write failing `model.json` loader tests**

```python
def test_model_config_matches_released_oaf_identity():
    config = load_model_config(Path("runtime/oaf_tf1/model.json"))
    assert config.backend_id == "magenta-egmd-tf1-94529798-8hit-v1"
    assert config.model_id == "magenta-egmd-ckpt-569400-v1"
    assert config.upstream_source_commit == "94529798dfbbb14c27ddfd76f23027dc8e2ce185"
    assert config.checkpoint.archive_sha256 == "09765ae0ff19c7d769a3c20e158eba3b9cd279429b02e498b1e911d16f82e2c0"
    assert {component.name for component in config.checkpoint.components} == {
        "model.ckpt-569400.data-00000-of-00001",
        "model.ckpt-569400.index",
        "model.ckpt-569400.meta",
    }
```

For malformed config, copy the JSON to `tmp_path`, replace `archive_sha256` with `"bad"`, and assert `OafModelConfigError` is raised with `archive_sha256` in the message.

- [ ] **Step 3: Write checkpoint preparation tests with a synthetic ZIP**

Create a test helper that writes exactly three component files into a ZIP, reads the ZIP bytes, calculates archive/component hashes, and returns an `OafModelConfig` built with those calculated values. Use it to test:

- download-disabled local `archive_path` extracts all three verified components;
- verify-only call reuses an already verified cache;
- changing one byte in the archive produces `CheckpointAcquisitionError` before extraction is accepted;
- changing a component hash in the config produces `CheckpointAcquisitionError`;
- verify-only on an absent cache raises `CheckpointAcquisitionError("checkpoint cache is missing")`.

The cache assertion is exact: returned directory equals `<cache_root>/sha256/<archive_sha256>`.

- [ ] **Step 4: Run new tests and confirm neutral APIs are absent**

```bash
uv run pytest tests/benchmark/test_artifact_io.py tests/benchmark/test_checkpoint_acquisition.py -q
```

Expected: import/API failures.

- [ ] **Step 5: Implement only retained generic artifact behavior**

`artifact_io.py` contains:

```python
@dataclass(frozen=True)
class PublishedArtifact:
    path: Path
    sha256: str


class ArtifactPublicationError(OSError):
    pass
```

Implement no-follow regular-file reads and immutable file create/reuse. Do not port seal directory transactions, rollback, attestation, or manifest machinery.

Update retained imports in `input_view.py`, `checkpoint_acquisition.py`, and `prediction_artifact.py`. If `DirectoryAnchor` is unused after report/transcription call sites are removed, delete the parameter/import instead of recreating it. If one retained HPA-323/HPA-324 path still requires anchored reads, implement only the small anchor type and direct read function used by that path, with a matching test in `test_artifact_io.py`.

- [ ] **Step 6: Create `model.json` and dependency-light config loader**

`runtime/oaf_tf1/model.json` uses schema `crux.oaf-model/v1` and the exact identities/hashes from the design spec.

`runtime/oaf_tf1/model.py` initially defines `CheckpointComponent`, `OafCheckpointConfig`, `OafModelConfig`, `OafModelConfigError`, and `load_model_config()`. This module must import only Python standard-library modules at import time in Task 2; Task 4 adds lazy numeric/TensorFlow imports inside model loading.

- [ ] **Step 7: Rewrite checkpoint acquisition around one direct function**

Reuse the useful existing ZIP/download/hash code, but remove backend lock/evidence/publication concepts. Cache verified components under:

```text
<cache_root>/sha256/<archive_sha256>/
```

`prepare_oaf_checkpoint()` returns that verified directory. It raises a direct `CheckpointAcquisitionError` on download, archive hash, member, extraction, or component hash failure.

- [ ] **Step 8: Simplify `prepare-backend`**

The command surface becomes:

```text
crux benchmark prepare-backend
  --backend [oaf]
  --download
  --archive PATH
  --cache-root DIRECTORY
```

`--backend` defaults to `oaf`; `--cache-root` defaults to `artifacts/benchmark/model-cache`. `--download` and `--archive` are mutually exclusive. Remove `--acquisition-request`, `--evidence-output`, and `--backend-lock`.

On success print one canonical JSON line with exactly `backend`, `checkpoint_path`, and `status="ready"`.

- [ ] **Step 9: Prove old container modules are no longer needed, then delete them**

```bash
git grep -n "backend_publication\|backend_prepare" -- src runtime tests .github tools
```

Migrate retained imports first. Delete `backend_prepare.py` and `backend_publication.py` once remaining matches are only old HPA-320/report code scheduled for deletion; deleting those old callers in the same commit is allowed when required to keep imports valid.

- [ ] **Step 10: Run focused tests and CLI help**

```bash
uv run pytest \
  tests/benchmark/test_artifact_io.py \
  tests/benchmark/test_checkpoint_acquisition.py \
  tests/test_cli_benchmark.py -q
uv run crux benchmark prepare-backend --help
```

Expected: PASS; help contains no seal/evidence/lock options.

- [ ] **Step 11: Commit**

```bash
git add src/benchmark src/cli/benchmark.py runtime/oaf_tf1/model.py \
  runtime/oaf_tf1/model.json tests
git commit -m "refactor: simplify OaF checkpoint preparation"
```

---

### Task 3: Persist HPA-324 mapped predictions as artifact v2 and unblock scorer input

**Precondition:** HPA-324 production code is present with `PredictionMap`, `ClassMapping`, `OAF_PREDICTION_MAP`, and `OAF_PREDICTION_MAP_ID`. If HPA-324 has not merged, integrate its reviewed production commit; do not copy its mapping table into HPA-423.

**Files:**
- Create: `src/benchmark/prediction_mapping.py`
- Rewrite: `src/benchmark/prediction_artifact.py`
- Rewrite: `src/benchmark/scorer_input.py`
- Add: `tests/benchmark/test_prediction_mapping.py`
- Rewrite: `tests/benchmark/test_prediction_artifact.py`
- Add: `tests/benchmark/test_scorer_input.py`
- Replace if active schema-golden coverage remains: `tests/benchmark/schema_goldens/crux.drum-prediction-events-v1.jsonl` with `tests/benchmark/schema_goldens/crux.drum-prediction-events-v2.jsonl`

**Interfaces:**
- `map_oaf_prediction(prediction: NativePrediction, prediction_map: PredictionMap = OAF_PREDICTION_MAP) -> PredictionMappingResult`
- `MappedPredictionEvent(native, canonical_class, common_class, mapping_status, prediction_map_version)`
- `MappedPrediction(audio, descriptor, events)`
- `read_scorer_events(content: bytes) -> tuple[BenchmarkEvent, ...]`

- [ ] **Step 1: Write real-shaped OaF mapping fixtures and failing tests**

Use this concrete helper:

```python
def oaf_event(group_id: str | None, midi_note: int) -> NativeEvent:
    return NativeEvent(
        time_sec=0.5,
        native_class_id=f"midi_{midi_note}",
        model_output_bin=midi_note - 21,
        native_midi_note=midi_note,
        native_metadata={"upstream_8hit_group_id": group_id},
        confidence=0.9,
        velocity_midi=100,
    )
```

Build a `NativePrediction` whose descriptor carries the full `backend_id` and native output-space identity expected by the HPA-324 map. Assert:

- `hihat`, MIDI 46 -> mapped, `canonical_class is None`, `common_class == "hihat"`;
- `toms`, MIDI 48 -> mapped, `canonical_class is None`, `common_class == "tom"`;
- `snare`, MIDI 38 -> mapped detailed/common `snare`;
- `sticks`, MIDI 75 -> unmapped with both class fields null;
- null/unknown group -> unmapped and counted by diagnostic key;
- every event has `prediction_map_version == OAF_PREDICTION_MAP_ID`;
- every native field is unchanged.

- [ ] **Step 2: Write failing artifact-v2 round-trip tests**

Render the hihat/tom/sticks mapped prediction and parse each physical JSONL event record. Require `schema == "crux.drum-prediction-events/v2"`, a `common_class` field, and exact semantics:

```text
hihat: mapping_status=mapped, canonical_class=null, common_class=hihat, map_version non-null
toms:  mapping_status=mapped, canonical_class=null, common_class=tom,   map_version non-null
sticks: mapping_status=unmapped, canonical_class=null, common_class=null, map_version non-null
```

Read the bytes back and assert native MIDI note, output bin, confidence, velocity, class ID, and metadata reproduce exactly.

- [ ] **Step 3: Write failing scorer-input tests**

Create v2 bytes from one mapped hihat event plus one unmapped sticks event. Assert:

```python
events = read_scorer_events(content)
assert len(events) == 1
assert events[0].chart_id == "song-1"
assert events[0].canonical_class == "hihat"
assert events[0].source == "prediction"
assert events[0].metadata["native_class_id"] == "midi_46"
assert events[0].metadata["input_view_id"] == "full_mix"
```

The unmapped sticks event must not become a fake scoring class.

- [ ] **Step 4: Run tests and confirm current v1/not-applied behavior fails**

```bash
uv run pytest \
  tests/benchmark/test_prediction_mapping.py \
  tests/benchmark/test_prediction_artifact.py \
  tests/benchmark/test_scorer_input.py -q
```

- [ ] **Step 5: Implement mapped prediction domain and OaF mapper**

Define the dataclasses from the Interfaces block plus `PredictionMappingDiagnostics(unmapped: Mapping[str, int])` and `PredictionMappingResult(prediction, diagnostics)`.

The mapper validates:

```python
prediction.descriptor.payload["backend_id"] == prediction_map.model_id
prediction.descriptor.payload["native_output_space_id"] == prediction_map.native_output_space_id
```

This matches the HPA-324 contract where `PredictionMap.model_id` identifies the locked backend/model path. Lookup is only:

```python
key = event.native_metadata.get("upstream_8hit_group_id")
class_mapping = prediction_map.classes.get(key) if key is not None else None
```

A map entry with non-null `common_class` is `mapped`; missing entries or entries with null common class are `unmapped`. Native events are always retained.

- [ ] **Step 6: Advance `prediction_artifact.py` to v2**

Set `PREDICTION_SCHEMA = "crux.drum-prediction-events/v2"`.

Keep header/event/terminal JSONL structure, but remove seal-only header fields: backend/runtime/parameter/model-artifact lock hashes, upstream values duplicated outside the descriptor, and heuristic nullability rules. Add `common_class` to event exact keys. The renderer accepts `MappedPrediction`; the reader reconstructs `MappedPrediction`.

No v1 compatibility branch is permitted.

- [ ] **Step 7: Replace `scorer_input.py` stub**

For each mapped event with non-null `common_class`, create:

```python
BenchmarkEvent(
    chart_id=artifact.prediction.audio.source_audio_id,
    time_sec=event.native.time_sec,
    canonical_class=event.common_class,
    source="prediction",
    metadata={
        "input_view_id": artifact.prediction.audio.input_view_id,
        "native_class_id": event.native.native_class_id,
        "native_midi_note": event.native.native_midi_note,
        "model_output_bin": event.native.model_output_bin,
        "detailed_canonical_class": event.canonical_class,
        "prediction_map_version": event.prediction_map_version,
    },
)
```

Sort and return a tuple. Delete `CanonicalMappingRequired`. Do not inspect backend ID here.

- [ ] **Step 8: Replace the active v1 golden if schema-golden coverage remains**

The v2 golden must contain one common-only mapped event and one unmapped event. Remove the v1 file and every active v1 assertion.

- [ ] **Step 9: Run focused tests**

```bash
uv run pytest \
  tests/benchmark/test_prediction_mapping.py \
  tests/benchmark/test_prediction_artifact.py \
  tests/benchmark/test_scorer_input.py \
  tests/benchmark/test_schema_goldens.py -q
```

Expected: PASS, including hihat/tom round-trip.

- [ ] **Step 10: Commit**

```bash
git add src/benchmark/prediction_mapping.py src/benchmark/prediction_artifact.py \
  src/benchmark/scorer_input.py tests/benchmark
git commit -m "feat: persist mapped OaF prediction classes"
```

---

### Task 4: Extract the TensorFlow OaF model engine from the sealed runtime

**Files:**
- Expand: `runtime/oaf_tf1/model.py`
- Modify: `runtime/oaf_tf1/Dockerfile`
- Keep: `runtime/oaf_tf1/patches/capture-emitted-frame.patch`
- Keep: `runtime/oaf_tf1/vendor/`
- Add: `runtime/oaf_tf1/tests/test_model.py`
- Migrate useful coverage from: `runtime/oaf_tf1/tests/test_tensor_coverage.py`
- Extraction source, delete in Task 6: `runtime/oaf_tf1/oaf_backend.py`

**Interfaces:**
- `OafModel.load(checkpoint_dir: Path, config: OafModelConfig | None = None) -> OafModel`
- `OafModel.restored_tensor_count -> int`
- `OafModel.transcribe(audio_path: Path) -> tuple[OafNativeEvent, ...]`
- `validate_required_tensors(required: Collection[str], restored: Collection[str]) -> int`

- [ ] **Step 1: Write model config/import-boundary tests**

Assert importing `runtime.oaf_tf1.model` does not add `tensorflow` to `sys.modules`. Then call `load_model_config()` and reassert the released IDs/hashes from Task 2.

- [ ] **Step 2: Write concrete native-event conversion tests using existing known OaF groups**

Reuse representative frame/post-processing fixtures from the old runtime tests. Require at least these identities after conversion:

```text
snare -> native_class_id=midi_38, model_output_bin=17, native_midi_note=38, group=snare
hihat -> native_class_id=midi_46, model_output_bin=25, native_midi_note=46, group=hihat
toms  -> native_class_id=midi_48, model_output_bin=27, native_midi_note=48, group=toms
```

Also assert time, confidence, and velocity are preserved from the representative input fixture.

- [ ] **Step 3: Write required-tensor tests**

Load the existing required tensor inventory from the extraction source. Assert a complete 78-name restored set returns `78`; remove one known tensor and assert `OafModelError` contains that tensor name.

- [ ] **Step 4: Run new model tests and confirm extraction is incomplete**

```bash
uv run pytest runtime/oaf_tf1/tests/test_model.py -q
```

- [ ] **Step 5: Move only inference-relevant code into `model.py`**

Add:

```python
@dataclass(frozen=True)
class OafNativeEvent:
    time_sec: float
    native_class_id: str
    model_output_bin: int
    native_midi_note: int
    upstream_8hit_group_id: str | None
    confidence: float
    velocity_midi: int


class OafModelError(RuntimeError):
    pass
```

Move graph construction, feature extraction, checkpoint restore, required-tensor validation, emitted-frame instrumentation handling, post-processing, and native event conversion. TensorFlow/Numpy/Magenta imports occur inside `OafModel.load()` or helpers called only from it.

Do not move backend locks, seal evidence, host identity, calibration requests, source-attestation manifests, or publication code.

- [ ] **Step 6: Simplify Docker build context while preserving inference requirements**

Keep the pinned base/wheelhouse/requirements, vendored Magenta source, and instrumentation patch. Remove copies of HPA-320 workflows, `tools/hpa320`, runner/build-context/source-attestation manifests, and calibration-only files. Keep a test target only if runtime unit tests use it.

- [ ] **Step 7: Run model tests and a Docker test build**

```bash
uv run pytest runtime/oaf_tf1/tests/test_model.py -q
docker build --target test -f runtime/oaf_tf1/Dockerfile -t crux-oaf-tf1-test .
```

The real checkpoint is not required yet.

- [ ] **Step 8: Commit**

```bash
git add runtime/oaf_tf1 src/benchmark/checkpoint_acquisition.py tests/benchmark/test_checkpoint_acquisition.py
git commit -m "refactor: extract reusable OaF model engine"
```

---

### Task 5: Replace the sealed protocol/process controller with one persistent sequential worker

**Files:**
- Create: `runtime/oaf_tf1/worker.py`
- Rewrite: `src/benchmark/backend_process.py`
- Add: `runtime/oaf_tf1/tests/test_worker.py`
- Rewrite: `tests/benchmark/test_backend_process.py`
- Delete after migration: `runtime/oaf_tf1/entrypoint.py`
- Delete after migration: `runtime/oaf_tf1/protocol.py`

**Interfaces:**
- Ready line: `{"type":"ready","backend_id":"magenta-egmd-tf1-94529798-8hit-v1","restored_tensor_count":78}`.
- Request line: `{"id":"request-1","audio_path":"relative/path.wav"}`.
- Success line: `{"id":"request-1","events":[...]}`.
- Error line: `{"id":"request-1","error":{"code":"inference_failed","message":"inference failed"}}`.
- Host: `WorkerProcess.start(command, startup_timeout_sec=120.0, request_timeout_sec=600.0)`, `.ready`, `.request(payload)`, `.close()`.

- [ ] **Step 1: Write worker lifecycle tests with a concrete fake model**

Create a fake model class in the test with a `calls` list, `restored_tensor_count=78`, and `transcribe()` that appends the path and returns one fixed `OafNativeEvent`. Inject a model factory into `worker.main()` or `worker.run()` and feed two JSON request lines through `io.StringIO`. Assert:

- factory call count is one;
- first output is ready;
- two responses have matching request IDs;
- fake model calls contain both `/input/...` paths in order;
- EOF returns zero.

Feed `{not-json}\n` as another test and assert a single error response with code `invalid_request`; feed a valid request whose fake model raises `OafModelError` and assert code `inference_failed`.

- [ ] **Step 2: Write process-wrapper tests using a generated fake worker script**

The test writes a Python script that prints a valid ready line, then echoes each request ID with an empty events list. Start it with `[sys.executable, str(script)]`; call `.request()` twice; assert both IDs return; call `.close()`; assert the child exits.

Write separate scripts that emit malformed ready JSON, exit before ready, and emit a response with the wrong request ID. Each must raise `WorkerProcessError` with a stable message identifying startup JSON, early exit, or ID mismatch.

- [ ] **Step 3: Run focused tests and confirm old APIs fail them**

```bash
uv run pytest runtime/oaf_tf1/tests/test_worker.py tests/benchmark/test_backend_process.py -q
```

- [ ] **Step 4: Implement `worker.py` as one synchronous line loop**

Load `OafModel` once, write ready, then process stdin line by line. Validate request object has exactly string `id` and relative string `audio_path`; reject absolute paths and `..`. Convert each `OafNativeEvent` into a JSON object with time/native IDs/group/confidence/velocity. Flush after every output line.

No protocol version, signatures, compatibility decoder, sequence/resume state, or diagnostic-redaction subsystem.

- [ ] **Step 5: Rewrite `backend_process.py` as a small synchronous wrapper**

Define:

```python
class WorkerProcessError(RuntimeError):
    pass


class WorkerProcess:
    @classmethod
    def start(
        cls,
        command: Sequence[str],
        *,
        startup_timeout_sec: float = 120.0,
        request_timeout_sec: float = 600.0,
    ) -> "WorkerProcess":
        ...
```

The method body in production must use `subprocess.Popen(..., stdin=PIPE, stdout=PIPE, stderr=PIPE, text=True)` and a small selector-based `readline` timeout helper. Store ready payload. `.request()` holds one `threading.Lock`, assigns/validates the provided `id`, writes one JSON line, reads one JSON line, and rejects ID mismatch/malformed response. `.close()` closes stdin, waits briefly for clean EOF, then terminates/kills only if necessary.

The `...` above is Python's function-body notation only in this interface declaration; the implementation step requires a real body and no placeholder code in the repository.

- [ ] **Step 6: Delete old `entrypoint.py` and `protocol.py` after import search**

```bash
git grep -n "runtime\.oaf_tf1\.protocol\|runtime\.oaf_tf1\.entrypoint\|from protocol" -- src runtime tests
```

Move any still-needed pure inference logic into `model.py` first. Then delete both old files.

- [ ] **Step 7: Run worker/process tests**

```bash
uv run pytest runtime/oaf_tf1/tests/test_worker.py tests/benchmark/test_backend_process.py -q
```

Expected: PASS.

- [ ] **Step 8: Commit**

```bash
git add runtime/oaf_tf1 src/benchmark/backend_process.py tests/benchmark/test_backend_process.py
git commit -m "refactor: simplify persistent OaF worker protocol"
```

---

### Task 6: Add the thin Docker-backed OaF adapter and activate it

**Files:**
- Create: `src/benchmark/backends/oaf.py`
- Add: `tests/benchmark/test_oaf_adapter.py`
- Modify: `src/benchmark/backend_registry.py`
- Modify: `tests/benchmark/test_backend_registry.py`
- Delete after extraction: `src/benchmark/backends/oaf_tf1.py`
- Delete after extraction: `runtime/oaf_tf1/oaf_backend.py`
- Delete/migrate: `tests/benchmark/test_oaf_backend.py`

**Interfaces:**
- `OafBackendConfig(checkpoint_dir: Path, input_root: Path, image: str = "crux-oaf-tf1:local")`
- `OafBackend.descriptor() -> BackendDescriptor`
- `OafBackend.transcribe(audio: CanonicalAudio) -> NativePrediction`
- `OafBackend.close() -> None`
- `create_backend(config: OafBackendConfig | None = None) -> OafBackend`

- [ ] **Step 1: Write adapter tests with a deterministic fake process factory**

The fake process exposes `.ready`, `.request()`, and `.close()`. Cover exact assertions:

- ready backend ID plus `restored_tensor_count=78` constructs successfully;
- wrong backend ID raises `OafBackendError("worker backend identity mismatch")`;
- restored count 77 raises `OafBackendError("worker restored tensor count mismatch")`;
- one success event becomes `NativeEvent(time_sec=..., native_class_id="midi_38", model_output_bin=17, native_midi_note=38, native_metadata={"upstream_8hit_group_id":"snare"}, confidence=..., velocity_midi=...)`;
- a worker `error` payload raises `OafBackendError` preserving its code;
- two `.transcribe()` calls reuse the same fake process instance;
- audio path outside `input_root` is rejected before `.request()` is called;
- `.close()` twice calls underlying close at most once.

- [ ] **Step 2: Run adapter tests and confirm module is absent**

```bash
uv run pytest tests/benchmark/test_oaf_adapter.py -q
```

- [ ] **Step 3: Build compact descriptor from `model.json` identity**

The descriptor payload is exactly:

```python
{
    "architecture_id": config.architecture_id,
    "backend_id": config.backend_id,
    "descriptor_schema": OAF_DESCRIPTOR_SCHEMA,
    "model_id": config.model_id,
    "native_metadata_schema_id": config.native_metadata_schema_id,
    "native_output_space_id": config.native_output_space_id,
    "prediction_schema": "crux.drum-prediction-events/v2",
    "training_data_map_id": config.training_data_map_id,
    "upstream_source_commit": config.upstream_source_commit,
}
```

- [ ] **Step 4: Implement Docker worker launch and native decoding**

Build one command equivalent to:

```bash
docker run --rm -i --network none \
  -v <checkpoint_dir>:/model:ro \
  -v <input_root>:/input:ro \
  crux-oaf-tf1:local
```

Use `WorkerProcess.start(command)`. Convert the canonical path to an input-root-relative POSIX path. Validate numeric fields/ranges and native identity before constructing `NativePrediction(audio=audio, descriptor=self.descriptor(), events=events)`.

- [ ] **Step 5: Activate registry default configuration**

`create_backend()` without explicit config loads `model.json`, derives verified checkpoint path `artifacts/benchmark/model-cache/sha256/<archive_sha256>`, sets `input_root=Path.cwd()`, and uses image `crux-oaf-tf1:local`.

No environment/config framework is added.

- [ ] **Step 6: Delete old sealed host/runtime backend modules**

```bash
git grep -n "backends\.oaf_tf1\|runtime\.oaf_tf1\.oaf_backend" -- src runtime tests
```

Move any remaining inference code into `model.py`, then delete `src/benchmark/backends/oaf_tf1.py` and `runtime/oaf_tf1/oaf_backend.py`. Replace useful behavior coverage with model/worker/adapter tests and delete seal-only `test_oaf_backend.py` cases.

- [ ] **Step 7: Run combined backend tests**

```bash
uv run pytest \
  tests/benchmark/test_oaf_adapter.py \
  tests/benchmark/test_backend_registry.py \
  tests/benchmark/test_backend_process.py \
  runtime/oaf_tf1/tests/test_model.py \
  runtime/oaf_tf1/tests/test_worker.py -q
```

Expected: PASS without the released checkpoint.

- [ ] **Step 8: Commit**

```bash
git add src/benchmark runtime/oaf_tf1 tests/benchmark
git commit -m "feat: activate reusable OaF backend adapter"
```

---

### Task 7: Replace report-heavy backend CLI paths with `smoke-backend` and one workflow

**Files:**
- Modify: `src/cli/benchmark.py`
- Modify: `tests/test_cli_benchmark.py`
- Add: `.github/workflows/oaf-smoke.yml`
- Add: `tests/benchmark/test_oaf_smoke_workflow.py`
- Delete after migration: `src/benchmark/transcription.py`
- Delete after migration: `src/benchmark/backend_reports.py`
- Delete after migration: `src/benchmark/backend_attestation.py`
- Delete/migrate: `tests/benchmark/test_transcription.py`
- Delete: `tests/benchmark/test_backend_reports.py`

**Interfaces:**
- `crux benchmark smoke-backend --backend oaf`
- Output prediction: `artifacts/benchmark/oaf-smoke/prediction.jsonl`
- One machine-readable stdout summary containing status/identity/count/hash/timing fields.
- `.github/workflows/oaf-smoke.yml`: one manual Ubuntu 24.04 smoke job.

- [ ] **Step 1: Write CLI smoke tests with a fake backend**

Create a fake backend returning one hihat plus one sticks native event. Patch `default_backend_registry()` to return it and patch `time.perf_counter()` to deterministic start/end values. Invoke the CLI against a temporary copy of the canonical WAV fixture and output path. Assert:

- exit code 0 when at least one event maps;
- output reads as prediction artifact v2;
- summary has `mapped_event_count == 1`, `unmapped_event_count == 1`;
- `real_time_factor == elapsed_seconds / (audio_frame_count / sample_rate)`;
- `prediction_sha256` matches output bytes;
- fake backend `.close()` was called once.

Add a test with only `sticks` and require nonzero exit plus no successful prediction publication. Assert Click command registry contains neither `verify-backend` nor `transcribe-one` after implementation.

- [ ] **Step 2: Write workflow structure tests before adding the workflow**

Parse the workflow and assert:

```text
workflow_dispatch has no inputs
exact job keys == {smoke}
runs-on == ubuntu-24.04
permissions do not grant id-token/attestations write
one command contains "docker build"
one command contains "smoke-backend"
one upload-artifact step uploads artifacts/benchmark/oaf-smoke/prediction.jsonl
no step text contains tools.hpa320, seal, candidate, calibration, or measurement
```

- [ ] **Step 3: Run CLI/workflow tests and confirm old command surface fails them**

```bash
uv run pytest tests/test_cli_benchmark.py tests/benchmark/test_oaf_smoke_workflow.py -q
```

- [ ] **Step 4: Implement `smoke-backend` without a report orchestration layer**

Flow is exact:

```text
load canonical fixture as CanonicalAudio
create registry backend
start timer
transcribe once
map through map_oaf_prediction
require mapped_event_count > 0
render v2 bytes
publish immutable output
stop timer
derive duration and RTF
print canonical JSON summary
close backend in finally
```

Default fixture: `tests/fixtures/oaf_tf1_smoke/canonical.wav`. Default checkpoint/cache/image come from the OaF adapter. Default output: `artifacts/benchmark/oaf-smoke/prediction.jsonl`.

Summary keys are exactly:

```text
status
backend_id
upstream_source_commit
checkpoint_archive_sha256
restored_tensor_count
fixture_sha256
mapped_event_count
unmapped_event_count
prediction_path
prediction_sha256
elapsed_seconds
real_time_factor
```

- [ ] **Step 5: Remove old backend verification/transcribe report paths**

Delete `verify-backend` and `transcribe-one` Click commands and their report emitters. When no retained import remains, delete `transcription.py`, `backend_reports.py`, and `backend_attestation.py` plus report-specific tests.

If a generic file helper remains useful, move it to `artifact_io.py`; do not preserve a report module for one helper.

- [ ] **Step 6: Add `.github/workflows/oaf-smoke.yml`**

One `workflow_dispatch` job does checkout, uv setup, `prepare-backend --backend oaf --download`, one Docker build tagged `crux-oaf-tf1:local`, one `smoke-backend --backend oaf`, writes the parsed stdout summary to `$GITHUB_STEP_SUMMARY`, and uploads only the prediction JSONL. Use `contents: read` only.

- [ ] **Step 7: Run CLI/workflow tests**

```bash
uv run pytest tests/test_cli_benchmark.py tests/benchmark/test_oaf_smoke_workflow.py -q
```

Expected: PASS.

- [ ] **Step 8: Commit**

```bash
git add src/cli/benchmark.py src/benchmark .github/workflows/oaf-smoke.yml tests
git commit -m "feat: add minimal OaF smoke workflow"
```

---

### Task 8: Delete the remaining HPA-320 estate and run repository verification

**Files:**
- Delete: `.github/workflows/hpa320-native-bootstrap.yml`
- Delete: `.github/workflows/hpa320-native-candidate.yml`
- Delete: `.github/workflows/hpa320-native-measurement.yml`
- Delete: `tools/hpa320/` entire directory
- Delete: `src/benchmark/backend_lock.py`
- Delete: remaining calibration/attestation/build-context runtime files no longer imported from `runtime/oaf_tf1/`
- Delete: corresponding seal/attestation/schema tests and goldens
- Modify: `docs/drumery-dtx-midi-benchmarking-reference.md` where it describes removed active commands

**Interfaces:**
- Produces no new abstraction; proves the new backend is the only active OaF path.

- [ ] **Step 1: Enumerate all remaining active old-architecture references**

```bash
git grep -n -E \
  'BackendVerification|TensorCoverageCheck|SmokeCheck|SealState|BackendNotSealed|heuristic-onset-v1|backend_lock|backend_reports|backend_attestation|tools\.hpa320|hpa320-native|calibration_(entrypoint|protocol)' \
  -- src runtime tests .github tools
```

Every result must be either removed in this task or moved to the already-defined active module that owns the behavior. Do not create a compatibility shim.

- [ ] **Step 2: Delete the obsolete workflows, tools, locks, calibration/attestation runtime files, tests, and goldens**

Retain only inference-relevant vendored Magenta source, dependencies/wheelhouse, instrumentation patch, smoke fixture, model/config/worker tests, and Tasks 1-7 active modules.

- [ ] **Step 3: Update active benchmark documentation**

Replace current seal/verify instructions with:

```bash
uv run crux benchmark prepare-backend --backend oaf --download
docker build -f runtime/oaf_tf1/Dockerfile -t crux-oaf-tf1:local .
uv run crux benchmark smoke-backend --backend oaf
```

Historical `docs/superpowers/` records may remain unchanged.

- [ ] **Step 4: Prove no active old architecture or prediction-v1 reference remains**

```bash
git grep -n -E \
  'BackendVerification|TensorCoverageCheck|SmokeCheck|SealState|BackendNotSealed|heuristic-onset-v1|backend_lock|backend_reports|backend_attestation|tools\.hpa320|hpa320-native|calibration_(entrypoint|protocol)' \
  -- src runtime tests .github tools

git grep -n 'crux.drum-prediction-events/v1' -- src runtime tests .github
```

Expected: no active matches.

- [ ] **Step 5: Run focused backend suite**

```bash
uv run pytest \
  tests/benchmark/test_backend_contract_acceptance.py \
  tests/benchmark/test_backend_identity.py \
  tests/benchmark/test_backend_registry.py \
  tests/benchmark/test_artifact_io.py \
  tests/benchmark/test_checkpoint_acquisition.py \
  tests/benchmark/test_prediction_mapping.py \
  tests/benchmark/test_prediction_artifact.py \
  tests/benchmark/test_scorer_input.py \
  tests/benchmark/test_backend_process.py \
  tests/benchmark/test_oaf_adapter.py \
  tests/benchmark/test_oaf_smoke_workflow.py \
  runtime/oaf_tf1/tests/test_model.py \
  runtime/oaf_tf1/tests/test_worker.py \
  tests/test_cli_benchmark.py -q
```

Expected: PASS without the real released checkpoint.

- [ ] **Step 6: Run repository validation**

```bash
uv run pytest
uv run ruff check src tests
uv run ruff format --check src tests
uv run pylint --errors-only --disable=E1120,E0401 src
git diff --check main...HEAD
```

Record only a pre-existing baseline failure when the failing bytes are unchanged from `main`; do not widen HPA-423 into unrelated cleanup.

- [ ] **Step 7: Build final runtime image**

```bash
docker build -f runtime/oaf_tf1/Dockerfile -t crux-oaf-tf1:local .
```

Expected: success; image starts the new worker.

- [ ] **Step 8: Commit**

```bash
git add -A
git commit -m "refactor: remove obsolete OaF sealing estate"
```

---

### Task 9: Run and review the authentic released-checkpoint smoke

**Files/outputs:**
- Workflow: `.github/workflows/oaf-smoke.yml`
- Uploaded artifact: `artifacts/benchmark/oaf-smoke/prediction.jsonl`
- Operational evidence: GitHub job summary only

**Interfaces:**
- Consumes the complete Tasks 1-8 backend path.
- Produces one successful reviewed smoke run that unlocks HPA-326.

- [ ] **Step 1: Dispatch `OaF Smoke` manually on the exact implementation commit**

No workflow input is supplied beyond selecting the branch/ref in GitHub's standard UI.

- [ ] **Step 2: Verify summary invariants**

Require:

```text
backend_id = magenta-egmd-tf1-94529798-8hit-v1
upstream_source_commit = 94529798dfbbb14c27ddfd76f23027dc8e2ce185
checkpoint_archive_sha256 = 09765ae0ff19c7d769a3c20e158eba3b9cd279429b02e498b1e911d16f82e2c0
restored_tensor_count = 78
mapped_event_count > 0
prediction_sha256 matches 64 lowercase hexadecimal digits
elapsed_seconds > 0
real_time_factor > 0
```

- [ ] **Step 3: Validate downloaded artifact through production reader**

```bash
uv run python - <<'PY'
from pathlib import Path
from src.benchmark.prediction_artifact import read_prediction_artifact

artifact = read_prediction_artifact(Path("prediction.jsonl").read_bytes())
assert artifact.event_count > 0
assert any(event.common_class is not None for event in artifact.prediction.events)
print(artifact.artifact_sha256)
PY
```

The printed hash must equal the job summary hash.

- [ ] **Step 4: If smoke fails, fix only the new path with a focused regression test**

Do not restore preseal state, host evidence, candidate/measurement workflows, or compatibility adapters. After any fix rerun Task 8 repository validation before redispatch.

- [ ] **Step 5: Record HPA-423 completion and unblock HPA-326**

Post the successful run URL, exact commit SHA, restored tensor count, mapped/unmapped counts, and RTF to Linear. Mark HPA-423 Done only after both the hihat/tom artifact-v2 round-trip tests and authentic smoke pass.

---

## Dependency and Parallelism Notes

Tasks 1, 2, 4, and 5 can proceed without HPA-324 production code. Task 3 is the only hard HPA-324 seam because taxonomy/map policy must have exactly one owner. Task 6 can be developed against `NativePrediction` independently, but HPA-423 is not complete until Task 3 is integrated. Tasks 7-9 require the complete mapping/persistence path.

One implementation branch with task-sized commits is sufficient; do not create child infrastructure merely to parallelize the work.

## Self-Review

- **Spec coverage:** Tasks 1-9 cover every revised HPA-423 acceptance criterion.
- **Codebase drift:** Task 2 extracts the generic helpers now used outside sealing before deleting `backend_publication.py`.
- **HPA-324 ownership:** Task 3 consumes the reviewed prediction map and never defines a duplicate table.
- **Prediction correctness:** common-only hi-hat/tom and unmapped sticks are hard round-trip tests before HPA-326.
- **Runtime:** one persistent sequential worker; no premature parallelism or retry framework.
- **Compatibility:** no seal compatibility or prediction-v1 reader is planned.
- **Security scope:** no new attestation/signing/security-only subsystem is planned.
- **Placeholder scan:** every implementation/test step has concrete inputs, assertions, commands, or exact interfaces; the only ellipsis notation is explicitly identified as Python interface-body syntax rather than an unfinished plan item.
