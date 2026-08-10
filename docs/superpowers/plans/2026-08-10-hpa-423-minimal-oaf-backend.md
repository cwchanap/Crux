# HPA-423 Minimal Reusable OaF Backend Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the unusable sealed OaF path with one reusable persistent OaF backend that verifies the released checkpoint, persists HPA-324-compatible prediction events, and passes one real native smoke inference.

**Architecture:** Keep the existing canonical-audio/native-event domain model and extract the working TensorFlow inference code into `runtime/oaf_tf1/model.py`. A small line-JSON worker loads that model once; a thin host adapter owns the persistent process and returns `NativePrediction`; an OaF-specific mapper consumes HPA-324 taxonomy data and writes prediction artifact v2 with common classes for scoring. Remove seal/attestation/report/heuristic machinery instead of adapting it.

**Tech Stack:** Python 3.12 host tooling/tests, existing pinned Python 3.7/TensorFlow 1 OaF container dependencies, Click CLI, Docker, pytest, Ruff, Pylint, GitHub Actions `ubuntu-24.04`.

## Global Constraints

- Preserve `OAF_BACKEND_ID = "magenta-egmd-tf1-94529798-8hit-v1"`.
- Preserve upstream Magenta revision `94529798dfbbb14c27ddfd76f23027dc8e2ce185`.
- Preserve checkpoint archive SHA-256 `09765ae0ff19c7d769a3c20e158eba3b9cd279429b02e498b1e911d16f82e2c0` and the three published component hashes already committed in the repository.
- Require all 78 inference tensors to restore before the worker reports ready.
- Preserve `CanonicalAudio.source_audio_id`, `source_audio_sha256`, `input_view_id`, and `input_audio_sha256`.
- Preserve OaF `native_class_id="midi_<note>"` and `native_metadata["upstream_8hit_group_id"]` identity.
- HPA-324 owns taxonomy/map data. Do not duplicate a temporary taxonomy or OaF class table inside HPA-423.
- Prediction persistence is a breaking replacement to `crux.drum-prediction-events/v2`; do not add a v1 compatibility reader.
- One persistent worker, one request at a time. No worker pool, queue, retry framework, or distributed execution.
- Do not add OIDC, Sigstore, host attestation, runtime seals, deterministic OCI publication, or security-only policy code.
- Keep TensorFlow imports lazy/outside host module import paths.
- Do not change the pinned TensorFlow/OaF dependency set unless extraction proves a dependency is unused and removing it is independently safe.
- Use the repository validation commands from `CLAUDE.md`: `uv run pytest`, `uv run ruff check src tests`, `uv run ruff format --check src tests`, and error-focused Pylint on changed host modules. Existing repository-wide full-warning Pylint noise is not an HPA-423 blocker.

---

## File Structure After HPA-423

The intended active backend surface is deliberately small:

```text
src/benchmark/
  artifact_io.py                 # generic retained file read/publish helpers
  backend_identity.py            # compact model descriptor + canonical JSON/hash helpers
  backend_process.py             # small synchronous persistent line-JSON subprocess wrapper
  backend_registry.py            # static {"oaf": factory} mapping
  checkpoint_acquisition.py      # model.json-driven checkpoint download/verify/cache
  prediction_mapping.py          # OaF native -> HPA-324 mapped prediction conversion
  prediction_artifact.py         # prediction JSONL v2 read/render/publish
  scorer_input.py                # v2 common-class -> BenchmarkEvent
  backends/
    base.py                       # CanonicalAudio, NativeEvent, NativePrediction, Protocol
    oaf.py                        # thin host adapter
runtime/oaf_tf1/
  model.json                     # single OaF identity/checkpoint data file
  model.py                       # TF graph/checkpoint/inference/native event engine
  worker.py                      # persistent sequential stdin/stdout worker
  Dockerfile                     # minimal runtime/test image
  requirements.lock              # existing pinned runtime deps
  wheelhouse/                    # existing pinned wheels
  vendor/                        # existing Magenta source
  patches/capture-emitted-frame.patch
.github/workflows/
  oaf-smoke.yml                  # one manual native smoke workflow
```

HPA-324 supplies `src/benchmark/taxonomy.py`; HPA-423 consumes it but does not own that file's policy data.

---

### Task 1: Cut the shared backend contract and registry free from sealing

**Files:**
- Modify: `src/benchmark/backends/base.py`
- Modify: `src/benchmark/backends/__init__.py`
- Modify: `src/benchmark/backend_identity.py`
- Modify: `src/benchmark/backend_registry.py`
- Modify: `tests/benchmark/test_backend_registry.py`
- Modify: `tests/benchmark/test_backend_identity.py`
- Modify: `tests/benchmark/test_backend_contract_acceptance.py`
- Delete: `tests/benchmark/schema_goldens/crux.heuristic-backend-descriptor-v1.json`
- Delete: `tests/benchmark/schema_goldens/crux.transcription-backend-descriptor-v1.json`

**Interfaces:**
- Produces: `CanonicalAudio`, `NativeEvent`, `NativePrediction`, `TranscriptionBackend` with no verification lifecycle.
- Produces: `BackendDescriptor(payload: Mapping[str, str], sha256: str)` and a compact OaF descriptor builder/validator.
- Produces: `default_backend_registry()` whose only supported selection is short key `"oaf"`.
- Consumes: no HPA-324 code yet.

- [ ] **Step 1: Write contract tests that describe the reduced domain model**

Update `tests/benchmark/test_backend_contract_acceptance.py` so the public protocol is exactly:

```python
from inspect import getmembers, isfunction

from src.benchmark.backends import CanonicalAudio, NativePrediction, TranscriptionBackend


def test_transcription_backend_has_only_runtime_transcription_methods():
    methods = {
        name
        for name, value in getmembers(TranscriptionBackend)
        if isfunction(value) and not name.startswith("_")
    }
    assert methods == {"descriptor", "transcribe", "close"}


def test_canonical_audio_keeps_source_and_input_identity_fields():
    assert {
        "source_audio_id",
        "source_audio_sha256",
        "input_view_id",
        "input_audio_sha256",
    } <= set(CanonicalAudio.__dataclass_fields__)


def test_native_prediction_has_no_seal_or_lock_fields():
    assert set(NativePrediction.__dataclass_fields__) == {"audio", "descriptor", "events"}
```

Also assert removed names are no longer exported from `src.benchmark.backends`:

```python
import src.benchmark.backends as backends

for name in ("BackendVerification", "TensorCoverageCheck", "SmokeCheck"):
    assert not hasattr(backends, name)
```

- [ ] **Step 2: Write registry tests for one explicit OaF factory and no heuristic/seal state**

Use a monkeypatched factory so no TensorFlow/Docker code is imported:

```python
def test_registry_selects_only_oaf(monkeypatch):
    sentinel = object()
    monkeypatch.setattr("src.benchmark.backend_registry._create_oaf_backend", lambda: sentinel)
    registry = default_backend_registry()
    assert registry.default_backend_id == "oaf"
    assert registry.create("oaf") is sentinel


def test_registry_rejects_removed_heuristic():
    registry = default_backend_registry()
    with pytest.raises(BackendUnavailable):
        registry.create("heuristic-onset-v1")
```

Assert `SealState`, `BackendNotSealed`, `BackendIntegrityUnavailable`, `HEURISTIC_BACKEND_ID`, and `LEGACY_TF2_BACKEND_ID` are absent.

- [ ] **Step 3: Run the focused tests and confirm they fail against the sealed contract**

Run:

```bash
uv run pytest \
  tests/benchmark/test_backend_contract_acceptance.py \
  tests/benchmark/test_backend_registry.py \
  tests/benchmark/test_backend_identity.py -q
```

Expected: failures mention verification/lock fields, `seal_state`, heuristic registration, or the old descriptor shape.

- [ ] **Step 4: Reduce `backends/base.py` and exports**

Keep the existing `CanonicalAudio` and `NativeEvent` fields unchanged. Replace `NativePrediction` with:

```python
@dataclass(frozen=True)
class NativePrediction:
    audio: CanonicalAudio
    descriptor: BackendDescriptor
    events: tuple[NativeEvent, ...]


class TranscriptionBackend(Protocol):
    def descriptor(self) -> BackendDescriptor: ...
    def transcribe(self, audio: CanonicalAudio) -> NativePrediction: ...
    def close(self) -> None: ...
```

Remove verification/report/publication types from this module and `backends/__init__.py`.

- [ ] **Step 5: Replace the frozen OaF descriptor v1 with compact v2 identity**

In `backend_identity.py` define:

```python
OAF_DESCRIPTOR_SCHEMA = "crux.transcription-backend-descriptor/v2"
OAF_DESCRIPTOR_KEYS = frozenset(
    {
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
)
```

Keep `BackendDescriptor`, `canonical_json_bytes`, `strict_json_loads`, `sha256_hex`, `require_sha256`, and `quantize_six` because active benchmark code already uses them.

Delete heuristic descriptor constants and all lock/runtime-image/protocol descriptor fields. Add:

```python
def build_oaf_descriptor(payload: Mapping[str, str]) -> BackendDescriptor:
    return build_descriptor(payload, OAF_DESCRIPTOR_KEYS, OAF_DESCRIPTOR_SCHEMA)
```

Validation must still assert the stable OaF backend/model/source identities but must not require any seal hash.

- [ ] **Step 6: Replace the registry with a small static factory mapping**

Target shape:

```python
OFFICIAL_BACKEND_ID = OAF_BACKEND_ID


class BackendUnavailable(ValueError):
    pass


def _create_oaf_backend() -> TranscriptionBackend:
    from src.benchmark.backends.oaf import create_backend
    return create_backend()


@dataclass(frozen=True)
class BackendRegistry:
    default_backend_id: str
    factories: Mapping[str, Callable[[], TranscriptionBackend]]

    def create(self, backend_id: str | None) -> TranscriptionBackend:
        key = self.default_backend_id if backend_id is None else backend_id
        factory = self.factories.get(key)
        if factory is None:
            raise BackendUnavailable(f"unknown backend: {key}")
        return factory()


def default_backend_registry() -> BackendRegistry:
    return BackendRegistry(default_backend_id="oaf", factories={"oaf": _create_oaf_backend})
```

The temporary import of `src.benchmark.backends.oaf` may fail until Task 6 if a real registry creation is attempted; focused registry tests monkeypatch the lazy factory until then.

- [ ] **Step 7: Remove descriptor goldens that encode the abandoned exact seal shape**

Delete the two old descriptor golden files listed above and remove their registrations/assertions from `tests/benchmark/test_schema_goldens.py` if present. Do not create a new exact-field golden solely to replace them; `test_backend_identity.py` directly covers the compact descriptor.

- [ ] **Step 8: Run the focused contract tests**

Run:

```bash
uv run pytest \
  tests/benchmark/test_backend_contract_acceptance.py \
  tests/benchmark/test_backend_registry.py \
  tests/benchmark/test_backend_identity.py \
  tests/benchmark/test_schema_goldens.py -q
```

Expected: PASS.

- [ ] **Step 9: Commit the contract cut**

```bash
git add src/benchmark/backends src/benchmark/backend_identity.py \
  src/benchmark/backend_registry.py tests/benchmark

git commit -m "refactor: remove OaF seal lifecycle from backend contract"
```

---

### Task 2: Extract generic artifact I/O and simplify checkpoint preparation

**Files:**
- Create: `src/benchmark/artifact_io.py`
- Create: `runtime/oaf_tf1/model.json`
- Modify: `src/benchmark/input_view.py`
- Modify: `src/benchmark/checkpoint_acquisition.py`
- Modify: `src/benchmark/prediction_artifact.py` (imports only; v2 behavior comes in Task 3)
- Modify: `src/cli/benchmark.py` (`prepare-backend` only)
- Modify: `tests/benchmark/test_checkpoint_acquisition.py`
- Modify: `tests/test_cli_benchmark.py`
- Add: `tests/benchmark/test_artifact_io.py`
- Delete after migration: `src/benchmark/backend_prepare.py`
- Delete after migration: `src/benchmark/backend_publication.py`

**Interfaces:**
- Produces: `read_regular_file_no_follow(path: Path) -> bytes`.
- Produces: `publish_immutable_file(path: Path, content: bytes) -> PublishedArtifact` where `PublishedArtifact` is defined in `artifact_io.py`.
- Produces: `OafModelConfig`/`load_oaf_model_config()` without importing TensorFlow.
- Produces: `prepare_oaf_checkpoint(config, cache_root, *, download, archive_path=None) -> Path`.

- [ ] **Step 1: Write tests for only the generic artifact helpers that retained code needs**

In `tests/benchmark/test_artifact_io.py` cover:

```python
def test_read_regular_file_no_follow_reads_normal_file(tmp_path): ...
def test_read_regular_file_no_follow_rejects_symlink(tmp_path): ...
def test_publish_immutable_file_writes_once_and_returns_sha(tmp_path): ...
def test_publish_immutable_file_reuses_identical_existing_bytes(tmp_path): ...
def test_publish_immutable_file_rejects_different_existing_bytes(tmp_path): ...
```

Keep the implementation small: file-level `O_NOFOLLOW`, parent creation, exclusive create, and content hash verification. Do not port directory transaction/attestation rollback machinery.

- [ ] **Step 2: Write model-config tests before moving checkpoint identity**

Add to `tests/benchmark/test_checkpoint_acquisition.py`:

```python
def test_model_config_loads_released_oaf_identity():
    config = load_oaf_model_config(Path("runtime/oaf_tf1/model.json"))
    assert config.backend_id == "magenta-egmd-tf1-94529798-8hit-v1"
    assert config.upstream_source_commit == "94529798dfbbb14c27ddfd76f23027dc8e2ce185"
    assert config.checkpoint.archive_sha256 == "09765ae0ff19c7d769a3c20e158eba3b9cd279429b02e498b1e911d16f82e2c0"
    assert len(config.checkpoint.components) == 3
```

Also test malformed SHA, missing field, wrong backend ID, and duplicate component names.

- [ ] **Step 3: Write checkpoint preparation tests around behavior, not evidence publication**

Use a tiny synthetic ZIP with injected config identities so tests do not download the real archive. Cover:

```python
def test_prepare_checkpoint_extracts_verified_components(tmp_path): ...
def test_prepare_checkpoint_reuses_verified_cache(tmp_path): ...
def test_prepare_checkpoint_rejects_archive_hash_mismatch(tmp_path): ...
def test_prepare_checkpoint_rejects_component_hash_mismatch(tmp_path): ...
def test_prepare_checkpoint_verify_only_reports_missing_cache(tmp_path): ...
```

The test-facing implementation may accept a loaded `OafModelConfig` so fixture hashes can be generated locally.

- [ ] **Step 4: Run new tests and confirm they fail because the neutral APIs do not exist**

```bash
uv run pytest \
  tests/benchmark/test_artifact_io.py \
  tests/benchmark/test_checkpoint_acquisition.py -q
```

Expected: import/API failures.

- [ ] **Step 5: Create `artifact_io.py` and migrate active imports**

Move only the minimum generic behavior. Define:

```python
@dataclass(frozen=True)
class PublishedArtifact:
    path: Path
    sha256: str


def read_regular_file_no_follow(path: Path) -> bytes: ...


def publish_immutable_file(path: Path, content: bytes) -> PublishedArtifact: ...
```

Update `input_view.py`, `checkpoint_acquisition.py`, and `prediction_artifact.py` to import from `artifact_io.py`.

If `DirectoryAnchor` survives only because old transcription/report code still uses it, do not copy it here yet; that old caller is removed in Task 7/8. If `input_view.py` genuinely needs an anchored read after its active call sites are inspected, implement one small `DirectoryAnchor` context type in `artifact_io.py` and cover it directly rather than porting the full publication module.

- [ ] **Step 6: Add `runtime/oaf_tf1/model.json` as the single OaF identity source**

Use the exact stable values from the revised spec, including the three component hashes. The file is canonical ordinary JSON; it does not need a schema-golden publication framework.

Implement config dataclasses in `checkpoint_acquisition.py` for host-side preparation:

```python
@dataclass(frozen=True)
class CheckpointComponent:
    name: str
    sha256: str


@dataclass(frozen=True)
class OafCheckpointConfig:
    url: str
    archive_name: str
    archive_sha256: str
    components: tuple[CheckpointComponent, ...]


@dataclass(frozen=True)
class OafModelConfig:
    backend_id: str
    model_id: str
    architecture_id: str
    upstream_source_commit: str
    training_data_map_id: str
    native_output_space_id: str
    native_metadata_schema_id: str
    checkpoint: OafCheckpointConfig


def load_oaf_model_config(path: Path = Path("runtime/oaf_tf1/model.json")) -> OafModelConfig: ...
```

Task 4 may move this loader into `runtime/oaf_tf1/model.py` after that module exists; until then, keep it host-side and dependency-free. There must remain only one JSON data source.

- [ ] **Step 7: Collapse checkpoint acquisition to one direct function**

Reuse the existing safe ZIP/download/hash code but remove lock/evidence/publication inputs. Target API:

```python
def prepare_oaf_checkpoint(
    config: OafModelConfig,
    cache_root: Path,
    *,
    download: bool,
    archive_path: Path | None = None,
) -> Path:
    """Return the verified checkpoint directory or raise CheckpointAcquisitionError."""
```

Store verified components under:

```text
<cache_root>/sha256/<archive_sha256>/
```

The directory key may use the archive SHA rather than the old derived model-artifact-set hash because no downstream contract needs the latter.

Do not render/persist acquisition evidence.

- [ ] **Step 8: Simplify the `prepare-backend` CLI surface**

Replace the old options with:

```python
@benchmark.command("prepare-backend")
@click.option("--backend", type=click.Choice(["oaf"]), default="oaf", show_default=True)
@click.option("--download", is_flag=True)
@click.option("--archive", type=click.Path(path_type=Path, dir_okay=False), default=None)
@click.option(
    "--cache-root",
    type=click.Path(path_type=Path, file_okay=False),
    default=Path("artifacts/benchmark/model-cache"),
    show_default=True,
)
def prepare_backend_command(backend: str, download: bool, archive: Path | None, cache_root: Path): ...
```

Reject `--download` + `--archive`. Print one JSON summary containing `backend`, `checkpoint_path`, and `status`. Remove `--acquisition-request`, `--evidence-output`, and `--backend-lock`.

- [ ] **Step 9: Delete the seal-oriented container modules once imports are migrated**

Before deletion run:

```bash
git grep -n "backend_publication\|backend_prepare" -- src runtime tests .github tools
```

Expected retained matches before deletion: only files being migrated in this task plus old HPA-320/report code scheduled for later deletion.

Delete `backend_prepare.py` now when no active retained caller remains. Delete `backend_publication.py` only after all retained imports point to `artifact_io.py`; old seal-only callers may be deleted in the same commit if required to make the import graph valid.

- [ ] **Step 10: Run focused tests and CLI help regression**

```bash
uv run pytest \
  tests/benchmark/test_artifact_io.py \
  tests/benchmark/test_checkpoint_acquisition.py \
  tests/test_cli_benchmark.py -q
uv run crux benchmark prepare-backend --help
```

Expected: tests PASS; help has no seal/evidence/lock options.

- [ ] **Step 11: Commit artifact/checkpoint simplification**

```bash
git add src/benchmark runtime/oaf_tf1/model.json src/cli/benchmark.py tests

git commit -m "refactor: simplify OaF checkpoint preparation"
```

---

### Task 3: Make prediction persistence scoreable with the HPA-324 map

**Precondition:** `src/benchmark/taxonomy.py` from HPA-324 is present with `PredictionMap`, `ClassMapping`, `OAF_PREDICTION_MAP`, and `OAF_PREDICTION_MAP_ID`. If HPA-324 has not merged, bring its reviewed production commit into the implementation branch; do not create duplicate constants in HPA-423.

**Files:**
- Create: `src/benchmark/prediction_mapping.py`
- Modify: `src/benchmark/prediction_artifact.py`
- Modify: `src/benchmark/scorer_input.py`
- Modify: `tests/benchmark/test_prediction_artifact.py`
- Add: `tests/benchmark/test_prediction_mapping.py`
- Add: `tests/benchmark/test_scorer_input.py`
- Replace: `tests/benchmark/schema_goldens/crux.drum-prediction-events-v1.jsonl` with `tests/benchmark/schema_goldens/crux.drum-prediction-events-v2.jsonl` only if `test_schema_goldens.py` still provides value for the active prediction contract.

**Interfaces:**
- Consumes: HPA-324 `OAF_PREDICTION_MAP` keyed by `upstream_8hit_group_id`.
- Produces: `MappedPredictionEvent`, `MappedPrediction`, `PredictionMappingDiagnostics`, `map_oaf_prediction()`.
- Produces: prediction artifact v2 read/render/publish.
- Produces: `read_scorer_events(content: bytes) -> tuple[BenchmarkEvent, ...]` using common class.

- [ ] **Step 1: Write OaF mapping tests with real native event shapes**

In `tests/benchmark/test_prediction_mapping.py` create a helper:

```python
def oaf_event(group_id: str, midi_note: int) -> NativeEvent:
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

Cover:

```python
def test_hihat_maps_common_only():
    result = map_oaf_prediction(native_prediction(oaf_event("hihat", 46)))
    event = result.prediction.events[0]
    assert event.canonical_class is None
    assert event.common_class == "hihat"
    assert event.mapping_status == "mapped"
    assert event.prediction_map_version == OAF_PREDICTION_MAP_ID
    assert event.native.native_class_id == "midi_46"


def test_toms_map_common_only(): ...  # common_class == "tom"
def test_sticks_are_preserved_as_unmapped(): ...
def test_unknown_or_null_group_is_preserved_as_unmapped(): ...
```

- [ ] **Step 2: Write artifact-v2 round-trip tests before changing the serializer**

Assert the event JSON contains:

```json
{
  "canonical_class": null,
  "common_class": "hihat",
  "mapping_status": "mapped",
  "prediction_map_version": "crux.prediction-map/oaf-egmd-8hit-v1"
}
```

and that read-back preserves `midi_46`, native MIDI note, output bin, confidence, velocity, and metadata.

Add the tom equivalent and an unmapped `sticks` event where both class fields are null but the map version remains non-null.

- [ ] **Step 3: Write scorer-input tests**

```python
def test_scorer_input_uses_persisted_common_class():
    events = read_scorer_events(rendered_hihat_prediction_v2())
    assert len(events) == 1
    assert events[0].canonical_class == "hihat"
    assert events[0].source == "prediction"


def test_scorer_input_omits_unmapped_events_without_backend_branch():
    events = read_scorer_events(rendered_mapped_plus_sticks_prediction_v2())
    assert [event.canonical_class for event in events] == ["snare"]
```

Use `source_audio_id` as `BenchmarkEvent.chart_id`; include `input_view_id`, native identity, and detailed canonical class in metadata.

- [ ] **Step 4: Run the focused tests and confirm v1/not-applied behavior fails them**

```bash
uv run pytest \
  tests/benchmark/test_prediction_mapping.py \
  tests/benchmark/test_prediction_artifact.py \
  tests/benchmark/test_scorer_input.py -q
```

Expected: missing mapping module/common-class field and `CanonicalMappingRequired` failures.

- [ ] **Step 5: Implement the mapped prediction domain in `prediction_mapping.py`**

Use the HPA-324 types rather than strings where possible:

```python
@dataclass(frozen=True)
class MappedPredictionEvent:
    native: NativeEvent
    canonical_class: DetailedDrumClass | None
    common_class: CommonDrumClass | None
    mapping_status: Literal["mapped", "unmapped"]
    prediction_map_version: str


@dataclass(frozen=True)
class MappedPrediction:
    audio: CanonicalAudio
    descriptor: BackendDescriptor
    events: tuple[MappedPredictionEvent, ...]


@dataclass(frozen=True)
class PredictionMappingDiagnostics:
    unmapped: Mapping[str, int]


@dataclass(frozen=True)
class PredictionMappingResult:
    prediction: MappedPrediction
    diagnostics: PredictionMappingDiagnostics
```

Implement:

```python
def map_oaf_prediction(
    prediction: NativePrediction,
    prediction_map: PredictionMap = OAF_PREDICTION_MAP,
) -> PredictionMappingResult:
    ...
```

Validate `prediction.descriptor.payload["model_id"] == prediction_map.model_id` and native output-space identity before applying the table. The semantic key is only `event.native_metadata.get("upstream_8hit_group_id")`.

- [ ] **Step 6: Advance `prediction_artifact.py` to v2 and remove seal header fields**

Set:

```python
PREDICTION_SCHEMA = "crux.drum-prediction-events/v2"
```

The v2 header keeps:

```text
schema
record_type
artifact_role
backend_descriptor
backend_descriptor_sha256
architecture_id
model_id
native_metadata_schema_id
native_output_space_id
source_audio_id
source_audio_sha256
input_view_id
input_audio_sha256
byte_length
sample_rate
channel_count
sample_width_bytes
audio_frame_count
```

Remove backend/runtime/parameter/model-artifact lock hashes and special heuristic nullability.

The event exact-key set adds `common_class` and accepts only the mapped/unmapped semantics from the revised spec. `render_prediction_artifact()` now accepts `MappedPrediction`, and `read_prediction_artifact()` reconstructs `MappedPrediction`.

- [ ] **Step 7: Replace `scorer_input.py` stub**

Implement:

```python
def read_scorer_events(content: bytes) -> tuple[BenchmarkEvent, ...]:
    artifact = read_prediction_artifact(content)
    source_audio_id = artifact.prediction.audio.source_audio_id
    result = []
    for event in artifact.prediction.events:
        if event.mapping_status != "mapped" or event.common_class is None:
            continue
        result.append(
            BenchmarkEvent(
                chart_id=source_audio_id,
                time_sec=event.native.time_sec,
                canonical_class=event.common_class,
                source="prediction",
                metadata={
                    "input_view_id": artifact.prediction.audio.input_view_id,
                    "native_class_id": event.native.native_class_id,
                    "native_midi_note": event.native.native_midi_note,
                    "model_output_bin": event.native.model_output_bin,
                    "canonical_class": event.canonical_class,
                    "prediction_map_version": event.prediction_map_version,
                },
            )
        )
    return tuple(sorted(result))
```

Delete `CanonicalMappingRequired`.

- [ ] **Step 8: Replace the active prediction golden only if retained schema-golden coverage still uses it**

If `tests/benchmark/test_schema_goldens.py` validates active artifact schemas, replace the v1 file with a compact v2 golden containing at least one common-only mapped event and one unmapped event. Do not keep the v1 golden.

- [ ] **Step 9: Run mapping/artifact/scorer tests**

```bash
uv run pytest \
  tests/benchmark/test_prediction_mapping.py \
  tests/benchmark/test_prediction_artifact.py \
  tests/benchmark/test_scorer_input.py \
  tests/benchmark/test_schema_goldens.py -q
```

Expected: PASS, including hi-hat/tom round-trip.

- [ ] **Step 10: Commit the scoreable prediction seam**

```bash
git add src/benchmark/prediction_mapping.py src/benchmark/prediction_artifact.py \
  src/benchmark/scorer_input.py tests/benchmark

git commit -m "feat: persist mapped OaF prediction classes"
```

---

### Task 4: Extract the reusable TensorFlow OaF model engine

**Files:**
- Create: `runtime/oaf_tf1/model.py`
- Modify: `runtime/oaf_tf1/Dockerfile`
- Modify: `runtime/oaf_tf1/apply_instrumentation_patch.py` only if imports/paths must change
- Keep: `runtime/oaf_tf1/patches/capture-emitted-frame.patch`
- Keep: `runtime/oaf_tf1/vendor/`
- Add: `runtime/oaf_tf1/tests/test_model.py`
- Migrate/remove relevant tests from: `runtime/oaf_tf1/tests/test_tensor_coverage.py`
- Extraction source, delete after Task 6: `runtime/oaf_tf1/oaf_backend.py`

**Interfaces:**
- Produces: dependency-light `load_model_config()` reading `model.json`.
- Produces: `OafModel.load(checkpoint_dir: Path) -> OafModel`.
- Produces: `OafModel.restored_tensor_count` and `OafModel.transcribe(path: Path) -> tuple[OafNativeEvent, ...]`.
- Consumes: existing vendored Magenta source/instrumentation patch and verified checkpoint directory.

- [ ] **Step 1: Write config-loading tests in the runtime package**

`runtime/oaf_tf1/tests/test_model.py`:

```python
def test_load_model_config_matches_released_identity(): ...
def test_load_model_config_rejects_bad_checkpoint_hash(tmp_path): ...
```

Move the source of the config dataclasses/loader from Task 2 into `runtime/oaf_tf1/model.py`, then make host `checkpoint_acquisition.py` import that dependency-light loader. Keep TensorFlow imports inside `OafModel.load()` so importing config remains cheap.

- [ ] **Step 2: Write native event conversion tests from representative existing OaF outputs**

Use data already covered by the old `oaf_backend.py` tests. Assert an emitted snare event becomes:

```python
OafNativeEvent(
    time_sec=expected_time,
    native_class_id="midi_38",
    model_output_bin=17,
    native_midi_note=38,
    upstream_8hit_group_id="snare",
    confidence=expected_confidence,
    velocity_midi=expected_velocity,
)
```

Do the same for at least one hi-hat/tom group so the group metadata used by Task 3 remains proven.

- [ ] **Step 3: Write a missing-tensor test at the new model boundary**

Extract the old tensor inventory/restoration check into a helper that can be unit tested with fake restored names:

```python
def validate_required_tensors(required: Collection[str], restored: Collection[str]) -> int: ...
```

Assert it returns `78` for the complete known inventory and raises `OafModelError("required tensors not restored: ...")` when one is missing.

- [ ] **Step 4: Run runtime model tests and confirm the new module is absent**

```bash
uv run pytest runtime/oaf_tf1/tests/test_model.py -q
```

Expected: import failure before implementation.

- [ ] **Step 5: Move inference-only code from `oaf_backend.py` into `model.py`**

Define:

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


class OafModel:
    @classmethod
    def load(cls, checkpoint_dir: Path, config: OafModelConfig | None = None) -> "OafModel": ...

    @property
    def restored_tensor_count(self) -> int: ...

    def transcribe(self, audio_path: Path) -> tuple[OafNativeEvent, ...]: ...
```

Copy/move only graph construction, feature extraction, checkpoint restoration, emitted-frame processing, post-processing, and native event conversion. Do not copy backend locks, seal evidence, source-attestation validation, host fingerprints, calibration requests, or publication logic.

- [ ] **Step 6: Simplify the Dockerfile around inference**

Keep the pinned base image, wheelhouse, requirements lock, vendored Magenta source, and instrumentation patch. Remove:

- UID/GID policy unless the ordinary runtime actually needs it;
- copies of `.github/workflows/hpa320-*`;
- copies of `tools/hpa320/`;
- source/runner/build-context attestation manifests;
- calibration-specific stages/modes.

Runtime should end with a normal entrypoint added in Task 5, for example:

```dockerfile
ENTRYPOINT ["/opt/crux/venv/bin/python", "-m", "runtime.oaf_tf1.worker"]
```

If module layout inside the image differs, use the equivalent direct `/opt/crux/runtime/worker.py` invocation consistently in Dockerfile and tests.

- [ ] **Step 7: Run model tests and a container test build**

```bash
uv run pytest runtime/oaf_tf1/tests/test_model.py -q
docker build --target test -f runtime/oaf_tf1/Dockerfile -t crux-oaf-tf1-test .
```

Do not require the real released checkpoint in this step.

- [ ] **Step 8: Commit the model extraction**

```bash
git add runtime/oaf_tf1 src/benchmark/checkpoint_acquisition.py tests/benchmark/test_checkpoint_acquisition.py

git commit -m "refactor: extract reusable OaF model engine"
```

---

### Task 5: Replace the sealed protocol with one persistent sequential worker/process

**Files:**
- Create: `runtime/oaf_tf1/worker.py`
- Rewrite: `src/benchmark/backend_process.py`
- Add: `runtime/oaf_tf1/tests/test_worker.py`
- Rewrite: `tests/benchmark/test_backend_process.py`
- Delete after migration: `runtime/oaf_tf1/entrypoint.py`
- Delete after migration: `runtime/oaf_tf1/protocol.py`

**Interfaces:**
- Worker stdin request: `{"id": str, "audio_path": str}`.
- Worker ready stdout: `{"type":"ready","backend_id":...,"restored_tensor_count":78}`.
- Worker success stdout: `{"id":...,"events":[...]}`.
- Worker error stdout: `{"id":...,"error":{"code":str,"message":str}}`.
- Host: `WorkerProcess.start(command)`, `.ready`, `.request(payload)`, `.close()`.

- [ ] **Step 1: Write worker tests with a fake model factory**

Inject model loading into `worker.main()` so tests do not import TensorFlow:

```python
def test_worker_loads_model_once_and_serves_two_requests(): ...
def test_worker_returns_direct_error_for_bad_json(): ...
def test_worker_returns_model_error_for_failed_inference(): ...
def test_worker_exits_zero_on_stdin_eof(): ...
```

The two-request test must assert the fake model factory was called exactly once.

- [ ] **Step 2: Write synchronous process-wrapper tests using a tiny fake Python worker script**

Cover:

```python
def test_process_reads_ready_and_serves_two_requests(tmp_path): ...
def test_process_rejects_malformed_ready(tmp_path): ...
def test_process_rejects_malformed_response(tmp_path): ...
def test_process_reports_early_exit(tmp_path): ...
def test_process_close_finishes_clean_worker(tmp_path): ...
```

No threads are required. Use `selectors.DefaultSelector` or equivalent to bound startup/request reads without the old reader-state machine.

- [ ] **Step 3: Run focused tests and confirm the simple APIs do not exist**

```bash
uv run pytest \
  runtime/oaf_tf1/tests/test_worker.py \
  tests/benchmark/test_backend_process.py -q
```

Expected: failures against old protocol/process APIs.

- [ ] **Step 4: Implement `worker.py` as a small line loop**

Keep JSON encoding ordinary/deterministic enough for tests but do not require canonical-byte negotiation. Pseudocode:

```python
def serve(model: OafModel, stdin: TextIO, stdout: TextIO) -> int:
    write_json_line(stdout, ready_payload(model))
    for line in stdin:
        request_id = None
        try:
            request = json.loads(line)
            request_id, path = validate_request(request)
            events = model.transcribe(Path("/input") / path)
            write_json_line(stdout, success_payload(request_id, events))
        except ExpectedWorkerError as error:
            write_json_line(stdout, error_payload(request_id, error))
    return 0
```

Reject absolute/`..` input paths. The model is loaded once before `serve()`.

- [ ] **Step 5: Rewrite `backend_process.py` around blocking line JSON with bounded reads**

Target public surface:

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
    ) -> "WorkerProcess": ...

    @property
    def ready(self) -> Mapping[str, object]: ...

    def request(self, payload: Mapping[str, object]) -> Mapping[str, object]: ...

    def close(self) -> None: ...
```

Use one `threading.Lock` only to prevent concurrent requests. Do not port host evidence, stderr redaction, canonical protocol schemas, process-state enums, or background reader threads.

- [ ] **Step 6: Delete old protocol/entrypoint modules once no retained imports remain**

```bash
git grep -n "runtime\.oaf_tf1\.protocol\|runtime\.oaf_tf1\.entrypoint\|from protocol" -- src runtime tests
```

Expected: no retained matches before deletion.

Delete `entrypoint.py` and `protocol.py`; move any still-needed pure WAV/native-event logic into `model.py` or `worker.py` with direct tests.

- [ ] **Step 7: Run worker/process tests**

```bash
uv run pytest \
  runtime/oaf_tf1/tests/test_worker.py \
  tests/benchmark/test_backend_process.py -q
```

Expected: PASS.

- [ ] **Step 8: Commit the persistent worker simplification**

```bash
git add runtime/oaf_tf1 src/benchmark/backend_process.py \
  tests/benchmark/test_backend_process.py

git commit -m "refactor: simplify persistent OaF worker protocol"
```

---

### Task 6: Add the thin OaF host adapter and activate it in the registry

**Files:**
- Create: `src/benchmark/backends/oaf.py`
- Add: `tests/benchmark/test_oaf_adapter.py`
- Modify: `src/benchmark/backend_registry.py`
- Modify: `tests/benchmark/test_backend_registry.py`
- Delete after extraction: `src/benchmark/backends/oaf_tf1.py`
- Delete after extraction: `runtime/oaf_tf1/oaf_backend.py`
- Delete/migrate obsolete: `tests/benchmark/test_oaf_backend.py`

**Interfaces:**
- Consumes: `WorkerProcess`, verified checkpoint directory, canonical input root, compact OaF descriptor.
- Produces: `OafBackend(OafBackendConfig)` implementing `TranscriptionBackend`.
- Produces: `create_backend() -> OafBackend` for registry use.

- [ ] **Step 1: Write adapter ready/success/error tests with an injected process factory**

Define test configuration:

```python
@dataclass(frozen=True)
class OafBackendConfig:
    checkpoint_dir: Path
    input_root: Path
    image: str = "crux-oaf-tf1:local"
```

Tests:

```python
def test_adapter_accepts_ready_identity_and_78_tensors(tmp_path): ...
def test_adapter_rejects_wrong_ready_backend_id(tmp_path): ...
def test_adapter_rejects_incomplete_tensor_restore(tmp_path): ...
def test_adapter_converts_worker_event_to_native_event(tmp_path): ...
def test_adapter_surfaces_worker_error(tmp_path): ...
def test_adapter_reuses_one_worker_for_two_transcriptions(tmp_path): ...
def test_adapter_rejects_audio_outside_input_root(tmp_path): ...
```

- [ ] **Step 2: Run adapter tests and confirm the new adapter is absent**

```bash
uv run pytest tests/benchmark/test_oaf_adapter.py -q
```

Expected: import failure.

- [ ] **Step 3: Implement descriptor creation from `model.json` identity**

Create a helper in `oaf.py` or `backend_identity.py` that builds the compact descriptor:

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

No Docker image digest or protocol version enters the descriptor.

- [ ] **Step 4: Implement the Docker-backed adapter**

`_start_worker()` builds one command equivalent to:

```bash
docker run --rm -i --network none \
  -v <checkpoint_dir>:/model:ro \
  -v <input_root>:/input:ro \
  crux-oaf-tf1:local
```

`transcribe()` converts `CanonicalAudio.path` to an input-root-relative POSIX path, sends it, validates each returned event, and creates `NativePrediction(audio=audio, descriptor=self.descriptor(), events=...)`.

Keep `close()` idempotent.

- [ ] **Step 5: Activate the real adapter factory in `backend_registry.py`**

`default_backend_registry().create("oaf")` must now import/create the real adapter. Make the default adapter configuration use:

```text
checkpoint_dir = artifacts/benchmark/model-cache/sha256/<archive_sha256>
input_root = current working directory
image = crux-oaf-tf1:local
```

Prefer allowing `create_backend(config=...)` in direct callers/tests rather than adding environment/config frameworks.

- [ ] **Step 6: Delete the old sealed host/runtime backend modules after import search**

Run:

```bash
git grep -n "backends\.oaf_tf1\|runtime\.oaf_tf1\.oaf_backend" -- src runtime tests
```

Move any last inference logic into `model.py`, then delete:

```text
src/benchmark/backends/oaf_tf1.py
runtime/oaf_tf1/oaf_backend.py
```

Replace useful old behavior tests with the focused model/worker/adapter tests; delete seal-only `test_oaf_backend.py` cases.

- [ ] **Step 7: Run adapter/registry/model/worker tests together**

```bash
uv run pytest \
  tests/benchmark/test_oaf_adapter.py \
  tests/benchmark/test_backend_registry.py \
  tests/benchmark/test_backend_process.py \
  runtime/oaf_tf1/tests/test_model.py \
  runtime/oaf_tf1/tests/test_worker.py -q
```

Expected: PASS without a real checkpoint.

- [ ] **Step 8: Commit the active OaF adapter switch**

```bash
git add src/benchmark runtime/oaf_tf1 tests/benchmark

git commit -m "feat: activate reusable OaF backend adapter"
```

---

### Task 7: Replace report-heavy backend CLI paths with one smoke command and workflow

**Files:**
- Modify: `src/cli/benchmark.py`
- Modify: `tests/test_cli_benchmark.py`
- Add: `.github/workflows/oaf-smoke.yml`
- Add: `tests/benchmark/test_oaf_smoke_workflow.py`
- Delete after CLI migration: `src/benchmark/transcription.py`
- Delete after CLI migration: `src/benchmark/backend_reports.py`
- Delete after CLI migration: `src/benchmark/backend_attestation.py`
- Delete/migrate: `tests/benchmark/test_transcription.py`
- Delete: `tests/benchmark/test_backend_reports.py`

**Interfaces:**
- Produces: `crux benchmark smoke-backend --backend oaf`.
- Produces one mapped prediction v2 artifact and one machine-readable summary.
- Produces one manual `oaf-smoke.yml` workflow.

- [ ] **Step 1: Write CLI tests for the target smoke flow with a fake backend**

Patch registry/backend creation and mapping so the CLI test stays fast. Cover:

```python
def test_smoke_backend_writes_prediction_v2_and_summary(tmp_path, monkeypatch): ...
def test_smoke_backend_fails_when_no_mapped_events(tmp_path, monkeypatch): ...
def test_smoke_backend_closes_backend_on_failure(tmp_path, monkeypatch): ...
def test_removed_verify_backend_command_is_not_registered(): ...
def test_removed_transcribe_one_command_is_not_registered(): ...
```

The summary must include:

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

- [ ] **Step 2: Write workflow structure tests before creating the workflow**

Parse `.github/workflows/oaf-smoke.yml` with the same YAML strategy used by existing workflow tests. Assert:

- event is only `workflow_dispatch` with no custom inputs;
- one job named `smoke` on `ubuntu-24.04`;
- no `id-token` or `attestations` write permission;
- exactly one Docker build command;
- exactly one `smoke-backend` invocation;
- exactly one artifact upload of the prediction path;
- no `tools.hpa320`, seal, calibration, candidate, or measurement command.

- [ ] **Step 3: Run the CLI/workflow tests and confirm they fail against old commands**

```bash
uv run pytest tests/test_cli_benchmark.py tests/benchmark/test_oaf_smoke_workflow.py -q
```

Expected: missing smoke command/workflow and still-present verify/transcribe commands.

- [ ] **Step 4: Implement `smoke-backend` directly in `src/cli/benchmark.py`**

Do not create another report orchestration layer. The command can use one small helper/dataclass inside the CLI module if needed.

Flow:

```python
start = time.perf_counter()
audio = load_direct_audio(fixture_path, source_audio_id="oaf-smoke", input_view_id="full_mix")
backend = default_backend_registry().create("oaf")
try:
    native = backend.transcribe(audio)
    mapped = map_oaf_prediction(native)
    require mapped mapped-event count > 0
    content = render_prediction_artifact(mapped.prediction)
    publish_immutable_file(output_path, content)
finally:
    backend.close()
elapsed = time.perf_counter() - start
duration = audio.audio_frame_count / audio.sample_rate
rtf = elapsed / duration
```

Use fixed defaults from the spec and allow only `--backend oaf` plus optional `--output`/`--fixture` if tests/development genuinely need override paths. Do not expose seal/report knobs.

- [ ] **Step 5: Remove old backend CLI/report orchestration**

Delete `verify-backend` and `transcribe-one` registrations and their report helper functions. Once no retained imports remain, delete `transcription.py`, `backend_reports.py`, and `backend_attestation.py` plus report-specific tests.

If another active non-HPA-320 module imports a generic helper from one of those files, move only that helper to its natural domain module and test it; do not preserve the report module for one utility.

- [ ] **Step 6: Add `.github/workflows/oaf-smoke.yml`**

The job sequence is:

```text
checkout
setup uv/python
uv sync/install needed host extras
crux benchmark prepare-backend --backend oaf --download
docker build -f runtime/oaf_tf1/Dockerfile -t crux-oaf-tf1:local .
crux benchmark smoke-backend --backend oaf
append parsed summary fields to $GITHUB_STEP_SUMMARY
upload artifacts/benchmark/oaf-smoke/prediction.jsonl
```

Use normal `contents: read` permissions only.

- [ ] **Step 7: Run CLI/workflow tests**

```bash
uv run pytest tests/test_cli_benchmark.py tests/benchmark/test_oaf_smoke_workflow.py -q
```

Expected: PASS.

- [ ] **Step 8: Commit the smoke entrypoint**

```bash
git add src/cli/benchmark.py src/benchmark .github/workflows/oaf-smoke.yml tests

git commit -m "feat: add minimal OaF smoke workflow"
```

---

### Task 8: Delete the remaining HPA-320 seal estate and run full verification

**Files:**
- Delete: `.github/workflows/hpa320-native-bootstrap.yml`
- Delete: `.github/workflows/hpa320-native-candidate.yml`
- Delete: `.github/workflows/hpa320-native-measurement.yml`
- Delete: `tools/hpa320/` entire directory
- Delete: `src/benchmark/backend_lock.py`
- Delete: remaining calibration/attestation/build-context runtime files under `runtime/oaf_tf1/` that are no longer imported
- Delete: corresponding seal/attestation/schema tests and goldens
- Modify: `docs/drumery-dtx-midi-benchmarking-reference.md` only where it describes removed active commands/lifecycle

**Interfaces:**
- Consumes: all active APIs established in Tasks 1-7.
- Produces: no new abstraction; this task proves the old architecture is unreachable and the new one is the sole active path.

- [ ] **Step 1: Enumerate remaining active seal references before deleting**

Run:

```bash
git grep -n -E \
  'BackendVerification|TensorCoverageCheck|SmokeCheck|SealState|BackendNotSealed|heuristic-onset-v1|backend_lock|backend_reports|backend_attestation|tools\.hpa320|hpa320-native|calibration_(entrypoint|protocol)' \
  -- src runtime tests .github tools
```

Classify every result as either:

- file scheduled for deletion in this task; or
- accidental retained dependency that must be removed/moved before deletion.

Do not allow a third category of compatibility shim.

- [ ] **Step 2: Delete the obsolete workflow/tool/runtime/test files mechanically**

Delete the three workflows and `tools/hpa320/` directory. Delete `backend_lock.py` and all tests whose only subject is locks, seal publication, host attestation, native candidate/bootstrap/measurement evidence, deterministic OCI archives, or removed schema goldens.

Retain only inference-relevant vendored Magenta source, requirements/wheelhouse, instrumentation patch, model/config/worker tests, smoke fixture, and new active modules.

- [ ] **Step 3: Update active user-facing benchmark documentation**

In `docs/drumery-dtx-midi-benchmarking-reference.md`, replace any active instructions for `verify-backend`, seal publication, or HPA-320 native workflows with:

```bash
uv run crux benchmark prepare-backend --backend oaf --download
docker build -f runtime/oaf_tf1/Dockerfile -t crux-oaf-tf1:local .
uv run crux benchmark smoke-backend --backend oaf
```

Historical superpowers design/plan docs may remain as historical records; do not rewrite old design history simply to remove string matches.

- [ ] **Step 4: Prove no active code/test/workflow references remain**

Re-run:

```bash
git grep -n -E \
  'BackendVerification|TensorCoverageCheck|SmokeCheck|SealState|BackendNotSealed|heuristic-onset-v1|backend_lock|backend_reports|backend_attestation|tools\.hpa320|hpa320-native|calibration_(entrypoint|protocol)' \
  -- src runtime tests .github tools
```

Expected: no matches, except an intentionally retained word in a test explicitly asserting absence. Prefer removing even those if they add no value.

Also run:

```bash
git grep -n 'crux.drum-prediction-events/v1' -- src runtime tests .github
```

Expected: no active v1 references.

- [ ] **Step 5: Run the focused backend suite first**

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

Expected: PASS without the real checkpoint.

- [ ] **Step 6: Run repository validation**

```bash
uv run pytest
uv run ruff check src tests
uv run ruff format --check src tests
uv run pylint --errors-only --disable=E1120,E0401 src

git diff --check main...HEAD
```

Expected: all commands exit 0, except any pre-existing explicitly documented repository baseline that is byte-identical to `main`; record such a baseline rather than broadening HPA-423 to fix unrelated lint.

- [ ] **Step 7: Build the final runtime image locally without a real inference**

```bash
docker build -f runtime/oaf_tf1/Dockerfile -t crux-oaf-tf1:local .
```

Expected: build succeeds and the image entrypoint is the new worker.

- [ ] **Step 8: Commit final cleanup**

```bash
git add -A

git commit -m "refactor: remove obsolete OaF sealing estate"
```

---

### Task 9: Run and review the one authentic released-checkpoint smoke

**Files:**
- No production code should be required if Tasks 1-8 are correct.
- Output artifact: `artifacts/benchmark/oaf-smoke/prediction.jsonl` in the GitHub Actions run.
- Operational evidence: GitHub job summary only.

**Interfaces:**
- Consumes: `.github/workflows/oaf-smoke.yml` and the complete active backend path.
- Produces: one reviewed successful smoke run that unlocks HPA-326.

- [ ] **Step 1: Push the implementation branch and dispatch `OaF Smoke` manually**

Dispatch the workflow with no custom inputs against the exact implementation commit under review.

- [ ] **Step 2: Verify the job summary values**

Require:

```text
backend_id = magenta-egmd-tf1-94529798-8hit-v1
upstream_source_commit = 94529798dfbbb14c27ddfd76f23027dc8e2ce185
checkpoint_archive_sha256 = 09765ae0ff19c7d769a3c20e158eba3b9cd279429b02e498b1e911d16f82e2c0
restored_tensor_count = 78
mapped_event_count > 0
prediction_sha256 = 64 lowercase hex digits
elapsed_seconds > 0
real_time_factor > 0
```

- [ ] **Step 3: Download/read the uploaded prediction artifact and validate it through production code**

Run locally against the artifact bytes:

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

Confirm its hash matches the job summary.

- [ ] **Step 4: If the smoke fails, repair only the new path and repeat the normal review/test cycle**

Do not restore `preseal`, old host evidence, candidate/measurement workflows, or compatibility adapters. Any code fix starts with a focused failing test in the task-owned module, then repeats Task 8 repository validation before redispatch.

- [ ] **Step 5: Record completion on HPA-423 and unblock HPA-326**

Post the successful workflow run URL, commit SHA, mapped/unmapped event counts, restored tensor count, and RTF to Linear HPA-423. Mark HPA-423 Done only after the real OaF-shaped hi-hat/tom v2 round-trip test and the authentic smoke both pass.

---

## Implementation Dependency / Parallelism Notes

Tasks 1, 2, 4, and 5 can proceed without HPA-324 production code. Task 3 is the only hard merge seam with HPA-324 because taxonomy/map policy must have exactly one owner. Task 6 can be developed against `NativePrediction` independently but should not be considered HPA-423 complete until Task 3 is integrated. Tasks 7-9 require the complete mapping/persistence path.

Do not create child infrastructure merely to parallelize this plan. A single implementation branch with task-sized commits is sufficient.

## Plan Self-Review Checklist

- Spec coverage: each revised HPA-423 acceptance criterion is assigned to Tasks 1-9.
- HPA-324 boundary: taxonomy/map data is consumed, never duplicated.
- Codebase drift: `backend_publication.py` generic helpers are explicitly extracted before deletion.
- Prediction correctness: common-only hi-hat/tom and unmapped `sticks` round-trips are hard tests before HPA-326.
- Performance: one persistent sequential worker; RTF measured but no premature parallelism.
- Compatibility: no seal or prediction-v1 compatibility path is planned.
- Security scope: no new attestation/signing/security-only subsystem is planned.
- Placeholder scan: no `TBD`, `TODO`, or unspecified implementation step remains.
