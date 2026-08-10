# HPA-423 Minimal Reusable OaF Backend Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the unusable sealed OaF prototype with one reusable persistent OaF backend, preserve the authentic released model behavior through an exact smoke oracle, and make predictions directly scoreable by the HPA-324 common taxonomy.

**Architecture:** Prune dead outer seal machinery first, add the new checkpoint/model/worker pieces alongside the still-importable old path, then perform one atomic breaking cutover of shared backend types, registry, adapter, mapping, prediction artifact v2, scorer bridge, Docker runtime, and remaining old consumers. The smoke command then validates the real checkpoint; the first successful result establishes a compact exact native-event oracle and a second run must match it before HPA-423 is complete.

**Tech Stack:** Python 3.12 host tooling/tests, existing pinned Python 3.7/TensorFlow 1 runtime dependencies, Click, Docker, pytest, Ruff, Pylint, GitHub Actions `ubuntu-24.04`.

## Global Constraints

- Preserve backend ID `magenta-egmd-tf1-94529798-8hit-v1`.
- Preserve checkpoint model ID `magenta-egmd-ckpt-569400-v1` as a distinct identity.
- Preserve upstream revision `94529798dfbbb14c27ddfd76f23027dc8e2ce185`.
- Preserve checkpoint archive SHA-256 `09765ae0ff19c7d769a3c20e158eba3b9cd279429b02e498b1e911d16f82e2c0`.
- Preserve component SHA-256 values:
  - data: `6312cd6f8e29ee6f21f4a763822d554f46393e0e336812ba4a8cb324d2e8b0b5`;
  - index: `475ca21993e102d52e2e6c7066b598b9b9e9a097bfb2fa6b770d40eedaa6139a`;
  - meta: `e36105b27c17f2367292682df825b8ad16eb887ccad465bbb5ddf955ef75d422`.
- Require all 78 inference tensors before the new worker reports ready.
- Preserve `CanonicalAudio.source_audio_id`, `source_audio_sha256`, `input_view_id`, and `input_audio_sha256`.
- Preserve native OaF `midi_<note>` identity and `native_metadata["upstream_8hit_group_id"]`.
- HPA-324 owns `PredictionMap.backend_id`, `native_output_space_id`, and the OaF class table.
- OaF map application validates descriptor `backend_id` plus `native_output_space_id`; it never uses checkpoint `model_id` as the map selector.
- Prediction artifact v2 is a breaking replacement; do not add a v1 reader.
- Prediction-side `BenchmarkEvent.canonical_class` means the **common scoring class**. Detailed prediction class is diagnostic metadata only.
- One persistent sequential worker. No pool, queue, retry framework, or distributed execution.
- Keep TensorFlow imports lazy.
- Keep `assert_no_reachable_stochastic_ops()` in model startup.
- Final `model.json` sets `max_input_audio_frames` explicitly to `null`; HPA-423 does not invent an unmeasured full-song cap.
- Every committed task must leave ordinary imports and `uv run pytest` green.
- New model/worker tests must live under `tests/runtime/oaf_tf1/` so normal CI executes them.

---

## Final Active File Map

### Create / retain

- `src/benchmark/artifact_io.py`
- `src/benchmark/backend_identity.py`
- `src/benchmark/backend_registry.py`
- `src/benchmark/checkpoint_acquisition.py`
- `src/benchmark/mapping.py`
- `src/benchmark/prediction_artifact.py`
- `src/benchmark/scorer_input.py`
- `src/benchmark/oaf_smoke_oracle.py`
- `src/benchmark/worker_process.py`
- `src/benchmark/backends/base.py`
- `src/benchmark/backends/oaf.py`
- `runtime/oaf_tf1/model.json`
- `runtime/oaf_tf1/model.py`
- `runtime/oaf_tf1/worker.py`
- `runtime/oaf_tf1/Dockerfile`
- `runtime/oaf_tf1/requirements.lock`
- `runtime/oaf_tf1/wheelhouse/runtime/`
- `runtime/oaf_tf1/vendor/`
- `runtime/oaf_tf1/patches/capture-emitted-frame.patch`
- `tests/runtime/oaf_tf1/test_model.py`
- `tests/runtime/oaf_tf1/test_worker.py`
- `.github/workflows/oaf-smoke.yml`
- `tests/fixtures/oaf_tf1_smoke/canonical.wav`
- `tests/fixtures/oaf_tf1_smoke/smoke-oracle.json` after Task F

### HPA-324 dependency

- `src/benchmark/taxonomy.py`

### Final removals

- HPA-320 native workflows;
- `tools/hpa320/`;
- backend/runtime locks, reports, attestation, seal preparation/publication modules;
- old sealed OaF host/runtime adapter/process/protocol;
- old checkpoint request/config files;
- Docker test target and its test lock/wheelhouse;
- HPA-320 schema goldens except none are retained under their old schemas;
- prediction artifact v1.

---

## Task A: Prune Independently Dead HPA-320 Outer Estate

**Goal:** Remove the workflows/tools/build-context surface that the replacement does not need without changing shared backend types or the old extraction sources yet.

**Files:**
- Delete: `.github/workflows/hpa320-native-bootstrap.yml`
- Delete: `.github/workflows/hpa320-native-candidate.yml`
- Delete: `.github/workflows/hpa320-native-measurement.yml`
- Delete: `tools/hpa320/` entire directory
- Delete: tests whose sole subject is those workflows/tools/attestations/build-context/OCI publication/smoke generation
- Modify: `runtime/oaf_tf1/Dockerfile` to remove copies of deleted workflows/tools only; keep the old runtime entrypoint until Task D
- Modify: `tests/benchmark/schema_goldens/manifest.json` to remove rows whose validators live only under deleted `tools.hpa320.*`
- Delete the matching tool-only schema-golden files

**Preserve for extraction until Task D:**
- `runtime/oaf_tf1/oaf_backend.py`
- `runtime/oaf_tf1/entrypoint.py`
- `runtime/oaf_tf1/protocol.py`
- `src/benchmark/backends/oaf_tf1.py`
- `src/benchmark/backend_process.py`
- `src/benchmark/backend_lock.py`
- `src/benchmark/backend_prepare.py`
- `src/benchmark/backend_publication.py`
- `src/benchmark/backend_reports.py`
- `src/benchmark/backend_attestation.py`
- `src/benchmark/transcription.py`

- [ ] **Step 1: Prove the deletion boundary before editing**

Run:

```bash
git grep -n -E \
  'tools\.hpa320|tools/hpa320|hpa320-native-(bootstrap|candidate|measurement)' \
  -- src runtime tests .github tools pyproject.toml
```

The only retained production/runtime match allowed after this task is a literal historical path in an extraction source. No retained import may point at `tools.hpa320`.

- [ ] **Step 2: Delete the three workflows and `tools/hpa320/`**

Delete their direct tests, including native workflow, attestation, OCI/build-context, candidate/calibration, and smoke-generator tests whose production owner is also deleted.

Do not delete `tests/fixtures/oaf_tf1_smoke/canonical.wav`.

- [ ] **Step 3: Remove tool-only schema-golden rows**

From `tests/benchmark/schema_goldens/manifest.json`, remove every row whose only validator is one of:

```text
tools.hpa320.seal_oaf_backend
tools.hpa320.oaf_system_packages
tools.hpa320.oaf_build_context
tools.hpa320.oaf_host_attestation
```

Delete each referenced golden file in the same change.

Do **not** yet remove lock/report/prediction/runner rows owned by retained old modules; Task D replaces the complete final manifest atomically.

- [ ] **Step 4: Remove deleted tool/workflow copies from the Dockerfile**

Remove `COPY tools/hpa320/...` and HPA-320 workflow copy lines. Keep the old final `ENTRYPOINT` in Task A so the extraction source remains intact until the new worker exists.

- [ ] **Step 5: Run the green-commit gate**

```bash
uv run pytest
uv run ruff check src tests
uv run ruff format --check src tests
git diff --check
```

Expected: PASS. If a collection/import error names a deleted tool, remove the obsolete test/validator that owned that tool rather than adding a compatibility shim.

- [ ] **Step 6: Commit**

```bash
git add -A
git commit -m "refactor: prune obsolete OaF seal workflows and tools"
```

---

## Task B: Add Neutral Artifact I/O and Model-Driven Checkpoint Preparation

**Goal:** Establish the new model/checkpoint source and neutral file helpers while leaving old HPA-320 APIs temporarily available for the old extraction path.

**Files:**
- Create: `src/benchmark/artifact_io.py`
- Create: `runtime/oaf_tf1/model.json`
- Create: `runtime/oaf_tf1/model.py` with configuration types/loader only
- Extend: `src/benchmark/checkpoint_acquisition.py`
- Modify: `src/cli/benchmark.py` `prepare-backend` command only
- Add: `tests/benchmark/test_artifact_io.py`
- Add/modify: `tests/benchmark/test_checkpoint_acquisition.py`
- Modify: `tests/test_cli_benchmark.py`

**Do not yet change:**
- `publish_prediction_artifact()` anchor/role signature;
- `input_view.py` anchor parameters;
- old checkpoint request/evidence functions imported by `backend_prepare.py`.

Those move at Task D when their live callers disappear.

### Interfaces

```python
@dataclass(frozen=True)
class PublishedArtifact:
    path: Path
    sha256: str


def read_regular_file_no_follow(path: Path) -> bytes: ...

def publish_immutable_file(path: Path, content: bytes) -> PublishedArtifact: ...
```

```python
@dataclass(frozen=True)
class OafModelConfig:
    backend_id: str
    model_id: str
    architecture_id: str
    upstream_source_commit: str
    training_data_map_id: str
    native_output_space_id: str
    native_metadata_schema_id: str
    max_input_audio_frames: int | None
    checkpoint: OafCheckpointConfig
```

```python
def load_model_config(path: Path = Path("runtime/oaf_tf1/model.json")) -> OafModelConfig: ...

def prepare_oaf_checkpoint(
    config: OafModelConfig,
    cache_root: Path,
    *,
    download: bool,
    archive_path: Path | None = None,
) -> Path: ...
```

- [ ] **Step 1: Write artifact-I/O tests**

```python
def test_read_regular_file_no_follow_rejects_symlink(tmp_path):
    target = tmp_path / "target"
    target.write_bytes(b"abc")
    link = tmp_path / "link"
    link.symlink_to(target)
    with pytest.raises(OSError):
        read_regular_file_no_follow(link)


def test_publish_immutable_file_reuses_identical_bytes(tmp_path):
    path = tmp_path / "artifact.json"
    content = b'{"ok":true}\n'
    first = publish_immutable_file(path, content)
    second = publish_immutable_file(path, content)
    assert first == second
    assert first.path == path
    assert first.sha256 == hashlib.sha256(content).hexdigest()


def test_publish_immutable_file_rejects_different_existing_bytes(tmp_path):
    path = tmp_path / "artifact.json"
    publish_immutable_file(path, b"one")
    with pytest.raises(ArtifactPublicationError):
        publish_immutable_file(path, b"two")
```

- [ ] **Step 2: Write exact model-config tests**

Require:

```python
config = load_model_config()
assert config.backend_id == "magenta-egmd-tf1-94529798-8hit-v1"
assert config.model_id == "magenta-egmd-ckpt-569400-v1"
assert config.upstream_source_commit == "94529798dfbbb14c27ddfd76f23027dc8e2ce185"
assert config.max_input_audio_frames is None
assert config.checkpoint.archive_sha256 == "09765ae0ff19c7d769a3c20e158eba3b9cd279429b02e498b1e911d16f82e2c0"
```

Require the exact three component names and hashes from Global Constraints. Copy the config to `tmp_path`, replace one hash with `"bad"`, and require `OafModelConfigError` naming that field.

- [ ] **Step 3: Write synthetic-ZIP checkpoint tests**

Build an in-test ZIP with the three expected member names and test:

```text
local archive -> verified cache succeeds
verify-only existing cache -> succeeds
missing verify-only cache -> CheckpointAcquisitionError("checkpoint cache is missing")
changed archive hash -> fails
changed component hash -> fails
returned path -> <cache_root>/sha256/<archive_sha256>
```

- [ ] **Step 4: Implement `artifact_io.py`**

Port only no-follow regular read and immutable-file create/reuse mechanics. Do not port `DirectoryAnchor`, directory transaction rollback, attestation, role metadata, or seal publication.

- [ ] **Step 5: Add `model.json` and the dependency-light loader**

Use exactly the design-spec payload, including:

```json
"max_input_audio_frames": null
```

`model.py` must import only Python standard library modules at module import time in Task B.

- [ ] **Step 6: Extend `checkpoint_acquisition.py` with the new final API**

Reuse current member/hash parsing and useful ZIP/download mechanics from `backend_prepare.py`. Keep old request/evidence functions temporarily because the old adapter still imports them; mark them for deletion in Task D. Do not create another module.

- [ ] **Step 7: Switch only `prepare-backend` to the new API**

Final option surface:

```text
crux benchmark prepare-backend
  --backend oaf
  --download
  --archive PATH
  --cache-root DIRECTORY
```

Defaults:

```text
backend = oaf
cache-root = artifacts/benchmark/model-cache
```

`--download` and `--archive` are mutually exclusive. Remove acquisition-request/evidence/backend-lock CLI options.

Success stdout contains exactly:

```json
{"backend":"oaf","checkpoint_path":"...","status":"ready"}
```

- [ ] **Step 8: Run validation**

```bash
uv run pytest tests/benchmark/test_artifact_io.py \
  tests/benchmark/test_checkpoint_acquisition.py tests/test_cli_benchmark.py -q
uv run pytest
uv run ruff check src tests
uv run ruff format --check src tests
```

Expected: PASS while the old extraction path still imports its old request/evidence API.

- [ ] **Step 9: Commit**

```bash
git add src/benchmark/artifact_io.py src/benchmark/checkpoint_acquisition.py \
  src/cli/benchmark.py runtime/oaf_tf1/model.json runtime/oaf_tf1/model.py tests
git commit -m "refactor: add model-driven OaF checkpoint path"
```

---

## Task C: Extract the Model Engine and New Persistent Worker Alongside the Old Path

**Goal:** Build and test the risky inference extraction before deleting the source implementation.

**Files:**
- Expand: `runtime/oaf_tf1/model.py`
- Create: `runtime/oaf_tf1/worker.py`
- Create: `src/benchmark/worker_process.py`
- Create: `tests/runtime/oaf_tf1/test_model.py`
- Create: `tests/runtime/oaf_tf1/test_worker.py`
- Create: `tests/benchmark/test_worker_process.py`
- Keep unchanged until Task D: old runtime/backend process modules and final Docker entrypoint

### Model interfaces

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


class OafModel:
    @classmethod
    def load(cls, checkpoint_dir: Path, config: OafModelConfig | None = None) -> "OafModel": ...
    @property
    def restored_tensor_count(self) -> int: ...
    def transcribe(self, audio_path: Path) -> tuple[OafNativeEvent, ...]: ...
```

### Worker protocol

```text
ready   {"type":"ready","backend_id":"magenta-egmd-tf1-94529798-8hit-v1","restored_tensor_count":78}
request {"id":"request-1","audio_path":"song.wav"}
success {"id":"request-1","events":[...]}
error   {"id":"request-1","error":{"code":"inference_failed","message":"inference failed"}}
```

- [ ] **Step 1: Write model import and 78-tensor tests**

```python
def test_model_import_does_not_load_tensorflow():
    sys.modules.pop("tensorflow", None)
    importlib.import_module("runtime.oaf_tf1.model")
    assert "tensorflow" not in sys.modules
```

Migrate the existing required inventory fixture. A complete restored set returns 78; removing one known required tensor raises `OafModelError` naming that tensor.

- [ ] **Step 2: Preserve the stochastic-graph gate**

Port `assert_no_reachable_stochastic_ops()` unchanged in responsibility. Unit-test a small fake operation graph where a reachable operation type is stochastic and require `OafModelError`; a deterministic graph passes.

Call this gate during real `OafModel.load()` after prediction fetches are built.

- [ ] **Step 3: Write native-conversion regressions**

Using extracted pure helpers, require:

```text
snare -> midi_38 / bin 17 / MIDI 38 / group snare
hihat -> midi_46 / bin 25 / MIDI 46 / group hihat
tom -> midi_48 / bin 27 / MIDI 48 / group toms
```

Require time, confidence, and velocity from the representative input capture to reproduce exactly.

- [ ] **Step 4: Extract only inference behavior into `model.py`**

Move graph construction, feature extraction, checkpoint restoration, tensor validation, emitted-frame handling, post-processing, and native event conversion. Do not move lock/seal/attestation/calibration/publication code.

- [ ] **Step 5: Write worker tests with an injected fake model factory**

Feed two request lines and EOF through `io.StringIO`. Require:

```text
model factory called once
ready is first line
two response IDs match requests
both audio paths are served in order
EOF returns 0
```

Malformed JSON returns `invalid_request`; injected `OafModelError` returns `inference_failed`.

- [ ] **Step 6: Implement `worker.py`**

Load one model, emit ready, serve one request at a time, flush after every response. Reject absolute paths and `..` paths. Keep no protocol version/signature/resume/pool machinery.

- [ ] **Step 7: Write/implement `WorkerProcess` tests**

Generate small Python fake-worker scripts in `tmp_path` and require:

```text
valid ready + two requests -> success
malformed ready -> WorkerProcessError("worker ready response is invalid")
early exit -> WorkerProcessError("worker exited before ready")
wrong response id -> WorkerProcessError("worker response id mismatch")
close -> child exits
```

`WorkerProcess` uses one request lock and simple line timeout handling.

- [ ] **Step 8: Run the extraction gate under normal repository CI paths**

```bash
uv run pytest tests/runtime/oaf_tf1 tests/benchmark/test_worker_process.py -q
uv run pytest
uv run ruff check runtime/oaf_tf1/model.py runtime/oaf_tf1/worker.py \
  src/benchmark/worker_process.py tests/runtime/oaf_tf1 tests/benchmark/test_worker_process.py
```

Expected: PASS. Do not rely on `runtime/oaf_tf1/tests/` or a Docker test stage for new coverage.

- [ ] **Step 9: Commit**

```bash
git add runtime/oaf_tf1/model.py runtime/oaf_tf1/worker.py \
  src/benchmark/worker_process.py tests/runtime/oaf_tf1 tests/benchmark/test_worker_process.py
git commit -m "refactor: extract OaF model and persistent worker"
```

---

## Task D: Atomic Backend, Mapping, Persistence, and Runtime Cutover

**Goal:** Move all incompatible shared pieces together so the commit remains green.

**Precondition:** HPA-324 production taxonomy is present with `PredictionMap.backend_id`, `native_output_space_id`, `OAF_PREDICTION_MAP`, and `OAF_PREDICTION_MAP_ID`.

### Create/modify

- Modify: `src/benchmark/backends/base.py`
- Modify: `src/benchmark/backends/__init__.py`
- Modify: `src/benchmark/backend_identity.py`
- Modify: `src/benchmark/backend_registry.py`
- Create: `src/benchmark/backends/oaf.py`
- Modify: `src/benchmark/mapping.py`
- Rewrite: `src/benchmark/prediction_artifact.py`
- Rewrite: `src/benchmark/scorer_input.py`
- Modify: `src/benchmark/input_view.py`
- Modify: `runtime/oaf_tf1/Dockerfile`
- Modify: `tests/benchmark/schema_goldens/manifest.json`
- Add: `tests/benchmark/schema_goldens/crux.drum-prediction-events-v2.jsonl`
- Add/modify: contract, adapter, mapping, artifact, scorer, CLI/schema tests

### Delete

- `src/benchmark/backends/oaf_tf1.py`
- `src/benchmark/backend_process.py`
- `src/benchmark/backend_lock.py`
- `src/benchmark/backend_prepare.py`
- `src/benchmark/backend_publication.py`
- `src/benchmark/backend_reports.py`
- `src/benchmark/backend_attestation.py`
- `src/benchmark/transcription.py`
- `runtime/oaf_tf1/oaf_backend.py`
- `runtime/oaf_tf1/entrypoint.py`
- `runtime/oaf_tf1/protocol.py`
- `runtime/oaf_tf1/calibration_entrypoint.py`
- `runtime/oaf_tf1/calibration_protocol.py`
- `runtime/oaf_tf1/requirements-test.lock`
- `runtime/oaf_tf1/wheelhouse/test/`
- `runtime/oaf_tf1/tests/` old runtime-local tests after useful cases are migrated
- `config/benchmark/backends/magenta-egmd-tf1-94529798-8hit-v1.checkpoint-acquisition-request.json`
- `config/benchmark/backends/magenta-egmd-tf1-94529798-8hit-v1.calibration-bootstrap-request.json`
- old seal/lock/report/attestation/runner/checkpoint-request/evidence schema goldens and obsolete tests

### Final backend contract

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

- [ ] **Step 1: Write post-cutover backend/registry/descriptor tests**

Require `NativePrediction` fields exactly `{audio, descriptor, events}` and no public `verify`, `BackendVerification`, `TensorCoverageCheck`, or `SmokeCheck`.

Descriptor keys are exactly:

```text
architecture_id
backend_id
descriptor_schema
model_id
native_metadata_schema_id
native_output_space_id
prediction_schema
training_data_map_id
upstream_source_commit
```

Require descriptor v2 and prediction schema v2.

Registry:

```python
registry = default_backend_registry()
assert registry.default_backend_id == "oaf"
assert set(registry.factories) == {"oaf"}
with pytest.raises(BackendUnavailable, match="unknown backend"):
    registry.create("heuristic-onset-v1")
```

- [ ] **Step 2: Put OaF mapping into existing `mapping.py`**

Move the persistence-side types into `prediction_artifact.py`:

```python
@dataclass(frozen=True)
class MappedPredictionEvent:
    native: NativeEvent
    canonical_class: str | None
    common_class: str | None
    mapping_status: Literal["mapped", "unmapped"]
    prediction_map_version: str


@dataclass(frozen=True)
class MappedPrediction:
    audio: CanonicalAudio
    descriptor: BackendDescriptor
    events: tuple[MappedPredictionEvent, ...]
```

Extend existing `mapping.py`:

```python
def map_oaf_prediction(
    prediction: NativePrediction,
    prediction_map: PredictionMap = OAF_PREDICTION_MAP,
) -> tuple[MappedPrediction, MappingDiagnostics]: ...
```

Validate exactly:

```python
prediction.descriptor.payload["backend_id"] == prediction_map.backend_id
prediction.descriptor.payload["native_output_space_id"] == prediction_map.native_output_space_id
```

Reuse `MappingDiagnostics`; do not create `PredictionMappingDiagnostics` or `PredictionMappingResult`.

Test hi-hat, toms, snare, sticks, unknown/null group, and the backend/native-output mismatch cases.

- [ ] **Step 3: Write and implement the thin OaF adapter**

Fake-process tests require:

```text
backend id + restored count 78 -> accepted
wrong backend id -> rejected
restored count 77 -> rejected
snare worker event -> exact NativeEvent
worker error -> direct OafBackendError
2 transcribes -> same WorkerProcess
outside-input-root path -> rejected
close twice -> underlying close once
```

Launch command mounts checkpoint `/model:ro`, input root `/input:ro`, uses `--network none`, and runs image `crux-oaf-tf1:local`.

- [ ] **Step 4: Advance prediction artifact to v2**

Keep JSONL header/event/terminal. Remove seal-only header fields. Add `common_class` and enforce:

```text
mapped => map version non-null, common class non-null, detailed may be null
unmapped => map version non-null, common/detailed null
```

Round-trip a real-shaped hi-hat, tom, snare, and sticks event while preserving every native field.

`publish_prediction_artifact()` now uses:

```python
publish_immutable_file(path, content)
```

with no `role=` and no anchor.

- [ ] **Step 5: Remove `DirectoryAnchor` from retained input/prediction APIs**

Remove `anchor` from:

```text
input_view.load_direct_audio
input_view.load_derived_audio
input_view._load_manifest
prediction_artifact.publish_prediction_artifact
```

All old callers that passed anchors are deleted in this Task D cutover. `artifact_io.read_regular_file_no_follow()` remains path-only.

- [ ] **Step 6: Replace `scorer_input.py` stub and lock common-class semantics**

For each mapped event:

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

Name the regression:

```text
test_scorer_bridge_uses_common_class_in_benchmark_event_canonical_class
```

Unmapped events never become scoring events.

- [ ] **Step 7: Switch Docker runtime deliberately**

Final Docker environment/entrypoint:

```dockerfile
ENV CUDA_VISIBLE_DEVICES=-1 \
    PYTHONHASHSEED=0

ENTRYPOINT ["/opt/crux/venv/bin/python", "-s", "/opt/crux/runtime/worker.py"]
```

Remove `/usr/bin/env -i` and all `*_NUM_THREADS=1` pins. Keep CPU-only behavior. Delete the Docker `test` stage plus test lock/wheelhouse.

- [ ] **Step 8: Remove old checkpoint/config authority and old modules**

Delete the two `config/benchmark/backends/` request files and the old request/evidence loader surface from `checkpoint_acquisition.py`. `model.json` becomes the only stable source.

Delete every old module/file listed in Task D after import search confirms the replacement owner.

- [ ] **Step 9: Replace the schema-golden manifest explicitly**

After HPA-324 integration and before Task E, `manifest.json` contains exactly these rows:

```text
crux.input-view-manifest/v1
crux.drum-prediction-events/v2
crux.dtx-reference-event/v1
crux.reference-chart-manifest/v1
crux.reference-timing-manifest/v1
crux.benchmark-reference-manifest/v1
```

Task E adds the new smoke-oracle v2 row. Descriptor v2 is tested directly, not via a golden.

- [ ] **Step 10: Prove no active old path remains**

```bash
git grep -n -E \
  'BackendVerification|TensorCoverageCheck|SmokeCheck|SealState|BackendNotSealed|heuristic-onset-v1|backend_lock|backend_reports|backend_attestation|backend_prepare|backend_publication|backend_process|backends\.oaf_tf1|runtime\.oaf_tf1\.(oaf_backend|entrypoint|protocol)|crux.drum-prediction-events/v1' \
  -- src runtime tests .github config
```

Expected: no active match.

- [ ] **Step 11: Run the atomic cutover gate**

```bash
uv run pytest
uv run ruff check src tests
uv run ruff format --check src tests
uv run pylint --errors-only --disable=E1120,E0401 src
git diff --check

docker build -f runtime/oaf_tf1/Dockerfile -t crux-oaf-tf1:local .
```

Expected: all pass without needing the released checkpoint.

- [ ] **Step 12: Commit**

```bash
git add -A
git commit -m "refactor: cut over to reusable OaF backend"
```

---

## Task E: Add Exact-Oracle Smoke Command and Manual Workflow

**Goal:** Validate real inference through the normal adapter and prepare the behavioral oracle gate.

**Files:**
- Create: `src/benchmark/oaf_smoke_oracle.py`
- Modify: `src/cli/benchmark.py`
- Add: `tests/benchmark/test_oaf_smoke_oracle.py`
- Modify: `tests/test_cli_benchmark.py`
- Add: `tests/benchmark/schema_goldens/crux.oaf-smoke-oracle-v2.json`
- Modify: `tests/benchmark/schema_goldens/manifest.json`
- Add: `.github/workflows/oaf-smoke.yml`
- Add: `tests/benchmark/test_oaf_smoke_workflow.py`

### Oracle schema

```text
crux.oaf-smoke-oracle/v2
```

Top-level keys:

```text
schema
backend_id
input_audio_sha256
native_events
```

Event keys:

```text
time_sec_binary64
native_class_id
model_output_bin
native_midi_note
upstream_8hit_group_id
confidence_binary64
velocity_midi
```

- [ ] **Step 1: Write exact oracle render/read/compare tests**

Use `struct.pack(">d", value).hex()` for time/confidence. Require read(render(events)) reproduces every native event exactly and changing one bit/digit in any event field makes comparison fail.

Register `crux.oaf-smoke-oracle/v2` as the seventh and final row in `schema_goldens/manifest.json`.

- [ ] **Step 2: Implement `oaf_smoke_oracle.py`**

Expose:

```python
def render_smoke_oracle(prediction: MappedPrediction) -> bytes: ...
def read_smoke_oracle(content: bytes) -> SmokeOracle: ...
def assert_smoke_oracle_matches(prediction: MappedPrediction, oracle: SmokeOracle) -> None: ...
```

Compare only retained post-adapter native fields. Do not reintroduce old worker `frame_index` into the shared contract.

- [ ] **Step 3: Write CLI timing/oracle tests with a fake backend**

Patch `time.perf_counter()` so:

```text
start = 10.0
return from backend.transcribe = before 12.0
mapping/publication operations execute after timer stop
```

Require `inference_elapsed_seconds == 2.0` regardless of mocked time spent in rendering/publication.

With one mapped hi-hat + one sticks event require:

```text
exit 0
mapped_event_count = 1
unmapped_event_count = 1
prediction artifact v2 exists
RTF = 2.0 / fixture_duration_seconds
```

When `--oracle path` is supplied:

```text
matching oracle -> oracle_status=matched, exit 0
missing oracle -> nonzero
mismatching oracle -> nonzero
```

Without the option require `oracle_status=not_checked`.

- [ ] **Step 4: Implement `smoke-backend`**

Command:

```bash
uv run crux benchmark smoke-backend --backend oaf [--oracle PATH]
```

Timer scope is exactly:

```python
started = time.perf_counter()
native_prediction = backend.transcribe(audio)
inference_elapsed_seconds = time.perf_counter() - started
```

Then map, validate mapped count, render/publish prediction, compare optional oracle, and print one canonical summary.

Summary keys:

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
inference_elapsed_seconds
real_time_factor
oracle_status
```

- [ ] **Step 5: Write the initial workflow without an oracle flag**

`.github/workflows/oaf-smoke.yml`:

```text
workflow_dispatch only
no custom inputs
one smoke job
runs-on ubuntu-24.04
permissions contents: read
prepare-backend --backend oaf --download once
docker build once
smoke-backend --backend oaf once
upload prediction JSONL only
write summary fields to GITHUB_STEP_SUMMARY
```

Require no attestation/OIDC/Sigstore/calibration/measurement step.

- [ ] **Step 6: Run validation**

```bash
uv run pytest tests/benchmark/test_oaf_smoke_oracle.py \
  tests/benchmark/test_oaf_smoke_workflow.py tests/test_cli_benchmark.py -q
uv run pytest
uv run ruff check src tests
uv run ruff format --check src tests
```

- [ ] **Step 7: Commit**

```bash
git add src/benchmark/oaf_smoke_oracle.py src/cli/benchmark.py \
  tests/benchmark tests/test_cli_benchmark.py .github/workflows/oaf-smoke.yml
git commit -m "feat: add oracle-capable OaF smoke workflow"
```

---

## Task F: Establish the Real Oracle and Prove the Extracted Model Twice

**Goal:** Run authentic inference once to establish expected native behavior, commit that expectation, then prove a fresh run matches it exactly.

**Outputs:**
- first workflow prediction artifact;
- committed `tests/fixtures/oaf_tf1_smoke/smoke-oracle.json`;
- second successful workflow run with `oracle_status=matched`.

- [ ] **Step 1: Dispatch the first smoke on the exact implementation commit**

Use the workflow UI with no custom inputs. Require:

```text
backend_id = magenta-egmd-tf1-94529798-8hit-v1
upstream_source_commit = 94529798dfbbb14c27ddfd76f23027dc8e2ce185
checkpoint_archive_sha256 = 09765ae0ff19c7d769a3c20e158eba3b9cd279429b02e498b1e911d16f82e2c0
restored_tensor_count = 78
mapped_event_count > 0
inference_elapsed_seconds > 0
real_time_factor > 0
oracle_status = not_checked
```

Download the uploaded prediction JSONL.

- [ ] **Step 2: Derive the oracle with production code**

Place the downloaded file at `/tmp/hpa423-prediction.jsonl` and run:

```bash
uv run python - <<'PY'
from pathlib import Path
from src.benchmark.oaf_smoke_oracle import render_smoke_oracle
from src.benchmark.prediction_artifact import read_prediction_artifact

prediction = read_prediction_artifact(Path("/tmp/hpa423-prediction.jsonl").read_bytes()).prediction
Path("tests/fixtures/oaf_tf1_smoke/smoke-oracle.json").write_bytes(
    render_smoke_oracle(prediction)
)
PY
```

Validate it immediately:

```bash
uv run pytest tests/benchmark/test_oaf_smoke_oracle.py -q
```

- [ ] **Step 3: Update the fixed workflow to enable the oracle gate**

Change the smoke invocation to:

```bash
uv run crux benchmark smoke-backend --backend oaf \
  --oracle tests/fixtures/oaf_tf1_smoke/smoke-oracle.json
```

Update the workflow test to require that exact oracle argument.

- [ ] **Step 4: Run repository verification and commit the oracle**

```bash
uv run pytest
uv run ruff check src tests
uv run ruff format --check src tests
uv run pylint --errors-only --disable=E1120,E0401 src
git diff --check

git add tests/fixtures/oaf_tf1_smoke/smoke-oracle.json \
  .github/workflows/oaf-smoke.yml tests/benchmark/test_oaf_smoke_workflow.py
git commit -m "test: freeze OaF smoke output oracle"
```

- [ ] **Step 5: Dispatch a second smoke on the oracle commit**

Require all Step 1 invariants plus:

```text
oracle_status = matched
```

The workflow must fail if the native event count/order, binary64 time/confidence, native ID/bin/MIDI/group, or velocity differs from the committed oracle.

- [ ] **Step 6: Record completion in Linear**

Post:

```text
exact commit SHA
first smoke run URL
second oracle-gated smoke run URL
restored tensor count = 78
mapped/unmapped counts
inference RTF
oracle_status = matched
```

Move HPA-423 to Done only after the second run passes. Then HPA-326 may start.

---

## Final Self-Review Checklist

Before declaring the implementation plan complete, confirm:

- [ ] Task A does not shrink shared types or delete old extraction sources.
- [ ] Task B temporarily extends, rather than breaks, old checkpoint APIs so its commit stays green.
- [ ] New runtime tests live under `tests/runtime/oaf_tf1/` and therefore run in bare CI pytest.
- [ ] Task C preserves the 78-tensor and stochastic-op gates before deleting source code.
- [ ] Task D is the only broad incompatible commit and runs the full suite/image build before commit.
- [ ] `map_oaf_prediction` extends `mapping.py` and reuses `MappingDiagnostics`.
- [ ] Final `model.json` is the only model/checkpoint source and explicitly says `max_input_audio_frames: null`.
- [ ] Final input/prediction APIs contain no `DirectoryAnchor` parameter.
- [ ] Final Docker runtime has no `env -i` or single-thread caps and starts `worker.py` directly.
- [ ] Final schema-golden manifest contains exactly the seven rows listed across Tasks D/E.
- [ ] Smoke timing excludes mapping/serialization/publication.
- [ ] Exact output behavior is established from one authentic run and independently matched by a second run.
- [ ] No seal compatibility, prediction-v1 compatibility, worker pool, queue, retry system, or new security subsystem has been introduced.
