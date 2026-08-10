# HPA-423 Minimal Reusable OaF Backend Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the unusable sealed OaF path with one reusable persistent OaF backend that verifies the released checkpoint, persists HPA-324-compatible prediction events, and passes one real native smoke inference.

**Architecture:** Extract the useful checkpoint/model/worker mechanics while the old shared seal-shaped interfaces remain temporarily importable, then make one deliberate breaking cutover that moves every live consumer together. The final path is `CanonicalAudio -> OafBackend -> persistent worker -> OafModel`, followed by HPA-324-owned class mapping, prediction artifact v2, and common-class scorer input. No compatibility reader, seal lifecycle, heuristic backend, report framework, or security-attestation subsystem survives.

**Tech Stack:** Python 3.12 host tooling/tests, existing pinned Python 3.7/TensorFlow 1 OaF container dependencies, Click CLI, Docker, pytest, Ruff, Pylint, GitHub Actions `ubuntu-24.04`.

## Global Constraints

- Preserve `OAF_BACKEND_ID = "magenta-egmd-tf1-94529798-8hit-v1"`.
- Preserve upstream Magenta revision `94529798dfbbb14c27ddfd76f23027dc8e2ce185`.
- Preserve checkpoint archive SHA-256 `09765ae0ff19c7d769a3c20e158eba3b9cd279429b02e498b1e911d16f82e2c0`.
- Preserve checkpoint component SHA-256 values `6312cd6f8e29ee6f21f4a763822d554f46393e0e336812ba4a8cb324d2e8b0b5`, `475ca21993e102d52e2e6c7066b598b9b9e9a097bfb2fa6b770d40eedaa6139a`, and `e36105b27c17f2367292682df825b8ad16eb887ccad465bbb5ddf955ef75d422`.
- Require all 78 inference tensors to restore before the new worker reports ready.
- Preserve `CanonicalAudio.source_audio_id`, `source_audio_sha256`, `input_view_id`, and `input_audio_sha256`.
- Preserve OaF `native_class_id="midi_<note>"` and `native_metadata["upstream_8hit_group_id"]` identity.
- HPA-324 owns taxonomy/map data. `PredictionMap.backend_id`, not checkpoint `model_id`, selects the backend/native-output contract.
- The OaF mapper validates both descriptor `backend_id == prediction_map.backend_id` and matching `native_output_space_id` before lookup.
- Prediction persistence is a breaking replacement to `crux.drum-prediction-events/v2`; do not add a v1 compatibility reader.
- Prediction-side `BenchmarkEvent.canonical_class` is the **common scoring class**. Detailed canonical prediction identity belongs only in `metadata["detailed_canonical_class"]`; HPA-325 must not confuse these levels.
- One persistent worker, one request at a time. No worker pool, queue, retry framework, or distributed execution.
- Do not add OIDC, Sigstore, host attestation, runtime seals, deterministic OCI publication, or security-only policy code.
- Keep TensorFlow imports lazy/outside host module import paths.
- Do not change the pinned TensorFlow/OaF dependency set unless extraction proves a dependency is unused and removing it is independently safe.
- Every task commit must leave ordinary imports and `uv run pytest` green. Focused RED tests are allowed while implementing a task, but never commit a state that depends on a later task to repair broken live consumers.
- Use repository validation from `CLAUDE.md`: `uv run pytest`, `uv run ruff check src tests`, `uv run ruff format --check src tests`, and error-focused Pylint on changed host modules.

---

## Target Active File Structure

```text
src/benchmark/
  artifact_io.py
  backend_identity.py
  backend_registry.py
  checkpoint_acquisition.py
  worker_process.py
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

### Task 1: Prune independently dead HPA-320 outer estate without cutting shared live interfaces

**Files:**
- Delete when the pre-delete grep proves no retained importer: `.github/workflows/hpa320-native-bootstrap.yml`
- Delete when the pre-delete grep proves no retained importer: `.github/workflows/hpa320-native-candidate.yml`
- Delete when the pre-delete grep proves no retained importer: `.github/workflows/hpa320-native-measurement.yml`
- Delete when the pre-delete grep proves no retained importer: `tools/hpa320/`
- Delete matching workflow/tool-only tests and goldens whose only purpose is the removed outer estate
- Modify: `runtime/oaf_tf1/Dockerfile` only to remove copies of files deleted in this task; do not switch its runtime entrypoint yet

**Interfaces:**
- Produces no new runtime API.
- Preserves the current `backends/base.py`, `NativePrediction` lock fields, `backend_process.py`, old OaF adapter, `prediction_artifact.py`, and `transcription.py` until Task 6 moves every live consumer together.

- [ ] **Step 1: Enumerate outer-estate references before deleting anything**

Run:

```bash
git grep -n -E \
  'tools\.hpa320|tools/hpa320|hpa320-native-(bootstrap|candidate|measurement)|calibration_(entrypoint|protocol)' \
  -- src runtime tests .github tools pyproject.toml
```

Classify every match into exactly one of these buckets:

```text
A = workflow/tool/test file deleted in this task
B = Dockerfile copy line deleted in this task
C = retained live importer; defer that source file and its dependency to Task 6
```

Do not edit a retained live module merely to force deletion in Task 1.

- [ ] **Step 2: Write/adjust deletion-scope tests**

In the existing workflow/build-context tests, remove assertions for workflows/tools being deleted. Add one small repository-scope test that scans active `.github/workflows/*.yml` and fails if any workflow name starts with `hpa320-native-` after Task 1.

Example assertion:

```python
workflow_names = {path.name for path in Path(".github/workflows").glob("*.yml")}
assert not {name for name in workflow_names if name.startswith("hpa320-native-")}
```

- [ ] **Step 3: Delete only bucket A/B outer-estate files**

Delete the three HPA-320 native workflows, tool modules/tests that are not imported by retained source, and Dockerfile copy lines that reference those deleted files.

If `tools/hpa320` has a retained live importer from bucket C, leave only that exact required file subtree until Task 6 instead of inventing a compatibility shim. Record the deferred path in the Task 6 deletion list.

- [ ] **Step 4: Prove no dangling outer-estate imports**

Run the grep again:

```bash
git grep -n -E \
  'tools\.hpa320|tools/hpa320|hpa320-native-(bootstrap|candidate|measurement)' \
  -- src runtime tests .github tools pyproject.toml
```

Expected: only explicitly deferred live HPA-320 extraction sources remain; no reference points at a deleted path.

- [ ] **Step 5: Run repository tests and lint for touched files**

```bash
uv run pytest
uv run ruff check src tests
uv run ruff format --check src tests
git diff --check
```

Expected: PASS. Task 1 must not rely on later contract work to restore imports.

- [ ] **Step 6: Commit**

```bash
git add -A
git commit -m "refactor: prune obsolete OaF seal workflows and tools"
```

---

### Task 2: Extract neutral artifact I/O and simplify the existing checkpoint acquisition path

**Files:**
- Create: `src/benchmark/artifact_io.py`
- Create: `runtime/oaf_tf1/model.json`
- Create: `runtime/oaf_tf1/model.py` with dependency-light config types/loader only
- Modify: `src/benchmark/input_view.py`
- Rewrite in place: `src/benchmark/checkpoint_acquisition.py`
- Modify: `src/benchmark/prediction_artifact.py` publication helper/import only; keep v1 domain shape until Task 6
- Modify: `src/cli/benchmark.py` `prepare-backend` command
- Add: `tests/benchmark/test_artifact_io.py`
- Rewrite: `tests/benchmark/test_checkpoint_acquisition.py`
- Modify: `tests/benchmark/test_prediction_artifact.py` only for neutral publisher behavior
- Modify: `tests/test_cli_benchmark.py`
- Keep temporarily: `src/benchmark/backend_prepare.py` if the old OaF adapter still imports it
- Keep temporarily: `src/benchmark/backend_publication.py` for old live HPA-320 callers not yet cut over

**Interfaces:**

```text
artifact_io.read_regular_file_no_follow(path: Path) -> bytes
artifact_io.publish_immutable_file(path: Path, content: bytes) -> PublishedArtifact(path, sha256)
model.load_model_config(path) -> OafModelConfig
checkpoint_acquisition.prepare_oaf_checkpoint(config, cache_root, download=False, archive_path=None) -> Path
```

- [ ] **Step 1: Write failing neutral artifact-I/O tests**

Add:

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
    assert first.path == path
    assert first.sha256 == hashlib.sha256(b"{\"ok\":true}\n").hexdigest()
    assert path.read_bytes() == b"{\"ok\":true}\n"


def test_publish_immutable_file_rejects_changed_existing_bytes(tmp_path):
    path = tmp_path / "artifact.json"
    publish_immutable_file(path, b"one")
    with pytest.raises(ArtifactPublicationError):
        publish_immutable_file(path, b"two")
```

- [ ] **Step 2: Write the publisher-callsite regression before changing `PublishedArtifact`**

Add/adjust a `publish_prediction_artifact` test that patches `artifact_io.publish_immutable_file` with a two-argument spy and requires exactly:

```python
published = publish_prediction_artifact(path, prediction)
assert spy.call_args.args == (path, render_prediction_artifact(prediction))
assert spy.call_args.kwargs == {}
assert published.path == path
```

This is the hard gate for removing the old `role="prediction"`/anchor call shape. Do not leave a half-migrated `role=` keyword for Task 6.

- [ ] **Step 3: Write failing `model.json` loader tests**

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

For malformed config, copy `model.json` to `tmp_path`, replace `archive_sha256` with `"bad"`, and assert `OafModelConfigError` contains `archive_sha256`.

- [ ] **Step 4: Write checkpoint preparation tests with a synthetic ZIP**

Create a helper that writes exactly three component files into a ZIP, calculates archive/component hashes, and builds an `OafModelConfig` from those values. Require:

```text
local archive extraction succeeds
verify-only reuse succeeds
mutated archive bytes fail before accepted extraction
wrong component hash fails
missing verify-only cache raises "checkpoint cache is missing"
returned path == <cache_root>/sha256/<archive_sha256>
```

- [ ] **Step 5: Verify RED for the new APIs**

```bash
uv run pytest \
  tests/benchmark/test_artifact_io.py \
  tests/benchmark/test_checkpoint_acquisition.py \
  tests/benchmark/test_prediction_artifact.py -q
```

Expected: new neutral APIs/model config are absent.

- [ ] **Step 6: Implement only retained generic artifact behavior**

`artifact_io.py` defines:

```python
@dataclass(frozen=True)
class PublishedArtifact:
    path: Path
    sha256: str


class ArtifactPublicationError(OSError):
    pass
```

Implement no-follow regular reads and immutable create/reuse. Port an anchor type only if `input_view.py` still requires it after its own simplification; test that exact retained API if so. Do not port seal directory transaction/rollback/attestation code.

Update `publish_prediction_artifact()` in this task to call `publish_immutable_file(path, content)` with no `role=` or seal-specific anchor argument.

- [ ] **Step 7: Create `model.json` and dependency-light config loader**

Use schema `crux.oaf-model/v1` and the exact IDs/hashes from the design spec. `runtime/oaf_tf1/model.py` defines `CheckpointComponent`, `OafCheckpointConfig`, `OafModelConfig`, `OafModelConfigError`, and `load_model_config()` using standard-library imports only at module import time.

- [ ] **Step 8: Rewrite existing `checkpoint_acquisition.py` in place**

Do not add another acquisition module. Reuse the current ZIP/member/hash mechanics plus useful extraction code from `backend_prepare.py`, but make `model.json` the only stable identity source for the retained path.

`prepare_oaf_checkpoint()` verifies archive/member/component identities and caches under:

```text
<cache_root>/sha256/<archive_sha256>/
```

Remove acquisition-request/evidence/backend-lock/publication behavior from this retained API. Leave `backend_prepare.py` untouched only if the old adapter still imports it; Task 6 deletes that old path.

- [ ] **Step 9: Simplify `prepare-backend` without touching old transcription commands yet**

Final command surface:

```text
crux benchmark prepare-backend
  --backend [oaf]
  --download
  --archive PATH
  --cache-root DIRECTORY
```

Defaults: `--backend oaf`, `--cache-root artifacts/benchmark/model-cache`. `--download` and `--archive` are mutually exclusive. Remove `--acquisition-request`, `--evidence-output`, and `--backend-lock`.

Success stdout is one canonical JSON object containing exactly:

```text
backend
checkpoint_path
status
```

with `status="ready"`.

- [ ] **Step 10: Run focused and repository validation**

```bash
uv run pytest \
  tests/benchmark/test_artifact_io.py \
  tests/benchmark/test_checkpoint_acquisition.py \
  tests/benchmark/test_prediction_artifact.py \
  tests/test_cli_benchmark.py -q
uv run pytest
uv run ruff check src tests
uv run ruff format --check src tests
```

Expected: PASS. Old live HPA-320 modules may still exist, but retained paths use the new neutral helpers where planned.

- [ ] **Step 11: Commit**

```bash
git add src/benchmark/artifact_io.py src/benchmark/input_view.py \
  src/benchmark/checkpoint_acquisition.py src/benchmark/prediction_artifact.py \
  src/cli/benchmark.py runtime/oaf_tf1/model.json runtime/oaf_tf1/model.py tests
git commit -m "refactor: simplify OaF checkpoint and artifact I/O"
```

---

### Task 3: Extract the TensorFlow OaF model engine alongside the old sealed runtime

**Files:**
- Expand: `runtime/oaf_tf1/model.py`
- Keep as extraction source until Task 6: `runtime/oaf_tf1/oaf_backend.py`
- Keep: `runtime/oaf_tf1/patches/capture-emitted-frame.patch`
- Keep: `runtime/oaf_tf1/vendor/`
- Add: `runtime/oaf_tf1/tests/test_model.py`
- Migrate useful pure coverage from: `runtime/oaf_tf1/tests/test_tensor_coverage.py`

**Interfaces:**

```text
OafModel.load(checkpoint_dir, config=None) -> OafModel
OafModel.restored_tensor_count -> int
OafModel.transcribe(audio_path) -> tuple[OafNativeEvent, ...]
validate_required_tensors(required, restored) -> 78 or raises OafModelError
```

- [ ] **Step 1: Write import-boundary tests**

```python
def test_model_module_does_not_import_tensorflow_eagerly():
    sys.modules.pop("tensorflow", None)
    importlib.import_module("runtime.oaf_tf1.model")
    assert "tensorflow" not in sys.modules
```

Reassert the Task 2 model config identities in the same file.

- [ ] **Step 2: Write representative native-event conversion tests**

Reuse representative frame/post-processing fixtures from the old runtime tests and require:

```text
snare -> native_class_id=midi_38, output_bin=17, midi=38, group=snare
hihat -> native_class_id=midi_46, output_bin=25, midi=46, group=hihat
toms  -> native_class_id=midi_48, output_bin=27, midi=48, group=toms
```

Also assert time, confidence, and velocity preserve the representative input values.

- [ ] **Step 3: Write the 78-tensor gate test**

Load the existing required tensor inventory from the extraction source. Require a complete restored set returns `78`; remove one known tensor and require `OafModelError` contains its name.

- [ ] **Step 4: Verify RED**

```bash
uv run pytest runtime/oaf_tf1/tests/test_model.py -q
```

- [ ] **Step 5: Move only inference-relevant code into `model.py`**

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
```

Move graph construction, feature extraction, checkpoint restore, required-tensor validation, emitted-frame instrumentation handling, post-processing, and native event conversion. TensorFlow/Numpy/Magenta imports occur only inside model loading/inference helpers.

Do not move backend locks, seal evidence, host identity, calibration requests, source-attestation manifests, or publication code.

- [ ] **Step 6: Run old and new model tests together**

```bash
uv run pytest runtime/oaf_tf1/tests/test_model.py runtime/oaf_tf1/tests/test_tensor_coverage.py -q
uv run pytest
```

The old runtime remains present, so this extraction commit must not break it.

- [ ] **Step 7: Commit**

```bash
git add runtime/oaf_tf1/model.py runtime/oaf_tf1/tests/test_model.py \
  runtime/oaf_tf1/tests/test_tensor_coverage.py
git commit -m "refactor: extract reusable OaF model engine"
```

---

### Task 4: Add the new persistent worker and host process helper alongside the old controller

**Files:**
- Create: `runtime/oaf_tf1/worker.py`
- Create: `src/benchmark/worker_process.py`
- Add: `runtime/oaf_tf1/tests/test_worker.py`
- Add: `tests/benchmark/test_worker_process.py`
- Keep until Task 6: `runtime/oaf_tf1/entrypoint.py`
- Keep until Task 6: `runtime/oaf_tf1/protocol.py`
- Keep until Task 6: `src/benchmark/backend_process.py`

**Interfaces:**

```text
Ready:   {"type":"ready","backend_id":"magenta-egmd-tf1-94529798-8hit-v1","restored_tensor_count":78}
Request: {"id":"request-1","audio_path":"relative/path.wav"}
Success: {"id":"request-1","events":[...]}
Error:   {"id":"request-1","error":{"code":"inference_failed","message":"inference failed"}}
WorkerProcess.start(command, startup_timeout_sec=120.0, request_timeout_sec=600.0)
WorkerProcess.ready
WorkerProcess.request(payload)
WorkerProcess.close()
```

- [ ] **Step 1: Write worker lifecycle tests with a concrete fake model**

Use a fake model factory with a call counter, `restored_tensor_count=78`, and `transcribe()` returning one fixed `OafNativeEvent`. Feed two request lines through `io.StringIO`. Assert:

```text
factory called once
first output is ready
two responses retain request IDs
fake model receives both /input/... paths in order
EOF returns zero
```

Feed `{not-json}\n` and require one `invalid_request` error. Make the fake model raise `OafModelError` and require `inference_failed`.

- [ ] **Step 2: Write process-helper tests using generated fake scripts**

Generate a Python script that prints a valid ready line and echoes each request ID with an empty event list. Start it with `[sys.executable, str(script)]`, call `.request()` twice, then `.close()` and assert child exit.

Generate separate scripts for:

```text
malformed ready JSON -> WorkerProcessError("worker ready response is invalid")
exit before ready -> WorkerProcessError("worker exited before ready")
wrong response ID -> WorkerProcessError("worker response id mismatch")
```

- [ ] **Step 3: Verify RED**

```bash
uv run pytest runtime/oaf_tf1/tests/test_worker.py tests/benchmark/test_worker_process.py -q
```

- [ ] **Step 4: Implement `worker.py` as one synchronous line loop**

Load `OafModel` once. Write ready, then process stdin line by line. A request contains exactly string `id` and relative string `audio_path`; reject absolute paths and `..`. Convert each `OafNativeEvent` to JSON with native IDs/group/confidence/velocity and flush after every response.

No protocol version negotiation, signatures, compatibility decoder, resume state, or diagnostic-redaction subsystem.

- [ ] **Step 5: Implement `worker_process.py` without touching old `backend_process.py`**

Use `subprocess.Popen(..., stdin=PIPE, stdout=PIPE, stderr=PIPE, text=True)` plus a small selector-based line timeout. `.request()` holds one lock, writes one JSON line, reads one response, validates ID, and rejects malformed/error responses. `.close()` closes stdin, waits briefly, then terminates/kills only if needed.

- [ ] **Step 6: Run focused and repository tests**

```bash
uv run pytest runtime/oaf_tf1/tests/test_worker.py tests/benchmark/test_worker_process.py -q
uv run pytest
```

The old OaF path still imports `backend_process.py`; keeping the new helper separate is what makes this commit green.

- [ ] **Step 7: Commit**

```bash
git add runtime/oaf_tf1/worker.py runtime/oaf_tf1/tests/test_worker.py \
  src/benchmark/worker_process.py tests/benchmark/test_worker_process.py
git commit -m "feat: add minimal persistent OaF worker"
```

---

### Task 5: Add the HPA-324 OaF prediction mapper without changing persistence yet

**Precondition:** HPA-324 production code is present with `PredictionMap.backend_id`, `ClassMapping`, `OAF_PREDICTION_MAP`, and `OAF_PREDICTION_MAP_ID`. If HPA-324 has not merged, integrate its reviewed production commit; do not copy its map table into HPA-423.

**Files:**
- Create: `src/benchmark/prediction_mapping.py`
- Add: `tests/benchmark/test_prediction_mapping.py`
- Do not rewrite `prediction_artifact.py` or `scorer_input.py` until Task 6.

**Interfaces:**

```text
map_oaf_prediction(prediction, prediction_map=OAF_PREDICTION_MAP) -> PredictionMappingResult
MappedPredictionEvent(native, canonical_class, common_class, mapping_status, prediction_map_version)
MappedPrediction(audio, descriptor, events)
PredictionMappingDiagnostics(unmapped)
```

- [ ] **Step 1: Write real-shaped OaF fixtures**

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

Build a current `NativePrediction` fixture with whatever legacy lock fields are still required by the pre-cutover dataclass; the mapper must ignore those fields.

- [ ] **Step 2: Write exact map identity tests**

Construct a valid descriptor/map and assert mapping succeeds. Then change each selector independently:

```python
bad_backend_map = replace(OAF_PREDICTION_MAP, backend_id="other-backend")
with pytest.raises(PredictionMappingError, match="backend identity mismatch"):
    map_oaf_prediction(prediction, bad_backend_map)

bad_space_map = replace(OAF_PREDICTION_MAP, native_output_space_id="other-space")
with pytest.raises(PredictionMappingError, match="native output space mismatch"):
    map_oaf_prediction(prediction, bad_space_map)
```

Do **not** compare `prediction_map.backend_id` with descriptor `model_id`. Add an assertion that the fixture has different backend/model IDs so this regression cannot hide:

```python
assert prediction.descriptor.payload["backend_id"] != prediction.descriptor.payload["model_id"]
```

- [ ] **Step 3: Write class-mapping tests**

Require:

```text
hihat MIDI 46 -> mapped, canonical=None, common=hihat
toms MIDI 48 -> mapped, canonical=None, common=tom
snare MIDI 38 -> mapped, canonical=snare, common=snare
sticks MIDI 75 -> unmapped, both classes null
unknown/null group -> unmapped and counted
prediction_map_version == OAF_PREDICTION_MAP_ID on every event
all native fields unchanged
```

- [ ] **Step 4: Verify RED**

```bash
uv run pytest tests/benchmark/test_prediction_mapping.py -q
```

- [ ] **Step 5: Implement the pure mapper**

Validate exactly:

```python
prediction.descriptor.payload["backend_id"] == prediction_map.backend_id
prediction.descriptor.payload["native_output_space_id"] == prediction_map.native_output_space_id
```

Lookup only:

```python
key = event.native_metadata.get("upstream_8hit_group_id")
class_mapping = prediction_map.classes.get(key) if key is not None else None
```

A map entry with non-null `common_class` is mapped. Missing entries or entries with null common class are unmapped. Native events are always retained.

- [ ] **Step 6: Run focused and repository tests**

```bash
uv run pytest tests/benchmark/test_prediction_mapping.py -q
uv run pytest
```

No persistence/contract code changed, so the old repository path remains green.

- [ ] **Step 7: Commit**

```bash
git add src/benchmark/prediction_mapping.py tests/benchmark/test_prediction_mapping.py
git commit -m "feat: map native OaF prediction classes"
```

---

### Task 6: Perform the atomic backend/prediction cutover

This is the only intentionally broad task. The shared contract, descriptor, registry, new adapter, prediction artifact v2, scorer bridge, old live consumers, and seal-oriented helper modules must move together because they cannot be independently committed without leaving broken imports or incompatible constructors.

**Files — create/modify:**
- Modify: `src/benchmark/backends/base.py`
- Modify: `src/benchmark/backends/__init__.py`
- Modify: `src/benchmark/backend_identity.py`
- Modify: `src/benchmark/backend_registry.py`
- Create: `src/benchmark/backends/oaf.py`
- Rewrite: `src/benchmark/prediction_artifact.py`
- Rewrite: `src/benchmark/scorer_input.py`
- Modify: `runtime/oaf_tf1/Dockerfile` to run `worker.py`
- Modify: `src/cli/benchmark.py` to remove old `verify-backend` and `transcribe-one`; keep Task 2 `prepare-backend`

**Files — delete after import search:**
- `src/benchmark/backends/oaf_tf1.py`
- `runtime/oaf_tf1/oaf_backend.py`
- `runtime/oaf_tf1/entrypoint.py`
- `runtime/oaf_tf1/protocol.py`
- `src/benchmark/backend_process.py`
- `src/benchmark/transcription.py`
- `src/benchmark/backend_reports.py`
- `src/benchmark/backend_attestation.py`
- `src/benchmark/backend_lock.py`
- `src/benchmark/backend_prepare.py`
- `src/benchmark/backend_publication.py`
- old seal/verification/heuristic tests and schema goldens replaced by the new tests below

**Tests:**
- Modify: `tests/benchmark/test_backend_contract_acceptance.py`
- Modify: `tests/benchmark/test_backend_registry.py`
- Modify: `tests/benchmark/test_backend_identity.py`
- Add: `tests/benchmark/test_oaf_adapter.py`
- Rewrite: `tests/benchmark/test_prediction_artifact.py`
- Add/Rewrite: `tests/benchmark/test_scorer_input.py`
- Modify: `tests/benchmark/test_schema_goldens.py`
- Delete/migrate old `test_oaf_backend.py`, `test_transcription.py`, `test_backend_reports.py`, `test_backend_process.py`, lock/attestation/seal-only tests
- Replace active prediction golden with `tests/benchmark/schema_goldens/crux.drum-prediction-events-v2.jsonl` if schema-golden coverage remains

**Interfaces after cutover:**

```text
NativePrediction(audio, descriptor, events)
TranscriptionBackend.descriptor/transcribe/close
BackendDescriptor schema crux.transcription-backend-descriptor/v2
BackendRegistry default/factory key oaf only
OafBackendConfig(checkpoint_dir, input_root, image="crux-oaf-tf1:local")
OafBackend.transcribe(audio) -> NativePrediction
prediction artifact schema crux.drum-prediction-events/v2
read_scorer_events(content) -> tuple[BenchmarkEvent, ...]
```

- [ ] **Step 1: Write the post-cutover contract tests before changing production types**

```python
def test_native_prediction_has_only_transcription_domain_fields():
    assert set(NativePrediction.__dataclass_fields__) == {"audio", "descriptor", "events"}


def test_canonical_audio_retains_source_and_input_identity():
    assert {
        "source_audio_id",
        "source_audio_sha256",
        "input_view_id",
        "input_audio_sha256",
    } <= set(CanonicalAudio.__dataclass_fields__)


def test_verification_types_are_removed():
    import src.benchmark.backends as backends
    assert not hasattr(backends, "BackendVerification")
    assert not hasattr(backends, "TensorCoverageCheck")
    assert not hasattr(backends, "SmokeCheck")
    assert not hasattr(TranscriptionBackend, "verify")
```

- [ ] **Step 2: Write compact descriptor/registry tests**

Require descriptor keys exactly:

```python
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
```

Require `OAF_DESCRIPTOR_SCHEMA == "crux.transcription-backend-descriptor/v2"`, `prediction_schema == "crux.drum-prediction-events/v2"`, and no lock/image/protocol fields.

Registry tests:

```python
registry = default_backend_registry()
assert registry.default_backend_id == "oaf"
assert set(registry.factories) == {"oaf"}
with pytest.raises(BackendUnavailable, match="unknown backend"):
    registry.create("heuristic-onset-v1")
```

Assert `SealState`, `BackendNotSealed`, `BackendIntegrityUnavailable`, `HEURISTIC_BACKEND_ID`, and `LEGACY_TF2_BACKEND_ID` are absent.

- [ ] **Step 3: Write adapter tests using a fake `WorkerProcess` factory**

Cover:

```text
ready backend ID + restored_tensor_count=78 -> success
wrong backend ID -> "worker backend identity mismatch"
restored count 77 -> "worker restored tensor count mismatch"
worker snare event -> exact NativeEvent identity
worker error -> OafBackendError preserving error code
two transcribe calls -> same process instance
audio outside input_root -> rejected before request
close twice -> underlying close at most once
```

The compact descriptor is built from `model.json` identity.

- [ ] **Step 4: Write artifact-v2 hihat/tom/sticks round-trip tests**

Use `map_oaf_prediction()` from Task 5 to produce a mapped prediction containing hihat, toms, and sticks. Require:

```text
schema = crux.drum-prediction-events/v2
hihat: mapping_status=mapped, canonical=null, common=hihat, map version non-null
toms: mapping_status=mapped, canonical=null, common=tom, map version non-null
sticks: mapping_status=unmapped, canonical=null, common=null, map version non-null
```

Read bytes back and assert every native MIDI note/output bin/confidence/velocity/class ID/metadata value reproduces exactly.

The v2 header must not contain backend/runtime/parameter/model-artifact lock hashes or heuristic nullability fields.

- [ ] **Step 5: Write scorer semantics tests that lock HPA-325 behavior**

Create v2 bytes with one mapped hihat and one unmapped sticks event:

```python
events = read_scorer_events(content)
assert len(events) == 1
assert events[0].chart_id == "song-1"
assert events[0].canonical_class == "hihat"
assert events[0].source == "prediction"
assert events[0].metadata["detailed_canonical_class"] is None
assert events[0].metadata["native_class_id"] == "midi_46"
assert events[0].metadata["input_view_id"] == "full_mix"
```

Add a mapped snare example and require:

```python
assert snare_event.canonical_class == "snare"  # common scorer class
assert snare_event.metadata["detailed_canonical_class"] == "snare"
```

Name the test `test_scorer_bridge_uses_common_class_in_benchmark_event_canonical_class` so HPA-325 cannot reasonably reinterpret the field later.

- [ ] **Step 6: Verify RED across the complete cutover surface**

```bash
uv run pytest \
  tests/benchmark/test_backend_contract_acceptance.py \
  tests/benchmark/test_backend_identity.py \
  tests/benchmark/test_backend_registry.py \
  tests/benchmark/test_oaf_adapter.py \
  tests/benchmark/test_prediction_mapping.py \
  tests/benchmark/test_prediction_artifact.py \
  tests/benchmark/test_scorer_input.py -q
```

Expected: current shared contract/registry/artifact/scorer path fails the new assertions.

- [ ] **Step 7: Shrink shared backend types and descriptor**

`NativePrediction` becomes only `audio`, `descriptor`, `events`. `TranscriptionBackend` contains `descriptor`, `transcribe`, `close` only. Remove verification/smoke/tensor/publication types from `backends/__init__.py`.

Replace descriptor v1 with compact v2, remove heuristic descriptor constants and seal identity fields, and keep canonical JSON/hash utilities.

- [ ] **Step 8: Create and activate `backends/oaf.py`**

Use `worker_process.WorkerProcess`. Validate ready backend ID and restored tensor count before accepting the worker. Mount verified checkpoint at `/model:ro`, input root at `/input:ro`, use `--network none`, convert canonical input path to an input-root-relative POSIX path, decode native events, and return simplified `NativePrediction`.

Default registry contains only `{"oaf": create_backend}`. Unknown backend raises `BackendUnavailable` directly.

- [ ] **Step 9: Advance `prediction_artifact.py` to v2 using neutral artifact I/O**

Renderer/reader consume `MappedPrediction`. Keep header/event/terminal JSONL structure. Add `common_class`, enforce mapped/unmapped semantics, preserve native fields, and remove seal-only header/nullability logic. `publish_prediction_artifact()` uses `artifact_io.publish_immutable_file(path, content)` with no `role=` keyword.

Delete the v1 compatibility path entirely.

- [ ] **Step 10: Replace `scorer_input.py` stub**

For each mapped event with non-null `common_class`, construct:

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

- [ ] **Step 11: Switch Docker runtime to the new worker and remove every old live consumer in the same change**

Update `runtime/oaf_tf1/Dockerfile` to copy/start `model.py` + `worker.py` and required vendored/instrumented runtime files only.

Before deletion run:

```bash
git grep -n -E \
  'BackendVerification|TensorCoverageCheck|SmokeCheck|backends\.oaf_tf1|runtime\.oaf_tf1\.(oaf_backend|entrypoint|protocol)|backend_process|backend_reports|backend_attestation|backend_lock|backend_prepare|backend_publication' \
  -- src runtime tests .github tools
```

Update/delete every active match in this task. Delete the old modules listed in the Task 6 file list plus their obsolete tests/goldens. Remove old `verify-backend` and `transcribe-one` Click commands with `transcription.py`; Task 7 adds `smoke-backend` separately.

- [ ] **Step 12: Run the full cutover gate before committing**

```bash
uv run pytest
uv run ruff check src tests
uv run ruff format --check src tests
uv run pylint --errors-only --disable=E1120,E0401 src
git diff --check
```

Also run the old-architecture grep again. Expected: no active source/runtime/test imports of removed seal modules or heuristic backend support.

- [ ] **Step 13: Build the new runtime image**

```bash
docker build -f runtime/oaf_tf1/Dockerfile -t crux-oaf-tf1:local .
```

Expected: image builds and starts `worker.py`; real checkpoint inference is deferred to Task 9.

- [ ] **Step 14: Commit the atomic cutover**

```bash
git add -A
git commit -m "refactor: cut over to reusable OaF backend"
```

---

### Task 7: Add `smoke-backend` and one manual workflow

**Files:**
- Modify: `src/cli/benchmark.py`
- Modify: `tests/test_cli_benchmark.py`
- Add: `.github/workflows/oaf-smoke.yml`
- Add: `tests/benchmark/test_oaf_smoke_workflow.py`

**Interfaces:**

```text
crux benchmark smoke-backend --backend oaf
output prediction = artifacts/benchmark/oaf-smoke/prediction.jsonl
one canonical stdout summary
one workflow_dispatch Ubuntu 24.04 job
```

- [ ] **Step 1: Write CLI smoke tests with a fake backend**

Create a fake backend returning one hihat plus one sticks native event. Patch `default_backend_registry()` and `time.perf_counter()` deterministically. Invoke against a temporary canonical WAV fixture. Require:

```text
exit 0 when at least one event maps
output reads as prediction artifact v2
mapped_event_count == 1
unmapped_event_count == 1
RTF == elapsed_seconds / (audio_frame_count / sample_rate)
prediction_sha256 matches bytes
backend.close called once
```

A sticks-only result exits nonzero and does not publish a successful prediction. Assert Click command registry contains neither `verify-backend` nor `transcribe-one`.

- [ ] **Step 2: Write workflow structure tests**

Parse the YAML and require:

```text
workflow_dispatch has no inputs
exact job key == smoke
runs-on == ubuntu-24.04
permissions == contents: read
one command contains "prepare-backend --backend oaf --download"
one command contains "docker build"
one command contains "smoke-backend --backend oaf"
one upload-artifact step uploads artifacts/benchmark/oaf-smoke/prediction.jsonl
no step contains tools.hpa320, seal, candidate, calibration, measurement, attest, or sigstore
```

- [ ] **Step 3: Verify RED**

```bash
uv run pytest tests/test_cli_benchmark.py tests/benchmark/test_oaf_smoke_workflow.py -q
```

- [ ] **Step 4: Implement `smoke-backend` without report orchestration**

Exact flow:

```text
load fixed canonical fixture as CanonicalAudio
create oaf backend
start timer
transcribe once
map_oaf_prediction
require mapped_event_count > 0
render/publish v2 prediction
derive elapsed/duration/RTF
print canonical JSON summary
close backend in finally
```

Default fixture: `tests/fixtures/oaf_tf1_smoke/canonical.wav`. Default output: `artifacts/benchmark/oaf-smoke/prediction.jsonl`.

Summary keys exactly:

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

- [ ] **Step 5: Add `.github/workflows/oaf-smoke.yml`**

One `workflow_dispatch` job does checkout, uv setup, checkpoint preparation, one Docker build tagged `crux-oaf-tf1:local`, one smoke invocation, writes parsed summary fields to `$GITHUB_STEP_SUMMARY`, and uploads only the prediction JSONL.

- [ ] **Step 6: Run focused and full validation**

```bash
uv run pytest tests/test_cli_benchmark.py tests/benchmark/test_oaf_smoke_workflow.py -q
uv run pytest
uv run ruff check src tests
uv run ruff format --check src tests
```

- [ ] **Step 7: Commit**

```bash
git add src/cli/benchmark.py tests/test_cli_benchmark.py \
  .github/workflows/oaf-smoke.yml tests/benchmark/test_oaf_smoke_workflow.py
git commit -m "feat: add minimal OaF smoke workflow"
```

---

### Task 8: Final old-estate scan, repository verification, and runtime build

**Files:**
- Delete any explicitly deferred Task 1 HPA-320 extraction file that Task 6 made unreachable
- Delete remaining seal/attestation/calibration tests/goldens with no active owner
- Modify: `docs/drumery-dtx-midi-benchmarking-reference.md` where it describes removed active commands

- [ ] **Step 1: Run final active-code grep**

```bash
git grep -n -E \
  'BackendVerification|TensorCoverageCheck|SmokeCheck|SealState|BackendNotSealed|heuristic-onset-v1|backend_lock|backend_reports|backend_attestation|backend_prepare|backend_publication|backend_process|tools\.hpa320|hpa320-native|calibration_(entrypoint|protocol)|crux.drum-prediction-events/v1' \
  -- src runtime tests .github tools
```

Expected: no active match. If a match is an inference-relevant retained behavior, move it to the already-defined owner rather than adding a compatibility shim.

- [ ] **Step 2: Update active benchmark documentation**

Replace active seal/verify instructions with:

```bash
uv run crux benchmark prepare-backend --backend oaf --download
docker build -f runtime/oaf_tf1/Dockerfile -t crux-oaf-tf1:local .
uv run crux benchmark smoke-backend --backend oaf
```

Historical `docs/superpowers/` records may remain unchanged.

- [ ] **Step 3: Run focused backend suite**

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
  tests/benchmark/test_worker_process.py \
  tests/benchmark/test_oaf_adapter.py \
  tests/benchmark/test_oaf_smoke_workflow.py \
  runtime/oaf_tf1/tests/test_model.py \
  runtime/oaf_tf1/tests/test_worker.py \
  tests/test_cli_benchmark.py -q
```

- [ ] **Step 4: Run repository gates**

```bash
uv run pytest
uv run ruff check src tests
uv run ruff format --check src tests
uv run pylint --errors-only --disable=E1120,E0401 src
git diff --check main...HEAD
```

Record only a pre-existing baseline failure when the failing bytes are unchanged from `main`; do not widen HPA-423 into unrelated cleanup.

- [ ] **Step 5: Build final runtime image**

```bash
docker build -f runtime/oaf_tf1/Dockerfile -t crux-oaf-tf1:local .
```

Expected: success; image starts the new worker.

- [ ] **Step 6: Commit any final cleanup/docs change**

```bash
git add -A
git commit -m "refactor: finish OaF seal estate removal"
```

Skip the commit if Task 8 produces no file changes after verification.

---

### Task 9: Run and review the authentic released-checkpoint smoke

**Files/outputs:**
- Workflow: `.github/workflows/oaf-smoke.yml`
- Uploaded artifact: `artifacts/benchmark/oaf-smoke/prediction.jsonl`
- Operational evidence: GitHub job summary only

- [ ] **Step 1: Dispatch `OaF Smoke` manually on the exact implementation commit**

No workflow custom input is supplied; select the implementation branch/ref through GitHub's standard workflow UI.

- [ ] **Step 2: Verify summary invariants**

Require:

```text
backend_id = magenta-egmd-tf1-94529798-8hit-v1
upstream_source_commit = 94529798dfbbb14c27ddfd76f23027dc8e2ce185
checkpoint_archive_sha256 = 09765ae0ff19c7d769a3c20e158eba3b9cd279429b02e498b1e911d16f82e2c0
restored_tensor_count = 78
mapped_event_count > 0
prediction_sha256 is 64 lowercase hex digits
elapsed_seconds > 0
real_time_factor > 0
```

- [ ] **Step 3: Validate the downloaded artifact through the production reader**

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

Post the successful run URL, exact commit SHA, restored tensor count, mapped/unmapped counts, and RTF to Linear. Mark HPA-423 Done only after both the real OaF-shaped hihat/tom artifact-v2 tests and authentic smoke pass.

---

## Dependency and Parallelism Notes

- Tasks 1-4 can proceed without HPA-324 production code.
- Task 5 is the first hard HPA-324 seam and consumes `PredictionMap.backend_id` plus the reviewed OaF map.
- Task 6 requires Tasks 2-5 and is intentionally one atomic breaking cutover because the shared contract, adapter, registry, persistence, scorer bridge, and old live consumers must change together.
- Tasks 7-9 require the complete cutover.
- One implementation branch with task-sized commits is sufficient; do not create child infrastructure merely to parallelize the work.

## Self-Review

- **Spec coverage:** Tasks 1-9 cover every revised HPA-423 acceptance criterion.
- **Incremental health:** no task commit intentionally leaves live imports/constructors broken; the broad contract change is isolated to the atomic Task 6 cutover.
- **Codebase drift:** Task 2 extracts generic helpers now used outside sealing, updates the prediction publisher's removed `role=` call shape immediately, and rewrites the existing `checkpoint_acquisition.py` rather than creating a parallel path.
- **HPA-324 identity:** Task 5 validates `PredictionMap.backend_id` against descriptor `backend_id`; checkpoint `model_id` is explicitly different and never used as the map selector.
- **HPA-324 ownership:** Task 5 consumes the reviewed prediction map and never defines a duplicate table.
- **Prediction correctness:** common-only hi-hat/tom and unmapped sticks are hard v2 round-trip tests before HPA-326.
- **Scorer semantics:** Task 6 locks `BenchmarkEvent.canonical_class` to the common scorer level and preserves detailed class only in metadata for HPA-325.
- **Runtime:** one persistent sequential worker; no premature parallelism or retry framework.
- **Compatibility:** no seal compatibility or prediction-v1 reader is planned.
- **Security scope:** no new attestation/signing/security-only subsystem is planned.
- **Placeholder scan:** every implementation/test step names concrete inputs, assertions, commands, or exact output semantics; no implementation step depends on a later unspecified repair.
