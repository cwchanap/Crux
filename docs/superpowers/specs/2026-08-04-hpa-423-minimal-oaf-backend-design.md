# HPA-423: Minimal Reusable OaF Backend Design

**Issue:** HPA-423  
**Status:** Revised for implementation  
**Originally drafted:** 2026-08-04  
**Revised:** 2026-08-10  
**Planning baseline:** `main` at `b4aa8060e86b64ddfd211a46c81f4522d4a69323`

## Summary

Replace the unfinished HPA-320 sealed OaF prototype with the smallest reusable backend path that can load the authentic released Magenta OaF Drums checkpoint, keep one model instance alive across repeated requests, persist scoreable mapped predictions, and support the later HPA-326 corpus run.

This is a breaking internal replacement. There are no external compatibility commitments. Development speed, low operating cost, clear module boundaries, and easy integration of later transcription models matter more than supply-chain hardening, release attestation, compatibility readers, or exhaustive recovery behavior.

The implementation is primarily **subtraction and extraction**:

- keep the working OaF graph/checkpoint/event-conversion code;
- keep the useful persistent-process shape;
- keep the four existing audio identity fields;
- keep the existing checkpoint hashes and 78-tensor restoration gate;
- keep the existing prediction artifact as the persistence seam, but advance it to v2;
- remove sealing, attestation, verification reports, lock state, heuristic fallback, calibration publication, and protocol hardening that are not needed for a hobby benchmark.

The replacement is staged so each committed task remains import-clean and repository-testable. Independently dead outer seal estate is removed first, but shared types/modules that still have live consumers are cut over only once their replacements exist. This staging is not backward compatibility: no old interface or artifact reader remains in the final state.

## Current Repository State

The planning baseline already contains most of the raw mechanisms, but they are trapped behind the old seal lifecycle.

### The official OaF backend cannot run through the registry

`src/benchmark/backend_registry.py` registers `magenta-egmd-tf1-94529798-8hit-v1` as `seal_state="preseal"` and raises `BackendNotSealed` before creating it. There is no functioning sealed fallback to preserve.

The same registry still exposes `heuristic-onset-v1`, even though its implementation module does not exist. The heuristic ID and its special validation/nullability rules are dead weight and are removed entirely.

### The shared backend contract is seal-shaped

`src/benchmark/backends/base.py` currently contains:

- `verify()`;
- `BackendVerification`;
- `TensorCoverageCheck`;
- `SmokeCheck`;
- backend/runtime/parameter/model-artifact lock hashes;
- shared publication/report types.

Those responsibilities do not belong in a reusable transcription interface.

### The reusable OaF implementation already exists, but is tangled

`runtime/oaf_tf1/oaf_backend.py` contains the real graph construction, checkpoint restoration, feature extraction, inference, emitted-frame handling, post-processing, and native-event conversion. `REQUIRED_INFERENCE_COUNT` is already 78.

`runtime/oaf_tf1/entrypoint.py`, `runtime/oaf_tf1/protocol.py`, and `src/benchmark/backend_process.py` already prove the persistent-worker shape, but include strict environment equality, sealed identities, host evidence, compatibility/schema enforcement, diagnostic redaction, and a large process state machine that HPA-423 does not need.

The current Dockerfile also copies HPA-320 workflows/tools/manifests into the runtime image for attestation. Those copies are removed. The existing Magenta instrumentation patch used by actual native event conversion is inference-relevant and remains.

### Prediction persistence is almost the right seam

`src/benchmark/prediction_artifact.py` already persists native identity and reserves:

- `canonical_class`;
- `mapping_status`;
- `prediction_map_version`.

But v1 hard-codes those fields to `canonical_class=null`, `mapping_status="not_applied"`, and `prediction_map_version=null`.

`src/benchmark/scorer_input.py` therefore reads the artifact and unconditionally raises `CanonicalMappingRequired`.

HPA-324 has now made this gap concrete: OaF `hihat` and `toms` map only to the common scoring classes `hihat` and `tom`. A prediction artifact that persists only `canonical_class` cannot represent those valid mapped events. HPA-423 therefore owns the prediction serialization/conversion side of the HPA-324 seam.

### One cleanup detail changed since the first HPA-423 draft

The old issue text listed `src/benchmark/backend_publication.py` for unconditional deletion. That is no longer safe on current `main`: generic helpers from that file are imported by `input_view.py`, `prediction_artifact.py`, and `checkpoint_acquisition.py`.

HPA-423 will extract only the still-needed generic file/publication helpers into a neutral small module and then delete the seal-oriented remainder. This is the only material cleanup exception introduced by post-draft codebase drift.

## Decision Drivers

1. **Make OaF runnable now.** The benchmark should not wait for a publication/sealing ceremony that provides no value to this project.
2. **Reuse proven inference code.** Moving working model logic is safer and faster than reimplementing Magenta OaF behavior.
3. **Keep model load persistent.** Reloading TensorFlow for every song would make the roughly 400-song corpus unnecessarily slow.
4. **Keep the shared contract small.** Later MuScriptor/IDM adapters must not inherit OaF seal concepts.
5. **Persist scoreable predictions once.** HPA-326 must not need to rerun expensive inference because class mapping was omitted from the artifact.
6. **Scale only from measurements.** One sequential worker is the default until HPA-326 runtime evidence shows otherwise.
7. **Prefer direct failures over policy frameworks.** Expected development failures should be understandable without custom attestation/error taxonomies.
8. **Keep intermediate commits healthy.** A task may use focused RED/GREEN tests while being developed, but its commit must not intentionally leave live imports or the repository suite broken for a later task to repair.

## Target Architecture

```text
                     HPA-324 taxonomy/map policy
                              |
                              v
CanonicalAudio -> OafBackend -> NativePrediction -> OaF prediction mapper
                    |                              |
                    |                              v
                    |                   prediction artifact v2
                    |                              |
                    |                              v
                    |                       scorer_input.py
                    |
                    v
            WorkerProcess (private)
                    |
                    v
            runtime/oaf_tf1/worker.py
                    |
                    v
            runtime/oaf_tf1/model.py
                    |
                    v
      released OaF checkpoint + vendored Magenta
```

The host/backend boundary ends at native prediction. Taxonomy policy stays outside the TensorFlow worker.

## 1. Remove the Seal Lifecycle from the Active Architecture

Cleanup is **outer-estate first, shared-contract cutover once**.

The HPA-320 workflows/tools/calibration/publication pieces that have no retained live import are removed as soon as import/call-site search proves them unreachable. By contrast, shared seal-shaped types/modules such as `NativePrediction` lock fields, `prediction_artifact.py`, the old OaF adapter, `transcription.py`, and `backend_process.py` remain temporarily only while they are still live imports. They are removed together in one breaking cutover after the replacement model/worker/adapter/mapping components exist.

This avoids two bad outcomes at once:

- preserving the old architecture as compatibility baggage; and
- committing a branch where ordinary imports/full tests stay red for several later tasks.

### Delete outright when unreachable

After import/call-site search proves each item is no longer referenced by retained code, remove:

- `.github/workflows/hpa320-native-bootstrap.yml`;
- `.github/workflows/hpa320-native-candidate.yml`;
- `.github/workflows/hpa320-native-measurement.yml`;
- `tools/hpa320/`;
- `src/benchmark/backend_lock.py`;
- `src/benchmark/backend_reports.py`;
- `src/benchmark/backend_attestation.py`;
- the report/verification-heavy `src/benchmark/transcription.py` path and its old CLI commands;
- runtime calibration/candidate/attestation manifests and entrypoints that are not required by inference;
- seal/schema goldens and tests that exist only for the removed lifecycle;
- the old `src/benchmark/backends/oaf_tf1.py` after the new adapter owns active OaF creation;
- the old seal-oriented `src/benchmark/backend_process.py` after the new small worker-process helper owns active process control.

Git history is the archive. No compatibility readers remain.

### Extract before deleting

Two old modules contain useful non-seal mechanics.

#### `backend_publication.py`

Create a neutral `src/benchmark/artifact_io.py` containing only the retained generic responsibilities currently used outside sealing:

- `DirectoryAnchor` only if an active caller still needs it after simplification;
- `read_regular_file_no_follow()`;
- a small immutable-file publisher used by prediction artifacts.

Move `PublishedArtifact` out of the backend contract if a publisher result type is still useful. If it becomes the simple `(path, sha256)` result, update `publish_prediction_artifact()` in the same extraction so no retained call keeps the old `role=` or seal-specific anchor arguments.

Do not carry over directory transaction, attestation, rollback, or seal-publication machinery unless an active retained caller demonstrably requires it.

The old `backend_publication.py` may remain temporarily for old HPA-320 callers until the atomic backend cutover; the final active callers must use `artifact_io.py` only.

#### `checkpoint_acquisition.py` and `backend_prepare.py`

`src/benchmark/checkpoint_acquisition.py` already exists and already owns the released archive/component identities plus request/evidence loading. Do **not** create a second acquisition path beside it.

Simplify the existing `checkpoint_acquisition.py` in place around `runtime/oaf_tf1/model.json`, reusing only the useful download/extract/hash mechanics from `backend_prepare.py`. Remove request/evidence/backend-lock/publication concepts from the retained path. Delete `backend_prepare.py` only after its old OaF caller is removed at cutover.

The new path verifies:

- archive SHA-256;
- exact required archive members;
- each published checkpoint component SHA-256.

It does not publish acquisition evidence, use a backend lock, or implement atomic authority transactions.

## 2. Minimal Shared Backend Contract

Keep `src/benchmark/backends/base.py`, but reduce it to transcription-domain types **at the atomic backend cutover**, not earlier while live consumers still construct/read the seal-shaped `NativePrediction`.

The cutover commit updates or removes every active constructor/reader/import of the removed fields and verification types in the same change. No committed task intentionally leaves `prediction_artifact.py`, `transcription.py`, the old OaF adapter, or process/report modules importing removed names.

### `CanonicalAudio`

Keep the existing four identity fields unchanged:

```python
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
```

These identities are already needed by HPA-328 input-view ablations and HPA-325 score grouping. They must not be collapsed into one hash.

### `NativeEvent`

Keep the current native event fields:

```python
@dataclass(frozen=True)
class NativeEvent:
    time_sec: float
    native_class_id: str
    model_output_bin: int | None
    native_midi_note: int | None
    native_metadata: Mapping[str, str | None]
    confidence: float | None
    velocity_midi: int | None
```

For OaF, native identity remains `native_class_id="midi_<note>"`; the 8-hit semantic group remains `native_metadata["upstream_8hit_group_id"]`.

### `NativePrediction`

Reduce to:

```python
@dataclass(frozen=True)
class NativePrediction:
    audio: CanonicalAudio
    descriptor: BackendDescriptor
    events: tuple[NativeEvent, ...]
```

No lock hashes or publication state are carried in the backend result.

### `TranscriptionBackend`

```python
class TranscriptionBackend(Protocol):
    def descriptor(self) -> BackendDescriptor: ...
    def transcribe(self, audio: CanonicalAudio) -> NativePrediction: ...
    def close(self) -> None: ...
```

Remove `verify()` entirely.

Backend-specific startup failures use direct adapter exceptions. There is no shared verification lifecycle.

## 3. Keep Backend Identity, Strip Seal Identity

Retain `BackendDescriptor` and its canonical SHA-256 because it is a compact model identity that HPA-326 can persist and compare without a generic plugin framework.

Advance the OaF descriptor to a breaking v2 exact shape containing only model/prediction identity:

```text
architecture_id
backend_id
model_id
native_metadata_schema_id
native_output_space_id
prediction_schema
training_data_map_id
upstream_source_commit
descriptor_schema
```

Remove:

- backend lock hash;
- runtime lock hash;
- runtime image digest;
- model artifact set hash;
- protocol schema;
- heuristic descriptor/schema/constants.

The worker protocol is private and changes with the host adapter, so it is not part of stable backend identity.

`OAF_BACKEND_ID` remains the full stable identity `magenta-egmd-tf1-94529798-8hit-v1`. Registry/CLI selection may use the short key `oaf`; the descriptor always records the full ID.

## 4. One OaF Model Configuration

Replace the preseal acquisition-request/lock constellation with one small data file:

```text
runtime/oaf_tf1/model.json
```

It is the single repository source for stable OaF model inputs:

```json
{
  "schema": "crux.oaf-model/v1",
  "backend_id": "magenta-egmd-tf1-94529798-8hit-v1",
  "model_id": "magenta-egmd-ckpt-569400-v1",
  "architecture_id": "magenta-oaf-model-tpu-drums-v1",
  "upstream_source_commit": "94529798dfbbb14c27ddfd76f23027dc8e2ce185",
  "training_data_map_id": "magenta-egmd-data-8hit-94529798-v1",
  "native_output_space_id": "magenta-oaf-midi88-a0-v1",
  "native_metadata_schema_id": "magenta-oaf-native-metadata-v1",
  "checkpoint": {
    "url": "https://storage.googleapis.com/magentadata/models/onsets_frames_transcription/e-gmd_checkpoint.zip",
    "archive_name": "e-gmd_checkpoint.zip",
    "archive_sha256": "09765ae0ff19c7d769a3c20e158eba3b9cd279429b02e498b1e911d16f82e2c0",
    "components": {
      "model.ckpt-569400.data-00000-of-00001": "6312cd6f8e29ee6f21f4a763822d554f46393e0e336812ba4a8cb324d2e8b0b5",
      "model.ckpt-569400.index": "475ca21993e102d52e2e6c7066b598b9b9e9a097bfb2fa6b770d40eedaa6139a",
      "model.ckpt-569400.meta": "e36105b27c17f2367292682df825b8ad16eb887ccad465bbb5ddf955ef75d422"
    }
  }
}
```

The existing acquisition request is migrated into this file rather than kept as a parallel source.

`runtime/oaf_tf1/model.py` owns the small config loader. TensorFlow imports remain lazy so host-side checkpoint preparation can reuse the loader without importing TensorFlow.

## 5. OaF Model Engine

Create `runtime/oaf_tf1/model.py` by moving inference-relevant code out of `oaf_backend.py`.

It owns only:

- loading `model.json`;
- building the intended vendored Magenta OaF graph;
- restoring the released checkpoint;
- proving all 78 required inference tensors restore;
- audio feature extraction;
- inference;
- upstream post-processing;
- emitted-frame handling required for event timing;
- conversion to native OaF events.

Retain the vendored Magenta source and the existing instrumentation patch needed by the working inference/event path. Remove source-attestation manifests and runtime checks whose only purpose was to prove sealed build provenance.

Model startup returns a small identity/status object containing the descriptor and restored tensor count. It does not return a verification report.

The real TensorFlow checkpoint is not used by ordinary unit tests.

## 6. Persistent Sequential Worker

Create `runtime/oaf_tf1/worker.py` from the useful parts of `entrypoint.py` and `protocol.py`.

The worker lifecycle is:

1. load the model once;
2. write one ready JSON line;
3. read one request JSON line at a time;
4. transcribe the requested mounted canonical WAV;
5. write one success or error JSON line;
6. exit when stdin reaches EOF.

Example ready response:

```json
{"type":"ready","backend_id":"magenta-egmd-tf1-94529798-8hit-v1","restored_tensor_count":78}
```

Example request:

```json
{"id":"song-1","audio_path":"song-1.wav"}
```

Example success:

```json
{
  "id": "song-1",
  "events": [
    {
      "time_sec": 0.42,
      "native_class_id": "midi_38",
      "model_output_bin": 17,
      "native_midi_note": 38,
      "upstream_8hit_group_id": "snare",
      "confidence": 0.91,
      "velocity_midi": 97
    }
  ]
}
```

Example error:

```json
{"id":"song-1","error":{"code":"inference_failed","message":"inference failed"}}
```

The protocol deliberately has no version negotiation, compatibility decoder, signatures, resume sequence, policy object, canonical-byte requirement, or diagnostic redaction layer.

Keep basic request IDs and malformed-response detection because they make failures understandable at negligible complexity.

## 7. Small Host Process Wrapper and OaF Adapter

Add a new small private line-JSON process helper alongside the old seal-oriented `backend_process.py` while the old adapter still imports it. The target helper may be named `worker_process.py`; it becomes the only process helper after cutover and the old `backend_process.py` is deleted.

The new helper only needs to:

- start the worker/container with stdin/stdout/stderr pipes;
- read the ready line;
- write one request and read one response while holding a simple lock;
- surface EOF/malformed JSON/nonzero startup as direct errors;
- close stdin and terminate/kill only if normal shutdown does not finish promptly.

No worker pools or concurrent in-flight requests are added.

Create `src/benchmark/backends/oaf.py` at the atomic backend cutover. It owns:

- checking that `CanonicalAudio.path` is under the configured input root;
- starting the fixed local OaF image/container;
- mounting the verified checkpoint directory read-only;
- mounting the input root read-only;
- waiting for the ready response;
- verifying the ready backend ID and restored tensor count;
- sending transcription requests;
- decoding native OaF events;
- returning `NativePrediction`;
- closing the persistent worker.

The adapter does not own taxonomy policy, corpus traversal, retries, resume logic, score calculation, or prediction storage layout.

## 8. Static Backend Selection, No Heuristic

At the atomic backend cutover, reduce `backend_registry.py` to a small static factory map.

Conceptually:

```python
BACKEND_FACTORIES = {
    "oaf": create_oaf_backend,
}
```

Unknown keys fail directly. There is no `SealState`, `BackendNotSealed`, integrity gate, heuristic fallback, legacy TF2 registry entry, plugin discovery, entry-point loading, or service container.

The FastAPI app's separate TF2 fallback in `src/app/transcriber.py` is unrelated and remains out of scope.

## 9. HPA-324 Prediction Mapping Seam

HPA-324 owns:

- `src/benchmark/taxonomy.py`;
- detailed/common taxonomy;
- `ClassMapping` / `PredictionMap` data structures;
- `PredictionMap.backend_id` as the backend/native-output identity selector;
- `OAF_PREDICTION_MAP_ID = "crux.prediction-map/oaf-egmd-8hit-v1"`;
- the frozen OaF group-to-class table.

HPA-423 owns:

- applying the OaF map to native predictions;
- validating the map applies to the descriptor;
- carrying map identity into every persisted event;
- prediction artifact v2 serialization/deserialization;
- scorer input conversion.

Do not duplicate HPA-324 taxonomy constants in HPA-423.

### Map identity rule

Before lookup, HPA-423 validates exactly:

```python
prediction.descriptor.payload["backend_id"] == prediction_map.backend_id
prediction.descriptor.payload["native_output_space_id"] == prediction_map.native_output_space_id
```

The checkpoint descriptor field `model_id=magenta-egmd-ckpt-569400-v1` is separate and is **not** compared to `PredictionMap.backend_id`.

### OaF lookup key

The OaF mapper uses:

```python
key = event.native_metadata["upstream_8hit_group_id"]
```

It does not use `native_class_id` as the semantic key because that field intentionally remains the source MIDI identity.

### Mapped event domain

Add a persistence-side mapped event type that retains the complete native event plus mapping result:

```python
@dataclass(frozen=True)
class MappedPredictionEvent:
    native: NativeEvent
    canonical_class: str | None
    common_class: str | None
    mapping_status: Literal["mapped", "unmapped"]
    prediction_map_version: str
```

Mapping produces diagnostics with unmapped counts. Unmapped events remain in the persisted artifact; they are never silently discarded.

## 10. Prediction Artifact v2

Advance the active schema to:

```text
crux.drum-prediction-events/v2
```

Keep the existing JSONL header/event/terminal shape, but simplify the header by removing seal-only fields and add `common_class` to every event row.

Required event semantics are exactly:

```text
mapped:
  prediction_map_version != null
  common_class != null
  canonical_class may be null

unmapped:
  prediction_map_version != null
  common_class == null
  canonical_class == null
```

Native event fields remain unchanged:

```text
native_class_id
model_output_bin
native_midi_note
native_metadata
confidence
velocity_midi
time_sec
```

The artifact must round-trip a real OaF-shaped hi-hat event with `common_class="hihat"` and a real OaF-shaped tom event with `common_class="tom"` while preserving their native MIDI/group identities.

There is no v1 compatibility reader. Breaking internal schema replacement is intentional.

## 11. Scorer Input

Replace the unconditional exception in `scorer_input.py`.

`read_scorer_events()` reads prediction artifact v2 and returns only mapped events as `BenchmarkEvent` instances using the persisted **common class** as the scorer class.

The existing `BenchmarkEvent.canonical_class` field is therefore a compatibility-shaped scorer field name: on this prediction bridge it contains the **common comparison class**, not HPA-324's detailed canonical class. The detailed value, when present, is retained only as `metadata["detailed_canonical_class"]`.

HPA-325 must treat prediction `BenchmarkEvent.canonical_class` and reference `CommonReferenceEvent.common_class` as the same headline class level. HPA-325 must not reinterpret `BenchmarkEvent.canonical_class` as the detailed taxonomy or derive common class from it.

The scorer path has no OaF-specific branch. Native/detailed identities are retained in metadata for diagnosis.

Unmapped events are not fabricated into a score class; their presence remains visible in the prediction artifact and mapping diagnostics.

## 12. Checkpoint Preparation Command

Keep the existing `prepare-backend` command name but simplify its contract around the rewritten-in-place `src/benchmark/checkpoint_acquisition.py`.

Target usage:

```bash
uv run crux benchmark prepare-backend --backend oaf --download
```

Default cache root:

```text
artifacts/benchmark/model-cache
```

The command:

1. loads `runtime/oaf_tf1/model.json`;
2. downloads the released archive when requested or verifies an existing cache;
3. checks archive and component hashes;
4. reports the verified checkpoint directory;
5. exits nonzero on download/hash/extraction failure.

Remove acquisition-evidence output, backend-lock input, seal terminology, and report publication.

## 13. Smoke Command

Replace the old `verify-backend` / seal-report validation path with:

```bash
uv run crux benchmark smoke-backend --backend oaf
```

The command uses fixed/default development locations rather than a large option surface:

- fixture: `tests/fixtures/oaf_tf1_smoke/canonical.wav`;
- checkpoint cache: `artifacts/benchmark/model-cache`;
- image tag: `crux-oaf-tf1:local`;
- output: `artifacts/benchmark/oaf-smoke/prediction.jsonl`.

It:

1. loads the canonical fixture through the normal `CanonicalAudio` path;
2. creates the normal OaF adapter;
3. starts one persistent worker;
4. performs one real transcription;
5. maps the native events through HPA-324's OaF prediction map;
6. requires at least one mapped valid event;
7. persists prediction artifact v2;
8. computes elapsed seconds, audio duration, and real-time factor;
9. prints one machine-readable summary;
10. closes the backend in `finally`.

Real-time factor is:

```text
elapsed_inference_seconds / fixture_duration_seconds
```

The fixture is small, so this number is indicative only. HPA-326 owns the meaningful corpus projection.

## 14. Manual GitHub Smoke Workflow

Add one workflow:

```text
.github/workflows/oaf-smoke.yml
```

Requirements:

- `workflow_dispatch` only;
- no custom inputs;
- one `ubuntu-24.04` job;
- normal checkout and uv setup;
- prepare/download verified checkpoint once;
- build the OaF image once with tag `crux-oaf-tf1:local`;
- run `smoke-backend` once;
- upload only the prediction JSONL artifact;
- write a job summary.

The summary records:

- Git commit and ref;
- backend ID;
- upstream source revision;
- checkpoint archive SHA-256;
- restored tensor count;
- fixture SHA-256;
- mapped event count;
- unmapped event count;
- prediction SHA-256;
- elapsed seconds;
- real-time factor.

No OIDC, Sigstore, GitHub artifact attestation, deterministic OCI export, second image build, calibration matrix, host fingerprint, or custom `summary.json` artifact is added.

## Error Handling

Handle only expected development failures:

- missing/malformed `model.json`;
- checkpoint download failure;
- archive/component hash mismatch;
- graph/checkpoint load failure;
- fewer than 78 required inference tensors restored;
- worker startup/early exit;
- fixture/audio decode failure;
- malformed worker response;
- inference failure;
- invalid prediction mapping result;
- prediction artifact validation/write failure.

Use direct exceptions with useful messages. CI may preserve normal tracebacks.

Do not add retries, recovery journals, diagnostic redaction, partial-artifact repair, signal/OOM taxonomies, cgroup handling, or filesystem race policy unless a real recurring failure later requires them.

## Testing Strategy

Keep ordinary tests fast; the real checkpoint runs only in the manual smoke workflow.

### Contract/registry tests

- backend protocol has only descriptor/transcribe/close after the cutover;
- four `CanonicalAudio` identity fields remain;
- registry selects `oaf`;
- unknown backend fails;
- no seal/heuristic constants remain after the cutover.

### Model/config tests

- known `model.json` loads expected model/checkpoint identities;
- obvious malformed config/hashes fail;
- representative native event conversion retains MIDI/group identity;
- missing required tensors fails clearly.

### Worker tests

- one mocked model instance is created once;
- two sequential requests use the same model;
- malformed JSON returns a direct error;
- stdin EOF exits cleanly.

### Host/process tests

- fake worker ready/success/error flow;
- multiple sequential requests over one process;
- shutdown closes cleanly;
- malformed ready/response fails directly.

### Mapping/artifact tests

- map backend/native-output identity is validated before lookup;
- kick/snare mapped event;
- OaF hi-hat maps to `canonical_class=None`, `common_class="hihat"`;
- OaF tom maps to `canonical_class=None`, `common_class="tom"`;
- `sticks` persists as unmapped;
- every event records `prediction_map_version`;
- native fields survive write/read exactly;
- scorer input uses persisted common class without an OaF branch;
- scorer bridge stores detailed canonical identity only in metadata.

### CLI/workflow tests

- `prepare-backend` option surface no longer accepts seal inputs;
- `smoke-backend` works with a fake backend and writes one prediction artifact/summary;
- workflow has one job, no custom dispatch inputs, no attestation permissions, one build, one smoke invocation, and one uploaded prediction artifact.

### Commit health

Every task commit runs its focused tests and a repository import/full-test gate appropriate to the touched surface. A task may not intentionally defer broken imports/constructors to a later task. The atomic cutover is one task precisely because the contract, adapter, registry, prediction persistence, scorer bridge, and old live consumers change together.

## Implementation Order

The implementation order is intentionally different from the original August 4 draft and from the earlier over-aggressive contract-first revision.

1. **Prune independently dead outer HPA-320 estate and heuristic-only surfaces without changing shared backend/prediction types that still have live consumers.**
2. **Extract generic artifact I/O and simplify the existing `checkpoint_acquisition.py` in place around `model.json`.** Keep old `backend_publication.py`/`backend_prepare.py` only while old live callers still need them.
3. **Extract the OaF model engine and simplify the container build context.** Keep the old runtime backend only as extraction source until cutover.
4. **Add the new small persistent worker and a new small host worker-process helper alongside the old seal-oriented process controller.**
5. **Add the pure HPA-324 OaF prediction mapper against the still-existing `NativePrediction` fields it actually consumes.** Do not rewrite persistence yet.
6. **Perform one atomic backend cutover:** shrink shared backend types/descriptor, create and activate `backends/oaf.py`, reduce registry to `oaf`, advance prediction artifact to v2, un-stub scorer input, update the prediction publisher to neutral artifact I/O, remove old `transcription.py`/OaF adapter/process/lock/report/attestation/preparation/publication live paths, and update all affected tests in the same commit.
7. **Add simplified checkpoint/smoke CLI commands and the one manual workflow.**
8. **Perform final import/call-site cleanup, repository verification, and final image build.**
9. **Run the real released-checkpoint smoke.** If it fails, repair the new path. Do not restore the abandoned seal path.

This preserves subtraction-first intent without using knowingly broken intermediate commits as an implementation technique.

## HPA-324 Coordination

HPA-423 and HPA-324 may be implemented in parallel, but their merge boundary is explicit:

- HPA-324 can implement taxonomy/reference eligibility independently.
- HPA-423 can implement outer cleanup, config/checkpoint, model, and worker independently.
- HPA-423's mapper requires the HPA-324 `PredictionMap.backend_id`/native-output contract and OaF map data.
- HPA-423's artifact-v2/scorer cutover consumes that mapper in the same breaking cutover commit.
- HPA-326 must not start until the real OaF-shaped hi-hat/tom v2 round-trip passes and HPA-423's real smoke passes.

Do not duplicate temporary taxonomy/map constants in HPA-423 to avoid this coordination point.

## Downstream Ownership

### HPA-326 owns

- corpus iteration;
- per-song status/persistence orchestration;
- simple resume behavior;
- full-song/corpus runtime measurement;
- deciding whether multiple workers are justified.

It must reuse the HPA-423 OaF adapter, persistent worker, prediction mapper, and prediction artifact v2.

### HPA-325 owns

- scoring tolerances;
- per-class/per-song/aggregate reports;
- score cohort semantics.

It consumes HPA-423 scorer input and HPA-324 reference common projection. For prediction-side `BenchmarkEvent`, `canonical_class` means the **common scoring class**; any detailed canonical prediction identity is diagnostic metadata only. HPA-325 must not mix the detailed and common levels.

### HPA-328 owns

- Spleeter/HTDemucs input views and their comparisons.

The four audio identity fields preserved here are required for that work.

## Non-Goals

HPA-423 does not implement:

- full-corpus inference;
- resume or retry behavior;
- parallel/multi-worker inference;
- queues or distributed scheduling;
- threshold tuning;
- fine-tuning/retraining;
- a TensorFlow 2 replacement;
- scoring metrics or tolerance policy;
- reference taxonomy/eligibility policy;
- security attestation or release signing;
- deterministic OCI reproduction;
- host/cgroup/resource profiling;
- backward compatibility for seal or prediction-v1 artifacts;
- a generic backend plugin framework;
- changes to the FastAPI app's separate TF2 fallback.

## Acceptance Criteria

- [ ] Active shared backend types contain only backend identity, the four audio identity fields, native prediction events, transcription, and cleanup.
- [ ] The shared-contract removal lands atomically with every active constructor/reader/import updated or removed; no committed task intentionally leaves the live repository broken for a later task.
- [ ] `verify()`, `BackendVerification`, `TensorCoverageCheck`, `SmokeCheck`, shared lock hashes, seal state, and heuristic registry support are removed.
- [ ] `OAF_BACKEND_ID` remains `magenta-egmd-tf1-94529798-8hit-v1`; the short registry key `oaf` cannot fall back to another backend.
- [ ] Generic artifact I/O still required by active benchmark modules is extracted from `backend_publication.py`; the seal-oriented remainder is removed.
- [ ] `publish_prediction_artifact()` uses the neutral artifact publisher and no retained call requires the removed `PublishedArtifact.role`/`role=` contract.
- [ ] One `runtime/oaf_tf1/model.json` is the single source for OaF model/checkpoint identities.
- [ ] Existing `src/benchmark/checkpoint_acquisition.py` is simplified in place; no second checkpoint-acquisition implementation remains.
- [ ] Checkpoint archive SHA-256 and all required component hashes are verified without backend/seal locks or acquisition-evidence publication.
- [ ] OaF graph/checkpoint/inference code is isolated in `runtime/oaf_tf1/model.py` from GitHub, sealing, attestation, and corpus orchestration.
- [ ] All 78 required inference tensors restore before the worker announces ready.
- [ ] One persistent worker loads the model once and serves at least two sequential mocked requests in tests.
- [ ] One small host adapter returns native predictions and is the only active OaF registry implementation.
- [ ] OaF prediction mapping validates `PredictionMap.backend_id` against descriptor `backend_id` and validates `native_output_space_id`; descriptor checkpoint `model_id` is not misused as the map selector.
- [ ] Prediction artifact v2 persists `canonical_class`, `common_class`, `mapping_status`, and non-null `prediction_map_version` while preserving native event identity.
- [ ] Real OaF-shaped hi-hat and tom events round-trip with `common_class="hihat"` and `common_class="tom"` respectively.
- [ ] Unmapped OaF events remain persisted and counted rather than silently dropped.
- [ ] `scorer_input.py` returns mapped `BenchmarkEvent` values using persisted common class and contains no OaF-specific scoring branch.
- [ ] HPA-325 treats prediction `BenchmarkEvent.canonical_class` as the common scoring class and detailed prediction class only as diagnostic metadata.
- [ ] `crux benchmark prepare-backend --backend oaf --download` verifies/prepares the released checkpoint with the simplified contract.
- [ ] `crux benchmark smoke-backend --backend oaf` uses the normal adapter/worker, writes one v2 prediction artifact, and reports elapsed time plus real-time factor.
- [ ] `.github/workflows/oaf-smoke.yml` has no custom inputs, one Ubuntu 24.04 job, one image build, one real inference, one uploaded prediction artifact, and no attestation/signing path.
- [ ] The canonical smoke fixture produces at least one mapped valid event through the authentic released checkpoint.
- [ ] Final import/call-site search finds no active references to HPA-320 seal workflows, locks, reports, attestation, calibration publication, or heuristic backend support.
- [ ] No compatibility layer for removed seal interfaces/artifacts or prediction artifact v1 is added.
