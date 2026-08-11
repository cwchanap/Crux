# HPA-423: Minimal Reusable OaF Backend Design

**Issue:** HPA-423  
**Status:** Revised for implementation  
**Date:** 2026-08-10  
**Planning baseline:** `main` at `b4aa8060e86b64ddfd211a46c81f4522d4a69323`

## Summary

Replace the unfinished HPA-320 sealed OaF prototype with the smallest reusable backend path that can load the authentic released Magenta OaF Drums checkpoint, keep one model instance alive across repeated requests, persist scoreable predictions, and support HPA-326 corpus inference.

This is a breaking internal replacement. There are no external compatibility commitments. Development speed, low cost, clear ownership, and future model integration matter more than supply-chain attestation, compatibility readers, or exhaustive recovery behavior.

The implementation is primarily subtraction and extraction:

- reuse the working OaF graph, checkpoint restore, emitted-frame handling, and native-event conversion;
- preserve the released checkpoint/source identity and the 78 required inference-tensor gate;
- preserve the four existing `CanonicalAudio` source/input identity fields;
- keep one sequential persistent worker;
- advance prediction persistence to a scoreable common-class v2 artifact;
- remove sealing, locks, verification reports, attestation, calibration publication, heuristic fallback, and legacy protocol policy;
- retain one cheap exact-output oracle gate so extracting the TensorFlow path is verified by behavior, not just by “some event was produced.”

## Current Repository State

The current official path is not usable. `backend_registry.py` registers the official OaF backend as `preseal`, so registry creation stops before the implementation factory. `heuristic-onset-v1` is also registered even though its implementation module does not exist.

Most reusable mechanics already exist:

- `runtime/oaf_tf1/oaf_backend.py` owns real graph construction, feature extraction, checkpoint restore, post-processing, and native-event conversion;
- `REQUIRED_INFERENCE_COUNT` is already 78;
- `assert_no_reachable_stochastic_ops()` already verifies the prediction fetch graph has no reachable stochastic TensorFlow operation;
- `entrypoint.py` / `protocol.py` already prove a persistent worker shape;
- `backend_process.py` already proves the host can keep one subprocess alive across sequential requests;
- `prediction_artifact.py` already persists native identity and reserves mapping fields, but v1 forces them to `not_applied` / null;
- `scorer_input.py` therefore still raises `CanonicalMappingRequired` unconditionally.

The repository also contains an HPA-320 smoke-oracle schema and exact native-event comparison logic. The real oracle fixture referenced by `test_oaf_real_checkpoint.py` is not committed, so HPA-423 cannot simply reuse a completed oracle. It can, however, reuse the exact binary64 comparison idea to establish a new post-adapter oracle after the first successful smoke.

## Design Principles

1. **Subtraction over adaptation.** Do not make the seal path work; remove it.
2. **Extract, do not rewrite, OaF inference.** Keep the proven graph/checkpoint/event behavior.
3. **Every committed task stays green.** Focused tests may be red during TDD, but a commit must not intentionally depend on a later task to repair imports or constructors.
4. **One worker until measurement says otherwise.** HPA-326 owns meaningful corpus-scale performance work.
5. **One owner for taxonomy policy.** HPA-324 owns taxonomy/map data; HPA-423 applies and persists it.
6. **Persist scoreable predictions once.** HPA-326 must not rerun inference because `common_class` was omitted.
7. **Keep one behavioral extraction oracle.** A real model extraction must prove output equivalence, not merely non-empty output.
8. **Avoid hidden performance policy.** Container thread behavior and timer scope are explicit decisions.

## Target Architecture

```text
                         HPA-324 taxonomy.py
                                |
                                v
CanonicalAudio -> OafBackend -> NativePrediction
                    |                 |
                    |                 v
                    |          mapping.map_oaf_prediction
                    |                 |
                    |                 v
                    |          MappedPrediction
                    |                 |
                    |                 +--> prediction artifact v2
                    |                 +--> scorer_input.py
                    |
                    v
              WorkerProcess
                    |
                    v
        runtime/oaf_tf1/worker.py
                    |
                    v
         runtime/oaf_tf1/model.py
                    |
                    v
    released checkpoint + vendored Magenta
```

The TensorFlow worker emits native model information only. Taxonomy mapping remains host-side.

## 1. Implementation Staging

The implementation uses six reviewable tasks:

A. prune independently dead HPA-320 outer estate;
B. add neutral artifact I/O plus model-driven checkpoint preparation while old live interfaces remain intact;
C. extract the model engine and new worker/process helper alongside the old path;
D. perform one atomic breaking cutover of shared contract, registry, adapter, mapping, artifact v2, scorer bridge, Docker runtime, and remaining old consumers;
E. add `smoke-backend`, exact-oracle support, and one manual workflow;
F. run one smoke to establish the real oracle, commit it, then run a second oracle-gated smoke.

This is not backward compatibility. Temporary old interfaces exist only while still required by the branch itself and disappear in Task D.

## 2. Final Shared Backend Contract

Keep `CanonicalAudio` exactly with the current source/input identities:

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

Keep the current `NativeEvent` fields:

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

Final `NativePrediction` is only:

```python
@dataclass(frozen=True)
class NativePrediction:
    audio: CanonicalAudio
    descriptor: BackendDescriptor
    events: tuple[NativeEvent, ...]
```

Final backend protocol is only:

```python
class TranscriptionBackend(Protocol):
    def descriptor(self) -> BackendDescriptor: ...
    def transcribe(self, audio: CanonicalAudio) -> NativePrediction: ...
    def close(self) -> None: ...
```

Remove `verify()`, `BackendVerification`, tensor/smoke report types, lock hashes, and publication result types from the shared backend contract in the same atomic cutover that updates/removes every live consumer.

## 3. Backend Identity

Keep a compact OaF descriptor with exact fields:

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

Use:

```text
descriptor_schema = crux.transcription-backend-descriptor/v2
backend_id = magenta-egmd-tf1-94529798-8hit-v1
model_id = magenta-egmd-ckpt-569400-v1
prediction_schema = crux.drum-prediction-events/v2
```

Remove runtime/backend lock hashes, image digest, protocol schema, and heuristic descriptor support.

`backend_id` and checkpoint `model_id` are deliberately different identities.

## 4. One OaF Model Configuration

`runtime/oaf_tf1/model.json` becomes the only final repository source for stable OaF model/checkpoint inputs:

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
  "max_input_audio_frames": null,
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

`max_input_audio_frames` is explicitly `null` because no accepted measurement has established a trustworthy full-song ceiling. HPA-423 keeps canonical WAV validation but does not invent an unmeasured length cap. HPA-326 may add a measured bound if corpus evidence requires one.

During staging, old request JSON may remain only while old HPA-320 modules still import it. At the atomic cutover delete:

- `config/benchmark/backends/magenta-egmd-tf1-94529798-8hit-v1.checkpoint-acquisition-request.json`;
- `config/benchmark/backends/magenta-egmd-tf1-94529798-8hit-v1.calibration-bootstrap-request.json`;
- their request/evidence schema goldens and validators.

No duplicate identity source remains after Task D.

## 5. Checkpoint Preparation and Artifact I/O

Keep `src/benchmark/checkpoint_acquisition.py`; extend it during staging and simplify it in place at cutover. Do not create a second acquisition module.

Final retained behavior:

- load `model.json`;
- download or open the released ZIP;
- verify archive SHA-256;
- require exact checkpoint members;
- verify every published component SHA-256;
- cache the verified component directory;
- return the checkpoint path.

Delete acquisition evidence, backend locks, and authority publication.

Create `src/benchmark/artifact_io.py` with only the generic retained behavior:

```python
@dataclass(frozen=True)
class PublishedArtifact:
    path: Path
    sha256: str
```

plus path-only `read_regular_file_no_follow(path)` and immutable-file publication.

Do **not** recreate `DirectoryAnchor`. At the atomic cutover remove the unused anchor parameters from `input_view.load_direct_audio`, `input_view.load_derived_audio`, its private manifest reader, and `publish_prediction_artifact`. Old anchor-using HPA-320 callers are removed in the same cutover.

`publish_prediction_artifact()` moves to `publish_immutable_file(path, content)` in that cutover; no `role="prediction"` argument survives.

## 6. OaF Model Engine

Create `runtime/oaf_tf1/model.py` by extracting inference-relevant behavior from the current 1,600+ line `oaf_backend.py`.

Retain:

- vendored Magenta graph construction;
- released checkpoint restore;
- the exact 78 required inference-variable gate;
- feature extraction and post-processing;
- emitted-frame handling;
- native pitch/group/time/confidence/velocity conversion;
- `assert_no_reachable_stochastic_ops()` as a correctness gate on the prediction graph.

Do not move locks, host evidence, seal candidate logic, calibration publication, source attestations, or operational reports.

TensorFlow/Numpy/Magenta imports remain lazy so host-side import/tests do not require the TF1 runtime.

## 7. Persistent Worker and Container Runtime

Create one synchronous line-JSON worker. It loads `OafModel` once, writes one ready record, serves one request at a time, and exits on stdin EOF.

Keep request IDs and direct malformed/inference errors. Do not add protocol version negotiation, compatibility decoders, signatures, resumability, queues, or worker pools.

The final Docker runtime is explicit:

```dockerfile
ENV CUDA_VISIBLE_DEVICES=-1 \
    PYTHONHASHSEED=0

ENTRYPOINT ["/opt/crux/venv/bin/python", "-s", "/opt/crux/runtime/worker.py"]
```

Drop `/usr/bin/env -i` and the inherited `MKL_NUM_THREADS`, `OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS`, `TF_NUM_INTEROP_THREADS`, and `TF_NUM_INTRAOP_THREADS` single-thread caps. Those were seal reproducibility policy, not model-correctness requirements, and they would make the smoke RTF an accidentally single-threaded performance number.

Keep `CUDA_VISIBLE_DEVICES=-1` so this baseline remains CPU-only. `assert_no_reachable_stochastic_ops()` plus the exact smoke oracle provide the retained correctness checks.

Delete the Docker `test` target, `runtime/oaf_tf1/requirements-test.lock`, and `runtime/oaf_tf1/wheelhouse/test/` after the new tests are host-runnable.

New model/worker tests live under:

```text
tests/runtime/oaf_tf1/
```

rather than `runtime/oaf_tf1/tests/`, so the repository's existing `pytest.ini` / bare CI `pytest` automatically runs them without widening `testpaths` to arbitrary runtime files.

## 8. Host Process and OaF Adapter

Create a small `src/benchmark/worker_process.py` while the old controller still exists. It owns only subprocess lifecycle, ready line, one locked request/response exchange, timeout, and shutdown.

At cutover create `src/benchmark/backends/oaf.py` using that helper. It:

- verifies ready backend ID and restored tensor count 78;
- mounts the verified checkpoint read-only;
- mounts the input root read-only;
- sends an input-root-relative canonical WAV path;
- decodes native events;
- returns `NativePrediction`;
- reuses the same process across sequential calls.

No taxonomy, corpus traversal, retries, or scoring logic belongs in the adapter.

## 9. HPA-324 Mapping Seam

HPA-324 owns `PredictionMap` and the OaF group table. The planning owner contract uses:

```python
@dataclass(frozen=True)
class PredictionMap:
    map_id: str
    backend_id: str
    native_output_space_id: str
    classes: Mapping[str, ClassMapping]
```

HPA-423 validates:

```python
prediction.descriptor.payload["backend_id"] == prediction_map.backend_id
prediction.descriptor.payload["native_output_space_id"] == prediction_map.native_output_space_id
```

Do not compare the map selector to checkpoint `model_id`.

Put `map_oaf_prediction()` in the existing `src/benchmark/mapping.py` beside `map_dtx_events` / `map_midi_events`. Reuse the existing `MappingDiagnostics(unmapped=...)` convention.

The mapping API is:

```python
map_oaf_prediction(
    prediction: NativePrediction,
    prediction_map: PredictionMap = OAF_PREDICTION_MAP,
) -> tuple[MappedPrediction, MappingDiagnostics]
```

Keep `MappedPredictionEvent` / `MappedPrediction` in `prediction_artifact.py`, because those are the domain values serialized by that module. Do not add a parallel `PredictionMappingDiagnostics` / `PredictionMappingResult` abstraction.

OaF lookup remains solely:

```python
key = event.native_metadata["upstream_8hit_group_id"]
```

Unmapped events remain in the artifact and are counted.

## 10. Prediction Artifact v2 and Scorer Bridge

Advance prediction persistence to `crux.drum-prediction-events/v2`.

Keep header/event/terminal JSONL structure. Remove seal-only header fields and add `common_class`.

Mapped event:

```text
prediction_map_version != null
common_class != null
canonical_class may be null
mapping_status = mapped
```

Unmapped event:

```text
prediction_map_version != null
common_class = null
canonical_class = null
mapping_status = unmapped
```

Native fields remain unchanged.

`scorer_input.py` returns only mapped events. For prediction-side `BenchmarkEvent`, the existing field name `canonical_class` means the **common scoring class**. Detailed prediction identity, when available, lives only in:

```text
metadata["detailed_canonical_class"]
```

HPA-325 must score prediction `BenchmarkEvent.canonical_class` against HPA-324 `CommonReferenceEvent.common_class`; it must not reinterpret the prediction field as the detailed taxonomy.

## 11. Exact Smoke Oracle

A non-empty prediction is not a sufficient extraction test. HPA-423 keeps one exact post-adapter native-event oracle.

The old HPA-320 oracle concept is reused, but not its exact v1 wire shape: that schema includes `frame_index`, which is not part of the retained shared `NativeEvent` boundary. Carrying it forward would leak a worker-internal detail into the new host contract.

Define a compact `crux.oaf-smoke-oracle/v2` with:

```text
schema
backend_id
input_audio_sha256
native_events[]
```

Each oracle event contains exactly:

```text
time_sec_binary64
native_class_id
model_output_bin
native_midi_note
upstream_8hit_group_id
confidence_binary64
velocity_midi
```

Time and confidence are compared by exact IEEE-754 binary64 encoding. Other fields compare exactly. Event ordering/count also compare exactly.

The first successful real-checkpoint smoke runs without an oracle and produces the normal prediction artifact. Using a small production oracle renderer, derive `tests/fixtures/oaf_tf1_smoke/smoke-oracle.json` from that downloaded prediction and commit it. Then update the workflow to pass:

```text
--oracle tests/fixtures/oaf_tf1_smoke/smoke-oracle.json
```

A second run must report `oracle_status=matched`. HPA-423 is not Done until this second run passes.

The old `crux.oaf-smoke-oracle/v1` schema/golden is removed at cutover; v2 is the only surviving smoke-oracle schema.

## 12. Smoke Command and Timing

Add:

```bash
uv run crux benchmark smoke-backend --backend oaf [--oracle PATH]
```

Fixed defaults:

- fixture `tests/fixtures/oaf_tf1_smoke/canonical.wav`;
- checkpoint cache `artifacts/benchmark/model-cache`;
- image `crux-oaf-tf1:local`;
- prediction `artifacts/benchmark/oaf-smoke/prediction.jsonl`.

The inference timer is scoped exactly:

```python
started = time.perf_counter()
native_prediction = backend.transcribe(audio)
inference_elapsed_seconds = time.perf_counter() - started
```

Mapping, JSONL rendering, immutable publication, oracle loading/comparison, and summary output occur **after** that timer stops.

Real-time factor is:

```text
inference_elapsed_seconds / fixture_duration_seconds
```

The workflow reports this as a smoke diagnostic only. HPA-326 full-song/corpus measurements remain authoritative for worker-count or throughput decisions.

When `--oracle` is supplied, the path is required to exist and exact native events must match or the command fails.

## 13. Manual Workflow

Add `.github/workflows/oaf-smoke.yml`:

- `workflow_dispatch` only;
- no custom inputs;
- one `ubuntu-24.04` job;
- `contents: read` only;
- prepare checkpoint once;
- build image once;
- run smoke once;
- upload only prediction JSONL;
- write job summary.

First-run summary includes `oracle_status=not_checked`; after the oracle commit the fixed workflow invocation includes `--oracle` and requires `oracle_status=matched`.

Summary records:

```text
commit/ref
backend_id
upstream_source_commit
checkpoint_archive_sha256
restored_tensor_count
fixture_sha256
mapped_event_count
unmapped_event_count
prediction_sha256
inference_elapsed_seconds
real_time_factor
oracle_status
```

No OIDC, Sigstore, artifact attestation, deterministic OCI export, calibration matrix, host fingerprint, or second image build.

## 14. Final Schema-Golden State

At the atomic cutover, make `tests/benchmark/schema_goldens/manifest.json` explicit rather than incrementally retaining HPA-320 rows.

The surviving benchmark rows are:

```text
crux.input-view-manifest/v1
crux.drum-prediction-events/v2
crux.dtx-reference-event/v1
crux.reference-chart-manifest/v1
crux.reference-timing-manifest/v1
crux.benchmark-reference-manifest/v1
crux.oaf-smoke-oracle/v2
```

The HPA-324 benchmark-reference row is present because HPA-423's mapping cutover consumes the HPA-324 production contract. Delete HPA-320 lock/report/attestation/checkpoint-request/evidence/runner/calibration/source-manifest goldens.

Descriptor v2 remains covered directly by exact unit tests rather than another golden row.

## Non-Goals

HPA-423 does not add:

- full-corpus inference;
- resume/retry behavior;
- worker pools or queues;
- distributed execution;
- threshold tuning or fine-tuning;
- a TensorFlow 2 replacement;
- scoring metrics/tolerances;
- reference taxonomy/eligibility policy;
- release/security attestation;
- deterministic OCI reproduction;
- backward compatibility for seal or prediction-v1 artifacts;
- generic plugin/config frameworks;
- changes to the separate FastAPI TF2 fallback.

## Acceptance Criteria

- [ ] Every task commit leaves ordinary imports and repository tests green.
- [ ] The final shared backend protocol is descriptor/transcribe/close only.
- [ ] All four existing source/input audio identity fields remain.
- [ ] `model.json` is the only final model/checkpoint identity source; old backend config requests are deleted.
- [ ] `max_input_audio_frames` is explicitly null rather than silently losing the old lock-supplied value.
- [ ] Checkpoint archive and all component hashes are verified.
- [ ] All 78 required inference tensors restore before ready.
- [ ] `assert_no_reachable_stochastic_ops()` remains in the model startup correctness path.
- [ ] The final Docker entrypoint runs `worker.py`, uses CPU-only `CUDA_VISIBLE_DEVICES=-1`, and does not inherit single-thread seal caps.
- [ ] New model/worker tests run under normal CI from `tests/runtime/oaf_tf1/`; the Docker test target/test wheelhouse are removed.
- [ ] One persistent worker serves repeated sequential requests.
- [ ] The registry contains only `oaf`; seal state and heuristic support are gone.
- [ ] OaF mapping lives in `mapping.py`, reuses `MappingDiagnostics`, and validates HPA-324 backend/native-output identity.
- [ ] Prediction artifact v2 persists common-only hi-hat/tom mappings and unmapped events while preserving native identity.
- [ ] Prediction-side `BenchmarkEvent.canonical_class` is locked by tests/documentation to the common scoring level.
- [ ] Final `DirectoryAnchor` parameters are removed from retained input/prediction APIs.
- [ ] `publish_prediction_artifact()` uses neutral immutable publication with no `role=` contract.
- [ ] Final schema-golden manifest contains only the explicitly listed surviving rows.
- [ ] Smoke RTF times `backend.transcribe()` only.
- [ ] First real smoke produces a valid mapped prediction through the authentic released checkpoint.
- [ ] A committed exact binary64 native-event smoke oracle is derived from that first result.
- [ ] A second real smoke matches the committed oracle exactly and reports `oracle_status=matched`.
- [ ] No active HPA-320 seal/lock/report/attestation/calibration/heuristic path remains.
- [ ] No compatibility reader for removed seal interfaces or prediction v1 is added.
