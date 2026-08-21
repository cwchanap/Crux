# HPA-396 Frozen IDM Stem Pilot Design

**Status:** Proposed  
**Date:** 2026-08-20  
**Linear:** HPA-396 — Evaluate frozen Inverse Drum Machine on fixed drum-stem benchmark pilot

## Decision summary

Implement Inverse Drum Machine (IDM) as the third **benchmark-only** transcription backend over the frozen HPA-328 HTDemucs pilot population.

The architecture remains deliberately narrow:

1. pin the official `bernardo-torres/inverse-drum-machine` repository at commit `456656868538205ef756912c7cf5b0fd936de8af` and the committed `idm-44-train-kits` checkpoint/config bytes;
2. keep IDM in one isolated `uv` runtime rather than adding its older research dependency stack to Crux's Python 3.12 environment;
3. reuse the existing persistent `WorkerProcess` controller so the model loads once per pilot;
4. use IDM's encoder transcription activations plus its evaluation sigmoid/peak-picking path directly, with no manual onset override and no required synthesis/separation output;
5. extend the existing closed backend descriptor, prediction-v2, taxonomy, mapping, and zero-hit identity tables for exactly one third known backend family;
6. consume `load_separation_pilot_manifest()` and the retained HPA-328 HTDemucs canonical input/OaF prediction artifacts; never rerun HTDemucs or OaF;
7. rebuild OaF and IDM cohorts through the unchanged HPA-325 scorer and publish paired OaF→IDM deltas through the existing public comparison primitives;
8. keep direct full-mix IDM inference to one fixed five-song diagnostic selected from the real HPA-328 handoff **before any IDM corpus/full-mix inference**;
9. stop before scored execution if runtime, checkpoint, license, WAV-input, or peak-picking evidence cannot be frozen reproducibly.

Do **not** add a generic model-runtime manager, generic corpus-runner base class, prediction schema v3, plugin registry, new scorer, required reconstructed IDM stems, model-selection/tuning loop, full-corpus IDM run, production backend registration, training, fine-tuning, or manual transcription corrections.

## Review-frozen decisions

The following decisions are part of the design now, not implementation-time choices.

### Zero-hit prediction identity is mandatory

A valid IDM input may produce zero picked onsets. HPA-325 derives an empty prediction artifact's map identity through the closed `ZERO_HIT_PREDICTION_MAPS` table, because no event record exists from which to recover `prediction_map_version`.

HPA-396 must therefore add:

```python
ZERO_HIT_PREDICTION_MAPS = {
    OAF_BACKEND_ID: OAF_PREDICTION_MAP_ID,
    MUSCRIPTOR_BACKEND_ID: MUSCRIPTOR_PREDICTION_MAP_ID,
    IDM_BACKEND_ID: IDM_PREDICTION_MAP_ID,
}
```

This is required in the same change that adds the IDM prediction family. It is not deferred until a zero-hit song is encountered.

### Native failure codes map to the existing HPA-325 reason enum

Do not extend `CohortFailureReason`. Preserve the HPA-396 native detail code in the run snapshot and native failure histogram, but map failed rows to the existing scorer vocabulary:

| HPA-396 native code | HPA-325 failure reason |
| --- | --- |
| `worker_start_failed` | `backend_unavailable` |
| `worker_protocol_failed` | `backend_unavailable` |
| `inference_failed` | `inference_failed` |
| `native_event_invalid` | `inference_failed` |
| `upstream_stem_unavailable` | `inference_failed` |
| `retained_input_invalid` | `prediction_artifact_invalid` |
| `retained_oaf_prediction_invalid` | `prediction_artifact_invalid` |
| `prediction_artifact_invalid` | `prediction_artifact_invalid` |
| `prediction_output_conflict` | `prediction_artifact_invalid` |
| `prediction_publish_failed` | `prediction_artifact_invalid` |

`upstream_stem_unavailable` deliberately maps to `inference_failed`, not `prediction_missing`. Existing Crux runners already classify unavailable source input as an inference failure; `prediction_missing` remains reserved for a prediction artifact that should exist but does not.

A worker close failure is run-level finalization evidence, not an item-level scorer reason.

### Worker WAV decoding is frozen and never uses librosa resampling

The primary HPA-328 input is already canonical 44.1 kHz mono PCM16 WAV. The isolated IDM environment contains `librosa 0.9.x`, whose default `librosa.load(path)` would resample to 22.05 kHz. That path is forbidden.

The worker input contract is:

```text
loader: soundfile-preserve-wav/v1
container: WAV
sample rate: 44100 Hz
channels: 1
subtype: PCM_16
resampling: forbidden
mixdown: forbidden
normalization: forbidden
```

The worker uses `soundfile.info()` to reject any input that is not WAV/44.1 kHz/mono/PCM16, then uses `soundfile.read(..., dtype="float32", always_2d=True)` without an `sr` conversion. It passes the resulting mono tensor directly to `model.encoder()`.

These loader facts are recorded in `crux.idm-model/v1`, and Task 0 must prove that the pinned official encoder accepts this tensor path.

### Request timeout is frozen in run identity

`WorkerProcess.start()` defaults to 30 seconds, which is not acceptable for the experimental IDM pilot. HPA-396 freezes:

```text
IDM_REQUEST_TIMEOUT_SECONDS = 1800.0
IDM_WORKER_CLOSE_TIMEOUT_SECONDS = 30.0
```

The 1800-second request ceiling matches the existing HPA-328 separator ceiling and is selected before IDM scores are observed. It is part of `crux.idm-inference-config/v1`, so changing it changes run identity rather than silently changing the paired population.

The worker is always launched with an explicit command sequence:

```python
[
    os.fspath(runtime_python),
    os.fspath(worker_path),
    "--model-root",
    os.fspath(model_root),
]
```

Never pass the worker path as a lone `Path` or string to `WorkerProcess.start()`, because that would bind execution to Crux's `sys.executable` instead of the isolated IDM interpreter.

### Peak metadata is preserved before the pilot

`idm-peak-event-metadata-v1` contains exactly:

```text
frame_index
native_velocity
```

The shared `NativeEvent.native_metadata` contract currently stores string/null values. HPA-396 therefore persists these diagnostics as canonical strings rather than widening prediction-v2:

```text
frame_index     = non-negative base-10 integer string
native_velocity = canonical six-decimal-or-shorter numeric string, > 0 and <= 2
```

`native_velocity` is the model's velocity activation before the MIDI diagnostic projection, quantized only at Crux's existing six-decimal persisted-artifact boundary. `velocity_midi` remains the deterministic `0..127` projection. `confidence` remains the post-sigmoid onset activation used by upstream peak picking.

Prediction-v2 validation special-cases only this third metadata schema's value formats. Do not create a generic metadata-validator plugin system and do not widen OaF/MuScriptor metadata semantics.

### IDM mapping keys on `native_class_id`

`map_idm_prediction()` looks up `prediction_map.classes[native.native_class_id]`.

It does not use OaF's `native_metadata["upstream_8hit_group_id"]` and does not use MuScriptor's MIDI pitch.

### Concrete modules mirror existing concrete modules

`IdmBackend` mirrors the host-side lifecycle shape in `backends/oaf.py`: process factory injection, ready validation, request/error decoding, poison semantics, and bounded close. It is not an in-process `MuscriptorBackend` copy.

`idm_comparison.py` mirrors the report-driver shape in `muscriptor_comparison.py` while calling only the public helpers in `published_comparison.py`. HPA-396 does not unify the existing comparison modules.

## Current dependency reality

The HPA-328 implementation is merged, but its production experiment is not yet available. PR #26 explicitly records an operational hold: Task 11 was not run, no production separator locks were generated, and no native separator inference was performed.

Therefore:

- HPA-396 implementation work through the frozen runtime/model/adapter/schema/runner tests can proceed;
- the real HPA-396 stem pilot cannot start until an immutable production `crux.oaf-separation-pilot/v1` handoff exists;
- concrete HPA-396 smoke song IDs cannot be truthfully named in this planning PR because the real HPA-328 population does not yet exist;
- as soon as the real handoff exists, the five smoke IDs are selected from that membership using only pre-IDM evidence, written to `runtime/idm/smoke.json`, committed, and only then may any IDM HPA-328/full-mix inference start.

Do not substitute the synthetic MuScriptor smoke IDs `1..5` or test-fixture HPA-328 IDs.

## Existing evidence and seams

### HPA-328 handoff contract

`src/benchmark/separation_handoff.py` owns the immutable closeout boundary intended for HPA-396. `load_separation_pilot_manifest()` validates:

- canonical `crux.oaf-separation-pilot/v1` JSONL;
- exact 20–30-song HPA-327 population;
- sorted unique simfile IDs;
- consistent reviewed-subset/reference/timing/OaF/run identity;
- full-mix, Spleeter, and HTDemucs per-song status/evidence;
- retained stem, canonical input, and prediction hashes;
- fixed comparison artifact identity.

HPA-396 consumes this immutable handoff. It does not parse HPA-328's mutable `run.json`, infer membership from directories, or invoke a separator.

### Shared backend protocol

Reuse unchanged from `src/benchmark/backends/base.py`:

```python
CanonicalAudio
NativeEvent
NativePrediction
TranscriptionBackend
```

IDM fits this protocol and remains outside `default_backend_registry()`.

### Persistent worker

Reuse `src/benchmark/worker_process.py` unchanged unless a test proves a model-neutral wording issue. It already owns:

- one `ready` record;
- JSON request/response IDs;
- timeout/poison semantics;
- bounded stderr diagnostics;
- deterministic close/terminate/kill behavior.

### Prediction v2

`src/benchmark/prediction_artifact.py` retains the existing event shape:

```text
time_sec
native_class_id
model_output_bin
native_midi_note
native_metadata
confidence
velocity_midi
canonical_class
common_class
mapping_status
prediction_map_version
```

IDM is the third closed family. No prediction-v3 migration is required.

### Published comparison primitives

Reuse the public functions in `src/benchmark/published_comparison.py`:

```python
pairable_success_ids(...)
paired_song_rows(...)
paired_class_rows(...)
metric_delta(...)
aggregate_delta_rows(...)
population(...)
runtime(...)
```

The headline OaF↔IDM pair must call:

```python
pairable_success_ids(
    oaf,
    idm,
    selected_ids,
    require_identical_input_hash=True,
    left_label="oaf",
    right_label="idm",
)
```

The strict input-hash default is intentional. HPA-328 needed an opt-out only because it compared full-mix bytes to derived stem bytes.

## Upstream IDM facts frozen by this design

Use exactly:

```text
repository: https://github.com/bernardo-torres/inverse-drum-machine
revision: 456656868538205ef756912c7cf5b0fd936de8af
package version: 0.1.0
model: idm-44-train-kits
model config: pretrained/idm-44-train-kits/checkpoints/model.yaml
checkpoint: pretrained/idm-44-train-kits/checkpoints/val-epoch=518-global_step=0.ckpt
```

The repository declares Python `^3.10`, PyTorch/torchaudio `^2.1.1`, librosa `^0.9.1`, and a research dependency set. It does not commit a Poetry lock file.

The selected `model.yaml` fixes:

```text
sample rate: 44100 Hz
mel n_fft: 1024
mel hop_length: 256
mel bins: 128
transcription onset activation: none (native logits)
transcription velocity activation: exp_sigmoid
ordered classes: 9
decoder evaluation onset activation: sigmoid
decoder evaluation peak picking: PeakPicking defaults
```

Ordered `train_classes`:

```text
CY_CR
CY_RD
HH_CHH
HH_OHH
KD
SD
TT_HFT
TT_HMT
TT_LMT
```

Expected encoder frame rate is `44100 / 256 = 172.265625 Hz`; the worker verifies the observed runtime value.

The repository root license is Apache-2.0. The freeze must also establish the licensing basis for the committed checkpoint bytes. Missing or contradictory weight-license evidence stops scored execution.

## Approach selection

### A. Isolated persistent IDM worker — selected

Create a standalone runtime under `runtime/idm/` and launch a small worker with that runtime's Python. The worker loads the model once and handles the fixed pilot through `WorkerProcess`.

This isolates the research dependency stack without adding a container or repeated startup cost.

### B. Main-environment IDM dependency — rejected

Crux uses Python 3.12 and librosa `>=0.10`, while IDM requests the older 0.9 line. Pulling that stack into the main lock is unnecessary coupling for one experimental comparator.

### C. One process per song — rejected

Reloading PyTorch/model 20–30 times contaminates RTF and adds runtime without simplifying an already-existing worker protocol.

### D. Docker or generalized external-runtime framework — rejected

The fixed public checkpoint and isolated lock do not justify another container/runtime-attestation subsystem. Keep the contract specific to IDM.

## Isolated runtime and model freeze

### Runtime project

Create:

```text
runtime/idm/pyproject.toml
runtime/idm/uv.lock
runtime/idm/worker.py
```

The project has one direct dependency on the exact Git revision. Task 0 selects one supported Python minor and freezes it. Normal Crux `uv run` does not install IDM.

Operational setup:

```bash
uv sync --project runtime/idm --frozen
```

The runtime `.venv` is uncommitted.

### Model cache

Acquire only the exact committed `model.yaml` and checkpoint into a caller-selected local model root. Commit identities, not duplicate weight bytes. Runtime verifies both files before worker startup.

### `crux.idm-model/v1`

`runtime/idm/model.json` is a strict canonical lock containing:

```text
schema
repository_url
repository_revision
package_name
package_version
code_license
weight_license
runtime_lock_sha256
python_version
model_name
model_config_relative_path
model_config_sha256
model_config_byte_length
checkpoint_relative_path
checkpoint_sha256
checkpoint_byte_length
model_id
device
dtype
sample_rate_hz
input_channel_count
input_container
input_subtype
audio_loader_revision
resampling
mixdown
mel_n_fft
mel_hop_length
mel_n_mels
activation_rate_hz
train_classes
onset_activation
peak_pick_div_max
peak_pick_div_avg
peak_pick_div_wait
peak_pick_div_threshold
peak_pick_normalize
velocity_activation
velocity_max_value
velocity_to_midi_revision
native_velocity_persistence_revision
manual_onset_override
reconstructed_stems
masking
chunking_mode
native_output_space_id
native_metadata_schema_id
training_data_map_id
```

Fixed v1 semantics include:

```text
repository_revision = 456656868538205ef756912c7cf5b0fd936de8af
package_name = inverse-drum-machine
package_version = 0.1.0
model_name = idm-44-train-kits
sample_rate_hz = 44100
input_channel_count = 1
input_container = WAV
input_subtype = PCM_16
audio_loader_revision = soundfile-preserve-wav/v1
resampling = forbidden
mixdown = forbidden
mel_n_fft = 1024
mel_hop_length = 256
mel_n_mels = 128
train_classes = [CY_CR, CY_RD, HH_CHH, HH_OHH, KD, SD, TT_HFT, TT_HMT, TT_LMT]
onset_activation = sigmoid-over-native-logits-before-upstream-peak-picking
peak_pick_div_max = 20
peak_pick_div_avg = 10
peak_pick_div_wait = 16
peak_pick_div_threshold = 5
peak_pick_normalize = false
velocity_activation = exp_sigmoid(exponent=10,max_value=2,threshold=1e-7)
velocity_to_midi_revision = clamp-half-round-midi127/v1
native_velocity_persistence_revision = quantize-six-canonical-string/v1
manual_onset_override = false
reconstructed_stems = false
masking = none
chunking_mode = none
```

Device/dtype are frozen before the scored pilot based on feasibility only, never F1.

### Model ID

```text
idm-44-train-kits-<revision12>-<checkpoint12>
```

The lock derives it from exact source revision and checkpoint SHA-256.

### `crux.idm-inference-config/v1`

Keep run policy thin and avoid duplicating peak/model facts already on the model lock:

```text
schema
backend_descriptor_sha256
model_lock_sha256
adapter_revision
prediction_map_version
input_view_id
request_timeout_seconds
```

For the primary stem pilot:

```text
adapter_revision = crux.idm-adapter/v1
prediction_map_version = crux.prediction-map/idm-44-train-kits-v1
input_view_id = crux.oaf-htdemucs-drums-mono44k1-pcm16/v1
request_timeout_seconds = 1800
```

The canonical config hash participates in run/prediction identity.

## Worker transcription contract

### Startup

The host launches the worker with the exact isolated interpreter as an argv sequence. The worker:

1. imports `idm.inference.load_model` from the locked runtime;
2. loads `idm-44-train-kits` once from the verified local model root;
3. verifies sample rate, frame rate, ordered classes, and expected model facts;
4. emits a `ready` record.

`IdmBackend` validates the ready payload against `runtime/idm/model.json` and the expected descriptor before retaining the process.

### WAV decode

For every request:

```python
info = soundfile.info(audio_path)
require info.format == "WAV"
require info.subtype == "PCM_16"
require info.samplerate == 44100
require info.channels == 1
samples, sample_rate = soundfile.read(
    audio_path,
    dtype="float32",
    always_2d=True,
)
require sample_rate == 44100
require samples.shape[1] == 1
audio = torch.from_numpy(samples[:, 0]).to(device).unsqueeze(0)
```

No librosa load, resample, mixdown, normalization, or rewrite is permitted.

### Transcription path

```python
encoder_outs = model.encoder(audio)
onset_logits = encoder_outs["activations"]["onset"]
native_velocity = encoder_outs["activations"]["velocity"]
onset_scores = model.decoder.activation(onset_logits)
onsets = model.decoder.peak_picking_val(
    onset_scores,
    activation_rate=encoder_outs["activation_rate"],
)
```

Do not call the synthesis, masking, or manual override path.

### Raw worker events

For every picked event return:

```text
class_index
native_class_id
frame_index
time_sec
onset_score
native_velocity
```

All values must be finite and internally consistent. `frame_index` is derived from the same picked activation frame used for time/score/velocity lookup; do not reconstruct it later from quantized `time_sec`.

### Host-side `NativeEvent`

Example:

```python
NativeEvent(
    time_sec=1.25,
    native_class_id="KD",
    model_output_bin=4,
    native_midi_note=None,
    native_metadata={
        "frame_index": "215",
        "native_velocity": "1.337421",
    },
    confidence=0.83,
    velocity_midi=85,
)
```

`model_output_bin` is the locked class index. `native_midi_note` is always null.

Velocity projection:

```text
velocity_midi = round(clamp(native_velocity / 2.0, 0.0, 1.0) * 127)
```

Velocity/confidence remain diagnostic only.

## Descriptor and prediction-v2 extension

Add the third descriptor family:

```text
backend_id: idm-44-train-kits-v1
architecture_id: inverse-drum-machine-v0.1.0
model_id: idm-44-train-kits-<revision12>-<checkpoint12>
native_output_space_id: idm-44-train-kits-9class-v1
native_metadata_schema_id: idm-peak-event-metadata-v1
prediction_schema: crux.drum-prediction-events/v2
training_data_map_id: idm-training-contract-44-train-kits-v1
upstream_source_commit: 456656868538205ef756912c7cf5b0fd936de8af
```

The backend requires exact descriptor equality with the lock-derived descriptor.

### IDM native event invariants

Prediction-v2 validation for IDM requires:

```text
model_output_bin: integer 0..8
native_class_id: exact train_classes[model_output_bin]
native_midi_note: null
confidence: non-null finite 0..1
velocity_midi: non-null integer 0..127
native_metadata keys: exactly frame_index, native_velocity
frame_index: canonical non-negative integer string
native_velocity: canonical persisted decimal string, >0 and <=2
```

OaF and MuScriptor invariants remain byte-for-byte/semantically unchanged.

## Frozen IDM mapping

```text
IDM_PREDICTION_MAP_ID = crux.prediction-map/idm-44-train-kits-v1
```

| Native class | Upstream meaning | Detailed Crux class | Common class |
| --- | --- | --- | --- |
| `KD` | kick | `kick` | `kick` |
| `SD` | snare | `snare` | `snare` |
| `HH_CHH` | closed hi-hat | `closed_hihat` | `hihat` |
| `HH_OHH` | open hi-hat | `open_hihat` | `hihat` |
| `CY_CR` | crash | `crash` | `crash` |
| `CY_RD` | ride | `ride` | `ride` |
| `TT_HMT` | hi tom | `high_tom` | `tom` |
| `TT_LMT` | mid tom | `null` | `tom` |
| `TT_HFT` | low tom | `low_or_floor_tom` | `tom` |

`TT_LMT` remains common-only because Crux has no mid-tom detailed category. Do not infer a false detailed class.

Every native event is persisted. Unexpected classes become unmapped evidence.

## Primary pilot input contract

The primary population is exactly the real immutable HPA-328 handoff.

For each row:

- a successful HTDemucs row must have retained stem/input/OaF prediction evidence;
- re-read the retained canonical HTDemucs input bytes and verify the handoff SHA before IDM inference;
- the `CanonicalAudio` passed to IDM uses the HPA-328 input identity unchanged;
- do not rerun the separator or materialize another WAV;
- re-read and verify retained OaF prediction artifacts for the control cohort;
- non-successful HPA-328 rows stay in the HPA-396 population with explicit native failure state.

The shared input-view ID remains:

```text
crux.oaf-htdemucs-drums-mono44k1-pcm16/v1
```

It names the bytes, not the consumer model.

## Runner

Add one concrete `src/benchmark/idm_pilot_run.py`. Do not generalize `muscriptor_corpus_run.py` or `separation_pilot.py`.

### Primary request

```python
@dataclass(frozen=True)
class IdmPilotRunRequest:
    handoff_manifest_path: Path
    stem_cache_root: Path
    output_dir: Path
    model_lock_path: Path
    model_root: Path
    runtime_python: Path
    resume: bool = False
    crux_commit: str | None = None
```

There is no `source_cache_dir` on the primary stem request. Full-mix source resolution belongs only to the separate smoke request.

### Full-mix smoke request

```python
@dataclass(frozen=True)
class IdmFullMixSmokeRequest:
    handoff_manifest_path: Path
    smoke_manifest_path: Path
    source_cache_dir: Path
    output_dir: Path
    model_lock_path: Path
    model_root: Path
    runtime_python: Path
    crux_commit: str | None = None
```

Smoke plumbing cannot leak into the stem-pilot identity or required inputs.

### Resume

Resume only exact run identity. Re-read retained input/prediction bytes before reuse. A missing or changed artifact becomes explicit failure evidence; do not silently re-separate or rerun OaF.

## Scoring and comparison

Use unchanged:

```python
score_cohort()
write_cohort_reports()
```

Build one OaF cohort and one IDM cohort over the same fixed HPA-328 population. Each failed row maps through the fixed table above.

`idm_comparison.py`:

1. reads published OaF/IDM reports through public report readers;
2. requires matching reference/timing/taxonomy/scoring identities;
3. calls `pairable_success_ids(..., require_identical_input_hash=True)`;
4. emits paired per-song and per-class deltas;
5. reports each full population before the successful intersection;
6. includes native IDM failure histogram, mapped/unmapped coverage, runtime, peak memory where practical, and velocity diagnostics;
7. never merges the five full-mix smoke rows into the headline stem comparison.

No automatic winner policy belongs here.

## Five-song full-mix compatibility smoke

### Membership freeze gate

The smoke manifest uses the MuScriptor-style compact shape with a distinct schema:

```json
{
  "schema": "crux.idm-smoke/v1",
  "cases": [
    {"reason": "short", "simfile_id": 123},
    {"reason": "long", "simfile_id": 456},
    {"reason": "dense", "simfile_id": 789},
    {"reason": "sparse", "simfile_id": 1011},
    {"reason": "non_drum_heavy", "simfile_id": 1213}
  ]
}
```

The numbers above are **shape examples only and must never be committed**.

Because the real HPA-328 handoff does not yet exist, exact IDs are intentionally unresolved at planning time. The operational gate is:

1. obtain and validate the production HPA-328 handoff;
2. select exactly five members using only pre-IDM evidence such as duration/reference hit counts plus source listening where needed for the reason label;
3. require five unique IDs, each present in the handoff;
4. commit `runtime/idm/smoke.json`;
5. verify the smoke-manifest tests;
6. only then allow any IDM inference on HPA-328 stem/full-mix corpus inputs.

No OaF score, separator score, or IDM score may influence membership.

The full-mix diagnostic uses the authoritative full mix only for those five IDs and a separate input-view identity. It does not change primary stem-pilot configuration or membership.

## Failure semantics

### Fatal preflight

Fail before item execution for:

- malformed model/runtime lock;
- model/config/checkpoint hash mismatch;
- missing/contradictory license evidence;
- descriptor mismatch;
- runtime Python not the locked environment;
- worker ready/model-fact mismatch;
- invalid/missing production HPA-328 handoff for a real pilot.

### Item-local / persisted failures

Preserve the native code on the HPA-396 snapshot and map to HPA-325 through the frozen table. Do not collapse native histograms into scorer reasons.

### Poison/finalization

Protocol failures poison the worker according to existing `WorkerProcess`/OaF lifecycle semantics. Close failure is finalization evidence and does not invent an item-level reason.

## Risks and hard gates

### HPA-328 operational handoff absent

The merged HPA-328 code currently has no production Task 11 handoff. This blocks smoke-ID freeze and real HPA-396 execution, but not offline implementation/tests. Never fabricate membership from fixtures.

### Weight-license ambiguity

The root repository is Apache-2.0, but scored use requires a defensible basis for the committed checkpoint bytes. Missing/contradictory evidence stops the experiment.

### PeakPicking API/runtime drift

The design relies on the pinned commit's evaluation path: decoder sigmoid plus `peak_picking_val` using the encoder activation rate. Task 0 must execute this exact path. If it does not work at the pinned revision, stop rather than reimplement/tune it silently.

### Accidental librosa resampling

Any `librosa.load(path)` path would default-resample in the isolated dependency set. The worker uses the frozen soundfile loader and rejects noncanonical input. Tests must prove no 22.05 kHz path is used.

### Timeout-dependent pairability

A request timeout changes success population and therefore paired evidence. `1800` seconds is frozen in the inference-config identity before scores. Do not change it after inspecting results.

### Zero-hit scoring

An empty valid IDM artifact must score successfully through `ZERO_HIT_PREDICTION_MAPS[IDM_BACKEND_ID]`. This is a required schema/cohort test before the pilot.

### Metadata loss

Frame index and pre-MIDI native velocity are cheap evidence available at inference time and expensive to recover later. Persist both in v1 metadata before real execution.

## Verification strategy

Offline CI uses fakes and synthetic artifacts; it never installs/downloads the isolated IDM runtime.

Required coverage includes:

- model/runtime lock canonical parsing and exact model-file hashes;
- soundfile input contract and rejection of wrong sample rate/channel/subtype;
- Task 0 real encoder/PeakPicking compatibility gate;
- descriptor third-family validation;
- prediction-v2 IDM render/read/render with frame/native-velocity metadata;
- OaF/MuScriptor prediction regression;
- zero-hit IDM artifact scoring;
- exact class-index↔native-class invariants;
- `map_idm_prediction()` keyed by `native_class_id`;
- `TT_LMT -> ClassMapping(None, "tom")`;
- worker ready/request/error/poison/close behavior;
- explicit runtime-python argv sequence;
- fixed request-timeout identity;
- handoff population and retained input/OaF prediction hash binding;
- native→HPA-325 failure mapping;
- resume without separator/OaF rerun;
- strict identical-input pairing;
- full population + paired intersection reporting;
- smoke manifest requires five unique real handoff IDs and remains separate from headline comparison;
- output determinism and CLI 0/1/2 semantics.

## Non-goals

- Generic backend registry/plugin architecture.
- Generic runner/comparison framework.
- Prediction v3.
- Full-corpus IDM evaluation.
- IDM training/fine-tuning.
- Manual onset correction.
- Required IDM reconstructed stems or separator SDR evaluation.
- Feeding IDM-generated stems into OaF.
- Re-running HTDemucs or OaF.
- Changing HPA-325 scoring/matching/tolerances.
- Velocity-aware scoring.
- Automated model/input-view winner selection.
- Production DTX chart generation/editor integration.
