# HPA-396 Frozen IDM Stem Pilot Design

**Status:** Proposed  
**Date:** 2026-08-20  
**Linear:** HPA-396 — Evaluate frozen Inverse Drum Machine on fixed drum-stem benchmark pilot

## Decision summary

Implement Inverse Drum Machine (IDM) as the third **benchmark-only** transcription backend over the fixed HPA-328 HTDemucs pilot population.

The design stays narrow:

1. pin official `bernardo-torres/inverse-drum-machine` at commit `456656868538205ef756912c7cf5b0fd936de8af` and model `idm-44-train-kits`;
2. run IDM in one isolated Python 3.11 `uv` environment behind the existing persistent `WorkerProcess`;
3. use the model encoder's native onset/velocity activations plus the pinned decoder sigmoid/`PeakPicking` evaluation path, with no manual onset override and no required synthesis/masking output;
4. extend the existing closed backend descriptor, prediction-v2, taxonomy, mapping, and zero-hit tables for exactly one third backend family;
5. consume the immutable HPA-328 handoff and its retained HTDemucs input/OaF prediction artifacts; never rerun HTDemucs or OaF;
6. rebuild both cohorts with unchanged HPA-325 `score_cohort()` / `write_cohort_reports()` and pair only identical source/input hashes through public `published_comparison` helpers;
7. keep full-mix IDM work to a separately named fixed five-song smoke selected from the real HPA-328 population before any IDM corpus inference;
8. stop before scored execution if runtime, model, license, WAV-input, or peak-picking evidence cannot be frozen reproducibly.

Do **not** add prediction v3, a plugin/registry system, a generic runner/comparison framework, a second scorer, required reconstructed IDM stems, score-driven tuning, full-corpus IDM execution, production API integration, training, fine-tuning, or manual event correction.

## Current dependency reality

The HPA-328 implementation is merged, but its production experiment is not. The merged PR records an operational hold: Task 11 was not run, no production separator locks were generated, and no native separator inference/handoff was produced.

Consequences:

- offline HPA-396 implementation/tests may proceed;
- real HPA-396 stem/full-mix execution remains blocked until a production `crux.oaf-separation-pilot/v1` handoff exists;
- concrete smoke IDs cannot be truthfully named in this planning PR;
- test-fixture IDs and MuScriptor smoke IDs `1..5` must never be substituted for the production population.

The design therefore freezes the **selection rule** now and freezes the resulting five real IDs immediately when the production handoff exists, before any IDM HPA-328/full-mix inference.

## Review-frozen closed decisions

### Mandatory zero-hit prediction map

HPA-325 derives map identity for an empty prediction artifact from `ZERO_HIT_PREDICTION_MAPS`. A valid IDM input may have zero picked peaks, so the third entry is required:

```python
ZERO_HIT_PREDICTION_MAPS = {
    OAF_BACKEND_ID: OAF_PREDICTION_MAP_ID,
    MUSCRIPTOR_BACKEND_ID: MUSCRIPTOR_PREDICTION_MAP_ID,
    IDM_BACKEND_ID: IDM_PREDICTION_MAP_ID,
}
```

This lands with the IDM prediction family; it is not conditional on observing a zero-hit song.

### Fixed native failure mapping

Preserve native HPA-396 detail codes in the mutable run snapshot and comparison histogram, but map failed `CohortItem`s onto the existing HPA-325 enum:

| HPA-396 native code | HPA-325 reason |
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

Do not extend `CohortFailureReason`.

`upstream_stem_unavailable` maps to `inference_failed`, not `prediction_missing`: existing Crux corpus runners already classify unavailable source input as inference failure, while `prediction_missing` means an expected prediction artifact itself is absent.

Worker close failure is run-level finalization evidence, not an item-level scorer reason.

### Frozen WAV loader

HPA-328 canonical input is 44.1 kHz mono PCM16 WAV. The isolated IDM environment contains librosa 0.9.x, whose default `librosa.load(path)` would resample to 22.05 kHz. That path is forbidden.

Freeze this input contract in `crux.idm-model/v1`:

```text
audio_loader_revision = soundfile-preserve-wav/v1
input_container = WAV
sample_rate_hz = 44100
input_channel_count = 1
input_subtype = PCM_16
resampling = forbidden
mixdown = forbidden
normalization = forbidden
```

Worker request handling uses `soundfile.info()` to reject any mismatch, followed by:

```python
samples, sr = soundfile.read(path, dtype="float32", always_2d=True)
assert sr == 44100
assert samples.shape[1] == 1
audio = torch.from_numpy(samples[:, 0]).to(device).unsqueeze(0)
```

Task 0 must prove the pinned official encoder accepts this exact tensor path.

### Frozen timeout and exact runtime interpreter

Freeze:

```text
IDM_REQUEST_TIMEOUT_SECONDS = 1800.0
IDM_WORKER_CLOSE_TIMEOUT_SECONDS = 30.0
```

`1800` is selected before scores and included in `crux.idm-inference-config/v1`; changing it changes run identity.

Always launch `WorkerProcess.start()` with a command sequence beginning with the isolated runtime Python:

```python
[
    os.fspath(runtime_python),
    os.fspath(worker_path),
    "--model-root",
    os.fspath(model_root),
]
```

Never pass a lone worker `Path`/string because `WorkerProcess.start()` would prepend Crux's `sys.executable`.

### Peak metadata is not discarded

`idm-peak-event-metadata-v1` contains exactly:

```text
frame_index
native_velocity
```

The shared metadata value contract is string/null, so preserve these as canonical strings rather than widening prediction-v2:

```text
frame_index     = non-negative base-10 integer string
native_velocity = canonical persisted decimal string, >0 and <=2
```

`native_velocity` is captured before the MIDI diagnostic projection, at Crux's six-decimal persisted-artifact boundary. `velocity_midi` remains the fixed `0..127` projection. `confidence` remains the post-sigmoid onset score used by upstream peak picking.

Prediction-v2 receives one small IDM-specific value-format branch. Do not build a generic metadata-validator framework.

### IDM mapping key

`map_idm_prediction()` looks up:

```python
prediction_map.classes[native.native_class_id]
```

It does not use OaF metadata or MuScriptor MIDI pitch.

### Concrete module shape

- `IdmBackend` mirrors OaF's process-factory/ready/request/poison/close host adapter shape, not in-process MuScriptor.
- `idm_pilot_run.py` is a concrete fixed-pilot runner, not a generic corpus runner.
- `idm_comparison.py` mirrors the driver shape of `muscriptor_comparison.py` and calls public `published_comparison.py` helpers; HPA-396 does not unify comparison modules.

## Existing reusable seams

Reuse unchanged unless a focused test proves otherwise:

```text
src/benchmark/backends/base.py
  CanonicalAudio
  NativeEvent
  NativePrediction
  TranscriptionBackend

src/benchmark/worker_process.py
  WorkerProcess

src/benchmark/separation_handoff.py
  load_separation_pilot_manifest()

src/benchmark/cohort_scoring.py
  score_cohort()

src/benchmark/reports.py
  write_cohort_reports()

src/benchmark/published_comparison.py
  pairable_success_ids()
  paired_song_rows()
  paired_class_rows()
  metric_delta()
  aggregate_delta_rows()
  population()
  runtime()
```

The headline pair uses:

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

The strict input-hash requirement is intentional; HPA-328's opt-out existed only for full-mix-versus-stem comparisons.

IDM remains outside `default_backend_registry()`.

## Frozen upstream boundary

Use exactly:

```text
repository: https://github.com/bernardo-torres/inverse-drum-machine
revision: 456656868538205ef756912c7cf5b0fd936de8af
package version: 0.1.0
model: idm-44-train-kits
model config: pretrained/idm-44-train-kits/checkpoints/model.yaml
checkpoint: pretrained/idm-44-train-kits/checkpoints/val-epoch=518-global_step=0.ckpt
official demo input: demo/mix.wav
```

The upstream package requests Python `^3.10`, PyTorch/torchaudio `^2.1.1`, librosa `^0.9.1`, and a research dependency set. It has no committed Poetry lock, so Crux supplies a standalone `uv.lock` for the pinned Git revision.

The selected model config fixes:

```text
sample rate = 44100
mel n_fft = 1024
mel hop_length = 256
mel bins = 128
onset head activation = none (logits)
velocity activation = exp_sigmoid
ordered train classes = 9
decoder evaluation onset activation = sigmoid
peak picking = PeakPicking defaults
```

Ordered classes:

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

Expected activation rate is `44100 / 256 = 172.265625 Hz`; runtime verifies the observed value.

Root code license is Apache-2.0. The freeze must establish a defensible checkpoint-weight license basis separately; missing or contradictory evidence stops scored execution.

## Approach

### Selected: isolated persistent worker

Create:

```text
runtime/idm/pyproject.toml
runtime/idm/uv.lock
runtime/idm/worker.py
```

Pin Python `3.11.*` and the exact Git revision. Load IDM once and handle the 20–30-song pilot through `WorkerProcess`.

This keeps old research dependencies out of Crux's main Python 3.12 environment while avoiding repeated model startup.

### Rejected

- **Main optional dependency:** avoid coupling root dependency resolution to IDM's older librosa/research stack.
- **One process per song:** repeated PyTorch/model startup adds cost and pollutes RTF.
- **Docker/general runtime framework:** unnecessary for one public, small, fixed comparator.

## Model and runtime identity

### `crux.idm-model/v1`

`runtime/idm/model.json` records exact source/runtime/model/extraction semantics:

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

Fixed semantic values include:

```text
repository_revision = 456656868538205ef756912c7cf5b0fd936de8af
package_name = inverse-drum-machine
package_version = 0.1.0
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

Device/dtype are selected once by feasibility, never F1.

Model ID:

```text
idm-44-train-kits-<revision12>-<checkpoint12>
```

### `crux.idm-inference-config/v1`

Keep run policy thin; do not duplicate peak/model fields already on the lock:

```text
schema
backend_descriptor_sha256
model_lock_sha256
adapter_revision
prediction_map_version
input_view_id
request_timeout_seconds
```

Primary values:

```text
adapter_revision = crux.idm-adapter/v1
prediction_map_version = crux.prediction-map/idm-44-train-kits-v1
input_view_id = crux.oaf-htdemucs-drums-mono44k1-pcm16/v1
request_timeout_seconds = 1800
```

The config SHA participates in run/prediction identity.

## Worker transcription contract

Startup loads the exact model once, verifies model/sample-rate/frame-rate/class facts, and emits one `ready` record.

For each canonical WAV request:

```python
info = soundfile.info(audio_path)
# require WAV / PCM_16 / 44100 / mono
samples, sr = soundfile.read(audio_path, dtype="float32", always_2d=True)
audio = torch.from_numpy(samples[:, 0]).to(device).unsqueeze(0)
encoder_outs = model.encoder(audio)
onset_logits = encoder_outs["activations"]["onset"]
native_velocity = encoder_outs["activations"]["velocity"]
onset_scores = model.decoder.activation(onset_logits)
onsets = model.decoder.peak_picking_val(
    onset_scores,
    activation_rate=encoder_outs["activation_rate"],
)
```

Do not execute manual override, one-shot synthesis, sequencer output, or masking.

For each picked peak return:

```text
class_index
native_class_id
frame_index
time_sec
onset_score
native_velocity
```

The frame index is captured from the picked activation itself; do not reconstruct it later from quantized time.

Host conversion example:

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

Velocity MIDI projection is frozen as:

```text
round(clamp(native_velocity / 2.0, 0.0, 1.0) * 127)
```

Velocity/confidence never affect onset F1.

## Descriptor / prediction-v2 extension

Add exactly one third descriptor family:

```text
backend_id = idm-44-train-kits-v1
architecture_id = inverse-drum-machine-v0.1.0
model_id = idm-44-train-kits-<revision12>-<checkpoint12>
native_output_space_id = idm-44-train-kits-9class-v1
native_metadata_schema_id = idm-peak-event-metadata-v1
prediction_schema = crux.drum-prediction-events/v2
training_data_map_id = idm-training-contract-44-train-kits-v1
upstream_source_commit = 456656868538205ef756912c7cf5b0fd936de8af
```

IDM native-event validation requires:

```text
model_output_bin = integer 0..8
native_class_id = exact train_classes[model_output_bin]
native_midi_note = null
confidence = non-null finite 0..1
velocity_midi = non-null integer 0..127
native_metadata keys = exactly frame_index,native_velocity
frame_index = canonical non-negative integer string
native_velocity = canonical decimal string >0 and <=2
```

OaF and MuScriptor invariants remain unchanged.

## Frozen IDM taxonomy map

```text
IDM_PREDICTION_MAP_ID = crux.prediction-map/idm-44-train-kits-v1
```

| Native | Upstream meaning | Detailed class | Common class |
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

Crux has no mid-tom detailed class, so `TT_LMT` stays common-only. Every native event is persisted; unexpected native classes remain unmapped evidence.

## HPA-328 input and reference contract

The production `load_separation_pilot_manifest()` output defines membership and retained audio/prediction identities, but the handoff does not replace the HPA-324/HPA-323 source files needed to reconstruct reference mappings.

Primary request therefore keeps explicit paths/roots:

```python
@dataclass(frozen=True)
class IdmPilotRunRequest:
    separation_handoff_path: Path
    reference_manifest_path: Path
    timing_manifest_path: Path
    separation_artifact_root: Path
    stem_cache_root: Path
    output_dir: Path
    model_lock_path: Path
    model_root: Path
    runtime_python: Path
    resume: bool = False
    crux_commit: str | None = None
```

`reference_manifest_path` / `timing_manifest_path` must hash/version-match handoff identity. `separation_artifact_root` and `stem_cache_root` are explicit retained-artifact owners; never scan directories to discover alternatives.

For every successful HTDemucs row:

- re-read retained canonical HTDemucs input bytes no-follow and verify the handoff SHA;
- pass the same `CanonicalAudio` source/input identities to IDM;
- re-read retained OaF prediction bytes and verify source/input/header identity;
- never rerun HTDemucs or OaF.

Non-success HPA-328 rows stay in population accounting with explicit HPA-396 failure state.

The primary input-view ID remains:

```text
crux.oaf-htdemucs-drums-mono44k1-pcm16/v1
```

## Runner / resume

`src/benchmark/idm_pilot_run.py` is a concrete fixed-pilot runner.

Run identity binds:

```text
handoff manifest SHA/version
reference manifest SHA/version
timing manifest SHA/version
IDM descriptor SHA
IDM model lock SHA
IDM inference config SHA
input-view ID
Crux commit
```

Resume only exact identity. Re-read retained upstream/input/prediction bytes before reuse. Missing/changed retained evidence becomes explicit failure; no automatic re-separation/OaF repair exists.

The worker is persistent and sequential. A protocol-poisoned worker is not automatically restarted in HPA-396.

## Scoring / comparison

Build complete OaF and IDM cohorts over the same fixed population using unchanged HPA-325 rails.

`idm_comparison.py`:

1. reads published reports through public readers;
2. requires matching reference/timing/taxonomy/scoring identities;
3. uses strict identical source/input pairing;
4. publishes deterministic paired per-song/per-class deltas;
5. reports each full eligible/success/failure population before the intersection;
6. preserves native IDM failure histogram, mapped/unmapped/native-class coverage, runtime/RTF/peak memory where present, and velocity diagnostics;
7. never merges full-mix smoke into the headline stem result.

No automatic winner policy is added.

## Five-song full-mix smoke

### Deterministic pre-IDM selection rule

When the real production HPA-328 handoff exists, form candidates from rows whose HTDemucs input is successful/resumed and whose reference mapping can be loaded. Compute only pre-IDM facts:

```text
source_duration_sec
common_reference_event_count
simfile_id
```

Select without replacement in this exact order:

1. `short`: minimum `(source_duration_sec, simfile_id)`;
2. `long`: maximum duration, tie broken by lowest `simfile_id`;
3. `sparse`: minimum `(common_reference_event_count, simfile_id)` among remaining rows;
4. `dense`: maximum event count, tie broken by lowest `simfile_id`, among remaining rows;
5. `median_duration`: remaining row closest to the median candidate duration, tie broken by lowest `simfile_id`.

If fewer than five eligible candidates remain, full-mix smoke is operationally blocked; do not broaden membership or use fixtures.

Immediately write the resulting real IDs to:

```json
{"cases":[{"reason":"short","simfile_id":1},{"reason":"long","simfile_id":2},{"reason":"sparse","simfile_id":3},{"reason":"dense","simfile_id":4},{"reason":"median_duration","simfile_id":5}],"schema":"crux.idm-smoke/v1"}
```

The numeric values above are schema-shape examples only. The committed file must contain actual production handoff IDs and is committed **before any IDM request using HPA-328 stem/full-mix inputs**.

No OaF, separator, or IDM score is available to the selection function.

### Separate request

Full-mix source access does not leak into the primary stem request:

```python
@dataclass(frozen=True)
class IdmFullMixSmokeRequest:
    separation_handoff_path: Path
    reference_manifest_path: Path
    timing_manifest_path: Path
    smoke_manifest_path: Path
    source_cache_dir: Path
    output_dir: Path
    model_lock_path: Path
    model_root: Path
    runtime_python: Path
    crux_commit: str | None = None
```

Materialize authoritative full mix only for those five IDs with the historical OaF full-mix canonicalization and a separate input-view/config identity. Report separately.

## Risks / hard gates

### Missing production HPA-328 evidence

Current merged HPA-328 code has no production Task 11 handoff. This blocks real smoke-ID freeze and real HPA-396 execution. Offline implementation/tests may proceed; never fabricate membership.

### Weight-license ambiguity

Missing or contradictory checkpoint-license evidence stops scored inference.

### PeakPicking API mismatch

Task 0 must execute the exact pinned decoder sigmoid + `peak_picking_val` path. Failure is an upstream compatibility blocker, not permission to tune/reimplement silently.

### librosa resampling

Default `librosa.load(path)` would alter the fixed input. The soundfile loader and format tests make this impossible by contract.

### Timeout-dependent pairability

A timeout changes success/pairable population. `1800` seconds is frozen in inference identity before score inspection.

### Zero-hit scoring

An empty valid IDM prediction must still score through the mandatory zero-hit map entry.

### Peak metadata loss

Frame index and pre-MIDI native velocity are cheap at inference time and expensive to recover. Persist both before any real pilot.

## Verification

Offline CI uses synthetic/fake IDM processes and never installs/downloads the isolated runtime/model.

Required coverage:

- runtime/model lock canonical identity;
- real Task 0 upstream demo + soundfile tensor + PeakPicking feasibility gate;
- wrong WAV sample rate/channel/subtype rejection;
- exact isolated-runtime argv and fixed timeout forwarding;
- third descriptor/prediction-v2 family;
- exact frame/native-velocity metadata format;
- OaF/MuScriptor regressions;
- mandatory zero-hit IDM scoring;
- exact class-index↔native-class invariant;
- `map_idm_prediction()` keyed on `native_class_id`;
- `TT_LMT -> ClassMapping(None, "tom")`;
- worker ready/request/error/poison/close behavior;
- production-handoff/reference/timing/root identity checks;
- fixed native→HPA-325 failure table;
- exact resume without separator/OaF rerun;
- strict identical-input comparison;
- complete population plus paired intersection reporting;
- deterministic pre-IDM smoke selection/manifest validation;
- smoke/source-cache separation from the primary request;
- deterministic outputs and CLI 0/1/2 semantics.

## Non-goals

- Generic backend/plugin registry.
- Generic runner/comparison framework.
- Prediction v3.
- Full-corpus IDM evaluation.
- IDM training/fine-tuning.
- Manual onset correction.
- Required IDM reconstructed stems or SDR evaluation.
- Feeding IDM-generated stems to OaF.
- Rerunning HTDemucs or OaF.
- Changing HPA-325 scorer/matcher/tolerances.
- Velocity-aware scoring.
- Automated model/input-view winner selection.
- Production DTX chart/editor integration.
