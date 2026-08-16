# HPA-395 MuScriptor Full-Mix Benchmark Design

**Status:** Proposed  
**Date:** 2026-08-15  
**Linear:** HPA-395 — Integrate and benchmark frozen MuScriptor drum-only inference against OaF

## Decision summary

Implement MuScriptor as a **benchmark comparator only**, not a production backend architecture:

1. freeze official MuScriptor `v0.3.0` plus one exact gated Hugging Face checkpoint before scored inference;
2. add one optional `MuscriptorBackend` implementing the existing `TranscriptionBackend` protocol and consume native `NoteStartEvent` onsets with hard `instruments=["drums"]`;
3. promote the three model-neutral execution seams that now have two concrete callers: verified source resolution, canonical full-mix materialization, and immutable prediction-path construction;
4. keep `crux.drum-prediction-events/v2`, extending both write- and read-path validation for exactly two known backend families while preserving every OaF invariant;
5. freeze one code-owned MuScriptor MIDI-pitch map before inference and persist unsupported pitches as unmapped evidence;
6. add one concrete `muscriptor_corpus_run.py`, reuse HPA-325/HPA-327 reporting, and compare OaF/MuScriptor by joining the already-published HPA-325 score CSVs rather than re-scoring them.

Do **not** add a generic runner/base class, production registry entry, second prediction schema, mapping DSL, database, queue, second scorer, model-selection DSL, stem separation, tuning, or fine-tuning.

## Why HPA-395 is next

HPA-395 is the highest-priority unstarted Crux issue whose dependency chain is complete. HPA-325 scoring, HPA-326 broad OaF inference, and HPA-327 reviewed-subset work are complete. HPA-395 is High priority and blocks HPA-329 final findings.

HPA-328 is also unblocked but Medium priority. HPA-396 still depends on HPA-328. HPA-562 becomes useful once a second comparison population exists.

## Goals

HPA-395 must provide:

- one reproducible MuScriptor package/checkpoint/config identity;
- hard drum-only deterministic decoding with no post-score tuning;
- raw native onset and MIDI-pitch provenance;
- the same authoritative source/reference identity and byte-identical Crux canonical full-mix input as OaF;
- HPA-325 broad and HPA-327 reviewed-subset reports at 30/50/100 ms, raw and aligned;
- visible mapped/unmapped native pitch coverage;
- immutable resumable prediction artifacts and explicit per-song failures;
- paired per-song **and per-class** OaF deltas derived from the published HPA-325 report artifacts;
- runtime/device/dtype/memory evidence where cheap;
- separate code-license and weight-license evidence.

## Non-goals

This ticket does not:

- replace OaF as the primary baseline;
- expose MuScriptor through `default_backend_registry()` or the production API;
- benchmark several checkpoints and choose by F1;
- use `transcribe_to_midi()` or beat/downbeat correction;
- change HPA-323/HPA-324/HPA-327 membership or lineage;
- change HPA-325 tolerance, matcher, alignment, or aggregate semantics;
- generalize comparison to arbitrary models; HPA-562 owns that later.

## Frozen upstream boundary

Use the official release, not moving `main`:

```text
package: muscriptor==0.3.0
release tag: v0.3.0
release commit: d73147e75e5b9b0c0a79ebe154587db4fd603e0c
code license: MIT
candidate checkpoint size: medium
```

MuScriptor 0.3.0 exposes:

```python
TranscriptionModel.load_model(weights_path=..., device=..., dtype=...)
TranscriptionModel.transcribe(
    audio,
    use_sampling=False,
    temperature=1.0,
    cfg_coef=1.0,
    instruments=["drums"],
    batch_size=1,
    no_eos_is_ok=True,
    beam_size=1,
    prelude_forcing=True,
)
```

The event stream contains `NoteStartEvent`, `NoteEndEvent`, and `ProgressEvent`. Drum hits carry `instrument == "drums"`, native MIDI `pitch`, and `start_time`. Upstream splits audio into five-second chunks and internally targets 16 kHz audio.

The official code and model weights have different licenses. Missing or contradictory weight-license evidence is fatal before inference.

## Exact checkpoint and model identity

MuScriptor's convenient `"medium"` / unqualified `hf://...` path follows mutable repository state. Scored runs must instead use an exact local checkpoint.

The one-time freeze script must:

1. resolve `MuScriptor/muscriptor-medium` to an exact Hugging Face commit SHA;
2. download `model.safetensors` and `config.json` at that exact revision;
3. compute SHA-256 and byte length for both;
4. verify official weight-license metadata;
5. write `runtime/muscriptor/model.json` from the resolved evidence.

Medium is the first candidate. It may be replaced once by small only when a pre-score feasibility run demonstrates that medium is impractical on the selected hardware. F1 must not participate in that decision.

### `model_id` grammar

The descriptor cannot contain an unspecified future string. Freeze the grammar now:

```text
muscriptor-<variant>-<revision12>-<checkpoint12>
```

where:

```text
variant      = medium | small
revision12   = first 12 lowercase hex characters of checkpoint_revision
checkpoint12 = first 12 lowercase hex characters of checkpoint_sha256
```

Examples are shape-only; the actual digest fragments come from the authenticated freeze:

```text
muscriptor-medium-0123456789ab-fedcba987654
muscriptor-small-0123456789ab-fedcba987654
```

`normalize_known_backend_descriptor()` validates that grammar for MuScriptor. The MuScriptor runner separately derives the exact expected `model_id` from the loaded model lock and requires descriptor equality. OaF keeps its existing exact constant-equality validation.

## Model lock remains explicit

Keep a closed `crux.muscriptor-model/v1` lock. HPA-395 explicitly requires the lock to contain both acquired identity and the deterministic execution contract, so do **not** remove fixed decode/preprocessing fields merely because they have one valid v1 value.

The lock contains at minimum:

```text
package_name/package_version
upstream_source_commit
code_license
weight_license
checkpoint_repo_id
checkpoint_revision
checkpoint_filename/checkpoint_sha256/checkpoint_byte_length
checkpoint_config_filename/checkpoint_config_sha256/checkpoint_config_byte_length
checkpoint_variant
model_id
device
dtype
input_sample_rate_hz
chunk_duration_sec
use_sampling
temperature
cfg_coef
instruments
batch_size
no_eos_is_ok
beam_size
prelude_forcing
native_output_space_id
native_metadata_schema_id
training_data_map_id
```

The same inference-semantic values are folded into `crux.muscriptor-inference-config/v1` and its hash. Duplication here is intentional evidence required by the ticket: the model lock records the frozen chosen backend contract; the run config binds that contract to a concrete cohort execution.

No scored CLI flag overrides these fields.

## Dependency isolation

Add MuScriptor as an optional benchmark dependency:

```toml
[project.optional-dependencies]
muscriptor = ["muscriptor==0.3.0"]
```

The adapter lazily imports MuScriptor. Normal Crux CLI/API/tests work without MuScriptor installed. CI uses fake model/event objects and never downloads gated weights.

If `uv lock` proves MuScriptor cannot coexist with Python 3.12, add a small subprocess runtime only then. Do not preemptively create a second image/protocol.

## Existing seams and required promotions

### Backend protocol

Reuse unchanged from `src/benchmark/backends/base.py`:

```text
CanonicalAudio
NativeEvent
NativePrediction
TranscriptionBackend
```

`MuscriptorBackend` lives at `src/benchmark/backends/muscriptor.py` and is not registered in `default_backend_registry()`.

### Verified source resolution

`oaf_corpus_run.py` currently owns model-neutral `ResolvedSourceAudio`, `_remote_from_source_mapping()`, `_source_audio_parts()`, and `_resolve_source_audio()` while importing the OaF runtime at module load.

Now that MuScriptor is a second caller, move that resolver/value object to `src/benchmark/corpus_cache.py` and expose only:

```python
resolve_source_audio(..., load_body: bool = True) -> ResolvedSourceAudio
```

Preserve current cache-index fallback, exact source digest binding, header-duration probe, `load_body=False`, and per-item re-pin behavior. Both concrete runners use this one implementation.

### Canonical full-mix materialization

Promote the current OaF conversion to `src/benchmark/input_view.py` **with the path-containment guard intact**:

```python
def materialize_full_mix_audio(
    source_audio: ResolvedSourceAudio,
    output_path: Path,
    *,
    input_root: Path,
    input_view_id: str,
    max_input_audio_frames: int | None,
) -> CanonicalAudio:
    ...
```

The helper must:

- require `ResolvedSourceAudio` and `Path` arguments;
- require `output_path.resolve()` to stay beneath `input_root.resolve()`;
- preserve `librosa.load(..., sr=44100, mono=True, res_type="soxr_hq")`;
- preserve `soundfile.write(..., 44100, format="WAV", subtype="PCM_16")`;
- call `load_materialized_audio()` for exact WAV validation/hash.

Keep the historical persisted input-view ID exactly:

```text
crux.oaf-full-mix-mono44k1-pcm16/v1
```

MuScriptor receives the exact same canonical WAV bytes. Its 16 kHz preprocessing remains model-internal. Renaming the historical input-view ID would destroy pairing with persisted HPA-326 evidence.

The OaF call site still validates/loads its `OafModelConfig` and passes `config.max_input_audio_frames`; the neutral helper does not import OaF config types.

### Immutable prediction path

Move the existing descriptor/config/source-keyed `prediction_path()` unchanged from `oaf_corpus_run.py` to `src/benchmark/prediction_artifact.py`:

```python
def prediction_path(
    output_dir: Path,
    *,
    simfile_id: int,
    source_audio_sha256: str,
    backend_descriptor_sha256: str,
    inference_config_sha256: str,
) -> Path:
    ...
```

Both runners use that exact path convention. Do not introduce a path strategy abstraction.

## MuScriptor adapter

Construction validates the model lock, package version, local checkpoint/config hashes, requested device/dtype, and frozen descriptor before loading one model for the run.

For each canonical WAV, call the exact frozen `transcribe()` arguments and consume the stream once. Persist only `NoteStartEvent` as hits:

```python
NativeEvent(
    time_sec=event.start_time,
    native_class_id=f"drums:midi_{event.pitch}",
    model_output_bin=None,
    native_midi_note=event.pitch,
    native_metadata={"instrument_group": event.instrument},
    confidence=None,
    velocity_midi=None,
)
```

Rules:

- `instrument` must be exactly `"drums"`;
- pitch is integer MIDI `0..127`;
- onset is finite and nonnegative;
- `NoteEndEvent` does not create a second hit;
- `ProgressEvent` is ignored for scoring;
- do not synthesize confidence/velocity;
- do not serialize/reparse MIDI;
- do not apply beat/downbeat timing correction.

## Descriptor and prediction-v2 compatibility

Keep descriptor schema `crux.transcription-backend-descriptor/v2`. Extend `normalize_known_backend_descriptor()` to exactly two known families.

MuScriptor fixed identities are:

```text
backend_id: muscriptor-v0.3.0-drums-v1
architecture_id: muscriptor-transformer-v0.3.0
native_output_space_id: muscriptor-drums-midi128-v1
native_metadata_schema_id: muscriptor-note-start-metadata-v1
prediction_schema: crux.drum-prediction-events/v2
```

`model_id` follows the grammar above. `training_data_map_id` is an opaque upstream-version identity rather than an unsupported claim about training composition.

Use a small per-family descriptor policy:

```text
family -> exact key set + exact constant identities + optional patterned fields
```

OaF has no patterned fields and keeps current constant equality. MuScriptor pattern-checks only `model_id`; its runner later checks the exact lock-derived value.

### Write path and read path must both dispatch

Prediction v2 currently contains two redundant OaF-only backend guards. Remove both after the shared descriptor normalizer succeeds:

1. `_build_header()` write-path OaF-only `backend_id` guard;
2. `_validate_header()` read-path OaF-only `backend_id` guard.

Do **not** alter `_validate_header()`'s model-neutral descriptor↔header equality checks for:

```text
architecture_id
model_id
native_metadata_schema_id
native_output_space_id
training_data_map_id
upstream_source_commit
```

Add `muscriptor-note-start-metadata-v1` to `NATIVE_METADATA_SCHEMAS` with exactly:

```python
{"instrument_group": {"drums"}}
```

Then dispatch native event invariants by the already-validated backend family:

- OaF keeps all existing non-null bin/MIDI/confidence/velocity, `midi_<note>`, and metadata invariants;
- MuScriptor requires MIDI pitch, null bin/confidence/velocity, exact drum metadata, and `native_class_id == "drums:midi_<pitch>"`.

Never make OaF-required fields globally optional.

Keep the existing prediction-v2 schema golden and registry **byte-for-byte unchanged**. MuScriptor canonical render/read/render coverage belongs in focused tests.

### Zero-hit prediction-map identity

Do not source an empty artifact's map identity from the expected cohort identity; that would make the later identity comparison tautological and weaken OaF.

Keep `_artifact_identity_from_artifacts()` signature unchanged and add one closed backend-to-map table:

```python
ZERO_HIT_PREDICTION_MAPS = {
    OAF_BACKEND_ID: OAF_PREDICTION_MAP_ID,
    MUSCRIPTOR_BACKEND_ID: MUSCRIPTOR_PREDICTION_MAP_ID,
}
```

For non-empty artifacts, continue deriving the map from event records and reject mixed maps. For empty artifacts, look up by validated `backend_id`; an unknown backend still raises `ValueError("empty prediction artifact has no prediction_map_version")`. `_validate_artifact_identity()` remains the independent comparison against `CohortIdentity.prediction_map_version`.

## Frozen MuScriptor pitch mapping

The mapping is part of the scored model identity because HPA-325 scores only mapped prediction events. Freeze it before real inference:

```text
MUSCRIPTOR_PREDICTION_MAP_ID = crux.prediction-map/muscriptor-drums-v1
```

Define **one** string-keyed `PredictionMap.classes` table. Do not also define a second integer-keyed copy.

| MIDI key | GM name | Detailed class | Common class |
| ---: | --- | --- | --- |
| `35` | Acoustic Bass Drum | `kick` | `kick` |
| `36` | Bass Drum 1 | `kick` | `kick` |
| `38` | Acoustic Snare | `snare` | `snare` |
| `40` | Electric Snare | `snare` | `snare` |
| `41` | Low Floor Tom | `low_or_floor_tom` | `tom` |
| `43` | High Floor Tom | `low_or_floor_tom` | `tom` |
| `45` | Low Tom | `low_or_floor_tom` | `tom` |
| `47` | Low-Mid Tom | `low_or_floor_tom` | `tom` |
| `48` | Hi-Mid Tom | `high_tom` | `tom` |
| `50` | High Tom | `high_tom` | `tom` |
| `42` | Closed Hi-Hat | `closed_hihat` | `hihat` |
| `44` | Pedal Hi-Hat | `closed_hihat` | `hihat` |
| `46` | Open Hi-Hat | `open_hihat` | `hihat` |
| `49` | Crash Cymbal 1 | `crash` | `crash` |
| `57` | Crash Cymbal 2 | `crash` | `crash` |
| `51` | Ride Cymbal 1 | `ride` | `ride` |
| `53` | Ride Bell | `ride` | `ride` |
| `59` | Ride Cymbal 2 | `ride` | `ride` |

Everything else is intentionally unmapped in v1, including side-stick, clap, splash/Chinese cymbal, cowbell, tambourine, and other percussion without an unambiguous existing Crux detailed class.

`map_muscriptor_prediction()` follows `map_oaf_prediction()` semantics and begins with the same two identity guards:

```python
prediction.descriptor.payload["backend_id"] == prediction_map.backend_id
prediction.descriptor.payload["native_output_space_id"] == prediction_map.native_output_space_id
```

For each native hit, look up:

```python
key = str(native.native_midi_note)
```

Every event is retained. Unsupported pitches get `mapping_status="unmapped"` and null canonical/common classes. Do not reuse `map_midi_events()`, which drops unmapped notes.

Changing the table later requires a new prediction-map ID and newly mapped prediction artifacts.

## MuScriptor corpus run

Add `src/benchmark/muscriptor_corpus_run.py` with concrete schemas:

```text
crux.muscriptor-corpus-run/v1
crux.muscriptor-inference-config/v1
```

Keep two concrete runners. No base class.

Run identity binds:

```text
reference/timing identities
model-lock SHA
checkpoint revision/SHA
backend descriptor SHA
inference-config SHA
MUSCRIPTOR_PREDICTION_MAP_ID
shared input-view ID
adapter revision
Crux commit
include/exclude scope
```

Execution order mirrors HPA-326 semantically:

```text
preflight
-> durable initial snapshot
-> verified source pass
-> per-item re-pin
-> canonical materialize
-> resume-or-infer
-> map/publish/checkpoint
-> close
-> runtime evidence
-> HPA-325 broad score/report
```

Per-song source/decode/inference/publication failures remain item-local while model identity remains trustworthy. Model/checkpoint/descriptor/config integrity failures stop the run.

## Fixed smoke gate

Commit `runtime/muscriptor/smoke.json` before first inference with exactly five pre-model roles:

```text
short
long
dense
sparse
non_drum_heavy
```

At the end of the model-lock implementation task, perform one authenticated freeze/preflight run against the official gated source. This verifies the assumed companion filenames, exact revision resolution, and license metadata shape before the descriptor/runner implementation is considered ready to execute against real weights.

If credentials are unavailable in an implementation environment, offline unit work may continue, but the task remains operationally incomplete and **no real inference or scored result may run** until this preflight succeeds.

The smoke gate verifies package/checkpoint/device readiness, hard drum-only output, native pitch/onset validity, repeatability, five-second boundary behavior, resume identity, runtime, and memory. It may reject medium for feasibility but may not tune by F1.

## Reviewed-subset scoring

Keep HPA-327 membership immutable. Extract one model-neutral core that owns both subset lineage and filtering/scoring:

```python
def score_reviewed_subset_cohort(
    parent_identity: CohortIdentity,
    parent_items: tuple[CohortItem, ...],
    reference: LoadedReferenceSetManifest,
    timing: LoadedReferenceTimingManifest,
    subset: LoadedReviewedSubsetManifest,
    *,
    output_dir: Path,
) -> ScoreReviewedSubsetOutcome:
    ...
```

The core validates:

- subset source-reference hash/version == supplied HPA-324 manifest;
- subset source-timing hash/version == supplied HPA-323 manifest;
- every subset member exists in the parent population;
- subset cohort ID derives from parent cohort ID + subset manifest SHA;
- HPA-325 diagnostics are requested only for successful selected songs.

Output-directory alias protection remains in the model-specific wrapper because only that wrapper knows the parent run's broad report path.

Model-specific wrappers own only:

```text
parse_*_corpus_run()
snapshot ↔ supplied HPA-323/HPA-324 validation
build_*_cohort_from_snapshot()
parent broad-report output alias protection
delegation
```

Then they delegate to the shared core. No inference occurs during subset scoring.

## Narrow OaF pairing from published reports

Do **not** re-score persisted events inside HPA-395 comparison. HPA-325 already published the authoritative score rows.

`muscriptor_comparison.py` reads:

```text
<oaf reports>/per_song.csv
<oaf reports>/per_class.csv
<oaf reports>/items.csv
<muscriptor reports>/per_song.csv
<muscriptor reports>/per_class.csv
<muscriptor reports>/items.csv
```

and the two run snapshots.

Before joining, validate each report's identity columns against its run snapshot/cohort identity, including model ID, model-lock SHA, prediction-map version, input-view ID, scoring version, and cohort ID. Reject malformed/duplicate score keys.

Pairing identity comes from the run snapshots, not score CSV rows:

### Run-level identity

```text
reference_manifest_sha256
reference_timing_manifest_sha256/reference_timing_version
input_view_id (or the run-level inference config field that carries it)
```

### Successful item-level identity

```text
simfile_id
source_audio_sha256
input_audio_sha256
```

`input_view_id` is **not** read from an item row.

For the same `simfile_id` + source-audio identity, any run-level input-view mismatch or item-level input-audio hash mismatch is a fatal canonical-input integrity error, not a silent exclusion.

Join successful `per_song.csv` rows on:

```text
(simfile_id, tolerance_ms, mode)
```

and emit `MuScriptor - OaF` precision/recall/F1 deltas.

Join successful `per_class.csv` rows on:

```text
(simfile_id, tolerance_ms, mode, common_class)
```

and emit paired per-class precision/recall/F1 deltas. Missing class support stays explicit rather than inventing a zero row unless HPA-325 already published one.

Report each model's full success/failure population from `items.csv`/run evidence, plus paired/intersection/exclusion counts. Optional reviewed-subset comparison filters the exact HPA-327 IDs before the joins.

No significance testing, ranking, bootstrap intervals, arbitrary model labels, or second scoring path. HPA-562 can generalize later.

## Runtime evidence

Persist only cheap evidence:

- device/dtype;
- checkpoint identity/size;
- measured wall time/audio duration;
- aggregate RTF and projected wall time when duration coverage is complete;
- peak process RSS where cheap;
- accelerator peak memory only when directly exposed.

Missing device-specific memory is `None`, not a failure. No telemetry subsystem.

## CLI surface

Add only:

```text
crux benchmark run-muscriptor-corpus
crux benchmark score-muscriptor-reviewed-subset
crux benchmark compare-oaf-muscriptor
```

`run-muscriptor-corpus` mirrors HPA-326 manifest/timing/cache/output/include/exclude/resume flags. Model size, device, dtype, temperature, beam, instrument list, and mapping version remain frozen, not CLI knobs.

Extend the existing `tests/test_cli_benchmark.py`; do not create a second CLI harness.

## Failure policy

Fatal before inference:

- malformed/unfrozen model lock;
- wrong package/source identity;
- missing/inconsistent license evidence;
- checkpoint/config hash or size mismatch;
- unavailable locked device/dtype;
- reference/timing lineage mismatch;
- invalid scope;
- descriptor/config/map identity mismatch.

Item-local while model integrity remains valid:

- source decode/materialization failure;
- per-song inference failure;
- invalid native event for that song;
- immutable prediction publication failure.

If a failure invalidates loaded model state, stop rather than pretending later rows are independent successes.

## Acceptance

HPA-395 is complete when:

- exact package/checkpoint/config/license/device/dtype/decode settings are frozen before scores;
- the authenticated freeze path has been exercised against the official gated source;
- inference always uses hard drum-only settings;
- raw MuScriptor onset + native pitch persist directly in prediction v2;
- the exact v1 pitch map is committed before real inference and unsupported pitches remain visible;
- both prediction-v2 write and read paths accept MuScriptor while existing OaF invariants/golden remain unchanged;
- zero-hit OaF and MuScriptor artifacts retain independently derived map identity;
- shared source resolution/materialization/path helpers preserve existing OaF behavior and containment guards;
- overlapping same-source OaF/MuScriptor rows have identical run-level input-view and item-level canonical-input hash;
- broad and HPA-327 reviewed reports use unchanged HPA-325 semantics;
- comparison joins the published per-song/per-class reports rather than re-scoring;
- paired per-song/per-class deltas, population failures, runtime evidence, and license constraints are reported;
- no score-informed checkpoint, decode, mapping, or post-processing choice occurs.

## Reuse survey

| Need | Decision | Existing seam |
| --- | --- | --- |
| backend protocol | REUSE | `backends/base.py` |
| production registry | DO NOT EXTEND | `backend_registry.py` |
| model lock | EXTEND pattern | OaF model lock; MuScriptor source is different |
| source resolution | PROMOTE | `oaf_corpus_run.py` -> `corpus_cache.py` |
| full-mix bytes | PROMOTE | OaF materializer -> `input_view.py`, keep containment guard |
| prediction path | PROMOTE | `oaf_corpus_run.py` -> `prediction_artifact.py` |
| descriptor | EXTEND | two frozen families; MuScriptor `model_id` patterned + lock-bound |
| prediction v2 | EXTEND | both `_build_header()` and `_validate_header()` + event dispatch |
| zero-hit map | EXTEND | closed backend→map dict; preserve independent check |
| pitch mapping | EXTEND | one string-keyed `PredictionMap`; persist unmapped |
| scoring/reports | REUSE | HPA-325 published CSVs |
| reviewed subset | EXTEND narrow core | lineage + membership + score delegation |
| corpus runner | NEW concrete module | no generic runner |
| paired deltas | NEW narrow join | published `per_song.csv` + `per_class.csv`, no re-score |
| CLI tests | EXTEND | `tests/test_cli_benchmark.py` |

## Explicitly rejected review suggestion

Do not remove the deterministic decode/preprocessing fields from `runtime/muscriptor/model.json`. HPA-395's Linear contract explicitly requires the MuScriptor model lock to contain device/dtype, audio decoding/resampling behavior, `instruments=["drums"]`, deterministic decode settings, chunk duration/boundary behavior, and adapter identity. Moving those facts exclusively to code constants/run config would make the implementation smaller but would no longer satisfy the ticket's frozen-lock evidence contract.
