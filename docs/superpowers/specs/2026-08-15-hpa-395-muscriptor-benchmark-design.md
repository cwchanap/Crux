# HPA-395 MuScriptor Full-Mix Benchmark Design

**Status:** Proposed  
**Date:** 2026-08-15  
**Linear:** HPA-395 — Integrate and benchmark frozen MuScriptor drum-only inference against OaF

## Decision summary

Implement MuScriptor as a **benchmark comparator only**, not a new production backend architecture:

1. freeze official MuScriptor `v0.3.0` plus one exact gated Hugging Face checkpoint before any scored run;
2. add one `MuscriptorBackend` implementing the existing `TranscriptionBackend` protocol and consume its native `NoteStartEvent` stream with hard `instruments=["drums"]`;
3. reuse the exact HPA-326 44.1 kHz mono PCM16 full-mix materialization so OaF and MuScriptor are bound to identical pre-model input bytes;
4. keep `crux.drum-prediction-events/v2` and HPA-325 scoring, extending only their frozen backend-specific identity/validation seams;
5. freeze one explicit MuScriptor MIDI-pitch mapping before inference, seeded from the existing MIDI map plus only clear General MIDI aliases;
6. promote the already-model-neutral source resolver and prediction-path helper now that they have a second caller, while keeping two concrete runners;
7. add one MuScriptor-specific persisted corpus runner, reuse HPA-327 reviewed-subset membership, and publish a narrow exact-identity OaF-vs-MuScriptor delta report.

Do **not** add a generic experiment runner, model registry, plugin framework, production backend selector, second scorer, database, job queue, MIDI round-trip, beat-grid correction, stem separation, model tuning, or fine-tuning.

## Why HPA-395 is next

HPA-395 is the highest-priority unstarted Crux issue whose dependency chain is complete. HPA-325 scoring, HPA-326 broad OaF inference, and HPA-327 reviewed-subset work are all complete. HPA-395 is High priority and blocks HPA-329 final findings.

HPA-328 is also unblocked but Medium priority. HPA-396 still depends on HPA-328. HPA-562 becomes more useful once a second comparison population exists. HPA-395 therefore gives the next useful model-quality signal without opening another prerequisite chain.

## Goals

HPA-395 must provide:

- one reproducible MuScriptor package/checkpoint/config identity;
- hard drum-only decoding with no post-score tuning;
- raw native onset and MIDI-pitch provenance;
- a frozen, versioned native-pitch-to-Crux mapping decided before scored results;
- the same authoritative source/reference identity and byte-identical Crux canonical full-mix input as OaF;
- HPA-325 broad and HPA-327 reviewed-subset reports at 30/50/100 ms, raw and aligned;
- visible mapped/unmapped native drum-pitch coverage;
- immutable resumable prediction artifacts and explicit per-song failures;
- exact-identity paired OaF deltas;
- wall time/RTF/device/dtype/peak-memory evidence;
- separate code-license and weight-license evidence.

## Non-goals

This ticket does not:

- replace OaF as the baseline;
- expose MuScriptor through `default_backend_registry()` or the production API;
- select a checkpoint or decoding setting by benchmark F1;
- use `transcribe_to_midi()` or beat/downbeat correction;
- change HPA-323/HPA-324/HPA-327 membership or lineage;
- change the reference taxonomy, tolerance windows, matcher, or report semantics;
- generalize comparison to arbitrary N models; HPA-562 owns that later.

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

The event stream contains `NoteStartEvent`, `NoteEndEvent`, and `ProgressEvent`; drum hits carry `instrument == "drums"`, native MIDI `pitch`, and `start_time`. Upstream splits audio into five-second chunks and internally targets 16 kHz audio.

The planning-time upstream documentation identifies code and model weights under different licenses. The scored preflight must verify the exact official weight-license metadata and persist it separately; a missing or contradictory license is fatal before inference.

## Exact checkpoint freeze

MuScriptor's convenient `"medium"`/`hf://...` flow resolves Hugging Face content without an explicit immutable revision. That is unsuitable as the scored benchmark identity.

Add a one-time acquisition script that:

1. resolves `MuScriptor/muscriptor-medium` to an exact Hugging Face commit SHA;
2. downloads `model.safetensors` and companion `config.json` at that exact revision;
3. computes SHA-256 and byte size for both;
4. verifies official model-license metadata;
5. writes `runtime/muscriptor/model.json` only from those resolved values.

Scored code loads MuScriptor with the verified **local safetensors path**, never a mutable alias.

Mirror the existing OaF model-lock rail with a closed `crux.muscriptor-model/v1` object containing package/source identity, checkpoint repo/revision/files/hashes/sizes, code/weight licenses, device/dtype, internal sample rate/chunk duration, exact decoding arguments, output/metadata-space identities, and adapter/inference identity inputs.

No scored CLI flag overrides these fields. An intentional configuration change creates a different lock/config/run identity.

### Medium-first feasibility rule

Medium is the fixed first candidate. Before broad/reviewed scoring, run a committed pre-score smoke set chosen only from reference/audit/source evidence. Medium may be replaced once by small only when runtime/memory/device stability makes medium technically impractical. Do not compare checkpoint F1 to make that choice.

## Dependency isolation

Add MuScriptor only as an optional benchmark dependency:

```toml
[project.optional-dependencies]
muscriptor = ["muscriptor==0.3.0"]
```

The adapter lazily imports MuScriptor. Normal Crux CLI/API/tests must work without PyTorch/MuScriptor installed. Unit tests inject fake model/event objects; CI does not authenticate to Hugging Face or fetch the real checkpoint.

If `uv lock` proves the package cannot coexist with the current Python 3.12 environment, introduce a small dedicated runtime only then. A subprocess protocol is not part of the initial design.

## Existing seams to reuse

### `TranscriptionBackend`

Reuse `src/benchmark/backends/base.py`:

```text
CanonicalAudio
NativeEvent
NativePrediction
TranscriptionBackend
```

`MuscriptorBackend` lives at `src/benchmark/backends/muscriptor.py` and is not added to the default production registry.

### Exact full-mix input

HPA-326 currently materializes authoritative source audio as:

```text
44,100 Hz
mono
WAV PCM_16
librosa res_type="soxr_hq"
```

Promote only this model-neutral materialization to `src/benchmark/input_view.py` and characterize output bytes before moving it. Preserve the existing historical input-view ID:

```text
crux.oaf-full-mix-mono44k1-pcm16/v1
```

The name remains OaF-shaped because it is already persisted evidence. Renaming it would destroy exact pairing with HPA-326.

MuScriptor receives this canonical WAV path. Its 16 kHz resampling remains internal model preprocessing. The result is a shared `input_audio_sha256` before model-native preprocessing diverges.

### Verified source resolution

`oaf_corpus_run.py` also currently owns a model-neutral `ResolvedSourceAudio` value plus verified-cache resolution/re-pin helpers. A second corpus runner makes this a real two-caller seam, and importing it from `oaf_corpus_run.py` would pull the OaF runtime into the MuScriptor path at module load.

Move the existing behavior, without semantic changes, to `src/benchmark/corpus_cache.py`:

```python
@dataclass(frozen=True)
class ResolvedSourceAudio:
    path: Path
    source_audio_id: str
    source_audio_sha256: str
    duration_sec: float
    content: bytes | None = None


def resolve_source_audio(
    source: RemoteObject | ManifestRowView | Mapping[str, object],
    cache_dir: Path,
    cache_index: CacheIndexStore | None = None,
    *,
    index: CacheIndexStore | None = None,
    source_audio_key: str | None = None,
    source_audio_content_hash: str | None = None,
    source_endpoint_sha256: str | None = None,
    source_bucket: str | None = None,
    load_body: bool = True,
) -> ResolvedSourceAudio:
    ...
```

Move the private `_remote_from_source_mapping()` and `_source_audio_parts()` with it as implementation details. OaF and MuScriptor both call the neutral helper. There is still no shared runner class.

### Prediction artifact location

`oaf_corpus_run.prediction_path()` is already backend-neutral: it keys immutable prediction v2 bytes by song, source hash, backend descriptor hash, and inference-config hash.

Move it unchanged to `src/benchmark/prediction_artifact.py`:

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

Both concrete runners reuse it. Do not create a second path convention.

### Reference/scoring/subset rails

Reuse unchanged where possible:

- `load_reference_set_manifest()`;
- `preflight_reference_mappings()`;
- `load_reference_timing_manifest()`;
- `score_cohort()`;
- `write_cohort_reports()`;
- `load_reviewed_subset_manifest()`.

Do not parse DTX again and do not create a MuScriptor-specific scorer.

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

- instrument must be exactly `"drums"` after requesting hard drum-only decoding;
- pitch must be integer MIDI 0..127;
- onset must be finite and nonnegative;
- `NoteEndEvent` is stream integrity/provenance only and creates no hit;
- `ProgressEvent` is ignored for scoring;
- no synthetic confidence/velocity;
- no MIDI serialization/reparse;
- no beat/downbeat timing correction.

## Descriptor and prediction-v2 compatibility

Keep descriptor schema `crux.transcription-backend-descriptor/v2`. Extend `normalize_known_backend_descriptor()` from one frozen OaF family to exactly two known families: existing OaF and MuScriptor v0.3.0.

MuScriptor identities are:

```text
backend_id: muscriptor-v0.3.0-drums-v1
architecture_id: muscriptor-transformer-v0.3.0
native_output_space_id: muscriptor-drums-midi128-v1
native_metadata_schema_id: muscriptor-note-start-metadata-v1
prediction_schema: crux.drum-prediction-events/v2
```

`model_id` is derived from the frozen checkpoint variant/revision/hash in `runtime/muscriptor/model.json`; the MuScriptor runner requires the descriptor to match that lock-derived model identity exactly. `training_data_map_id` is an opaque upstream-version identity, not a claim about training composition that the official release does not precisely publish.

Keep prediction schema v2. The extension must cover **both header and event validation**:

1. add `muscriptor-note-start-metadata-v1` to `NATIVE_METADATA_SCHEMAS` with exactly `{"instrument_group": {"drums"}}`;
2. `_build_header()` still calls `normalize_known_backend_descriptor()`, but after that succeeds it no longer has an extra `backend_id must be OaF` guard;
3. `_build_header()` still requires the descriptor's `prediction_schema == crux.drum-prediction-events/v2` and a known metadata schema;
4. `_normalize_event()` dispatches by the already-validated backend ID, not by global nullability;
5. OaF retains every existing non-null bin/MIDI/confidence/velocity, `midi_<note>`, and metadata invariant;
6. MuScriptor requires native MIDI, null bin/confidence/velocity, exact drum metadata, and `native_class_id == "drums:midi_<pitch>"`.

Do not make OaF-required fields optional globally.

The current prediction-v2 schema golden is one complete OaF artifact and the schema-golden registry intentionally has one fixed entry per schema. **Keep that golden and manifest unchanged.** Add MuScriptor canonical render/read/render coverage in focused prediction-artifact tests instead of mixing two backend headers into one golden or adding a duplicate schema entry.

### Zero-hit scorer closure

HPA-325 currently infers the prediction-map ID for empty prediction artifacts only for OaF. Make the internal artifact-identity reconstruction accept the expected `CohortIdentity`; for an empty but otherwise valid artifact use `identity.prediction_map_version`. This is a model-neutral closure and leaves non-empty artifact validation unchanged.

## Frozen MuScriptor pitch mapping

The pitch table is part of the scored model identity because HPA-325 scores only mapped prediction events. Therefore HPA-395 freezes the mapping **before any scored MuScriptor result is inspected**.

Use:

```text
MUSCRIPTOR_PREDICTION_MAP_ID = crux.prediction-map/muscriptor-drums-v1
```

Seed the table from the existing `DEFAULT_MIDI_NOTE_MAP`, then add only clear General MIDI aliases that collapse into the existing Crux classes:

| MIDI | GM name | Detailed class | Common class |
| ---: | --- | --- | --- |
| 35 | Acoustic Bass Drum | `kick` | `kick` |
| 36 | Bass Drum 1 | `kick` | `kick` |
| 38 | Acoustic Snare | `snare` | `snare` |
| 40 | Electric Snare | `snare` | `snare` |
| 41 | Low Floor Tom | `low_or_floor_tom` | `tom` |
| 43 | High Floor Tom | `low_or_floor_tom` | `tom` |
| 45 | Low Tom | `low_or_floor_tom` | `tom` |
| 47 | Low-Mid Tom | `low_or_floor_tom` | `tom` |
| 48 | Hi-Mid Tom | `high_tom` | `tom` |
| 50 | High Tom | `high_tom` | `tom` |
| 42 | Closed Hi-Hat | `closed_hihat` | `hihat` |
| 44 | Pedal Hi-Hat | `closed_hihat` | `hihat` |
| 46 | Open Hi-Hat | `open_hihat` | `hihat` |
| 49 | Crash Cymbal 1 | `crash` | `crash` |
| 57 | Crash Cymbal 2 | `crash` | `crash` |
| 51 | Ride Cymbal 1 | `ride` | `ride` |
| 53 | Ride Bell | `ride` | `ride` |
| 59 | Ride Cymbal 2 | `ride` | `ride` |

Everything else is intentionally unmapped in v1, including side-stick, clap, splash/Chinese cymbal, cowbell, tambourine, and other percussion for which the frozen Crux taxonomy has no unambiguous detailed class.

Implement the table as a closed code-owned mapping in `taxonomy.py`; do not expose a config file or mapping DSL. `map_muscriptor_prediction()` follows `map_oaf_prediction()` semantics: every native hit is retained in the artifact, and unsupported pitches are emitted with `mapping_status="unmapped"` and null canonical/common classes. Do **not** reuse `map_midi_events()`, which drops unmapped notes.

Changing this table later requires a new prediction-map ID and rescoring newly mapped artifacts. The implementation must not publish scores against an unfrozen map.

## MuScriptor corpus run

Add `src/benchmark/muscriptor_corpus_run.py` with concrete schemas:

```text
crux.muscriptor-corpus-run/v1
crux.muscriptor-inference-config/v1
```

Keep it separate from the large OaF runner instead of creating a generic runner base class.

Required shared seams are now explicit:

- verified source-audio resolution/re-pin from `corpus_cache.py`;
- shared full-mix materialization from `input_view.py`;
- descriptor/config/source-keyed immutable prediction path from `prediction_artifact.py`;
- HPA-325 cohort/report assembly.

Run identity binds reference/timing manifests, model-lock SHA, checkpoint revision/SHA, descriptor SHA, inference-config SHA, input-view ID, adapter revision, Crux commit, and include/exclude scope.

Execution order mirrors HPA-326 semantically:

```text
preflight
-> durable initial snapshot
-> verified source pass
-> per-item re-pin/materialize/resume-or-infer/map/publish/checkpoint
-> close model
-> runtime evidence
-> persisted CohortItems
-> HPA-325 broad score/report
```

Per-song source/decode/inference/publication problems remain explicit item failures when the model identity is still trustworthy. Model/checkpoint/descriptor/config integrity failures stop the run.

No shared runner base class, restart engine, queue, worker protocol, or generic error-policy framework is introduced.

## Fixed smoke gate

Commit `runtime/muscriptor/smoke.json` before first inference with exactly five pre-model roles:

```text
short
long
dense
sparse
non_drum_heavy
```

Verify package/checkpoint/device readiness, drum-only events, native pitch/onset validity, repeatability on at least one repeated song, behavior near five-second chunk boundaries, resume identity, runtime, and memory. The smoke gate can reject medium for feasibility but cannot tune by F1.

The smoke must run only after the `MUSCRIPTOR_PREDICTION_MAP_ID` and exact v1 pitch table above are committed.

## Reviewed-subset scoring

Keep HPA-327 membership immutable. Extract one small model-neutral core from `reviewed_subset.py` that receives a reconstructed parent `CohortIdentity`/`CohortItem` population plus the loaded subset, filters exact IDs, derives the subset cohort ID, and invokes HPA-325 with event diagnostics for successful selected songs.

Existing OaF and new MuScriptor wrappers each remain responsible for parsing their own run and validating HPA-323/HPA-324/HPA-327 lineage before calling the core.

## Narrow OaF pairing

Add `muscriptor_comparison.py` and `compare-oaf-muscriptor` only for this ticket. Reconstruct persisted OaF and MuScriptor cohorts and pair a song only when both succeeded and all of these match:

```text
simfile_id
reference manifest identity
timing manifest identity
source_audio_sha256
input_view_id
input_audio_sha256
```

For the same `simfile_id` and source-audio identity, an `input_view_id` or `input_audio_sha256` difference is a **fatal canonical-input integrity error**, not a silent exclusion: HPA-395 explicitly claims byte-identical Crux input before model-native preprocessing.

For each 30/50/100 ms tolerance and raw/aligned mode emit per-song OaF F1, MuScriptor F1, and `MuScriptor - OaF` delta plus pair count and mean/median delta. Optional HPA-327 filtering reuses the exact subset manifest.

Do not add significance testing, ranking, bootstrap intervals, or arbitrary model labels; HPA-562 can generalize later.

## Runtime evidence

Persist cheap operational evidence:

- device/dtype;
- checkpoint identity/size;
- measured wall time and audio duration;
- aggregate RTF and projected full wall time when coverage is complete;
- peak process RSS where cheaply available;
- accelerator peak memory only when the selected PyTorch device exposes it directly.

Missing device-specific memory is `None`, not a failure. No telemetry subsystem.

## CLI surface

Add only:

```text
crux benchmark run-muscriptor-corpus
crux benchmark score-muscriptor-reviewed-subset
crux benchmark compare-oaf-muscriptor
```

The runner mirrors HPA-326 manifest/timing/cache/output/include/exclude/resume flags. Model size, temperature, beam, instrument list, device, dtype, and mapping version are frozen and are not CLI knobs.

CLI coverage extends the existing `tests/test_cli_benchmark.py`; do not create a second CLI harness.

## Failure policy

Fatal before inference:

- malformed/unfrozen model lock;
- wrong package version;
- missing/inconsistent license evidence;
- checkpoint/config hash/size mismatch;
- unavailable locked device/dtype;
- reference/timing lineage mismatch;
- invalid scope;
- descriptor/config hash mismatch;
- MuScriptor prediction-map identity/table does not match the frozen v1 contract.

Item-local where model integrity remains valid:

- source decode/materialization failure;
- per-song MuScriptor inference failure;
- invalid native event for that song;
- immutable prediction publication failure.

If a failure invalidates the loaded model state, stop instead of pretending later rows are independent successes.

## Reuse survey

| Need | Decision | Existing seam |
| --- | --- | --- |
| backend protocol | REUSE | `backends/base.py` |
| production registry | DO NOT EXTEND | `backend_registry.py` |
| model lock / freeze | EXTEND PATTERN | OaF lock shape; OaF zip acquisition remains model-specific |
| full-mix bytes | PROMOTE REQUIRED | OaF materializer -> `input_view.py` |
| source resolve/re-pin | PROMOTE REQUIRED | `oaf_corpus_run.py` -> `corpus_cache.py` |
| prediction path | PROMOTE REQUIRED | `oaf_corpus_run.py` -> `prediction_artifact.py` |
| descriptor | EXTEND | `backend_identity.py` two frozen families |
| prediction artifact | EXTEND | v2 header + event backend dispatch; OaF golden unchanged |
| seed MIDI mapping | REUSE + FREEZE | `mapping.DEFAULT_MIDI_NOTE_MAP` |
| MuScriptor pitch map | EXTEND | explicit table above; persist-unmapped semantics from `map_oaf_prediction()` |
| zero-hit map identity | EXTEND | `cohort_scoring.py` uses expected cohort map for empty artifacts |
| scoring/reports | REUSE | HPA-325 |
| reviewed membership | REUSE + small core extraction | HPA-327 |
| corpus execution | NEW concrete module | no generic runner |
| paired deltas | NEW narrow helper | OaF/MuScriptor only |
| CLI tests | EXTEND | `tests/test_cli_benchmark.py` |

## Acceptance

HPA-395 is complete when:

- the exact package/checkpoint/config/licenses are frozen before scores;
- inference always uses the frozen hard drum-only settings;
- the exact MuScriptor v1 pitch table/map ID is frozen before inference and is present in every persisted event identity;
- raw MuScriptor onset + native pitch persist directly in prediction v2;
- unsupported pitches remain visible as unmapped coverage rather than disappearing from the artifact;
- paired OaF/MuScriptor songs have byte-identical Crux canonical full-mix input identity;
- no same-source OaF/MuScriptor row has an input-view or input-audio hash mismatch;
- broad and HPA-327 reviewed reports use unchanged HPA-325 semantics;
- complete per-model success/failure counts remain visible separately from paired intersections;
- narrow paired deltas are reproducible from persisted artifacts without rerunning inference;
- runtime/device/dtype/memory and code/weight-license evidence are recorded;
- no score-informed checkpoint, decoding, mapping, or post-processing change occurred.

## Stop conditions

Do not publish scored results while any of these is unresolved:

- checkpoint/license/revision/hash is not exact;
- medium/small choice is not frozen from pre-score feasibility evidence;
- hard `instruments=["drums"]` is unavailable in pinned v0.3.0;
- `MUSCRIPTOR_PREDICTION_MAP_ID` or the exact v1 pitch table is not committed/frozen;
- shared canonical full-mix bytes differ from the current OaF output for the same source;
- a same-source OaF/MuScriptor row has a different input-view or input-audio hash;
- existing OaF prediction-v2 golden changes;
- HPA-323/HPA-324/HPA-327 lineage does not match;
- implementation starts copying the neutral source resolver or prediction-path logic instead of sharing the now-two-caller seam;
- a proposed abstraction exists only for hypothetical future models rather than two current callers.

## Expected long-lived additions

Keep new production modules to the concrete ticket needs:

```text
src/benchmark/muscriptor_model.py
src/benchmark/backends/muscriptor.py
src/benchmark/muscriptor_corpus_run.py
src/benchmark/muscriptor_comparison.py
```

Existing modules receive narrow extensions/extractions only. If implementation starts creating a generic model registry, experiment object model, shared runner base class, or second report stack, stop and simplify.
