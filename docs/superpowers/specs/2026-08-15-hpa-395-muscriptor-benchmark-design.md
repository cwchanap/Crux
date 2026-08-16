# HPA-395 MuScriptor Full-Mix Benchmark Design

**Status:** Proposed  
**Date:** 2026-08-15  
**Linear:** HPA-395 — Integrate and benchmark frozen MuScriptor drum-only inference against OaF

## Decision summary

Implement HPA-395 as a comparator-only MuScriptor path on the existing HPA-324/HPA-325/HPA-326 benchmark rails:

1. freeze the official MuScriptor `v0.3.0` release and one gated Hugging Face checkpoint before any scored run;
2. add a small `MuscriptorBackend` that implements the existing `TranscriptionBackend` protocol, loads one model per run, calls `transcribe(..., instruments=["drums"])`, and persists only raw `NoteStartEvent` onsets;
3. reuse the exact HPA-326 44.1 kHz mono PCM16 full-mix materialization so OaF and MuScriptor are bound to identical `input_audio_sha256` values before their model-native preprocessing diverges;
4. extend the existing `crux.drum-prediction-events/v2` artifact and HPA-325 cohort scorer to accept a second frozen backend without changing the OaF v2 bytes or scorer semantics;
5. add one MuScriptor-specific corpus runner, broad + reviewed-subset scoring, and a narrow OaF-vs-MuScriptor paired-delta report.

Do **not** add a generic experiment runner, plugin framework, model registry, production backend selector, asynchronous job system, database, second scorer, MIDI round-trip, beat-grid timing correction, threshold tuning, or fine-tuning machinery.

## Why HPA-395 is next

HPA-395 is the highest-priority unstarted Crux issue whose dependency chain is now complete:

- HPA-325 cohort scoring is complete;
- HPA-326 broad OaF corpus inference is complete;
- HPA-327 fixed reviewed reference subset is complete;
- HPA-395 is High priority and directly blocks HPA-329 final findings.

HPA-328 is also unblocked, but it is Medium priority. HPA-396 still depends on HPA-328. HPA-562 becomes more useful after at least one comparison population exists. HPA-395 therefore gives the next useful model-quality signal with the least dependency risk.

## Goals

HPA-395 must provide:

- one frozen, reproducible MuScriptor model identity;
- a hard drum-only decoding constraint with no model tuning after scores are inspected;
- native event provenance that preserves the original MuScriptor drum MIDI pitch;
- the same authoritative source/reference identities and exact pre-model full-mix WAV bytes as OaF;
- the existing HPA-325 30/50/100 ms raw/aligned scoring outputs for the broad corpus;
- the same scoring outputs for the HPA-327 reviewed subset with event diagnostics retained;
- visible unmapped/unsupported native drum pitches rather than silent dropping;
- resumable immutable prediction artifacts and run snapshots;
- a narrow paired OaF-vs-MuScriptor delta output for matching successful songs;
- wall time, RTF, device/dtype, and practical peak-memory evidence for the MuScriptor run;
- a clear code-license versus model-weight-license record.

## Non-goals

HPA-395 does not:

- replace OaF as the Crux baseline;
- expose MuScriptor through `default_backend_registry()` or production API selection;
- fine-tune or calibrate MuScriptor;
- search checkpoints, temperatures, beams, prompts, or mapping variants against benchmark scores;
- use MuScriptor's MIDI writer as the benchmark boundary;
- use beat/downbeat correction or `transcribe_to_midi()` timing adjustment;
- separate stems;
- implement the generalized multi-model comparison publication owned by HPA-562;
- change the reference corpus, reviewed-subset membership, taxonomy, tolerance windows, or HPA-325 matching logic.

## Planning-time upstream freeze

The implementation must start from the official MuScriptor release, not moving `main`:

```text
package: muscriptor==0.3.0
release tag: v0.3.0
release commit: d73147e75e5b9b0c0a79ebe154587db4fd603e0c
code license: MIT
preferred checkpoint size: medium
```

At `v0.3.0`, the public API provides:

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

The stream contains `NoteStartEvent`, `NoteEndEvent`, and `ProgressEvent`. A `NoteStartEvent` contains `pitch`, `start_time`, `index`, and `instrument`. Drum hits are represented as `instrument == "drums"` with a native MIDI pitch. The upstream model splits audio into five-second chunks.

The planning-time upstream documentation identifies the model weights as CC BY-NC 4.0. Implementation preflight must verify and persist the exact code and weight license strings before the scored run. License disagreement is a fatal preflight error, not a warning.

## Checkpoint freeze and model lock

### Why the built-in default download is not sufficient

MuScriptor 0.3.0 resolves `hf://MuScriptor/muscriptor-medium/model.safetensors` with `hf_hub_download()` without an explicit revision. That is convenient interactively but not enough for a benchmark whose identity must remain stable.

The scored runner must never resolve a mutable Hugging Face default branch.

### Frozen acquisition flow

Add a small preflight/acquisition path under `runtime/muscriptor/` that:

1. authenticates through the normal Hugging Face client/environment;
2. resolves `MuScriptor/muscriptor-medium` to an exact 40-character commit SHA;
3. downloads `model.safetensors` and companion `config.json` at that exact revision;
4. computes the local SHA-256 and byte size for both files;
5. writes the checked-in model lock only after those values are known;
6. loads MuScriptor by **local safetensors path**, never by `"medium"` or an unqualified `hf://` URL during a scored run.

Use `runtime/muscriptor/model.json`, mirroring the existing `runtime/oaf_tf1/model.json` rail rather than introducing another configuration subsystem.

The MuScriptor lock is a strict `crux.muscriptor-model/v1` JSON object containing at least:

```text
schema
backend_id
model_id
architecture_id
package_name
package_version
upstream_source_commit
code_license
weight_license
checkpoint.repo_id
checkpoint.revision
checkpoint.filename
checkpoint.sha256
checkpoint.byte_length
checkpoint.config_filename
checkpoint.config_sha256
checkpoint.config_byte_length
device
dtype
input_sample_rate_hz          # MuScriptor internal target, 16000
chunk_duration_sec            # 5.0 for v0.3.0
use_sampling                  # false
temperature                   # 1.0, persisted even though sampling is disabled
cfg_coef                      # 1.0
instruments                   # exactly ["drums"]
batch_size                    # 1
no_eos_is_ok                  # true
beam_size                     # 1
prelude_forcing               # true
native_output_space_id
native_metadata_schema_id
training_data_map_id
```

No scored CLI flag overrides these values in v1. A later intentional configuration change means a new model-lock hash and therefore a different run identity.

### Medium-first feasibility rule

Medium is the fixed default candidate. Before any benchmark score is published, run the predeclared smoke population and record runtime/memory evidence.

Only if medium is technically impractical on the target benchmark hardware may implementation switch once to the small checkpoint. That decision must happen before broad or reviewed-subset scores are inspected and must produce a newly frozen model lock. Do not compare medium versus small on benchmark F1 to choose the winner.

## Dependency isolation

Add MuScriptor as an optional benchmark dependency rather than a root runtime requirement:

```toml
[project.optional-dependencies]
muscriptor = ["muscriptor==0.3.0"]
```

The backend module imports MuScriptor lazily. Normal Crux tests and the production API must not require PyTorch or gated weights.

CI tests use injected fake model/event objects. CI must not authenticate to Hugging Face, download the real checkpoint, or perform GPU inference.

If `uv lock` proves that the optional package cannot coexist with the current Python 3.12 environment, stop and use a tiny dedicated runtime only then. Do not preemptively add a subprocess protocol; the direct optional dependency is the simplest path and preserves the required native event stream.

## Existing seams to reuse

### Backend-neutral protocol

Reuse `src/benchmark/backends/base.py` unchanged where possible:

```python
CanonicalAudio
NativeEvent
NativePrediction
TranscriptionBackend
```

`MuscriptorBackend` belongs at `src/benchmark/backends/muscriptor.py`. It is a benchmark comparator implementation, but it is **not** registered in `default_backend_registry()`.

### Exact full-mix input view

HPA-326 currently materializes every authoritative source to:

```text
44,100 Hz
mono
WAV
PCM_16
librosa res_type="soxr_hq"
```

and `prediction_artifact.py` already defines that as the canonical persisted input shape.

Promote only the model-neutral part of `_materialize_oaf_full_mix()` into `src/benchmark/input_view.py`:

```python
def materialize_full_mix_audio(
    source_path_or_bytes: Path | bytes,
    output_path: Path,
    *,
    source_audio_id: str,
    source_audio_sha256: str,
    max_input_audio_frames: int | None,
) -> CanonicalAudio:
    ...
```

Keep the existing input-view identifier and bytes stable. OaF calls the promoted helper with its existing maximum-frame guard. MuScriptor calls it with no OaF-specific frame limit.

This is intentionally a characterization refactor: the OaF `input_audio_sha256` for the same source must remain byte-identical before and after extraction.

MuScriptor receives that canonical WAV path. Its own documented 16 kHz preprocessing remains inside MuScriptor and is captured in the model lock/inference config. This gives the cross-model comparison the strongest possible input identity without rewriting the completed OaF baseline.

### Reference and scoring rails

Reuse unchanged:

- `load_reference_set_manifest()`;
- `preflight_reference_mappings()`;
- `load_reference_timing_manifest()`;
- `score_cohort()`;
- `write_cohort_reports()`;
- `load_reviewed_subset_manifest()`;
- the HPA-327 reviewed-subset lineage checks.

Do not parse DTX again and do not create a MuScriptor-specific scorer.

## Backend adapter

### Construction

`MuscriptorBackend` takes a parsed frozen model lock plus resolved local checkpoint path. Construction:

1. validates the lock;
2. validates installed package version `0.3.0`;
3. validates the local checkpoint/config hashes against the lock;
4. imports `TranscriptionModel` and event classes lazily;
5. loads exactly one `TranscriptionModel` with the frozen device/dtype;
6. builds the frozen backend descriptor.

If the package, checkpoint, requested device, or hard drum-only API is unavailable, fail before processing the first song.

### Transcription

For each `CanonicalAudio`, call the frozen API parameters from the model lock. Consume the iterator once.

Persist a `NativeEvent` only for `NoteStartEvent`:

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

- `event.instrument` must be exactly `"drums"`; anything else is a backend-contract failure because hard drum-only decoding was requested;
- pitch must be an integer in MIDI range 0..127;
- onset must be finite and nonnegative;
- `NoteEndEvent` is consumed/validated for stream integrity but does not create a second hit;
- `ProgressEvent` is ignored for scoring;
- no synthetic velocity or confidence is invented;
- no MIDI file is written and re-read;
- no beat/downbeat correction is applied.

`close()` releases the model reference and performs best-effort accelerator cache cleanup only if upstream/PyTorch exposes a safe no-op-compatible call. Do not add a worker process merely for teardown.

## Descriptor and prediction artifact

### Descriptor

Keep `crux.transcription-backend-descriptor/v2`. Extend `normalize_known_backend_descriptor()` from one hard-coded OaF identity to a two-branch frozen identity validator:

```text
OaF -> existing identity rules unchanged
MuScriptor -> frozen v0.3.0/model-lock identity rules
```

Do not create a descriptor v3; the current fields already express backend/model/source/output identity.

Suggested MuScriptor identities:

```text
backend_id: muscriptor-v0.3.0-drums-v1
architecture_id: muscriptor-transformer-v0.3.0
native_output_space_id: muscriptor-drums-midi128-v1
native_metadata_schema_id: muscriptor-note-start-metadata-v1
training_data_map_id: muscriptor-published-training-mix-v0.3.0
prediction_schema: crux.drum-prediction-events/v2
```

The exact `model_id` binds the chosen medium or allowed small checkpoint and its frozen checkpoint revision/hash through the model lock/inference config.

### Keep prediction schema v2

The current prediction v2 record already has all fields HPA-395 needs. Extend backend-specific validation instead of adding columns or a new schema version.

OaF retains its current invariants:

- non-null output bin, MIDI note, confidence, and velocity;
- OaF MIDI/bin identity rules;
- `upstream_8hit_group_id` metadata.

MuScriptor uses:

- `model_output_bin = null`;
- `native_midi_note = required`;
- `confidence = null`;
- `velocity_midi = null`;
- exact `instrument_group = "drums"` metadata;
- `native_class_id == f"drums:midi_{native_midi_note}"`.

Dispatch native-event validation from the validated backend descriptor/metadata schema. Do not weaken OaF validation globally.

Add a MuScriptor schema golden and retain the existing OaF golden so a single loader proves both backend branches round-trip through `read_prediction_artifact()`.

## Taxonomy mapping

Add `MUSCRIPTOR_PREDICTION_MAP_ID = "crux.prediction-map/muscriptor-drums-v1"` and an explicit pitch mapping in `taxonomy.py`/`mapping.py`.

The mapping must preserve each original pitch even when multiple pitches collapse to one common class. The event artifact keeps `native_midi_note`; only the scoring projection collapses to the existing common taxonomy.

Start with the General MIDI drum families that map unambiguously to the existing six common classes. Keep the table explicit and reviewed in code. Examples include kick, snare, hihat, crash, ride, and tom families. Do not infer mappings from pitch names at runtime.

A native drum pitch not present in the frozen table is persisted as:

```text
mapping_status = "unmapped"
canonical_class = null
common_class = null
```

and remains visible through HPA-325 native coverage counts. Never drop it before persistence.

Add `map_muscriptor_prediction()` beside `map_oaf_prediction()`; do not create a generic mapping DSL.

## HPA-325 compatibility closures

Two narrow scorer changes are required for a second backend:

1. `prediction_artifact.py` must validate MuScriptor event nullability/identity as described above;
2. `cohort_scoring._artifact_identity_from_artifacts()` must derive the prediction-map identity for an empty artifact from the cohort/descriptor branch rather than assuming only OaF can produce an empty prediction.

Everything after `PredictionArtifact` stays model-neutral. `prediction_to_benchmark_events()` already scores mapped common-class events without reading OaF-specific metadata.

Add characterization tests proving all existing OaF HPA-325/HPA-326 artifacts still validate and score identically.

## MuScriptor corpus runner

Add `src/benchmark/muscriptor_corpus_run.py`. Keep it separate from the large OaF runner instead of turning both into a generic framework in this ticket.

Reuse or narrowly promote only these OaF-owned operations that are genuinely backend-neutral:

- verified source-audio cache resolution;
- canonical full-mix materialization;
- immutable prediction path calculation if its path scheme is already backend-descriptor/config keyed;
- snapshot normalization/durable write helpers only if extraction is smaller than duplication.

Do not extract worker lifecycle, OaF error policy, TF1 configuration, or OaF run schema.

### Run identity

Use:

```text
schema: crux.muscriptor-corpus-run/v1
config schema: crux.muscriptor-inference-config/v1
```

The run identity binds:

- HPA-324 reference manifest SHA/version;
- HPA-323 timing manifest SHA/version;
- frozen model-lock SHA;
- backend descriptor SHA;
- exact checkpoint SHA/revision;
- inference-config SHA;
- canonical full-mix input-view ID;
- adapter revision;
- include/exclude scope.

The inference config records the exact frozen decoding parameters and MuScriptor preprocessing/chunk semantics. No value comes from a scoring result.

### Execution

Run one persistent model sequentially over the selected population. The flow mirrors HPA-326 at the semantic level:

1. validate manifests, scope, model lock, checkpoint, and config before model construction;
2. resolve and re-pin each verified source body;
3. materialize the exact shared full-mix WAV;
4. validate resumable immutable prediction evidence;
5. lazily construct MuScriptor on the first item that actually requires inference;
6. transcribe, map, persist, checkpoint run state;
7. continue item-locally on decode/inference/prediction publication failures;
8. stop the run on model/descriptor/config integrity failures;
9. finalize the broad cohort through HPA-325.

Use the existing HPA-325 failure families. Do not add backend-specific score statuses unless a current category cannot represent the failure.

## Smoke gate

Before the broad corpus, commit one small `runtime/muscriptor/smoke.json` describing a fixed pre-score smoke set with named roles:

```text
short
long
dense
sparse
non_drum_heavy
```

IDs are chosen from the already-fixed reference population using only reference/audit/source listening information, never model scores. Each role records the simfile ID and a short pre-model reason.

Run the same frozen backend/adapter over that set and verify:

- package/checkpoint/device readiness;
- hard drum-only output;
- native pitch/onset validity;
- repeatability on at least one repeated song;
- five-second chunk boundary behavior on the long song;
- resume identity/hash behavior;
- practical wall time and peak memory.

The smoke gate may decide medium is technically impractical. It must not tune decoding based on F1.

## Reviewed-subset scoring

Keep HPA-327 membership immutable.

Extract one small model-neutral helper from `reviewed_subset.py` that takes a parent `CohortIdentity` + `CohortItem` population and the loaded reviewed-subset manifest, validates membership/lineage, filters exact IDs, derives the subset cohort ID, and invokes HPA-325 with diagnostics for successful selected rows.

Then:

- existing `score-oaf-reviewed-subset` calls the helper after `build_oaf_cohort_from_snapshot()`;
- new `score-muscriptor-reviewed-subset` calls the same helper after `build_muscriptor_cohort_from_snapshot()`.

This removes duplicated selection/scoring policy without inventing a backend framework.

## Narrow paired OaF comparison

HPA-395 needs immediate OaF deltas; HPA-562 later owns the generalized multi-model publication.

Add a narrow `compare-oaf-muscriptor` command/helper that reads the two persisted run populations, rebuilds HPA-325 results, and joins only songs that satisfy all HPA-395 pairing requirements:

```text
same simfile_id
same HPA-324 reference manifest identity
same HPA-323 timing identity
same source_audio_sha256
same canonical input_view_id
same input_audio_sha256
both status == success
```

For each 30/50/100 ms tolerance and raw/aligned mode, emit per-song:

```text
simfile_id
oaf_f1
muscriptor_f1
delta_f1
```

Also emit aggregate paired counts and mean/median delta. Use the same helper for the reviewed subset by passing the subset membership filter.

Do not add ranking, significance tests, bootstrap confidence intervals, win-rate policy, or generic model labels in HPA-395. HPA-562 can generalize after HPA-328/HPA-396 create more populations.

## Runtime evidence

Persist in the MuScriptor run snapshot/report metadata:

- device;
- dtype;
- checkpoint byte size;
- total measured wall time;
- measured audio duration;
- aggregate RTF;
- projected full wall time when duration coverage is complete;
- peak process RSS where portable/cheap to measure;
- CUDA/MPS-specific peak accelerator memory only when the selected device exposes it safely.

Peak memory is evidence, not a CI gate. Do not add a cross-platform telemetry abstraction.

## CLI surface

Add only the benchmark commands needed to reproduce the ticket:

```text
crux benchmark run-muscriptor-corpus
crux benchmark score-muscriptor-reviewed-subset
crux benchmark compare-oaf-muscriptor
```

`run-muscriptor-corpus` mirrors the stable HPA-326 arguments:

```text
--manifest
--timing-manifest
--cache-dir
--output-dir
--include-simfile-id (repeatable)
--exclude-simfile-id (repeatable)
--resume
```

It does **not** expose model size, temperature, beam size, device, dtype, instrument list, chunk size, or mapping version. Those are frozen in the model lock/config.

The paired command takes explicit persisted OaF and MuScriptor run paths plus the same reference/timing manifests and optional reviewed-subset manifest. It never reruns inference.

## Failure policy

### Fatal before inference

Fail the whole run before model construction when:

- model lock is malformed or not frozen;
- installed MuScriptor version differs from the lock;
- code/weight license metadata is missing or inconsistent;
- checkpoint/config file is missing or hash/size mismatches;
- requested locked device/dtype is unavailable;
- reference/timing lineage fails;
- include/exclude scope is invalid;
- the backend descriptor or inference config cannot reproduce its frozen hash.

### Item-local

Mark one song failed and continue for:

- source decode/materialization failure;
- MuScriptor inference exception that does not invalidate the loaded model identity;
- invalid native event values for that song;
- immutable prediction publication failure.

If the model becomes unusable after an item failure, stop rather than pretending later items are independent successes.

## Testing strategy

The implementation uses TDD and fake MuScriptor objects for deterministic tests.

Required coverage includes:

- strict model-lock loader and exact checkpoint hash/revision handling;
- missing optional dependency and unavailable-device failure;
- exact frozen `transcribe()` arguments;
- `NoteStartEvent` -> `NativeEvent` conversion;
- `NoteEndEvent`/`ProgressEvent` handling;
- rejection of non-drum output under hard drum mode;
- pitch range and finite-onset validation;
- MuScriptor mapping including unmapped pitches;
- prediction-v2 round trip for both OaF and MuScriptor;
- unchanged OaF prediction golden/validation behavior;
- exact shared full-mix bytes before/after helper extraction;
- run-id/config determinism;
- resume accepts exact artifacts and rejects edited/mixed identity;
- HPA-325 broad scoring with empty and non-empty MuScriptor predictions;
- reviewed-subset lineage/filtering/diagnostics for both backends;
- paired join rejects source/input/reference mismatches;
- CLI exit codes and canonical JSON summaries.

The real gated checkpoint is exercised only in the operator smoke/broad benchmark, not in unit CI.

## Reuse survey

| Need | Decision | Existing seam |
| --- | --- | --- |
| model protocol | REUSE | `src/benchmark/backends/base.py` |
| production backend registry | DO NOT EXTEND | `src/benchmark/backend_registry.py` remains OaF default |
| source cache verification | PROMOTE/REUSE narrowly | current HPA-326 source resolver |
| canonical full-mix bytes | PROMOTE | `_materialize_oaf_full_mix()` -> `input_view.py` helper |
| descriptor | EXTEND | `backend_identity.py` two frozen identities |
| persisted events | EXTEND | `crux.drum-prediction-events/v2`, backend-specific invariants |
| taxonomy mapping | EXTEND | `taxonomy.py` + `mapping.py` explicit MuScriptor pitch map |
| scoring | REUSE | `score_cohort()` unchanged mathematically |
| reports | REUSE | `write_cohort_reports()` |
| reviewed subset | REUSE + small core extraction | `reviewed_subset.py` |
| corpus execution | NEW, model-specific | `muscriptor_corpus_run.py`, no generic runner |
| paired deltas | NEW, narrow | OaF/MuScriptor only; HPA-562 generalizes later |

## Acceptance mapping

HPA-395 is complete when:

- one exact MuScriptor release/checkpoint/config is frozen before scores;
- scored inference always uses `instruments=["drums"]` and frozen deterministic settings;
- raw `NoteStartEvent.start_time` values and native drum pitches persist directly into prediction v2;
- OaF and MuScriptor consume byte-identical Crux canonical full-mix inputs for paired songs;
- broad and reviewed-subset reports come from the unchanged HPA-325 scoring semantics at 30/50/100 ms;
- unmapped MuScriptor drum pitches are visible in coverage;
- failed songs remain explicit and do not disappear from the population;
- paired OaF deltas include only exact identity matches;
- runtime/device/memory and code/weight license evidence are reported;
- existing OaF prediction/scoring tests remain green;
- no model-selection or tuning decision used benchmark outcomes.

## Follow-on boundary

After HPA-395:

- HPA-329 can consume MuScriptor broad/reviewed outputs for final findings;
- HPA-562 can generalize the narrow paired-delta report once additional comparison populations exist;
- HPA-328 remains the independent drum-stem ablation path;
- HPA-396 remains downstream of HPA-328.

Do not pull those tickets into this implementation.