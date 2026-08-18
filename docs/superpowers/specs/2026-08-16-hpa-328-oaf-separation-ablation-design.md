# HPA-328 OaF Drum-Stem Separation Ablation Design

**Status:** Proposed  
**Date:** 2026-08-16  
**Reviewed against main:** 2026-08-17, `d9a124835a020b630db088112fa733b33baee619`  
**Linear:** HPA-328 — Run a cost-controlled drum-stem separation ablation for validated OaF Drums

## Decision summary

Implement HPA-328 as one fixed paired-input experiment around the validated OaF benchmark path:

1. use the **entire canonical HPA-327 reviewed subset** as pilot membership;
2. reuse the persisted HPA-326 full-mix OaF run as the control and never rerun full-mix OaF;
3. run exactly **Spleeter 4-stem drums** and **standard `htdemucs` drums** in v1;
4. keep both separators outside Crux's Python dependency graph and invoke two concrete pinned external commands;
5. retain exact separator-produced WAV bytes by authoritative source SHA-256 plus separator-lock SHA-256;
6. canonicalize retained stems through the existing 44.1 kHz mono PCM16 OaF input boundary while keeping source, stem, and canonical-input identities distinct;
7. reuse the exact OaF inference write path: `backend.transcribe()` → `map_oaf_prediction()` → `publish_prediction_artifact()` with `classify_oaf_backend_error()` for backend failures;
8. reuse HPA-325 scoring and reports without adding a scorer or failure-reason enum;
9. compare full mix against each separator only on their paired-success intersections, while keeping independent population counts visible;
10. publish an immutable HPA-396 handoff containing the exact HTDemucs stem and OaF-on-HTDemucs identities;
11. record the recommendation explicitly as an operator decision rather than implementing an automatic winner formula.

Do **not** build a separator plugin registry, generic experiment runner, RPC/service layer, queue, worker pool, database, third separator, SDR scorer, bootstrap/significance layer, retry/tuning engine, or full-corpus separator mode.

## Latest review resolution

The latest review is accepted in full. It tightens implementation bindings but does not change the experiment.

### 1. Derived-view resume must not reuse the full-mix-only matcher

Current `oaf_corpus_run._prediction_artifact_matches()` rejects any prediction whose `input_view_id` is not `OAF_FULL_MIX_INPUT_VIEW_ID`, then separately compares it with the materialized `CanonicalAudio.input_view_id`. That helper is therefore correct for HPA-326 but wrong for HPA-328 derived views.

HPA-328 must not copy or call that full-mix equality. Resume validation instead has two layers:

1. reuse the existing view-agnostic `_prediction_artifact_matches_run_row()` to bind raw persisted bytes to the prior run-row evidence;
2. validate the artifact against the current materialized `CanonicalAudio`, requiring descriptor/source/input SHA identity and:

```python
prediction.audio.input_view_id == audio.input_view_id
```

The expected view comes from the materialized derived input, never from `OAF_FULL_MIX_INPUT_VIEW_ID`.

### 2. Derived inference uses the existing OaF write path exactly

A prediction path helper is only a location function. HPA-328 must not invent a second publisher or bypass native-event mapping.

For every prediction miss the live path is exactly:

```python
native = backend.transcribe(audio)
mapped, diagnostics = map_oaf_prediction(native)
published = publish_prediction_artifact(prediction_target, mapped)
```

`OafBackendError` is handled through `classify_oaf_backend_error()`. `ArtifactPublicationError` / `PredictionArtifactError` remain prediction-artifact failures. No generic runner extraction is needed.

### 3. Full-mix control uses the existing high-level reviewed-subset scorer

`score_oaf_reviewed_subset()` already owns:

- parsing the persisted HPA-326 run;
- HPA-323/HPA-324 lineage checks;
- parent cohort reconstruction through `build_oaf_cohort_from_snapshot()`;
- exact HPA-327 membership filtering;
- selected-song diagnostics;
- HPA-325 report writing.

HPA-328 therefore calls:

```python
score_oaf_reviewed_subset(
    ScoreReviewedSubsetRequest(
        reference_manifest_path=...,
        timing_manifest_path=...,
        subset_manifest_path=...,
        run_path=oaf_run_path,
        output_dir=run_dir / "views" / "full_mix" / "reports",
    )
)
```

HPA-328 still parses the parent run snapshot separately for run identity and per-row full-mix evidence, but does not restitch `score_reviewed_subset_cohort()` itself.

### 4. HPA-325 failure reasons stay closed

Do not extend `CohortFailureReason`. HPA-328 retains detailed separator/runtime failure codes in its own `run.json`, then collapses them to the existing scorer vocabulary with one frozen table:

```text
separation_failed          -> inference_failed
stem_invalid               -> inference_failed
canonical_input_failed     -> inference_failed
inference_failed           -> inference_failed
prediction_invalid         -> prediction_artifact_invalid
prediction_output_conflict -> prediction_artifact_invalid
prediction_publish_failed  -> prediction_artifact_invalid
```

`success` and `resumed` become HPA-325 success rows. No separator-specific reason reaches HPA-325.

### 5. Headline comparison aggregates use paired-success membership

HPA-325 `summary.json` aggregates are correct independent cohort evidence, but two summary event-micro F1 values cannot simply be subtracted if successful memberships differ.

The shared report reader therefore includes the independent summary aggregates, including per-tolerance/mode event-micro `tp/fp/fn/precision/recall/f1`.

For each HPA-328 comparison:

```text
full_mix vs spleeter
full_mix vs htdemucs
```

headline paired event-micro metrics are recomputed from the **published per-song TP/FP/FN counts** on the exact successful-song intersection. This is aggregation of published counts, not event rescoring.

Independent view populations and independent HPA-325 summary aggregates remain visible separately.

### 6. Separate implementation schema work from real runtime freezing

Unit development does not require installed Spleeter or Demucs. The lock parser/freeze script task uses synthetic fixture locks and tests only.

Actual upstream smoke reproduction, model-file hashing, and real checked-in `runtime/separators/*/model.json` generation happen at the final operational gate, before any benchmark scores are inspected.

The report-reader extraction and HPA-328 comparison are also separate reviewable tasks: first preserve MuScriptor comparison behavior while promoting `read_cohort_reports()`, then implement the HPA-328-specific joins.

## Why HPA-328 is next

HPA-328 is fully unblocked: HPA-325, HPA-326, and HPA-327 are complete. It directly blocks HPA-396 and contributes required evidence to HPA-329. HPA-395 is merged, so implementation starts directly from current `main`.

## Non-goals

HPA-328 does not:

- train or fine-tune a separator or OaF;
- treat separator SDR as the outcome metric;
- add a third separator;
- process the approximately 400-song corpus;
- change OaF checkpoint, threshold, peak picking, taxonomy, prediction map, or scoring tolerances;
- run Inverse Drum Machine;
- add a new scorer or `CohortFailureReason` value;
- add significance/bootstrap analysis;
- build distributed execution;
- add backward compatibility for an experimental artifact schema.

## Existing seams to reuse

Use current `main` directly:

```text
src.benchmark.corpus_cache.resolve_source_audio
src.benchmark.corpus_cache.ResolvedSourceAudio
src.benchmark.input_view.materialize_full_mix_audio
src.benchmark.input_view.load_materialized_audio
src.benchmark.prediction_artifact.prediction_path
src.benchmark.prediction_artifact.read_prediction_artifact
src.benchmark.prediction_artifact.publish_prediction_artifact
src.benchmark.mapping.map_oaf_prediction
src.benchmark.oaf_corpus_run.classify_oaf_backend_error
src.benchmark.oaf_corpus_run._prediction_artifact_matches_run_row
src.benchmark.reviewed_subset.score_oaf_reviewed_subset
src.benchmark.reference_set_manifest.preflight_reference_mappings
src.benchmark.cohort_scoring.score_cohort
src.benchmark.reports.write_cohort_reports
src.benchmark.artifact_io.publish_immutable_file
src.benchmark.corpus_manifest.render_manifest
src.benchmark.corpus_manifest.publish_manifest
```

Two small shared extractions remain justified:

1. the scorer-local prediction-source adaptation and non-success `CohortItem` construction duplicated by OaF and MuScriptor, because HPA-328 becomes a third caller;
2. model-neutral reading of already-published HPA-325 reports, because `muscriptor_comparison.py` currently duplicates schemas owned by `reports.py` and HPA-328 becomes another reader.

Do not generalize corpus runners or pairing policies.

## Fixed pilot membership

Use every row in the supplied canonical `crux.reviewed-reference-subset/v1` manifest. HPA-327 already enforces a reviewed 20–30-song population selected without model scores, so HPA-328 adds no sample, seed, replacement, or score-aware filter.

Require exact lineage equality among:

```text
HPA-327 reviewed subset
HPA-324 reference manifest
HPA-323 timing manifest
HPA-326 persisted OaF run
```

Persist each HPA-327 row's `source_row_sha256` into HPA-328 `run.json` and the final handoff. A subset member missing from HPA-326 is fatal preflight. A persisted full-mix failure remains an explicit control failure; membership is never replaced.

## Full-mix control

Use `score_oaf_reviewed_subset()` to write:

```text
<run>/views/full_mix/reports/
```

The request points to the exact HPA-324 manifest, HPA-323 timing manifest, HPA-327 subset, and HPA-326 `run.json`.

Separately parse the HPA-326 snapshot for:

- parent run ID;
- OaF descriptor/model-lock/checkpoint/config identity;
- full-mix per-row status;
- source/input/prediction hashes required in HPA-328 `run.json` and final HPA-396 handoff.

No full-mix backend is created.

## Separator lock and runtime contract

### Frozen v1 choices

```text
Spleeter: official 4-stem configuration, drums stem
Demucs: standard htdemucs model, drums stem
```

Do not compare model variants, shift counts, overlap values, clip modes, or multiple Spleeter configurations and choose from benchmark scores.

### Canonical lock schema

Each real separator lock eventually uses `crux.separator-lock/v1`:

```text
separator_id
repository_url
repository_revision
package_name
package_version
model_id
model_files[{name, sha256}]
code_license
model_license
argv
expected_drum_stem_relative_path
output_container
```

`argv` contains score-relevant arguments after the interpreter executable. Absolute machine paths do not enter identity. The canonical lock bytes' SHA-256 are the separator config identity.

During unit implementation, use fixture locks under `tests/fixtures/separators/`. Do not require or fake production hashes in `runtime/separators/`.

At the operational gate:

1. reproduce one known-audio Spleeter command;
2. reproduce one known-audio HTDemucs command;
3. capture exact package/repository/model files actually used;
4. generate canonical real lock files through `scripts/freeze_separator_runtime.py`;
5. commit the lock JSON files before running HPA-328 against reference scores.

Use exactly two public adapter functions:

```python
run_spleeter_drums(...)
run_htdemucs_drums(...)
```

Tests inject subprocess runners; CI never installs separators.

For deterministic output discovery, stage the authoritative source under a fixed temporary basename such as `input.wav`.

## Exact stem cache

For a source and lock, derive the cache identity from:

```python
{
    "source_audio_sha256": source_audio_sha256,
    "separator_id": separator_id,
    "separator_config_sha256": lock.sha256,
}
```

Persist exact produced WAV bytes at:

```text
derived/stems/<separator-id>/<source-sha>/<separator-lock-sha>/drums.wav
```

Publish with `publish_immutable_file()`. A cache hit requires matching source/config identity, exact persisted stem SHA, readable regular bytes, and successful QC revalidation. A conflicting file at the same identity fails closed.

Retain the native lossless WAV through HPA-396. The canonical OaF WAV remains temporary.

## Practical fixed stem QC

Validate every generated or resumed stem before OaF:

1. readable regular nonempty file;
2. decodable audio with at least one frame;
3. finite samples only;
4. one or two channels only;
5. record sample rate/channel count; non-44.1 kHz or mono is a warning;
6. fail when duration difference exceeds `max(0.5 sec, source_duration * 0.005)`;
7. fail at RMS `<= -80 dBFS`;
8. record clipping warning and clipped-sample fraction at absolute peak `>= 0.9999`;
9. never inspect DTX references or OaF scores during QC.

These thresholds are frozen constants, not CLI flags.

## OaF derived-input contract

Keep the existing full-mix view:

```text
crux.oaf-full-mix-mono44k1-pcm16/v1
```

Add exactly:

```text
crux.oaf-spleeter4-drums-mono44k1-pcm16/v1
crux.oaf-htdemucs-drums-mono44k1-pcm16/v1
```

For every derived item preserve three separate identities:

```text
authoritative source:
  source_audio_id
  source_audio_sha256

retained separator artifact:
  stem_path
  stem_sha256

canonical OaF input:
  input_view_id
  input_audio_sha256
```

Do not replace authoritative `source_audio_sha256` with the stem hash.

Add `materialize_derived_audio()` beside `materialize_full_mix_audio()`. It reads the retained stem, uses the same `librosa` `soxr_hq` mono 44.1-kHz conversion and `soundfile` PCM16 writer, then calls `load_materialized_audio()` with the authoritative source identity. Factor only the canonicalization body necessary to keep full-mix behavior byte-identical.

Extend OaF's inference config builder only with:

```python
input_view_id: str = OAF_FULL_MIX_INPUT_VIEW_ID
```

All other inference identity fields remain unchanged. `build_run_id()` stays full-mix-specific; HPA-328 owns its own run ID.

## Run snapshot and resume

Use one mutable canonical `crux.oaf-separation-run/v1` snapshot. Run identity binds:

```text
reviewed-subset/reference/timing hashes
parent HPA-326 run ID
OaF descriptor/model-lock/checkpoint identities
Spleeter lock SHA
HTDemucs lock SHA
both derived input-view IDs
SCORING_VERSION
Crux commit
```

Rows are sorted by `simfile_id` and contain:

```text
source_row_sha256
source audio ID/SHA/duration
full-mix parent status/input/prediction identity
Spleeter status + failure code + stem/QC/runtime + canonical input + prediction/runtime
HTDemucs status + failure code + stem/QC/runtime + canonical input + prediction/runtime
```

Closed derived statuses remain:

```text
pending
separation_failed
stem_invalid
inference_failed
prediction_invalid
success
resumed
```

Detailed run failure codes may additionally include `canonical_input_failed`, `prediction_output_conflict`, and `prediction_publish_failed`.

### Resume validation

A matching stem is reused only after exact hash and QC validation.

For a prediction candidate:

1. read with `read_prediction_artifact()`;
2. use `_prediction_artifact_matches_run_row()` when prior persisted prediction evidence exists;
3. require descriptor equality with the frozen OaF descriptor;
4. require source ID/SHA equality with the resolved authoritative source;
5. require `prediction.audio.input_view_id == audio.input_view_id`;
6. require `prediction.audio.input_audio_sha256 == audio.input_audio_sha256`;
7. require all event `prediction_map_version` values equal `OAF_PREDICTION_MAP_ID`.

Never call or copy `_prediction_artifact_matches()`'s `OAF_FULL_MIX_INPUT_VIEW_ID` guard for derived views.

## Derived inference execution

Use one persistent OaF backend for all prediction misses, constructed lazily.

Per successful stem:

```python
native = backend.transcribe(audio)
if not isinstance(native, NativePrediction):
    raise OafBackendError("native prediction is invalid", code="native_event_invalid")

mapped, _ = map_oaf_prediction(native)
published = publish_prediction_artifact(prediction_target, mapped)
```

Backend exceptions go through `classify_oaf_backend_error()`. Publication/artifact errors map to HPA-328 prediction failure codes. No second mapper/publisher or generic inference abstraction is introduced.

Separator failures remain independent: Spleeter failure does not prevent HTDemucs for the same song and vice versa. A poisoned OaF worker stops later inference in that invocation and leaves resumable state; it does not trigger automatic restart/tuning.

## Persisted prediction → scorer bridge

Persisted OaF prediction artifacts keep the authoritative source object key. HPA-325 requires scorer-facing artifacts to use `source_audio_id == simfile_id`.

Promote the already duplicated OaF/MuScriptor in-memory adaptation into one model-neutral helper in `cohort_scoring.py`. The helper assumes the caller already validated raw persisted bytes against its run row, then changes only scorer-local `source_audio_id`, rerenders/reloads the temporary artifact, and calls `cohort_item_from_artifacts()`.

Also promote the duplicated non-success `CohortItem` constructor. Do not weaken `_artifact_identity_from_artifacts()` or mutate persisted bytes.

### Frozen HPA-328 failure mapping

Before calling HPA-325, use exactly:

```python
SEPARATION_FAILURE_TO_COHORT_REASON = {
    "separation_failed": "inference_failed",
    "stem_invalid": "inference_failed",
    "canonical_input_failed": "inference_failed",
    "inference_failed": "inference_failed",
    "prediction_invalid": "prediction_artifact_invalid",
    "prediction_output_conflict": "prediction_artifact_invalid",
    "prediction_publish_failed": "prediction_artifact_invalid",
}
```

No new `CohortFailureReason` is allowed.

## Scoring

Preflight HPA-324/HPA-323 once with `preflight_reference_mappings()` before separator execution.

Publish:

```text
views/full_mix/reports
views/spleeter4_drums/reports
views/htdemucs_drums/reports
```

Full mix is produced through `score_oaf_reviewed_subset()`.

Derived views construct HPA-325 items through the scorer bridge and frozen failure map above, then call only:

```python
score_cohort(...)
write_cohort_reports(...)
```

HPA-325 remains the sole owner of 30/50/100-ms raw/aligned event matching, diagnostics, per-class scoring, cohort aggregates, and report rendering.

## Shared published-report reader

`reports.py` owns the HPA-325 writer schemas. `muscriptor_comparison.py` currently duplicates those schemas and parsers.

Promote one model-neutral `read_cohort_reports()` boundary in a separate implementation task. It reads:

```text
summary.json
items.csv
per_song.csv
per_class.csv
```

It validates exact schemas, canonical numeric domains, report identity, duplicates, and successful-item score membership.

The typed result includes:

- `CohortIdentity`;
- independent population evidence;
- per-tolerance/mode independent summary aggregates, including event-micro TP/FP/FN/precision/recall/F1;
- item rows;
- per-song rows including TP/FP/FN and timing diagnostics;
- per-class rows.

Retarget MuScriptor comparison to this reader with characterization tests requiring unchanged published outputs. Backend-family checks, run parsing, source/input pairability, and MuScriptor-specific labels stay local to `muscriptor_comparison.py`.

## HPA-328 comparison semantics

HPA-328 owns only two fixed comparisons:

```text
full_mix vs spleeter
full_mix vs htdemucs
```

For each:

1. report independent full-view populations and independent HPA-325 summary aggregates;
2. derive the paired-success song set from both `items.csv` populations;
3. require matching score key grids on that paired set;
4. emit per-song and per-class precision/recall/F1 deltas;
5. recompute headline event-micro TP/FP/FN on that paired set by summing the **published per-song counts** for each tolerance/mode;
6. derive paired precision/recall/F1 from those summed counts and report `derived - full_mix` headline deltas.

Do not subtract independent cohort event-micro F1 values when successful memberships differ.

This aggregation does not reopen prediction artifacts or run event matching.

### FP/FN per minute

For every paired per-song row:

```python
minutes = source_duration_sec / 60
fp_per_min = fp / minutes
fn_per_min = fn / minutes
```

`source_duration_sec` comes from authoritative HPA-328 run-row evidence and must be positive and finite. Emit absolute values and derived-minus-full-mix deltas using six-place canonical numeric conventions.

Also carry HPA-325 timing diagnostics, bounded most-helped/most-harmed cases, separator/OaF wall time and RTF, retained stem/prediction/report bytes, and optional explicit compute/storage cost rates.

Do not add significance tests or automatic winner ranking.

Comparison output remains:

```text
comparison/summary.json
comparison/per_song_delta.csv
comparison/per_class_delta.csv
comparison/summary.md
```

## Recommendation/finalization

After frozen reports are reviewed, record exactly one operator decision:

```text
keep_full_mix
use_spleeter
use_htdemucs
gather_more_evidence
prioritize_another_model
```

Store a nonempty rationale and optional nonnegative compute/storage cost rates. Rates affect display/evidence only.

## Immutable HPA-396 handoff

Publish canonical `crux.oaf-separation-pilot/v1` JSONL through existing manifest rails.

Every HPA-327 song is present, including failed views. Each row includes:

```text
simfile_id
source_row_sha256
reviewed subset SHA
reference manifest SHA/version
reference timing SHA/version
taxonomy/lane/prediction-map/scoring versions
source audio ID/SHA
OaF descriptor/model/model-lock/checkpoint/adapter identities
full-mix status/input/prediction path+SHA
Spleeter status/lock/stem/input/prediction identities
HTDemucs status/lock/stem/input/prediction identities
comparison artifact hashes
decision/rationale/cost rates
Crux commit
```

Success/resume requires all relevant artifact identities. Failed views require unavailable fields to be null. Finalization re-reads and hashes retained stems, predictions, and comparison files before publication.

The handoff exposes exact HTDemucs stem and OaF prediction identities to HPA-396 without requiring mutable HPA-328 state or an input-view manifest. HPA-396 may rematerialize from the retained native stem through `materialize_derived_audio()`.

## CLI

Add only:

```text
crux benchmark run-oaf-separation-pilot
crux benchmark finalize-oaf-separation-pilot
```

Run inputs:

```text
--manifest
--timing-manifest
--subset-manifest
--oaf-run
--cache-dir
--output-dir
--spleeter-python
--demucs-python
--resume
```

Finalize inputs:

```text
--run
--subset-manifest
--output-manifest
--decision
--rationale
--compute-cost-per-hour
--storage-cost-per-gb-month
```

No model/QC/tolerance/third-separator/full-corpus flags. Run exit codes follow the existing 0 complete / 1 partial / 2 fatal convention.

## Testing and implementation gates

Use TDD around existing seams:

- characterize OaF/MuScriptor scorer adaptation before extraction;
- characterize full-mix OaF inference/config bytes before derived-input work;
- unit-test separator lock schema/freeze script with fixture locks only;
- unit-test cache/QC/process behavior with injected subprocess runners;
- acceptance-test that full-mix reports come from `score_oaf_reviewed_subset()` and no full-mix backend call occurs;
- test derived resume with a Spleeter/HTDemucs input view and prove the full-mix-only `_prediction_artifact_matches()` is not used;
- test derived inference calls the existing map/publish/error-classification path;
- test the exact HPA-328 failure-code → HPA-325 reason table;
- characterize MuScriptor comparison before extracting `read_cohort_reports()`;
- test shared reader preserves independent summary aggregates;
- test paired headline event-micro metrics are summed from paired-success per-song counts;
- test HPA-396 handoff consumes exact HTDemucs stem/prediction identities;
- keep real separator smoke/model acquisition outside CI.

### Final operational gate

Before any separator benchmark scores are inspected:

1. reproduce real Spleeter and HTDemucs known-audio smoke runs;
2. freeze exact real model/package/repository evidence;
3. generate and commit `runtime/separators/spleeter/model.json` and `runtime/separators/htdemucs/model.json`;
4. record the HPA-328 implementation commit and both lock hashes;
5. run the exact HPA-327 pilot;
6. review required evidence and finalize one closed decision;
7. run repository-wide pytest, Ruff, format, Pylint errors-only, and `git diff --check`.

## Risks and gates

- **Dependency conflict:** isolated interpreter commands only.
- **Resume identity bug:** derived view is matched to materialized `CanonicalAudio.input_view_id`; never full-mix constant.
- **Write-path drift:** derived predictions use existing OaF transcribe/map/publish/classify functions.
- **Source/scorer identity mismatch:** persisted bytes validated first, then scorer-local source-ID adaptation.
- **Failure-vocabulary drift:** one frozen HPA-328 → HPA-325 reason table; no enum extension.
- **Comparison membership bug:** independent summaries are not used as paired headlines; paired TP/FP/FN are summed from the success intersection.
- **Comparison-schema drift:** one reader beside the HPA-325 writer; no third field-schema copy.
- **Selection bias:** exact HPA-327 membership only.
- **Artifact drift:** source/config/stem/input/prediction hashes checked on resume/finalize.
- **Runaway compute:** no full-corpus path or third separator.
- **Over-abstraction:** shared extractions stop at real duplicated seams; runners and comparison policies remain concrete.

## Acceptance mapping

- Same fixed songs and OaF configuration → HPA-327 membership + frozen OaF identity.
- Full mix/Spleeter/HTDemucs results → existing full-mix subset scorer + two derived HPA-325 cohorts.
- Separation cache and explicit failures → content-addressed stems + per-view run ledger.
- Accuracy/runtime/storage/cost → independent and paired-success comparison evidence.
- FP/FN per minute → published per-song counts normalized by authoritative source duration.
- HTDemucs handoff → immutable rows with exact native stem and OaF prediction identities.
- No training/full-corpus/framework expansion → explicit non-goals and narrow CLI.
