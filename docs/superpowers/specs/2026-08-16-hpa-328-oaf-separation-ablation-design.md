# HPA-328 OaF Drum-Stem Separation Ablation Design

**Status:** Proposed  
**Date:** 2026-08-16  
**Reviewed against main:** 2026-08-17, `d9a124835a020b630db088112fa733b33baee619`  
**Linear:** HPA-328 — Run a cost-controlled drum-stem separation ablation for validated OaF Drums

## Decision summary

Implement HPA-328 as one fixed paired-input experiment around the validated OaF benchmark path:

1. use the **entire canonical HPA-327 reviewed subset** as pilot membership;
2. reuse persisted HPA-326 full-mix OaF predictions as the control instead of rerunning them;
3. run exactly **Spleeter 4-stem drums** and **standard `htdemucs` drums** in v1;
4. keep both separators outside Crux's Python environment and invoke two concrete pinned external commands;
5. cache exact separator-produced drum WAVs by authoritative source SHA-256 + separator-lock SHA-256;
6. canonicalize retained stems through the existing 44.1 kHz mono PCM16 OaF boundary while preserving authoritative source identity separately from exact stem/input hashes;
7. reuse normal OaF prediction-v2 artifacts and HPA-325 scoring/reports;
8. reuse the scorer-local persisted-artifact bridge and a shared HPA-325 report reader rather than creating third copies of behavior that now exists in both OaF and MuScriptor paths;
9. publish narrow full-mix↔separator deltas plus an immutable JSONL handoff containing exact HTDemucs stems and OaF-on-HTDemucs identities for HPA-396;
10. record the final recommendation explicitly rather than implementing an automatic winner formula.

Do **not** build a generic experiment runner, separator plugin registry, RPC/service layer, queue, database, third separator, SDR evaluator, model tuning path, or full-corpus separator mode.

## Latest-main review

HPA-395 PR #24 is now merged. The implementation is no longer waiting for it. The exact neutral seams planned for reuse exist on `main`:

```text
src.benchmark.corpus_cache.resolve_source_audio
src.benchmark.corpus_cache.ResolvedSourceAudio
src.benchmark.input_view.materialize_full_mix_audio
src.benchmark.prediction_artifact.prediction_path
src.benchmark.reviewed_subset.score_reviewed_subset_cohort
```

The review also exposed two integration details that the original HPA-328 plan must close.

### Scorer-local source ID is a real contract

Persisted OaF and MuScriptor prediction artifacts retain the authoritative source object key in `CanonicalAudio.source_audio_id`. HPA-325's `cohort_item_from_artifacts()` intentionally requires the scorer-facing artifact to expose `source_audio_id == simfile_id`.

Both landed corpus runners therefore currently do the same in-memory adaptation after validating the persisted bytes:

```python
scorer_audio = replace(prediction.prediction.audio, source_audio_id=simfile_id)
scorer_prediction = replace(prediction.prediction, audio=scorer_audio)
scorer_artifact = read_prediction_artifact(render_prediction_artifact(scorer_prediction))
```

HPA-328 must **not** call `cohort_item_from_artifacts()` directly on its persisted derived-view artifacts. Since HPA-328 would be the third caller, promote only this scorer-local adaptation plus the model-neutral non-success item constructor into `cohort_scoring.py`, then retarget OaF and MuScriptor to those helpers. This is a narrow scoring bridge, not a generic runner abstraction.

### Published HPA-325 report parsing now has a second consumer

`muscriptor_comparison.py` landed robust CSV/report identity validation for `items.csv`, `per_song.csv`, and `per_class.csv`, but it currently duplicates the HPA-325 field schemas from `reports.py` and keeps the parsers private.

HPA-328 should not create a third copy. When implementing the separation comparison, promote only the model-neutral **report-directory reader** into `reports.py` (or a focused sibling if import direction requires it), preserving the existing MuScriptor comparison output byte-for-byte. Model-pairing rules remain in `muscriptor_comparison.py`; HPA-328 keeps its own fixed three-view comparison logic.

These are the only new shared extractions justified by current `main`.

## Why HPA-328 is next

HPA-328 is fully unblocked: HPA-325, HPA-326, and HPA-327 are complete. It directly blocks HPA-396 and contributes required evidence to HPA-329. HPA-395 has now landed, so HPA-328 implementation can start directly from current `main` without parallel-branch coordination.

## Approaches considered

### Import Spleeter and Demucs into Crux

Rejected. Spleeter is TensorFlow-heavy and Demucs is PyTorch-based. Adding both to the Python 3.12 application environment increases resolver, CI, and install cost for a 20–30-song experiment.

### Build a generic separator/worker framework

Rejected. Two required commands do not justify another backend registry, RPC protocol, worker service, or experiment DSL.

### Two frozen external commands behind a small HPA-328 seam

Selected. Crux validates two canonical lock files, invokes each isolated interpreter, validates/publishes the expected drum stem, and records runtime/failure evidence. Tests inject subprocess runners so real separator packages never become CI dependencies.

## Goals

HPA-328 must produce:

- score-blind fixed pilot membership;
- exact reference/source/full-mix/Spleeter/HTDemucs lineage per song;
- reproducible separator package/model/command evidence;
- content-addressed retained stems;
- practical stem QC and explicit per-view failures;
- unchanged OaF inference/scoring over the two derived views;
- HPA-325 reports for full mix, Spleeter drums, and HTDemucs drums;
- paired per-song/per-class deltas at 30/50/100 ms;
- FP/FN per minute, timing/alignment diagnostics, runtime, storage, and estimated cost evidence;
- an explicit recommended input path;
- an immutable handoff that lets HPA-396 reuse exact HTDemucs bytes without regenerating or resampling the pilot population.

## Non-goals

HPA-328 does not train/fine-tune anything, evaluate separator SDR as the primary objective, add a third separator, process the approximately 400-song corpus, change OaF checkpoint/thresholds/mapping/scoring tolerances, run IDM, add significance/bootstrap machinery, build distributed execution, or optimize backward compatibility for an experimental schema.

## Existing seams to reuse

Use current `main` directly:

```text
src.benchmark.corpus_cache.resolve_source_audio
src.benchmark.corpus_cache.ResolvedSourceAudio
src.benchmark.input_view.materialize_full_mix_audio
src.benchmark.input_view.load_materialized_audio
src.benchmark.prediction_artifact.prediction_path
src.benchmark.prediction_artifact.read_prediction_artifact
src.benchmark.prediction_artifact.render_prediction_artifact
src.benchmark.reviewed_subset.score_reviewed_subset_cohort
src.benchmark.reference_set_manifest.preflight_reference_mappings
src.benchmark.cohort_scoring.cohort_item_from_artifacts
src.benchmark.cohort_scoring.score_cohort
src.benchmark.reports.write_cohort_reports
```

Add only:

- one derived-audio materializer that accepts a retained stem path while preserving the authoritative source identity;
- an optional `input_view_id` argument on OaF's existing inference-config builder, defaulting to the current full-mix ID;
- two small scorer bridge helpers extracted from the duplicated OaF/MuScriptor code;
- one shared published HPA-325 report reader extracted when HPA-328 becomes its second consumer.

Do not generalize the OaF/MuScriptor corpus runners.

## Fixed pilot membership

Use every row in the supplied canonical `crux.reviewed-reference-subset/v1` manifest. HPA-327 already enforces a reviewed 20–30-song population selected without model scores, so HPA-328 must not add another sample, seed, replacement, or score-aware filter.

Require exact lineage equality among:

```text
HPA-327 reviewed subset
HPA-324 reference manifest
HPA-323 timing manifest
HPA-326 persisted OaF run
```

Persist each HPA-327 row's `source_row_sha256` into the HPA-328 run and final handoff. A subset member missing from HPA-326 is fatal preflight. A persisted full-mix item failure remains an explicit control failure; membership is never silently replaced.

## Full-mix control

Reconstruct HPA-326 `run.json` + immutable prediction artifacts and rescore them on HPA-327 membership through `score_reviewed_subset_cohort()`.

HPA-328 never invokes OaF for the full-mix view. This saves compute and guarantees the control is the already published baseline.

## Separator lock and runtime contract

### Frozen v1 choices

```text
Spleeter: official 4-stem configuration, drums stem
Demucs: standard htdemucs model, drums stem
```

Do not try alternate Demucs variants, shifts, overlaps, clip modes, or multiple Spleeter configurations and select one based on benchmark score.

Each separator has one checked-in canonical `crux.separator-lock/v1` JSON containing:

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

Only stable model filenames/hashes enter the lock; absolute developer paths do not. `argv` contains score-relevant arguments after the interpreter executable. The canonical lock bytes' SHA-256 are the separator config identity.

Use exactly two public adapter functions:

```python
run_spleeter_drums(...)
run_htdemucs_drums(...)
```

Shared private process/QC helpers are fine. Do not create a plugin interface.

For deterministic output discovery, stage each source under a fixed temporary basename such as `input.wav`, so the lock can name stable expected outputs rather than depending on arbitrary source filenames.

## Exact stem cache

For a source and lock, derive identity from:

```python
{
    "source_audio_sha256": source_audio_sha256,
    "separator_id": separator_id,
    "separator_config_sha256": lock.sha256,
}
```

Persist the exact produced WAV at:

```text
derived/stems/<separator-id>/<source-sha>/<separator-lock-sha>/drums.wav
```

A cache hit requires matching source/config identity, exact persisted stem SHA, a readable regular file, and successful QC revalidation. Existing different bytes at the same identity fail closed; never overwrite them.

Retain native lossless WAV through HPA-396. A later cleanup may archive retained winners after dependent work completes; that is outside HPA-328.

## Practical fixed stem QC

Validate every generated or resumed stem before OaF:

1. regular/readable/nonempty file;
2. decodable audio with at least one frame;
3. all samples finite;
4. one or two channels only;
5. record sample rate/channel count; non-44.1 kHz or mono is a warning;
6. fail if duration differs from authoritative source by more than `max(0.5 sec, source_duration * 0.005)`;
7. fail if RMS is at/below `-80 dBFS`;
8. record clipping warning/fraction when absolute peak reaches `0.9999`;
9. never inspect DTX references or OaF scores during QC.

These values are frozen constants, not CLI flags. A failure is local to that separator view.

## OaF derived-input contract

Keep control view:

```text
crux.oaf-full-mix-mono44k1-pcm16/v1
```

Add:

```text
crux.oaf-spleeter4-drums-mono44k1-pcm16/v1
crux.oaf-htdemucs-drums-mono44k1-pcm16/v1
```

For every derived item preserve independently:

```text
authoritative source:
  source_audio_id
  source_audio_sha256

retained separator artifact:
  stem_path
  stem_sha256

actual canonical OaF input:
  input_view_id
  input_audio_sha256
```

Do not replace authoritative `source_audio_sha256` with the stem hash.

Add `materialize_derived_audio()` beside the current `materialize_full_mix_audio()`. It reads the retained stem, uses the same `librosa` `soxr_hq` mono 44.1 kHz conversion and `soundfile` PCM16 writer, then calls `load_materialized_audio()` with the authoritative source identity. Factor only the shared canonicalization body needed to keep full-mix bytes unchanged.

Extend OaF's inference-config builder with only:

```python
input_view_id: str = OAF_FULL_MIX_INPUT_VIEW_ID
```

All descriptor/model-lock/checkpoint/adapter/prediction-map/canonicalization fields remain identical. `build_run_id()` stays full-mix-specific and unchanged; HPA-328 has its own run identity.

## Persisted prediction → scorer bridge

Persisted derived-view prediction artifacts keep the actual authoritative source object identity. Before calling HPA-325, validate the persisted artifact against the HPA-328 run row, then create a scorer-local artifact whose only changed field is:

```python
CanonicalAudio.source_audio_id = str(simfile_id)
```

Promote this behavior from the existing OaF/MuScriptor copies into a public model-neutral helper in `cohort_scoring.py`. The helper must not mutate persisted bytes or change source SHA/input view/input SHA/descriptor/map identity.

Also promote the duplicated model-neutral constructor for failed/skipped/quarantined `CohortItem` values. Detailed separator errors stay in HPA-328 `run.json`; HPA-325 continues to receive only its existing closed failure reasons.

## Run snapshot and resume

Use one mutable canonical `crux.oaf-separation-run/v1` snapshot. Run identity binds:

```text
HPA-327/HPA-324/HPA-323 hashes
parent HPA-326 run ID
frozen OaF descriptor/model-lock/checkpoint
Spleeter lock SHA
HTDemucs lock SHA
both derived view IDs
SCORING_VERSION
Crux commit
```

Rows are sorted by `simfile_id` and include:

```text
source_row_sha256
source audio ID/SHA/duration
parent full-mix status/prediction identity
Spleeter status + lock + stem/QC/runtime + canonical input + OaF prediction/runtime
HTDemucs status + lock + stem/QC/runtime + canonical input + OaF prediction/runtime
```

Closed derived-view statuses:

```text
pending
separation_failed
stem_invalid
inference_failed
prediction_invalid
success
resumed
```

Resume requires exact run identity. Reuse stems only after content/QC validation and predictions only after immutable artifact source/input/descriptor/config identity checks.

One persistent OaF backend services all prediction misses. Existing HPA-326 backend error classification stays authoritative: item-local errors stay local; poison stops further OaF work for that invocation and leaves exact resume state. No automatic restart/tuning loop.

## Scoring

Preflight HPA-324/HPA-323 once with `preflight_reference_mappings()` before separator execution.

Publish three HPA-325 report sets:

```text
views/full_mix/reports
views/spleeter4_drums/reports
views/htdemucs_drums/reports
```

The full mix comes from persisted HPA-326. Derived views build scorer items through the shared persisted-artifact bridge above. Failed separator/inference views remain failed cohort rows instead of being dropped.

Only `score_cohort()` and `write_cohort_reports()` perform metric scoring. HPA-325 remains the owner of 30/50/100 ms, raw/aligned matching, event diagnostics, per-class aggregation, and population conventions.

## Shared published-report reader, specific comparison

`reports.py` owns HPA-325 output schemas, while `muscriptor_comparison.py` currently repeats them for reading. Before HPA-328 creates another reader, add one model-neutral read boundary that:

- parses `summary.json`, `items.csv`, `per_song.csv`, and `per_class.csv`;
- validates exact column schemas and canonical numeric domains;
- validates row identity against an expected `CohortIdentity`;
- rejects duplicates and score rows for non-success items;
- returns typed/read-only evidence for comparison code.

Retarget `muscriptor_comparison.py` to this reader with characterization tests proving its published output is unchanged.

Do **not** move MuScriptor/OaF backend-family checks or pair-specific rules into the shared reader.

HPA-328 then produces only:

```text
comparison/summary.json
comparison/per_song_delta.csv
comparison/per_class_delta.csv
comparison/summary.md
```

Join full-mix↔Spleeter and full-mix↔HTDemucs on existing score dimensions. Report independent population counts, paired-success counts, precision/recall/F1 values and deltas, per-class deltas, timing diagnostics, bounded helped/harmed cases, separator/OaF runtime, retained bytes, and optional explicit cost rates.

### FP/FN per minute

HPA-325 owns raw FP/FN counts but not per-minute normalization. HPA-328 derives without rescoring:

```python
minutes = source_duration_sec / 60
fp_per_min = fp / minutes
fn_per_min = fn / minutes
```

Require positive finite authoritative duration from `run.json`. Emit absolute rates and derived-minus-full-mix deltas using existing six-place numeric conventions.

Do not add significance tests, bootstrap intervals, or automatic winner ranking.

## Recommendation/finalization

A separate finalization step records the operator conclusion after reviewing the frozen report. Closed decisions:

```text
keep_full_mix
use_spleeter
use_htdemucs
gather_more_evidence
prioritize_another_model
```

Finalization records a nonempty rationale and optional nonnegative compute/storage rates. Rates affect evidence display only and never membership/inference/scoring.

## Immutable HPA-396 handoff

Publish canonical JSONL through existing manifest rails:

```text
crux.oaf-separation-pilot/v1
```

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
full-mix status/input-view/input hash/prediction path+SHA
Spleeter status/lock/stem path+SHA/input-view/input hash/prediction path+SHA
HTDemucs status/lock/stem path+SHA/input-view/input hash/prediction path+SHA
comparison artifact hashes
decision/rationale/cost rates
Crux commit
```

Success/resume requires all relevant artifact identities. Failed views require unavailable fields to be null. Finalization re-reads and hashes retained stems, predictions, and comparison artifacts before publication.

`source_row_sha256` must equal the HPA-327 row's frozen source-row hash. HPA-396 therefore gets exact row/reference/source/HTDemucs/OaF identity without consulting mutable HPA-328 state.

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

No model/QC/tolerance/third-separator/full-corpus flags. Run exit codes follow existing convention: 0 complete, 1 partial item/view failures, 2 fatal identity/preflight/publication failure. Finalization never invokes a separator or OaF backend.

## Failure semantics

### Fatal before expensive work

Manifest/subset/parent-run lineage mismatch, missing pilot member in parent population, invalid separator lock/model evidence, mixed OaF identity, or protected output alias.

### Item-local separator view

Process failure, missing/invalid/near-silent/duration-invalid stem, derived materialization failure, OaF item-local failure, or invalid/conflicting prediction artifact.

### Poisoned OaF process

Use existing HPA-326 backend error classification. Stop later OaF inference in that invocation, checkpoint remaining work, and resume later without regenerating valid stems.

## Testing strategy

Use TDD around small boundaries:

- characterize current full-mix materialization/inference identity before shared refactors;
- characterize OaF and MuScriptor scorer adaptation before extracting scorer helpers;
- characterize MuScriptor comparison bytes before extracting the report reader;
- unit-test separator locks, command rendering, cache identity, and QC without installing separators;
- acceptance-test that full-mix OaF is never called;
- acceptance-test exact 20–30 membership and independent per-view failures;
- acceptance-test derived prediction artifacts preserve authoritative source identity while scorer-local artifacts use `simfile_id`;
- test resume at stem and prediction boundaries;
- test handoff loader can supply HPA-396 exact HTDemucs stem/prediction identities without mutable run state;
- finish with repository-wide pytest, Ruff, format, Pylint errors-only, and `git diff --check`.

Real separator model reproduction and pilot execution are operational acceptance gates, not CI dependencies.

## Risks and gates

- **Dependency conflict:** isolated interpreter commands only.
- **Source/scorer identity mismatch:** persisted artifact validated first, then scorer-local source-ID adaptation through one shared helper.
- **Comparison-schema drift:** one HPA-325 reader owned beside the writer; no third CSV schema copy.
- **Selection bias:** HPA-327 membership only; no score-derived inputs.
- **Artifact drift:** exact source/config/stem/prediction hashes checked on resume/finalize.
- **Runaway compute:** no full-corpus path or third separator.
- **Over-abstraction:** only the two duplicated scorer helpers and duplicated report reader are promoted; runners and comparison policies remain concrete.

## Acceptance mapping

- Same songs/model/config across views → fixed HPA-327 membership + frozen OaF identity.
- Paired full mix/Spleeter/HTDemucs results → three HPA-325 report sets + fixed two comparisons.
- Cached separation artifacts/failures → content-addressed stem cache + per-view ledger.
- Accuracy/runtime/storage/cost → comparison artifacts + explicit rates.
- HTDemucs handoff → immutable `crux.oaf-separation-pilot/v1` rows with exact stem/prediction identities.
- No fine-tuning/full-corpus/framework work → explicit non-goals and narrow CLI.