# HPA-328 OaF Drum-Stem Separation Ablation Design

**Status:** Proposed  
**Date:** 2026-08-16  
**Linear:** HPA-328 — Run a cost-controlled drum-stem separation ablation for validated OaF Drums

## Decision summary

Implement HPA-328 as one fixed paired-input experiment around the validated OaF benchmark path:

1. use the **entire canonical HPA-327 reviewed subset** as pilot membership;
2. reuse persisted HPA-326 full-mix OaF predictions as the control instead of rerunning them;
3. run exactly **Spleeter 4-stem drums** and **standard `htdemucs` drums** in v1;
4. keep both separators outside Crux's Python environment and invoke two concrete pinned external commands;
5. cache exact separator-produced drum WAVs by authoritative source SHA-256 + separator-lock SHA-256;
6. canonicalize retained stems through the same 44.1 kHz mono PCM16 OaF boundary while preserving authoritative full-mix source identity separately from exact stem/input hashes;
7. reuse unchanged OaF prediction mapping/artifacts and HPA-325 scoring/reports;
8. publish narrow full-mix↔separator deltas plus an immutable JSONL handoff containing exact HTDemucs stems and OaF-on-HTDemucs identities for HPA-396;
9. record the final recommendation explicitly rather than implementing an arbitrary automatic winner formula.

Do **not** build a generic experiment runner, separator plugin registry, RPC/service layer, queue, database, third separator, SDR evaluator, model tuning path, or full-corpus separator mode.

## Why HPA-328 is next

HPA-328 is fully unblocked: HPA-325, HPA-326, and HPA-327 are Done. It directly blocks HPA-396 and HPA-329, so it is the next useful independent slice while HPA-395 MuScriptor work continues.

HPA-395 PR #24 currently promotes model-neutral source/materialization/prediction/subset-scoring seams that HPA-328 also needs. HPA-328 is not logically dependent on MuScriptor, but implementation should start from refreshed `main` after #24 lands to reuse those seams rather than create duplicate helpers.

## Approaches considered

### Import Spleeter and Demucs into Crux

Rejected. Spleeter is TensorFlow-heavy and Demucs is PyTorch-based. Adding both to the Python 3.12 application environment increases resolver/CI/install cost for a 20–30-song experiment.

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

## Reuse existing benchmark seams

After #24 lands, reuse these neutral helpers:

```text
src.benchmark.corpus_cache.resolve_source_audio
src.benchmark.corpus_cache.ResolvedSourceAudio
src.benchmark.input_view.materialize_full_mix_audio
src.benchmark.prediction_artifact.prediction_path
src.benchmark.reviewed_subset.score_reviewed_subset_cohort
src.benchmark.reference_set_manifest.preflight_reference_mappings
src.benchmark.cohort_scoring.score_cohort
src.benchmark.reports.write_cohort_reports
```

HPA-328 adds only one narrow derived-audio materializer and an optional `input_view_id` argument to the existing OaF inference-config builder. Full-mix default bytes/identity must stay unchanged.

## Fixed pilot membership

Use every row in the supplied canonical `crux.reviewed-reference-subset/v1` manifest. HPA-327 already enforces the reviewed 20–30-song population and model-blind selection, so HPA-328 must not add another sample/seed/replacement policy.

Require exact lineage equality among:

```text
HPA-327 reviewed subset
HPA-324 reference manifest
HPA-323 timing manifest
HPA-326 persisted OaF run
```

Persist each HPA-327 row's `source_row_sha256` into the HPA-328 run/handoff so downstream HPA-396 has row-level reference identity in addition to manifest/version identity.

A subset member missing from HPA-326 is fatal preflight. A persisted full-mix item failure stays an explicit control failure; membership is never silently replaced.

## Full-mix control

Reconstruct HPA-326 `run.json` + immutable prediction artifacts and rescore them on the HPA-327 membership through the existing reviewed-subset/HPA-325 path.

HPA-328 never creates an OaF backend for the full-mix view. This both saves compute and guarantees the control is the already published baseline.

## Separator lock and runtime contract

### Frozen v1 model choices

```text
Spleeter: official 4-stem configuration, drums stem
Demucs: standard htdemucs model, drums stem
```

The official Spleeter repository includes `4stems.json`. Demucs upstream documentation identifies `htdemucs` as the default Hybrid Transformer model and its separation output includes `drums.wav`. The upstream Demucs repository is archived, strengthening the need to retain exact revision/model evidence.

Do not try alternate Demucs model variants, shifts, overlap, clip modes, or multiple Spleeter configurations and select one based on score.

### Canonical lock

Each separator has one checked-in `crux.separator-lock/v1` JSON containing:

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

Only stable model filenames/hashes enter the lock; absolute developer paths do not. `argv` contains score-relevant arguments after the interpreter executable. The interpreter path is runtime plumbing and is separately supplied by CLI.

The canonical lock bytes' SHA-256 are the separator config identity.

Use exactly two public adapter functions:

```python
run_spleeter_drums(...)
run_htdemucs_drums(...)
```

Shared private process/QC helpers are fine. Do not create a plugin interface.

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

### Why retain WAV rather than convert to FLAC during the pilot

HPA-396 explicitly needs the exact HTDemucs stem file/hash used by HPA-328. The pilot is only 20–30 songs, so retaining the native lossless WAV avoids making an archive transcode part of inference identity. A later cleanup may archive retained winners after HPA-396; that is outside HPA-328.

## Practical fixed stem QC

Validate every generated or resumed stem before OaF:

1. regular/readable/nonempty file;
2. decodable audio with at least one frame;
3. all samples finite;
4. one or two channels only;
5. record sample rate/channel count; non-44.1 kHz or mono is a warning, not hidden special-case processing;
6. fail if duration differs from authoritative source by more than `max(0.5 sec, source_duration * 0.005)`;
7. fail if RMS is at/below `-80 dBFS`;
8. record clipping warning/fraction when absolute peak reaches `0.9999`;
9. never inspect DTX references or OaF scores during QC.

These values are frozen constants, not CLI flags. A failure is local to that separator view and never removes the song from the other view/population ledger.

## OaF derived-input contract

Keep existing control view:

```text
crux.oaf-full-mix-mono44k1-pcm16/v1
```

Add:

```text
crux.oaf-spleeter4-drums-mono44k1-pcm16/v1
crux.oaf-htdemucs-drums-mono44k1-pcm16/v1
```

For every derived item preserve three identities independently:

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

Add `materialize_derived_audio()` beside the existing full-mix helper. It reads the retained stem, uses the same `librosa` `soxr_hq` mono 44.1 kHz conversion and `soundfile` PCM16 WAV writer, then calls `load_materialized_audio()` with the authoritative source identity. The canonical WAV is temporary; stem and prediction artifacts are durable.

Extend OaF's inference-config builder with only:

```python
input_view_id: str = OAF_FULL_MIX_INPUT_VIEW_ID
```

All descriptor/model-lock/checkpoint/adapter/prediction-map/canonicalization fields remain identical. The view-specific config hash naturally produces distinct immutable prediction artifact paths.

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

One persistent OaF backend services all prediction misses. Existing HPA-326 backend error classification stays authoritative: item-local errors stay local; a poisoned worker stops further OaF work for that invocation and leaves exact resume state. No automatic restart/tuning loop.

## Scoring

Preflight HPA-324/HPA-323 once with `preflight_reference_mappings()` before separator execution.

Publish three HPA-325 report sets:

```text
views/full_mix/reports
views/spleeter4_drums/reports
views/htdemucs_drums/reports
```

The full mix comes from persisted HPA-326. Derived views construct normal `CohortItem` values from the same reference mappings and prediction-v2 artifacts. Failed separator/inference views remain failed cohort rows instead of being dropped.

Only `score_cohort()` and `write_cohort_reports()` perform metric scoring. HPA-325 remains the owner of 30/50/100 ms, raw/aligned matching, event diagnostics, per-class aggregation, and population conventions.

## Narrow paired comparison

HPA-562 owns any future generic cross-model/cross-input comparison. HPA-328 reads the already published HPA-325 artifacts and produces only:

```text
comparison/summary.json
comparison/per_song_delta.csv
comparison/per_class_delta.csv
comparison/summary.md
```

Validate full report identity from `summary.json`, then join full-mix↔Spleeter and full-mix↔HTDemucs rows on their existing score dimensions.

Report:

- independent population/success/failure counts per view;
- paired-success count for each comparison;
- precision/recall/F1 values and derived-minus-full-mix deltas at 30/50/100 ms for existing raw/aligned modes;
- per-class deltas;
- `median_abs_error_ms`, `p95_abs_error_ms`, and alignment offset from HPA-325;
- bounded most-helped/most-harmed songs/classes;
- separator wall time, derived OaF wall time, real-time factor, retained stem bytes, prediction/report bytes;
- optional compute/storage cost using explicit nonnegative operator rates.

### FP/FN per minute

HPA-325 owns raw FP/FN counts but not per-minute normalization. HPA-328 derives it without rescoring:

```python
minutes = source_duration_sec / 60
fp_per_min = fp / minutes
fn_per_min = fn / minutes
```

`source_duration_sec` comes from the authoritative resolved source recorded in `run.json`. Require a positive finite duration. Emit absolute per-view rates and derived-minus-full-mix deltas using existing six-place numeric conventions.

Do not add significance tests, bootstrap intervals, or automatic winner ranking.

## Recommendation/finalization

A separate finalization step records the operator's conclusion after the frozen report is reviewed. Closed decisions:

```text
keep_full_mix
use_spleeter
use_htdemucs
gather_more_evidence
prioritize_another_model
```

Finalization also records a nonempty rationale and optional nonnegative compute/storage rates. Rates affect only cost display/evidence and never membership/inference/scoring.

No automatic policy decides whether a small F1 gain is worth separation latency/storage/complexity.

## Immutable HPA-396 handoff

Publish canonical JSONL through the existing manifest rails:

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

No model/QC/tolerance/third-separator/full-corpus flags.

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

Run exit codes follow existing convention: 0 complete, 1 partial item/view failures, 2 fatal identity/preflight/publication failure. Finalization never invokes a separator or OaF backend.

## Failure semantics

### Fatal before expensive work

Manifest/subset/parent-run lineage mismatch, missing pilot member in the parent population, invalid separator lock/model evidence, mixed OaF identity, or protected output alias.

### Item-local separator view

Process failure, missing/invalid/near-silent/duration-invalid stem, derived materialization failure, OaF item-local inference error, or immutable prediction conflict/invalidity. The other separator view remains eligible.

### Poisoned OaF worker

Reuse HPA-326 classification: checkpoint, stop later OaF inference for the invocation, report partial state, resume later with exact identity. No automatic restart policy.

## Test strategy

Characterize full-mix OaF bytes before shared seam changes. Use fake subprocess runners/tiny WAV fixtures for separator command/cache/QC tests and fake OaF backend for orchestration/resume tests; real separator packages are not CI dependencies.

Acceptance tests must prove:

- exact HPA-327 membership;
- no full-mix OaF inference call;
- identical OaF model semantics except input view;
- independent per-view failures retained;
- exact stem/prediction resume validation;
- HPA-325 reports include complete populations and existing tolerances/modes;
- paired-success intersection is used only for deltas;
- FP/FN per minute uses recorded source duration, not a scorer fork;
- final JSONL has exact row/source/HTDemucs/OaF identities and deterministic rendering.

## Acceptance mapping

- Same songs/model/config → exact HPA-327 membership + one frozen OaF identity.
- Full mix/Spleeter/HTDemucs paired results → persisted control + two derived HPA-325 cohorts.
- Cache/failures → source+lock stem cache and explicit per-view statuses.
- Accuracy/runtime/storage/cost → narrow paired report including FP/FN/minute and timing diagnostics.
- HPA-396 handoff → immutable JSONL with `source_row_sha256`, exact HTDemucs stem hashes/paths, and OaF-on-HTDemucs prediction identities.
- No tuning/full corpus/framework → fixed locks/QC/scorer and intentionally narrow CLI/modules.

## Deferred work

Defer production separator preprocessing, full-corpus separation, third separators, generic paired comparison (HPA-562), IDM execution (HPA-396), post-HPA-396 stem archival/pruning, cloud/distributed separator execution, SDR metrics, and automatic decision policies until evidence justifies them.
