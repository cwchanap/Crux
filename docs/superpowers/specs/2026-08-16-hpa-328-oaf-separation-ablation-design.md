# HPA-328 OaF Drum-Stem Separation Ablation Design

**Status:** Proposed  
**Date:** 2026-08-16  
**Reviewed against main:** 2026-08-17, `d9a124835a020b630db088112fa733b33baee619`  
**Linear:** HPA-328 — Run a cost-controlled drum-stem separation ablation for validated OaF Drums

## Decision summary

Implement HPA-328 as one fixed paired-input experiment around the validated OaF benchmark path:

1. use the entire canonical HPA-327 reviewed subset as pilot membership;
2. reuse persisted HPA-326 full-mix OaF predictions as the control and never rerun full-mix OaF;
3. compare exactly Spleeter 4-stem drums and standard `htdemucs` drums;
4. keep both separators in isolated external runtimes;
5. retain exact separator-produced WAV stems for HPA-396;
6. canonicalize stems through the existing 44.1 kHz mono PCM16 OaF boundary while keeping source, stem, and canonical-input identity distinct;
7. reuse the existing OaF transcribe → map → publish path and HPA-325 scorer;
8. reuse shared persisted-artifact matching and published-report comparison primitives instead of creating HPA-328 copies;
9. compare each separator with full mix on the exact paired-success population;
10. publish one immutable HPA-396 handoff and one operator-recorded recommendation.

Do not add a separator plugin registry, generic experiment runner, RPC/service layer, queue, database, third separator, SDR scorer, significance/bootstrap layer, model tuning, or a full-corpus separator path.

## Latest review resolution

The latest review is **accepted except for one deliberately narrowed refactor**. The experiment does not change.

### 1. Share the existing comparison engine, not only its report reader

`muscriptor_comparison.py` already owns the mechanical pieces HPA-328 needs: successful-population intersection, metric deltas, per-song/per-class joins, deterministic CSV writing, canonical summary writing, and Markdown rendering.

Task 8A therefore promotes the model-neutral pieces into a focused `src/benchmark/published_comparison.py` beside the new `read_cohort_reports()` boundary. MuScriptor keeps its existing behavior and output bytes through characterization tests.

The one real policy difference becomes explicit:

```python
def pairable_success_ids(
    left: PublishedRunEvidence,
    right: PublishedRunEvidence,
    selected_ids: set[str] | None,
    *,
    require_identical_input_hash: bool = True,
) -> tuple[set[str], dict[str, int]]: ...
```

MuScriptor uses the default `True`, because both models consume the same canonical full-mix bytes. HPA-328 passes `False`, because different derived `input_audio_sha256` values are the experiment; source-audio SHA equality remains required.

Shared helpers also cover deterministic metric deltas, paired song/class row construction, CSV writing, and Markdown rendering. HPA-328-specific comparison code is limited to:

- two pair drivers: full mix ↔ Spleeter and full mix ↔ HTDemucs;
- paired event-micro TP/FP/FN aggregation;
- FP/FN per minute;
- native HPA-328 failure-code histograms;
- runtime and retained-byte totals.

Use the established filenames `paired_per_song.csv` and `paired_per_class.csv`. Because HPA-328 has two comparisons, write them under:

```text
comparison/spleeter/paired_per_song.csv
comparison/spleeter/paired_per_class.csv
comparison/htdemucs/paired_per_song.csv
comparison/htdemucs/paired_per_class.csv
comparison/summary.json
comparison/summary.md
```

### 2. Promote view-aware prediction matching instead of writing a third matcher

The current OaF artifact matcher contains a redundant full-mix equality immediately before comparing the artifact view with the supplied `CanonicalAudio.input_view_id`. Its current caller always supplies full-mix audio, so removing the redundant equality preserves HPA-326 behavior and makes the check useful for derived inputs.

Move the model-neutral matching logic next to `PredictionArtifact` in `prediction_artifact.py`:

```python
def prediction_artifact_matches_audio(
    artifact: PredictionArtifact,
    *,
    source_audio_id: str,
    source_audio_sha256: str,
    audio: CanonicalAudio,
    descriptor: BackendDescriptor,
    prediction_map_version: str,
) -> bool: ...


def prediction_artifact_matches_run_row(
    artifact: PredictionArtifact,
    row: Mapping[str, object],
    *,
    expected_input_view_id: str,
) -> bool: ...
```

OaF, MuScriptor, and HPA-328 call these public helpers. HPA-328 does not import runner-private symbols or hand-write another identity boolean.

### 3. Do not move all runner helpers into `run_snapshot_common.py`

The review correctly identifies duplication between the OaF and MuScriptor corpus runners, but moving all eighteen byte-identical helpers is wider than HPA-328 needs. Several encode the existing flat corpus-run shape (`include/exclude`, flat item dispositions, skip/quarantine helpers) and are not natural inputs to HPA-328's fixed membership plus nested Spleeter/HTDemucs state.

HPA-328 instead:

- reuses existing public primitives such as `require_sha256()`, `format_manifest_timestamp()`, `read_regular_file_no_follow()`, `atomic_replace_bytes()`, `prediction_path()`, the promoted prediction matchers, and the promoted scorer bridge;
- promotes only functions that gain a real third caller;
- keeps `crux.oaf-separation-run/v1` parse/render/nullability logic local to `separation_pilot.py`.

This avoids both a third copy of useful leaf behavior and an unnecessary run framework.

### 4. Add a pre-implementation separator feasibility gate

Before implementation work begins, stand up both isolated runtimes and run one known WAV through each official `-m` command. The gate only proves:

```text
interpreter exists
module imports
command exits successfully
drums.wav exists and decodes
```

It does not hash model files or create benchmark locks. If either required separator cannot be reproduced, stop and revise the design before building the two-separator run schema.

Real package/repository/model hashing and production lock generation remain at the final pre-score gate.

### 5. Add a fixed one-shot separator timeout

Spleeter and Demucs are one-shot child processes, so do not reuse `WorkerProcess`. Use one fixed ceiling:

```python
SEPARATOR_TIMEOUT_SECONDS = 1800.0
SEPARATOR_TERMINATE_GRACE_SECONDS = 5.0
```

The timeout is not a CLI option. Spawn in a new process session; on timeout terminate the process group, wait the grace interval, then kill if required. Persist native detail code `separator_timeout`, set the view status to `separation_failed`, checkpoint, and continue with the other view/song.

### 6. Preserve native failure families in comparison output

HPA-325 correctly collapses separator/stem/canonical-input failures into `inference_failed`, but HPA-328 still needs to explain whether lost population came from separation, stem QC, canonicalization, OaF inference, or prediction publication.

`comparison/summary.json` and `summary.md` therefore include, for each derived view, a histogram of the native HPA-328 failure codes from `run.json`, alongside the paired-success size and the independent HPA-325 population counts.

### 7. Remove speculative dollar-cost and top-five plumbing

Do not persist or expose compute/storage rate configuration. HPA-328 records the evidence it actually measures:

```text
separator wall time seconds
OaF wall time seconds
retained stem bytes
prediction/report bytes
```

No `CostRates`, no `--compute-cost-per-hour`, no `--storage-cost-per-gb-month`, and no cost fields in the immutable handoff. A future consumer can derive dollars if a real marginal rate exists.

Also do not materialize separate top-five helped/harmed lists. The pilot has at most 30 reviewed songs; deterministic paired CSVs already expose the complete sorted evidence.

## Existing decisions retained

### Fixed membership and control

Use every row in `crux.reviewed-reference-subset/v1`. Require exact HPA-327/HPA-324/HPA-323 lineage and require every subset member to exist in the parent HPA-326 run.

Use `score_oaf_reviewed_subset()` to write:

```text
<run>/views/full_mix/reports/
```

Parse HPA-326 `run.json` separately only for parent/run-row identity evidence. Never create an OaF backend for the full-mix control.

### Separator lock

Each real separator uses one canonical `crux.separator-lock/v1` containing:

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

Unit tasks use synthetic fixture locks. The final pre-score gate generates and commits the real locks from reproduced runtimes before benchmark scores are inspected.

### Stem cache and QC

Store exact native WAV bytes at:

```text
derived/stems/<separator-id>/<source-sha>/<separator-lock-sha>/drums.wav
```

Publish through `publish_immutable_file()`. Cache hits require matching identity, exact bytes, and QC revalidation.

Fixed QC:

- readable nonempty decodable file;
- finite samples;
- one or two channels;
- duration delta `<= max(0.5 sec, source_duration * 0.005)`;
- RMS above `-80 dBFS`;
- clipping at absolute peak `>= 0.9999` recorded as warning/evidence, not an automatic rejection;
- no reference or OaF score is consulted.

### OaF derived views

Keep:

```text
crux.oaf-full-mix-mono44k1-pcm16/v1
```

Add exactly:

```text
crux.oaf-spleeter4-drums-mono44k1-pcm16/v1
crux.oaf-htdemucs-drums-mono44k1-pcm16/v1
```

`build_inference_config()` gains only an explicit `input_view_id` parameter, defaulting to full mix. The field remains part of the inference-config hash, so the immutable `prediction_path()` cannot collide across views. `build_run_id()` stays full-mix-specific; HPA-328 owns its own run identity.

### Derived inference

For every prediction miss use exactly:

```python
native = backend.transcribe(audio)
mapped, _ = map_oaf_prediction(native)
published = publish_prediction_artifact(prediction_target, mapped)
```

Route `OafBackendError` through `classify_oaf_backend_error()`. Do not create another mapper, publisher, or runner base class.

### Scoring and failure mapping

Promote the scorer-local source-ID adaptation already duplicated by OaF and MuScriptor. Persisted artifacts keep authoritative source-object identity; scorer-local artifacts may change only `source_audio_id` to `simfile_id` after persisted bytes have been validated.

Do not extend `CohortFailureReason`. Freeze:

```text
separation_failed          -> inference_failed
stem_invalid               -> inference_failed
canonical_input_failed     -> inference_failed
inference_failed           -> inference_failed
prediction_invalid         -> prediction_artifact_invalid
prediction_output_conflict -> prediction_artifact_invalid
prediction_publish_failed  -> prediction_artifact_invalid
```

Detailed native codes remain in HPA-328 `run.json`.

### Paired comparison

Independent HPA-325 `summary.json` aggregates remain independent evidence. Headline comparison metrics are recomputed from published per-song TP/FP/FN counts on the exact paired-success population; this is aggregation, not event rescoring.

Derive FP/FN per minute using authoritative source duration:

```python
minutes = source_duration_sec / 60
fp_per_min = fp / minutes
fn_per_min = fn / minutes
```

### HPA-396 handoff

Publish `crux.oaf-separation-pilot/v1` through existing `render_manifest()` / `publish_manifest()` rails. Every HPA-327 member remains present. Rows bind:

```text
simfile_id
source_row_sha256
subset/reference/timing identities
source audio ID/SHA
OaF backend/model/model-lock/checkpoint/adapter identities
full-mix status/input/prediction identity
Spleeter status/lock/stem/input/prediction identity
HTDemucs status/lock/stem/input/prediction identity
comparison artifact hashes
decision
rationale
Crux commit
```

No dollar-cost fields are part of v1.

### Operator decision

Final decision remains one of:

```text
keep_full_mix
use_spleeter
use_htdemucs
gather_more_evidence
prioritize_another_model
```

The finalizer records a nonempty rationale. No automatic winner rule is added.

## CLI

Run command:

```text
crux benchmark run-oaf-separation-pilot
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

Finalize command:

```text
crux benchmark finalize-oaf-separation-pilot
--run
--subset-manifest
--output-manifest
--decision
--rationale
```

No separator model/QC/tolerance/full-corpus/third-separator/cost-rate flags.

## Testing and acceptance gates

- **Task 0 feasibility:** both isolated required separators produce decodable `drums.wav` before implementation begins.
- **Full-mix regression:** full-mix OaF materialization/config/prediction resume stays byte/identity compatible.
- **Matcher regression:** OaF and MuScriptor resume behavior remains unchanged after public matcher extraction; derived views pass the same matcher through supplied input-view identity.
- **MuScriptor comparison regression:** shared comparison primitive extraction preserves existing published comparison output.
- **Timeout:** injected timeout terminates the one-shot separator child and checkpoints `separation_failed` without blocking the other view.
- **Population:** every HPA-327 member appears in all three HPA-325 population ledgers.
- **Comparison:** paired-success sizes, native failure histograms, paired TP/FP/FN/F1, FP/FN-per-minute, runtime, and retained bytes are explicit.
- **Handoff:** HPA-396 can obtain exact HTDemucs stem/path/hash and OaF-on-HTDemucs prediction identity without mutable HPA-328 state.
- **Operational freeze:** real separator locks are generated and committed before scored execution.

## Explicit non-goals

No `run_snapshot_common.py` sweep, generic comparison framework, separator plugin system, cloud execution, retry/backoff engine, full-corpus separator processing, dollar-cost model, or presentation-only top-five ranking.