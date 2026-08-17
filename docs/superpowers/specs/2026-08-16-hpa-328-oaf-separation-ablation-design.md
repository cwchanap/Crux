# HPA-328 OaF Drum-Stem Separation Ablation Design

**Status:** Proposed  
**Date:** 2026-08-16  
**Linear:** HPA-328 — Run a cost-controlled drum-stem separation ablation for validated OaF Drums

## Decision summary

Implement HPA-328 as one small paired-input experiment around the already validated OaF benchmark path:

1. use the **entire canonical HPA-327 reviewed subset** as the fixed pilot membership instead of inventing another selector;
2. reuse the persisted HPA-326 full-mix OaF predictions as the control — never rerun full-mix OaF merely for this experiment;
3. run exactly two required separators in v1: **Spleeter 4-stem drums** and **standard `htdemucs` drums**;
4. keep Spleeter and Demucs outside Crux's Python environment and invoke them through two concrete, pinned external-command adapters;
5. cache each exact separator-produced drum stem by authoritative source-audio SHA-256 plus separator-lock/config SHA-256;
6. canonicalize each retained stem through the same 44.1 kHz mono PCM16 OaF input boundary, while preserving the authoritative full-mix source identity and the exact retained stem hash separately;
7. run the unchanged OaF backend, prediction mapping, prediction artifact v2, HPA-325 scoring, and report writer on the two derived input views;
8. publish narrow paired full-mix-versus-Spleeter and full-mix-versus-HTDemucs reports plus one immutable JSONL handoff manifest for HPA-396;
9. keep the final recommendation operator-authored from a closed decision enum instead of encoding an arbitrary automatic winner rule.

Do **not** add a generic experiment runner, separator plugin registry, service/RPC layer, queue, worker pool, database, full-corpus separation path, third separator, SDR evaluator, model tuning, or compatibility layer.

## Why HPA-328 is next

HPA-328's three explicit blockers are complete:

- HPA-325 provides deterministic single-cohort scoring and reports;
- HPA-326 provides the validated broad OaF run and immutable prediction artifacts;
- HPA-327 provides the fixed reviewed 20–30-song subset and reviewed-subset scoring seam.

HPA-328 is also useful on the remaining critical path:

- it directly unlocks HPA-396, which must consume the exact HTDemucs stem hashes and the frozen OaF-on-HTDemucs results;
- it contributes one of the required result families for HPA-329's final benchmark recommendation;
- it can proceed independently of HPA-395's MuScriptor inference work.

HPA-395 is already active in PR #24. That PR is promoting several model-neutral HPA-326 seams that HPA-328 also needs. HPA-328 is **not** logically blocked on MuScriptor, but its implementation should start from refreshed `main` after #24 lands so it reuses those neutral helpers instead of creating conflicting copies.

## Approaches considered

### A. Import Spleeter and Demucs into the Crux Python 3.12 environment

Rejected.

Crux currently carries its own application/runtime dependencies and OaF isolation. Spleeter is TensorFlow-heavy, while Demucs is PyTorch-based. Pulling both into the main environment increases resolver pressure, installation time, CI weight, and maintenance cost for a 20–30-song experiment.

### B. Build a generic separator plugin/worker framework

Rejected.

A backend registry, RPC protocol, Docker worker abstraction, or generic experiment engine would be more infrastructure than HPA-328 needs. There are exactly two required separator commands and no production separator feature yet.

### C. Two pinned external commands behind one tiny experiment-specific seam

Selected.

Crux owns two concrete command builders, validates their frozen lock evidence, invokes each process, validates the expected drum-stem artifact, and records runtime/failure evidence. Tests inject a subprocess runner rather than installing the real tools.

This isolates dependencies without creating another framework.

## Goals

HPA-328 must provide:

- score-blind pilot membership fixed before separator/OaF results are inspected;
- exact full-mix, Spleeter, and HTDemucs input lineage for every pilot song;
- reproducible separator identity/configuration evidence;
- content-addressed retained drum-stem artifacts;
- practical stem-quality failure detection;
- frozen OaF inference on both separator views with no threshold/model/map changes;
- HPA-325 reports for all three input views on the same reviewed subset;
- paired per-song and per-class accuracy deltas;
- separation and OaF runtime/storage evidence;
- an explicit recommendation among the Linear decision-gate choices;
- an immutable HPA-396 handoff containing exact HTDemucs stem and OaF prediction identities.

## Non-goals

HPA-328 does not:

- train or fine-tune a separator or OaF;
- benchmark separator SDR or source-separation quality as an end in itself;
- add an optional third separator in v1;
- process the approximately 400-song corpus through separators;
- change OaF checkpoint, architecture, thresholds, peak picking, taxonomy, mapping, or scoring tolerances;
- run Inverse Drum Machine;
- add statistical-significance/bootstrap machinery;
- build a cloud-distributed execution system;
- make separator infrastructure part of the production Crux backend;
- optimize for backward compatibility with an experimental artifact schema.

## Implementation sequencing with HPA-395

PR #24 currently extracts/promotes model-neutral helpers that should be reused by HPA-328:

- `src.benchmark.corpus_cache.resolve_source_audio()` and `ResolvedSourceAudio`;
- `src.benchmark.input_view.materialize_full_mix_audio()`;
- `src.benchmark.prediction_artifact.prediction_path()`;
- `src.benchmark.reviewed_subset.score_reviewed_subset_cohort()`.

HPA-328 implementation should rebase onto the mainline version containing those changes before touching the same seams. If their final names move during HPA-395 review, update this design/plan during rebase rather than preserving stale aliases.

This is sequencing to reduce merge conflict, not a new Linear dependency.

## Fixed pilot membership

### Use HPA-327 exactly

The pilot membership is every row in the canonical `crux.reviewed-reference-subset/v1` manifest supplied to the command.

That manifest already enforces a reviewed 20–30-song population and was selected using reference-only features before model scores could affect membership. HPA-328 therefore does not need another random sample, seed, overlap rule, or 20–30-song selector.

The run requires exact lineage equality among:

```text
HPA-327 reviewed subset
HPA-324 reference manifest
HPA-323 timing manifest
HPA-326 persisted OaF full-mix run
```

A missing subset member in the parent HPA-326 run is fatal preflight. A parent item whose OaF prediction is missing/invalid remains an explicit full-mix control failure; HPA-328 does not silently replace the song.

### Full-mix control

The control is reconstructed from HPA-326 `run.json` plus immutable prediction artifacts and rescored on the HPA-327 membership through the existing reviewed-subset/HPA-325 path.

HPA-328 must not invoke the OaF backend for the full-mix view. This saves compute and guarantees the control is exactly the already published baseline rather than a fresh run that could differ because of environment drift.

## Separator identities and isolated runtimes

### Required v1 separators

Freeze exactly these semantic model choices before scored outputs are inspected:

```text
Spleeter: official 4-stem configuration, use the drums stem
Demucs: standard htdemucs model, use the drums stem
```

The upstream Spleeter repository contains an official `4stems.json` configuration. The upstream Demucs documentation identifies `htdemucs` as the default Hybrid Transformer model and emits a `drums.wav` stem. The Demucs upstream repository is archived, which makes retaining exact revision/package/model evidence especially important.

Do not compare multiple Spleeter configurations, Demucs model variants, shift counts, overlap settings, or clip modes and choose the best benchmark score.

### Per-separator lock

Each separator has one checked-in canonical lock using a common small schema:

```text
schema
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

The freeze/probe step may discover local model-cache paths, but only stable model filenames and SHA-256 values enter the lock. Absolute developer-machine paths do not.

`argv` captures the score-relevant command/config arguments after the interpreter executable. The runtime receives the interpreter path separately so a local venv location does not become benchmark identity.

The implementation computes:

```python
separator_config_sha256 = sha256(canonical_json_bytes(lock_payload)).hexdigest()
```

The exact package version, repository revision, model bytes, and command are frozen **before** the pilot is scored.

### Concrete process adapters

Use two functions, not a plugin interface:

```python
run_spleeter_drums(...)
run_htdemucs_drums(...)
```

Both accept an injected command runner for tests and return the same small `SeparatedStem` value. Shared process/error/QC helpers may be private functions in the same module.

The production command shapes are fixed by the locks. The HPA-328 CLI may select the two Python interpreter paths used to launch the isolated environments, but it does not expose separator model/config tuning flags.

## Stem cache and artifact identity

### Cache key

For each song and separator:

```python
stem_cache_id = sha256(
    canonical_json_bytes(
        {
            "source_audio_sha256": source_audio_sha256,
            "separator_id": separator_id,
            "separator_config_sha256": separator_config_sha256,
        }
    )
).hexdigest()
```

Store the exact produced drum stem beneath the HPA-328 output root:

```text
derived/stems/<separator_id>/<source_audio_sha256>/<separator_config_sha256>/drums.wav
```

The cache is a hit only when:

- the lock/config identity matches;
- the authoritative source-audio SHA-256 matches;
- the retained stem is a regular readable file;
- its content SHA-256 matches the persisted run row;
- stem QC can still parse the artifact.

A conflicting existing artifact fails closed. Do not overwrite a different stem at the same identity.

### Why keep exact WAV stems during the pilot

Both separator paths naturally produce lossless WAV artifacts and HPA-396 requires the **exact HTDemucs stem file and hash** used by the OaF comparison. For this small 20–30-song pilot, retain the exact produced WAV bytes through HPA-396 rather than adding a WAV→FLAC archive transform into benchmark identity.

After HPA-396 finishes, a later cleanup may losslessly archive retained winners as FLAC if storage matters. That is outside HPA-328 and must preserve the published original hash/path evidence.

## Practical stem QC

Validate every newly generated or resumed cached stem before OaF inference.

The v1 checks are deliberately practical rather than exhaustive:

1. output file exists, is regular/readable, and has nonzero byte length;
2. audio decoder succeeds and returns at least one frame;
3. all decoded samples are finite;
4. decoded channel count is 1 or 2; other layouts fail;
5. sample rate and channel count are recorded; values differing from the expected 44.1 kHz stereo separator output produce warnings rather than hidden normalization;
6. duration difference from the authoritative source must not exceed `max(0.5 seconds, source_duration * 0.005)`;
7. RMS at or below `-80 dBFS` is `near_silent` and fails the item/view;
8. peak absolute amplitude at or above `0.9999` is recorded as a clipping warning with clipped-sample fraction; it does not alone reject the stem because Demucs may legitimately rescale/clamp output;
9. QC never uses reference events or OaF scores.

The thresholds are frozen constants in v1. Do not expose them as CLI flags.

A QC failure is item-local for that separator view. It does not remove the song from the other view or from the paired population accounting.

## OaF input-view contract

### Existing full mix

Keep the existing control identity unchanged:

```text
crux.oaf-full-mix-mono44k1-pcm16/v1
```

### Derived views

Use two fixed semantic input-view IDs:

```text
crux.oaf-spleeter4-drums-mono44k1-pcm16/v1
crux.oaf-htdemucs-drums-mono44k1-pcm16/v1
```

The run-level ID names the semantic transform. Exact separator model/config bytes are carried independently by `separator_config_sha256`, and exact item audio bytes are carried by the stem and canonical-input SHA-256 values.

### Preserve source identity separately from stem identity

For each separator item retain three distinct identities:

```text
authoritative full mix:
  source_audio_id
  source_audio_sha256

retained separator output:
  stem_path
  stem_sha256

canonical OaF input derived from that stem:
  input_view_id
  input_audio_sha256
```

Do not overwrite `source_audio_sha256` with the stem hash. The PredictionArtifact `CanonicalAudio` continues to identify the authoritative full mix as the source and the canonicalized stem as the actual model input.

Add one narrow derived-view materializer beside the HPA-395 full-mix materializer. It reads the retained stem, uses the same `librosa` `soxr_hq` mono/44.1 kHz conversion and `soundfile` PCM16 WAV write, and then calls `load_materialized_audio()` with the authoritative source identity.

The canonical WAV is temporary. The retained separator stem and immutable prediction artifact are durable.

### Frozen OaF inference identity

Extend the existing OaF inference-config builder only enough to accept an explicit `input_view_id`, defaulting to the current full-mix value so HPA-326 bytes and tests remain unchanged.

All other OaF fields stay identical across views:

```text
backend descriptor
model_id
model lock hash
checkpoint archive hash
adapter revision
prediction map version
canonicalization revision
request/worker behavior
```

The different input-view ID produces a different inference-config hash and therefore a distinct immutable prediction artifact path without introducing prediction schema v3.

## HPA-328 run snapshot and resume

Use one mutable, canonical run snapshot:

```text
crux.oaf-separation-run/v1
```

The run identity binds:

```text
reviewed subset manifest SHA-256
reference manifest SHA-256
reference timing manifest SHA-256
HPA-326 full-mix parent run ID
OaF descriptor/model-lock/checkpoint identities
Spleeter lock SHA-256
HTDemucs lock SHA-256
both derived input-view IDs
scoring version
Crux commit
```

The item ledger is sorted by `simfile_id`. Each row contains:

```text
source identities
full-mix parent prediction identity/status
spleeter status + stem/QC/runtime + canonical input + OaF prediction identity/runtime
htdemucs status + stem/QC/runtime + canonical input + OaF prediction identity/runtime
```

Closed per-view dispositions:

```text
pending
separation_failed
stem_invalid
inference_failed
prediction_invalid
success
resumed
```

Resume validates exact run identity. It may reuse a stem only after content/QC validation and may reuse a prediction only after reading the immutable artifact and matching source/input/descriptor/config identities.

Do not automatically retry alternate separator settings, restart the OaF worker repeatedly, or silently regenerate conflicting artifacts.

One persistent OaF backend processes the two derived views sequentially. Existing HPA-326 backend error classification remains authoritative: item-local failures stay local; a poisoned worker stops further OaF inference for the invocation and leaves remaining work resumable.

## Scoring

### Reference/mapping preflight

Load and validate the same HPA-324/HPA-323 reference/timing manifests with `preflight_reference_mappings()` before any separator command runs.

Require the reviewed subset lineage to match those exact manifests.

### Three cohorts, one scorer

Produce HPA-325 reports for:

```text
full_mix
spleeter4_drums
htdemucs_drums
```

Full mix is reconstructed from the HPA-326 parent run and filtered to HPA-327 through the shared reviewed-subset scorer.

The derived views construct `CohortItem` values directly from the same reference mappings and persisted OaF prediction artifacts. Failures stay in the cohort population ledger; they are not dropped to make a cleaner comparison.

Call only:

```python
score_cohort(...)
write_cohort_reports(...)
```

Do not add a separator-specific matcher or scoring implementation.

All 30/50/100 ms raw/aligned metrics, FP/FN counts, timing diagnostics, per-class rows, and aggregate conventions therefore remain HPA-325-defined.

## Paired comparison report

HPA-562 still owns a general cross-model/cross-input comparison layer. HPA-328 only needs a narrow, fixed three-view report.

Read the already published HPA-325 `per_song.csv`, `per_class.csv`, `items.csv`, and `summary.json` files. Validate their cohort/reference/model/input identities, then join:

```text
full_mix ↔ spleeter4_drums
full_mix ↔ htdemucs_drums
```

Emit:

```text
comparison/summary.json
comparison/per_song_delta.csv
comparison/per_class_delta.csv
comparison/summary.md
```

The report includes:

- independent success/failure counts per view;
- paired successful-song count;
- F1/precision/recall deltas at 30/50/100 ms for raw and diagnostic-aligned modes already present in HPA-325;
- per-class deltas;
- songs/classes helped or harmed;
- FP/FN-per-minute and timing/alignment diagnostics from HPA-325 outputs where available;
- separator wall time, OaF wall time, real-time factor, retained stem bytes, prediction/report bytes;
- optional estimated compute/storage cost using operator-supplied nonnegative rates at finalization.

Do not bootstrap confidence intervals, significance-test deltas, or automatically rank a winner.

## Recommendation and finalization

The experiment runner produces evidence. A separate cheap finalization step records the human decision after reviewing the generated report.

Closed decision values mirror Linear:

```text
keep_full_mix
use_spleeter
use_htdemucs
gather_more_evidence
prioritize_another_model
```

`use_spleeter` and `use_htdemucs` are the concrete forms of Linear's “use a selected separator before OaF” option.

The finalizer also accepts a short rationale and optional nonnegative cost rates:

```text
compute_cost_per_hour
storage_cost_per_gb_month
```

These rates affect only the cost display/final decision evidence. They never affect membership, inference, scoring, or metric deltas.

No automatic decision function is added: the tradeoff between modest F1 changes, runtime, storage, and operational complexity is a product judgment, not a stable scoring formula.

## Immutable HPA-396 handoff manifest

Publish one JSONL manifest through the existing canonical manifest rails:

```text
crux.oaf-separation-pilot/v1
```

It contains **every HPA-327 pilot song**, including failed views, so downstream code never infers population from successful artifacts.

Each row contains the shared/reference identities plus per-view evidence:

```text
schema_version
corpus_version
simfile_id
reviewed_subset_manifest_sha256
reference_manifest_sha256
reference_manifest_version
reference_timing_manifest_sha256
reference_timing_version
taxonomy_version
lane_map_version
prediction_map_version
scoring_version
source_audio_id
source_audio_sha256
oaf_backend_descriptor_sha256
oaf_model_id
oaf_model_lock_sha256
oaf_checkpoint_archive_sha256
oaf_adapter_revision

full_mix_status
full_mix_input_view_id
full_mix_input_audio_sha256
full_mix_prediction_path
full_mix_prediction_sha256

spleeter_status
spleeter_separator_config_sha256
spleeter_stem_path
spleeter_stem_sha256
spleeter_input_view_id
spleeter_input_audio_sha256
spleeter_prediction_path
spleeter_prediction_sha256

htdemucs_status
htdemucs_separator_config_sha256
htdemucs_stem_path
htdemucs_stem_sha256
htdemucs_input_view_id
htdemucs_input_audio_sha256
htdemucs_prediction_path
htdemucs_prediction_sha256
```

Missing artifact fields are `null` only when that view has a non-success status. The loader validates the exact status/nullability combinations and content hashes for artifacts it is asked to resolve.

The manifest-level finalized evidence also records or binds:

```text
Spleeter lock SHA-256
HTDemucs lock SHA-256
comparison report SHA-256 values
decision
rationale
cost basis/rates
Crux commit
```

If repeating these shared values on each row makes the canonical JSONL easier to validate and consume, repeat them; do not introduce a second database/header protocol just to deduplicate 20–30 rows.

HPA-396 must consume `htdemucs_stem_path`, `htdemucs_stem_sha256`, the exact source/reference identities, and the OaF-on-HTDemucs prediction identity from this manifest. It must not regenerate stems or select another pilot membership.

## CLI

Add two thin commands:

```text
crux benchmark run-oaf-separation-pilot
crux benchmark finalize-oaf-separation-pilot
```

### Run command

Required inputs:

```text
--manifest
--timing-manifest
--subset-manifest
--oaf-run
--cache-dir
--output-dir
--spleeter-python
--demucs-python
```

Optional:

```text
--resume
```

No separator model/config/QC thresholds are CLI flags.

The command emits one canonical JSON summary with run ID/path, per-view success/failure counts, report paths, and exit code using the repository's existing 0/1/2 convention:

```text
0 = complete required pilot
1 = partial pilot with item/view failures
2 = fatal preflight/identity/publication error
```

### Finalize command

Required:

```text
--run
--decision
--rationale
--output-manifest
```

Optional cost-rate flags may be supplied. Finalization never invokes a separator or OaF backend.

## Failure semantics

### Fatal before separator/backend construction

Examples:

- manifest/timing/subset lineage mismatch;
- parent HPA-326 run mismatch;
- subset member absent from parent population;
- invalid separator lock/hash/model evidence;
- unknown/mixed run identity;
- output path aliases a protected parent report/artifact location.

### Item-local separator-view failure

Examples:

- separator process exits nonzero;
- expected drum output missing;
- stem cannot decode;
- near-silent or major-duration-mismatch stem;
- canonical stem materialization fails;
- OaF item-local inference error;
- immutable prediction artifact conflict/invalidity.

The other separator view for the song remains eligible to run.

### Poisoned OaF worker

Reuse HPA-326's closed backend error disposition. Stop later OaF inference in the current invocation, persist evidence, and rely on exact-identity resume. Do not add an automatic restart policy in HPA-328.

## Test strategy

### Characterization first

Before changing shared HPA-326/HPA-395 seams, pin current behavior for:

- full-mix OaF inference-config bytes/ID;
- HPA-326 full-mix canonicalization;
- prediction artifact path/identity;
- reviewed-subset parent reconstruction/report bytes.

The existing full-mix path must remain byte-identical after adding explicit derived input-view support.

### Separator adapter tests

Use fake subprocess executables/runners and tiny audio fixtures. Test:

- exact command construction from the frozen lock;
- interpreter path is not part of separator config identity;
- nonzero exit, missing output, malformed audio;
- deterministic cache path/hash;
- cache conflict/no-overwrite behavior;
- QC duration/silence/channel/clipping evidence;
- no reference/score input enters separator execution.

Real Spleeter/Demucs packages are not CI dependencies.

### Runner tests

Use fake separator adapters and fake OaF backend to cover:

- exact HPA-327 membership only;
- no full-mix OaF call;
- both derived views use the same OaF descriptor/map/scorer;
- one persistent derived-view OaF backend;
- item-local vs poison behavior;
- stem/prediction resume;
- immutable conflict handling;
- deterministic run snapshot ordering/serialization.

### Scoring/comparison tests

Use synthetic persisted HPA-325 reports to cover:

- exact identity joins;
- independent failure counts retained;
- paired-success intersection only for deltas;
- per-song/per-class deltas;
- helped/harmed direction;
- 30/50/100 ms raw/aligned rows preserved;
- runtime/storage/cost arithmetic.

### Handoff tests

Add a schema golden and loader tests for:

- 20–30 exact subset population;
- all shared lineage/lock/OaF/scoring identities;
- unique simfile IDs;
- success/failure field nullability;
- exact HTDemucs stem/path/hash handoff;
- final decision enum;
- deterministic rerender;
- rejection of edited generated identities.

## Acceptance mapping

HPA-328 acceptance maps directly to this design:

- **same fixed songs / same OaF** — exact HPA-327 manifest + one frozen OaF descriptor/model lock;
- **full mix + Spleeter + HTDemucs paired results** — persisted HPA-326 control plus two derived HPA-325 cohorts;
- **cache by source/config identity** — content-addressed exact stem cache;
- **explicit failures** — per-view closed dispositions and population ledger;
- **accuracy/runtime/storage/cost report** — narrow paired report + measured resource evidence;
- **HPA-396 handoff** — immutable `crux.oaf-separation-pilot/v1` JSONL with exact HTDemucs bytes and OaF prediction identities;
- **no tuning** — locked separator commands/QC thresholds and unchanged OaF/scoring configuration;
- **no full corpus** — CLI accepts only the reviewed-subset manifest and has no broad-corpus mode;
- **no framework creep** — two concrete separator adapters and one pilot orchestrator only.

## Deferred work

Defer until evidence justifies it:

- promoting a separator to a production preprocessing step;
- full-corpus separation;
- a third separator;
- generic paired-comparison framework (HPA-562);
- IDM adapter/inference (HPA-396);
- pruning/transcoding retained stems after HPA-396;
- cloud/distributed separator execution;
- source-separation SDR metrics;
- automatic decision policies.
