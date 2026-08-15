# HPA-327 Reviewed Reference Subset Design

**Status:** Proposed  
**Date:** 2026-08-15  
**Linear:** HPA-327 — Curate and audit a fixed reviewed reference subset

## Decision summary

Implement HPA-327 as four small pieces:

1. a deterministic, reference-only selector that freezes a 30-song candidate slate before model scores can influence membership;
2. one editable CSV review ledger for the manual chart/audio/timing audit;
3. one finalized canonical `crux.reviewed-reference-subset/v1` manifest containing the accepted 20–40 songs;
4. one narrow OaF rescore path that filters the already persisted HPA-326 cohort and reuses HPA-325 scoring/report generation without rerunning inference.

Do **not** add a database, reviewer UI, metadata service, clustering pipeline, configurable sampling DSL, generic experiment framework, automatic chart repair, second scorer, or model-result-aware selection.

## Why HPA-327 is next

HPA-321, HPA-322, HPA-323, HPA-324, HPA-325, HPA-326, and HPA-423 are complete, so HPA-327 is fully unblocked.

It also unlocks the most useful remaining benchmark work:

- HPA-395 MuScriptor requires the fixed reviewed subset;
- HPA-328 uses or overlaps it for the separator pilot;
- HPA-329 needs it for final comparative interpretation.

Starting those tasks first would leave subset selection unresolved or risk choosing diagnostic songs after seeing model results. HPA-327 should freeze the sample now.

## Goals

HPA-327 must provide:

- deterministic pre-score candidate membership and ordering;
- exact HPA-323/HPA-324 source identities and per-row hashes;
- a practical manual checklist for chart/audio/timing fidelity;
- explicit include/exclude reasons and known limitations;
- a reusable machine-readable subset manifest;
- reviewed-subset OaF reports produced from the same prediction artifacts and HPA-325 scorer as the broad result.

The subset complements the broad corpus. It does not replace it.

## Non-goals

HPA-327 does not:

- create a fine-tuning dataset;
- correct the approximately 400-chart corpus;
- change prediction artifacts;
- rerun OaF merely to score a subset;
- select songs from F1, confidence, runtime, FP/FN counts, or any other model output;
- automatically infer genre or acoustic/electronic character;
- preserve backward compatibility for later review schemas.

If review discovers a source problem, preserve the original broad result and publish a later source/subset revision rather than mutating the baseline.

## Existing seams to reuse

### HPA-324 reference boundary

Use `src/benchmark/reference_set_manifest.py` directly:

- `load_reference_set_manifest()` for canonical HPA-324 loading and lineage;
- `read_native_reference_events()` for event-path/hash/source-identity validation;
- `ReferenceSetRowView.eligibility_status` for the eligible population.

Do not create another reference-manifest reader.

### HPA-323 timing boundary

Use `load_reference_timing_manifest()` and require its hash/version to match the HPA-324 lineage exactly, using the same relationship HPA-326 already enforces.

Native reference paths remain relative to:

```python
timing_output_root = timing_manifest_path.parent.parent
```

Automated selection requires no R2 access.

### Reference mapping

Use `map_reference_events()` for common projected events and mapping diagnostics. Candidate features come from those persisted reference events only. There is no second taxonomy projection.

### HPA-325 scoring

Use `score_cohort()` and `write_cohort_reports()` unchanged for subset reports.

### HPA-326 persisted cohort

`oaf_corpus_run.py` already reconstructs `CohortItem` values from `run.json` and immutable prediction artifacts inside finalization. Extract that reconstruction into one public helper so broad finalization and subset rescoring share it.

This is the only HPA-326 refactor required.

## Fixed v1 selection policy

### Inputs

Candidate preparation consumes only:

```text
--manifest          exact HPA-324 benchmark-reference manifest
--timing-manifest   exact HPA-323 timing manifest referenced by HPA-324
--output-file       editable review CSV to create
```

It must not accept a run path, prediction path, HPA-325 report, score threshold, model/backend ID, or score-derived include/exclude filter. The absence of model-result inputs is part of the anti-bias contract.

### Frozen constants

```python
REVIEW_SUBSET_REVISION = "hpa327-v1"
REVIEW_TARGET_COUNT = 30
REVIEW_MIN_COUNT = 20
REVIEW_MAX_COUNT = 40
REVIEW_SELECTION_SEED = "crux-hpa327-v1"
```

Do not expose seed or target-count flags in v1. An intentional policy change creates a new subset revision instead of tuning this one interactively.

### Eligible population

Only HPA-324 `eligible` rows enter selection. Every eligible row must successfully reconstruct its native reference artifact and `ReferenceMappingResult`; a broken eligible artifact is fatal preflight inconsistency, matching HPA-326.

Population handling is fixed:

- fewer than 20 eligible rows: fail;
- 20–29 eligible rows: select all;
- 30 or more eligible rows: select exactly 30.

### Source-row hash

Record an exact per-row identity:

```python
source_row_sha256 = sha256(
    canonical_json_bytes(
        {key: value for key, value in source_row.items() if key != "corpus_version"}
    )
).hexdigest()
```

This supplements, not replaces, the HPA-324 manifest SHA-256.

### Derived pre-inference features

For each eligible row derive only:

```text
common_event_count
reference_event_span_sec
common_event_density_per_sec
common_class_count
has_timing_warning
uses_nonstandard_master_filename
```

Definitions:

- `common_event_count`: number of common projected events;
- `reference_event_span_sec`: last minus first common-event time, or `0` for one event;
- `common_event_density_per_sec`: `common_event_count / max(reference_event_span_sec, 1.0)`;
- `common_class_count`: distinct common classes present;
- `has_timing_warning`: whether carried HPA-323 `timing_warnings` is nonempty;
- `uses_nonstandard_master_filename`: selected basename equals `real.dtx` or `full.dtx`, case-insensitively.

Do not add genre lookup, embeddings, BPM extraction, separator analysis, or acoustic classification. Musical character is recorded manually during review.

### Exact rank bands

Create density and class-richness thirds independently.

For a population of `n`, sort rows ascending by `(feature_value, simfile_id)`. For zero-based sorted index `i`, assign:

```python
band_index = min(2, (i * 3) // n)
```

Map `0 -> low`, `1 -> medium`, `2 -> high`.

This makes ties deterministic through `simfile_id` and removes percentile-boundary ambiguity.

The selection stratum is:

```text
(density_band, class_richness_band, has_timing_warning, uses_nonstandard_master_filename)
```

Within each nonempty stratum, sort by:

```python
sha256(f"{REVIEW_SELECTION_SEED}:{source_row_sha256}".encode()).hexdigest()
```

Iterate nonempty strata in lexicographic order, taking one row from each per round until the target is filled. `candidate_rank` is the resulting 1-based acceptance order.

This is intentionally one fixed policy, not a sampling framework.

## Editable review ledger

Candidate preparation writes one exact-header CSV. Generated columns are immutable; audit columns are filled manually.

### Generated columns

```text
subset_revision
selection_seed
candidate_rank
simfile_id
source_reference_manifest_sha256
source_reference_manifest_version
source_timing_manifest_sha256
source_timing_manifest_version
source_row_sha256
selected_chart_key
selected_chart_content_hash
source_audio_key
source_audio_content_hash
common_event_count
reference_event_span_sec
common_event_density_per_sec
common_class_count
has_timing_warning
uses_nonstandard_master_filename
```

Float-derived cells use the repository's existing six-decimal token convention.

### Manual columns

```text
reviewer
reviewed_at
chart_selection_confirmed
audio_revision_confirmed
bgm_alignment_confirmed
technical_mapping_confirmed
musical_fidelity
drum_character
known_limitations
decision
reason_codes
notes
```

Closed values:

```text
confirmation fields   true | false
musical_fidelity      close | usable_with_limits | not_representative
drum_character        acoustic | electronic | hybrid | unknown
decision              include | exclude
```

`reviewed_at` is RFC3339 UTC.

`reason_codes` is a semicolon-separated list from:

```text
chart_selection_mismatch
audio_revision_mismatch
bgm_alignment_problem
chart_audio_drift
chart_simplification
chart_authored_error
unusual_lane_convention
not_representative
other
```

`other` requires nonempty `notes`. `known_limitations` and `notes` may otherwise be empty.

### Review rule

Review every candidate before model scores are used for membership decisions.

An included row must have all four confirmations `true` and fidelity `close` or `usable_with_limits`. A failed confirmation or `not_representative` requires `exclude` plus at least one reason code.

The ledger records drum character and limitations but v1 does not automatically optimize replacements around those categories.

If pre-score review leaves fewer than 20 acceptable rows or clearly inadequate diagnostic coverage, create a new subset revision with explicit replacement rationale. Do not mutate `hpa327-v1` after model scores are inspected.

## Finalization

Finalization consumes:

```text
--manifest          same HPA-324 manifest used for preparation
--timing-manifest   same HPA-323 manifest used for preparation
--review-file       completed candidate CSV
--output-dir        publication root
```

Before publication, require:

1. exact CSV header and unique ranks/simfile IDs;
2. frozen revision and seed;
3. membership/order exactly reproducible from the supplied manifests;
4. every generated identity/feature cell exactly reproducible;
5. reviewer, timestamp, audit enums, and decision completed for every row;
6. included rows satisfy confirmations/fidelity;
7. excluded rows contain valid reasons;
8. `other` rows have notes;
9. included count is 20–40;
10. every included row remains HPA-324 eligible.

A partially reviewed or edited-identity CSV is simply not publishable; the operator fixes it and reruns finalization. There is no repair engine.

### Canonical subset manifest

Publish accepted rows only with:

```text
schema_version = "crux.reviewed-reference-subset/v1"
corpus_version
subset_revision
candidate_rank
simfile_id
source_reference_manifest_sha256
source_reference_manifest_version
source_timing_manifest_sha256
source_timing_manifest_version
source_row_sha256
selected_chart_key
selected_chart_content_hash
source_audio_key
source_audio_content_hash
reviewer
reviewed_at
musical_fidelity
drum_character
known_limitations
reason_codes
notes
```

Use existing `render_manifest()`, `publish_manifest()`, and `publish_latest_manifest()`. The content-derived `corpus_version` is the immutable publication identity; `subset_revision` is the semantic review-policy revision.

Preserve the completed CSV byte-for-byte beside the publication output as `review-ledger.csv`. The manifest is the downstream machine contract; the CSV is the complete human audit record, including exclusions.

## Reviewed-subset scoring from persisted OaF artifacts

### Shared cohort reconstruction

Extract:

```python
def build_oaf_cohort_from_snapshot(
    snapshot: Mapping[str, object],
    *,
    mappings: Mapping[int, ReferenceMappingResult | None],
    output_dir: Path,
) -> tuple[CohortIdentity, tuple[CohortItem, ...]]:
    ...
```

`None` mappings are required for parent-run quarantined rows. The helper performs the same identity reconstruction and `_cohort_item_from_run_row()` adaptation currently embedded in `_finalize_scoring_and_outcome()`.

Broad HPA-326 finalization then calls this helper followed by `score_cohort()` exactly as today.

### Subset rescore command

Add:

```text
crux benchmark score-oaf-reviewed-subset \
  --run RUN_JSON \
  --manifest HPA324_MANIFEST \
  --timing-manifest HPA323_MANIFEST \
  --subset-manifest HPA327_MANIFEST \
  --output-dir REPORT_DIR
```

The domain function, kept in the existing OaF corpus-run boundary rather than a new generic runner, must:

1. parse the persisted HPA-326 snapshot;
2. load/reconcile HPA-324 and HPA-323 exactly as HPA-326 does;
3. reconstruct the full persisted cohort without constructing `OafBackend`;
4. load the HPA-327 manifest and require exact source-manifest/timing identity;
5. require every subset simfile ID to exist in the parent run;
6. filter to exact candidate-rank order, retaining failed/skipped/quarantined items when selected;
7. derive a distinct deterministic cohort ID;
8. call existing `score_cohort()` and `write_cohort_reports()`;
9. write only subset reports to the requested directory.

Broad run reports and prediction artifacts are untouched.

### Subset cohort ID

Retain every parent `CohortIdentity` dimension except `cohort_id`.

Define:

```python
subset_cohort_id = sha256(
    canonical_json_bytes(
        {
            "parent_run_id": parent_run_id,
            "reviewed_subset_manifest_sha256": subset_manifest_sha256,
        }
    )
).hexdigest()
```

No prediction artifact is copied or rewritten.

## CLI shape and outcomes

Add three thin commands under the existing benchmark group:

```text
prepare-reviewed-subset
finalize-reviewed-subset
score-oaf-reviewed-subset
```

Click callbacks own paths and concise JSON rendering only.

Use the existing `0 | 1 | 2` convention:

- prepare: `0` success, `2` invalid lineage/population/write failure;
- finalize: `0` published, `2` incomplete/invalid review or publication failure;
- subset score: `0` all selected items successful, `1` selected cohort contains item-level failure/skip/quarantine, `2` identity/artifact/report failure.

Manual exclusions do not need a special exit code.

## Error handling

Prefer explicit failure to recovery machinery.

Fatal preparation/finalization errors include mixed or noncanonical lineage, broken eligible event artifacts, fewer than 20 eligible rows, edited generated ledger fields, incomplete reviews, duplicate rows, invalid enums/reasons, accepted population outside 20–40, and publication failure.

A mismatched run/reference/subset identity, unreadable run snapshot, or missing subset member in the parent run is fatal during rescoring. Parent-run item failures remain normal HPA-325 cohort states and are not dropped.

## Testing strategy

### Candidate selection

Test:

- only eligible HPA-324 rows participate;
- no model/run input exists in the preparation API;
- 20–29 uses all, 30+ selects 30, fewer than 20 fails;
- row hashes are stable;
- input manifest ordering does not change membership/rank;
- exact thirds formula is deterministic under tied features;
- stratum round-robin is deterministic;
- timing-warning and `real.dtx`/`full.dtx` cases are represented by their intended flags;
- CSV headers and numeric tokens are exact.

### Finalization

Test:

- a fully completed ledger publishes;
- any generated-field edit is rejected;
- blank/invalid reviewer, timestamp, confirmation, fidelity, decision, reason, or notes-for-`other` is rejected;
- included rows require all confirmations and acceptable fidelity;
- excluded rows require reasons;
- accepted count remains 20–40;
- canonical manifest round-trip and source lineage are exact;
- `review-ledger.csv` preserves the submitted bytes.

### OaF rescore

Test:

- extracted cohort reconstruction preserves existing HPA-326 broad finalization results;
- subset scoring never constructs/invokes `OafBackend`;
- membership and candidate-rank ordering are exact;
- selected failed/skipped/quarantined rows remain present;
- broad reports remain unchanged;
- subset cohort ID is parent-run + subset identity only;
- HPA-325 scorer/report writer remains the sole scoring path;
- run/subset/reference mismatches fail closed.

### Acceptance fixture

Exercise prepare -> finalize -> rescore using synthetic canonical HPA-323/HPA-324 data plus a persisted OaF-run fixture. No R2, Docker, TensorFlow, or network dependency belongs in the automated acceptance path.

Do not fake the real human audit in tests.

## Operational acceptance

Code alone does not complete HPA-327.

For the real corpus:

1. generate and preserve the v1 candidate ledger before score-informed membership decisions;
2. manually inspect every candidate's selected chart, matching source audio, BGM alignment, mapping, and musical fidelity;
3. record reviewer/timestamp/confirmations/fidelity/drum character/limitations/decision/reasons/notes;
4. finalize the immutable 20–40-song subset and complete ledger;
5. verify the accepted sample contains materially different density, class, timing-warning, master-filename, and manually observed musical-character conditions;
6. rescore the existing OaF run on that exact membership;
7. preserve the reviewed-subset reports while leaving the broad run/reports unchanged.

If real HPA-323/HPA-324 artifacts are unavailable during implementation, the code may land after synthetic acceptance, but HPA-327 remains In Progress until the real review evidence is completed.

## Risks and mitigations

### Selection bias

Freeze membership from reference-only inputs and a code-owned seed; candidate preparation has no model-result input.

### Overengineering

Use six simple features and one fixed selector. Leave genre/acoustic classification and richer analytics out.

### Review drift

Finalization re-derives candidate membership and every generated cell. Any replacement requires a new subset revision.

### Source corrections

Document/exclude problems in the ledger and preserve the broad result; correct sources only through a later manifest/subset revision.

### Scoring divergence

Extract only persisted-cohort reconstruction from HPA-326, then call HPA-325 scoring/reporting unchanged.

## Completion criteria

HPA-327 is complete when:

- deterministic reference-only candidate generation exists and is tested;
- the real v1 candidate slate is frozen before score-informed membership changes;
- every real candidate has a completed review record;
- the accepted subset contains 20–40 songs with explicit source identities and exclusions;
- the canonical subset manifest and completed ledger are preserved;
- the same persisted OaF predictions generate HPA-325 reviewed-subset reports without rerunning inference;
- broad-corpus artifacts remain unchanged;
- no model-result-aware selection, training, or automatic chart repair is introduced.
