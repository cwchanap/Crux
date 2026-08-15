# HPA-327 Reviewed Reference Subset Design

**Status:** Proposed  
**Date:** 2026-08-15  
**Linear:** HPA-327 — Curate and audit a fixed reviewed reference subset

## Decision summary

Implement HPA-327 as four small pieces:

1. a deterministic, reference-only selector that freezes a 30-song candidate slate before model scores can influence membership;
2. one spreadsheet-friendly review CSV for the manual chart/audio/timing audit, treated as an operator editing surface rather than a source-of-truth identity format;
3. one canonical `crux.reviewed-reference-subset/v1` manifest containing the accepted 20–30 songs, with a loader and schema golden;
4. one narrow OaF rescore path that reconstructs the already persisted HPA-326 cohort and reuses HPA-325 scoring/report generation without rerunning inference.

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
- a cheap continuation path that preserves valid completed reviews when replacements are needed before scores are inspected;
- a reusable machine-readable subset manifest with enough selection evidence for HPA-395/HPA-328/HPA-329;
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

If review discovers a source problem, preserve the original broad result and publish a later source/subset version rather than mutating the baseline.

## Existing seams to reuse

### HPA-324 reference boundary

Use `src/benchmark/reference_set_manifest.py` directly:

- `load_reference_set_manifest()` for canonical HPA-324 loading and lineage;
- `read_native_reference_events()` for event-path/hash/source-identity validation;
- `ReferenceSetRowView.eligibility_status` for the eligible population;
- `ReferenceSetRowView.common_scored_event_count` as the published common-event accounting check.

Do not create another reference-manifest reader.

### HPA-323 timing boundary

Use `load_reference_timing_manifest()` and require its hash/version to match the HPA-324 lineage exactly.

`timing_warnings` comes from `LoadedReferenceTimingRow.view.timing_warnings`; it is not part of `ReferenceSetRowView`.

Native reference paths remain relative to:

```python
timing_output_root = timing_manifest_path.parent.parent
```

Automated selection requires no R2 access.

### Shared reference preflight

HPA-326 already has `_preflight_reference_mappings()` with the exact checks HPA-327 needs:

- HPA-324/HPA-323 lineage equality;
- per-simfile timing-row reconciliation;
- selected chart/audio identity equality;
- native event artifact validation;
- `map_reference_events()` reconstruction;
- `None` mappings for legitimate quarantined rows;
- fatal failure for broken eligible reference artifacts.

Promote this logic without changing semantics to:

```python
src.benchmark.reference_set_manifest.preflight_reference_mappings(
    reference_manifest: LoadedReferenceSetManifest,
    timing_manifest: LoadedReferenceTimingManifest,
    *,
    timing_output_root: Path,
) -> dict[int, ReferenceMappingResult | None]
```

`reference_set_manifest.py` is the correct model-independent home because it already owns native-event loading, reference-manifest rows, timing-manifest integration, and imports `map_reference_events()`.

Do **not** move this helper into `reference_set.py`: `reference_set_manifest.py` already imports `reference_set.py`, so importing manifest/timing loaders back into `reference_set.py` would create a circular dependency.

HPA-326 and HPA-327 both import the promoted helper from `reference_set_manifest.py`. This keeps candidate preparation model-blind and avoids importing `oaf_corpus_run.py`, which currently imports librosa, soundfile, the TF1 model runtime, and OaF backend code at module load time.

### HPA-325 scoring

Use `score_cohort()` and `write_cohort_reports()` unchanged for subset reports.

HPA-325 already supports bounded event diagnostics through `diagnostics_for`. Broad HPA-326 scoring keeps `diagnostics_for=()`. The reviewed subset passes the IDs of its successful items so HPA-329 has matched/FP/FN evidence for the manually audited population.

### HPA-326 persisted cohort

`oaf_corpus_run.py` already reconstructs `CohortItem` values from `run.json` and immutable prediction artifacts inside `_finalize_scoring_and_outcome()`.

Extract only that reconstruction into:

```python
def build_oaf_cohort_from_snapshot(
    snapshot: Mapping[str, object],
    *,
    mappings: Mapping[int, ReferenceMappingResult | None],
    output_dir: Path,
) -> tuple[CohortIdentity, tuple[CohortItem, ...]]:
    ...
```

`None` mappings remain required for parent-run quarantined rows.

Broad HPA-326 finalization derives `output_dir` exactly as it does today (`output_dir or run_path.parents[2]`) and calls the helper. The subset rescore command passes `run_path.parents[2]` explicitly.

HPA-327 therefore needs two narrow no-behavior-change HPA-326 extractions:

1. promote reference preflight into `reference_set_manifest.py`;
2. extract persisted cohort reconstruction next to `_cohort_item_from_run_row()`.

No prediction, execution, retry, worker, or report behavior changes are part of those refactors.

## Fixed v1 selection policy

### Inputs

Candidate preparation consumes only:

```text
--manifest          exact HPA-324 benchmark-reference manifest
--timing-manifest   exact HPA-323 timing manifest referenced by HPA-324
--output-file       review CSV to create
[--prior-ledger]    optional earlier HPA-327 review CSV to preserve valid work
```

It must not accept a run path, prediction path, HPA-325 report, score threshold, model/backend ID, or score-derived include/exclude filter. The absence of model-result inputs is part of the anti-bias contract.

`--prior-ledger` is review evidence, not a policy parameter. It is allowed only before model scores are used for membership decisions.

### Frozen constants

```python
REVIEW_POLICY_VERSION = "hpa327-v1"
REVIEW_TARGET_COUNT = 30
REVIEW_MIN_COUNT = 20
REVIEW_MAX_COUNT = 30
REVIEW_SELECTION_SEED = "crux-hpa327-v1"
```

Do not expose seed or target-count flags in v1. A selection-policy change creates a new policy version instead of tuning this one interactively.

The finalized subset's content-derived `corpus_version` is the subset instance/version. `REVIEW_POLICY_VERSION` identifies the frozen selection/review rules; it is not a mutable subset revision counter.

### Eligible population and mapping accounting

Only HPA-324 `eligible` rows enter selection.

Use `preflight_reference_mappings()` once. Every eligible row must yield a mapping. Then require:

```python
len(mapping.common_events) == loaded_reference_row.view.common_scored_event_count
```

A mismatch is a fatal persisted-reference inconsistency. Feature extraction never silently recomputes a different event population from HPA-324.

Population handling is fixed:

- fewer than 20 eligible rows: fail;
- 20–29 eligible rows: select all;
- 30 or more eligible rows: select 30.

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
selects_real_or_full_chart
```

Definitions:

- `common_event_count`: number of common projected events;
- `reference_event_span_sec`: last minus first common-event time, or `0` for one event;
- `common_event_density_per_sec`: `common_event_count / max(reference_event_span_sec, 1.0)`;
- `common_class_count`: distinct common classes present;
- `has_timing_warning`: whether the matching HPA-323 timing row has any `timing_warnings`;
- `selects_real_or_full_chart`: selected chart basename equals `real.dtx` or `full.dtx`, case-insensitively.

The last flag is intentionally descriptive rather than calling `real`/`full` "nonstandard": the repository's current filename priority is `real`, `full`, `mas`, `ext`, `adv`, `bas`.

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
(density_band, class_richness_band, has_timing_warning, selects_real_or_full_chart)
```

### Seeded stratum order

Do not iterate strata lexicographically. With up to 36 nonempty strata and a 30-song target, lexicographic truncation would systematically starve whichever band names sort last.

For each nonempty stratum, build the canonical key:

```python
stratum_key = (
    f"{density_band}|{class_richness_band}|"
    f"{int(has_timing_warning)}|{int(selects_real_or_full_chart)}"
)
```

Order strata by:

```python
sha256(f"{REVIEW_SELECTION_SEED}:{stratum_key}".encode()).hexdigest()
```

Within each stratum, order rows by:

```python
sha256(f"{REVIEW_SELECTION_SEED}:{source_row_sha256}".encode()).hexdigest()
```

Round-robin over that seeded stratum order, taking one row per nonempty stratum per round. Continue producing a deterministic candidate stream even after the first 30 rows; the continuation is used only by `--prior-ledger` replacement preparation.

`candidate_rank` is the 1-based order within the current prepared ledger. Scoring does not preserve this order; HPA-325 keeps canonical `simfile_id` order.

### Carry-forward without redoing accepted manual work

An initial preparation takes the first target-count rows from the deterministic candidate stream.

When `--prior-ledger` is supplied:

1. read the prior CSV bytes and compute `prior_review_ledger_sha256`;
2. parse the prior CSV by `simfile_id`;
3. use the prior `source_row_sha256` only as a carry-forward guard, comparing it to the freshly derived current row hash;
4. carry forward the 12 manual audit fields for rows whose source hash is unchanged and whose prior decision is `include` with a valid completed review;
5. treat unchanged previously reviewed `exclude` rows as consumed so they are not offered again;
6. if a prior candidate's source hash changed, do not carry its review and allow the current row to appear again as an unreviewed candidate;
7. fill the remaining candidate slots from the deterministic candidate stream, skipping unchanged rows already reviewed in the prior ledger;
8. preserve the previous relative order of carried included rows, then append replacement rows in deterministic stream order;
9. stop at `REVIEW_TARGET_COUNT` or when no unused eligible rows remain.

This preserves scarce human listening work while keeping replacement choice independent of model scores. The seed and policy remain frozen.

If fewer than `REVIEW_MIN_COUNT` candidate rows can be produced after carry-forward and replacement, preparation fails.

## Review CSV

The CSV is an editing surface for approximately 30 manual audits, not a second identity database.

### Generated display/evidence columns

Preparation writes:

```text
review_policy_version
selection_seed
prior_review_ledger_sha256
candidate_rank
simfile_id
source_reference_manifest_sha256
source_reference_manifest_version
source_timing_manifest_sha256
source_timing_manifest_version
source_row_sha256
selected_chart_key
selected_chart_content_hash
selected_chart_cache_path
source_audio_key
source_audio_content_hash
source_audio_cache_path
common_event_count
reference_event_span_sec
common_event_density_per_sec
common_class_count
density_band
class_richness_band
has_timing_warning
selects_real_or_full_chart
```

`prior_review_ledger_sha256` is empty for the initial slate and the exact SHA-256 of the supplied prior review CSV for a continuation slate.

`selected_chart_cache_path` comes from the HPA-324 row.

`source_audio_cache_path` is the stable content-addressed relative cache path derived from the source audio hash:

```python
f"sha256/{source_audio_content_hash[:2]}/{source_audio_content_hash}"
```

Do not persist a machine-specific absolute cache path.

Float-derived display values use the repository's existing canonical numeric convention directly:

```python
canonical_json_bytes(quantize_six(value)).decode("ascii")
```

Do not import the private `reports._csv_decimal()` helper.

### Manual audit columns

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

The ledger records drum character and limitations but v1 does not choose replacements by hand around those categories. Replacement membership comes from the deterministic continuation stream.

## Finalization

Finalization consumes:

```text
--manifest          same HPA-324 manifest used for preparation
--timing-manifest   same HPA-323 manifest used for preparation
--review-file       completed candidate CSV
--output-dir        publication root
[--prior-ledger]    required only when this review file came from continuation preparation
```

For a continuation ledger, finalization must receive the same prior ledger bytes used by preparation. It hashes those bytes and re-runs the same carry-forward/replacement selection before validating the current review. This is necessary because continuation membership depends on which earlier rows were already reviewed; the current HPA-323/HPA-324 manifests alone cannot reproduce that slate.

### Trust boundary

The editable CSV's generated cells are hints/evidence for the reviewer, not source authority.

Finalization trusts only:

- `simfile_id` to associate a review with a freshly reconstructed candidate row;
- the 12 manual audit columns.

It does **not** reject a review merely because a spreadsheet rewrote a generated boolean, decimal token, line ending, or hash cell. Instead it re-runs the selector/preflight, using the same optional prior ledger when applicable, and re-derives every generated value from HPA-323/HPA-324.

`source_row_sha256` is read from a prior ledger only for the optional carry-forward guard described above; it is never accepted as current source identity without recomputation.

Before publication, require:

1. the expected column set, unique canonical integer `simfile_id` values, and no duplicate reviews;
2. every review row names a member of the freshly reproduced current candidate slate;
3. every current candidate has exactly one review row;
4. reviewer, timestamp, audit enums, and decision are completed for every row;
5. included rows satisfy confirmations/fidelity;
6. excluded rows contain valid reasons;
7. `other` rows have notes;
8. included count is 20–30;
9. every included row remains HPA-324 eligible with the same freshly reconstructed source identity;
10. continuation finalization's supplied prior-ledger hash matches the freshly reproduced slate lineage.

A partially reviewed or stale-membership CSV is not publishable. The operator fixes it or generates a continuation ledger before looking at model scores. There is no repair engine.

### Canonical complete review ledger

After validation, finalization re-renders `review-ledger.csv` from:

- freshly reconstructed generated columns;
- the validated manual review fields.

The canonical output uses UTF-8, `\n` line endings, stable header order, and `csv.DictWriter` quoting. The submitted spreadsheet bytes are not treated as canonical evidence.

Compute:

```python
review_ledger_sha256 = sha256(review_ledger_bytes).hexdigest()
```

and bind that hash into the accepted subset manifest.

### Canonical subset manifest

Publish accepted rows only as `crux.reviewed-reference-subset/v1` through existing `render_manifest()`, `publish_manifest()`, and `publish_latest_manifest()`.

Each accepted row carries:

```text
schema_version
corpus_version
review_policy_version
prior_review_ledger_sha256
review_ledger_sha256
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
density_band
class_richness_band
has_timing_warning
selects_real_or_full_chart
reviewer
reviewed_at
musical_fidelity
drum_character
known_limitations
reason_codes
notes
```

`prior_review_ledger_sha256` is null for an initial subset and the exact prior-ledger hash for a continuation subset.

The six selection features plus both band labels remain in the published artifact so HPA-395, HPA-328, and HPA-329 can inspect why the fixed sample is diverse without re-deriving selection metadata from upstream manifests.

The complete ledger preserves exclusions and confirmation details; the subset manifest is the downstream accepted-membership contract.

### Loader and schema golden

Implement:

```python
def load_reviewed_subset_manifest(path: Path) -> LoadedReviewedSubsetManifest:
    ...
```

in `src/benchmark/reviewed_subset.py`.

The loader must use:

```python
read_canonical_manifest_core(
    path,
    schema_version=REVIEWED_REFERENCE_SUBSET_SCHEMA,
    validate_rows=...,
)
```

so canonical JSONL framing, exact input SHA-256, schema enforcement, and `render_manifest()` byte round-trip stay on the existing manifest rails.

Add `crux.reviewed-reference-subset/v1` to `tests/benchmark/schema_goldens/manifest.json` with a golden under `tests/benchmark/schema_goldens/`, and expose the module's `validate_schema_golden()` entrypoint expected by `test_schema_goldens.py`.

Downstream HPA-395/HPA-328/HPA-329 and the OaF subset scorer use this loader rather than parsing JSONL independently.

## Reviewed-subset scoring from persisted OaF artifacts

### Inputs

Add:

```text
crux benchmark score-oaf-reviewed-subset \
  --run RUN_JSON \
  --manifest HPA324_MANIFEST \
  --timing-manifest HPA323_MANIFEST \
  --subset-manifest HPA327_MANIFEST \
  --output-dir REPORT_DIR
```

Keep orchestration in `src/benchmark/reviewed_subset.py`; do not grow `oaf_corpus_run.py` into a second runner.

### Flow

The domain function:

1. reads `RUN_JSON` and validates it with `parse_oaf_corpus_run()`;
2. loads HPA-324 and HPA-323 through their existing loaders;
3. reconstructs mappings with `preflight_reference_mappings()` from `reference_set_manifest.py`;
4. requires run/reference/timing identity equality;
5. loads HPA-327 with `load_reviewed_subset_manifest()` and requires the same source-manifest/timing identities;
6. calls `build_oaf_cohort_from_snapshot(snapshot, mappings=mappings, output_dir=run_path.parents[2])` without constructing `OafBackend`;
7. requires every subset `simfile_id` to exist in the parent run population;
8. filters the full persisted cohort to exact subset membership, retaining selected failed/skipped/quarantined items;
9. derives a distinct deterministic subset `cohort_id`;
10. passes only successful selected IDs to `score_cohort(..., diagnostics_for=successful_subset_ids)`;
11. calls `write_cohort_reports()` once;
12. writes only subset reports to the requested output directory.

HPA-325 sorts cohort items and report rows by `simfile_id`; `candidate_rank` remains selection provenance in the subset manifest and does not fork scorer/report ordering.

Broad run reports and prediction artifacts remain untouched.

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

- prepare: `0` success, `2` invalid lineage/population/review-continuation/write failure;
- finalize: `0` published, `2` incomplete/invalid review or publication failure;
- subset score: `0` all selected items successful, `1` selected cohort contains item-level failure/skip/quarantine, `2` identity/artifact/report failure.

Manual exclusions do not need a special exit code.

## Error handling

Prefer explicit failure to recovery machinery.

Fatal preparation/finalization errors include:

- mixed or noncanonical HPA-323/HPA-324 lineage;
- broken eligible event artifacts;
- recomputed common-event count differing from HPA-324 `common_scored_event_count`;
- fewer than 20 eligible/current candidates;
- invalid, unknown, or inconsistent prior-ledger review data;
- missing/mismatched prior ledger during continuation finalization;
- stale or duplicate review membership;
- incomplete reviews;
- invalid enums/reasons;
- accepted population outside 20–30;
- publication failure.

A mismatched run/reference/subset identity, unreadable run snapshot, or missing subset member in the parent run is fatal during rescoring. Parent-run item failures remain normal HPA-325 cohort states and are not dropped.

## Testing strategy

### Shared no-behavior-change extractions

Before new subset behavior depends on them:

- move `_preflight_reference_mappings()` to public `reference_set_manifest.preflight_reference_mappings()` and run existing HPA-326 acceptance tests unchanged;
- extract `build_oaf_cohort_from_snapshot()` and characterize current broad finalization so identical persisted input produces the same `CohortIdentity`, item states, population, and reports;
- keep existing `test_oaf_corpus_run_acceptance.py` and `test_cohort_scoring_acceptance.py` green without fixture rewrites that hide behavior changes.

### Reusable synthetic fixture

Introduce a canonical synthetic HPA-323/HPA-324 + persisted OaF-run fixture early and reuse it through prepare, finalize, and rescore tests. The end-to-end acceptance chain must be runnable before the last implementation task.

No R2, Docker, TensorFlow, or network dependency belongs in this fixture.

### Candidate selection

Test:

- only eligible HPA-324 rows participate;
- no model/run input exists in the preparation API;
- published HPA-324 `common_scored_event_count` must match reconstructed common events;
- 20–29 uses all, 30+ selects 30, fewer than 20 fails;
- row hashes are stable;
- input manifest ordering does not change membership/rank;
- exact thirds formula is deterministic under tied features;
- seeded stratum order is deterministic;
- a population with more than 30 nonempty strata does not systematically drop one named density/class band due to lexical ordering;
- timing-warning is read from the HPA-323 row;
- `real.dtx`/`full.dtx` cases set `selects_real_or_full_chart`;
- generated chart/audio cache paths are correct;
- float-derived CSV tokens use `quantize_six()` + `canonical_json_bytes()`.

### Carry-forward

Test:

- unchanged valid included rows keep their manual audit fields;
- unchanged excluded rows are not re-offered;
- changed source-row hashes invalidate carry-forward and allow re-review;
- replacement rows come only from the next unused deterministic candidates;
- a prior ledger never changes the seed or selector policy;
- malformed prior manual fields fail closed;
- continuation finalization requires the same prior ledger and rejects a different/missing prior hash.

### Finalization and manifest loading

Test:

- a fully completed review publishes;
- spreadsheet changes to generated display cells do not become source authority;
- stale or unknown `simfile_id` membership is rejected;
- blank/invalid reviewer, timestamp, confirmation, fidelity, decision, reason, or notes-for-`other` is rejected;
- included rows require all confirmations and acceptable fidelity;
- excluded rows require reasons;
- accepted count remains 20–30;
- canonical `review-ledger.csv` is regenerated from fresh generated fields plus review data;
- subset rows contain all six features and both band labels;
- current and optional prior ledger hashes bind the manifest to the audit lineage;
- `load_reviewed_subset_manifest()` uses `read_canonical_manifest_core()` and rejects noncanonical/mixed/duplicate input;
- the schema golden is registered and validates through `test_schema_goldens.py`.

### OaF rescore

Test:

- subset scoring never constructs or invokes `OafBackend`;
- membership is exact while HPA-325 keeps `simfile_id` ordering;
- selected failed/skipped/quarantined rows remain present;
- only successful selected IDs are passed through `diagnostics_for`;
- broad reports remain unchanged;
- subset cohort ID is parent-run + subset identity only;
- HPA-325 scorer/report writer remains the sole scoring path;
- run/subset/reference mismatches fail closed;
- `output_dir` for persisted prediction reconstruction is derived from `run_path.parents[2]`.

Do not fake the real human audit in tests.

## Operational acceptance

Code alone does not complete HPA-327.

For the real corpus:

1. generate and preserve the initial v1 candidate ledger before score-informed membership decisions;
2. manually inspect every candidate's selected chart, matching source audio, BGM alignment, mapping, and musical fidelity;
3. record reviewer/timestamp/confirmations/fidelity/drum character/limitations/decision/reasons/notes;
4. if review leaves fewer than 20 acceptable songs or clearly inadequate diagnostic coverage, generate a continuation ledger from `--prior-ledger` **before** consulting model scores, preserving unchanged included reviews and filling only deterministic unused replacements;
5. finalize the initial ledger directly, or finalize a continuation with the same prior ledger used for its preparation; preserve the resulting immutable 20–30-song subset, canonical complete ledger, and schema-valid manifest;
6. verify the published subset artifact itself shows materially different density, class-richness, timing-warning, `real`/`full` chart, and manually observed musical-character conditions;
7. rescore the existing OaF run on that exact membership with event diagnostics for successful subset songs;
8. preserve the reviewed-subset reports while leaving the broad run/reports unchanged.

If real HPA-323/HPA-324 artifacts are unavailable during implementation, the code may land after synthetic acceptance, but HPA-327 remains In Progress until the real review evidence is completed.

## Risks and mitigations

### Selection bias

Freeze membership from reference-only inputs and a code-owned seed. Seed both stratum ordering and within-stratum ordering so deterministic truncation does not structurally prefer a named band.

### Wasted manual review during replacement

Use optional pre-score `--prior-ledger` continuation. Carry unchanged included audits forward and choose only replacement rows from the unused deterministic candidate stream. Continuation finalization receives the same prior ledger so the replacement slate is reproducible.

### Overengineering

Use six simple features, two rank bands, and one fixed selector. Leave genre/acoustic classification and richer analytics out.

### Review drift

Finalization ignores editable generated cells as authority and reconstructs membership/source evidence from HPA-323/HPA-324 plus the explicit prior-ledger input for continuation. Manual review fields are the only operator-authored decisions.

### Source corrections

Document/exclude problems in the ledger and preserve the broad result. If source bytes change, source-row hashing prevents stale manual review from carrying forward automatically.

### Scoring divergence

Land the two HPA-326 extractions as characterization-tested no-behavior-change refactors before subset behavior depends on them, then call HPA-325 scoring/reporting unchanged.

## Completion criteria

HPA-327 is complete when:

- deterministic reference-only candidate generation exists and is tested;
- seeded stratum order has no lexical band preference;
- the real v1 candidate slate is frozen before score-informed membership changes;
- every accepted real candidate has a completed valid review record;
- any pre-score replacement pass preserves unchanged included reviews instead of restarting the audit;
- the accepted subset contains 20–30 songs with explicit source identities, selection features/bands, and exclusions preserved in the canonical ledger;
- `crux.reviewed-reference-subset/v1` has a canonical loader and schema golden;
- the same persisted OaF predictions generate HPA-325 reviewed-subset reports with selected-song diagnostics and without rerunning inference;
- broad-corpus artifacts remain unchanged;
- no model-result-aware selection, training, or automatic chart repair is introduced.
