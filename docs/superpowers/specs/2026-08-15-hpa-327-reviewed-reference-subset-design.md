# HPA-327 Reviewed Reference Subset Design

**Status:** Proposed  
**Date:** 2026-08-15  
**Linear:** HPA-327 — Curate and audit a fixed reviewed reference subset

## Decision summary

Implement HPA-327 as one deterministic pre-score candidate selector, one human-review ledger, one finalized reviewed-subset manifest, and one narrow rescore path over persisted OaF run artifacts.

The workflow is deliberately small:

1. load the exact HPA-324 benchmark-reference manifest and its HPA-323 timing lineage;
2. reconstruct eligible reference events without reading any model prediction or score;
3. derive a few pre-inference stratification features already available from reference artifacts;
4. deterministically choose a fixed 30-song candidate slate using a code-owned seed and balanced strata;
5. emit an editable review CSV with immutable source identity columns and blank audit fields;
6. manually review every candidate before consulting model scores;
7. validate the completed review CSV and publish the accepted 20–40 songs as a canonical `crux.reviewed-reference-subset/v1` manifest;
8. filter an already persisted HPA-326 OaF cohort to that exact membership and reuse HPA-325 scoring/report generation without rerunning inference.

Do **not** add a database, reviewer UI, generic experiment framework, clustering pipeline, metadata service, configurable sampling DSL, model-result-aware selection, automatic chart repair, or second scoring implementation.

## Why HPA-327 is next

HPA-321, HPA-322, HPA-323, HPA-324, HPA-325, HPA-326, and HPA-423 are complete. HPA-327 is therefore fully unblocked.

It is also the highest-leverage remaining dependency in the benchmark epic:

- HPA-395 MuScriptor requires the reviewed subset in addition to the completed OaF/scoring pipeline;
- HPA-328 uses or overlaps the same fixed subset for the separator pilot;
- HPA-329 requires the reviewed subset for final comparative interpretation.

Starting HPA-395 or HPA-328 first would either leave subset selection unresolved or risk selecting diagnostic songs after seeing model results. HPA-327 should freeze the human-reviewed reference sample now, while selection can still be proven independent of prediction quality.

## Goals

HPA-327 must provide:

- a deterministic, reproducible pre-score candidate set;
- explicit source hashes and selection rules;
- a practical manual checklist for chart/audio/timing fidelity;
- a versioned record of include/exclude decisions and limitations;
- a canonical accepted-subset manifest usable by later model/input comparisons;
- a way to produce reviewed-subset OaF reports from the same persisted predictions and scorer as the broad corpus.

The subset is diagnostic evidence, not a replacement for the broad-corpus benchmark.

## Non-goals

HPA-327 does not:

- create a fine-tuning or training dataset;
- correct the full chart corpus;
- modify any prediction artifact;
- rerun OaF inference merely to score a subset;
- select songs from OaF F1, error counts, confidence, runtime, or any other model output;
- infer genre or acoustic/electronic character automatically;
- build a general survey/review application;
- introduce backward-compatibility machinery for future review schemas.

If a reviewed source needs correction, preserve the original broad result and publish a new source/subset revision later. Do not mutate the baseline manifest in place.

## Existing seams to reuse

### HPA-324 reference manifest

`src/benchmark/reference_set_manifest.py` already provides the model-independent eligibility boundary:

- `load_reference_set_manifest()` validates canonical HPA-324 rows and lineage;
- `LoadedReferenceSetManifest` exposes exact manifest SHA-256 and corpus version;
- eligible rows carry HPA-322 chart identity and HPA-323 timing/audio identity through unchanged;
- `read_native_reference_events()` validates the referenced native-event artifact path, content hash, and source identity.

HPA-327 must use these functions rather than introducing another reference-manifest reader.

### HPA-323 timing manifest

`src/benchmark/reference_timing_manifest.py::load_reference_timing_manifest()` is the authoritative HPA-323 loader. HPA-327 supplies the exact timing manifest referenced by HPA-324 and verifies the same hash/version relationship used by HPA-326.

The timing output root remains:

```python
timing_output_root = timing_manifest_path.parent.parent
```

No R2 access is needed for automated candidate selection.

### Reference mapping

`src/benchmark.reference_set.map_reference_events()` already produces:

- mapped native drum events;
- common projected events;
- ignored/unmapped lane diagnostics;
- duplicate-common-event accounting.

Candidate feature extraction uses these persisted reference events only. Do not parse model output and do not add a second taxonomy projection.

### HPA-325 scorer and reports

`src/benchmark/cohort_scoring.py` already owns the scorer-facing cohort contract and `score_cohort()`. `src/benchmark/reports.py::write_cohort_reports()` already owns deterministic per-song, per-class, aggregate, coverage, and summary outputs.

Reviewed-subset scoring must call those same functions. There is no reviewed-subset metric implementation.

### HPA-326 persisted run

`src/benchmark/oaf_corpus_run.py` already reconstructs HPA-325 `CohortItem` values from a persisted `run.json` and immutable prediction artifacts before final scoring.

The current reconstruction is private to HPA-326 finalization. HPA-327 should extract one narrow public helper that reconstructs a cohort from a validated run snapshot plus reference mappings. Both broad finalization and reviewed-subset rescoring use that helper.

This is the only HPA-326 refactor required by HPA-327.

## Fixed v1 selection policy

### Inputs

Candidate selection consumes only:

```text
--manifest          exact HPA-324 benchmark-reference manifest
--timing-manifest   exact HPA-323 timing manifest referenced by HPA-324
--output-file       editable review CSV to create
```

It explicitly does **not** accept:

- `run.json`;
- prediction directories;
- HPA-325 reports;
- F1 thresholds;
- model/backend IDs;
- include/exclude score filters.

The absence of model-result inputs is an intentional anti-bias contract.

### Frozen constants

Version 1 defines:

```python
REVIEW_SUBSET_REVISION = "hpa327-v1"
REVIEW_TARGET_COUNT = 30
REVIEW_MIN_COUNT = 20
REVIEW_MAX_COUNT = 40
REVIEW_SELECTION_SEED = "crux-hpa327-v1"
```

Do not expose CLI flags for seed or target count in v1. A future intentional selection-policy change produces a new revision instead of tuning v1 interactively.

### Eligible population

Only HPA-324 rows with `reference_eligibility_status == "eligible"` enter candidate selection.

For every eligible row, HPA-327 must reconstruct its native reference artifact and `ReferenceMappingResult` before selection. An eligible row whose artifact no longer validates is a fatal preflight inconsistency, matching HPA-326's fail-closed reference behavior.

If fewer than 20 eligible rows exist, candidate preparation fails. If there are 20–29 eligible rows, use all eligible rows. Otherwise select exactly 30.

### Manifest item hash

For each HPA-324 row, compute:

```python
source_row_sha256 = sha256(
    canonical_json_bytes(
        {key: value for key, value in source_row.items() if key != "corpus_version"}
    )
).hexdigest()
```

This is the exact candidate identity recorded in the review table. It is not a new persisted source artifact and does not replace the HPA-324 manifest SHA-256.

### Derived pre-inference features

Keep feature extraction deliberately narrow. For each eligible reference mapping derive:

```text
common_event_count
reference_event_span_sec
common_event_density_per_sec
common_class_count
has_timing_warning
uses_nonstandard_master_filename
```

Definitions:

- `common_event_count`: length of the common projected reference events;
- `reference_event_span_sec`: last common event time minus first common event time, or `0` for a single event;
- `common_event_density_per_sec`: `common_event_count / max(reference_event_span_sec, 1.0)`;
- `common_class_count`: number of distinct common classes present;
- `has_timing_warning`: true when the carried HPA-323 `timing_warnings` list is nonempty;
- `uses_nonstandard_master_filename`: true when the selected chart basename is `real.dtx` or `full.dtx`, case-insensitively.

Do not add genre lookup, audio embedding, separator analysis, BPM extraction, or acoustic/electronic classification to candidate generation. Those would widen dependencies for little selection value. Musical character is recorded manually during review instead.

### Balanced strata

Derive two rank-based bands over the eligible population:

- density band: `low | medium | high` from deterministic thirds of `(density, simfile_id)` ordering;
- class-richness band: `low | medium | high` from deterministic thirds of `(common_class_count, simfile_id)` ordering.

The selection stratum is:

```text
(density_band, class_richness_band, has_timing_warning, uses_nonstandard_master_filename)
```

Within each nonempty stratum, order candidates by:

```python
sha256(f"{REVIEW_SELECTION_SEED}:{source_row_sha256}".encode()).hexdigest()
```

Then iterate strata in lexicographic order and take one candidate from each nonempty stratum per round until the target count is reached.

This is not statistical sampling infrastructure. It is one small deterministic policy that prevents the fixed sample from collapsing around the most common easy-looking reference shape while preserving reproducibility.

### Candidate rank

`candidate_rank` is the 1-based order in which the round-robin selector accepts rows. It is persisted in the review CSV and remains unchanged after manual review.

## Editable review ledger

Candidate preparation writes one CSV with an exact header and one row per selected candidate. The first section is generated and must not be edited; the second section is completed manually.

### Generated identity and feature columns

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

Float-derived CSV values use the repository's existing canonical six-decimal token convention.

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

Closed values are:

```text
chart_selection_confirmed   true | false
audio_revision_confirmed    true | false
bgm_alignment_confirmed     true | false
technical_mapping_confirmed true | false
musical_fidelity            close | usable_with_limits | not_representative
drum_character              acoustic | electronic | hybrid | unknown
decision                    include | exclude
```

`reviewed_at` is an RFC3339 UTC timestamp.

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

`known_limitations` and `notes` are free text and may be empty. `other` requires a nonempty note.

### Manual review rule

Every candidate is reviewed before any model score is used for membership decisions.

An included row must have all four confirmation booleans true and `musical_fidelity` in `close | usable_with_limits`. A row with a failed source/timing/mapping confirmation or `not_representative` must be excluded with at least one reason code.

The review may record acoustic/electronic/hybrid character and limitations, but v1 does not automatically replace candidates to optimize those categories.

If exclusions leave fewer than 20 included rows or reveal unacceptable coverage before model scores are opened, generate a **new subset revision** with documented replacement rationale. Do not mutate `hpa327-v1` membership after inspecting benchmark scores.

## Finalization and immutable reviewed subset

Finalization consumes:

```text
--manifest          same HPA-324 manifest used for preparation
--timing-manifest   same HPA-323 manifest used for preparation
--review-file       completed candidate CSV
--output-dir        reviewed-subset publication root
```

Before publication, validate:

1. exact CSV header and unique candidate ranks/simfile IDs;
2. `subset_revision` and selection seed equal the frozen v1 constants;
3. candidate membership and order exactly reproduce deterministic preparation from the supplied manifests;
4. every generated identity/feature cell equals a freshly reconstructed value;
5. every row has a reviewer, timestamp, closed audit values, and decision;
6. included rows satisfy the confirmation/fidelity rule;
7. excluded rows contain at least one valid reason code;
8. `other` reason rows contain notes;
9. included count is 20–40;
10. every included source row is still HPA-324 eligible.

A partially filled or edited-identity CSV is not publishable. The operator fixes the CSV and reruns finalization.

### Manifest schema

Publish accepted rows only as canonical JSONL:

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

The `reason_codes` field is retained for included rows because `usable_with_limits` may legitimately carry documented chart simplifications or lane conventions even when the reference is still useful.

Use the existing `render_manifest()`, `publish_manifest()`, and `publish_latest_manifest()` primitives. `corpus_version` remains the content-derived publication identity; `subset_revision` is the semantic review-policy revision.

The completed CSV is also copied byte-for-byte into the publication directory as the human review ledger. Publication does not rewrite the operator's review text.

No database or mutable subset registry is added.

## Reviewed-subset scoring from persisted OaF artifacts

### Required HPA-326 extraction

Refactor the current HPA-326 finalization path to expose one public helper with this responsibility:

```python
def build_oaf_cohort_from_snapshot(
    snapshot: Mapping[str, object],
    *,
    mappings: Mapping[int, ReferenceMappingResult],
    output_dir: Path,
) -> tuple[CohortIdentity, tuple[CohortItem, ...]]:
    ...
```

It performs the same identity reconstruction and per-run-row `CohortItem` adaptation currently performed inside `_finalize_scoring_and_outcome()`.

HPA-326 broad finalization calls the new helper and then `score_cohort()` exactly as today. No behavior or report schema changes are intended.

### Subset rescore command

Add a thin model-specific command for the existing OaF run rather than a generic multi-backend runner:

```text
crux benchmark score-oaf-reviewed-subset \
  --run RUN_JSON \
  --manifest HPA324_MANIFEST \
  --timing-manifest HPA323_MANIFEST \
  --subset-manifest HPA327_MANIFEST \
  --output-dir REPORT_DIR
```

The command:

1. parses and validates the persisted HPA-326 run snapshot;
2. loads HPA-324 and HPA-323 and reuses HPA-326 preflight reference reconstruction;
3. requires the run's reference/timing identity to match those manifests;
4. loads the HPA-327 subset and requires its source identities to match the same manifests;
5. reconstructs the full persisted `CohortItem` tuple without invoking `OafBackend`;
6. requires every subset simfile ID to exist in the run population;
7. filters items to exact subset membership in candidate-rank order;
8. gives the filtered items a distinct deterministic cohort ID derived from the parent run ID plus subset manifest hash;
9. calls existing `score_cohort()` and `write_cohort_reports()`;
10. writes only reviewed-subset reports to the supplied output directory.

The broad run reports remain untouched.

### Cohort identity

The subset score keeps the parent run's:

- reference manifest SHA-256;
- timing version;
- taxonomy/lane-map versions;
- backend/model/descriptor/model-lock identity;
- prediction-map version;
- input-view ID;
- scoring version.

Only `cohort_id` changes so broad and reviewed-subset reports cannot be mistaken for the same population.

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

## CLI shape

Add three thin commands under the existing `benchmark` group:

```text
prepare-reviewed-subset
finalize-reviewed-subset
score-oaf-reviewed-subset
```

Keep domain logic out of Click callbacks. The CLI owns paths, concise JSON outcome rendering, and the repository's existing `0 | 1 | 2` exit convention only.

Suggested outcome semantics:

- `prepare-reviewed-subset`: `0` when the candidate ledger is written; `2` for invalid lineage/population/publication failure;
- `finalize-reviewed-subset`: `0` when a valid 20–40-song manifest is published; `2` for incomplete/invalid review or publication failure;
- `score-oaf-reviewed-subset`: `0` when reports are written; `1` when the filtered persisted cohort contains item-level failed/skipped/quarantined rows; `2` for identity, artifact, or report-publication failure.

Do not add a special exit code for manual exclusions; exclusions are already represented by absence from the finalized subset plus the review ledger.

## Data flow

```text
HPA-324 manifest + HPA-323 timing/events
                |
                v
      deterministic feature extraction
                |
                v
      fixed 30-song candidate selector
                |
                v
        editable review CSV
                |
        manual source audit
                |
                v
       finalization + validation
                |
                v
  reviewed-subset manifest + review ledger
                |
                +--------------------------+
                                           |
HPA-326 run.json + prediction artifacts    |
                |                          |
                v                          v
       reconstruct full persisted OaF cohort
                |
          filter exact membership
                |
                v
       HPA-325 score_cohort + reports
```

Candidate preparation has no edge from the HPA-326 side of this diagram.

## Error handling

Prefer explicit correctness failures over recovery machinery.

### Fatal preparation/finalization errors

Return exit 2 for:

- noncanonical or mixed HPA-323/HPA-324 lineage;
- eligible reference artifact mismatch;
- fewer than 20 eligible rows;
- edited generated review identity/feature cells;
- missing/invalid review fields;
- duplicate candidate rows;
- included population outside 20–40;
- publication failure.

There is no retry engine or automatic candidate replacement.

### Manual exclusions

Manual exclusions are ordinary reviewed evidence, not runtime failures. They remain visible in the finalized CSV review ledger with reason codes.

### Subset score errors

A mismatched run/reference/subset identity, unreadable run snapshot, or missing subset member in the parent run is fatal exit 2.

Item-level parent-run failures remain item-level cohort states. The subset scorer reports them through HPA-325 and exits 1 rather than silently dropping them.

## Testing strategy

### Candidate selection tests

Cover:

- only HPA-324 eligible rows are considered;
- candidate preparation never accepts/reads a run or prediction input;
- exact target count is 30 for populations of at least 30;
- 20–29 eligible rows use the full eligible population;
- fewer than 20 fail;
- source-row SHA-256 is stable;
- input manifest row order does not change candidate membership/rank;
- rank-band assignment and stratum round-robin are deterministic;
- timing-warning and `real.dtx`/`full.dtx` exceptions reach distinct strata;
- generated CSV uses exact columns and canonical numeric tokens.

### Finalization tests

Cover:

- untouched generated fields plus completed valid reviews publish successfully;
- edited source hash, selected chart/audio identity, feature, candidate rank, seed, or membership is rejected;
- blank reviewer/timestamp/decision is rejected;
- invalid enum/reason code is rejected;
- included rows require all confirmations and acceptable fidelity;
- excluded rows require a reason;
- `other` requires notes;
- included count must remain 20–40;
- output JSONL round-trips canonically and carries exact HPA-323/HPA-324 lineage;
- completed review CSV is preserved byte-for-byte beside publication output.

### OaF rescore tests

Cover:

- extracted `build_oaf_cohort_from_snapshot()` preserves broad HPA-326 finalization results;
- reviewed-subset scoring never constructs or invokes `OafBackend`;
- subset membership is exact and ordered by candidate rank;
- parent-run failure/quarantine rows remain present when selected;
- parent run reports are unchanged;
- cohort ID changes only by parent run + subset identity;
- per-song/per-class/aggregate rows come from the existing HPA-325 scorer/report writer;
- mismatched subset/run/reference identity fails closed.

### CLI acceptance

Use synthetic canonical HPA-323/HPA-324 fixtures and a persisted OaF run fixture to exercise prepare -> finalize -> rescore without R2, Docker, TensorFlow, or network access.

The real manual review itself is not faked by automated tests.

## Operational acceptance

HPA-327 is not complete merely because the code can generate a review sheet.

For the real corpus:

1. run candidate preparation before inspecting OaF song-level scores for membership decisions;
2. commit or otherwise preserve the generated v1 candidate review CSV before review begins;
3. manually inspect all candidates against their selected chart and source audio;
4. record reviewer, timestamp, confirmations, fidelity, drum character, limitations, decision, reasons, and notes;
5. finalize and preserve the immutable reviewed-subset manifest plus completed ledger;
6. verify 20–40 included songs remain and the sample contains materially different density, class, timing-warning, filename-exception, and manually recorded musical-character conditions;
7. rescore the persisted OaF run on the exact subset and preserve the resulting reports;
8. leave the broad-corpus run and reports unchanged.

If the real HPA-323/HPA-324 artifacts are unavailable in the implementation environment, the code can land after synthetic acceptance, but HPA-327 remains In Progress until this operational review evidence is completed.

## Risks and mitigations

### Selection bias

**Risk:** a manually chosen diagnostic set drifts toward songs known to score well or badly.

**Mitigation:** candidate membership is produced from reference-only inputs with a frozen seed and no prediction/run argument.

### Overengineering stratification

**Risk:** the ticket turns into a metadata/clustering project.

**Mitigation:** v1 derives six small reference-only features and one fixed round-robin policy. Human musical character stays in the review ledger.

### Review drift

**Risk:** reviewers silently change candidate membership while inspecting scores.

**Mitigation:** generated identities/ranks are re-derived at finalization; replacements require a new subset revision. The completed ledger records all exclusions.

### Source corrections discovered during review

**Risk:** an authored/source problem tempts the reviewer to rewrite the baseline.

**Mitigation:** exclude/document the item in the review ledger, preserve the broad result, and create a new source/subset revision for any later correction.

### Duplicate scoring code

**Risk:** subset reporting diverges from broad scoring.

**Mitigation:** extract only persisted-cohort reconstruction from HPA-326, then call the existing HPA-325 scorer and report writer unchanged.

## Completion criteria

HPA-327 is complete when:

- deterministic reference-only candidate generation exists and is tested;
- the real v1 candidate set is frozen before score-informed membership changes;
- every real candidate has a completed manual review record;
- the finalized accepted subset contains 20–40 songs with explicit source identities and exclusions;
- the canonical reviewed-subset manifest and completed review ledger are preserved;
- the same persisted OaF predictions generate reviewed-subset HPA-325 reports without rerunning inference;
- broad-corpus artifacts remain unchanged;
- no model-result-aware selection, training, or chart repair is introduced.
