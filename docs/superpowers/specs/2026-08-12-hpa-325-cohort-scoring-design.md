# HPA-325 Single-Cohort Scoring and Reports Design

## Context

HPA-325 is the next Crux benchmark task after the HPA-323 reference clock, HPA-324 taxonomy/reference-eligibility foundation, and HPA-423 reusable OaF backend/prediction seam landed on `main`.

The Linear issue still describes `src/benchmark/scorer_input.py` as a stub, but current `main` already contains `prediction_to_benchmark_events()` / `read_scorer_events()` and prediction artifact v2 persists `common_class`. HPA-324 also already provides `map_reference_events()` and `project_common_reference_events()` for reference-side common projection.

HPA-325 therefore does **not** rebuild model mapping, prediction persistence, reference eligibility, inference, or onset matching. Its job is to turn one model/input-view cohort of persisted artifacts into deterministic song/class scores, cohort aggregates, bounded diagnostic evidence, coverage/failure accounting, and reproducible reports.

## Why HPA-325 Next

The shared inputs it needs are now present:

- HPA-323 persists audio-relative native reference events.
- HPA-324 freezes the detailed/common taxonomy and model-independent reference eligibility.
- HPA-423 persists mapped OaF prediction artifact v2 and exposes the common-class scorer bridge.
- `src/benchmark/scoring.py` already owns onset matching, TP/FP/FN retention, timing error, and raw versus global-offset diagnostic scoring.

Landing the scoring/report contract before HPA-326 broad-corpus inference prevents the execution ticket from inventing metrics, identity rules, or output encodings during the expensive corpus run.

## Goals

1. Score one model + input-view cohort at 30 ms, 50 ms, and 100 ms.
2. Preserve raw and diagnostically aligned modes as separate named results.
3. Produce per-song and per-class metrics from the existing matcher without a second matching pass.
4. Produce event-micro, song-macro, class-macro, per-class aggregate, and deterministic song-F1 distribution summaries.
5. Keep the complete success/failed/skipped/quarantined population visible with stable grouped reason codes.
6. Report reference native/common/ignored/unmapped/duplicate-collapse counts and prediction native/mapped/unmapped coverage.
7. Detect mechanically mixed cohorts by reconciling prediction-event input-view and prediction-map identities against `CohortIdentity`.
8. Keep matched/FP/FN diagnostics available for targeted inspection without materializing corpus-wide diagnostics by default.
9. Make identical inputs produce byte-identical canonical JSON/JSONL and stable CSV/Markdown content.
10. Expose one small in-memory handoff that HPA-326 can populate from its future run manifest without HPA-325 defining that run schema.

## Non-Goals

- OaF or other model inference.
- A prediction-run manifest writer; HPA-326 owns corpus execution/run persistence.
- Cross-model or cross-input paired comparisons; HPA-562 owns them.
- Threshold tuning or score-driven model configuration changes.
- Rewriting `score_events()`, `_match_class()`, or alignment search unless an adversarial regression proves a correctness defect.
- Corpus-wide diagnostic-event retention by default.
- Parquet, databases, dashboards, generic report plugins, a metrics service, queues, or concurrency.
- `publish_immutable_file()` for regenerated score reports.
- Backward compatibility for the current internal `ChartReport` / `write_reports` API; only tests consume it.
- Combining broad-corpus, reviewed-subset, and pilot populations in one result. Each scope is a separate cohort.

## Approaches Considered

### A. Extend `reports.py` only

Put cohort validation, scoring policy, aggregation, and rendering into the existing report writer.

This minimizes file count but mixes domain scoring semantics with serialization and makes HPA-326 depend on a renderer as its integration boundary.

### B. Pure cohort scorer + focused report writer — selected

Add one `cohort_scoring.py` module for validation, bounded diagnostics, scoring, coverage, canonical result ordering, and aggregation. Keep `reports.py` as deterministic rendering only. Reuse `scoring.py`, `reference_set.py`, `scorer_input.py`, `require_sha256()`, `quantize_six()`, and canonical JSON rendering.

This is the smallest reusable seam HPA-326 needs: construct `CohortItem` values, call `score_cohort()`, then call `write_cohort_reports()`.

### C. Generic benchmark pipeline/framework

Introduce stage registries, report backends, generic manifest loaders, model plugins, or a metrics store.

Rejected. Current domain seams already cover model mapping, reference projection, and matching. A framework would add indirection without removing work.

## Architecture

### 1. `scorer_input.py` remains the artifact projection boundary

Keep the existing prediction projection unchanged:

```python
prediction_to_benchmark_events(
    artifact: PredictionArtifact,
) -> tuple[BenchmarkEvent, ...]
```

Add the symmetric reference adapter:

```python
reference_to_benchmark_events(
    simfile_id: str,
    common_events: tuple[CommonReferenceEvent, ...],
) -> tuple[BenchmarkEvent, ...]
```

`CommonReferenceEvent.canonical_audio_time` is a `Decimal`, while `BenchmarkEvent.time_sec` and the existing scorer operate on `float` (including float offsets and NumPy histograms). The adapter intentionally converts with `float(event.canonical_audio_time)` and uses `common_class` as `BenchmarkEvent.canonical_class`.

It must not rerun DTX parsing, timing, `map_dtx_events()`, or any taxonomy mapping.

### 2. `cohort_scoring.py` owns one closed in-memory cohort contract

Use frozen dataclasses and closed literals:

```python
CohortExecutionStatus = Literal[
    "success",
    "failed",
    "skipped",
    "quarantined",
]

CohortFailureReason = Literal[
    "reference_quarantined",
    "backend_unavailable",
    "inference_failed",
    "prediction_artifact_invalid",
    "prediction_missing",
    "explicitly_skipped",
]

COHORT_FAILURE_REASONS = frozenset(get_args(CohortFailureReason))
ScoreMode = Literal["raw", "aligned"]
```

These are scorer-facing stable categories, not HPA-326's future detailed runtime error taxonomy. HPA-326 may retain richer failure detail in its own run manifest while adapting each item to one stable HPA-325 reason code.

A resume hit with a valid prediction artifact is adapted as `success`, not `skipped`: HPA-325 scores available artifacts rather than execution attempts. `skipped` is reserved for an explicitly excluded row with no prediction artifact in this cohort.

```python
@dataclass(frozen=True)
class CohortIdentity:
    cohort_id: str
    reference_manifest_sha256: str
    reference_timing_version: str
    taxonomy_version: str
    lane_map_version: str
    backend_id: str
    model_id: str
    model_lock_sha256: str
    backend_descriptor_sha256: str
    prediction_map_version: str
    input_view_id: str
    scoring_version: str = SCORING_VERSION

@dataclass(frozen=True)
class CohortCoverage:
    reference_native_event_count: int
    reference_common_event_count: int
    reference_ignored_event_count: int
    reference_unmapped_event_count: int
    reference_duplicate_collapsed_count: int
    prediction_native_event_count: int | None
    prediction_mapped_event_count: int | None
    prediction_unmapped_event_count: int | None
    prediction_native_class_counts: tuple[tuple[str, int], ...] = ()

@dataclass(frozen=True)
class CohortItem:
    simfile_id: str
    status: CohortExecutionStatus
    reference_events: tuple[BenchmarkEvent, ...]
    prediction_events: tuple[BenchmarkEvent, ...] | None
    coverage: CohortCoverage
    warnings: tuple[str, ...] = ()
    failure_reason: CohortFailureReason | None = None
```

`CohortItem` stays in memory. HPA-325 does not persist another run manifest.

### 3. Identity validation reuses existing contracts and rejects mixed prediction cohorts

`CohortIdentity` calls `require_sha256()` directly for its SHA-256 fields. `StrictJsonError` already subclasses `ValueError`, so no wrapper or second lowercase-hex validator is added.

For each successful item, every prediction `BenchmarkEvent` must carry:

```python
event.metadata["input_view_id"] == identity.input_view_id
event.metadata["prediction_map_version"] == identity.prediction_map_version
```

Missing or mismatched values fail cohort validation before scoring. This prevents a report labelled as one full-mix/map cohort from silently scoring stem events or events normalized under another map version.

The existing persisted prediction adapter already stamps those metadata fields; HPA-325 only reconciles them.

### 4. Coverage balances on both sides

`coverage_from_artifacts()` consumes the HPA-324 `ReferenceMappingResult` and optional `PredictionArtifact`.

Reference coverage carries `ReferenceMappingDiagnostics.duplicate_common_event_count` as `reference_duplicate_collapsed_count` and validates:

```text
reference_native_event_count
  = reference_common_event_count
  + reference_duplicate_collapsed_count
  + reference_ignored_event_count
  + reference_unmapped_event_count
```

Prediction coverage validates:

```text
prediction_native_event_count
  = prediction_mapped_event_count
  + prediction_unmapped_event_count
```

For successful rows, `prediction_mapped_event_count == len(prediction_events)`.

Prediction mapping coverage is `mapped/native` only when native count is greater than zero. A `0/0` artifact has **undefined** coverage and renders `None` / an empty CSV cell, not `1.0`.

### 5. Per-class metrics reuse exactly one matcher result

For each successful song/tolerance, call `score_events_with_alignment()` once. Its `raw` and `aligned` `ScoreResult`s become separate `SongScore` rows.

Per-class rows are derived from that same `ScoreResult`'s matches, unmatched references, and unmatched predictions. Because `scoring.py` already partitions events by class before matching, class TP/FP/FN sums reconcile with the song totals by construction.

Do not invoke a second matcher for per-class reports.

### 6. Canonical ordering is established once in `score_cohort()`

The scorer returns deterministic tuple order independent of caller item order:

```text
items:          simfile_id
song_scores:    simfile_id, tolerance_ms, mode-order(raw, aligned)
per_class:      common_class
aggregates:     tolerance_ms, mode-order(raw, aligned)
diagnostics:    simfile_id, tolerance_ms, mode-order, outcome, class, times
```

Writers consume result tuples in that canonical order instead of inventing independent sort rules.

### 7. Event diagnostics are opt-in and bounded

The original plan materialized every matched/FP/FN event for every song × tolerance × mode. That scales with the full event population and is unnecessary for aggregates.

Change the public scorer seam to:

```python
def score_cohort(
    identity: CohortIdentity,
    items: tuple[CohortItem, ...],
    tolerances_ms: tuple[int, ...] = DEFAULT_TOLERANCES_MS,
    diagnostics_for: tuple[str, ...] = (),
) -> CohortScoreResult: ...
```

`diagnostics_for=()` means no event diagnostics. Requested IDs must be unique item IDs in the supplied cohort. Only those songs materialize `EventDiagnostic` rows.

This keeps the broad HPA-326 run cheap while preserving the exact inspection artifact for explicitly selected outliers or acceptance fixtures.

For aligned diagnostics, retain both:

```text
prediction_time_sec          # original persisted time
scored_prediction_time_sec   # offset-adjusted time used by matcher
```

Recover the original aligned prediction time as:

```python
original_prediction_time = scored_prediction_time - score_result.summary.offset_sec
```

### 8. Aggregate definitions reuse existing percentile semantics

For each tolerance/mode:

- **Event micro:** sum TP/FP/FN across successful songs, then compute P/R/F1 once.
- **Song macro F1:** arithmetic mean of successful song F1 values.
- **Class macro F1:** arithmetic mean across supported aggregate common classes.
- **Per-class aggregate:** summed TP/FP/FN and support by class.
- **Song F1 distribution:** min, p10, p25, median, p75, p90, max.
- **Population:** total/success/failed/skipped/quarantined counts and grouped stable `CohortFailureReason` counts.

Promote the existing private `scoring._percentile()` to public `scoring.percentile()` without changing its upper-nearest-rank behavior. `score_events()` continues using the same function for p95 timing error, and HPA-325 reuses it for p10/p25/p75/p90. Use `statistics.median()` only for the median.

A zero-success cohort retains population data; P/R/F1 and macro/distribution values remain undefined (`None`).

## Report Model

### 9. Give each fact one persisted home

Keep six artifacts because the item failure/warning ledger remains a distinct row-level requirement, but remove duplicated data across them:

```text
<output_dir>/
  summary.json
  items.csv
  per_song.csv
  per_class.csv
  event_diagnostics.jsonl
  summary.md
```

#### `summary.json`

Cohort-level facts only:

```text
schema
identity
tolerances_ms
population
aggregates
```

Each aggregate contains micro/macro metrics, F1 distribution, successful-song count, and cohort per-class aggregate rows.

It does **not** duplicate item ledger rows.

#### `items.csv`

The sole complete item ledger:

```text
cohort_id
simfile_id
status
failure_reason
warnings
reference_native_event_count
reference_common_event_count
reference_ignored_event_count
reference_unmapped_event_count
reference_duplicate_collapsed_count
prediction_native_event_count
prediction_mapped_event_count
prediction_unmapped_event_count
prediction_mapping_coverage
prediction_native_class_counts
```

Full cohort identity is joined through `cohort_id` and lives in `summary.json`.

#### `per_song.csv`

Successful score rows only. HPA-325 explicitly requires `model_id`, model-lock hash, prediction-map version, and `input_view_id` as first-class dimensions in every score row, so keep that required subset rather than either the old 12-field prefix or `cohort_id` alone:

```text
cohort_id
model_id
model_lock_sha256
prediction_map_version
input_view_id
scoring_version
simfile_id
tolerance_ms
mode
tp
fp
fn
precision
recall
f1
prediction_to_reference_ratio
median_abs_error_ms
p95_abs_error_ms
offset_ms
warnings
```

#### `per_class.csv`

Song-scope class rows only; cohort class aggregates already live in `summary.json`:

```text
cohort_id
model_id
model_lock_sha256
prediction_map_version
input_view_id
scoring_version
simfile_id
tolerance_ms
mode
common_class
tp
fp
fn
reference_support
prediction_support
precision
recall
f1
```

No `scope` column is needed.

#### `event_diagnostics.jsonl`

Only requested `diagnostics_for` songs, keyed by `cohort_id`:

```text
cohort_id
simfile_id
tolerance_ms
mode
outcome
common_class
reference_time_sec
prediction_time_sec
scored_prediction_time_sec
timing_error_sec
```

No full identity prefix or native prediction payload duplication.

#### `summary.md`

Human-readable rendering from the same `CohortScoreResult`, with identity, population, aggregates, class aggregates, F1 distribution, and raw-50ms song extremes.

### 10. Numeric encoding uses the existing prediction-artifact contract

`canonical_json_bytes()` accepts `Decimal`, not Python `float`. Every float-derived report value crosses one boundary using existing `quantize_six()`:

```python
def _report_decimal(value: float | None) -> Decimal | None:
    return None if value is None else quantize_six(value)


def _csv_decimal(value: float | None) -> str:
    decimal = _report_decimal(value)
    return "" if decimal is None else canonical_json_bytes(decimal).decode("ascii")
```

Rules:

- JSON/JSONL metrics, ratios, offsets, distributions, and times use quantized `Decimal`.
- CSV/Markdown use the exact canonical decimal token derived from the same `Decimal`.
- Undefined values remain JSON `null` / empty CSV cells.
- Counts remain integers.
- Never pass Python `float` to `canonical_json_bytes()`.
- Never encode report numbers with `str(float)` or `repr(float)`.
- Nonfinite floats fail loudly through `quantize_six()`.

A golden JSONL line and CSV `0.5` cells pin the encoding.

## Data Flow

```text
HPA-324 native reference artifact
  -> map_reference_events()
  -> ReferenceMappingResult
  -> common reference events + balanced coverage
  -> reference_to_benchmark_events()

prediction artifact v2
  -> read_prediction_artifact()
  -> prediction_to_benchmark_events()
  -> native/mapped/unmapped coverage

HPA-326/future execution row
  -> CohortItem(status + stable reason + events + coverage)

CohortIdentity + CohortItem[]
  -> validation (hashes + coverage + per-event cohort identity)
  -> canonical ordering
  -> score_cohort(..., diagnostics_for=selected_ids)
  -> CohortScoreResult
  -> write_cohort_reports()
  -> summary.json / row CSVs / bounded JSONL / Markdown
```

## Identity and Reproducibility

Every `CohortScoreResult` carries:

- HPA-324 reference manifest SHA-256;
- HPA-323 reference timing version;
- taxonomy and DTX lane-map versions;
- backend ID;
- checkpoint/model ID;
- model-lock SHA-256;
- backend descriptor SHA-256;
- prediction-map version;
- input-view ID;
- scoring version;
- tolerance set and scoring modes.

Define:

```python
SCORING_VERSION = "crux.single-cohort-scoring/v1"
DEFAULT_TOLERANCES_MS = (30, 50, 100)
```

A mapping correction changes upstream mapping identity and can rescore persisted events without inference.

## Error Handling

- Invalid prediction artifacts fail at their existing reader boundary; HPA-325 adds no compatibility reader.
- `require_sha256()` validates cohort hashes directly.
- A success item requires nonempty reference events and a prediction tuple; the prediction tuple may be empty.
- Successful prediction metadata must match cohort input-view and prediction-map identities.
- A non-success item has no scoreable prediction tuple and carries one closed `CohortFailureReason` appropriate to its status.
- Reference and prediction coverage must balance before scoring.
- Missing confidence/velocity metadata on `BenchmarkEvent` does not affect onset/class scoring.
- Warnings remain warnings and do not alter membership or grouped reason counts.
- Undefined P/R/F1 and 0/0 mapping coverage remain `None`.
- Regenerated reports use ordinary deterministic writes; no immutable publication layer is required.

## Risks / Hard Gates

### Gate A — Diagnostic volume

Broad-corpus scoring must call `score_cohort()` with the default `diagnostics_for=()`. Event diagnostics are generated only for explicitly selected song IDs. A plan or implementation that materializes all event diagnostics by default does not satisfy HPA-325.

### Gate B — Mixed cohort identity

Every successful prediction event must match the cohort's `input_view_id` and `prediction_map_version`. A mismatch fails before scoring rather than producing confidently mislabeled reports.

### Gate C — Numeric serialization

No Python float may reach `canonical_json_bytes()`. The JSONL/CSV encoding goldens are required before the report writer is considered complete.

### Gate D — CI-equivalent lint scope

Final implementation verification must lint all of `src`, including the new `src/benchmark/cohort_scoring.py`, using the same errors-only Pylint scope as CI. `pylint src/app src/cli` alone is insufficient.

## Testing Strategy

### Matcher and adapter pins

Add/retain focused tests for:

- simultaneous same-class hits;
- dense same-class hits;
- empty prediction song;
- class present on only one side;
- exact duplicate-after-common-collapse behavior;
- missing confidence/velocity metadata at scorer level;
- large-offset raw/aligned separation;
- `Decimal` common-reference time converted to float in the reference adapter.

Do not change matcher semantics if these pass.

### Cohort contract tests

Prove:

- closed failure-reason invariant;
- direct reuse of `require_sha256()` behavior;
- reference/prediction coverage balance equations;
- 0/0 mapping coverage is undefined;
- prediction event `input_view_id` / `prediction_map_version` reconciliation;
- invalid diagnostics IDs are rejected;
- default diagnostics are empty;
- canonical result ordering is independent of input item order.

### Cohort scoring tests

Prove:

- 30/50/100 ms × raw/aligned matrix;
- whole-song totals reconcile with per-class totals;
- event micro, song macro, class macro;
- public reused percentile semantics for p10/p25/p75/p90;
- zero-success behavior;
- diagnostics are generated only for requested songs.

### Artifact acceptance

Round-trip HPA-324-shaped references and an OaF prediction artifact v2 through existing adapters, derive balanced coverage, request diagnostics for that one fixture ID, score without invoking a backend, and write all reports.

### Report determinism

Pin:

- exact canonical JSONL numeric line;
- exact CSV `0.5` cell;
- JSON null / empty CSV for undefined values including 0/0 coverage;
- byte-identical outputs across two directories;
- byte-identical outputs when input item order is reversed.

## Scope / Maintenance Notes

The design intentionally avoids a CLI and disk-level run loader. HPA-326 owns execution and its run manifest, then adapts those rows into `CohortItem`.

The result is deliberately small: one scorer module, one report writer, one reference adapter, and one safe public-name promotion for the existing percentile helper.

## Acceptance Mapping

- Persisted predictions are consumed through HPA-423's existing mapper; no model is invoked.
- 30/50/100 ms and raw/aligned modes are explicit.
- Per-song, per-class, event-micro, song-macro, class-macro, and F1-distribution metrics are present.
- Full population counts use stable closed reason codes.
- Reference and prediction coverage reconcile explicitly, including duplicate common collapse.
- Mixed input-view/map cohorts are rejected before scoring.
- Matched/FP/FN evidence remains available through bounded, opt-in diagnostics.
- Model ID, model-lock hash, prediction-map version, and input-view ID remain first-class on score rows without repeating every cohort hash everywhere.
- Mapping changes rescore persisted artifacts without inference.
- Canonical numeric encoding and result ordering make report bytes deterministic.
- No runner, CLI, run manifest, queue, plugin registry, database, Parquet layer, or immutable report publisher is introduced.
