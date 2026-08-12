# HPA-325 Single-Cohort Scoring and Reports Design

## Context

HPA-325 is the next Crux benchmark task after the HPA-324 taxonomy/reference-eligibility foundation and HPA-423 reusable OaF backend landed on `main`.

The Linear issue contains one stale implementation assumption: `src/benchmark/scorer_input.py` is no longer a stub. HPA-423 already added `prediction_to_benchmark_events()` / `read_scorer_events()` and prediction artifact v2 persists `common_class`. HPA-324 also already provides `map_reference_events()` and `project_common_reference_events()` for reference-side common-class projection.

Therefore HPA-325 must not rebuild model mapping, prediction persistence, reference eligibility, inference, or the working onset matcher. Its remaining job is to turn one model/input-view cohort into deterministic song/class scores, cohort aggregates, coverage/population diagnostics, inspectable matched/FP/FN evidence, and reproducible JSON/CSV/Markdown outputs.

## Why HPA-325 Next

The foundation required by HPA-325 is present:

- HPA-323 persists audio-relative native reference events.
- HPA-324 freezes the detailed/common taxonomy and model-independent reference eligibility.
- HPA-423 persists mapped OaF prediction artifact v2 and exposes the common-class scorer bridge.
- `src/benchmark/scoring.py` already owns onset matching, TP/FP/FN retention, timing error, raw/aligned scoring, and the percentile rank convention used by existing score summaries.

HPA-325 should land before HPA-326 broad-corpus execution so the OaF run has a stable scoring/report consumer rather than inventing metrics and serialization rules during the expensive corpus run.

## Goals

1. Score one model + input-view cohort at 30 ms, 50 ms, and 100 ms.
2. Preserve raw and diagnostically aligned modes as separate results.
3. Produce per-song and per-class metrics without changing matcher/alignment behavior.
4. Produce event-micro, song-macro, class-macro, and per-song F1 distribution aggregates.
5. Keep quarantined/failed/skipped population rows and grouped reasons visible instead of silently scoring only successes.
6. Report native/mapped/unmapped prediction coverage and reference mapping counts where artifacts exist.
7. Persist matched, false-positive, and false-negative diagnostics while retaining original and aligned prediction times.
8. Make identical inputs produce byte-identical canonical JSON/JSONL and stable CSV/Markdown bytes.
9. Reuse the repository's existing SHA-256 validation, numeric quantization, and percentile conventions rather than defining parallel ones.
10. Expose a small in-memory handoff that HPA-326 can populate from its future run manifest without HPA-325 defining a runner/queue framework.

## Non-Goals

- OaF or other model inference.
- A prediction-run manifest writer; HPA-326 owns corpus execution/run persistence.
- Cross-model or cross-input paired comparisons; HPA-562 owns them.
- Threshold tuning or score-driven model configuration changes.
- Rewriting `score_events()`, `_match_class()`, or alignment search.
- Parquet, databases, dashboards, generic report plugins, or a metrics service.
- Backward compatibility for `ChartReport` / `write_reports`; only tests consume them.
- Combining broad-corpus, reviewed-subset, and pilot scopes in one result. Each is a separate cohort.
- Immutable publication for regenerated score reports; ordinary deterministic files are sufficient.

## Approaches Considered

### A. Extend `reports.py` only

Keep orchestration, aggregation, and serialization in the existing report file.

This minimizes file count but mixes scoring policy with rendering and makes the report writer own cohort semantics. It is harder to reuse from HPA-326 and later model work.

### B. Pure cohort scorer + focused report writer — selected

Add one `cohort_scoring.py` module for validation/scoring/aggregation and rewrite `reports.py` as deterministic rendering only. Reuse `scoring.py`, `reference_set.py`, and `scorer_input.py`, adding only the reference-side scorer adapter and promoting the already-existing percentile helper to a public name.

This keeps the architecture small while giving HPA-326 one clear seam: construct `CohortItem` values, call `score_cohort()`, then `write_cohort_reports()`.

### C. Generic benchmark pipeline/framework

Introduce typed stages, plugin registries, report backends, generic manifest loaders, and model adapters.

Rejected. There is one immediate scorer consumer and existing backend/taxonomy seams are already sufficient.

## Architecture

### 1. `scorer_input.py` remains the artifact projection boundary

Keep the existing prediction projection:

```python
prediction_to_benchmark_events(artifact: PredictionArtifact) -> tuple[BenchmarkEvent, ...]
```

Add the corresponding reference projection helper:

```python
reference_to_benchmark_events(
    simfile_id: str,
    common_events: tuple[CommonReferenceEvent, ...],
) -> tuple[BenchmarkEvent, ...]
```

The adapter uses `CommonReferenceEvent.canonical_audio_time` as event time and `common_class` as `BenchmarkEvent.canonical_class`. It does not go back through `map_dtx_events()` or chart-time conversion.

### 2. `cohort_scoring.py` owns scoring policy, coverage input, and aggregation

Use frozen dataclasses with closed `Literal` status/mode contracts:

```python
CohortExecutionStatus = Literal["success", "failed", "skipped", "quarantined"]
ScoreMode = Literal["raw", "aligned"]

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
    scoring_version: str

@dataclass(frozen=True)
class CohortCoverage:
    reference_native_event_count: int
    reference_common_event_count: int
    reference_ignored_event_count: int
    reference_unmapped_event_count: int
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
    failure_reason: str | None = None
```

`CohortItem` is an in-memory seam, not a persisted run manifest. HPA-326 later adapts its own run rows into this type.

For success rows, prediction coverage must reconcile with scoreable prediction events: `native = mapped + unmapped` and `mapped == len(prediction_events)`. Failed/skipped/quarantined rows remain in population accounting without fabricated scores.

### 3. Identity validation reuses `require_sha256()`

`CohortIdentity` validates `reference_manifest_sha256`, `model_lock_sha256`, and `backend_descriptor_sha256` through the existing `src.benchmark.backend_identity.require_sha256()` helper. If the cohort API should surface `ValueError`, catch `StrictJsonError` and raise `ValueError` with the same message.

Do not copy lowercase-hex validation into `cohort_scoring.py`.

Other identity strings remain explicit nonempty strings; HPA-325 does not invent another general identity framework.

### 4. Per-class metrics reuse one matcher result

For each successful song/tolerance/mode, compute the whole-song `ScoreResult` once. Derive per-class TP/FP/FN from that same result's matched/unmatched collections rather than rerunning a different matcher.

This guarantees whole-song totals equal the sum of per-class totals.

### 5. Aggregate definitions reuse the scorer percentile convention

For one tolerance and mode:

- **Event-micro P/R/F1:** sum TP/FP/FN across successful items, then calculate metrics once.
- **Song-macro F1:** arithmetic mean of successful song F1 values. An eligible song with empty predictions has F1 `0.0` and stays in the mean.
- **Class-macro F1:** arithmetic mean across common classes with reference or prediction support.
- **Per-class aggregate:** summed TP/FP/FN/support per common class.
- **Per-song F1 distribution:** min, p10, p25, median, p75, p90, max.
- **Population counts:** total/success/failed/skipped/quarantined plus grouped reason counts.

Promote the existing `scoring._percentile()` helper to public `scoring.percentile()` and use it for p10/p25/p75/p90. This is a name promotion of an existing numeric convention, not a matcher/alignment behavior change. Keep `statistics.median()` for the median.

Undefined P/R/F1 remain `None`; HPA-325 does not coalesce them to `0.0`.

### 6. Native/common coverage

Do not infer native coverage from `BenchmarkEvent`, because unmapped native predictions are intentionally absent from scorer input. `CohortCoverage` carries the small artifact-level accounting needed for reports:

- reference native/common/ignored/unmapped counts;
- prediction native/mapped/unmapped counts when a valid prediction artifact exists;
- native class counts sorted by native class ID;
- common-class support from song/class score rows.

No separate coverage artifact is introduced.

### 7. Event diagnostics stay small

Persist one deterministic `event_diagnostics.jsonl` containing only event-level evidence plus `cohort_id`:

```text
cohort_id
simfile_id
mode
tolerance_ms
outcome = matched | false_positive | false_negative
common_class
reference_time_sec (nullable)
prediction_time_sec (nullable)
scored_prediction_time_sec (nullable)
timing_error_sec (nullable)
```

Full cohort identity appears once in `summary.json`. `items.csv`, `per_song.csv`, and `per_class.csv` retain the full traceability prefix because they are durable cohort/score rows; event diagnostics join back through `cohort_id` and do not repeat eleven cohort-constant fields on every matched/FP/FN line.

For aligned mode, recover the original persisted prediction time as `scored_prediction_time_sec - offset_sec`. For raw mode both prediction times are identical.

Do not duplicate full native prediction metadata; the immutable prediction artifact remains the source of truth.

### 8. One numeric encoding contract for every report artifact

`canonical_json_bytes()` accepts repository `JsonValue`, which deliberately excludes Python `float`. Every float-derived metric, ratio, offset, timing error, and timestamp must therefore cross one report-number boundary before JSON/JSONL/CSV rendering.

Reuse `quantize_six()` from `src.benchmark.backend_identity`:

```python
def _report_decimal(value: float | None) -> Decimal | None:
    return None if value is None else quantize_six(value)


def _csv_decimal(value: float | None) -> str:
    decimal = _report_decimal(value)
    return "" if decimal is None else canonical_json_bytes(decimal).decode("ascii")
```

Rules:

- JSON/JSONL float-derived values become `Decimal` through `quantize_six()`; `None` remains JSON `null`.
- CSV uses the exact canonical decimal token from the same quantized `Decimal`; `None` becomes an empty cell.
- Counts remain integers.
- Strings/statuses remain strings.
- No `str(float)`, `repr(float)`, custom rounding, or `json.dumps()` numeric path is allowed.

This matches prediction artifact numeric semantics and prevents HPA-326 from inheriting a second float renderer.

### 9. Reports

Rewrite `reports.py` around:

```python
write_cohort_reports(result: CohortScoreResult, output_dir: Path) -> ReportArtifacts
```

Output layout:

```text
<output_dir>/
  summary.json
  items.csv
  per_song.csv
  per_class.csv
  event_diagnostics.jsonl
  summary.md
```

`summary.json` contains the full cohort identity once, population, item coverage/failures, aggregate rows, and F1 distributions. `items.csv`, `per_song.csv`, and `per_class.csv` repeat the traceability prefix. `event_diagnostics.jsonl` carries only `cohort_id` plus event evidence. Markdown is rendered from the same result object.

Use ordinary file writes for these regenerable reports; do not add `publish_immutable_file()`.

## Data Flow

```text
HPA-324 native reference artifact + reference manifest row
  -> map_reference_events()
  -> common reference events + mapping counts
  -> reference_to_benchmark_events()

prediction artifact v2
  -> read_prediction_artifact()
  -> prediction_to_benchmark_events()
  -> native/mapped/unmapped counts + native class counts

HPA-326/future execution row
  -> CohortItem(status + events + coverage + warnings/failure)

CohortItem[] + CohortIdentity
  -> score_cohort()
  -> CohortScoreResult
  -> write_cohort_reports()
  -> quantize_six numeric boundary
  -> canonical JSON / fixed CSV / canonical JSONL / Markdown
```

## Identity and Reproducibility

Every cohort result carries:

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

Define `SCORING_VERSION = "crux.single-cohort-scoring/v1"` and default tolerances `(30, 50, 100)` ms. A semantic scoring change increments the version.

A mapping correction changes mapping identity upstream and can rescore persisted artifacts without inference.

## Error Handling

- Invalid prediction artifacts fail at the existing artifact reader boundary; no compatibility readers.
- A `success` item requires a prediction tuple; an empty tuple is valid.
- A non-success item must not carry prediction events and must carry a failure reason.
- An eligible success item requires nonempty reference events.
- Successful prediction coverage must balance and match the scoreable prediction-event count.
- Invalid SHA-256 identities reuse `require_sha256()` and surface as cohort `ValueError`.
- Nonfinite report floats fail through `quantize_six()` rather than being rendered.
- Missing confidence/velocity do not affect onset/class scoring.
- Report writing is all-or-error at this stage; no best-effort partial report set.

## Testing Strategy

### Pin matcher behavior first

Add/retain regressions for simultaneous same-class hits, dense hits, empty prediction songs, classes present on one side, duplicate-after-common-collapse input, missing confidence/velocity metadata at scorer level, and large-offset raw/aligned separation.

Do not change matcher/alignment behavior if these pass.

### Cohort scoring tests

Prove the 30/50/100 ms × raw/aligned matrix, song/class reconciliation, micro/macro definitions, percentile values, coverage balancing, full population accounting, and zero-success `None` semantics.

### Helper reuse tests

- Invalid identity hashes fail through the shared SHA-256 contract.
- `scoring.percentile()` retains the existing upper-nearest-rank behavior after the private-to-public rename.

### Report encoding goldens

Pin at least one canonical JSONL line and one CSV numeric cell. A `0.5` float must render as the canonical token `0.5`, not a binary-float representation or a separate fixed-width decimal format. `None` renders JSON `null` and an empty CSV field.

### Artifact integration test

Round-trip HPA-324 reference projection and OaF prediction artifact v2, derive coverage from artifacts, score without invoking a backend, and write reports.

### Report determinism

Write the same result into separate directories and assert all six file byte streams are identical. Reverse input item order and require the same bytes.

## Scope / Maintenance Notes

HPA-325 intentionally avoids a CLI and disk-level prediction-run loader. HPA-326 owns corpus execution and its run manifest and should adapt those rows into `CohortItem`.

The selected design is intentionally smaller than the previous draft: reuse `require_sha256()` and the scorer percentile helper, centralize report numeric encoding, and keep event diagnostics joined by `cohort_id` rather than repeating cohort-constant identity fields.

## Acceptance Mapping

- Persisted predictions are consumed through the landed HPA-423 mapper; no model is invoked.
- 30/50/100 ms and raw/aligned modes are explicit result dimensions.
- Per-song, per-class, event-micro, song-macro, class-macro, and F1 distributions are produced.
- Native/mapped/unmapped coverage and full item status/failure populations are persisted explicitly.
- Matched/FP/FN diagnostics retain original/scored prediction times and join through `cohort_id`.
- Adversarial matcher fixtures are pinned before aggregate work.
- Population, coverage, song, class, and aggregate counts reconcile by construction and tests.
- Full identity is in `summary.json` and score/ledger CSV rows; diagnostic JSONL avoids redundant identity stamping.
- Mapping changes rescore persisted events because mapping/inference remain separate.
- All report floats use the same `quantize_six()` + canonical decimal encoding already used by prediction artifacts.
- Canonical JSON/JSONL plus fixed CSV ordering provide deterministic output.
