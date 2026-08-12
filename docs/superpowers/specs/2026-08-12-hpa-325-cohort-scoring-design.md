# HPA-325 Single-Cohort Scoring and Reports Design

## Context

HPA-325 is the next Crux benchmark task after the HPA-324 taxonomy/reference-eligibility foundation and HPA-423 reusable OaF backend landed on `main`.

The Linear issue contains one stale implementation assumption: `src/benchmark/scorer_input.py` is no longer a stub. HPA-423 already added `prediction_to_benchmark_events()` / `read_scorer_events()` and prediction artifact v2 persists `common_class`. HPA-324 also already provides `map_reference_events()` and `project_common_reference_events()` for reference-side common-class projection.

Therefore HPA-325 must not rebuild model mapping, prediction persistence, reference eligibility, inference, or the working onset matcher. Its remaining job is to turn one model/input-view cohort into deterministic song/class scores, cohort aggregates, coverage/population diagnostics, inspectable matched/FP/FN evidence, and reproducible JSON/CSV/Markdown outputs.

## Why HPA-325 Next

The foundation required by HPA-325 is now present:

- HPA-323 audio-relative native reference events are persisted.
- HPA-324 freezes the detailed/common taxonomy and model-independent reference eligibility.
- HPA-423 persists mapped OaF prediction artifact v2 and exposes the common-class scorer bridge.
- `src/benchmark/scoring.py` already owns onset matching, TP/FP/FN retention, timing error, and raw versus global-offset diagnostic scoring.

HPA-325 should land before HPA-326 broad-corpus execution so the OaF run has a stable scoring/report consumer rather than inventing metrics during or after the corpus run.

## Goals

1. Score one model + input-view cohort at 30 ms, 50 ms, and 100 ms.
2. Preserve raw and diagnostically aligned modes as separate results.
3. Produce per-song and per-class metrics without changing the working matcher.
4. Produce event-micro, song-macro, class-macro, and per-song F1 distribution aggregates.
5. Keep quarantined/failed/skipped population rows and grouped reasons visible instead of silently scoring only successes.
6. Report native/mapped/unmapped prediction coverage and reference mapping counts for every item where those artifacts exist.
7. Persist matched, false-positive, and false-negative diagnostics for later inspection while retaining both original and aligned prediction times.
8. Make identical inputs produce byte-identical canonical JSON/JSONL and stable CSV/Markdown content.
9. Keep model/reference/timing/mapping/scoring identities first-class in persisted score rows.
10. Expose a small in-memory handoff that HPA-326 can populate from its future run manifest without HPA-325 defining a runner/queue framework.

## Non-Goals

- OaF or other model inference.
- A prediction-run manifest writer; HPA-326 owns corpus execution/run persistence.
- Cross-model or cross-input paired comparisons; HPA-562 owns them.
- Threshold tuning or score-driven model configuration changes.
- Rewriting `score_events()`, `_match_class()`, or alignment search unless a new adversarial regression proves a correctness defect.
- Parquet, databases, dashboards, generic report plugins, or a metrics service.
- Backward compatibility for the existing small `reports.py` API; there are no production consumers to preserve.
- Combining broad-corpus, reviewed-subset, and pilot scopes in one result. Run one cohort per scope so each report keeps one unambiguous population.

## Approaches Considered

### A. Extend `reports.py` only

Keep all orchestration, aggregation, and serialization in the existing report file.

This minimizes file count but mixes scoring policy with rendering and would make the already simple report writer become the owner of cohort semantics. It is fast initially but harder to reuse from HPA-326 and later MuScriptor/IDM work.

### B. Pure cohort scorer + focused report writer — selected

Add one `cohort_scoring.py` module for scoring/aggregation and rewrite `reports.py` as deterministic rendering only. Reuse `scoring.py`, `reference_set.py`, and `scorer_input.py` unchanged except for a tiny reference-to-`BenchmarkEvent` adapter.

This keeps the architecture small while giving HPA-326 a clear seam: construct `CohortItem` values from its run/reference artifacts, call `score_cohort()`, then `write_cohort_reports()`.

### C. Generic benchmark pipeline/framework

Introduce typed stages, plugin registries, report backends, generic manifest loaders, and model adapters.

Rejected. There is one immediate scorer consumer and one known future model path. The existing backend/taxonomy seams are already sufficient. A general pipeline would add indirection without removing current work.

## Architecture

### 1. `scorer_input.py` remains the artifact projection boundary

Keep existing prediction projection:

```python
prediction_to_benchmark_events(artifact: PredictionArtifact) -> tuple[BenchmarkEvent, ...]
```

Add only the corresponding reference projection helper:

```python
reference_to_benchmark_events(
    simfile_id: str,
    common_events: tuple[CommonReferenceEvent, ...],
) -> tuple[BenchmarkEvent, ...]
```

The adapter uses `CommonReferenceEvent.canonical_audio_time` as the event time and `common_class` as `BenchmarkEvent.canonical_class`. It must not rerun DTX parsing or invent a second taxonomy map.

### 2. `cohort_scoring.py` owns scoring policy, coverage input, and aggregation

Use a small set of frozen dataclasses:

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

`CohortItem` is deliberately an in-memory boundary, not another persisted manifest schema. HPA-326 can later adapt its run rows plus the HPA-324 reference manifest/prediction artifact into this type.

For a successful item, prediction coverage counts are required and must reconcile with `prediction_events`: mapped count equals the number of scoreable prediction events and `native = mapped + unmapped`. For a failed/skipped/quarantined item, prediction counts may be `None` if no valid prediction artifact exists. Reference counts come from HPA-324 mapping/reference-manifest data rather than being rediscovered here.

Successful items are scored for every tolerance and both modes. Failed/skipped/quarantined rows stay in population accounting but do not fabricate zero model scores.

### 3. Per-class metrics reuse the same matcher

For each successful song/tolerance/mode, compute the whole-song `ScoreResult` once. Per-class metrics are derived from that result's matched/FP/FN event sets grouped by common class rather than rerunning a different matching algorithm.

This guarantees that whole-song totals equal the sum of per-class totals.

### 4. Aggregate definitions

For one tolerance and mode:

- **Event-micro P/R/F1:** sum TP/FP/FN across successful items, then calculate metrics once.
- **Song-macro F1:** arithmetic mean of per-song F1 across successful eligible items. HPA-324 quarantines references with no scored drum events, so a successful eligible song must have reference support. An empty prediction therefore has F1 `0.0`, not an omitted song.
- **Class-macro F1:** arithmetic mean across common classes having reference or prediction support in the successful cohort. Classes with zero total support on both sides are excluded.
- **Per-class aggregate:** summed TP/FP/FN/support per common class.
- **Per-song F1 distribution:** deterministic `min`, p10, p25, median, p75, p90, and `max` across successful song F1 values. Empty successful populations return `None` for all distribution values.
- **Population counts:** total input rows plus success/failed/skipped/quarantined counts and grouped reason counts.

If a cohort has zero successful items, micro/macro/distribution values are empty/`None`; population accounting still renders successfully.

### 5. Native/common coverage

Do not infer native coverage from scoreable `BenchmarkEvent` rows because the scorer intentionally drops unmapped native predictions. `CohortCoverage` carries the small amount of artifact-level accounting needed for reports.

For each item report:

- reference native/common/ignored/unmapped counts;
- prediction native/mapped/unmapped counts when a prediction artifact exists;
- prediction mapping coverage = mapped/native when native count is nonzero;
- native class counts from the prediction artifact, sorted by native class ID;
- common-class reference/prediction support from the existing per-class score rows.

This satisfies native-output/common-canonical coverage reporting without adding another persisted mapping artifact.

### 6. Event diagnostics

Persist one deterministic JSONL artifact, `event_diagnostics.jsonl`, containing matched/FP/FN event evidence needed for later inspection.

For aligned mode the current scorer returns offset-adjusted prediction copies. The diagnostic therefore records both:

```text
prediction_time_sec          # original persisted prediction time
scored_prediction_time_sec   # after the diagnostic global offset
```

For raw mode those values are identical. Recover the original aligned prediction time as `scored_prediction_time_sec - offset_sec`; do not mutate or repersist the prediction artifact.

Each diagnostic row includes:

```text
cohort/model/input/map/scoring identity
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

Do not duplicate full native prediction metadata here; the immutable prediction artifact remains the source of truth.

### 7. Reports

Rewrite `reports.py` around one public entry point:

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

`items.csv` is the explicit full-population ledger: one row per input item with status, failure reason, warnings, reference/prediction coverage, and native class counts. `per_song.csv` contains successful score rows for every tolerance/mode. `per_class.csv` contains song-level and cohort-level common-class rows.

`summary.json` is canonical JSON using existing `canonical_json_bytes()` and includes identity, all item rows, population, aggregate rows, and F1 distributions. JSONL diagnostics use the same canonical serializer. CSV uses explicit fixed field lists and stable row sorting. Markdown is human-readable but generated from the same result object, not independently recomputed.

Every CSV score row and diagnostic JSONL row repeats the load-bearing cohort/model/reference/input/mapping/scoring identity fields needed to trace that row without relying on a directory name.

No generic writer registry is introduced.

## Data Flow

```text
HPA-324 native reference artifact + reference manifest row
  -> map_reference_events()
  -> common reference events + reference counts
  -> reference_to_benchmark_events()

prediction artifact v2
  -> read_prediction_artifact()
  -> prediction_to_benchmark_events()
  -> native/mapped/unmapped counts + native class counts

HPA-326/future caller execution row
  -> CohortItem(status + events + coverage + warnings/failure)

CohortItem[] + CohortIdentity
  -> score_cohort()
  -> CohortScoreResult
  -> write_cohort_reports()
  -> canonical JSON / CSV / JSONL / Markdown
```

## Identity and Reproducibility

Every cohort result carries:

- HPA-324 reference manifest SHA-256;
- HPA-323 reference timing version;
- taxonomy version and DTX lane-map version;
- backend ID;
- checkpoint/model ID;
- model-lock SHA-256;
- backend descriptor SHA-256;
- prediction-map version;
- input-view ID;
- scoring version;
- tolerance set and scoring modes.

Define one code-owned `SCORING_VERSION = "crux.single-cohort-scoring/v1"` and fixed default tolerances `(30, 50, 100)` ms. A future semantic scoring change increments this version.

A mapping correction changes mapping identity upstream and can therefore rescore persisted artifacts without inference.

Broad corpus, reviewed subset, and later pilot scopes are separate `cohort_id` invocations using the same scorer. HPA-325 does not merge those populations into one unlabeled result.

## Error Handling

- Invalid persisted prediction artifacts fail at the existing artifact reader boundary; HPA-325 does not add permissive compatibility readers.
- A `success` item with no prediction event tuple is invalid input and raises `ValueError`.
- A non-success item with prediction events is invalid input; callers must classify one execution outcome consistently.
- An eligible success item with zero reference events is invalid because HPA-324 would quarantine that source.
- Successful prediction coverage must balance and match the scoreable prediction-event count.
- Missing confidence/velocity do not affect scoring because `BenchmarkEvent` scoring uses time + common class only.
- Warnings and model failure reasons are retained as report data; they do not mutate cohort membership.
- Report publication writes complete deterministic files; no best-effort partial report set is needed at this stage.

## Testing Strategy

### Pin the matcher before building aggregates

Add focused regressions for:

- simultaneous same-class hits;
- dense same-class hits (existing test retained);
- empty prediction song;
- empty class on one side;
- duplicate-after-common-collapse input;
- missing confidence/velocity metadata at the scorer adapter;
- large-offset raw/aligned separation (existing test retained).

If these pass with the current matcher, do not change `scoring.py`.

### Cohort scoring tests

Use small synthetic `BenchmarkEvent` tuples to prove:

- 30/50/100 ms × raw/aligned output matrix;
- whole-song totals reconcile with per-class totals;
- micro/macro definitions;
- p10/p25/median/p75/p90 song-F1 distribution values;
- unsupported common classes are excluded from class macro;
- coverage accounting balances;
- failed/skipped/quarantined rows remain in population counts;
- zero successful cohort renders metrics/distributions as `None` without losing failures.

### Artifact integration test

Round-trip a real-shaped HPA-324 reference event projection and OaF prediction artifact v2 through the existing readers/adapters, derive coverage directly from those artifacts, then score them without invoking a backend.

### Report determinism

Write the same `CohortScoreResult` twice into separate directories and assert all six file byte streams are identical. Reverse input item order and require the same bytes.

## Scope / Maintenance Notes

This design intentionally avoids a CLI and disk-level prediction-run loader in HPA-325. HPA-326 owns corpus execution and its run manifest. HPA-326 should adapt its run/reference/prediction rows into the small `CohortItem` boundary rather than HPA-325 guessing a not-yet-landed run schema.

That keeps HPA-325 independently testable now while preserving a clean, low-cost integration seam for HPA-326 and later model adapters.

## Acceptance Mapping

- Persisted predictions are consumed through the already-landed HPA-423 mapper; no model is invoked.
- 30/50/100 ms and raw/aligned modes are explicit result dimensions.
- Per-song, per-class, event-micro, song-macro, class-macro, and F1 distribution metrics are produced.
- Native/mapped/unmapped coverage and full item status/failure populations are persisted explicitly.
- Matched/FP/FN evidence is retained in `event_diagnostics.jsonl`, with original and scored prediction times separated for aligned mode.
- Adversarial matcher fixtures are pinned before aggregate work.
- Population, coverage, song, class, and aggregate counts are reconciled by construction and tests.
- Model/reference/timing/mapping/input/scoring identities are carried through persisted score rows.
- Mapping changes rescore persisted events because mapping/inference remain separate.
- Canonical JSON/JSONL plus fixed CSV ordering provide deterministic output.
