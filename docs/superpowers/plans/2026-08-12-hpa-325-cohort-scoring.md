# HPA-325 Single-Cohort Scoring and Reports Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn one frozen model/input-view cohort of persisted reference and prediction events into deterministic per-song, per-class, aggregate, and event-diagnostic benchmark reports at 30/50/100 ms without invoking inference or rebuilding the existing matcher/mapping seams.

**Architecture:** Keep `src/benchmark/scoring.py` as the sole onset-matching/alignment implementation. Add one small `cohort_scoring.py` module that validates in-memory cohort rows, invokes the existing scorer, derives per-class and cohort aggregates from the same `ScoreResult`, and retains population failures. Extend `scorer_input.py` only with the reference-side common-event adapter. Rewrite the small existing `reports.py` API around deterministic cohort rendering; HPA-326 will later adapt its run-manifest rows into `CohortItem` rather than HPA-325 inventing that future schema.

**Tech Stack:** Python 3.12, frozen dataclasses, `typing.Literal`, `statistics.fmean`, existing `BenchmarkEvent`/`ScoreSummary`/`ScoreResult`, existing canonical JSON helpers, CSV, pathlib, pytest, Ruff, Pylint.

## Global Constraints

- Consume the merged HPA-323 audio-relative native reference artifacts; do not rerun DTX parsing, timing, BGM resolution, chart selection, R2 access, or audio decoding.
- Consume HPA-324 `map_reference_events()` / `CommonReferenceEvent` and the frozen common taxonomy; do not define another lane or class map.
- Consume HPA-423 prediction artifact v2 and `prediction_to_benchmark_events()`; do not rebuild `NativeEvent` mapping or prediction persistence.
- The scored identity is one model/checkpoint + one `input_view_id` cohort at a time.
- Default tolerances are exactly `30`, `50`, and `100` milliseconds.
- Raw and diagnostically aligned scores are separate named modes; alignment never replaces the raw result.
- Headline matching uses onset time + common class only. Confidence and velocity remain diagnostic metadata and never affect matching.
- Failed, skipped, and quarantined items remain visible in population accounting; never silently reduce the cohort to successes.
- Do not change `score_events()`, `_match_class()`, or alignment code unless a focused adversarial test first proves incorrect behavior.
- Do not add inference, worker pools, queues, retries, a prediction-run manifest, cross-model comparison, Parquet, a database, a metrics service, a report plugin registry, or a generic benchmark pipeline.
- Breaking the old internal `reports.py` API is allowed; do not add a compatibility wrapper for `ChartReport` / `write_reports`.
- Every persisted JSON or JSONL output uses existing canonical JSON serialization. CSV rows and Markdown sections have fixed deterministic ordering.
- Repository verification follows `CLAUDE.md`: `uv run pytest`, `uv run ruff check src tests`, `uv run ruff format --check src tests`, and `uv run pylint src/app src/cli`.

---

## File Map

### Create

- `src/benchmark/cohort_scoring.py` — single-cohort validation, per-song scoring, per-class derivation, aggregates, and population accounting.
- `tests/benchmark/test_scorer_input.py` — focused reference-side scorer adapter tests; existing HPA-423 prediction bridge coverage remains in `test_task_d_contract.py`.
- `tests/benchmark/test_cohort_scoring.py` — scoring matrix, aggregation, and reconciliation contract.
- `tests/benchmark/test_cohort_scoring_acceptance.py` — persisted reference/prediction artifact to reports integration without inference.

### Modify

- `src/benchmark/scorer_input.py` — add only `reference_to_benchmark_events()`.
- `src/benchmark/reports.py` — replace the flat chart writer with deterministic cohort report rendering.
- `tests/benchmark/test_scoring.py` — pin remaining adversarial matcher behavior before relying on it for aggregate reports.
- `tests/benchmark/test_reports.py` — replace legacy `ChartReport` tests with cohort output and byte-determinism tests.

### Explicitly unchanged

- `src/benchmark/scoring.py`
- `src/benchmark/models.py`
- `src/benchmark/reference_set.py`
- `src/benchmark/reference_set_manifest.py`
- `src/benchmark/prediction_artifact.py`
- `src/benchmark/mapping.py`
- `src/benchmark/backends/oaf.py`
- `src/cli/benchmark.py`

---

## Task 1: Pin the existing matcher and add the reference scorer adapter

**Files:**
- Modify: `tests/benchmark/test_scoring.py`
- Modify: `src/benchmark/scorer_input.py`
- Create: `tests/benchmark/test_scorer_input.py`

**Interfaces:**
- Consumes: `CommonReferenceEvent` from `src.benchmark.reference_set`; existing `BenchmarkEvent` and prediction adapters.
- Produces:

```python
def reference_to_benchmark_events(
    simfile_id: str,
    common_events: tuple[CommonReferenceEvent, ...],
) -> tuple[BenchmarkEvent, ...]: ...
```

- [ ] **Step 1: Add simultaneous-hit and empty-song matcher regressions**

Append focused tests to `tests/benchmark/test_scoring.py`:

```python
def test_simultaneous_same_class_hits_match_one_to_one() -> None:
    result = score_events(
        [event(1.0, "kick", "gt_a"), event(1.0, "kick", "gt_b")],
        [event(1.0, "kick", "pred_a"), event(1.0, "kick", "pred_b")],
        0.03,
    )

    assert result.summary.true_positives == 2
    assert result.summary.false_positives == 0
    assert result.summary.false_negatives == 0


def test_empty_prediction_song_counts_every_reference_as_false_negative() -> None:
    result = score_events(
        [event(1.0, "kick", "gt"), event(2.0, "snare", "gt")],
        [],
        0.05,
    )

    assert result.summary.true_positives == 0
    assert result.summary.false_positives == 0
    assert result.summary.false_negatives == 2
    assert result.summary.f1 == 0.0


def test_class_present_on_only_one_side_is_retained() -> None:
    result = score_events(
        [event(1.0, "kick", "gt")],
        [event(1.0, "snare", "pred")],
        0.05,
    )

    assert [item.canonical_class for item in result.unmatched_ground_truth] == ["kick"]
    assert [item.canonical_class for item in result.unmatched_predictions] == ["snare"]
```

Keep the existing dense-hit and large-offset tests. These fixtures establish the matcher behavior HPA-325 aggregates must preserve.

- [ ] **Step 2: Run matcher regressions before touching scoring code**

Run:

```bash
uv run pytest tests/benchmark/test_scoring.py -q
```

Expected: PASS. If one of the new fixtures fails, stop and review the matcher as a correctness bug before proceeding; do not hide it in aggregate code.

- [ ] **Step 3: Write the failing reference adapter tests**

Create `tests/benchmark/test_scorer_input.py`:

```python
from decimal import Decimal

from src.benchmark.reference_set import CommonReferenceEvent
from src.benchmark.scorer_input import reference_to_benchmark_events


def test_reference_common_events_become_ground_truth_benchmark_events() -> None:
    common = (
        CommonReferenceEvent(Decimal("0.500000"), "kick", ()),
        CommonReferenceEvent(Decimal("1.250000"), "hihat", ()),
    )

    result = reference_to_benchmark_events("42", common)

    assert [(event.chart_id, event.time_sec, event.canonical_class, event.source) for event in result] == [
        ("42", 0.5, "kick", "ground_truth"),
        ("42", 1.25, "hihat", "ground_truth"),
    ]


def test_reference_adapter_keeps_common_projection_order() -> None:
    common = (
        CommonReferenceEvent(Decimal("1.000000"), "kick", ()),
        CommonReferenceEvent(Decimal("1.000000"), "snare", ()),
    )

    result = reference_to_benchmark_events("song", common)

    assert [(event.time_sec, event.canonical_class) for event in result] == [
        (1.0, "kick"),
        (1.0, "snare"),
    ]
```

- [ ] **Step 4: Run the adapter tests to verify RED**

Run:

```bash
uv run pytest tests/benchmark/test_scorer_input.py -q
```

Expected: FAIL because `reference_to_benchmark_events` does not exist yet.

- [ ] **Step 5: Implement the minimal reference adapter**

In `src/benchmark/scorer_input.py`, import `CommonReferenceEvent` and add:

```python
def reference_to_benchmark_events(
    simfile_id: str,
    common_events: tuple[CommonReferenceEvent, ...],
) -> tuple[BenchmarkEvent, ...]:
    if not isinstance(simfile_id, str) or not simfile_id:
        raise ValueError("simfile_id must be a nonempty string")
    return tuple(
        BenchmarkEvent(
            chart_id=simfile_id,
            time_sec=float(event.canonical_audio_time),
            canonical_class=event.common_class,
            source="ground_truth",
            metadata={},
        )
        for event in common_events
    )
```

Export it in `__all__` beside the existing prediction functions. Do not accept native reference events here; callers must first use HPA-324's common projection.

- [ ] **Step 6: Run focused scorer-input and HPA-423 bridge tests**

Run:

```bash
uv run pytest tests/benchmark/test_scorer_input.py \
  tests/benchmark/test_task_d_contract.py -q
```

Expected: PASS. The existing OaF artifact-to-common-class bridge must remain unchanged.

- [ ] **Step 7: Commit the stable scorer input boundary**

```bash
git add src/benchmark/scorer_input.py tests/benchmark/test_scorer_input.py \
  tests/benchmark/test_scoring.py
git commit -m "test: pin cohort scorer inputs"
```

---

## Task 2: Add the single-cohort scoring model and result matrix

**Files:**
- Create: `src/benchmark/cohort_scoring.py`
- Create: `tests/benchmark/test_cohort_scoring.py`

**Interfaces:**
- Consumes: `BenchmarkEvent`, `ScoreSummary`, `ScoreResult`, and `score_events_with_alignment()`.
- Produces:

```python
SCORING_VERSION = "crux.single-cohort-scoring/v1"
DEFAULT_TOLERANCES_MS = (30, 50, 100)

CohortExecutionStatus = Literal["success", "failed", "skipped", "quarantined"]
ScoreMode = Literal["raw", "aligned"]

@dataclass(frozen=True)
class CohortIdentity:
    cohort_id: str
    reference_manifest_sha256: str
    backend_id: str
    model_id: str
    backend_descriptor_sha256: str
    prediction_map_version: str
    input_view_id: str
    scoring_version: str = SCORING_VERSION

@dataclass(frozen=True)
class CohortItem:
    simfile_id: str
    status: CohortExecutionStatus
    reference_events: tuple[BenchmarkEvent, ...]
    prediction_events: tuple[BenchmarkEvent, ...] | None
    warnings: tuple[str, ...] = ()
    failure_reason: str | None = None

@dataclass(frozen=True)
class ClassScore:
    common_class: str
    summary: ScoreSummary
    reference_support: int
    prediction_support: int

@dataclass(frozen=True)
class SongScore:
    simfile_id: str
    tolerance_ms: int
    mode: ScoreMode
    summary: ScoreSummary
    prediction_to_reference_ratio: float
    per_class: tuple[ClassScore, ...]
    warnings: tuple[str, ...]

@dataclass(frozen=True)
class EventDiagnostic:
    simfile_id: str
    tolerance_ms: int
    mode: ScoreMode
    outcome: Literal["matched", "false_positive", "false_negative"]
    common_class: str
    reference_time_sec: float | None
    prediction_time_sec: float | None
    timing_error_sec: float | None

@dataclass(frozen=True)
class CohortScoreResult:
    identity: CohortIdentity
    tolerances_ms: tuple[int, ...]
    song_scores: tuple[SongScore, ...]
    event_diagnostics: tuple[EventDiagnostic, ...]
    population: PopulationSummary
    aggregates: tuple[CohortAggregate, ...]
```

`PopulationSummary` and `CohortAggregate` are completed in Task 3, but define the dataclasses in this module from the start so there is one result model.

- [ ] **Step 1: Write input-validation tests**

Start `tests/benchmark/test_cohort_scoring.py` with a tiny event helper and tests:

```python
from src.benchmark.cohort_scoring import CohortIdentity, CohortItem, score_cohort
from src.benchmark.models import BenchmarkEvent


def benchmark_event(song: str, time_sec: float, common_class: str, source: str) -> BenchmarkEvent:
    return BenchmarkEvent(song, time_sec, common_class, source)


def identity() -> CohortIdentity:
    return CohortIdentity(
        cohort_id="oaf-full-mix-v1",
        reference_manifest_sha256="a" * 64,
        backend_id="magenta-egmd-tf1-94529798-8hit-v1",
        model_id="magenta-egmd-ckpt-569400-v1",
        backend_descriptor_sha256="b" * 64,
        prediction_map_version="crux.prediction-map/oaf-egmd-8hit-v1",
        input_view_id="full-mix-v1",
    )


def test_success_requires_prediction_events() -> None:
    item = CohortItem(
        simfile_id="1",
        status="success",
        reference_events=(benchmark_event("1", 1.0, "kick", "ground_truth"),),
        prediction_events=None,
    )

    with pytest.raises(ValueError, match="success item requires prediction_events"):
        score_cohort(identity(), (item,))


def test_non_success_must_not_carry_prediction_events() -> None:
    item = CohortItem(
        simfile_id="1",
        status="failed",
        reference_events=(benchmark_event("1", 1.0, "kick", "ground_truth"),),
        prediction_events=(benchmark_event("1", 1.0, "kick", "prediction"),),
        failure_reason="worker_error",
    )

    with pytest.raises(ValueError, match="non-success item must not carry prediction_events"):
        score_cohort(identity(), (item,))
```

Also require nonempty reference events for `success`, unique `simfile_id` values, positive sorted unique tolerances, and identity `scoring_version == SCORING_VERSION`.

- [ ] **Step 2: Write the 30/50/100 ms × raw/aligned matrix test**

Use one song whose prediction is 40 ms late:

```python
def test_score_cohort_emits_fixed_tolerance_and_mode_matrix() -> None:
    item = CohortItem(
        simfile_id="1",
        status="success",
        reference_events=(benchmark_event("1", 1.0, "kick", "ground_truth"),),
        prediction_events=(benchmark_event("1", 1.04, "kick", "prediction"),),
    )

    result = score_cohort(identity(), (item,))

    assert [(score.tolerance_ms, score.mode) for score in result.song_scores] == [
        (30, "aligned"),
        (30, "raw"),
        (50, "aligned"),
        (50, "raw"),
        (100, "aligned"),
        (100, "raw"),
    ]
    raw_30 = next(score for score in result.song_scores if score.tolerance_ms == 30 and score.mode == "raw")
    aligned_30 = next(
        score for score in result.song_scores if score.tolerance_ms == 30 and score.mode == "aligned"
    )
    assert raw_30.summary.f1 == 0.0
    assert aligned_30.summary.f1 == 1.0
```

Sort by `(simfile_id, tolerance_ms, mode)` everywhere so rendering never depends on insertion order.

- [ ] **Step 3: Run the new tests to verify RED**

```bash
uv run pytest tests/benchmark/test_cohort_scoring.py -q
```

Expected: FAIL because the cohort module does not exist.

- [ ] **Step 4: Implement the result dataclasses and validation**

Create `src/benchmark/cohort_scoring.py`. Keep validation local and explicit; do not introduce a generic validation framework.

Use:

```python
SCORING_VERSION = "crux.single-cohort-scoring/v1"
DEFAULT_TOLERANCES_MS = (30, 50, 100)
```

Validate `CohortIdentity` strings as nonempty and both SHA fields as exactly 64 lowercase hexadecimal characters. Validate `CohortItem` status/event invariants in `score_cohort()` so callers receive one deterministic failure boundary.

- [ ] **Step 5: Implement one scorer call per song/tolerance**

For every successful item and tolerance:

```python
aligned = score_events_with_alignment(
    list(item.reference_events),
    list(item.prediction_events),
    tolerance_sec=tolerance_ms / 1000.0,
)
```

Turn `aligned.raw` and `aligned.aligned` into separate `SongScore` rows. Do not call `score_events()` again for per-class metrics.

`prediction_to_reference_ratio` is:

```python
len(item.prediction_events) / len(item.reference_events)
```

A success item cannot have zero references, so no divide-by-zero policy is needed.

- [ ] **Step 6: Derive class rows from each existing `ScoreResult`**

Add a private helper:

```python
def _class_scores(result: ScoreResult) -> tuple[ClassScore, ...]: ...
```

For each class present in matches or either unmatched side:

- TP = matched rows whose ground-truth class equals the class;
- FP = unmatched predictions for the class;
- FN = unmatched references for the class;
- reference support = TP + FN;
- prediction support = TP + FP.

Build `ScoreSummary(tp, fp, fn)` for class metrics. Timing-error percentile fields may remain `None` in `ClassScore`; HPA-325 requires per-class P/R/F1/support, not duplicate per-class timing distributions.

- [ ] **Step 7: Build event diagnostics from the same `ScoreResult`**

For matched events:

```python
EventDiagnostic(
    simfile_id=item.simfile_id,
    tolerance_ms=tolerance_ms,
    mode=mode,
    outcome="matched",
    common_class=match.ground_truth.canonical_class,
    reference_time_sec=match.ground_truth.time_sec,
    prediction_time_sec=match.prediction.time_sec,
    timing_error_sec=match.timing_error_sec,
)
```

For false positives, reference time/error are `None`. For false negatives, prediction time/error are `None`.

Sort diagnostics by `(simfile_id, tolerance_ms, mode, outcome, common_class, reference_time_sec or -1, prediction_time_sec or -1)`.

- [ ] **Step 8: Prove per-class reconciliation**

Add a test with kick + snare + one unmatched hihat and assert for every `SongScore`:

```python
assert sum(row.summary.true_positives for row in score.per_class) == score.summary.true_positives
assert sum(row.summary.false_positives for row in score.per_class) == score.summary.false_positives
assert sum(row.summary.false_negatives for row in score.per_class) == score.summary.false_negatives
```

Also assert a class occurring on only one side remains in `per_class` with the expected zero metric.

- [ ] **Step 9: Run focused scoring tests**

```bash
uv run pytest tests/benchmark/test_cohort_scoring.py tests/benchmark/test_scoring.py -q
```

Expected: PASS without modifying `src/benchmark/scoring.py`.

- [ ] **Step 10: Commit the single-song/matrix scorer**

```bash
git add src/benchmark/cohort_scoring.py tests/benchmark/test_cohort_scoring.py
git commit -m "feat: add single-cohort scoring"
```

---

## Task 3: Add cohort aggregates and full population accounting

**Files:**
- Modify: `src/benchmark/cohort_scoring.py`
- Modify: `tests/benchmark/test_cohort_scoring.py`

**Interfaces:**
- Consumes: `SongScore`, `ClassScore`, and all input `CohortItem` rows.
- Produces:

```python
@dataclass(frozen=True)
class PopulationSummary:
    total_count: int
    success_count: int
    failed_count: int
    skipped_count: int
    quarantined_count: int
    reason_counts: tuple[tuple[str, int], ...]

@dataclass(frozen=True)
class CohortAggregate:
    tolerance_ms: int
    mode: ScoreMode
    event_micro: ScoreSummary
    song_macro_f1: float | None
    class_macro_f1: float | None
    per_class: tuple[ClassScore, ...]
    successful_song_count: int
```

- [ ] **Step 1: Write population-accounting tests**

Add a cohort containing one `success`, one `failed`, one `skipped`, and one `quarantined` item. Give the failed/skipped/quarantined rows explicit reasons:

```python
assert result.population.total_count == 4
assert result.population.success_count == 1
assert result.population.failed_count == 1
assert result.population.skipped_count == 1
assert result.population.quarantined_count == 1
assert result.population.reason_counts == (
    ("already_complete", 1),
    ("reference_quarantined", 1),
    ("worker_error", 1),
)
```

Warnings are not failure reasons and are not counted here.

- [ ] **Step 2: Write exact aggregate semantic tests**

Use two successful songs:

- song A: kick TP, snare FN;
- song B: kick TP plus one kick FP.

For a fixed tolerance/mode assert:

```python
assert aggregate.event_micro.true_positives == 2
assert aggregate.event_micro.false_positives == 1
assert aggregate.event_micro.false_negatives == 1
assert aggregate.event_micro.f1 == pytest.approx(2 * (2 / 3) * (2 / 3) / ((2 / 3) + (2 / 3)))
assert aggregate.song_macro_f1 == pytest.approx((song_a_f1 + song_b_f1) / 2)
```

For class macro, aggregate kick and snare independently, include both because at least one side has support, then average their class F1 values.

- [ ] **Step 3: Write the zero-success cohort test**

A cohort containing only failed/quarantined items must still return six aggregate rows (three tolerances × two modes) with:

```python
assert aggregate.successful_song_count == 0
assert aggregate.event_micro.true_positives == 0
assert aggregate.event_micro.false_positives == 0
assert aggregate.event_micro.false_negatives == 0
assert aggregate.event_micro.f1 is None
assert aggregate.song_macro_f1 is None
assert aggregate.class_macro_f1 is None
assert aggregate.per_class == ()
```

This preserves complete population reporting without inventing scores.

- [ ] **Step 4: Run aggregate tests to verify RED**

```bash
uv run pytest tests/benchmark/test_cohort_scoring.py -q
```

Expected: FAIL on missing aggregate/population behavior.

- [ ] **Step 5: Implement population accounting**

Build counts from every input item before filtering successes. Require a non-success item to have a nonempty `failure_reason`; require a success item to have `failure_reason is None`.

Store grouped reasons as a sorted tuple of `(reason, count)` pairs. Do not create an enum of model-specific failure reasons in HPA-325; HPA-326 owns execution reason vocabulary.

- [ ] **Step 6: Implement event-micro and per-class aggregate counts**

Group `SongScore` rows by `(tolerance_ms, mode)`. Event micro is constructed only from summed TP/FP/FN:

```python
ScoreSummary(
    true_positives=sum(score.summary.true_positives for score in rows),
    false_positives=sum(score.summary.false_positives for score in rows),
    false_negatives=sum(score.summary.false_negatives for score in rows),
)
```

Aggregate per-class rows by common class the same way and recompute support from the summed counts. Do not average per-song precision/recall for event-micro metrics.

- [ ] **Step 7: Implement song-macro and class-macro F1**

Use a tiny helper:

```python
def _mean_f1(values: tuple[float | None, ...]) -> float | None:
    scored = tuple(value for value in values if value is not None)
    return fmean(scored) if scored else None
```

A successful eligible song with an empty prediction has `ScoreSummary.f1 == 0.0`, so it remains in song macro. For class macro, include aggregated classes having reference or prediction support and average their non-`None` F1 values. Classes absent from both sides never appear and are not padded into the macro.

- [ ] **Step 8: Assert aggregate reconciliation across every dimension**

Add a parametrized test over all six `(tolerance_ms, mode)` rows proving:

```python
song_rows = [row for row in result.song_scores if row.tolerance_ms == aggregate.tolerance_ms and row.mode == aggregate.mode]
assert aggregate.event_micro.true_positives == sum(row.summary.true_positives for row in song_rows)
assert aggregate.event_micro.false_positives == sum(row.summary.false_positives for row in song_rows)
assert aggregate.event_micro.false_negatives == sum(row.summary.false_negatives for row in song_rows)
```

Also reconcile each aggregate class against its song-level class rows.

- [ ] **Step 9: Run the full cohort unit suite**

```bash
uv run pytest tests/benchmark/test_cohort_scoring.py -q
```

Expected: PASS.

- [ ] **Step 10: Commit aggregates and accounting**

```bash
git add src/benchmark/cohort_scoring.py tests/benchmark/test_cohort_scoring.py
git commit -m "feat: aggregate cohort benchmark scores"
```

---

## Task 4: Replace the flat report writer with deterministic cohort artifacts

**Files:**
- Modify: `src/benchmark/reports.py`
- Modify: `tests/benchmark/test_reports.py`

**Interfaces:**
- Consumes: `CohortScoreResult`.
- Produces:

```python
@dataclass(frozen=True)
class ReportArtifacts:
    summary_json: Path
    per_song_csv: Path
    per_class_csv: Path
    event_diagnostics_jsonl: Path
    summary_markdown: Path


def write_cohort_reports(
    result: CohortScoreResult,
    output_dir: Path,
) -> ReportArtifacts: ...
```

- [ ] **Step 1: Replace the old report test with the five-file contract**

Build one small `CohortScoreResult` by calling `score_cohort()` rather than hand-constructing duplicate report structures. Assert:

```python
artifacts = write_cohort_reports(result, tmp_path)
assert artifacts.summary_json == tmp_path / "summary.json"
assert artifacts.per_song_csv == tmp_path / "per_song.csv"
assert artifacts.per_class_csv == tmp_path / "per_class.csv"
assert artifacts.event_diagnostics_jsonl == tmp_path / "event_diagnostics.jsonl"
assert artifacts.summary_markdown == tmp_path / "summary.md"
assert all(path.is_file() for path in dataclasses.astuple(artifacts))
```

Delete imports/tests for legacy `ChartReport` and `write_reports`.

- [ ] **Step 2: Write summary identity/population assertions**

Parse `summary.json` with the strict JSON loader and assert it contains, at minimum:

```text
schema = crux.single-cohort-report/v1
cohort_identity
population
aggregates
```

Each aggregate row contains tolerance, mode, event-micro metrics, song-macro F1, class-macro F1, successful song count, and per-class aggregates. Keep the top-level key set fixed in the test.

- [ ] **Step 3: Write fixed CSV header assertions**

`per_song.csv` fields, in this exact order:

```text
simfile_id,tolerance_ms,mode,tp,fp,fn,precision,recall,f1,
prediction_to_reference_ratio,median_abs_error_ms,p95_abs_error_ms,offset_ms,warnings
```

`per_class.csv` fields, in this exact order:

```text
scope,simfile_id,tolerance_ms,mode,common_class,tp,fp,fn,
reference_support,prediction_support,precision,recall,f1
```

`scope` is `song` or `cohort`. Cohort rows use an empty `simfile_id` cell. This avoids creating two separate per-class files while keeping the dimension explicit.

- [ ] **Step 4: Write event JSONL shape assertions**

Every line of `event_diagnostics.jsonl` must have the exact key set:

```text
common_class
mode
outcome
prediction_time_sec
reference_time_sec
simfile_id
timing_error_sec
tolerance_ms
```

Use canonical JSON line serialization and assert the file ends in one newline when nonempty. An empty diagnostic set produces an empty file; do not fabricate a header record.

- [ ] **Step 5: Write the byte-determinism test**

Call `write_cohort_reports()` with the same result into `tmp_path / "first"` and `tmp_path / "second"`. For each of the five relative file names:

```python
assert (first / name).read_bytes() == (second / name).read_bytes()
```

Then call `score_cohort()` with the same `CohortItem` values in reversed input order and prove the rendered bytes remain identical. This pins sorting at the domain boundary rather than relying on caller order.

- [ ] **Step 6: Run report tests to verify RED**

```bash
uv run pytest tests/benchmark/test_reports.py -q
```

Expected: FAIL against the legacy report API.

- [ ] **Step 7: Rewrite `reports.py` around the cohort result**

Use existing `canonical_json_bytes()` for `summary.json` and every JSONL row. Do not use a second JSON normalization policy.

Convert metric fields using small explicit helpers:

```python
def _metric(summary: ScoreSummary) -> dict[str, object]:
    return {
        "tp": summary.true_positives,
        "fp": summary.false_positives,
        "fn": summary.false_negatives,
        "precision": summary.precision,
        "recall": summary.recall,
        "f1": summary.f1,
    }
```

Milliseconds are derived only at rendering time from second-based domain values.

- [ ] **Step 8: Keep Markdown as a view, not another metrics implementation**

Render sections in fixed order:

```text
# Single-Cohort Drum Benchmark
## Identity
## Population
## Aggregate Results
## Per-Class Aggregate Results
## Song Extremes
```

For `Song Extremes`, show up to five lowest and five highest raw 50 ms successful song F1 rows, sorted deterministically by `(f1, simfile_id)`. If there are no successful rows, write `No successful songs.`. Do not add statistical outlier detection in this ticket.

- [ ] **Step 9: Run reports and cohort tests together**

```bash
uv run pytest tests/benchmark/test_reports.py tests/benchmark/test_cohort_scoring.py -q
```

Expected: PASS.

- [ ] **Step 10: Commit deterministic reports**

```bash
git add src/benchmark/reports.py tests/benchmark/test_reports.py
git commit -m "feat: write cohort benchmark reports"
```

---

## Task 5: Prove persisted HPA-324/HPA-423 artifacts score without inference

**Files:**
- Create: `tests/benchmark/test_cohort_scoring_acceptance.py`
- Modify only if a test exposes a missing narrow adapter: `src/benchmark/scorer_input.py`, `src/benchmark/cohort_scoring.py`, or `src/benchmark/reports.py`

**Interfaces:**
- Consumes: HPA-324 `map_reference_events()`, HPA-423 `map_oaf_prediction()` + prediction artifact v2 + scorer bridge, `score_cohort()`, and `write_cohort_reports()`.
- Produces: one end-to-end test proving persisted-event rescoring; no backend or Docker process is allowed.

- [ ] **Step 1: Build a native HPA-323-shaped reference fixture in the test**

Reuse the actual `NativeReferenceEvent` constructor from `reference_timing.py` and create two events at known audio-relative times on mapped DTX lanes, for example kick lane `13` and hi-hat lane `18`. Use stable source/chart/audio identity fields required by the dataclass.

Then:

```python
reference_mapping = map_reference_events(native_reference_events)
reference_events = reference_to_benchmark_events("42", reference_mapping.common_events)
assert [event.canonical_class for event in reference_events] == ["kick", "hihat"]
```

Do not hand-map lanes in the acceptance test.

- [ ] **Step 2: Build and round-trip a real OaF-shaped prediction artifact**

Follow the existing `tests/benchmark/test_task_d_contract.py` descriptor/audio/native-event fixture style. Create kick and hihat native events with non-null OaF confidence/velocity, run:

```python
mapped_prediction, diagnostics = map_oaf_prediction(native_prediction)
content = render_prediction_artifact(mapped_prediction)
artifact = read_prediction_artifact(content)
prediction_events = prediction_to_benchmark_events(artifact)
```

Assert `diagnostics.unmapped == {}` and the prediction events use common classes.

- [ ] **Step 3: Score the persisted-event pair with no backend call**

Create one `CohortItem(status="success", ...)`, score it, and assert at raw 50 ms:

```python
assert score.summary.true_positives == 2
assert score.summary.false_positives == 0
assert score.summary.false_negatives == 0
assert score.summary.f1 == 1.0
```

The test imports no `OafBackend` and starts no worker/Docker process.

- [ ] **Step 4: Prove mapping-version identity reaches the cohort**

Set `CohortIdentity.prediction_map_version` from `OAF_PREDICTION_MAP_ID` and assert the rendered summary contains exactly that value. This is the rescoring seam: changing map identity upstream changes the score identity without rerunning inference.

- [ ] **Step 5: Write and validate all report artifacts**

Call `write_cohort_reports()` and assert:

- all five artifacts exist;
- `summary.json` has the expected reference-manifest hash, backend/model identity, map ID, input-view ID, and `SCORING_VERSION`;
- population totals reconcile to one successful input;
- raw 50 ms aggregate TP/FP/FN equal the per-song row;
- event diagnostics contain two matched rows for raw 50 ms.

- [ ] **Step 6: Run the acceptance path and existing HPA-324/HPA-423 contract suites**

```bash
uv run pytest \
  tests/benchmark/test_cohort_scoring_acceptance.py \
  tests/benchmark/test_reference_set.py \
  tests/benchmark/test_task_d_contract.py -q
```

Expected: PASS.

- [ ] **Step 7: Commit the persisted-artifact acceptance proof**

```bash
git add tests/benchmark/test_cohort_scoring_acceptance.py \
  src/benchmark/scorer_input.py src/benchmark/cohort_scoring.py src/benchmark/reports.py
git commit -m "test: prove persisted cohort rescoring"
```

If the three source files were unchanged in this task, omit them from `git add`.

---

## Task 6: Final reconciliation and repository verification

**Files:**
- Verify all HPA-325 files above.
- No new production files should be introduced in this task.

- [ ] **Step 1: Run the full benchmark-focused suite**

```bash
uv run pytest tests/benchmark -q
```

Expected: PASS.

- [ ] **Step 2: Run the complete repository test suite**

```bash
uv run pytest
```

Expected: PASS.

- [ ] **Step 3: Run repository lint and format gates from `CLAUDE.md`**

```bash
uv run ruff check src tests
uv run ruff format --check src tests
uv run pylint src/app src/cli
```

Expected: PASS under the repository's documented gate set.

- [ ] **Step 4: Check whitespace and scope**

```bash
git diff --check main...HEAD
git status --short
git diff --stat main...HEAD
```

Expected production scope:

```text
src/benchmark/cohort_scoring.py
src/benchmark/scorer_input.py
src/benchmark/reports.py
tests/benchmark/test_scorer_input.py
tests/benchmark/test_scoring.py
tests/benchmark/test_cohort_scoring.py
tests/benchmark/test_cohort_scoring_acceptance.py
tests/benchmark/test_reports.py
```

Plus the already-reviewed HPA-325 design/plan docs if implementation occurs on the planning branch. Any additional production file requires an explicit scope reason before merge.

- [ ] **Step 5: Confirm all acceptance invariants in one focused run**

```bash
uv run pytest \
  tests/benchmark/test_scorer_input.py \
  tests/benchmark/test_scoring.py \
  tests/benchmark/test_cohort_scoring.py \
  tests/benchmark/test_cohort_scoring_acceptance.py \
  tests/benchmark/test_reports.py -q
```

Expected: PASS with coverage of:

- persisted reference and prediction scorer adapters;
- simultaneous/dense/empty matcher behavior;
- 30/50/100 ms × raw/aligned matrix;
- per-song/per-class reconciliation;
- micro/song-macro/class-macro semantics;
- failed/skipped/quarantined population accounting;
- matched/FP/FN diagnostics;
- byte-identical report reproduction;
- persisted-artifact rescoring without model inference.

- [ ] **Step 6: Commit only if verification required cleanup edits**

```bash
git add src/benchmark tests/benchmark
git commit -m "test: finalize HPA-325 scoring contract"
```

Skip this commit if verification produced no edits.

## Completion Boundary

HPA-325 is complete when the deterministic in-memory scorer and five report artifacts are implemented and verified as above. Do **not** add a corpus-run command or define a prediction-run manifest to finish HPA-325. HPA-326 should reuse this boundary by adapting each of its run rows into `CohortItem` and passing the frozen cohort identity into `score_cohort()` after prediction persistence.

That handoff keeps HPA-326 focused on execution/resume/runtime measurement and keeps HPA-325 focused on scoring semantics, which prevents either ticket from owning two overlapping orchestration layers.
