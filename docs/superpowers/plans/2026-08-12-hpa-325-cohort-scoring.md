# HPA-325 Single-Cohort Scoring and Reports Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn one frozen model/input-view cohort of persisted reference and prediction events into deterministic per-song, per-class, aggregate, coverage, population, and event-diagnostic benchmark reports at 30/50/100 ms without invoking inference or rebuilding the existing matcher/mapping seams.

**Architecture:** Keep `src/benchmark/scoring.py` as the sole onset-matching/alignment implementation. Extend `scorer_input.py` only with the reference common-event adapter, then add one `cohort_scoring.py` module that validates in-memory cohort rows, derives artifact coverage, invokes the existing scorer, derives class/cohort aggregates from the same `ScoreResult`, and retains every population outcome. Rewrite the small existing `reports.py` API around deterministic cohort rendering. HPA-326 will later adapt its run-manifest rows into `CohortItem` rather than HPA-325 inventing that future schema.

**Tech Stack:** Python 3.12, frozen dataclasses, `typing.Literal`, `statistics.fmean`, `math.ceil`, existing `BenchmarkEvent`/`ScoreSummary`/`ScoreResult`, HPA-324 reference mapping, HPA-423 prediction artifacts, existing canonical JSON helpers, CSV, pathlib, pytest, Ruff, Pylint.

## Global Constraints

- Consume merged HPA-323 audio-relative native reference artifacts; do not rerun DTX parsing, timing, BGM resolution, chart selection, R2 access, or audio decoding.
- Consume HPA-324 `map_reference_events()` / `ReferenceMappingResult` / `CommonReferenceEvent` and frozen taxonomy IDs; do not define another lane or class map.
- Consume HPA-423 prediction artifact v2 and `prediction_to_benchmark_events()`; do not rebuild `NativeEvent` mapping or prediction persistence.
- One invocation scores exactly one model/checkpoint + one `input_view_id` + one named population scope. Broad, reviewed, and pilot populations are separate cohort invocations.
- Default tolerances are exactly `30`, `50`, and `100` milliseconds.
- Raw and diagnostically aligned scores are separate named modes; alignment never replaces the raw result.
- Headline matching uses onset time + common class only. Confidence and velocity remain diagnostic metadata and never affect matching.
- Failed, skipped, and quarantined items remain visible as item rows and population counts; never silently reduce the cohort to successes.
- Persist reference native/common/ignored/unmapped counts and prediction native/mapped/unmapped/native-class coverage when the corresponding artifacts exist.
- Every persisted score/diagnostic row carries the load-bearing cohort, model-lock, reference/timing, mapping, input-view, and scoring identities.
- Do not change `score_events()`, `_match_class()`, or alignment code unless a focused adversarial test first proves incorrect behavior.
- Do not add inference, worker pools, queues, retries, a prediction-run manifest, cross-model comparison, Parquet, a database, a metrics service, a report plugin registry, or a generic benchmark pipeline.
- Breaking the old internal `reports.py` API is allowed; do not add compatibility wrappers for `ChartReport` / `write_reports`.
- Every persisted JSON or JSONL output uses existing canonical JSON serialization. CSV rows and Markdown sections have fixed deterministic ordering.
- Repository verification follows `CLAUDE.md`: `uv run pytest`, `uv run ruff check src tests`, `uv run ruff format --check src tests`, and `uv run pylint src/app src/cli`.

---

## File Map

### Create

- `src/benchmark/cohort_scoring.py` — cohort identities/items, artifact coverage, scoring matrix, per-class derivation, F1 distributions, aggregates, and population accounting.
- `tests/benchmark/test_scorer_input.py` — focused reference-side scorer adapter tests; existing HPA-423 prediction bridge coverage remains in `test_task_d_contract.py`.
- `tests/benchmark/test_cohort_scoring.py` — validation, coverage, score matrix, diagnostics, aggregation, and reconciliation contract.
- `tests/benchmark/test_cohort_scoring_acceptance.py` — persisted HPA-324/HPA-423 artifact to reports integration without inference.

### Modify

- `src/benchmark/scorer_input.py` — add only `reference_to_benchmark_events()`.
- `src/benchmark/reports.py` — replace the flat chart writer with deterministic cohort report rendering.
- `tests/benchmark/test_scoring.py` — pin remaining adversarial matcher behavior before relying on it for aggregate reports.
- `tests/benchmark/test_reports.py` — replace legacy `ChartReport` tests with six-artifact, traceability, and byte-determinism tests.

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

## Task 1: Pin the matcher and add the reference scorer adapter

**Files:**
- Modify: `tests/benchmark/test_scoring.py`
- Modify: `src/benchmark/scorer_input.py`
- Create: `tests/benchmark/test_scorer_input.py`

**Interfaces:**
- Consumes: `CommonReferenceEvent` from `src.benchmark.reference_set`; existing `BenchmarkEvent` and prediction scorer adapters.
- Produces:

```python
def reference_to_benchmark_events(
    simfile_id: str,
    common_events: tuple[CommonReferenceEvent, ...],
) -> tuple[BenchmarkEvent, ...]: ...
```

- [ ] **Step 1: Add simultaneous-hit and empty-song matcher regressions**

Append to `tests/benchmark/test_scoring.py`:

```python
def test_simultaneous_same_class_hits_match_one_to_one() -> None:
    result = score_events(
        [event(1.0, "kick", "gt_a"), event(1.0, "kick", "gt_b")],
        [event(1.0, "kick", "pred_a"), event(1.0, "kick", "pred_b")],
        0.03,
    )
    assert (result.summary.true_positives, result.summary.false_positives, result.summary.false_negatives) == (2, 0, 0)


def test_empty_prediction_song_counts_every_reference_as_false_negative() -> None:
    result = score_events(
        [event(1.0, "kick", "gt"), event(2.0, "snare", "gt")],
        [],
        0.05,
    )
    assert (result.summary.true_positives, result.summary.false_positives, result.summary.false_negatives) == (0, 0, 2)
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

Keep the existing dense-hit and large-offset fixtures unchanged.

- [ ] **Step 2: Run matcher regressions before touching scoring code**

```bash
uv run pytest tests/benchmark/test_scoring.py -q
```

Expected: PASS. If a new fixture fails, treat that as a scoring correctness defect and resolve it before aggregate work; do not compensate in `cohort_scoring.py`.

- [ ] **Step 3: Write reference common-event adapter tests**

Create `tests/benchmark/test_scorer_input.py`:

```python
from decimal import Decimal

import pytest

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


def test_reference_adapter_keeps_exact_common_projection_order() -> None:
    common = (
        CommonReferenceEvent(Decimal("1.000000"), "kick", ()),
        CommonReferenceEvent(Decimal("1.000000"), "snare", ()),
    )
    assert [(event.time_sec, event.canonical_class) for event in reference_to_benchmark_events("song", common)] == [
        (1.0, "kick"),
        (1.0, "snare"),
    ]


def test_reference_adapter_requires_nonempty_simfile_id() -> None:
    with pytest.raises(ValueError, match="simfile_id must be a nonempty string"):
        reference_to_benchmark_events("", ())
```

- [ ] **Step 4: Verify RED**

```bash
uv run pytest tests/benchmark/test_scorer_input.py -q
```

Expected: FAIL because `reference_to_benchmark_events` is missing.

- [ ] **Step 5: Implement the minimal adapter**

In `src/benchmark/scorer_input.py`:

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

Add it to `__all__`. Do not accept native reference events here; HPA-324's exact common collapse must happen first.

- [ ] **Step 6: Run the new adapter and existing HPA-423 bridge tests**

```bash
uv run pytest tests/benchmark/test_scorer_input.py tests/benchmark/test_task_d_contract.py -q
```

Expected: PASS, including OaF hihat/tom common-class round-trip coverage already present in `test_task_d_contract.py`.

- [ ] **Step 7: Commit**

```bash
git add src/benchmark/scorer_input.py tests/benchmark/test_scorer_input.py tests/benchmark/test_scoring.py
git commit -m "test: pin cohort scorer inputs"
```

---

## Task 2: Define cohort identity, item, and artifact-coverage contracts

**Files:**
- Create: `src/benchmark/cohort_scoring.py`
- Create: `tests/benchmark/test_cohort_scoring.py`

**Interfaces:**
- Consumes: `ReferenceMappingResult`, `PredictionArtifact`, `BenchmarkEvent`.
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


def coverage_from_artifacts(
    reference: ReferenceMappingResult,
    prediction: PredictionArtifact | None,
) -> CohortCoverage: ...
```

- [ ] **Step 1: Write one shared identity fixture**

Start `tests/benchmark/test_cohort_scoring.py`:

```python
from src.benchmark.cohort_scoring import CohortIdentity
from src.benchmark.taxonomy import DTX_LANE_MAP_VERSION, OAF_PREDICTION_MAP_ID, TAXONOMY_VERSION


def identity() -> CohortIdentity:
    return CohortIdentity(
        cohort_id="oaf-full-mix-v1",
        reference_manifest_sha256="a" * 64,
        reference_timing_version="sha256:" + "b" * 64,
        taxonomy_version=TAXONOMY_VERSION,
        lane_map_version=DTX_LANE_MAP_VERSION,
        backend_id="magenta-egmd-tf1-94529798-8hit-v1",
        model_id="magenta-egmd-ckpt-569400-v1",
        model_lock_sha256="c" * 64,
        backend_descriptor_sha256="d" * 64,
        prediction_map_version=OAF_PREDICTION_MAP_ID,
        input_view_id="full-mix-v1",
    )
```

- [ ] **Step 2: Write artifact coverage tests using real domain objects**

Reuse the `NativeReferenceEvent` shape already established by `tests/benchmark/test_reference_set.py`:

```python
def native_reference(lane_id: str, time_sec: float) -> NativeReferenceEvent:
    return NativeReferenceEvent(
        simfile_id=7,
        selected_chart_key="7/chart.dtx",
        selected_chart_content_hash="1" * 64,
        source_audio_key="7/audio.ogg",
        source_audio_content_hash="2" * 64,
        source_order=0,
        measure=1,
        position=0.0,
        lane_id=lane_id,
        note_id="01",
        chart_time_sec=time_sec,
        audio_time_sec=time_sec,
    )
```

Build a reference mapping containing mapped lane `13`, ignored lane `54`, and no unknown lane. For prediction coverage, reuse the OaF descriptor/audio/native-event fixture style in `tests/benchmark/test_task_d_contract.py`, map it, and round-trip with `read_prediction_artifact(render_prediction_artifact(mapped))`.

Assert:

```python
assert coverage.reference_native_event_count == 2
assert coverage.reference_common_event_count == 1
assert coverage.reference_ignored_event_count == 1
assert coverage.reference_unmapped_event_count == 0
assert coverage.prediction_native_event_count == 2
assert coverage.prediction_mapped_event_count == 1
assert coverage.prediction_unmapped_event_count == 1
assert coverage.prediction_native_class_counts == (("midi_36", 1), ("midi_75", 1))
```

For `prediction=None`, prediction counts are `None` and native-class counts are empty.

- [ ] **Step 3: Verify RED**

```bash
uv run pytest tests/benchmark/test_cohort_scoring.py -q
```

Expected: FAIL because `cohort_scoring.py` does not exist.

- [ ] **Step 4: Implement `CohortIdentity`, `CohortCoverage`, and `coverage_from_artifacts()`**

Compute reference native count as:

```python
len(reference.mapped_events) + sum(reference.diagnostics.ignored.values()) + sum(reference.diagnostics.unmapped.values())
```

Prediction coverage when an artifact exists:

```python
native_events = artifact.prediction.events
mapped_count = sum(event.mapping_status == "mapped" for event in native_events)
unmapped_count = sum(event.mapping_status == "unmapped" for event in native_events)
```

Count `event.native.native_class_id` with `Counter` and store `tuple(sorted(counter.items()))`. Do not create a separate coverage artifact.

- [ ] **Step 5: Write input validation tests**

Add tests requiring:

- nonempty identity strings;
- exactly 64 lowercase hex characters for `reference_manifest_sha256`, `model_lock_sha256`, and `backend_descriptor_sha256`;
- `scoring_version == SCORING_VERSION`;
- unique nonempty `simfile_id` values;
- a `success` item has nonempty references and a prediction tuple (empty tuple is valid); 
- a non-success item has `prediction_events is None` and a nonempty `failure_reason`;
- a success item has `failure_reason is None`;
- successful prediction coverage is non-`None`, balances `native == mapped + unmapped`, and `mapped == len(prediction_events)`;
- reference common count equals `len(reference_events)` for every item.

Use explicit `ValueError` messages and test them. Do not introduce a validation framework.

- [ ] **Step 6: Run coverage/validation tests**

```bash
uv run pytest tests/benchmark/test_cohort_scoring.py -q
```

Expected: PASS for the implemented contract-only tests.

- [ ] **Step 7: Commit**

```bash
git add src/benchmark/cohort_scoring.py tests/benchmark/test_cohort_scoring.py
git commit -m "feat: define cohort scoring contract"
```

---

## Task 3: Score every tolerance/mode and derive class/event diagnostics from one matcher result

**Files:**
- Modify: `src/benchmark/cohort_scoring.py`
- Modify: `tests/benchmark/test_cohort_scoring.py`

**Interfaces:**
- Consumes: validated `CohortItem` rows and `score_events_with_alignment()`.
- Produces:

```python
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
    scored_prediction_time_sec: float | None
    timing_error_sec: float | None


def score_cohort(
    identity: CohortIdentity,
    items: tuple[CohortItem, ...],
    tolerances_ms: tuple[int, ...] = DEFAULT_TOLERANCES_MS,
) -> CohortScoreResult: ...
```

- [ ] **Step 1: Write the 30/50/100 ms × raw/aligned matrix test**

Use one successful song with a 40 ms-late kick prediction and valid coverage. Assert the sorted result matrix is:

```python
[
    (30, "aligned"), (30, "raw"),
    (50, "aligned"), (50, "raw"),
    (100, "aligned"), (100, "raw"),
]
```

At raw 30 ms F1 is `0.0`; at aligned 30 ms F1 is `1.0`; at raw 50/100 ms F1 is `1.0`.

- [ ] **Step 2: Write per-class reconciliation tests**

Use kick + snare references with one unmatched hihat prediction. For every song result:

```python
assert sum(row.summary.true_positives for row in score.per_class) == score.summary.true_positives
assert sum(row.summary.false_positives for row in score.per_class) == score.summary.false_positives
assert sum(row.summary.false_negatives for row in score.per_class) == score.summary.false_negatives
```

Assert hihat remains present even with zero reference support.

- [ ] **Step 3: Write the missing-confidence/velocity regression at the scorer level**

Construct prediction `BenchmarkEvent` objects with `metadata={}` and prove `score_cohort()` produces the same counts/F1 as events whose metadata contains confidence/velocity. Do not attempt to create an invalid OaF artifact with null fields; the shared scorer contract is intentionally metadata-independent.

- [ ] **Step 4: Write the aligned diagnostic-time regression**

For reference time `1.0`, persisted prediction time `1.1`, and aligned offset `-0.1`, assert the aligned matched diagnostic contains:

```python
assert diagnostic.reference_time_sec == pytest.approx(1.0)
assert diagnostic.prediction_time_sec == pytest.approx(1.1)
assert diagnostic.scored_prediction_time_sec == pytest.approx(1.0)
assert diagnostic.timing_error_sec == pytest.approx(0.0)
```

This pins original prediction provenance separately from the offset-adjusted scorer copy.

- [ ] **Step 5: Verify RED**

```bash
uv run pytest tests/benchmark/test_cohort_scoring.py -q
```

Expected: FAIL on missing score rows/diagnostics.

- [ ] **Step 6: Implement one scorer call per successful song/tolerance**

For every successful item and tolerance:

```python
aligned = score_events_with_alignment(
    list(item.reference_events),
    list(item.prediction_events),
    tolerance_sec=tolerance_ms / 1000.0,
)
```

Turn `aligned.raw` and `aligned.aligned` into separate `SongScore` rows. Do not call the matcher again for classes.

`prediction_to_reference_ratio` is `len(prediction_events) / len(reference_events)`; references are guaranteed nonempty.

- [ ] **Step 7: Derive per-class rows from the existing `ScoreResult`**

For each class present in matched or either unmatched set:

- TP = matched rows for that class;
- FP = unmatched predictions for that class;
- FN = unmatched references for that class;
- reference support = TP + FN;
- prediction support = TP + FP.

Use `ScoreSummary(tp, fp, fn)` for per-class P/R/F1. Keep class timing distribution out of scope because HPA-325 only requires class P/R/F1/support.

- [ ] **Step 8: Derive diagnostics from the same `ScoreResult`**

For raw mode, persisted and scored prediction times are identical. For aligned mode:

```python
original_prediction_time = scored_prediction_time - score_result.summary.offset_sec
```

Apply that conversion to matched and unmatched prediction rows. False negatives have both prediction fields `None`; false positives have reference/error fields `None`.

Sort diagnostics by `(simfile_id, tolerance_ms, mode, outcome, common_class, reference_time_sec or -1, prediction_time_sec or -1)`.

- [ ] **Step 9: Run focused scoring tests**

```bash
uv run pytest tests/benchmark/test_cohort_scoring.py tests/benchmark/test_scoring.py -q
```

Expected: PASS without modifying `src/benchmark/scoring.py`.

- [ ] **Step 10: Commit**

```bash
git add src/benchmark/cohort_scoring.py tests/benchmark/test_cohort_scoring.py
git commit -m "feat: score single benchmark cohorts"
```

---

## Task 4: Add population accounting, cohort aggregates, and F1 distributions

**Files:**
- Modify: `src/benchmark/cohort_scoring.py`
- Modify: `tests/benchmark/test_cohort_scoring.py`

**Interfaces:**
- Consumes: all input `CohortItem` rows plus `SongScore` rows.
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
class F1Distribution:
    minimum: float | None
    p10: float | None
    p25: float | None
    median: float | None
    p75: float | None
    p90: float | None
    maximum: float | None

@dataclass(frozen=True)
class CohortAggregate:
    tolerance_ms: int
    mode: ScoreMode
    event_micro: ScoreSummary
    song_macro_f1: float | None
    class_macro_f1: float | None
    song_f1_distribution: F1Distribution
    per_class: tuple[ClassScore, ...]
    successful_song_count: int

@dataclass(frozen=True)
class CohortScoreResult:
    identity: CohortIdentity
    tolerances_ms: tuple[int, ...]
    items: tuple[CohortItem, ...]
    song_scores: tuple[SongScore, ...]
    event_diagnostics: tuple[EventDiagnostic, ...]
    population: PopulationSummary
    aggregates: tuple[CohortAggregate, ...]
```

- [ ] **Step 1: Write full-population tests**

Use one success, one failed, one skipped, and one quarantined item. Assert:

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
assert [item.simfile_id for item in result.items] == sorted(item.simfile_id for item in input_items)
```

Warnings are retained on item rows but are not failure reasons.

- [ ] **Step 2: Write event-micro, song-macro, and class-macro tests**

Use two successful songs:

- song A: kick TP, snare FN;
- song B: kick TP plus one kick FP.

At a fixed tolerance/mode assert aggregate TP/FP/FN are `(2, 1, 1)`. Assert `song_macro_f1` equals the arithmetic mean of the two song F1s. Aggregate kick and snare independently, then assert `class_macro_f1` equals the mean of supported class F1 values.

Never average per-song precision/recall to obtain event-micro metrics.

- [ ] **Step 3: Write exact F1 distribution tests**

Use successful song F1 values `(0.0, 0.25, 0.5, 0.75, 1.0)` and require:

```python
assert distribution.minimum == 0.0
assert distribution.p10 == 0.25
assert distribution.p25 == 0.25
assert distribution.median == 0.5
assert distribution.p75 == 0.75
assert distribution.p90 == 1.0
assert distribution.maximum == 1.0
```

Implement p10/p25/p75/p90 with the same deterministic upper-nearest-rank convention used by the existing scoring percentile helper:

```python
index = min(len(values) - 1, ceil((len(values) - 1) * percentile))
```

Use `statistics.median()` for the median. One-song distributions repeat the same value at every percentile.

- [ ] **Step 4: Write the zero-success population test**

A cohort with only failed/quarantined rows still returns six aggregate rows. Require zero micro counts with `f1 is None`, `song_macro_f1 is None`, `class_macro_f1 is None`, empty class rows, and every `F1Distribution` field `None`.

- [ ] **Step 5: Verify RED**

```bash
uv run pytest tests/benchmark/test_cohort_scoring.py -q
```

Expected: FAIL on missing aggregate/distribution/population behavior.

- [ ] **Step 6: Implement population and reason counting before success filtering**

Build status counts and `Counter` failure reasons from all items. Store `result.items` sorted by `simfile_id` so report output does not depend on caller order.

- [ ] **Step 7: Implement event-micro and per-class aggregate counts**

Group `SongScore` rows by `(tolerance_ms, mode)`. Event micro is:

```python
ScoreSummary(
    true_positives=sum(score.summary.true_positives for score in rows),
    false_positives=sum(score.summary.false_positives for score in rows),
    false_negatives=sum(score.summary.false_negatives for score in rows),
)
```

Aggregate per-class counts the same way and recompute support from summed counts.

- [ ] **Step 8: Implement macro F1 and distribution helpers**

Use:

```python
def _mean_f1(values: tuple[float | None, ...]) -> float | None:
    scored = tuple(value for value in values if value is not None)
    return fmean(scored) if scored else None
```

A successful eligible song with an empty prediction has F1 `0.0`, so it remains in song macro/distribution. For class macro, only aggregate classes with reference or prediction support appear.

- [ ] **Step 9: Reconcile every aggregate against its song rows**

Parametrize all six `(tolerance_ms, mode)` rows and assert aggregate TP/FP/FN equal summed song counts. For each aggregate class, reconcile against all matching song-level class rows.

- [ ] **Step 10: Run the cohort suite**

```bash
uv run pytest tests/benchmark/test_cohort_scoring.py -q
```

Expected: PASS.

- [ ] **Step 11: Commit**

```bash
git add src/benchmark/cohort_scoring.py tests/benchmark/test_cohort_scoring.py
git commit -m "feat: aggregate cohort benchmark results"
```

---

## Task 5: Replace the flat report writer with six deterministic artifacts

**Files:**
- Modify: `src/benchmark/reports.py`
- Modify: `tests/benchmark/test_reports.py`

**Interfaces:**
- Consumes: `CohortScoreResult`.
- Produces:

```python
REPORT_SCHEMA = "crux.single-cohort-report/v1"

@dataclass(frozen=True)
class ReportArtifacts:
    summary_json: Path
    items_csv: Path
    per_song_csv: Path
    per_class_csv: Path
    event_diagnostics_jsonl: Path
    summary_markdown: Path


def write_cohort_reports(
    result: CohortScoreResult,
    output_dir: Path,
) -> ReportArtifacts: ...
```

- [ ] **Step 1: Replace the legacy report test with the six-file contract**

Build the result through `score_cohort()` and assert paths:

```python
assert artifacts.summary_json == tmp_path / "summary.json"
assert artifacts.items_csv == tmp_path / "items.csv"
assert artifacts.per_song_csv == tmp_path / "per_song.csv"
assert artifacts.per_class_csv == tmp_path / "per_class.csv"
assert artifacts.event_diagnostics_jsonl == tmp_path / "event_diagnostics.jsonl"
assert artifacts.summary_markdown == tmp_path / "summary.md"
```

Delete legacy `ChartReport` / `write_reports` tests and imports.

- [ ] **Step 2: Define and test the repeated traceability prefix**

Every CSV score row and event JSONL row repeats these fields:

```text
cohort_id
reference_manifest_sha256
reference_timing_version
taxonomy_version
lane_map_version
backend_id
model_id
model_lock_sha256
backend_descriptor_sha256
prediction_map_version
input_view_id
scoring_version
```

Use one `_identity_fields(result.identity)` helper in `reports.py`; do not retype the mapping in each writer.

- [ ] **Step 3: Test `items.csv` as the complete cohort ledger**

After the identity prefix, fixed fields are:

```text
simfile_id,status,failure_reason,warnings,
reference_native_event_count,reference_common_event_count,
reference_ignored_event_count,reference_unmapped_event_count,
prediction_native_event_count,prediction_mapped_event_count,
prediction_unmapped_event_count,prediction_mapping_coverage,prediction_native_class_counts
```

Encode warnings as `|`-joined sorted strings. Encode native class counts as `class=count|class=count`. If prediction counts are unavailable, leave corresponding cells empty. `prediction_mapping_coverage` is mapped/native when native > 0, `1.0` when both are zero for a successful empty prediction artifact, and empty when no prediction artifact exists.

- [ ] **Step 4: Test `per_song.csv` fixed fields**

After identity prefix:

```text
simfile_id,tolerance_ms,mode,tp,fp,fn,precision,recall,f1,
prediction_to_reference_ratio,median_abs_error_ms,p95_abs_error_ms,offset_ms,warnings
```

Only successful score rows appear here; failures remain explicit in `items.csv` and `summary.json`.

- [ ] **Step 5: Test `per_class.csv` fixed fields**

After identity prefix:

```text
scope,simfile_id,tolerance_ms,mode,common_class,tp,fp,fn,
reference_support,prediction_support,precision,recall,f1
```

`scope` is `song` or `cohort`; cohort rows have an empty `simfile_id` cell.

- [ ] **Step 6: Test canonical event JSONL with original/aligned times**

After the identity fields each line has:

```text
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

Use `canonical_json_bytes(..., trailing_newline=True)` for each line. Empty diagnostics produce an empty file; do not invent a header record.

- [ ] **Step 7: Test `summary.json` exact top-level contract**

Require top-level keys:

```text
schema
identity
tolerances_ms
population
items
aggregates
```

Each aggregate contains event-micro metrics, song/class macro F1, F1 distribution, successful song count, and cohort per-class rows. Each item includes status/failure/warnings/coverage. Use canonical JSON bytes, not `json.dumps(indent=...)`.

- [ ] **Step 8: Write byte-determinism tests**

Write the same result into `first/` and `second/` and compare all six file byte streams. Then reverse input `CohortItem` order, rescore, and require the same six bytes again.

- [ ] **Step 9: Verify RED**

```bash
uv run pytest tests/benchmark/test_reports.py -q
```

Expected: FAIL against the old writer.

- [ ] **Step 10: Rewrite `reports.py` around the cohort result**

Use `canonical_json_bytes()` for JSON and JSONL. Use explicit `csv.DictWriter(fieldnames=...)` lists. Milliseconds are derived only while rendering from second-based domain values.

A small metric helper is sufficient:

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

- [ ] **Step 11: Render Markdown from the same result object**

Fixed sections:

```text
# Single-Cohort Drum Benchmark
## Identity
## Population
## Aggregate Results
## Per-Class Aggregate Results
## Song F1 Distribution
## Song Extremes
```

For `Song Extremes`, show up to five lowest and five highest **raw 50 ms** successful song F1 rows, sorted by `(f1, simfile_id)`. If no successful song exists, write `No successful songs.`. Do not add outlier detection or chart generation.

- [ ] **Step 12: Run report/cohort tests**

```bash
uv run pytest tests/benchmark/test_reports.py tests/benchmark/test_cohort_scoring.py -q
```

Expected: PASS.

- [ ] **Step 13: Commit**

```bash
git add src/benchmark/reports.py tests/benchmark/test_reports.py
git commit -m "feat: write cohort benchmark reports"
```

---

## Task 6: Prove HPA-324/HPA-423 persisted artifacts rescore end-to-end without inference

**Files:**
- Create: `tests/benchmark/test_cohort_scoring_acceptance.py`
- Modify only if the test exposes a missing narrow contract: `src/benchmark/scorer_input.py`, `src/benchmark/cohort_scoring.py`, or `src/benchmark/reports.py`

**Interfaces:**
- Consumes: `map_reference_events()`, `reference_to_benchmark_events()`, `map_oaf_prediction()`, prediction artifact v2, `prediction_to_benchmark_events()`, `coverage_from_artifacts()`, `score_cohort()`, and `write_cohort_reports()`.
- Produces: one persisted-event acceptance proof; importing/starting `OafBackend`, Docker, or `WorkerProcess` is forbidden in this test.

- [ ] **Step 1: Build HPA-323-shaped reference events with an exact common collapse**

Use:

```python
def native_reference(lane_id: str, time_sec: float, source_order: int) -> NativeReferenceEvent:
    return NativeReferenceEvent(
        simfile_id=42,
        selected_chart_key="42/real.dtx",
        selected_chart_content_hash="1" * 64,
        source_audio_key="42/bgm.ogg",
        source_audio_content_hash="2" * 64,
        source_order=source_order,
        measure=1,
        position=0.0,
        lane_id=lane_id,
        note_id="01",
        chart_time_sec=time_sec,
        audio_time_sec=time_sec,
    )
```

Create kick lane `13` at `0.5`, high-tom lane `14` at `1.0`, and low/floor-tom lane `15` also at `1.0`. After `map_reference_events()` the two native tom events collapse to one common tom event. Assert:

```python
assert len(reference_mapping.mapped_events) == 3
assert len(reference_mapping.common_events) == 2
assert reference_mapping.diagnostics.duplicate_common_event_count == 1
```

Then convert common events with `reference_to_benchmark_events("42", ...)`.

- [ ] **Step 2: Build and round-trip matching real-shaped OaF predictions**

Follow `tests/benchmark/test_task_d_contract.py` for the in-memory canonical WAV, descriptor payload, and `NativePrediction`. Create one kick and one `toms` event at the same reference times, call:

```python
mapped_prediction, mapping_diagnostics = map_oaf_prediction(native_prediction)
artifact = read_prediction_artifact(render_prediction_artifact(mapped_prediction))
prediction_events = prediction_to_benchmark_events(artifact)
```

Assert both predictions are mapped to common `kick`/`tom` and `mapping_diagnostics.unmapped == {}`.

- [ ] **Step 3: Derive coverage from the actual reference mapping and prediction artifact**

```python
coverage = coverage_from_artifacts(reference_mapping, artifact)
assert coverage.reference_native_event_count == 3
assert coverage.reference_common_event_count == 2
assert coverage.reference_unmapped_event_count == 0
assert coverage.prediction_native_event_count == 2
assert coverage.prediction_mapped_event_count == 2
assert coverage.prediction_unmapped_event_count == 0
```

This proves duplicate common collapse and native prediction coverage survive as separate diagnostics.

- [ ] **Step 4: Score without a backend call**

Create one `CohortItem(status="success", ...)` and an identity using the actual taxonomy/lane/prediction-map constants. At raw 50 ms assert TP/FP/FN `(2, 0, 0)` and F1 `1.0`.

The test must not import `src.benchmark.backends.oaf.OafBackend` or `src.benchmark.worker_process`.

- [ ] **Step 5: Prove identity and full-population data in reports**

Write reports and parse canonical `summary.json`. Assert identity includes reference manifest hash, reference timing version, taxonomy/lane map, backend/model, model lock, backend descriptor hash, prediction map, input view, and `SCORING_VERSION` exactly as passed.

Assert `items[0]` retains all reference/prediction coverage counts and `population.total_count == population.success_count == 1`.

- [ ] **Step 6: Reconcile report rows**

At raw 50 ms:

- `per_song.csv` TP/FP/FN are `2/0/0`;
- cohort aggregate TP/FP/FN are `2/0/0`;
- per-class kick and tom reference supports sum to `2`;
- event diagnostics contain exactly two `matched` raw-50-ms rows;
- every CSV/diagnostic row carries `OAF_PREDICTION_MAP_ID`, `input_view_id`, model lock, and `SCORING_VERSION`.

- [ ] **Step 7: Run acceptance plus foundation contract suites**

```bash
uv run pytest \
  tests/benchmark/test_cohort_scoring_acceptance.py \
  tests/benchmark/test_reference_set.py \
  tests/benchmark/test_task_d_contract.py -q
```

Expected: PASS.

- [ ] **Step 8: Commit**

```bash
git add tests/benchmark/test_cohort_scoring_acceptance.py \
  src/benchmark/scorer_input.py src/benchmark/cohort_scoring.py src/benchmark/reports.py
git commit -m "test: prove persisted cohort rescoring"
```

If the source files were unchanged in this task, omit them from `git add`.

---

## Task 7: Final reconciliation and repository verification

**Files:**
- Verify all HPA-325 files above.
- No new production files should be introduced here.

- [ ] **Step 1: Run the complete benchmark suite**

```bash
uv run pytest tests/benchmark -q
```

Expected: PASS.

- [ ] **Step 2: Run the complete repository suite**

```bash
uv run pytest
```

Expected: PASS.

- [ ] **Step 3: Run repository lint/format gates**

```bash
uv run ruff check src tests
uv run ruff format --check src tests
uv run pylint src/app src/cli
```

Expected: PASS under the documented repository gate set.

- [ ] **Step 4: Check whitespace and scope**

```bash
git diff --check main...HEAD
git status --short
git diff --stat main...HEAD
```

Expected production/test scope:

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

Plus the HPA-325 design/plan docs if implementation occurs on the planning branch. Any additional production file requires an explicit scope reason before merge.

- [ ] **Step 5: Run the focused acceptance contract**

```bash
uv run pytest \
  tests/benchmark/test_scorer_input.py \
  tests/benchmark/test_scoring.py \
  tests/benchmark/test_cohort_scoring.py \
  tests/benchmark/test_cohort_scoring_acceptance.py \
  tests/benchmark/test_reports.py -q
```

Expected coverage:

- persisted reference and prediction scorer adapters;
- simultaneous/dense/empty matcher behavior;
- exact duplicate common-reference collapse;
- 30/50/100 ms × raw/aligned matrix;
- original vs aligned prediction diagnostic times;
- per-song/per-class reconciliation;
- native/mapped/unmapped coverage;
- event-micro/song-macro/class-macro semantics;
- p10/p25/median/p75/p90 song-F1 distributions;
- full failed/skipped/quarantined population accounting;
- matched/FP/FN diagnostics;
- row-level identity traceability;
- byte-identical report reproduction independent of input ordering;
- persisted-artifact rescoring without inference.

- [ ] **Step 6: Commit only if verification required cleanup**

```bash
git add src/benchmark tests/benchmark
git commit -m "test: finalize HPA-325 scoring contract"
```

Skip this commit when verification produces no edits.

## Completion Boundary

HPA-325 is complete when the deterministic in-memory scorer and six report artifacts are implemented and verified as above. Do **not** add a corpus-run command or define a prediction-run manifest to finish HPA-325. HPA-326 should reuse this boundary by adapting each run/reference/prediction row into `CohortItem` and passing one frozen `CohortIdentity` into `score_cohort()` after prediction persistence.

That handoff keeps HPA-326 focused on execution/resume/runtime measurement and HPA-325 focused on scoring semantics, preventing two overlapping orchestration layers.
