# HPA-325 Single-Cohort Scoring and Reports Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn one frozen model/input-view cohort of persisted reference and prediction events into deterministic per-song, per-class, aggregate, coverage, population, and event-diagnostic reports at 30/50/100 ms without invoking inference or rebuilding the existing matcher/mapping seams.

**Architecture:** Keep `src/benchmark/scoring.py` as the sole onset-matching/alignment implementation; only promote its existing private percentile helper to a public name so aggregate distributions reuse the same rank convention. Add one small `cohort_scoring.py` module for validated in-memory cohort rows, artifact coverage, score aggregation, and population accounting. Extend `scorer_input.py` only with the reference-side common-event adapter. Rewrite `reports.py` as a deterministic renderer that reuses `quantize_six()`, `require_sha256()`, and `canonical_json_bytes()` rather than inventing parallel numeric or identity encoders.

**Tech Stack:** Python 3.12, frozen dataclasses, `typing.Literal`, `statistics.fmean`/`median`, existing `BenchmarkEvent`/`ScoreSummary`/`ScoreResult`, existing `require_sha256`/`quantize_six`/canonical JSON helpers, CSV, pathlib, pytest, Ruff, Pylint.

## Global Constraints

- Consume merged HPA-323 audio-relative native reference artifacts; do not rerun DTX parsing, timing, BGM resolution, chart selection, R2 access, or audio decoding.
- Consume HPA-324 `map_reference_events()` / `CommonReferenceEvent` and the frozen common taxonomy; do not define another lane or class map.
- Consume HPA-423 prediction artifact v2 and `prediction_to_benchmark_events()`; do not rebuild `NativeEvent` mapping or prediction persistence.
- The scored identity is one model/checkpoint + one `input_view_id` + one population scope per cohort.
- Default tolerances are exactly `30`, `50`, and `100` milliseconds.
- Raw and diagnostically aligned scores are separate named modes; alignment never replaces raw scoring.
- Headline matching uses onset time + common class only. Confidence and velocity remain diagnostic metadata and never affect matching.
- Failed, skipped, and quarantined items remain visible in population accounting; never silently reduce the cohort to successes.
- Do not change `score_events()`, `_match_class()`, alignment search, or their semantics. The only `scoring.py` change allowed is renaming/exporting the existing percentile helper and updating its internal call site.
- Reuse `require_sha256()` for SHA-256 identity validation. Do not copy lowercase-hex validators.
- Reuse the existing scoring percentile convention. Do not create a second percentile implementation.
- Every float-derived report number crosses one `quantize_six()` rendering boundary. JSON/JSONL receive `Decimal`; CSV receives the same canonical decimal token. Never pass Python `float` to `canonical_json_bytes()`.
- Undefined P/R/F1 remain `None`; do not coalesce them to `0.0`.
- Full identity appears once in `summary.json` and on `items.csv` / `per_song.csv` / `per_class.csv` score/ledger rows. `event_diagnostics.jsonl` carries `cohort_id` only plus event evidence.
- Do not add inference, worker pools, queues, retries, a prediction-run manifest, cross-model comparison, Parquet, a database, a metrics service, a report plugin registry, or a generic benchmark pipeline.
- Breaking `ChartReport` / `write_reports` is allowed; do not add a compatibility wrapper.
- Reports are regenerable files; do not use `publish_immutable_file()`.
- Repository verification follows `CLAUDE.md`: `uv run pytest`, `uv run ruff check src tests`, `uv run ruff format --check src tests`, and `uv run pylint src/app src/cli`.

---

## File Map

### Create

- `src/benchmark/cohort_scoring.py` — identity/item validation, artifact coverage, song/class scoring derivation, aggregates, F1 distributions, and population accounting.
- `tests/benchmark/test_scorer_input.py` — focused reference-side scorer adapter tests.
- `tests/benchmark/test_cohort_scoring.py` — identity, coverage, scoring matrix, aggregation, percentile, and reconciliation tests.
- `tests/benchmark/test_cohort_scoring_acceptance.py` — persisted HPA-324/HPA-423 artifact-to-report proof with no backend.

### Modify

- `src/benchmark/scorer_input.py` — add only `reference_to_benchmark_events()`.
- `src/benchmark/scoring.py` — promote `_percentile()` to `percentile()` and use that name internally; no matcher/alignment behavior change.
- `src/benchmark/reports.py` — replace flat chart rendering with deterministic cohort rendering and the shared report-number boundary.
- `tests/benchmark/test_scoring.py` — pin remaining matcher behavior and public percentile convention.
- `tests/benchmark/test_reports.py` — replace legacy report tests with schema/encoding/determinism tests.

### Explicitly unchanged

- `src/benchmark/models.py`
- `src/benchmark/reference_set.py`
- `src/benchmark/reference_set_manifest.py`
- `src/benchmark/prediction_artifact.py`
- `src/benchmark/mapping.py`
- `src/benchmark/backends/oaf.py`
- `src/cli/benchmark.py`

---

## Task 1: Pin matcher behavior and add the reference scorer adapter

**Files:**
- Modify: `tests/benchmark/test_scoring.py`
- Modify: `src/benchmark/scorer_input.py`
- Create: `tests/benchmark/test_scorer_input.py`

**Interfaces:**
- Consumes: `CommonReferenceEvent`, `BenchmarkEvent`, existing prediction scorer adapters.
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

Keep the existing dense-hit and large-offset tests.

- [ ] **Step 2: Run matcher regressions before touching scoring code**

```bash
uv run pytest tests/benchmark/test_scoring.py -q
```

Expected: PASS. If a new matcher regression fails, treat that as a separate correctness finding before continuing; do not compensate in cohort aggregation.

- [ ] **Step 3: Write the failing reference adapter tests**

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

    assert [(item.chart_id, item.time_sec, item.canonical_class, item.source) for item in result] == [
        ("42", 0.5, "kick", "ground_truth"),
        ("42", 1.25, "hihat", "ground_truth"),
    ]


def test_reference_adapter_rejects_empty_simfile_id() -> None:
    with pytest.raises(ValueError, match="simfile_id must be a nonempty string"):
        reference_to_benchmark_events("", ())
```

- [ ] **Step 4: Verify RED**

```bash
uv run pytest tests/benchmark/test_scorer_input.py -q
```

Expected: FAIL because `reference_to_benchmark_events` does not exist.

- [ ] **Step 5: Implement the minimal reference adapter**

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
            time_sec=float(item.canonical_audio_time),
            canonical_class=item.common_class,
            source="ground_truth",
            metadata={},
        )
        for item in common_events
    )
```

Add it to `__all__`. Do not accept native DTX/reference events here; callers must first use HPA-324 common projection.

- [ ] **Step 6: Run scorer-input + HPA-423 bridge tests**

```bash
uv run pytest tests/benchmark/test_scorer_input.py \
  tests/benchmark/test_task_d_contract.py tests/benchmark/test_scoring.py -q
```

Expected: PASS, including existing OaF hihat/tom common-class round-trip coverage.

- [ ] **Step 7: Commit**

```bash
git add src/benchmark/scorer_input.py tests/benchmark/test_scorer_input.py \
  tests/benchmark/test_scoring.py
git commit -m "test: pin cohort scorer inputs"
```

---

## Task 2: Define cohort identity, item, and artifact-coverage contracts

**Files:**
- Create: `src/benchmark/cohort_scoring.py`
- Create: `tests/benchmark/test_cohort_scoring.py`

**Interfaces:**
- Consumes: `ReferenceMappingResult`, `PredictionArtifact`, `BenchmarkEvent`, `require_sha256()`.
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

- [ ] **Step 1: Write a shared identity fixture**

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

- [ ] **Step 2: Write shared SHA-validation tests**

Parametrize the three raw SHA fields:

```python
@pytest.mark.parametrize(
    "field",
    ["reference_manifest_sha256", "model_lock_sha256", "backend_descriptor_sha256"],
)
def test_identity_rejects_invalid_sha256_through_shared_contract(field: str) -> None:
    values = dataclasses.asdict(identity())
    values[field] = "ABC"

    with pytest.raises(ValueError, match=f"{field} must be lowercase SHA-256"):
        CohortIdentity(**values)
```

This test intentionally locks the existing `require_sha256()` message. Do not write a local regex/hex loop.

- [ ] **Step 3: Write artifact coverage tests using real domain objects**

Reuse `NativeReferenceEvent` fixtures from `tests/benchmark/test_reference_set.py` and OaF mapped prediction fixtures from `test_task_d_contract.py`. Build mapped lane `13`, ignored lane `54`, one mapped prediction, and one unmapped prediction.

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

- [ ] **Step 4: Verify RED**

```bash
uv run pytest tests/benchmark/test_cohort_scoring.py -q
```

Expected: FAIL because `cohort_scoring.py` does not exist.

- [ ] **Step 5: Implement identity validation by reusing `require_sha256()`**

In `CohortIdentity.__post_init__`, require all non-hash identity strings to be nonempty, require `scoring_version == SCORING_VERSION`, then validate hashes:

```python
for field in (
    "reference_manifest_sha256",
    "model_lock_sha256",
    "backend_descriptor_sha256",
):
    try:
        require_sha256(getattr(self, field), field)
    except StrictJsonError as error:
        raise ValueError(str(error)) from None
```

No new SHA helper.

- [ ] **Step 6: Implement `coverage_from_artifacts()`**

Reference native count:

```python
len(reference.mapped_events) + sum(reference.diagnostics.ignored.values()) + sum(
    reference.diagnostics.unmapped.values()
)
```

Prediction counts:

```python
native_events = artifact.prediction.events
mapped_count = sum(item.mapping_status == "mapped" for item in native_events)
unmapped_count = sum(item.mapping_status == "unmapped" for item in native_events)
native_class_counts = tuple(
    sorted(Counter(item.native.native_class_id for item in native_events).items())
)
```

Do not create another coverage artifact.

- [ ] **Step 7: Write and implement item/tolerance validation**

Tests require:

- unique nonempty `simfile_id` values;
- sorted unique positive tolerances;
- `success` rows have nonempty references and `prediction_events is not None`; empty prediction tuple is valid;
- non-success rows have `prediction_events is None` and nonempty `failure_reason`;
- success rows have `failure_reason is None`;
- success prediction coverage is non-`None`, balances `native == mapped + unmapped`, and `mapped == len(prediction_events)`;
- reference common count equals `len(reference_events)`.

Use direct `ValueError`; no validation framework.

- [ ] **Step 8: Run the contract suite**

```bash
uv run pytest tests/benchmark/test_cohort_scoring.py -q
```

Expected: PASS for identity/coverage/validation tests.

- [ ] **Step 9: Commit**

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

- [ ] **Step 1: Write the fixed tolerance/mode matrix test**

Use one success row with a 40 ms-late kick prediction. Require sorted score dimensions:

```python
[
    (30, "aligned"), (30, "raw"),
    (50, "aligned"), (50, "raw"),
    (100, "aligned"), (100, "raw"),
]
```

Require raw 30 ms F1 `0.0`, aligned 30 ms F1 `1.0`, and raw 50/100 ms F1 `1.0`.

- [ ] **Step 2: Write per-class reconciliation tests**

Use kick + snare references and an unmatched hihat prediction. For every `SongScore`:

```python
assert sum(row.summary.true_positives for row in score.per_class) == score.summary.true_positives
assert sum(row.summary.false_positives for row in score.per_class) == score.summary.false_positives
assert sum(row.summary.false_negatives for row in score.per_class) == score.summary.false_negatives
```

Require hihat to remain present with zero reference support.

- [ ] **Step 3: Write scorer metadata-independence test**

Compare predictions with `metadata={}` against otherwise identical predictions carrying confidence/velocity metadata. Counts and F1 must match. Do not create an invalid OaF artifact with null OaF fields; this is a scorer-level contract.

- [ ] **Step 4: Write aligned original/scored-time regression**

For reference `1.0`, persisted prediction `1.1`, and aligned offset `-0.1`:

```python
assert diagnostic.reference_time_sec == pytest.approx(1.0)
assert diagnostic.prediction_time_sec == pytest.approx(1.1)
assert diagnostic.scored_prediction_time_sec == pytest.approx(1.0)
assert diagnostic.timing_error_sec == pytest.approx(0.0)
```

- [ ] **Step 5: Verify RED**

```bash
uv run pytest tests/benchmark/test_cohort_scoring.py -q
```

Expected: FAIL on missing score rows/diagnostics.

- [ ] **Step 6: Implement one matcher invocation per success/tolerance**

```python
aligned = score_events_with_alignment(
    list(item.reference_events),
    list(item.prediction_events),
    tolerance_sec=tolerance_ms / 1000.0,
)
```

Turn `aligned.raw` and `aligned.aligned` into separate `SongScore` rows. Do not invoke the matcher again for classes.

- [ ] **Step 7: Derive per-class rows from the same `ScoreResult`**

For every class present in matches or either unmatched set:

- TP = matched rows for class;
- FP = unmatched predictions for class;
- FN = unmatched references for class;
- reference support = TP + FN;
- prediction support = TP + FP.

Use `ScoreSummary(tp, fp, fn)` for class P/R/F1. Do not add class timing distributions.

- [ ] **Step 8: Derive diagnostics from the same `ScoreResult`**

For raw mode, original/scored prediction time are identical. For aligned mode:

```python
original_prediction_time = scored_prediction_time - score_result.summary.offset_sec
```

Apply this to matched and FP rows. FN rows have prediction fields `None`; FP rows have reference/error fields `None`.

Sort diagnostics by `(simfile_id, tolerance_ms, mode, outcome, common_class, reference_time_sec or -1, prediction_time_sec or -1)`.

- [ ] **Step 9: Run focused scoring tests**

```bash
uv run pytest tests/benchmark/test_cohort_scoring.py tests/benchmark/test_scoring.py -q
```

Expected: PASS without matcher/alignment changes.

- [ ] **Step 10: Commit**

```bash
git add src/benchmark/cohort_scoring.py tests/benchmark/test_cohort_scoring.py
git commit -m "feat: score single benchmark cohorts"
```

---

## Task 4: Reuse the scorer percentile helper and add aggregates/population

**Files:**
- Modify: `src/benchmark/scoring.py`
- Modify: `tests/benchmark/test_scoring.py`
- Modify: `src/benchmark/cohort_scoring.py`
- Modify: `tests/benchmark/test_cohort_scoring.py`

**Interfaces:**
- Consumes: existing `_percentile` implementation, all `CohortItem` and `SongScore` rows.
- Produces:

```python
def percentile(values: list[float], percentile: float) -> float: ...

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

- [ ] **Step 1: Pin the existing percentile convention under its public name**

In `tests/benchmark/test_scoring.py` import `percentile` and add:

```python
def test_percentile_uses_existing_upper_nearest_rank_convention() -> None:
    values = [0.0, 0.25, 0.5, 0.75, 1.0]

    assert percentile(values, 0.10) == 0.25
    assert percentile(values, 0.25) == 0.25
    assert percentile(values, 0.75) == 0.75
    assert percentile(values, 0.90) == 1.0
```

- [ ] **Step 2: Verify RED for the public name**

```bash
uv run pytest tests/benchmark/test_scoring.py::test_percentile_uses_existing_upper_nearest_rank_convention -q
```

Expected: FAIL because only `_percentile` exists.

- [ ] **Step 3: Promote `_percentile` to `percentile` without semantic change**

Rename:

```python
def percentile(values: list[float], percentile: float) -> float:
    if not values:
        raise ValueError("values must not be empty")
    index = min(len(values) - 1, ceil((len(values) - 1) * percentile))
    return values[index]
```

Update `score_events()` to call `percentile(errors, 0.95)`. Do not leave a compatibility alias; repository search shows no other consumer.

- [ ] **Step 4: Write full-population accounting test**

Use one success, one failed, one skipped, and one quarantined item:

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

Warnings remain warnings, not failure reasons.

- [ ] **Step 5: Write event-micro/song-macro/class-macro test**

Use two successful songs:

- song A: kick TP, snare FN;
- song B: kick TP plus one kick FP.

Require aggregate TP/FP/FN `(2, 1, 1)`, song macro as arithmetic mean of song F1 values, and class macro as arithmetic mean of supported aggregate class F1 values.

- [ ] **Step 6: Write exact F1 distribution tests using `percentile()`**

For successful song F1 values `(0.0, 0.25, 0.5, 0.75, 1.0)`:

```python
assert distribution.minimum == 0.0
assert distribution.p10 == 0.25
assert distribution.p25 == 0.25
assert distribution.median == 0.5
assert distribution.p75 == 0.75
assert distribution.p90 == 1.0
assert distribution.maximum == 1.0
```

Use `statistics.median()` only for median; p10/p25/p75/p90 must call `scoring.percentile()`.

- [ ] **Step 7: Write zero-success test**

A cohort containing only failed/quarantined rows still returns six aggregate dimensions. Require micro counts zero with P/R/F1 `None`, song/class macro `None`, empty class rows, and every `F1Distribution` field `None`.

- [ ] **Step 8: Verify RED for aggregate behavior**

```bash
uv run pytest tests/benchmark/test_cohort_scoring.py tests/benchmark/test_scoring.py -q
```

Expected: aggregate tests FAIL; percentile public-name test PASS after Step 3.

- [ ] **Step 9: Implement population and aggregate helpers**

Population counts are built before filtering successes. Event micro sums TP/FP/FN and constructs one `ScoreSummary`. Aggregate classes sum the same class rows across songs.

Macro helper:

```python
def _mean_f1(values: tuple[float | None, ...]) -> float | None:
    scored = tuple(value for value in values if value is not None)
    return fmean(scored) if scored else None
```

Distribution helper uses sorted successful song F1 values:

```python
F1Distribution(
    minimum=values[0],
    p10=percentile(values, 0.10),
    p25=percentile(values, 0.25),
    median=median(values),
    p75=percentile(values, 0.75),
    p90=percentile(values, 0.90),
    maximum=values[-1],
)
```

- [ ] **Step 10: Add reconciliation assertions in tests**

For all six `(tolerance_ms, mode)` aggregates, require TP/FP/FN equal summed song counts. For each aggregate class, reconcile against song-level class rows.

- [ ] **Step 11: Run the aggregate suite**

```bash
uv run pytest tests/benchmark/test_scoring.py tests/benchmark/test_cohort_scoring.py -q
```

Expected: PASS.

- [ ] **Step 12: Commit**

```bash
git add src/benchmark/scoring.py tests/benchmark/test_scoring.py \
  src/benchmark/cohort_scoring.py tests/benchmark/test_cohort_scoring.py
git commit -m "feat: aggregate cohort benchmark results"
```

---

## Task 5: Replace the flat report writer and freeze numeric/identity encoding

**Files:**
- Modify: `src/benchmark/reports.py`
- Modify: `tests/benchmark/test_reports.py`

**Interfaces:**
- Consumes: `CohortScoreResult`, `quantize_six()`, `canonical_json_bytes()`.
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

Build a result through `score_cohort()` and require:

```python
assert artifacts.summary_json == tmp_path / "summary.json"
assert artifacts.items_csv == tmp_path / "items.csv"
assert artifacts.per_song_csv == tmp_path / "per_song.csv"
assert artifacts.per_class_csv == tmp_path / "per_class.csv"
assert artifacts.event_diagnostics_jsonl == tmp_path / "event_diagnostics.jsonl"
assert artifacts.summary_markdown == tmp_path / "summary.md"
```

Delete legacy `ChartReport` / `write_reports` tests and imports.

- [ ] **Step 2: Define one traceability helper for durable CSV rows**

`summary.json` stores the full identity once. `items.csv`, `per_song.csv`, and `per_class.csv` repeat this prefix:

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

Implement one `_identity_fields(identity)` helper and reuse it for the three CSV writers.

`event_diagnostics.jsonl` does **not** use `_identity_fields`; it carries `cohort_id` only and joins to `summary.json`.

- [ ] **Step 3: Write the report-number helper tests before writer implementation**

In `tests/benchmark/test_reports.py` pin:

```python
assert _report_decimal(0.5) == Decimal("0.500000")
assert _report_decimal(None) is None
assert _csv_decimal(0.5) == "0.5"
assert _csv_decimal(None) == ""
```

Also require a nonfinite input to raise `StrictJsonError` through `quantize_six()`.

- [ ] **Step 4: Verify RED for numeric helpers**

```bash
uv run pytest tests/benchmark/test_reports.py -q
```

Expected: FAIL because the new report helpers/writer do not exist.

- [ ] **Step 5: Implement the single numeric boundary**

In `src/benchmark/reports.py`:

```python
def _report_decimal(value: float | None) -> Decimal | None:
    return None if value is None else quantize_six(value)


def _csv_decimal(value: float | None) -> str:
    decimal = _report_decimal(value)
    return "" if decimal is None else canonical_json_bytes(decimal).decode("ascii")
```

Rules for every writer:

- float-derived metrics/times/ratios/coverage percentages -> `_report_decimal()` for JSON/JSONL;
- the same values -> `_csv_decimal()` for CSV/Markdown numeric tokens;
- `None` -> JSON `null` / empty CSV cell;
- counts -> integer;
- never pass Python `float` to `canonical_json_bytes()`;
- never use `str(float)` or `repr(float)` as report encoding.

- [ ] **Step 6: Test `items.csv` as the complete cohort ledger**

After the identity prefix, fixed fields are:

```text
simfile_id,status,failure_reason,warnings,
reference_native_event_count,reference_common_event_count,
reference_ignored_event_count,reference_unmapped_event_count,
prediction_native_event_count,prediction_mapped_event_count,
prediction_unmapped_event_count,prediction_mapping_coverage,prediction_native_class_counts
```

Warnings are sorted and `|`-joined. Native class counts are `class=count|class=count`. Missing prediction counts are empty cells. Mapping coverage is mapped/native; when a successful artifact has zero native events, encode `1.0`; when no artifact exists, leave empty.

Require a fixture with one-half coverage to produce the exact CSV cell:

```python
assert row["prediction_mapping_coverage"] == "0.5"
```

- [ ] **Step 7: Test `per_song.csv` and `per_class.csv` fields**

`per_song.csv`, after identity prefix:

```text
simfile_id,tolerance_ms,mode,tp,fp,fn,precision,recall,f1,
prediction_to_reference_ratio,median_abs_error_ms,p95_abs_error_ms,offset_ms,warnings
```

`per_class.csv`, after identity prefix:

```text
scope,simfile_id,tolerance_ms,mode,common_class,tp,fp,fn,
reference_support,prediction_support,precision,recall,f1
```

`scope` is `song` or `cohort`; cohort rows have empty `simfile_id`.

Use a fixture where precision is one-half and lock:

```python
assert raw_50_row["precision"] == "0.5"
```

Do not emit `0.500000` or a binary-float representation.

- [ ] **Step 8: Test the slim canonical event JSONL schema with a golden line**

Each line contains exactly:

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

Use `canonical_json_bytes(..., trailing_newline=True)` per line. Empty diagnostics produce an empty file; no header record.

For a raw matched kick at 0.5 s, require this exact canonical line to occur:

```python
expected = (
    b'{"cohort_id":"oaf-full-mix-v1","common_class":"kick","mode":"raw",'
    b'"outcome":"matched","prediction_time_sec":0.5,"reference_time_sec":0.5,'
    b'"scored_prediction_time_sec":0.5,"simfile_id":"1","timing_error_sec":0,'
    b'"tolerance_ms":50}'
)
assert expected in (tmp_path / "event_diagnostics.jsonl").read_bytes().splitlines()
```

This golden locks `quantize_six()` + canonical decimal rendering and the smaller diagnostic identity shape.

- [ ] **Step 9: Test `summary.json` exact top-level contract**

Require:

```text
schema
identity
tolerances_ms
population
items
aggregates
```

All float-derived values inside items/aggregates are `Decimal` before `canonical_json_bytes()`. Parse with `strict_json_loads(..., require_canonical=True)` and assert a one-half metric becomes `Decimal("0.5")`; undefined metrics become `None`.

- [ ] **Step 10: Implement the three CSV writers, JSONL writer, and summary renderer**

Use explicit fixed `csv.DictWriter(fieldnames=...)` lists and stable row sorting. Build JSON/JSONL dictionaries only from JSON-safe scalars: ints/strings/bools/`None`/quantized `Decimal`.

For score metrics, use a helper returning quantized values:

```python
def _metric(summary: ScoreSummary) -> dict[str, JsonValue]:
    return {
        "tp": summary.true_positives,
        "fp": summary.false_positives,
        "fn": summary.false_negatives,
        "precision": _report_decimal(summary.precision),
        "recall": _report_decimal(summary.recall),
        "f1": _report_decimal(summary.f1),
    }
```

Apply the same boundary to ratios, milliseconds, offsets, F1 distributions, and event diagnostic times/errors.

- [ ] **Step 11: Render Markdown from the same result and numeric tokens**

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

Use the same canonical numeric token helper rather than independent float formatting. For `Song Extremes`, show up to five lowest/highest raw 50 ms successful songs sorted by `(f1, simfile_id)`. If none, write `No successful songs.`.

- [ ] **Step 12: Write byte-determinism tests**

Write the same result into `first/` and `second/`; compare all six file byte streams. Reverse input `CohortItem` order, rescore, and require the same bytes again.

- [ ] **Step 13: Run report/cohort tests**

```bash
uv run pytest tests/benchmark/test_reports.py tests/benchmark/test_cohort_scoring.py -q
```

Expected: PASS with no Python float reaching canonical JSON.

- [ ] **Step 14: Commit**

```bash
git add src/benchmark/reports.py tests/benchmark/test_reports.py
git commit -m "feat: write deterministic cohort reports"
```

---

## Task 6: Prove persisted HPA-324/HPA-423 artifacts rescore end-to-end without inference

**Files:**
- Create: `tests/benchmark/test_cohort_scoring_acceptance.py`
- Modify only if the test exposes a missing narrow contract: `src/benchmark/scorer_input.py`, `src/benchmark/cohort_scoring.py`, or `src/benchmark/reports.py`

**Interfaces:**
- Consumes: `map_reference_events()`, `reference_to_benchmark_events()`, `map_oaf_prediction()`, prediction artifact v2, `prediction_to_benchmark_events()`, `coverage_from_artifacts()`, `score_cohort()`, and `write_cohort_reports()`.
- Produces: one persisted-event acceptance proof. Importing/starting `OafBackend`, Docker, or `WorkerProcess` is forbidden.

- [ ] **Step 1: Build HPA-323-shaped reference events with exact common collapse**

Create simultaneous tom lanes `14` and `15` at 1.0 s plus a kick at 2.0 s. Run `map_reference_events()` and assert:

```python
assert len(reference_mapping.mapped_events) == 3
assert len(reference_mapping.common_events) == 2
assert reference_mapping.diagnostics.duplicate_common_event_count == 1
```

This proves duplicate-after-collapse behavior before scoring.

- [ ] **Step 2: Convert the HPA-324 common references through the new adapter**

```python
reference_events = reference_to_benchmark_events(
    "7",
    reference_mapping.common_events,
)
assert [(item.time_sec, item.canonical_class) for item in reference_events] == [
    (1.0, "tom"),
    (2.0, "kick"),
]
```

- [ ] **Step 3: Build and round-trip a real-shaped OaF prediction artifact v2**

Reuse the descriptor/audio/native-event style from `tests/benchmark/test_task_d_contract.py`. Include one `toms` event at 1.0 s, one `kick` at 2.0 s, and one `sticks` unmapped event.

```python
mapped_prediction, diagnostics = map_oaf_prediction(native_prediction)
artifact = read_prediction_artifact(render_prediction_artifact(mapped_prediction))
prediction_events = prediction_to_benchmark_events(artifact)

assert diagnostics.unmapped == {"sticks": 1}
assert [(item.time_sec, item.canonical_class) for item in prediction_events] == [
    (1.0, "tom"),
    (2.0, "kick"),
]
```

- [ ] **Step 4: Derive coverage from the same artifacts**

```python
coverage = coverage_from_artifacts(reference_mapping, artifact)
assert coverage.reference_native_event_count == 3
assert coverage.reference_common_event_count == 2
assert coverage.prediction_native_event_count == 3
assert coverage.prediction_mapped_event_count == 2
assert coverage.prediction_unmapped_event_count == 1
```

- [ ] **Step 5: Score without invoking a backend**

Construct one `CohortItem(status="success", ...)`, call `score_cohort()`, and require raw 30/50/100 ms F1 `1.0` and aggregate counts `(2, 0, 0)`.

- [ ] **Step 6: Prove mapping/version identity affects report bytes without inference**

Score/write once with the normal `prediction_map_version`, then score/write the same persisted events with a deliberately different identity string such as `crux.prediction-map/test-v2`. Do not remap or transcribe. Require score counts equal but `summary.json` bytes differ and contain the changed identity.

This is the rescoring/provenance proof: prediction inference is not rerun when score identity changes.

- [ ] **Step 7: Assert canonical numeric report output**

Parse `summary.json` with `strict_json_loads(..., require_canonical=True)`. Assert report floats are parsed as `Decimal`, not rejected, and the event diagnostics contain the canonical `0.5`/`1` style tokens where applicable.

- [ ] **Step 8: Assert no backend/runtime imports in the acceptance module**

Keep imports limited to domain artifacts/mappers/scorer/reports. Do not import `src.benchmark.backends.oaf`, `WorkerProcess`, or Docker helpers.

- [ ] **Step 9: Run focused acceptance**

```bash
uv run pytest tests/benchmark/test_cohort_scoring_acceptance.py \
  tests/benchmark/test_reports.py tests/benchmark/test_cohort_scoring.py \
  tests/benchmark/test_scorer_input.py tests/benchmark/test_scoring.py -q
```

Expected: PASS with no inference/container work.

- [ ] **Step 10: Commit**

```bash
git add tests/benchmark/test_cohort_scoring_acceptance.py
git commit -m "test: prove persisted cohort rescoring"
```

---

## Task 7: Full verification and scope check

**Files:**
- No intended production changes beyond Tasks 1-6.

- [ ] **Step 1: Run full tests**

```bash
uv run pytest
```

Expected: PASS.

- [ ] **Step 2: Run repository lint/format gates**

```bash
uv run ruff check src tests
uv run ruff format --check src tests
uv run pylint src/app src/cli
```

Expected: PASS according to the repository's `CLAUDE.md` gates.

- [ ] **Step 3: Check diff hygiene**

```bash
git diff --check main...HEAD
```

Expected: no whitespace errors.

- [ ] **Step 4: Verify no forbidden architecture appeared**

Inspect `git diff --stat main...HEAD` and imports. Confirm there is no:

- CLI for HPA-325;
- inference/backend runtime change;
- run-manifest schema;
- queue/retry/concurrency layer;
- generic benchmark/report plugin framework;
- Parquet/database layer;
- copied SHA-256 validator;
- copied percentile implementation;
- custom float JSON renderer;
- immutable publication layer;
- compatibility wrapper for `ChartReport` / `write_reports`.

- [ ] **Step 5: Verify report traceability and compact diagnostics**

Check one generated fixture report and confirm:

- `summary.json` contains full identity once;
- `items.csv`, `per_song.csv`, and `per_class.csv` contain the full traceability prefix;
- `event_diagnostics.jsonl` contains only `cohort_id` plus event fields;
- every float-derived JSON/JSONL number came through `quantize_six()`;
- CSV numeric cells use the same canonical decimal tokens;
- undefined metrics remain JSON null / empty CSV cells.

- [ ] **Step 6: Commit any verification-only fixture correction if needed**

Only if a deterministic expected fixture required correction:

```bash
git add tests
 git commit -m "test: align cohort report fixtures"
```

If no correction is needed, do not create an empty commit.

---

## Self-Review

### Spec coverage

- Persisted prediction bridge: reused from HPA-423; not rebuilt.
- Reference common projection: reused from HPA-324 with one scorer adapter.
- Matcher/alignment: unchanged; adversarial behavior pinned first.
- 30/50/100 ms and raw/aligned: explicit result matrix.
- Per-song/per-class/micro/song-macro/class-macro/F1 distribution: covered.
- Native/mapped/unmapped coverage: derived from persisted artifacts.
- Full failed/skipped/quarantined population: retained in item ledger and summary.
- Matched/FP/FN evidence: compact deterministic JSONL with original/scored prediction times.
- Identity: full once in summary and repeated on durable CSV score/ledger rows; diagnostics join by `cohort_id`.
- SHA validation: shared `require_sha256()`.
- Percentiles: shared `scoring.percentile()` after a public-name promotion.
- Numeric encoding: shared `quantize_six()` + `canonical_json_bytes()`; no float JSON path.
- Determinism: golden JSONL/CSV numeric tokens plus byte-for-byte rerender tests.
- Rescoring without inference: persisted-artifact acceptance proof.
- HPA-326 runner/run-manifest work: deferred.
- HPA-562 paired comparisons: deferred.

### Placeholder scan

No TBD/TODO or implementation placeholders remain. Each code-producing task has explicit interfaces, failing tests, implementation shape, verification commands, and commit boundaries.

### Type consistency

- `CohortItem.prediction_events` is `tuple[BenchmarkEvent, ...] | None`; empty tuple means a successful zero-event prediction, `None` means no valid prediction artifact for a non-success row.
- `CohortCoverage` prediction counts follow the same distinction.
- Domain score values remain Python floats/`None`; conversion to `Decimal` happens only at report rendering.
- `ScoreSummary` keeps its existing `None` semantics.
- Event diagnostics keep original and scored prediction times as separate domain floats until the report boundary.

## Execution Handoff

Plan complete at `docs/superpowers/plans/2026-08-12-hpa-325-cohort-scoring.md`.

Recommended implementation path: use `superpowers:subagent-driven-development` task-by-task, with review after each task. Do not start HPA-326 corpus inference until the HPA-325 scorer/report contract is implemented and verified.
