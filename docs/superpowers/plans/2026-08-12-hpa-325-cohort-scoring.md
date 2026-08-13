# HPA-325 Single-Cohort Scoring and Reports Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn one frozen model/input-view cohort of persisted reference and prediction events into deterministic per-song, per-class, aggregate, coverage, failure, and optionally bounded event-diagnostic reports at 30/50/100 ms without invoking inference or rebuilding existing mapping/matching seams.

**Architecture:** Keep `src/benchmark/scoring.py` as the sole onset-matching/alignment implementation. Add one small `cohort_scoring.py` module for the in-memory cohort contract, identity/coverage validation, canonical ordering, bounded diagnostics, per-class derivation, population accounting, and aggregates. Extend `scorer_input.py` only with the reference-side common-event adapter. Rewrite the small `reports.py` API as deterministic rendering over one canonical result object; HPA-326 later adapts its run rows into `CohortItem` rather than HPA-325 inventing a run schema.

**Tech Stack:** Python 3.12, frozen dataclasses, `typing.Literal`, `statistics.fmean`/`median`, existing `BenchmarkEvent`/`ScoreSummary`/`ScoreResult`, existing `require_sha256()` / `quantize_six()` / canonical JSON helpers, CSV, pathlib, pytest, Ruff, Pylint.

## Global Constraints

- Consume merged HPA-323 audio-relative native reference artifacts; do not rerun DTX parsing, timing, BGM resolution, chart selection, R2 access, or audio decoding.
- Consume HPA-324 `map_reference_events()` / `CommonReferenceEvent`; do not define another lane/class map or mapped-reference artifact.
- Consume HPA-423 prediction artifact v2 and `prediction_to_benchmark_events()`; do not rebuild `NativeEvent` mapping or prediction persistence.
- Keep `score_events()` / `_match_class()` / alignment behavior unchanged unless a focused adversarial regression first proves a correctness defect.
- Score one model/checkpoint + one `input_view_id` cohort at a time.
- Default tolerances are exactly `30`, `50`, and `100` milliseconds.
- Raw and diagnostically aligned modes are separate named rows; alignment never replaces raw scoring.
- Headline matching uses onset time + common class only. Confidence/velocity never affect matching.
- Failed/skipped/quarantined rows remain visible in population accounting; never silently reduce a cohort to successes.
- Event diagnostics are opt-in by simfile ID and empty by default; never materialize broad-corpus diagnostics automatically.
- Every Python-float-derived persisted report number crosses `quantize_six()` before canonical JSON/JSONL or CSV/Markdown rendering.
- Undefined metrics and undefined 0/0 mapping coverage remain `None` / JSON `null` / empty CSV cells.
- `summary.json` owns cohort-level identity/population/aggregates. `items.csv` owns the item ledger. Cohort class aggregates live only in `summary.json`.
- `items.csv` and `event_diagnostics.jsonl` carry `cohort_id` only for cohort identity. `per_song.csv` and `per_class.csv` additionally carry the Linear-required score-row dimensions: `model_id`, `model_lock_sha256`, `prediction_map_version`, `input_view_id`, and `scoring_version`.
- Do not add inference, a CLI/run manifest, worker pool, queue, retry framework, generic pipeline, report registry, database, Parquet, or immutable report publication.
- Breaking `ChartReport` / `write_reports` is allowed; do not add a compatibility shim.

## Risks / Hard Gates

### Gate A — Diagnostic volume

`score_cohort(..., diagnostics_for=())` must produce zero event diagnostics. Diagnostics are materialized only for explicitly requested successful song IDs. HPA-326 broad-corpus scoring uses the empty default.

### Gate B — Cohort identity

Every successful prediction event must carry `metadata["input_view_id"]` and `metadata["prediction_map_version"]` matching `CohortIdentity`. Mixed input-view/map events fail before scoring.

### Gate C — Coverage balance

Reference accounting must include duplicate common collapse and satisfy:

```text
native = common + duplicate-collapsed + ignored + unmapped
```

Prediction accounting must satisfy:

```text
native = mapped + unmapped
```

### Gate D — Numeric encoding

No Python `float` may reach `canonical_json_bytes()`. Task 5 must pin one exact JSONL line and exact CSV `0.5` cells before report implementation can pass.

### Gate E — CI lint scope

Final implementation verification must run the CI-equivalent `ruff check .` and errors-only Pylint over all `src`, including the new benchmark module.

---

## File Map

### Create

- `src/benchmark/cohort_scoring.py` — closed cohort contract, validation, coverage, canonical ordering, bounded diagnostics, scoring, and aggregates.
- `tests/benchmark/test_scorer_input.py` — reference-side adapter tests.
- `tests/benchmark/test_cohort_scoring.py` — cohort contract/scoring/aggregate tests.
- `tests/benchmark/test_cohort_scoring_acceptance.py` — persisted HPA-324/HPA-423 artifact acceptance without inference.

### Modify

- `src/benchmark/scorer_input.py` — add only `reference_to_benchmark_events()`.
- `src/benchmark/scoring.py` — promote `_percentile` to public `percentile` without semantic change.
- `src/benchmark/reports.py` — replace the flat chart writer with deterministic cohort rendering.
- `tests/benchmark/test_scoring.py` — pin remaining matcher behavior and public percentile name.
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

## Task 1: Pin the matcher and add the reference scorer adapter

**Files:**
- Modify: `tests/benchmark/test_scoring.py`
- Modify: `src/benchmark/scorer_input.py`
- Create: `tests/benchmark/test_scorer_input.py`

**Interfaces:**
- Consumes: existing `BenchmarkEvent`, `CommonReferenceEvent`, and prediction scorer adapter.
- Produces:

```python
def reference_to_benchmark_events(
    simfile_id: str,
    common_events: tuple[CommonReferenceEvent, ...],
) -> tuple[BenchmarkEvent, ...]: ...
```

- [ ] **Step 1: Add simultaneous/empty/one-sided-class matcher pins**

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


def test_empty_prediction_song_counts_all_references_as_false_negatives() -> None:
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

Keep existing dense-hit and large-offset tests unchanged.

- [ ] **Step 2: Run matcher pins before touching scoring code**

```bash
uv run pytest tests/benchmark/test_scoring.py -q
```

Expected: PASS. If a new fixture fails, stop and review the matcher as a correctness defect before aggregate work.

- [ ] **Step 3: Write RED tests for reference projection**

Create `tests/benchmark/test_scorer_input.py`:

```python
from decimal import Decimal

from src.benchmark.reference_set import CommonReferenceEvent
from src.benchmark.scorer_input import reference_to_benchmark_events


def test_reference_common_events_become_ground_truth_events() -> None:
    common = (
        CommonReferenceEvent(Decimal("0.500000"), "kick", ()),
        CommonReferenceEvent(Decimal("1.250000"), "hihat", ()),
    )

    result = reference_to_benchmark_events("42", common)

    assert [
        (item.chart_id, item.time_sec, item.canonical_class, item.source)
        for item in result
    ] == [
        ("42", 0.5, "kick", "ground_truth"),
        ("42", 1.25, "hihat", "ground_truth"),
    ]
    assert all(type(item.time_sec) is float for item in result)


def test_reference_adapter_preserves_common_projection_order() -> None:
    common = (
        CommonReferenceEvent(Decimal("1.000000"), "kick", ()),
        CommonReferenceEvent(Decimal("1.000000"), "snare", ()),
    )
    result = reference_to_benchmark_events("song", common)
    assert [(item.time_sec, item.canonical_class) for item in result] == [
        (1.0, "kick"),
        (1.0, "snare"),
    ]
```

- [ ] **Step 4: Verify RED**

```bash
uv run pytest tests/benchmark/test_scorer_input.py -q
```

Expected: FAIL because the reference adapter does not exist.

- [ ] **Step 5: Implement the minimal sibling adapter**

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

Export it in `__all__`. Do not accept native reference events and do not call DTX timing/mapping helpers here.

- [ ] **Step 6: Run adapter + existing HPA-423 bridge coverage**

```bash
uv run pytest tests/benchmark/test_scorer_input.py \
  tests/benchmark/test_task_d_contract.py tests/benchmark/test_scoring.py -q
```

Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add src/benchmark/scorer_input.py tests/benchmark/test_scorer_input.py \
  tests/benchmark/test_scoring.py
git commit -m "test: pin cohort scorer inputs"
```

---

## Task 2: Define the closed cohort identity, reason, and coverage contract

**Files:**
- Create: `src/benchmark/cohort_scoring.py`
- Create: `tests/benchmark/test_cohort_scoring.py`

**Interfaces:**
- Consumes: `ReferenceMappingResult`, `PredictionArtifact`, `BenchmarkEvent`, existing identity validators.
- Produces:

```python
SCORING_VERSION = "crux.single-cohort-scoring/v1"
DEFAULT_TOLERANCES_MS = (30, 50, 100)
SCORE_MODES = ("raw", "aligned")

CohortExecutionStatus = Literal["success", "failed", "skipped", "quarantined"]
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
    artifact_identity: CohortArtifactIdentity | None = None
    reference_artifact: ReferenceMappingResult | None = None
    prediction_artifact: PredictionArtifact | None = None
```

Add the narrow in-memory artifact handoff:

```python
def cohort_item_from_artifacts(
    identity: CohortIdentity,
    simfile_id: str,
    reference: ReferenceMappingResult,
    prediction: PredictionArtifact,
    *,
    warnings: tuple[str, ...] = (),
) -> CohortItem: ...
```

Successful items retain the actual `ReferenceMappingResult` and
`PredictionArtifact` objects used to derive their events and coverage, plus a
frozen `CohortArtifactIdentity` derived from their descriptor/audio/map and
reference song identity. Validation recomputes events, coverage, and identity
from those retained artifacts before scoring; caller-supplied event tuples or
identity values cannot bless forged evidence. An empty OaF prediction artifact
remains scoreable; its map identity comes from the existing OaF mapping
identity because no event record exists. Non-success items keep their
no-prediction shape and no artifact evidence.

Before deriving successful-item evidence, validation re-reads
`prediction_artifact.content` through `read_prediction_artifact()` and requires
the canonical result to equal the retained artifact, including event count,
hashes, descriptor, audio, and event projection. It also re-projects
`reference.mapped_events` through `project_common_reference_events()` and
requires the retained common events and duplicate-collapse diagnostic to agree.


```python
def coverage_from_artifacts(
    reference: ReferenceMappingResult,
    prediction: PredictionArtifact | None,
) -> CohortCoverage: ...
```

- [ ] **Step 1: Write the closed reason-code invariant**

```python
def test_cohort_failure_reason_set_is_closed() -> None:
    assert COHORT_FAILURE_REASONS == {
        "reference_quarantined",
        "backend_unavailable",
        "inference_failed",
        "prediction_artifact_invalid",
        "prediction_missing",
        "explicitly_skipped",
    }
    assert COHORT_FAILURE_REASONS == frozenset(get_args(CohortFailureReason))
```

These are grouped scoring categories. HPA-326 may retain detailed worker text in its own run manifest; do not put detail into grouped HPA-325 reason strings.

- [ ] **Step 2: Write a shared identity fixture and direct SHA validation tests**

```python
def identity() -> CohortIdentity:
    return CohortIdentity(
        cohort_id="oaf-full-mix-v1",
        reference_manifest_sha256="a" * 64,
        reference_timing_version="sha256:" + "b" * 64,
        taxonomy_version=TAXONOMY_VERSION,
        lane_map_version=DTX_LANE_MAP_VERSION,
        backend_id=OAF_BACKEND_ID,
        model_id="magenta-egmd-ckpt-569400-v1",
        model_lock_sha256="c" * 64,
        backend_descriptor_sha256="d" * 64,
        prediction_map_version=OAF_PREDICTION_MAP_ID,
        input_view_id="full-mix-v1",
    )
```

Parametrize the SHA fields and require the existing message:

```python
@pytest.mark.parametrize(
    "field",
    ["reference_manifest_sha256", "model_lock_sha256", "backend_descriptor_sha256"],
)
def test_identity_reuses_shared_sha256_validation(field: str) -> None:
    values = dataclasses.asdict(identity())
    values[field] = "ABC"
    with pytest.raises(ValueError, match=f"{field} must be lowercase SHA-256"):
        CohortIdentity(**values)
```

- [ ] **Step 3: Implement identity validation using `require_sha256()` directly**

In `CohortIdentity.__post_init__` validate nonempty identity strings and exact `SCORING_VERSION`, then:

```python
for field in (
    "reference_manifest_sha256",
    "model_lock_sha256",
    "backend_descriptor_sha256",
):
    require_sha256(getattr(self, field), field)
```

Do **not** catch/re-wrap `StrictJsonError`; it already subclasses `ValueError`. Do not add a local SHA regex/helper.

- [ ] **Step 4: Write reference/prediction coverage tests from real domain objects**

Build a `ReferenceMappingResult` containing:

- mapped tom lanes `14` and `15` at the same exact time (one duplicate collapse);
- one mapped kick lane `13`;
- one ignored lane `54`;
- no unknown lane.

Use an OaF-shaped mapped prediction artifact containing one mapped and one unmapped native event.

Require:

```python
assert coverage.reference_native_event_count == 4
assert coverage.reference_common_event_count == 2
assert coverage.reference_duplicate_collapsed_count == 1
assert coverage.reference_ignored_event_count == 1
assert coverage.reference_unmapped_event_count == 0
assert (
    coverage.reference_native_event_count
    == coverage.reference_common_event_count
    + coverage.reference_duplicate_collapsed_count
    + coverage.reference_ignored_event_count
    + coverage.reference_unmapped_event_count
)
assert coverage.prediction_native_event_count == 2
assert coverage.prediction_mapped_event_count == 1
assert coverage.prediction_unmapped_event_count == 1
```

For `prediction=None`, all prediction counts are `None` and native-class counts are empty.

- [ ] **Step 5: Verify RED for coverage**

```bash
uv run pytest tests/benchmark/test_cohort_scoring.py -q
```

Expected: FAIL because `cohort_scoring.py` is incomplete/missing.

- [ ] **Step 6: Implement `coverage_from_artifacts()`**

Reference native count:

```python
reference_native_event_count = (
    len(reference.mapped_events)
    + sum(reference.diagnostics.ignored.values())
    + sum(reference.diagnostics.unmapped.values())
)
```

Carry:

```python
reference_common_event_count = len(reference.common_events)
reference_duplicate_collapsed_count = reference.diagnostics.duplicate_common_event_count
```

Prediction counts:

```python
native_events = prediction.prediction.events
mapped_count = sum(item.mapping_status == "mapped" for item in native_events)
unmapped_count = sum(item.mapping_status == "unmapped" for item in native_events)
native_class_counts = tuple(
    sorted(Counter(item.native.native_class_id for item in native_events).items())
)
```

Do not persist another coverage artifact.

- [ ] **Step 7: Write item/status/coverage validation tests**

Require:

- nonempty unique `simfile_id` values in a cohort;
- success: nonempty references, `prediction_events is not None` (empty tuple valid), and `failure_reason is None`;
- failed: no prediction tuple and reason in `backend_unavailable | inference_failed | prediction_artifact_invalid | prediction_missing`;
- skipped: no prediction tuple and reason exactly `explicitly_skipped`;
- quarantined: no prediction tuple and reason exactly `reference_quarantined`;
- reference coverage counts are nonnegative and satisfy the balance equation;
- `coverage.reference_common_event_count == len(reference_events)`;
- when prediction counts exist, they are nonnegative and `native == mapped + unmapped`;
- success requires prediction counts and `prediction_mapped_event_count == len(prediction_events)`.

A future HPA-326 resume hit that reuses a valid prediction artifact is adapted as `success`, not `skipped`.

- [ ] **Step 8: Write mixed-cohort prediction metadata tests**

Use prediction `BenchmarkEvent`s shaped like the existing scorer adapter output:

```python
metadata={
    "input_view_id": "full-mix-v1",
    "prediction_map_version": OAF_PREDICTION_MAP_ID,
}
```

Require success validation to reject either mismatch:

```python
with pytest.raises(ValueError, match="input_view_id"):
    validate_cohort_items(identity(), (mixed_view_item,))

with pytest.raises(ValueError, match="prediction_map_version"):
    validate_cohort_items(identity(), (mixed_map_item,))
```

Missing metadata also fails. Do not infer identity from cohort labels.

- [ ] **Step 9: Implement narrow validation helpers**

Use direct loops and `ValueError`; do not add a validation framework. For each successful prediction event:

```python
if event.metadata.get("input_view_id") != identity.input_view_id:
    raise ValueError("prediction event input_view_id does not match cohort")
if event.metadata.get("prediction_map_version") != identity.prediction_map_version:
    raise ValueError("prediction event prediction_map_version does not match cohort")
```

- [ ] **Step 10: Run contract tests**

```bash
uv run pytest tests/benchmark/test_cohort_scoring.py \
  tests/benchmark/test_task_d_contract.py -q
```

Expected: PASS.

- [ ] **Step 11: Commit**

```bash
git add src/benchmark/cohort_scoring.py tests/benchmark/test_cohort_scoring.py
git commit -m "feat: define cohort scoring contract"
```

---

## Task 3: Add canonical song/class scoring with opt-in diagnostics

**Files:**
- Modify: `src/benchmark/cohort_scoring.py`
- Modify: `tests/benchmark/test_cohort_scoring.py`

**Interfaces:**
- Consumes: validated `CohortItem`s and `score_events_with_alignment()`.
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


def _score_success_items(
    items: tuple[CohortItem, ...],
    tolerances_ms: tuple[int, ...],
    diagnostics_for: frozenset[str],
) -> tuple[tuple[SongScore, ...], tuple[EventDiagnostic, ...]]: ...
```

- [ ] **Step 1: Write fixed tolerance/mode matrix tests**

Use one success with a kick prediction 40 ms late. Require canonical row order:

```python
[
    (30, "raw"),
    (30, "aligned"),
    (50, "raw"),
    (50, "aligned"),
    (100, "raw"),
    (100, "aligned"),
]
```

Raw 30 ms F1 is `0.0`; aligned 30 ms and raw 50/100 ms F1 are `1.0`.

- [ ] **Step 2: Write per-class reconciliation test**

Use kick + snare references plus an unmatched hihat prediction. For every song result:

```python
assert sum(row.summary.true_positives for row in score.per_class) == score.summary.true_positives
assert sum(row.summary.false_positives for row in score.per_class) == score.summary.false_positives
assert sum(row.summary.false_negatives for row in score.per_class) == score.summary.false_negatives
assert [row.common_class for row in score.per_class] == sorted(
    row.common_class for row in score.per_class
)
```

Require hihat even with zero reference support.

- [ ] **Step 3: Write scorer metadata-independence test**

Compare otherwise identical prediction `BenchmarkEvent`s with/without confidence/velocity metadata. Counts/F1 are identical. Do not create an invalid OaF persisted artifact merely to test missing optional model fields at the scorer layer.

- [ ] **Step 4: Write default-empty diagnostics test**

```python
song_scores, diagnostics = _score_success_items(
    (success_a, success_b),
    DEFAULT_TOLERANCES_MS,
    diagnostics_for=frozenset(),
)
assert song_scores
assert diagnostics == ()
```

This is the broad-corpus safety default.

- [ ] **Step 5: Write selected-song diagnostic test**

Request only song `"2"` and require every diagnostic row to have `simfile_id == "2"`; no row for song `"1"` is materialized.

For an aligned prediction:

```python
assert diagnostic.reference_time_sec == pytest.approx(1.0)
assert diagnostic.prediction_time_sec == pytest.approx(1.1)
assert diagnostic.scored_prediction_time_sec == pytest.approx(1.0)
assert diagnostic.timing_error_sec == pytest.approx(0.0)
```

- [ ] **Step 6: Verify RED**

```bash
uv run pytest tests/benchmark/test_cohort_scoring.py -q
```

Expected: FAIL on missing score/diagnostic behavior.

- [ ] **Step 7: Implement one matcher call per success/tolerance**

```python
aligned = score_events_with_alignment(
    list(item.reference_events),
    list(item.prediction_events),
    tolerance_sec=tolerance_ms / 1000.0,
)
```

Turn `aligned.raw` and `aligned.aligned` into two rows in fixed `SCORE_MODES = ("raw", "aligned")` order. Do not run a second matcher for classes.

- [ ] **Step 8: Derive per-class rows from the same `ScoreResult`**

For each class present in matches or either unmatched set:

- TP = matching rows for class;
- FP = unmatched predictions for class;
- FN = unmatched references for class;
- reference support = TP + FN;
- prediction support = TP + FP.

Construct `ScoreSummary(tp, fp, fn)` and sort `ClassScore` rows by `common_class`.

- [ ] **Step 9: Derive diagnostics only for requested IDs**

Do nothing when `item.simfile_id not in diagnostics_for`.

Raw mode: original/scored prediction times are equal.

Aligned mode:

```python
original_prediction_time = scored_prediction_time - score_result.summary.offset_sec
```

FN rows have prediction fields `None`; FP rows have reference/error fields `None`.

- [ ] **Step 10: Establish canonical song/diagnostic ordering here**

Sort song scores by:

```python
(simfile_id, tolerance_ms, mode_rank)
```

with `raw=0`, `aligned=1`.

Sort diagnostics by:

```python
(
    simfile_id,
    tolerance_ms,
    mode_rank,
    outcome,
    common_class,
    -1.0 if reference_time_sec is None else reference_time_sec,
    -1.0 if prediction_time_sec is None else prediction_time_sec,
)
```

Writers must not define another semantic order.

- [ ] **Step 11: Run focused scoring tests**

```bash
uv run pytest tests/benchmark/test_cohort_scoring.py tests/benchmark/test_scoring.py -q
```

Expected: PASS without modifying matcher/alignment semantics.

- [ ] **Step 12: Commit**

```bash
git add src/benchmark/cohort_scoring.py tests/benchmark/test_cohort_scoring.py
git commit -m "feat: score single benchmark cohorts"
```

---

## Task 4: Reuse the percentile helper and add public cohort aggregation

**Files:**
- Modify: `src/benchmark/scoring.py`
- Modify: `tests/benchmark/test_scoring.py`
- Modify: `src/benchmark/cohort_scoring.py`
- Modify: `tests/benchmark/test_cohort_scoring.py`

**Interfaces:**
- Consumes: existing `_percentile` semantics plus Task 3 song rows.
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
    reason_counts: tuple[tuple[CohortFailureReason, int], ...]

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


def score_cohort(
    identity: CohortIdentity,
    items: tuple[CohortItem, ...],
    tolerances_ms: tuple[int, ...] = DEFAULT_TOLERANCES_MS,
    diagnostics_for: tuple[str, ...] = (),
) -> CohortScoreResult: ...
```

- [ ] **Step 1: Pin the existing percentile convention under a public name**

In `tests/benchmark/test_scoring.py`:

```python
def test_percentile_uses_existing_upper_nearest_rank_convention() -> None:
    values = [0.0, 0.25, 0.5, 0.75, 1.0]
    assert percentile(values, 0.10) == 0.25
    assert percentile(values, 0.25) == 0.25
    assert percentile(values, 0.75) == 0.75
    assert percentile(values, 0.90) == 1.0
```

- [ ] **Step 2: Verify RED, then promote `_percentile` without semantic change**

```bash
uv run pytest tests/benchmark/test_scoring.py::test_percentile_uses_existing_upper_nearest_rank_convention -q
```

Expected: FAIL because only `_percentile` exists.

Rename the existing function to `percentile` and update `score_events()`'s p95 call. Do not keep an alias; repository search shows no external consumer.

- [ ] **Step 3: Write input-order and diagnostics-request validation tests for public `score_cohort()`**

Build items in reverse ID order and require:

```python
result = score_cohort(identity(), (item_2, item_1))
assert [item.simfile_id for item in result.items] == ["1", "2"]
assert [
    (row.simfile_id, row.tolerance_ms, row.mode)
    for row in result.song_scores
] == sorted_by_declared_contract
```

Require:

- tolerances are positive, unique, and sorted;
- `diagnostics_for` contains unique, nonempty IDs;
- every diagnostics ID names a successful input item;
- default diagnostics are empty.

- [ ] **Step 4: Write full-population/closed-reason tests**

Use success + failed + skipped + quarantined rows and require:

```python
assert result.population.total_count == 4
assert result.population.success_count == 1
assert result.population.failed_count == 1
assert result.population.skipped_count == 1
assert result.population.quarantined_count == 1
assert result.population.reason_counts == (
    ("explicitly_skipped", 1),
    ("inference_failed", 1),
    ("reference_quarantined", 1),
)
```

Group only stable reason codes; warnings never become reason buckets.

- [ ] **Step 5: Write event-micro/song-macro/class-macro tests**

Use two successful songs:

- song A: kick TP + snare FN;
- song B: kick TP + kick FP.

Require event-micro `(TP,FP,FN) == (2,1,1)`, song macro equals the arithmetic mean of song F1s, and class macro equals the arithmetic mean of supported aggregate class F1s.

- [ ] **Step 6: Write F1 distribution tests using `scoring.percentile()`**

For song F1 values `(0.0, 0.25, 0.5, 0.75, 1.0)` require:

```python
assert distribution.minimum == 0.0
assert distribution.p10 == 0.25
assert distribution.p25 == 0.25
assert distribution.median == 0.5
assert distribution.p75 == 0.75
assert distribution.p90 == 1.0
assert distribution.maximum == 1.0
```

Use `statistics.median()` for the median; p10/p25/p75/p90 call the promoted `percentile()`.

- [ ] **Step 7: Write zero-success tests**

A cohort with failed/quarantined rows only still returns six aggregate dimensions. Require zero event counts with undefined P/R/F1, undefined macro/distribution fields, empty aggregate class rows, and intact population counts.

- [ ] **Step 8: Implement `score_cohort()` canonicalization before scoring**

Validation order:

1. validate `CohortIdentity` (dataclass post-init already covers hash/string rules);
2. validate tolerances;
3. validate unique item IDs/status/reason/coverage/event metadata;
4. validate `diagnostics_for`;
5. sort `items` by `simfile_id`;
6. call `_score_success_items()` with a `frozenset(diagnostics_for)`;
7. aggregate canonical song rows;
8. return canonical tuples.

Do not defer item ordering to the writer.

- [ ] **Step 9: Implement population/aggregate helpers**

Event micro:

```python
ScoreSummary(
    true_positives=sum(row.summary.true_positives for row in rows),
    false_positives=sum(row.summary.false_positives for row in rows),
    false_negatives=sum(row.summary.false_negatives for row in rows),
)
```

Macro helper:

```python
def _mean_f1(values: tuple[float | None, ...]) -> float | None:
    scored = tuple(value for value in values if value is not None)
    return fmean(scored) if scored else None
```

Aggregate class rows by summed TP/FP/FN and sort by `common_class`.

- [ ] **Step 10: Run scoring/aggregate suites**

```bash
uv run pytest tests/benchmark/test_scoring.py tests/benchmark/test_cohort_scoring.py -q
```

Expected: PASS.

- [ ] **Step 11: Commit**

```bash
git add src/benchmark/scoring.py src/benchmark/cohort_scoring.py \
  tests/benchmark/test_scoring.py tests/benchmark/test_cohort_scoring.py
git commit -m "feat: aggregate cohort benchmark results"
```

---

## Task 5: Replace the flat report writer with non-duplicated deterministic artifacts

**Files:**
- Modify: `src/benchmark/reports.py`
- Modify: `tests/benchmark/test_reports.py`

**Interfaces:**
- Consumes: canonical `CohortScoreResult`, `quantize_six()`, `canonical_json_bytes()`.
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

- [ ] **Step 1: Replace legacy report tests with the six-file contract**

Require paths:

```python
assert artifacts.summary_json == tmp_path / "summary.json"
assert artifacts.items_csv == tmp_path / "items.csv"
assert artifacts.per_song_csv == tmp_path / "per_song.csv"
assert artifacts.per_class_csv == tmp_path / "per_class.csv"
assert artifacts.event_diagnostics_jsonl == tmp_path / "event_diagnostics.jsonl"
assert artifacts.summary_markdown == tmp_path / "summary.md"
```

Delete legacy `ChartReport` / `write_reports` test imports.

- [ ] **Step 2: Pin the single numeric boundary before any writer code**

In `tests/benchmark/test_reports.py`:

```python
assert _report_decimal(0.5) == Decimal("0.500000")
assert _report_decimal(None) is None
assert _csv_decimal(0.5) == "0.5"
assert _csv_decimal(None) == ""
```

Require nonfinite input to raise `StrictJsonError` through `quantize_six()`.

- [ ] **Step 3: Verify RED, then implement numeric helpers**

```bash
uv run pytest tests/benchmark/test_reports.py -q
```

Expected: FAIL.

Implement:

```python
def _report_decimal(value: float | None) -> Decimal | None:
    return None if value is None else quantize_six(value)


def _csv_decimal(value: float | None) -> str:
    decimal = _report_decimal(value)
    return "" if decimal is None else canonical_json_bytes(decimal).decode("ascii")
```

Every float-derived persisted number uses these helpers. No `str(float)` / `repr(float)`.

- [ ] **Step 4: Test `summary.json` as cohort-level facts only**

Require exact top-level keys:

```text
schema
identity
tolerances_ms
population
aggregates
```

There is **no** `items` key. Each aggregate owns its cohort per-class aggregate rows and F1 distribution.

Parse with `strict_json_loads(..., require_canonical=True)`. Assert one-half becomes `Decimal("0.5")` and undefined metrics remain `None`.

- [ ] **Step 5: Test `items.csv` as the sole item ledger**

Fixed fields:

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

Rules:

- warnings sorted and `|`-joined;
- native-class counts `class=count|class=count`;
- missing prediction counts => empty fields;
- mapping coverage = mapped/native only when native > 0;
- native == 0 => empty coverage field (undefined, not `1.0`).

Require:

```python
assert half_coverage_row["prediction_mapping_coverage"] == "0.5"
assert empty_prediction_row["prediction_mapping_coverage"] == ""
```

- [ ] **Step 6: Test the reduced but Linear-compliant `per_song.csv` identity columns**

Fixed fields:

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

Do not repeat reference/backend descriptor hashes here; those join through `cohort_id` to `summary.json`.

Use a half-precision fixture:

```python
assert raw_50_row["precision"] == "0.5"
```

- [ ] **Step 7: Test `per_class.csv` as song rows only**

Fixed fields:

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

No `scope` column and no cohort rows. Cohort class aggregates live only in `summary.json`.

- [ ] **Step 8: Test slim, bounded canonical event JSONL with an exact golden line**

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

For a raw matched kick at 0.5 seconds, require this exact canonical line:

```python
expected = (
    b'{"cohort_id":"oaf-full-mix-v1","common_class":"kick","mode":"raw",'
    b'"outcome":"matched","prediction_time_sec":0.5,"reference_time_sec":0.5,'
    b'"scored_prediction_time_sec":0.5,"simfile_id":"1","timing_error_sec":0,'
    b'"tolerance_ms":50}'
)
assert expected in (tmp_path / "event_diagnostics.jsonl").read_bytes().splitlines()
```

A result built with default `diagnostics_for=()` produces an empty JSONL file.

- [ ] **Step 9: Implement report payload builders from canonical result order**

Writers iterate result tuples in their existing order. They do not semantically resort items/song/class rows.

Score metric helper:

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

Apply `_report_decimal()` / `_csv_decimal()` to ratios, coverage, milliseconds, offsets, F1 distributions, and diagnostic times/errors.

Use `canonical_json_bytes()` for `summary.json` and each JSONL record; explicit `csv.DictWriter(fieldnames=...)` for CSVs.

- [ ] **Step 10: Render Markdown from the same result/numeric tokens**

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

Use the same canonical numeric token helper instead of independent float formatting. Song extremes show up to five lowest/highest raw-50ms successful songs sorted by `(f1, simfile_id)`; if none, write `No successful songs.`.

- [ ] **Step 11: Write byte-determinism tests**

1. Write the same `CohortScoreResult` to `first/` and `second/`; compare all six byte streams.
2. Reverse input item order, call `score_cohort()` again, and require identical six output files.
3. Require JSON/JSONL canonical parsing succeeds.

This test now diagnoses scorer-order bugs in Task 4 rather than relying on writer sorting.

- [ ] **Step 12: Run report/cohort tests**

```bash
uv run pytest tests/benchmark/test_reports.py tests/benchmark/test_cohort_scoring.py -q
```

Expected: PASS with no Python float reaching canonical JSON and no duplicated item/cohort-class data.

- [ ] **Step 13: Commit**

```bash
git add src/benchmark/reports.py tests/benchmark/test_reports.py
git commit -m "feat: write deterministic cohort reports"
```

---

## Task 6: Prove persisted HPA-324/HPA-423 artifacts rescore end-to-end without inference

**Files:**
- Create: `tests/benchmark/test_cohort_scoring_acceptance.py`
- Modify only if the test exposes a narrow missing contract: `src/benchmark/scorer_input.py`, `src/benchmark/cohort_scoring.py`, or `src/benchmark/reports.py`

**Interfaces:**
- Consumes: existing reference mapping, prediction artifact v2, both scorer adapters, coverage derivation, `score_cohort()`, and report writer.
- Produces: one persisted-artifact proof without importing/starting `OafBackend`, Docker, or `WorkerProcess`.

- [ ] **Step 1: Build HPA-323-shaped references with exact common collapse**

Create simultaneous lanes `14` and `15` at 1.0 s plus kick `13` at 2.0 s. Run `map_reference_events()`:

```python
assert len(reference_mapping.mapped_events) == 3
assert len(reference_mapping.common_events) == 2
assert reference_mapping.diagnostics.duplicate_common_event_count == 1
```

- [ ] **Step 2: Convert common references through the new adapter**

```python
reference_events = reference_to_benchmark_events(
    "7",
    reference_mapping.common_events,
)
assert [(item.canonical_class, item.time_sec) for item in reference_events] == [
    ("tom", 1.0),
    ("kick", 2.0),
]
```

- [ ] **Step 3: Round-trip a real-shaped OaF prediction artifact v2**

Use `NativePrediction`/descriptor fixtures matching `test_task_d_contract.py`, map with `map_oaf_prediction()`, render/read prediction artifact v2, then:

```python
prediction_events = prediction_to_benchmark_events(artifact)
```

Include one common tom and kick matching the references plus one unmapped native sticks event so coverage proves native vs mapped accounting.

- [ ] **Step 4: Derive balanced coverage including duplicate collapse**

```python
coverage = coverage_from_artifacts(reference_mapping, artifact)
assert coverage.reference_duplicate_collapsed_count == 1
assert (
    coverage.reference_native_event_count
    == coverage.reference_common_event_count
    + coverage.reference_duplicate_collapsed_count
    + coverage.reference_ignored_event_count
    + coverage.reference_unmapped_event_count
)
assert coverage.prediction_native_event_count == (
    coverage.prediction_mapped_event_count + coverage.prediction_unmapped_event_count
)
```

- [ ] **Step 5: Build one valid cohort item and identity from artifact metadata**

Use `input_view_id` and `prediction_map_version` matching the scorer event metadata so the mixed-cohort guard is exercised on the success path.

- [ ] **Step 6: Request diagnostics for this fixture only**

```python
result = score_cohort(
    identity,
    (item,),
    diagnostics_for=("7",),
)
assert result.event_diagnostics
assert {row.simfile_id for row in result.event_diagnostics} == {"7"}
```

Then rescore with default diagnostics and require `event_diagnostics == ()` while aggregate metrics are identical.

- [ ] **Step 7: Write and validate all reports**

```python
artifacts = write_cohort_reports(result, tmp_path / "reports")
```

Require:

- `summary.json` has identity/population/aggregates but no items;
- `items.csv` includes duplicate-collapse coverage;
- `per_song.csv` includes mandated score-row identity fields;
- `per_class.csv` contains only song-scope rows;
- diagnostics JSONL contains only `cohort_id` plus event evidence;
- no backend/worker module is imported during the test.

- [ ] **Step 8: Prove a mapping identity change rescoring path does not invoke inference**

Construct a second `CohortIdentity` with a different `prediction_map_version` and corresponding copied prediction-event metadata, then rescore the already-persisted events. The test must remain entirely in mapping/artifact/scorer/report code.

- [ ] **Step 9: Run acceptance suite**

```bash
uv run pytest tests/benchmark/test_cohort_scoring_acceptance.py \
  tests/benchmark/test_task_d_contract.py tests/benchmark/test_reference_set.py -q
```

Expected: PASS without Docker/backend startup.

- [ ] **Step 10: Commit**

```bash
git add tests/benchmark/test_cohort_scoring_acceptance.py
git commit -m "test: prove persisted cohort rescoring"
```

---

## Task 7: Run CI-equivalent verification and freeze the HPA-326 handoff

**Files:**
- Modify only if verification exposes a defect in the files already named by Tasks 1–6.

- [ ] **Step 1: Run focused HPA-325 tests**

```bash
uv run pytest tests/benchmark/test_scorer_input.py \
  tests/benchmark/test_scoring.py \
  tests/benchmark/test_cohort_scoring.py \
  tests/benchmark/test_reports.py \
  tests/benchmark/test_cohort_scoring_acceptance.py -q
```

Expected: PASS.

- [ ] **Step 2: Run the full test suite**

```bash
uv run pytest
```

Expected: PASS.

- [ ] **Step 3: Run CI-equivalent Ruff checks**

```bash
uv run ruff check .
uv run ruff format --check src tests
```

Expected: PASS.

- [ ] **Step 4: Run CI-equivalent Pylint across all source modules**

```bash
uv run pylint --errors-only --disable=E1120,E0401 src
```

Expected: no Pylint errors, including `src/benchmark/cohort_scoring.py`.

Do not substitute `uv run pylint src/app src/cli`; that misses every new HPA-325 source file.

- [ ] **Step 5: Check diff hygiene**

```bash
git diff --check main...HEAD
```

Expected: no whitespace errors.

- [ ] **Step 6: Run final contract-focused regressions**

```bash
uv run pytest \
  tests/benchmark/test_task_d_contract.py \
  tests/benchmark/test_reference_set.py \
  tests/benchmark/test_prediction_artifact.py \
  tests/benchmark/test_scorer_input.py \
  tests/benchmark/test_cohort_scoring.py \
  tests/benchmark/test_reports.py \
  tests/benchmark/test_cohort_scoring_acceptance.py -q
```

Expected: PASS.

- [ ] **Step 7: Verify the HPA-326 handoff contract**

Confirm by tests/source inspection:

```text
HPA-326 provides:
- CohortIdentity
- one CohortItem per reference/run row
- persisted reference/prediction artifacts used to derive coverage/events

HPA-325 provides:
- score_cohort(identity, items, diagnostics_for=selected_ids)
- write_cohort_reports(result, output_dir)

Broad HPA-326 baseline rule:
- call score_cohort() with default diagnostics_for=()
- classify valid resumed prediction artifacts as success for scoring
- retain detailed runtime failure text in HPA-326's run manifest
- map failures to closed HPA-325 CohortFailureReason categories
```

No queue, runner, retry, or manifest loader belongs in HPA-325.

- [ ] **Step 8: Commit only verification-driven fixes if any**

Use a focused conventional commit describing the actual fix. Do not create a no-op verification commit.

---

## Completion Checklist

- [ ] Existing matcher/alignment semantics remain unchanged.
- [ ] Reference `Decimal` time converts intentionally to scorer `float`.
- [ ] HPA-423 persisted prediction scorer bridge remains unchanged and covered.
- [ ] `CohortFailureReason` is closed and grouped counts cannot split on free-form details.
- [ ] `require_sha256()` is called directly; no copied hash validator/wrapper.
- [ ] Reference coverage carries duplicate-collapse count and balances exactly.
- [ ] Prediction coverage balances exactly; 0/0 mapping coverage is undefined.
- [ ] Prediction event input-view/map identities must match the cohort.
- [ ] Caller item order cannot alter result/report ordering.
- [ ] Diagnostics are opt-in and broad-corpus default is empty.
- [ ] Per-class song rows derive from the same `ScoreResult` as song totals.
- [ ] Public `scoring.percentile()` retains the existing percentile convention.
- [ ] `summary.json` contains cohort facts only; item ledger lives only in `items.csv`.
- [ ] Cohort per-class aggregate rows live only in `summary.json`; `per_class.csv` is song scope only.
- [ ] `items.csv`/diagnostics use `cohort_id` only; score CSVs carry only the explicitly required score-row identity subset.
- [ ] All float-derived persisted values use `quantize_six()` + canonical decimal rendering.
- [ ] Exact JSONL and CSV numeric goldens lock encoding.
- [ ] No CLI, run manifest, queue, pipeline framework, Parquet/database, or immutable report publisher is introduced.
- [ ] Full tests, CI-equivalent Ruff, CI-equivalent all-`src` Pylint, and diff hygiene pass before implementation is called complete.
