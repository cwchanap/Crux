# HPA-305 Per-Class Comparison Asymmetry Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let published per-class comparisons tolerate legitimate prediction-only class rows on either side, report exactly how many class rows were excluded from paired deltas, remain fail-closed for reference-supported evidence, and prove the recorded HPA-328 comparison can publish.

**Architecture:** Keep class-join semantics centralized in `src/benchmark/published_comparison.py::paired_class_rows()`. The helper validates the union of pairable class keys, returns paired rows plus per-side prediction-only exclusion counts, and renders only shared rows. The three existing comparison callers merge those counts into their existing `pairing.exclusions` maps; HPA-325 scoring, report schemas, `paired_song_rows()`, and separation execution stay unchanged.

**Tech Stack:** Python 3.12, existing HPA-325 published report models/readers, pytest, Ruff, Pylint, existing Crux benchmark artifact readers.

**Spec:** `docs/superpowers/specs/2026-08-29-hpa-305-per-class-comparison-asymmetry-design.md`

## Global Constraints

- Keep HPA-305 in one PR: planning, Task 0 evidence, tests, production fix, and closeout stay on PR #32.
- Task 0 is a hard gate. Do not change tests or production code until the six recorded simfile-241 full-mix `ride` rows are inspected and recorded.
- Proceed only if every recorded row has `reference_support == 0` and `prediction_support > 0`.
- If Task 0 fails, HPA-305 remains the owning ticket and PR #32 remains the single PR; re-scope the same branch from artifact evidence instead of opening another implementation PR.
- Keep `paired_song_rows()` exact-grid and unchanged.
- `paired_class_rows()` returns paired rows plus per-side prediction-only class-row exclusion counts.
- Shared class keys validate and compare `reference_support` only; do not add dead shared `prediction_support` validation.
- One-sided keys validate both supports and are valid only for `reference_support == 0 && prediction_support > 0`.
- Add module-level `_class_key_sort_key()` next to `_song_key_sort_key()`; do not use an assigned lambda or retain duplicate class sort definitions.
- Keep `_support_count()` as a small defensive duck-typed helper. Production `PublishedClassRow` supports have already been validated by report readers.
- Merge class exclusion counts into the existing exclusions map in MuScriptor, IDM, and separation comparisons; do not add a second reporting structure.
- Exclusion counters count omitted per-class score rows (`song × tolerance × mode × common_class`), not unique class names.
- Do not synthesize missing class rows. Source reports and event diagnostics remain the evidence for prediction-only false positives.
- Do not modify HPA-325 scoring, `cohort_scoring.py`, HPA-325 report schemas, taxonomy/mapping, separator execution, or OaF inference.
- Do not add a join-policy option, framework, schema version, compatibility wrapper, new dependency, or second PR.

## File Structure

| File | Responsibility in HPA-305 |
| --- | --- |
| `src/benchmark/published_comparison.py` | Central class-key validation, deterministic sorting, paired row rendering, and class exclusion counts. |
| `src/benchmark/muscriptor_comparison.py` | Merge class exclusions into the existing OaF/MuScriptor exclusions map. |
| `src/benchmark/idm_comparison.py` | Merge class exclusions into the existing OaF/IDM exclusions map. |
| `src/benchmark/separation_comparison.py` | Merge per-view class exclusions inside `_pair_summary()` before `comparison_summary()`. |
| `tests/benchmark/test_published_comparison.py` | Pin symmetric omission/counting, support mismatch, malformed one-sided evidence, and song-grid strictness. |
| `tests/benchmark/test_muscriptor_comparison_coverage.py` | Retarget direct migration-seam tests for the new return shape and positive-reference failure message. |
| `tests/benchmark/test_muscriptor_comparison.py` | Prove class exclusion keys reach the OaF/MuScriptor summary. |
| `tests/benchmark/test_idm_comparison.py` | Prove class exclusion keys reach the OaF/IDM summary. |
| `tests/benchmark/test_separation_comparison.py` | Prove real FP-only source evidence is omitted from paired rows but counted in each affected pair summary. |

No new module, schema, or fixture file is required.

---

### Task 0: Prove the recorded simfile-241 rows are prediction-only before touching code

**Files:** workstation-local production artifacts only; no repository change.

**Interfaces:**
- Input: `artifacts/benchmark/oaf-separation-pilot/runs/oaf-separation-8e66abde20b8f590/views/full_mix/reports/per_class.csv`
- Pass contract: six rows for `simfile_id=241`, `common_class=ride`, covering 30/50/100 ms × raw/aligned; every row has zero reference support and positive prediction support.
- Stop fallback input: `views/spleeter/reports/event_diagnostics.jsonl` from the same run.

- [ ] **Step 1: Confirm the reproduced full-mix report is present**

Run:

```bash
RUN_DIR=artifacts/benchmark/oaf-separation-pilot/runs/oaf-separation-8e66abde20b8f590
FULL_MIX_CLASS_REPORT="$RUN_DIR/views/full_mix/reports/per_class.csv"
test -f "$FULL_MIX_CLASS_REPORT"
```

Expected: exit 0. If absent, restore the PR #31 reproduced artifacts before doing any code work.

- [ ] **Step 2: Inspect and assert all six `ride` support tuples**

Run:

```bash
uv run python - <<'PY'
import csv
from pathlib import Path

path = Path(
    "artifacts/benchmark/oaf-separation-pilot/runs/"
    "oaf-separation-8e66abde20b8f590/views/full_mix/reports/per_class.csv"
)
expected_keys = {
    (30, "raw"),
    (30, "aligned"),
    (50, "raw"),
    (50, "aligned"),
    (100, "raw"),
    (100, "aligned"),
}

with path.open("r", encoding="utf-8", newline="") as handle:
    rows = [
        row
        for row in csv.DictReader(handle)
        if row["simfile_id"] == "241" and row["common_class"] == "ride"
    ]

actual_keys = {(int(row["tolerance_ms"]), row["mode"]) for row in rows}
assert actual_keys == expected_keys, (actual_keys, expected_keys)

for row in sorted(rows, key=lambda item: (int(item["tolerance_ms"]), item["mode"])):
    reference_support = int(row["reference_support"])
    prediction_support = int(row["prediction_support"])
    print(
        f"241 ride {row['tolerance_ms']}ms {row['mode']}: "
        f"reference_support={reference_support} prediction_support={prediction_support}"
    )
    assert reference_support == 0, row
    assert prediction_support > 0, row

print("HPA-305 preflight PASS: six prediction-only full-mix ride rows")
PY
```

Expected: six support lines and the PASS line.

- [ ] **Step 3: Record PASS before implementation, or execute the STOP fallback**

If Step 2 passes, post the six tuples and PASS line on HPA-305 and PR #32 before Task 1.

If Step 2 fails, do not add the tests below and do not modify `paired_class_rows()`. Keep HPA-305/PR #32 as the owner and inspect Spleeter diagnostics first:

```bash
uv run python - <<'PY'
import json
from pathlib import Path

path = Path(
    "artifacts/benchmark/oaf-separation-pilot/runs/"
    "oaf-separation-8e66abde20b8f590/views/spleeter/reports/event_diagnostics.jsonl"
)
rows = []
for line in path.read_text(encoding="utf-8").splitlines():
    row = json.loads(line)
    if row["simfile_id"] == "241" and row["common_class"] == "ride":
        rows.append(row)
for row in rows:
    print(
        row["tolerance_ms"],
        row["mode"],
        row["outcome"],
        row["reference_time_sec"],
        row["prediction_time_sec"],
    )
print(f"spleeter ride diagnostics={len(rows)}")
PY
```

Interpretation:

- If `ride` false negatives exist but Spleeter `per_class.csv` has no `ride` row, rewrite this same HPA-305 spec/plan around the HPA-325 scorer/report emission path, starting with `_class_scores()` and `write_cohort_reports()`.
- If Spleeter diagnostics also contain no `ride` evidence despite positive full-mix reference support, compare the HPA-323/HPA-324 reference mapping evidence used by the two cohorts before changing scoring or comparison code.

There is intentionally no Task 0 commit.

---

### Task 1: TDD the shared reference-aware class join and exclusion result

**Files:**
- Modify: `tests/benchmark/test_published_comparison.py`
- Modify: `tests/benchmark/test_muscriptor_comparison_coverage.py`
- Modify: `src/benchmark/published_comparison.py`

**Interfaces:**

Current public helper:

```python
paired_class_rows(
    left,
    right,
    pairable_ids,
    *,
    left_label="oaf",
    right_label="muscriptor",
)
```

New return contract:

```python
tuple[list[dict[str, str]], dict[str, int]]
```

The second element always contains:

```python
{
    f"{left_label}_only_prediction_class": left_only_count,
    f"{right_label}_only_prediction_class": right_only_count,
}
```

- [ ] **Step 1: Add shared-helper test rows and the six-key production regression**

In `tests/benchmark/test_published_comparison.py`, import `paired_class_rows` and add:

```python
def _class_row(
    *,
    reference_support: object = 1,
    prediction_support: object = 1,
) -> SimpleNamespace:
    return SimpleNamespace(
        reference_support=reference_support,
        prediction_support=prediction_support,
        precision=Decimal("0.5"),
        recall=Decimal("0.5"),
        f1=Decimal("0.5"),
    )


def test_paired_class_rows_counts_left_only_prediction_rows() -> None:
    keys = {
        ("241", tolerance_ms, mode, "ride")
        for tolerance_ms in (30, 50, 100)
        for mode in ("raw", "aligned")
    }
    left = {
        key: _class_row(reference_support=0, prediction_support=1)
        for key in keys
    }

    rows, exclusions = paired_class_rows(
        left,
        {},
        {"241"},
        left_label="full_mix",
        right_label="spleeter",
    )

    assert rows == []
    assert exclusions == {
        "full_mix_only_prediction_class": 6,
        "spleeter_only_prediction_class": 0,
    }
```

- [ ] **Step 2: Add symmetric and fail-closed tests**

Add:

```python
def test_paired_class_rows_counts_right_only_prediction_row() -> None:
    key = ("241", 50, "raw", "ride")
    rows, exclusions = paired_class_rows(
        {},
        {key: _class_row(reference_support=0, prediction_support=2)},
        {"241"},
        left_label="full_mix",
        right_label="spleeter",
    )
    assert rows == []
    assert exclusions == {
        "full_mix_only_prediction_class": 0,
        "spleeter_only_prediction_class": 1,
    }


def test_paired_class_rows_rejects_shared_reference_support_mismatch() -> None:
    key = ("241", 50, "raw", "ride")
    with pytest.raises(ComparisonIntegrityError, match="reference_support mismatch"):
        paired_class_rows(
            {key: _class_row(reference_support=1)},
            {key: _class_row(reference_support=2)},
            {"241"},
            left_label="full_mix",
            right_label="spleeter",
        )


def test_paired_class_rows_rejects_one_sided_zero_zero_row() -> None:
    key = ("241", 50, "raw", "ride")
    with pytest.raises(ComparisonIntegrityError, match="has no prediction support"):
        paired_class_rows(
            {key: _class_row(reference_support=0, prediction_support=0)},
            {},
            {"241"},
            left_label="full_mix",
            right_label="spleeter",
        )


def test_paired_class_rows_rejects_malformed_one_sided_support() -> None:
    key = ("241", 50, "raw", "ride")
    with pytest.raises(ComparisonIntegrityError, match="reference_support is malformed"):
        paired_class_rows(
            {key: _class_row(reference_support=True, prediction_support=1)},
            {},
            {"241"},
            left_label="full_mix",
            right_label="spleeter",
        )
```

Keep `test_paired_song_rows_rejects_key_grid_mismatch()` unchanged.

- [ ] **Step 3: Retarget the existing MuScriptor direct-helper coverage**

In `tests/benchmark/test_muscriptor_comparison_coverage.py`:

1. Change `test_paired_class_rows_skips_non_pairable_ids` to unpack the result and assert the default zero counters:

```python
rows, exclusions = _paired_class_rows(oaf, muscriptor, pairable_ids=set())
assert rows == []
assert exclusions == {
    "oaf_only_prediction_class": 0,
    "muscriptor_only_prediction_class": 0,
}
```

2. Keep `test_paired_class_rows_rejects_asymmetric_pairable_key_grid` with its existing left-only `snare` row whose `reference_support=1`, but change the expected message to `reference-supported class row is missing`.

Do not weaken that row to prediction-only evidence.

- [ ] **Step 4: Run focused tests and confirm RED behavior**

Run:

```bash
uv run pytest -q \
  tests/benchmark/test_published_comparison.py \
  tests/benchmark/test_muscriptor_comparison_coverage.py::test_paired_class_rows_skips_non_pairable_ids \
  tests/benchmark/test_muscriptor_comparison_coverage.py::test_paired_class_rows_rejects_asymmetric_pairable_key_grid
```

Expected before production changes: the new return/count tests fail, the shared-support mismatch is not rejected, and the old MuScriptor message/return shape no longer match.

- [ ] **Step 5: Add one canonical class sort key and the defensive support reader**

In `src/benchmark/published_comparison.py`, add next to `_song_key_sort_key()`:

```python
def _class_key_sort_key(
    value: tuple[str, int, str, str],
) -> tuple[int, int, int, str]:
    return int(value[0]), value[1], _MODES[value[2]], value[3]
```

Keep `_row_value()` and add:

```python
def _support_count(
    row: object,
    field: str,
    *,
    key: tuple[str, int, str, str],
    label: str,
) -> int:
    value = _row_value(row, field)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        _fail(f"per_class {field} is malformed for {label} key={key}")
    return value
```

This helper guards the duck-typed shared helper boundary; production report readers remain the canonical parser/validator.

- [ ] **Step 6: Replace exact-grid validation with reference-aware union validation**

Change `paired_class_rows()` return annotation to:

```python
) -> tuple[list[dict[str, str]], dict[str, int]]:
```

Replace the current grid-mismatch block with this shape:

```python
    left_keys = {key for key in left if str(key[0]) in pairable_ids}
    right_keys = {key for key in right if str(key[0]) in pairable_ids}
    shared_keys = left_keys & right_keys
    left_only = left_keys - right_keys
    right_only = right_keys - left_keys

    for key in sorted(shared_keys, key=_class_key_sort_key):
        left_reference_support = _support_count(
            left[key], "reference_support", key=key, label=left_label
        )
        right_reference_support = _support_count(
            right[key], "reference_support", key=key, label=right_label
        )
        if left_reference_support != right_reference_support:
            _fail(
                "per_class reference_support mismatch"
                f" for key={key} ({left_label}={left_reference_support},"
                f" {right_label}={right_reference_support})"
            )

    for present, present_keys, present_label, missing_label in (
        (left, left_only, left_label, right_label),
        (right, right_only, right_label, left_label),
    ):
        for key in sorted(present_keys, key=_class_key_sort_key):
            reference_support = _support_count(
                present[key], "reference_support", key=key, label=present_label
            )
            prediction_support = _support_count(
                present[key], "prediction_support", key=key, label=present_label
            )
            if reference_support > 0:
                _fail(
                    "per_class reference-supported class row is missing"
                    f" from {missing_label} for key={key}"
                    f" ({present_label}_reference_support={reference_support})"
                )
            if prediction_support <= 0:
                _fail(
                    "per_class one-sided class row has no prediction support"
                    f" for key={key}"
                    f" ({present_label}_prediction_support={prediction_support})"
                )

    rows: list[dict[str, str]] = []
    for key in sorted(shared_keys, key=_class_key_sort_key):
```

Keep the existing row-rendering dictionary unchanged below the loop. Do not validate shared `prediction_support` solely for validation's sake.

Return:

```python
    return rows, {
        f"{left_label}_only_prediction_class": len(left_only),
        f"{right_label}_only_prediction_class": len(right_only),
    }
```

- [ ] **Step 7: Run focused tests and the lint gate**

Run:

```bash
uv run pytest -q \
  tests/benchmark/test_published_comparison.py \
  tests/benchmark/test_muscriptor_comparison_coverage.py::test_paired_class_rows_skips_non_pairable_ids \
  tests/benchmark/test_muscriptor_comparison_coverage.py::test_paired_class_rows_rejects_asymmetric_pairable_key_grid
uv run ruff check src/benchmark/published_comparison.py
uv run ruff format --check src/benchmark/published_comparison.py
```

Expected: all pass. The module-level sort helper avoids Ruff E731 and removes the old nested/inline class sort duplication.

- [ ] **Step 8: Commit the shared helper change**

```bash
git add \
  src/benchmark/published_comparison.py \
  tests/benchmark/test_published_comparison.py \
  tests/benchmark/test_muscriptor_comparison_coverage.py
git commit -m "fix: classify prediction-only class rows"
```

---

### Task 2: Thread class exclusion counts through all three comparison callers

**Files:**
- Modify: `src/benchmark/muscriptor_comparison.py`
- Modify: `src/benchmark/idm_comparison.py`
- Modify: `src/benchmark/separation_comparison.py`
- Modify: `tests/benchmark/test_muscriptor_comparison.py`
- Modify: `tests/benchmark/test_idm_comparison.py`
- Modify: `tests/benchmark/test_separation_comparison.py`

**Interfaces:**
- Consumes Task 1 `paired_class_rows() -> (class_rows, class_exclusions)`.
- Produces the same comparison summaries/artifact names, with the two class exclusion counters merged into each existing `pairing.exclusions` mapping.

- [ ] **Step 1: Update MuScriptor and IDM caller plumbing**

In `compare_oaf_muscriptor()` replace the class helper call with:

```python
class_rows, class_exclusions = _paired_class_rows(
    oaf_classes,
    muscriptor_classes,
    pairable_ids,
)
exclusions = {**exclusions, **class_exclusions}
```

Keep the existing `_summary(..., exclusions, song_rows, class_rows, ...)` call otherwise unchanged.

In `compare_oaf_idm()` replace the class helper call with:

```python
class_rows, class_exclusions = paired_class_rows(
    oaf_classes,
    idm_classes,
    pairable_ids,
    left_label="oaf",
    right_label="idm",
)
exclusions = {**exclusions, **class_exclusions}
```

Keep the existing `comparison_summary(..., exclusions, song_rows, class_rows, ...)` call otherwise unchanged.

- [ ] **Step 2: Update separation `_pair_summary()` without adding another join**

Replace only its class helper/result plumbing:

```python
class_rows, class_exclusions = paired_class_rows(
    full_classes,
    view_classes,
    pairable_ids,
    left_label="full_mix",
    right_label=view_name,
)
merged_exclusions = {**exclusions, **class_exclusions}
```

Pass `merged_exclusions` to `comparison_summary()` instead of the original `exclusions` mapping. Keep `_pair_summary()` as the only separation caller of the shared helper.

- [ ] **Step 3: Pin zero-counter plumbing in existing MuScriptor and IDM comparison tests**

In `test_compare_joins_published_song_and_class_rows_without_rescoring` after loading `summary`, add:

```python
assert summary["pairing"]["exclusions"]["oaf_only_prediction_class"] == 0
assert summary["pairing"]["exclusions"]["muscriptor_only_prediction_class"] == 0
```

In `test_compare_oaf_idm_publishes_deterministic_pair_and_diagnostics`, after its existing `summary = json.loads(...)`, add:

```python
assert summary["pairing"]["exclusions"]["oaf_only_prediction_class"] == 0
assert summary["pairing"]["exclusions"]["idm_only_prediction_class"] == 0
```

These assertions prove the two model callers do not silently discard Task 1's exclusion result even when the baseline fixtures have no asymmetry.

- [ ] **Step 4: Add a test-only parent full-mix `ride` false positive to the separation publish regression**

In `tests/benchmark/test_separation_comparison.py`, add:

```python
def _add_parent_prediction_only_ride(fixture) -> str:
    from src.benchmark.backends import NativeEvent, NativePrediction
    from src.benchmark.mapping import map_oaf_prediction
    from src.benchmark.oaf_corpus_run import parse_oaf_corpus_run, write_oaf_corpus_run
    from src.benchmark.prediction_artifact import (
        read_prediction_artifact,
        render_prediction_artifact,
    )

    snapshot = parse_oaf_corpus_run(fixture.run_path.read_bytes())
    row = next(
        item
        for item in snapshot["items"]
        if item["execution_disposition"] == "inferred"
    )
    prediction_path = fixture.oaf_output_dir / row["prediction_path"]
    artifact = read_prediction_artifact(prediction_path.read_bytes())

    ride = NativeEvent(
        time_sec=0.75,
        native_class_id="midi_51",
        model_output_bin=51,
        native_midi_note=51,
        native_metadata={"upstream_8hit_group_id": "ride"},
        confidence=0.8,
        velocity_midi=90,
    )
    native = NativePrediction(
        audio=artifact.prediction.audio,
        descriptor=artifact.prediction.descriptor,
        events=tuple(event.native for event in artifact.prediction.events) + (ride,),
    )
    mapped, _ = map_oaf_prediction(native)
    content = render_prediction_artifact(mapped)
    prediction_path.write_bytes(content)
    updated = read_prediction_artifact(content)
    row["prediction_artifact_sha256"] = updated.artifact_sha256
    write_oaf_corpus_run(fixture.run_path, snapshot)
    return str(row["simfile_id"])
```

The helper changes only test fixture bytes and the matching run-snapshot artifact hash.

- [ ] **Step 5: Strengthen the existing separation publish test with source evidence, paired omission, and counts**

Inside `test_comparison_publishes_paired_csvs_summary_and_native_evidence`, immediately after building the `failed_count=0` fixture:

```python
prediction_only_simfile_id = _add_parent_prediction_only_ride(fixture)
```

After the pilot succeeds, read full-mix `per_class.csv` and require six source rows:

```python
with (outcome.run_path.parent / "views/full_mix/reports/per_class.csv").open(
    "r", encoding="utf-8", newline=""
) as handle:
    full_mix_ride_rows = [
        row
        for row in csv.DictReader(handle)
        if row["simfile_id"] == prediction_only_simfile_id
        and row["common_class"] == "ride"
    ]

assert len(full_mix_ride_rows) == 6
assert all(row["reference_support"] == "0" for row in full_mix_ride_rows)
assert all(int(row["prediction_support"]) > 0 for row in full_mix_ride_rows)
```

For each derived view, prove the paired class CSV omits the source-only `ride` rows and the pair summary counts all six omissions:

```python
for view_name in ("spleeter", "htdemucs"):
    with (comparison_dir / view_name / "paired_per_class.csv").open(
        "r", encoding="utf-8", newline=""
    ) as handle:
        paired_rows = list(csv.DictReader(handle))
    assert not any(
        row["simfile_id"] == prediction_only_simfile_id
        and row["common_class"] == "ride"
        for row in paired_rows
    )
    exclusions = summary["pairing"][view_name]["exclusions"]
    assert exclusions["full_mix_only_prediction_class"] == 6
    assert exclusions[f"{view_name}_only_prediction_class"] == 0
```

Keep the existing population/resource/event-micro assertions unchanged.

- [ ] **Step 6: Run all caller suites together**

```bash
uv run pytest -q \
  tests/benchmark/test_published_comparison.py \
  tests/benchmark/test_muscriptor_comparison.py \
  tests/benchmark/test_muscriptor_comparison_coverage.py \
  tests/benchmark/test_separation_comparison.py \
  tests/benchmark/test_idm_comparison.py \
  tests/benchmark/test_cross_comparison.py
```

Expected: PASS. This is the atomic return-signature/caller-plumbing gate.

- [ ] **Step 7: Run focused formatting/lint for every production caller**

```bash
uv run ruff check \
  src/benchmark/published_comparison.py \
  src/benchmark/muscriptor_comparison.py \
  src/benchmark/idm_comparison.py \
  src/benchmark/separation_comparison.py
uv run ruff format --check \
  src/benchmark/published_comparison.py \
  src/benchmark/muscriptor_comparison.py \
  src/benchmark/idm_comparison.py \
  src/benchmark/separation_comparison.py
```

Expected: PASS.

- [ ] **Step 8: Commit caller plumbing and publish regression**

```bash
git add \
  src/benchmark/muscriptor_comparison.py \
  src/benchmark/idm_comparison.py \
  src/benchmark/separation_comparison.py \
  tests/benchmark/test_muscriptor_comparison.py \
  tests/benchmark/test_idm_comparison.py \
  tests/benchmark/test_separation_comparison.py
git commit -m "fix: report asymmetric class exclusions"
```

---

### Task 3: Re-run comparison only, close repository verification, and record production evidence

**Files:** no new repository files expected; update PR #32 and HPA-305 with command evidence only.

**Interfaces:**
- Consumes `SeparationComparisonRequest` and `compare_oaf_separation()` from `src/benchmark/separation_comparison.py`.
- Produces the existing HPA-328 comparison directory and summary with explicit class exclusion counters, without rerunning separator/OaF inference.

- [ ] **Step 1: Run focused static checks on all touched files**

```bash
uv run ruff check \
  src/benchmark/published_comparison.py \
  src/benchmark/muscriptor_comparison.py \
  src/benchmark/idm_comparison.py \
  src/benchmark/separation_comparison.py \
  tests/benchmark/test_published_comparison.py \
  tests/benchmark/test_muscriptor_comparison_coverage.py \
  tests/benchmark/test_muscriptor_comparison.py \
  tests/benchmark/test_idm_comparison.py \
  tests/benchmark/test_separation_comparison.py
uv run ruff format --check \
  src/benchmark/published_comparison.py \
  src/benchmark/muscriptor_comparison.py \
  src/benchmark/idm_comparison.py \
  src/benchmark/separation_comparison.py \
  tests/benchmark/test_published_comparison.py \
  tests/benchmark/test_muscriptor_comparison_coverage.py \
  tests/benchmark/test_muscriptor_comparison.py \
  tests/benchmark/test_idm_comparison.py \
  tests/benchmark/test_separation_comparison.py
uv run pylint --errors-only --disable=E1120,E0401 --jobs=1 \
  src/benchmark/published_comparison.py \
  src/benchmark/muscriptor_comparison.py \
  src/benchmark/idm_comparison.py \
  src/benchmark/separation_comparison.py
```

Expected: all commands exit 0.

- [ ] **Step 2: Run only the recorded production comparison**

```bash
export SEPARATION_RUN="artifacts/benchmark/oaf-separation-pilot/runs/oaf-separation-8e66abde20b8f590/run.json"
: "${REFERENCE_MANIFEST:?set REFERENCE_MANIFEST to the PR #31 reproduced HPA-324 manifest}"
: "${TIMING_MANIFEST:?set TIMING_MANIFEST to the PR #31 reproduced HPA-323 manifest}"
: "${REVIEWED_SUBSET_MANIFEST:?set REVIEWED_SUBSET_MANIFEST to the PR #31 reproduced HPA-327 manifest}"
: "${CORPUS_CACHE:?set CORPUS_CACHE to the PR #31 verified HPA-321 cache}"
```

Then:

```bash
uv run python - <<'PY'
import os
from pathlib import Path

from src.benchmark.separation_comparison import (
    SeparationComparisonRequest,
    compare_oaf_separation,
)

run_path = Path(os.environ["SEPARATION_RUN"])
outcome = compare_oaf_separation(
    SeparationComparisonRequest(
        run_path=run_path,
        reference_manifest_path=Path(os.environ["REFERENCE_MANIFEST"]),
        timing_manifest_path=Path(os.environ["TIMING_MANIFEST"]),
        subset_manifest_path=Path(os.environ["REVIEWED_SUBSET_MANIFEST"]),
        output_dir=run_path.parent / "comparison",
        cache_dir=Path(os.environ["CORPUS_CACHE"]),
    )
)
print(outcome)
PY
```

Expected: a `SeparationComparisonOutcome` is printed and comparison artifacts publish without a simfile-241 `ride` integrity error.

Do not rerun Spleeter, HTDemucs, or OaF inference.

- [ ] **Step 3: Verify the production Spleeter exclusion evidence**

```bash
uv run python - <<'PY'
import csv
import json
from pathlib import Path

run_dir = Path(
    "artifacts/benchmark/oaf-separation-pilot/runs/"
    "oaf-separation-8e66abde20b8f590"
)
with (run_dir / "comparison/spleeter/paired_per_class.csv").open(
    "r", encoding="utf-8", newline=""
) as handle:
    rows = list(csv.DictReader(handle))
assert not any(
    row["simfile_id"] == "241" and row["common_class"] == "ride"
    for row in rows
)

summary = json.loads((run_dir / "comparison/summary.json").read_text(encoding="utf-8"))
exclusions = summary["pairing"]["spleeter"]["exclusions"]
assert exclusions["full_mix_only_prediction_class"] == 6, exclusions
assert exclusions["spleeter_only_prediction_class"] == 0, exclusions
print("HPA-305 production comparison PASS: six source-only ride rows counted")
PY
```

Expected: the PASS line above. Record the HTDemucs class exclusion counters as observed, but do not assume they are zero unless the production evidence says so.

- [ ] **Step 4: Confirm the HPA-328 finalizer prerequisite is restored**

```bash
RUN_DIR="${SEPARATION_RUN%/run.json}"
test -f "$RUN_DIR/comparison/summary.json"
test -f "$RUN_DIR/comparison/spleeter/paired_per_song.csv"
test -f "$RUN_DIR/comparison/spleeter/paired_per_class.csv"
test -f "$RUN_DIR/comparison/htdemucs/paired_per_song.csv"
test -f "$RUN_DIR/comparison/htdemucs/paired_per_class.csv"
```

Expected: all checks exit 0. HPA-305 does not choose or alter HPA-328's benchmark decision/rationale.

- [ ] **Step 5: Run repository-wide verification**

```bash
uv run pytest -q
uv run ruff check .
uv run ruff format --check src tests scripts
uv run pylint --errors-only --disable=E1120,E0401 --jobs=1 src
git diff --check origin/main...HEAD
git status --short
```

Expected: all test/static gates pass and no generated benchmark artifacts are staged.

- [ ] **Step 6: Scope audit**

```bash
git diff --stat origin/main...HEAD
git diff --name-only origin/main...HEAD
```

Expected implementation files beyond the planning docs:

```text
src/benchmark/published_comparison.py
src/benchmark/muscriptor_comparison.py
src/benchmark/idm_comparison.py
src/benchmark/separation_comparison.py
tests/benchmark/test_published_comparison.py
tests/benchmark/test_muscriptor_comparison_coverage.py
tests/benchmark/test_muscriptor_comparison.py
tests/benchmark/test_idm_comparison.py
tests/benchmark/test_separation_comparison.py
```

Reject implementation changes in `cohort_scoring.py`, `reports.py`, taxonomy/mapping, schemas, or separator/OaF execution unless new artifact evidence invalidates this plan.

- [ ] **Step 7: Record production closeout**

Post on PR #32 and HPA-305:

- the six Task 0 support tuples;
- focused caller-suite result;
- repository-wide test/static result;
- production `SeparationComparisonOutcome`;
- Spleeter `full_mix_only_prediction_class=6` and `spleeter_only_prediction_class=0`;
- observed HTDemucs class exclusion counters;
- confirmation that simfile 241 `ride` remains in source full-mix `per_class.csv` but not in the paired Spleeter class product;
- confirmation that HPA-328's comparison/finalizer prerequisite is restored.

Do not mark HPA-305 complete if Task 0 was never recorded or the real comparison was not rerun on the final implementation tree.

---

## Risks and Stop Conditions

**Task 0 invalidates the design:** HPA-305/PR #32 stay open and are re-scoped from `views/spleeter/reports/event_diagnostics.jsonl` as described in Task 0 Step 3. Do not merge a partial comparison relaxation.

**Shared helper return change:** all three callers are updated in Task 2 and gated by MuScriptor, IDM, separation, and cross-comparison suites. No compatibility wrapper is added for this internal seam.

**Exclusion interpretation:** the two new counters count omitted score rows, not unique class identities. Keep the naming/documentation stable so later report consumers do not treat `6` as six distinct drum classes.

## Plan Self-Review

**Spec coverage:** Task 0 enforces the hypothesis before code. Task 1 centralizes reference-aware pairing, deterministic sorting, support validation, and exclusion counts. Task 2 threads those counts through every current caller and proves zero/nonzero summary behavior. Task 3 validates the real run and repository gates.

**Review corrections:** Python is pinned to 3.12. `_class_key_sort_key()` replaces the assigned lambda/nested duplicate. Shared prediction support is not redundantly validated. One malformed direct-helper test is sufficient because production report readers already validate support types/ranges.

**Reference invariant:** class-constrained matching plus `_class_scores()` unmatched-reference inclusion means any reference-supported class must publish a row in every successfully scored view. That is why only prediction-only one-sided rows are droppable.

**Paired-product accounting:** one-sided source rows are not synthesized; their omission is counted in `pairing.exclusions`, and source `per_class.csv`/diagnostics retain the actual false-positive evidence.

**Type consistency:** `paired_class_rows()` has one new tuple return contract used identically by MuScriptor, IDM, and separation callers. The private `_paired_class_rows` migration alias naturally shares that contract.

**Placeholder scan:** no undefined implementation bodies, `TBD`, or `TODO` steps remain. Production environment variables are explicit references to the already reproduced PR #31 evidence.

**YAGNI:** no dense grid, policy configuration, new result type, schema bump, scorer rewrite, or compatibility layer. A tuple plus the existing exclusions maps is sufficient.

**Single-PR delivery:** planning and implementation stay on PR #32; Task 0 is the only gate before code changes.
