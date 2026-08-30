# HPA-305 Per-Class Comparison Asymmetry Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let published per-class comparisons tolerate legitimate prediction-only class rows on either side while remaining fail-closed for missing or inconsistent reference-supported evidence, then prove the recorded HPA-328 separation comparison can publish.

**Architecture:** Keep the production change entirely in `src/benchmark/published_comparison.py::paired_class_rows()`. Validate the union of pairable class keys, omit only one-sided rows whose published support proves prediction-only evidence, render only shared keys, and leave HPA-325 scoring, report schemas, `paired_song_rows()`, and separation-specific join wiring unchanged.

**Tech Stack:** Python 3.13, existing HPA-325 published report models, pytest, Ruff, Pylint, existing Crux benchmark CLI/artifact readers.

**Spec:** `docs/superpowers/specs/2026-08-29-hpa-305-per-class-comparison-asymmetry-design.md`

## Global Constraints

- Keep HPA-305 in one PR: design, implementation plan, tests, production fix, and verification evidence stay on PR #32.
- Task 0 is a hard gate. Do not change tests or production code until the six recorded simfile-241 full-mix `ride` rows are inspected and recorded.
- Proceed only if every recorded one-sided `ride` row has `reference_support == 0` and `prediction_support > 0`; otherwise stop HPA-305's comparison-layer implementation and investigate upstream reference/scoring evidence.
- Change production behavior only in `src/benchmark/published_comparison.py::paired_class_rows()` plus one small private support-reading helper in that module.
- Keep `paired_song_rows()` exact-grid and unchanged.
- Do not modify HPA-325 scoring, `src/benchmark/cohort_scoring.py`, taxonomy/mapping, report schemas, comparison artifact schemas, separator execution, or `src/benchmark/separation_comparison.py` join wiring.
- A shared class key must have equal authoritative `reference_support` on both sides.
- A one-sided class key may be omitted only when its present row has non-boolean integer `reference_support == 0` and non-boolean integer `prediction_support > 0`.
- A one-sided row with positive reference support, zero prediction support, negative/malformed support, or a shared row with unequal reference support raises `ComparisonIntegrityError`.
- Do not synthesize missing `PublishedClassRow` values; `paired_per_class.csv` remains a paired-delta product over rows actually published by both cohorts.
- Cover both left-only and right-only prediction asymmetry because separation, MuScriptor, and IDM all consume the shared helper.
- Retarget the existing MuScriptor positive-reference asymmetry regression instead of deleting it.
- Reuse the already reproduced HPA-328 artifacts for verification. Do not rerun Spleeter, HTDemucs, or OaF merely to repair comparison semantics.
- Do not add a join-policy option, framework, schema version, compatibility layer, new dependency, or second PR.

## File Structure

| File | Responsibility in HPA-305 |
| --- | --- |
| `src/benchmark/published_comparison.py` | Validate shared/one-sided class support and render only safely pairable class rows. |
| `tests/benchmark/test_published_comparison.py` | Pin symmetric prediction-only omission, shared-support equality, malformed support, and strict song-grid semantics. |
| `tests/benchmark/test_muscriptor_comparison_coverage.py` | Preserve the existing positive-reference one-sided fail-closed regression with the new error contract. |
| `tests/benchmark/test_separation_comparison.py` | Prove the HPA-328 publish path succeeds when full mix has a real FP-only class absent from a derived view. |

No new production module, schema, or fixture file is required.

---

### Task 0: Prove the recorded simfile-241 rows are prediction-only before touching code

**Files:** workstation-local production artifacts only; no repository change.

**Interfaces:**
- Input: `artifacts/benchmark/oaf-separation-pilot/runs/oaf-separation-8e66abde20b8f590/views/full_mix/reports/per_class.csv`
- Evidence contract: six `(tolerance_ms, mode, reference_support, prediction_support)` rows for `simfile_id=241`, `common_class=ride`.
- Produces: a recorded PASS/STOP result on HPA-305 and PR #32.

- [ ] **Step 1: Confirm the exact reproduced report is present**

Run:

```bash
RUN_DIR=artifacts/benchmark/oaf-separation-pilot/runs/oaf-separation-8e66abde20b8f590
FULL_MIX_CLASS_REPORT="$RUN_DIR/views/full_mix/reports/per_class.csv"
test -f "$FULL_MIX_CLASS_REPORT"
```

Expected: exit 0. If the reproduced artifact is unavailable, stop before Task 1 and restore/reproduce the PR #31 evidence first; do not encode the hypothesis in tests without the production check.

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

Expected: six support lines followed by:

```text
HPA-305 preflight PASS: six prediction-only full-mix ride rows
```

- [ ] **Step 3: Record the gate result before implementation**

Post the six printed tuples and the PASS line verbatim on both HPA-305 and PR #32. If any assertion in Step 2 fails, record the actual row(s), stop this plan, and do not modify `paired_class_rows()`.

There is intentionally no commit for Task 0.

---

### Task 1: TDD the shared reference-aware asymmetric class join

**Files:**
- Modify: `tests/benchmark/test_published_comparison.py:10-85`
- Modify: `tests/benchmark/test_muscriptor_comparison_coverage.py:1022-1037`
- Modify: `src/benchmark/published_comparison.py:177-292`

**Interfaces:**
- Consumes:

```python
paired_class_rows(
    left: Mapping[tuple[str, int, str, str], PublishedClassRow | object],
    right: Mapping[tuple[str, int, str, str], PublishedClassRow | object],
    pairable_ids: set[str],
    *,
    left_label: str = "oaf",
    right_label: str = "muscriptor",
) -> list[dict[str, str]]
```

- Produces: the same public signature and paired CSV row shape, with reference-aware union validation before rendering shared keys.
- Adds one private helper:

```python
def _support_count(
    row: object,
    field: str,
    *,
    key: tuple[str, int, str, str],
    label: str,
) -> int:
    ...
```

- [ ] **Step 1: Add shared-helper test data and the six-key left-only production regression**

In `tests/benchmark/test_published_comparison.py`, import `paired_class_rows` next to `paired_song_rows`, then add this local row helper:

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
```

Add the recorded left-only shape across all six score keys:

```python
def test_paired_class_rows_omits_left_only_prediction_only_rows() -> None:
    keys = {
        ("241", tolerance_ms, mode, "ride")
        for tolerance_ms in (30, 50, 100)
        for mode in ("raw", "aligned")
    }
    left = {
        key: _class_row(reference_support=0, prediction_support=1)
        for key in keys
    }

    rows = paired_class_rows(
        left,
        {},
        {"241"},
        left_label="full_mix",
        right_label="spleeter",
    )

    assert rows == []
```

- [ ] **Step 2: Add the symmetric right-only regression and fail-closed support tests**

Add:

```python
def test_paired_class_rows_omits_right_only_prediction_only_row() -> None:
    key = ("241", 50, "raw", "ride")

    rows = paired_class_rows(
        {},
        {key: _class_row(reference_support=0, prediction_support=2)},
        {"241"},
        left_label="full_mix",
        right_label="spleeter",
    )

    assert rows == []


def test_paired_class_rows_rejects_shared_reference_support_mismatch() -> None:
    key = ("241", 50, "raw", "ride")
    with pytest.raises(ComparisonIntegrityError, match="reference_support mismatch"):
        paired_class_rows(
            {key: _class_row(reference_support=1, prediction_support=1)},
            {key: _class_row(reference_support=2, prediction_support=1)},
            {"241"},
            left_label="full_mix",
            right_label="spleeter",
        )


def test_paired_class_rows_rejects_one_sided_zero_support_without_predictions() -> None:
    key = ("241", 50, "raw", "ride")
    with pytest.raises(ComparisonIntegrityError, match="has no prediction support"):
        paired_class_rows(
            {key: _class_row(reference_support=0, prediction_support=0)},
            {},
            {"241"},
            left_label="full_mix",
            right_label="spleeter",
        )


@pytest.mark.parametrize("bad_support", (True, -1, "0", None))
def test_paired_class_rows_rejects_malformed_support(bad_support: object) -> None:
    key = ("241", 50, "raw", "ride")
    with pytest.raises(ComparisonIntegrityError, match="reference_support is malformed"):
        paired_class_rows(
            {key: _class_row(reference_support=bad_support, prediction_support=1)},
            {key: _class_row(reference_support=0, prediction_support=1)},
            {"241"},
            left_label="full_mix",
            right_label="spleeter",
        )
```

The existing `test_paired_song_rows_rejects_key_grid_mismatch()` stays unchanged.

- [ ] **Step 3: Retarget the existing MuScriptor positive-reference asymmetry regression**

In `tests/benchmark/test_muscriptor_comparison_coverage.py`, keep the existing left-only `snare` row with `reference_support=1`. Change only the expected error contract:

```python
with pytest.raises(
    ComparisonIntegrityError,
    match="reference-supported class row is missing",
):
    _paired_class_rows(
        {key: row, extra_key: extra_row},
        {key: row},
        pairable_ids={"1"},
    )
```

Do not weaken the row to `reference_support=0`; this test remains the legacy fail-closed tripwire that blind intersection would break.

- [ ] **Step 4: Run the focused tests and confirm RED behavior**

Run:

```bash
uv run pytest -q \
  tests/benchmark/test_published_comparison.py \
  tests/benchmark/test_muscriptor_comparison_coverage.py::test_paired_class_rows_rejects_asymmetric_pairable_key_grid
```

Expected before production code changes:

- left-only prediction-only test fails with the current `per_class score key grid mismatch`;
- right-only prediction-only test fails with the same current mismatch;
- shared unequal reference-support test fails because current code silently renders it;
- one-sided `0/0` test fails with the current generic grid mismatch rather than the new contract;
- malformed-support test fails because current code does not validate support at this helper boundary;
- the retargeted existing MuScriptor test fails because the old message is still generic.

If the new tests do not fail for these reasons, inspect the current helper before proceeding.

- [ ] **Step 5: Implement the minimal support reader and union validation**

In `src/benchmark/published_comparison.py`, keep `_row_value()` unchanged and add immediately after it:

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

Replace only the key-grid block and render iteration inside `paired_class_rows()` with the reference-aware shape:

```python
    class_key_sort = lambda value: (int(value[0]), value[1], _MODES[value[2]], value[3])
    left_keys = {key for key in left if str(key[0]) in pairable_ids}
    right_keys = {key for key in right if str(key[0]) in pairable_ids}
    shared_keys = left_keys & right_keys

    for key in sorted(shared_keys, key=class_key_sort):
        left_reference_support = _support_count(
            left[key], "reference_support", key=key, label=left_label
        )
        right_reference_support = _support_count(
            right[key], "reference_support", key=key, label=right_label
        )
        _support_count(left[key], "prediction_support", key=key, label=left_label)
        _support_count(right[key], "prediction_support", key=key, label=right_label)
        if left_reference_support != right_reference_support:
            _fail(
                "per_class reference_support mismatch"
                f" for key={key} ({left_label}={left_reference_support},"
                f" {right_label}={right_reference_support})"
            )

    for present, present_keys, present_label, missing_label in (
        (left, left_keys - right_keys, left_label, right_label),
        (right, right_keys - left_keys, right_label, left_label),
    ):
        for key in sorted(present_keys, key=class_key_sort):
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
                    f" for key={key} ({present_label}_prediction_support={prediction_support})"
                )

    rows: list[dict[str, str]] = []
    for key in sorted(shared_keys, key=class_key_sort):
```

Keep the existing row rendering body unchanged except that it now iterates `shared_keys`. Do not synthesize a row for either one-sided set.

- [ ] **Step 6: Run the focused tests and confirm GREEN behavior**

Run:

```bash
uv run pytest -q \
  tests/benchmark/test_published_comparison.py \
  tests/benchmark/test_muscriptor_comparison_coverage.py::test_paired_class_rows_rejects_asymmetric_pairable_key_grid
```

Expected: PASS.

- [ ] **Step 7: Commit the shared-helper fix**

Run:

```bash
git add \
  src/benchmark/published_comparison.py \
  tests/benchmark/test_published_comparison.py \
  tests/benchmark/test_muscriptor_comparison_coverage.py
git commit -m "fix: allow prediction-only class asymmetry"
```

---

### Task 2: Prove the HPA-328 publish path with a real FP-only full-mix class

**Files:**
- Modify: `tests/benchmark/test_separation_comparison.py` near `test_comparison_publishes_paired_csvs_summary_and_native_evidence`.
- No production change in `src/benchmark/separation_comparison.py`.

**Interfaces:**
- Consumes the existing `build_reviewed_subset_oaf_fixture()`, `_subset_path()`, `_task6_seams()`, real HPA-325 scoring wrapper, and `run_oaf_separation_pilot()` integration path.
- Produces one synthetic fixture mutation in the test module only: parent full-mix OaF predicts `ride` once while the reference and derived OaF predictions do not.
- The full comparison still reaches `_pair_summary() -> paired_class_rows()`; no test-only join is substituted.

- [ ] **Step 1: Add a test-only helper that injects one valid full-mix `ride` false positive**

Add imports locally in the helper so the existing module collection cost stays small:

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

This updates both immutable-artifact bytes and the synthetic parent run's expected artifact hash so the existing production readers continue to validate the fixture. It does not change the shared fixture builder.

- [ ] **Step 2: Strengthen the existing publish integration test with the asymmetry**

Inside `test_comparison_publishes_paired_csvs_summary_and_native_evidence`, immediately after building the `failed_count=0` fixture, call:

```python
prediction_only_simfile_id = _add_parent_prediction_only_ride(fixture)
```

Keep the existing derived `aligned_backend_factory`: it emits only the fixture's kick prediction, so Spleeter/HTDemucs have no `ride` row while full mix does.

After the pilot succeeds, prove the source report contains the legal one-sided row:

```python
import csv

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

Then prove the paired delta product deliberately omits that one-sided class while still publishing:

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
```

Keep the existing artifact-set, population, resource, and summary assertions unchanged.

- [ ] **Step 3: Run the separation integration test**

Run:

```bash
uv run pytest -q \
  tests/benchmark/test_separation_comparison.py::test_comparison_publishes_paired_csvs_summary_and_native_evidence
```

Expected: PASS after Task 1. On the pre-fix helper this strengthened test would terminate in `ComparisonIntegrityError` at the final comparison.

- [ ] **Step 4: Run every current shared-helper caller/regression suite together**

Run:

```bash
uv run pytest -q \
  tests/benchmark/test_published_comparison.py \
  tests/benchmark/test_muscriptor_comparison.py \
  tests/benchmark/test_muscriptor_comparison_coverage.py \
  tests/benchmark/test_separation_comparison.py \
  tests/benchmark/test_idm_comparison.py \
  tests/benchmark/test_cross_comparison.py
```

Expected: PASS. This is the caller-symmetry gate: HPA-305 must not repair the full-mix-left/Spleeter-right case while leaving model comparisons broken for right-only prediction classes.

- [ ] **Step 5: Commit the publish-path regression**

Run:

```bash
git add tests/benchmark/test_separation_comparison.py
git commit -m "test: cover asymmetric separation class comparison"
```

---

### Task 3: Re-run comparison only, close repository verification, and record production evidence

**Files:**
- No new production files expected.
- Update PR #32 / HPA-305 with evidence after commands complete.
- Generated benchmark outputs remain outside git.

**Interfaces:**
- `SeparationComparisonRequest(...)`
- `compare_oaf_separation(request) -> SeparationComparisonOutcome`
- Existing `crux benchmark finalize-oaf-separation-pilot` command if an HPA-328 decision/rationale is already recorded.

- [ ] **Step 1: Run focused static checks on the final touched files**

Run:

```bash
uv run ruff check \
  src/benchmark/published_comparison.py \
  tests/benchmark/test_published_comparison.py \
  tests/benchmark/test_muscriptor_comparison_coverage.py \
  tests/benchmark/test_separation_comparison.py
uv run ruff format --check \
  src/benchmark/published_comparison.py \
  tests/benchmark/test_published_comparison.py \
  tests/benchmark/test_muscriptor_comparison_coverage.py \
  tests/benchmark/test_separation_comparison.py
uv run pylint --errors-only --disable=E1120,E0401 --jobs=1 \
  src/benchmark/published_comparison.py
```

Expected: all commands exit 0.

- [ ] **Step 2: Reuse the PR #31 production paths and run only `compare_oaf_separation()`**

The reproduced run itself is fixed:

```bash
export SEPARATION_RUN="artifacts/benchmark/oaf-separation-pilot/runs/oaf-separation-8e66abde20b8f590/run.json"
```

Reuse the exact published HPA-324/HPA-323/HPA-327/cache paths from the PR #31 reproduction workspace. Require them before invocation rather than silently substituting regenerated evidence:

```bash
: "${REFERENCE_MANIFEST:?set REFERENCE_MANIFEST to the PR #31 reproduced HPA-324 manifest}"
: "${TIMING_MANIFEST:?set TIMING_MANIFEST to the PR #31 reproduced HPA-323 manifest}"
: "${REVIEWED_SUBSET_MANIFEST:?set REVIEWED_SUBSET_MANIFEST to the PR #31 reproduced HPA-327 manifest}"
: "${CORPUS_CACHE:?set CORPUS_CACHE to the PR #31 verified HPA-321 cache}"
```

Run:

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

Expected:

- no `per_class score key grid mismatch`;
- no `reference-supported class row is missing` for simfile 241 `ride`;
- a `SeparationComparisonOutcome` is printed;
- `artifacts/benchmark/oaf-separation-pilot/runs/oaf-separation-8e66abde20b8f590/comparison/summary.json` exists;
- both Spleeter and HTDemucs paired CSV directories are published.

Do not rerun separator or OaF inference for this step.

- [ ] **Step 3: Verify the production paired class output intentionally omits the one-sided `ride` keys**

Run:

```bash
uv run python - <<'PY'
import csv
from pathlib import Path

root = Path(
    "artifacts/benchmark/oaf-separation-pilot/runs/"
    "oaf-separation-8e66abde20b8f590/comparison/spleeter"
)
with (root / "paired_per_class.csv").open("r", encoding="utf-8", newline="") as handle:
    rows = list(csv.DictReader(handle))
assert not any(
    row["simfile_id"] == "241" and row["common_class"] == "ride"
    for row in rows
)
print("HPA-305 production comparison PASS: simfile 241 ride remains source-only evidence")
PY
```

Expected: the PASS line above.

- [ ] **Step 4: Confirm HPA-328 finalization is no longer blocked by comparison publication**

If the reproduced HPA-328 workspace already contains its evidence-backed decision/rationale, rerun the existing finalizer with those exact values:

```bash
uv run crux benchmark finalize-oaf-separation-pilot \
  --run "$SEPARATION_RUN" \
  --subset-manifest "$REVIEWED_SUBSET_MANIFEST" \
  --output-manifest "$SEPARATION_HANDOFF" \
  --decision "$DECISION" \
  --rationale "$RATIONALE"
```

Do not invent a new HPA-328 benchmark decision inside HPA-305. If no prior decision/rationale is recorded, the HPA-305 gate is satisfied by successful comparison publication plus confirmation that the finalizer's missing comparison prerequisite is now present; leave the benchmark decision to HPA-328's owner/evidence process.

- [ ] **Step 5: Run repository-wide verification**

Run:

```bash
uv run pytest -q
uv run ruff check .
uv run ruff format --check src tests scripts
uv run pylint --errors-only --disable=E1120,E0401 --jobs=1 src
git diff --check origin/main...HEAD
git status --short
```

Expected: test/static gates pass; `git status --short` contains no unintended generated benchmark artifacts staged for commit.

- [ ] **Step 6: Scope audit**

Run:

```bash
git diff --stat origin/main...HEAD
git diff --name-only origin/main...HEAD
```

Expected implementation files beyond the already committed planning docs:

```text
src/benchmark/published_comparison.py
tests/benchmark/test_published_comparison.py
tests/benchmark/test_muscriptor_comparison_coverage.py
tests/benchmark/test_separation_comparison.py
```

Reject any implementation diff in `cohort_scoring.py`, `reports.py`, `separation_comparison.py`, taxonomy/mapping, schemas, or separator/OaF execution unless a new independently demonstrated defect invalidates this plan.

- [ ] **Step 7: Record the production closeout**

Post on PR #32 and HPA-305:

- the six Task 0 support tuples;
- focused shared-caller suite result;
- repository-wide test/static result;
- production `SeparationComparisonOutcome`;
- confirmation that simfile 241 `ride` is omitted only from the paired delta product while remaining present in full-mix `per_class.csv`;
- HPA-328 finalizer result, or an explicit note that comparison publication is restored and only HPA-328's independent decision choice remains.

Do not mark HPA-305 complete if Task 0 was never recorded or the real comparison was not rerun on the final implementation tree.

---

## Plan Self-Review

**Spec coverage:** Task 0 enforces the production hypothesis gate before code. Task 1 implements symmetric reference-aware class pairing and support validation while preserving song-grid strictness. Task 2 covers the actual HPA-328 publish path and all current shared-helper caller families. Task 3 proves the recorded production run publishes and closes the repository/operational gates.

**Test-surface correction:** The pre-existing MuScriptor coverage test is explicitly retained and retargeted; HPA-305 does not leave a stale assertion on the old generic grid-mismatch message. Both left-only and right-only prediction asymmetry are pinned in the shared helper so IDM/model callers receive the same semantics as separation.

**Malformed-evidence boundary:** `_support_count()` converts malformed support evidence into `ComparisonIntegrityError` before arithmetic/comparison. One-sided `0/0` rows remain invalid because HPA-325 `_class_scores()` cannot legitimately emit a class absent from both reference and prediction events.

**YAGNI:** One private support reader plus an in-place `paired_class_rows()` change is sufficient. There is no dense class grid, join-policy configuration, new comparison engine, schema bump, scorer change, or compatibility layer.

**Single-PR delivery:** Planning and implementation stay on PR #32. The implementation is blocked on Task 0, not split into a second PR.
