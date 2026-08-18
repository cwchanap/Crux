# HPA-328 Task 8A implementation report

## Result

**PASS.** Task 8A now has a strict typed reader for published HPA-325 cohort
reports and a narrow model-neutral comparison module. MuScriptor consumes the
shared reader and comparison primitives while retaining its existing four
artifact names, CSV columns, summary shape, Markdown, and default identical
input-hash policy. No HPA-328 comparison outputs or backend/subset policy were
added.

## Implementation

- `src/benchmark/reports.py`
  - Adds `PublishedCohortReports` and typed published row/aggregate models.
  - Adds `read_cohort_reports()` beside `write_cohort_reports()`.
  - Uses the writer-owned `_ITEM_FIELDNAMES`, `_PER_SONG_FIELDNAMES`, and
    `_PER_CLASS_FIELDNAMES` for CSV schemas.
  - Strictly validates canonical JSON schemas, numeric values/ranges, report
    identities, item status/reason semantics, duplicate IDs/score keys, and
    score rows belonging only to successful items.
  - Parses `summary.json` event-micro aggregates independently from the CSV
    score rows.
- `src/benchmark/published_comparison.py`
  - Moves the existing model-neutral pairability, metric deltas, paired
    song/class joins, aggregate rows, population/runtime summaries,
    deterministic CSV, canonical summary, and Markdown rendering.
  - Parameterizes output labels and
    `require_identical_input_hash` (default `True`). Source-audio identity
    remains mandatory; the explicit opt-out supports derived input views.
- `src/benchmark/muscriptor_comparison.py`
  - Retains backend-family, run-lineage, and subset-policy validation locally.
  - Retargets report parsing and model-neutral comparison calls to the shared
    modules. A compatibility-only private parser path remains for existing
    in-memory fixtures that predate `summary.json`; published directories with
    `summary.json` always use the strict reader.
- Tests cover valid round-trip reading, independent event-micro parsing,
  malformed schema/numbers/identities/duplicates/non-success score rows, and
  label/input-hash parameterization.

## TDD evidence

### Baseline

Before edits, the characterization suite passed:

```text
rtk uv run pytest -q tests/benchmark/test_muscriptor_comparison.py \
  tests/benchmark/test_muscriptor_comparison_coverage.py
79 passed in 2.14s
```

### RED

The new reader tests were run before implementation:

```text
rtk uv run pytest -q tests/benchmark/test_reports.py -k 'read_cohort_reports'
```

They failed at collection because `ReportIntegrityError`/`read_cohort_reports`
did not yet exist. This was the intentional RED state for the reader seam.

### GREEN

The required focused suite passes after implementation:

```text
rtk uv run pytest -q tests/benchmark/test_reports.py \
  tests/benchmark/test_muscriptor_comparison.py \
  tests/benchmark/test_muscriptor_comparison_coverage.py
97 passed in 2.51s
```

## Verification

- `rtk uv run ruff check --fix` on all five changed Python files — PASS;
  final check reported `All checks passed!`.
- `rtk uv run ruff format --check` on all five changed Python files — PASS.
- `rtk uv run pylint --errors-only src/benchmark/reports.py
  src/benchmark/published_comparison.py
  src/benchmark/muscriptor_comparison.py` — PASS.
- `rtk git diff --check` — PASS.

## Changed files

- `src/benchmark/reports.py`
- `src/benchmark/published_comparison.py`
- `src/benchmark/muscriptor_comparison.py`
- `tests/benchmark/test_reports.py`
- `tests/benchmark/test_muscriptor_comparison_coverage.py`
- `.superpowers/sdd/hpa-328-oaf-separation-ablation-plan/task-8a-report.md`

## Hashes

- Base commit: `4e256ff852561b5b6ba6c602c8eca34d102c8b30`
- Implementation commit: `b5612c1c98252a2e03905f27627b8c5492b0e283`
- `src/benchmark/reports.py`: `c3c566cc41d0cdf4f56f4e10a5334ecbcc2ce273731cc501b317a59ab6f44a5c`
- `src/benchmark/published_comparison.py`: `bb47615d5793e6a892235fc3b0bff1db1250d0afe9a899a3319a7f09a7aef191`
- `src/benchmark/muscriptor_comparison.py`: `15d3a3136f2e34a156b34ce0710c686723d90e691422caa30feb7a028ee4746c`
- `tests/benchmark/test_reports.py`: `8f3ad3f544f727456beb99c0bea25b2f2c87012e47d00baf714ad0ac92b88ae1`
- `tests/benchmark/test_muscriptor_comparison_coverage.py`: `0753b316edea5a6e0d6b1acf6c255afdc4ed1ab5e0a6697a613ae00e07891239`

## Commit

- `refactor: share published benchmark comparison`

## Limitations

- Verification uses deterministic local fixtures; no production corpus,
  backend, model checkpoint, or HPA-328 separator runtime was invoked.
- The full repository suite was not rerun in this task; Tasks 8B and later
  HPA-328 comparison/handoff/CLI work remain untouched.
- The strict reader intentionally covers the four published report files
  needed by Task 8A and does not parse `event_diagnostics.jsonl` or
  `summary.md`.
