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
  - Retargets every production report-directory load and model-neutral
    comparison call to the shared modules. The private parser wrappers used by
    direct coverage tests also delegate to the strict reader; incomplete or
    legacy live report directories have no comparison fallback.
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

## Remediation round 1

The first review identified five strictness and reuse gaps. This remediation
keeps the same narrow Task8A surface and does not add HPA-328 comparison
outputs.

### RED evidence

The characterization baseline before the remediation changes was:

```text
rtk uv run pytest -q tests/benchmark/test_reports.py \
  tests/benchmark/test_muscriptor_comparison.py \
  tests/benchmark/test_muscriptor_comparison_coverage.py
97 passed in 2.52s
```

New regressions were then run before their production fixes:

```text
rtk uv run pytest -q tests/benchmark/test_reports.py -k 'incomplete or noncanonical'
5 failed
rtk uv run pytest -q tests/benchmark/test_muscriptor_comparison_coverage.py \
  -k 'source_hash or shared_summary or shared_markdown or without_summary'
5 failed, 1 passed
```

The failures covered missing summary rejection, missing/invalid source audio
identity, incomplete per-song and aggregate grids, schema/identity/label
parameterization, and canonical CSV tokens.

### GREEN evidence

The remediation now:

- always loads a live report directory through `read_cohort_reports()`;
  missing or legacy `summary.json` therefore fails rather than falling back to
  partial CSV parsing;
- validates every successful pair's source audio identity as a nonempty,
  lowercase SHA-256 before applying the optional input-hash policy;
- requires the complete tolerance × raw/aligned aggregate grid and the
  complete successful-item × tolerance × raw/aligned `per_song` grid;
- exposes shared schema, summary identity, title, and both comparison labels
  while retaining MuScriptor's exact default schema, artifact names, CSV shape,
  and Markdown heading; and
- rejects noncanonical CSV integer/decimal spellings such as `01`, `0.50`,
  and `1e-1`.

The focused GREEN run was:

```text
rtk uv run pytest -q tests/benchmark/test_reports.py \
  tests/benchmark/test_muscriptor_comparison.py \
  tests/benchmark/test_muscriptor_comparison_coverage.py
108 passed in 2.78s
```

Additional checks:

```text
rtk uv run ruff check src/benchmark/reports.py \
  src/benchmark/published_comparison.py src/benchmark/muscriptor_comparison.py \
  tests/benchmark/test_reports.py tests/benchmark/test_muscriptor_comparison.py \
  tests/benchmark/test_muscriptor_comparison_coverage.py
All checks passed!
rtk uv run ruff format --check src/benchmark/reports.py \
  src/benchmark/published_comparison.py src/benchmark/muscriptor_comparison.py \
  tests/benchmark/test_reports.py tests/benchmark/test_muscriptor_comparison.py \
  tests/benchmark/test_muscriptor_comparison_coverage.py
6 files already formatted
rtk git diff --check
PASS
```

### Remediation limitations and hash

The remediation is verified with deterministic local report/run fixtures only;
no production corpus, backend, checkpoint, or separator runtime was invoked.
Task8B comparison output publication and later HPA-328 handoff work remain
untouched. The strict reader still intentionally excludes
`event_diagnostics.jsonl` and `summary.md`.

Remediation commit and final file hashes are recorded after commit:

- Commit: `d747926a92ff8e4da13c3ff840e95faf3bf3494e`
- `src/benchmark/reports.py`: `47de6be16f15e8b8e9b2583084e81ae06101b1570d3b762cda48ccdffe2ac655`
- `src/benchmark/published_comparison.py`: `7a95620fd56a01839a49756590c97bab75fb36ff4ccd633ab92644e919bad16e`
- `src/benchmark/muscriptor_comparison.py`: `9b3ce4a7d4a75af084a11e70b2c3292beb1c8d17748d6b2f0f79b9aaf28b872e`
- `tests/benchmark/test_reports.py`: `42ca9056f1ada6dfbfc660f0e647670d3a340d88ffe0f6194c8ddca78dad87df`
- `tests/benchmark/test_muscriptor_comparison.py`: `e43de918e94aa44d57e97f67f785af700ea5a36da2d29a78b3715817916bb79e`
- `tests/benchmark/test_muscriptor_comparison_coverage.py`: `ace1e215e6523c288b73aef69b6cccea02a7ea7fbd01deeef66c6b63d8342bea`
