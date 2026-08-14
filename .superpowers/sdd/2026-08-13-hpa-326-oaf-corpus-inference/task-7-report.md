# HPA-326 Task 7 implementation report

## Implementation

Task 7 now adapts persisted OaF run rows to the existing HPA-325 scorer
contracts and finalizes each run through the existing scoring/report APIs. The
implementation:

- builds validated `CohortItem` rows for inferred/resumed, failed, skipped,
  quarantined, missing-prediction, and unavailable-reference cases;
- preserves mapping-derived reference events/coverage for non-success rows and
  emits the pinned zero-reference coverage when the upstream reference is not
  available;
- derives frozen `CohortIdentity` values from the HPA-324/HPA-323 run header and
  invokes `score_cohort(identity, tuple(items), diagnostics_for=())` followed by
  `write_cohort_reports(score_result, run_dir / "reports")`;
- projects aggregate RTF and full eligible-corpus wall time while excluding
  resume hits with no retained timing and recording duration coverage counts;
- persists projection values through the existing canonical run renderer,
  which routes binary floats through `quantize_six()`; and
- verifies that a clean resume validates exact artifacts, regenerates reports,
  and never constructs/transcribes an inference backend.

The persisted Task 6 prediction bytes remain unchanged. During scorer handoff,
source-keyed prediction audio IDs are normalized in memory to the HPA-325
simfile ID binding required by `cohort_item_from_artifacts()`.

## TDD evidence

- RED: the projection tests initially failed collection because
  `_project_runtime` did not exist.
- GREEN: the projection implementation made the corpus-run suite pass (34
  tests at that checkpoint).
- RED: the scorer/report acceptance test initially observed zero calls to the
  existing scorer seam.
- RED: the first adaptation tests failed collection because
  `_cohort_item_from_run_row` did not exist.
- GREEN: the scorer adaptation, existing scorer/report calls, runtime
  persistence, and zero-inference resume acceptance coverage now pass in the
  focused suite.

## Verification

- `rtk uv run pytest tests/benchmark/test_oaf_corpus_run.py tests/benchmark/test_oaf_corpus_run_acceptance.py tests/benchmark/test_cohort_scoring.py tests/benchmark/test_reports.py -q` — PASS, 161 passed.
- `rtk uv run pytest tests/benchmark/test_oaf_corpus_run_acceptance.py::test_run_oaf_corpus_resume_regenerates_reports_without_inference -q` — PASS, 1 passed.
- `rtk uv run pytest tests/benchmark/test_oaf_corpus_run.py::test_render_quantizes_runtime_projection_fields -q` — PASS, 1 passed.
- `rtk uv run ruff check src/benchmark/oaf_corpus_run.py tests/benchmark/test_oaf_corpus_run.py tests/benchmark/test_oaf_corpus_run_acceptance.py` — PASS.
- `rtk uv run black --check src/benchmark/oaf_corpus_run.py tests/benchmark/test_oaf_corpus_run.py tests/benchmark/test_oaf_corpus_run_acceptance.py` — PASS.
- `rtk uv run pylint src/benchmark/oaf_corpus_run.py` — PASS, 10.00/10.
- `rtk git diff --check` — PASS.

## Changed files

- `src/benchmark/oaf_corpus_run.py`
- `tests/benchmark/test_oaf_corpus_run.py`
- `tests/benchmark/test_oaf_corpus_run_acceptance.py`
- `.superpowers/sdd/2026-08-13-hpa-326-oaf-corpus-inference/task-7-report.md`

## Commit

- `feat: score OaF corpus control runs`

## Concerns and scope

- The full repository test suite was not rerun in this task; the required
  focused runner, scorer, and report suites plus module quality checks pass.
- Task 8 CLI wiring and HPA-325 scorer/report internals remain untouched.

## Review fix round 1

Addressed the three review findings without expanding beyond Task 7:

- raw prediction bytes are now checked against the persisted row's artifact
  SHA, source-audio identity/hash, input-view identity, and input-audio hash
  before any source-key-to-simfile-ID normalization; resume reuse applies the
  same persisted-evidence check;
- validated resume hits restore persisted `wall_time_sec` and `rtf` before
  projection, so rescoring a completed run retains its measured aggregate;
  and
- finite zero wall time is counted as a measured inference while source
  duration remains strictly positive.

### Review-fix TDD evidence

- RED: `rtk uv run pytest tests/benchmark/test_oaf_corpus_run_acceptance.py::test_scorer_rejects_raw_artifact_mismatch_before_source_binding -q` — FAIL, 1 failed (`success` was returned instead of `failed`).
- RED: `rtk uv run pytest tests/benchmark/test_oaf_corpus_run.py::test_project_runtime_counts_zero_elapsed_inference -q` — FAIL, 1 failed (`None` was returned instead of zero measured wall time).
- RED: `rtk uv run pytest tests/benchmark/test_oaf_corpus_run_acceptance.py::test_run_oaf_corpus_resume_retains_projection_timing -q` — FAIL, 1 failed (`aggregate_rtf` was `None` instead of approximately `1/3`).
- GREEN: `rtk uv run pytest tests/benchmark/test_oaf_corpus_run_acceptance.py::test_scorer_rejects_raw_artifact_mismatch_before_source_binding tests/benchmark/test_oaf_corpus_run.py::test_project_runtime_counts_zero_elapsed_inference tests/benchmark/test_oaf_corpus_run_acceptance.py::test_run_oaf_corpus_resume_retains_projection_timing -q` — PASS, 3 passed.
- GREEN: `rtk uv run pytest tests/benchmark/test_oaf_corpus_run.py tests/benchmark/test_oaf_corpus_run_acceptance.py tests/benchmark/test_cohort_scoring.py tests/benchmark/test_reports.py -q` — PASS, 164 passed.

### Review-fix verification

- `rtk uv run ruff check src/benchmark/oaf_corpus_run.py tests/benchmark/test_oaf_corpus_run.py tests/benchmark/test_oaf_corpus_run_acceptance.py` — PASS.
- `rtk uv run black --check src/benchmark/oaf_corpus_run.py tests/benchmark/test_oaf_corpus_run.py tests/benchmark/test_oaf_corpus_run_acceptance.py` — PASS.
- `rtk uv run pylint src/benchmark/oaf_corpus_run.py` — PASS, 10.00/10.
- `rtk git diff --check` — PASS.

### Review-fix commit

- `fix: harden OaF corpus resume scoring`
