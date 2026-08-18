# HPA-328 Task 7 implementation report

## Implementation

Task 7 now publishes each fixed derived OaF view through the existing HPA-325
cohort scorer. The implementation:

- adds the exact frozen `SEPARATION_FAILURE_TO_COHORT_REASON` mapping;
- retains the complete reviewed-subset population in both derived cohorts,
  including failed separator/OaF members;
- re-reads and validates persisted prediction artifacts against each native
  HPA-328 view row before calling
  `cohort_item_from_validated_prediction_artifact()`;
- builds every non-success scorer item through
  `cohort_item_without_prediction()` and the frozen collapsed reason mapping;
- scores with `score_cohort(identity, items, diagnostics_for=successful_ids)`
  and writes reports only with `write_cohort_reports()`; and
- leaves native status, failure codes, and evidence in `run.json` unchanged.

The Task 6 separator, canonical-input, OaF, resume, and publication paths are
otherwise unchanged. Derived reports are written under
`views/spleeter/reports` and `views/htdemucs/reports`.

## TDD evidence

- Baseline: `uv run pytest -q tests/benchmark/test_separation_pilot_acceptance.py tests/benchmark/test_cohort_scoring.py` — PASS, 104 passed.
- RED: the new acceptance tests failed as expected because the frozen mapping
  export and derived report population did not yet exist (1 passed, 2 failed).
- GREEN: the final focused Task 6/Task 7/HPA-325 suite — PASS, 129 passed.

## Verification

- `uv run ruff check src/benchmark/separation_pilot.py tests/benchmark/test_separation_pilot_acceptance.py` — PASS.
- `uv run black --check src/benchmark/separation_pilot.py tests/benchmark/test_separation_pilot_acceptance.py` — PASS.
- `uv run pylint --errors-only --disable=E1120,E0401 src/benchmark/separation_pilot.py tests/benchmark/test_separation_pilot_acceptance.py` — PASS.
- `git diff --check` — PASS.

## Changed files

- `src/benchmark/separation_pilot.py`
- `tests/benchmark/test_separation_pilot_acceptance.py`
- `.superpowers/sdd/hpa-328-oaf-separation-ablation-plan/task-7-report.md`

## Hashes

- Base commit: `52b42063d79b134cd7e9980e179a40ac10882031`
- `src/benchmark/separation_pilot.py`: `17f6fff0f7f75d1974a6408feeec7741d6d8ebffbd946d3c322790d0de4f4d07`
- `tests/benchmark/test_separation_pilot_acceptance.py`: `2d03812fdeba4f1b47d4aad866702a239bf1378c48c6da65eac3136769489252`

## Commit

- `feat: score separated OaF cohorts`

## Limitations

- Verification uses only synthetic offline fixtures and fake separator/OaF
  seams; no operational corpus, separator runtime, or OaF checkpoint was
  accessed.
- The full repository suite was not rerun in this task. HPA-328 comparison,
  handoff, CLI, and later tasks remain untouched.
