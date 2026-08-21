# HPA-396 Task 6 report

## Scope

Implemented the approved offline Task 6 slice only. The checkout has no
production HPA-328 handoff, so no production smoke membership was frozen, no
`runtime/idm/smoke.json` was created, and no real IDM/HPA-328 inference was
invoked. The implementation is ready to consume a validated handoff later
without treating fixture IDs as production membership.

## RED

Before implementing the smoke API and tests:

```text
uv run pytest tests/benchmark/test_idm_pilot_run.py -q
```

Observed collection failure: `ImportError` because
`IDM_FULL_MIX_INPUT_VIEW_ID` and the Task 6 smoke API did not yet exist.

## GREEN

Focused Task 6 tests:

```text
uv run pytest tests/benchmark/test_idm_pilot_run.py -q
29 passed
```

Task 4 acceptance/runner and related IDM regressions:

```text
uv run pytest tests/benchmark/test_idm_pilot_run.py \
  tests/benchmark/test_idm_pilot_run_acceptance.py \
  tests/benchmark/test_idm_comparison.py -q
58 passed
```

Style and focused static checks:

```text
uv run ruff check src/benchmark/idm_pilot_run.py tests/benchmark/test_idm_pilot_run.py
uv run black --check src/benchmark/idm_pilot_run.py tests/benchmark/test_idm_pilot_run.py
uv run pylint --errors-only --disable=E1120,E0401 --jobs=1 src/benchmark/idm_pilot_run.py
git diff --check
```

All passed. `runtime/idm/smoke.json` is absent by design.

## Implementation and self-review

- Added strict canonical smoke manifest parsing/rendering/writing with exactly
  five positive unique IDs, one case for each binding reason, canonical order
  `short`, `long`, `sparse`, `dense`, `median_duration`, and optional loaded
  handoff membership validation.
- Added deterministic pre-IDM selection from successful/resumed HPA-328 rows
  using only source duration and reference common-event counts; ties select the
  lowest `simfile_id` after the required primary sort. No model scores are read.
- Added `IdmFullMixSmokeRequest` with the source cache owned only by this
  diagnostic request; `IdmPilotRunRequest` remains unchanged and has no source
  cache field.
- Reused `CacheIndexStore`, `resolve_source_audio`, the historical full-mix
  canonicalizer, the frozen IDM backend, prediction artifacts, and unchanged
  cohort scoring/report writers. The full-mix input view, inference config,
  run identity, run directory, and report directory are separate from the
  stem pilot and are never routed through `idm_comparison`.
- Added deterministic synthetic coverage for selection, canonical manifest
  validation, full-mix materialization, source/cache ownership, output-root
  alias rejection, fake-backend inference, scoring, and separate reports.

## Operational block

The production handoff prerequisite remains unmet. Before operational use,
load the real HPA-328 handoff, derive the five eligible IDs with
`select_idm_smoke_cases()`, validate and write the resulting manifest, commit
the real `runtime/idm/smoke.json`, then run the separate full-mix diagnostic
through the attested runtime. This task intentionally does not perform those
steps or invent IDs.

## Files

- `src/benchmark/idm_pilot_run.py`
- `tests/benchmark/test_idm_pilot_run.py`
- `.superpowers/sdd/2026-08-20-hpa-396-idm-stem-pilot/task-6-report.md`
