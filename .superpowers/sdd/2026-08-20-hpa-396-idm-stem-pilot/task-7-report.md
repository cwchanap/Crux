# HPA-396 Task 7 report

## Scope

Implemented the fixed `crux benchmark run-idm-pilot` CLI wiring only. The
command requires the immutable HPA-328 handoff/reference/timing inputs,
retained separation/stem roots, output root, isolated IDM runtime, model lock,
and model root. It accepts `--resume` and exposes no scope, seed, tuning,
device, or dtype controls. Optional full-mix smoke execution requires both
`--smoke-manifest` and `--source-cache-dir` and is routed through its separate
request/report path; the primary `IdmPilotRunRequest` remains stem-only.

## RED

Before implementing the command wiring:

```text
uv run pytest tests/test_cli_benchmark.py tests/test_cli_benchmark_coverage.py -q
```

Observed six new IDM CLI failures: the command and its expected request,
comparison, smoke-group, outcome, option-scope, and lazy-import seams did not
yet exist.

## GREEN

Focused CLI tests:

```text
uv run pytest tests/test_cli_benchmark.py tests/test_cli_benchmark_coverage.py -q
37 passed
```

Full HPA-396 verification set:

```text
uv run pytest \
  tests/benchmark/test_idm_model.py \
  tests/benchmark/test_idm_backend.py \
  tests/benchmark/test_idm_pilot_run.py \
  tests/benchmark/test_idm_pilot_run_acceptance.py \
  tests/benchmark/test_idm_comparison.py \
  tests/test_cli_benchmark.py \
  tests/test_cli_benchmark_coverage.py -q
154 passed
```

Focused static checks:

```text
uv run ruff check src/cli/benchmark.py tests/test_cli_benchmark.py tests/test_cli_benchmark_coverage.py
uv run black --check src/cli/benchmark.py tests/test_cli_benchmark.py tests/test_cli_benchmark_coverage.py
uv run pylint --errors-only --disable=E1120,E0401 --jobs=1 src/cli
git diff --check
```

All passed.

## Implementation and self-review

- Added lazy IDM runner/comparison/smoke imports and a thin Click callback that
  delegates request construction/orchestration to a private execution helper.
- Added canonical JSON output for primary counts, native failure histogram,
  timing diagnostics, comparison path/error, and optional smoke outcome. The
  final process status preserves the fixed `0` complete, `1` partial, and `2`
  fatal convention.
- Added deterministic tests for all required inputs, resume propagation,
  frozen option scope, canonical output, outcome propagation, smoke option
  pairing and request separation, and `--help` without importing IDM modules.

## Operational limitation

The checkout has no production HPA-328 immutable handoff or retained inputs,
so no real IDM inference or full-mix smoke was run. Operational execution
remains blocked until those artifacts, the attested isolated runtime/model,
and the production smoke manifest are available.

## Files

- `src/cli/benchmark.py`
- `tests/test_cli_benchmark.py`
- `tests/test_cli_benchmark_coverage.py`
- `.superpowers/sdd/2026-08-20-hpa-396-idm-stem-pilot/task-7-report.md`

## Fix Round 1

### RED

Added a regression before changing the command:

```text
uv run pytest tests/test_cli_benchmark.py::test_run_idm_pilot_skips_smoke_when_comparison_fails -q
1 failed
```

The failure showed that a nonfatal primary run still invoked the optional
full-mix smoke after `compare_oaf_idm` raised, allowing the smoke call to mask
the comparison failure.

### GREEN

The smoke branch now requires `comparison_error is None`, preserving the
canonical fatal comparison payload and exit code 2 while making zero smoke
calls after comparison failure.

```text
uv run pytest tests/test_cli_benchmark.py tests/test_cli_benchmark_coverage.py -q
38 passed

uv run pytest \
  tests/benchmark/test_idm_model.py \
  tests/benchmark/test_idm_backend.py \
  tests/benchmark/test_idm_pilot_run.py \
  tests/benchmark/test_idm_pilot_run_acceptance.py \
  tests/benchmark/test_idm_comparison.py \
  tests/test_cli_benchmark.py \
  tests/test_cli_benchmark_coverage.py -q
155 passed
```

Ruff, Black, Pylint (`src/cli`), and `git diff --check` also passed.
