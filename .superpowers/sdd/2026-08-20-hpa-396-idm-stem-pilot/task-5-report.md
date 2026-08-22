# HPA-396 Task 5 Report

## RED

Command:

```text
uv run pytest tests/benchmark/test_idm_comparison.py -q
```

Observed failure: pytest stopped during collection with
`ModuleNotFoundError: No module named 'src.benchmark.idm_comparison'`.

## GREEN

Focused implementation and regression command:

```text
uv run pytest tests/benchmark/test_idm_comparison.py tests/benchmark/test_published_comparison.py -q
```

Observed result: `20 passed`.

Additional checks:

```text
uv run ruff check src/benchmark/idm_comparison.py tests/benchmark/test_idm_comparison.py
uv run black --check src/benchmark/idm_comparison.py tests/benchmark/test_idm_comparison.py
git diff --check
```

All passed.

## Self-review

- `compare_oaf_idm()` reads the concrete IDM run snapshot and sibling `reports/oaf` / `reports/idm` directories.
- OaF and IDM identities are independently reconstructed and checked for reference, timing, taxonomy, lane-map, scoring, and frozen stem input identity.
- Pairing calls public `pairable_success_ids(..., require_identical_input_hash=True, left_label="oaf", right_label="idm")`; score joins and artifact writes also use only public `published_comparison.py` helpers.
- Full per-model populations remain in `summary.json` before the successful intersection. The summary retains native IDM failure counts as a histogram, mapped/unmapped/native-class coverage, runtime/RTF/peak-memory fields when present, and optional native-velocity diagnostics.
- Outputs are the four deterministic artifacts required by the task: `summary.json`, `summary.md`, `paired_per_song.csv`, and `paired_per_class.csv`.
- No MuScriptor private helper, generic comparison registry, scorer change, CLI wiring, or progress-ledger edit was added.

## Files

- `src/benchmark/idm_comparison.py`
- `tests/benchmark/test_idm_comparison.py`
- `.superpowers/sdd/2026-08-20-hpa-396-idm-stem-pilot/task-5-report.md`

## Concerns / limitations

- Tests use deterministic synthetic published reports; no production HPA-328 handoff or real IDM runtime was executed in this offline task.
- Velocity diagnostics are intentionally best-effort evidence from persisted IDM prediction artifacts and remain separate from headline scoring. Missing artifact paths produce unavailable diagnostics rather than altering pairing or scores.

## Fix Round 1

### RED

Added regressions for all four review findings, then ran:

```text
uv run pytest tests/benchmark/test_idm_comparison.py -q
```

Observed `11 failed, 3 passed`: the pre-fix comparison accepted equal missing
input hashes, did not bind the run ID to immutable header identity, trusted an
inconsistent native-failure map, and accepted ancestor/shadow or wrong-row
velocity artifacts.

### GREEN

Focused Task 5 verification:

```text
uv run pytest tests/benchmark/test_idm_comparison.py -q
14 passed
```

Task 5 plus published-comparison and relevant Task 4 regressions:

```text
uv run pytest tests/benchmark/test_idm_comparison.py \
  tests/benchmark/test_published_comparison.py \
  tests/benchmark/test_idm_pilot_run.py \
  tests/benchmark/test_idm_pilot_run_acceptance.py \
  tests/benchmark/test_prediction_artifact.py \
  tests/benchmark/test_prediction_artifact_coverage.py -q
110 passed
```

Style and diff checks:

```text
uv run black --check src/benchmark/idm_comparison.py tests/benchmark/test_idm_comparison.py
uv run ruff check src/benchmark/idm_comparison.py tests/benchmark/test_idm_comparison.py
uv run pylint --errors-only --disable=E1120,E0401 --jobs=1 src/benchmark/idm_comparison.py
git diff --check
```

All passed.

### Fix-round self-review

- Every successful OaF/IDM evidence row now requires valid canonical source and
  input SHA-256 values before calling the unchanged public pairing helper.
- The comparison recomputes `run_id` through public `build_run_id()` using all
  immutable header identity inputs and rejects stale or mutated lineage.
- Native failure counts are derived from failed item rows; a stored histogram
  must reconcile exactly, while an omitted map is reconstructed.
- Velocity diagnostics read only the Task-4 output root and require public
  `prediction_artifact_matches_run_row()` validation before extracting native
  velocity metadata. Shadow and wrong-row artifacts remain unavailable.
- No private MuScriptor helper, generic registry, scorer change, CLI wiring, or
  progress-ledger edit was added.

### Fix-round files

- `src/benchmark/idm_comparison.py`
- `tests/benchmark/test_idm_comparison.py`
- `.superpowers/sdd/2026-08-20-hpa-396-idm-stem-pilot/task-5-report.md`
