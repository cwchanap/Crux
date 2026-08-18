# HPA-328 Task 8B implementation report

## Result

**PASS.** Task 8B now publishes only HPA-328 separation-comparison evidence:
two deterministic paired CSV pairs, one canonical summary JSON, and one
combined Markdown summary. Pairing requires matching authoritative
`source_audio_sha256` and explicitly opts out of derived
`input_audio_sha256` equality. The comparison reuses the Task 8A reader,
pairing, joins, deltas, CSV, summary, and Markdown primitives.

## Implementation

- `src/benchmark/separation_comparison.py`
  - Validates the persisted HPA-328 run and all three HPA-325 report
    identities/lineages.
  - Pairs full mix with Spleeter and HTDemucs using
    `require_identical_input_hash=False` while retaining mandatory source
    hash validation.
  - Adds the only new metric calculation: paired event-micro sums of the
    published per-song TP/FP/FN counts for each tolerance/mode, followed by
    precision/recall/F1 derivation.
  - Computes FP/FN per minute from positive finite authoritative run-row
    durations only.
  - Publishes native run failure-code histograms, collapsed HPA-325
    populations, paired-set sizes, measured separator/OaF wall time, and
    retained stem/prediction/report bytes. It contains no cost or top-five
    fields.
  - Writes exactly the requested comparison artifact layout with deterministic
    paired row order.
- `src/benchmark/separation_pilot.py`
  - Adds one run-scoped comparison call after the two derived HPA-325 scoring
    passes, gated on all six reports being present. No execution/scoring
    semantics were changed.
- `tests/benchmark/test_separation_comparison.py`
  - Covers summed event-micro counts and authoritative duration rates,
    distinct derived input hashes, pilot wiring, and an end-to-end fake/offline
    artifact publication path.

## TDD evidence

### Baseline

Before edits, the existing Task 6 pilot suites passed:

```text
uv run pytest -q tests/benchmark/test_separation_pilot.py \
  tests/benchmark/test_separation_pilot_acceptance.py
26 passed in 50.11s
```

### RED

The new comparison tests were first run before the module existed:

```text
uv run pytest -q tests/benchmark/test_separation_comparison.py
ERROR during collection: ModuleNotFoundError: src.benchmark.separation_comparison
```

After adding the aggregation seam, the pairing/aggregation tests went green;
the pilot wiring test then intentionally failed because
`separation_pilot` had no comparison call seam. The narrow pilot wiring was
added and the focused suite was rerun.

### GREEN

The required focused suite passes:

```text
uv run pytest -q tests/benchmark/test_separation_comparison.py \
  tests/benchmark/test_separation_pilot_acceptance.py
7 passed in 9.71s
```

The existing Task 6 regression suite also passes:

```text
uv run pytest -q tests/benchmark/test_separation_pilot.py
23 passed in 46.61s
```

The integration test exercises the real pilot/report/comparison path with
offline seams only and verifies all six required output files, paired-set
size, native evidence fields, and retained byte totals.

## Verification

- Ruff check on the three changed Python files — PASS.
- Ruff format check on the three changed Python files — PASS.
- Pylint `--errors-only` on the three changed Python files — PASS.
- `git diff --check` — PASS.
- The feature commit's repository pre-commit checks (Ruff, Ruff format, and
  Pylint errors-only) — PASS.

## Changed files

- `src/benchmark/separation_comparison.py`
- `src/benchmark/separation_pilot.py`
- `tests/benchmark/test_separation_comparison.py`
- `.superpowers/sdd/hpa-328-oaf-separation-ablation-plan/task-8b-report.md`

## Hashes

- Base commit: `e84dbe65a905e2855a040ef337ee10216ffddba1`
- Feature commit: `b1cc985b4047b8d6cf4a034aabb57cfe711122f9`
- `src/benchmark/separation_comparison.py`:
  `f2da0f0c1f2fd0599856839adefb47ddee27a7813de866f798afaa8ce0b9cb0e`
- `src/benchmark/separation_pilot.py`:
  `18694def74c2df787b0e0df2d6461dd07f04d89fba19c992aace7d99c70b7033`
- `tests/benchmark/test_separation_comparison.py`:
  `e2ae17b0605dc8d8670b82e1e5b16ac2e14728da20210ae5171ba3da58e0a70f`

## Commit

- `feat: compare OaF separation views`
- Commit: `b1cc985b4047b8d6cf4a034aabb57cfe711122f9`

## Limitations

- Verification uses deterministic fake/offline fixtures; no production
  corpus, separator runtime, OaF checkpoint, network, or full-mix
  re-inference was invoked.
- The full repository suite was not rerun; HPA-396 handoff and later
  operator-decision work remain untouched.
- The comparison consumes the strict Task 8A HPA-325 report reader and does
  not add a legacy/partial-report fallback or alter Task 8A reader semantics.

## Review remediation: authoritative durations

Round-1 review identified that invalid or missing pairable-song durations were
omitted from the FP/FN-per-minute denominator while their published counts
remained in the numerator. The event-micro aggregator now validates every
pairable song before deriving any rate or returning comparison evidence. A
missing, null, nonfinite, zero, or negative authoritative duration raises
`ComparisonIntegrityError`, so the comparison fails before `_write_outputs`.

### RED

The new regression was run before the guard was added:

```text
uv run pytest -q tests/benchmark/test_separation_comparison.py -k positive_finite_duration
5 failed, 4 deselected
```

Each case incorrectly completed without raising: missing, null, nonfinite,
zero, and negative duration.

### GREEN

After the narrow validation guard:

```text
uv run pytest -q tests/benchmark/test_separation_comparison.py -k positive_finite_duration
5 passed, 4 deselected

uv run pytest -q tests/benchmark/test_separation_comparison.py tests/benchmark/test_separation_pilot_acceptance.py
12 passed
```

Focused Ruff, Ruff format, Pylint `--errors-only`, and `git diff --check`
also passed. The repository commit hooks passed the same checks.

### Remediation hashes

- Remediation commit: `64e6b305773db8a6fd50caaaf47b9836680a0897`
- `src/benchmark/separation_comparison.py`:
  `fdeacc142e15397cddf95c3c508b2a2c127e6ff788ac80ea36ac1545a9db8102`
- `tests/benchmark/test_separation_comparison.py`:
  `e8e5db6c3ddc8820debfeda8fde99f4564f9f34f1a5c9f9b05f9e67d3b567671`
