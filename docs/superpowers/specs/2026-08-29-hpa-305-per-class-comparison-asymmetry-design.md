# HPA-305 Per-Class Comparison Asymmetry Design

## Context

HPA-305 records a deterministic failure from the reproduced HPA-328 separation pilot run `oaf-separation-8e66abde20b8f590` after the HPA-424 source-binding fix landed. All 30 fixed-subset Spleeter and HTDemucs separations, derived OaF inference, and HPA-325 cohort scoring complete, but the final separation comparison aborts in `src/benchmark/published_comparison.py::paired_class_rows()` because the full-mix and Spleeter `per_class.csv` key grids differ for simfile 241, class `ride`, at all three tolerances and both score modes.

The comparison currently treats any one-sided per-class key as corrupt evidence. That is stricter than the scorer contract and stricter than the earlier MuScriptor comparator, which intentionally joined class rows on the intersection because models can have asymmetric prediction-only class support.

HPA-627 remains operationally blocked on gated Hugging Face access, so HPA-305 is the next executable Crux benchmark task.

## Root-cause hypothesis

HPA-325 does not omit a class merely because a model predicted zero events for it. `_class_scores()` builds the class set from matched events, unmatched reference events, and unmatched prediction events. Therefore a reference-supported class with zero predictions still produces a per-class row with nonzero `reference_support` and zero `prediction_support`.

The production failure is instead expected to be a valid prediction-only asymmetry:

- the authoritative reference for simfile 241 has zero `ride` support;
- full-mix OaF predicts one or more false-positive `ride` events, so HPA-325 emits a `ride` row with `reference_support == 0` and `prediction_support > 0`;
- Spleeter predicts no `ride` events, so no `ride` row exists;
- exact key-grid equality rejects the resulting one-sided row.

Because class matching is class-constrained, `reference_support = TP + FN` is the authoritative ground-truth count for that class. A one-sided row with positive reference support therefore cannot be treated as harmless prediction asymmetry.

## Hard production preflight — implementation Step 0

Do not change tests or production comparison behavior until this preflight is recorded.

Inspect the six full-mix-only simfile-241 `ride` rows from the recorded run:

- 30 ms raw
- 30 ms aligned
- 50 ms raw
- 50 ms aligned
- 100 ms raw
- 100 ms aligned

Record each `(reference_support, prediction_support)` tuple on HPA-305 and PR #32.

Proceed with this design only if every row satisfies:

- `reference_support == 0`; and
- `prediction_support > 0`.

If any missing-counterpart row has positive reference support, stop HPA-305's comparison-layer implementation and investigate reference/scoring evidence instead. If a row is `0/0`, also stop: `_class_scores()` would not legitimately emit a class that appears in neither reference nor prediction evidence, so silently omitting such a row would hide malformed scorer/report evidence.

This preflight is Task 0 of the implementation plan, not a post-implementation production check.

## Goals

- Let paired class comparison accept legitimate prediction-only class asymmetry on either side.
- Preserve strict failure for missing reference-supported class evidence.
- Preserve strict failure for malformed one-sided class evidence.
- Preserve strict per-song grid equality.
- Keep HPA-325 scoring and report schemas unchanged.
- Unblock `compare_oaf_separation()` and HPA-328 finalization on the recorded production run.

## Non-goals

- Do not make HPA-325 emit a dense taxonomy grid.
- Do not change score formulas, taxonomy, event mapping, tolerances, alignment, or report schemas.
- Do not add a generic comparison policy framework or configurable join mode.
- Do not rerun separator or OaF inference merely to repair comparison semantics.
- Do not change MuScriptor, IDM, or cross-comparison product behavior beyond the shared helper semantics they already consume.
- Do not add backward-compatibility handling for historical comparison artifacts.

## Options considered

### A. Reference-aware asymmetric class join — selected

Keep `paired_song_rows()` unchanged. In `paired_class_rows()`, reason over the union of the two class-key sets:

1. Shared keys remain pairable.
2. Shared keys must have equal valid `reference_support`; otherwise fail closed because both reports score the same authoritative reference population.
3. A key present on only one side is valid only when the present row has `reference_support == 0` and `prediction_support > 0`.
4. Valid one-sided prediction-only keys are excluded from paired class delta rows, matching the earlier model-comparison intersection semantics.
5. A one-sided key with positive reference support, zero prediction support, or malformed support fields remains an integrity error.

This fixes the actual comparison contract without changing scorer output.

### B. Emit a dense HPA-325 class grid

Have the scorer generate every taxonomy class for every successful song, tolerance, and mode. This would force equal key grids, but it changes the canonical single-cohort report contract and expands every report to solve a downstream join concern. Reject as unnecessary.

### C. Restore blind class-key intersection

Simply join `left_keys & right_keys`, as the earlier MuScriptor comparator did. This is smaller but could hide a real missing reference-supported class. Reject because the reference-aware guard is cheap and retains fail-closed evidence validation.

## Detailed design

### Shared comparison helper

Modify only `src/benchmark/published_comparison.py::paired_class_rows()` production behavior.

The function already receives validated `PublishedClassRow`-like objects and the pairable successful song IDs. Continue filtering both input maps to `pairable_ids` first.

Build:

- `left_keys` — all class score keys for pairable songs on the left;
- `right_keys` — all class score keys for pairable songs on the right;
- `shared_keys = left_keys & right_keys`;
- `left_only = left_keys - right_keys`;
- `right_only = right_keys - left_keys`.

Use one small local/private support-reading seam rather than raw comparisons through `_row_value()`. For both `reference_support` and `prediction_support`, require a non-boolean, nonnegative integer. Malformed support evidence raises `ComparisonIntegrityError`; do not leak a `TypeError` from a comparison helper that callers inconsistently wrap.

For each shared key:

- require valid support fields on both rows;
- require `left.reference_support == right.reference_support`;
- otherwise raise `ComparisonIntegrityError` with the key and both reference-support values.

Different reference support means the reports are not comparable even if both keys exist.

For each key in both `left_only` and `right_only`, validate the present row symmetrically:

- `reference_support == 0` and `prediction_support > 0` => valid prediction-only asymmetry; do not emit a paired delta row;
- otherwise => raise `ComparisonIntegrityError` identifying the key, present side, missing side, and support values.

Render paired rows only for `shared_keys`, retaining the current deterministic `(simfile_id, tolerance_ms, mode, common_class)` ordering and existing metric/support fields.

Do not synthesize fake `PublishedClassRow` values. The comparison output remains explicitly paired evidence rather than pretending an absent row was published by the scorer.

### Per-song comparison

`paired_song_rows()` remains exact-grid and unchanged. HPA-325 deterministically emits one song row per successful song × tolerance × mode, so a one-sided song key is still corrupt/incomplete evidence.

### Existing callers

Do not add caller-specific join logic.

`compare_oaf_separation()` already reaches the shared helper through its pair-summary path. MuScriptor and IDM comparisons also consume `paired_class_rows()`. The helper therefore must accept valid prediction-only asymmetry on either the left or right side and keep reference-supported asymmetry fail-closed for every caller.

`src/benchmark/separation_comparison.py` receives no production join implementation change.

## Data and artifact behavior

No persisted schema changes.

Existing HPA-325 artifacts remain canonical inputs:

- `summary.json`
- `items.csv`
- `per_song.csv`
- `per_class.csv`
- `event_diagnostics.jsonl`
- `summary.md`

Existing comparison artifact names and schemas remain unchanged. Valid one-sided prediction-only class rows are not represented in `paired_per_class.csv` because that file is a paired-row product. Their underlying native/per-class evidence remains available in the source cohort reports and diagnostics.

## Error handling

Continue using `ComparisonIntegrityError` for comparison evidence failures.

Distinct failure cases should cover:

- malformed/non-integer/negative `reference_support` or `prediction_support`;
- shared class key with different authoritative `reference_support`;
- one-sided class key with positive `reference_support`;
- one-sided class key with `prediction_support <= 0`.

Do not broadly weaken or remove integrity validation.

## Testing

Testing starts only after the production Step 0 preflight confirms the hypothesis.

### `tests/benchmark/test_published_comparison.py`

Keep the shared helper's new semantic coverage together near the existing song-grid test:

1. Left-only prediction-only class (`reference_support == 0`, `prediction_support > 0`) is valid and omitted from paired rows.
2. Right-only prediction-only class is valid and omitted from paired rows.
3. Shared class key with unequal `reference_support` fails.
4. Malformed support evidence fails as `ComparisonIntegrityError` rather than leaking a raw `TypeError`.
5. Existing shared-key rendering remains unchanged.
6. Existing per-song key-grid mismatch remains strict.

At least one omit test should use the simfile-241 shape across all six 30/50/100 ms × raw/aligned keys so the recorded failure cannot regress through a one-key-only fixture.

### `tests/benchmark/test_muscriptor_comparison_coverage.py`

Retarget the existing `test_paired_class_rows_rejects_asymmetric_pairable_key_grid` coverage case rather than deleting it. Its left-only `snare` row has positive reference support and must continue to fail under Option A, but assert the new reference-supported-asymmetry error instead of the old generic key-grid-mismatch message.

This preserves an existing fail-closed regression used by a current shared-helper caller.

### `tests/benchmark/test_separation_comparison.py`

Add one integration-level regression proving the HPA-328 separation comparison publishes successfully when full mix contains a prediction-only class row and a derived view has no corresponding row.

Prefer the existing synthetic separation fixtures and published-report seams. Do not introduce a second report/scorer harness solely for this bug.

No separate IDM integration test is required: left-only and right-only shared-helper tests cover the symmetric contract consumed by IDM, while the existing caller suites remain regression gates.

## Implementation verification

After Step 0 and the focused RED/GREEN tests:

1. Run the focused shared comparison, MuScriptor comparison coverage, separation comparison, and IDM comparison suites that import/use the shared helper.
2. Run the normal repository static/test gates appropriate to the touched Python files.
3. Re-run `compare_oaf_separation()` against `oaf-separation-8e66abde20b8f590` using the already reproduced evidence; do not rerun expensive separator/OaF inference unless those artifacts are unavailable.
4. Require comparison publication to complete rather than exit 2.
5. Run `finalize-oaf-separation-pilot` and verify HPA-328 can reach its intended final outcome.

## Expected implementation surface

Production:

- `src/benchmark/published_comparison.py`

Tests:

- `tests/benchmark/test_published_comparison.py`
- `tests/benchmark/test_muscriptor_comparison_coverage.py`
- `tests/benchmark/test_separation_comparison.py`

No production change is expected in `src/benchmark/separation_comparison.py`, `src/benchmark/cohort_scoring.py`, or report schemas.

Documentation/planning stays on this same HPA-305 branch and PR.

## Delivery

Use one PR for HPA-305. This draft starts with the design spec. After the written design is approved, add the implementation plan to this same branch/PR; after planning review, implementation and production verification also land on this same PR. Do not open a second implementation PR.

Planned branch: `agent/hpa-305-per-class-comparison-asymmetry`.
