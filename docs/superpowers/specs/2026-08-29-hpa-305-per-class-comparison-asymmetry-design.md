# HPA-305 Per-Class Comparison Asymmetry Design

## Context

HPA-305 records a deterministic failure from the reproduced HPA-328 separation pilot run `oaf-separation-8e66abde20b8f590` after the HPA-424 source-binding fix landed. All 30 fixed-subset Spleeter and HTDemucs separations, derived OaF inference, and HPA-325 cohort scoring complete, but the final separation comparison aborts in `src/benchmark/published_comparison.py::paired_class_rows()` because the full-mix and Spleeter `per_class.csv` key grids differ for simfile 241, class `ride`, at all three tolerances and both score modes.

The comparison currently treats any one-sided per-class key as corrupt evidence. That is stricter than the scorer contract and stricter than the earlier MuScriptor comparator, which intentionally joined class rows on the intersection because models can have asymmetric prediction-only class support.

HPA-627 remains operationally blocked on gated Hugging Face access, so HPA-305 is the next executable Crux benchmark task.

## Root-cause hypothesis and required preflight

HPA-325 does not omit a class merely because a model predicted zero events for it. `_class_scores()` builds the class set from matched events, unmatched reference events, and unmatched prediction events. Therefore a reference-supported class with zero predictions still produces a per-class row with nonzero `reference_support` and zero `prediction_support`.

The production failure is instead expected to be a valid prediction-only asymmetry:

- the authoritative reference for simfile 241 has zero `ride` support;
- full-mix OaF predicts one or more false-positive `ride` events, so HPA-325 emits a `ride` row with `reference_support == 0`;
- Spleeter predicts no `ride` events, so no `ride` row exists;
- exact key-grid equality rejects the resulting one-sided row.

This hypothesis must be checked against the recorded production reports before changing code. Inspect all six missing `ride` keys in the full-mix report and require `reference_support == 0`. If any missing counterpart has positive reference support, stop this ticket's comparison-layer fix and investigate reference/scoring evidence instead; a reference-supported class must never be silently discarded.

## Goals

- Let paired class comparison accept legitimate prediction-only class asymmetry.
- Preserve strict failure for missing reference-supported class evidence.
- Preserve strict per-song grid equality.
- Keep HPA-325 scoring and report schemas unchanged.
- Unblock `compare_oaf_separation()` and HPA-328 finalization on the recorded production run.

## Non-goals

- Do not make HPA-325 emit a dense taxonomy grid.
- Do not change score formulas, taxonomy, event mapping, tolerances, alignment, or report schemas.
- Do not add a generic comparison policy framework or configurable join mode.
- Do not rerun separator or OaF inference merely to repair published comparison semantics.
- Do not change MuScriptor, IDM, or cross-comparison product behavior beyond the shared helper semantics they already consume.
- Do not add backward-compatibility handling for historical comparison artifacts.

## Options considered

### A. Reference-aware asymmetric class join — selected

Keep `paired_song_rows()` unchanged. In `paired_class_rows()`, reason over the union of the two class-key sets:

1. Shared keys remain pairable.
2. Shared keys must have equal `reference_support`; otherwise fail closed because both reports score the same authoritative reference population.
3. A key present on only one side is allowed only when the present row has `reference_support == 0`.
4. Allowed one-sided prediction-only keys are excluded from paired class delta rows, matching the earlier model-comparison intersection semantics.
5. A one-sided key with positive reference support remains an integrity error.

This fixes the actual comparison contract without changing scorer output.

### B. Emit a dense HPA-325 class grid

Have the scorer generate every taxonomy class for every successful song, tolerance, and mode. This would force equal key grids, but it changes the canonical single-cohort report contract and expands every report to solve a downstream join concern. Reject as unnecessary.

### C. Restore blind class-key intersection

Simply join `left_keys & right_keys`, as the earlier MuScriptor comparator did. This is smaller but could hide a real missing reference-supported class. Reject because the reference-aware guard is cheap and retains fail-closed evidence validation.

## Detailed design

### Shared comparison helper

Modify only `src/benchmark/published_comparison.py::paired_class_rows()` behavior.

The function already receives validated `PublishedClassRow`-like objects and the pairable successful song IDs. Continue filtering both input maps to `pairable_ids` first.

Build:

- `left_keys` — all class score keys for pairable songs on the left;
- `right_keys` — all class score keys for pairable songs on the right;
- `shared_keys = left_keys & right_keys`;
- `left_only = left_keys - right_keys`;
- `right_only = right_keys - left_keys`.

For each shared key, require `left.reference_support == right.reference_support`. Different reference support means the reports are not comparable and must raise `ComparisonIntegrityError` with the key and both supports.

For each one-sided key, inspect the present row's `reference_support`:

- `0` => valid prediction-only asymmetry; do not emit a paired delta row;
- `> 0` => raise `ComparisonIntegrityError`, reporting which side is missing the reference-supported row.

Render paired rows only for `shared_keys`, retaining the current deterministic `(simfile_id, tolerance_ms, mode, common_class)` ordering and existing metric/support fields.

Do not synthesize fake `PublishedClassRow` values. The comparison output remains explicitly paired evidence rather than pretending an absent row was published by the scorer.

### Per-song comparison

`paired_song_rows()` remains exact-grid and unchanged. HPA-325 deterministically emits one song row per successful song × tolerance × mode, so a one-sided song key is still corrupt/incomplete evidence.

### Separation comparison

`src/benchmark/separation_comparison.py` should not get a second join implementation. It continues to call the shared `paired_class_rows()` helper. Its integration coverage should prove a separation comparison can publish when one view has an FP-only class absent from the other.

The shared semantics are also appropriate for model comparisons: prediction-only classes can differ by backend, while authoritative reference support must remain consistent.

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

New failure cases/messages should distinguish:

- shared class key with different authoritative `reference_support`;
- class key missing from one side even though the present side has positive `reference_support`.

Do not broadly weaken or remove integrity validation.

## Testing

### `tests/benchmark/test_published_comparison.py`

Add focused shared-helper coverage:

1. A simfile-241-shaped regression with `ride` present only on one side across 30/50/100 ms × raw/aligned, each present row having `reference_support == 0`; `paired_class_rows()` succeeds and omits those six keys from paired rows.
2. One-sided class key with positive `reference_support`; comparison fails.
3. Shared class key with unequal `reference_support`; comparison fails.
4. Existing shared-key rendering remains unchanged.
5. Existing per-song key-grid mismatch test remains strict.

### `tests/benchmark/test_separation_comparison.py`

Add one integration-level regression proving the HPA-328 separation comparison publishes successfully when full mix contains a prediction-only class row and a derived view has no corresponding row.

Prefer the existing synthetic separation fixtures and published-report seams. Do not introduce a second report/scorer harness solely for this bug.

## Production verification

Use the already reproduced HPA-328 evidence rather than rerunning expensive separation/inference unless those artifacts are unavailable.

1. Inspect the six full-mix-only simfile-241 `ride` rows from `oaf-separation-8e66abde20b8f590` and record their `reference_support`/`prediction_support` on HPA-305 and the PR.
2. Require all missing-counterpart rows to have `reference_support == 0` before applying the comparison fix.
3. Run `compare_oaf_separation()` against the recorded run on the implementation tree.
4. Require comparison publication to complete rather than exit 2.
5. Run `finalize-oaf-separation-pilot` and verify HPA-328 can reach its intended final outcome.

If step 2 fails, do not implement or merge a permissive comparison change under this design; move the investigation upstream to reference/scoring evidence.

## Expected implementation surface

Production:

- `src/benchmark/published_comparison.py`

Tests:

- `tests/benchmark/test_published_comparison.py`
- `tests/benchmark/test_separation_comparison.py`

Documentation/planning stays on this same HPA-305 branch and PR.

## Delivery

Use one PR for HPA-305. The draft begins with this design spec and the implementation plan. After planning review, the implementation and production verification land on the same branch/PR; do not open a second implementation PR.

Planned branch: `agent/hpa-305-per-class-comparison-asymmetry`.
