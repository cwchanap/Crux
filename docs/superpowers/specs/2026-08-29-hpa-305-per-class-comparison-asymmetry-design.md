# HPA-305 Per-Class Comparison Asymmetry Design

## Context

HPA-305 records a deterministic failure from the reproduced HPA-328 separation pilot run `oaf-separation-8e66abde20b8f590` after the HPA-424 source-binding fix landed. All 30 fixed-subset Spleeter and HTDemucs separations, derived OaF inference, and HPA-325 cohort scoring complete, but the final separation comparison aborts in `src/benchmark/published_comparison.py::paired_class_rows()` because the full-mix and Spleeter `per_class.csv` key grids differ for simfile 241, class `ride`, at all three tolerances and both score modes.

The comparison currently treats any one-sided per-class key as corrupt evidence. That is stricter than the scorer contract and stricter than the earlier MuScriptor comparator, which intentionally joined class rows on the intersection because models can have asymmetric prediction-only class support.

HPA-627 remains operationally blocked on gated Hugging Face access, so HPA-305 is the next executable Crux benchmark task.

## Root-cause invariant

HPA-325 does not omit a class merely because a model predicted zero events for it. `_class_scores()` builds its class set from matched events, unmatched reference events, and unmatched prediction events. Matching itself is class-constrained in `src/benchmark/scoring.py`, so `reference_support = TP + FN` is exactly the number of authoritative reference events for that class and is model/view-independent when both cohorts use the same reference set.

That gives the load-bearing invariant for this design: **if a successful scored song has any reference support for a class, every compared view must publish that class row even when it predicts zero events.** A missing per-class key after successful scoring can therefore be legitimate only when the present row is prediction-only (`reference_support == 0`, `prediction_support > 0`), unless the published evidence itself is malformed.

The production failure is expected to have exactly that shape:

- the authoritative reference for simfile 241 has zero `ride` support;
- full-mix OaF predicts one or more false-positive `ride` events, so HPA-325 emits a `ride` row with `reference_support == 0` and `prediction_support > 0`;
- Spleeter predicts no `ride` events, so no `ride` row exists;
- exact key-grid equality rejects the resulting one-sided row.

A one-sided row with positive reference support is not harmless asymmetry and must remain fatal. Shared rows with unequal reference support are also incomparable and must fail closed.

## Hard production preflight — implementation Task 0

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

If any missing-counterpart row has positive reference support, stop the comparison-layer implementation. If a row is `0/0`, also stop: `_class_scores()` would not legitimately emit a class that appears in neither reference nor prediction evidence.

This preflight is Task 0 of the implementation plan, not a post-implementation check.

## Goals

- Let paired class comparison accept legitimate prediction-only class asymmetry on either side.
- Preserve strict failure for missing reference-supported class evidence.
- Preserve strict failure for malformed one-sided class evidence.
- Preserve strict per-song grid equality.
- Publish explicit counts for prediction-only class rows omitted from paired class deltas.
- Keep HPA-325 scoring and report schemas unchanged.
- Unblock `compare_oaf_separation()` and restore HPA-328 finalization prerequisites on the recorded production run.

## Non-goals

- Do not make HPA-325 emit a dense taxonomy grid.
- Do not change score formulas, taxonomy, event mapping, tolerances, alignment, or HPA-325 report schemas.
- Do not add a generic comparison policy framework or configurable join mode.
- Do not rerun separator or OaF inference merely to repair comparison semantics.
- Do not synthesize missing class rows.
- Do not add backward-compatibility handling for historical comparison artifacts.

## Options considered

### A. Reference-aware asymmetric class join with explicit exclusions — selected

Keep `paired_song_rows()` unchanged. In `paired_class_rows()`, reason over the union of the two class-key sets:

1. Shared keys remain pairable.
2. Shared keys must have equal valid `reference_support`.
3. A key present on only one side is valid only when the present row has `reference_support == 0` and `prediction_support > 0`.
4. Valid one-sided prediction-only keys are excluded from paired class delta rows.
5. The helper returns per-side counts for those omitted prediction-only class rows, and each caller merges those counts into its existing `pairing.exclusions` map.
6. A one-sided key with positive reference support, zero prediction support, or malformed support fields remains an integrity error.

This fixes the comparison contract without changing scorer output and keeps dropped paired-row evidence visible in summaries.

### B. Emit a dense HPA-325 class grid

Have the scorer generate every taxonomy class for every successful song, tolerance, and mode. This would force equal key grids, but it changes the canonical single-cohort report contract and expands every report to solve a downstream join concern. Reject as unnecessary.

### C. Restore blind class-key intersection

Simply join `left_keys & right_keys`, as the earlier MuScriptor comparator did. This is smaller but could hide a real missing reference-supported class and would not explain how many rows disappeared from the paired product. Reject because the reference-aware guard and exclusion counts are cheap.

## Detailed design

### Shared comparison helper

Change `src/benchmark/published_comparison.py::paired_class_rows()` to preserve its inputs and labels but return:

```python
tuple[list[dict[str, str]], dict[str, int]]
```

The first element is the existing paired row list. The second is the per-side prediction-only exclusion counts.

Continue filtering both input maps to `pairable_ids` first. Build:

- `left_keys`
- `right_keys`
- `shared_keys = left_keys & right_keys`
- `left_only = left_keys - right_keys`
- `right_only = right_keys - left_keys`

Add one module-level `_class_key_sort_key()` next to `_song_key_sort_key()` and use it for every class-key sort. Do not retain the current nested sort function or duplicate inline sort lambdas.

Keep one small private `_support_count()` helper for the duck-typed helper boundary. It requires a non-boolean, nonnegative integer and converts malformed values into `ComparisonIntegrityError`. Current production `PublishedClassRow` values are already validated by the report readers; this helper is defensive consistency with the other duck-typed accessors in `published_comparison.py`, not a replacement production parser.

For each shared key:

- validate only `reference_support` on both rows because that is the field used to decide comparability;
- require `left.reference_support == right.reference_support`;
- otherwise raise `ComparisonIntegrityError` with the key and both values.

Do not redundantly validate shared `prediction_support`; production rows already came through the report reader and paired rendering does not branch on that field.

For each key in `left_only` and `right_only`, validate the present row symmetrically:

- require valid `reference_support` and `prediction_support`;
- `reference_support == 0` and `prediction_support > 0` => valid prediction-only asymmetry;
- otherwise raise `ComparisonIntegrityError` identifying the key, present side, missing side, and supports.

Render paired rows only for `shared_keys` in the existing deterministic order and with the existing fields.

Return class exclusion counts with the current labels, for example:

```python
{
    f"{left_label}_only_prediction_class": len(left_only),
    f"{right_label}_only_prediction_class": len(right_only),
}
```

These values count omitted **per-class score rows** (`song × tolerance × mode × common_class`), not unique class names.

### Caller exclusion plumbing

All three current callers already receive an `exclusions` dict from `pairable_success_ids()` immediately before building `comparison_summary()`:

- `src/benchmark/muscriptor_comparison.py`
- `src/benchmark/idm_comparison.py`
- `src/benchmark/separation_comparison.py`

Each caller unpacks `paired_class_rows()` into `class_rows, class_exclusions`, merges `class_exclusions` into the existing pair exclusions, and passes the merged mapping to `comparison_summary()`.

For separation, keep the merge inside `_pair_summary()` so each full-mix-vs-view pair gets its own labels such as `full_mix_only_prediction_class` and `spleeter_only_prediction_class`.

Do not add a second summary field or caller-specific class-join behavior.

### Per-song comparison

`paired_song_rows()` remains exact-grid and unchanged. HPA-325 deterministically emits one song row per successful song × tolerance × mode, so a one-sided song key remains corrupt/incomplete evidence.

### Do not synthesize missing rows

`paired_per_class.csv` is a paired-delta product, so a class row absent from one source report should not be fabricated. Besides misrepresenting source evidence, synthesizing a zero row would increase `aggregate_delta_rows()` `row_count` while contributing no real paired delta, distorting the denominator used to describe the paired class population.

The actual false-positive evidence is not lost: it remains in the source cohort `per_class.csv`, event diagnostics, and separation paired event-micro FP/min metrics. The new exclusion counters make the omission from the paired class product explicit.

## Data and artifact behavior

No HPA-325 artifact or comparison schema version changes.

Existing comparison artifact names remain unchanged. The existing free-form `pairing.exclusions` mapping gains two class-row counters for each pair. `paired_class_row_count` continues to count only rows actually present on both sides.

For the recorded HPA-305 case, Spleeter comparison should report six `full_mix_only_prediction_class` exclusions for the six 30/50/100 ms × raw/aligned `ride` rows, while the source full-mix `per_class.csv` retains those rows.

## Error handling

Continue using `ComparisonIntegrityError` for comparison evidence failures.

Distinct failure cases cover:

- malformed/non-integer/negative support read through the duck-typed helper boundary;
- shared class key with different authoritative `reference_support`;
- one-sided class key with positive `reference_support`;
- one-sided class key with `prediction_support <= 0`.

Do not broadly weaken integrity validation.

## Testing

Testing starts only after production Task 0 confirms the hypothesis.

### `tests/benchmark/test_published_comparison.py`

Add focused shared-helper coverage:

1. Left-only simfile-241 prediction-only rows across all six score keys are omitted and counted on the left.
2. Right-only prediction-only row is omitted and counted on the right.
3. Shared class key with unequal `reference_support` fails.
4. One-sided `0/0` row fails.
5. One malformed support case fails as `ComparisonIntegrityError`.
6. Existing shared-key rendering remains unchanged and returns zero class exclusions.
7. Existing per-song key-grid mismatch remains strict.

### `tests/benchmark/test_muscriptor_comparison_coverage.py`

Retarget the existing positive-reference asymmetry test to the new fail-closed message. Update its direct helper-return assertions for the new `(rows, exclusions)` return shape; do not delete the migration-seam coverage.

### Caller summary coverage

Use existing real comparison tests rather than building new caller frameworks:

- `tests/benchmark/test_muscriptor_comparison.py`: baseline comparison summary contains zero `oaf_only_prediction_class` / `muscriptor_only_prediction_class` counters.
- `tests/benchmark/test_idm_comparison.py`: baseline comparison summary contains zero `oaf_only_prediction_class` / `idm_only_prediction_class` counters.
- `tests/benchmark/test_separation_comparison.py`: strengthen the existing publish integration fixture so full mix has a valid FP-only `ride` class absent from derived views; assert the paired CSV omits it and each affected pair summary records the corresponding nonzero exclusion count.

The current MuScriptor, IDM, separation, and cross-comparison suites remain regression gates for the shared return-signature change.

## Risks and fallback

### Risk: Task 0 disproves the design

If any recorded full-mix `ride` row has positive reference support or is `0/0`, **HPA-305 remains the owning ticket and PR #32 remains the single PR**, but this comparison-layer design is abandoned before tests/code changes.

The first fallback artifact to inspect is:

`views/spleeter/reports/event_diagnostics.jsonl`

Filter simfile 241 / `ride` across the same six tolerance/mode keys.

- If Spleeter diagnostics contain `ride` false negatives while `per_class.csv` lacks the class row, re-scope HPA-305/PR #32 to the HPA-325 scorer/report emission path, starting with `_class_scores()` and `write_cohort_reports()`.
- If Spleeter diagnostics also lack `ride` despite positive full-mix reference support, compare the HPA-323/HPA-324 reference mapping evidence used by both cohorts before changing scoring or comparison logic.

Do not open a second HPA-305 implementation PR. Rewrite the same spec/plan only after the artifact evidence identifies the actual layer.

### Risk: shared helper return signature reaches multiple callers

The return change is intentionally made now while there are only three callers. Update all three atomically and gate on their existing comparison suites. No compatibility wrapper is needed for this internal seam.

## Implementation verification

After Task 0 and focused RED/GREEN tests:

1. Run shared helper, MuScriptor, IDM, separation, and cross-comparison suites.
2. Run repository static/test gates for all touched Python files.
3. Re-run only `compare_oaf_separation()` against `oaf-separation-8e66abde20b8f590`; do not rerun separator/OaF inference.
4. Require comparison publication to complete and the Spleeter pair summary to report the production prediction-only exclusions.
5. Confirm the HPA-328 comparison bundle/finalizer prerequisite is restored without inventing a new benchmark decision inside HPA-305.

## Expected implementation surface

Production:

- `src/benchmark/published_comparison.py`
- `src/benchmark/muscriptor_comparison.py`
- `src/benchmark/idm_comparison.py`
- `src/benchmark/separation_comparison.py`

Tests:

- `tests/benchmark/test_published_comparison.py`
- `tests/benchmark/test_muscriptor_comparison_coverage.py`
- `tests/benchmark/test_muscriptor_comparison.py`
- `tests/benchmark/test_idm_comparison.py`
- `tests/benchmark/test_separation_comparison.py`

No production change is expected in `src/benchmark/cohort_scoring.py`, `src/benchmark/reports.py`, taxonomy/mapping, or separator/OaF execution.

## Delivery

Use one PR for HPA-305. This draft contains the design spec and implementation plan. After planning review, Task 0, implementation, and production verification also land on this same PR. Do not open a second implementation PR.

Planned branch: `agent/hpa-305-per-class-comparison-asymmetry`.
