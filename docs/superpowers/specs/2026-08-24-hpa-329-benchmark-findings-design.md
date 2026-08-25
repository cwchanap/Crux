# HPA-329 Comparative Benchmark Findings Design

**Status:** Proposed  
**Date:** 2026-08-24  
**Linear:** HPA-329 — Publish comparative Crux benchmark findings and decide the next model-assisted charting step

## Decision

HPA-329 is a one-time benchmark closeout. Its product is one versioned Markdown report:

```text
docs/benchmark/2026-08-24-drum-transcription-findings.md
```

Do not add a report generator, CLI, notebook/dashboard, scorer, comparison engine, inference path, significance framework, or compatibility layer. The same draft PR carries planning now and the report later; the repository remains planning-only until the **final evidence gate** passes.

The report consumes existing benchmark authorities rather than rebuilding them:

- HPA-325 `PublishedCohortReports` for absolute aggregates, distributions, population, coverage, per-song, and per-class evidence;
- HPA-562 for the paired scopes it actually publishes;
- HPA-327/HPA-328/HPA-396 persisted reviewed/pilot evidence for scope, diagnostics, runtime, storage, native coverage, and failure evidence;
- checked-in model locks for model/license identity.

## Frozen evidence-source map

| HPA-329 scope | Absolute metrics | Paired deltas |
| --- | --- | --- |
| `broad_full_mix` | HPA-325 OaF + MuScriptor broad reports | HPA-562 `comparisons/oaf-muscriptor/` |
| `reviewed_subset_full_mix` | HPA-325 reports regenerated with `score-oaf-reviewed-subset` and `score-muscriptor-reviewed-subset` | Not required. If the final decision genuinely needs reviewed pairing, reuse `compare-oaf-muscriptor --subset-manifest`. |
| `reviewed_separator_pilot` | HPA-328 full-mix/Spleeter/HTDemucs view reports | HPA-562 `comparisons/oaf-separation/` |
| `idm_htdemucs_pilot` | HPA-396 OaF/IDM pilot reports | HPA-562 `comparisons/oaf-idm/` |

HPA-562's OaF/MuScriptor comparison is broad-only because `publish_cross_comparisons()` calls it with `subset_manifest_path=None`. Never reuse that pair as reviewed-subset evidence.

HPA-562 also records `reviewed_subset_cross_verified == false` for the IDM comparison. HPA-329 must preserve that qualification and must not imply that the IDM and separator publications share one cross-verified HPA-327 membership.

## Existing reader contract

### Broad OaF/MuScriptor

Use `muscriptor_comparison._load_evidence()` as the disposable run/report integrity check because it already validates backend family, run identity, report status population, and source/input evidence. Its returned `_RunEvidence.reports` is **not** the absolute-metric object; it contains only reduced item/song/class comparison data.

For absolute metrics, bind the typed report separately using the same identity path `_load_evidence()` uses:

```python
broad_identity = _report_identity_from_snapshot(evidence.snapshot)
broad_reports = read_cohort_reports(
    run_path.parent / "reports",
    expected_identity=broad_identity,
)
```

`PublishedCohortReports` is the authority for `PublishedAggregate`, population, distribution, and aggregate per-class values.

Do not use `_compat_report_identity()` for production evidence; its dummy backend-descriptor hash exists only as a narrow legacy test seam.

### Reviewed OaF/MuScriptor

Regenerate the reviewed reports from the production parent run plus the accepted subset manifest. This is the one canonical policy; do not alternate between consuming an old reviewed directory and regenerating it.

Use the existing commands:

```text
crux benchmark score-oaf-reviewed-subset
crux benchmark score-muscriptor-reviewed-subset
```

They rerun no model inference. Capture their returned reviewed `cohort_id`s, then load the reports with `read_cohort_reports()` and the parent typed identity with only `cohort_id` replaced. This matches `reviewed_subset.py`, which derives the subset identity as `replace(parent_identity, cohort_id=subset_cohort_id)`.

### Paired publication

Freshly run `publish_cross_comparisons()` (or its existing CLI) from the production run inputs into a new output directory. `CrossComparisonOutcome.pairable_success_counts` is the authoritative four-key pair-count source. Do not treat a hand-opened stale HPA-562 `summary.json` as equivalent validation.

## Evidence gates

### Per-scope prechecks

These checks may be performed independently while upstream evidence is still arriving. They are local validation only; they do **not** authorize committing a partial findings report.

1. **OaF broad/reviewed:** production OaF run loads, typed broad report loads, reviewed report regenerates and loads.
2. **OaF separator pilot:** HPA-328 run/view reports/handoff load and retain the expected pilot lineage.
3. **IDM pilot:** HPA-396 run/comparison/model evidence loads and preserves the identical HTDemucs-input contract.
4. **MuScriptor broad/reviewed:** checked-in MuScriptor lock exists and exactly matches the production run; typed broad report loads; reviewed report regenerates and loads.

This split allows upstream evidence problems to be found early without changing the approved planning-only PR behavior.

### Final evidence gate

No findings Markdown file is committed until all of these are true:

1. all four per-scope prechecks pass;
2. `runtime/muscriptor/model.json` loads with `load_muscriptor_model_lock()` and its `model_id`, exact lock SHA-256, checkpoint revision, and checkpoint SHA-256 equal the production MuScriptor run;
3. HPA-562 can be freshly republished from the production OaF, MuScriptor, separation, and IDM runs and returns all four pairable-success counts;
4. reference/timing/taxonomy/lane/scoring/model/prediction-map/input-view identity remains coherent under the existing HPA-562 validation;
5. reviewed and HPA-328 diagnostic files contain real diagnostic rows, not merely an existing zero-byte `event_diagnostics.jsonl`.

If the final gate fails, record the missing evidence on HPA-329 and leave the PR planning-only. Never use fixtures or synthetic results as substitute benchmark evidence.

## Diagnostic evidence

`write_cohort_reports()` always creates `event_diagnostics.jsonl`, so existence alone proves nothing.

Require non-empty diagnostic content for the reviewed OaF/MuScriptor reports and for the HPA-328 view reports used for event-level examples. When Task 6 selects a concrete example, require at least one diagnostic row for that cited song/view before making an event-level causal claim.

Reviewed reports already request diagnostics for every successful selected song; if a reviewed diagnostic file is empty or incomplete, rerun the existing reviewed-subset command. Do not call `score_cohort()` from an HPA-329 scratch path.

For IDM, use the existing paired per-song/per-class, native coverage, failure, and runtime evidence. Do not add after-the-fact IDM event diagnostics.

## Report contract

The final report contains these sections:

1. Executive summary
2. Evidence identity and scope
3. OaF baseline capability
4. OaF separation conclusion
5. MuScriptor comparison
6. Inverse Drum Machine comparison
7. Cross-model failure taxonomy
8. Decision
9. Follow-up work
10. Reproducibility appendix

The executive summary is written last.

### Absolute metrics

At 30/50/100 ms, raw then aligned, copy the canonical HPA-325 values for:

- event-micro precision/recall/F1;
- song-macro F1;
- class-macro F1;
- successful-song count;
- song F1 distribution;
- material per-class precision/recall/F1 and support;
- complete success/failure/skipped/quarantined population and coverage.

Do not average `per_song.csv` into event-micro F1. Aligned scoring is diagnostic only.

### Paired metrics

Use only the existing comparison publications for pairing/intersection/exclusion/delta evidence. Broad and pilot populations remain visibly separate; there is no global leaderboard.

### Failure taxonomy

Use a small reviewed sample from persisted diagnostic rows and paired/native evidence. Distinguish recognition limits, accompaniment interference, timing semantics, mapping/unsupported distinctions, authored-chart/source ambiguity, decoding/adapter behavior, input-domain mismatch, missing native attributes, operational failures, and license/deployment constraints. Mark ambiguous cases as ambiguous.

### Decision

Evaluate all eight HPA-329 options and select exactly one primary next step. A more expensive separator/model/research/fine-tuning path must justify its extra cost with measured benefit. Create follow-up Linear issues only after this decision and only when the report evidence directly justifies them.

## Reproducibility and verification

Do not add `scripts/hpa329_headline_rows.py`; HPA-329 remains a one-report closeout with no new Python file.

The final report's reproducibility appendix must commit the exact reader/CLI invocation used to validate its numbers, including the concrete production paths/identities. Task 8 reruns that committed recipe fresh and compares generated headline rows/counts against the report in the same verification step; it must not depend on a `/tmp` file surviving between commits or sessions.

This makes verification repeatable without creating a maintained report/extractor surface.

Before review:

- every headline row traces to a typed HPA-325 field or validated paired summary/outcome;
- all populations and pairable counts reconcile;
- diagnostic examples have actual diagnostic rows where event-level claims are made;
- IDM retains the non-cross-verified subset caveat;
- exactly one decision option is primary;
- `git diff --check` passes;
- the PR contains only the two planning docs plus the final findings Markdown report.

## Non-goals

- New scoring or inference.
- A report/extractor framework or committed Python helper.
- Full-corpus IDM expansion.
- Fine-tuning/retraining.
- Production backend replacement.
- DTXWeb/editor integration.
- Automatic winner selection.
- Statistical significance/bootstrap machinery.
- Compatibility support for older report schemas.
