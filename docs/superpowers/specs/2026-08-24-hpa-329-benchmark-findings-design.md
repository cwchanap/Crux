# HPA-329 Comparative Benchmark Findings Design

**Status:** Proposed
**Date:** 2026-08-24
**Linear:** HPA-329 — Publish comparative Crux benchmark findings and decide the next model-assisted charting step

## Decision summary

HPA-329 is a one-time benchmark closeout, not a new benchmark subsystem.

The implementation should publish one versioned Markdown findings report from the evidence already produced by HPA-325, HPA-327, HPA-328, HPA-395, HPA-396, and HPA-562. It should not add a findings-report framework, notebook platform, dashboard, generic analytics engine, new scorer, new comparison engine, or new inference path.

The same draft PR that carries this design and implementation plan will later carry the final findings report after the production-evidence gate is satisfied. There should not be a second HPA-329 PR.

The final report will live at:

```text
docs/benchmark/2026-08-24-drum-transcription-findings.md
```

The report is the HPA-329 product. Temporary extraction/checking snippets remain uncommitted and must call the repository's existing readers/CLIs rather than reimplement parsing, pairing, or scoring.

## Why this is the next actionable task

HPA-627 has higher nominal priority but is operationally blocked by gated Hugging Face access required to produce `runtime/muscriptor/model.json`. Its ticket explicitly records that the authenticated weight download cannot be completed from the current environment.

HPA-329 is the next logical benchmark child after HPA-562. All of HPA-329's code-producing predecessors are merged, including the HPA-562 paired-comparison publication layer. HPA-329 should consume those artifacts rather than recreate their joins.

However, HPA-329 is **not yet executable to completion**. The merged HPA-562 PR states that no real paired publication bundle was generated because the required production OaF, MuScriptor, separation, IDM, manifest, and publication inputs were unavailable. The design therefore separates planning from an explicit production-evidence gate and forbids fabricated/synthetic headline numbers.

## Existing authorities

HPA-329 must treat these existing artifacts and readers as authoritative:

```text
src/benchmark/reports.py
  read_cohort_reports()
  PublishedCohortReports
  PublishedAggregate
  PublishedSongRow
  PublishedClassRow

src/benchmark/muscriptor_comparison.py
  _load_evidence()              # one-off HPA-329 validation reuse; do not copy it
  compare_oaf_muscriptor()

src/benchmark/reviewed_subset.py
  score_oaf_reviewed_subset()
  score_muscriptor_reviewed_subset()

src/benchmark/separation_comparison.py
  compare_oaf_separation()

src/benchmark/idm_comparison.py
  compare_oaf_idm()

src/benchmark/cross_comparison.py
  publish_cross_comparisons()
  CrossComparisonOutcome

src/benchmark/muscriptor_model.py
  load_muscriptor_model_lock()

src/benchmark/muscriptor_corpus_run.py
  parse_muscriptor_corpus_run()
  compute_model_lock_sha256()

runtime/oaf_tf1/model.json
runtime/muscriptor/model.json
  frozen model-lock evidence when HPA-627 is complete
```

The HPA-325 report is the authority for absolute cohort metrics such as event-micro F1, song-macro F1, class-macro F1, per-class metrics, song-score distributions, and complete cohort populations. HPA-562 is authoritative only for the paired comparisons and scopes it actually publishes: broad OaF/MuScriptor full mix, the HPA-328 separator pilot, and the HPA-396 IDM pilot.

HPA-329 must not recompute headline metrics from per-song means when HPA-325 already publishes the canonical aggregate, and it must not build a second paired join.

## Frozen evidence-source map

Use this map throughout the gate and final report. It closes the scope ambiguity between reviewed-subset absolute reports and the broad-only OaF/MuScriptor comparison published by HPA-562.

| HPA-329 scope | Absolute metrics | Paired deltas |
| --- | --- | --- |
| `broad_full_mix` | HPA-325 OaF + MuScriptor broad reports | HPA-562 `comparisons/oaf-muscriptor/` |
| `reviewed_subset_full_mix` | HPA-325 reports produced by `score-oaf-reviewed-subset` and `score-muscriptor-reviewed-subset` | Not required for HPA-329. If the written decision later genuinely needs a reviewed pair, reuse `compare-oaf-muscriptor --subset-manifest`; do not add it to HPA-562 or build another join. |
| `reviewed_separator_pilot` | HPA-328 full-mix/Spleeter/HTDemucs view reports | HPA-562 `comparisons/oaf-separation/` |
| `idm_htdemucs_pilot` | HPA-396 OaF/IDM pilot reports | HPA-562 `comparisons/oaf-idm/` |

HPA-562 deliberately invokes `compare_oaf_muscriptor(..., subset_manifest_path=None)`, so its OaF/MuScriptor pair is broad-only. HPA-329 must never treat that comparison as reviewed-subset evidence.

The HPA-562 IDM scope also carries `reviewed_subset_cross_verified = false`. HPA-329 must preserve that qualification: the HPA-396 run validates its own pilot lineage, but HPA-562 does not prove that the IDM pilot and HPA-328 separator publication share one cross-verified HPA-327 membership.

## Approaches considered

### A. One versioned Markdown closeout over existing artifacts — selected

Use the existing HPA-325 cohort reports and HPA-562 pairwise bundle as source evidence, verify their identities/hashes with the existing readers, and write one traceable Markdown findings report.

This is the smallest solution. HPA-329 is inherently a one-time decision document, and the repository already owns deterministic scoring and pairwise publication.

### B. Add `src/benchmark/findings_report.py` plus a new CLI — rejected

A concrete generator could make the Markdown reproducible, but it would duplicate substantial report parsing, cross-scope presentation, and narrative policy for a report that is expected to be produced once. It would also create a maintenance surface without enabling a follow-on runtime feature.

If the benchmark later becomes a recurring release process, a separate ticket can extract the repeated mechanical portions after repetition actually exists.

### C. Build a notebook/dashboard/general analysis framework — rejected

This adds the most machinery and the least value for a hobby-project benchmark closeout. It also risks creating a second source of truth for metrics that HPA-325/HPA-562 already publish deterministically.

## Production-evidence gate

No final findings report may be populated with benchmark conclusions until all of the following are true.

1. **The checked-in MuScriptor lock matches the scored run.** Load `runtime/muscriptor/model.json` with `load_muscriptor_model_lock()`, parse the production MuScriptor run with `parse_muscriptor_corpus_run()`, and require at minimum:
   - `lock.model_id == run["model_id"]`;
   - `compute_model_lock_sha256(lock_path) == run["model_lock_sha256"]`;
   - `lock.checkpoint_revision == run["checkpoint_revision"]`;
   - `lock.checkpoint_sha256 == run["checkpoint_sha256"]`.
   A unit test over a synthetic `tmp_path` lock is not evidence that the checked-in production lock matches the run. If HPA-627 is still blocked and the file is absent, stop here.
2. **Real broad runs exist.** OaF and MuScriptor production run/report artifacts exist for the authoritative broad corpus, including complete success/failure populations.
3. **Real reviewed/pilot evidence exists.** The accepted HPA-327 subset, the two reviewed-subset HPA-325 report sets, HPA-328 separation publication/handoff, and HPA-396 IDM pilot outputs exist and are production artifacts rather than test fixtures.
4. **A real HPA-562 bundle can be republished from production inputs.** Invoke the existing HPA-562 publication path from the immutable production run inputs and treat `CrossComparisonOutcome.pairable_success_counts` (or the CLI JSON rendered directly from that outcome) as the four authoritative pairable counts. Do not hand-validate only the top-level JSON while bypassing `publish_cross_comparisons()`.
5. **Shared identity is coherent.** Manifest, timing, taxonomy, lane-map, scoring, OaF model-lock, prediction-map, and input-view contracts pass the existing comparison/publication validation.
6. **Source reports load through the existing readers.** Broad OaF/MuScriptor evidence should use the same `_load_evidence()` path used by the existing comparison. Reviewed reports should be generated/validated through the existing reviewed-subset CLIs/functions and then loaded with `read_cohort_reports()` using the exact derived subset cohort identity. Pilot report identities must come from their owning persisted run/comparison paths. Do not use `_compat_report_identity()` or a dummy `backend_descriptor_sha256` for production evidence.

If any gate item is false, HPA-329 remains planning-only. Record the missing evidence in Linear and do not publish placeholder numbers, synthetic results, or conclusions inferred from unit-test fixtures.

## Report structure

The final Markdown report has the following fixed sections.

### 1. Executive summary

A short answer to three questions:

- What is the best measured pretrained path for assisted DTX drafting today?
- What is the dominant remaining failure mode?
- What single next step is justified by the evidence?

This section is written last, after the evidence sections and decision table are complete.

### 2. Evidence identity and scope

Record the exact Crux commit and the immutable identities needed to trace every headline number:

- corpus/reference manifest path, version, and SHA-256;
- timing manifest path, version, and SHA-256;
- taxonomy, lane-map, and scoring versions;
- model IDs and model-lock hashes;
- prediction-map versions;
- input-view IDs;
- production run paths;
- HPA-325 report paths;
- HPA-562 bundle path and nested comparison artifact hashes;
- reviewed-subset identity;
- separator/IDM retained-input identities where applicable.

The report must explicitly distinguish:

```text
broad_full_mix
reviewed_subset_full_mix
reviewed_separator_pilot
idm_htdemucs_pilot
```

There is no unlabeled global leaderboard.

### 3. OaF baseline capability

Report broad-corpus and reviewed-subset results separately.

For 30/50/100 ms and raw/aligned modes, show the canonical HPA-325:

- event-micro precision/recall/F1;
- song-macro F1;
- class-macro F1;
- successful-song count;
- song F1 distribution;
- per-class precision/recall/F1 and support for the classes that materially explain the conclusion.

Also include population counts, quarantine/failure counts, coverage, and the observed raw-versus-aligned gap. The aligned view is diagnostic only; the report must not imply that alignment repaired the reference clock or that aligned scores are interchangeable with raw headline scores.

### 4. OaF separation conclusion

Use the fixed HPA-328 reviewed pilot only.

Compare OaF full mix with Spleeter and HTDemucs using the existing paired-success rows and aggregate deltas. Include:

- exact pairable-success intersection and exclusions;
- song/class improvements and regressions;
- runtime and retained-storage evidence;
- operational complexity relative to direct OaF;
- whether either separator earns its extra cost.

Do not extrapolate pilot-only gains into a full-corpus headline without saying that the scope is a pilot.

### 5. MuScriptor comparison

Report MuScriptor independently first, then the **broad full-mix** paired comparison with OaF.

Include:

- broad-corpus and reviewed-subset HPA-325 aggregates;
- complete success/failure population for both scopes;
- exact broad OaF/MuScriptor paired intersection/exclusions from HPA-562;
- broad paired song/class deltas;
- native/output-class coverage and unmapped drum pitches;
- timing behavior;
- runtime and memory evidence;
- lack of velocity in the released transcription output;
- checkpoint weight license and its suitability for the intended Crux use.

Reviewed-subset OaF and MuScriptor evidence remains two independent HPA-325 cohorts by default. A reviewed paired comparison is optional and must use the existing `compare-oaf-muscriptor --subset-manifest` path if the final decision actually needs it; its absence is not a production-evidence gate failure.

Accuracy and deployability are separate conclusions.

### 6. Inverse Drum Machine comparison

Keep IDM scoped to the fixed HPA-396 HTDemucs-stem pilot.

Include:

- exact paired HTDemucs input scope;
- OaF/IDM paired intersection/exclusions;
- song/class deltas;
- native class coverage and velocity availability;
- failure rate;
- runtime/memory where available;
- installation/runtime maturity;
- direct-full-mix smoke evidence only in a clearly separate diagnostic subsection;
- the HPA-562 qualification that `reviewed_subset_cross_verified` is false for the IDM scope.

The report may recommend a separately scoped full-corpus IDM experiment, but must not present the pilot as an existing full-corpus result or as cross-verified to the HPA-328 reviewed membership at HPA-562 level.

### 7. Cross-model failure taxonomy

Use representative persisted evidence to distinguish at least these categories:

- recognition/model limits;
- accompaniment interference;
- DTX/audio timing or alignment semantics;
- class/lane mapping and unsupported distinctions;
- authored chart simplification/source ambiguity;
- decoding/peak/chunk/adapter behavior;
- input-domain mismatch;
- missing confidence/velocity/native distinctions;
- operational/runtime failures;
- license/deployment constraints.

Event-level examples must come from already-published `event_diagnostics.jsonl` in the reviewed-subset and separator report directories. The reviewed-subset scoring path and HPA-328 separation views already request diagnostics for every successful selected song. Parse those canonical JSONL rows directly; `read_cohort_reports()` intentionally does not include event diagnostics in `PublishedCohortReports`.

Do **not** call `score_cohort()` from an HPA-329 scratch script to manufacture diagnostics. If a required OaF/MuScriptor reviewed report is missing or lacks its diagnostics artifact, regenerate that report with the existing `score-oaf-reviewed-subset` / `score-muscriptor-reviewed-subset` path. For IDM, stay on the existing paired per-song/per-class outputs plus native coverage/runtime evidence; do not add post-hoc event diagnostics.

A disagreement with authored DTX is not automatically a model error. Ambiguous examples must be labeled ambiguous.

### 8. Decision

Evaluate the eight HPA-329 options exactly as written in the Linear ticket:

1. use OaF directly;
2. use OaF after a selected separator;
3. prototype MuScriptor-assisted drafting;
4. continue with IDM research;
5. improve deterministic post-processing/mapping;
6. prepare for fine-tuning later;
7. improve corpus/reference quality first;
8. evaluate another pretrained model.

Select exactly one primary next step. For each rejected option, give one concise evidence-based reason.

Any fine-tuning recommendation must name the stable measured error pattern that training is expected to address.

### 9. Follow-up work

Create Linear follow-up issues only after the decision is written and only for work directly justified by the measurements. Do not pre-create a fine-tuning, editor-integration, full-corpus IDM, separator rollout, or new-model ticket merely because it appears in the decision menu.

### 10. Reproducibility appendix

List the commands/readers used to extract/check the report, source artifact paths/hashes, and the exact Crux commit. The appendix is sufficient for a later reader to trace each headline table row back to persisted evidence without rerunning model inference.

## Numeric and presentation rules

- Copy canonical HPA-325 aggregate decimals from validated typed report objects; do not recompute event-micro F1 from song rows.
- Use HPA-562 paired outputs only for the scopes HPA-562 publishes. Its OaF/MuScriptor comparison is broad-only.
- Do not build another paired join.
- Keep 30/50/100 ms rows in fixed ascending tolerance order and `raw` before `aligned`.
- Keep broad, reviewed-subset, separator-pilot, and IDM-pilot scopes visibly separated.
- Report missing metrics as `N/A`; never infer zero.
- Preserve complete failure populations before discussing paired intersections.
- Discuss class coverage independently from common-canonical F1.
- Discuss license suitability independently from accuracy.
- Prefer the simplest pipeline that meets the measured quality need.

## Error handling

HPA-329 fails closed as a reporting task.

- Missing required evidence: stop before conclusions.
- Mixed lineage/identity: stop and fix or regenerate the upstream publication; do not hand-merge results.
- Missing or mismatched model lock: stop and resolve the owning lock ticket.
- Missing reviewed diagnostics: regenerate through the existing reviewed-subset command, not a scratch scorer call.
- Missing class/runtime/license evidence: mark the conclusion incomplete rather than guessing.
- Ambiguous reference example: label it ambiguous and exclude it from causal certainty, not from the benchmark population.

No compatibility layer is needed for older report schemas. This report targets the current benchmark artifacts only.

## Validation

Because the product is a Markdown findings report, verification is evidence-oriented rather than a new unit-test suite.

Before HPA-329 can move to review:

1. every headline table identifies its source typed report/comparison;
2. every listed source path exists and its recorded hash/identity matches the production evidence;
3. the checked-in MuScriptor lock's model ID and exact lock hash match the production MuScriptor run;
4. the HPA-562 bundle is freshly validated/republished through `publish_cross_comparisons()`, and its `CrossComparisonOutcome.pairable_success_counts` match the report exactly;
5. broad and pilot scopes are never combined into an unlabeled ranking;
6. reviewed OaF/MuScriptor absolute metrics come from the two reviewed-subset HPA-325 reports, not from HPA-562;
7. the report preserves the IDM `reviewed_subset_cross_verified = false` qualification;
8. event-level failure examples come from existing canonical diagnostics, not an HPA-329 rescore;
9. the report includes complete per-model population/failure counts before paired metrics;
10. the selected recommendation is one of the eight ticket options and rejected alternatives are explained;
11. rerun the same typed source-reader pass used to populate the report and compare every headline cell/count against the source field (`event_micro.*`, `song_macro_f1`, `class_macro_f1`, `pairing.pairable_success_intersection`, population fields, and runtime/storage fields as applicable); grep-only verification is not sufficient;
12. any created follow-up Linear issue is directly supported by a measured finding;
13. `git diff --check` passes and the HPA-329 PR contains no unrelated production-code change.

## Non-goals

- New model inference.
- New scoring logic.
- New pairwise comparison logic.
- A required reviewed-subset OaF/MuScriptor pair when the decision does not need one.
- A recurring reporting framework or dashboard.
- Fine-tuning or retraining.
- Full-corpus IDM expansion.
- Production backend replacement.
- DTXWeb/editor integration.
- Automatic winner selection.
- Statistical significance/bootstrap machinery.
- Compatibility support for older benchmark-report schemas.
