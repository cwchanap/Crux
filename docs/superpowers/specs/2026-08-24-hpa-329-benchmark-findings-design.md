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

The report is the HPA-329 product. Any temporary extraction/checking scripts used while writing it remain uncommitted unless an already-existing reader exposes a tiny reusable seam that materially reduces duplication. The default is still no production-code change.

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

src/benchmark/published_comparison.py
  pairable_success_ids()
  paired_song_rows()
  paired_class_rows()
  aggregate_delta_rows()
  population()

src/benchmark/muscriptor_comparison.py
  HPA-395 OaF/MuScriptor comparison publication

src/benchmark/separation_comparison.py
  HPA-328 OaF full-mix/Spleeter/HTDemucs comparison publication

src/benchmark/idm_comparison.py
  HPA-396 OaF/IDM identical-HTDemucs-input comparison publication

src/benchmark/cross_comparison.py
  HPA-562 top-level publication and scope matrix

runtime/oaf_tf1/model.json
runtime/muscriptor/model.json
  frozen model-lock evidence when HPA-627 is complete
```

The HPA-325 report is the authority for absolute cohort metrics such as event-micro F1, song-macro F1, class-macro F1, per-class metrics, and song-score distributions. HPA-562 is the authority for pairwise scope, pairing populations, exclusions, paired deltas, and the top-level model × input-view population matrix.

HPA-329 must not recompute headline metrics from per-song means when HPA-325 already publishes the canonical aggregate.

## Approaches considered

### A. One versioned Markdown closeout over existing artifacts — selected

Use the existing HPA-325 cohort reports and HPA-562 pairwise bundle as source evidence, verify their identities/hashes, and write one traceable Markdown findings report.

This is the smallest solution. HPA-329 is inherently a one-time decision document, and the repository already owns deterministic scoring and pairwise publication.

### B. Add `src/benchmark/findings_report.py` plus a new CLI — rejected

A concrete generator could make the Markdown reproducible, but it would duplicate substantial report parsing, cross-scope presentation, and narrative policy for a report that is expected to be produced once. It would also create a maintenance surface without enabling a follow-on runtime feature.

If the benchmark later becomes a recurring release process, a separate ticket can extract the repeated mechanical portions from the HPA-329 work after repetition actually exists.

### C. Build a notebook/dashboard/general analysis framework — rejected

This adds the most machinery and the least value for a hobby-project benchmark closeout. It also risks creating a second source of truth for metrics that HPA-325/HPA-562 already publish deterministically.

## Production-evidence gate

No final findings report may be populated with benchmark conclusions until all of the following are true.

1. **MuScriptor lock exists.** `runtime/muscriptor/model.json` is present, validates through the existing MuScriptor lock test, and represents the exact checkpoint used for the scored run.
2. **Real broad runs exist.** OaF and MuScriptor production run/report artifacts exist for the authoritative broad corpus, including complete success/failure populations.
3. **Real reviewed/pilot evidence exists.** The accepted HPA-327 subset, HPA-328 separation publication/handoff, and HPA-396 IDM pilot outputs exist and are production artifacts rather than test fixtures.
4. **A real HPA-562 bundle exists.** `summary.json` uses `crux.paired-benchmark-publication/v1`, its nested artifact index validates, and its four pairable-success counts come from the production comparisons.
5. **Shared identity is coherent.** Manifest, timing, taxonomy, lane-map, scoring, OaF model-lock, prediction-map, and input-view contracts pass the existing HPA-562 validation.
6. **Source reports are available.** The HPA-325 report directories needed for absolute metrics are available for the broad OaF and MuScriptor cohorts and the reviewed/pilot views discussed in the report.

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

Report MuScriptor independently first, then the paired comparison with OaF.

Include:

- broad-corpus and reviewed-subset HPA-325 aggregates;
- complete success/failure population;
- exact OaF/MuScriptor paired intersection/exclusions;
- paired song/class deltas;
- native/output-class coverage and unmapped drum pitches;
- timing behavior;
- runtime and memory evidence;
- lack of velocity in the released transcription output;
- checkpoint weight license and its suitability for the intended Crux use.

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
- direct-full-mix smoke evidence only in a clearly separate diagnostic subsection.

The report may recommend a separately scoped full-corpus IDM experiment, but must not present the pilot as an existing full-corpus result.

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

Examples should come from persisted reviewed-subset diagnostics and paired per-song/per-class outputs. Where a model/view lacks pre-rendered event diagnostics, it is acceptable to rescore **persisted predictions only** for a small set of reviewed song IDs using the existing HPA-325 scorer with diagnostics enabled. Never rerun inference and never change model/mapping/scoring identities for diagnosis.

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

- Copy canonical HPA-325 aggregate decimals from validated report objects/files; do not recompute event-micro F1 from song rows.
- Use the existing HPA-562 paired outputs for paired deltas/intersections; do not build another join.
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
- Missing model lock: stop and resolve the owning lock ticket.
- Missing class/runtime/license evidence: mark the conclusion incomplete rather than guessing.
- Ambiguous reference example: label it ambiguous and exclude it from causal certainty, not from the benchmark population.

No compatibility layer is needed for older report schemas. This report targets the current benchmark artifacts only.

## Validation

Because the product is a Markdown findings report, verification is evidence-oriented rather than a new unit-test suite.

Before HPA-329 can move to review:

1. every headline table identifies its source report/comparison;
2. every listed source path exists and its recorded hash/identity matches the production evidence;
3. HPA-562 pairable-success counts match the report text/tables exactly;
4. broad and pilot scopes are never combined into an unlabeled ranking;
5. the report includes complete per-model population/failure counts before paired metrics;
6. the selected recommendation is one of the eight ticket options and rejected alternatives are explained;
7. any created follow-up Linear issue is directly supported by a measured finding;
8. `git diff --check` passes and the HPA-329 PR contains no unrelated production-code change.

## Non-goals

- New model inference.
- New scoring logic.
- New pairwise comparison logic.
- A recurring reporting framework or dashboard.
- Fine-tuning or retraining.
- Full-corpus IDM expansion.
- Production backend replacement.
- DTXWeb/editor integration.
- Automatic winner selection.
- Statistical significance/bootstrap machinery.
- Compatibility support for older benchmark-report schemas.
