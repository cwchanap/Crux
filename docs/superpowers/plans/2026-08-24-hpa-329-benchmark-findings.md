# HPA-329 Comparative Benchmark Findings Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Publish one traceable Markdown findings report that closes the pretrained drum-transcription benchmark phase and selects one evidence-supported next step for Crux.

**Architecture:** Do not add a findings-report subsystem. Read absolute cohort metrics from the existing HPA-325 report artifacts and paired populations/deltas from the existing HPA-562 publication, verify that all evidence belongs to the same frozen benchmark lineage, then write one versioned Markdown report. Keep the current planning branch and draft PR for the eventual findings report; if production evidence is missing, stop at the evidence gate and leave the PR planning-only.

**Tech Stack:** Python 3.12, existing Crux benchmark readers/validators, Markdown, shell/git. No new runtime dependency, notebook stack, database, dashboard, scorer, comparison engine, or inference path.

**Spec:** `docs/superpowers/specs/2026-08-24-hpa-329-benchmark-findings-design.md`

## Global Constraints

- Keep all HPA-329 planning, evidence review, and the final report on `agent/hpa-329-benchmark-findings-plan` and its single draft PR. Do not open a second HPA-329 PR.
- The final deliverable is `docs/benchmark/2026-08-24-drum-transcription-findings.md`.
- `runtime/muscriptor/model.json` must exist and identify the exact MuScriptor checkpoint used for scored evidence before any final MuScriptor conclusion is published.
- Do not use fixtures, synthetic runs, test goldens, or unit-test populations as benchmark evidence.
- HPA-325 `summary.json`/validated `PublishedCohortReports` are authoritative for absolute event-micro, song-macro, class-macro, distribution, per-class, and population metrics.
- HPA-562 `crux.paired-benchmark-publication/v1` is authoritative for paired scope, pairable-success counts, exclusions, paired per-song/per-class rows, and the model × input-view population matrix.
- Do not recompute event-micro F1 by averaging song rows or reconstruct a second paired join.
- Keep `broad_full_mix`, `reviewed_subset_full_mix`, `reviewed_separator_pilot`, and `idm_htdemucs_pilot` visibly separate. There is no global leaderboard across those scopes.
- Present complete success/failure/quarantine populations before paired-intersection metrics.
- Treat aligned scoring as diagnostics; do not present it as a replacement for raw audio-relative ground truth.
- Discuss common-canonical score, native class coverage, runtime/reliability, and license suitability as separate dimensions.
- If event-level examples are needed and not already published, rescore only persisted predictions for a small reviewed-song set with the existing HPA-325 diagnostics seam. Never rerun inference.
- Select exactly one of HPA-329's eight decision options as the primary next step. Create follow-up Linear issues only after that decision and only when measured evidence justifies them.
- No backward compatibility for old benchmark-report schemas.
- No generic findings generator, reporting framework, notebook platform, dashboard, automatic winner policy, significance/bootstrap layer, fine-tuning, full-corpus IDM run, or editor integration in HPA-329.

---

## File structure

```text
Already created in this planning PR
  docs/superpowers/specs/2026-08-24-hpa-329-benchmark-findings-design.md
  docs/superpowers/plans/2026-08-24-hpa-329-benchmark-findings.md

Create only after Task 0 passes
  docs/benchmark/2026-08-24-drum-transcription-findings.md
```

No production Python file or test file is planned. Temporary local extraction snippets are disposable analysis tools and remain uncommitted.

---

### Task 0: Prove the production evidence is complete before writing conclusions

**Files:**
- Read: `runtime/oaf_tf1/model.json`
- Read: `runtime/muscriptor/model.json`
- Read: production HPA-325 cohort report directories
- Read: production HPA-327 reviewed-subset manifest
- Read: production HPA-328 separation run/comparison/handoff artifacts
- Read: production HPA-396 IDM run/comparison artifacts
- Read: production HPA-562 publication bundle
- Modify only if the gate fails: Linear HPA-329 discussion, not repository files

**Interfaces:**
- Consumes: the immutable run/report/publication outputs produced by HPA-325/HPA-327/HPA-328/HPA-395/HPA-396/HPA-562.
- Produces: a recorded set of exact evidence paths and identities that every later task uses.

- [ ] **Step 1: Verify the MuScriptor lock exists and validates**

Run from the repository root:

```bash
test -f runtime/muscriptor/model.json
uv run pytest tests/benchmark/test_muscriptor_model.py -q
```

Expected: the file exists and the focused lock suite passes. If the file is absent because HPA-627 is still blocked on gated Hugging Face access, stop HPA-329 here. Do not substitute metadata-only revision information for the canonical lock.

- [ ] **Step 2: Resolve the exact production evidence paths from the completed upstream runs**

Collect and record these concrete paths before continuing:

```text
OaF broad run + HPA-325 report directory
MuScriptor broad run + HPA-325 report directory
OaF reviewed-subset HPA-325 report directory
MuScriptor reviewed-subset HPA-325 report directory
HPA-327 accepted subset manifest
HPA-328 separation run/comparison/handoff
HPA-396 IDM run/comparison
HPA-562 paired publication directory
reference-set manifest
reference-timing manifest
```

The values come from the actual completed upstream executions/handoffs. Do not invent repository-default paths: the runners accept caller-owned output directories, so the real paths are part of the evidence being frozen in the final report.

- [ ] **Step 3: Verify the HPA-562 publication is real and complete**

Inspect the production HPA-562 `summary.json` and require:

```text
schema == crux.paired-benchmark-publication/v1
pairable_success_counts has exactly:
  oaf_muscriptor_full_mix
  oaf_separation_pilot.spleeter
  oaf_separation_pilot.htdemucs
  oaf_idm_htdemucs
headline_matrix.csv exists
all indexed nested artifacts exist and match their recorded SHA-256
```

Prefer regenerating/validating the bundle through the existing `crux benchmark publish-paired-comparisons` path from the immutable production run inputs rather than manually repairing a stale bundle. Any HPA-562 identity failure is an upstream evidence problem, not something to paper over in HPA-329.

- [ ] **Step 4: Prove the source cohort reports are production reports, not fixtures**

For each HPA-325 report directory, validate the six report artifacts with the existing `read_cohort_reports()` contract and the run-derived `CohortIdentity`. Confirm the report schema is `crux.single-cohort-report/v1` and that the report identity matches the associated run/model/input view.

Use the existing application readers in a one-off Python invocation; do not write a second parser. The check must fail if `summary.json`, `items.csv`, `per_song.csv`, or `per_class.csv` has mixed identity or malformed canonical data.

- [ ] **Step 5: Prove all required scopes are represented**

Before proceeding, the evidence set must contain:

```text
broad_full_mix: OaF + MuScriptor
reviewed_subset_full_mix: OaF + MuScriptor
reviewed_separator_pilot: OaF full mix + Spleeter + HTDemucs
idm_htdemucs_pilot: OaF + IDM on identical retained HTDemucs stems
```

If a scope is missing, stop instead of silently shrinking the requested HPA-329 analysis.

- [ ] **Step 6: Record the gate result**

If all checks pass, record the exact evidence paths plus their immutable identities for use in Task 1.

If any check fails, add a concise Linear HPA-329 comment naming the missing artifact/identity and leave the draft PR planning-only. There is intentionally no repository commit for a failed Task 0.

---

### Task 1: Create the report skeleton and freeze its evidence identity

**Files:**
- Create: `docs/benchmark/2026-08-24-drum-transcription-findings.md`

**Interfaces:**
- Consumes: the concrete evidence path set from Task 0.
- Produces: one report with a fixed scope vocabulary and a traceability table reused by all later sections.

- [ ] **Step 1: Create the fixed report outline**

Create the file with exactly these top-level sections:

```markdown
# Crux Drum Transcription Benchmark Findings

## Executive summary
## Evidence identity and scope
## OaF baseline capability
## OaF separation conclusion
## MuScriptor comparison
## Inverse Drum Machine comparison
## Cross-model failure taxonomy
## Decision
## Follow-up work
## Reproducibility appendix
```

Leave `Executive summary` semantically empty until Task 7; do not write a provisional winner.

- [ ] **Step 2: Add the scope table**

Use these four scope labels and no alternatives:

```markdown
| scope | models/views | population meaning |
| --- | --- | --- |
| broad_full_mix | OaF full mix, MuScriptor full mix | complete technically eligible broad corpus with model failures preserved |
| reviewed_subset_full_mix | OaF full mix, MuScriptor full mix | accepted HPA-327 reviewed membership |
| reviewed_separator_pilot | OaF full mix, Spleeter drums, HTDemucs drums | fixed HPA-328 pilot only |
| idm_htdemucs_pilot | OaF HTDemucs stem, IDM HTDemucs stem | fixed HPA-396 identical-input pilot only |
```

This table is the report-level guard against an unlabeled leaderboard.

- [ ] **Step 3: Add the evidence identity table**

For every production artifact, record the actual value for the applicable fields:

```text
artifact role
path
schema/version
SHA-256 when the upstream contract publishes one
run/cohort id
model id
model-lock SHA-256
prediction-map version
input-view id
reference manifest version/SHA-256
reference timing version/SHA-256
taxonomy version
lane-map version
scoring version
```

Do not copy a value from another model/view merely because it is expected to match; use the existing validation to prove equality and list the authoritative value.

- [ ] **Step 4: Record the exact Crux revision used to validate the report**

Run:

```bash
git rev-parse HEAD
```

Record that SHA in the reproducibility appendix as the analysis implementation revision. If later commits change any benchmark reader used by this report, update the recorded SHA during Task 8.

- [ ] **Step 5: Commit the traceable report shell**

```bash
git add docs/benchmark/2026-08-24-drum-transcription-findings.md
git commit -m "docs: start HPA-329 benchmark findings"
```

---

### Task 2: Populate OaF broad-corpus and reviewed-subset capability from HPA-325

**Files:**
- Modify: `docs/benchmark/2026-08-24-drum-transcription-findings.md`

**Interfaces:**
- Consumes: validated OaF broad and reviewed-subset `PublishedCohortReports`.
- Produces: absolute OaF capability tables and the raw-versus-aligned diagnostic interpretation.

- [ ] **Step 1: Extract the six canonical aggregate rows per OaF scope**

For each OaF report, use `read_cohort_reports()` and its `PublishedAggregate` values. Keep fixed order:

```text
30 raw
30 aligned
50 raw
50 aligned
100 raw
100 aligned
```

For each row copy the canonical persisted values for:

```text
event-micro precision
 event-micro recall
 event-micro F1
 song-macro F1
 class-macro F1
 successful-song count
 song F1 min/p10/p25/median/p75/p90/max
```

Do not derive event-micro values from `per_song.csv`.

- [ ] **Step 2: Add complete population/coverage context before score interpretation**

For broad and reviewed scopes, report:

```text
total
success
failed
skipped
quarantined
failure reason counts
prediction mapping coverage/native-output coverage evidence
```

Call out selection bias explicitly: the reviewed subset is a diagnostic quality-control view, not a replacement for broad-corpus performance.

- [ ] **Step 3: Add the per-class evidence that explains the conclusion**

Start from the HPA-325 aggregate `per_class` rows. Include all classes in an appendix/table when practical, and in prose call out only material classes that explain the observed strengths/weaknesses. Preserve reference/prediction support so a high or low F1 cannot be interpreted without population size.

- [ ] **Step 4: Explain the raw/aligned gap without promoting aligned scores to the headline**

Use the persisted raw/aligned aggregates and timing diagnostics. State whether alignment materially changes the interpretation and what that implies about residual timing/source semantics. Keep raw scores as the ground-truth-facing benchmark view; aligned scores are diagnostic evidence.

- [ ] **Step 5: Commit the OaF evidence section**

```bash
git add docs/benchmark/2026-08-24-drum-transcription-findings.md
git commit -m "docs: summarize OaF benchmark capability"
```

---

### Task 3: Populate the fixed OaF separation-pilot conclusion from HPA-562/HPA-328

**Files:**
- Modify: `docs/benchmark/2026-08-24-drum-transcription-findings.md`

**Interfaces:**
- Consumes: HPA-562 `comparisons/oaf-separation/` plus HPA-328 runtime/storage/handoff evidence.
- Produces: one pilot-scoped full-mix vs Spleeter vs HTDemucs conclusion.

- [ ] **Step 1: Copy the exact paired population evidence**

Use HPA-562/HPA-328 to report for Spleeter and HTDemucs:

```text
pairable_success_intersection
left/right-only successes
source/input exclusions defined by the upstream comparison
paired song-row count
paired class-row count
```

These counts must match top-level HPA-562 `pairable_success_counts` exactly.

- [ ] **Step 2: Summarize paired score movement without rebuilding the join**

Read the existing:

```text
comparisons/oaf-separation/spleeter/paired_per_song.csv
comparisons/oaf-separation/spleeter/paired_per_class.csv
comparisons/oaf-separation/htdemucs/paired_per_song.csv
comparisons/oaf-separation/htdemucs/paired_per_class.csv
```

Use the published aggregate deltas and representative rows to identify which separator/classes/songs improve or regress. Do not recompute a new paired population.

- [ ] **Step 3: Add measured cost/complexity evidence**

Use HPA-328 persisted runtime and retained-byte evidence to compare direct OaF with separator + OaF. Discuss wall time, storage, extra runtime/environment/model dependencies, and operational steps. Do not invent dollar costs when the upstream artifacts did not measure them.

- [ ] **Step 4: Write one explicit separator conclusion**

State one of:

```text
keep OaF full mix
use Spleeter before OaF
use HTDemucs before OaF
pilot is inconclusive
```

Tie the statement to both quality delta and added cost. Keep the conclusion labeled `reviewed_separator_pilot`; do not imply full-corpus validation.

- [ ] **Step 5: Commit**

```bash
git add docs/benchmark/2026-08-24-drum-transcription-findings.md
git commit -m "docs: conclude OaF separation pilot"
```

---

### Task 4: Populate MuScriptor independent and paired findings

**Files:**
- Modify: `docs/benchmark/2026-08-24-drum-transcription-findings.md`

**Interfaces:**
- Consumes: validated MuScriptor broad/reviewed HPA-325 reports, HPA-562 OaF/MuScriptor comparison, MuScriptor run/model-lock evidence.
- Produces: separate accuracy, coverage, reliability, runtime, and license conclusions for MuScriptor.

- [ ] **Step 1: Report MuScriptor independently on broad and reviewed scopes**

Use the same six-row 30/50/100 raw/aligned aggregate shape as Task 2. Include complete success/failure population before any OaF pairing.

- [ ] **Step 2: Add the exact OaF/MuScriptor paired comparison**

Use HPA-562 `comparisons/oaf-muscriptor/` for:

```text
pairable success count
exclusions
paired song deltas
paired class deltas
```

Keep paired deltas separate from each model's independent population so a model-specific failure cannot disappear from the headline.

- [ ] **Step 3: Add native-output and timing evidence**

Report native pitch/class coverage, mapped/unmapped counts, timing behavior, and representative errors from the persisted run/reports. Explicitly state that the released MuScriptor transcription path does not preserve velocity.

- [ ] **Step 4: Add runtime/memory and license suitability**

Use measured run evidence for runtime, real-time factor, process/device memory where available, and operational reliability. Record the checkpoint weight license from the canonical model lock and explain what `cc-by-nc-4.0` means for the intended Crux usage without treating license suitability as an accuracy metric.

- [ ] **Step 5: Commit**

```bash
git add docs/benchmark/2026-08-24-drum-transcription-findings.md
git commit -m "docs: summarize MuScriptor comparison"
```

---

### Task 5: Populate the fixed IDM HTDemucs-stem pilot findings

**Files:**
- Modify: `docs/benchmark/2026-08-24-drum-transcription-findings.md`

**Interfaces:**
- Consumes: HPA-562 `comparisons/oaf-idm/` plus HPA-396 run/model/runtime evidence.
- Produces: an IDM pilot conclusion that cannot be confused with broad/full-mix evidence.

- [ ] **Step 1: State the identical-input pilot contract before metrics**

Record the exact pilot membership/input-view identity and the upstream proof that OaF and IDM consumed identical retained HTDemucs stem hashes. Label the entire section `idm_htdemucs_pilot`.

- [ ] **Step 2: Report population and paired deltas**

Use HPA-562/HPA-396 for exact success/failure population, pairable-success intersection/exclusions, and existing paired song/class deltas. Do not compare IDM's pilot absolute number directly with a broad-corpus OaF/MuScriptor headline as though populations were equal.

- [ ] **Step 3: Add native capability and operational evidence**

Report checkpoint-native class coverage, velocity availability, mapping coverage, failure reasons, install/runtime maturity, measured runtime, and memory where available.

- [ ] **Step 4: Keep any direct-full-mix smoke result in a separate diagnostic subsection**

If HPA-396 contains a direct-full-mix compatibility smoke, label it as smoke/compatibility evidence only. It is not a full-corpus or scored-peer result.

- [ ] **Step 5: State whether a larger IDM experiment is justified**

Write `yes`, `no`, or `inconclusive` with one evidence-based reason. Do not start or create a full-corpus IDM task until Task 7 chooses the primary next step or explicitly justifies that follow-up.

- [ ] **Step 6: Commit**

```bash
git add docs/benchmark/2026-08-24-drum-transcription-findings.md
git commit -m "docs: summarize IDM stem pilot"
```

---

### Task 6: Build a small evidence-backed failure taxonomy from reviewed examples

**Files:**
- Modify: `docs/benchmark/2026-08-24-drum-transcription-findings.md`
- Optional local-only analysis output: uncommitted temporary files under the developer's scratch directory

**Interfaces:**
- Consumes: HPA-327 reviewed evidence, HPA-325 event diagnostics, HPA-562 paired per-song/per-class rows, persisted prediction/reference artifacts.
- Produces: representative examples classified by cause with ambiguity called out explicitly.

- [ ] **Step 1: Select a bounded diagnostic sample from already-published evidence**

Choose a small set that covers useful contrasts rather than a new statistical study:

```text
one strong matched case
one large false-positive/overprediction case
one large false-negative/underprediction case
one large separator improvement or regression
one material cross-model disagreement
one ambiguous timing/source/reference case if present
```

Prefer reviewed-subset songs and use score/delta extremes only to select examples for diagnosis, never to change benchmark membership.

- [ ] **Step 2: Reuse existing event diagnostics where present**

Inspect persisted matched/FP/FN diagnostic rows plus native mapping/coverage evidence. Do not infer event-level causes from aggregate F1 alone.

- [ ] **Step 3: If diagnostics are missing, rescore persisted predictions only for the selected IDs**

Use the existing HPA-325 `score_cohort(..., diagnostics_for=...)` / report path over persisted predictions and the same frozen reference/taxonomy/scoring identities. This is diagnostic rescoring only:

```text
no model inference
no separator rerun
no map change
no tolerance change outside the frozen 30/50/100 set
no benchmark membership change
```

Do not commit a new diagnostic framework.

- [ ] **Step 4: Classify examples into the ticket's failure taxonomy**

Use these labels:

```text
recognition/model limit
accompaniment interference
DTX/audio timing semantics
class/lane mapping or unsupported distinction
authored chart simplification/source ambiguity
decoding/peak/chunk/adapter behavior
input-domain mismatch
missing confidence/velocity/native distinction
operational/runtime failure
license/deployment constraint
```

For each example, cite the evidence that supports the classification. If two causes remain plausible, mark the example ambiguous rather than forcing one label.

- [ ] **Step 5: Summarize the dominant measured limitations**

Rank only limitations that have evidence in the reviewed examples and population-level metrics. Do not claim a causal percentage split that the benchmark did not measure.

- [ ] **Step 6: Commit**

```bash
git add docs/benchmark/2026-08-24-drum-transcription-findings.md
git commit -m "docs: classify benchmark failure modes"
```

---

### Task 7: Make the eight-option decision, then write the executive summary and justified follow-ups

**Files:**
- Modify: `docs/benchmark/2026-08-24-drum-transcription-findings.md`
- Modify after the written decision only: Linear HPA-329 / new Linear follow-up issues when justified

**Interfaces:**
- Consumes: Tasks 2-6 conclusions.
- Produces: exactly one primary HPA-329 next step, rejected-alternative rationale, concise executive summary, and zero or more evidence-justified follow-up tickets.

- [ ] **Step 1: Add the fixed decision table**

Use all eight HPA-329 options without renaming them:

```markdown
| option | verdict | evidence |
| --- | --- | --- |
| Use OaF directly | ... | ... |
| Use OaF after a selected separator | ... | ... |
| Prototype MuScriptor-assisted drafting | ... | ... |
| Continue with IDM research | ... | ... |
| Improve deterministic post-processing/mapping | ... | ... |
| Prepare for fine-tuning later | ... | ... |
| Improve corpus/reference quality first | ... | ... |
| Evaluate another pretrained model | ... | ... |
```

Exactly one row is marked `Primary next step`; every other row gets a concise rejected/deferred reason based on the report evidence.

- [ ] **Step 2: Check the primary choice against the hobby-project cost rule**

Prefer the simplest path that satisfies the measured quality need. A separator, larger model, new research path, or fine-tuning must earn its extra compute, storage, integration, runtime, or licensing cost with a material measured benefit.

- [ ] **Step 3: If fine-tuning is selected or deferred as likely, name the precise measured error pattern**

Never write a generic `fine-tune for better accuracy` rationale. Name the stable domain-specific recognition/mapping pattern that remains after timing, source quality, input separation, and model choice have been controlled.

- [ ] **Step 4: Write the executive summary last**

Answer in a few paragraphs:

```text
best measured pretrained path today
dominant remaining limitation
one primary next step
why the closest alternative lost
```

Keep broad/pilot scope qualifications in the summary itself.

- [ ] **Step 5: Create only follow-up Linear issues directly required by the selected decision**

Before creating any follow-up, point to the exact report section/measurement that justifies it. Do not pre-create fine-tuning, separator rollout, IDM expansion, editor integration, or another-model work unless the selected decision requires it.

Link any created issue from HPA-329 and add its identifier to `Follow-up work` in the report.

- [ ] **Step 6: Commit**

```bash
git add docs/benchmark/2026-08-24-drum-transcription-findings.md
git commit -m "docs: decide next Crux transcription step"
```

---

### Task 8: Verify traceability, close the report PR, and hand the result back to Linear

**Files:**
- Modify if verification finds a report defect: `docs/benchmark/2026-08-24-drum-transcription-findings.md`
- No production-code changes

**Interfaces:**
- Consumes: complete HPA-329 report and all source evidence.
- Produces: review-ready single-PR closeout with no unsupported claims.

- [ ] **Step 1: Trace every headline row back to an immutable source**

For each absolute score table, verify the source HPA-325 report path/cohort identity. For each paired table, verify the HPA-562 nested comparison path and pairable count. For each runtime/storage/license statement, verify the owning run/model-lock/handoff evidence.

No headline number may exist only in prose or a scratch calculation.

- [ ] **Step 2: Reconcile all population and pairing counts**

Require:

```text
complete per-model populations balance
HPA-562 four pairable-success counts equal the report
paired row counts/exclusions match nested summaries
reviewed/pilot memberships match their owning manifests/runs
```

Fix the report or upstream evidence; never hand-adjust a number to make totals fit.

- [ ] **Step 3: Scan for scope and conclusion mistakes**

Reject the report if it:

```text
ranks pilot-only IDM beside broad-corpus models without qualification
uses aligned metrics as the raw headline
omits failed/quarantined population before paired metrics
conflates class coverage with canonical F1
conflates checkpoint license with code license
calls ambiguous DTX disagreement a definite model error
selects more than one primary next step
```

- [ ] **Step 4: Scan for incomplete planning language**

Run:

```bash
grep -nE 'TBD|TODO|placeholder|fill in|to be determined' \
  docs/benchmark/2026-08-24-drum-transcription-findings.md && exit 1 || true
```

Expected: no matches that indicate an unfinished report.

- [ ] **Step 5: Run repository diff hygiene and confirm the task stayed docs-only**

```bash
git diff --check main...HEAD
git diff --name-only main...HEAD
```

Expected HPA-329-owned files:

```text
docs/superpowers/specs/2026-08-24-hpa-329-benchmark-findings-design.md
docs/superpowers/plans/2026-08-24-hpa-329-benchmark-findings.md
docs/benchmark/2026-08-24-drum-transcription-findings.md
```

If an upstream fix became necessary, land/track it in its owning issue rather than silently widening HPA-329.

- [ ] **Step 6: Refresh the recorded analysis revision after the final report commit**

Run:

```bash
git rev-parse HEAD
```

Update the reproducibility appendix if the report currently names an earlier analysis SHA, commit that one-line correction, then re-run `git diff --check`.

- [ ] **Step 7: Move the draft PR to review and update Linear**

When the report passes every gate:

```text
mark the existing HPA-329 draft PR ready for review
move HPA-329 to In Review
add the PR/report link and one-paragraph executive conclusion to HPA-329
```

Do not mark HPA-329 Done until the report PR is accepted/merged and any child-completion/removal-with-reason condition required to close HPA-319 has been checked.

- [ ] **Step 8: After merge, close HPA-329 and evaluate HPA-319 closeout**

Confirm all HPA-319 children are completed or explicitly removed with reason, then mark HPA-329 Done and close HPA-319 only when its parent acceptance criteria are satisfied. HPA-627 cannot be silently ignored if the final report depends on MuScriptor scored evidence.
