# HPA-329 Comparative Benchmark Findings Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Publish one traceable Markdown findings report that closes the pretrained drum-transcription benchmark phase and selects one evidence-supported next step for Crux.

**Architecture:** Do not add a findings-report subsystem. Load absolute cohort metrics through the existing HPA-325 readers, regenerate reviewed-subset reports through their existing commands when needed, and republish paired evidence through the existing HPA-562 coordinator. Keep broad full-mix, reviewed full-mix, separator-pilot, and IDM-pilot evidence separate. If production evidence is missing or any identity check fails, stop at Task 0 and leave the PR planning-only.

**Tech Stack:** Python 3.12, existing Crux benchmark readers/validators and CLI commands, Markdown, shell/git. No new runtime dependency, notebook stack, database, dashboard, scorer, comparison engine, or inference path.

**Spec:** `docs/superpowers/specs/2026-08-24-hpa-329-benchmark-findings-design.md`

## Global Constraints

- Keep all HPA-329 planning, evidence review, and the final report on `agent/hpa-329-benchmark-findings-plan` and its single draft PR. Do not open a second HPA-329 PR.
- The final deliverable is `docs/benchmark/2026-08-24-drum-transcription-findings.md`.
- `runtime/muscriptor/model.json` must exist and its exact file hash/model/checkpoint identity must match the production MuScriptor run before any final MuScriptor conclusion is published.
- Do not use fixtures, synthetic runs, test goldens, or unit-test populations as benchmark evidence.
- HPA-325 `PublishedCohortReports` are authoritative for absolute event-micro, song-macro, class-macro, distribution, per-class, coverage, and population metrics.
- HPA-562 is authoritative for the paired scopes it actually publishes: broad OaF/MuScriptor full mix, OaF separator pilot, and OaF/IDM HTDemucs pilot. Its OaF/MuScriptor comparison is **not** a reviewed-subset pair.
- `reviewed_subset_full_mix` uses the two HPA-325 reports produced by `score-oaf-reviewed-subset` and `score-muscriptor-reviewed-subset`. A reviewed OaF/MuScriptor pair is not required for HPA-329; if the written decision later genuinely needs one, reuse `compare-oaf-muscriptor --subset-manifest`.
- Do not recompute event-micro F1 by averaging song rows or reconstruct a second paired join.
- Keep `broad_full_mix`, `reviewed_subset_full_mix`, `reviewed_separator_pilot`, and `idm_htdemucs_pilot` visibly separate. There is no global leaderboard across those scopes.
- Preserve HPA-562's IDM qualification: `reviewed_subset_cross_verified` is false for `oaf_idm_htdemucs`.
- Present complete success/failure/quarantine populations before paired-intersection metrics.
- Treat aligned scoring as diagnostics; do not present it as a replacement for raw audio-relative ground truth.
- Discuss common-canonical score, native class coverage, runtime/reliability, and license suitability as separate dimensions.
- Use already-published `event_diagnostics.jsonl` for reviewed/subset/separation event examples. Do not call `score_cohort()` from an HPA-329 scratch script. If reviewed OaF/MuScriptor diagnostics are missing, regenerate the reviewed report with the existing subset command.
- Select exactly one of HPA-329's eight decision options as the primary next step. Create follow-up Linear issues only after that decision and only when measured evidence justifies them.
- No backward compatibility for old benchmark-report schemas.
- No generic findings generator, reporting framework, notebook platform, dashboard, automatic winner policy, significance/bootstrap layer, fine-tuning, full-corpus IDM run, or editor integration in HPA-329.

## Frozen evidence-source map

| HPA-329 scope | Absolute metrics | Paired deltas |
| --- | --- | --- |
| `broad_full_mix` | HPA-325 OaF + MuScriptor broad reports | HPA-562 `comparisons/oaf-muscriptor/` |
| `reviewed_subset_full_mix` | HPA-325 reports from the two reviewed-subset commands | Not required. Optional only through existing `compare-oaf-muscriptor --subset-manifest` if the final decision needs it. |
| `reviewed_separator_pilot` | HPA-328 full-mix/Spleeter/HTDemucs view reports | HPA-562 `comparisons/oaf-separation/` |
| `idm_htdemucs_pilot` | HPA-396 OaF/IDM pilot reports | HPA-562 `comparisons/oaf-idm/` |

---

## File structure

```text
Already created in this planning PR
  docs/superpowers/specs/2026-08-24-hpa-329-benchmark-findings-design.md
  docs/superpowers/plans/2026-08-24-hpa-329-benchmark-findings.md

Create only after Task 0 passes
  docs/benchmark/2026-08-24-drum-transcription-findings.md
```

No production Python file or test file is planned. Temporary local extraction/checking snippets are disposable analysis tools and remain uncommitted.

---

### Task 0: Load the production evidence through existing readers/CLIs, or stop

**Files:**
- Read: `runtime/oaf_tf1/model.json`
- Read: `runtime/muscriptor/model.json`
- Read: production OaF/MuScriptor run snapshots
- Read/regenerate: production reviewed-subset HPA-325 reports
- Read: production HPA-327 reviewed-subset manifest
- Read: production HPA-328 separation run/comparison/handoff artifacts
- Read: production HPA-396 IDM run/comparison artifacts
- Regenerate/validate: production HPA-562 publication bundle
- Modify only if the gate fails: Linear HPA-329 discussion, not repository files

**Interfaces:**
- Consumes:
  - `load_muscriptor_model_lock(Path) -> MuscriptorModelLock`
  - `compute_model_lock_sha256(Path) -> str`
  - `parse_muscriptor_corpus_run(bytes) -> Mapping[str, object]`
  - `muscriptor_comparison._load_evidence(...)` for the exact broad run/report validation path already used by the comparator
  - `score_oaf_reviewed_subset()` / `score_muscriptor_reviewed_subset()` through their existing CLI commands
  - `read_cohort_reports(report_dir, expected_identity=...)`
  - `publish_cross_comparisons(CrossComparisonRequest) -> CrossComparisonOutcome`
- Produces: one recorded set of concrete production paths, typed cohort objects, the fresh HPA-562 outcome, and exact identities used by every later task.

- [ ] **Step 1: Resolve the concrete production paths before validating anything**

Record the real paths from the completed upstream executions/handoffs:

```text
OAF_RUN
MUSCRIPTOR_RUN
REFERENCE_MANIFEST
TIMING_MANIFEST
SUBSET_MANIFEST
OAF_REVIEWED_REPORTS
MUSCRIPTOR_REVIEWED_REPORTS
SEPARATION_RUN
SEPARATION_CACHE (optional)
IDM_RUN
HPA562_REPUBLISH_DIR (a new, non-existing directory)
```

The runner output roots are caller-owned; do not invent a repository-default artifact location. If any required run/manifest path cannot be resolved, stop and record that missing evidence on HPA-329.

- [ ] **Step 2: Load the checked-in MuScriptor lock and require exact equality with the production run**

Run a one-off check using the production `MUSCRIPTOR_RUN` path:

```python
from pathlib import Path

from src.benchmark.artifact_io import read_regular_file_no_follow
from src.benchmark.muscriptor_corpus_run import (
    compute_model_lock_sha256,
    parse_muscriptor_corpus_run,
)
from src.benchmark.muscriptor_model import load_muscriptor_model_lock

lock_path = Path("runtime/muscriptor/model.json")
run_path = Path(MUSCRIPTOR_RUN)

lock = load_muscriptor_model_lock(lock_path)
snapshot = parse_muscriptor_corpus_run(read_regular_file_no_follow(run_path))

assert snapshot["model_id"] == lock.model_id
assert snapshot["model_lock_sha256"] == compute_model_lock_sha256(lock_path)
assert snapshot["checkpoint_revision"] == lock.checkpoint_revision
assert snapshot["checkpoint_sha256"] == lock.checkpoint_sha256
```

Do not replace this with `pytest tests/benchmark/test_muscriptor_model.py`: that suite exercises synthetic `tmp_path` locks and does not prove the checked-in production lock matches the scored run. If `runtime/muscriptor/model.json` is absent because HPA-627 is still blocked, stop HPA-329 here.

- [ ] **Step 3: Load the broad OaF and MuScriptor run/report evidence through the comparator's existing loader**

Use the exact validation seam already used by the OaF/MuScriptor comparator instead of reconstructing `CohortIdentity` manually:

```python
from pathlib import Path

from src.benchmark.backend_identity import MUSCRIPTOR_BACKEND_ID, OAF_BACKEND_ID
from src.benchmark.muscriptor_comparison import _load_evidence

oaf_broad = _load_evidence(
    Path(OAF_RUN),
    expected_backend_id=OAF_BACKEND_ID,
    argument="oaf-run",
)
muscriptor_broad = _load_evidence(
    Path(MUSCRIPTOR_RUN),
    expected_backend_id=MUSCRIPTOR_BACKEND_ID,
    argument="muscriptor-run",
)
```

This deliberately reuses a private existing seam in a disposable HPA-329 analysis invocation; do not copy its parser into HPA-329. Do not use `_compat_report_identity()` for production evidence because its dummy backend-descriptor hash is only a narrow legacy test seam.

- [ ] **Step 4: Regenerate/validate the two reviewed-subset HPA-325 reports through the existing commands**

Run the current reviewed-subset paths over the persisted parent runs; these commands do not rerun model inference:

```bash
uv run crux benchmark score-oaf-reviewed-subset \
  --run "$OAF_RUN" \
  --manifest "$REFERENCE_MANIFEST" \
  --timing-manifest "$TIMING_MANIFEST" \
  --subset-manifest "$SUBSET_MANIFEST" \
  --output-dir "$OAF_REVIEWED_REPORTS"

uv run crux benchmark score-muscriptor-reviewed-subset \
  --run "$MUSCRIPTOR_RUN" \
  --manifest "$REFERENCE_MANIFEST" \
  --timing-manifest "$TIMING_MANIFEST" \
  --subset-manifest "$SUBSET_MANIFEST" \
  --output-dir "$MUSCRIPTOR_REVIEWED_REPORTS"
```

Capture each command's `cohort_id`. Load each report directory with `read_cohort_reports()` using the corresponding broad typed identity with only `cohort_id` replaced by that command's returned reviewed-subset cohort ID:

```python
from dataclasses import replace
from pathlib import Path

from src.benchmark.reports import read_cohort_reports

oaf_reviewed = read_cohort_reports(
    Path(OAF_REVIEWED_REPORTS),
    expected_identity=replace(oaf_broad.reports.identity, cohort_id=OAF_REVIEWED_COHORT_ID),
)
muscriptor_reviewed = read_cohort_reports(
    Path(MUSCRIPTOR_REVIEWED_REPORTS),
    expected_identity=replace(
        muscriptor_broad.reports.identity,
        cohort_id=MUSCRIPTOR_REVIEWED_COHORT_ID,
    ),
)
```

Do not require a reviewed OaF/MuScriptor paired comparison in Task 0. The two reviewed HPA-325 reports are the required `reviewed_subset_full_mix` evidence.

- [ ] **Step 5: Republish HPA-562 from the production run inputs and use its outcome as the pair-count source**

Run the existing `publish-paired-comparisons` path into the new `HPA562_REPUBLISH_DIR`, or call `publish_cross_comparisons()` directly with the same inputs. The output directory must not already exist.

The successful outcome must expose exactly these four non-negative counts:

```text
oaf_muscriptor_full_mix
oaf_separation_pilot.spleeter
oaf_separation_pilot.htdemucs
oaf_idm_htdemucs
```

Treat `CrossComparisonOutcome.pairable_success_counts` (or the CLI JSON rendered directly from that outcome) as authoritative. Do not hand-open a stale `summary.json`, check its schema, and call that equivalent validation: `publish_cross_comparisons()` already re-runs the owning comparison validators, cross-validates shared identity, checks the closed artifact set, and publishes atomically.

- [ ] **Step 6: Freeze the four-scope evidence map**

Record this exact mapping with the concrete paths resolved in Steps 1-5:

```text
broad_full_mix
  absolute: oaf_broad.reports + muscriptor_broad.reports
  pair:     HPA562_REPUBLISH_DIR/comparisons/oaf-muscriptor

reviewed_subset_full_mix
  absolute: oaf_reviewed + muscriptor_reviewed
  pair:     none required

reviewed_separator_pilot
  absolute: HPA-328 full_mix/spleeter/htdemucs view reports
  pair:     HPA562_REPUBLISH_DIR/comparisons/oaf-separation

idm_htdemucs_pilot
  absolute: HPA-396 OaF/IDM pilot reports
  pair:     HPA562_REPUBLISH_DIR/comparisons/oaf-idm
  caution:  reviewed_subset_cross_verified == false
```

HPA-562's `comparisons/oaf-muscriptor/` is broad-only because the coordinator invokes `compare_oaf_muscriptor(..., subset_manifest_path=None)`. Never reuse it as reviewed-subset pairing.

- [ ] **Step 7: Confirm required event-diagnostic artifacts already exist where HPA-329 will use them**

Require canonical `event_diagnostics.jsonl` in:

```text
OAF_REVIEWED_REPORTS
MUSCRIPTOR_REVIEWED_REPORTS
HPA-328 full-mix/Spleeter/HTDemucs view report directories used for examples
```

The reviewed-subset scorer and HPA-328 separation scorer already request diagnostics for every successful selected song. `read_cohort_reports()` does not parse this file, so its presence is checked separately. Do not require or manufacture IDM event diagnostics.

- [ ] **Step 8: Record the gate result**

If every check passes, record the exact paths, model/run/cohort IDs, lock hashes, manifest/timing hashes, input views, the fresh HPA-562 output path, and its four pairable counts for Task 1.

If any check fails, add a concise Linear HPA-329 comment naming the missing/mismatched artifact and leave the draft PR planning-only. There is intentionally no repository commit for a failed Task 0.

---

### Task 1: Create the report skeleton and freeze its evidence identity

**Files:**
- Create: `docs/benchmark/2026-08-24-drum-transcription-findings.md`

**Interfaces:**
- Consumes: the concrete evidence set from Task 0.
- Produces: one report with a fixed scope vocabulary, evidence-source table, and traceability table reused by all later sections.

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

- [ ] **Step 2: Add the fixed scope/evidence-source table**

Use:

```markdown
| scope | absolute source | paired source | population meaning |
| --- | --- | --- | --- |
| broad_full_mix | HPA-325 OaF + MuScriptor broad reports | HPA-562 oaf-muscriptor | complete technically eligible broad corpus with model failures preserved |
| reviewed_subset_full_mix | HPA-325 OaF + MuScriptor reviewed reports | none required | accepted HPA-327 reviewed membership |
| reviewed_separator_pilot | HPA-328 full mix/Spleeter/HTDemucs view reports | HPA-562 oaf-separation | fixed HPA-328 pilot only |
| idm_htdemucs_pilot | HPA-396 OaF/IDM pilot reports | HPA-562 oaf-idm | fixed HPA-396 identical-input pilot; HPA-562 does not cross-verify reviewed membership against HPA-328 |
```

This table is the report-level guard against both an unlabeled leaderboard and the broad-vs-reviewed OaF/MuScriptor mix-up.

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

Do not copy a value from another model/view merely because it is expected to match; use Task 0 validation to prove equality and list the authoritative value.

- [ ] **Step 4: Record the exact Crux revision used to validate the report**

Run:

```bash
git rev-parse HEAD
```

Record that SHA in the reproducibility appendix. If later commits change any benchmark reader used by this report, update the recorded SHA during Task 8.

- [ ] **Step 5: Commit the traceable report shell**

```bash
git add docs/benchmark/2026-08-24-drum-transcription-findings.md
git commit -m "docs: start HPA-329 benchmark findings"
```

---

### Task 2: Populate OaF broad-corpus and reviewed-subset capability from HPA-325

**Files:**
- Modify: `docs/benchmark/2026-08-24-drum-transcription-findings.md`
- Local-only: `/tmp/hpa329-headline-rows.txt`

**Interfaces:**
- Consumes: `oaf_broad.reports` and `oaf_reviewed` from Task 0.
- Produces: absolute OaF capability tables and exact Markdown headline rows that Task 8 can regenerate and compare.

- [ ] **Step 1: Emit the six canonical aggregate rows per OaF scope from typed `PublishedAggregate` objects**

Keep fixed order:

```text
30 raw
30 aligned
50 raw
50 aligned
100 raw
100 aligned
```

For each row emit and paste the canonical persisted values for:

```text
event-micro precision
 event-micro recall
 event-micro F1
 song-macro F1
 class-macro F1
 successful-song count
 song F1 min/p10/p25/median/p75/p90/max
```

Append the exact generated Markdown table rows to `/tmp/hpa329-headline-rows.txt`; do not retype decimals from memory or derive event-micro values from `per_song.csv`.

- [ ] **Step 2: Add complete population/coverage context before score interpretation**

For broad and reviewed scopes, copy from the typed report objects/items:

```text
total
success
failed
skipped
quarantined
failure reason counts
prediction mapping coverage/native-output coverage evidence
```

Append any headline population table rows to `/tmp/hpa329-headline-rows.txt` as exact lines.

Call out selection bias explicitly: the reviewed subset is a diagnostic quality-control view, not a replacement for broad-corpus performance.

- [ ] **Step 3: Add the per-class evidence that explains the conclusion**

Start from each HPA-325 aggregate `per_class`. Include all classes in an appendix/table when practical, and in prose call out only material classes that explain observed strengths/weaknesses. Preserve reference/prediction support.

- [ ] **Step 4: Explain the raw/aligned gap without promoting aligned scores to the headline**

Use the persisted raw/aligned aggregates and timing diagnostics. State whether alignment materially changes the interpretation and what that implies about residual timing/source semantics. Keep raw scores as the ground-truth-facing benchmark view.

- [ ] **Step 5: Commit**

```bash
git add docs/benchmark/2026-08-24-drum-transcription-findings.md
git commit -m "docs: summarize OaF benchmark capability"
```

---

### Task 3: Populate the fixed OaF separation-pilot conclusion from HPA-562/HPA-328

**Files:**
- Modify: `docs/benchmark/2026-08-24-drum-transcription-findings.md`
- Append local verification rows: `/tmp/hpa329-headline-rows.txt`

**Interfaces:**
- Consumes: fresh HPA-562 `comparisons/oaf-separation/`, `CrossComparisonOutcome.pairable_success_counts`, and HPA-328 runtime/storage/view reports.
- Produces: one pilot-scoped full-mix vs Spleeter vs HTDemucs conclusion.

- [ ] **Step 1: Copy the exact paired population evidence**

Use the validated HPA-562 nested summary and outcome for Spleeter and HTDemucs:

```text
pairable_success_intersection
left/right-only successes
source/input exclusions defined by the upstream comparison
paired song-row count
paired class-row count
```

The pairable intersections must equal `CrossComparisonOutcome.pairable_success_counts["oaf_separation_pilot.spleeter"]` and `["oaf_separation_pilot.htdemucs"]`. Append the exact report lines to `/tmp/hpa329-headline-rows.txt`.

- [ ] **Step 2: Summarize paired score movement without rebuilding the join**

Read the already-generated:

```text
comparisons/oaf-separation/spleeter/paired_per_song.csv
comparisons/oaf-separation/spleeter/paired_per_class.csv
comparisons/oaf-separation/htdemucs/paired_per_song.csv
comparisons/oaf-separation/htdemucs/paired_per_class.csv
```

Use the published aggregate deltas and representative rows. Do not recompute a new paired population.

- [ ] **Step 3: Add measured cost/complexity evidence**

Use HPA-328 persisted runtime and retained-byte evidence to compare direct OaF with separator + OaF. Discuss wall time, storage, extra environment/model dependencies, and operational steps. Do not invent dollar costs.

- [ ] **Step 4: Write one explicit separator conclusion**

State one of:

```text
keep OaF full mix
use Spleeter before OaF
use HTDemucs before OaF
pilot is inconclusive
```

Tie it to both quality delta and added cost. Keep it labeled `reviewed_separator_pilot`.

- [ ] **Step 5: Commit**

```bash
git add docs/benchmark/2026-08-24-drum-transcription-findings.md
git commit -m "docs: conclude OaF separation pilot"
```

---

### Task 4: Populate MuScriptor independent and broad-paired findings

**Files:**
- Modify: `docs/benchmark/2026-08-24-drum-transcription-findings.md`
- Append local verification rows: `/tmp/hpa329-headline-rows.txt`

**Interfaces:**
- Consumes: `muscriptor_broad.reports`, `muscriptor_reviewed`, fresh HPA-562 broad OaF/MuScriptor comparison, and the validated MuScriptor run/model lock.
- Produces: separate accuracy, coverage, reliability, runtime, and license conclusions for MuScriptor.

- [ ] **Step 1: Report MuScriptor independently on broad and reviewed scopes**

Use the same six-row typed HPA-325 aggregate shape as Task 2. Include complete success/failure population before any OaF pairing. Append the exact generated Markdown headline rows to `/tmp/hpa329-headline-rows.txt`.

- [ ] **Step 2: Add only the broad OaF/MuScriptor paired comparison from HPA-562**

Use fresh HPA-562 `comparisons/oaf-muscriptor/` for:

```text
pairable success count
exclusions
paired song deltas
paired class deltas
```

This is `broad_full_mix` only. Do **not** present it as reviewed-subset pairing. Reviewed OaF and MuScriptor remain independent HPA-325 cohorts unless the final decision specifically requires an additional pair.

If that additional reviewed pair becomes necessary, run the already-existing `compare-oaf-muscriptor --subset-manifest` command against the same parent runs/manifests and label the resulting artifact `reviewed_subset_full_mix`; do not add a new join or modify HPA-562.

- [ ] **Step 3: Add native-output and timing evidence**

Report native pitch/class coverage, mapped/unmapped counts, timing behavior, and representative errors from persisted run/reports. Explicitly state that the released MuScriptor transcription path does not preserve velocity.

- [ ] **Step 4: Add runtime/memory and license suitability**

Use measured run evidence for runtime, real-time factor, process/device memory where available, and operational reliability. Record the checkpoint weight license from the checked-in lock that Task 0 proved matches the run. Treat license suitability separately from accuracy.

- [ ] **Step 5: Commit**

```bash
git add docs/benchmark/2026-08-24-drum-transcription-findings.md
git commit -m "docs: summarize MuScriptor comparison"
```

---

### Task 5: Populate the fixed IDM HTDemucs-stem pilot findings

**Files:**
- Modify: `docs/benchmark/2026-08-24-drum-transcription-findings.md`
- Append local verification rows: `/tmp/hpa329-headline-rows.txt`

**Interfaces:**
- Consumes: fresh HPA-562 `comparisons/oaf-idm/`, its top-level `scope_identity`, and HPA-396 run/model/runtime evidence.
- Produces: an IDM pilot conclusion that cannot be confused with broad/full-mix or cross-verified reviewed-subset evidence.

- [ ] **Step 1: State the identical-input pilot contract and scope caution before metrics**

Record the exact pilot input-view identity and the upstream proof that OaF and IDM consumed identical retained HTDemucs stem hashes. Label the section `idm_htdemucs_pilot`.

Also copy the HPA-562 scope qualification:

```text
pilot_lineage == validated_hpa396_run
reviewed_subset_cross_verified == false
```

Do not claim that HPA-562 proved the IDM pilot and HPA-328 separator publication use one cross-verified HPA-327 reviewed membership.

- [ ] **Step 2: Report population and paired deltas**

Use HPA-562/HPA-396 for exact success/failure population, pairable-success intersection/exclusions, and existing paired song/class deltas. The intersection must equal `CrossComparisonOutcome.pairable_success_counts["oaf_idm_htdemucs"]`. Append exact headline rows to `/tmp/hpa329-headline-rows.txt`.

- [ ] **Step 3: Add native capability and operational evidence**

Report checkpoint-native class coverage, velocity availability, mapping coverage, failure reasons, install/runtime maturity, measured runtime, and memory where available.

- [ ] **Step 4: Keep any direct-full-mix smoke result separate**

If HPA-396 contains a direct-full-mix compatibility smoke, label it smoke/compatibility evidence only. It is not a full-corpus or scored-peer result.

- [ ] **Step 5: State whether a larger IDM experiment is justified**

Write `yes`, `no`, or `inconclusive` with one evidence-based reason. Do not create a full-corpus IDM issue until Task 7 chooses the primary next step or directly justifies that follow-up.

- [ ] **Step 6: Commit**

```bash
git add docs/benchmark/2026-08-24-drum-transcription-findings.md
git commit -m "docs: summarize IDM stem pilot"
```

---

### Task 6: Build a small failure taxonomy from existing diagnostics and paired rows only

**Files:**
- Modify: `docs/benchmark/2026-08-24-drum-transcription-findings.md`
- Read only: reviewed/separation `event_diagnostics.jsonl`, HPA-562 paired CSVs, native-coverage/runtime artifacts

**Interfaces:**
- Consumes: canonical diagnostics already emitted by reviewed-subset scoring and HPA-328 view scoring; paired song/class rows; native coverage/failure evidence.
- Produces: representative examples classified by cause with ambiguity called out explicitly.

- [ ] **Step 1: Select a bounded diagnostic sample from already-published evidence**

Choose a small set that covers useful contrasts:

```text
one strong matched case
one large false-positive/overprediction case
one large false-negative/underprediction case
one large separator improvement or regression
one material cross-model disagreement
one ambiguous timing/source/reference case if present
```

Prefer reviewed-subset songs. Score/delta extremes may select examples for diagnosis but never change benchmark membership.

- [ ] **Step 2: Read canonical event diagnostics directly for reviewed/subset/separation examples**

`read_cohort_reports()` intentionally excludes event diagnostics from `PublishedCohortReports`, so read each `event_diagnostics.jsonl` line with the existing strict JSON loader and require the published fields:

```text
cohort_id
simfile_id
tolerance_ms
mode
outcome
common_class
reference_time_sec
prediction_time_sec
scored_prediction_time_sec
timing_error_sec
```

Use these persisted matched/FP/FN rows plus native mapping/coverage evidence. Do not infer event-level causes from aggregate F1 alone.

- [ ] **Step 3: Do not create an HPA-329 diagnostic rescore path**

There is no `score_cohort(..., diagnostics_for=...)` call in HPA-329.

If OaF or MuScriptor reviewed diagnostics are missing, rerun the existing `score-oaf-reviewed-subset` / `score-muscriptor-reviewed-subset` command from Task 0 against the persisted predictions. If a separation example lacks event diagnostics, stay on its existing paired song/class evidence rather than rerunning separation or inference. For IDM, use HPA-562 paired CSVs plus native coverage/failure/runtime evidence only; do not add event diagnostics after the fact.

- [ ] **Step 4: Classify examples into the ticket's failure taxonomy**

Use:

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

For each example, cite the persisted evidence. If two causes remain plausible, mark it ambiguous rather than forcing one label.

- [ ] **Step 5: Summarize the dominant measured limitations**

Rank only limitations supported by reviewed examples and population-level metrics. Do not claim a causal percentage split that the benchmark did not measure.

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

### Task 8: Re-read typed evidence, verify every headline cell, then hand the result back to Linear

**Files:**
- Modify if verification finds a report defect: `docs/benchmark/2026-08-24-drum-transcription-findings.md`
- No production-code changes

**Interfaces:**
- Consumes: complete HPA-329 report and the same Task 0 typed readers/outcome.
- Produces: review-ready single-PR closeout with no unsupported or mistyped headline values.

- [ ] **Step 1: Re-run the full Task 0 reader/publication pass**

Freshly reload:

```text
checked-in MuScriptor lock + production MuScriptor run equality
broad OaF/MuScriptor via _load_evidence()
reviewed OaF/MuScriptor via read_cohort_reports() with exact reviewed cohort IDs
fresh HPA-562 publish_cross_comparisons() outcome
HPA-328/HPA-396 owning report/runtime/handoff evidence
```

Do not validate from values copied earlier into notes.

- [ ] **Step 2: Regenerate exact Markdown headline rows from typed fields and require line-for-line equality**

Regenerate `/tmp/hpa329-headline-rows.final.txt` from the same source objects used in Tasks 2-5. Every absolute row must take values directly from fields such as:

```text
PublishedAggregate.event_micro.precision/recall/f1
PublishedAggregate.song_macro_f1
PublishedAggregate.class_macro_f1
PublishedAggregate.successful_song_count
PublishedAggregate.song_f1_distribution.*
PublishedCohortReports.population.*
```

Every paired count must come from the fresh `CrossComparisonOutcome`/validated nested summary fields such as `pairing.pairable_success_intersection`; runtime/storage/license rows must come from their owning persisted evidence.

For every generated exact Markdown line, require exactly one identical line in the report:

```bash
while IFS= read -r row; do
  test "$(grep -Fxc -- "$row" docs/benchmark/2026-08-24-drum-transcription-findings.md)" -eq 1 || exit 1
done < /tmp/hpa329-headline-rows.final.txt
```

A copied 50 ms value from the 30 ms row must fail this check. Grep for unfinished prose later is only supplementary verification.

- [ ] **Step 3: Reconcile populations, pairing, and scope contracts**

Require:

```text
complete per-model populations balance
fresh HPA-562 four pairable-success counts equal the report
paired row counts/exclusions match nested summaries
reviewed absolute rows come from reviewed HPA-325 reports, not HPA-562 oaf-muscriptor
reviewed/pilot memberships match their owning manifests/runs
IDM section explicitly preserves reviewed_subset_cross_verified == false
```

Fix the report or upstream evidence; never hand-adjust a number to make totals fit.

- [ ] **Step 4: Scan for conclusion mistakes**

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

- [ ] **Step 5: Scan for incomplete planning language**

Run:

```bash
grep -nE 'TBD|TODO|placeholder|fill in|to be determined' \
  docs/benchmark/2026-08-24-drum-transcription-findings.md && exit 1 || true
```

Expected: no unfinished-report matches.

- [ ] **Step 6: Run repository diff hygiene and confirm the task stayed docs-only**

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

- [ ] **Step 7: Refresh the recorded analysis revision after the final report commit**

Run:

```bash
git rev-parse HEAD
```

Update the reproducibility appendix if it names an earlier analysis SHA, commit that one-line correction, then rerun the headline-row verification and `git diff --check`.

- [ ] **Step 8: Move the draft PR to review and update Linear**

When every gate passes:

```text
mark the existing HPA-329 draft PR ready for review
move HPA-329 to In Review
add the PR/report link and one-paragraph executive conclusion to HPA-329
```

Do not mark HPA-329 Done until the report PR is accepted/merged and any child-completion/removal-with-reason condition required to close HPA-319 has been checked.

- [ ] **Step 9: After merge, close HPA-329 and evaluate HPA-319 closeout**

Confirm all HPA-319 children are completed or explicitly removed with reason, then mark HPA-329 Done and close HPA-319 only when its parent acceptance criteria are satisfied. HPA-627 cannot be silently ignored if the final report depends on MuScriptor scored evidence.
