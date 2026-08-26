# HPA-329 Comparative Benchmark Findings Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Publish one traceable Markdown findings report that closes the pretrained drum-transcription benchmark phase and selects one evidence-supported next step for Crux.

**Architecture:** Reuse the existing HPA-325 typed readers and HPA-562 comparison publisher. Validate each scope independently to find upstream problems early, but keep the PR planning-only until the final all-evidence gate passes. Then add the report, populate it from typed evidence, and rerun the committed reproducibility recipe before review.

**Tech Stack:** Python 3.12, existing Crux benchmark readers/validators and CLI commands, Markdown, shell/git. No new runtime dependency or Python helper file.

**Spec:** `docs/superpowers/specs/2026-08-24-hpa-329-benchmark-findings-design.md`

## Global constraints

The spec is authoritative for scope ownership, reader choice, diagnostics policy, decision rules, and non-goals. In particular:

- one PR for HPA-329;
- no findings Markdown commit until the spec's final evidence gate passes;
- no new scorer, join, report generator, CLI, notebook/dashboard, or committed extractor script;
- `reviewed_subset_full_mix` comes from the two reviewed HPA-325 reports, not HPA-562;
- HPA-562's IDM `reviewed_subset_cross_verified == false` qualification must remain visible.

## Files

```text
Already in the planning PR
  docs/superpowers/specs/2026-08-24-hpa-329-benchmark-findings-design.md
  docs/superpowers/plans/2026-08-24-hpa-329-benchmark-findings.md

Create only after Task 0B passes
  docs/benchmark/2026-08-24-drum-transcription-findings.md
```

---

### Task 0A: Run independent per-scope evidence prechecks

**Files:** read production artifacts only; no repository change.

**Interfaces:**
- `load_muscriptor_model_lock(Path) -> MuscriptorModelLock`
- `compute_model_lock_sha256(Path) -> str`
- `parse_muscriptor_corpus_run(bytes) -> Mapping[str, object]`
- `muscriptor_comparison._load_evidence(...) -> _RunEvidence`
- `muscriptor_comparison._report_identity_from_snapshot(snapshot) -> CohortIdentity`
- `read_cohort_reports(report_dir, expected_identity=...) -> PublishedCohortReports`
- reviewed-subset CLI commands from `src/cli/benchmark.py`

- [ ] **Step 1: Resolve the real production paths**

Record these shell variables from the completed upstream runs/handoffs:

```bash
export OAF_RUN=...
export MUSCRIPTOR_RUN=...
export REFERENCE_MANIFEST=...
export TIMING_MANIFEST=...
export SUBSET_MANIFEST=...
export OAF_REVIEWED_REPORTS=...
export MUSCRIPTOR_REVIEWED_REPORTS=...
export SEPARATION_RUN=...
export SEPARATION_CACHE=...   # optional; leave unset if not needed
export IDM_RUN=...
export HPA562_REPUBLISH_ROOT=...  # fixed parent directory for HPA-562 publications
```

These are execution inputs, not repository defaults. Exporting them lets the Python
prechecks read them through `os.environ` (shell variables are not Python globals, so
bare `Path(OAF_RUN)` would raise `NameError`).

`publish_cross_comparisons()` rejects an existing output directory, so never record a
concrete publication directory. Instead, derive a fresh one under the fixed root
immediately before each publication:

```bash
HPA562_REPUBLISH_DIR="$HPA562_REPUBLISH_ROOT/hpa562-$(date -u +%Y%m%dT%H%M%SZ)-$$"
```

If a required path cannot be resolved, record the missing upstream evidence on HPA-329.

- [ ] **Step 2: Validate broad OaF and bind its typed HPA-325 report**

Run one disposable Python invocation using the existing loader and the real report identity:

```python
import os
from pathlib import Path
from src.benchmark.backend_identity import OAF_BACKEND_ID
from src.benchmark.muscriptor_comparison import (
    _load_evidence,
    _report_identity_from_snapshot,
)
from src.benchmark.reports import read_cohort_reports

oaf_evidence = _load_evidence(
    Path(os.environ["OAF_RUN"]), expected_backend_id=OAF_BACKEND_ID, argument="oaf-run"
)
oaf_identity = _report_identity_from_snapshot(oaf_evidence.snapshot)
oaf_broad = read_cohort_reports(
    Path(os.environ["OAF_RUN"]).parent / "reports",
    expected_identity=oaf_identity,
)
```

Do not read absolute aggregates from `oaf_evidence.reports`; that reduced `_Reports` object has no aggregate/population/identity fields.

- [ ] **Step 3: Validate the MuScriptor lock against the production run, then bind its typed broad report**

Run:

```python
import os
from pathlib import Path
from src.benchmark.artifact_io import read_regular_file_no_follow
from src.benchmark.backend_identity import MUSCRIPTOR_BACKEND_ID
from src.benchmark.muscriptor_comparison import (
    _load_evidence,
    _report_identity_from_snapshot,
)
from src.benchmark.muscriptor_corpus_run import (
    compute_model_lock_sha256,
    parse_muscriptor_corpus_run,
)
from src.benchmark.muscriptor_model import load_muscriptor_model_lock
from src.benchmark.reports import read_cohort_reports

lock_path = Path("runtime/muscriptor/model.json")
run_path = Path(os.environ["MUSCRIPTOR_RUN"])
lock = load_muscriptor_model_lock(lock_path)
snapshot = parse_muscriptor_corpus_run(read_regular_file_no_follow(run_path))
assert snapshot["model_id"] == lock.model_id
assert snapshot["model_lock_sha256"] == compute_model_lock_sha256(lock_path)
assert snapshot["checkpoint_revision"] == lock.checkpoint_revision
assert snapshot["checkpoint_sha256"] == lock.checkpoint_sha256

muscriptor_evidence = _load_evidence(
    run_path,
    expected_backend_id=MUSCRIPTOR_BACKEND_ID,
    argument="muscriptor-run",
)
muscriptor_identity = _report_identity_from_snapshot(muscriptor_evidence.snapshot)
muscriptor_broad = read_cohort_reports(
    run_path.parent / "reports",
    expected_identity=muscriptor_identity,
)
```

If the checked-in lock is absent because HPA-627 is blocked, this MuScriptor precheck fails. OaF/separation/IDM prechecks may still be run locally, but the PR remains planning-only.

- [ ] **Step 4: Regenerate the reviewed OaF and MuScriptor reports from the parent runs**

This is the canonical reviewed-report policy; do not consume an arbitrary old reviewed directory.

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

Export `OAF_REVIEWED_COHORT_ID` and `MUSCRIPTOR_REVIEWED_COHORT_ID` from the two command outputs, then load:

```python
import os
from dataclasses import replace
from pathlib import Path
from src.benchmark.reports import read_cohort_reports

oaf_reviewed = read_cohort_reports(
    Path(os.environ["OAF_REVIEWED_REPORTS"]),
    expected_identity=replace(
        oaf_identity, cohort_id=os.environ["OAF_REVIEWED_COHORT_ID"]
    ),
)
muscriptor_reviewed = read_cohort_reports(
    Path(os.environ["MUSCRIPTOR_REVIEWED_REPORTS"]),
    expected_identity=replace(
        muscriptor_identity,
        cohort_id=os.environ["MUSCRIPTOR_REVIEWED_COHORT_ID"],
    ),
)
```

- [ ] **Step 5: Validate diagnostic content, not file existence**

For the two regenerated reviewed directories and each HPA-328 view directory that Task 6 may cite, require at least one canonical JSONL row:

```bash
test "$(grep -cve '^$' "$REPORT_DIR/event_diagnostics.jsonl")" -gt 0
```

Also parse every non-empty line with `strict_json_loads(..., require_canonical=True)` and require exactly the published diagnostic fields:

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

Existence alone is insufficient because `write_cohort_reports()` creates the file even when it contains zero diagnostics.

- [ ] **Step 6: Validate HPA-328 and HPA-396 owning evidence**

Use their existing run/comparison/handoff/model readers. Confirm:

```text
reviewed_separator_pilot: full_mix + spleeter + htdemucs view reports are readable
idm_htdemucs_pilot: OaF and IDM are bound to the retained identical HTDemucs input contract
```

Do not create a new HPA-329 reader for either path.

- [ ] **Step 7: Record precheck failures only in Linear**

Per-scope prechecks may complete in any order. They are for early diagnosis only; do not create or partially populate the findings Markdown yet.

---

### Task 0B: Pass the final all-evidence decision gate

**Files:** read/regenerate production evidence only; no repository change.

**Interfaces:** `publish_cross_comparisons(CrossComparisonRequest) -> CrossComparisonOutcome`.

- [ ] **Step 1: Require all four Task 0A scope prechecks to pass**

No exception: the final decision and executive summary require OaF, MuScriptor, separation, and IDM evidence.

- [ ] **Step 2: Freshly republish HPA-562**

Run the existing `publish-paired-comparisons` CLI or call `publish_cross_comparisons()` directly with the production runs/manifests. First derive a fresh publication path under `HPA562_REPUBLISH_ROOT` (see Task 0A Step 1); `publish_cross_comparisons()` rejects an existing output directory, so the derivation is part of every publication run.

Require exactly these four non-negative outcome keys:

```text
oaf_muscriptor_full_mix
oaf_separation_pilot.spleeter
oaf_separation_pilot.htdemucs
oaf_idm_htdemucs
```

The fresh `CrossComparisonOutcome.pairable_success_counts` is the pair-count authority.

- [ ] **Step 3: Freeze the evidence-source map from the spec**

Record the concrete typed objects and fresh comparison paths for the four spec scopes. Reviewed OaF/MuScriptor pairing remains optional and is not a gate requirement.

- [ ] **Step 4: Gate outcome**

Only after Steps 1-3 pass may Task 1 create the report file. Otherwise comment on HPA-329 and keep PR #30 planning-only.

---

### Task 1: Create the report shell and reproducibility appendix

**Files:** create `docs/benchmark/2026-08-24-drum-transcription-findings.md`.

- [ ] **Step 1: Create the fixed top-level sections from the spec**

Use the ten section names exactly as specified. Leave `Executive summary` empty until Task 7.

- [ ] **Step 2: Add the four-scope table from the spec**

Do not restate the scope rules elsewhere in the report unless a section needs a local caution.

- [ ] **Step 3: Add the evidence identity table**

For each production artifact record its concrete path, schema/version, published hash where applicable, run/cohort ID, model ID, model-lock hash, prediction-map version, input-view ID, reference/timing identity, taxonomy/lane/scoring versions.

- [ ] **Step 4: Start the reproducibility appendix with the exact commands actually used in Tasks 0A/0B**

Include the real resolved paths/IDs and the exact reader/CLI invocation. The recorded HPA-562 publication step must keep the production inputs fixed and derive a fresh output directory under `HPA562_REPUBLISH_ROOT` on every invocation, so the committed recipe stays exactly rerunnable. Do not refer to an uncommitted helper script.

Also record the fixed machine-checked table formats used later:

```text
aggregate row:
| scope | tolerance_ms | mode | event_micro_precision | event_micro_recall | event_micro_f1 | song_macro_f1 | class_macro_f1 | successful_song_count |

population row:
| scope | total | success | failed | skipped | quarantined |

pair-count row:
| comparison_id | pairable_success_count |
```

Task 8 reconstructs these exact lines directly from typed source objects and the fresh HPA-562 outcome.

- [ ] **Step 5: Commit**

```bash
git add docs/benchmark/2026-08-24-drum-transcription-findings.md
git commit -m "docs: start HPA-329 benchmark findings"
```

---

### Task 2: Populate OaF broad and reviewed capability

**Files:** modify the findings Markdown.

**Interfaces:** `oaf_broad: PublishedCohortReports`, `oaf_reviewed: PublishedCohortReports`.

- [ ] **Step 1: Add fixed aggregate tables**

Use six rows per scope in order `30 raw`, `30 aligned`, `50 raw`, `50 aligned`, `100 raw`, `100 aligned` with the aggregate row shape defined in Task 1.

Copy values from `PublishedAggregate`; never derive event-micro from song rows.

- [ ] **Step 2: Add population/coverage rows**

Use the population row shape from Task 1 and `PublishedCohortReports.population`. Include failure reason counts and native/mapping coverage evidence before interpreting paired results.

- [ ] **Step 3: Add per-class/distribution evidence and explain raw vs aligned**

Use aggregate per-class values/support and song F1 distributions. Keep aligned scores diagnostic only.

- [ ] **Step 4: Commit**

```bash
git add docs/benchmark/2026-08-24-drum-transcription-findings.md
git commit -m "docs: summarize OaF benchmark capability"
```

---

### Task 3: Populate the OaF separation pilot

**Files:** modify the findings Markdown.

**Interfaces:** fresh HPA-562 separation comparison + HPA-328 reports/runtime/storage/handoff.

- [ ] **Step 1: Record pairable counts/exclusions and paired song/class deltas from the existing publication**

The pairable intersections must equal the fresh outcome keys for Spleeter and HTDemucs. Use the Task 1 pair-count row format for machine-checked counts. The exclusion helper only distinguishes left-only success, right-only success, and source/canonical-input mismatch — it does not preserve failed/skipped/quarantined populations.

- [ ] **Step 2: Add full-population aggregate/population rows for all three input views**

The HPA-328 view reports are this scope's absolute-metric authority. Bind the full_mix, spleeter, and htdemucs view reports via `read_cohort_reports` and emit the Task 1 aggregate and population row shapes for each view before any pilot conclusion, so a separator cannot look better only on the surviving pairable intersection while having materially worse execution coverage.

- [ ] **Step 3: Add measured runtime/storage/operational evidence**

Do not invent dollar costs.

- [ ] **Step 4: State one pilot conclusion**

Choose exactly one: keep full mix, use Spleeter, use HTDemucs, or pilot inconclusive. Label it `reviewed_separator_pilot`.

- [ ] **Step 5: Commit**

```bash
git add docs/benchmark/2026-08-24-drum-transcription-findings.md
git commit -m "docs: conclude OaF separation pilot"
```

---

### Task 4: Populate MuScriptor broad/reviewed and broad-paired evidence

**Files:** modify the findings Markdown.

**Interfaces:** `muscriptor_broad`, `muscriptor_reviewed`, checked-in lock, fresh HPA-562 broad pair.

- [ ] **Step 1: Add the same typed aggregate/population table shapes used by Task 2**

- [ ] **Step 2: Add only the broad HPA-562 OaF/MuScriptor pair**

Do not label it as reviewed pairing. If the final decision specifically requires reviewed pairing, invoke the existing `compare-oaf-muscriptor --subset-manifest` command and label that separate artifact `reviewed_subset_full_mix`.

- [ ] **Step 3: Add native coverage/timing/runtime/memory/license evidence**

State that released MuScriptor transcription does not preserve velocity. Keep `weight_license` and `code_license` separate.

- [ ] **Step 4: Commit**

```bash
git add docs/benchmark/2026-08-24-drum-transcription-findings.md
git commit -m "docs: summarize MuScriptor comparison"
```

---

### Task 5: Populate IDM HTDemucs-stem pilot evidence

**Files:** modify the findings Markdown.

**Interfaces:** fresh HPA-562 IDM pair + HPA-396 run/model/runtime/native evidence.

- [ ] **Step 1: State the identical-input pilot contract and scope caution first**

Include `reviewed_subset_cross_verified == false` and the HPA-396 lineage qualification.

- [ ] **Step 2: Add absolute aggregate/population/pairing/native/runtime evidence**

The HPA-396 OaF/IDM pilot reports are this scope's absolute-metric authority: emit the Task 1 aggregate and population row shapes for both reports before interpreting the pairable intersection. The pairable intersection must equal the fresh `oaf_idm_htdemucs` outcome count. Use the Task 1 pair-count row format for the machine-checked count.

- [ ] **Step 3: Keep direct-full-mix smoke separate and decide whether larger IDM work is justified**

Answer yes/no/inconclusive; do not create the follow-up yet.

- [ ] **Step 4: Commit**

```bash
git add docs/benchmark/2026-08-24-drum-transcription-findings.md
git commit -m "docs: summarize IDM stem pilot"
```

---

### Task 6: Build the failure taxonomy from existing diagnostics

**Files:** modify the findings Markdown; read persisted diagnostics/paired/native evidence only.

- [ ] **Step 1: Select a bounded reviewed sample**

Cover strong match, overprediction, underprediction, separator improvement/regression, material cross-model disagreement, and one ambiguous source/timing case when available.

- [ ] **Step 2: Require evidence for every event-level example**

For reviewed/separation examples, require at least one parsed diagnostic row for the cited song/view. If reviewed diagnostics are missing, rerun the existing reviewed-subset command. Do not call `score_cohort()` from HPA-329.

For IDM, use paired/native/failure/runtime evidence only.

- [ ] **Step 3: Classify without false certainty**

Use the taxonomy from the spec and label ambiguous cases explicitly.

- [ ] **Step 4: Commit**

```bash
git add docs/benchmark/2026-08-24-drum-transcription-findings.md
git commit -m "docs: classify benchmark failure modes"
```

---

### Task 7: Select one next step and write the executive summary

**Files:** modify the findings Markdown; create Linear follow-ups only after the written decision.

- [ ] **Step 1: Evaluate all eight Linear options in one table**

Exactly one row is `Primary next step`; every other option gets a concise evidence-based rejected/deferred reason.

- [ ] **Step 2: Apply the hobby-project cost rule**

A separator, larger model, new research path, or fine-tuning must earn its extra compute/storage/integration/license cost.

- [ ] **Step 3: Write the executive summary last**

State the best measured path, dominant limitation, primary next step, and why the closest alternative lost. Preserve broad/pilot qualifications.

- [ ] **Step 4: Create only evidence-required Linear follow-ups**

Link any created issue from HPA-329 and from `Follow-up work`.

- [ ] **Step 5: Commit**

```bash
git add docs/benchmark/2026-08-24-drum-transcription-findings.md
git commit -m "docs: decide next Crux transcription step"
```

---

### Task 8: Rerun typed verification from the committed appendix

**Files:** modify the report only if verification finds a defect.

**Interfaces:** same readers/outcome as Tasks 0A/0B; no new script file.

- [ ] **Step 1: Rerun the exact committed reproducibility commands in a fresh shell**

Revalidate the MuScriptor lock/run equality, broad typed reports, regenerated reviewed typed reports, HPA-328/HPA-396 owning evidence, diagnostic content, and a fresh HPA-562 publication into a newly derived output directory under the fixed `HPA562_REPUBLISH_ROOT`, exactly as recorded in the committed appendix.

- [ ] **Step 2: Regenerate and check every absolute headline row in the same invocation**

Use the fixed Task 1 aggregate/population row formats. Render Decimal/`None` values with:

```python
from src.benchmark.backend_identity import canonical_json_bytes

def metric(value):
    return "N/A" if value is None else canonical_json_bytes(value).decode("ascii")
```

Open the findings Markdown in that invocation. For every expected aggregate/population row constructed directly from `PublishedAggregate` and `PublishedCohortReports.population` — including the Task 3 separator-view rows and Task 5 HPA-396 rows — assert `report_text.count(expected_row) == 1`.

This check is generated fresh from the typed objects; it does not read an expected-row `/tmp` file produced by an earlier task/session.

- [ ] **Step 3: Regenerate and check paired counts/scopes**

For each fresh `CrossComparisonOutcome.pairable_success_counts` entry, construct the Task 1 pair-count row and assert it occurs exactly once. Verify nested pairing/exclusion counts against their validated summaries, reviewed absolute rows against the reviewed HPA-325 objects, and the IDM caveat against the report text.

- [ ] **Step 4: Verify diagnostic examples**

Every event-level reviewed/separation example must still resolve to at least one parsed diagnostic row for the cited song/view.

- [ ] **Step 5: Keep the exact Step 2/3 verification invocation in the report appendix**

After the report is final, copy the exact command/snippet used by Steps 2-3 into `Reproducibility appendix`, with the actual artifact paths/IDs. A later reader can therefore rerun the numeric verification without reconstructing an extractor from prose.

- [ ] **Step 6: Run prose/scope and diff hygiene**

Reject unfinished placeholders, an unlabeled pilot/broad leaderboard, aligned-as-headline wording, hidden failure populations, license/accuracy conflation, definite claims for ambiguous examples, or more than one primary decision.

Then run:

```bash
git diff --check main...HEAD
git diff --name-only main...HEAD
```

Expected HPA-329 files only:

```text
docs/superpowers/specs/2026-08-24-hpa-329-benchmark-findings-design.md
docs/superpowers/plans/2026-08-24-hpa-329-benchmark-findings.md
docs/benchmark/2026-08-24-drum-transcription-findings.md
```

- [ ] **Step 7: Refresh the recorded analysis revision and rerun Steps 1-6**

Commit the reproducibility-appendix correction first, so `HEAD` is the final-analysis revision containing the completed report. Run `git rev-parse HEAD` and record that pre-recording revision in the appendix, explicitly stating that the recorded hash identifies the final-analysis revision and that the report does not contain its own commit SHA — committing the recording necessarily produces a new `HEAD`. Commit the recording, then rerun Steps 1-6 verification fresh from that final revision.

- [ ] **Step 8: Move to review and update Linear**

Mark the same draft PR ready for review, move HPA-329 to In Review, and add the report link plus concise executive conclusion. Mark HPA-329 Done only after merge and then evaluate HPA-319 closeout, including HPA-627's required disposition.
