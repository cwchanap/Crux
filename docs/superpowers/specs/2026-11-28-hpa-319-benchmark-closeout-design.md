# HPA-319 — Benchmark evidence closeout design

Date: 2026-11-28 · Status: proposed; execution is gated on real evidence

[Linear HPA-319](https://linear.app/cwchanap/issue/HPA-319/benchmark-oaf-drums-and-selected-pretrained-alternatives-against) · Base: `37cdb9ea778965550f988cf2b4090c35fadcce80`

## Why this is next

The Crux project has no open PR and no remaining unstarted non-parent issue. Every child in the project list is marked Done, including HPA-627 immediately after PR #38 merged. HPA-319 remains Todo. Its acceptance is about benchmark evidence, not merely landed software. Treat the existing parent as the owner of a one-time evidence closeout; do not invent a new feature or silently reinterpret closed tickets as executed experiments.

| Evidence | Verified at planning time | Required disposition |
| --- | --- | --- |
| Frozen OaF broad full mix | [Archived summary](https://github.com/cwchanap/Crux/blob/599033663ca3bb495d8faaac13180206eaafadb4/evidence/oaf-broad/summary.json): cohort `oaf-2d67684ad190163e`, 320 total, 182 successes, 138 quarantined, zero failed/skipped, raw 50 ms micro F1 0.234846. | Recover the matching full run/reports and reference identities; validate before reuse. The archived summary alone is not proof of recoverable predictions. |
| OaF Docker-free candidate | [PR #36](https://github.com/cwchanap/Crux/pull/36) and `experiments/oaf_tf215/RESULTS.md`: real 5/20-song runtime comparisons and a two-song resumable candidate run. Broad/reviewed/separation continuation explicitly open. | Keep candidate and historical TF1 evidence distinct; not a replacement for the frozen control. |
| OaF separator pilot | [PR #32](https://github.com/cwchanap/Crux/pull/32) records a real 30-member rerun and a comparator fix; HPA-328 is Done. | Recover and validate complete paired views, comparison and HTDemucs handoff, not just a PR or a failed intermediate run. |
| IDM stem pilot | [PR #27](https://github.com/cwchanap/Crux/pull/27) reported missing real handoff; `runtime/idm/smoke.json` is absent at the planning base. | Locate actual pilot artifacts or complete the existing fixed pilot when prerequisites are available. |
| MuScriptor | [PR #38](https://github.com/cwchanap/Crux/pull/38) landed offline acquisition; its [last review](https://github.com/cwchanap/Crux/pull/38#pullrequestreview-6096530986) explicitly notes the missing canonical lock and GPU acceptance. `runtime/muscriptor/model.json` is absent at the merge commit. | Authenticated exact-revision freeze, real GPU smoke and broad/reviewed evidence remain unverified. |
| Paired publication and final findings | [PR #29](https://github.com/cwchanap/Crux/pull/29) reports no production paired bundle; [PR #30](https://github.com/cwchanap/Crux/pull/30) merged the findings plan, not the promised report. `docs/benchmark/2026-08-24-drum-transcription-findings.md` is absent at the planning base. | Recover/validate or publish real evidence, then follow the existing findings recipe. |

These observations establish missing checked-in artifacts and historical gates; they do not prove that private Drive/Colab/cache outputs never existed. No production execution, access to credentials, or full artifact recovery was attempted in this planning pass.

## Selected approach

Recover first, compute second. Reuse the existing HPA-627, HPA-395, HPA-328, HPA-396, HPA-562 and HPA-329 plans and their already-implemented entrypoints. No new benchmark architecture. Use one Markdown closeout ledger (this spec's evidence table is the initial inventory; the execution record in the implementation plan is the durable audit) plus the existing HPA-329 findings report **only** if its original all-evidence gate passes.

The parent is the coordination owner for the remaining operational evidence, not a second implementation of any child. Do not reopen children or create duplicate tickets/PRs in this planning pass. If an actual defect requiring independent implementation is discovered, record its scope and stop that path rather than grow this PR. Any narrowly required canonical lock/smoke artifacts and non-skipping lock assertion follow their existing owning plans in this same closeout PR; do not regenerate an existing valid artifact or add a new schema.

### Evidence and identity rules

* Use authoritative R2 `set.def`/DTX references and fixed HPA-327 membership. Resolve the real production paths and digest each input/run/report/handoff via existing validators. Match corpus/reference/timing/taxonomy/lane/scoring/input identity; preserve full success/failure/quarantine populations.
* The archived TF1 cohort is the first recovery target. PR #36's recovered September 180-eligible manifest and later candidate runs are not interchangeable with the archived 182-success cohort. Compare actual identities, never reconcile by counts alone. If no complete compatible archived run is recoverable, measure one bounded smoke/projection and resume/re-run the frozen baseline only with the approved original manifest and a feasible host. Do not silently use TF2.15 as TF1. Separation still requires the frozen TF1 parent.
* Before scoring MuScriptor, follow HPA-627's exact historical medium revision `f32236969308476e01fd3aae67357de5feb05a2d`, Python 3.12 locked optional runtime, Linux x86-64 CUDA/float32 and authenticated license/download gate. Use the existing freeze script and preparation command; record actual checkpoint/config hashes, lock hash, nonempty native drum inference, five-song repeatability, and measured broad-run projection before starting. Stop on missing access or infeasibility; no score-driven configuration switch. Keep CC BY-NC 4.0 weight restrictions and null velocity explicit.
* Recover the fixed HPA-328 pilot, or resume only its missing work using the original committed identities and matching TF1 control. Run IDM only on the retained exact HTDemucs stems; use its predeclared smoke/official lock and same reference lineage. Preserve the HPA-562 `reviewed_subset_cross_verified == false` qualification unless supported by the existing validator's actual output. Do not expand IDM or separation to the broad corpus.
* Use `read_cohort_reports`, the existing reviewed-subset commands and `publish-paired-comparisons` as specified in [HPA-329 design](../specs/2026-08-24-hpa-329-benchmark-findings-design.md). Preserve broad, reviewed, separator and IDM scopes, raw versus diagnostic aligned scores, coverage, pairing/exclusions and license/cost. Use existing canonical reports, not MLflow metrics as authority. No new dashboard is required.
* Store large private source files, weights, predictions and generated reports in the existing authorized artifact locations, not Git. Record concrete stable paths and hashes, invocation/commit and safe output in the execution record. Do not invent credentials, store tokens, copy songs into Git, or use fixture values as benchmark results.

### Gate and outcomes

For each scope, record **validated**, **blocked** (specific attempted preflight/error, or missing access/inputs and next action), or **not executed**; a scope blocked at preflight is not a population of model inference failures. The HPA-329 final gate still requires all four scopes plus a matching canonical MuScriptor lock, real diagnostics, and fresh HPA-562 paired counts. If blocked, keep HPA-319 open and the PR draft, with a precise blocker and remaining requirements. The parent allows explicit model blocker reports, but that does not waive its independent broad/reviewed comparison and population criteria. A reduced-scope milestone requires an explicit separately recorded scope disposition; do not declare full acceptance or a model winner from missing results.

Only after the full gate passes, produce the existing HPA-329 report with one evidence-backed decision, and reconcile every parent criterion with actual evidence or an already explicitly documented scope removal. Keep the parent open until the exact PR head is validated and merged.

## Non-goals

No new model, fine-tuning, chart generation/editor UI, scoring/mapping/threshold changes, general experiment runner, reporting framework, broad IDM/separator experiment, or compatibility layer. No new paid infrastructure by default. Use the existing authorized execution environment and small smoke before a broad run. No human acceptance and no UI screenshot requirement for this non-UI task.
