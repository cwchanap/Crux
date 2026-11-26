# HPA-828 — Revalidate the intermittent IDM wheel-builder test

- Linear: [HPA-828](https://linear.app/cwchanap/issue/HPA-828/revalidate-intermittent-idm-wheel-builder-test-failure)
- Original report: [GitHub #28](https://github.com/cwchanap/Crux/issues/28)
- Design base: `282586003c17923b97d8b0f33ac062b5f5328b90` (main)
- Status: proposed; execution evidence not yet collected.

## Problem and current evidence

The HPA-396 full suite failed the single test
`tests/runtime/test_idm_wheel_builder.py::test_commit_blobs_ignore_staged_and_unstaged_source_edits`
once (3,143 other passes). It then passed alone and the full rerun passed
(3,144 tests). The original Git/index-lock explanation was a hypothesis,
not a diagnosed root cause. The original issue is closed because tracking
moved to Linear, not because the flake was experimentally eliminated.

Two relevant changes are already present:
[ef43d758](https://github.com/cwchanap/Crux/commit/ef43d75813615bb0097b016fafb5d6ea597ab754)
and [8dc072c](https://github.com/cwchanap/Crux/commit/8dc072c16d8fcff892a6a47225a290eaa64b90fa).
The current test constructs ZIP entries with a fixed 1980 timestamp and
uses an isolated environment for fixture Git commands. The builder itself
reads pinned commit blobs through Git subprocesses, not the dirty worktree;
its Git calls inherit the process environment. That distinction is an
observation, not a reason to change production behavior preemptively.

## Decision: evidence first, no speculative fix

Use the current regression test, runtime-only Python 3.11 CI lane and
normal Python 3.12 suite. Repeated independent pytest processes exercise
the existing scenario under ordinary and modest parallel load. A clean
bounded run supports closing the revalidation issue without code changes;
it does not prove the failure is impossible or establish a root cause.
If a failure recurs, retain the failed attempt and diagnose it before
choosing a minimal change.

| Approach | Decision |
| --- | --- |
| Revalidate existing code and record results | Selected: low cost, tests the current contract directly. |
| Add retries, sleeps, Git serialization or timestamp changes now | Rejected: masks failures or duplicates changes already landed. |
| Build a permanent stress-test harness or rerun the model benchmark | Rejected: unrelated cost and infrastructure. |

## Existing contract and scope

- `tests/runtime/test_idm_wheel_builder.py`: reuse the named test and
  its committed/staged/unstaged fixture, deterministic ZIP construction,
  source-manifest and wheel-byte equality, and tampered-wheel rejection.
- `runtime/idm/build_pinned_wheel.py`: retain commit-based extraction,
  source verification, pinned upstream and build/provenance behavior.
  Change only a demonstrated production bug; prefer a test-only fix
  for a proven fixture problem.
- `.github/workflows/ci.yml`: existing `idm-runtime` job uses Python
  3.11 and the runtime test directory; the main test job uses Python
  3.12 with coverage. No permanent CI or dependency changes are planned.

## Evidence and disposition

Use a clean checkout of the candidate revision. Record exact revision,
OS/filesystem type where available, Python/Git/pytest versions, commands,
attempt counts, concurrency, exit statuses, and elapsed time. Keep per-run
logs outside the checkout, including failures; publish a concise evidence
summary and link retained logs/CI runs in the same PR. Avoid logging
credentials or the full process environment.

Baseline: 20 independent focused attempts on each of Python 3.11 and
3.12, the existing Python 3.11 runtime suite, and two Python 3.12 full
suite runs (one ordinary and one alongside four independent focused
workers, five attempts per worker). The concurrent workers use distinct
temporary directories and no coverage; do not run two coverage-producing
full suites together. This is a finite engineering sample, not a
statistical guarantee. Stop and investigate the first genuine failure;
record partial counts and distinguish dependency/runner problems from
test failures. Infrastructure failures do not count as successful runs.

- **No reproduced failure:** report exact clean counts, acknowledge
  residual uncertainty, and close the issue with evidence only.
- **Reproduced failure:** preserve traceback, Git command/exit/stderr
  when applicable, and expected vs actual artifact/assertion details.
  Add focused diagnostics only where needed, isolate the demonstrated
  cause, add a regression if not already covered, and repeat the matrix
  on the final revision. Do not hide failed attempts with automatic
  retry. If still unexplained, document the blocker and leave the issue
  open rather than claiming a fix.

No UI, model inference, R2 access, gated weights, upstream wheel rebuild,
model locks, or benchmark score changes are required. No user
acceptance or visual parity check is applicable to this test-only task.

## Acceptance

- [ ] Existing focused test and runtime CI lane pass on the recorded
  revision, with the specified independent-attempt counts.
- [ ] Both full-suite runs and all concurrent focused attempts complete
  without unexplained failures on the final revision; any original
  reproduction remains documented.
- [ ] End-to-end automated evidence executes the real local Git fixture
  through committed source extraction, staged/unstaged mutation,
  manifest and deterministic wheel verification, and tamper rejection;
  use an execution-capable MCP test runner and verify available
  GitHub Actions results/logs through GitHub MCP. This is the offline
  regression scenario, not a claim of real upstream model inference.
- [ ] If production builder code changes, add an end-to-end regression
  for the affected build/verification path (using local fixture inputs
  where sufficient) and rerun the same suite; no speculative change.
- [ ] Evidence records the outcome, residual uncertainty or diagnosed
  cause; no blanket retries, unnecessary dependencies, or unrelated
  changes. One PR contains the plan, evidence, and any justified fix.
