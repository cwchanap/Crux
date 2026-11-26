# HPA-828 — Implementation and revalidation plan

Design: [HPA-828 design](../specs/2026-11-25-hpa-828-idm-wheel-flake-design.md).
Linear: [HPA-828](https://linear.app/cwchanap/issue/HPA-828/revalidate-intermittent-idm-wheel-builder-test-failure).
Start from main at `282586003c17923b97d8b0f33ac062b5f5328b90`;
verify main again before implementation and use this draft PR for all
evidence and any necessary fix. This plan has not run the tests.

## 1. Prepare a clean baseline

- [ ] Confirm the named test and fixed timestamps/isolated fixture
  Git environment still exist. Review the original report, current
  builder's Git subprocess helpers and CI commands. Do not change them.
- [ ] Record `git rev-parse HEAD`, clean `git status --porcelain`,
  OS, Python, Git and pytest versions, and filesystem type if available.
  Use isolated Python 3.11 and 3.12 environments. For the former use
  the runtime-only CI setup (pytest); for the latter reuse the minimal
  test dependencies from `.github/workflows/ci.yml` and the editable
  package install, or the equivalent existing project environment.
  Keep heavy optional model dependencies out of the experiment.
- [ ] Put logs in a temporary directory outside the worktree; keep
  individual exit codes and durations. Do not commit temporary files,
  wheels, credentials or coverage artifacts.

## 2. Run the existing regression repeatedly

Use the following command in each environment (from repository root):

```sh
python -m pytest -q -o addopts=-ra --confcutdir tests/runtime \
  tests/runtime/test_idm_wheel_builder.py::test_commit_blobs_ignore_staged_and_unstaged_source_edits
```

- [ ] Execute 20 independent processes with Python 3.11 and 20 with
  Python 3.12, sequentially per environment. Save stdout/stderr, exit
  code and elapsed time for each, rather than looping in one pytest
  session. Do not silently retry or count a failed process as a pass.
  Use separate log directories for each environment.
- [ ] Run the complete runtime lane with Python 3.11, exactly as CI:
  `python -m pytest -q -o addopts=-ra --confcutdir tests/runtime tests/runtime/`.
  Record result and count.
- [ ] Check that the existing test verifies the committed bytes even
  after staged and unstaged mutations; manifest, verified source rows
  and wheel bytes are identical; tampered bytes are rejected. The
  regression scenario already runs real local Git and ZIP operations,
  so do not create a second synthetic end-to-end suite.

## 3. Exercise representative suite load

- [ ] With Python 3.12, run the current CI test command in a clean
  checkout/environment:

  ```sh
  PRELOAD_MODEL=0 PYTHONPATH="$PWD" python -m pytest -q \
    --cov=src --cov-report=xml:coverage.xml --cov-report=term-missing
  ```

  Record pass/fail/skip counts and preserve logs. Existing skips must
  be identified, not treated as passes or new failures.
- [ ] Repeat the full suite once with four concurrent focused workers
  using Python 3.12. Each worker runs five independent invocations
  of the step 2 focused command. Start the full suite and workers
  together, use a unique log per invocation and distinct
  `--basetemp` paths per invocation (under a temporary directory),
  retain exit codes for the full suite and every worker, and join all
  children. Set `-o addopts=-ra` on focused workers to avoid coverage
  file collisions. Use a modest 4-worker fan-out; do not create new
  CI jobs or require pytest-xdist. If runner resources make this
  impossible, report the limitation rather than silently reduce load.
  Keep coverage outputs out of the commit.
- [ ] Record the actual concurrency and whether worker/full-suite
  execution overlapped. The intended clean sample is 20/20 isolated
  focused runs on each Python version, one runtime suite, 2 full
  suites and 20/20 loaded focused runs. Stop launching new attempts
  after a failure is observed, let already-running processes finish
  and record the partial denominator.

## 4. Conditional diagnosis only if the failure recurs

- [ ] Save the first failure before any rerun. Classify it: fixture
  Git command error, production Git command error, assertion/ZIP
  mismatch, timeout, or unrelated infrastructure/full-suite failure.
- [ ] For a failing Git subprocess, capture argv, exit status and
  decoded captured stderr/stdout (sanitize sensitive information).
  If current pytest output is insufficient, add narrow test-side
  diagnostics for `CalledProcessError` and preserve the exception
  chain; use focused verbose traceback and targeted diagnostics.
  For a ZIP mismatch capture entry metadata and SHA-256s; for fixture
  failures capture relevant temp-repo state without editing it.
- [ ] Isolate one variable at a time and reproduce before attributing
  cause. Fixed ZIP timestamps and isolated fixture Git config are
  existing protections, not proof of what originally failed.
- [ ] If evidence supports a minimal fix, add it and a relevant
  regression (extend the existing test if sufficient); if production
  behavior changes, add automated end-to-end coverage of that path
  with real local Git/wheel inputs or the pinned build if actually
  necessary. Rerun steps 2–3 on the final revision. No blanket retry,
  global Git serialization, or skip. If no cause is confirmed,
  document the reproducible symptom and leave the ticket open.

## 5. Record and verify the result in this PR

- [ ] Add a short evidence section to this plan or PR body with
  tested commit, environments, actual run counts, concurrency, durations,
  failures/skips, commands, links to retained logs/CI runs, and outcome.
  For each matrix row report attempted/passed/failed, including all
  pre-fix failures. Do not claim the original cause was resolved
  merely because current repetitions pass.
- [ ] Use an execution-capable MCP runner for the focused offline
  end-to-end scenario and suite/matrix acceptance. Inspect the existing
  CI job status and available logs via GitHub MCP when CI runs; the
  workflow skips draft PR jobs, so draft status alone is not evidence.
  If the execution runner or CI evidence is unavailable, record the
  missing gate; do not mark acceptance as passed. No UI screenshots
  or human acceptance required.
- [ ] Run the repository lint/format/Pylint checks if source/test
  code was changed; verify `git diff --check` and no generated
  artifacts or unrelated changes in all cases. Keep existing CI
  behavior untouched.
- [ ] Clean bounded sample: evidence-only closure (residual flake
  risk acknowledged). Proven fix: same PR contains code and evidence.
  Unresolved or incomplete matrix: leave HPA-828 open with explicit
  reason. Link final evidence to Linear; do not close solely on plan
  approval or an old green rerun.

## Out of scope

Benchmark/model runs, MuScriptor gated model access, new tooling
dependencies, performance tuning, infrastructure redesign, and
permanent stress harnesses. One ticket, one PR; no follow-up PR
required for a clean evidence-only disposition.
