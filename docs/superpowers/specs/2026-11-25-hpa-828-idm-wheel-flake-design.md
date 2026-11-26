# HPA-828 — IDM wheel-builder flake: diagnosis and existing fix

- [Linear HPA-828](https://linear.app/cwchanap/issue/HPA-828/revalidate-intermittent-idm-wheel-builder-test-failure) · [Original report #28](https://github.com/cwchanap/Crux/issues/28) · [Design review](https://github.com/cwchanap/Crux/pull/37#pullrequestreview-6067735175)
- Reviewed main: `282586003c17923b97d8b0f33ac062b5f5328b90`
- Disposition: existing fix; documentation and focused confirmation only.

## Diagnosis

The historical [7975d80 test](https://github.com/cwchanap/Crux/blob/7975d80/tests/runtime/test_idm_wheel_builder.py) wrote a synthetic wheel using `writestr(relative.as_posix(), blob)`, which assigns a wall-clock ZIP entry timestamp, then asserted byte equality of two wheels. Several Git subprocesses ran between writes. Crossing a DOS timestamp's two-second boundary changes archive bytes even if every source blob is identical. This explains the intermittent nature and sensitivity to elapsed time.

[ef43d75](https://github.com/cwchanap/Crux/commit/ef43d75813615bb0097b016fafb5d6ea597ab754) replaced the helper's entry timestamp with fixed `ZipInfo(date_time=(1980, 1, 1, 0, 0, 0))`, explicitly preserved DEFLATE compression, and added an assertion for that timestamp before the wheel-byte comparison. The [current test](https://github.com/cwchanap/Crux/blob/282586003c17923b97d8b0f33ac062b5f5328b90/tests/runtime/test_idm_wheel_builder.py) still has both guards. The tampered-wheel fixture may use a wall-clock timestamp, but it is not compared byte-for-byte. The production builder uses pinned Git blobs and delegates the real wheel build to uv with `SOURCE_DATE_EPOCH=0`; this reproduced defect is in the synthetic test helper, not the production build path.

The original Git/index-lock suggestion was unsupported. Each test invocation creates its own Git repository under `tmp_path`; the later [8dc072c](https://github.com/cwchanap/Crux/commit/8dc072c16d8fcff892a6a47225a290eaa64b90fa) also isolated fixture Git configuration and added the Python 3.11 runtime CI lane. There is no demonstrated shared-index contention, so do not serialize Git operations. The timestamp defect is conclusively reproducible and fixed by construction; the original one-off failure's traceback is unavailable, so historical attribution is strongly supported, not independently proven for that exact occurrence.

## Evidence

The linked reviewer ran the actual historical and current tests in scratch copies (macOS); these are reviewer-reported results:

| Revision | Forced 2 s clock step | Natural clock |
| --- | --- | --- |
| 7975d80 | Fails at wheel-byte assertion | 28/300 failed |
| 2825860 | Passes | Passes in the reported control |

During review resolution, source and commit diffs were independently checked and a minimal scratch ZIP experiment (Python 3.13.5) confirmed that identical committed payloads with timestamps two seconds apart produce unequal old-style wheel bytes, while fixed timestamps give identical bytes. This experiment isolates the mechanism; it is not a full regression run.

The reviewer also reports 53 successful CI runs since August 23, 2026, with both test lanes. That aggregate was not independently enumerated. GitHub MCP independently confirmed [PR #36 CI run 45673844492](https://github.com/cwchanap/Crux/actions/runs/45673844492) on head `88edbdb1e6012904d9e10bf2177224fda5e7cd10` passed both the Python 3.11 runtime and Python 3.12 full test jobs (PR merged as `2825860`). The [historical failed run 42980232928](https://github.com/cwchanap/Crux/actions/runs/42980232928) had a successful runtime job and a failed full-suite job; the reviewer reports unrelated failing tests. CI is corroboration, not proof of the mechanism.

## Scope and acceptance

One PR, evidence-only. Do not change the production builder, add a duplicate regression, retry, sleep, serialize Git, add CI jobs, or run a stress matrix. Reuse the existing focused end-to-end regression with real local Git and ZIP operations, checking committed/staged/unstaged source extraction, manifest equality, fixed timestamp, byte equality and tamper rejection. Run it once on each Python 3.11 and 3.12; no model, upstream build, R2 or UI work.

- [ ] Record exact source revision, Python versions, pytest commands and exit results for the two focused runs; retain output or CI links. Perform automated end-to-end verification through MCP (e.g. execution-capable runner or GitHub Actions with MCP-verified job evidence). Evidence format is not dependent on a particular server; if MCP is unavailable, record the missing verification rather than silently waive it.
- [ ] Record the deterministic historical failure mechanism, existing fix, and reviewer evidence separately from independently verified results. The plan includes a small reproducible scratch recipe; do not commit a new test/harness.
- [ ] Update Linear to replace the unsupported Git-lock hypothesis and superseded stress matrix. Close as the diagnosed timestamp defect already fixed after focused confirmation; retain the caveat about the original missing traceback. If a fresh failure appears, preserve it and keep the issue open; investigate only that demonstrated symptom.

No visual acceptance applies.
