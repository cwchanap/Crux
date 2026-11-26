# HPA-828 — IDM wheel-builder flake: diagnosis and existing fix

- [Linear HPA-828](https://linear.app/cwchanap/issue/HPA-828/revalidate-intermittent-idm-wheel-builder-test-failure) · [Original report #28](https://github.com/cwchanap/Crux/issues/28) · [Design review](https://github.com/cwchanap/Crux/pull/37#pullrequestreview-6067735175)
- Reviewed main: `282586003c17923b97d8b0f33ac062b5f5328b90`
- Disposition: existing fix; technical verification complete on main; documentation-only closeout.

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

During review resolution, source and commit diffs were independently checked and a minimal scratch ZIP experiment (Python 3.13.5, repeated during round 2) confirmed that identical committed payloads with timestamps two seconds apart produce unequal old-style wheel bytes, while fixed timestamps give identical bytes. This experiment isolates the mechanism; it is not a full regression run.

The reviewer reports 53 successful CI runs since August 23, 2026, with both test lanes; that aggregate was not independently enumerated. The earlier PR #36 branch run also passed. More importantly, [main push CI run 45741055726](https://github.com/cwchanap/Crux/actions/runs/45741055726) supplies exact-revision evidence. Both job logs independently confirm checkout of `282586003c17923b97d8b0f33ac062b5f5328b90`.

| CI lane | Command | Result |
| --- | --- | --- |
| [idm-runtime, Python 3.11](https://github.com/cwchanap/Crux/actions/runs/45741055726/job/139645873722) | `python -m pytest -q -o addopts=-ra --confcutdir tests/runtime tests/runtime/` | 32 passed in 0.33 s |
| [test, Python 3.12](https://github.com/cwchanap/Crux/actions/runs/45741055726/job/139645874023) | `python -m pytest -q --cov=src --cov-report=xml:coverage.xml --cov-report=term-missing` | 3,835 passed in 369.13 s |

The existing runtime regression is in both suites. Its real local Git/ZIP operations cover committed/staged/unstaged source extraction, manifest equality, fixed timestamp, byte equality and tamper rejection. This is sufficient automated end-to-end acceptance for the diagnosed test defect; separate local runs would duplicate green CI on the identical code. The round-two reviewer also ran the scratch recipe against the old and fixed revisions and reports the expected historical assertion failure and current pass.

## Disposition

Technical acceptance is complete. The existing fix and guard are sufficient. This PR documents the diagnosis and evidence; it changes no production/test code. No new test, retry, sleep, serialization, load matrix, dependencies, CI jobs, model inference, upstream build, R2 work or visual acceptance is required. The missing original traceback remains a caveat, not a reason to keep revalidating the reproduced timestamp defect.

The remaining repository workflow is normal documentation PR CI and merge, then close HPA-828 with the recorded evidence. If unrelated failures appear, investigate their actual symptoms rather than reopen the unsupported index-lock hypothesis.
