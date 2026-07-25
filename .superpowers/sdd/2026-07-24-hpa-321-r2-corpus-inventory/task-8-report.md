# Task 8 Report: R2 Corpus Synchronization Orchestration

## Implemented

- Added the public `sync_r2_corpus` transaction boundary with explicit dependency,
  store, wall-clock, monotonic-clock, run-ID, environment, and progress injection.
- Preserved the required fail-before-network order for the optional dependency,
  configuration and local-path validation, provenance validation, and cache-index
  validation.
- Added a real-run whole-transaction writer lock from cache-index load through
  immutable manifest, durable report, and both pointer publications. Dry runs use a
  read-only index snapshot without taking the writer lock, reading bodies, or mutating
  cache or manifest state.
- Added deterministic complete, partial, failed, `dry_run_complete`, and
  `dry_run_partial` status/exit mapping. Empty, malformed, ambiguous, row-failure, and
  operation-failure conditions remain visible and produce partial artifacts where
  allowed.
- Integrated inventory, cache synchronization, provenance, manifest rendering, and
  immutable publication without changing their existing contracts.
- Added canonical per-attempt reports with exact fixed counters, all cache-miss
  reasons, sorted filters/rows/actions/errors, safe source identity, sanitized
  allowlisted failures, and nullable manifest fields for dry/fatal attempts.
- Added canonical UUID4 report names with fixed six-digit UTC timestamp precision,
  atomic sibling-temporary replacement, file and directory `fsync`, and an atomic
  `latest-report.json` pointer containing the report digest.
- Kept `latest.json` deferred until the immutable manifest and report are durable.
  Fatal report or manifest-pointer failures restore the previous pointer bytes, while
  successful real and dry writers retain last-completed-writer pointer semantics.
- Added bounded callback-only progress at deterministic phase boundaries and every
  100 items, final item, or five monotonic seconds. Messages contain only phases,
  counts, and byte totals.
- Added a reusable boto3-free fake store with exact list/HEAD/body state, configured
  operation failures, recorded calls, and fixed-size streamed body chunks.

## Files

- Added `src/benchmark/r2_corpus_sync.py`.
- Added `tests/benchmark/test_r2_corpus_sync.py`.
- Added this Task 8 report.

## TDD Evidence

- The initial focused RED run failed during collection because
  `src.benchmark.r2_corpus_sync` did not exist.
- The first implementation reached 21 focused passing tests for the core
  complete/partial/fatal/dry-run workflows.
- The counter mutation probe then failed because a selected object with a HEAD failure
  incorrectly incremented `downloads_failed`; the minimal fix restricts failed
  download counts to actions that had an actual cache-miss download plan.
- The fatal latest-report regression failed because a pointer error after manifest
  publication left `latest.json` moved; the fix snapshots and durably restores the
  prior pointer before returning the sanitized report fallback.
- A second fatal publication regression failed after a simulated
  `publish_latest_manifest` replacement-then-error; the fix restores the prior pointer
  before publishing the fatal report.
- The final focused suite covers 38 complete, partial, fatal, dry-run, validation,
  pagination, locking, counters, cache-miss, report naming, durability, redaction,
  progress, concurrent-writer, pointer-rollback, and repeatability behaviors.

## Verification

- `rtk uv run --extra r2 pytest tests/benchmark/test_r2_corpus_sync.py -q` —
  38 passed.
- `rtk uv run ruff check src/benchmark/r2_corpus_sync.py
  tests/benchmark/test_r2_corpus_sync.py` — passed.
- `rtk uv run black --check src/benchmark/r2_corpus_sync.py
  tests/benchmark/test_r2_corpus_sync.py` — passed; both files unchanged.
- `rtk uv run --extra r2 pytest -q` — 485 passed.
- `rtk git diff --check` — passed before the full-suite gate.

## Notes

- No live credentialed R2 request was made. Task 8 is verified with the injected
  object-store protocol; the approved plan reserves live acceptance for the later
  end-to-end task.
- Immutable manifest content installed before a later report failure is deliberately
  retained as an unreferenced reusable artifact and omitted from `SyncOutcome`, while
  `latest.json` remains unchanged.

## Commit

- Planned conventional commit: `feat: orchestrate R2 corpus synchronization`.
