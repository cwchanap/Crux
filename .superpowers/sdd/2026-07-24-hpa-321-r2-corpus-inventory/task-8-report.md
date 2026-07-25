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
- Added a durable POSIX output-publication lock shared by real and dry runs. It uses
  no-follow descriptor validation and serializes the immutable real manifest, report,
  `latest.json`, and `latest-report.json` as one process-safe publication unit while
  the real-run cache writer lock remains held.
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
  Both pointers are snapshotted with no-follow descriptor reads and durably restored
  together after replacement or directory-fsync failures, so a failed outcome cannot
  leave a success pointer behind. Successful real and dry processes retain
  last-completed-writer pointer semantics.
- If either pointer publication fails after the timestamped success or partial report
  is durable, both pointers are restored first and that same attempt report is
  atomically rewritten as canonical `failed`/exit-2 output with the sanitized
  publication error. A failed rewrite durably removes the stale success report and
  never moves `latest-report.json`.
- Added bounded callback-only progress at deterministic phase boundaries and every
  100 items, final item, or five monotonic seconds. Inventory HEAD and cache-download
  hooks now feed the thread-safe tracker as futures complete instead of replaying
  progress after work. Messages contain only phases, counts, and byte totals.
- Progress callbacks are non-authoritative observability. The tracker catches a
  callback exception at phase, live-item, or final-outcome delivery, permanently
  disables further callbacks for that attempt, and preserves the synchronization's
  real result without exposing callback details.
- Added lexical no-follow validation for existing output and cache path components
  before store creation, bucket validation, network access, cache mutation, or report
  publication. Broken symlinks, symlinked ancestors, non-directory components, and
  invalid leaves fail as `invalid_config` with zero local writes.
- Contained invalid or raising wall-clock and run-ID factories as `internal_error`.
  Fatal reports use trusted stdlib UTC/UUID4 fallbacks, and retain the primary internal
  error if fallback report publication also fails.
- Added a mutation-free platform preflight for `O_NOFOLLOW`, `O_DIRECTORY`, and the
  required `fcntl` locking interface. Unsupported platforms fail before store
  creation, network activity, output/cache creation, or fatal-report publication.
  Layered fatal fallbacks deduplicate identical artifact error codes.
- Added a reusable boto3-free fake store with exact list/HEAD/body state, configured
  operation failures, recorded calls, and fixed-size streamed body chunks.

## Files

- Updated `src/benchmark/r2_corpus_sync.py`,
  `src/benchmark/r2_inventory.py`, and `src/benchmark/corpus_cache.py`.
- Updated `tests/benchmark/test_r2_corpus_sync.py`,
  `tests/benchmark/test_r2_inventory.py`, and
  `tests/benchmark/test_corpus_cache.py`.
- Updated this Task 8 report.

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
- Review-fix RED runs demonstrated three dual-pointer rollback failures, including a
  replacement followed by directory-fsync failure, and two independent processes
  publishing concurrently without serialization.
- Live-progress RED tests held one HEAD or download future blocked after another
  completed; the callbacks were initially absent or replayed only after the blocked
  work finished. The inventory and cache hooks now report each completed future live.
- Path/factory RED tests covered symlinked and broken cache/output components,
  non-directory leaves, invalid/raising clocks, and invalid/raising run-ID factories.
- Round-2 RED tests found four stale-report outcomes after pointer
  replacement/durability failures. The fix restores both pointers and replaces the
  only timestamped attempt report with failed/exit-2 content, or removes it when the
  fatal rewrite fails.
- Three callback RED tests showed phase exceptions escaping, live-item exceptions
  changing a dry success into failure, and fatal-path exceptions replacing the
  primary error. A fourth isolated RED test covered the final `failed` event.
- Platform RED tests showed missing POSIX flags becoming internal errors after local
  work, an incompatible `fcntl` surface being ignored, and duplicate
  `artifact_write_failed` outcomes. Preflight and same-code fallback deduplication
  now contain those cases.
- The final R2-focused run covers 188 inventory, cache, and orchestration behaviors;
  the subsequent publication subset covers eight dual-pointer, process-lock,
  no-follow, and unsupported-platform behaviors.

## Verification

- `rtk uv run pytest tests/benchmark/test_r2_inventory.py
  tests/benchmark/test_corpus_cache.py tests/benchmark/test_r2_corpus_sync.py -q`
  — 188 passed.
- `rtk uv run pytest tests/benchmark/test_r2_corpus_sync.py -k
  'output_publication_lock or output_lock_requires or two_process_ or
  latest_report_failure_restores or latest_manifest_failure_restores or
  latest_report_directory_fsync_failure' -q` — 8 passed.
- `rtk uv run ruff check` on all six modified source/test files — passed.
- `rtk uv run black --check` on all six modified source/test files — passed.
- `rtk git diff --check` — passed before the full-suite gate.
- `rtk uv run --extra r2 pytest -q` — 504 passed.
- Round-2 `rtk uv run pytest tests/benchmark/test_r2_corpus_sync.py -q` —
  61 passed.
- Round-2 `rtk uv run ruff check` and `rtk uv run black --check` on
  `src/benchmark/r2_corpus_sync.py` and
  `tests/benchmark/test_r2_corpus_sync.py` — passed.
- Round-2 `rtk git diff --check` — passed before the full-suite gate.
- Round-2 `rtk uv run --extra r2 pytest -q` — 510 passed.

## Notes

- No live credentialed R2 request was made. Task 8 is verified with the injected
  object-store protocol; the approved plan reserves live acceptance for the later
  end-to-end task.
- Immutable manifest content installed before a later report failure is deliberately
  retained as an unreferenced reusable artifact and omitted from `SyncOutcome`, while
  `latest.json` remains unchanged.
- The output lock file is a persistent regular file named
  `.r2-corpus-publication.lock`; persistent identity avoids rename/unlink lock races.

## Commit

- Planned conventional commit: `fix: contain R2 sync failure paths`.
