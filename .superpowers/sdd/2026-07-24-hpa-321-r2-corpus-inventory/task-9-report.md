# Task 9 Report: R2 Corpus Sync CLI

## Implemented

- Registered `crux benchmark sync-r2-corpus` through the existing package entrypoint.
- Added only local command options: cache/output directories, repeatable include/exclude IDs,
  optional provenance, and dry-run. Endpoint, credential, profile, retry, concurrency, timeout,
  and maximum-ID flags are intentionally absent.
- Preserved the optional dependency boundary by importing the boto3-free orchestrator module at
  CLI module scope; no live credentialed R2 access was made.
- Maps `SyncOutcome` process results explicitly through `ctx.exit`: complete/dry-run-complete `0`,
  partial/dry-run-partial `1`, and fatal `2`. Click's own option parsing errors remain standard
  usage exit `2`.
- Emits only a sorted one-line JSON final summary on stdout, with status, exit code, optional
  corpus version/report path, manifest publication state, and all outcome counters. Progress stays
  on stderr; a fatal result without a report produces exactly one sanitized stderr fallback line.
- Added fake-backed CLI integration tests for help discoverability, bounds, request construction,
  defaults, provenance forwarding, repeatable filters, explicit exits, separated streams,
  counter-bearing summaries, dry-run non-publication, and endpoint/credential redaction.
- Added the operator reference covering optional installation, environment-only configuration,
  R2 authority, defaults, filtering, cache/resume behavior, dry-run mutation limits, artifacts and
  pointers, status/exit contract, provenance identity, HPA-322 scope, and the credentialed smoke
  acceptance requirement.

## TDD Evidence

- The initial focused CLI RED run had 11 expected failures: the command was undiscoverable and the
  directly monkeypatchable orchestrator import did not exist.
- The minimal command registration, request mapping, JSON summary, stderr progress, and explicit
  exits made the same focused suite pass without touching orchestration or R2 adapter code.

## Verification

- `rtk uv run --extra r2 pytest tests/test_cli_benchmark.py -q` — 23 passed.
- `rtk uv run ruff check src/cli/benchmark.py tests/test_cli_benchmark.py` — passed.
- `rtk uv run black --check src/cli/benchmark.py tests/test_cli_benchmark.py` — passed.
- `rtk uv run --extra r2 crux benchmark sync-r2-corpus --help` — command registered with only the
  documented local options.
- `rtk rg -n "sync-r2-corpus|latest-report|overall_status|setdef_dtx_txt_v1|HPA-322"
  docs/drumery-dtx-midi-benchmarking-reference.md` — required documentation boundaries present.
- `rtk git diff --check` — passed.
- `rtk uv run --extra r2 pytest -q` — 522 passed.

## Notes

- No live credentialed R2 smoke run was performed. The approved design reserves it for acceptance;
  this task uses injected fake outcomes and the existing fake-backed synchronization suite.

## Commit

- Planned conventional commit: `feat: expose R2 corpus sync command`.
