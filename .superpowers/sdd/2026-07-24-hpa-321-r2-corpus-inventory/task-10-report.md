# Task 10 Report: Deterministic R2 Corpus Acceptance

## Scope

Implemented only Task 10 acceptance coverage and final operator documentation:

- added an in-memory paginated corpus fixture covering nested markers, exact Unicode
  keys, marker-only simfile `3`, quarantined aliases `4/` and `04/`, a non-selected
  audio object, and four selected chart-definition bodies;
- proved cold-run/warm-run repeatability and one-object changed-history behavior
  through the public `sync_r2_corpus` entry point;
- independently recomputed `corpus_version` and the manifest SHA-256 from exact
  manifest bytes with local standard-library JSON/SHA-256 code, without production
  canonicalization helpers or constants;
- covered partial status, canonical cache-index bytes, extensionless
  content-addressed bodies, immutable historical manifests/cache bodies, provenance
  and error identity changes, report/pointer semantics, endpoint/credential
  redaction, and the base-install lazy-import boundary;
- documented the cooperative-writer local-cache threat boundary resolved by the
  Task 6 controller.

No production source behavior changed.

## TDD and Focused Verification

The acceptance behavior already existed at the Task 10 base, so the initial test run
was green. A deliberate mutation removed `.txt` selection to prove that the new
acceptance test detects the relevant regression, then the production line was
restored exactly.

- `rtk uv run --extra r2 pytest tests/benchmark/test_r2_corpus_acceptance.py -q`
  - the first sandboxed attempt could not read the existing uv cache:
    `failed to open .../.cache/uv/sdists-v9/.git: Operation not permitted`;
  - the same command rerun with existing uv-cache access completed the initial
    existing-behavior check: `3 passed`;
- deliberate mutation check using the same command:
  - expected red: `1 failed, 2 passed`;
  - failure was the missing `1/音源/readme.TXT` GET;
- restored mutation, same command:
  - green: `3 passed`;
- `rtk uv run ruff check tests/benchmark/test_r2_corpus_acceptance.py`
  - passed;
- `rtk uv run black --check tests/benchmark/test_r2_corpus_acceptance.py`
  - requested reformatting;
- `rtk uv run black tests/benchmark/test_r2_corpus_acceptance.py`
  - reformatted the new test;
- `rtk uv run --extra r2 pytest tests/benchmark/test_r2_corpus_models.py tests/benchmark/test_r2_inventory.py tests/benchmark/test_corpus_provenance.py tests/benchmark/test_corpus_cache.py tests/benchmark/test_corpus_manifest.py tests/benchmark/test_r2_corpus_sync.py tests/benchmark/test_r2_corpus_acceptance.py tests/test_cli_benchmark.py -q`
  - passed: `335 passed in 4.56s`.

## Final Repository Quality Gate

Each final command below was run exactly once after focused green:

- `rtk uv run --extra r2 pytest`
  - passed: `529 passed in 16.08s`;
- `rtk uv run ruff check src tests`
  - passed: `All checks passed!`;
- `rtk uv run black --check src tests`
  - exit `1`;
  - only pre-existing, unchanged files were reported:
    `tests/benchmark/test_runner.py` and
    `tests/benchmark/test_render_audio.py`;
  - `61 files would be left unchanged`;
- `rtk uv run --extra r2 pylint src/app src/cli src/benchmark`
  - exit `28`, rating `9.60/10`;
  - reported the existing repository Pylint backlog across unchanged application,
    CLI, and benchmark modules;
  - the new acceptance test is outside the command's source-only targets, and Task
    10 changed no source module;
- `rtk uv run --extra r2 crux benchmark sync-r2-corpus --help`
  - exit `0`;
  - rendered only the documented local options and no endpoint, credential, cache
    profile, concurrency, timeout, or retry flags.

Per the brief, the unrelated Black/Pylint backlog was recorded without broadening
Task 10 into cleanup of existing files.

## Credentialed R2 Acceptance Gate

Checked prerequisite presence without printing values:

```text
rtk uv run python -c '<presence-only environment check>'
all_smoke_inputs_present=False
```

The credentialed smoke sequence was not run because
`CRUX_R2_ENDPOINT_URL`, `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, and
`CRUX_R2_SMOKE_ID` were not all present. No live-R2 result, ETag observation, or
full-corpus dry-run evidence is claimed. The deterministic fake-store acceptance
proof is complete locally; the live smoke remains a pending manual acceptance gate.

## Operator Documentation

The R2 inventory section now explicitly records:

- optional `r2` installation and its transitive `boto3`, `botocore`, `s3transfer`,
  `jmespath`, and `urllib3` footprint;
- placeholder-only credential examples;
- durable per-body checkpoints and resume-by-rerun semantics;
- the cooperating-writer advisory-lock boundary and hostile same-UID rebinding
  exclusion from HPA-321's threat model;
- HPA-322 ownership of chart parsing/selection, audio selection, inference, and
  scoring;
- the pending credentialed smoke requirement.
