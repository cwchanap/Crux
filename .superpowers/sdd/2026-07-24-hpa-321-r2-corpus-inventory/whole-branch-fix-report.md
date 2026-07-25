# HPA-321 Whole-Branch Fix Report

## Outcome

Addressed the four Important findings from the final whole-branch review:

1. Provenance loading now rejects missing, unreadable, undecodable, and non-strict-UTF-8
   documents with sanitized provenance-invalid behavior before store creation, network access,
   or cache mutation.
2. Cache-index loading now distinguishes an absent index from a broken symlink, rejects every
   non-regular leaf, reads through an `O_NOFOLLOW` descriptor, and verifies the descriptor/path
   device and inode binding before and after the read.
3. Provider timestamps must be timezone-aware at list, HEAD, and download adapter boundaries;
   shared manifest timestamp formatting also rejects naive values.
4. Download preparation and installation now run in batches no larger than
   `download_concurrency`. Known repair-authorized candidates run before unrelated misses, while
   final object/action results are reconstructed in original corpus order.

The two explicitly deferred Minor findings were not changed:

- mutable `ambiguous_prefixes`
- corpus-level HEAD executor

## RED Evidence

All regressions were run against the pre-fix implementation before production edits:

- Provenance loader/orchestration selection:
  - command: `rtk uv run --extra r2 pytest -q tests/benchmark/test_corpus_provenance.py tests/benchmark/test_r2_corpus_sync.py -k 'strict_utf8 or missing_unreadable or invalid_provenance_fails_before_store'`
  - result: 7 failed
  - observed failures included accepted escaped lone surrogates, raw `FileNotFoundError`, store
    creation for invalid provenance, and `internal_error` instead of `provenance_invalid`.
- Cache-index adversarial selection:
  - command: `rtk uv run --extra r2 pytest -q tests/benchmark/test_corpus_cache.py -k 'absent_index or non_regular_leaf or nofollow_descriptor'`
  - result: 5 failed
  - observed failures included treating a broken symlink as absent, following a valid symlink,
    using `Path.read_text`, and no descriptor/path replacement detection.
- Timezone selection:
  - command: `rtk uv run --extra r2 pytest -q tests/benchmark/test_r2_inventory.py tests/benchmark/test_r2_corpus_models.py -k 'naive_datetimes or rejects_naive_values'`
  - result: 4 failed
  - list, HEAD, download, and formatter all accepted a naive datetime.
- Bounded-staging and repair-order selection:
  - command: `rtk uv run --extra r2 pytest -q tests/benchmark/test_corpus_cache.py -k 'bounds_owned_staging or repair_candidate_runs_before'`
  - result: 2 failed
  - the large-corpus probe retained 37 staged descriptors at concurrency 3, and the repair
    candidate completed after unrelated misses.

## GREEN Evidence

Focused regression reruns after each minimal fix:

- provenance selection: 7 passed
- cache-index adversarial selection: 5 passed
- timezone selection: 4 passed
- bounded-staging and cross-batch repair selection: 2 passed

Requested broader subsystem verification:

```text
rtk uv run --extra r2 pytest -q \
  tests/benchmark/test_corpus_provenance.py \
  tests/benchmark/test_r2_inventory.py \
  tests/benchmark/test_corpus_cache.py \
  tests/benchmark/test_r2_corpus_sync.py \
  tests/benchmark/test_corpus_manifest.py \
  tests/benchmark/test_r2_corpus_models.py
```

Result: 323 passed.

Changed-file quality gates:

```text
rtk uv run ruff check <all 9 changed Python files>
rtk uv run black --check <all 9 changed Python files>
rtk git diff --check
```

Results:

- Ruff: all checks passed
- Black: 9 files unchanged
- diff check: passed
- commit hooks: Ruff, Ruff format, and Pylint errors-only passed

Full requested suite:

```text
rtk uv run --extra r2 pytest -q
```

Result: 547 passed in 16.41 seconds.

The formatter-stabilized provenance test file was rerun after the full suite:

```text
rtk uv run --extra r2 pytest -q tests/benchmark/test_corpus_provenance.py
```

Result: 40 passed.

## Commits

- `a7596877de744d08167cc19cf78d8ff5b4f3ded1` —
  `fix: harden R2 corpus synchronization`
- This report is committed in the immediate documentation follow-up commit.

## External-System Confirmation

Live R2 was not contacted. All verification used in-process fake or mocked object-store clients;
no live credentials, bucket validation, listing, HEAD, or download request was issued.
