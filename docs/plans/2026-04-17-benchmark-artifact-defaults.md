# Benchmark Artifact Defaults Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Default benchmark CLI artifacts to `artifacts/benchmark/<run-name-or-input-dir-name>/`
when `--output-dir` is omitted, while keeping `--output-dir` as an override.

**Architecture:** Keep output path resolution in the CLI layer. Add a small shared resolver in
`src/cli/options.py`, make benchmark commands accept optional `--output-dir` plus optional
`--run-name`, and continue passing a concrete `Path` into the existing benchmark services.

**Tech Stack:** Python 3.12, Click, pytest, Ruff, Black, Pylint

---

### Task 1: Add failing CLI tests for default artifact resolution

**Files:**
- Modify: `tests/test_cli_benchmark.py`

**Step 1: Write the failing test**

Add CLI tests covering:
- `prepare-corpus` without `--output-dir` writes into `artifacts/benchmark/<raw-dir-name>/`
- `score-midi` without `--output-dir` but with `--run-name` writes into
  `artifacts/benchmark/<run-name>/`

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_cli_benchmark.py::test_prepare_corpus_defaults_output_dir_from_raw_dir_name tests/test_cli_benchmark.py::test_score_midi_defaults_output_dir_from_run_name -v`
Expected: FAIL because `--output-dir` is currently required and no default resolver exists

**Step 3: Commit**

Do not commit yet. Continue to Task 2 after confirming the red state.

### Task 2: Implement shared benchmark output resolution

**Files:**
- Modify: `src/cli/options.py`
- Modify: `src/cli/benchmark.py`

**Step 1: Write minimal implementation**

Add:
- optional `--run-name`
- optional `--output-dir`
- helper that resolves `artifacts/benchmark/<name>/` from a provided source directory basename
  unless `--output-dir` is given

Update benchmark commands that write outputs to use the resolver.

**Step 2: Run tests to verify they pass**

Run: `uv run pytest tests/test_cli_benchmark.py::test_prepare_corpus_defaults_output_dir_from_raw_dir_name tests/test_cli_benchmark.py::test_score_midi_defaults_output_dir_from_run_name -v`
Expected: PASS

### Task 3: Ignore repo-local artifacts and verify explicit override still works

**Files:**
- Modify: `.gitignore`
- Modify: `tests/test_cli_benchmark.py`

**Step 1: Write the failing test**

Add a CLI test showing explicit `--output-dir` still overrides the default artifact path.

**Step 2: Run test to verify it fails if needed**

Run: `uv run pytest tests/test_cli_benchmark.py::test_prepare_corpus_explicit_output_dir_overrides_default -v`
Expected: PASS immediately if existing explicit behavior is preserved; if so, keep the test as a
regression guard and move on.

**Step 3: Write minimal implementation**

Add `artifacts/` to `.gitignore`.

**Step 4: Run focused verification**

Run:
- `uv run pytest tests/test_cli_benchmark.py -v`
- `uv run ruff check src/cli/options.py src/cli/benchmark.py tests/test_cli_benchmark.py`
- `uv run black --check src/cli/options.py src/cli/benchmark.py tests/test_cli_benchmark.py`
- `uv run pylint --errors-only src/cli/options.py src/cli/benchmark.py`

Expected: all pass

### Task 4: Verify repo-level CLI behavior and commit

**Files:**
- Modify: none

**Step 1: Run final verification**

Run:
- `uv run pytest tests/test_cli_main.py tests/test_cli_benchmark.py -v`
- `git status --short`

Expected: tests pass and only intended files are modified

**Step 2: Commit**

```bash
git add .gitignore src/cli/options.py src/cli/benchmark.py tests/test_cli_benchmark.py docs/plans/2026-04-17-benchmark-artifact-defaults-design.md docs/plans/2026-04-17-benchmark-artifact-defaults.md
git commit -m "feat: default benchmark artifacts to repo outputs"
```
