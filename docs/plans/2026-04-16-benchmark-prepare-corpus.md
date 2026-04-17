# Benchmark Prepare-Corpus Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add a preparation command that converts raw DTX song folders into a parsed benchmark corpus with one highest-priority chart and one allowed drum stem per song.

**Architecture:** Add a new preparation service under `src/benchmark` that scans raw song folders, applies deterministic chart/audio selection rules, copies selected files into `charts/` and `audio/` output directories, and writes `manifest.json` plus `invalid.json`. Wire it into the benchmark Click CLI as `prepare-corpus` without changing the existing parsed-corpus benchmark flow.

**Tech Stack:** Python 3.12, Click, pathlib, shutil, json, pytest.

---

### Task 1: Preparation Service Data Model And Selection Rules

**Files:**
- Create: `src/benchmark/prepare.py`
- Test: `tests/benchmark/test_prepare.py`

**Step 1: Write failing tests**

Cover:
- selecting `mas.dtx` over `ext/adv/bas`
- selecting `2 Drums.mp3` or `drum.mp3`
- rejecting folders with no allowed drum audio
- rejecting folders with both allowed drum filenames present

**Step 2: Run tests to verify failure**

Run: `uv run pytest tests/benchmark/test_prepare.py -v`

**Step 3: Implement minimal preparation logic**

Add:
- `DRUM_AUDIO_FILENAMES`
- chart priority constant
- raw-folder scan helpers
- validation model for valid and invalid items

**Step 4: Run tests to verify pass**

Run: `uv run pytest tests/benchmark/test_prepare.py -v`

**Step 5: Commit**

```bash
git add src/benchmark/prepare.py tests/benchmark/test_prepare.py
git commit -m "feat: add benchmark corpus preparation rules"
```

### Task 2: Parsed Corpus Writer

**Files:**
- Modify: `src/benchmark/prepare.py`
- Modify: `tests/benchmark/test_prepare.py`

**Step 1: Write failing tests**

Cover:
- copying selected chart into `output/charts/<song_id>.dtx`
- copying selected audio into `output/audio/<song_id>.<ext>`
- writing `manifest.json`
- writing `invalid.json`

**Step 2: Run tests to verify failure**

Run: `uv run pytest tests/benchmark/test_prepare.py -v`

**Step 3: Implement writer**

Add a function like `prepare_corpus(raw_dir, output_dir)` that:
- scans raw folders
- copies valid selections
- writes manifest/invalid reports
- returns a summary

**Step 4: Run tests to verify pass**

Run: `uv run pytest tests/benchmark/test_prepare.py -v`

**Step 5: Commit**

```bash
git add src/benchmark/prepare.py tests/benchmark/test_prepare.py
git commit -m "feat: write parsed benchmark corpus"
```

### Task 3: CLI Command

**Files:**
- Modify: `src/cli/options.py`
- Modify: `src/cli/benchmark.py`
- Test: `tests/test_cli_benchmark.py`

**Step 1: Write failing tests**

Add a CLI test for:

```text
crux benchmark prepare-corpus --raw-dir <raw_root> --output-dir <parsed_root>
```

Assert:
- exit code `0`
- parsed chart/audio files exist
- manifest exists

**Step 2: Run tests to verify failure**

Run: `uv run pytest tests/test_cli_benchmark.py -v`

**Step 3: Implement CLI**

Add:
- `raw_dir_option` to `src/cli/options.py`
- `prepare-corpus` command to `src/cli/benchmark.py`

**Step 4: Run tests to verify pass**

Run: `uv run pytest tests/test_cli_benchmark.py -v`

**Step 5: Commit**

```bash
git add src/cli/options.py src/cli/benchmark.py tests/test_cli_benchmark.py
git commit -m "feat: add benchmark prepare-corpus command"
```

### Task 4: Final Verification

**Files:**
- Modify as needed based on test or lint feedback.

**Step 1: Run focused preparation and CLI tests**

Run:

```bash
uv run pytest tests/benchmark/test_prepare.py tests/test_cli_benchmark.py -v
```

**Step 2: Run benchmark-focused integration tests**

Run:

```bash
uv run pytest tests/benchmark tests/test_cli_main.py tests/test_cli_benchmark.py -v
```

**Step 3: Run repo-standard lint checks**

Run:

```bash
uv run ruff check src tests
uv run black --check src tests
uv run pylint --errors-only src/app src/cli src/benchmark
```

**Step 4: Manual smoke test**

Run `prepare-corpus` against a real raw folder root and confirm:
- parsed chart/audio files are created
- manifest and invalid reports exist
- later benchmark commands can operate on the parsed output
