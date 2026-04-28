# DTX MIDI Benchmark — PR Review Fixes Design

**Date:** 2026-04-27
**Branch:** `feature/dtx-midi-benchmark`
**PR:** #2

## Goal

Address all critical and important issues identified in the comprehensive PR review. Deliver in two commits: (1) behavioral correctness fixes, (2) test coverage gaps and documentation corrections.

---

## Commit 1: Behavioral Fixes

### `src/app/transcriber.py`

**CRITICAL-1 — `_run_tf2_model_inference` swallows all exceptions**
- Remove the `except Exception` fallback in `_run_tf2_model_inference`. Let exceptions propagate so the benchmark runner can catch and skip the chart with a proper error log.
- The fallback onset detector remains available but must only be invoked by an explicit caller decision, not silently on any error.

**CRITICAL-3 — `_build_model` broad catch returns None silently**
- Narrow `_build_model`'s `except Exception` to known recoverable TF errors (`OSError`, `ValueError`). All other exceptions propagate.
- In `__init__`, if `_build_model` raises, log at ERROR level and re-raise rather than silently setting `self.model = None`.

**HIGH-4 — Division by zero on silent audio**
- Before `(log_mel - log_mel.mean()) / log_mel.std()`, check `std < 1e-8`. If true, raise `ValueError` with a clear message so the benchmark runner records it as a skipped chart.

**HIGH-6 — "Dummy model for testing" in production path**
- Remove the misleading comment `# For testing, create a dummy model` from `_download_model`.
- `_download_model` returning `None` when all URLs fail now causes `_build_model` to return `None` through the `not self.model_path` early exit (which already logs a warning). No separate dummy path needed.

### `src/benchmark/runner.py`

**CRITICAL-2 — `transcribe()` call aborts entire run**
- Wrap `midi_bytes = transcribe(audio_path)` in try/except. On exception: log ERROR with chart id and exception, append to a `failed_charts` list, continue loop.
- After the loop, log a summary warning if `failed_charts` is non-empty.

**HIGH-2/HIGH-3 — No per-chart recovery in `export_reference_midis` and `run_score_midi`**
- `export_reference_midis`: wrap per-chart ops in try/except. On exception: log ERROR with `dtx_path`, increment a `failed` counter, continue. Return type stays `int` (success count); log a summary warning for failures at the end (no interface change, CLI unaffected).
- `run_score_midi`: wrap per-chart ops in try/except. On exception: log ERROR with `item.chart_id`, append to `failed_charts`, continue. Always call `write_reports(reports, output_dir)` with whatever partial results exist before re-raising or returning.

### `src/benchmark/timing.py`

**Code-reviewer issue 6 — Duplicate BPM at same beat raises ValueError**
- Replace the `raise ValueError` for duplicate-beat tempo events with last-wins deduplication: overwrite `resolved[-1]` and append to `chart.warnings`.
- This is already the pattern for beat-zero BPM overrides; extend it to all beats.

### `src/benchmark/scoring.py`

**Code-reviewer issue 2 — Metadata aliasing in `score_events`**
- When constructing adjusted prediction events with offset applied, use `dataclasses.replace(event, metadata=dict(event.metadata))` to avoid sharing the metadata dict reference.

### `src/benchmark/midi_io.py`

**Code-reviewer issue 3 / MEDIUM-5 — Silent empty drum track**
- After iterating instruments, if `events` is empty and `midi.instruments` is non-empty, log `WARNING` naming the non-drum instruments found.

### `src/benchmark/prepare.py`

**MEDIUM-3 — Unguarded `shutil.copy2`**
- Wrap both `shutil.copy2` calls in try/except `OSError`. On failure: append an `InvalidPreparedCorpusItem` with `reason="failed to copy corpus files"` and `details={"exception": str(exc)}`, then `continue` without appending to `manifest_entries`.

### `src/benchmark/render_audio.py`

**MEDIUM-6 — Cache populated before existence check**
- In `_render_placements`, check `if placement.sample_path not in cached_samples` before calling `_load_sample`. Only load and cache on a miss.

**MEDIUM-4 — Sample-rate mismatch error drops context**
- Enrich the `ValueError` message to include the expected rate, the offending filename, and the mismatched rate.

**MEDIUM-2 — Path traversal blocked silently**
- Add `logger.warning(...)` with `song_dir_resolved` and `sample_path` before returning `None` on traversal detection.

**HIGH-1 — Unreadable sample file indistinguishable from absent**
- In `_resolve_sample_path`'s `except` block for `sf.info`, add `logger.debug(...)` with `sample_path` and the exception.

### `src/benchmark/reports.py`

**MEDIUM-7 — Empty reports list writes 3-column CSV**
- Define a module-level `_REPORT_FIELDNAMES` constant derived from a sentinel `_report_row` call.
- Use `_REPORT_FIELDNAMES` as the authoritative `fieldnames` argument in `_write_per_chart_csv`, regardless of whether `rows` is empty.

### `src/benchmark/dtx_parser.py`

**MEDIUM-1 — Bare `float()` on VOLUME/POSITION headers**
- Wrap `float(value)` in try/except `ValueError` for both `VOLUME` and `POSITION` header parsing.
- On failure: append a warning string to `chart.warnings` and skip the entry (consistent with BPM table warning pattern).

### `src/cli/benchmark.py`

Fix 4 help strings:
- `inspect-dtx`: `"""Print parsed DTX chart metadata: event count, BPM, measure changes, and lane list."""`
- `validate-corpus`: `"""Check DTX charts and prediction MIDIs for missing files, stray files, and duplicate stems."""`
- `export-reference-midi`: `"""Export MIDI files derived from DTX charts for manual inspection (not used for scoring)."""`
- `--align/--no-align`: describe that the flag controls whether the global offset is computed, applied, and emitted as a separate aligned report row.

---

## Commit 2: Tests + Documentation

### New Tests

**`tests/benchmark/test_scoring.py`**

1. `test_alignment_empty_ground_truth` — call `score_events_with_alignment([], [pred], tol)`, assert `result.raw.summary.offset_sec == 0.0` and `result.aligned.summary.offset_sec == 0.0`.
2. `test_alignment_no_shared_classes` — GT has only `"kick"` events, predictions have only `"snare"` events. Assert offset is `0.0` (no shared class → skip alignment).

**`tests/benchmark/test_render_audio.py`**

3. `test_render_placements_sample_rate_mismatch` — construct two placements with different sample rates via monkeypatching `_load_sample`. Assert the song appears in the `invalid` list of the render plan output with a reason string containing both rates.

**`tests/benchmark/test_dtx_parser.py`**

4. `test_dtx_parser_odd_length_pattern` — pass a DTX text line `#00011: 010` (3-char pattern) to `parse_dtx_text`. Assert `ValueError` is raised.

**`tests/benchmark/test_runner.py`**

5. `test_run_score_midi_no_align` — run `run_score_midi(..., align=False)`. Assert returned reports contain only `"raw"` mode rows (no `"aligned"` rows). Assert report count equals `num_charts × num_tolerances × 1` (not `× 2`).

**`tests/test_cli_benchmark.py`**

6. `test_transcribe_and_score_cli` — invoke the `transcribe-and-score` CLI via Click test runner with a mock `transcribe` callable injected. Assert exit code 0, assert output contains the chart id and a score value.

### Documentation

**`docs/drumery-dtx-midi-benchmarking-reference.md`**

Restructure the "Supported DTX Subset in the Parser" section:

1. Rename to: **"DTX Parsing in the Benchmark Pipeline"**
2. Add subsection: **"Python `dtx_parser.py` (used for scoring)"** — describe what `parse_dtx_text` actually handles: `TITLE`, `ARTIST`, `BPM`, `BPMxx`, `WAVxx`, `VOLUMExx`, `POSITIONxx` headers; channel `02` measure-length changes; channels `03`/`08` BPM events; note chips on any channel. Note that `splitlines()` handles LF, CRLF, and CR. Note that `LINE_RE` accepts both `#` and `*` line prefixes.
3. Rename the existing TypeScript content to: **"Background: Drumery TypeScript Parser (not used for scoring)"** — make clear this describes the web tool's parser for reference context only.
4. Fix `--align/--no-align` option description: "Controls whether a global time-offset is computed and applied. With `--align` (default), both raw and aligned scores are written. With `--no-align`, only raw scores are written."

---

## Acceptance Criteria

- All existing tests continue to pass after Commit 1.
- 6 new tests added in Commit 2 all pass.
- `uv run ruff check src tests` and `uv run ruff format --check src tests` pass on both commits.
- No `except Exception` broad catches remain in the benchmark pipeline's hot paths.
- Documentation no longer references TypeScript parser behavior in the Python parser description section.
