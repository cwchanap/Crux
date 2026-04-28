# DTX MIDI Benchmark Fixes Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix all critical and important correctness issues identified in the PR review of `feature/dtx-midi-benchmark`, delivered in two commits.

**Architecture:** Commit 1 fixes behavioral correctness across 10 source files (no new files). Commit 2 adds 6 new tests across existing test files and rewrites one documentation file. All changes are contained within the existing module boundaries.

**Tech Stack:** Python 3.12, pytest, pretty-midi, numpy, soundfile, click

---

## File Map

**Commit 1 — modified only:**
- `src/app/transcriber.py` — remove broad exception catches, guard std() divide-by-zero
- `src/benchmark/runner.py` — add per-chart error recovery to all three loop functions
- `src/benchmark/timing.py` — replace duplicate-BPM ValueError with last-wins + warning
- `src/benchmark/scoring.py` — copy metadata dict when building adjusted predictions
- `src/benchmark/midi_io.py` — warn when no drum track found in non-empty MIDI
- `src/benchmark/prepare.py` — guard shutil.copy2 with OSError handling
- `src/benchmark/render_audio.py` — fix cache miss, enrich error messages, add traversal log
- `src/benchmark/reports.py` — derive CSV fieldnames from sentinel row, not hardcoded fallback
- `src/benchmark/dtx_parser.py` — guard VOLUME/POSITION float() with try/except
- `src/cli/benchmark.py` — fix 4 help strings

**Commit 2 — modified only:**
- `tests/benchmark/test_scoring.py` — 2 new tests
- `tests/benchmark/test_render_audio.py` — 1 new test
- `tests/benchmark/test_dtx_parser.py` — 1 new test
- `tests/benchmark/test_runner.py` — 1 new test
- `tests/test_cli_benchmark.py` — 1 new test
- `docs/drumery-dtx-midi-benchmarking-reference.md` — restructure parser section, fix 6 factual errors

---

## Task 1: Fix `transcriber.py` — remove broad exception swallowing

**Files:**
- Modify: `src/app/transcriber.py`

This task covers CRITICAL-1 (`_run_tf2_model_inference` swallows all exceptions), CRITICAL-3 (`_build_model` and `__init__` broad catch → None), HIGH-4 (std() divide-by-zero), and HIGH-6 (misleading "dummy model" comment).

- [ ] **Step 1: Remove the broad except in `_run_tf2_model_inference`**

In `src/app/transcriber.py`, find `_run_tf2_model_inference` (around line 497). Replace the entire method body:

```python
def _run_tf2_model_inference(self, audio: np.ndarray, sr: int) -> Dict:
    spec = self._compute_spectrogram_for_model(audio, sr)
    spec_input = spec[np.newaxis, :, :, np.newaxis]
    outputs = self.model(spec_input, training=False)
    return self._process_tf2_model_outputs(outputs, self.MODEL_SAMPLE_RATE)
```

- [ ] **Step 2: Narrow `_build_model` exception catch**

In `_build_model`, the `except Exception` block (around line 244) catches any error and returns None. Change it to only catch `OSError` and `ValueError`, which are the expected file-not-found and weight-format errors. All other exceptions (TF graph errors, shape errors, etc.) should propagate:

```python
        except (OSError, ValueError) as e:  # pylint: disable=broad-except
            logging.error("Failed to build TF2 model: %s", e)
            return None
```

- [ ] **Step 3: Remove broad catch in `__init__`**

In `__init__` (around line 111–115), replace:

```python
            # Build model
            try:
                self.model = self._build_model()
            except Exception as e:  # pylint: disable=broad-except
                logger.warning("Could not load model: %s. Using fallback method.", e)
                self.model = None
```

With:

```python
            # Build model
            self.model = self._build_model()
```

- [ ] **Step 4: Remove misleading "dummy model" comment in `_download_model`**

Around line 200–203, replace:

```python
        logger.error("Failed to download model from any source")
        # For testing, create a dummy model
        logger.warning("Creating dummy model for testing purposes")
        return None
```

With:

```python
        logger.error("Failed to download model from any source")
        return None
```

- [ ] **Step 5: Guard divide-by-zero in `_extract_features`**

In `_extract_features` (around line 537), replace:

```python
        # Normalize
        log_mel = (log_mel - log_mel.mean()) / log_mel.std()
```

With:

```python
        # Normalize
        std = log_mel.std()
        if std < 1e-8:
            raise ValueError(
                f"audio spectrogram has near-zero standard deviation ({std:.2e}); "
                "input may be silent or corrupt"
            )
        log_mel = (log_mel - log_mel.mean()) / std
```

- [ ] **Step 6: Verify existing tests still pass**

Run: `uv run pytest tests/test_transcriber_fallback.py tests/test_transcriber_helpers.py -v`

Expected: all tests PASS. (These tests use `load_model=False` so they bypass model loading.)

---

## Task 2: Fix `runner.py` — per-chart error recovery

**Files:**
- Modify: `src/benchmark/runner.py`

This task covers CRITICAL-2 (`transcribe()` aborts entire run) and HIGH-2/HIGH-3 (no per-chart recovery in `export_reference_midis` and `run_score_midi`).

- [ ] **Step 1: Add per-chart recovery to `run_score_midi`**

Replace the body of `run_score_midi` (lines 34–59) with:

```python
    reports: list[ChartReport] = []
    failed_charts: list[str] = []
    for item in validation.valid_items:
        try:
            chart = parse_dtx_file(item.dtx_path, chart_id=item.chart_id)
            ground_truth, _ = map_dtx_events(dtx_events_to_timed_events(chart))
            predictions, _ = map_midi_events(
                parse_prediction_midi(item.prediction_midi_path, item.chart_id)
            )

            if export_reference_midi:
                write_reference_midi(
                    ground_truth, output_dir / "reference_midi" / f"{item.chart_id}.mid"
                )

            for tolerance in tolerance_ms:
                tolerance_sec = tolerance / 1000
                if align:
                    result = score_events_with_alignment(ground_truth, predictions, tolerance_sec)
                    reports.append(ChartReport(item.chart_id, tolerance, "raw", result.raw.summary))
                    reports.append(
                        ChartReport(item.chart_id, tolerance, "aligned", result.aligned.summary)
                    )
                else:
                    result = score_events(ground_truth, predictions, tolerance_sec)
                    reports.append(ChartReport(item.chart_id, tolerance, "raw", result.summary))
        except Exception:  # pylint: disable=broad-except
            logger.exception("Failed to score chart %r; skipping", item.chart_id)
            failed_charts.append(item.chart_id)

    if failed_charts:
        logger.warning(
            "Scoring failed for %d chart(s): %s", len(failed_charts), ", ".join(failed_charts)
        )
    write_reports(reports, output_dir)
    return reports
```

- [ ] **Step 2: Add per-chart recovery to `export_reference_midis`**

Replace the body of `export_reference_midis` (lines 64–73) with:

```python
    output_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    failed = 0
    for dtx_path in sorted(
        p for p in charts_dir.iterdir() if p.is_file() and p.suffix.lower() in CHART_SUFFIXES
    ):
        try:
            chart = parse_dtx_file(dtx_path, chart_id=dtx_path.stem)
            ground_truth, _ = map_dtx_events(dtx_events_to_timed_events(chart))
            write_reference_midi(ground_truth, output_dir / f"{dtx_path.stem}.mid")
            count += 1
        except Exception:  # pylint: disable=broad-except
            logger.exception("Failed to export reference MIDI for %s; skipping", dtx_path.name)
            failed += 1

    if failed:
        logger.warning("Export failed for %d chart(s)", failed)
    return count
```

- [ ] **Step 3: Add per-chart recovery to `run_transcribe_and_score`**

In `run_transcribe_and_score`, find the transcription loop (around line 100–110). Replace the loop body:

```python
    missing_audio: list[str] = []
    failed_charts: list[str] = []
    for dtx_path in sorted(
        p for p in charts_dir.iterdir() if p.is_file() and p.suffix.lower() in CHART_SUFFIXES
    ):
        try:
            audio_path = _find_audio(audio_dir, dtx_path.stem)
        except FileNotFoundError:
            missing_audio.append(dtx_path.stem)
            continue
        try:
            midi_bytes = transcribe(audio_path)
        except Exception:  # pylint: disable=broad-except
            logger.exception("Transcription failed for chart %r; skipping", dtx_path.stem)
            failed_charts.append(dtx_path.stem)
            continue
        (predictions_dir / f"{dtx_path.stem}.mid").write_bytes(midi_bytes)
        shutil.copy2(dtx_path, matched_charts_dir / dtx_path.name)

    if missing_audio:
        logger.warning(
            "Skipping %d chart(s) with missing audio: %s",
            len(missing_audio),
            ", ".join(missing_audio),
        )
    if failed_charts:
        logger.warning(
            "Transcription failed for %d chart(s): %s",
            len(failed_charts),
            ", ".join(failed_charts),
        )
```

- [ ] **Step 4: Verify runner tests still pass**

Run: `uv run pytest tests/benchmark/test_runner.py -v`

Expected: all existing tests PASS.

---

## Task 3: Fix `timing.py` — duplicate BPM last-wins

**Files:**
- Modify: `src/benchmark/timing.py`

- [ ] **Step 1: Replace the duplicate-beat ValueError with last-wins**

In `_tempo_points`, replace the entire for-loop body (lines 79–92). The current code has a special-case `beat_zero_overridden` flag. Replace with a uniform last-wins rule for all beats:

```python
    resolved: list[tuple[float, float, float]] = []
    current_time = 0.0
    previous_beat = points[0][0]
    previous_bpm = points[0][1]
    for beat, bpm in points:
        if resolved and math.isclose(beat, resolved[-1][0], abs_tol=1e-9):
            chart.warnings.append(f"duplicate tempo at beat {beat:.6f}; using last value")
            resolved[-1] = (resolved[-1][0], resolved[-1][1], bpm)
            previous_bpm = bpm
            continue
        if beat > previous_beat:
            current_time += (beat - previous_beat) * 60.0 / previous_bpm
        resolved.append((beat, current_time, bpm))
        previous_beat = beat
        previous_bpm = bpm
    return resolved
```

Also update the function signature to accept `chart` so warnings can be appended. The current signature is:

```python
def _tempo_points(
    chart: ParsedDtxChart, measure_starts: list[float]
) -> list[tuple[float, float, float]]:
```

`chart` is already in the signature — no change needed there.

- [ ] **Step 2: Verify timing tests pass**

Run: `uv run pytest tests/benchmark/test_timing.py -v`

Expected: all tests PASS.

---

## Task 4: Fix `scoring.py` — metadata dict aliasing

**Files:**
- Modify: `src/benchmark/scoring.py`

- [ ] **Step 1: Add `dataclasses` import**

At the top of `scoring.py`, add `import dataclasses` to the existing imports block.

- [ ] **Step 2: Copy metadata when building adjusted predictions**

In `score_events`, replace the `adjusted_predictions` list comprehension (lines 31–40):

```python
    adjusted_predictions = [
        dataclasses.replace(
            event,
            time_sec=event.time_sec + offset_sec,
            metadata=dict(event.metadata),
        )
        for event in predictions
    ]
```

- [ ] **Step 3: Verify scoring tests pass**

Run: `uv run pytest tests/benchmark/test_scoring.py -v`

Expected: all tests PASS.

---

## Task 5: Fix `midi_io.py` — warn on empty drum track

**Files:**
- Modify: `src/benchmark/midi_io.py`

- [ ] **Step 1: Add warning after the instrument loop**

In `parse_prediction_midi`, after the for-loop over instruments and before `return sorted(events)`, add:

```python
    if not events and midi.instruments:
        non_drum = [i.name for i in midi.instruments if not i.is_drum]
        logger.warning(
            "parse_prediction_midi: no drum track found in %s (chart_id=%r); "
            "non-drum instruments: %s",
            path,
            chart_id,
            non_drum or "<none>",
        )
```

- [ ] **Step 2: Verify midi_io tests pass**

Run: `uv run pytest tests/benchmark/test_midi_io.py -v`

Expected: all tests PASS.

---

## Task 6: Fix `prepare.py` — guard shutil.copy2

**Files:**
- Modify: `src/benchmark/prepare.py`

- [ ] **Step 1: Wrap copy2 calls in try/except**

In `prepare_corpus`, replace the two `shutil.copy2` calls and the `manifest_entries.append` (lines 69–81) with:

```python
        try:
            shutil.copy2(item.selected_chart, parsed_chart_path)
            shutil.copy2(item.selected_audio, parsed_audio_path)
        except OSError as exc:
            scan_result.invalid_items.append(
                InvalidPreparedCorpusItem(
                    raw_folder=item.raw_folder,
                    reason="failed to copy corpus files",
                    details={"exception": str(exc)},
                )
            )
            continue
        manifest_entries.append(
            {
                "song_id": item.song_id,
                "raw_folder": str(item.raw_folder),
                "selected_chart": item.selected_chart.name,
                "selected_chart_level": item.selected_chart_level,
                "selected_audio": item.selected_audio.name,
                "parsed_chart_path": str(parsed_chart_path),
                "parsed_audio_path": str(parsed_audio_path),
            }
        )
```

Note: `InvalidPreparedCorpusItem` is already imported at the top of the file. Verify the import exists; if not, add it from `src.benchmark.prepare`.

- [ ] **Step 2: Verify prepare tests pass**

Run: `uv run pytest tests/benchmark/test_prepare.py -v`

Expected: all tests PASS.

---

## Task 7: Fix `render_audio.py` — cache, error messages, logging

**Files:**
- Modify: `src/benchmark/render_audio.py`

- [ ] **Step 1: Fix cache miss in `_render_placements`**

In `_render_placements`, replace the first for-loop (lines 336–347) with:

```python
    for placement in placements:
        if placement.sample_path not in cached_samples:
            sample, placement_sample_rate = _load_sample(placement.sample_path)
            cached_samples[placement.sample_path] = sample
            if sample_rate is None:
                sample_rate = placement_sample_rate
            elif sample_rate != placement_sample_rate:
                raise ValueError(
                    f"sample rate mismatch: expected {sample_rate} Hz but "
                    f"{placement.sample_path.name} has {placement_sample_rate} Hz"
                )
            output_channels = max(output_channels, sample.shape[1])
        else:
            sample = cached_samples[placement.sample_path]

        start_frame = max(0, int(round(placement.time_sec * sample_rate)))
        max_frame = max(max_frame, start_frame + sample.shape[0])
```

- [ ] **Step 2: Log path traversal attempt in `_resolve_sample_path`**

In `_resolve_sample_path`, replace the bare `return None` after the traversal check (lines 229–232) with:

```python
    try:
        sample_path.relative_to(song_dir_resolved)
    except ValueError:
        logger.warning(
            "Sample path escapes song directory (song_dir=%s, sample=%s); skipping",
            song_dir_resolved,
            sample_path,
        )
        return None
```

- [ ] **Step 3: Log unreadable sample file in `_resolve_sample_path`**

Replace the bare `return None` in the `sf.info` exception handler (lines 236–238) with:

```python
    try:
        sf.info(sample_path)
    except (OSError, RuntimeError, ValueError, sf.LibsndfileError) as exc:
        logger.debug(
            "Sample file exists but is unreadable (path=%s): %s",
            sample_path,
            exc,
        )
        return None
```

- [ ] **Step 4: Verify render_audio tests pass**

Run: `uv run pytest tests/benchmark/test_render_audio.py -v`

Expected: all tests PASS.

---

## Task 8: Fix `reports.py` — correct empty CSV schema

**Files:**
- Modify: `src/benchmark/reports.py`

- [ ] **Step 1: Add module-level fieldnames constant**

After the `_report_row` function definition, add:

```python
_REPORT_FIELDNAMES = list(
    _report_row(ChartReport("", 0, "", ScoreSummary(0, 0, 0)))
)
```

- [ ] **Step 2: Use constant in `_write_per_chart_csv`**

Replace `_write_per_chart_csv` with:

```python
def _write_per_chart_csv(reports: list[ChartReport], path: Path) -> None:
    rows = [_report_row(report) for report in reports]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=_REPORT_FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)
```

- [ ] **Step 3: Verify reports tests pass**

Run: `uv run pytest tests/benchmark/test_reports.py -v`

Expected: all tests PASS.

---

## Task 9: Fix `dtx_parser.py` — guard VOLUME/POSITION float()

**Files:**
- Modify: `src/benchmark/dtx_parser.py`

- [ ] **Step 1: Replace bare float() calls with guarded parsing**

In `parse_dtx_text`, find the VOLUME and POSITION elif branches (lines 116–119). Replace:

```python
        elif key.startswith("VOLUME") and len(key) == 8:
            volume_table[key[6:].upper()] = float(value)
        elif key.startswith("POSITION") and len(key) == 10:
            position_table[key[8:].upper()] = float(value)
```

With:

```python
        elif key.startswith("VOLUME") and len(key) == 8:
            try:
                volume_table[key[6:].upper()] = float(value)
            except ValueError:
                warnings.append(f"ignoring non-numeric VOLUME value for {key}: {value!r}")
        elif key.startswith("POSITION") and len(key) == 10:
            try:
                position_table[key[8:].upper()] = float(value)
            except ValueError:
                warnings.append(f"ignoring non-numeric POSITION value for {key}: {value!r}")
```

- [ ] **Step 2: Verify dtx_parser tests pass**

Run: `uv run pytest tests/benchmark/test_dtx_parser.py -v`

Expected: all tests PASS.

---

## Task 10: Fix `cli/benchmark.py` help strings + Commit 1

**Files:**
- Modify: `src/cli/benchmark.py`

- [ ] **Step 1: Fix `validate-corpus` help string**

Change line 84:
```python
    """Validate benchmark corpus folder matching."""
```
To:
```python
    """Check DTX charts and prediction MIDIs for missing files, stray files, and duplicate stems."""
```

- [ ] **Step 2: Fix `inspect-dtx` help string**

Change line 96:
```python
    """Inspect parsed DTX timing and lane statistics."""
```
To:
```python
    """Print parsed DTX chart metadata: event count, BPM, measure changes, and lane list."""
```

- [ ] **Step 3: Fix `export-reference-midi` help string**

Change line 113:
```python
    """Export benchmark-owned reference MIDI artifacts."""
```
To:
```python
    """Export MIDI files derived from DTX charts for manual inspection (not used for scoring)."""
```

- [ ] **Step 4: Fix `score-midi` `--align` option help text**

In `src/cli/benchmark.py` line 49, replace:

```python
@click.option("--align/--no-align", default=True, show_default=True)
```

With:

```python
@click.option(
    "--align/--no-align",
    default=True,
    show_default=True,
    help="Compute and apply a global time-offset correction. Emits both raw and aligned report rows when enabled.",
)

- [ ] **Step 5: Run all tests to confirm Commit 1 is clean**

Run: `uv run pytest -x -q`

Expected: all tests PASS with no failures.

- [ ] **Step 6: Check linting**

Run: `uv run ruff check src tests && uv run ruff format --check src tests`

Expected: no errors. If format errors appear, run `uv run ruff format src tests` first.

- [ ] **Step 7: Commit Commit 1**

```bash
git add src/app/transcriber.py src/benchmark/runner.py src/benchmark/timing.py \
        src/benchmark/scoring.py src/benchmark/midi_io.py src/benchmark/prepare.py \
        src/benchmark/render_audio.py src/benchmark/reports.py src/benchmark/dtx_parser.py \
        src/cli/benchmark.py
git commit -m "fix: improve error handling and correctness across benchmark pipeline"
```

---

## Task 11: Add scoring edge-case tests

**Files:**
- Modify: `tests/benchmark/test_scoring.py`

- [ ] **Step 1: Add test for empty ground truth**

Append to `tests/benchmark/test_scoring.py`:

```python
def test_alignment_empty_ground_truth_returns_zero_offset():
    pred = [event(1.0, "kick", "prediction")]
    result = score_events_with_alignment([], pred, 0.05)

    assert result.raw.summary.offset_sec == 0.0
    assert result.aligned.summary.offset_sec == 0.0
    assert result.raw.summary.true_positives == 0
    assert result.raw.summary.false_positives == 1
```

- [ ] **Step 2: Add test for no shared drum classes**

```python
def test_alignment_no_shared_classes_returns_zero_offset():
    gt = [event(1.0, "kick", "ground_truth")]
    pred = [event(1.0, "snare", "prediction")]
    result = score_events_with_alignment(gt, pred, 0.05)

    assert result.raw.summary.offset_sec == 0.0
    assert result.aligned.summary.offset_sec == 0.0
```

- [ ] **Step 3: Run the new tests to verify they pass**

Run: `uv run pytest tests/benchmark/test_scoring.py::test_alignment_empty_ground_truth_returns_zero_offset tests/benchmark/test_scoring.py::test_alignment_no_shared_classes_returns_zero_offset -v`

Expected: both PASS.

---

## Task 12: Add render_audio sample-rate mismatch test

**Files:**
- Modify: `tests/benchmark/test_render_audio.py`

- [ ] **Step 1: Add the sample-rate mismatch test**

Append to `tests/benchmark/test_render_audio.py`:

```python
def test_render_plan_item_reports_sample_rate_mismatch_as_invalid(
    tmp_path: Path, monkeypatch
):
    song = tmp_path / "Song"
    song.mkdir()
    (song / "mas.dtx").write_text(
        "\n".join(["#BPM: 120", "#WAV01: kick.wav", "#WAV02: snare.wav", "#00111: 01", "#00112: 01"]),
        encoding="utf-8",
    )
    _write_sample(song / "kick.wav", [1.0, 0.0], sample_rate=8000)
    _write_sample(song / "snare.wav", [0.5, 0.0], sample_rate=44100)  # mismatched rate

    result = plan_render_song(song, tmp_path / "out")

    # The song should end up invalid because _render_placements raises ValueError
    # on the sample-rate mismatch, which _render_plans catches and records.
    assert not result.valid_items
    assert len(result.invalid_items) == 1
    invalid = result.invalid_items[0]
    assert "sample rate mismatch" in invalid.details.get("message", "")
```

- [ ] **Step 2: Run the new test**

Run: `uv run pytest tests/benchmark/test_render_audio.py::test_render_plan_item_reports_sample_rate_mismatch_as_invalid -v`

Expected: PASS.

---

## Task 13: Add DTX odd-length pattern test

**Files:**
- Modify: `tests/benchmark/test_dtx_parser.py`

- [ ] **Step 1: Add the odd-length test**

Append to `tests/benchmark/test_dtx_parser.py`:

```python
def test_parse_dtx_raises_on_odd_length_pattern():
    import pytest
    from src.benchmark.dtx_parser import parse_dtx_text

    with pytest.raises(ValueError, match="odd length"):
        parse_dtx_text("#BPM: 120\n#00011: 010\n", chart_id="test")
```

- [ ] **Step 2: Run the new test**

Run: `uv run pytest tests/benchmark/test_dtx_parser.py::test_parse_dtx_raises_on_odd_length_pattern -v`

Expected: PASS.

---

## Task 14: Add runner `align=False` test

**Files:**
- Modify: `tests/benchmark/test_runner.py`

- [ ] **Step 1: Add the no-align test**

Append to `tests/benchmark/test_runner.py`:

```python
def test_run_score_midi_no_align_emits_only_raw_reports(tmp_path: Path):
    charts = tmp_path / "charts"
    predictions = tmp_path / "predictions"
    output = tmp_path / "out"
    charts.mkdir()
    predictions.mkdir()
    (charts / "foo.dtx").write_text("#BPM: 120\n#00013: 0100\n", encoding="utf-8")
    write_prediction(predictions / "foo.mid")

    reports = run_score_midi(charts, predictions, output, tolerance_ms=[50], align=False)

    modes = {report.mode for report in reports}
    assert modes == {"raw"}, f"expected only 'raw' mode, got {modes}"
    # 1 chart × 1 tolerance × 1 mode = 1 report
    assert len(reports) == 1
```

- [ ] **Step 2: Run the new test**

Run: `uv run pytest tests/benchmark/test_runner.py::test_run_score_midi_no_align_emits_only_raw_reports -v`

Expected: PASS.

---

## Task 15: Add `transcribe-and-score` CLI test

**Files:**
- Modify: `tests/test_cli_benchmark.py`

- [ ] **Step 1: Add the CLI test**

Append to `tests/test_cli_benchmark.py`:

```python
def test_transcribe_and_score_cli_runs_and_reports_chart_count(
    tmp_path: Path, monkeypatch
):
    from src.benchmark.models import ScoreSummary
    from src.benchmark.reports import ChartReport

    fake_reports = [ChartReport("foo", 50, "raw", ScoreSummary(1, 0, 0))]
    monkeypatch.setattr(
        "src.cli.benchmark.run_transcribe_and_score",
        lambda *args, **kwargs: fake_reports,
    )

    result = CliRunner().invoke(
        main,
        [
            "benchmark",
            "transcribe-and-score",
            "--charts-dir", str(tmp_path),
            "--audio-dir", str(tmp_path),
            "--output-dir", str(tmp_path / "out"),
        ],
    )

    assert result.exit_code == 0, result.output
    assert "1 chart" in result.output
```

`CliRunner`, `main` are already imported at the top of the file. No new imports needed.

- [ ] **Step 2: Run the new test**

Run: `uv run pytest tests/test_cli_benchmark.py::test_transcribe_and_score_cli_runs_and_reports_chart_count -v`

Expected: PASS.

---

## Task 16: Rewrite documentation + Commit 2

**Files:**
- Modify: `docs/drumery-dtx-midi-benchmarking-reference.md`

- [ ] **Step 1: Restructure the parser section**

Find the section titled "Supported DTX Subset in the Parser" (or similar). Replace it with two clearly separated subsections:

```markdown
## DTX Parsing in the Benchmark Pipeline

### Python `dtx_parser.py` (used for scoring)

The benchmark uses `src/benchmark/dtx_parser.py` exclusively for all DTX parsing. It handles:

**File encoding:** Attempts UTF-8 first; falls back to Shift-JIS (common for Japanese DTX files). Line splitting uses Python's `str.splitlines()`, which correctly handles LF (`\n`), CRLF (`\r\n`), and CR (`\r`) files.

**Line prefixes:** Both `#` and `*` are accepted as line-directive prefixes.

**Header fields parsed:**
- `#TITLE` — song title (string; semicolons preserved)
- `#ARTIST` — artist name (string; semicolons preserved)
- `#BPM` — base tempo in BPM (positive float)
- `#BPMxx` — BPM lookup table entry (positive float; hex key)
- `#WAVxx` — sample filename (string; hex key)
- `#VOLUMExx` — per-sample volume scalar (float; non-numeric values produce a warning and are skipped)
- `#POSITIONxx` — per-sample pan position (float; non-numeric values produce a warning and are skipped)

**Tempo events:**
- Channel `02` — measure-length multiplier (modifies how many beats a measure spans)
- Channel `03` — inline BPM, encoded as a two-digit hex integer (e.g. `78` hex = 120 BPM)
- Channel `08` — BPM lookup index into the `#BPMxx` table

**Note chips:** Any channel not handled above is treated as a note event (lane = channel id, note_id = chip value).

**Duplicate tempo events:** If two BPM chips resolve to the same beat position, the later value wins and a warning is added to `chart.warnings`.

### Background: Drumery TypeScript Parser (not used for scoring)

The Drumery web application uses a separate TypeScript parser (`dtx.ts`, `note.ts`) for chart rendering. This parser is **not** used anywhere in the benchmark pipeline. Its behavior may differ from `dtx_parser.py` in edge cases (whitespace handling, comment stripping, supported channels). The sections below describe the TypeScript parser for reference only.
```

- [ ] **Step 2: Fix `--align/--no-align` description**

Find the paragraph describing `--align`/`--no-align` in the docs. Replace it with:

```markdown
`--align` / `--no-align` controls whether a global time-offset correction is computed and applied before scoring. With `--align` (the default), the pipeline:

1. Computes a cross-correlation histogram across shared drum classes to find the best global offset
2. Applies that offset to all predictions
3. Emits two report rows per chart per tolerance window: `raw` (unshifted) and `aligned` (offset-corrected)

With `--no-align`, only `raw` report rows are emitted and no offset is computed.
```

- [ ] **Step 3: Run the full test suite to confirm Commit 2 is clean**

Run: `uv run pytest -x -q`

Expected: all tests PASS.

- [ ] **Step 4: Check linting**

Run: `uv run ruff check src tests && uv run ruff format --check src tests`

Expected: no errors.

- [ ] **Step 5: Commit Commit 2**

```bash
git add tests/benchmark/test_scoring.py tests/benchmark/test_render_audio.py \
        tests/benchmark/test_dtx_parser.py tests/benchmark/test_runner.py \
        tests/test_cli_benchmark.py \
        docs/drumery-dtx-midi-benchmarking-reference.md
git commit -m "test: add coverage for benchmark edge cases and fix documentation"
```
