# Benchmark Mapping Calibration Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add a one-off script that calibrates the TF2 model's 88 output bins against the prepared benchmark corpus and writes ranked mapping diagnostics under `artifacts/benchmark/`.

**Architecture:** Keep calibration outside the product CLI. Reuse the existing transcriber for model inference and the benchmark modules for ground truth conversion and scoring. The script should evaluate candidate mappings empirically and emit machine-readable reports only.

**Tech Stack:** Python 3.13, NumPy, librosa, TensorFlow model wrapper in `src/app/transcriber.py`, benchmark modules under `src/benchmark/`.

---

### Task 1: Add calibration script skeleton

**Files:**
- Create: `scripts/calibrate_egmd_mapping.py`

**Step 1: Write minimal argument parsing and output directory resolution**

Define flags for:

- `--charts-dir`
- `--audio-dir`
- `--output-dir`
- `--tolerance-ms`

**Step 2: Write minimal corpus loading**

Load matching `.dtx` and audio files from the prepared corpus.

**Step 3: Commit**

```bash
git add scripts/calibrate_egmd_mapping.py
git commit -m "feat: add mapping calibration script skeleton"
```

### Task 2: Reuse benchmark ground truth conversion

**Files:**
- Modify: `scripts/calibrate_egmd_mapping.py`

**Step 1: Parse each chart and build canonical benchmark events**

Use:

- `parse_dtx_file`
- `dtx_events_to_timed_events`
- `map_dtx_events`

**Step 2: Store per-chart class counts for diagnostics**

### Task 3: Reuse TF2 model inference

**Files:**
- Modify: `scripts/calibrate_egmd_mapping.py`

**Step 1: Instantiate `DrumTranscriber` once**

**Step 2: Run the current TF2 model and capture raw onset probabilities**

Reuse:

- `_compute_spectrogram_for_model`
- `model(...)`
- `_find_onset_peaks`

**Step 3: Convert peak bins into provisional `BenchmarkEvent` values**

### Task 4: Implement empirical mapping search

**Files:**
- Modify: `scripts/calibrate_egmd_mapping.py`

**Step 1: Score each bin against each canonical class by temporal overlap**

**Step 2: Build candidate mappings from top-scoring bins**

**Step 3: Evaluate candidate mappings with `score_events_with_alignment`**

### Task 5: Emit reports and verify on prepared corpus

**Files:**
- Modify: `scripts/calibrate_egmd_mapping.py`

**Step 1: Write JSON outputs**

- `summary.json`
- `best_mapping.json`
- `per_chart.json`

**Step 2: Run on `artifacts/benchmark/Test DTX/`**

```bash
uv run python scripts/calibrate_egmd_mapping.py \
  --charts-dir "artifacts/benchmark/Test DTX/charts" \
  --audio-dir "artifacts/benchmark/Test DTX/audio"
```

**Step 3: Commit**

```bash
git add scripts/calibrate_egmd_mapping.py
git commit -m "feat: add one-off mapping calibration script"
```
