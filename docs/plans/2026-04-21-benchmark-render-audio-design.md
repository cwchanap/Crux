# Benchmark Render Audio Design

## Goal

Add a reusable `crux benchmark render-audio` command that synthesizes drum-only benchmark audio
from raw DTX song folders using the selected chart and its referenced sample chips.

## User-Facing Command

The new command will live under the existing grouped Click CLI:

```text
crux benchmark render-audio
```

It supports both:

- single-song mode via `--song-dir`
- batch mode via `--raw-dir`

It follows the existing repo-local benchmark artifact convention:

- default output root: `artifacts/benchmark/<run-name-or-input-dir-name>/`

Default outputs:

- prepared-corpus-compatible audio under `audio/<song>.wav`
- standalone render artifacts under `renders/<song>.wav`
- `manifest.json`
- `invalid.json`

## Selection Rules

Chart selection must match `prepare-corpus` exactly:

- select the highest available chart using `mas > ext > adv > bas`

This prevents render-time chart choice from diverging from benchmark preparation.

## Architecture

Add a dedicated render module rather than overloading `prepare-corpus`.

### CLI

- `src/cli/benchmark.py`
  - add `render-audio`
- `src/cli/options.py`
  - add reusable `song_dir` option
  - keep output resolution aligned with existing benchmark commands

### Benchmark modules

- `src/benchmark/render_audio.py`
  - song selection orchestration
  - batch orchestration
  - render planning
  - mixing
  - manifest / invalid report writing
- `src/benchmark/prepare.py`
  - reuse the existing highest-chart selection helpers instead of duplicating the rule
- `src/benchmark/dtx_parser.py`
  - extend parsed chart metadata to expose render-relevant directives:
    - `#WAVxx`
    - `#VOLUMExx`
    - `#POSITIONxx`

The benchmark scoring pipeline remains unchanged. `render-audio` only produces drum stems that
later steps can consume.

## Render Pipeline

For each valid raw song folder:

1. Select the highest-priority chart using the existing benchmark rule.
2. Parse the selected DTX chart, including timing, chip table, chip volume, and chip position
   directives.
3. Convert note events into absolute event times using the benchmark timing map.
4. Resolve each note’s `note_id` through the parsed `#WAVxx` chip table to a concrete sample file.
5. Load the referenced sample audio.
6. Apply supported per-chip transformations:
   - volume scaling from `#VOLUMExx`
   - playback offset from `#POSITIONxx` if supported cleanly
7. Mix the sample into an output buffer at the computed event time.
8. Write the final render as `.wav`.

The renderer should interpret the chart literally from its chip definitions. It should not guess
missing samples from filenames.

## Modes

### Single-song mode

Inputs:

- `--song-dir`

Outputs:

- `audio/<song>.wav`
- `renders/<song>.wav`
- `manifest.json`
- `invalid.json`

### Batch mode

Inputs:

- `--raw-dir`

Outputs:

- one render per valid song
- aggregate `manifest.json`
- aggregate `invalid.json`

## Validation

Each invalid song should be reported with an explicit reason, including:

- no recognized benchmark chart
- multiple charts for the selected level
- missing `#WAVxx` chip for a referenced note id
- chip references a missing sample file
- unreadable or unsupported sample file
- no renderable note events
- output write failure

Songs that fail validation should be excluded from the render manifest and listed under
`invalid.json`.

## Scope And Limitations

Version 1 should support the subset needed for the current benchmark corpus:

- chart selection via `mas > ext > adv > bas`
- benchmark timing map from parsed DTX
- `#WAVxx` chip lookup
- `#VOLUMExx` gain scaling
- `#POSITIONxx` playback offset if it can be applied deterministically
- sample decoding for formats present in the raw song folders, especially `.ogg`, `.mp3`, and
  `.xa` when supported by the runtime audio stack

This is not intended to emulate the full DTX engine. It is a deterministic benchmark renderer.

If a sample format such as `.xa` cannot be decoded reliably in the current Python stack, the
renderer should fail explicitly for that song rather than silently dropping those hits.

## Testing

Unit tests should cover:

- parser support for `#WAVxx`, `#VOLUMExx`, `#POSITIONxx`
- render-plan generation from timed note events and chip metadata
- overlapping-hit mixing with small synthetic sample fixtures
- CLI behavior for single-song and batch modes

Integration coverage should include:

- a tiny synthetic raw song folder that renders a real `.wav`

Real-data verification target after implementation:

1. render `Kyuuka ressha no madobe de`
2. verify the rendered stem exists and has plausible duration
3. place it into the parsed benchmark corpus
4. rerun calibration and transcription benchmarking on both `Kyuuka` and `Soukyuu`

## Success Criteria

- `crux benchmark render-audio` works in both single-song and batch modes
- outputs default to benchmark artifact directories consistent with the rest of the CLI
- valid songs produce deterministic `.wav` drum stems
- invalid songs fail explicitly with actionable diagnostics
- rendered stems can be fed into the benchmark preparation / evaluation workflow
