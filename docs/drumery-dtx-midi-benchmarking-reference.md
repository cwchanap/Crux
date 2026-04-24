# Drumery DTX and MIDI Benchmarking Reference

## Purpose

This document describes the DTX subset implemented by Drumery's parser and the exact
DTX-to-MIDI / MIDI-to-DTX mapping used by the current codebase. It is intended as a
benchmarking reference for evaluating drum transcription or drum-track-to-MIDI conversion
against DTX charts.

This is not a generic DTX specification. It is an implementation-grounded reference based on:

- `../drumery/packages/common/src/lib/chart/dtx.ts`
- `../drumery/packages/common/src/lib/chart/note.ts`
- `../drumery/packages/common/src/lib/game/scenes/BaseGame.ts`
- `../drumery/packages/common/src/lib/game/scenes/Preview.ts`

## Key Takeaway

If you want to benchmark a model against existing Drumery DTX charts, do not assume that:

- the converter implements the full DTX format
- the lane ids used by the editor/game match the lane ids used by the DTX-to-MIDI tool
- a DTX chart can be losslessly round-tripped through MIDI

The safest benchmarking strategy is:

1. Parse the DTX chart directly.
2. Normalize its lane ids into your own canonical drum classes.
3. Compare the normalized events against model MIDI output after applying the same canonical mapping.

## Files and Code Paths

### DTX parser and exporter

- `../drumery/packages/common/src/lib/chart/dtx.ts`
- `../drumery/packages/common/src/lib/chart/note.ts`

### DTX/MIDI web tools

- `../drumery/packages/dtx-web/src/routes/(tool)/tool/dtx-to-midi/+page.svelte`
- `../drumery/packages/dtx-web/src/routes/(tool)/tool/midi-to-dtx/+page.svelte`

### Internal game/editor lane conventions

- `../drumery/packages/common/src/lib/game/scenes/BaseGame.ts`
- `../drumery/packages/common/src/lib/game/scenes/Preview.ts`

## Supported DTX Subset in the Parser

### File decoding

When a DTX file is loaded from `File`, the parser tries these encodings in order:

- `shift-jis`
- `utf-8`
- `utf-16le`
- `utf-16be`

The fallback encoding is `shift-jis`.

Validation is shallow. The file is treated as DTX-like if the decoded text contains one of:

- `#TITLE:`
- `#ARTIST:`
- `#BPM:`
- `#WAV`

### Line splitting

`parseFromText()` splits only on `\r\n`.

Implication:

- CRLF files are handled as expected.
- LF-only files may not be parsed correctly as multi-line DTX files.

### Header fields actually parsed

The parser reads these directives:

- `#TITLE`
- `#ARTIST`
- `#DLEVEL`
- `#BPM`
- `#PREIMAGE`
- `#PREVIEW`

It does not parse `#COMMENT` from source DTX, even though export can write `#COMMENT`.

### Sound chip directives actually parsed

The parser reads:

- `#WAVxx`
- `#VOLUMExx`
- `#POSITIONxx`

Where `xx` is a 2-character base-36 chip id.

Examples:

- `#WAV01: kick.wav`
- `#WAVA1: hihat.wav`
- `#VOLUME01: 80`
- `#POSITION02: 50`

Important implementation detail:

- sound-chip parsing expects `": "` with a trailing space after the colon
- `#WAV01:kick.wav` is not handled consistently by the sound-chip parser

### BPM change directives

The parser can extract `#BPMxx` directives into a lookup table, including:

- `#BPM: 120`
- `#BPM01: 140.5`

But the DTX-to-MIDI exporter does not apply BPM changes during MIDI generation. Only the base
`this.bpm` value is used.

### Note line grammar implemented by the parser

The parser expects note lines in this form:

```text
#mmmll: pattern
```

Where:

- `mmm` is a 3-digit measure number
- `ll` is a 2-character lane id
- `pattern` is an even-length string of 2-character note ids
- `00` means "empty slot"

Example:

```text
#00011: 01020000
```

The parser:

- splits the pattern into 2-character chunks
- assigns each chunk a fractional position within the measure
- filters out `00`

Example:

```text
#00011: 01020000
```

becomes:

- note `01` at position `0`
- note `02` at position `0.25`

### Supported subdivisions

Subdivision support is implicit through pattern length. Tests cover:

- 16th-note-like grids
- 24th-note grids
- 32nd-note grids
- 48th-note grids
- 64th-note grids

Positions are represented as fractions of the measure and normalized for precision.

### Important parser limitations

- note lines must start with `#`
- note parsing expects `": "` with a trailing space
- measure-length changes are not handled here
- extended DTX channel semantics are not interpreted
- note ids are treated as opaque 2-character values until later conversion

## Lane Id Conventions: The Biggest Benchmarking Trap

There are multiple lane-id conventions in the codebase.

### Internal game/editor lanes

The gameplay/editor code uses drum-style DTX lane ids such as:

- `11` = closed hi-hat
- `12` = snare
- `13` = bass drum
- `14` = high tom
- `15` = low tom
- `16` = cymbal
- `17` = floor tom
- `18` = open hi-hat
- `19` = ride
- `1A` = left cymbal
- `1B` = left pedal
- `1C` = left bass drum

These appear in `BaseGame.ts` and `Preview.ts`.

### DTX-to-MIDI tool lanes

The DTX-to-MIDI tool and MIDI conversion helper use a different built-in mapping based on:

- `01`
- `02`
- `03`
- `04`
- `05`
- `06`
- `07`
- `08`
- `09`
- `0A`
- `0B`
- `0C`

This is not aligned with the internal/editor drum lane ids above.

### Benchmarking implication

If your source DTX charts use the conventional drum lane ids (`11`, `12`, `13`, etc.), the
converter's built-in `01`-`0C` map is not a safe ground truth. You should normalize lanes
yourself before scoring model output.

## DTX to MIDI Mapping Implemented by Drumery

### MIDI file structure

The exporter writes:

- Standard MIDI header `MThd`
- format `0`
- track count `1`
- ticks per quarter note `480`

The track contains:

- one tempo meta event
- note on / note off events
- one end-of-track meta event

### Timing model

The exporter assumes:

- fixed 4/4 time
- `ticksPerMeasure = 480 * 4 = 1920`

For each parsed note:

- `measureStart = measure * 1920`
- `noteTime = measureStart + (position * 1920)`

Every note is emitted as:

- MIDI channel `9` (zero-based drum channel, i.e. channel 10 in MIDI UI terms)
- velocity `100`
- duration `120` ticks

### Lane to MIDI note map used by the converter

| Lane | MIDI note | Drum label |
| --- | ---: | --- |
| `01` | 36 | Bass Drum |
| `02` | 38 | Snare |
| `03` | 42 | Closed Hi-Hat |
| `04` | 46 | Open Hi-Hat |
| `05` | 49 | Crash Cymbal |
| `06` | 51 | Ride Cymbal |
| `07` | 45 | Low Tom |
| `08` | 47 | Mid Tom |
| `09` | 50 | High Tom |
| `0A` | 44 | Pedal Hi-Hat |
| `0B` | 57 | Crash 2 |
| `0C` | 59 | Ride 2 |

Unknown lanes fall back to MIDI note `60`.

### What the exporter ignores

The DTX note id inside the pattern is not mapped to MIDI pitch. For export purposes:

- any non-`00` note is treated as "a hit exists at this lane/time"
- pitch comes from the lane id mapping
- WAV-chip identity is not preserved in MIDI

This matters for benchmarking:

- a DTX chart with multiple chip ids on the same lane will collapse to a single drum class in MIDI

### Known exporter failure mode

The current serializer updates `currentTime` to `noteTime + duration` after each note. That can
produce invalid delta times for:

- simultaneous hits on different lanes
- overlapping hits closer than the fixed duration

Since drum chords are common, this makes the built-in DTX-to-MIDI export unsafe as a benchmark
reference for exact MIDI event generation. For benchmarking, prefer direct DTX event extraction
plus your own normalized lane mapping.

## MIDI to DTX Mapping Implemented by Drumery

### MIDI parsing support

The MIDI parser reads:

- header chunk `MThd`
- track chunks `MTrk`
- note on events
- note off events
- meta events
- running status for channel messages

It does not robustly implement the full MIDI spec. Unsupported/other events are skipped with
minimal logic, so complex files may parse incorrectly.

### Metadata created when converting MIDI to DTX

Default values:

- title: `Converted from MIDI`
- artist: `Unknown`
- level: `5`
- bpm: `120`
- comment: `Converted from MIDI file`

If tempo meta events (`0xFF 0x51`) are found, BPM is replaced by the latest one encountered by
the current track scan.

### MIDI note to DTX lane map used by the converter

| MIDI note | Lane | Drum label |
| ---: | --- | --- |
| 36 | `01` | Bass Drum |
| 38 | `02` | Snare |
| 42 | `03` | Closed Hi-Hat |
| 46 | `04` | Open Hi-Hat |
| 49 | `05` | Crash Cymbal |
| 51 | `06` | Ride Cymbal |
| 45 | `07` | Low Tom |
| 47 | `08` | Mid Tom |
| 50 | `09` | High Tom |
| 44 | `0A` | Pedal Hi-Hat |
| 57 | `0B` | Crash 2 |
| 59 | `0C` | Ride 2 |

Only note-on events with velocity greater than zero are converted into DTX hits.

### Timing model for MIDI to DTX

The converter assumes:

- fixed 4/4 time
- `ticksPerMeasure = ticksPerQuarter * 4`

For each converted note:

- `measure = floor(currentTime / ticksPerMeasure)`
- `positionInMeasure = (currentTime % ticksPerMeasure) / ticksPerMeasure`

### What is lost during MIDI to DTX conversion

All converted notes are emitted with:

- DTX note id `01`

The converter does not generate:

- real `#WAVxx` assignments for each instrument
- original chip ids
- per-hit velocity mapping
- per-hit sample linkage

So MIDI-to-DTX in Drumery produces a structural lane/timing representation, not a faithful
simfile reconstruction.

## Round-Trip Expectations

You should not expect this round trip to be lossless:

```text
DTX -> MIDI -> DTX
```

Information lost in the round trip includes:

- original WAV chip ids
- multi-sample distinctions on a single lane
- custom/extended DTX channel semantics
- measure-length semantics outside the supported subset
- exact simultaneity when using the built-in DTX-to-MIDI exporter

## Recommended Canonical Mapping for Benchmarking

For model evaluation, define your own canonical drum classes and map both DTX and MIDI into them.

A practical canonical set is:

- kick
- snare
- closed_hihat
- open_hihat
- crash
- ride
- low_tom
- mid_tom
- high_tom
- pedal_hihat
- crash_2
- ride_2

Then maintain two normalization tables:

- one from DTX lane ids in your chart corpus to canonical classes
- one from MIDI note numbers in model output to canonical classes

If your Drumery charts use editor/game lane ids, a likely starting point is:

| DTX lane | Canonical class |
| --- | --- |
| `11` | closed_hihat |
| `12` | snare |
| `13` | kick |
| `14` | high_tom |
| `15` | low_tom |
| `16` | crash |
| `17` | floor_tom |
| `18` | open_hihat |
| `19` | ride |
| `1A` | left_cymbal |
| `1B` | pedal_hihat |
| `1C` | left_kick |

You can then merge or collapse these further depending on your benchmark definition.

## Practical Benchmarking Advice

- Use DTX as the timing/lane ground truth, not Drumery's exported MIDI.
- Parse DTX note events directly and normalize lanes yourself.
- Treat note ids (`01`, `02`, etc. inside the pattern) as sample ids, not drum classes.
- Benchmark on note onset timing plus normalized instrument class.
- Avoid relying on round-trip fidelity as an evaluation signal.
- If you need a benchmark MIDI reference, generate it with your own exporter from parsed DTX notes.

## Summary

Drumery currently provides:

- a narrow DTX parser
- a custom drum-lane MIDI mapper
- a structural MIDI-to-DTX converter

For benchmarking a drum transcription model, the useful ground truth is the parsed DTX note timing
plus a lane normalization layer that you define explicitly. The built-in DTX-to-MIDI and
MIDI-to-DTX conversions are helpful utilities, but they should not be treated as a lossless or
canonical interchange layer.

## Reproducible Benchmark Workflow

This section documents the benchmark workflow implemented in this repository so results can be
reproduced consistently.

### Recommended workflow

The recommended path is:

```text
raw song folders -> prepare-corpus -> parsed charts/audio -> transcribe-and-score or score-midi
```

Use this path when a song folder already contains a usable drum stem such as `2 Drums.mp3` or
`drum.mp3`.

In this workflow:

- raw DTX chip/sample assets such as `bass.xa` or `snare.ogg` are not needed for scoring
- the parsed corpus becomes the stable working directory for future benchmark runs
- the DTX chart remains the scoring authority

### Benchmark inputs

The benchmark uses these inputs:

- ground truth chart: parsed from `.dtx`
- model input audio: one drum-only audio file per chart
- prediction: model-generated MIDI or precomputed MIDI

The benchmark does not require raw DTX chip assets unless you are using the optional
`render-audio` fallback described later in this document.

### Raw corpus assumptions for `prepare-corpus`

`crux benchmark prepare-corpus` scans a raw song-folder corpus and selects one benchmark item per
song folder.

Selection rules:

- chart priority is `mas > ext > adv > bas`
- exactly one selected chart file must exist at the winning level
- the drum audio filename must be one of:
  - `2 Drums.mp3`
  - `drum.mp3`

If a song folder does not meet those rules, it is excluded from the parsed corpus and recorded in
`invalid.json`.

### Step 1: Prepare a parsed corpus

Run:

```bash
uv run crux benchmark prepare-corpus \
  --raw-dir /path/to/raw-dtx-corpus \
  --run-name my-benchmark-corpus
```

If `--output-dir` is omitted, outputs default to:

```text
artifacts/benchmark/<run-name-or-input-dir-name>/
```

Example output layout:

```text
artifacts/benchmark/my-benchmark-corpus/
  charts/
    Song A.dtx
  audio/
    Song A.mp3
  manifest.json
  invalid.json
```

`manifest.json` records the selected chart and copied audio for each valid song. `invalid.json`
records every rejected raw folder and the reason it was excluded.

After this step, future benchmark work should use the parsed corpus rather than the original raw
folder structure.

### Step 2: Inspect or validate the parsed corpus

Optional checks:

Inspect one parsed chart:

```bash
uv run crux benchmark inspect-dtx \
  artifacts/benchmark/my-benchmark-corpus/charts/Song\ A.dtx
```

Validate a parsed chart directory against precomputed prediction MIDI files:

```bash
uv run crux benchmark validate-corpus \
  --charts-dir artifacts/benchmark/my-benchmark-corpus/charts \
  --predictions-dir /path/to/predictions
```

### Step 3A: Run end-to-end transcription and scoring

Use this when you want the benchmark to run the model and score its generated MIDI:

```bash
uv run crux benchmark transcribe-and-score \
  --charts-dir artifacts/benchmark/my-benchmark-corpus/charts \
  --audio-dir artifacts/benchmark/my-benchmark-corpus/audio \
  --run-name my-transcription-run \
  --tolerance-ms 30 \
  --tolerance-ms 50 \
  --tolerance-ms 100
```

Outputs:

```text
artifacts/benchmark/my-transcription-run/
  predictions/
    Song A.mid
  summary.json
  per_chart.csv
  summary.md
```

The current transcriber expects the benchmark model weights to be available at:

```text
models/e-gmd/tf2_model.weights.h5
```

### Step 3B: Score precomputed MIDI only

Use this when model MIDI has already been generated elsewhere:

```bash
uv run crux benchmark score-midi \
  --charts-dir artifacts/benchmark/my-benchmark-corpus/charts \
  --predictions-dir /path/to/predictions \
  --run-name my-score-run \
  --tolerance-ms 30 \
  --tolerance-ms 50 \
  --tolerance-ms 100
```

Optional flags:

- `--align/--no-align` controls whether global offset alignment is reported
- `--export-reference-midi` writes benchmark-owned reference MIDI artifacts alongside the reports

### Optional: Export benchmark-owned reference MIDI

If you want a benchmark-owned MIDI representation of the parsed DTX charts for debugging or manual
inspection, export it explicitly:

```bash
uv run crux benchmark export-reference-midi \
  --charts-dir artifacts/benchmark/my-benchmark-corpus/charts \
  --run-name my-reference-midi
```

This is useful for inspection, but parsed DTX remains the scoring source of truth.

## Optional Fallback: `render-audio`

`render-audio` is not part of the core benchmark path when a usable drum stem already exists.

It exists for the fallback case where:

- a raw song folder has a usable chart
- but there is no benchmark-ready drum audio file to feed into `prepare-corpus`

In that case, `render-audio` can synthesize a drum-only `.wav` from the raw DTX chart and its
referenced sample chips.

Single-song mode:

```bash
uv run crux benchmark render-audio \
  --song-dir /path/to/raw-song-folder \
  --run-name my-render
```

Batch mode:

```bash
uv run crux benchmark render-audio \
  --raw-dir /path/to/raw-dtx-corpus \
  --run-name my-render-batch
```

Outputs:

```text
artifacts/benchmark/my-render/
  audio/
    Song A.wav
  renders/
    Song A.wav
  manifest.json
  invalid.json
```

Use `render-audio` only when you need to manufacture a drum stem. If a song already has
`2 Drums.mp3` or `drum.mp3`, prefer `prepare-corpus` directly.

## Reproducibility Notes

For reproducible runs:

- keep the parsed corpus fixed once prepared
- record the benchmark command, `--run-name`, and tolerance windows
- keep the model weights file fixed across comparisons
- compare runs using the emitted `summary.json` and `per_chart.csv`

If you add more songs later, rerun `prepare-corpus` to create a new parsed corpus artifact rather
than mutating old results in place without tracking the change.

## Known Limitations

- `prepare-corpus` currently only recognizes `2 Drums.mp3` and `drum.mp3` as allowed raw drum
  stem filenames
- one raw song folder yields at most one benchmark chart, selected by `mas > ext > adv > bas`
- `render-audio` is optional and may fail on raw sample formats that the current Python audio
  stack cannot decode reliably
- in the current environment, `.xa` raw sample files are a known example of that limitation
