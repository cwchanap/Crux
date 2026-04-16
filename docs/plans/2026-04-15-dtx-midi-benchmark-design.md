# DTX MIDI Benchmark Design

## Purpose

Build a benchmark for evaluating drum transcription accuracy against a corpus of DTX
charts and generated drum-only audio renders. The benchmark scores model output by
onset timing and normalized drum class. Velocity and exact MIDI-event equivalence are
out of scope for the first version.

Parsed DTX events are the benchmark ground truth. Generated reference MIDI is an
optional artifact for debugging and listening, not the scoring authority.

## Goals

- Evaluate onset and normalized drum-class accuracy for a few hundred DTX charts.
- Support generated drum-only audio as the controlled model input.
- Report both model-compatible collapsed classes and richer DTX-native diagnostics.
- Report both raw and auto-aligned scores for each chart.
- Support BPM changes and measure-length changes in DTX timing.
- Provide a scalable Click CLI under `src/cli` that can grow beyond benchmarking.
- Keep parser, timing, scoring, and reporting logic reusable and independently testable.

## Non-Goals

- Do not treat Drumery's DTX-to-MIDI exporter as ground truth.
- Do not rely on DTX-to-MIDI-to-DTX round trips as an evaluation signal.
- Do not score velocity in v1.
- Do not integrate benchmark runs into the FastAPI API/UI in v1.
- Do not require the real TensorFlow model in unit tests.

## CLI Architecture

Use `src/cli` as the single home for project CLI entry points. Add a scalable Click
root command that can support benchmark and non-benchmark functionality over time.

Suggested structure:

```text
src/cli/
  __init__.py
  main.py
  options.py
  benchmark.py
  convert.py
```

Suggested public command shape:

```text
crux
  benchmark score-midi
  benchmark transcribe-and-score
  benchmark export-reference-midi
  benchmark inspect-dtx
  benchmark validate-corpus
  convert checkpoint
```

`src/cli/main.py` owns the root Click group. `src/cli/benchmark.py` owns the
benchmark subgroup and calls benchmark service modules. `src/cli/options.py` contains
reusable Click option decorators for shared paths, mappings, output, tolerance
windows, alignment, verbosity, and report formats.

Keep the existing `convert-checkpoint` script compatible while adding the grouped
command path:

```toml
[project.scripts]
crux = "src.cli.main:main"
convert-checkpoint = "src.cli.convert:main"
```

## Benchmark Core Architecture

Core benchmark logic should live outside `src/cli`:

```text
src/benchmark/
  corpus.py
  dtx_parser.py
  timing.py
  mapping.py
  midi_io.py
  scoring.py
  reports.py
  config.py
```

Responsibilities:

- `corpus`: discover benchmark items by folder basename matching.
- `dtx_parser`: parse DTX headers, WAV definitions, note channels, BPM definitions,
  BPM change channels, and measure-length channel `02`.
- `timing`: convert DTX measure/position events into seconds using tempo maps and
  measure-length multipliers.
- `mapping`: normalize DTX lanes and MIDI notes into canonical classes.
- `midi_io`: parse prediction MIDI note-on events and write optional reference MIDI.
- `scoring`: match prediction events to ground truth and compute metrics.
- `reports`: write JSON, CSV, and Markdown artifacts.
- `config`: load mapping, scoring, and output configuration.

## Input Discovery

The first version uses folder basename matching:

```text
charts/foo.dtx
audio/foo.wav
predictions/foo.mid
```

`score-midi` requires chart and prediction files. `transcribe-and-score` requires
chart and audio files, then writes or scores generated predictions. Missing
counterparts are validation failures, not silent skips.

## Event Model

DTX parsing first emits structural events:

```text
DtxEvent(chart_id, measure, position, lane_id, note_id)
```

Timing and mapping then produce benchmark events:

```text
BenchmarkEvent(chart_id, time_sec, canonical_class, source, metadata)
```

Prediction MIDI is parsed into note-on events and normalized into the same
`BenchmarkEvent` shape. MIDI note duration is ignored; only onset time and class
matter.

## Timing Model

Support these DTX timing features:

- Base tempo from `#BPM`.
- Extended tempo table from `#BPMxx`.
- BPM change channel `08`, using two-character note IDs that reference `#BPMxx`.
- Numeric BPM change channel `03`, if present.
- Measure-length multiplier from channel `02`.
- Pattern subdivision from the number of two-character cells in each lane pattern.

Use a beat-based model. A normal measure is four quarter notes. Channel `02` scales
the measure duration. A note position is a fraction of that scaled measure. BPM
changes become ordered tempo events at exact measure-relative positions; note times
are integrated segment by segment through the tempo map.

Measure-length handling should follow DTXMania-style behavior: channel `02` entries
set a measure-length multiplier that carries forward until changed.

## Canonical Mapping

Report two class layers:

- Model-compatible collapsed classes: `kick`, `snare`, `closed_hihat`,
  `open_hihat`, `crash`, `ride`, `low_tom`, `mid_tom`, `high_tom`.
- Richer DTX-native diagnostics, including pedal hi-hat, second crashes/rides, and
  left/right variants when the source lanes distinguish them.

The default DTX mapping should include Drumery editor/game lanes such as `11`-`1C`
and the Drumery DTX-to-MIDI tool lanes `01`-`0C`, but the mapping must be
configurable. The default MIDI mapping should use General MIDI drum notes emitted by
the current transcriber.

## Scoring

Score events independently per canonical class. A prediction for one class can never
consume a ground-truth event from another class.

For each configured onset-tolerance window:

1. Partition ground truth and predictions by canonical class.
2. Perform one-to-one nearest-time matching within each class and tolerance.
3. Count matched pairs as true positives.
4. Count unmatched ground-truth events as false negatives.
5. Count unmatched prediction events as false positives.
6. Record signed and absolute timing error for every match.

The CLI should require explicit tolerance configuration. Reports must state every
tolerance window used and optionally mark one as the primary headline tolerance.

## Raw And Aligned Scores

Each chart produces two score sets:

```text
raw
aligned
```

Raw scoring uses prediction times exactly as emitted. Aligned scoring estimates one
global offset per chart and applies it to all prediction events before rescoring. The
alignment must not warp time.

A practical v1 alignment method:

1. Generate candidate offsets from near-neighbor ground-truth and prediction pairs
   across compatible classes.
2. Choose the offset that maximizes matches under the primary tolerance.
3. Break ties by lower median absolute timing error.
4. Rescore with the chosen offset.

Reports always include the chosen offset in milliseconds.

## Reports

Write these outputs:

```text
summary.json
per_chart.csv
per_class.csv
summary.md
reference_midi/  # optional
```

`summary.json` is the canonical machine-readable artifact. CSV files support
analysis. Markdown supports quick human inspection.

Metrics are reported at three levels:

- Per chart: precision, recall, F1, TP, FP, FN, median absolute timing error, p95
  absolute timing error, and alignment offset.
- Per class: the same metrics aggregated across the corpus.
- Corpus headline: micro-averaged precision, recall, and F1 across all configured
  tolerances.

## Validation And Error Handling

`validate-corpus` should scan the folder layout and report:

- Missing matched files.
- Duplicate chart IDs after basename normalization.
- DTX parse failures, encoding failures, malformed note patterns, invalid BPM
  definitions, and unsupported scoring-relevant channels.
- Unmapped DTX lanes and unmapped MIDI notes.
- Timing anomalies such as non-positive BPM, negative measure length, conflicting
  channel `02` entries, or BPM events referencing unknown `#BPMxx` IDs.
- Empty ground-truth charts and empty predictions.

Default behavior is conservative:

- A chart with invalid timing or unmapped scoring lanes is excluded from headline
  scores and listed under `invalid_charts`.
- A chart with valid ground truth but empty predictions is included and scores as all
  false negatives.
- A chart with prediction events in unmapped MIDI notes is included, but those events
  are counted under diagnostics and excluded from the canonical score unless mapping
  config says otherwise.

Strictness controls:

```text
--fail-on-invalid / --keep-going
--unmapped-lanes fail|warn|ignore
--unmapped-midi fail|warn|ignore
```

## Testing

Unit tests should cover the benchmark core without loading TensorFlow or model
weights:

- DTX parsing with CRLF and LF line endings, `#BPM`, `#BPMxx`, BPM channels `03` and
  `08`, channel `02` measure-length changes, simultaneous hits, and malformed lines.
- Timing conversion across BPM changes, measure-length changes, and BPM changes
  inside shortened or extended measures.
- Canonical mapping for Drumery editor lanes, DTX-to-MIDI tool lanes, richer
  DTX-native diagnostics, and unmapped lanes.
- MIDI prediction parsing using minimal generated MIDI bytes or `pretty_midi`.
- Scoring behavior for exact matches, class mismatches, duplicate predictions, false
  positives, false negatives, multiple tolerance windows, and raw versus aligned
  scoring.
- Report writers for stable JSON, CSV, and Markdown output shapes.
- Click CLI command tests with `click.testing.CliRunner` and tiny temp fixture
  folders.

Integration tests should run `score-midi` against a small synthetic corpus with:

- One straight 4/4 constant-BPM chart.
- One chart with BPM changes.
- One chart with measure-length changes and simultaneous hits.

`transcribe-and-score` should be tested with a mocked transcriber.

## Open Implementation Notes

- Decide the exact package import path during implementation. The repository already
  exposes scripts from `src.cli`, so the design keeps that convention.
- Confirm whether the corpus contains DTX features beyond channel `02`, `03`, and
  `08`. If so, add validation diagnostics before expanding scoring semantics.
- Keep the benchmark deterministic: stable sorting, explicit tolerances, explicit
  mapping config, and machine-readable output should make run-to-run diffs meaningful.
