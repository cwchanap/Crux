# HPA-323: Audio-Relative DTX Reference Timing Design

## Context

HPA-323 is the next task on the Crux benchmark critical path after HPA-322.
HPA-322 selects one authoritative DTX chart per simfile and publishes an immutable
reference-chart manifest. HPA-323 must turn those selected charts into trustworthy,
audio-relative native reference events before any model corpus inference or scoring
runs.

The existing timing implementation is close but has three correctness gaps:

1. channel `02` measure lengths apply to one measure only instead of remaining active
   until superseded;
2. channel `01` is parsed as a generic note event and then disappears as an unmapped
   lane rather than becoming an explicit BGM start;
3. drum events remain in chart time because no selected BGM start is subtracted.

This issue is intentionally planned while HPA-322 is active, but implementation must
start from the merged HPA-322 interface. HPA-323 remains blocked by HPA-322 until the
reference-chart manifest and selected-chart cache contract exist.

A second practical gap is that HPA-321 deliberately caches only chart-definition files
(`set.def`, `.dtx`, and `.txt`). The selected full-mix audio is inventoried but normally
is not cached. HPA-323 therefore needs a targeted cache fill for only the DTX-referenced
audio objects. It must not expand HPA-321 into downloading every audio object in the
corpus.

## Goals

- Match DTXMania channel `02` semantics: a measure-length value applies to its measure
  and every following measure until another channel `02` value replaces it.
- Parse every non-zero channel `01` token as a typed BGM event with source order,
  measure, fractional position, and WAV note ID.
- Resolve BGM note IDs through the selected chart's `#WAVxx` table.
- Resolve referenced audio relative to the selected DTX object key, with a narrow
  simfile-root compatibility fallback.
- Prefer exact object-key matches and allow only a unique case-insensitive fallback.
- Download and cache only the exact selected source-audio objects that are missing from
  the existing content-addressed cache.
- Decode source-audio metadata and record duration using the already-installed
  `soundfile` dependency.
- Compute every native DTX event's chart time and audio-relative time from one shared
  timing map.
- Exclude events outside decoded audio bounds while preserving exact counts, reason
  codes, and source metadata.
- Persist immutable native reference-event artifacts before HPA-324 maps lanes into a
  canonical taxonomy.
- Publish a derived immutable manifest that preserves HPA-322 lineage and records the
  source-audio and timing identities required by later benchmark stages.
- Keep raw and auto-aligned scoring separate. HPA-323 fixes the raw reference clock;
  the existing global alignment remains a diagnostic only.

## Non-goals

- Selecting the authoritative DTX chart; HPA-322 owns that decision.
- Defining canonical drum classes, lane collapse, duplicate-after-collapse behavior, or
  final corpus eligibility; HPA-324 owns those policies.
- Running OaF, MuScriptor, IDM, source separation, or scoring.
- Automatically repairing malformed DTX timing or choosing a different chart when the
  selected chart is invalid.
- Guessing a BGM start when channel `01` is absent.
- Treating `bgm.ogg` as an unconditional filename.
- Downloading every audio file listed in R2.
- Adding a database, service, workflow engine, generic plugin system, or concurrency
  beyond the existing small targeted cache helper.
- Preserving backward compatibility for the incorrect chart-time reference artifacts;
  timing semantics are versioned and old artifacts must not be reused.

## Considered Approaches

### 1. Derived timing stage with targeted audio caching — recommended

Add one command after `select-reference-charts`. It consumes the immutable HPA-322
manifest and HPA-321 cache, parses the selected chart, identifies the exact referenced
audio object, fills only missing selected audio bodies into the existing cache, then
publishes native audio-relative event artifacts and a derived manifest.

Advantages:

- fixes timing before inference, so HPA-326 cannot create predictions against an invalid
  reference identity;
- reuses the authoritative object inventory and content-addressed cache;
- avoids downloading unrelated audio;
- persists native reference events once so HPA-324 can remap without reparsing DTX;
- remains deterministic and mostly offline after the first cache fill;
- keeps HPA-321 inventory, HPA-322 chart selection, and HPA-323 timing as separate,
  reviewable stages.

This is the recommended approach.

### 2. Apply the BGM shift inside legacy scoring

`run_score_midi` could load the selected audio and subtract the BGM start immediately
before scoring. This is smaller in the short term, but it would leave corpus inference,
reference MIDI export, HPA-324 mapping, and later model comparisons with different
reference clocks. It would also require reparsing DTX every time a mapping changes.
This approach is rejected.

### 3. Embed all timed events directly in the manifest

Embedding thousands of event objects in each manifest row would avoid separate files,
but it would turn the lineage manifest into a large event database and force every
consumer to parse the entire corpus. Separate content-addressed per-simfile JSONL event
artifacts keep rows compact and allow later tasks to read only the needed songs. This
approach is rejected.

## Operator Interface

Add one command:

```bash
uv run crux benchmark build-reference-timing \
  --manifest artifacts/benchmark/reference-charts/manifests/<sha256>.jsonl \
  --cache-dir artifacts/benchmark/r2-corpus/cache \
  --output-dir artifacts/benchmark/reference-timing
```

Options:

- `--manifest PATH` is required and identifies the immutable HPA-322
  `crux.reference-chart-manifest/v1` input.
- `--cache-dir PATH` defaults to the HPA-321 cache location derived from the input
  manifest layout, but may be supplied explicitly.
- `--output-dir PATH` defaults to `artifacts/benchmark/reference-timing`.

The command uses the existing R2 environment variables only when one or more selected
audio objects are missing from the local cache. A complete cache permits a fully
offline rerun. Missing or invalid R2 configuration while a cache fill is required is a
fatal command error; object-specific download failures quarantine only the affected
rows and still publish successful rows.

The command emits progress and warnings to stderr and one JSON summary to stdout:

```json
{
  "corpus_version": "sha256:...",
  "events_published": 398,
  "exit_code": 1,
  "manifest_path": "artifacts/benchmark/reference-timing/manifests/<sha>.jsonl",
  "quarantined": 2,
  "ready": 398,
  "status": "partial"
}
```

Exit codes:

- `0`: every selected HPA-322 row produced a timing-ready event artifact;
- `1`: the derived manifest was published, but one or more rows were quarantined;
- `2`: invalid input, unavailable required R2 configuration, or artifact-publication
  failure prevented a usable result.

## Architecture

The implementation adds four focused boundaries and one small cache extension.

### Typed DTX BGM events

`src/benchmark/dtx_parser.py` gains a frozen `DtxBgmEvent` record:

```python
@dataclass(frozen=True)
class DtxBgmEvent:
    chart_id: str
    measure: int
    position: float
    note_id: str
    source_order: int
```

`ParsedDtxChart` gains `bgm_events: list[DtxBgmEvent]`. Channel `01` is parsed into
that list and never enters the generic `events` collection. The parser preserves every
non-zero source token and its source order. It does not resolve WAV paths or choose a
BGM event.

All other existing parser behavior remains unchanged. HPA-322's shared text decoder and
DLEVEL fields are consumed as merged prerequisites rather than reimplemented.

### Shared timing map

`src/benchmark/timing.py` gains a small immutable `DtxTimingMap` that owns resolved
measure lengths and tempo points. Both generic DTX events and typed BGM events use the
same `time_sec(event)` method.

The resolved measure-length sequence carries one active value forward:

```text
active_length = 1.0
for each measure in order:
    if the chart defines channel 02 for this measure:
        active_length = defined value
    resolved_lengths[measure] = active_length
```

The active value affects both the measure's start and event positions within that
measure. A later channel `02` value replaces it beginning with that measure.

The timing map preserves the existing behavior for base BPM, channel `03`, channel
`08`, fractional positions, and deterministic same-beat tempo ordering. Existing
`dtx_events_to_timed_events` remains as a compatibility wrapper for legacy tests and
commands, but delegates to the new map.

### BGM and source-audio resolution

`src/benchmark/reference_timing.py` owns the pure row-level policy:

- validate the selected-chart fields and cached selected chart body;
- build the DTX timing map;
- resolve each BGM note ID through `chart.wav_table`;
- normalize DTX path separators;
- resolve the referenced path relative to the selected chart's object-key directory;
- use an exact object-key match first;
- otherwise use one unique case-insensitive match;
- only when the relative lookup has no match, retry from the simfile root as an explicit
  compatibility fallback;
- reject absolute paths and traversal outside the simfile prefix;
- return the exact source-audio object record required by the cache layer.

`bgm.ogg` receives no special authority. It is selected when and only when a typed BGM
event references it and path resolution finds the corresponding object.

### Targeted source-audio cache fill

`src/benchmark/corpus_cache.py` gains a narrowly scoped explicit-key entry point that
reuses the existing cache index, ETag/size validation, content hashing, and immutable
content installation. Existing `sync_cache` behavior remains unchanged for HPA-321.

The new helper receives the already-inventoried `RemoteObject` records and an exact set
of selected keys. It does not list R2 or broaden the suffix policy. HPA-323 performs one
batch cache fill after all rows have completed pure BGM resolution, so duplicate audio
keys download once.

After the cache fill, the HPA-323 row uses the verified `sha256` and `cache_path` returned
by the cache layer. The matching object entry in the derived row is updated with those
verified cache fields so HPA-326 can reuse the same audio body without another fetch.

### Reference-event and manifest publication

`src/benchmark/reference_timing_manifest.py` orchestrates the stage:

1. read and validate the exact HPA-322 JSONL bytes;
2. preserve `source_manifest_sha256` and the HPA-322 `corpus_version`;
3. parse selected charts and resolve BGM/audio candidates;
4. fill exact missing source-audio cache keys;
5. inspect cached audio metadata;
6. compute audio-relative native events;
7. publish one content-addressed event JSONL per timing-ready simfile;
8. publish one derived immutable manifest with the existing canonical renderer.

Click remains responsible only for option parsing, progress display, JSON summary, and
exit-code mapping.

## Input Contract

The HPA-322 input must:

- be UTF-8 JSONL with one object per non-empty line;
- use `schema_version: "crux.reference-chart-manifest/v1"`;
- contain one shared `corpus_version`;
- contain unique integer `simfile_id` values;
- preserve the HPA-321 `object_prefix`, R2 source identity, and `objects` array;
- contain `selection_status`, `selected_chart_key`, and
  `selected_chart_content_hash`.

Rows already quarantined by HPA-322 remain quarantined with
`upstream_chart_selection_unavailable`; HPA-323 does not attempt fallback chart
selection.

For a selected row, the selected chart object must still have a verified cache body
whose exact SHA-256 matches `selected_chart_content_hash`. A mismatch quarantines the
row rather than reparsing a different file.

All input rows must identify one source bucket and source endpoint hash. Mixed R2
sources are rejected as a fatal input error because one cache-fill transaction cannot
safely infer multiple credential domains.

## Channel `02` Timing Semantics

The resolved timing map includes every measure needed by:

- generic DTX events;
- typed BGM events;
- BPM events;
- explicit measure-length directives.

A channel `02` value on measure `M` is active for measure `M` itself and every following
measure until the next explicit value. `_event_beat` must use the resolved length for the
event's measure; fixing only measure-start accumulation is insufficient.

Fixtures cover:

- one shortened measure persisting across several following measures;
- a later channel `02` restoring `1.0`;
- multiple sticky changes in sequence;
- a channel `03` or `08` BPM change inside an altered measure;
- a BGM event and a drum event using the same altered-measure timing map.

## Channel `01` and BGM Selection Policy

Every non-zero channel `01` token becomes a `DtxBgmEvent`. `bgm_event_count` records the
raw number of parsed events.

Each BGM event must resolve successfully through `#WAVxx`. An unknown note ID, empty WAV
value, unsafe path, missing object, ambiguous case-insensitive match, or unavailable
object metadata quarantines the row. HPA-323 does not ignore unresolved BGM events,
because an unresolved event could be the actual full mix.

Resolved events are grouped by `(audio_object_key, chart_time_sec)`, using exact
floating-point values produced by one timing map. Selection rules:

1. zero groups: quarantine as `bgm_event_missing`;
2. one group: select it;
3. one group containing repeated identical source tokens: select the first source-order
   event and emit `duplicate_bgm_event` as a warning;
4. more than one group: quarantine as `ambiguous_bgm_start`.

The design deliberately does not choose the earliest event, longest file, or a preferred
filename when distinct BGM starts exist. Those heuristics could silently align the
benchmark to the wrong audio revision.

## Audio Metadata

The verified cached source-audio body is inspected with `soundfile.info`.

Record:

- duration in seconds as `frames / samplerate`;
- sample rate;
- channel count;
- frame count.

A zero-frame file, non-positive sample rate, unsupported/undecodable file, size/hash
mismatch, or missing cache body quarantines the row with an actionable reason code.
No full waveform decode is required in this stage.

## Audio-Relative Events and Bounds Policy

For every generic native DTX event:

```text
chart_time_sec = timing_map.time_sec(event)
audio_time_sec = chart_time_sec - selected_bgm_chart_time_sec
```

HPA-323 preserves native lane ID, note ID, measure, fractional position, and source
order. It does not map to canonical classes.

Bounds use one decoded audio frame as numerical tolerance:

```text
frame_tolerance_sec = 1 / source_audio_sample_rate
```

Policy:

- `-frame_tolerance_sec <= audio_time_sec < 0`: clamp to `0.0` and warn;
- `audio_duration_sec < audio_time_sec <= audio_duration_sec + frame_tolerance_sec`:
  clamp to `audio_duration_sec` and warn;
- values below or above those tolerances are excluded from the published event artifact
  and counted as pre-audio or post-audio events;
- non-finite times quarantine the row;
- a row with no in-bounds native events after exclusion is quarantined as
  `no_in_bounds_reference_events`;
- otherwise out-of-bounds exclusions remain visible warnings and counts. HPA-324 owns
  any stricter eligibility threshold by count or proportion.

This policy fixes the reference clock without prematurely deciding model-independent
corpus eligibility beyond obvious unusability.

## Native Reference-Event Artifact

Each timing-ready row publishes one JSONL file under:

```text
<output-dir>/events/<sha256>.jsonl
```

Every line uses `schema_version: "crux.dtx-reference-event/v1"` and contains:

```text
simfile_id
selected_chart_key
selected_chart_content_hash
source_audio_key
source_audio_content_hash
timing_semantics_version
source_order
measure
position
lane_id
note_id
chart_time_sec
audio_time_sec
```

Rows are sorted by `(audio_time_sec, measure, position, lane_id, note_id, source_order)`.
The exact canonical JSONL bytes determine the filename SHA-256. Repeated runs with
identical inputs reuse the same immutable file.

The artifact intentionally preserves non-BGM native DTX events even when HPA-324 later
classifies a lane as control or unmapped. Channel `01`, channel `02`, and tempo channels
never enter this artifact because they are parsed as typed control data.

## Derived Manifest Fields

The output schema is `crux.reference-timing-manifest/v1`. Each row preserves the HPA-322
inventory and selection fields, replaces the old top-level `corpus_version` through the
existing canonical renderer, and adds:

```text
source_manifest_sha256
source_reference_chart_version
timing_semantics_version = crux.dtx-audio-timing/v1
timing_status = ready | quarantined
timing_reason_codes[]
timing_warnings[]
source_audio_key
source_audio_content_hash
source_audio_duration_sec
source_audio_sample_rate
source_audio_channels
source_audio_frames
bgm_event_count
selected_bgm_note_id
selected_bgm_chart_time_sec
reference_events_path
reference_events_sha256
reference_event_count
pre_audio_event_count
post_audio_event_count
```

Selected audio fields and event-artifact fields are `null` for quarantined rows. Counts
are integers and reason-code arrays are deterministically sorted.

The matching `objects[]` entry is enriched with the verified audio `cache_status`,
`sha256`, and `cache_path`. Other object records remain byte-for-byte equivalent at the
JSON value level.

## Error Handling

Fatal command errors:

- malformed or unsupported input manifest;
- duplicate simfile IDs or mixed source identities;
- missing required R2 configuration when selected audio cache misses exist;
- invalid cache index;
- output or manifest publication failure.

Row-level quarantine reasons include:

- `upstream_chart_selection_unavailable`;
- `selected_chart_cache_invalid`;
- `bgm_event_missing`;
- `unresolved_bgm_wav`;
- `unsafe_bgm_audio_path`;
- `source_audio_missing`;
- `source_audio_key_ambiguous`;
- `ambiguous_bgm_start`;
- `source_audio_download_failed`;
- `source_audio_cache_invalid`;
- `source_audio_decode_failed`;
- `non_finite_reference_time`;
- `no_in_bounds_reference_events`.

One bad simfile does not discard timing-ready rows. Operational error messages never
include credentials, signed requests, or source file contents.

## Testing

Focused unit tests cover:

- typed channel `01` parsing and exclusion from generic events;
- channel `02` persistence, replacement, and altered-measure BPM changes;
- one timing map shared by BGM and native events;
- WAV note lookup and safe relative path normalization;
- selected-chart-directory resolution and root compatibility fallback;
- exact-case and unique case-insensitive source-audio matches;
- unresolved and ambiguous BGM cases;
- duplicate identical BGM events versus distinct BGM starts;
- targeted cache fill selecting only exact audio keys and preserving existing HPA-321
  suffix behavior;
- complete-cache reruns that do not construct an R2 store;
- `soundfile.info` duration, frame, sample-rate, and channel metadata;
- frame-tolerance clamps, pre-audio exclusion, post-audio exclusion, and no-event
  quarantine;
- deterministic event JSONL hashing and immutable reuse;
- manifest lineage, object-record cache enrichment, counters, and partial publication;
- CLI exit codes `0`, `1`, and `2`.

One acceptance fixture creates a minimal HPA-322 manifest and cache with:

- a chart whose sticky channel `02` length changes affect its BGM and drum events;
- a referenced `bgm.ogg` cache miss supplied by a fake R2 store;
- a chart with a nested DTX-relative audio path;
- a row with multiple conflicting BGM starts;
- pre-audio and post-audio events.

The real CLI publishes event artifacts and a derived manifest without external network
access.

## Delivery Sequence

1. Add typed BGM parsing and the shared sticky-measure timing map.
2. Add safe BGM path resolution and pure row-level timing decisions.
3. Add exact-key targeted source-audio cache fill using the existing cache machinery.
4. Add audio metadata inspection, bounds handling, and deterministic native event
   artifacts.
5. Add derived-manifest publication and lineage.
6. Wire the CLI and acceptance fixture, then run focused and full validation.

No HPA-324 taxonomy, HPA-326 corpus inference, or HPA-325 scoring implementation belongs
in this PR.