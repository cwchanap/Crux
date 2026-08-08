# HPA-323: Audio-Relative DTX Reference Timing Design

## Context

HPA-322 is complete and merged as PR #10. It publishes
`crux.reference-chart-manifest/v1` and establishes the inventory, cache, object-key,
DTX decoding, and schema-validation contracts HPA-323 should consume.

HPA-323 is now unblocked. Its scope is narrow: turn each selected authoritative DTX
chart into immutable, audio-relative native reference events before corpus inference or
scoring begins.

Three timing gaps remain on `main`:

1. channel `02` measure length resets to `1.0` on every unspecified measure;
2. channel `01` still falls through generic note parsing instead of becoming typed BGM
   control data;
3. native DTX events remain in chart time because no selected BGM start is subtracted.

HPA-321 still caches chart-definition files only. HPA-323 must fill the existing cache
for only the exact DTX-referenced source-audio objects without widening global cache
policy.

## Merged HPA-322 Contracts

Reuse these merged interfaces:

- `parse_manifest_timestamp` from `r2_corpus_models.py`;
- `ManifestRowView`, `manifest_row_view_from_row`, and `inventory_from_manifest_row`
  from `corpus_manifest.py` for embedded HPA-321 payloads;
- `resolve_verified_cache_body` and `read_verified_cache_body` from
  `corpus_cache.py`;
- `resolve_inventory_object_key` and `ResolvedObjectKey` from
  `inventory_object_keys.py`;
- `parse_dtx_bytes` from `dtx_parser.py`;
- `render_manifest`, `publish_manifest`, and `publish_latest_manifest` from
  `corpus_manifest.py`.

A merged detail changes the old plan: `manifest_row_view_from_row` accepts only the exact
HPA-321 `crux.r2-corpus-manifest/v1` key set. HPA-322 rows contain selection fields and
use `crux.reference-chart-manifest/v1`. Production HPA-323 must not pass an HPA-322 row
directly to the HPA-321 parser.

Expose one small public adapter in `reference_chart_manifest.py`:

```python
@dataclass(frozen=True)
class ReferenceChartRowView:
    source: ManifestRowView
    corpus_version: str
    selection_status: Literal["selected", "quarantined"]
    selection_reason_codes: tuple[str, ...]
    selection_warnings: tuple[str, ...]
    selected_chart: RemoteObject | None


def reference_chart_row_view_from_row(
    row: Mapping[str, object],
) -> ReferenceChartRowView: ...
```

It reuses the existing HPA-322 `_validate_reference_row` and selected-chart identity
checks. This is an API exposure, not a new parser or data model.

## Verified Channel `02` Semantics

Sticky channel `02` remains required. In DTXManiaXG Ver.K commit
`2e7839d93c00ef528407bebdcf829dafb8c8c804`, the active `dbBarLength` remains after a
channel `02` chip. Automatic reset to `1.0` is conditional on BMS/BME and does not run
for DTX.

Primary source:

- `CDTX.cs`, timing loop and `case 0x02`:
  <https://github.com/kairera0467/DTXManiaXG_VerK/blob/2e7839d93c00ef528407bebdcf829dafb8c8c804/DTXMania%E3%83%97%E3%83%AD%E3%82%B8%E3%82%A7%E3%82%AF%E3%83%88/%E3%82%B3%E3%83%BC%E3%83%89/%E3%82%B9%E3%82%B3%E3%82%A2%E3%80%81%E6%9B%B2/CDTX.cs>

HPA-323 therefore implements DTX semantics: channel `02` is active from its measure
until another channel `02` supersedes it. BMS/BME timing is outside this benchmark.

## Goals

- Parse channel `01` as typed BGM control events.
- Add deterministic source order to native pattern events.
- Build one timing map with sticky DTX channel `02` semantics.
- Rename chart-time APIs so the clock is explicit at every call site.
- Validate HPA-322 rows through `ReferenceChartRowView`.
- Read selected DTX bytes with `read_verified_cache_body` and `parse_dtx_bytes`.
- Resolve source audio through `#WAVxx` using `resolve_inventory_object_key`.
- Measure real-corpus BGM group/root-fallback distribution before freezing multi-group
  policy.
- Fill only exact selected source-audio cache misses.
- Inspect audio metadata with `soundfile.info` without waveform decode.
- Shift native DTX events into audio time while preserving native identities and bounds
  diagnostics.
- Publish content-addressed native event JSONL and an immutable timing manifest.
- Follow the merged schema-golden convention for both new stable artifact schemas.
- Preserve all HPA-322 lineage and selection fields without overwriting them.
- Keep HPA-324 taxonomy and final eligibility outside this issue.

## Non-goals

- Selecting or changing the authoritative chart.
- Reimplementing HPA-322 row, cache, or key-resolution validation.
- Defining canonical drum classes or duplicate-after-collapse behavior.
- Running OaF, MuScriptor, IDM, separation, inference, or scoring.
- Automatically repairing authored timing.
- Downloading every audio object.
- Embedding full event arrays in the timing manifest.
- Adding a database, service, workflow engine, plugin system, or new concurrency layer.
- Preserving the ambiguous API name `dtx_events_to_timed_events`.

## Operator Interface

```bash
uv run crux benchmark build-reference-timing \
  --manifest artifacts/benchmark/reference-charts/manifests/<sha256>.jsonl \
  --cache-dir artifacts/benchmark/r2-corpus/cache \
  --output-dir artifacts/benchmark/reference-timing
```

`--manifest` and `--cache-dir` are required. `--output-dir` defaults to
`artifacts/benchmark/reference-timing`. A complete cache permits a fully offline run;
R2 dependencies and credentials are resolved only when selected source-audio misses
exist.

Exit codes mirror the existing derived-manifest stages: `0` complete, `1` partial with a
published manifest, `2` fatal input/configuration/publication failure.

## Pre-Implementation Corpus Measurement Gate

Before freezing BGM multi-group policy, run a read-only analysis over a real merged
HPA-322 manifest and cache.

The temporary analysis may call private `_validate_reference_row` only as a one-off
validation seam. Production code uses the public adapter above.

For each selected row:

1. obtain the embedded validated `ManifestRowView`;
2. locate the selected `RemoteObject` by `selected_chart_key`;
3. read the exact chart bytes with `read_verified_cache_body`, checking
   `selected_chart_content_hash`;
4. parse with `parse_dtx_bytes`;
5. before typed BGM parsing lands, use current generic events with `lane_id == "01"` as
   the observations;
6. resolve each `#WAVxx` path with `resolve_inventory_object_key` relative to the
   selected-chart directory;
7. after relative `missing`, measure whether one simfile-root retry resolves it;
8. group resolved observations by `(audio_object_key, measure, position)`.

Record zero/one/multiple group rows, unresolved WAVs, casefold matches, and root-fallback
matches. If multiple groups are common, inspect representative authored charts and amend
this design before implementing selection. Do not adopt an earliest-group heuristic only
to increase yield.

## Architecture

### 1. Typed controls and explicit clocks

Add `source_order: int = 0` to `DtxEvent`, add frozen `DtxBgmEvent`, and store channel
`01` events separately in `ParsedDtxChart.bgm_events`.

`timing.py` gains `DtxTimingMap`, which resolves active measure lengths, measure starts,
and tempo points once. BPM, BGM, and native events all use `time_sec(event)`.

Rename without an alias:

```text
dtx_events_to_timed_events
    -> dtx_events_to_chart_time_events
```

The return type remains `list[BenchmarkEvent]`. Update `render_audio.py`, `runner.py`,
and all tests. `render_audio` correctly uses chart time because it synthesizes chart
samples from zero. Legacy folder/MIDI scoring remains explicitly chart-time and is not
the HPA-325 benchmark path.

### 2. Reference-chart row view

`reference_chart_manifest.py` exposes `ReferenceChartRowView` by wrapping the merged
HPA-322 validator. A selected view contains the exact selected `RemoteObject`; a
quarantined view contains `selected_chart=None`.

No `manifest_inventory.py` or parallel source-object type is introduced.

### 3. BGM and source-audio resolution

`reference_timing.py` resolves each BGM WAV reference exclusively through
`resolve_inventory_object_key`:

- `exact` / `casefold` -> use the supplied `RemoteObject`;
- `invalid_path` -> `unsafe_bgm_audio_path`;
- `ambiguous` -> `source_audio_key_ambiguous`;
- `missing` -> optional measured root retry, otherwise `source_audio_missing`.

HPA-323 does not implement separator normalization, traversal handling, prefix checks,
or an exact/casefold index. Group identity is `(remote.key, measure, position)`, never
floating-point time.

### 4. Exact-key source-audio cache fill

Extract `sync_cache` behind one private selector and add
`sync_explicit_cache_keys(...)`. Existing `sync_cache`, `is_selected`, and
`CACHE_PROFILE` stay unchanged. The new path selects exact requested keys only and
reuses existing locking, conditional GET, hashing, cache-index, and installation logic.

### 5. One verification per source body

Selected DTX is read once with `read_verified_cache_body` and parsed with
`parse_dtx_bytes`.

Selected audio is verified once with `resolve_verified_cache_body` when already cached.
Only unavailable audio enters targeted fill; only rows whose inventory changes during
fill are verified afterward.

### 6. Audio metadata and native event artifacts

Use `soundfile.info` for duration, sample rate, channels, and frames.

For every native event:

```text
chart_time_sec = timing_map.time_sec(event)
audio_time_sec = chart_time_sec - selected_bgm_chart_time_sec
```

Use one audio frame as bounds tolerance. Clamp only near the boundaries; exclude larger
pre/post events and count them. Non-finite times or zero retained events quarantine.

Publish canonical JSONL at `events/<sha256>.jsonl`. Expose only a thin
`publish_immutable_content(...)` wrapper around `corpus_manifest._publish_immutable` so
event artifacts retain corpus durability and `ManifestPublicationError` semantics.

Stable event schema: `crux.dtx-reference-event/v1`.

### 7. Derived timing manifest

`reference_timing_manifest.py` reads canonical HPA-322 JSONL, validates every row with
`reference_chart_row_view_from_row`, requires unique simfile IDs and one source identity,
and reproduces the input derived `corpus_version` through `render_manifest`.

All HPA-322 fields are preserved at the JSON-value level except the top-level
`corpus_version`, which becomes the new timing-manifest version. In particular, preserve
HPA-322's existing:

```text
source_manifest_sha256
source_corpus_version
```

Those fields identify HPA-321 lineage and must not be overwritten. Add separate immediate
lineage fields:

```text
source_reference_chart_manifest_sha256
source_reference_chart_version
```

where the first hashes the exact HPA-322 input bytes and the second stores the HPA-322
input `corpus_version`.

`ReferenceTimingOutcome` has no report artifact:

```python
@dataclass(frozen=True)
class ReferenceTimingOutcome:
    status: Literal["complete", "partial", "failed"]
    exit_code: Literal[0, 1, 2]
    manifest: PublishedManifest | None
    ready_count: int
    quarantined_count: int
    events_published: int
```

## Stable Schemas

Follow the convention established by merged HPA-322. Add and register:

- `tests/benchmark/schema_goldens/crux.dtx-reference-event-v1.jsonl`;
- `tests/benchmark/schema_goldens/crux.reference-timing-manifest-v1.jsonl`.

`reference_timing.py` validates the event golden. `reference_timing_manifest.py`
validates a timing-manifest golden with one ready and one quarantined row and a valid
derived corpus version.

## Derived Timing Fields

Output schema: `crux.reference-timing-manifest/v1`.

Add to the preserved HPA-322 row:

```text
source_reference_chart_manifest_sha256
source_reference_chart_version
timing_semantics_version = crux.dtx-audio-timing/v1
timing_status
timing_reason_codes[]
timing_warnings[]
source_audio_key
source_audio_content_hash
source_audio_duration_sec
source_audio_sample_rate
source_audio_channels
source_audio_frames
bgm_event_count
bgm_group_count
selected_bgm_note_id
selected_bgm_chart_time_sec
reference_events_path
reference_events_sha256
reference_event_count
pre_audio_event_count
post_audio_event_count
```

## Error Handling

Fatal stage errors include malformed/non-canonical HPA-322 input, unsupported schema,
duplicate/mixed source identity, invalid cache index, missing R2 configuration when
needed, and immutable publication failure.

Row quarantine reasons remain:

```text
upstream_chart_selection_unavailable
selected_chart_cache_invalid
selected_chart_parse_failed
timing_map_invalid
bgm_event_missing
unresolved_bgm_wav
unsafe_bgm_audio_path
source_audio_missing
source_audio_key_ambiguous
ambiguous_bgm_start
source_audio_download_failed
source_audio_cache_invalid
source_audio_decode_failed
non_finite_reference_time
no_in_bounds_reference_events
```

`ambiguous_bgm_start` is retained only if the measurement gate supports that policy.

## Risk Verification Matrix

| Risk | Required proof |
|---|---|
| HPA-322 row fed directly to HPA-321 parser | Production loader uses `ReferenceChartRowView` |
| Immediate lineage overwrites HPA-321 lineage | Dedicated `source_reference_chart_*` fields |
| Incorrect channel `02` semantics | Primary DTXMania source + sticky/replacement fixtures |
| BGM policy quarantines too much corpus | Preflight distribution + sampled multi-group review |
| HPA-323 forks object-key behavior | All lookups use `resolve_inventory_object_key` |
| Timing rename breaks legacy consumers | Full repository suite immediately after timing change |
| Existing audio is hashed twice | Already-verified rows bypass post-fill verification |
| Targeted fill mutates unrelated objects | Before/after inventory equality tests |
| Complete cache still requires R2 | Dependency/store factories are not called |
| Stable artifact contract drifts | Event and timing-manifest schema goldens |

## Delivery Sequence

0. Measure real BGM layouts from the merged HPA-322 output and freeze policy.
1. Add typed BGM events, sticky timing, and explicit chart-time naming; run full tests.
2. Expose `ReferenceChartRowView` and implement pure BGM resolution.
3. Add exact-key source-audio cache fill.
4. Add metadata, bounds, immutable native events, and event schema golden.
5. Add one-verification orchestration, derived timing manifest, and timing schema golden.
6. Wire the CLI and run CI-equivalent verification.
