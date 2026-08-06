# HPA-323: Audio-Relative DTX Reference Timing Design

## Context

HPA-323 follows HPA-322 on the Crux benchmark critical path. HPA-322 selects one
authoritative DTX chart and publishes `crux.reference-chart-manifest/v1`. HPA-323
turns each selected chart into immutable, audio-relative native reference events before
corpus inference or scoring begins.

Three correctness gaps exist today:

1. Crux resets channel `02` measure length to `1.0` on every unspecified measure;
2. channel `01` becomes a generic event and later disappears as unmapped;
3. native events remain in chart time because the selected BGM start is not
   subtracted.

HPA-321 caches chart-definition files only, so HPA-323 must fill the cache for the
exact DTX-referenced source audio without widening the global cache policy.

Implementation begins only after HPA-322 merges its shared manifest-row,
cache-body, and inventory object-key contracts.

## Verified Channel `02` Semantics

Sticky channel `02` is not an assumption. In DTXManiaXG Ver.K commit
`2e7839d93c00ef528407bebdcf829dafb8c8c804`, timing calculation keeps the active
`dbBarLength` after a channel `02` chip. The automatic reset to `1.0` is explicitly
guarded by `e種別 == BMS || e種別 == BME`; it does not run for DTX.

Primary source:

- `CDTX.cs`, timing loop and `case 0x02`:
  <https://github.com/kairera0467/DTXManiaXG_VerK/blob/2e7839d93c00ef528407bebdcf829dafb8c8c804/DTXMania%E3%83%97%E3%83%AD%E3%82%B8%E3%82%A7%E3%82%AF%E3%83%88/%E3%82%B3%E3%83%BC%E3%83%89/%E3%82%B9%E3%82%B3%E3%82%A2%E3%80%81%E6%9B%B2/CDTX.cs>

Therefore HPA-323 implements DTX semantics: a channel `02` value is active from its
measure until another channel `02` value supersedes it. BMS/BME behavior is outside
this DTX benchmark stage.

## Goals

- Parse channel `01` as typed BGM control events.
- Add deterministic source order to native pattern events.
- Build one timing map with sticky DTX channel `02` semantics.
- Rename chart-time APIs so the clock is explicit at every call site.
- Consume HPA-322's `parse_manifest_timestamp`, `manifest_row_view_from_row`,
  `inventory_from_manifest_row`, `resolve_verified_cache_body`, and
  `resolve_inventory_object_key` contracts.
- Resolve source audio through the selected DTX `#WAVxx` table and R2 inventory.
- Measure real-corpus BGM group and root-fallback distribution before freezing
  ambiguity policy.
- Fill only exact selected source-audio cache misses.
- Inspect metadata with `soundfile.info` without waveform decode.
- Shift native events into audio time, preserve native identity, and report bounds.
- Publish content-addressed native event JSONL and
  `crux.reference-timing-manifest/v1`.
- Keep HPA-324 taxonomy and final eligibility outside this issue.

## Non-goals

- Selecting or changing the authoritative chart.
- Defining canonical drum classes or duplicate-after-collapse behavior.
- Running OaF, MuScriptor, IDM, separation, inference, or scoring.
- Automatically repairing authored timing.
- Downloading every audio object.
- Embedding full event arrays in the manifest.
- Reimplementing HPA-322 row parsing, cache validation, or object-key matching.
- Adding a database, service, workflow engine, plugin system, or new concurrency layer.
- Preserving the ambiguous name `dtx_events_to_timed_events`.

## Operator Interface

```bash
uv run crux benchmark build-reference-timing \
  --manifest artifacts/benchmark/reference-charts/manifests/<sha256>.jsonl \
  --cache-dir artifacts/benchmark/r2-corpus/cache \
  --output-dir artifacts/benchmark/reference-timing
```

All three paths are explicit:

- `--manifest PATH` is required;
- `--cache-dir PATH` is required;
- `--output-dir PATH` defaults to `artifacts/benchmark/reference-timing`.

Requiring `--cache-dir` avoids coupling correctness to a particular placement of the
input manifest. A complete cache permits a fully offline run. R2 dependencies and
credentials are resolved only when selected source-audio misses exist.

Exit codes:

- `0`: every input row is timing-ready;
- `1`: a derived manifest was published with one or more quarantines;
- `2`: fatal input, cache/configuration, or publication failure.

## Pre-Implementation Corpus Measurement Gate

Before implementing BGM selection policy, run a read-only analysis over the merged
HPA-322 manifest and cache. Publish a local report containing:

```text
selected_rows
rows_with_0_bgm_groups
rows_with_1_bgm_group
rows_with_multiple_bgm_groups
rows_with_unresolved_wav
rows_needing_case_insensitive_match
rows_needing_simfile_root_fallback
```

A BGM group uses discrete identity `(audio_object_key, measure, position)`. The report
also lists a bounded sample of multi-group rows with note IDs, object keys, measures,
positions, and source order.

Every candidate `#WAVxx` value is resolved with HPA-322's
`resolve_inventory_object_key` relative to the selected chart directory. The analysis
may perform a second call with the simfile-root directory after a relative `missing`
result so it can measure root fallback. It must not implement a separate path
normalizer, containment check, exact matcher, or casefold matcher.

Decision gate:

- if zero/one-group behavior dominates and multi-group rows are exceptional, retain
  conservative quarantine for distinct groups;
- if multi-group rows are common, inspect representative authored charts and revise
  this design before implementing selection;
- remove the second root-level resolver call if the report proves it unnecessary;
- do not automatically adopt an earliest-group rule merely because it increases yield.

No immutable timing artifact is published until the policy is frozen from evidence.

## Architecture

### 1. Typed controls and explicit clocks

`DtxEvent` gains `source_order: int = 0`.

`dtx_parser.py` gains:

```python
@dataclass(frozen=True)
class DtxBgmEvent:
    chart_id: str
    measure: int
    position: float
    note_id: str
    source_order: int
```

Channel `01` enters `ParsedDtxChart.bgm_events` and never enters generic events.

`timing.py` gains `DtxTimingMap`, which resolves active measure lengths, measure starts,
and tempo points once. BPM, BGM, and native events all use `time_sec(event)`.

Rename:

```text
dtx_events_to_timed_events
    -> dtx_events_to_chart_time_events
```

Update `render_audio.py`, `runner.py`, and tests. No compatibility alias is retained.
`render_audio` correctly needs chart time because it synthesizes chart samples from
zero. Legacy `runner.py` remains chart-time scoring and is explicitly documented as
non-authoritative for HPA-325; HPA-325 consumes HPA-323 audio-time artifacts instead.

The new reference stage exposes clearly named audio-time construction through
`build_audio_relative_events`.

### 2. Shared HPA-322 contracts

HPA-323 imports:

- `parse_manifest_timestamp` from `r2_corpus_models.py`;
- `manifest_row_view_from_row` and `inventory_from_manifest_row` from
  `corpus_manifest.py`;
- `resolve_verified_cache_body` from `corpus_cache.py`;
- `resolve_inventory_object_key` and `ResolvedObjectKey` from
  `inventory_object_keys.py`.

`ManifestRowView` supplies inventory, provenance, cache profile, endpoint, bucket, and
discovery identity in one validated read. HPA-323 must not parse those fields again.

No `manifest_inventory.py`, parallel source-object type, private path resolver, or
hand-rolled filesystem verifier belongs in HPA-323.

### 3. BGM and source-audio resolution

`reference_timing.py` receives a parsed chart, timing map, selected chart key,
`ManifestRowView`, and the policy frozen by the measurement gate.

For each BGM event it:

1. resolves `note_id` through `chart.wav_table`;
2. calls `resolve_inventory_object_key` relative to the selected chart directory;
3. maps `invalid_path`, `ambiguous`, and `missing` to HPA-323 reason codes;
4. when the measured policy retains root fallback and the relative result is
   `missing`, calls the same helper with the simfile-root directory;
5. returns the exact `RemoteObject` supplied by the shared resolver.

HPA-323 does not normalize separators, walk `..`, check prefix containment, search
exact keys, or build casefold indexes itself. Those semantics are frozen by HPA-322.

The preflight measurement determines whether the second root lookup remains and
whether multiple discrete groups quarantine. No filename, duration, or alphabetical
heuristic selects source audio.

### 4. Exact-key cache fill

`corpus_cache.py` extracts its existing `sync_cache` body behind a selector and adds:

```python
def sync_explicit_cache_keys(
    simfiles: tuple[SimfileInventory, ...],
    store: R2ObjectStore,
    index: CacheIndexStore,
    config: R2Config,
    selected_keys: frozenset[str],
    item_progress: Callable[[int, int, int], None] | None = None,
) -> CacheSyncResult: ...
```

Existing `sync_cache` and `is_selected` behavior remains unchanged. The orchestration
passes complete inventories only for rows with misses and merges returned inventories by
`simfile_id`.

### 5. One verification per source body

First pass:

- verify the selected chart once and parse it;
- resolve source audio;
- call `resolve_verified_cache_body` for the audio once;
- store the verified path when successful;
- queue only non-verified audio rows for cache fill.

After cache fill, call `resolve_verified_cache_body` only for rows whose inventory was
replaced by the merge. Already verified rows are not hashed a second time.

### 6. Audio metadata and native events

Use `soundfile.info` to record duration, sample rate, channels, and frames. A zero-frame
or undecodable body quarantines the row.

For every native event:

```text
chart_time_sec = timing_map.time_sec(event)
audio_time_sec = chart_time_sec - selected_bgm_chart_time_sec
```

One audio frame is the numerical bounds tolerance. Near-zero/near-duration values clamp;
larger pre/post-audio values are excluded and counted; non-finite values or zero retained
events quarantine.

Publish canonical JSONL under `events/<sha256>.jsonl` using the existing immutable byte
publisher. Preserve chart/audio hashes, lane, note, measure, position, source order, and
both clocks.

### 7. Derived manifest orchestration

`reference_timing_manifest.py`:

1. reads exact HPA-322 bytes and validates source identity through
   `manifest_row_view_from_row`;
2. preserves upstream quarantines;
3. verifies and parses selected charts;
4. maps chart parse failures to `selected_chart_parse_failed`;
5. maps timing construction failures to `timing_map_invalid`;
6. resolves/queues audio through the shared object-key contract;
7. performs targeted cache fill only when needed;
8. inspects audio and publishes native events;
9. publishes the derived manifest through existing canonical helpers.

Reconciliation:

```text
ready + quarantined = input rows
events_published = ready
```

## Derived Manifest Fields

Output schema: `crux.reference-timing-manifest/v1`.

Add:

```text
source_manifest_sha256
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

Fatal:

- malformed/unsupported input manifest;
- duplicate IDs or mixed source identities;
- invalid cache index;
- missing/invalid R2 configuration when misses exist;
- publication failure.

Row quarantines:

- `upstream_chart_selection_unavailable`;
- `selected_chart_cache_invalid`;
- `selected_chart_parse_failed`;
- `timing_map_invalid`;
- `bgm_event_missing`;
- `unresolved_bgm_wav`;
- `unsafe_bgm_audio_path` for shared resolver `invalid_path`;
- `source_audio_missing` for shared resolver `missing`;
- `source_audio_key_ambiguous` for shared resolver `ambiguous`;
- `ambiguous_bgm_start` when retained by the measured policy;
- `source_audio_download_failed`;
- `source_audio_cache_invalid`;
- `source_audio_decode_failed`;
- `non_finite_reference_time`;
- `no_in_bounds_reference_events`.

## Risk Verification Matrix

| Risk | Required proof |
|---|---|
| Incorrect DTX channel `02` semantics | Primary DTXMania source reference plus sticky/replacement fixtures |
| BGM policy quarantines an unacceptable share | Pre-implementation corpus distribution report and sampled multi-group review |
| HPA-323 forks object-key behavior | Tests call the HPA-322 resolver and contain no private normalization/matching helper |
| Timing change breaks legacy consumers | Rename clock API and run the full repository suite in the timing task |
| Existing cache is unnecessarily rehashed | Test complete-cache rows are verified once and never enter post-fill verification |
| Targeted fill mutates unrelated objects | Compare non-selected object records before/after merge |
| Complete cache still requires R2 | Dependency/store factories must not be called |
| Row exception aborts the corpus | Valid sibling publishes while parse/timing failures produce exit `1` |
| Legacy scoring silently becomes authoritative | Explicit chart-time name/comment; HPA-325 consumes audio-time artifacts only |

## Delivery Sequence

0. Measure real BGM group/fallback distribution using the shared resolver and freeze the policy.
1. Add typed BGM events, sticky timing, and explicit clock names; run the full suite.
2. Implement BGM policy by mapping shared resolver outcomes, not by reimplementing key resolution.
3. Add exact-key audio cache fill.
4. Add metadata, bounds, and immutable native events.
5. Add one-verification orchestration and derived manifest publication.
6. Wire the required-path CLI and acceptance fixture.
