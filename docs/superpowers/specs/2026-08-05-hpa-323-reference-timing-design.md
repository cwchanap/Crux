# HPA-323: Audio-Relative DTX Reference Timing Design

## Context

HPA-322 is complete and merged as PR #10. The merged implementation publishes
`crux.reference-chart-manifest/v1` and establishes the inventory, cache, object-key,
DTX decoding, and schema-validation contracts that HPA-323 should consume.

HPA-323 is therefore no longer waiting on an unknown HPA-322 shape. Its job is now
narrow: turn each selected authoritative DTX chart into immutable, audio-relative native
reference events before corpus inference or scoring begins.

Three timing gaps remain on `main`:

1. Crux resets channel `02` measure length to `1.0` on every unspecified measure;
2. channel `01` still falls through generic note parsing instead of becoming typed BGM
   control data;
3. native DTX events remain in chart time because no selected BGM start is subtracted.

HPA-321 still caches chart-definition files only. HPA-323 must fill the existing cache
for only the exact DTX-referenced source-audio objects without widening global corpus
cache policy.

## Merged HPA-322 Contracts

HPA-323 must reuse the following merged interfaces rather than rebuilding them:

- `parse_manifest_timestamp` from `r2_corpus_models.py`;
- `ManifestRowView`, `manifest_row_view_from_row`, and `inventory_from_manifest_row`
  from `corpus_manifest.py` for embedded HPA-321 payloads;
- `resolve_verified_cache_body` and `read_verified_cache_body` from
  `corpus_cache.py`;
- `is_chart_key` and `is_set_def_key` from `corpus_cache.py`;
- `resolve_inventory_object_key` and `ResolvedObjectKey` from
  `inventory_object_keys.py`;
- `parse_dtx_bytes` and the shared DTXMania decoder from `dtx_parser.py`;
- `render_manifest`, `publish_manifest`, and `publish_latest_manifest` from
  `corpus_manifest.py`.

One merged detail requires a plan correction: `manifest_row_view_from_row` validates the
exact HPA-321 `crux.r2-corpus-manifest/v1` key set. An HPA-322 row contains the extra
selection fields and uses `crux.reference-chart-manifest/v1`, so HPA-323 must **not**
call `manifest_row_view_from_row` directly on an HPA-322 row.

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

The adapter reuses the existing `_validate_reference_row` and selected-chart identity
checks. It does not introduce a new schema parser or object model. The raw validated row
mapping remains available to the timing-manifest loader for lossless pass-through.

## Verified Channel `02` Semantics

Sticky channel `02` remains a required correctness fix. In DTXManiaXG Ver.K commit
`2e7839d93c00ef528407bebdcf829dafb8c8c804`, timing keeps the active `dbBarLength`
after a channel `02` chip. Automatic reset to `1.0` is explicitly conditional on BMS or
BME and does not run for DTX.

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
- Validate HPA-322 rows through one public `ReferenceChartRowView` adapter.
- Read selected DTX bytes with `read_verified_cache_body` and `parse_dtx_bytes`.
- Resolve source audio through `#WAVxx` using `resolve_inventory_object_key`.
- Measure real-corpus BGM group and root-fallback distribution before freezing
  multi-group policy.
- Fill only exact selected source-audio cache misses.
- Inspect source-audio metadata with `soundfile.info` without waveform decode.
- Shift native DTX events into audio time while preserving native identity and bounds
  diagnostics.
- Publish content-addressed native event JSONL and an immutable
  `crux.reference-timing-manifest/v1`.
- Follow the merged schema-golden convention for both new stable artifact schemas.
- Keep HPA-324 taxonomy and final eligibility outside this issue.

## Non-goals

- Selecting or changing the authoritative chart.
- Reimplementing HPA-322 manifest, cache, or key-resolution validation.
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

- `--manifest PATH` is required.
- `--cache-dir PATH` is required.
- `--output-dir PATH` defaults to `artifacts/benchmark/reference-timing`.

A complete cache permits a fully offline run. R2 dependencies and credentials are
resolved only when selected source-audio misses exist.

Exit codes mirror the existing derived-manifest stages:

- `0`: every input row is timing-ready;
- `1`: a derived manifest was published with one or more quarantines;
- `2`: fatal input, cache/configuration, or publication failure.

## Pre-Implementation Corpus Measurement Gate

Before freezing BGM multi-group policy, run a read-only analysis over a real merged
HPA-322 manifest and its cache. This is still useful after HPA-322 because the selected
chart set is now deterministic.

The temporary analysis may call the private HPA-322 `_validate_reference_row` only as a
one-off validation seam; production HPA-323 uses the public adapter described above.
For each selected row:

1. obtain the validated embedded `ManifestRowView`;
2. locate the selected `RemoteObject` by `selected_chart_key`;
3. call `read_verified_cache_body(..., expected_sha256=selected_chart_content_hash)`;
4. parse with `parse_dtx_bytes`;
5. before typed BGM parsing lands, use the existing generic events with
   `lane_id == "01"` as the channel `01` observations;
6. resolve each referenced `#WAVxx` value with `resolve_inventory_object_key` relative
   to the selected-chart directory;
7. after a relative `missing`, measure whether one root-level retry resolves it;
8. group resolved observations by `(audio_object_key, measure, position)`.

The report records:

```text
selected_rows
rows_with_0_bgm_groups
rows_with_1_bgm_group
rows_with_multiple_bgm_groups
rows_with_unresolved_wav
rows_needing_case_insensitive_match
rows_needing_simfile_root_fallback
```

Decision gate:

- exceptional multi-group rows -> retain `ambiguous_bgm_start` quarantine;
- common multi-group rows -> inspect representative authored charts and amend this
  design before implementing BGM selection;
- zero root-fallback rows -> remove the compatibility retry;
- do not choose the earliest group without authored-chart evidence.

No immutable HPA-323 artifact is published until this policy is frozen.

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

`ParsedDtxChart` gains `bgm_events`; channel `01` no longer enters generic events.

`timing.py` gains `DtxTimingMap`, which resolves active measure lengths, measure-start
beats, and tempo points once. BPM, BGM, and native events all use `time_sec(event)`.

Rename without a compatibility alias:

```text
dtx_events_to_timed_events
    -> dtx_events_to_chart_time_events
```

The return type remains `list[BenchmarkEvent]`. Update `render_audio.py`, `runner.py`,
and every test/import. `render_audio` correctly uses chart time because it synthesizes
chart samples from zero. Legacy folder/MIDI scoring remains chart-time and is explicitly
non-authoritative for HPA-325; HPA-325 consumes HPA-323 audio-time artifacts.

### 2. HPA-322 reference-row view

`reference_chart_manifest.py` exposes `ReferenceChartRowView` and
`reference_chart_row_view_from_row` by wrapping the validation HPA-322 already ships.
The adapter also resolves the selected chart object from the validated embedded
inventory when `selection_status == "selected"`.

This is an API exposure, not a new parsing layer. HPA-323 never constructs a parallel
source-object or manifest model.

### 3. BGM and source-audio resolution

`reference_timing.py` receives a parsed chart, timing map, selected chart key, and
`ReferenceChartRowView`.

For each BGM event it resolves the WAV table entry using
`resolve_inventory_object_key`. It maps the shared statuses as follows:

- `exact` / `casefold`: use the supplied `RemoteObject`;
- `invalid_path`: `unsafe_bgm_audio_path`;
- `ambiguous`: `source_audio_key_ambiguous`;
- `missing`: optional measured root fallback, otherwise `source_audio_missing`.

HPA-323 does not normalize separators, walk `..`, enforce prefix containment, or build
its own exact/casefold index. Those semantics are owned by HPA-322.

Group BGM observations by `(remote.key, event.measure, event.position)`, never floating
point time. The preflight measurement decides whether distinct groups quarantine.

### 4. Exact-key source-audio cache fill

`corpus_cache.py` extracts the existing `sync_cache` implementation behind one private
selector and adds:

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

`sync_cache` keeps using the existing `is_selected` policy. The new wrapper selects only
exact requested keys. It does not change `CACHE_PROFILE` or list R2.

### 5. One verification per source body

Selected chart:

- read and verify once through `read_verified_cache_body`;
- parse those exact bytes with `parse_dtx_bytes`.

Selected audio:

- call `resolve_verified_cache_body` once when already cached and retain the path;
- queue only unavailable audio for targeted fill;
- after fill, verify only rows whose inventory was replaced by the fill result.

Already verified audio is not hashed a second time.

### 6. Audio metadata and native event artifacts

Use `soundfile.info` to record duration, sample rate, channels, and frames.

For every native event:

```text
chart_time_sec = timing_map.time_sec(event)
audio_time_sec = chart_time_sec - selected_bgm_chart_time_sec
```

One audio frame is the bounds tolerance. Near-zero and near-duration values clamp;
larger pre/post-audio values are excluded and counted; non-finite values or zero
retained events quarantine.

Publish canonical JSONL under:

```text
<output-dir>/events/<sha256>.jsonl
```

Expose only a thin public `publish_immutable_content(...)` wrapper around the existing
`corpus_manifest._publish_immutable` so event files keep corpus durability and
`ManifestPublicationError` semantics.

The stable event schema is `crux.dtx-reference-event/v1`.

### 7. Derived timing manifest

`reference_timing_manifest.py` reads exact HPA-322 JSONL bytes with canonical strict
JSON handling, validates each row through `reference_chart_row_view_from_row`, requires
unique simfile IDs and one source identity, and verifies the input derived
`corpus_version` with `render_manifest` just as HPA-322 verifies its source manifest.

For selected rows it then performs chart parsing, BGM resolution, optional exact-key
cache fill, audio inspection, event publication, and timing enrichment. Upstream
quarantined rows remain quarantined without chart/audio work.

All HPA-322 row fields are preserved at the JSON value level except the top-level
`corpus_version`, which is replaced by the new derived version. This includes selection
method/reasons/warnings, selected chart cache identity, title, artist, DLEVEL, and
provenance.

Reconciliation is exact:

```text
ready + quarantined = input rows
events_published = ready
```

`ReferenceTimingOutcome` has no separate report artifact:

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

The merged HPA-322 implementation registered
`crux.reference-chart-manifest/v1` in `tests/benchmark/schema_goldens/manifest.json`.
HPA-323 follows that established convention rather than treating its new schemas as an
exception.

Add:

- `tests/benchmark/schema_goldens/crux.dtx-reference-event-v1.jsonl`;
- `tests/benchmark/schema_goldens/crux.reference-timing-manifest-v1.jsonl`;
- both entries to `tests/benchmark/schema_goldens/manifest.json`.

`reference_timing.py` validates the event golden; `reference_timing_manifest.py`
validates a timing-manifest golden containing at least one ready and one quarantined
row, including a valid derived corpus version.

## Derived Manifest Fields

Output schema: `crux.reference-timing-manifest/v1`.

Add to each preserved HPA-322 row:

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

Fatal stage errors:

- malformed or non-canonical HPA-322 manifest;
- unsupported HPA-322 schema;
- duplicate simfile IDs or mixed source identities;
- invalid cache index;
- missing/invalid R2 configuration when selected audio misses exist;
- immutable event or manifest publication failure.

Row-level quarantine reasons:

- `upstream_chart_selection_unavailable`;
- `selected_chart_cache_invalid`;
- `selected_chart_parse_failed`;
- `timing_map_invalid`;
- `bgm_event_missing`;
- `unresolved_bgm_wav`;
- `unsafe_bgm_audio_path`;
- `source_audio_missing`;
- `source_audio_key_ambiguous`;
- `ambiguous_bgm_start` when retained by measured policy;
- `source_audio_download_failed`;
- `source_audio_cache_invalid`;
- `source_audio_decode_failed`;
- `non_finite_reference_time`;
- `no_in_bounds_reference_events`.

## Risk Verification Matrix

| Risk | Required proof |
|---|---|
| HPA-322 row adapter drifts from merged schema | Adapter delegates to existing HPA-322 validator + golden tests |
| Incorrect DTX channel `02` semantics | Primary DTXMania source + sticky/replacement fixtures |
| BGM policy quarantines too much corpus | Preflight distribution + sampled multi-group review |
| HPA-323 forks object-key behavior | All lookup tests use `resolve_inventory_object_key` |
| Timing rename breaks legacy consumers | Full repository suite immediately after Task 1 |
| Existing audio is hashed twice | Already-verified rows bypass post-fill verification |
| Targeted fill mutates unrelated objects | Before/after inventory equality tests |
| Complete cache still requires R2 | Dependency/store factories are not called |
| Row failure aborts corpus | Valid sibling still publishes with exit `1` |
| New schemas drift silently | Both stable artifacts registered in schema goldens |

## Delivery Sequence

0. Measure real BGM layouts using the merged HPA-322 contracts and freeze policy.
1. Add typed BGM events, sticky timing, and explicit clock names; run the full suite.
2. Add the public HPA-322 reference-row view and pure BGM resolution.
3. Add exact-key source-audio cache fill.
4. Add metadata, bounds, immutable native events, and event schema golden.
5. Add one-verification orchestration, derived manifest, and timing schema golden.
6. Wire the CLI and run the complete CI-equivalent verification.
