# HPA-323: Audio-Relative DTX Reference Timing Design

## Context

HPA-322 is complete and merged as PR #10. It publishes
`crux.reference-chart-manifest/v1` and establishes the inventory, cache, object-key,
DTX decoding, and schema-validation contracts HPA-323 should consume.

HPA-323 is unblocked. Its scope is narrow: turn each selected authoritative DTX chart
into immutable, audio-relative native reference events before corpus inference or scoring
begins.

Three timing gaps remain on `main`:

1. channel `02` measure length resets to `1.0` on every unspecified measure;
2. channel `01` still falls through generic note parsing instead of becoming typed BGM
   control data;
3. native DTX events remain in chart time because no selected BGM start is subtracted.

HPA-321/HPA-322 still cache chart-definition files only. HPA-323 must fill the existing
cache for only exact DTX-referenced source-audio objects without widening the global
cache policy.

## Merged HPA-322 Contracts

Reuse these merged interfaces:

- `parse_manifest_timestamp` from `r2_corpus_models.py`;
- `ManifestRowView`, `manifest_row_view_from_row`, and `inventory_from_manifest_row`
  from `corpus_manifest.py` for embedded HPA-321 payloads;
- `resolve_verified_cache_body` and `read_verified_cache_body` from
  `corpus_cache.py`;
- `is_selected`, `is_chart_key`, and `is_set_def_key` from `corpus_cache.py`;
- `resolve_inventory_object_key` and `ResolvedObjectKey` from
  `inventory_object_keys.py`;
- `parse_dtx_bytes` and the shared DTXMania decoder from `dtx_parser.py`;
- `render_manifest`, `publish_manifest`, and `publish_latest_manifest` from
  `corpus_manifest.py`;
- the schema-golden registry in `tests/benchmark/schema_goldens/manifest.json`.

`manifest_row_view_from_row` accepts only the exact HPA-321
`crux.r2-corpus-manifest/v1` key set. HPA-322 rows contain selection fields and use
`crux.reference-chart-manifest/v1`, so production HPA-323 must not pass an HPA-322 row
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
    selected_chart_content_hash: str | None

    @property
    def simfile_id(self) -> int:
        return self.source.inventory.simfile_id


def reference_chart_row_view_from_row(
    row: Mapping[str, object],
) -> ReferenceChartRowView: ...
```

The adapter reuses HPA-322's existing reference-row validation and selected-chart
identity checks. `selected_chart_content_hash` is exposed with a narrowed type because it
is passed as `expected_sha256` when reading the selected chart. `simfile_id` is a
property over the already-typed inventory rather than duplicated storage.

This is API exposure, not a second parser or a parallel manifest model.

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

## Existing Channel `01` Blast Radius

Removing channel `01` from `ParsedDtxChart.events` is behaviorally inert for existing
playable/render paths:

- `render_audio.py` skips events whose lane is not in `DEFAULT_DTX_LANE_MAP`;
- channel `01` is absent from that lane map;
- `map_dtx_events` therefore treats existing channel `01` events only as unmapped
  diagnostics, not playable drum hits.

HPA-323 changes channel `01` from discarded generic metadata into typed BGM control
data; it does not remove an existing playable event.

## Goals

- Parse channel `01` as typed BGM control events.
- Add deterministic source order to native pattern events.
- Build one timing map with sticky DTX channel `02` semantics.
- Rename chart-time APIs so the clock is explicit at every call site.
- Validate HPA-322 rows through `ReferenceChartRowView`.
- Read selected DTX bytes with `read_verified_cache_body` and `parse_dtx_bytes`.
- Resolve source audio through `#WAVxx` using `resolve_inventory_object_key`.
- Use a closed `TimingReasonCode` type for stable reason codes.
- Add exact-key source-audio cache fill without changing `CACHE_PROFILE`.
- Commit a reproducible corpus diagnostic that uses the real HPA-323 row adapter,
  resolver, timing map, and exact-key cache path.
- Measure BGM group distribution, root-fallback need, audio-extension distribution,
  sampled audio decodability, and channel-`02` timing impact before freezing policy.
- Inspect source-audio metadata with `soundfile.info` without waveform decode.
- Shift native DTX events into audio time while preserving native identities and bounds
  diagnostics.
- Publish content-addressed native event JSONL and an immutable timing manifest.
- Follow the merged schema-golden convention for both new stable artifact schemas.
- Preserve every HPA-322 lineage and selection field without overwriting it.
- Keep HPA-324 taxonomy and final eligibility outside this issue.

## Non-goals

- Selecting or changing the authoritative chart.
- Reimplementing HPA-322 row, cache, or key-resolution validation.
- Defining canonical drum classes or duplicate-after-collapse behavior.
- Running OaF, MuScriptor, IDM, separation, inference, or scoring.
- Automatically repairing authored timing.
- Downloading every audio object in the corpus.
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
`artifacts/benchmark/reference-timing`.

The required cache path is deliberate. HPA-322's sibling CLI defaults to
`manifest.parent.parent / "cache"`, which is correct for an HPA-321 manifest under
`r2-corpus/manifests/`, but an HPA-322 input lives under
`reference-charts/manifests/`; inheriting that default would silently point HPA-323 at
`reference-charts/cache` instead of the authoritative `r2-corpus/cache`.

A complete cache permits a fully offline timing build. R2 dependencies and credentials
are resolved only when selected source-audio misses exist.

Exit codes mirror the existing derived-manifest stages:

- `0`: every input row is timing-ready;
- `1`: a derived manifest was published with one or more quarantines;
- `2`: fatal input/configuration/publication failure.

Because upstream HPA-322 quarantines also produce exit `1`, the outcome and CLI summary
include `upstream_quarantined_count`. HPA-323-specific quarantine count is therefore
observable as `quarantined_count - upstream_quarantined_count` without inventing a new
exit-code convention.

## Architecture

### 1. Typed controls and explicit clocks

Add `source_order: int = 0` to `DtxEvent`, add frozen `DtxBgmEvent`, and store channel
`01` events separately in `ParsedDtxChart.bgm_events`.

`timing.py` gains `DtxTimingMap`, extending the existing measure-start, event-beat,
tempo-point, and integration helpers rather than replacing them. It resolves active
measure lengths once and uses them for BPM, BGM, and native events.

Rename without an alias:

```text
dtx_events_to_timed_events
    -> dtx_events_to_chart_time_events
```

The return type remains `list[BenchmarkEvent]`. Update `render_audio.py`, `runner.py`,
and all tests. `render_audio` correctly uses chart time because it synthesizes chart
samples from zero. Legacy folder/MIDI scoring remains explicitly chart-time and is not
the HPA-325 benchmark path.

### 2. Closed timing reason-code contract

Match HPA-322's established convention:

```python
TimingReasonCode = Literal[
    "upstream_chart_selection_unavailable",
    "selected_chart_cache_invalid",
    "selected_chart_parse_failed",
    "timing_map_invalid",
    "bgm_event_missing",
    "unresolved_bgm_wav",
    "unsafe_bgm_audio_path",
    "source_audio_missing",
    "source_audio_key_ambiguous",
    "ambiguous_bgm_start",
    "source_audio_download_failed",
    "source_audio_cache_invalid",
    "source_audio_decode_failed",
    "non_finite_reference_time",
    "no_in_bounds_reference_events",
]
```

The timing-manifest validator derives its closed set with
`frozenset(get_args(TimingReasonCode))`. Producers and tests use the type rather than
free-form strings.

### 3. Reference-chart row view

`reference_chart_manifest.py` exposes `ReferenceChartRowView` by wrapping the merged
HPA-322 validator. A selected view contains the exact selected `RemoteObject` and a
non-null typed selected-chart content hash; a quarantined view contains both as `None`.

Production HPA-323 code does not index the raw HPA-322 mapping for selected chart
identity. The raw validated mapping is retained only by the timing-manifest loader for
lossless field pass-through.

### 4. Policy-neutral BGM reference collection

`reference_timing.py` resolves each BGM WAV reference exclusively through
`resolve_inventory_object_key`:

- `exact` / `casefold` -> use the supplied `RemoteObject`;
- `invalid_path` -> `unsafe_bgm_audio_path`;
- `ambiguous` -> `source_audio_key_ambiguous`;
- `missing` -> optionally try the simfile-root compatibility path, otherwise
  `source_audio_missing`.

Group identity is `(remote.key, measure, position)`, never floating-point time.

The resolver can collect/group references before multi-group policy is frozen. Selection
of one BGM group is a separate pure policy step. This lets the corpus diagnostic call the
same production resolver without maintaining a throwaway parallel implementation.

`ResolvedBgm.used_root_fallback` is **not** part of the pre-policy interface. If the
corpus diagnostic proves root fallback is needed, the final resolver may retain that
branch and communicate it through a deterministic warning. If no row needs it, the
branch and any related field are omitted.

### 5. Exact-key source-audio cache fill

Extract `sync_cache` behind one private selector and add
`sync_explicit_cache_keys(...)`. Existing `sync_cache`, `is_selected`, and
`CACHE_PROFILE` stay unchanged. The new path selects exact requested keys only and
reuses existing locking, conditional GET, hashing, cache-index, and installation logic.

This cache capability is implemented before the corpus diagnostic because HPA-321 and
HPA-322 normally leave audio objects uncached. A diagnostic that attempts `sf.info` on
"cached BGM audio" before exact-key fill exists would produce incomplete evidence.

### 6. Reproducible corpus diagnostic and policy gate

Add a committed tool:

```text
tools/hpa323/analyze_reference_timing.py
```

It calls the real `ReferenceChartRowView`, typed BGM parser, timing map,
`resolve_inventory_object_key`, and `sync_explicit_cache_keys`; it does not reach into
private HPA-322 validators or duplicate path/group logic.

The tool produces a deterministic JSON report under a caller-supplied output path and
records:

```text
selected_rows
upstream_quarantined_rows
rows_with_0_bgm_groups
rows_with_1_bgm_group
rows_with_multiple_bgm_groups
rows_with_unresolved_wav
rows_needing_case_insensitive_match
rows_needing_simfile_root_fallback
bgm_extension_counts
sampled_audio_count
sampled_audio_decodable_count
sampled_audio_undecodable_count
sampled_audio_undecodable_by_extension
charts_with_channel_02
charts_with_multiple_channel_02_changes
max_channel_02_time_delta_sec
channel_02_delta_examples[]
multi_group_examples[]
```

Audio decodability sampling uses exact candidate audio keys. Missing sampled bodies are
filled through `sync_explicit_cache_keys`, then opened through the verified cache path and
probed with `soundfile.info`. Catch the same expected failure family already used by the
repo's render path: `OSError`, `RuntimeError`, `ValueError`, and
`sf.LibsndfileError`.

The report records extensions from authored `#WAVxx` values, but format support is
proved by `sf.info`, not inferred from the extension.

Channel-`02` blast radius is measured without copying the legacy timing engine. For each
chart, build the corrected timing map normally. For comparison, construct a diagnostic
chart copy whose `measure_lengths` explicitly sets every measure to
`chart.measure_lengths.get(measure, 1.0)`; running the same new timing map over that copy
reproduces the old per-measure reset semantics. Compare native event times by
`source_order` and report the maximum delta.

Policy gate:

- exceptional multi-group rows -> retain `ambiguous_bgm_start` quarantine;
- common multi-group rows -> inspect representative authored charts and amend the design
  before finalizing BGM selection;
- zero root-fallback rows -> remove the root fallback entirely;
- material audio undecodability -> decide supported-format/quarantine policy before
  native event publication;
- material channel-`02` deltas -> inspect representative affected charts before landing
  the timing change;
- never choose the earliest BGM group only to increase yield.

The report is a reproducible diagnostic, not a stable schema artifact and not part of
`ReferenceTimingOutcome`.

### 7. One verification per source body

Selected DTX is read once with `read_verified_cache_body` and parsed with
`parse_dtx_bytes`.

Selected audio is verified once with `resolve_verified_cache_body` when already cached.
Only unavailable audio enters targeted fill; only rows whose inventory changes during
fill are verified afterward.

### 8. Audio metadata and native event artifacts

Use `soundfile.info` for duration, sample rate, channels, and frames, with the same
expected exception family used by `render_audio.py` for unreadable audio.

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

### 9. Derived timing manifest

`reference_timing_manifest.py` reads canonical HPA-322 JSONL, validates every row with
`reference_chart_row_view_from_row`, requires unique simfile IDs and one source identity,
and reproduces the input derived `corpus_version` through `render_manifest`.

All HPA-322 fields are preserved at the JSON-value level except the top-level
`corpus_version`, which becomes the new timing-manifest version. Preserve HPA-322's
existing HPA-321 lineage:

```text
source_manifest_sha256
source_corpus_version
```

Add separate immediate lineage:

```text
source_reference_chart_manifest_sha256
source_reference_chart_version
```

where the first hashes the exact HPA-322 input bytes and the second stores the HPA-322
input `corpus_version`.

`ReferenceTimingOutcome` mirrors the existing 0/1/2 convention but adds upstream
visibility:

```python
@dataclass(frozen=True)
class ReferenceTimingOutcome:
    status: Literal["complete", "partial", "failed"]
    exit_code: Literal[0, 1, 2]
    manifest: PublishedManifest | None
    ready_count: int
    quarantined_count: int
    upstream_quarantined_count: int
    events_published: int
```

No report artifact or `report_path` belongs in this outcome.

## Stable Schemas

Follow the convention established by merged HPA-322. Add and register:

- `tests/benchmark/schema_goldens/crux.dtx-reference-event-v1.jsonl`;
- `tests/benchmark/schema_goldens/crux.reference-timing-manifest-v1.jsonl`.

`reference_timing.py` validates the event golden. `reference_timing_manifest.py`
validates a timing-manifest golden with one ready and one quarantined row and a valid
derived corpus version. Timing reason codes are validated against
`TimingReasonCode`.

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

Row quarantine reasons are exactly `TimingReasonCode`. `ambiguous_bgm_start` remains in
the type but is emitted only if the corpus diagnostic freezes conservative multi-group
quarantine as the policy.

## Risk Verification Matrix

| Risk | Required proof |
|---|---|
| HPA-322 row fed directly to HPA-321 parser | Production loader uses `ReferenceChartRowView` |
| Selected-chart safety identity leaks back to raw dicts | Typed `selected_chart_content_hash` on the row view |
| Immediate lineage overwrites HPA-321 lineage | Dedicated `source_reference_chart_*` fields |
| Incorrect channel `02` semantics | Primary source + fixtures + corpus delta report |
| BGM policy quarantines too much corpus | Committed diagnostic + sampled multi-group review |
| Source audio is unsupported by libsndfile | Extension counts + sampled `sf.info` results |
| HPA-323 forks object-key behavior | All lookups use `resolve_inventory_object_key` |
| Timing rename breaks legacy consumers | Full repository suite immediately after timing change |
| Existing audio is hashed twice | Already-verified rows bypass post-fill verification |
| Targeted fill mutates unrelated objects | Before/after inventory equality tests |
| Complete cache still requires R2 | Dependency/store factories are not called |
| Upstream quarantine hides HPA-323 health | `upstream_quarantined_count` in outcome/CLI |
| Stable artifact contract drifts | Closed `TimingReasonCode` + both schema goldens |
| One orchestration commit becomes unreviewable | Split pure manifest contract from R2/cache orchestration |

## Delivery Sequence

1. Add typed BGM events, sticky timing, and explicit chart-time naming; run full tests.
2. Expose `ReferenceChartRowView`, add `TimingReasonCode`, and implement policy-neutral
   BGM grouping/resolution.
3. Add exact-key source-audio cache fill.
4. Add the committed corpus diagnostic; measure BGM layouts, root fallback, audio
   decodability, and channel-`02` impact; freeze policy.
5. Add metadata, bounds, immutable native events, and event schema golden.
6. Add pure timing-manifest loading/lineage/row rendering and the timing schema golden.
7. Add cache/R2 orchestration, event publication, and the end-to-end acceptance fixture.
8. Wire the CLI, including `upstream_quarantined_count`, and run CI-equivalent
   verification.
