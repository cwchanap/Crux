# HPA-323: Audio-Relative DTX Reference Timing Design

## Context

HPA-323 is the next Crux benchmark task after HPA-322. HPA-322 selects one
authoritative DTX chart per simfile and publishes
`crux.reference-chart-manifest/v1`. HPA-323 turns those selected charts into
trustworthy, audio-relative native reference events before any corpus inference or
scoring runs.

The current timing path has three correctness gaps:

1. channel `02` measure lengths reset to `1.0` on the following measure instead of
   remaining active until superseded;
2. channel `01` is parsed as a generic note event and later disappears as unmapped
   instead of becoming an explicit BGM control event;
3. native DTX events remain in chart time because no selected BGM start is
   subtracted.

HPA-321 also deliberately caches only `set.def`, `.dtx`, and `.txt`. HPA-323 must
therefore fill the cache for only the exact DTX-referenced source-audio objects. It
must not broaden the HPA-321 default cache-selection policy or download every audio
object in the corpus.

Implementation starts only after HPA-322 is merged. HPA-323 consumes the merged
manifest and cache contracts rather than guessing around an active branch.

## Goals

- Make channel `02` sticky beginning with its own measure and continuing until a
  later channel `02` value supersedes it.
- Parse each non-zero channel `01` token as typed BGM control data with measure,
  fractional position, WAV note ID, and deterministic source order.
- Use one timing map for BPM controls, BGM events, and native DTX events.
- Load manifest object records once into the existing `RemoteObject` and
  `SimfileInventory` types.
- Reuse one shared cache-body verifier for HPA-322 selected charts and HPA-323
  selected audio.
- Resolve source audio from the selected DTX `#WAVxx` reference, relative to the
  selected DTX object key, with a narrow simfile-root compatibility fallback.
- Prefer exact R2 object-key matches, then one unique case-insensitive match.
- Cache only exact selected source-audio keys through the existing HPA-321 cache
  machinery.
- Inspect audio metadata with `soundfile.info` without decoding full waveforms.
- Shift native DTX events from chart time to source-audio time.
- Exclude events outside decoded audio bounds while preserving counts, reason
  codes, and native identities.
- Publish content-addressed per-simfile native event JSONL plus an immutable
  `crux.reference-timing-manifest/v1`.
- Keep raw DTX-derived timing separate from the existing auto-alignment diagnostic.
- Keep the implementation small, sequential, and easy to rerun locally.

## Non-goals

- Selecting or changing the authoritative chart.
- Defining canonical drum classes, lane collapse, duplicate-after-collapse
  behavior, or final corpus eligibility; HPA-324 owns those policies.
- Running OaF, MuScriptor, IDM, source separation, inference, or scoring.
- Choosing a BGM start by filename, earliest event, longest file, or another
  heuristic when distinct starts conflict.
- Repairing malformed DTX files or authored timing.
- Downloading every audio object.
- Embedding full event arrays in the manifest.
- Adding a database, service, workflow engine, plugin framework, or new
  general-purpose concurrency layer.
- Preserving backward compatibility for incorrect chart-time artifacts.

## Chosen Approach

Add one derived reference-timing stage after HPA-322:

```text
HPA-322 manifest
    -> load existing inventory types
    -> verify selected chart body
    -> parse chart and build sticky timing map
    -> resolve one BGM/audio identity
    -> fill exact selected audio cache misses
    -> verify audio body and inspect metadata
    -> build bounded audio-relative native events
    -> publish event JSONL and derived manifest
```

This keeps remote inventory, chart selection, timing, taxonomy, inference, and
scoring as separate reviewable stages. It also prevents HPA-326 from running
expensive corpus inference against an invalid reference clock.

Applying the BGM shift inside legacy `run_score_midi` is rejected because other
consumers would continue using chart time. Embedding events in the manifest is
rejected because it would turn the lineage manifest into a large event store.

## Operator Interface

```bash
uv run crux benchmark build-reference-timing \
  --manifest artifacts/benchmark/reference-charts/manifests/<sha256>.jsonl \
  --cache-dir artifacts/benchmark/r2-corpus/cache \
  --output-dir artifacts/benchmark/reference-timing
```

Options:

- `--manifest PATH` is required and must be a
  `crux.reference-chart-manifest/v1` JSONL file.
- `--cache-dir PATH` is optional. For the standard layout where the input lives at
  `<benchmark-root>/reference-charts/manifests/<sha>.jsonl`, the default is exactly
  `<benchmark-root>/r2-corpus/cache`, computed as:

  ```python
  manifest.parent.parent.parent / "r2-corpus" / "cache"
  ```

  This is intentionally different from HPA-322's input-manifest cache default.
- `--output-dir PATH` defaults to
  `artifacts/benchmark/reference-timing`.

A complete cache permits a fully offline run. R2 dependencies, credentials, and a
store are resolved only when selected source-audio cache misses exist.

The command writes progress and warnings to stderr and one JSON object to stdout:

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

- `0`: every input row is timing-ready;
- `1`: a derived manifest was published and at least one row was quarantined;
- `2`: a fatal input, configuration, cache-index, or publication error prevented a
  usable result.

## Architecture

### 1. Typed BGM Events and Shared Timing Map

`src/benchmark/dtx_parser.py` gains:

```python
@dataclass(frozen=True)
class DtxBgmEvent:
    chart_id: str
    measure: int
    position: float
    note_id: str
    source_order: int
```

`ParsedDtxChart` gains `bgm_events: list[DtxBgmEvent]`. Channel `01` never enters
the generic `events` list.

`DtxEvent` gains `source_order: int = 0` so native event order is deterministic
without breaking direct test construction.

`src/benchmark/timing.py` gains an immutable `DtxTimingMap` containing resolved
measure lengths, measure-start beats, and tempo points. Its
`time_sec(event)` method accepts generic events, BPM events, and BGM events.

Sticky measure lengths are resolved once:

```text
active_length = 1.0
for each measure in order:
    if measure has channel 02:
        active_length = channel 02 value
    resolved_lengths[measure] = active_length
```

The resolved length affects both the measure start and the event's fractional
position within the measure. Existing base BPM, channel `03`, channel `08`,
fractional-position, and same-beat ordering behavior remains covered by parity
tests.

Legacy `dtx_events_to_timed_events` delegates to the same timing map but remains a
chart-time compatibility wrapper. HPA-323's new stage owns the BGM shift.

### 2. Shared Manifest Inventory and Cache-Body Verification

HPA-323 must not introduce another source-object dataclass. Manifest objects are
loaded into the existing HPA-321 types:

- `RemoteObject`;
- `SimfileInventory`;
- `SyncError`.

Add `src/benchmark/manifest_inventory.py` with focused adapters:

```python
def parse_manifest_timestamp(value: object) -> datetime: ...

def inventory_from_manifest_row(
    row: Mapping[str, object],
) -> SimfileInventory: ...

def resolve_verified_cache_body(
    cache_dir: Path,
    remote: RemoteObject,
    *,
    expected_sha256: str | None = None,
) -> Path: ...
```

`inventory_from_manifest_row` parses one row once. It converts ISO UTC timestamps
to timezone-aware `datetime`, validates cache-status values, reconstructs
row-level `SyncError` records, attaches object-specific errors by `object_key`, and
returns one `SimfileInventory`.

`resolve_verified_cache_body` is the only selected-source body verifier used by
HPA-322 and HPA-323. It requires:

- `cache_status == "verified"`;
- a lowercase SHA-256;
- a non-empty relative `cache_path`;
- a regular file below `cache_dir`;
- exact byte size;
- exact SHA-256;
- optional equality with `expected_sha256`.

When HPA-322 merges with a private `_CachedObject` or private verifier, HPA-323's
first integration step extracts that logic into this module and refactors
`reference_chart_selection.py` to use it. No third verifier is added.

### 3. BGM and Source-Audio Resolution

`src/benchmark/reference_timing.py` receives a parsed chart, timing map, selected
chart key, object prefix, and `tuple[RemoteObject, ...]`.

For every typed BGM event it:

1. resolves `event.note_id` through `chart.wav_table`;
2. normalizes backslashes to `/`;
3. rejects empty, absolute, drive-prefixed, or escaping paths;
4. resolves relative to the selected chart directory;
5. tries an exact key match;
6. otherwise accepts one unique `casefold()` match with a warning;
7. only after a relative miss, retries from the simfile root;
8. returns the exact `RemoteObject` member from the inventory.

Because resolution selects from `inventory.objects`, a later selected key missing
from that inventory is an internal invariant violation, not a normal row state.

Resolved BGM events are grouped by discrete source identity:

```text
(audio_object.key, event.measure, event.position)
```

`chart_time_sec` is computed once from the selected event and stored as data, not
used as the group identity.

Selection rules:

1. no BGM events: `bgm_event_missing`;
2. any unresolved BGM token: quarantine with its specific reason;
3. one discrete group: select its lowest-source-order event;
4. repeated tokens in that group: warn `duplicate_bgm_event`;
5. more than one discrete group: `ambiguous_bgm_start`.

No filename, earliest-time, longest-duration, or preferred-name heuristic may
resolve a true conflict.

### 4. Exact-Key Source-Audio Cache Fill

`src/benchmark/corpus_cache.py` extracts its existing worker behind a selector and
adds:

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

The wrapper selects `remote.key in selected_keys`. Existing `sync_cache` and
`is_selected` behavior remains unchanged.

The orchestration layer passes only inventories containing selected audio misses,
not the full corpus. Each inventory remains complete, so non-selected object
records survive the cache rebuild unchanged. Returned inventories are merged back
by `simfile_id`.

The stage never broadens HPA-321's default suffix policy and never lists R2.
Duplicate selected keys are deduplicated before the call.

### 5. Audio Metadata and Native Event Artifacts

The verified source-audio body is inspected with `soundfile.info`. Record:

- duration as `frames / samplerate`;
- sample rate;
- channel count;
- frame count.

A zero-frame file, non-positive metadata, unsupported format, or decode failure
quarantines the row.

For every generic native DTX event:

```text
chart_time_sec = timing_map.time_sec(event)
audio_time_sec = chart_time_sec - selected_bgm_chart_time_sec
```

Bounds use one audio frame of tolerance:

```text
frame_tolerance_sec = 1 / source_audio_sample_rate
```

Policy:

- small negative values within one frame clamp to `0.0`;
- small post-duration values within one frame clamp to the duration;
- larger pre-audio and post-audio values are excluded and counted;
- non-finite values quarantine the row;
- zero retained events quarantine as `no_in_bounds_reference_events`;
- otherwise exclusions remain warnings for HPA-324 to assess.

Each ready row publishes canonical JSONL at:

```text
<output-dir>/events/<sha256>.jsonl
```

Every event uses `crux.dtx-reference-event/v1` and preserves:

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

Events are sorted deterministically. The existing corpus immutable-byte publisher
is exposed through a thin public wrapper so event artifacts keep HPA-321
`ManifestPublicationError` and durability semantics.

### 6. Manifest Orchestration

`src/benchmark/reference_timing_manifest.py` performs two row passes.

#### Load and first pass

- Read exact input bytes once and compute `source_manifest_sha256`.
- Validate one schema, corpus version, bucket, endpoint hash, and unique simfile
  IDs.
- Build one `SimfileInventory` per row through `inventory_from_manifest_row`.
- Preserve upstream quarantined rows without chart parsing or R2 access.
- Verify the selected chart through `resolve_verified_cache_body`.
- Parse the selected DTX and build the timing map.
- Quarantine row-local parse failures as `selected_chart_parse_failed`.
- Quarantine row-local timing failures as `timing_map_invalid`.
- Resolve the BGM against `inventory.objects`.
- Collect exact selected audio keys and only inventories containing cache misses.

#### Cache fill

When no selected audio body misses, do not import optional R2 dependencies, resolve
credentials, or create a store.

When misses exist:

- resolve the existing R2 dependency and config;
- require the config bucket and endpoint hash to match the manifest;
- load the existing cache index;
- create and validate the store;
- hold the existing cache writer lock;
- call `sync_explicit_cache_keys` with miss inventories and exact keys;
- merge returned inventories by `simfile_id`.

#### Second pass and publication

For every resolved row:

- retrieve the selected audio `RemoteObject` from the current inventory;
- verify it with the same `resolve_verified_cache_body`;
- inspect metadata;
- build bounded native events;
- publish event JSONL;
- enrich only that object's `cache_status`, `sha256`, and `cache_path` in the
  derived row.

Then render and publish the immutable derived manifest through the existing
canonical helpers.

Reconciliation is exact:

```text
ready + quarantined = input rows
events_published = ready
```

## Derived Manifest

Output schema: `crux.reference-timing-manifest/v1`.

Each row preserves HPA-322 inventory and selection values and adds:

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

Selected-audio and event-artifact fields are `null` for quarantined rows.
Reason-code and warning arrays are deterministically ordered.

## Error Handling

Fatal command errors:

- malformed or unsupported input manifest;
- duplicate simfile IDs;
- mixed corpus, bucket, or endpoint identities;
- invalid cache index;
- missing or invalid R2 configuration when cache misses exist;
- output or manifest publication failure.

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
- `ambiguous_bgm_start`;
- `source_audio_download_failed`;
- `source_audio_cache_invalid`;
- `source_audio_decode_failed`;
- `non_finite_reference_time`;
- `no_in_bounds_reference_events`.

One bad simfile never discards ready rows. Source contents, credentials, and signed
requests never appear in diagnostics.

## Testing

Focused tests cover:

- typed channel `01` and deterministic native source order;
- sticky channel `02`, replacement, altered-measure BPM, and shared BGM/event
  timing;
- manifest timestamp parsing and `RemoteObject`/`SimfileInventory` reconstruction;
- object and row `SyncError` reconstruction;
- shared cache-body verification for chart and audio;
- HPA-322 selector parity after extracting the verifier;
- DTX-relative and root-fallback audio paths;
- exact and unique case-insensitive object matches;
- discrete BGM grouping and true multi-start ambiguity;
- exact-key cache fill with non-selected objects unchanged;
- only miss inventories passed to cache sync;
- returned cache fields merged only into selected objects;
- complete-cache runs that never create an R2 store;
- selected-chart parse and timing-map failures quarantining only their rows;
- audio metadata and frame-tolerance bounds;
- deterministic event and manifest hashes;
- exact default cache path from a standard HPA-322 manifest location;
- exit `0`, `1`, and `2` behavior.

The acceptance fixture includes a sticky-measure chart, nested audio path, one
selected audio cache miss, one conflicting BGM row, one parse-failure row, and
pre/post-audio events. It invokes the real CLI with a fake R2 store and no external
network.

## Risks and Mitigations

### Inventory-to-cache round-trip drift

Risk: timestamp, error, cache-status, or object ordering changes during
manifest-to-dataclass-to-manifest conversion.

Mitigation: round-trip unit fixtures compare all non-selected object JSON values and
assert only the selected audio cache fields change.

### Sticky timing regression

Risk: old tests or commands accidentally depend on the incorrect per-measure reset.

Mitigation: replace the incorrect fixture, retain parity tests for BPM and
fractional positions, and assert the legacy wrapper delegates to the shared timing
map without applying a BGM shift.

### Unnecessary R2 dependency

Risk: an offline rerun still imports boto3 or requests credentials.

Mitigation: a complete-cache orchestration test injects a dependency check and store
factory that fail immediately if called.

### Partial versus fatal outcome confusion

Risk: one malformed chart or bad timing map aborts the corpus.

Mitigation: parse/timing/decode failures are row quarantines; schema, mixed source
identity, cache-index, required config, and publication failures are fatal. Tests
assert reconciliation and exit codes.

### Wrong default cache root

Risk: simplifying the default to the HPA-322 formula points at
`reference-charts/cache`.

Mitigation: CLI request-wiring and acceptance tests assert the exact
`<benchmark-root>/r2-corpus/cache` path.

## Delivery Sequence

1. Typed BGM parsing and sticky shared timing map.
2. Shared manifest-inventory adapter, shared cache verifier, and BGM resolution
   using `RemoteObject`.
3. Exact-key cache selector extension.
4. Audio metadata, bounds, and immutable native events.
5. Manifest orchestration, row quarantine, and derived publication.
6. CLI, exact default path, acceptance fixture, and full Ruff-based validation.

No HPA-324 taxonomy, HPA-325 scoring, or HPA-326 inference belongs in this work.
