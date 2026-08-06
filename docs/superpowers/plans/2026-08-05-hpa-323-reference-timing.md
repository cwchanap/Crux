# HPA-323 Audio-Relative DTX Reference Timing Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Consume the immutable HPA-322 reference-chart manifest, correct DTX timing semantics, resolve and cache the DTX-referenced full-mix audio, and publish immutable audio-relative native reference-event artifacts for HPA-324.

**Architecture:** Extend the parser with typed BGM events and deterministic source order, replace ad hoc timing helpers with one shared sticky-measure timing map, and add a focused reference-timing stage. The stage resolves exact source-audio objects from manifest metadata, fills only missing selected audio bodies through the existing R2 cache, then publishes content-addressed per-song event JSONL and a derived immutable manifest.

**Tech Stack:** Python 3.12, dataclasses, pathlib/PurePosixPath, JSONL, hashlib, Click, soundfile, the existing optional R2/boto3 adapter, pytest.

## Global Constraints

- Start implementation only after HPA-322 is merged; consume its actual `crux.reference-chart-manifest/v1` contract.
- Channel `02` is sticky beginning with its own measure and remains active until superseded.
- Channel `01` is typed BGM control data and must never enter native drum-event artifacts.
- Resolve source audio from the DTX `#WAVxx` reference; never hard-code `bgm.ogg`.
- Prefer exact object-key matches, then one unique case-insensitive match.
- Resolve relative to the selected chart directory first; use simfile-root fallback only after a relative miss.
- Cache only exact selected source-audio keys; do not broaden HPA-321's default suffix policy.
- Preserve native DTX lane, note, measure, position, and source-order identities. HPA-324 owns canonical mapping and final eligibility.
- Publish timing semantics as `crux.dtx-audio-timing/v1`.
- Raw benchmark timing uses the DTX-derived audio clock. Existing auto-alignment remains a separately labelled diagnostic.
- Keep processing sequential after the targeted cache fill. Do not add a database, service, workflow framework, or new general concurrency layer.
- Use TDD, focused commits, and the repository's `uv run` commands.

---

## File Map

### Create

- `src/benchmark/reference_timing.py` — BGM resolution, source-audio metadata, bounds handling, and native event rendering.
- `src/benchmark/reference_timing_manifest.py` — HPA-322 manifest loading, targeted cache orchestration, event publication, derived manifest, counters, and outcome.
- `tests/benchmark/test_reference_timing.py` — BGM path, ambiguity, audio metadata, bounds, and deterministic-event tests.
- `tests/benchmark/test_reference_timing_manifest.py` — input validation, cache orchestration, lineage, counters, and publication tests.
- `tests/benchmark/test_reference_timing_acceptance.py` — end-to-end CLI fixture with a fake R2 store.

### Modify

- `src/benchmark/models.py` — add deterministic source order to `DtxEvent`.
- `src/benchmark/dtx_parser.py` — parse typed channel `01` events.
- `tests/benchmark/test_dtx_parser.py` — typed BGM and source-order tests.
- `src/benchmark/timing.py` — add `DtxTimingMap` and sticky channel `02` semantics.
- `tests/benchmark/test_timing.py` — persistence, replacement, altered-measure BPM, and BGM parity tests.
- `src/benchmark/corpus_cache.py` — expose exact-key cache selection while preserving default HPA-321 behavior.
- `tests/benchmark/test_corpus_cache.py` — exact-key cache tests and suffix-policy regression.
- `src/benchmark/corpus_manifest.py` — expose the existing immutable byte publisher through a public wrapper.
- `tests/benchmark/test_corpus_manifest.py` — immutable-content wrapper tests.
- `src/cli/benchmark.py` — add `build-reference-timing`.
- `tests/test_cli_benchmark.py` — CLI help, wiring, summary, and exit-code tests.

---

### Task 1: Parse typed BGM events and build one sticky-measure timing map

**Files:**
- Modify: `src/benchmark/models.py`
- Modify: `src/benchmark/dtx_parser.py`
- Modify: `src/benchmark/timing.py`
- Modify: `tests/benchmark/test_dtx_parser.py`
- Modify: `tests/benchmark/test_timing.py`

**Interfaces:**
- Produces: `DtxEvent.source_order: int` with default `0`.
- Produces: `DtxBgmEvent(chart_id, measure, position, note_id, source_order)`.
- Produces: `ParsedDtxChart.bgm_events: list[DtxBgmEvent]`.
- Produces: `DtxTimingMap.time_sec(event: DtxEvent | DtxBpmEvent | DtxBgmEvent) -> float`.
- Produces: `build_dtx_timing_map(chart: ParsedDtxChart) -> DtxTimingMap`.
- Consumed by: Tasks 2, 4, and 5.

- [ ] **Step 1: Write failing typed-channel tests**

Append to `tests/benchmark/test_dtx_parser.py`:

```python
def test_channel_01_is_typed_bgm_not_generic_event() -> None:
    chart = parse_dtx_text(
        "#WAV01: bgm.ogg\n#00101: 0100\n#00111: 0001\n",
        chart_id="song",
    )

    assert [(e.measure, e.position, e.note_id) for e in chart.bgm_events] == [
        (1, 0.0, "01")
    ]
    assert [(e.lane_id, e.note_id) for e in chart.events] == [("11", "01")]


def test_pattern_events_receive_monotonic_source_order() -> None:
    chart = parse_dtx_text(
        "#00111: 0102\n#00101: 0304\n#00112: 0500\n",
        chart_id="song",
    )

    orders = [e.source_order for e in chart.events]
    orders.extend(e.source_order for e in chart.bgm_events)
    assert sorted(orders) == [0, 1, 2, 3, 4]
```

- [ ] **Step 2: Verify the parser tests fail**

```bash
uv run pytest tests/benchmark/test_dtx_parser.py -k "channel_01 or source_order" -q
```

Expected: missing `bgm_events` and `source_order`, and channel `01` remains generic.

- [ ] **Step 3: Add typed BGM and source-order records**

In `src/benchmark/models.py`:

```python
@dataclass(frozen=True)
class DtxEvent:
    chart_id: str
    measure: int
    position: float
    lane_id: str
    note_id: str
    source_order: int = 0
```

In `src/benchmark/dtx_parser.py`:

```python
@dataclass(frozen=True)
class DtxBgmEvent:
    chart_id: str
    measure: int
    position: float
    note_id: str
    source_order: int
```

Add `bgm_events: list[DtxBgmEvent] = field(default_factory=list)` to
`ParsedDtxChart`.

- [ ] **Step 4: Parse pattern events through one helper**

Replace the generic-only note helper with:

```python
def _parse_pattern_events(
    chart_id: str,
    measure: int,
    channel: str,
    value: str,
    next_source_order: int,
) -> tuple[list[DtxEvent], list[DtxBgmEvent], int]:
    generic: list[DtxEvent] = []
    bgm: list[DtxBgmEvent] = []
    chunks = _chunks(value)
    for index, note_id in enumerate(chunks):
        if note_id == "00":
            continue
        position = index / len(chunks)
        if channel == "01":
            bgm.append(DtxBgmEvent(chart_id, measure, position, note_id, next_source_order))
        else:
            generic.append(
                DtxEvent(
                    chart_id,
                    measure,
                    position,
                    channel,
                    note_id,
                    next_source_order,
                )
            )
        next_source_order += 1
    return generic, bgm, next_source_order
```

Keep pattern source order separate from BPM source order. Sort generic events by
`(measure, position, lane_id, note_id, source_order)` and BGM events by
`(measure, position, note_id, source_order)`.

- [ ] **Step 5: Run all parser tests**

```bash
uv run pytest tests/benchmark/test_dtx_parser.py -q
```

Expected: all pass, including HPA-322 decoder and DLEVEL tests after merge.

- [ ] **Step 6: Replace the incorrect measure-reset fixture**

Update `tests/benchmark/test_timing.py`:

```python
def test_measure_length_persists_until_superseded() -> None:
    chart = parse_dtx_text(
        "\n".join(
            [
                "#BPM: 120",
                "#00102: 0.5",
                "#00111: 01",
                "#00211: 01",
                "#00311: 01",
                "#00402: 1.0",
                "#00411: 01",
                "#00511: 01",
            ]
        ),
        "song",
    )

    timed = dtx_events_to_timed_events(chart)
    assert [e.time_sec for e in timed] == [2.0, 3.0, 4.0, 5.0, 7.0]


def test_bpm_change_inside_sticky_short_measure_uses_resolved_length() -> None:
    chart = parse_dtx_text(
        "#BPM: 120\n#BPM01: 60\n#00102: 0.5\n#00208: 0001\n#00311: 01\n",
        "song",
    )

    timed = dtx_events_to_timed_events(chart)
    assert timed[0].time_sec == 4.5


def test_bgm_and_generic_event_share_one_timing_map() -> None:
    chart = parse_dtx_text(
        "#BPM: 120\n#00102: 0.5\n#00201: 01\n#00211: 0001\n",
        "song",
    )

    timing = build_dtx_timing_map(chart)
    assert timing.time_sec(chart.bgm_events[0]) == 3.0
    assert timing.time_sec(chart.events[0]) == 3.5
```

- [ ] **Step 7: Verify timing tests fail**

```bash
uv run pytest tests/benchmark/test_timing.py -q
```

Expected: sticky-measure assertions fail and `build_dtx_timing_map` is missing.

- [ ] **Step 8: Implement the shared timing map**

```python
@dataclass(frozen=True)
class DtxTimingMap:
    resolved_measure_lengths: tuple[float, ...]
    measure_start_beats: tuple[float, ...]
    tempo_points: tuple[tuple[float, float, float], ...]

    def time_sec(self, event: DtxEvent | DtxBpmEvent | DtxBgmEvent) -> float:
        length = self.resolved_measure_lengths[event.measure]
        beat = self.measure_start_beats[event.measure] + event.position * length * 4.0
        return _time_at_beat(beat, list(self.tempo_points))
```

Resolve sticky lengths with:

```python
def _measure_lengths_by_measure(chart: ParsedDtxChart) -> list[float]:
    active = 1.0
    lengths: list[float] = []
    for measure in range(_max_measure(chart) + 2):
        if measure in chart.measure_lengths:
            active = chart.measure_lengths[measure]
        if active <= 0:
            raise ValueError(f"measure {measure} has non-positive length")
        lengths.append(active)
    return lengths
```

Extend `_max_measure` with BGM events. Make measure-start, BPM-event, BGM-event, and
generic-event time calculations consume the same resolved sequence.

- [ ] **Step 9: Delegate the legacy wrapper to the timing map**

Keep `dtx_events_to_timed_events` for old commands, but build its chart-time values via
`build_dtx_timing_map`. Include `source_order` in metadata. Do not apply the BGM shift in
this legacy wrapper; the new manifest stage owns the audio clock.

- [ ] **Step 10: Run focused checks and commit**

```bash
uv run pytest tests/benchmark/test_dtx_parser.py tests/benchmark/test_timing.py -q
uv run ruff check src/benchmark/models.py src/benchmark/dtx_parser.py src/benchmark/timing.py tests/benchmark/test_dtx_parser.py tests/benchmark/test_timing.py
uv run black --check src/benchmark/models.py src/benchmark/dtx_parser.py src/benchmark/timing.py tests/benchmark/test_dtx_parser.py tests/benchmark/test_timing.py
git add src/benchmark/models.py src/benchmark/dtx_parser.py src/benchmark/timing.py tests/benchmark/test_dtx_parser.py tests/benchmark/test_timing.py
git commit -m "fix: derive sticky DTX timing controls"
```

---

### Task 2: Resolve one authoritative BGM event and source-audio object

**Files:**
- Create: `src/benchmark/reference_timing.py`
- Create: `tests/benchmark/test_reference_timing.py`

**Interfaces:**
- Consumes: `ParsedDtxChart`, `DtxBgmEvent`, and `DtxTimingMap`.
- Produces: `SourceObject`.
- Produces: `ResolvedBgm(event, chart_time_sec, audio_object, raw_event_count, warnings)`.
- Produces: `BgmResolution(status, reason_codes, warnings, resolved)`.
- Produces: `resolve_bgm_reference(...) -> BgmResolution`.
- Consumed by: Task 5.

- [ ] **Step 1: Write failing path-resolution tests**

Create `tests/benchmark/test_reference_timing.py`:

```python
def test_resolve_bgm_uses_selected_chart_directory() -> None:
    chart = parse_dtx_text("#WAV01: audio/song.ogg\n#00101: 01\n", "song")
    resolution = resolve_bgm_reference(
        chart,
        build_dtx_timing_map(chart),
        selected_chart_key="42/charts/real.dtx",
        object_prefix="42/",
        objects=(source_object("42/charts/audio/song.ogg"),),
    )
    assert resolution.status == "resolved"
    assert resolution.resolved.audio_object.key == "42/charts/audio/song.ogg"


def test_resolve_bgm_uses_root_fallback_only_after_relative_miss() -> None:
    chart = parse_dtx_text("#WAV01: bgm.ogg\n#00101: 01\n", "song")
    resolution = resolve_bgm_reference(
        chart,
        build_dtx_timing_map(chart),
        selected_chart_key="42/charts/real.dtx",
        object_prefix="42/",
        objects=(source_object("42/bgm.ogg"),),
    )
    assert resolution.resolved.audio_object.key == "42/bgm.ogg"
    assert "source_audio_root_fallback" in resolution.warnings
```

Add tests for unknown WAV ID, empty WAV value, absolute path, traversal above the
simfile prefix, missing object, ambiguous case-insensitive matches, zero BGM events,
duplicate identical BGM events, and two distinct BGM start groups.

- [ ] **Step 2: Verify tests fail**

```bash
uv run pytest tests/benchmark/test_reference_timing.py -k bgm -q
```

Expected: module import failure.

- [ ] **Step 3: Add domain records**

```python
@dataclass(frozen=True)
class SourceObject:
    key: str
    size: int
    etag: str
    etag_is_weak: bool
    last_modified: str
    content_type: str | None
    cache_status: str | None = None
    sha256: str | None = None
    cache_path: str | None = None


@dataclass(frozen=True)
class ResolvedBgm:
    event: DtxBgmEvent
    chart_time_sec: float
    audio_object: SourceObject
    raw_event_count: int
    warnings: tuple[str, ...]


@dataclass(frozen=True)
class BgmResolution:
    status: Literal["resolved", "quarantined"]
    reason_codes: tuple[str, ...]
    warnings: tuple[str, ...]
    resolved: ResolvedBgm | None
```

- [ ] **Step 4: Implement safe object-key resolution**

Normalize backslashes to `/`. Reject empty, absolute, drive-prefixed, or escaping paths.
Resolve against `PurePosixPath(selected_chart_key).parent`, then against
`object_prefix` only after a relative miss.

For each lookup:

1. exact key match;
2. one unique `casefold()` match with `source_audio_case_fallback` warning;
3. multiple casefold matches -> `source_audio_key_ambiguous`;
4. no match -> try root fallback or return `source_audio_missing`.

Never search arbitrary directories by basename.

- [ ] **Step 5: Implement WAV lookup and group selection**

Resolve every channel `01` event through `chart.wav_table[event.note_id]`. Any unresolved
event quarantines the row; do not silently use another BGM event.

Group resolved events by `(audio_object.key, timing.time_sec(event))`:

- zero events -> `bgm_event_missing`;
- one group -> select the lowest-source-order event;
- repeated events in that group -> add `duplicate_bgm_event` warning;
- multiple groups -> `ambiguous_bgm_start`.

- [ ] **Step 6: Run checks and commit**

```bash
uv run pytest tests/benchmark/test_reference_timing.py -k bgm -q
uv run ruff check src/benchmark/reference_timing.py tests/benchmark/test_reference_timing.py
uv run black --check src/benchmark/reference_timing.py tests/benchmark/test_reference_timing.py
git add src/benchmark/reference_timing.py tests/benchmark/test_reference_timing.py
git commit -m "feat: resolve DTX referenced source audio"
```

---

### Task 3: Cache only exact selected source-audio keys

**Files:**
- Modify: `src/benchmark/corpus_cache.py`
- Modify: `tests/benchmark/test_corpus_cache.py`

**Interfaces:**
- Produces: `sync_explicit_cache_keys(simfiles, store, index, config, selected_keys, item_progress=None) -> CacheSyncResult`.
- Preserves: existing `sync_cache(...)` and `is_selected(...)` behavior.
- Consumed by: Task 5.

- [ ] **Step 1: Write failing exact-key tests**

Add a fixture with `42/real.dtx`, `42/bgm.ogg`, and `42/preview.ogg`. Select only
`42/bgm.ogg`, then assert:

```python
assert by_key["42/bgm.ogg"].cache_status == "verified"
assert by_key["42/real.dtx"].cache_status is None
assert by_key["42/preview.ogg"].cache_status is None
assert store.opened_keys == ["42/bgm.ogg"]
```

Also assert:

```python
assert is_selected("42/real.dtx") is True
assert is_selected("42/set.def") is True
assert is_selected("42/bgm.ogg") is False
```

Test an empty selected-key set and reject a set containing `""`.

- [ ] **Step 2: Verify tests fail**

```bash
uv run pytest tests/benchmark/test_corpus_cache.py -k "explicit_cache or default_policy" -q
```

Expected: missing `sync_explicit_cache_keys`.

- [ ] **Step 3: Extract one selector-driven worker**

Keep `sync_cache`'s signature and delegate to:

```python
def _sync_cache_selected(
    simfiles: tuple[SimfileInventory, ...],
    store: R2ObjectStore,
    index: CacheIndexStore,
    config: R2Config,
    dry_run: bool,
    selector: Callable[[RemoteObject], bool],
    item_progress: Callable[[int, int, int], None] | None = None,
) -> CacheSyncResult:
    ...
```

Use `selector` for both `total_selected` and per-object selection. Do not modify cache
locking, download validation, hashing, or installation.

- [ ] **Step 4: Add the explicit-key wrapper**

```python
def sync_explicit_cache_keys(
    simfiles: tuple[SimfileInventory, ...],
    store: R2ObjectStore,
    index: CacheIndexStore,
    config: R2Config,
    selected_keys: frozenset[str],
    item_progress: Callable[[int, int, int], None] | None = None,
) -> CacheSyncResult:
    if "" in selected_keys:
        raise ValueError("selected cache keys must be non-empty")
    return _sync_cache_selected(
        simfiles,
        store,
        index,
        config,
        False,
        selector=lambda remote: remote.key in selected_keys,
        item_progress=item_progress,
    )
```

- [ ] **Step 5: Run checks and commit**

```bash
uv run pytest tests/benchmark/test_corpus_cache.py -q
uv run ruff check src/benchmark/corpus_cache.py tests/benchmark/test_corpus_cache.py
uv run black --check src/benchmark/corpus_cache.py tests/benchmark/test_corpus_cache.py
git add src/benchmark/corpus_cache.py tests/benchmark/test_corpus_cache.py
git commit -m "feat: cache explicit benchmark objects"
```

---

### Task 4: Inspect source audio and build bounded native events

**Files:**
- Modify: `src/benchmark/reference_timing.py`
- Modify: `tests/benchmark/test_reference_timing.py`
- Modify: `src/benchmark/corpus_manifest.py`
- Modify: `tests/benchmark/test_corpus_manifest.py`

**Interfaces:**
- Produces: `SourceAudioMetadata(duration_sec, sample_rate, channels, frames)`.
- Produces: `NativeReferenceEvent` and `ReferenceEventBuild`.
- Produces: `inspect_source_audio(path: Path) -> SourceAudioMetadata`.
- Produces: `build_audio_relative_events(...) -> ReferenceEventBuild`.
- Produces: `render_reference_event_jsonl(events) -> bytes`.
- Produces: `publish_immutable_content(path, content, expected_sha256) -> None`.
- Consumed by: Task 5.

- [ ] **Step 1: Write failing audio-metadata tests**

```python
def test_inspect_source_audio_returns_frame_metadata(tmp_path: Path) -> None:
    path = tmp_path / "source.wav"
    soundfile.write(path, np.zeros(8000, dtype=np.float32), 8000)

    metadata = inspect_source_audio(path)
    assert metadata == SourceAudioMetadata(1.0, 8000, 1, 8000)
```

Add undecodable and zero-frame cases. They must raise
`SourceAudioDecodeError("source_audio_decode_failed")`.

- [ ] **Step 2: Write failing audio-relative bounds tests**

Cover:

- subtraction of selected BGM chart time;
- one-frame negative time clamped to `0.0`;
- earlier negative time excluded and counted;
- one-frame post-duration time clamped to duration;
- later post-duration time excluded and counted;
- non-finite time -> `non_finite_reference_time`;
- no retained native events -> `no_in_bounds_reference_events`.

Use `frame_tolerance_sec = 1.0 / sample_rate`.

- [ ] **Step 3: Verify tests fail**

```bash
uv run pytest tests/benchmark/test_reference_timing.py -k "source_audio or audio_relative" -q
```

Expected: missing interfaces.

- [ ] **Step 4: Implement metadata inspection**

```python
@dataclass(frozen=True)
class SourceAudioMetadata:
    duration_sec: float
    sample_rate: int
    channels: int
    frames: int


class SourceAudioDecodeError(ValueError):
    pass


def inspect_source_audio(path: Path) -> SourceAudioMetadata:
    try:
        info = soundfile.info(path)
    except (OSError, RuntimeError, ValueError):
        raise SourceAudioDecodeError("source_audio_decode_failed") from None
    if info.frames <= 0 or info.samplerate <= 0 or info.channels <= 0:
        raise SourceAudioDecodeError("source_audio_decode_failed")
    return SourceAudioMetadata(
        info.frames / info.samplerate,
        info.samplerate,
        info.channels,
        info.frames,
    )
```

- [ ] **Step 5: Implement native events and bounds**

Define `TIMING_SEMANTICS_VERSION = "crux.dtx-audio-timing/v1"` and
`REFERENCE_EVENT_SCHEMA = "crux.dtx-reference-event/v1"`.

For every generic DTX event:

```python
chart_time = timing.time_sec(event)
audio_time = chart_time - resolved_bgm.chart_time_sec
```

Preserve `simfile_id`, selected chart key/hash, source audio key/hash, source order,
measure, position, lane ID, note ID, chart time, and audio time. Do not map or deduplicate.
Sort by `(audio_time_sec, measure, position, lane_id, note_id, source_order)`.

- [ ] **Step 6: Render deterministic canonical JSONL**

Use `canonical_json_line` for each event and include both schema and timing-semantics
versions. Test exact repeated bytes and SHA-256 equality.

- [ ] **Step 7: Expose the existing immutable byte publisher**

In `src/benchmark/corpus_manifest.py`:

```python
def publish_immutable_content(path: Path, content: bytes, expected_sha256: str) -> None:
    _publish_immutable(path, content, expected_sha256)
```

Test new publication, identical reuse, and rejection of different bytes at the same
path. Do not duplicate or weaken `_publish_immutable`.

- [ ] **Step 8: Run checks and commit**

```bash
uv run pytest tests/benchmark/test_reference_timing.py tests/benchmark/test_corpus_manifest.py -q
uv run ruff check src/benchmark/reference_timing.py src/benchmark/corpus_manifest.py tests/benchmark/test_reference_timing.py tests/benchmark/test_corpus_manifest.py
uv run black --check src/benchmark/reference_timing.py src/benchmark/corpus_manifest.py tests/benchmark/test_reference_timing.py tests/benchmark/test_corpus_manifest.py
git add src/benchmark/reference_timing.py src/benchmark/corpus_manifest.py tests/benchmark/test_reference_timing.py tests/benchmark/test_corpus_manifest.py
git commit -m "feat: publish bounded native reference events"
```

---

### Task 5: Orchestrate manifest loading, cache fill, and derived publication

**Files:**
- Create: `src/benchmark/reference_timing_manifest.py`
- Create: `tests/benchmark/test_reference_timing_manifest.py`

**Interfaces:**
- Consumes: Tasks 1-4 and existing manifest publication helpers.
- Produces: `TimingRequest(manifest_path, cache_dir, output_dir)`.
- Produces: `TimingCounters(ready, quarantined, events_published)`.
- Produces: `TimingOutcome(status, exit_code, manifest, counters)`.
- Produces: `build_reference_timing_manifest(request, ...) -> TimingOutcome`.
- Consumed by: Task 6.

- [ ] **Step 1: Write failing input-validation tests**

Cover empty input, invalid UTF-8/JSON, non-object rows, wrong schema, duplicate simfile
IDs, mixed `corpus_version`, mixed bucket, mixed endpoint hash, and a selected chart with
no matching object record.

A representative assertion:

```python
with pytest.raises(ValueError, match="one corpus_version"):
    build_reference_timing_manifest(request_for(mixed_manifest, tmp_path))
```

- [ ] **Step 2: Write cache-orchestration tests**

- Complete chart/audio cache: inject a store factory that calls `pytest.fail` if used.
- Audio cache miss: inventory two audio objects, reference one, and assert only that key
  opens.
- Upstream HPA-322 quarantine: no chart parsing or R2 access.
- Per-object download failure: quarantine only rows using that audio object.

- [ ] **Step 3: Verify tests fail**

```bash
uv run pytest tests/benchmark/test_reference_timing_manifest.py -q
```

Expected: module import failure.

- [ ] **Step 4: Add request and outcome records**

```python
REFERENCE_TIMING_MANIFEST_SCHEMA = "crux.reference-timing-manifest/v1"

@dataclass(frozen=True)
class TimingRequest:
    manifest_path: Path
    cache_dir: Path
    output_dir: Path

@dataclass(frozen=True)
class TimingCounters:
    ready: int = 0
    quarantined: int = 0
    events_published: int = 0

@dataclass(frozen=True)
class TimingOutcome:
    status: Literal["complete", "partial"]
    exit_code: Literal[0, 1]
    manifest: PublishedManifest
    counters: TimingCounters
```

Fatal errors raise `ValueError`, `R2StoreError`, or `ManifestPublicationError`; the CLI
maps them to exit `2`.

- [ ] **Step 5: Load and validate exact HPA-322 bytes**

Read once, compute `source_manifest_sha256`, parse non-empty UTF-8 JSONL lines, and
require one `crux.reference-chart-manifest/v1` corpus version, bucket, endpoint hash,
and unique integer simfile IDs. Preserve the input version as
`source_reference_chart_version`.

Rows with `selection_status != "selected"` receive
`upstream_chart_selection_unavailable` and no further work.

- [ ] **Step 6: Validate selected chart cache bodies**

Require a matching object record with `cache_status == "verified"`, relative
`cache_path`, regular file under `cache_dir`, exact size, and exact SHA-256 equal to
`selected_chart_content_hash`. Quarantine violations as
`selected_chart_cache_invalid`; do not network-repair charts.

- [ ] **Step 7: First pass — parse charts and resolve selected audio keys**

For each selected row:

1. parse the selected DTX;
2. build the timing map;
3. convert `objects[]` into `SourceObject` values;
4. resolve the BGM reference;
5. retain resolved row state or quarantine;
6. collect one deduplicated exact audio-key set.

- [ ] **Step 8: Fill only missing selected audio bodies**

When every selected audio body is already verified, do not import optional R2
dependencies or create a store.

When misses exist:

1. `ensure_r2_dependency()`;
2. `R2Config.from_environ(environ)`;
3. verify bucket and endpoint hash match the manifest;
4. `CacheIndexStore.load(cache_dir)`;
5. create and validate the store;
6. hold `cache_writer_lock(cache_dir)`;
7. call `sync_explicit_cache_keys(..., selected_keys)`.

Inject `environ`, dependency check, and store factory using the same pattern as
`sync_r2_corpus`.

- [ ] **Step 9: Second pass — inspect audio and publish events**

For each resolved row:

1. verify post-cache body/hash;
2. call `inspect_source_audio`;
3. build bounded audio-relative events;
4. render canonical JSONL;
5. compute SHA-256;
6. publish to `events/<sha256>.jsonl` with `publish_immutable_content`.

- [ ] **Step 10: Build derived rows**

Preserve HPA-322 inventory/selection values and add:

```text
schema_version = crux.reference-timing-manifest/v1
source_manifest_sha256
source_reference_chart_version
timing_semantics_version
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
selected_bgm_note_id
selected_bgm_chart_time_sec
reference_events_path
reference_events_sha256
reference_event_count
pre_audio_event_count
post_audio_event_count
```

Update only the matching source-audio entry in `objects[]` with verified cache status,
SHA, and cache path.

- [ ] **Step 11: Publish and test reconciliation**

Use `render_manifest`, `publish_manifest`, and `publish_latest_manifest`.

Require:

```text
ready + quarantined = input rows
events_published = ready
```

All ready -> `complete`, exit `0`; any quarantine -> `partial`, exit `1`.
Test deterministic reruns, changed-audio identity, partial publication, and publication
failure without a fake outcome.

- [ ] **Step 12: Run checks and commit**

```bash
uv run pytest tests/benchmark/test_reference_timing_manifest.py -q
uv run ruff check src/benchmark/reference_timing_manifest.py tests/benchmark/test_reference_timing_manifest.py
uv run black --check src/benchmark/reference_timing_manifest.py tests/benchmark/test_reference_timing_manifest.py
git add src/benchmark/reference_timing_manifest.py tests/benchmark/test_reference_timing_manifest.py
git commit -m "feat: publish audio relative timing manifest"
```

---

### Task 6: Wire the CLI and acceptance fixture

**Files:**
- Modify: `src/cli/benchmark.py`
- Modify: `tests/test_cli_benchmark.py`
- Create: `tests/benchmark/test_reference_timing_acceptance.py`

**Interfaces:**
- Consumes: `TimingRequest` and `build_reference_timing_manifest`.
- Produces: `crux benchmark build-reference-timing`.
- Produces: one JSON stdout summary and exit codes `0`, `1`, or `2`.

- [ ] **Step 1: Write failing CLI tests**

```python
def test_build_reference_timing_help_lists_options() -> None:
    result = runner.invoke(main, ["benchmark", "build-reference-timing", "--help"])
    assert result.exit_code == 0
    assert "--manifest FILE" in result.stdout
    assert "--cache-dir DIRECTORY" in result.stdout
    assert "--output-dir DIRECTORY" in result.stdout
```

Add request-wiring, complete summary, partial summary, invalid-input exit `2`, and
complete-cache-without-R2 tests.

- [ ] **Step 2: Verify CLI tests fail**

```bash
uv run pytest tests/test_cli_benchmark.py -k build_reference_timing -q
```

Expected: unknown command.

- [ ] **Step 3: Add the Click command**

```python
@benchmark.command("build-reference-timing")
@click.option("--manifest", type=click.Path(path_type=Path, dir_okay=False), required=True)
@click.option("--cache-dir", type=click.Path(path_type=Path, file_okay=False), default=None)
@click.option(
    "--output-dir",
    type=click.Path(path_type=Path, file_okay=False),
    default=Path("artifacts/benchmark/reference-timing"),
    show_default=True,
)
@click.pass_context
def build_reference_timing_command(
    ctx: click.Context,
    manifest: Path,
    cache_dir: Path | None,
    output_dir: Path,
) -> None:
    from src.benchmark.corpus_manifest import ManifestPublicationError
    from src.benchmark.r2_inventory import R2StoreError
    from src.benchmark.reference_timing_manifest import (
        TimingRequest,
        build_reference_timing_manifest,
    )

    resolved_cache = (
        manifest.parent.parent.parent / "r2-corpus" / "cache"
        if cache_dir is None
        else cache_dir
    )
    try:
        outcome = build_reference_timing_manifest(
            TimingRequest(manifest, resolved_cache, output_dir)
        )
    except (ValueError, ManifestPublicationError, R2StoreError) as exc:
        click.echo(str(exc), err=True)
        ctx.exit(2)

    click.echo(json.dumps({
        "corpus_version": outcome.manifest.corpus_version,
        "events_published": outcome.counters.events_published,
        "exit_code": outcome.exit_code,
        "manifest_path": str(outcome.manifest.path),
        "quarantined": outcome.counters.quarantined,
        "ready": outcome.counters.ready,
        "status": outcome.status,
    }, sort_keys=True))
    if outcome.exit_code:
        ctx.exit(outcome.exit_code)
```

- [ ] **Step 4: Build the acceptance fixture**

Create five rows:

1. sticky measure length affecting BGM and drum times;
2. nested chart with a safe relative audio path;
3. selected audio cache miss served by a fake store;
4. conflicting BGM starts causing `ambiguous_bgm_start`;
5. pre/post-audio events with one retained in-bounds event.

Invoke the real Click command, assert exit `1`, inspect the output manifest and event
JSONL, and assert no external network access.

- [ ] **Step 5: Run focused and full validation**

```bash
uv run pytest tests/test_cli_benchmark.py -k build_reference_timing -q
uv run pytest tests/benchmark/test_reference_timing_acceptance.py -q
uv run pytest \
  tests/benchmark/test_dtx_parser.py \
  tests/benchmark/test_timing.py \
  tests/benchmark/test_corpus_cache.py \
  tests/benchmark/test_corpus_manifest.py \
  tests/benchmark/test_reference_timing.py \
  tests/benchmark/test_reference_timing_manifest.py \
  tests/benchmark/test_reference_timing_acceptance.py \
  tests/test_cli_benchmark.py -q
uv run pytest -q
uv run ruff check .
uv run black --check .
```

If current CI runs Pylint after HPA-322 merges, run the exact Pylint command from
`.github/workflows/ci.yml`.

- [ ] **Step 6: Verify artifact determinism**

Run the acceptance command twice against the same fixture/output and compare:

```bash
sha256sum artifacts/benchmark/reference-timing/manifests/*.jsonl
sha256sum artifacts/benchmark/reference-timing/events/*.jsonl
```

Expected: no second set of content-addressed files and identical hashes.

- [ ] **Step 7: Commit**

```bash
git add src/cli/benchmark.py tests/test_cli_benchmark.py tests/benchmark/test_reference_timing_acceptance.py
git commit -m "feat: expose reference timing pipeline"
```

---

## Final Review Checklist

- [ ] HPA-322 is merged and this implementation consumes its real schema.
- [ ] Channel `01` never appears in native event artifacts.
- [ ] Sticky channel `02` affects both measure starts and positions within each active measure.
- [ ] Existing BPM, fractional-position, and same-beat ordering tests still pass.
- [ ] Every BGM token resolves into the selected group or causes an actionable quarantine.
- [ ] No filename heuristic overrides the DTX WAV reference.
- [ ] Only exact selected source-audio keys are downloaded.
- [ ] Complete-cache reruns need no R2 dependency or credentials.
- [ ] Native lane, note, position, and source order survive into event artifacts.
- [ ] Pre/post-audio exclusions and frame-tolerance clamps reconcile exactly.
- [ ] `ready + quarantined` equals the HPA-322 input row count.
- [ ] Raw scoring remains distinct from aligned diagnostics.
- [ ] No HPA-324 taxonomy, HPA-326 inference, or HPA-325 scoring scope leaked into this work.