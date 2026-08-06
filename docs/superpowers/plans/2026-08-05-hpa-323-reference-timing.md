# HPA-323 Audio-Relative DTX Reference Timing Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Consume the immutable HPA-322 reference-chart manifest, correct DTX timing semantics, resolve and cache the DTX-referenced full-mix audio, and publish immutable audio-relative native reference-event artifacts for HPA-324.

**Architecture:** Extend the existing parser with typed BGM events and source order, replace ad hoc timing helpers with one shared sticky-measure timing map, and add a focused reference-timing stage. The stage resolves exact source-audio objects from manifest metadata, fills only missing selected audio bodies through the existing R2 cache, then publishes content-addressed per-song event JSONL and a derived immutable manifest.

**Tech Stack:** Python 3.12, dataclasses, pathlib/PurePosixPath, JSONL, hashlib, Click, soundfile, boto3 through the existing optional `r2` extra, pytest.

## Global Constraints

- Start implementation only after HPA-322 is merged; consume `crux.reference-chart-manifest/v1` rather than guessing its final code shape on an unmerged branch.
- Channel `02` is sticky beginning with its own measure and remains active until superseded.
- Channel `01` must never enter generic/native drum events; preserve it as typed BGM control data.
- Resolve source audio from the DTX `#WAVxx` reference; `bgm.ogg` has no unconditional authority.
- Prefer exact object-key matches, then one unique case-insensitive match.
- Use selected-chart-directory resolution first and simfile-root fallback only when the relative path has no match.
- Cache only exact selected source-audio keys; do not broaden HPA-321's default suffix policy.
- Preserve native DTX lane and note identities; HPA-324 owns canonical mapping and final eligibility.
- Publish a new timing semantics identity: `crux.dtx-audio-timing/v1`.
- Raw benchmark timing uses the DTX-derived audio clock. Auto-alignment remains separately labeled diagnostics.
- Process rows sequentially after the targeted cache fill. Do not add a database, service, workflow framework, or new general-purpose concurrency layer.
- Use TDD, focused commits, and the repository's `uv run` commands.

---

## File Map

### Create

- `src/benchmark/reference_timing.py` — pure BGM resolution, source-audio metadata, bounds handling, and native reference-event rendering.
- `src/benchmark/reference_timing_manifest.py` — HPA-322 manifest loading, targeted cache orchestration, event-artifact publication, derived-manifest publication, counters, and outcome.
- `tests/benchmark/test_reference_timing.py` — BGM path, ambiguity, audio metadata, bounds, and deterministic event tests.
- `tests/benchmark/test_reference_timing_manifest.py` — manifest validation, cache-fill orchestration, lineage, counters, and immutable publication tests.
- `tests/benchmark/test_reference_timing_acceptance.py` — offline end-to-end fixture through the real CLI.

### Modify

- `src/benchmark/models.py` — preserve deterministic source order on generic DTX events.
- `src/benchmark/dtx_parser.py` — parse typed channel `01` events and assign source order.
- `tests/benchmark/test_dtx_parser.py` — typed BGM and source-order tests.
- `src/benchmark/timing.py` — add `DtxTimingMap` and sticky channel `02` semantics.
- `tests/benchmark/test_timing.py` — persistence, replacement, altered-measure BPM, and BGM parity tests.
- `src/benchmark/corpus_cache.py` — expose exact-key cache selection while preserving the default HPA-321 policy.
- `tests/benchmark/test_corpus_cache.py` — exact-key cache tests and default-policy regression.
- `src/benchmark/corpus_manifest.py` — expose the existing immutable byte publisher through a public wrapper.
- `tests/benchmark/test_corpus_manifest.py` — public immutable-content wrapper tests.
- `src/cli/benchmark.py` — add `build-reference-timing`.
- `tests/test_cli_benchmark.py` — CLI help, request wiring, JSON summary, and exit-code tests.

---

### Task 1: Parse typed BGM events and build one sticky-measure timing map

**Files:**
- Modify: `src/benchmark/models.py`
- Modify: `src/benchmark/dtx_parser.py`
- Modify: `src/benchmark/timing.py`
- Modify: `tests/benchmark/test_dtx_parser.py`
- Modify: `tests/benchmark/test_timing.py`

**Interfaces:**
- Produces: `DtxEvent.source_order: int` with default `0` for direct construction compatibility.
- Produces: `DtxBgmEvent(chart_id, measure, position, note_id, source_order)`.
- Produces: `ParsedDtxChart.bgm_events: list[DtxBgmEvent]`.
- Produces: `DtxTimingMap.time_sec(event: DtxEvent | DtxBpmEvent | DtxBgmEvent) -> float`.
- Produces: `build_dtx_timing_map(chart: ParsedDtxChart) -> DtxTimingMap`.
- Consumed by: Tasks 2, 4, and 5.

- [ ] **Step 1: Write failing typed-channel and source-order tests**

Append focused tests to `tests/benchmark/test_dtx_parser.py`:

```python
def test_channel_01_is_typed_bgm_not_generic_event() -> None:
    chart = parse_dtx_text(
        "#WAV01: bgm.ogg\n#00101: 0100\n#00111: 0001\n",
        chart_id="song",
    )

    assert [(event.measure, event.position, event.note_id) for event in chart.bgm_events] == [
        (1, 0.0, "01")
    ]
    assert [(event.lane_id, event.note_id) for event in chart.events] == [("11", "01")]


def test_parser_assigns_monotonic_source_order_to_nonzero_pattern_events() -> None:
    chart = parse_dtx_text(
        "#00111: 0102\n#00101: 0304\n#00112: 0500\n",
        chart_id="song",
    )

    orders = [event.source_order for event in chart.events]
    bgm_orders = [event.source_order for event in chart.bgm_events]

    assert sorted(orders + bgm_orders) == [0, 1, 2, 3, 4]
```

- [ ] **Step 2: Run parser tests and verify failure**

```bash
uv run pytest tests/benchmark/test_dtx_parser.py -k "channel_01 or source_order" -q
```

Expected: assertions fail because channel `01` is generic and the event types have no
`source_order` or `bgm_events` fields.

- [ ] **Step 3: Add source order and the typed BGM record**

In `src/benchmark/models.py`, extend `DtxEvent` without breaking direct test fixtures:

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

In `src/benchmark/dtx_parser.py`, add:

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

Replace `_parse_note_events` with one helper that assigns a monotonically increasing
order only to non-zero tokens:

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
            bgm.append(
                DtxBgmEvent(chart_id, measure, position, note_id, next_source_order)
            )
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

Do not reuse the existing BPM `source_counter`; keep pattern-event source order separate
from tempo-source order because they serve different deterministic ties.

- [ ] **Step 4: Return sorted typed and generic events**

Sort generic events by:

```python
(event.measure, event.position, event.lane_id, event.note_id, event.source_order)
```

Sort BGM events by:

```python
(event.measure, event.position, event.note_id, event.source_order)
```

Return both lists from `parse_dtx_text`.

- [ ] **Step 5: Run parser tests**

```bash
uv run pytest tests/benchmark/test_dtx_parser.py -q
```

Expected: all parser tests pass, including HPA-322 DLEVEL/decoder tests after that work
is merged.

- [ ] **Step 6: Replace the old non-sticky timing expectation with persistence tests**

Update `tests/benchmark/test_timing.py`. The current
`test_measure_length_changes_timing` encodes the incorrect reset-to-`1.0` behavior.
Replace it with:

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

    assert [event.time_sec for event in timed] == [2.0, 3.0, 4.0, 5.0, 7.0]


def test_bpm_change_inside_sticky_short_measure_uses_resolved_length() -> None:
    chart = parse_dtx_text(
        "#BPM: 120\n#BPM01: 60\n#00102: 0.5\n#00208: 0001\n#00311: 01\n",
        "song",
    )

    timed = dtx_events_to_timed_events(chart)

    assert timed[0].time_sec == 5.0
```

Add a BGM parity test:

```python
def test_bgm_and_generic_event_share_one_timing_map() -> None:
    chart = parse_dtx_text(
        "#BPM: 120\n#00102: 0.5\n#00201: 01\n#00211: 0001\n",
        "song",
    )

    timing = build_dtx_timing_map(chart)

    assert timing.time_sec(chart.bgm_events[0]) == 3.0
    assert timing.time_sec(chart.events[0]) == 3.5
```

- [ ] **Step 7: Run timing tests and verify failure**

```bash
uv run pytest tests/benchmark/test_timing.py -q
```

Expected: sticky-measure assertions fail because `_measure_lengths_by_measure` and
`_event_beat` still default missing measures to `1.0`; the public timing-map API does not
exist.

- [ ] **Step 8: Implement `DtxTimingMap`**

In `src/benchmark/timing.py`, add:

```python
@dataclass(frozen=True)
class DtxTimingMap:
    resolved_measure_lengths: tuple[float, ...]
    measure_start_beats: tuple[float, ...]
    tempo_points: tuple[tuple[float, float, float], ...]

    def time_sec(self, event: DtxEvent | DtxBpmEvent | DtxBgmEvent) -> float:
        length = self.resolved_measure_lengths[event.measure]
        beat = (
            self.measure_start_beats[event.measure]
            + event.position * length * BEATS_PER_MEASURE
        )
        return _time_at_beat(beat, list(self.tempo_points))
```

Add `build_dtx_timing_map`:

```python
def build_dtx_timing_map(chart: ParsedDtxChart) -> DtxTimingMap:
    lengths = tuple(_measure_lengths_by_measure(chart))
    starts = tuple(_measure_start_beats(lengths))
    points = tuple(_tempo_points(chart, lengths, starts))
    return DtxTimingMap(lengths, starts, points)
```

Refactor private helpers to consume the resolved sequence rather than
`chart.measure_lengths`.

- [ ] **Step 9: Implement sticky measure lengths correctly**

```python
def _measure_lengths_by_measure(chart: ParsedDtxChart) -> list[float]:
    lengths: list[float] = []
    active = 1.0
    for measure in range(_max_measure(chart) + 2):
        if measure in chart.measure_lengths:
            active = chart.measure_lengths[measure]
        if active <= 0:
            raise ValueError(f"measure {measure} has non-positive length")
        lengths.append(active)
    return lengths
```

Extend `_max_measure` with `chart.bgm_events`. Update tempo-point construction so BPM
events also use `resolved_measure_lengths[event.measure]`.

- [ ] **Step 10: Keep the legacy wrapper on the shared map**

```python
def dtx_events_to_timed_events(chart: ParsedDtxChart) -> list[BenchmarkEvent]:
    timing = build_dtx_timing_map(chart)
    return [
        BenchmarkEvent(
            chart_id=event.chart_id,
            time_sec=timing.time_sec(event),
            canonical_class=event.lane_id,
            source="ground_truth",
            metadata={
                "lane_id": event.lane_id,
                "note_id": event.note_id,
                "source_order": event.source_order,
            },
        )
        for event in chart.events
    ]
```

Do not apply a BGM shift in this legacy wrapper. HPA-323's new stage owns the selected
audio clock; old folder-based commands remain chart-time compatibility paths until they
are retired separately.

- [ ] **Step 11: Run focused checks**

```bash
uv run pytest tests/benchmark/test_dtx_parser.py tests/benchmark/test_timing.py -q
uv run ruff check src/benchmark/models.py src/benchmark/dtx_parser.py src/benchmark/timing.py tests/benchmark/test_dtx_parser.py tests/benchmark/test_timing.py
uv run black --check src/benchmark/models.py src/benchmark/dtx_parser.py src/benchmark/timing.py tests/benchmark/test_dtx_parser.py tests/benchmark/test_timing.py
```

Expected: all pass.

- [ ] **Step 12: Commit Task 1**

```bash
git add src/benchmark/models.py src/benchmark/dtx_parser.py src/benchmark/timing.py tests/benchmark/test_dtx_parser.py tests/benchmark/test_timing.py
git commit -m "fix: derive sticky DTX timing controls"
```

---

### Task 2: Resolve one authoritative BGM event and source-audio object

**Files:**
- Create: `src/benchmark/reference_timing.py`
- Create: `tests/benchmark/test_reference_timing.py`

**Interfaces:**
- Consumes: `ParsedDtxChart`, `DtxBgmEvent`, and `DtxTimingMap` from Task 1.
- Produces: `SourceObject`.
- Produces: `ResolvedBgm(event, chart_time_sec, audio_object, raw_event_count, warnings)`.
- Produces: `BgmResolution(status, reason_codes, warnings, resolved)`.
- Produces: `resolve_bgm_reference(...) -> BgmResolution`.
- Consumed by: Task 5.

- [ ] **Step 1: Write failing exact and relative-resolution tests**

Create `tests/benchmark/test_reference_timing.py` with local object helpers and:

```python
def test_resolve_bgm_uses_selected_chart_directory() -> None:
    chart = parse_dtx_text(
        "#WAV01: audio/song.ogg\n#00101: 01\n",
        chart_id="song",
    )
    timing = build_dtx_timing_map(chart)

    resolution = resolve_bgm_reference(
        chart,
        timing,
        selected_chart_key="42/charts/real.dtx",
        object_prefix="42/",
        objects=(source_object("42/charts/audio/song.ogg"),),
    )

    assert resolution.status == "resolved"
    assert resolution.resolved.audio_object.key == "42/charts/audio/song.ogg"


def test_resolve_bgm_uses_root_compatibility_only_after_relative_miss() -> None:
    chart = parse_dtx_text("#WAV01: bgm.ogg\n#00101: 01\n", "song")

    resolution = resolve_bgm_reference(
        chart,
        build_dtx_timing_map(chart),
        selected_chart_key="42/charts/real.dtx",
        object_prefix="42/",
        objects=(source_object("42/bgm.ogg"),),
    )

    assert resolution.status == "resolved"
    assert resolution.resolved.audio_object.key == "42/bgm.ogg"
    assert "source_audio_root_fallback" in resolution.warnings
```

- [ ] **Step 2: Add failing safety and ambiguity tests**

Cover:

```python
def test_unresolved_wav_id_quarantines() -> None:
    chart = parse_dtx_text("#00101: 01\n", "song")
    resolution = resolve_bgm_reference(
        chart,
        build_dtx_timing_map(chart),
        selected_chart_key="42/real.dtx",
        object_prefix="42/",
        objects=(),
    )
    assert resolution.reason_codes == ("unresolved_bgm_wav",)


def test_distinct_bgm_starts_quarantine() -> None:
    chart = parse_dtx_text(
        "#WAV01: bgm.ogg\n#00101: 01\n#00201: 01\n",
        "song",
    )
    resolution = resolve_bgm_reference(
        chart,
        build_dtx_timing_map(chart),
        selected_chart_key="42/real.dtx",
        object_prefix="42/",
        objects=(source_object("42/bgm.ogg"),),
    )
    assert resolution.reason_codes == ("ambiguous_bgm_start",)
```

Also test absolute paths, `..` traversal above `object_prefix`, duplicate
case-insensitive matches, unknown audio object, and zero BGM events.

- [ ] **Step 3: Run tests and verify failure**

```bash
uv run pytest tests/benchmark/test_reference_timing.py -k bgm -q
```

Expected: collection fails because `src.benchmark.reference_timing` does not exist.

- [ ] **Step 4: Add the focused domain records**

Create `src/benchmark/reference_timing.py`:

```python
from __future__ import annotations

from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Literal

from src.benchmark.dtx_parser import DtxBgmEvent, ParsedDtxChart
from src.benchmark.timing import DtxTimingMap


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

- [ ] **Step 5: Implement safe DTX-relative object-key resolution**

Normalize backslashes to `/`. Reject empty, absolute, drive-prefixed, or traversal paths.
Resolve first against `PurePosixPath(selected_chart_key).parent`, then against the
simfile root only when no relative candidate exists.

For each lookup:

1. exact key match;
2. one unique `casefold()` match with `source_audio_case_fallback` warning;
3. multiple casefold matches -> `source_audio_key_ambiguous`;
4. no match -> continue to root fallback or return `source_audio_missing`.

Do not search by basename across arbitrary nested directories.

- [ ] **Step 6: Implement WAV lookup and group selection**

For every `chart.bgm_events` item:

```python
wav_value = chart.wav_table.get(event.note_id)
```

An absent or empty value adds `unresolved_bgm_wav` and quarantines the row after all
source events are inspected. Do not select a remaining resolvable event when another
channel `01` event is unresolved.

Resolve all events, calculate `timing.time_sec(event)`, and group by:

```python
(audio_object.key, chart_time_sec)
```

Selection:

- no channel `01`: `bgm_event_missing`;
- one group: select its lowest-source-order event;
- one group with multiple events: add `duplicate_bgm_event` warning;
- multiple groups: `ambiguous_bgm_start`.

Set `raw_event_count=len(chart.bgm_events)`.

- [ ] **Step 7: Run BGM-resolution tests**

```bash
uv run pytest tests/benchmark/test_reference_timing.py -k bgm -q
uv run ruff check src/benchmark/reference_timing.py tests/benchmark/test_reference_timing.py
uv run black --check src/benchmark/reference_timing.py tests/benchmark/test_reference_timing.py
```

Expected: all BGM tests pass.

- [ ] **Step 8: Commit Task 2**

```bash
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

- [ ] **Step 1: Write a failing exact-key cache test**

Add a test using existing fake store/cache fixtures:

```python
def test_sync_explicit_cache_keys_downloads_only_named_audio(tmp_path: Path) -> None:
    simfile = inventory_with_objects(
        remote_object("42/real.dtx", b"chart"),
        remote_object("42/bgm.ogg", b"audio"),
        remote_object("42/preview.ogg", b"preview"),
    )
    store = fake_store_for(simfile.objects)
    index = CacheIndexStore.load(tmp_path / "cache")

    result = sync_explicit_cache_keys(
        (simfile,),
        store,
        index,
        r2_config(),
        frozenset({"42/bgm.ogg"}),
    )

    by_key = {obj.key: obj for obj in result.simfiles[0].objects}
    assert by_key["42/bgm.ogg"].cache_status == "verified"
    assert by_key["42/real.dtx"].cache_status is None
    assert by_key["42/preview.ogg"].cache_status is None
    assert store.opened_keys == ["42/bgm.ogg"]
```

Use the repository's actual fixture names where they already exist; keep the assertions
and exact selected-key behavior unchanged.

- [ ] **Step 2: Add default-policy regression tests**

```python
def test_default_sync_cache_still_excludes_audio() -> None:
    assert is_selected("42/real.dtx") is True
    assert is_selected("42/set.def") is True
    assert is_selected("42/bgm.ogg") is False
```

Also test that a selected key absent from the inventory is ignored rather than invented,
and that duplicate selected keys across repeated input rows still download through the
cache index at most once.

- [ ] **Step 3: Run tests and verify failure**

```bash
uv run pytest tests/benchmark/test_corpus_cache.py -k "explicit_cache or default_sync_cache" -q
```

Expected: import or attribute failure because `sync_explicit_cache_keys` does not exist.

- [ ] **Step 4: Extract the selector inside `sync_cache`**

Keep the public default signature. Make it delegate to one private worker:

```python
def sync_cache(
    simfiles: tuple[SimfileInventory, ...],
    store: R2ObjectStore,
    index: CacheIndexStore,
    config: R2Config,
    dry_run: bool,
    item_progress: Callable[[int, int, int], None] | None = None,
) -> CacheSyncResult:
    return _sync_cache_selected(
        simfiles,
        store,
        index,
        config,
        dry_run,
        selector=lambda remote: is_selected(remote.key),
        item_progress=item_progress,
    )
```

The private worker receives:

```python
selector: Callable[[RemoteObject], bool]
```

Use it for both `total_selected` and per-object selection. Do not alter download,
locking, validation, content hashing, or cache-index behavior.

- [ ] **Step 5: Add the explicit-key wrapper**

```python
def sync_explicit_cache_keys(
    simfiles: tuple[SimfileInventory, ...],
    store: R2ObjectStore,
    index: CacheIndexStore,
    config: R2Config,
    selected_keys: frozenset[str],
    item_progress: Callable[[int, int, int], None] | None = None,
) -> CacheSyncResult:
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

Reject an empty string inside `selected_keys` with `ValueError`; an empty set returns the
unchanged simfiles and no actions without constructing network work.

- [ ] **Step 6: Run cache tests**

```bash
uv run pytest tests/benchmark/test_corpus_cache.py -q
uv run ruff check src/benchmark/corpus_cache.py tests/benchmark/test_corpus_cache.py
uv run black --check src/benchmark/corpus_cache.py tests/benchmark/test_corpus_cache.py
```

Expected: all existing and new cache tests pass.

- [ ] **Step 7: Commit Task 3**

```bash
git add src/benchmark/corpus_cache.py tests/benchmark/test_corpus_cache.py
git commit -m "feat: cache explicit benchmark objects"
```

---

### Task 4: Inspect source audio and build deterministic bounded native events

**Files:**
- Modify: `src/benchmark/reference_timing.py`
- Modify: `tests/benchmark/test_reference_timing.py`
- Modify: `src/benchmark/corpus_manifest.py`
- Modify: `tests/benchmark/test_corpus_manifest.py`

**Interfaces:**
- Produces: `SourceAudioMetadata(duration_sec, sample_rate, channels, frames)`.
- Produces: `NativeReferenceEvent`.
- Produces: `ReferenceEventBuild(events, pre_audio_event_count, post_audio_event_count, warnings, reason_codes)`.
- Produces: `inspect_source_audio(path: Path) -> SourceAudioMetadata`.
- Produces: `build_audio_relative_events(...) -> ReferenceEventBuild`.
- Produces: `render_reference_event_jsonl(...) -> bytes`.
- Produces: `publish_immutable_content(path, content, expected_sha256) -> None`.
- Consumed by: Task 5.

- [ ] **Step 1: Write failing source-audio metadata tests**

Generate a tiny valid WAV through `soundfile.write`:

```python
def test_inspect_source_audio_returns_exact_frame_metadata(tmp_path: Path) -> None:
    path = tmp_path / "source.wav"
    soundfile.write(path, np.zeros(8000, dtype=np.float32), 8000)

    metadata = inspect_source_audio(path)

    assert metadata.frames == 8000
    assert metadata.sample_rate == 8000
    assert metadata.channels == 1
    assert metadata.duration_sec == 1.0
```

Add tests for zero frames and an undecodable file raising
`SourceAudioDecodeError("source_audio_decode_failed")` without including file contents.

- [ ] **Step 2: Write failing bounds tests**

Use directly constructed charts/timing and metadata:

```python
def test_audio_relative_events_shift_by_selected_bgm_time() -> None:
    chart = parse_dtx_text(
        "#BPM: 120\n#WAV01: bgm.wav\n#00101: 01\n#00211: 01\n",
        "song",
    )
    timing = build_dtx_timing_map(chart)
    bgm = resolved_bgm(chart, timing, key="42/bgm.wav")

    built = build_audio_relative_events(
        simfile_id=42,
        chart=chart,
        timing=timing,
        resolved_bgm=bgm,
        selected_chart_key="42/real.dtx",
        selected_chart_content_hash="a" * 64,
        source_audio_content_hash="b" * 64,
        audio=SourceAudioMetadata(10.0, 8000, 1, 80000),
    )

    assert built.events[0].chart_time_sec == 4.0
    assert built.events[0].audio_time_sec == 2.0
```

Add separate tests for:

- one-frame negative value clamped to zero;
- earlier negative value excluded and counted;
- one-frame post-duration value clamped to duration;
- later post-duration value excluded and counted;
- NaN/Infinity quarantining with `non_finite_reference_time`;
- no in-bounds events quarantining with `no_in_bounds_reference_events`.

- [ ] **Step 3: Run tests and verify failure**

```bash
uv run pytest tests/benchmark/test_reference_timing.py -k "source_audio or audio_relative" -q
```

Expected: missing interfaces.

- [ ] **Step 4: Add source-audio metadata inspection**

In `src/benchmark/reference_timing.py`:

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
        duration_sec=info.frames / info.samplerate,
        sample_rate=info.samplerate,
        channels=info.channels,
        frames=info.frames,
    )
```

Do not fall back to full waveform decode in this stage.

- [ ] **Step 5: Add native reference-event records**

```python
REFERENCE_EVENT_SCHEMA = "crux.dtx-reference-event/v1"
TIMING_SEMANTICS_VERSION = "crux.dtx-audio-timing/v1"


@dataclass(frozen=True)
class NativeReferenceEvent:
    simfile_id: int
    selected_chart_key: str
    selected_chart_content_hash: str
    source_audio_key: str
    source_audio_content_hash: str
    source_order: int
    measure: int
    position: float
    lane_id: str
    note_id: str
    chart_time_sec: float
    audio_time_sec: float


@dataclass(frozen=True)
class ReferenceEventBuild:
    events: tuple[NativeReferenceEvent, ...]
    pre_audio_event_count: int
    post_audio_event_count: int
    warnings: tuple[str, ...]
    reason_codes: tuple[str, ...]
```

- [ ] **Step 6: Implement the frame-tolerance bounds policy**

For each `chart.events` item:

```python
chart_time = timing.time_sec(event)
audio_time = chart_time - resolved_bgm.chart_time_sec
frame_tolerance = 1.0 / audio.sample_rate
```

Rules:

- reject non-finite chart/audio times;
- clamp `[-frame_tolerance, 0)` to `0.0` and add
  `pre_audio_event_clamped_to_zero` once;
- exclude `< -frame_tolerance` and increment `pre_audio_event_count`;
- clamp `(duration, duration + frame_tolerance]` to duration and add
  `post_audio_event_clamped_to_duration` once;
- exclude `> duration + frame_tolerance` and increment
  `post_audio_event_count`;
- sort included events by
  `(audio_time_sec, measure, position, lane_id, note_id, source_order)`;
- when no events remain, return reason `no_in_bounds_reference_events`.

Do not map lanes or deduplicate simultaneous native events.

- [ ] **Step 7: Render canonical event JSONL**

Use `canonical_json_line` from `corpus_manifest.py`:

```python
def render_reference_event_jsonl(events: tuple[NativeReferenceEvent, ...]) -> bytes:
    return b"".join(
        canonical_json_line(
            {
                "schema_version": REFERENCE_EVENT_SCHEMA,
                "simfile_id": event.simfile_id,
                "selected_chart_key": event.selected_chart_key,
                "selected_chart_content_hash": event.selected_chart_content_hash,
                "source_audio_key": event.source_audio_key,
                "source_audio_content_hash": event.source_audio_content_hash,
                "timing_semantics_version": TIMING_SEMANTICS_VERSION,
                "source_order": event.source_order,
                "measure": event.measure,
                "position": event.position,
                "lane_id": event.lane_id,
                "note_id": event.note_id,
                "chart_time_sec": event.chart_time_sec,
                "audio_time_sec": event.audio_time_sec,
            }
        )
        for event in events
    )
```

Add a deterministic-byte test that builds the same event tuple twice and compares exact
bytes and SHA-256.

- [ ] **Step 8: Expose the existing immutable publisher safely**

In `src/benchmark/corpus_manifest.py`, add a public wrapper only:

```python
def publish_immutable_content(
    path: Path,
    content: bytes,
    expected_sha256: str,
) -> None:
    _publish_immutable(path, content, expected_sha256)
```

Do not duplicate or weaken `_publish_immutable`.

Add tests that publish new content, accept identical existing content, and reject a
same-path different body with `ManifestPublicationError`.

- [ ] **Step 9: Run focused checks**

```bash
uv run pytest tests/benchmark/test_reference_timing.py tests/benchmark/test_corpus_manifest.py -q
uv run ruff check src/benchmark/reference_timing.py src/benchmark/corpus_manifest.py tests/benchmark/test_reference_timing.py tests/benchmark/test_corpus_manifest.py
uv run black --check src/benchmark/reference_timing.py src/benchmark/corpus_manifest.py tests/benchmark/test_reference_timing.py tests/benchmark/test_corpus_manifest.py
```

Expected: all pass.

- [ ] **Step 10: Commit Task 4**

```bash
git add src/benchmark/reference_timing.py src/benchmark/corpus_manifest.py tests/benchmark/test_reference_timing.py tests/benchmark/test_corpus_manifest.py
git commit -m "feat: publish bounded native reference events"
```

---

### Task 5: Orchestrate HPA-322 manifest loading, cache fill, and derived publication

**Files:**
- Create: `src/benchmark/reference_timing_manifest.py`
- Create: `tests/benchmark/test_reference_timing_manifest.py`

**Interfaces:**
- Consumes: Task 1 timing, Task 2 BGM resolution, Task 3 exact-key cache fill, Task 4 audio/event helpers, and existing canonical manifest publication.
- Produces: `TimingRequest(manifest_path, cache_dir, output_dir)`.
- Produces: `TimingCounters(ready, quarantined, events_published)`.
- Produces: `TimingOutcome(status, exit_code, manifest, counters)`.
- Produces: `build_reference_timing_manifest(request, ...) -> TimingOutcome`.
- Consumed by: Task 6.

- [ ] **Step 1: Write failing source-manifest validation tests**

Create `tests/benchmark/test_reference_timing_manifest.py` with JSONL helpers and cover:

```python
def test_rejects_mixed_reference_chart_versions(tmp_path: Path) -> None:
    manifest = write_jsonl(
        tmp_path / "input.jsonl",
        [
            reference_chart_row(1, corpus_version="sha256:a"),
            reference_chart_row(2, corpus_version="sha256:b"),
        ],
    )

    with pytest.raises(ValueError, match="one corpus_version"):
        build_reference_timing_manifest(request_for(manifest, tmp_path))
```

Also cover empty input, invalid JSON, non-object rows, wrong schema, duplicate simfile IDs,
mixed `source_bucket`, mixed `source_endpoint_sha256`, and selected rows without matching
selected-chart object metadata.

- [ ] **Step 2: Write a failing complete-cache orchestration test**

Build a selected chart and source audio in the content-addressed cache with manifest
records already marked verified. Inject a store factory that fails if called:

```python
def test_complete_cache_does_not_construct_r2_store(tmp_path: Path) -> None:
    request = complete_cached_request(tmp_path)

    outcome = build_reference_timing_manifest(
        request,
        store_factory=lambda _config: pytest.fail("store must not be constructed"),
    )

    assert outcome.status == "complete"
    assert outcome.counters.ready == 1
```

- [ ] **Step 3: Write a failing targeted-cache-fill test**

Create two inventoried audio objects, but reference only one from channel `01`. Inject a
fake store and assert only the referenced key opens. Verify the derived row enriches the
matching `objects[]` record with `cache_status`, `sha256`, and `cache_path` while leaving
the preview object unchanged.

- [ ] **Step 4: Run tests and verify failure**

```bash
uv run pytest tests/benchmark/test_reference_timing_manifest.py -q
```

Expected: collection fails because the module does not exist.

- [ ] **Step 5: Add request, counters, and outcome records**

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
    exit_code: Literal[0, 1, 2]
    manifest: PublishedManifest | None
    counters: TimingCounters
```

Fatal invalid input remains `ValueError`; cache configuration/store adapter failures use
existing `R2StoreError`; publication failures use `ManifestPublicationError`. The CLI
maps them to exit `2`.

- [ ] **Step 6: Load exact input bytes and validate lineage**

Read the file once as bytes, compute `source_manifest_sha256`, decode UTF-8, and parse
non-empty lines. Require:

```text
schema_version = crux.reference-chart-manifest/v1
one corpus_version
unique integer simfile_id
one source_bucket
one source_endpoint_sha256
```

Retain the input corpus version as `source_reference_chart_version`.

Rows with `selection_status != "selected"` become quarantined timing rows with
`upstream_chart_selection_unavailable`; do not parse or download anything for them.

- [ ] **Step 7: Validate selected chart cache bodies locally**

Implement a local helper in this module rather than guessing a private HPA-322 function:

```python
def _verified_cached_path(
    cache_dir: Path,
    object_record: dict[str, object],
    *,
    expected_sha256: str,
) -> Path:
    ...
```

Require `cache_status == "verified"`, a relative `cache_path`, a regular file under the
resolved cache directory, exact size, and exact SHA-256. Reject absolute paths and `..`.
Return `selected_chart_cache_invalid` as a row quarantine reason.

Do not perform network repair for a selected chart; HPA-322 must already have verified
it.

- [ ] **Step 8: First pass — parse charts and resolve audio candidates**

For each selected row:

1. validate the chart cache body;
2. `parse_dtx_file` with `chart_id=str(simfile_id)`;
3. `build_dtx_timing_map`;
4. convert row `objects[]` to `SourceObject` tuples;
5. call `resolve_bgm_reference`;
6. retain resolved row state or row quarantine;
7. collect `resolved.audio_object.key` in one exact-key set.

Do not inspect audio until the cache pass finishes.

- [ ] **Step 9: Reconstruct cache models and fill missing selected audio**

Add private parsers that convert validated row object dictionaries into existing
`RemoteObject`, `SyncError`, and `SimfileInventory` records. Parse manifest timestamps
with:

```python
def _parse_manifest_timestamp(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError("last_modified must be timezone-aware")
    return parsed
```

Before constructing a store, check whether every selected audio object already has a
verified cache body/index entry. When all are cached, skip optional dependency and R2
configuration resolution.

When misses exist:

1. `ensure_r2_dependency()`;
2. `config = R2Config.from_environ(environ)`;
3. verify `config.bucket` and `config.source_endpoint_sha256` match the input manifest;
4. `index = CacheIndexStore.load(request.cache_dir)`;
5. create and validate the store;
6. hold `cache_writer_lock(request.cache_dir)`;
7. call `sync_explicit_cache_keys(..., selected_keys)`.

Use dependency, environment, and store-factory injection parameters matching
`sync_r2_corpus` so tests never need real credentials.

If an individual selected audio object returns a failed cache action, quarantine only
rows referencing that key with `source_audio_download_failed`. Configuration, dependency,
bucket-validation, or cache-index failures abort the command.

- [ ] **Step 10: Second pass — inspect audio and publish event artifacts**

For each row with a resolved BGM:

1. find the post-cache object record;
2. verify its cached body and SHA;
3. call `inspect_source_audio`;
4. call `build_audio_relative_events`;
5. quarantine on returned reason codes;
6. render JSONL and compute `event_sha256`;
7. publish to `output_dir / "events" / f"{event_sha256}.jsonl"` using
   `publish_immutable_content`;
8. retain relative path `events/<sha>.jsonl`, hash, and event count.

Create the `events` directory before publication and use the existing durability helper
for directory creation.

- [ ] **Step 11: Build derived rows**

Copy every input row except old `corpus_version` and `schema_version`. Update only the
selected source-audio record in `objects[]` with post-cache values. Add:

```python
{
    "schema_version": REFERENCE_TIMING_MANIFEST_SCHEMA,
    "source_manifest_sha256": source_manifest_sha256,
    "source_reference_chart_version": source_reference_chart_version,
    "timing_semantics_version": TIMING_SEMANTICS_VERSION,
    "timing_status": "ready" if artifact is not None else "quarantined",
    "timing_reason_codes": sorted(reason_codes),
    "timing_warnings": list(warnings),
    "source_audio_key": source_audio_key,
    "source_audio_content_hash": source_audio_sha256,
    "source_audio_duration_sec": audio.duration_sec if audio else None,
    "source_audio_sample_rate": audio.sample_rate if audio else None,
    "source_audio_channels": audio.channels if audio else None,
    "source_audio_frames": audio.frames if audio else None,
    "bgm_event_count": raw_bgm_event_count,
    "selected_bgm_note_id": selected_bgm_note_id,
    "selected_bgm_chart_time_sec": selected_bgm_chart_time_sec,
    "reference_events_path": artifact.relative_path if artifact else None,
    "reference_events_sha256": artifact.sha256 if artifact else None,
    "reference_event_count": artifact.event_count if artifact else 0,
    "pre_audio_event_count": pre_audio_count,
    "post_audio_event_count": post_audio_count,
}
```

For upstream selection quarantine, set selected audio/BGM fields to `None`, event count
to `0`, and preserve the HPA-322 selection reason fields unchanged.

- [ ] **Step 12: Publish the derived immutable manifest**

Use existing `render_manifest`, `publish_manifest`, and `publish_latest_manifest`.
Counters must reconcile exactly:

```text
ready + quarantined = input row count
events_published = ready
```

Status/exit mapping:

- all ready -> `complete`, `0`;
- any quarantine -> `partial`, `1`;
- fatal exception -> no fake `TimingOutcome`; the CLI emits exit `2`.

- [ ] **Step 13: Add deterministic and partial-publication tests**

Assert:

- identical input/cache bytes produce identical event hashes, manifest bytes, and
  derived `corpus_version`;
- changing the selected audio bytes changes both source-audio hash and manifest identity;
- duplicate rows are rejected before any file write;
- one ambiguous BGM row still publishes another ready row;
- one upstream HPA-322 quarantine remains visible and requires no cache/store access;
- selected audio object enrichment is limited to the exact matching object;
- `ready + quarantined` equals input count;
- publication failure does not return a fake manifest.

- [ ] **Step 14: Run focused checks**

```bash
uv run pytest tests/benchmark/test_reference_timing_manifest.py -q
uv run ruff check src/benchmark/reference_timing_manifest.py tests/benchmark/test_reference_timing_manifest.py
uv run black --check src/benchmark/reference_timing_manifest.py tests/benchmark/test_reference_timing_manifest.py
```

Expected: all pass.

- [ ] **Step 15: Commit Task 5**

```bash
git add src/benchmark/reference_timing_manifest.py tests/benchmark/test_reference_timing_manifest.py
git commit -m "feat: publish audio relative timing manifest"
```

---

### Task 6: Wire the CLI and offline acceptance fixture

**Files:**
- Modify: `src/cli/benchmark.py`
- Modify: `tests/test_cli_benchmark.py`
- Create: `tests/benchmark/test_reference_timing_acceptance.py`

**Interfaces:**
- Consumes: `TimingRequest` and `build_reference_timing_manifest` from Task 5.
- Produces: `crux benchmark build-reference-timing`.
- Produces: one JSON stdout summary and exit codes `0`, `1`, or `2`.

- [ ] **Step 1: Write failing CLI help and request-wiring tests**

```python
def test_build_reference_timing_help_lists_exact_options() -> None:
    result = runner.invoke(main, ["benchmark", "build-reference-timing", "--help"])

    assert result.exit_code == 0
    assert "--manifest FILE" in result.stdout
    assert "--cache-dir DIRECTORY" in result.stdout
    assert "--output-dir DIRECTORY" in result.stdout


def test_build_reference_timing_wires_request(monkeypatch, tmp_path: Path) -> None:
    captured = []
    monkeypatch.setattr(
        "src.benchmark.reference_timing_manifest.build_reference_timing_manifest",
        lambda request: captured.append(request) or complete_timing_outcome(tmp_path),
    )

    result = runner.invoke(
        main,
        [
            "benchmark",
            "build-reference-timing",
            "--manifest",
            str(tmp_path / "reference-charts" / "manifests" / "input.jsonl"),
            "--cache-dir",
            str(tmp_path / "cache"),
            "--output-dir",
            str(tmp_path / "out"),
        ],
    )

    assert result.exit_code == 0
    assert captured[0].cache_dir == tmp_path / "cache"
```

- [ ] **Step 2: Run CLI tests and verify failure**

```bash
uv run pytest tests/test_cli_benchmark.py -k build_reference_timing -q
```

Expected: unknown command.

- [ ] **Step 3: Add the Click command**

Follow the existing lazy-import pattern:

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

    click.echo(
        json.dumps(
            {
                "corpus_version": outcome.manifest.corpus_version,
                "events_published": outcome.counters.events_published,
                "exit_code": outcome.exit_code,
                "manifest_path": str(outcome.manifest.path),
                "quarantined": outcome.counters.quarantined,
                "ready": outcome.counters.ready,
                "status": outcome.status,
            },
            sort_keys=True,
        )
    )
    if outcome.exit_code:
        ctx.exit(outcome.exit_code)
```

Do not log environment values or signed request details.

- [ ] **Step 4: Add summary and fatal-exit tests**

Cover:

- complete output -> exit `0`, one JSON stdout object;
- partial output -> exit `1`, published manifest path retained;
- invalid manifest -> exit `2`, no success JSON;
- missing required R2 configuration -> exit `2`;
- complete cache -> command succeeds without `r2` optional imports.

- [ ] **Step 5: Build the offline acceptance fixture**

Create `tests/benchmark/test_reference_timing_acceptance.py` with five rows:

1. sticky measure length, channel `01`, and one in-bounds drum event;
2. nested selected chart with `../audio/song.wav` remaining inside the simfile prefix;
3. selected audio cache miss served by a fake R2 store;
4. two distinct BGM start groups producing `ambiguous_bgm_start`;
5. pre-audio and post-audio events with one remaining in-bounds event.

The fixture must:

- write a valid `crux.reference-chart-manifest/v1` JSONL;
- build the HPA-321 content-addressed cache and index for selected charts;
- provide valid small WAV bytes for source audio;
- invoke the real Click command;
- inject the fake store at the manifest function boundary;
- assert exit `1` because row 4 is quarantined;
- load the output manifest and verify exact counts/fields;
- read a published event artifact and verify audio-relative times;
- assert no external network call occurs.

- [ ] **Step 6: Run focused CLI and acceptance checks**

```bash
uv run pytest tests/test_cli_benchmark.py -k build_reference_timing -q
uv run pytest tests/benchmark/test_reference_timing_acceptance.py -q
uv run ruff check src/cli/benchmark.py tests/test_cli_benchmark.py tests/benchmark/test_reference_timing_acceptance.py
uv run black --check src/cli/benchmark.py tests/test_cli_benchmark.py tests/benchmark/test_reference_timing_acceptance.py
```

Expected: all pass.

- [ ] **Step 7: Run the complete HPA-323 focused suite**

```bash
uv run pytest \
  tests/benchmark/test_dtx_parser.py \
  tests/benchmark/test_timing.py \
  tests/benchmark/test_corpus_cache.py \
  tests/benchmark/test_corpus_manifest.py \
  tests/benchmark/test_reference_timing.py \
  tests/benchmark/test_reference_timing_manifest.py \
  tests/benchmark/test_reference_timing_acceptance.py \
  tests/test_cli_benchmark.py \
  -q
```

Expected: all pass.

- [ ] **Step 8: Run repository validation**

```bash
uv run pytest -q
uv run ruff check .
uv run black --check .
```

Expected: all pass. If Pylint is part of the current CI workflow after HPA-322 merges,
run the same Pylint command from `.github/workflows/ci.yml` and record only existing
unrelated warnings.

- [ ] **Step 9: Verify artifact determinism twice**

Run the acceptance command twice against the same fixture/output directory. Compare:

```bash
sha256sum artifacts/benchmark/reference-timing/manifests/*.jsonl
sha256sum artifacts/benchmark/reference-timing/events/*.jsonl
```

Expected: the second run creates no different content-addressed files and reports the
same manifest/event hashes.

- [ ] **Step 10: Commit Task 6**

```bash
git add src/cli/benchmark.py tests/test_cli_benchmark.py tests/benchmark/test_reference_timing_acceptance.py
git commit -m "feat: expose reference timing pipeline"
```

---

## Final Review Checklist

- [ ] HPA-322 is merged and the implementation consumes its actual manifest schema and
      selected-chart fields.
- [ ] Channel `01` never appears in generic/native event artifacts.
- [ ] Sticky channel `02` semantics affect both measure starts and positions inside the
      active measure.
- [ ] BPM, fractional-position, and same-beat ordering parity tests still pass.
- [ ] Every BGM source token either contributes to the selected single group or causes
      an actionable quarantine.
- [ ] No filename heuristic can override the DTX WAV reference.
- [ ] Only exact selected source-audio keys are downloaded.
- [ ] Complete-cache reruns require no R2 dependency or credentials.
- [ ] Native lane/note/source identities survive into event artifacts.
- [ ] Pre/post-audio exclusions and frame-tolerance clamps reconcile exactly.
- [ ] Timing-ready plus quarantined equals the HPA-322 input row count.
- [ ] Raw scoring is not auto-shifted or replaced by aligned diagnostics.
- [ ] No HPA-324 taxonomy, HPA-326 inference, or HPA-325 scoring scope leaked into the
      implementation.