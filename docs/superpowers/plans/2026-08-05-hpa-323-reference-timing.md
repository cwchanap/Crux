# HPA-323 Audio-Relative DTX Reference Timing Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Consume the merged HPA-322 `crux.reference-chart-manifest/v1`, derive trustworthy audio-relative native DTX events, and publish immutable artifacts for HPA-324.

**Architecture:** Extend the existing parser/timing engine, expose one thin typed view over HPA-322's merged row validator, use a closed timing-reason contract, reuse HPA-322's cache/key machinery, add exact-key audio cache fill, run a committed corpus diagnostic over the real production helpers, freeze BGM/format policy from that evidence, then publish native events and a derived timing manifest.

**Tech Stack:** Python 3.12, dataclasses, pathlib, strict canonical JSONL, Click, soundfile, existing R2/boto3 adapter, pytest, Ruff, Pylint.

## Global Constraints

- HPA-322 is merged as PR #10; code against merged `main`, not the old planning branch.
- Reuse `ManifestRowView`, `read_verified_cache_body`, `resolve_verified_cache_body`,
  `resolve_inventory_object_key`, `parse_dtx_bytes`, and existing manifest publishers.
- Do not call `manifest_row_view_from_row` directly on an HPA-322 reference-chart row;
  its key set is intentionally HPA-321-only.
- Expose one `ReferenceChartRowView` from `reference_chart_manifest.py`; do not add a
  second manifest/inventory module or source-object class.
- Expose `selected_chart_content_hash` on that typed view; use
  `row.source.inventory.simfile_id` / `row.simfile_id` property instead of duplicating
  simfile ID storage.
- Define `TimingReasonCode = Literal[...]` and validate stable reason arrays from that
  closed type.
- DTX channel `02` is sticky until superseded; this is backed by DTXManiaXG source.
- Channel `01` becomes typed BGM control data and never enters native playable events.
- Keep BGM grouping policy-neutral until the corpus diagnostic is run.
- Group by `(remote.key, measure, position)`, never floating-point time.
- Rename chart-time APIs; keep no `dtx_events_to_timed_events` compatibility alias.
- Require `--cache-dir` explicitly; do not inherit HPA-322's manifest-relative default.
- Verify selected DTX bytes once through `read_verified_cache_body`.
- Verify each selected audio body once unless exact-key fill changes its inventory record.
- Preserve HPA-321's global `is_selected` / `CACHE_PROFILE` behavior.
- Preserve every HPA-322 lineage/selection field; do not overwrite
  `source_manifest_sha256` or `source_corpus_version`.
- Follow the merged schema-golden convention for new stable schemas.
- Keep the corpus diagnostic committed and reproducible; do not use an uncommitted
  throwaway resolver/parser path.
- Split pure manifest contract work from cache/R2 orchestration.

## Primary Timing Evidence

DTXManiaXG Ver.K commit `2e7839d93c00ef528407bebdcf829dafb8c8c804`
keeps `dbBarLength` active after channel `02`; reset to `1.0` is conditional on BMS/BME,
not DTX:

<https://github.com/kairera0467/DTXManiaXG_VerK/blob/2e7839d93c00ef528407bebdcf829dafb8c8c804/DTXMania%E3%83%97%E3%83%AD%E3%82%B8%E3%82%A7%E3%82%AF%E3%83%88/%E3%82%B3%E3%83%BC%E3%83%89/%E3%82%B9%E3%82%B3%E3%82%A2%E3%80%81%E6%9B%B2/CDTX.cs>

---

## File Map

### Create

- `src/benchmark/reference_timing.py`
- `src/benchmark/reference_timing_manifest.py`
- `tests/benchmark/test_reference_timing.py`
- `tests/benchmark/test_reference_timing_manifest.py`
- `tests/benchmark/test_reference_timing_acceptance.py`
- `tests/benchmark/schema_goldens/crux.dtx-reference-event-v1.jsonl`
- `tests/benchmark/schema_goldens/crux.reference-timing-manifest-v1.jsonl`
- `tools/hpa323/analyze_reference_timing.py`
- `tests/tools/test_hpa323_analyze_reference_timing.py`

### Modify

- `src/benchmark/models.py`
- `src/benchmark/dtx_parser.py`
- `src/benchmark/timing.py`
- `src/benchmark/runner.py`
- `src/benchmark/render_audio.py`
- `src/benchmark/reference_chart_manifest.py`
- `src/benchmark/corpus_cache.py`
- `src/benchmark/corpus_manifest.py`
- `src/cli/benchmark.py`
- `tests/benchmark/test_dtx_parser.py`
- `tests/benchmark/test_timing.py`
- `tests/benchmark/test_reference_chart_manifest.py`
- `tests/benchmark/test_corpus_cache.py`
- `tests/benchmark/test_corpus_manifest.py`
- `tests/test_cli_benchmark.py`
- `tests/benchmark/schema_goldens/manifest.json`
- every existing test importing `dtx_events_to_timed_events`

---

## Task 1: Typed BGM events, sticky timing, and explicit clock names

**Files:**
- Modify: `src/benchmark/models.py`
- Modify: `src/benchmark/dtx_parser.py`
- Modify: `src/benchmark/timing.py`
- Modify: `src/benchmark/runner.py`
- Modify: `src/benchmark/render_audio.py`
- Modify: `tests/benchmark/test_dtx_parser.py`
- Modify: `tests/benchmark/test_timing.py`
- Modify: all tests importing `dtx_events_to_timed_events`

**Interfaces:**

```python
DtxEvent.source_order: int = 0

@dataclass(frozen=True)
class DtxBgmEvent:
    chart_id: str
    measure: int
    position: float
    note_id: str
    source_order: int

ParsedDtxChart.bgm_events: list[DtxBgmEvent]

class DtxTimingMap:
    def time_sec(self, event: DtxEvent | DtxBgmEvent | DtxBpmEvent) -> float: ...


def build_dtx_timing_map(chart: ParsedDtxChart) -> DtxTimingMap: ...


def dtx_events_to_chart_time_events(
    chart: ParsedDtxChart,
) -> list[BenchmarkEvent]: ...
```

- [ ] **Step 1: Write failing typed-channel tests**

```python
def test_channel_01_is_typed_bgm_not_native_event() -> None:
    chart = parse_dtx_text(
        "#WAV01: bgm.ogg\n#00101: 0100\n#00111: 0001\n",
        "song",
    )

    assert [(e.measure, e.position, e.note_id) for e in chart.bgm_events] == [
        (1, 0.0, "01")
    ]
    assert [(e.lane_id, e.note_id) for e in chart.events] == [("11", "01")]
```

Add one test proving monotonic `source_order` across non-zero channel `01` and playable
pattern tokens.

- [ ] **Step 2: Add regression proof that channel `01` was never playable**

Cover the existing behavior explicitly:

- `render_audio` ignores lane `01` because it is absent from `DEFAULT_DTX_LANE_MAP`;
- mapping classifies lane `01` only as unmapped diagnostics.

The purpose is to prove that removing `01` from `chart.events` changes control-data
visibility, not playable drum output.

- [ ] **Step 3: Verify parser tests fail before implementation**

```bash
uv run pytest tests/benchmark/test_dtx_parser.py -k "channel_01 or source_order" -q
```

- [ ] **Step 4: Implement typed pattern parsing**

Add `source_order` to `DtxEvent`. Add `DtxBgmEvent` next to `DtxBpmEvent`. Route channel
`01` into `bgm_events`; all other non-control pattern channels remain `DtxEvent`.
Keep pattern source order separate from BPM source order.

- [ ] **Step 5: Replace the incorrect channel `02` fixture**

```python
def test_dtx_measure_length_persists_until_superseded() -> None:
    chart = parse_dtx_text(
        "#BPM: 120\n"
        "#00102: 0.5\n"
        "#00111: 01\n"
        "#00211: 01\n"
        "#00311: 01\n"
        "#00402: 1.0\n"
        "#00411: 01\n"
        "#00511: 01\n",
        "song",
    )

    assert [e.time_sec for e in dtx_events_to_chart_time_events(chart)] == [
        2.0,
        3.0,
        4.0,
        5.0,
        7.0,
    ]
```

Also add:

- multiple sticky changes `0.5 -> 1.5 -> 1.0`;
- BPM change inside a sticky shortened measure with expected `4.5` seconds;
- BGM and playable event parity through one timing map;
- unchanged base BPM/channel `03`/channel `08`/fractional-position fixtures.

- [ ] **Step 6: Implement `DtxTimingMap` by extending existing helpers**

Carry active measure length forward. Use the resolved length for measure starts and
in-measure positions of BPM, BGM, and native events. Extend max-measure discovery with
BGM events. Preserve existing tempo tie behavior.

Do not create a second timing engine; factor the existing
`_measure_lengths_by_measure`, `_measure_start_beats`, `_event_beat`, `_tempo_points`,
and `_time_at_beat` logic behind `DtxTimingMap`.

- [ ] **Step 7: Rename the chart-time API**

Rename `dtx_events_to_timed_events` to `dtx_events_to_chart_time_events` with no alias.
Update every caller and import.

Add to legacy scoring:

```python
# Legacy folder/MIDI scoring uses chart time. HPA-325 consumes HPA-323
# audio-time reference artifacts and must not use this path.
```

- [ ] **Step 8: Run the full blast-radius checks**

```bash
uv run pytest -q
uv run ruff check .
uv run ruff format --check src tests
```

Any changed score/golden expectation must be investigated rather than blindly rewritten.
The real-corpus magnitude of the channel-`02` change is measured later by the committed
Task 4 diagnostic; this step proves repository behavior remains internally coherent.

- [ ] **Step 9: Commit Task 1**

```bash
git add src/benchmark/models.py src/benchmark/dtx_parser.py src/benchmark/timing.py \
  src/benchmark/runner.py src/benchmark/render_audio.py tests
git commit -m "fix: distinguish DTX chart and BGM timing"
```

---

## Task 2: Expose the HPA-322 row view, close reason codes, and resolve BGM groups

**Files:**
- Modify: `src/benchmark/reference_chart_manifest.py`
- Modify: `tests/benchmark/test_reference_chart_manifest.py`
- Create: `src/benchmark/reference_timing.py`
- Create: `tests/benchmark/test_reference_timing.py`

### Interfaces

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

Use the typed property for simfile ID; do not store a second integer field.

Define the stable reason-code contract:

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

Policy-neutral BGM grouping:

```python
@dataclass(frozen=True)
class BgmReferenceGroup:
    remote: RemoteObject
    measure: int
    position: float
    events: tuple[DtxBgmEvent, ...]


@dataclass(frozen=True)
class BgmReferenceSet:
    groups: tuple[BgmReferenceGroup, ...]
    reason_codes: tuple[TimingReasonCode, ...]
    warnings: tuple[str, ...]
    bgm_event_count: int


def resolve_bgm_reference_groups(
    chart: ParsedDtxChart,
    *,
    selected_chart_key: str,
    row: ReferenceChartRowView,
    allow_root_fallback: bool,
) -> BgmReferenceSet: ...
```

`ResolvedBgm.used_root_fallback` is deliberately absent until Task 4 proves that root
fallback should survive. If retained, production communicates its use through a stable
warning rather than precommitting to an unnecessary field.

- [ ] **Step 1: Write failing `ReferenceChartRowView` tests**

Use the merged HPA-322 golden rows. Assert selected and quarantined views, including:

```python
selected = reference_chart_row_view_from_row(selected_row)
assert selected.selection_status == "selected"
assert selected.selected_chart is not None
assert selected.selected_chart.key == selected_row["selected_chart_key"]
assert selected.selected_chart_content_hash == selected_row["selected_chart_content_hash"]
assert selected.simfile_id == selected.source.inventory.simfile_id

quarantined = reference_chart_row_view_from_row(quarantined_row)
assert quarantined.selection_status == "quarantined"
assert quarantined.selected_chart is None
assert quarantined.selected_chart_content_hash is None
```

Also assert malformed/identity-inconsistent rows still fail through the existing HPA-322
validator.

- [ ] **Step 2: Implement the adapter by reusing merged validation**

`reference_chart_row_view_from_row` must call the existing HPA-322 reference-row
validator. Resolve the selected `RemoteObject` from the already-validated inventory and
narrow the selected hash to `str` for selected rows.

Do not duplicate key-set, cache-path, digest, DLEVEL, or selected/nullability validation.

- [ ] **Step 3: Write and implement the closed reason-code contract**

Tests should assert:

```python
TIMING_REASON_CODES == frozenset(get_args(TimingReasonCode))
```

All `BgmReferenceSet` reason codes must be `TimingReasonCode`; no free-form strings.

- [ ] **Step 4: Write shared-resolver mapping tests**

Test mapping of `resolve_inventory_object_key` outcomes:

- `exact` / `casefold` -> use `result.remote`;
- `invalid_path` -> `unsafe_bgm_audio_path`;
- `ambiguous` -> `source_audio_key_ambiguous`;
- `missing` -> optional root retry, then `source_audio_missing`.

Do not repeat HPA-322's normalization/casefold matrix.

- [ ] **Step 5: Write grouping tests without freezing multi-group selection**

Always group by `(remote.key, event.measure, event.position)`, never float time. Cover:

- repeated tokens at one identity collapse to one group and retain source events;
- same file at two positions yields two groups;
- different files at one position yields two groups;
- unresolved WAV note ID -> `unresolved_bgm_wav`;
- zero BGM events -> `bgm_event_missing`.

Do **not** yet choose a winner for multiple groups.

- [ ] **Step 6: Implement policy-neutral group collection**

`resolve_bgm_reference_groups` receives only the typed row view and parsed chart. It uses
`row.source.inventory` and `row.selected_chart` and never indexes the raw HPA-322 mapping.

- [ ] **Step 7: Validate and commit Task 2**

```bash
uv run pytest \
  tests/benchmark/test_reference_chart_manifest.py \
  tests/benchmark/test_reference_timing.py -q
uv run ruff check \
  src/benchmark/reference_chart_manifest.py \
  src/benchmark/reference_timing.py \
  tests/benchmark/test_reference_chart_manifest.py \
  tests/benchmark/test_reference_timing.py
uv run ruff format --check \
  src/benchmark/reference_chart_manifest.py \
  src/benchmark/reference_timing.py \
  tests/benchmark/test_reference_chart_manifest.py \
  tests/benchmark/test_reference_timing.py
git add \
  src/benchmark/reference_chart_manifest.py \
  src/benchmark/reference_timing.py \
  tests/benchmark/test_reference_chart_manifest.py \
  tests/benchmark/test_reference_timing.py
git commit -m "feat: expose reference chart timing inputs"
```

---

## Task 3: Fill only exact selected audio cache keys

**Files:**
- Modify: `src/benchmark/corpus_cache.py`
- Modify: `tests/benchmark/test_corpus_cache.py`

**Interface:**

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

- [ ] **Step 1: Write selector-preservation tests**

Assert existing `sync_cache` still selects exactly the HPA-321 profile: `set.def`,
`.dtx`, and `.txt`. Audio remains outside `is_selected` and `CACHE_PROFILE`.

- [ ] **Step 2: Write explicit-key tests**

Cover:

- one selected audio key while unrelated objects remain unchanged;
- multiple exact keys including candidates from a multi-group chart;
- selected key absent from supplied inventory -> usage/invariant failure;
- cache hit;
- conditional remote change;
- download/install success;
- partial download failure;
- empty selected-key set.

- [ ] **Step 3: Verify tests fail before implementation**

```bash
uv run pytest tests/benchmark/test_corpus_cache.py -k explicit_cache -q
```

- [ ] **Step 4: Extract one internal selector-driven cache path**

Make `sync_cache` delegate to the existing body with `is_selected`; make
`sync_explicit_cache_keys` delegate with exact membership. Do not duplicate locking,
conditional GET, verification, hashing, install, progress, index, or inventory rebuild
logic.

- [ ] **Step 5: Validate and commit Task 3**

```bash
uv run pytest tests/benchmark/test_corpus_cache.py -q
uv run ruff check src/benchmark/corpus_cache.py tests/benchmark/test_corpus_cache.py
uv run ruff format --check src/benchmark/corpus_cache.py tests/benchmark/test_corpus_cache.py
git add src/benchmark/corpus_cache.py tests/benchmark/test_corpus_cache.py
git commit -m "feat: cache exact benchmark audio keys"
```

---

## Task 4: Commit the corpus diagnostic and freeze timing/BGM/format policy

This task replaces the old throwaway Task 0. It runs **after** Tasks 1-3 so it can use
the real typed parser, timing map, HPA-322 adapter, BGM group resolver, and exact-key
cache fill.

**Files:**
- Create: `tools/hpa323/analyze_reference_timing.py`
- Create: `tests/tools/test_hpa323_analyze_reference_timing.py`
- Modify: `src/benchmark/reference_timing.py` only if the frozen BGM policy needs a
  small selection helper/warning.
- Modify: `tests/benchmark/test_reference_timing.py` for the frozen policy.

**Command shape:**

```bash
uv run python tools/hpa323/analyze_reference_timing.py \
  --manifest artifacts/benchmark/reference-charts/manifests/<sha>.jsonl \
  --cache-dir artifacts/benchmark/r2-corpus/cache \
  --output artifacts/benchmark/reference-timing-analysis/bgm-layout.json \
  --audio-sample-limit 50
```

The tool may fill the **local** cache for sampled exact audio keys through
`sync_explicit_cache_keys`; it never changes R2 or broadens the default cache profile.

- [ ] **Step 1: Write diagnostic-unit tests around production seams**

Use fixture rows and fake store/cache. Assert the tool calls:

- `reference_chart_row_view_from_row`;
- `read_verified_cache_body` + `parse_dtx_bytes` for selected charts;
- `resolve_bgm_reference_groups` for BGM grouping;
- `sync_explicit_cache_keys` for sampled missing audio;
- `resolve_verified_cache_body` before `sf.info`;
- `build_dtx_timing_map` for both corrected and diagnostic-legacy comparisons.

No private HPA-322 validator or private path/casefold helper is allowed.

- [ ] **Step 2: Report BGM group and path-resolution distribution**

Write canonical sorted JSON containing:

```text
selected_rows
upstream_quarantined_rows
rows_with_0_bgm_groups
rows_with_1_bgm_group
rows_with_multiple_bgm_groups
rows_with_unresolved_wav
rows_needing_case_insensitive_match
rows_needing_simfile_root_fallback
multi_group_examples[]
```

Cap examples at 25 and retain simfile ID, chart key, object keys, note IDs, measures,
and positions only.

- [ ] **Step 3: Report authored audio extension distribution**

Count extension values from the selected charts' referenced BGM `#WAVxx` strings:

```text
bgm_extension_counts
```

Normalize extension case for counting only. Do not use extensions to decide decodability.

- [ ] **Step 4: Sample actual audio decodability**

For up to `--audio-sample-limit` unique resolved candidate audio objects:

1. fill the exact key if its verified body is unavailable;
2. verify the installed body;
3. call `soundfile.info(path)` on the content-addressed, extensionless body;
4. catch `(OSError, RuntimeError, ValueError, sf.LibsndfileError)`;
5. record:

```text
sampled_audio_count
sampled_audio_decodable_count
sampled_audio_undecodable_count
sampled_audio_undecodable_by_extension
```

The report distinguishes authored filename extension from actual decoder result.

- [ ] **Step 5: Measure real channel-`02` blast radius**

Record:

```text
charts_with_channel_02
charts_with_multiple_channel_02_changes
max_channel_02_time_delta_sec
channel_02_delta_examples[]
```

For each selected chart:

1. build the corrected `DtxTimingMap`;
2. construct a diagnostic chart copy where every measure up to the chart max has an
   explicit length `chart.measure_lengths.get(measure, 1.0)`;
3. run the **same corrected timing map** on that copy; explicit `1.0` values reproduce
   the old per-measure reset behavior without preserving a second timing engine;
4. compare native event times by `source_order`;
5. record the maximum absolute delta and a bounded set of affected examples.

- [ ] **Step 6: Verify deterministic diagnostics**

Run the tool twice against the same fixture and assert byte-identical report output.
The diagnostic is committed/re-runnable but is not a stable schema golden.

- [ ] **Step 7: Run the real corpus diagnostic**

Run against the actual HPA-322 output/cache and review:

- multiple BGM group frequency;
- root fallback frequency;
- extension distribution;
- `sf.info` failure rate and representative extensions;
- channel-`02` affected-row count and largest timing deltas.

- [ ] **Step 8: Freeze policy before Task 5**

Decision rules:

- exceptional multi-group rows -> `ambiguous_bgm_start` quarantine;
- common multi-group rows -> inspect representative authored charts and amend the
  design before selecting a rule;
- zero root-fallback rows -> remove the root retry branch entirely;
- material decoder failures -> freeze supported-format/quarantine behavior before
  native event publication;
- material channel-`02` deltas -> inspect representative charts before landing;
- never choose the earliest BGM group merely to increase yield.

If conservative multi-group quarantine is retained, add a small pure selector:

```python
def select_bgm_reference(
    references: BgmReferenceSet,
    timing_map: DtxTimingMap,
) -> BgmResolution: ...
```

One group selects its lowest-source-order event; repeated tokens in the same group emit a
warning. Multiple groups emit `ambiguous_bgm_start`.

If root fallback survives, add only the measured compatibility branch and a deterministic
warning. Keep `ResolvedBgm.used_root_fallback` out unless a later consumer actually needs
that boolean.

- [ ] **Step 9: Record evidence and commit Task 4**

Add the real report counts/policy to HPA-323 and the implementation PR description.
Commit the tool and frozen policy code/tests:

```bash
git add \
  tools/hpa323/analyze_reference_timing.py \
  tests/tools/test_hpa323_analyze_reference_timing.py \
  src/benchmark/reference_timing.py \
  tests/benchmark/test_reference_timing.py
git commit -m "test: measure reference timing corpus behavior"
```

---

## Task 5: Audio metadata, bounded native events, and event schema golden

**Files:**
- Modify: `src/benchmark/reference_timing.py`
- Modify: `tests/benchmark/test_reference_timing.py`
- Modify: `src/benchmark/corpus_manifest.py`
- Modify: `tests/benchmark/test_corpus_manifest.py`
- Create: `tests/benchmark/schema_goldens/crux.dtx-reference-event-v1.jsonl`
- Modify: `tests/benchmark/schema_goldens/manifest.json`

**Interfaces:**

```python
@dataclass(frozen=True)
class SourceAudioInfo:
    duration_sec: float
    sample_rate: int
    channels: int
    frames: int

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
class AudioRelativeReference:
    events: tuple[NativeReferenceEvent, ...]
    pre_audio_event_count: int
    post_audio_event_count: int


def inspect_source_audio(path: Path) -> SourceAudioInfo: ...


def build_audio_relative_events(
    chart: ParsedDtxChart,
    timing_map: DtxTimingMap,
    *,
    simfile_id: int,
    selected_chart_key: str,
    selected_chart_content_hash: str,
    source_audio_key: str,
    source_audio_content_hash: str,
    bgm_chart_time_sec: float,
    audio: SourceAudioInfo,
) -> AudioRelativeReference: ...
```

- [ ] **Step 1: Write metadata tests**

Use small generated WAV fixtures. Cover valid metadata, zero frames, and an unreadable
body.

Use the same expected exception family as `render_audio.py`:
`OSError`, `RuntimeError`, `ValueError`, `sf.LibsndfileError`.

- [ ] **Step 2: Write audio-relative bounds tests**

Cover:

- BGM shift;
- exact zero and exact duration;
- one-frame negative/late clamp;
- materially negative/late exclusions and counters;
- non-finite time -> `non_finite_reference_time`;
- zero retained events -> `no_in_bounds_reference_events`;
- source-order and native identity preservation.

- [ ] **Step 3: Implement metadata and event construction**

Use `soundfile.info` without waveform decode. One audio frame is the tolerance.

- [ ] **Step 4: Render deterministic event JSONL**

Define `REFERENCE_EVENT_SCHEMA = "crux.dtx-reference-event/v1"`. Use the existing
canonical JSON helper per event. Sort by deterministic native identity and verify repeated
rendering produces byte-identical output/hash.

- [ ] **Step 5: Expose the existing immutable byte publisher**

Add only:

```python
def publish_immutable_content(
    path: Path,
    content: bytes,
    expected_sha256: str,
) -> None:
    _publish_immutable(path, content, expected_sha256)
```

Do not duplicate durability/conflict logic.

- [ ] **Step 6: Add and register the event schema golden**

Create `crux.dtx-reference-event-v1.jsonl`, add it to the schema-golden registry, and
implement `validate_schema_golden` support in `reference_timing.py`.

The validator must reject unknown/missing keys, invalid timing reason-independent
identity fields, non-finite times, and invalid SHA-256 values.

- [ ] **Step 7: Validate and commit Task 5**

```bash
uv run pytest \
  tests/benchmark/test_reference_timing.py \
  tests/benchmark/test_corpus_manifest.py \
  tests/benchmark/test_schema_goldens.py -q
uv run ruff check \
  src/benchmark/reference_timing.py \
  src/benchmark/corpus_manifest.py \
  tests/benchmark/test_reference_timing.py \
  tests/benchmark/test_corpus_manifest.py \
  tests/benchmark/test_schema_goldens.py
uv run ruff format --check \
  src/benchmark/reference_timing.py \
  src/benchmark/corpus_manifest.py \
  tests/benchmark/test_reference_timing.py \
  tests/benchmark/test_corpus_manifest.py \
  tests/benchmark/test_schema_goldens.py
git add \
  src/benchmark/reference_timing.py \
  src/benchmark/corpus_manifest.py \
  tests/benchmark/test_reference_timing.py \
  tests/benchmark/test_corpus_manifest.py \
  tests/benchmark/schema_goldens/crux.dtx-reference-event-v1.jsonl \
  tests/benchmark/schema_goldens/manifest.json
git commit -m "feat: publish bounded reference events"
```

---

## Task 6a: Pure timing-manifest contract, lineage, rendering, and golden

This task is intentionally offline and reviewable without R2/cache orchestration.

**Files:**
- Create: `src/benchmark/reference_timing_manifest.py`
- Create: `tests/benchmark/test_reference_timing_manifest.py`
- Create: `tests/benchmark/schema_goldens/crux.reference-timing-manifest-v1.jsonl`
- Modify: `tests/benchmark/schema_goldens/manifest.json`

**Interfaces:**

```python
REFERENCE_TIMING_MANIFEST_SCHEMA = "crux.reference-timing-manifest/v1"
TIMING_SEMANTICS_VERSION = "crux.dtx-audio-timing/v1"

@dataclass(frozen=True)
class ReferenceTimingRequest:
    manifest_path: Path
    cache_dir: Path
    output_dir: Path

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

No `report_path`.

- [ ] **Step 1: Write canonical HPA-322 source-loading tests**

Require:

- canonical JSONL with one final newline;
- `crux.reference-chart-manifest/v1` rows;
- validation through `reference_chart_row_view_from_row`;
- unique simfile IDs;
- one shared HPA-322 input `corpus_version` and source identity;
- exact input-byte SHA-256;
- input `corpus_version` reproducible via `render_manifest` after removing that field.

Reject malformed schema, duplicate IDs, mixed source identities, invalid selected/null
shape, and inconsistent derived corpus version.

Do not call `manifest_row_view_from_row(hpa322_row)` directly.

- [ ] **Step 2: Test lineage preservation and field naming**

Preserve HPA-322 `source_manifest_sha256` and `source_corpus_version` unchanged. Remove
the HPA-322 top-level `corpus_version` before rendering the derived row and add:

```text
source_reference_chart_manifest_sha256
source_reference_chart_version
```

The first equals the exact HPA-322 input-byte SHA-256. The second equals the HPA-322
input `corpus_version`.

- [ ] **Step 3: Write pure derived-row tests**

Without any R2/store/cache behavior, test rendering of:

- ready row with complete source-audio/event metadata;
- upstream HPA-322 quarantined row ->
  `upstream_chart_selection_unavailable`;
- HPA-323 quarantined row with source/event fields null;
- exact pass-through of all other HPA-322 fields;
- deterministic warning/reason ordering;
- `TimingReasonCode` rejection for unknown strings.

- [ ] **Step 4: Add outcome accounting semantics**

Pure counters must support:

```text
ready + quarantined = input rows
upstream_quarantined_count <= quarantined_count
events_published = ready
```

Exit convention remains:

- no quarantines -> `0`;
- any upstream or HPA-323 quarantine with published manifest -> `1`;
- fatal loading/publication preparation -> `2`.

The separate `upstream_quarantined_count` provides signal without changing the existing
exit-code convention.

- [ ] **Step 5: Add timing-manifest schema golden**

Register `crux.reference-timing-manifest/v1` in the shared schema-golden registry.
Create a canonical golden with one ready and one quarantined row and a valid derived
corpus version.

`validate_schema_golden` must use `get_args(TimingReasonCode)` for reason validation.

- [ ] **Step 6: Validate and commit Task 6a**

```bash
uv run pytest \
  tests/benchmark/test_reference_timing_manifest.py \
  tests/benchmark/test_schema_goldens.py -q
uv run ruff check \
  src/benchmark/reference_timing_manifest.py \
  tests/benchmark/test_reference_timing_manifest.py \
  tests/benchmark/test_schema_goldens.py
uv run ruff format --check \
  src/benchmark/reference_timing_manifest.py \
  tests/benchmark/test_reference_timing_manifest.py \
  tests/benchmark/test_schema_goldens.py
git add \
  src/benchmark/reference_timing_manifest.py \
  tests/benchmark/test_reference_timing_manifest.py \
  tests/benchmark/schema_goldens/crux.reference-timing-manifest-v1.jsonl \
  tests/benchmark/schema_goldens/manifest.json
git commit -m "feat: define reference timing manifest contract"
```

---

## Task 6b: Cache/R2 orchestration, event publication, and acceptance

**Files:**
- Modify: `src/benchmark/reference_timing_manifest.py`
- Modify: `tests/benchmark/test_reference_timing_manifest.py`
- Create: `tests/benchmark/test_reference_timing_acceptance.py`

- [ ] **Step 1: Write first-pass verification tests**

For a selected HPA-322 row:

1. `ReferenceChartRowView.selected_chart` must be present;
2. call `read_verified_cache_body` exactly once with
   `expected_sha256=row.selected_chart_content_hash`;
3. parse those exact bytes with `parse_dtx_bytes`;
4. build `DtxTimingMap`;
5. resolve BGM groups and apply the Task 4 frozen selection policy;
6. resolve selected source-audio cache body once if already verified;
7. queue only missing/unverified selected audio.

Map row-local chart/cache/parse/timing/BGM failures to `TimingReasonCode` without aborting
valid sibling rows.

- [ ] **Step 2: Write no-R2-on-complete-cache tests**

When every selected audio body is already verified:

- optional R2 module import is not triggered;
- config/credential/store factories are not called;
- exact-key sync is not called.

- [ ] **Step 3: Write targeted-fill tests**

When misses exist:

- validate R2 config identity against the embedded source endpoint/bucket;
- load the existing cache index;
- create the existing store and lock through existing machinery;
- call `sync_explicit_cache_keys` with only exact selected audio keys/inventories;
- merge returned inventories by simfile ID;
- reverify only rows whose inventory changed;
- map failed selected audio downloads to `source_audio_download_failed` or
  `source_audio_cache_invalid` as appropriate;
- preserve unrelated object records exactly.

- [ ] **Step 4: Write metadata/event publication tests**

For resolved audio:

- inspect metadata;
- build bounded audio-relative native events;
- render event JSONL;
- publish `events/<sha256>.jsonl` through `publish_immutable_content`;
- identical second run reuses/verifies the existing immutable artifact;
- event publication failure is fatal because a ready manifest row cannot safely point to
  a missing/conflicting immutable artifact.

- [ ] **Step 5: Write orchestration accounting tests**

Cover combinations:

- all ready -> exit `0`;
- only upstream quarantines -> exit `1`, `upstream_quarantined_count > 0`;
- HPA-323-specific quarantine -> exit `1`, distinguishable as
  `quarantined_count - upstream_quarantined_count`;
- mixed upstream + HPA-323 quarantine;
- fatal config/index/publication -> exit `2`, `manifest is None`.

- [ ] **Step 6: Implement orchestration and immutable timing-manifest publication**

Use `render_manifest`, `publish_manifest`, and `publish_latest_manifest`. Do not add a
separate report artifact.

- [ ] **Step 7: Build the end-to-end offline acceptance fixture**

Cover at least:

1. selected chart + already-cached source audio -> ready;
2. selected chart + exact-key audio fill through fake store -> ready;
3. upstream HPA-322 quarantine -> preserved timing quarantine;
4. unresolved/ambiguous BGM row -> HPA-323 quarantine;
5. unreadable source audio -> `source_audio_decode_failed`;
6. repeated second run -> deterministic manifest/event identities.

Assert complete-cache acceptance does not require R2 extras.

- [ ] **Step 8: Validate and commit Task 6b**

```bash
uv run pytest \
  tests/benchmark/test_reference_timing.py \
  tests/benchmark/test_reference_timing_manifest.py \
  tests/benchmark/test_reference_timing_acceptance.py -q
uv run ruff check \
  src/benchmark/reference_timing.py \
  src/benchmark/reference_timing_manifest.py \
  tests/benchmark/test_reference_timing.py \
  tests/benchmark/test_reference_timing_manifest.py \
  tests/benchmark/test_reference_timing_acceptance.py
uv run ruff format --check \
  src/benchmark/reference_timing.py \
  src/benchmark/reference_timing_manifest.py \
  tests/benchmark/test_reference_timing.py \
  tests/benchmark/test_reference_timing_manifest.py \
  tests/benchmark/test_reference_timing_acceptance.py
git add \
  src/benchmark/reference_timing.py \
  src/benchmark/reference_timing_manifest.py \
  tests/benchmark/test_reference_timing.py \
  tests/benchmark/test_reference_timing_manifest.py \
  tests/benchmark/test_reference_timing_acceptance.py
git commit -m "feat: build audio-relative reference corpus"
```

---

## Task 7: CLI and final CI-equivalent verification

**Files:**
- Modify: `src/cli/benchmark.py`
- Modify: `tests/test_cli_benchmark.py`

- [ ] **Step 1: Write CLI path-contract tests**

Require:

```text
--manifest PATH
--cache-dir PATH
```

Default only `--output-dir`.

Add a test showing why `--cache-dir` cannot inherit `select-reference-charts` behavior:
for `.../reference-charts/manifests/<sha>.jsonl`,
`manifest.parent.parent / "cache"` points at `reference-charts/cache`, not
`r2-corpus/cache`.

- [ ] **Step 2: Write CLI summary tests**

Summary contains at least:

```text
status
exit_code
manifest_path
manifest_sha256
corpus_version
ready_count
quarantined_count
upstream_quarantined_count
events_published
```

Assert sorted canonical JSON output and exits `0`, `1`, and `2`.

One test should demonstrate the operational signal:

```text
quarantined_count = 7
upstream_quarantined_count = 5
```

so the operator can see two timing-stage quarantines despite exit `1` being shared with
upstream gaps.

- [ ] **Step 3: Implement lazy CLI wiring**

Add `build-reference-timing`, construct `ReferenceTimingRequest`, emit one summary, and
exit through `click.Context.exit`.

A complete-cache path must not import/create the optional R2 store.

- [ ] **Step 4: Run final validation**

Use the same checks as merged HPA-322 implementation PR #10:

```bash
uv run --extra r2 pytest -q
uv run ruff check .
uv run ruff format --check src tests
uv run pylint --errors-only --disable=E1120,E0401 src
```

Also run:

```bash
git diff --check main...HEAD
```

Verify the committed corpus diagnostic still reproduces byte-identical output for its
fixture.

- [ ] **Step 5: Final review checklist**

Reject the implementation if any of these remain:

- direct `manifest_row_view_from_row(hpa322_row)` call;
- raw HPA-322 dict lookup for selected chart hash or simfile ID in production timing
  code;
- free-form timing reason strings outside `TimingReasonCode`;
- private path/casefold resolver;
- second cache verifier;
- widened `is_selected` / `CACHE_PROFILE`;
- second verification of already-cached audio;
- overwrite of HPA-322 `source_manifest_sha256` or `source_corpus_version`;
- `report_path` field without a report artifact;
- uncommitted/throwaway corpus policy script;
- unconditional `used_root_fallback` field before evidence;
- one giant manifest+R2 orchestration commit.

- [ ] **Step 6: Commit Task 7**

```bash
git add src/cli/benchmark.py tests/test_cli_benchmark.py
git commit -m "feat: expose reference timing build"
```

---

## Risk Verification Matrix

| Risk | Required proof |
|---|---|
| HPA-322 row fed directly to HPA-321 parser | Production loader uses `ReferenceChartRowView` |
| Selected-chart identity leaks back to raw dicts | Typed selected hash + typed simfile property |
| Timing reason typo enters stable schema | `TimingReasonCode = Literal[...]` + validator uses `get_args` |
| Immediate lineage overwrites HPA-321 lineage | Dedicated `source_reference_chart_*` fields |
| Incorrect channel `02` semantics | Primary source + fixtures + committed corpus delta report |
| Timing change materially shifts many real charts unnoticed | `charts_with_*channel_02*` + max delta examples |
| BGM policy quarantines too much corpus | Committed group-distribution diagnostic + sampled review |
| Unsupported/corrupt source audio sinks yield late | Extension distribution + sampled verified-body `sf.info` |
| BGM diagnostic diverges from production resolver | Diagnostic imports production adapter/group resolver |
| HPA-323 forks object-key behavior | All object lookup uses `resolve_inventory_object_key` |
| Timing rename breaks legacy consumers | No alias + full repository suite immediately after rename |
| Existing audio is hashed twice | Already-verified rows bypass post-fill verification |
| Targeted fill mutates unrelated objects | Before/after inventory equality tests |
| Complete cache still requires R2 | Dependency/store factories are not called |
| Upstream gaps hide HPA-323 health | `upstream_quarantined_count` in outcome/CLI |
| Stable artifact contract drifts | Event + timing schema goldens |
| Manifest orchestration becomes unreviewable | Task 6a pure contract split from Task 6b R2/orchestration |

## Final Sequence

1. **Task 1** — typed BGM, `source_order`, sticky timing, clock rename.
2. **Task 2** — typed HPA-322 row view, `TimingReasonCode`, policy-neutral BGM groups.
3. **Task 3** — exact-key source-audio cache fill.
4. **Task 4** — committed real-corpus diagnostic; freeze multi-group, fallback, audio
   support, and inspect channel-`02` blast radius.
5. **Task 5** — audio metadata, bounded native events, immutable event artifact/golden.
6. **Task 6a** — pure timing-manifest loading, lineage, row rendering, manifest golden.
7. **Task 6b** — cache/R2 orchestration, event publication, acceptance fixture.
8. **Task 7** — CLI with upstream quarantine visibility and CI-equivalent verification.
