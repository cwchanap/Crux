# HPA-323 Audio-Relative DTX Reference Timing Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Consume the merged HPA-322 `crux.reference-chart-manifest/v1`, derive trustworthy audio-relative native DTX events, and publish immutable artifacts for HPA-324.

**Architecture:** Measure real BGM layouts first, extend the existing parser/timing engine with evidence-backed DTX semantics and explicit clock names, expose one thin public HPA-322 row view over the merged validator, reuse all merged cache/key contracts, cache only exact selected audio misses, and publish content-addressed native event JSONL plus a derived manifest.

**Tech Stack:** Python 3.12, dataclasses, pathlib, strict canonical JSONL, Click, soundfile, existing R2/boto3 adapter, pytest, Ruff, Pylint.

## Global Constraints

- HPA-322 is merged as PR #10; do not code against the old planning-only shape.
- Reuse `ManifestRowView`, `read_verified_cache_body`, `resolve_verified_cache_body`,
  `resolve_inventory_object_key`, `parse_dtx_bytes`, and the existing manifest publishers.
- Do not call `manifest_row_view_from_row` directly on an HPA-322 reference-chart row;
  its key set is intentionally HPA-321-only.
- Expose one `ReferenceChartRowView` from `reference_chart_manifest.py`; do not create
  another manifest/inventory module or source-object class.
- DTX channel `02` is sticky until superseded; this is backed by DTXManiaXG source.
- Channel `01` becomes typed BGM control data and never enters native playable events.
- Measure real BGM group/root-fallback distribution before freezing multi-group policy.
- Rename chart-time APIs; keep no ambiguous `dtx_events_to_timed_events` alias.
- Require `--cache-dir` explicitly.
- Verify selected DTX bytes once through `read_verified_cache_body`.
- Verify each selected audio body once unless targeted fill changes its inventory record.
- Preserve HPA-321's global `is_selected` / `CACHE_PROFILE` behavior.
- Preserve every HPA-322 lineage/selection field; do not overwrite
  `source_manifest_sha256` or `source_corpus_version`.
- Follow the merged schema-golden convention for new stable schemas.
- Run the full test suite immediately after the timing-semantics change.

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

### Task 0: Measure real BGM group and fallback distribution

**Files:**
- No committed production files.
- Local output: `artifacts/benchmark/reference-timing-analysis/bgm-layout.json`

**Consumes:** merged HPA-322 manifest/cache, `_validate_reference_row` for temporary
analysis only, `read_verified_cache_body`, `parse_dtx_bytes`, and
`resolve_inventory_object_key`.

- [ ] **Step 1: Run a read-only analysis over a real HPA-322 manifest**

For each canonical JSONL row:

1. require `schema_version == "crux.reference-chart-manifest/v1"`;
2. call the existing private `_validate_reference_row(row)` only in this temporary
   uncommitted script to obtain the embedded `ManifestRowView`;
3. skip `selection_status != "selected"` except for total counts;
4. locate the exact selected `RemoteObject` using `selected_chart_key`;
5. read it with:

```python
chart_bytes = read_verified_cache_body(
    cache_dir,
    selected_remote,
    source_endpoint_sha256=source_view.source_endpoint_sha256,
    bucket=source_view.source_bucket,
    expected_sha256=row["selected_chart_content_hash"],
)
chart = parse_dtx_bytes(
    chart_bytes,
    chart_id=str(row["simfile_id"]),
    source_name=str(row["selected_chart_key"]),
)
```

6. before Task 1 lands, treat current generic events with `lane_id == "01"` as the BGM
   observations;
7. resolve each observation's `#WAVxx` value using `resolve_inventory_object_key` relative
   to the selected-chart directory;
8. after a relative `missing`, call the same resolver from the simfile root and record
   whether that compatibility fallback is actually needed;
9. group resolved observations by `(remote.key, measure, position)`.

Do not implement another path normalizer, traversal check, exact matcher, or casefold
index in the analysis script.

- [ ] **Step 2: Write the measurement report**

```json
{
  "selected_rows": 0,
  "rows_with_0_bgm_groups": 0,
  "rows_with_1_bgm_group": 0,
  "rows_with_multiple_bgm_groups": 0,
  "rows_with_unresolved_wav": 0,
  "rows_needing_case_insensitive_match": 0,
  "rows_needing_simfile_root_fallback": 0,
  "multi_group_examples": []
}
```

Cap examples at 25 and retain only simfile ID, chart key, object keys, note IDs,
measures, and positions. Do not copy source bodies into the report.

- [ ] **Step 3: Freeze policy before Task 2**

- exceptional multi-group rows -> retain `ambiguous_bgm_start` quarantine;
- common multi-group rows -> inspect representative authored charts and amend the design
  before implementing BGM selection;
- zero root-fallback rows -> remove the root fallback entirely;
- do not choose the earliest group without authored-chart evidence.

Record counts and the frozen policy in HPA-323 and the implementation PR.

---

### Task 1: Typed BGM events, sticky timing, and explicit clock names

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

- [ ] **Step 2: Verify parser tests fail**

```bash
uv run pytest tests/benchmark/test_dtx_parser.py -k "channel_01 or source_order" -q
```

- [ ] **Step 3: Implement typed pattern parsing**

Add `source_order` to `DtxEvent`. Add `DtxBgmEvent` next to `DtxBpmEvent`. Route channel
`01` into `bgm_events`; all other non-control pattern channels remain `DtxEvent`.
Keep pattern source order separate from BPM source order.

- [ ] **Step 4: Replace the incorrect channel `02` fixture**

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
- BGM and playable event parity through one timing map.

- [ ] **Step 5: Implement `DtxTimingMap`**

Carry the active measure length forward. Use the resolved length for measure starts and
in-measure positions of BPM, BGM, and native events. Extend max-measure discovery with
BGM events. Preserve existing tempo tie behavior.

- [ ] **Step 6: Rename the chart-time API**

Rename `dtx_events_to_timed_events` to `dtx_events_to_chart_time_events` with no alias.
Update every caller and import.

Add to legacy scoring:

```python
# Legacy folder/MIDI scoring uses chart time. HPA-325 consumes HPA-323
# audio-time reference artifacts and must not use this path.
```

- [ ] **Step 7: Run the full blast-radius checks**

```bash
uv run pytest -q
uv run ruff check .
uv run ruff format --check src tests
```

Any changed score/golden expectation must be investigated rather than blindly rewritten.

- [ ] **Step 8: Commit Task 1**

```bash
git add src/benchmark/models.py src/benchmark/dtx_parser.py src/benchmark/timing.py \
  src/benchmark/runner.py src/benchmark/render_audio.py tests
git commit -m "fix: distinguish DTX chart and BGM timing"
```

---

### Task 2: Expose the merged HPA-322 row view and resolve BGM identity

**Files:**
- Modify: `src/benchmark/reference_chart_manifest.py`
- Modify: `tests/benchmark/test_reference_chart_manifest.py`
- Create: `src/benchmark/reference_timing.py`
- Create: `tests/benchmark/test_reference_timing.py`

**Interfaces:**

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

and:

```python
@dataclass(frozen=True)
class ResolvedBgm:
    remote: RemoteObject
    event: DtxBgmEvent
    chart_time_sec: float
    used_root_fallback: bool

@dataclass(frozen=True)
class BgmResolution:
    status: Literal["resolved", "quarantined"]
    selected: ResolvedBgm | None
    reason_codes: tuple[str, ...]
    warnings: tuple[str, ...]
    bgm_event_count: int
    bgm_group_count: int
```

- [ ] **Step 1: Write failing `ReferenceChartRowView` tests**

Use the merged HPA-322 golden rows. Assert:

```python
selected = reference_chart_row_view_from_row(selected_row)
assert selected.selection_status == "selected"
assert selected.selected_chart is not None
assert selected.selected_chart.key == selected_row["selected_chart_key"]
assert selected.source.inventory.simfile_id == selected_row["simfile_id"]

quarantined = reference_chart_row_view_from_row(quarantined_row)
assert quarantined.selection_status == "quarantined"
assert quarantined.selected_chart is None
```

Also assert malformed/identity-inconsistent rows still fail through the existing HPA-322
validator.

- [ ] **Step 2: Implement the adapter by reusing merged validation**

`reference_chart_row_view_from_row` must call the existing `_validate_reference_row`.
For a selected row, find the exact `RemoteObject` by the already-validated
`selected_chart_key`. Convert reason/warning lists to tuples. Do not duplicate key-set,
cache-path, digest, DLEVEL, or selected/nullability validation.

- [ ] **Step 3: Write BGM resolver-status tests**

Test mapping of shared `resolve_inventory_object_key` outcomes:

- `exact` / `casefold` -> use `result.remote`;
- `invalid_path` -> `unsafe_bgm_audio_path`;
- `ambiguous` -> `source_audio_key_ambiguous`;
- `missing` -> optional measured root retry, then `source_audio_missing`.

Do not repeat HPA-322's normalization/casefold matrix.

- [ ] **Step 4: Write group-policy tests from Task 0**

Always group by `(remote.key, event.measure, event.position)`, never float time. Cover
repeated tokens at one identity and multiple distinct groups according to the policy
frozen in Task 0.

- [ ] **Step 5: Implement pure BGM resolution**

`resolve_bgm_reference` receives `ReferenceChartRowView`, the parsed chart, timing map,
and selected chart key. Resolve all BGM note IDs through `wav_table`, call only the
shared object-key resolver, preserve event/group counts, and compute chart time only for
the selected event.

- [ ] **Step 6: Validate and commit Task 2**

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
git commit -m "feat: expose reference chart timing input"
```

---

### Task 3: Fill only exact selected source-audio cache keys

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

- [ ] **Step 1: Write failing explicit-key tests**

Use one inventory containing `real.dtx`, `bgm.ogg`, and `preview.ogg`; select only
`bgm.ogg`. Assert only the requested key is opened or mutated. Cover cache hit, download,
failed download, absent requested key, empty key, and unrelated object preservation.

- [ ] **Step 2: Verify failure**

```bash
uv run pytest tests/benchmark/test_corpus_cache.py -k explicit_cache -q
```

- [ ] **Step 3: Extract one selector-driven internal worker**

Keep public `sync_cache` unchanged and have it delegate with the existing `is_selected`.
Have `sync_explicit_cache_keys` delegate with exact key membership. Do not duplicate
locking, conditional GET, download validation, hashing, index publication, installation,
or inventory rebuilding.

- [ ] **Step 4: Validate and commit Task 3**

```bash
uv run pytest tests/benchmark/test_corpus_cache.py -q
uv run ruff check src/benchmark/corpus_cache.py tests/benchmark/test_corpus_cache.py
uv run ruff format --check src/benchmark/corpus_cache.py tests/benchmark/test_corpus_cache.py
git add src/benchmark/corpus_cache.py tests/benchmark/test_corpus_cache.py
git commit -m "feat: cache exact benchmark audio keys"
```

---

### Task 4: Build and publish bounded native event artifacts

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
    chart_id: str
    lane_id: str
    note_id: str
    measure: int
    position: float
    source_order: int
    chart_time_sec: float
    audio_time_sec: float

@dataclass(frozen=True)
class AudioRelativeReference:
    events: tuple[NativeReferenceEvent, ...]
    pre_audio_event_count: int
    post_audio_event_count: int


def inspect_source_audio(path: Path) -> SourceAudioInfo: ...


def build_audio_relative_events(...) -> AudioRelativeReference: ...


def render_reference_event_jsonl(...) -> bytes: ...
```

- [ ] **Step 1: Write source-audio metadata tests**

Generate a small WAV and assert duration, samplerate, channels, and frames from
`soundfile.info`. Cover undecodable and zero-frame files.

- [ ] **Step 2: Write bounds tests**

Cover BGM subtraction, exact boundaries, one-frame clamps, larger pre/post exclusions,
non-finite time, zero retained events, and source-order preservation.

- [ ] **Step 3: Implement audio-time events**

Use:

```text
chart_time_sec = timing_map.time_sec(event)
audio_time_sec = chart_time_sec - selected_bgm_chart_time_sec
```

Do not map lanes or deduplicate hits. Sort deterministically before rendering.

- [ ] **Step 4: Expose the corpus immutable-byte publisher**

Add only:

```python
def publish_immutable_content(
    path: Path,
    content: bytes,
    expected_sha256: str,
) -> None:
    _publish_immutable(path, content, expected_sha256)
```

Test new publication, identical reuse, wrong expected hash, and conflicting existing
bytes. Do not create another publication implementation.

- [ ] **Step 5: Add the event schema golden**

Define `REFERENCE_EVENT_SCHEMA = "crux.dtx-reference-event/v1"` and
`validate_schema_golden(schema, content)` in `reference_timing.py`. Register
`tests/benchmark/schema_goldens/crux.dtx-reference-event-v1.jsonl` in the central
schema-golden manifest.

The golden must be canonical JSONL and include all stable event identity/timing fields.

- [ ] **Step 6: Validate and commit Task 4**

```bash
uv run pytest \
  tests/benchmark/test_reference_timing.py \
  tests/benchmark/test_corpus_manifest.py \
  tests/benchmark/test_schema_goldens.py -q
uv run ruff check \
  src/benchmark/reference_timing.py src/benchmark/corpus_manifest.py \
  tests/benchmark/test_reference_timing.py tests/benchmark/test_corpus_manifest.py
uv run ruff format --check \
  src/benchmark/reference_timing.py src/benchmark/corpus_manifest.py \
  tests/benchmark/test_reference_timing.py tests/benchmark/test_corpus_manifest.py
git add \
  src/benchmark/reference_timing.py src/benchmark/corpus_manifest.py \
  tests/benchmark/test_reference_timing.py tests/benchmark/test_corpus_manifest.py \
  tests/benchmark/schema_goldens/crux.dtx-reference-event-v1.jsonl \
  tests/benchmark/schema_goldens/manifest.json
git commit -m "feat: publish native DTX reference events"
```

---

### Task 5: Orchestrate the derived timing manifest

**Files:**
- Create: `src/benchmark/reference_timing_manifest.py`
- Create: `tests/benchmark/test_reference_timing_manifest.py`
- Create: `tests/benchmark/test_reference_timing_acceptance.py`
- Create: `tests/benchmark/schema_goldens/crux.reference-timing-manifest-v1.jsonl`
- Modify: `tests/benchmark/schema_goldens/manifest.json`

**Interfaces:**

```python
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
    events_published: int


def build_reference_timing_manifest(
    request: ReferenceTimingRequest,
    *,
    environ: Mapping[str, str] | None = None,
    dependency_check: Callable[[], None] = ensure_r2_dependency,
    store_factory: StoreFactory = create_boto3_store,
) -> ReferenceTimingOutcome: ...
```

- [ ] **Step 1: Write canonical HPA-322 source-loading tests**

Read exact bytes once. For every non-empty line use `strict_json_loads(...,
require_canonical=True)` and `reference_chart_row_view_from_row`.

Reject:

- empty/non-canonical JSONL;
- wrong schema;
- duplicate simfile IDs;
- mixed HPA-322 derived `corpus_version`;
- mixed embedded source endpoint/bucket/cache/discovery identity;
- a file whose rows do not reproduce the same bytes and derived corpus version through
  `render_manifest`.

Do not call `manifest_row_view_from_row` directly on HPA-322 rows.

- [ ] **Step 2: Test lineage preservation and field naming**

Preserve HPA-322 `source_manifest_sha256` and `source_corpus_version` unchanged. Remove
the HPA-322 top-level `corpus_version` before rendering the derived row and add:

```text
source_reference_chart_manifest_sha256
source_reference_chart_version
```

where the first is SHA-256 of the exact HPA-322 input bytes and the second is the common
HPA-322 derived `corpus_version`.

Assert no upstream field is overwritten.

- [ ] **Step 3: Write first-pass tests**

For selected rows:

1. read selected chart bytes with `read_verified_cache_body`, using the selected
   `RemoteObject` from `ReferenceChartRowView` and the row's selected chart hash;
2. parse via `parse_dtx_bytes`;
3. build `DtxTimingMap`;
4. resolve BGM/audio identity;
5. call `resolve_verified_cache_body` once for selected audio;
6. retain verified paths and queue only unavailable audio for targeted fill.

Upstream quarantined rows must receive `upstream_chart_selection_unavailable` and do no
chart parsing or R2 work.

Map chart body/parser failures to `selected_chart_cache_invalid` /
`selected_chart_parse_failed`; map timing construction errors to `timing_map_invalid`.

- [ ] **Step 4: Write targeted-fill tests**

Prove:

- complete cache never calls dependency/store factories;
- missing audio creates/validates R2 config only when needed;
- source bucket/endpoint must match the embedded HPA-322 source identity;
- only exact selected keys are passed to `sync_explicit_cache_keys`;
- returned inventories merge by simfile ID;
- only rows whose inventory changed are verified after the merge;
- unrelated object records are preserved.

- [ ] **Step 5: Write event and derived-row tests**

Ready rows publish `events/<sha256>.jsonl` through `publish_immutable_content`. Derived
rows preserve every HPA-322 field and add the timing fields from the design.

Require:

```text
ready_count + quarantined_count = input row count
events_published = ready_count
```

All ready -> `complete`, exit `0`; any row quarantine -> `partial`, exit `1`; fatal
input/config/publication -> `failed`, exit `2` with `manifest is None`.

- [ ] **Step 6: Implement immutable timing-manifest publication**

Use `render_manifest`, `publish_manifest`, and `publish_latest_manifest`. Do not add a
separate report artifact or `report_path` field.

- [ ] **Step 7: Add the timing-manifest schema golden**

Define `REFERENCE_TIMING_MANIFEST_SCHEMA = "crux.reference-timing-manifest/v1"` and
`validate_schema_golden`. Add a canonical golden with one ready and one quarantined row,
and register it in `tests/benchmark/schema_goldens/manifest.json`.

The validator must verify the exact key set, status-dependent nullability, reason codes,
embedded HPA-322 reference row consistency, and the derived corpus version.

- [ ] **Step 8: Build the offline acceptance fixture**

Cover:

- selected chart read through the merged HPA-322 cache contract;
- complete audio cache requiring no R2;
- one exact-key audio fill using a fake store;
- one upstream HPA-322 quarantine;
- one BGM/timing quarantine;
- deterministic second-run event and manifest identities.

- [ ] **Step 9: Validate and commit Task 5**

```bash
uv run pytest \
  tests/benchmark/test_reference_chart_manifest.py \
  tests/benchmark/test_reference_timing.py \
  tests/benchmark/test_reference_timing_manifest.py \
  tests/benchmark/test_reference_timing_acceptance.py \
  tests/benchmark/test_schema_goldens.py -q
uv run ruff check \
  src/benchmark/reference_timing_manifest.py \
  tests/benchmark/test_reference_timing_manifest.py \
  tests/benchmark/test_reference_timing_acceptance.py
uv run ruff format --check \
  src/benchmark/reference_timing_manifest.py \
  tests/benchmark/test_reference_timing_manifest.py \
  tests/benchmark/test_reference_timing_acceptance.py
git add \
  src/benchmark/reference_timing_manifest.py \
  tests/benchmark/test_reference_timing_manifest.py \
  tests/benchmark/test_reference_timing_acceptance.py \
  tests/benchmark/schema_goldens/crux.reference-timing-manifest-v1.jsonl \
  tests/benchmark/schema_goldens/manifest.json
git commit -m "feat: publish audio-relative reference manifest"
```

---

### Task 6: Wire CLI and final verification

**Files:**
- Modify: `src/cli/benchmark.py`
- Modify: `tests/test_cli_benchmark.py`

- [ ] **Step 1: Write CLI contract tests**

Require `--manifest` and `--cache-dir`; default only `--output-dir`. Cover complete `0`,
partial `1`, fatal `2`, one sorted JSON summary, and no R2 import/store construction on a
complete cache.

Expected summary keys:

```text
corpus_version
events_published
exit_code
manifest_path
quarantined
ready
status
```

- [ ] **Step 2: Implement lazy CLI wiring**

Add `build-reference-timing`, construct `ReferenceTimingRequest`, emit one sorted JSON
summary, and exit through `click.Context.exit` when non-zero.

- [ ] **Step 3: Run CI-equivalent verification**

```bash
uv run pytest -q
uv run ruff check .
uv run ruff format --check src tests
uv run pylint --errors-only --disable=E1120,E0401 src
```

- [ ] **Step 4: Verify the planning assumptions against the implementation diff**

Confirm there is no:

- direct `manifest_row_view_from_row(hpa322_row)` call;
- new path/casefold resolver;
- second cache verifier;
- widening of `is_selected`;
- second verification of already-cached audio;
- overwrite of HPA-322 `source_manifest_sha256` or `source_corpus_version`;
- `report_path` field without a report artifact.

- [ ] **Step 5: Commit Task 6**

```bash
git add src/cli/benchmark.py tests/test_cli_benchmark.py
git commit -m "feat: expose reference timing build"
```

---

## Risk Verification Matrix

| Risk | Required proof |
|---|---|
| HPA-322 derived row fed to HPA-321 parser | All production loads use `reference_chart_row_view_from_row` |
| Immediate lineage overwrites HPA-321 lineage | Dedicated `source_reference_chart_*` fields + preservation test |
| Incorrect DTX channel `02` semantics | Primary DTXMania source + sticky/replacement fixtures |
| BGM policy quarantines too much corpus | Task 0 distribution + sampled multi-group review |
| HPA-323 forks object-key behavior | All lookups use `resolve_inventory_object_key` |
| Timing rename breaks legacy consumers | Full repository suite immediately after Task 1 |
| Existing audio is hashed twice | Already-verified rows bypass post-fill verification |
| Targeted fill mutates unrelated objects | Before/after inventory equality tests |
| Complete cache still requires R2 | Dependency/store factories are not called |
| Row failure aborts corpus | Valid sibling still publishes with exit `1` |
| Stable artifact contract drifts | Event and timing-manifest schema goldens |

## Final Review Checklist

- [ ] Task 0 measured the real selected corpus and froze BGM policy.
- [ ] HPA-322 rows use `ReferenceChartRowView`, not the HPA-321-only parser directly.
- [ ] Selected chart bytes use `read_verified_cache_body` + `parse_dtx_bytes`.
- [ ] No private object-key normalization/matching logic exists in HPA-323.
- [ ] Channel `01` is typed and excluded from native playable events.
- [ ] Channel `02` remains sticky until superseded.
- [ ] Chart-time APIs are explicitly named and return `BenchmarkEvent`.
- [ ] Complete-cache audio is verified once.
- [ ] Only exact selected audio keys are downloaded.
- [ ] HPA-322 lineage fields survive unchanged.
- [ ] Native event and timing-manifest schemas have registered goldens.
- [ ] Ready plus quarantined equals input count.
- [ ] Full tests, Ruff, Ruff format, and enabled Pylint pass.
