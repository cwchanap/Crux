# HPA-323 Audio-Relative DTX Reference Timing Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Consume the merged HPA-322 reference-chart manifest, derive trustworthy audio-relative native DTX events, and publish immutable artifacts for HPA-324.

**Architecture:** Measure real BGM usage first, extend the parser/timing engine with evidence-backed DTX semantics and explicit clock names, reuse all HPA-322 row/cache/key contracts, cache only exact selected audio misses, and publish content-addressed native event JSONL plus a derived manifest.

**Tech Stack:** Python 3.12, dataclasses, pathlib, JSONL, Click, soundfile, existing R2/boto3 adapter, pytest, Ruff, Pylint.

## Global Constraints

- Start after HPA-322 merges.
- Reuse `parse_manifest_timestamp`, `manifest_row_view_from_row`,
  `inventory_from_manifest_row`, `resolve_verified_cache_body`, and
  `resolve_inventory_object_key` from HPA-322.
- Do not create `manifest_inventory.py`, a parallel source-object type, a private
  object-key matcher, or another cache verifier.
- DTX channel `02` is sticky until superseded; this is verified by DTXManiaXG source.
- Channel `01` is typed BGM control data.
- Measure real BGM group and root-fallback distribution before freezing ambiguity policy.
- Rename chart-time APIs; no ambiguous `dtx_events_to_timed_events` alias remains.
- Require `--cache-dir` explicitly.
- Verify each audio body once unless its inventory changes during targeted fill.
- Preserve HPA-321's default `is_selected` policy.
- Run the full test suite in the timing-semantics task.

## Primary Timing Evidence

DTXManiaXG Ver.K commit `2e7839d93c00ef528407bebdcf829dafb8c8c804`
keeps `dbBarLength` active after channel `02`. Reset to `1.0` is conditional on BMS/BME,
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

### Modify

- `src/benchmark/models.py`
- `src/benchmark/dtx_parser.py`
- `src/benchmark/timing.py`
- `src/benchmark/runner.py`
- `src/benchmark/render_audio.py`
- `src/benchmark/corpus_cache.py`
- `src/cli/benchmark.py`
- corresponding tests

---

### Task 0: Measure BGM group and fallback distribution

**Files:**
- No production files.
- Output: `artifacts/benchmark/reference-timing-analysis/bgm-layout.json`

**Consumes:** merged HPA-322 manifest/cache, `manifest_row_view_from_row`,
`resolve_verified_cache_body`, `resolve_inventory_object_key`, and the DTX parser.

- [ ] **Step 1: Run a read-only analysis script**

For every HPA-322 selected row:

1. load one `ManifestRowView`;
2. verify and parse the selected chart;
3. extract non-zero channel `01` tokens through a temporary local parser helper if
   typed BGM events have not landed;
4. resolve every `#WAVxx` value by calling `resolve_inventory_object_key` relative to
   the selected-chart directory;
5. when that result is `missing`, make a second call with the simfile-root directory
   and record whether it resolves;
6. group by `(remote.key, measure, position)`.

Do not implement separator normalization, `..` handling, prefix containment, exact
matching, or casefold matching in the analysis script.

Write sorted JSON:

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

Cap examples at 25 rows and retain simfile ID, chart key, object keys, note IDs,
measures, positions, and source order.

- [ ] **Step 2: Review the report before implementation**

Decision:

- exceptional multi-group rows -> retain `ambiguous_bgm_start` quarantine;
- common multi-group rows -> inspect representative charts and amend HPA-323;
- zero root fallback rows -> remove the second root-level resolver call;
- never choose the earliest BGM group without authored-chart evidence.

- [ ] **Step 3: Record the decision**

Add report counts and the frozen policy to HPA-323 Linear and the implementation PR.
Do not commit corpus paths or source bodies.

---

### Task 1: Typed BGM events, sticky timing, and explicit clocks

**Files:**
- Modify: `src/benchmark/models.py`
- Modify: `src/benchmark/dtx_parser.py`
- Modify: `src/benchmark/timing.py`
- Modify: `src/benchmark/runner.py`
- Modify: `src/benchmark/render_audio.py`
- Modify: `tests/benchmark/test_dtx_parser.py`
- Modify: `tests/benchmark/test_timing.py`
- Modify: every test importing `dtx_events_to_timed_events`

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

def build_dtx_timing_map(chart: ParsedDtxChart) -> DtxTimingMap

def dtx_events_to_chart_time_events(chart: ParsedDtxChart) -> list[TimedEvent]
```

- [ ] **Step 1: Write typed channel `01` tests**

```python
def test_channel_01_is_typed_bgm_not_native_event() -> None:
    chart = parse_dtx_text(
        "#WAV01: bgm.ogg\n#00101: 0100\n#00111: 0001\n",
        "song",
    )

    assert [(event.measure, event.position, event.note_id) for event in chart.bgm_events] == [
        (1, 0.0, "01")
    ]
    assert [(event.lane_id, event.note_id) for event in chart.events] == [("11", "01")]
```

Add source-order coverage for multiple non-zero pattern tokens.

- [ ] **Step 2: Verify parser tests fail**

```bash
uv run pytest tests/benchmark/test_dtx_parser.py -k "channel_01 or source_order" -q
```

- [ ] **Step 3: Implement typed pattern parsing**

Add `source_order` to `DtxEvent`, create `DtxBgmEvent`, route channel `01` to
`bgm_events`, and keep BPM ordering separate.

- [ ] **Step 4: Replace the incorrect measure-length fixture**

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

    assert [event.time_sec for event in dtx_events_to_chart_time_events(chart)] == [
        2.0,
        3.0,
        4.0,
        5.0,
        7.0,
    ]
```

Also cover `0.5 -> 1.5 -> 1.0`, BPM change inside a shortened measure, and BGM/native
parity through one timing map.

- [ ] **Step 5: Implement `DtxTimingMap`**

Carry active measure length forward and use the resolved value for measure starts and
in-measure BPM/BGM/native positions.

- [ ] **Step 6: Rename the chart-time API**

Rename `dtx_events_to_timed_events` to `dtx_events_to_chart_time_events` with no alias.
Update all callers and add this comment to legacy scoring:

```python
# Legacy folder/MIDI scoring uses chart time. HPA-325 consumes HPA-323
# audio-time reference artifacts and must not use this path.
```

- [ ] **Step 7: Run the full repository checks**

```bash
uv run pytest -q
uv run ruff check .
uv run ruff format --check src tests
```

Investigate changed goldens or score expectations; do not update blindly.

- [ ] **Step 8: Commit Task 1**

```bash
git add src/benchmark/models.py src/benchmark/dtx_parser.py src/benchmark/timing.py \
  src/benchmark/runner.py src/benchmark/render_audio.py tests
git commit -m "fix: distinguish DTX chart and BGM timing"
```

---

### Task 2: Resolve BGM identity through the HPA-322 key contract

**Files:**
- Create: `src/benchmark/reference_timing.py`
- Create: `tests/benchmark/test_reference_timing.py`

**Interfaces:**

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

def resolve_bgm_reference(
    chart: ParsedDtxChart,
    timing_map: DtxTimingMap,
    *,
    selected_chart_key: str,
    row: ManifestRowView,
    allow_root_fallback: bool,
    quarantine_multiple_groups: bool,
) -> BgmResolution
```

- [ ] **Step 1: Write shared-resolver mapping tests**

Monkeypatch or inject `resolve_inventory_object_key` results and assert:

- `exact` and `casefold` return the supplied `RemoteObject`;
- `invalid_path` -> `unsafe_bgm_audio_path`;
- `missing` -> optional second root call, then `source_audio_missing`;
- `ambiguous` -> `source_audio_key_ambiguous`;
- root fallback warning is emitted only after relative `missing`.

Do not duplicate HPA-322 normalization/casefold test matrices here.

- [ ] **Step 2: Write BGM grouping tests**

Group by `(remote.key, measure, position)`, never float time. Cover duplicate source
tokens at one discrete identity and multiple distinct groups under the Task 0 policy.

- [ ] **Step 3: Verify tests fail**

```bash
uv run pytest tests/benchmark/test_reference_timing.py -k bgm -q
```

- [ ] **Step 4: Implement policy-only resolution**

Resolve note IDs through `wav_table`, call the shared resolver, optionally make the
second root call, preserve counts/warnings, and compute chart time only for the selected
event. Do not define a private path helper or casefold index.

- [ ] **Step 5: Validate and commit Task 2**

```bash
uv run pytest tests/benchmark/test_reference_timing.py -k bgm -q
uv run ruff check src/benchmark/reference_timing.py tests/benchmark/test_reference_timing.py
uv run ruff format --check src/benchmark/reference_timing.py tests/benchmark/test_reference_timing.py
git add src/benchmark/reference_timing.py tests/benchmark/test_reference_timing.py
git commit -m "feat: resolve DTX BGM references"
```

---

### Task 3: Fill only exact selected audio cache keys

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
) -> CacheSyncResult
```

- [ ] **Step 1: Write selector-preservation tests**

Assert `sync_cache` still selects `set.def`, `.dtx`, and `.txt` exactly as before.

- [ ] **Step 2: Write explicit-key tests**

Cover one selected audio key, unrelated objects unchanged, missing selected key,
conditional source changes, cache hit, download, and partial failure.

- [ ] **Step 3: Verify failure**

```bash
uv run pytest tests/benchmark/test_corpus_cache.py -k explicit_cache -q
```

- [ ] **Step 4: Extract one internal selectable-cache path**

Make `sync_cache` delegate with `is_selected`; make `sync_explicit_cache_keys` delegate
with exact membership. Do not duplicate download/install/index logic.

- [ ] **Step 5: Validate and commit Task 3**

```bash
uv run pytest tests/benchmark/test_corpus_cache.py -q
uv run ruff check src/benchmark/corpus_cache.py tests/benchmark/test_corpus_cache.py
uv run ruff format --check src/benchmark/corpus_cache.py tests/benchmark/test_corpus_cache.py
git add src/benchmark/corpus_cache.py tests/benchmark/test_corpus_cache.py
git commit -m "feat: cache exact benchmark audio keys"
```

---

### Task 4: Build bounded audio-relative native events

**Files:**
- Modify: `src/benchmark/reference_timing.py`
- Modify: `tests/benchmark/test_reference_timing.py`

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

def inspect_source_audio(path: Path) -> SourceAudioInfo

def build_audio_relative_events(
    chart: ParsedDtxChart,
    timing_map: DtxTimingMap,
    *,
    bgm_chart_time_sec: float,
    audio: SourceAudioInfo,
) -> AudioRelativeReference
```

- [ ] **Step 1: Write metadata tests**

Use small generated WAV fixtures. Cover valid metadata, undecodable body, and zero
frames without decoding the full waveform.

- [ ] **Step 2: Write bounds tests**

Cover BGM shift, exact zero/duration, one-frame clamp, larger negative/late exclusions,
non-finite time, zero retained events, and source-order preservation.

- [ ] **Step 3: Verify failure**

```bash
uv run pytest tests/benchmark/test_reference_timing.py -k "audio_info or audio_relative" -q
```

- [ ] **Step 4: Implement metadata and bounds**

Use `soundfile.info`. One frame is the tolerance. Clamp only within tolerance; otherwise
exclude and count. Raise typed row failures for non-finite time or zero retained events.

- [ ] **Step 5: Validate and commit Task 4**

```bash
uv run pytest tests/benchmark/test_reference_timing.py -q
uv run ruff check src/benchmark/reference_timing.py tests/benchmark/test_reference_timing.py
uv run ruff format --check src/benchmark/reference_timing.py tests/benchmark/test_reference_timing.py
git add src/benchmark/reference_timing.py tests/benchmark/test_reference_timing.py
git commit -m "feat: build audio-relative DTX events"
```

---

### Task 5: Orchestrate one-verification reference timing publication

**Files:**
- Create: `src/benchmark/reference_timing_manifest.py`
- Create: `tests/benchmark/test_reference_timing_manifest.py`
- Create: `tests/benchmark/test_reference_timing_acceptance.py`

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
    report_path: Path | None
    ready_count: int
    quarantined_count: int

def build_reference_timing_manifest(
    request: ReferenceTimingRequest,
) -> ReferenceTimingOutcome
```

- [ ] **Step 1: Write source-loading tests**

Use `manifest_row_view_from_row` for every HPA-322 row. Reject malformed schema,
duplicate IDs, mixed source identities, and invalid selected/quarantined nullability.
Do not parse source identity separately.

- [ ] **Step 2: Write first-pass orchestration tests**

Assert:

- selected chart is verified once and parsed;
- upstream quarantine is preserved;
- BGM resolution receives the shared row view;
- verified audio path is retained;
- only unavailable audio enters the fill queue;
- complete cache never creates an R2 store.

- [ ] **Step 3: Write post-fill tests**

Only rows whose inventory changed are verified after fill. Already verified rows are not
hashed again. Unrelated object records remain byte-for-byte equal.

- [ ] **Step 4: Write event artifact tests**

Publish canonical native event JSONL at `events/<sha256>.jsonl`. Reusing identical bytes
must verify the existing immutable artifact; changed bytes create a new identity.

- [ ] **Step 5: Write derived-row tests**

Add the exact fields from the design, preserve upstream selection fields, reconcile:

```text
ready + quarantined = input rows
events_published = ready
```

Map resolver outcomes and row failures to the frozen reason-code set.

- [ ] **Step 6: Implement orchestration**

Perform first pass, targeted fill, changed-row verification, metadata/event publication,
and manifest publication through existing canonical helpers. One row exception must not
abort valid siblings.

- [ ] **Step 7: Build the offline acceptance fixture**

Cover complete cache, one targeted audio fill with a fake store, one upstream
quarantine, one BGM-resolution quarantine, and deterministic second-run identities.

- [ ] **Step 8: Validate and commit Task 5**

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
git commit -m "feat: publish audio-relative references"
```

---

### Task 6: Wire CLI and final verification

**Files:**
- Modify: `src/cli/benchmark.py`
- Modify: `tests/test_cli_benchmark.py`

- [ ] **Step 1: Write CLI contract tests**

Require `--manifest` and `--cache-dir`; default only `--output-dir`. Cover complete `0`,
partial `1`, fatal `2`, one sorted JSON summary, and no R2 import on a complete cache.

- [ ] **Step 2: Implement lazy CLI wiring**

Add `build-reference-timing`, construct `ReferenceTimingRequest`, emit the summary, and
exit through `click.Context.exit`.

- [ ] **Step 3: Run full validation**

```bash
uv run pytest -q
uv run ruff check .
uv run ruff format --check src tests
```

Run the exact enabled Pylint command from `.github/workflows/ci.yml`.

- [ ] **Step 4: Commit Task 6**

```bash
git add src/cli/benchmark.py tests/test_cli_benchmark.py
git commit -m "feat: expose reference timing build"
```

---

## Risk Verification Matrix

| Risk | Required proof |
|---|---|
| Incorrect DTX channel `02` semantics | Primary DTXMania source plus sticky/replacement fixtures |
| BGM policy quarantines an unacceptable share | Corpus distribution report and sampled multi-group review |
| HPA-323 forks object-key semantics | No private normalization/matcher; tests inject shared resolver outcomes |
| Timing change breaks legacy consumers | Explicit rename and full repository suite |
| Existing cache is unnecessarily rehashed | Complete rows verified once and excluded from post-fill verification |
| Targeted fill mutates unrelated objects | Before/after equality tests |
| Complete cache still requires R2 | Store factory is not called |
| Row failure aborts corpus | Valid sibling publishes with exit `1` |
| Legacy scoring becomes authoritative | Chart-time comment; HPA-325 consumes audio-time artifacts |

## Final Review Checklist

- [ ] Task 0 uses the HPA-322 key resolver.
- [ ] HPA-323 imports one validated `ManifestRowView` per source row.
- [ ] No private object-key normalizer, containment checker, exact matcher, or casefold index exists.
- [ ] Channel `01` is typed and excluded from native playable events.
- [ ] Channel `02` remains sticky until superseded.
- [ ] Chart-time APIs are explicitly named.
- [ ] Complete-cache rows are verified once.
- [ ] Only exact selected audio keys are downloaded.
- [ ] Native event artifacts are content-addressed and deterministic.
- [ ] Ready plus quarantined equals input count.
- [ ] Full tests, Ruff, formatter, and enabled Pylint pass.
