# HPA-323 Audio-Relative DTX Reference Timing Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Consume the merged HPA-322 reference-chart manifest, derive trustworthy audio-relative native DTX events, and publish immutable artifacts for HPA-324.

**Architecture:** Measure real BGM usage first, extend the parser/timing engine with evidence-backed DTX semantics and explicit clock names, reuse HPA-322 manifest/cache contracts, cache only exact selected audio misses, and publish content-addressed native event JSONL plus a derived manifest.

**Tech Stack:** Python 3.12, dataclasses, pathlib, JSONL, Click, soundfile, existing R2/boto3 adapter, pytest, Ruff.

## Global Constraints

- Start after HPA-322 merges.
- Reuse `parse_manifest_timestamp`, `inventory_from_manifest_row`, and
  `resolve_verified_cache_body` from HPA-322.
- Do not create `manifest_inventory.py`, a parallel source-object type, or another
  cache verifier.
- DTX channel `02` is sticky until superseded; this is verified by DTXManiaXG source.
- Channel `01` is typed BGM control data.
- Measure real BGM group distribution before freezing ambiguity policy.
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
- `src/benchmark/corpus_manifest.py`
- `src/cli/benchmark.py`
- corresponding tests

---

### Task 0: Measure BGM group and fallback distribution

**Files:**
- No production files.
- Output: `artifacts/benchmark/reference-timing-analysis/bgm-layout.json`

**Consumes:** merged HPA-322 manifest, cache, `inventory_from_manifest_row`,
`resolve_verified_cache_body`, and the existing DTX parser.

- [ ] **Step 1: Run a read-only analysis script**

Use an inline `uv run python` script or a temporary uncommitted notebook. For every
HPA-322 selected row:

1. reconstruct the inventory;
2. verify and parse the selected chart;
3. extract non-zero channel `01` tokens through a temporary local parser helper if Task
   1 has not landed yet;
4. resolve `#WAVxx` values against inventory keys using selected-chart-relative exact
   matching, then record whether casefold or root fallback would be needed;
5. group by `(object_key, measure, position)`.

Write sorted JSON with:

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

Cap `multi_group_examples` at 25 rows, preserving simfile ID, selected chart key,
object keys, note IDs, measures, positions, and source order.

- [ ] **Step 2: Review the report before implementation**

Decision:

- exceptional multi-group rows -> keep `ambiguous_bgm_start` quarantine;
- common multi-group rows -> inspect representative DTX files and amend this design and
  HPA-323 before Task 2;
- do not automatically choose the earliest group without evidence that later groups are
  continuations/layers rather than competing starts.

Also remove casefold/root fallback from the implementation if the report proves it is
unused and unnecessary.

- [ ] **Step 3: Record the decision in HPA-323**

Add the report counts and frozen policy to Linear and the implementation PR description.
Do not commit corpus paths or source contents.

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
DtxBgmEvent(chart_id, measure, position, note_id, source_order)
ParsedDtxChart.bgm_events: list[DtxBgmEvent]
DtxTimingMap.time_sec(event)
build_dtx_timing_map(chart)
dtx_events_to_chart_time_events(chart)
```

- [ ] **Step 1: Write typed-channel tests**

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

Add monotonic non-zero pattern source-order coverage.

- [ ] **Step 2: Verify parser failure**

```bash
uv run pytest tests/benchmark/test_dtx_parser.py -k "channel_01 or source_order" -q
```

- [ ] **Step 3: Implement typed pattern parsing**

Add `source_order` to `DtxEvent`, frozen `DtxBgmEvent`, and `bgm_events` to the parsed
chart. Route channel `01` into BGM controls. Keep BPM source ordering separate.

- [ ] **Step 4: Replace the incorrect DTX timing fixture**

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

    assert [
        event.time_sec for event in dtx_events_to_chart_time_events(chart)
    ] == [2.0, 3.0, 4.0, 5.0, 7.0]
```

Add:

- `0.5 -> 1.5 -> 1.0` replacement;
- BPM change inside a sticky shortened measure with expected `4.5` seconds;
- BGM/native parity through one timing map;
- BMS/BME semantics are not implemented or asserted by this DTX parser.

- [ ] **Step 5: Implement `DtxTimingMap`**

Carry active measure length forward. Use the resolved length for starts and in-measure
positions of BPM, BGM, and native events.

- [ ] **Step 6: Rename the chart-time function**

Rename `dtx_events_to_timed_events` to `dtx_events_to_chart_time_events` without an
alias. Update `render_audio.py`, `runner.py`, and all tests.

Add a short comment in `runner.py`:

```python
# Legacy folder/MIDI scoring uses chart time. HPA-325 consumes HPA-323
# audio-time reference artifacts and must not use this path.
```

- [ ] **Step 7: Run the full suite and CI formatting gates**

```bash
uv run pytest -q
uv run ruff check .
uv run ruff format --check src tests
```

Any changed golden/score expectation must be investigated; do not update blindly.

- [ ] **Step 8: Commit**

```bash
git add src/benchmark/models.py src/benchmark/dtx_parser.py src/benchmark/timing.py src/benchmark/runner.py src/benchmark/render_audio.py tests
git commit -m "fix: distinguish DTX chart and BGM timing"
```

---

### Task 2: Resolve BGM identity from HPA-322 inventory

**Files:**
- Create: `src/benchmark/reference_timing.py`
- Create: `tests/benchmark/test_reference_timing.py`

**Consumes:** `ParsedDtxChart`, `DtxTimingMap`, `RemoteObject`, and the policy frozen in
Task 0.

**Produces:** `BgmResolution`, `ResolvedBgm`, and
`resolve_bgm_reference(...) -> BgmResolution`.

- [ ] **Step 1: Write path-resolution tests**

Cover exact selected-chart-relative match, unique casefold match if retained by Task 0,
root fallback if retained, unknown/empty WAV IDs, absolute paths, drive prefixes,
traversal, missing objects, and ambiguous casefold matches.

- [ ] **Step 2: Write group-policy tests from Task 0's decision**

Always group by `(remote.key, event.measure, event.position)`, never float time.
Test duplicate source tokens at one discrete identity and multiple distinct groups.

- [ ] **Step 3: Verify failure**

```bash
uv run pytest tests/benchmark/test_reference_timing.py -k bgm -q
```

- [ ] **Step 4: Implement pure resolution**

Resolve every BGM token through `#WAVxx` and inventory objects. Compute chart time only
for the selected event. Preserve raw event/group counts and warnings. Apply only the
policy frozen in Task 0.

- [ ] **Step 5: Validate and commit**

```bash
uv run pytest tests/benchmark/test_reference_timing.py -k bgm -q
uv run ruff check src/benchmark/reference_timing.py tests/benchmark/test_reference_timing.py
uv run ruff format --check src/benchmark/reference_timing.py tests/benchmark/test_reference_timing.py
git add src/benchmark/reference_timing.py tests/benchmark/test_reference_timing.py
git commit -m "feat: resolve DTX source audio identity"
```

---

### Task 3: Cache only exact selected source-audio keys

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

- [ ] **Step 1: Write exact-key tests**

Use `real.dtx`, `bgm.ogg`, and `preview.ogg`; select only `bgm.ogg`. Assert only it is
opened/verified and `is_selected("bgm.ogg")` remains false. Cover empty set, empty key,
absent key, cache hit, and failed download.

- [ ] **Step 2: Extract a selector-driven worker**

Keep public `sync_cache` unchanged. Move the existing body behind a private selector
without changing locking, validation, installation, progress, or inventory rebuilding.

- [ ] **Step 3: Add the explicit wrapper**

Require every selected key to be non-empty and present in the supplied inventories.

- [ ] **Step 4: Validate and commit**

```bash
uv run pytest tests/benchmark/test_corpus_cache.py -q
uv run ruff check src/benchmark/corpus_cache.py tests/benchmark/test_corpus_cache.py
uv run ruff format --check src/benchmark/corpus_cache.py tests/benchmark/test_corpus_cache.py
git add src/benchmark/corpus_cache.py tests/benchmark/test_corpus_cache.py
git commit -m "feat: cache explicit benchmark objects"
```

---

### Task 4: Audio metadata, bounds, and immutable native events

**Files:**
- Modify: `src/benchmark/reference_timing.py`
- Modify: `tests/benchmark/test_reference_timing.py`
- Modify: `src/benchmark/corpus_manifest.py`
- Modify: `tests/benchmark/test_corpus_manifest.py`

**Interfaces:**

```text
SourceAudioMetadata
inspect_source_audio(path)
build_audio_relative_events(...)
render_reference_event_jsonl(events)
publish_immutable_content(path, content, expected_sha256)
```

- [ ] **Step 1: Write metadata tests**

Write a one-second 8 kHz WAV and assert duration, sample rate, channels, and frames.
Cover undecodable and zero-frame inputs.

- [ ] **Step 2: Write bounds tests**

At 1 kHz, assert near-boundary values clamp within one frame while larger pre/post values
are excluded. Cover non-finite time and no retained events.

- [ ] **Step 3: Implement audio-time event construction**

Use:

```text
chart_time = timing.time_sec(event)
audio_time = chart_time - selected_bgm.chart_time_sec
```

Preserve native identities and both clocks. Do not map or deduplicate.

- [ ] **Step 4: Render deterministic JSONL**

Use `canonical_json_line` for every event. Sort deterministically and test repeated byte
identity.

- [ ] **Step 5: Expose the existing immutable publisher**

Add only a thin public wrapper over `_publish_immutable`; do not duplicate it.

- [ ] **Step 6: Validate and commit**

```bash
uv run pytest tests/benchmark/test_reference_timing.py tests/benchmark/test_corpus_manifest.py -q
uv run ruff check src/benchmark/reference_timing.py src/benchmark/corpus_manifest.py tests/benchmark/test_reference_timing.py tests/benchmark/test_corpus_manifest.py
uv run ruff format --check src/benchmark/reference_timing.py src/benchmark/corpus_manifest.py tests/benchmark/test_reference_timing.py tests/benchmark/test_corpus_manifest.py
git add src/benchmark/reference_timing.py src/benchmark/corpus_manifest.py tests/benchmark/test_reference_timing.py tests/benchmark/test_corpus_manifest.py
git commit -m "feat: publish bounded native reference events"
```

---

### Task 5: Orchestrate one-verification cache flow and derived publication

**Files:**
- Create: `src/benchmark/reference_timing_manifest.py`
- Create: `tests/benchmark/test_reference_timing_manifest.py`

**Interfaces:**

```python
@dataclass(frozen=True)
class TimingRequest:
    manifest_path: Path
    cache_dir: Path
    output_dir: Path

def build_reference_timing_manifest(request: TimingRequest, ...) -> TimingOutcome
```

- [ ] **Step 1: Write input and row-failure tests**

Cover malformed input, duplicate IDs, mixed source identities, upstream quarantine,
`selected_chart_parse_failed`, and `timing_map_invalid`. A valid sibling must still
publish with partial exit `1`.

- [ ] **Step 2: Write one-verification tests**

Inject/spy on `resolve_verified_cache_body` and prove:

- already verified audio is called once and its path is stored;
- cache-miss audio is called once before fill and once after its inventory changes;
- already verified rows are not called again after another row fills the cache;
- complete cache does not call dependency/store factories.

- [ ] **Step 3: Implement exact input loading**

Read bytes once, calculate source SHA, reconstruct inventories through HPA-322's reader,
and preserve one corpus/bucket/endpoint identity.

- [ ] **Step 4: Implement first pass**

For each selected row:

1. locate/verify selected chart;
2. parse chart and map exceptions to row reason codes;
3. build timing map separately;
4. resolve BGM/audio;
5. call the shared verifier once;
6. store verified path or queue the row for targeted fill.

- [ ] **Step 5: Fill misses only**

Resolve optional R2 dependencies/config only when the queue is non-empty. Pass only
complete miss inventories and exact keys to `sync_explicit_cache_keys`, then merge by
`simfile_id`.

- [ ] **Step 6: Re-verify changed rows only**

Call the shared verifier only for queued rows using merged inventories. Rows with stored
paths skip this step.

- [ ] **Step 7: Publish events and derived rows**

Inspect metadata, build audio-time events, publish immutable event files, update only the
selected audio object fields, and publish through existing manifest helpers.

Require:

```text
ready + quarantined = input rows
events_published = ready
```

- [ ] **Step 8: Validate and commit**

```bash
uv run pytest tests/benchmark/test_reference_timing_manifest.py -q
uv run ruff check src/benchmark/reference_timing_manifest.py tests/benchmark/test_reference_timing_manifest.py
uv run ruff format --check src/benchmark/reference_timing_manifest.py tests/benchmark/test_reference_timing_manifest.py
git add src/benchmark/reference_timing_manifest.py tests/benchmark/test_reference_timing_manifest.py
git commit -m "feat: publish audio relative timing manifest"
```

---

### Task 6: Required-path CLI and acceptance fixture

**Files:**
- Modify: `src/cli/benchmark.py`
- Modify: `tests/test_cli_benchmark.py`
- Create: `tests/benchmark/test_reference_timing_acceptance.py`

- [ ] **Step 1: Write CLI tests**

Assert help lists required `--manifest` and `--cache-dir`, optional/default
`--output-dir`, request wiring, summaries, and exit `0`, `1`, `2`.

- [ ] **Step 2: Add `build-reference-timing`**

```python
@benchmark.command("build-reference-timing")
@click.option("--manifest", type=click.Path(path_type=Path, dir_okay=False), required=True)
@click.option("--cache-dir", type=click.Path(path_type=Path, file_okay=False), required=True)
@click.option(
    "--output-dir",
    type=click.Path(path_type=Path, file_okay=False),
    default=Path("artifacts/benchmark/reference-timing"),
    show_default=True,
)
```

Map fatal `ValueError`, `ManifestPublicationError`, and `R2StoreError` to exit `2`.

- [ ] **Step 3: Build acceptance coverage**

Use the policy frozen in Task 0. Include sticky measure timing, nested relative audio,
one targeted cache miss, row-local parse/timing failure, and pre/post-audio counts.

- [ ] **Step 4: Run full validation**

```bash
uv run pytest -q
uv run ruff check .
uv run ruff format --check src tests
```

Run the current CI Pylint command when enabled.

- [ ] **Step 5: Verify determinism**

Run the acceptance command twice and assert no new content-addressed manifest/event files
and identical hashes.

- [ ] **Step 6: Commit**

```bash
git add src/cli/benchmark.py tests/test_cli_benchmark.py tests/benchmark/test_reference_timing_acceptance.py
git commit -m "feat: expose reference timing pipeline"
```

---

## Risk Verification Matrix

| Risk | Proof |
|---|---|
| Wrong DTX channel `02` semantics | Primary source plus sticky/replacement tests |
| BGM policy destroys corpus yield | Task 0 distribution report before policy implementation |
| Timing rename/regression breaks consumers | Task 1 full `pytest -q` |
| Complete cache hashes audio twice | Task 5 verifier-call assertions |
| Targeted fill mutates unrelated objects | Task 5 before/after object comparison |
| Complete cache still resolves R2 | Dependency/store factories fail if called |
| Bad row aborts corpus | Valid sibling publishes with partial exit `1` |
| Wrong ground-truth clock is silently used | Explicit chart-time name and HPA-325 boundary comment |

## Final Review Checklist

- [ ] HPA-322 shared contracts are merged and imported directly.
- [ ] Task 0 evidence freezes BGM ambiguity/fallback policy.
- [ ] Channel `02` behavior matches DTXMania DTX semantics.
- [ ] Chart-time and audio-time functions are unmistakably named.
- [ ] Full tests pass immediately after timing changes.
- [ ] Each unchanged cached audio body is verified exactly once.
- [ ] `--cache-dir` is explicit.
- [ ] No HPA-324 taxonomy or model execution leaked into this work.
