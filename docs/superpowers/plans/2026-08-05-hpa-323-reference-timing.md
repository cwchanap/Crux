# HPA-323 Audio-Relative DTX Reference Timing Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Consume the merged HPA-322 reference-chart manifest, correct DTX timing semantics, cache the exact DTX-referenced source audio, and publish immutable audio-relative native reference events for HPA-324.

**Architecture:** Extend the parser and timing engine, load manifest records into existing HPA-321 inventory types, share one cache-body verifier with HPA-322, resolve BGM identity from typed control events, fill only exact selected audio misses, and publish content-addressed event JSONL plus a derived manifest.

**Tech Stack:** Python 3.12, dataclasses, pathlib, datetime, JSONL, hashlib, Click, soundfile, existing R2/boto3 adapter, pytest, Ruff.

## Global Constraints

- Start only after HPA-322 is merged.
- Use `RemoteObject`, `SimfileInventory`, and `SyncError`; do not add a parallel source-object type.
- HPA-322 and HPA-323 must share one verified-cache-body helper.
- Channel `02` is sticky from its own measure until superseded.
- Channel `01` is typed BGM control data and never enters native event artifacts.
- Resolve source audio from `#WAVxx`; never hard-code `bgm.ogg`.
- Prefer exact object keys, then one unique case-insensitive match.
- Group BGM starts by `(key, measure, position)`, not float time.
- Cache only exact selected audio keys; preserve HPA-321's default suffix policy.
- Raw timing uses the DTX-derived audio clock; auto-alignment remains diagnostic.
- Keep processing sequential after targeted cache fill.
- Use the repository's Ruff-based CI gates.

## File Map

### Create

- `src/benchmark/manifest_inventory.py`
- `src/benchmark/reference_timing.py`
- `src/benchmark/reference_timing_manifest.py`
- `tests/benchmark/test_manifest_inventory.py`
- `tests/benchmark/test_reference_timing.py`
- `tests/benchmark/test_reference_timing_manifest.py`
- `tests/benchmark/test_reference_timing_acceptance.py`

### Modify

- `src/benchmark/models.py`
- `src/benchmark/dtx_parser.py`
- `src/benchmark/timing.py`
- `src/benchmark/reference_chart_selection.py`
- `src/benchmark/corpus_cache.py`
- `src/benchmark/corpus_manifest.py`
- `src/cli/benchmark.py`
- corresponding existing test files

---

### Task 1: Typed BGM events and sticky timing map

**Files:**
- Modify: `src/benchmark/models.py`
- Modify: `src/benchmark/dtx_parser.py`
- Modify: `src/benchmark/timing.py`
- Modify: `tests/benchmark/test_dtx_parser.py`
- Modify: `tests/benchmark/test_timing.py`

**Interfaces:**
- `DtxEvent.source_order: int = 0`
- `DtxBgmEvent(chart_id, measure, position, note_id, source_order)`
- `ParsedDtxChart.bgm_events`
- `DtxTimingMap.time_sec(event)`
- `build_dtx_timing_map(chart)`

- [ ] **Step 1: Write failing parser tests**

```python
def test_channel_01_is_typed_bgm() -> None:
    chart = parse_dtx_text(
        "#WAV01: bgm.ogg\n#00101: 0100\n#00111: 0001\n",
        "song",
    )

    assert [(event.measure, event.position, event.note_id) for event in chart.bgm_events] == [
        (1, 0.0, "01")
    ]
    assert [(event.lane_id, event.note_id) for event in chart.events] == [("11", "01")]


def test_pattern_events_keep_source_order() -> None:
    chart = parse_dtx_text(
        "#00111: 0102\n#00101: 0304\n#00112: 0500\n",
        "song",
    )

    orders = [event.source_order for event in chart.events]
    orders.extend(event.source_order for event in chart.bgm_events)

    assert sorted(orders) == [0, 1, 2, 3, 4]
```

- [ ] **Step 2: Verify failure**

```bash
uv run pytest tests/benchmark/test_dtx_parser.py -k "channel_01 or source_order" -q
```

- [ ] **Step 3: Implement typed events**

Add `source_order` to `DtxEvent`. Add frozen `DtxBgmEvent`. Replace the generic-only pattern parser with one helper that assigns source order to each non-zero token and sends channel `01` to `bgm_events`. Keep BPM source order separate.

- [ ] **Step 4: Replace the incorrect timing fixture**

```python
def test_measure_length_persists_until_superseded() -> None:
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

    timed = dtx_events_to_timed_events(chart)

    assert [event.time_sec for event in timed] == [2.0, 3.0, 4.0, 5.0, 7.0]


def test_bpm_change_inside_sticky_measure() -> None:
    chart = parse_dtx_text(
        "#BPM: 120\n#BPM01: 60\n#00102: 0.5\n#00208: 0001\n#00311: 01\n",
        "song",
    )

    assert dtx_events_to_timed_events(chart)[0].time_sec == 4.5
```

Also add `0.5 -> 1.5 -> 1.0` replacement and BGM/native-event parity fixtures.

- [ ] **Step 5: Implement `DtxTimingMap`**

Resolve active measure length once. Use the resolved length for measure starts, BPM events, BGM events, and native events. Extend `_max_measure` with BGM events. Make the legacy wrapper delegate to the map without applying a BGM shift.

- [ ] **Step 6: Validate and commit**

```bash
uv run pytest tests/benchmark/test_dtx_parser.py tests/benchmark/test_timing.py -q
uv run ruff check src/benchmark/models.py src/benchmark/dtx_parser.py src/benchmark/timing.py tests/benchmark/test_dtx_parser.py tests/benchmark/test_timing.py
uv run ruff format --check src/benchmark/models.py src/benchmark/dtx_parser.py src/benchmark/timing.py tests/benchmark/test_dtx_parser.py tests/benchmark/test_timing.py
git add src/benchmark/models.py src/benchmark/dtx_parser.py src/benchmark/timing.py tests/benchmark/test_dtx_parser.py tests/benchmark/test_timing.py
git commit -m "fix: derive sticky DTX timing controls"
```

---

### Task 2: Shared manifest inventory, cache verification, and BGM resolution

**Files:**
- Create: `src/benchmark/manifest_inventory.py`
- Create: `tests/benchmark/test_manifest_inventory.py`
- Create: `src/benchmark/reference_timing.py`
- Create: `tests/benchmark/test_reference_timing.py`
- Modify: `src/benchmark/reference_chart_selection.py`
- Modify: `tests/benchmark/test_reference_chart_selection.py`

**Interfaces:**

```python
def parse_manifest_timestamp(value: object) -> datetime
def inventory_from_manifest_row(row: Mapping[str, object]) -> SimfileInventory
def resolve_verified_cache_body(
    cache_dir: Path,
    remote: RemoteObject,
    *,
    expected_sha256: str | None = None,
) -> Path
def resolve_bgm_reference(
    chart: ParsedDtxChart,
    timing: DtxTimingMap,
    *,
    selected_chart_key: str,
    object_prefix: str,
    objects: tuple[RemoteObject, ...],
) -> BgmResolution
```

- [ ] **Step 1: Write failing inventory adapter tests**

Use a real HPA-321-shaped row. Assert UTC timestamp parsing, `RemoteObject` and `SimfileInventory` construction, row `SyncError` reconstruction, object-specific errors by `object_key`, original object order, and rejection of duplicate keys, malformed timestamps, invalid cache status, and malformed arrays.

- [ ] **Step 2: Implement the adapter**

Parse each row once. Convert manifest timestamps to aware UTC `datetime`. Construct existing types only. Reject object-scoped errors whose `object_key` is absent from the row.

- [ ] **Step 3: Write and implement shared body verification**

Test verified success plus `not_selected`, missing digest, absolute path, traversal, missing/non-regular file, size mismatch, digest mismatch, and expected-hash mismatch.

`resolve_verified_cache_body` must require a regular file below `cache_dir`, exact size, and exact SHA-256. It raises `ValueError("verified cache body unavailable")` on contract failure.

- [ ] **Step 4: Refactor HPA-322 to use the helper**

After HPA-322 merges, remove its private path/size/hash verification and call `resolve_verified_cache_body`. If `_CachedObject` remains for selection metadata, it must not own a second verifier. Run all HPA-322 selector tests.

- [ ] **Step 5: Write failing BGM path tests using `RemoteObject`**

Cover selected-chart-relative paths, root fallback after relative miss, exact match, unique case-insensitive match, ambiguous case-insensitive match, unknown WAV, empty value, absolute path, drive prefix, traversal, missing key, and zero events.

- [ ] **Step 6: Write discrete grouping tests**

```python
def test_different_positions_are_ambiguous_even_for_same_file() -> None:
    chart = parse_dtx_text(
        "#WAV01: bgm.ogg\n#WAV02: bgm.ogg\n#00101: 0102\n",
        "song",
    )

    result = resolve_bgm_reference(
        chart,
        build_dtx_timing_map(chart),
        selected_chart_key="42/real.dtx",
        object_prefix="42/",
        objects=(remote_object("42/bgm.ogg"),),
    )

    assert result.reason_codes == ("ambiguous_bgm_start",)
```

Add repeated tokens at the same `(key, measure, position)` and assert one selected event plus `duplicate_bgm_event`.

- [ ] **Step 7: Implement BGM policy**

Resolve every token against `inventory.objects`. Group by `(remote.key, event.measure, event.position)`. Compute chart time only after choosing the group. Any unresolved token or multiple groups quarantines the row.

- [ ] **Step 8: Validate and commit**

```bash
uv run pytest tests/benchmark/test_manifest_inventory.py tests/benchmark/test_reference_chart_selection.py tests/benchmark/test_reference_timing.py -q
uv run ruff check src/benchmark/manifest_inventory.py src/benchmark/reference_chart_selection.py src/benchmark/reference_timing.py tests/benchmark/test_manifest_inventory.py tests/benchmark/test_reference_chart_selection.py tests/benchmark/test_reference_timing.py
uv run ruff format --check src/benchmark/manifest_inventory.py src/benchmark/reference_chart_selection.py src/benchmark/reference_timing.py tests/benchmark/test_manifest_inventory.py tests/benchmark/test_reference_chart_selection.py tests/benchmark/test_reference_timing.py
git add src/benchmark/manifest_inventory.py src/benchmark/reference_chart_selection.py src/benchmark/reference_timing.py tests/benchmark/test_manifest_inventory.py tests/benchmark/test_reference_chart_selection.py tests/benchmark/test_reference_timing.py
git commit -m "feat: share manifest inventory resolution"
```

---

### Task 3: Exact-key source-audio cache fill

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

- [ ] **Step 1: Write failing exact-key tests**

Use one inventory containing `real.dtx`, `bgm.ogg`, and `preview.ogg`. Select only `bgm.ogg`. Assert only it becomes verified and only it is opened. Assert `is_selected("bgm.ogg")` remains false. Cover empty selected set, empty key, absent selected key, cache hit, and failed download.

- [ ] **Step 2: Extract a selector-driven worker**

Keep `sync_cache` unchanged publicly. Move its body behind a private worker accepting `Callable[[RemoteObject], bool]`. Use the selector for counting and selection; do not change locking, validation, hashing, installation, progress, or inventory rebuild.

- [ ] **Step 3: Add the wrapper**

Validate every selected key is non-empty and exists in the passed inventories. Call the worker with `remote.key in selected_keys`. This enforces the inventory-bound BGM invariant.

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
- `SourceAudioMetadata`
- `inspect_source_audio(path)`
- `build_audio_relative_events(...)`
- `render_reference_event_jsonl(events)`
- `publish_immutable_content(path, content, expected_sha256)`

- [ ] **Step 1: Write failing metadata tests**

Write a one-second 8 kHz WAV and assert duration, sample rate, channels, and frames. Test undecodable and zero-frame inputs mapping to `source_audio_decode_failed`.

- [ ] **Step 2: Write failing bounds tests**

At 1 kHz and one-second duration, assert `-0.0001` clamps to zero, `-0.1` is pre-audio, `1.0001` clamps to duration, and `1.1` is post-audio. Test non-finite time and no retained events.

- [ ] **Step 3: Implement metadata and bounds**

Use `soundfile.info`. Define `crux.dtx-audio-timing/v1` and `crux.dtx-reference-event/v1`. Preserve all native identities. Use one frame as tolerance. Sort deterministically.

- [ ] **Step 4: Render canonical JSONL**

Use `canonical_json_line` per event. Test repeated rendering produces identical bytes and SHA-256.

- [ ] **Step 5: Expose the existing immutable publisher**

```python
def publish_immutable_content(
    path: Path,
    content: bytes,
    expected_sha256: str,
) -> None:
    _publish_immutable(path, content, expected_sha256)
```

Test new publication, identical reuse, hash mismatch, and conflicting existing bytes.

- [ ] **Step 6: Validate and commit**

```bash
uv run pytest tests/benchmark/test_reference_timing.py tests/benchmark/test_corpus_manifest.py -q
uv run ruff check src/benchmark/reference_timing.py src/benchmark/corpus_manifest.py tests/benchmark/test_reference_timing.py tests/benchmark/test_corpus_manifest.py
uv run ruff format --check src/benchmark/reference_timing.py src/benchmark/corpus_manifest.py tests/benchmark/test_reference_timing.py tests/benchmark/test_corpus_manifest.py
git add src/benchmark/reference_timing.py src/benchmark/corpus_manifest.py tests/benchmark/test_reference_timing.py tests/benchmark/test_corpus_manifest.py
git commit -m "feat: publish bounded native reference events"
```

---

### Task 5: Two-pass orchestration and row quarantine

**Files:**
- Create: `src/benchmark/reference_timing_manifest.py`
- Create: `tests/benchmark/test_reference_timing_manifest.py`

**Interfaces:**
- `TimingRequest(manifest_path, cache_dir, output_dir)`
- `TimingCounters(ready, quarantined, events_published)`
- `TimingOutcome(status, exit_code, manifest, counters)`
- `build_reference_timing_manifest(request, *, environ=None, dependency_check=ensure_r2_dependency, store_factory=create_boto3_store)`

- [ ] **Step 1: Write fatal input tests**

Cover empty/invalid JSONL, wrong schema, duplicate IDs, mixed corpus versions, mixed buckets/endpoints, and selected chart key absent from inventory.

- [ ] **Step 2: Write row-local failure tests**

An invalid selected DTX must produce `selected_chart_parse_failed`. A chart whose timing map raises must produce `timing_map_invalid`. Include a valid sibling and assert partial exit `1` with its event artifact published.

- [ ] **Step 3: Write cache orchestration tests**

Prove:
- complete cache never calls dependency or store factories;
- only inventories with selected audio misses enter exact-key sync;
- non-selected object JSON values remain unchanged;
- returned verified fields update only the selected audio object;
- download failure quarantines only affected rows;
- upstream HPA-322 quarantine performs no parsing or R2 access.

- [ ] **Step 4: Implement exact loading**

Read bytes once, compute source hash, validate shared identities, and call `inventory_from_manifest_row` once per row. Keep raw row dictionaries only for output copying.

- [ ] **Step 5: Implement first pass**

For selected rows:
1. find exact selected chart `RemoteObject`;
2. verify through the shared helper;
3. parse DTX, mapping exceptions to `selected_chart_parse_failed`;
4. build timing map separately, mapping `ValueError` to `timing_map_invalid`;
5. resolve BGM against `inventory.objects`;
6. verify selected audio; verifier failure marks an exact cache miss.

- [ ] **Step 6: Implement targeted cache fill**

If no misses, skip R2 setup. Otherwise validate R2 config identity, load index, create/validate store, hold the existing writer lock, and call `sync_explicit_cache_keys` with only miss inventories and the exact miss set. Merge returned inventories by `simfile_id`. Failed selected objects become `source_audio_download_failed`.

- [ ] **Step 7: Implement second pass**

Re-verify selected audio with the shared helper, inspect metadata, build events, publish event JSONL, and retain metadata/counts. Map audio cache/decode/event errors to row quarantines.

- [ ] **Step 8: Build derived rows without drift**

Copy original `objects[]` order and values. Replace only `cache_status`, `sha256`, and `cache_path` on the exact selected audio key. Add timing fields and use nulls for quarantined rows.

- [ ] **Step 9: Reconcile and publish**

Require:

```text
ready + quarantined = input rows
events_published = ready
```

Use existing canonical render/publish/latest helpers. All ready -> exit `0`; any quarantine -> exit `1`; fatal exceptions reach CLI exit `2`.

- [ ] **Step 10: Validate and commit**

```bash
uv run pytest tests/benchmark/test_reference_timing_manifest.py -q
uv run ruff check src/benchmark/reference_timing_manifest.py tests/benchmark/test_reference_timing_manifest.py
uv run ruff format --check src/benchmark/reference_timing_manifest.py tests/benchmark/test_reference_timing_manifest.py
git add src/benchmark/reference_timing_manifest.py tests/benchmark/test_reference_timing_manifest.py
git commit -m "feat: publish audio relative timing manifest"
```

---

### Task 6: CLI, exact cache default, acceptance, and CI gates

**Files:**
- Modify: `src/cli/benchmark.py`
- Modify: `tests/test_cli_benchmark.py`
- Create: `tests/benchmark/test_reference_timing_acceptance.py`

- [ ] **Step 1: Write CLI and path tests**

Assert help options and request wiring. For:

```text
<root>/reference-charts/manifests/input.jsonl
```

assert the default is exactly:

```python
manifest.parent.parent.parent / "r2-corpus" / "cache"
```

Also test explicit override and exit `0`, `1`, and `2`.

- [ ] **Step 2: Add `build-reference-timing`**

Keep imports lazy. Emit one sorted JSON summary. Catch fatal `ValueError`, `ManifestPublicationError`, and `R2StoreError` at the CLI boundary.

- [ ] **Step 3: Build acceptance fixture**

Include:
1. sticky timing;
2. nested audio path;
3. one selected audio cache miss;
4. conflicting BGM starts;
5. invalid selected DTX;
6. pre/post-audio events with one retained event.

Assert partial exit, exact reconciliation, only one fake-store key opened, unchanged non-selected objects, audio-relative event time, correct schema, and no external network.

- [ ] **Step 4: Run exact CI gates**

```bash
uv run pytest tests/test_cli_benchmark.py -k build_reference_timing -q
uv run pytest tests/benchmark/test_reference_timing_acceptance.py -q
uv run pytest -q
uv run ruff check .
uv run ruff format --check src tests
uv run pylint --errors-only --disable=E1120,E0401 src
```

Do not use Black as the local format gate.

- [ ] **Step 5: Verify determinism**

Run the acceptance command twice and compare manifest/event SHA-256 values. Assert no duplicate content-addressed files.

- [ ] **Step 6: Commit**

```bash
git add src/cli/benchmark.py tests/test_cli_benchmark.py tests/benchmark/test_reference_timing_acceptance.py
git commit -m "feat: expose reference timing pipeline"
```

---

## Risk Verification Matrix

| Risk | Required proof |
|---|---|
| Inventory/cache round-trip mutates unrelated objects | Task 5 compares non-selected object JSON before/after |
| Wrong default points at `reference-charts/cache` | Task 6 exact-path test |
| Cache fill leaves stale selected fields | Task 5 merges returned `RemoteObject` fields only into selected key |
| Parse/timing error aborts corpus | Task 5 valid-sibling partial-result tests |
| Complete cache still uses R2 | Task 5 fail-fast dependency/store injections |
| Float recomputation creates false ambiguity | Task 2 discrete grouping tests |
| Legacy wrapper becomes audio-relative | Task 1 chart-time parity |
| Local formatting disagrees with CI | Task 6 Ruff format gate |

## Final Checklist

- [ ] HPA-322 is merged.
- [ ] No parallel source-object type exists.
- [ ] HPA-322 and HPA-323 share cache verification.
- [ ] Manifest records are parsed once into existing inventory types.
- [ ] Channel `01` never enters native event artifacts.
- [ ] Sticky channel `02` affects measure starts and in-measure positions.
- [ ] BGM identity uses key/measure/position.
- [ ] Only exact selected audio keys are downloaded.
- [ ] Only miss inventories enter targeted cache sync.
- [ ] Non-selected object values remain unchanged.
- [ ] Complete-cache reruns need no R2.
- [ ] Parse/timing failures quarantine only their rows.
- [ ] Native identities and bounds counts reconcile.
- [ ] Raw and aligned timing remain separate.
- [ ] Ruff, Pylint, and full tests pass.
- [ ] No HPA-324, HPA-325, or HPA-326 scope leaked.
