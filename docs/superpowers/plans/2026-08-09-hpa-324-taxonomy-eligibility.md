# HPA-324 Taxonomy and Reference Eligibility Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Freeze the versioned drum taxonomy and OaF prediction-map data, convert HPA-323's persisted native DTX events into immutable benchmark-reference events, and publish model-independent eligibility/quarantine results for HPA-325/HPA-326/HPA-327.

**Architecture:** Keep policy data small and code-defined in one `taxonomy.py` module. Reuse HPA-323's canonical event/manifest validation instead of reparsing DTX, timing, or audio. Build one offline derived reference-set stage that maps native lanes, preserves detailed and common classes, reports unknown lanes and exact-collapse diagnostics, then publishes content-addressed mapped-event JSONL plus one derived manifest. HPA-423 remains the owner of prediction conversion mechanics; HPA-324 supplies the stable taxonomy/map data it consumes.

**Tech Stack:** Python 3.13, frozen dataclasses, `typing.Literal`, immutable mappings, pathlib, existing strict canonical JSON/JSONL helpers, Click, pytest, Ruff, Pylint.

## Global Constraints

- HPA-322 and HPA-323 are complete; consume merged `main` contracts only.
- HPA-423 is active in parallel. HPA-324 owns mapping vocabulary/data; HPA-423 owns `NativeEvent -> BenchmarkEvent` conversion, prediction-map stamping, and `scorer_input.py` unblocking.
- Before the final prediction-seam verification task, rebase onto the merged HPA-423 implementation if it has landed. Do not independently reimplement its mechanism.
- Score **onset time plus instrument class only**. Preserve prediction velocity/confidence and DTX note/sample identity for diagnostics; never score DTX `#VOLUME` as hit velocity.
- Keep two explicit class levels: detailed canonical (`kick`, `snare`, `closed_hihat`, `open_hihat`, `crash`, `ride`, `high_tom`, `low_or_floor_tom`) and common comparison (`kick`, `snare`, `hihat`, `crash`, `ride`, `tom`).
- OaF's frozen 8-hit output must not fabricate open/closed hi-hat or high/low tom distinctions.
- Keep all persisted native HPA-323 events lossless. Deduplicate only the common scoring projection, and only for exact `(audio_time_sec, common_class)` identity.
- Do not use a fuzzy time bucket during mapping. HPA-325 owns matching tolerance.
- Do not rerun chart selection, DTX parsing, timing construction, BGM resolution, R2 access, or audio decoding in HPA-324.
- Do not silently discard unknown lanes. A lane is either mapped, explicitly ignored by reviewed policy, or causes reference quarantine.
- Start `IGNORED_NON_DRUM_LANES` empty. Add a lane only after the committed real-corpus diagnostic observes it and its non-drum meaning is verified. Uncertain lanes remain quarantined; maximizing corpus yield is not a reason to classify them.
- Keep MuScriptor and IDM exact map tables out of HPA-324. HPA-395 and HPA-396 must freeze their observed locked vocabularies before their scored runs.
- No generic plugin discovery, config DSL, external mapping database, service, queue, or new concurrency layer.
- Breaking internal API cleanup is allowed; do not retain aliases solely for backward compatibility.

---

## File Map

### Create

- `src/benchmark/taxonomy.py`
- `src/benchmark/reference_set.py`
- `src/benchmark/reference_set_manifest.py`
- `tools/hpa324/analyze_reference_lanes.py`
- `tests/benchmark/test_taxonomy.py`
- `tests/benchmark/test_reference_set.py`
- `tests/benchmark/test_reference_set_manifest.py`
- `tests/benchmark/test_reference_set_acceptance.py`
- `tests/tools/hpa324/test_analyze_reference_lanes.py`
- `tests/benchmark/schema_goldens/crux.benchmark-reference-event-v1.jsonl`
- `tests/benchmark/schema_goldens/crux.benchmark-reference-manifest-v1.jsonl`

### Modify

- `src/benchmark/mapping.py`
- `src/benchmark/reference_timing.py`
- `src/benchmark/reference_timing_manifest.py`
- `src/benchmark/render_audio.py`
- `src/cli/benchmark.py`
- `tests/benchmark/test_mapping.py`
- `tests/benchmark/test_reference_timing.py`
- `tests/benchmark/test_reference_timing_manifest.py`
- `tests/benchmark/test_render_audio.py`
- `tests/test_cli_benchmark.py`
- `tests/benchmark/schema_goldens/manifest.json`
- HPA-423 mapping tests only after rebasing onto its final API; do not pre-name a production mechanism file that HPA-423 may replace.

---

## Task 1: Freeze taxonomy and versioned map data

**Files:**
- Create: `src/benchmark/taxonomy.py`
- Create: `tests/benchmark/test_taxonomy.py`
- Modify: `src/benchmark/mapping.py`
- Modify: `src/benchmark/render_audio.py`
- Modify: `tests/benchmark/test_mapping.py`
- Modify: `tests/benchmark/test_render_audio.py`

**Interfaces:**

```python
DetailedDrumClass = Literal[
    "kick",
    "snare",
    "closed_hihat",
    "open_hihat",
    "crash",
    "ride",
    "high_tom",
    "low_or_floor_tom",
]

CommonDrumClass = Literal[
    "kick",
    "snare",
    "hihat",
    "crash",
    "ride",
    "tom",
]

TAXONOMY_VERSION = "crux.drum-taxonomy/v1"
DTX_LANE_MAP_VERSION = "crux.dtx-lane-map/v1"
OAF_PREDICTION_MAP_ID = "crux.prediction-map/oaf-egmd-8hit-v1"

@dataclass(frozen=True)
class ClassMapping:
    canonical_class: DetailedDrumClass | None
    common_class: CommonDrumClass | None

@dataclass(frozen=True)
class PredictionMap:
    map_id: str
    model_id: str
    native_output_space_id: str
    classes: Mapping[str, ClassMapping]

DTX_LANE_MAP: Mapping[str, ClassMapping]
DRUM_LANE_IDS: frozenset[str]
IGNORED_NON_DRUM_LANES: frozenset[str]
OAF_PREDICTION_MAP: PredictionMap
```

`OAF_PREDICTION_MAP.classes` is keyed by the normalized OaF 8-hit native class IDs that HPA-423 must expose as `NativeEvent.native_class_id`. Native MIDI/output-bin identity remains in `NativeEvent.native_midi_note` / `model_output_bin`; this map does not throw those fields away.

- [ ] **Step 1: Write exact taxonomy tests**

```python
def test_detailed_and_common_taxonomies_are_frozen() -> None:
    assert set(get_args(DetailedDrumClass)) == {
        "kick",
        "snare",
        "closed_hihat",
        "open_hihat",
        "crash",
        "ride",
        "high_tom",
        "low_or_floor_tom",
    }
    assert set(get_args(CommonDrumClass)) == {
        "kick",
        "snare",
        "hihat",
        "crash",
        "ride",
        "tom",
    }
```

Add exact assertions for `TAXONOMY_VERSION`, `DTX_LANE_MAP_VERSION`, and `OAF_PREDICTION_MAP_ID`.

- [ ] **Step 2: Write the DTX lane-map test**

Assert the exact v1 mapping:

```python
expected = {
    "11": ("closed_hihat", "hihat"),
    "12": ("snare", "snare"),
    "13": ("kick", "kick"),
    "14": ("high_tom", "tom"),
    "15": ("low_or_floor_tom", "tom"),
    "16": ("crash", "crash"),
    "17": ("low_or_floor_tom", "tom"),
    "18": ("open_hihat", "hihat"),
    "19": ("ride", "ride"),
    "1A": ("crash", "crash"),
    "1B": ("closed_hihat", "hihat"),
    "1C": ("kick", "kick"),
}
```

Require `DRUM_LANE_IDS == frozenset(expected)` and `IGNORED_NON_DRUM_LANES == frozenset()` initially.

- [ ] **Step 3: Write the OaF non-fabrication tests**

```python
def test_oaf_hihat_and_toms_only_claim_common_classes() -> None:
    assert OAF_PREDICTION_MAP.classes["hihat"] == ClassMapping(None, "hihat")
    assert OAF_PREDICTION_MAP.classes["toms"] == ClassMapping(None, "tom")


def test_oaf_sticks_stays_explicitly_unmapped() -> None:
    assert OAF_PREDICTION_MAP.classes["sticks"] == ClassMapping(None, None)
```

Also assert exact entries for `kick`, `snare`, `ride`, `ride_bell`, and `crash` and exact model ID `magenta-egmd-tf1-94529798-8hit-v1`.

- [ ] **Step 4: Run tests and verify RED**

```bash
uv run pytest tests/benchmark/test_taxonomy.py -q
```

Expected: import/module failures because `taxonomy.py` does not exist.

- [ ] **Step 5: Implement `taxonomy.py` with immutable mappings**

Use `MappingProxyType` for `DTX_LANE_MAP` and `OAF_PREDICTION_MAP.classes`. Keep values as `ClassMapping`; do not add parser/config-loader machinery.

- [ ] **Step 6: Make the existing legacy DTX mapper consume the new data**

Remove the duplicate `DtxClassMapping`/`DEFAULT_DTX_LANE_MAP` policy from `mapping.py` and import `DTX_LANE_MAP` from `taxonomy.py`.

`map_dtx_events` keeps its existing legacy `BenchmarkEvent` API because `runner.py` still uses it, but canonical class comes from `ClassMapping.canonical_class`, and metadata records the common projection:

```python
metadata = {
    **event.metadata,
    "lane_id": lane_id,
    "common_class": class_mapping.common_class,
}
```

If a supplied map entry has `canonical_class is None`, count it as unmapped instead of inventing a detailed class.

Keep `map_midi_events` only for the legacy folder/MIDI scorer. Update its old tom names to the new detailed names (`high_tom` / `low_or_floor_tom`) so it cannot emit classes outside `DetailedDrumClass`. Do not present this legacy General-MIDI table as a frozen MuScriptor map.

- [ ] **Step 7: Replace render-audio's mapping-policy dependency with lane membership**

Change `render_audio.py` from importing a mapping dictionary merely to test membership to importing `DRUM_LANE_IDS` from `taxonomy.py`.

```python
if event.lane_id not in DRUM_LANE_IDS:
    continue
```

This keeps rendering independent from canonical naming details.

- [ ] **Step 8: Run focused blast-radius tests**

```bash
uv run pytest tests/benchmark/test_taxonomy.py tests/benchmark/test_mapping.py tests/benchmark/test_render_audio.py -q
uv run ruff check src/benchmark/taxonomy.py src/benchmark/mapping.py src/benchmark/render_audio.py tests/benchmark/test_taxonomy.py tests/benchmark/test_mapping.py tests/benchmark/test_render_audio.py
```

Expected: PASS.

- [ ] **Step 9: Commit Task 1**

```bash
git add src/benchmark/taxonomy.py src/benchmark/mapping.py src/benchmark/render_audio.py \
  tests/benchmark/test_taxonomy.py tests/benchmark/test_mapping.py tests/benchmark/test_render_audio.py
git commit -m "feat: freeze benchmark drum taxonomy"
```

---

## Task 2: Expose read-only HPA-323 consumption APIs

**Files:**
- Modify: `src/benchmark/reference_timing.py`
- Modify: `src/benchmark/reference_timing_manifest.py`
- Modify: `tests/benchmark/test_reference_timing.py`
- Modify: `tests/benchmark/test_reference_timing_manifest.py`

**Interfaces:**

```python
def read_reference_events(content: bytes) -> tuple[NativeReferenceEvent, ...]: ...

@dataclass(frozen=True)
class ReferenceTimingRowView:
    simfile_id: int
    corpus_version: str
    timing_status: Literal["ready", "quarantined"]
    timing_reason_codes: tuple[TimingReasonCode, ...]
    timing_warnings: tuple[str, ...]
    reference_events_cache_path: str | None
    source_audio_key: str | None
    source_audio_content_hash: str | None


def reference_timing_row_view_from_row(
    row: Mapping[str, object],
) -> ReferenceTimingRowView: ...
```

These are read-only adapters over HPA-323's existing schema validators. They do not rerun timing or resolve source files.

- [ ] **Step 1: Add event round-trip tests**

```python
def test_read_reference_events_round_trips_canonical_render() -> None:
    content = render_reference_events((sample_native_reference_event(),))

    events = read_reference_events(content)

    assert render_reference_events(events) == content
```

Also test rejection of:

- non-canonical JSON;
- wrong key set;
- non-finite time;
- out-of-order/duplicate native identity already rejected by the existing sequence validator.

- [ ] **Step 2: Add timing-row adapter tests**

Build one valid HPA-323 `ready` row and one valid `quarantined` row using existing test helpers. Assert:

```python
assert ready_view.timing_status == "ready"
assert ready_view.reference_events_cache_path == "events/<sha>.jsonl"
assert quarantined_view.reference_events_cache_path is None
```

Add one test proving a malformed HPA-323 row is rejected by the same existing status-shape rules, not by a second relaxed validator.

- [ ] **Step 3: Run focused tests and verify RED**

```bash
uv run pytest tests/benchmark/test_reference_timing.py tests/benchmark/test_reference_timing_manifest.py \
  -k "read_reference_events or timing_row_view" -q
```

Expected: missing API failures.

- [ ] **Step 4: Implement `read_reference_events` by reusing existing validation**

In `reference_timing.py`:

1. parse each canonical JSONL row with `strict_json_loads(..., require_canonical=True)`;
2. call the existing `_validate_reference_event_row` and `_validate_reference_event_sequence` helpers;
3. convert the validated mapping into `NativeReferenceEvent`;
4. re-render with `render_reference_events` and require byte identity before returning.

Do not maintain a second list of schema keys or a second validation policy.

- [ ] **Step 5: Implement `ReferenceTimingRowView` as a thin adapter**

In `reference_timing_manifest.py`, use the existing `_validate_timing_status_shape(row)` first, then narrow the fields HPA-324 needs. Validate the top-level schema is `REFERENCE_TIMING_MANIFEST_SCHEMA` and reuse the embedded HPA-322/HPA-321 row validation already present in the module.

Do not expose mutable raw row mappings through the view.

- [ ] **Step 6: Run focused tests**

```bash
uv run pytest tests/benchmark/test_reference_timing.py tests/benchmark/test_reference_timing_manifest.py -q
uv run ruff check src/benchmark/reference_timing.py src/benchmark/reference_timing_manifest.py \
  tests/benchmark/test_reference_timing.py tests/benchmark/test_reference_timing_manifest.py
```

Expected: PASS.

- [ ] **Step 7: Commit Task 2**

```bash
git add src/benchmark/reference_timing.py src/benchmark/reference_timing_manifest.py \
  tests/benchmark/test_reference_timing.py tests/benchmark/test_reference_timing_manifest.py
git commit -m "feat: expose persisted reference timing readers"
```

---

## Task 3: Build the committed real-corpus lane diagnostic

**Files:**
- Create: `src/benchmark/reference_set_manifest.py` (loader/view portion only in this task)
- Create: `tools/hpa324/analyze_reference_lanes.py`
- Create: `tests/tools/hpa324/test_analyze_reference_lanes.py`
- Create: `tests/benchmark/test_reference_set_manifest.py` (loader tests begin here)

**Interfaces:**

```python
@dataclass(frozen=True)
class LoadedReferenceTimingManifest:
    manifest_sha256: str
    corpus_version: str
    rows: tuple[LoadedReferenceTimingRow, ...]

@dataclass(frozen=True)
class LoadedReferenceTimingRow:
    source_row: Mapping[str, object]
    view: ReferenceTimingRowView


def load_reference_timing_manifest(path: Path) -> LoadedReferenceTimingManifest: ...

@dataclass(frozen=True)
class LaneAudit:
    input_manifest_sha256: str
    row_count: int
    ready_row_count: int
    lane_event_counts: Mapping[str, int]
    unmapped_lane_event_counts: Mapping[str, int]
    unmapped_lane_simfile_counts: Mapping[str, int]
    common_collision_count: int
    common_collision_simfile_count: int
```

The diagnostic is reproducible tooling, not a new benchmark artifact schema. Its stdout is sorted ordinary JSON for review.

- [ ] **Step 1: Write canonical timing-manifest loader tests**

Cover:

- valid ready + quarantined rows load;
- exact input SHA-256 is recorded;
- rows must share one corpus version;
- duplicate simfile IDs reject;
- malformed/non-canonical JSONL rejects;
- empty manifest rejects.

- [ ] **Step 2: Write diagnostic fixture tests**

Use a temporary HPA-323 layout:

```text
reference-timing/
  events/<sha>.jsonl
  manifests/input.jsonl
```

Include:

- lane `13` mapped to kick;
- lane `18` mapped to open hi-hat/common hi-hat;
- one synthetic unknown lane `2A`;
- simultaneous `14` + `15` at the same timestamp, which becomes one common `tom` scoring identity.

Assert the audit reports the unknown lane and exactly one common collision without dropping either native event.

- [ ] **Step 3: Run tests and verify RED**

```bash
uv run pytest tests/benchmark/test_reference_set_manifest.py \
  tests/tools/hpa324/test_analyze_reference_lanes.py -q
```

Expected: missing module/tool failures.

- [ ] **Step 4: Implement the read-only timing-manifest loader**

`load_reference_timing_manifest` mirrors the canonical derived-manifest loading conventions already used by HPA-322/HPA-323:

- read exact bytes once;
- require canonical JSONL with one final newline;
- validate each row through `reference_timing_row_view_from_row`;
- reject duplicate simfile IDs and mixed corpus versions;
- retain each validated source mapping through `MappingProxyType` for later lossless pass-through;
- compute exact input SHA-256.

It must not read DTX/audio/R2.

- [ ] **Step 5: Implement the audit using production readers and taxonomy data**

For each timing-ready row:

1. resolve `reference_events_cache_path` relative to `manifest_path.parent.parent`;
2. require the path shape `events/<64 lowercase hex>.jsonl`;
3. read bytes and require SHA-256 equals the hash encoded in the filename;
4. call `read_reference_events`;
5. count mapped and unmapped native lanes using `DTX_LANE_MAP` / `IGNORED_NON_DRUM_LANES`;
6. project mapped events to exact `(audio_time_sec, common_class)` identities and count collisions.

Timing-quarantined rows count toward `row_count` but are skipped for lane-event inspection.

- [ ] **Step 6: Add the CLI entrypoint for the diagnostic tool**

```bash
uv run python -m tools.hpa324.analyze_reference_lanes \
  --manifest artifacts/benchmark/reference-timing/manifests/<sha256>.jsonl
```

Write one sorted JSON object to stdout and diagnostic errors to stderr; return nonzero for invalid input.

- [ ] **Step 7: Run focused tests**

```bash
uv run pytest tests/benchmark/test_reference_set_manifest.py \
  tests/tools/hpa324/test_analyze_reference_lanes.py -q
uv run ruff check src/benchmark/reference_set_manifest.py tools/hpa324 \
  tests/benchmark/test_reference_set_manifest.py tests/tools/hpa324
```

Expected: PASS.

- [ ] **Step 8: Run the diagnostic against the current real HPA-323 artifact before adding ignored lanes**

```bash
uv run python -m tools.hpa324.analyze_reference_lanes \
  --manifest artifacts/benchmark/reference-timing/manifests/<current-sha256>.jsonl \
  > /tmp/hpa324-reference-lanes.json
python -m json.tool /tmp/hpa324-reference-lanes.json
```

Review every `unmapped_lane_event_counts` entry. For an observed lane whose semantics are confidently non-drum, add that exact lane to `IGNORED_NON_DRUM_LANES` and add a test explaining the classification. If semantics are uncertain, leave the lane out; affected references must quarantine in Task 5. Do not infer playability from low event count or later model score.

If no real HPA-323 artifact is locally available, keep `IGNORED_NON_DRUM_LANES` empty and proceed; the implementation remains correct but may quarantine more rows until the audit is run.

- [ ] **Step 9: Commit Task 3**

```bash
git add src/benchmark/reference_set_manifest.py tools/hpa324 \
  tests/benchmark/test_reference_set_manifest.py tests/tools/hpa324 \
  src/benchmark/taxonomy.py tests/benchmark/test_taxonomy.py
git commit -m "feat: audit benchmark reference lanes"
```

---

## Task 4: Map native references and build the common scoring projection

**Files:**
- Create: `src/benchmark/reference_set.py`
- Create: `tests/benchmark/test_reference_set.py`
- Create: `tests/benchmark/schema_goldens/crux.benchmark-reference-event-v1.jsonl`
- Modify: `tests/benchmark/schema_goldens/manifest.json`

**Interfaces:**

```python
BENCHMARK_REFERENCE_EVENT_SCHEMA = "crux.benchmark-reference-event/v1"

@dataclass(frozen=True)
class MappedReferenceEvent:
    native: NativeReferenceEvent
    canonical_class: DetailedDrumClass
    common_class: CommonDrumClass

@dataclass(frozen=True)
class CommonReferenceEvent:
    audio_time_sec: float
    common_class: CommonDrumClass
    source_events: tuple[MappedReferenceEvent, ...]

@dataclass(frozen=True)
class ReferenceMappingDiagnostics:
    unmapped: Mapping[str, int]
    ignored: Mapping[str, int]
    duplicate_common_event_count: int

@dataclass(frozen=True)
class ReferenceMappingResult:
    mapped_events: tuple[MappedReferenceEvent, ...]
    common_events: tuple[CommonReferenceEvent, ...]
    diagnostics: ReferenceMappingDiagnostics


def map_reference_events(
    events: tuple[NativeReferenceEvent, ...],
) -> ReferenceMappingResult: ...


def render_mapped_reference_events(
    events: tuple[MappedReferenceEvent, ...],
) -> bytes: ...


def read_mapped_reference_events(content: bytes) -> tuple[MappedReferenceEvent, ...]: ...
```

- [ ] **Step 1: Write native-identity preservation tests**

Map one lane `18` event and assert:

```python
assert mapped.native == source_event
assert mapped.canonical_class == "open_hihat"
assert mapped.common_class == "hihat"
```

Repeat for tom, kick, snare, crash, and ride representatives.

- [ ] **Step 2: Write unknown/ignored lane tests**

Use an explicit test-only monkeypatch/map fixture rather than assuming the real audit added ignored lanes:

- unknown lane -> appears in `diagnostics.unmapped`, no mapped event;
- explicitly ignored lane -> appears in `diagnostics.ignored`, no mapped event;
- neither case mutates the source event tuple.

- [ ] **Step 3: Write exact duplicate-collapse tests**

```python
def test_common_projection_collapses_only_exact_time_and_class() -> None:
    events = (
        native_event(lane_id="14", audio_time_sec=1.0),
        native_event(lane_id="15", audio_time_sec=1.0),
        native_event(lane_id="15", audio_time_sec=1.001),
    )

    result = map_reference_events(events)

    assert [(event.audio_time_sec, event.common_class) for event in result.common_events] == [
        (1.0, "tom"),
        (1.001, "tom"),
    ]
    assert len(result.common_events[0].source_events) == 2
    assert result.diagnostics.duplicate_common_event_count == 1
```

Also test simultaneous open/closed hi-hat collapse and prove detailed `mapped_events` still contains both native hits.

- [ ] **Step 4: Write canonical mapped-event golden tests**

The stable row has exactly:

```text
schema
simfile_id
selected_chart_key
selected_chart_content_hash
source_audio_key
source_audio_content_hash
source_order
measure
position
lane_id
note_id
chart_time_sec
audio_time_sec
canonical_class
common_class
taxonomy_version
lane_map_version
```

Require `taxonomy_version == crux.drum-taxonomy/v1` and `lane_map_version == crux.dtx-lane-map/v1` in every row. Register the golden in `tests/benchmark/schema_goldens/manifest.json` with validator module `src.benchmark.reference_set`.

- [ ] **Step 5: Run tests and verify RED**

```bash
uv run pytest tests/benchmark/test_reference_set.py tests/benchmark/test_schema_goldens.py -q
```

Expected: missing reference-set module/schema failures.

- [ ] **Step 6: Implement pure mapping and exact common projection**

`map_reference_events` uses only `DTX_LANE_MAP` and `IGNORED_NON_DRUM_LANES`.

Projection algorithm:

```python
groups: dict[tuple[float, CommonDrumClass], list[MappedReferenceEvent]] = {}
for event in mapped_events:
    groups.setdefault((event.native.audio_time_sec, event.common_class), []).append(event)
```

Render groups in `(audio_time_sec, common_class)` order. `duplicate_common_event_count` is the number of native mapped events beyond the first across all exact duplicate groups:

```python
sum(len(group) - 1 for group in groups.values())
```

Do not round timestamps here.

- [ ] **Step 7: Implement mapped-event canonical JSONL read/render**

Follow HPA-323's canonical renderer/reader pattern. Preserve all native identity values verbatim and validate the exact stable key set and version IDs. Re-render on read and require byte identity.

- [ ] **Step 8: Run focused tests**

```bash
uv run pytest tests/benchmark/test_reference_set.py tests/benchmark/test_schema_goldens.py -q
uv run ruff check src/benchmark/reference_set.py tests/benchmark/test_reference_set.py
```

Expected: PASS.

- [ ] **Step 9: Commit Task 4**

```bash
git add src/benchmark/reference_set.py tests/benchmark/test_reference_set.py \
  tests/benchmark/schema_goldens/crux.benchmark-reference-event-v1.jsonl \
  tests/benchmark/schema_goldens/manifest.json
git commit -m "feat: map native benchmark references"
```

---

## Task 5: Publish model-independent reference eligibility

**Files:**
- Modify: `src/benchmark/reference_set_manifest.py`
- Modify: `tests/benchmark/test_reference_set_manifest.py`
- Create: `tests/benchmark/schema_goldens/crux.benchmark-reference-manifest-v1.jsonl`
- Modify: `tests/benchmark/schema_goldens/manifest.json`

**Interfaces:**

```python
BENCHMARK_REFERENCE_MANIFEST_SCHEMA = "crux.benchmark-reference-manifest/v1"

ReferenceEligibilityStatus = Literal[
    "eligible",
    "eligible_with_warnings",
    "quarantined",
]

EligibilityReasonCode = Literal[
    "upstream_reference_unavailable",
    "reference_event_artifact_invalid",
    "unclassified_reference_lane",
    "no_scored_drum_events",
]

@dataclass(frozen=True)
class ReferenceSetRequest:
    manifest_path: Path
    output_dir: Path

@dataclass(frozen=True)
class ReferenceSetOutcome:
    status: Literal["complete", "partial", "failed"]
    exit_code: Literal[0, 1, 2]
    manifest: PublishedManifest | None
    eligible_count: int
    eligible_with_warnings_count: int
    quarantined_count: int
    mapped_event_artifact_count: int
```

Each derived row adds:

```text
schema_version = crux.benchmark-reference-manifest/v1
source_reference_timing_manifest_sha256
source_reference_timing_version
taxonomy_version
lane_map_version
reference_eligibility_status
reference_eligibility_reason_codes
reference_eligibility_warnings
benchmark_reference_events_path
mapped_event_count
common_scored_event_count
ignored_event_count
unmapped_event_count
duplicate_common_event_count
```

All HPA-323 fields except its top-level derived `corpus_version` pass through verbatim.

- [ ] **Step 1: Write upstream-quarantine tests**

A timing-quarantined input row becomes:

```text
reference_eligibility_status = quarantined
reference_eligibility_reason_codes = [upstream_reference_unavailable]
benchmark_reference_events_path = null
mapped_event_count = 0
common_scored_event_count = 0
```

HPA-323's original `timing_reason_codes` remain unchanged in the carried-through row.

- [ ] **Step 2: Write artifact-integrity tests**

For timing-ready rows, quarantine with `reference_event_artifact_invalid` when:

- path is not `events/<sha256>.jsonl`;
- path escapes the timing artifact root;
- file is missing;
- filename hash does not match bytes;
- `read_reference_events` rejects content.

These are row-local quarantines, not fatal whole-run failures.

- [ ] **Step 3: Write mapping eligibility tests**

Cover all three statuses:

1. mapped events, no warnings -> `eligible`;
2. mapped events with ignored lanes or duplicate common collapse -> `eligible_with_warnings`;
3. any unclassified lane -> `quarantined` with `unclassified_reference_lane` and no scored artifact;
4. no mapped drum event after explicit ignores -> `quarantined` with `no_scored_drum_events`.

Warnings must be deterministic. Use these exact formats:

```text
ignored_reference_lane:<LANE>:count=<N>
duplicate_common_projection:count=<N>
```

Sort warning strings lexicographically.

- [ ] **Step 4: Write accounting tests**

Require:

```python
eligible_count + eligible_with_warnings_count + quarantined_count == total_input_rows
mapped_event_artifact_count == eligible_count + eligible_with_warnings_count
```

A published manifest with any quarantine returns exit `1`; no quarantines returns `0`. Fatal input manifest or output publication failure returns `2` and no manifest.

- [ ] **Step 5: Write immutable publication tests**

For an eligible row:

1. render `mapped_events`;
2. SHA-256 the bytes;
3. publish to `output_dir / "events" / f"{sha256}.jsonl"` using the existing immutable publisher;
4. set `benchmark_reference_events_path` to that relative path;
5. publish the derived manifest via existing `render_manifest`, `publish_manifest`, and `publish_latest_manifest`.

A repeated run with identical inputs must produce the same event bytes and manifest content.

- [ ] **Step 6: Add the benchmark-reference manifest schema golden**

Create a two-row canonical golden: one eligible, one quarantined. Register it in `tests/benchmark/schema_goldens/manifest.json` with validator module `src.benchmark.reference_set_manifest`.

The validator must enforce the closed status/reason-code sets, version fields, nullable path/count shapes, and accounting-relevant row invariants.

- [ ] **Step 7: Run tests and verify RED**

```bash
uv run pytest tests/benchmark/test_reference_set_manifest.py tests/benchmark/test_schema_goldens.py -q
```

Expected: missing eligibility/orchestration behavior.

- [ ] **Step 8: Implement row evaluation**

For each loaded HPA-323 row:

```text
if timing_status != ready:
    quarantine upstream_reference_unavailable
else:
    verify/read persisted native event artifact
    map_reference_events(native_events)
    if unmapped:
        quarantine unclassified_reference_lane
    elif no mapped events:
        quarantine no_scored_drum_events
    else:
        publish mapped event artifact
        eligible_with_warnings if ignored or duplicate diagnostics else eligible
```

Do not inspect model predictions or scores anywhere in this decision.

- [ ] **Step 9: Implement row pass-through and publication**

Use the HPA-323 source row as the base, remove only its top-level `corpus_version`, replace `schema_version`, add the HPA-324 fields, then let `render_manifest` derive the new corpus version.

Record immediate lineage:

```text
source_reference_timing_manifest_sha256 = exact input bytes SHA-256
source_reference_timing_version = HPA-323 input corpus_version
```

Preserve all older HPA-321/HPA-322 lineage fields unchanged.

- [ ] **Step 10: Run focused tests**

```bash
uv run pytest tests/benchmark/test_reference_set_manifest.py tests/benchmark/test_schema_goldens.py -q
uv run ruff check src/benchmark/reference_set_manifest.py tests/benchmark/test_reference_set_manifest.py
```

Expected: PASS.

- [ ] **Step 11: Commit Task 5**

```bash
git add src/benchmark/reference_set_manifest.py tests/benchmark/test_reference_set_manifest.py \
  tests/benchmark/schema_goldens/crux.benchmark-reference-manifest-v1.jsonl \
  tests/benchmark/schema_goldens/manifest.json
git commit -m "feat: classify benchmark reference eligibility"
```

---

## Task 6: Add the offline `build-reference-set` CLI and acceptance path

**Files:**
- Modify: `src/cli/benchmark.py`
- Modify: `tests/test_cli_benchmark.py`
- Create: `tests/benchmark/test_reference_set_acceptance.py`

**Interfaces:**

```bash
uv run crux benchmark build-reference-set \
  --manifest artifacts/benchmark/reference-timing/manifests/<sha256>.jsonl \
  --output-dir artifacts/benchmark/reference-set
```

Machine-readable stdout fields:

```text
corpus_version
eligible_count
eligible_with_warnings_count
exit_code
manifest_path
manifest_sha256
mapped_event_artifact_count
quarantined_count
status
```

- [ ] **Step 1: Write CLI tests**

Cover:

- `--manifest` required;
- `--output-dir` default is `artifacts/benchmark/reference-set`;
- exit `0` complete outcome;
- exit `1` partial outcome;
- exit `2` failed outcome;
- stdout is one canonical JSON object and no R2/config credential path is touched.

- [ ] **Step 2: Write offline acceptance fixture**

Build a temporary HPA-323 manifest with:

- one ready kick/snare reference;
- one ready simultaneous high/low tom reference producing a duplicate common projection warning;
- one upstream-quarantined reference.

Run `run_reference_set`, then assert:

```python
assert outcome.eligible_count == 1
assert outcome.eligible_with_warnings_count == 1
assert outcome.quarantined_count == 1
assert outcome.mapped_event_artifact_count == 2
assert outcome.exit_code == 1
```

Read both published mapped-event artifacts and prove they retain native lane/note identities.

- [ ] **Step 3: Add the rescore-without-inference acceptance test**

Construct one persisted HPA-323 native event artifact once. Map it with lane map v1, then create a test-only alternate map version that changes only one canonical projection and remap the same native bytes.

Assert:

- no audio/model/transcription function is called;
- native input SHA-256 is unchanged;
- mapped output/version changes as expected.

This is the concrete proof that a mapping correction requires rescoring/remapping persisted native events, not rerunning inference.

- [ ] **Step 4: Run tests and verify RED**

```bash
uv run pytest tests/test_cli_benchmark.py -k build_reference_set -q
uv run pytest tests/benchmark/test_reference_set_acceptance.py -q
```

Expected: missing command/orchestration failures.

- [ ] **Step 5: Implement CLI summary helper and command**

Follow `_emit_reference_timing_summary` style. Imports stay lazy inside the Click command.

```python
@benchmark.command("build-reference-set")
@click.option(
    "--manifest",
    "manifest_path",
    type=click.Path(path_type=Path, dir_okay=False),
    required=True,
)
@click.option(
    "--output-dir",
    type=click.Path(path_type=Path, file_okay=False),
    default=Path("artifacts/benchmark/reference-set"),
    show_default=True,
)
@click.pass_context
def build_reference_set_command(...): ...
```

No `--cache-dir`, R2 option, model option, tolerance option, or concurrency option belongs here.

- [ ] **Step 6: Run focused acceptance**

```bash
uv run pytest tests/benchmark/test_reference_set.py \
  tests/benchmark/test_reference_set_manifest.py \
  tests/benchmark/test_reference_set_acceptance.py \
  tests/test_cli_benchmark.py -k "reference_set or build_reference_set" -q
```

Then run the whole CLI test file because Click registration can have broad blast radius:

```bash
uv run pytest tests/test_cli_benchmark.py -q
```

Expected: PASS.

- [ ] **Step 7: Commit Task 6**

```bash
git add src/cli/benchmark.py tests/test_cli_benchmark.py \
  tests/benchmark/test_reference_set_acceptance.py
git commit -m "feat: publish benchmark reference set"
```

---

## Task 7: Verify the HPA-423 prediction-mapping seam without taking ownership of it

**Files:**
- Modify: `tests/benchmark/test_taxonomy.py`
- Modify: the HPA-423 prediction-mapping test file after its final path/API is known from merged `main`.
- Modify production HPA-423 mapping code **only if** the merged implementation does not consume an injected/versioned `PredictionMap`; keep any such change as a narrow seam adaptation, not a second mapper.

**Interfaces HPA-324 requires from HPA-423:**

```python
def map_native_events(
    events: tuple[NativeEvent, ...],
    *,
    prediction_map: PredictionMap,
) -> MappingResult: ...
```

Equivalent naming is acceptable if HPA-423 chose a different final API, but the semantics are not:

- every mapped event retains native class/output-bin/MIDI/confidence/velocity metadata;
- mapping result carries `prediction_map_version == prediction_map.map_id`;
- unknown classes remain visible in diagnostics;
- OaF `hihat` maps only to common `hihat`, not fabricated `closed_hihat`/`open_hihat`;
- OaF `toms` maps only to common `tom`, not fabricated `high_tom`/`low_or_floor_tom`;
- OaF `sticks` is visible as unmapped;
- `model_id` and `input_view_id` remain attached through prediction persistence/scorer input.

- [ ] **Step 1: Rebase onto current `main` after HPA-423 lands**

```bash
git fetch origin
git rebase origin/main
```

Inspect HPA-423's final mapper and tests before editing. Do not reintroduce files HPA-423 deleted.

- [ ] **Step 2: Add an OaF map-consumption test**

Use the final HPA-423 `NativeEvent` constructor. Build representative normalized native classes:

```text
kick
hihat
toms
ride_bell
sticks
```

Call the HPA-423 mapper with `OAF_PREDICTION_MAP` and assert:

```text
kick      -> canonical kick, common kick
hihat     -> canonical null, common hihat
toms      -> canonical null, common tom
ride_bell -> canonical ride, common ride
sticks    -> unmapped diagnostic
```

Require the persisted/returned map ID to equal `crux.prediction-map/oaf-egmd-8hit-v1`.

- [ ] **Step 3: Add the scorer-input regression required by the seam**

Using HPA-423's final scorer-input API, prove a prediction artifact mapped under `OAF_PREDICTION_MAP` can produce common scorer events without a model-specific branch in HPA-325.

Do not add HPA-325 metrics or tolerance matching here.

- [ ] **Step 4: Run HPA-423 + HPA-324 focused tests**

```bash
uv run pytest tests/benchmark/test_taxonomy.py tests/benchmark/test_reference_set.py \
  tests/benchmark/test_reference_set_manifest.py -q
```

Also run the exact HPA-423 mapper/scorer-input test files discovered after the rebase.

Expected: PASS.

- [ ] **Step 5: Commit the seam verification**

Stage only the HPA-324 taxonomy/test changes plus any minimal HPA-423 adapter needed to accept `PredictionMap`.

```bash
git commit -m "test: bind OaF prediction map to benchmark taxonomy"
```

If HPA-423 already consumes `PredictionMap` with the required semantics, this task may be test-only.

---

## Task 8: Final acceptance and cleanup

**Files:**
- All HPA-324 files above.

- [ ] **Step 1: Run the full benchmark/reference suite**

```bash
uv run pytest tests/benchmark -q
uv run pytest tests/test_cli_benchmark.py -q
uv run pytest tests/tools/hpa324 -q
```

Expected: PASS.

- [ ] **Step 2: Run repository validation**

```bash
uv run pytest -q
uv run ruff check .
uv run ruff format --check src tests tools
uv run pylint --errors-only --disable=E1120,E0401 src
```

Expected: PASS, apart from any already-documented repository baseline exception that is byte-identical to `main`.

- [ ] **Step 3: Verify schema goldens and deterministic rerendering**

```bash
uv run pytest tests/benchmark/test_schema_goldens.py -q
```

Run one acceptance fixture twice and compare published event/manifest SHA-256 values; identical source bytes and mapping versions must yield identical hashes.

- [ ] **Step 4: Search for stale taxonomy policy**

```bash
git grep -n "DEFAULT_DTX_LANE_MAP\|DtxClassMapping\|low_tom" -- src tests
```

Resolve active benchmark-policy duplicates. Historical docs may retain old wording; production benchmark code must use `taxonomy.py` as the single owner.

- [ ] **Step 5: Check scope**

```bash
git diff --stat origin/main...HEAD
git diff --check origin/main...HEAD
```

Confirm the implementation did not add:

- MuScriptor/IDM guessed class tables;
- velocity scoring;
- DTX/audio/R2 reprocessing;
- scoring metrics/tolerances;
- backend plugin/config frameworks;
- result-driven mapping changes.

- [ ] **Step 6: Commit any final mechanical cleanup**

Only if Step 4/5 found real issues:

```bash
git add <exact-cleanup-files>
git commit -m "chore: finalize benchmark taxonomy contracts"
```

---

## HPA-324 Completion Gate

Before moving HPA-324 to Done, verify all of the following from committed code/tests/artifacts:

1. DTX detailed/common taxonomy and lane-map v1 are frozen in one module.
2. OaF has a distinct immutable prediction-map ID and does not fabricate unsupported hi-hat/tom distinctions.
3. HPA-323 native event artifacts are consumed directly and can be remapped without rerunning inference.
4. Unknown lanes are visible; only reviewed explicit ignored lanes are skipped.
5. Exact duplicate common projections are deterministic and retain all native source events for diagnosis.
6. Every HPA-323 input row is exactly one of eligible, eligible-with-warnings, or quarantined.
7. Reference eligibility is independent of model score and model-specific inference success.
8. Eligible reference artifacts contain both detailed and common class identities plus taxonomy/lane-map versions.
9. HPA-423 consumes HPA-324's OaF map data and stamps its map identity on predictions.
10. HPA-395/HPA-396 remain responsible for freezing their exact MuScriptor/IDM maps before their later scored runs; no guessed future vocabulary was added here.
