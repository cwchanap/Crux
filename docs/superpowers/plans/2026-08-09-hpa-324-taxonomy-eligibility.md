# HPA-324 Taxonomy and Reference Eligibility Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Freeze the benchmark drum taxonomy and OaF prediction-map data, convert HPA-323 persisted native DTX events into immutable benchmark-reference events, expose one deterministic common scorer projection, and publish model-independent eligible/quarantined reference results for HPA-325/HPA-326/HPA-327.

**Architecture:** Keep policy data small and code-defined in `taxonomy.py`. Reuse HPA-323 canonical event/manifest validation instead of reparsing DTX, timing, or audio. Build one pure native-reference mapper plus one exported common projection, use those exact production functions in the real-corpus lane audit, then publish a derived reference manifest with binary `eligible|quarantined` status and orthogonal warnings. HPA-423 remains owner of prediction conversion mechanics; HPA-324 supplies OaF mapping data keyed by `upstream_8hit_group_id` without rewriting native MIDI identity.

**Tech Stack:** Python 3.13, frozen dataclasses, `typing.Literal`, `MappingProxyType`, `Decimal`, pathlib, existing strict canonical JSON/JSONL helpers, Click, pytest, Ruff, Pylint.

## Global Constraints

- HPA-322 and HPA-323 are complete; consume merged `main` contracts only.
- HPA-423 is active in parallel. HPA-324 owns mapping vocabulary/data; HPA-423 owns prediction conversion, map stamping, and `scorer_input.py` unblocking.
- Before Task 7, rebase onto merged HPA-423 if it has landed. Do not independently reimplement its mapper or redefine its final API.
- Score **onset time plus instrument class only**. Preserve prediction velocity/confidence and DTX note/sample identity for diagnostics; never score DTX `#VOLUME` as hit velocity.
- Keep two class levels: detailed canonical (`kick`, `snare`, `closed_hihat`, `open_hihat`, `crash`, `ride`, `high_tom`, `low_or_floor_tom`) and common comparison (`kick`, `snare`, `hihat`, `crash`, `ride`, `tom`).
- Define one total `DETAILED_TO_COMMON` projection. Any mapping with a non-null detailed class must agree with it.
- OaF map lookup uses `NativeEvent.native_metadata["upstream_8hit_group_id"]`; do not rewrite `NativeEvent.native_class_id="midi_<note>"`.
- Keep all persisted HPA-323 events lossless. Deduplicate only the common scorer projection.
- Common collapse identity uses canonical durable time (`Decimal(str(audio_time_sec))`) plus `common_class`; do not use raw float object identity, fuzzy buckets, or score tolerances.
- Do not rerun chart selection, DTX parsing, timing construction, BGM resolution, R2 access, or audio decoding in HPA-324.
- Do not silently discard unknown lanes. A lane is mapped, explicitly ignored after review, or quarantines the reference.
- Start `IGNORED_NON_DRUM_LANES` empty. Grow it only after the committed real-corpus audit verifies an observed lane is non-drum.
- Reference eligibility status is exactly `eligible|quarantined`. Warnings are an array on eligible rows, not a third status.
- Mapped benchmark-reference event rows mirror HPA-323 style and do not contain a per-row `schema` field.
- Keep exact MuScriptor and IDM map tables out of HPA-324. HPA-395/HPA-396 freeze their observed locked vocabularies before their scored runs.
- No generic plugin discovery, key-selector framework, config DSL, mapping database, service, queue, or concurrency layer.
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
- `src/benchmark/midi_io.py`
- `src/benchmark/reference_timing.py`
- `src/benchmark/reference_timing_manifest.py`
- `src/benchmark/render_audio.py`
- `src/cli/benchmark.py`
- `scripts/calibrate_egmd_mapping.py` only if taxonomy grep finds an active old-name assumption rather than a passive import
- `tests/benchmark/test_mapping.py`
- `tests/benchmark/test_midi_io.py`
- `tests/benchmark/test_reference_timing.py`
- `tests/benchmark/test_reference_timing_manifest.py`
- `tests/benchmark/test_render_audio.py`
- `tests/test_cli_benchmark.py`
- `tests/benchmark/schema_goldens/manifest.json`
- HPA-423 mapping tests after rebasing onto its actual merged API.

---

## Task 1: Freeze taxonomy and migrate existing mapping consumers

**Files:**
- Create: `src/benchmark/taxonomy.py`
- Create: `tests/benchmark/test_taxonomy.py`
- Modify: `src/benchmark/mapping.py`
- Modify: `src/benchmark/midi_io.py`
- Modify: `src/benchmark/render_audio.py`
- Modify: `tests/benchmark/test_mapping.py`
- Modify: `tests/benchmark/test_midi_io.py`
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
OAF_PREDICTION_MAP_METADATA_KEY = "upstream_8hit_group_id"

DETAILED_TO_COMMON: Mapping[DetailedDrumClass, CommonDrumClass]


def project_to_common(detailed: DetailedDrumClass) -> CommonDrumClass: ...


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

`OAF_PREDICTION_MAP.classes` is keyed by the eight values carried in `NativeEvent.native_metadata["upstream_8hit_group_id"]`. `native_class_id`, output bin, and native MIDI note remain independent native identity.

- [ ] **Step 1: Write exact taxonomy and projection tests**

```python
def test_detailed_to_common_projection_is_total() -> None:
    assert DETAILED_TO_COMMON == {
        "kick": "kick",
        "snare": "snare",
        "closed_hihat": "hihat",
        "open_hihat": "hihat",
        "crash": "crash",
        "ride": "ride",
        "high_tom": "tom",
        "low_or_floor_tom": "tom",
    }
    assert set(DETAILED_TO_COMMON) == set(get_args(DetailedDrumClass))
```

Add exact assertions for the two class Literals and all three version IDs.

- [ ] **Step 2: Write the DTX lane-map tests**

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

For every DTX mapping entry:

```python
assert mapping.common_class == project_to_common(mapping.canonical_class)
```

- [ ] **Step 3: Write OaF group-key tests**

Require the exact keys:

```python
assert set(OAF_PREDICTION_MAP.classes) == {
    "kick",
    "snare",
    "toms",
    "hihat",
    "ride",
    "ride_bell",
    "crash",
    "sticks",
}
```

Assert:

```python
assert OAF_PREDICTION_MAP.classes["hihat"] == ClassMapping(None, "hihat")
assert OAF_PREDICTION_MAP.classes["toms"] == ClassMapping(None, "tom")
assert OAF_PREDICTION_MAP.classes["sticks"] == ClassMapping(None, None)
```

For entries with a non-null detailed class, require the same `project_to_common` invariant.

- [ ] **Step 4: Write reverse-MIDI rename regressions**

Require `write_reference_midi` to handle the new detailed tom class instead of warning/skipping it:

```python
assert REFERENCE_CLASS_TO_MIDI["low_or_floor_tom"] == 45
assert "low_tom" not in REFERENCE_CLASS_TO_MIDI
```

Before removing `mid_tom`, search active producers:

```bash
git grep -n '"mid_tom"\|"low_tom"' -- src tests scripts
```

If no active producer emits `mid_tom`, remove it from `REFERENCE_CLASS_TO_MIDI`; do not keep it for compatibility alone.

- [ ] **Step 5: Run tests and verify RED**

```bash
uv run pytest tests/benchmark/test_taxonomy.py tests/benchmark/test_mapping.py \
  tests/benchmark/test_midi_io.py -q
```

Expected RED is missing taxonomy/new-name behavior.

- [ ] **Step 6: Implement `taxonomy.py` with immutable maps**

Use `MappingProxyType` for `DETAILED_TO_COMMON`, `DTX_LANE_MAP`, and `OAF_PREDICTION_MAP.classes`.

`project_to_common` is a direct lookup:

```python
def project_to_common(detailed: DetailedDrumClass) -> CommonDrumClass:
    return DETAILED_TO_COMMON[detailed]
```

Do not add a loader or generic validation framework.

- [ ] **Step 7: Migrate `mapping.py` to the single DTX policy owner**

Delete `DtxClassMapping` and `DEFAULT_DTX_LANE_MAP` as policy owners. Import `DTX_LANE_MAP` from `taxonomy.py`.

Keep the legacy `map_dtx_events` API because `runner.py` still uses it. Set `canonical_class` from `ClassMapping.canonical_class` and preserve the common class in metadata:

```python
metadata = {
    **event.metadata,
    "lane_id": lane_id,
    "common_class": class_mapping.common_class,
}
```

Update the old test that expected `metadata["native_class"]`; native lane/note identity already exists on the source event and should not be duplicated under a misleading class field.

Keep `DEFAULT_MIDI_NOTE_MAP` only for the legacy folder/MIDI scorer. Rename its emitted tom classes to `low_or_floor_tom`/`high_tom` so it cannot emit values outside `DetailedDrumClass`.

- [ ] **Step 8: Update render and reverse-MIDI consumers**

`render_audio.py` should import only `DRUM_LANE_IDS` for membership:

```python
if event.lane_id not in DRUM_LANE_IDS:
    continue
```

Update `REFERENCE_CLASS_TO_MIDI` in `midi_io.py` in the same task and extend `test_write_reference_midi_handles_simultaneous_hits` or add a focused `low_or_floor_tom` test in `tests/benchmark/test_midi_io.py`.

- [ ] **Step 9: Check the calibration script blast radius**

```bash
git grep -n 'DEFAULT_MIDI_NOTE_MAP\|map_dtx_events\|low_tom\|mid_tom' -- scripts/calibrate_egmd_mapping.py
```

If the script only consumes the maps dynamically, no code change is needed. If it asserts or serializes an old class spelling as policy, update that assumption and add the smallest regression available.

- [ ] **Step 10: Run focused validation**

```bash
uv run pytest tests/benchmark/test_taxonomy.py tests/benchmark/test_mapping.py \
  tests/benchmark/test_midi_io.py tests/benchmark/test_render_audio.py -q
uv run ruff check src/benchmark/taxonomy.py src/benchmark/mapping.py src/benchmark/midi_io.py \
  src/benchmark/render_audio.py tests/benchmark
```

- [ ] **Step 11: Commit Task 1**

```bash
git add src/benchmark/taxonomy.py src/benchmark/mapping.py src/benchmark/midi_io.py \
  src/benchmark/render_audio.py tests/benchmark/test_taxonomy.py \
  tests/benchmark/test_mapping.py tests/benchmark/test_midi_io.py \
  tests/benchmark/test_render_audio.py
git commit -m "feat: freeze benchmark drum taxonomy"
```

If Step 9 required an actual calibration-script edit, stage `scripts/calibrate_egmd_mapping.py` in this same commit.

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

These are read-only adapters over HPA-323 validators. They do not rerun timing or resolve source files.

- [ ] **Step 1: Write event reader round-trip tests**

```python
def test_read_reference_events_round_trips_canonical_render() -> None:
    content = render_reference_events((sample_native_reference_event(),))
    events = read_reference_events(content)
    assert render_reference_events(events) == content
```

Also reject non-canonical JSON, wrong key sets, non-finite time, and invalid native event ordering/duplicates through the existing validators.

- [ ] **Step 2: Write timing-row adapter tests**

Use existing valid ready/quarantined row fixtures. Require the adapter to call the same status-shape policy HPA-323 already uses.

```python
assert ready_view.timing_status == "ready"
assert ready_view.reference_events_cache_path.startswith("events/")
assert quarantined_view.reference_events_cache_path is None
```

- [ ] **Step 3: Run focused tests and verify RED**

```bash
uv run pytest tests/benchmark/test_reference_timing.py tests/benchmark/test_reference_timing_manifest.py \
  -k "read_reference_events or timing_row_view" -q
```

- [ ] **Step 4: Implement `read_reference_events` by reusing HPA-323 validation**

1. parse each row through `strict_json_loads(..., require_canonical=True)`;
2. call existing `_validate_reference_event_row` and `_validate_reference_event_sequence`;
3. construct `NativeReferenceEvent` values;
4. re-render with `render_reference_events` and require byte identity.

Do not maintain a second schema key list.

- [ ] **Step 5: Implement `ReferenceTimingRowView` as a thin adapter**

Call the existing timing row validator/status-shape logic first, then narrow only the fields HPA-324 needs. Do not expose mutable raw mappings through the view.

- [ ] **Step 6: Run focused validation**

```bash
uv run pytest tests/benchmark/test_reference_timing.py tests/benchmark/test_reference_timing_manifest.py -q
uv run ruff check src/benchmark/reference_timing.py src/benchmark/reference_timing_manifest.py \
  tests/benchmark/test_reference_timing.py tests/benchmark/test_reference_timing_manifest.py
```

- [ ] **Step 7: Commit Task 2**

```bash
git add src/benchmark/reference_timing.py src/benchmark/reference_timing_manifest.py \
  tests/benchmark/test_reference_timing.py tests/benchmark/test_reference_timing_manifest.py
git commit -m "feat: expose persisted reference timing readers"
```

---

## Task 3: Map native references and own the common scorer projection

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


def canonical_reference_time(time_sec: float) -> Decimal: ...


def project_common_reference_events(
    mapped_events: tuple[MappedReferenceEvent, ...],
) -> tuple[CommonReferenceEvent, ...]: ...


def map_reference_events(
    events: tuple[NativeReferenceEvent, ...],
) -> ReferenceMappingResult: ...


def render_mapped_reference_events(
    events: tuple[MappedReferenceEvent, ...],
) -> bytes: ...


def read_mapped_reference_events(content: bytes) -> tuple[MappedReferenceEvent, ...]: ...
```

- [ ] **Step 1: Write native-identity preservation tests**

Map representative kick, snare, open hi-hat, crash, ride, high-tom, and low/floor-tom lanes. For lane `18`:

```python
assert mapped.native == source_event
assert mapped.canonical_class == "open_hihat"
assert mapped.common_class == "hihat"
```

- [ ] **Step 2: Write unknown/ignored lane tests**

Use a test-only lane-map/ignore-set parameter or monkeypatch so tests do not depend on the real audit outcome.

- unknown -> `diagnostics.unmapped`, no mapped event;
- explicitly ignored -> `diagnostics.ignored`, no mapped event;
- source tuple remains unchanged.

- [ ] **Step 3: Write canonical time-key tests**

```python
def test_canonical_reference_time_matches_durable_decimal_text() -> None:
    assert canonical_reference_time(0.1) == Decimal("0.1")
    assert canonical_reference_time(1.001) == Decimal("1.001")
```

Round-trip one HPA-323 event through `render_reference_events`/`read_reference_events` and require the canonical key is unchanged.

- [ ] **Step 4: Write exact common-collapse tests through the public API**

```python
def test_common_projection_collapses_only_exact_canonical_time_and_class() -> None:
    mapped = map_reference_events(
        (
            native_event(lane_id="14", audio_time_sec=1.0),
            native_event(lane_id="15", audio_time_sec=1.0),
            native_event(lane_id="15", audio_time_sec=1.001),
        )
    ).mapped_events

    common = project_common_reference_events(mapped)

    assert [(event.audio_time_sec, event.common_class) for event in common] == [
        (1.0, "tom"),
        (1.001, "tom"),
    ]
    assert len(common[0].source_events) == 2
```

Also cover simultaneous open/closed hi-hat and prove all detailed mapped rows remain present.

- [ ] **Step 5: Write mapped-event schema-golden tests**

The stable row has exactly these keys and **no `schema` key**:

```text
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

Require `taxonomy_version == crux.drum-taxonomy/v1` and `lane_map_version == crux.dtx-lane-map/v1`. Register the golden under schema ID `crux.benchmark-reference-event/v1` with validator module `src.benchmark.reference_set`.

- [ ] **Step 6: Run tests and verify RED**

```bash
uv run pytest tests/benchmark/test_reference_set.py tests/benchmark/test_schema_goldens.py -q
```

- [ ] **Step 7: Implement pure mapping**

Map only through `DTX_LANE_MAP` and `IGNORED_NON_DRUM_LANES`. For every mapped DTX event, assert/derive:

```python
common_class = project_to_common(canonical_class)
```

Do not permit free-form mismatched detailed/common pairs.

- [ ] **Step 8: Implement the one common projection**

```python
groups: dict[tuple[Decimal, CommonDrumClass], list[MappedReferenceEvent]] = {}
for event in mapped_events:
    key = (canonical_reference_time(event.native.audio_time_sec), event.common_class)
    groups.setdefault(key, []).append(event)
```

Sort by `(canonical_time, common_class)`. `duplicate_common_event_count` equals:

```python
sum(len(group) - 1 for group in groups.values())
```

`map_reference_events` must call `project_common_reference_events`; do not duplicate collapse logic privately.

- [ ] **Step 9: Implement canonical mapped-event read/render**

Mirror HPA-323's renderer/reader pattern. Preserve all native identity fields and exact version IDs. Event rows contain no per-row schema marker. Re-render on read and require byte identity.

- [ ] **Step 10: Run focused validation**

```bash
uv run pytest tests/benchmark/test_reference_set.py tests/benchmark/test_schema_goldens.py -q
uv run ruff check src/benchmark/reference_set.py tests/benchmark/test_reference_set.py
```

- [ ] **Step 11: Commit Task 3**

```bash
git add src/benchmark/reference_set.py tests/benchmark/test_reference_set.py \
  tests/benchmark/schema_goldens/crux.benchmark-reference-event-v1.jsonl \
  tests/benchmark/schema_goldens/manifest.json
git commit -m "feat: map native benchmark references"
```

---

## Task 4: Build the committed real-corpus lane diagnostic

**Files:**
- Create: `src/benchmark/reference_set_manifest.py` (loader/view portion only)
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

The audit is reproducible tooling, not a new stable benchmark artifact schema.

- [ ] **Step 1: Write canonical timing-manifest loader tests**

Cover valid ready/quarantined rows, exact input SHA-256, one shared corpus version, duplicate simfile rejection, malformed/non-canonical JSONL rejection, and empty manifest rejection.

- [ ] **Step 2: Write diagnostic fixture tests using production mapping**

Create a temporary HPA-323 layout with mapped lane `13`, mapped lane `18`, synthetic unknown lane `2A`, and simultaneous lanes `14` + `15` at one timestamp.

The audit must call `map_reference_events`/`project_common_reference_events`, not reproduce lane/collapse logic.

Require one unknown lane and one common collision while both native tom events remain in the source artifact.

- [ ] **Step 3: Run tests and verify RED**

```bash
uv run pytest tests/benchmark/test_reference_set_manifest.py \
  tests/tools/hpa324/test_analyze_reference_lanes.py -q
```

- [ ] **Step 4: Implement read-only timing-manifest loading**

- read exact bytes once;
- require canonical JSONL;
- validate rows through `reference_timing_row_view_from_row`;
- reject duplicate simfile IDs and mixed corpus versions;
- retain validated source rows with `MappingProxyType` for later pass-through;
- compute exact input SHA-256.

Do not add a second timing status-shape policy.

- [ ] **Step 5: Implement the audit through production readers**

For each timing-ready row:

1. resolve `reference_events_cache_path` beneath the timing artifact root;
2. verify `events/<64 lowercase hex>.jsonl` shape and filename hash;
3. read bytes and call `read_reference_events`;
4. call `map_reference_events`;
5. aggregate mapped/unmapped/ignored lane diagnostics;
6. use the returned/public common projection to count collisions.

Timing-quarantined rows count toward total rows but are not opened.

- [ ] **Step 6: Add the audit CLI**

```bash
uv run python -m tools.hpa324.analyze_reference_lanes --manifest "$REFERENCE_TIMING_MANIFEST"
```

Write one sorted ordinary JSON object to stdout; diagnostic errors go to stderr; invalid input exits nonzero.

- [ ] **Step 7: Run focused validation**

```bash
uv run pytest tests/benchmark/test_reference_set_manifest.py \
  tests/tools/hpa324/test_analyze_reference_lanes.py -q
uv run ruff check src/benchmark/reference_set_manifest.py tools/hpa324 \
  tests/benchmark/test_reference_set_manifest.py tests/tools/hpa324
```

- [ ] **Step 8: Run the real-corpus audit before growing the ignore set**

Locate the current local HPA-323 manifest explicitly (for example from `reference-timing/latest.json`) and run the audit against that immutable manifest.

Review every `unmapped_lane_event_counts` entry. For an observed lane whose semantics are verified non-drum, add that exact lane to `IGNORED_NON_DRUM_LANES` and add a test explaining the policy. If semantics are uncertain, leave it unmapped; affected references quarantine in Task 5.

If no real HPA-323 artifact is locally available, keep `IGNORED_NON_DRUM_LANES` empty. Do not invent an ignore list from assumptions.

- [ ] **Step 9: Commit Task 4**

```bash
git add src/benchmark/reference_set_manifest.py tools/hpa324 tests/tools/hpa324 \
  tests/benchmark/test_reference_set_manifest.py src/benchmark/taxonomy.py tests/benchmark/test_taxonomy.py
git commit -m "feat: audit benchmark reference lanes"
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

ReferenceEligibilityStatus = Literal["eligible", "quarantined"]

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

A timing-quarantined input becomes HPA-324 quarantined with `upstream_reference_unavailable`, null mapped-event path, zero mapped/common counts, and unchanged upstream timing reasons.

- [ ] **Step 2: Write artifact-integrity tests**

For timing-ready rows, quarantine with `reference_event_artifact_invalid` when the artifact path is unsafe, file is missing, filename hash differs, or `read_reference_events` rejects content. These are row-local failures.

- [ ] **Step 3: Write binary-status eligibility tests**

Cover:

1. mapped events, no warnings -> `eligible`, warnings `[]`;
2. mapped events with ignored lanes -> `eligible`, warning array populated;
3. mapped events with exact common collapse -> `eligible`, warning array populated;
4. any unclassified lane -> `quarantined` with `unclassified_reference_lane` and no mapped artifact;
5. no mapped drum event after explicit ignores -> `quarantined` with `no_scored_drum_events`.

Use exact warning formats:

```text
ignored_reference_lane:<LANE>:count=<N>
duplicate_common_projection:count=<N>
```

Sort warnings lexicographically.

- [ ] **Step 4: Write accounting tests**

```python
assert eligible_count + quarantined_count == total_input_rows
assert mapped_event_artifact_count == eligible_count
```

No quarantines -> exit `0`; any quarantine with a published manifest -> exit `1`; fatal input/publication failure -> exit `2` and no manifest.

- [ ] **Step 5: Write publication tests**

For each eligible row:

1. render detailed mapped events;
2. hash the bytes;
3. publish `output_dir / "events" / f"{sha256}.jsonl"` through the existing immutable publisher;
4. set `benchmark_reference_events_path` to that relative path;
5. publish the derived manifest through `render_manifest`, `publish_manifest`, and `publish_latest_manifest`.

A repeated run with identical inputs must produce identical event/manifest bytes.

- [ ] **Step 6: Add the manifest schema golden**

Create one eligible-with-warnings row and one quarantined row. The eligible row still has `reference_eligibility_status="eligible"`.

The validator enforces the binary status set, closed reason codes, sorted warning strings, version fields, nullable path/count shapes, and row invariants.

- [ ] **Step 7: Run tests and verify RED**

```bash
uv run pytest tests/benchmark/test_reference_set_manifest.py tests/benchmark/test_schema_goldens.py -q
```

- [ ] **Step 8: Implement row evaluation**

```text
if timing_status != ready:
    quarantine upstream_reference_unavailable
else:
    verify/read persisted native event artifact
    result = map_reference_events(native_events)
    if result.diagnostics.unmapped:
        quarantine unclassified_reference_lane
    elif not result.mapped_events:
        quarantine no_scored_drum_events
    else:
        publish mapped event artifact
        status = eligible
        warnings = deterministic ignored/collapse diagnostics
```

Do not inspect model predictions or scores.

- [ ] **Step 9: Implement lossless upstream pass-through and immediate lineage**

Use the HPA-323 source row as the base, remove only top-level `corpus_version`, replace `schema_version`, add HPA-324 fields, and let `render_manifest` derive the new version.

Record:

```text
source_reference_timing_manifest_sha256 = exact input bytes SHA-256
source_reference_timing_version = HPA-323 input corpus_version
```

Preserve older HPA-321/HPA-322 lineage unchanged.

- [ ] **Step 10: Run focused validation**

```bash
uv run pytest tests/benchmark/test_reference_set_manifest.py tests/benchmark/test_schema_goldens.py -q
uv run ruff check src/benchmark/reference_set_manifest.py tests/benchmark/test_reference_set_manifest.py
```

- [ ] **Step 11: Commit Task 5**

```bash
git add src/benchmark/reference_set_manifest.py tests/benchmark/test_reference_set_manifest.py \
  tests/benchmark/schema_goldens/crux.benchmark-reference-manifest-v1.jsonl \
  tests/benchmark/schema_goldens/manifest.json
git commit -m "feat: classify benchmark reference eligibility"
```

---

## Task 6: Add offline `build-reference-set` CLI and acceptance

**Files:**
- Modify: `src/cli/benchmark.py`
- Modify: `tests/test_cli_benchmark.py`
- Create: `tests/benchmark/test_reference_set_acceptance.py`

**Interface:**

```bash
uv run crux benchmark build-reference-set \
  --manifest "$REFERENCE_TIMING_MANIFEST" \
  --output-dir artifacts/benchmark/reference-set
```

Machine-readable stdout fields:

```text
corpus_version
eligible_count
exit_code
manifest_path
manifest_sha256
mapped_event_artifact_count
quarantined_count
status
```

- [ ] **Step 1: Write CLI tests**

Cover required `--manifest`, default output dir, exits 0/1/2, one canonical JSON stdout object, and absence of R2/cache/model/tolerance/concurrency options.

- [ ] **Step 2: Write offline acceptance fixture**

Build one temporary HPA-323 manifest with:

- one ready kick/snare reference;
- one ready simultaneous high/low tom reference that remains `eligible` with a duplicate warning;
- one upstream-quarantined reference.

Require:

```python
assert outcome.eligible_count == 2
assert outcome.quarantined_count == 1
assert outcome.mapped_event_artifact_count == 2
assert outcome.exit_code == 1
```

Read both mapped artifacts and prove they retain native lane/note identity.

- [ ] **Step 3: Prove HPA-325 can reuse the public projection**

Reload a mapped artifact through `read_mapped_reference_events`, call `project_common_reference_events`, and require the common events equal those produced before persistence.

This is the anti-reimplementation gate for HPA-325.

- [ ] **Step 4: Add remap-without-inference acceptance**

Persist one HPA-323 native event artifact once. Map it under v1, then use a test-only alternate lane mapping/version to remap the same native bytes.

Require unchanged native input hash, changed mapped/version output, and no audio/model/transcription call.

- [ ] **Step 5: Run tests and verify RED**

```bash
uv run pytest tests/test_cli_benchmark.py -k build_reference_set -q
uv run pytest tests/benchmark/test_reference_set_acceptance.py -q
```

- [ ] **Step 6: Implement the command and summary**

Follow `_emit_reference_timing_summary` style with lazy imports. Do not add `--cache-dir`, R2, model, tolerance, or concurrency options.

- [ ] **Step 7: Run focused acceptance**

```bash
uv run pytest tests/benchmark/test_reference_set.py \
  tests/benchmark/test_reference_set_manifest.py \
  tests/benchmark/test_reference_set_acceptance.py -q
uv run pytest tests/test_cli_benchmark.py -q
```

- [ ] **Step 8: Commit Task 6**

```bash
git add src/cli/benchmark.py tests/test_cli_benchmark.py \
  tests/benchmark/test_reference_set_acceptance.py
git commit -m "feat: publish benchmark reference set"
```

---

## Task 7: Verify the HPA-423 prediction seam without redefining native identity

**Files:**
- Modify: `tests/benchmark/test_taxonomy.py`
- Modify: HPA-423 prediction mapping/scorer-input tests after their final paths are known from merged `main`.
- Modify production HPA-423 mapping code only if the merged implementation cannot consume the frozen OaF group map; keep any change a narrow seam adaptation.

There is deliberately no speculative production function signature here. Inspect merged HPA-423 first.

**Required semantics:**

- OaF lookup key comes from `native_metadata["upstream_8hit_group_id"]`;
- `native_class_id="midi_<note>"` remains unchanged;
- output bin and native MIDI note remain unchanged;
- mapping stamps `prediction_map_version == OAF_PREDICTION_MAP.map_id`;
- unknown/missing groups remain visible in diagnostics;
- `hihat` maps only to common `hihat`;
- `toms` maps only to common `tom`;
- `sticks` remains unmapped;
- `model_id` and `input_view_id` persist to scorer input.

- [ ] **Step 1: Rebase after HPA-423 lands**

```bash
git fetch origin
git rebase origin/main
```

Inspect the final HPA-423 mapper/tests before editing. Do not resurrect deleted seal/protocol files.

- [ ] **Step 2: Write real OaF-shaped seam fixtures**

Use events such as:

```python
NativeEvent(
    time_sec=0.5,
    native_class_id="midi_36",
    model_output_bin=15,
    native_midi_note=36,
    native_metadata={"upstream_8hit_group_id": "kick"},
    confidence=0.9,
    velocity_midi=100,
)
```

Add corresponding hihat/toms/ride_bell/sticks events with real MIDI-style `native_class_id` values. Do **not** construct `NativeEvent(native_class_id="kick")` merely to make the table convenient.

- [ ] **Step 3: Verify map consumption and native preservation**

Call the actual merged HPA-423 mapper with `OAF_PREDICTION_MAP` (or make the minimum seam adaptation required by its final API).

Require group-based class lookup, preserved MIDI/native identity, and stamped map ID.

- [ ] **Step 4: Verify scorer-input common events**

Using HPA-423's final scorer-input API, prove a mapped OaF prediction can produce common scorer events without an HPA-325 model-specific branch.

Do not add HPA-325 metrics or tolerance matching here.

- [ ] **Step 5: Run HPA-423 + HPA-324 focused tests**

```bash
uv run pytest tests/benchmark/test_taxonomy.py tests/benchmark/test_reference_set.py \
  tests/benchmark/test_reference_set_manifest.py -q
```

Also run the exact HPA-423 mapper/scorer-input test files discovered after rebase.

- [ ] **Step 6: Commit the seam verification**

Stage only HPA-324 taxonomy/tests plus any minimal HPA-423 seam adaptation and commit:

```bash
git commit -m "test: bind OaF group map to benchmark taxonomy"
```

If HPA-423 has not landed, do not invent its API. Leave Task 7 blocked until it does; HPA-324 map data remains independently testable.

---

## Task 8: Final acceptance and stale-policy cleanup

**Files:**
- All HPA-324 files above.

- [ ] **Step 1: Run benchmark/reference tests**

```bash
uv run pytest tests/benchmark -q
uv run pytest tests/test_cli_benchmark.py -q
uv run pytest tests/tools/hpa324 -q
```

- [ ] **Step 2: Run repository validation**

```bash
uv run pytest -q
uv run ruff check .
uv run ruff format --check src tests tools
uv run pylint --errors-only --disable=E1120,E0401 src
```

Any known repository baseline exception must be demonstrated unchanged from `main`; do not silently bless a new failure.

- [ ] **Step 3: Verify schema goldens and deterministic rerendering**

```bash
uv run pytest tests/benchmark/test_schema_goldens.py -q
```

Run one reference-set acceptance fixture twice and require identical event/manifest hashes.

- [ ] **Step 4: Search the full active blast radius for stale taxonomy policy**

```bash
git grep -n 'DEFAULT_DTX_LANE_MAP\|DtxClassMapping\|low_tom\|mid_tom\|REFERENCE_CLASS_TO_MIDI' -- src tests scripts
```

Resolve active old taxonomy owners/spellings. `REFERENCE_CLASS_TO_MIDI` itself is expected to remain, but its active keys must use the new taxonomy. Historical docs are outside this cleanup.

- [ ] **Step 5: Verify OaF identity was not rewritten**

```bash
git grep -n 'upstream_8hit_group_id\|native_class_id' -- src/benchmark runtime/oaf_tf1 tests/benchmark
```

Confirm HPA-324 did not turn group IDs into `native_class_id`. OaF group mapping must remain metadata-driven.

- [ ] **Step 6: Check scope**

```bash
git diff --stat origin/main...HEAD
git diff --check origin/main...HEAD
```

Confirm the implementation did not add guessed MuScriptor/IDM tables, velocity scoring, DTX/audio/R2 reprocessing, scoring tolerances, generic mapping frameworks, or result-driven mapping changes.

- [ ] **Step 7: Commit mechanical cleanup only when Steps 4-6 found concrete files to change**

Stage those concrete paths individually rather than using `git add -A`, then commit:

```bash
git commit -m "chore: finalize benchmark taxonomy contracts"
```

If Steps 4-6 find no cleanup, do not create an empty commit.

---

## HPA-324 Completion Gate

Before moving HPA-324 to Done, verify from committed code/tests/artifacts:

1. detailed/common taxonomy, total `DETAILED_TO_COMMON`, and DTX lane-map v1 are frozen in one module;
2. OaF has one immutable prediction-map ID keyed by `upstream_8hit_group_id`, while `native_class_id="midi_<note>"` remains native identity;
3. active legacy mapping/MIDI consumers use the new tom terminology without silent drops;
4. HPA-323 native artifacts are consumed directly and can be remapped without inference;
5. mapped benchmark-reference rows contain the exact HPA-323-derived identity + class/version fields and no per-row schema key;
6. unknown lanes are visible and only reviewed non-drum lanes are ignored;
7. `project_common_reference_events` is the single scorer-facing common projection and uses canonical decimal time identity;
8. every HPA-323 row is exactly `eligible` or `quarantined`; warnings are orthogonal to eligibility status;
9. reference eligibility is independent of model score and model-specific inference success;
10. after HPA-423 lands, its OaF mapping consumes group metadata, stamps the HPA-324 map ID, and preserves MIDI/native identity;
11. HPA-395/HPA-396 remain responsible for their exact MuScriptor/IDM maps; no guessed future vocabulary is added here.
