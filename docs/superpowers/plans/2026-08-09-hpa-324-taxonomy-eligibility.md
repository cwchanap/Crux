# HPA-324 Taxonomy and Reference Eligibility Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Freeze the benchmark drum taxonomy and mapping data, expose deterministic in-memory reference projection, audit the real HPA-323 corpus before freezing unknown-lane policy, publish a model-independent eligibility manifest, and ensure OaF predictions persist the common class HPA-325 will score.

**Architecture:** Keep taxonomy/mapping policy in one small `taxonomy.py`. Reuse HPA-323 persisted native reference events and validation. Map references in memory through one pure detailed mapper and one public common projection; do not publish a redundant mapped-event artifact. Publish only a derived eligibility manifest with version IDs and diagnostic counts. HPA-423 remains owner of prediction conversion, but its active prediction schema must persist `common_class` before HPA-326 is allowed to run.

**Tech Stack:** Python, frozen dataclasses, `typing.Literal`, `MappingProxyType`, `Decimal`, pathlib, existing strict canonical JSON/JSONL helpers, Click, pytest, Ruff, Pylint.

## Global Constraints

- HPA-322 and HPA-323 are complete; consume merged `main` contracts.
- HPA-423 is active in parallel. HPA-324 owns taxonomy/map policy; HPA-423 owns prediction mapping/serialization mechanics.
- Score onset time plus **common instrument class**. Detailed class is retained where justified for diagnosis; velocity/confidence are diagnostic only.
- `DETAILED_TO_COMMON` is total. Every map entry with a non-null detailed class must agree with `project_to_common`.
- OaF lookup uses `NativeEvent.native_metadata["upstream_8hit_group_id"]`; never rewrite `native_class_id="midi_<note>"`.
- Before HPA-326, the active prediction artifact must persist `common_class` for mapped events (`crux.drum-prediction-events/v2`).
- Preserve HPA-323 native reference artifacts as the only persisted event source. HPA-324 does **not** publish mapped-reference JSONL.
- Deduplicate only the common scorer projection, keyed by `(Decimal(str(audio_time_sec)), common_class)`.
- Do not use fuzzy time buckets. HPA-325 owns matching tolerances.
- Do not rerun chart selection, DTX parsing, timing, BGM resolution, R2 access, or audio decoding.
- Start `IGNORED_NON_DRUM_LANES` empty, but do not freeze eligibility policy until the real-corpus lane audit is committed and reviewed.
- Unknown lanes are never silently dropped. A lane is mapped, explicitly reviewed as non-drum, or quarantines the reference.
- Keep MuScriptor/IDM exact maps out of HPA-324. HPA-395/HPA-396 freeze them from their actual locked vocabularies.
- No generic plugin discovery, native-key selector framework, mapping DB, config DSL, service, queue, or concurrency layer.
- Breaking internal schemas is allowed. Do not add compatibility readers for prediction v1 merely to preserve old internal artifacts.

## Risks / Hard Gates

### Gate A — Prediction common class

Current `crux.drum-prediction-events/v1` has no `common_class`. HPA-326 must not begin until a real OaF-shaped hihat/tom prediction round-trips through the active prediction artifact with its common class intact.

### Gate B — Real-corpus lane policy

HPA-323 retains pattern channels outside 01/02/03/08. An unreviewed empty ignore set can therefore quarantine a large corpus while accounting still balances. The lane audit in Task 4 is a hard prerequisite for Task 5. If the real HPA-323 artifacts are unavailable, implementation stops at Task 4.

### Gate C — HPA-423 merge timing

Tasks 1-6 can be implemented independently. If HPA-423 is not merged when they finish, HPA-324 remains **In Progress**. Do not mark HPA-324 Done and do not start HPA-326 until Task 7 verifies the prediction schema/map seam.

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
- `tests/benchmark/schema_goldens/crux.benchmark-reference-manifest-v1.jsonl`
- `docs/superpowers/evidence/2026-08-09-hpa-324-reference-lane-audit.json`

### Modify

- `src/benchmark/mapping.py`
- `src/benchmark/midi_io.py`
- `src/benchmark/reference_timing.py`
- `src/benchmark/reference_timing_manifest.py`
- `src/benchmark/render_audio.py`
- `src/cli/benchmark.py`
- `tests/benchmark/test_mapping.py`
- `tests/benchmark/test_midi_io.py`
- `tests/benchmark/test_reference_timing.py`
- `tests/benchmark/test_reference_timing_manifest.py`
- `tests/benchmark/test_render_audio.py`
- `tests/test_cli_benchmark.py`
- `tests/benchmark/schema_goldens/manifest.json`
- `scripts/calibrate_egmd_mapping.py` only if the Task 1 grep finds an active old-class assumption

### HPA-423 integration files (Task 7, after rebase)

Current `main` seam:

- `src/benchmark/prediction_artifact.py`
- `src/benchmark/scorer_input.py`
- `tests/benchmark/test_prediction_artifact.py`
- `tests/benchmark/test_scorer_input.py`

If HPA-423 moves or replaces these files, use its merged equivalents; do not resurrect removed code.

---

## Task 1: Freeze taxonomy and migrate existing consumers

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

DETAILED_TO_COMMON: Mapping[DetailedDrumClass, CommonDrumClass]


def project_to_common(detailed: DetailedDrumClass) -> CommonDrumClass: ...


@dataclass(frozen=True)
class ClassMapping:
    canonical_class: DetailedDrumClass | None
    common_class: CommonDrumClass | None


@dataclass(frozen=True)
class PredictionMap:
    map_id: str
    backend_id: str
    native_output_space_id: str
    classes: Mapping[str, ClassMapping]

DTX_LANE_MAP: Mapping[str, ClassMapping]
DRUM_LANE_IDS: frozenset[str]
IGNORED_NON_DRUM_LANES: frozenset[str]
OAF_PREDICTION_MAP: PredictionMap
```

- [ ] **Step 1: Write the total projection test**

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

- [ ] **Step 2: Write exact DTX mapping tests**

Require:

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

For every non-null detailed entry:

```python
assert mapping.common_class == project_to_common(mapping.canonical_class)
```

- [ ] **Step 3: Bind the OaF map to the existing locked vocabulary**

In `tests/benchmark/test_taxonomy.py`, import the existing constants and require:

```python
from src.benchmark.backend_identity import OAF_BACKEND_ID
from src.benchmark.prediction_artifact import OAF_GROUP_IDS

assert set(OAF_PREDICTION_MAP.classes) == OAF_GROUP_IDS
assert OAF_PREDICTION_MAP.backend_id == OAF_BACKEND_ID
assert OAF_PREDICTION_MAP.classes["hihat"] == ClassMapping(None, "hihat")
assert OAF_PREDICTION_MAP.classes["toms"] == ClassMapping(None, "tom")
assert OAF_PREDICTION_MAP.classes["sticks"] == ClassMapping(None, None)
```

Keep this cross-module vocabulary assertion in tests so production `taxonomy.py` does not need to import `prediction_artifact.py`.

- [ ] **Step 4: Write reverse MIDI rename regressions**

In `tests/benchmark/test_midi_io.py`:

```python
assert REFERENCE_CLASS_TO_MIDI["low_or_floor_tom"] == 45
assert "low_tom" not in REFERENCE_CLASS_TO_MIDI
```

Add a `write_reference_midi` test proving `low_or_floor_tom` emits a tom note rather than the existing "Skipping unmapped canonical class" warning.

- [ ] **Step 5: Verify RED**

```bash
uv run pytest tests/benchmark/test_taxonomy.py tests/benchmark/test_mapping.py \
  tests/benchmark/test_midi_io.py -q
```

Expected: missing taxonomy/new-class behavior.

- [ ] **Step 6: Implement immutable taxonomy/maps**

Use `MappingProxyType` for `DETAILED_TO_COMMON`, `DTX_LANE_MAP`, and `OAF_PREDICTION_MAP.classes`. Start:

```python
IGNORED_NON_DRUM_LANES = frozenset()
```

Do not add a loader/config layer.

- [ ] **Step 7: Make `mapping.py` consume the single DTX map**

Delete `DtxClassMapping` / `DEFAULT_DTX_LANE_MAP` as policy owners. `map_dtx_events` keeps its legacy API but imports `DTX_LANE_MAP`.

Store the common projection in metadata:

```python
metadata = {
    **event.metadata,
    "lane_id": lane_id,
    "common_class": class_mapping.common_class,
}
```

Update the old `metadata["native_class"]` assertion accordingly.

Keep `DEFAULT_MIDI_NOTE_MAP` only for the legacy folder/MIDI scorer and make all emitted classes valid `DetailedDrumClass` values.

- [ ] **Step 8: Update render and reverse MIDI consumers**

`render_audio.py` imports `DRUM_LANE_IDS` for membership only.

Update `REFERENCE_CLASS_TO_MIDI` to the renamed detailed class. Before removing `mid_tom`, run:

```bash
git grep -n '"mid_tom"\|"low_tom"' -- src tests scripts
```

Remove `mid_tom` only if no active producer remains.

- [ ] **Step 9: Check calibration-script blast radius**

```bash
git grep -n 'DEFAULT_MIDI_NOTE_MAP\|map_dtx_events\|low_tom\|mid_tom' \
  -- scripts/calibrate_egmd_mapping.py
```

If the script only consumes `map_dtx_events`/`DEFAULT_MIDI_NOTE_MAP`, no direct edit is required. If it asserts an old class string, update that exact assumption in the same task.

- [ ] **Step 10: Run focused validation**

```bash
uv run pytest tests/benchmark/test_taxonomy.py tests/benchmark/test_mapping.py \
  tests/benchmark/test_midi_io.py tests/benchmark/test_render_audio.py -q
uv run ruff check src/benchmark/taxonomy.py src/benchmark/mapping.py \
  src/benchmark/midi_io.py src/benchmark/render_audio.py \
  tests/benchmark/test_taxonomy.py tests/benchmark/test_mapping.py \
  tests/benchmark/test_midi_io.py tests/benchmark/test_render_audio.py
```

- [ ] **Step 11: Commit**

```bash
git add src/benchmark/taxonomy.py src/benchmark/mapping.py src/benchmark/midi_io.py \
  src/benchmark/render_audio.py tests/benchmark/test_taxonomy.py \
  tests/benchmark/test_mapping.py tests/benchmark/test_midi_io.py \
  tests/benchmark/test_render_audio.py
git add scripts/calibrate_egmd_mapping.py 2>/dev/null || true
git commit -m "feat: freeze benchmark drum taxonomy"
```

---

## Task 2: Expose HPA-323 readers and reuse its manifest validation core

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


@dataclass(frozen=True)
class LoadedReferenceTimingRow:
    source_row: Mapping[str, object]
    view: ReferenceTimingRowView


@dataclass(frozen=True)
class LoadedReferenceTimingManifest:
    manifest_sha256: str
    corpus_version: str
    rows: tuple[LoadedReferenceTimingRow, ...]


def load_reference_timing_manifest(path: Path) -> LoadedReferenceTimingManifest: ...
```

- [ ] **Step 1: Write event-reader round-trip tests**

```python
content = render_reference_events((sample_native_reference_event(),))
assert render_reference_events(read_reference_events(content)) == content
```

Also reject non-canonical JSON, wrong keys, invalid time, and invalid sequence order/duplicates through the existing validators.

- [ ] **Step 2: Write timing-row view tests**

Use existing ready/quarantined row fixtures and prove the new view rejects anything `_validate_timing_status_shape` rejects.

- [ ] **Step 3: Write timing-manifest loader tests**

Cover:

- valid ready + quarantined rows;
- exact input SHA-256;
- one shared corpus version;
- duplicate simfile ID rejection;
- non-canonical/empty input rejection;
- byte-identical rerender requirement.

Also rerun existing `load_reference_chart_manifest` tests to prove the refactor does not relax HPA-322 validation.

- [ ] **Step 4: Verify RED**

```bash
uv run pytest tests/benchmark/test_reference_timing.py \
  tests/benchmark/test_reference_timing_manifest.py \
  -k "read_reference_events or timing_row_view or load_reference_timing" -q
```

- [ ] **Step 5: Implement `read_reference_events` by reusing existing validators**

Parse canonical rows, call `_validate_reference_event_row` and `_validate_reference_event_sequence`, construct `NativeReferenceEvent`, rerender, and require byte identity.

Do not add another key list.

- [ ] **Step 6: Extract only the canonical manifest-reading core**

Inside `reference_timing_manifest.py`, extract a private helper from the existing `load_reference_chart_manifest` for:

- file read;
- final-newline / nonblank canonical JSONL checks;
- per-row schema-version check supplied as an argument;
- strict canonical parse;
- exact-content SHA-256;
- byte-identical `render_manifest` verification.

Keep HPA-322-specific `ReferenceChartRowView` identity checks in `load_reference_chart_manifest`.

Do **not** add a generic outcome/invariant framework.

- [ ] **Step 7: Implement `ReferenceTimingRowView` and `load_reference_timing_manifest` in the source-owner module**

The loader uses the extracted canonical core, validates each row through `ReferenceTimingRowView`, rejects duplicate simfile IDs/mixed corpus versions, and retains immutable source rows for HPA-324 pass-through.

- [ ] **Step 8: Run focused + regression tests**

```bash
uv run pytest tests/benchmark/test_reference_timing.py \
  tests/benchmark/test_reference_timing_manifest.py -q
uv run ruff check src/benchmark/reference_timing.py src/benchmark/reference_timing_manifest.py \
  tests/benchmark/test_reference_timing.py tests/benchmark/test_reference_timing_manifest.py
```

- [ ] **Step 9: Commit**

```bash
git add src/benchmark/reference_timing.py src/benchmark/reference_timing_manifest.py \
  tests/benchmark/test_reference_timing.py tests/benchmark/test_reference_timing_manifest.py
git commit -m "feat: expose persisted reference timing readers"
```

---

## Task 3: Implement pure reference mapping and scorer projection

**Files:**
- Create: `src/benchmark/reference_set.py`
- Create: `tests/benchmark/test_reference_set.py`

**Interfaces:**

```python
@dataclass(frozen=True)
class MappedReferenceEvent:
    native: NativeReferenceEvent
    canonical_class: DetailedDrumClass
    common_class: CommonDrumClass


@dataclass(frozen=True)
class CommonReferenceEvent:
    canonical_audio_time: Decimal
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


def project_common_reference_events(
    mapped_events: tuple[MappedReferenceEvent, ...],
) -> tuple[CommonReferenceEvent, ...]: ...


def map_reference_events(
    events: tuple[NativeReferenceEvent, ...],
    *,
    lane_map: Mapping[str, ClassMapping] = DTX_LANE_MAP,
    ignored_lanes: frozenset[str] = IGNORED_NON_DRUM_LANES,
) -> ReferenceMappingResult: ...
```

- [ ] **Step 1: Write identity-preservation tests**

For lanes 13, 18, 14/15, 16, and 19, require the mapped event retains the exact `NativeReferenceEvent` and produces the expected detailed/common class.

- [ ] **Step 2: Write unknown/ignored tests**

Use explicit test inputs:

- unknown lane -> `diagnostics.unmapped`, no mapped event;
- explicit `ignored_lanes=frozenset({"2A"})` -> `diagnostics.ignored`, no mapped event;
- source tuple remains unchanged.

- [ ] **Step 3: Write durable exact-collapse tests**

```python
mapped = (
    mapped_event("14", 1.0),
    mapped_event("15", 1.0),
    mapped_event("15", 1.001),
)

common = project_common_reference_events(mapped)

assert [(e.canonical_audio_time, e.common_class) for e in common] == [
    (Decimal("1.0"), "tom"),
    (Decimal("1.001"), "tom"),
]
assert len(common[0].source_events) == 2
```

Also prove open/closed hi-hat collapses only at the exact same durable time.

- [ ] **Step 4: Verify RED**

```bash
uv run pytest tests/benchmark/test_reference_set.py -q
```

- [ ] **Step 5: Implement the public common projection first**

Key groups by:

```python
(Decimal(str(event.native.audio_time_sec)), event.common_class)
```

Sort by `(canonical_audio_time, common_class)`.

- [ ] **Step 6: Implement `map_reference_events` by composing the projection**

Map lanes through `lane_map`, report ignored/unmapped counts, then call `project_common_reference_events(mapped_events)`. Do not duplicate collapse logic inside `map_reference_events`.

`duplicate_common_event_count` is:

```python
sum(len(event.source_events) - 1 for event in common_events)
```

- [ ] **Step 7: Run focused validation**

```bash
uv run pytest tests/benchmark/test_reference_set.py -q
uv run ruff check src/benchmark/reference_set.py tests/benchmark/test_reference_set.py
```

- [ ] **Step 8: Commit**

```bash
git add src/benchmark/reference_set.py tests/benchmark/test_reference_set.py
git commit -m "feat: map benchmark references in memory"
```

---

## Task 4: Run and commit the real-corpus lane audit — HARD GATE

**Files:**
- Create: `tools/hpa324/analyze_reference_lanes.py`
- Create: `tests/tools/hpa324/test_analyze_reference_lanes.py`
- Create: `docs/superpowers/evidence/2026-08-09-hpa-324-reference-lane-audit.json`
- Modify: `src/benchmark/taxonomy.py` only after evidence review
- Modify: `tests/benchmark/test_taxonomy.py` only after evidence review

**Audit output:**

```text
source_reference_timing_manifest_sha256
source_reference_timing_version
row_count
ready_row_count
lane_event_counts
unmapped_lane_event_counts
unmapped_lane_simfile_counts
common_collision_count
common_collision_simfile_count
ignored_non_drum_lanes
prospective_eligible_row_count
prospective_quarantined_row_count
```

- [ ] **Step 1: Write audit fixture tests**

Use a temporary HPA-323 manifest with mapped lanes, one synthetic unknown lane, and simultaneous tom lanes. Assert lane counts, unknown visibility, collision counts, and prospective eligibility/quarantine counts.

- [ ] **Step 2: Verify RED**

```bash
uv run pytest tests/tools/hpa324/test_analyze_reference_lanes.py -q
```

- [ ] **Step 3: Implement the audit using production readers/functions only**

For timing-ready rows:

1. resolve inherited `reference_events_cache_path` beneath the timing artifact root;
2. verify the content-addressed filename hash;
3. call `read_reference_events`;
4. call `map_reference_events`;
5. aggregate lane/unmapped/ignored/collision and prospective row-status counts.

Do not parse DTX or implement a second mapper.

- [ ] **Step 4: Run focused validation**

```bash
uv run pytest tests/tools/hpa324/test_analyze_reference_lanes.py -q
uv run ruff check tools/hpa324 tests/tools/hpa324
```

- [ ] **Step 5: Locate and run against the real HPA-323 manifest**

Use the actual immutable reference-timing manifest selected by the local `reference-timing/latest.json` pointer or the known operator artifact path. Run:

```bash
uv run python -m tools.hpa324.analyze_reference_lanes \
  --manifest "$REFERENCE_TIMING_MANIFEST" \
  > docs/superpowers/evidence/2026-08-09-hpa-324-reference-lane-audit.json
```

If the real manifest/artifacts are unavailable, **stop here**. Do not proceed to Task 5.

- [ ] **Step 6: Review every observed unmapped lane**

For each observed unmapped lane, verify whether it is non-drum from authoritative DTX semantics/source evidence. Add only verified non-drum lane IDs to `IGNORED_NON_DRUM_LANES`, with a test asserting the reviewed policy.

Do not infer "non-drum" from rarity or from being outside the current map.

- [ ] **Step 7: Rerun the audit after policy changes**

Overwrite the evidence file with the final reviewed policy reflected in `ignored_non_drum_lanes`.

Require:

```text
ready_row_count = prospective_eligible_row_count + prospective_quarantined_row_count
prospective_eligible_row_count > 0
```

If prospective eligible count is zero, stop and revisit lane policy before Task 5.

- [ ] **Step 8: Commit the reviewed evidence and policy**

```bash
git add tools/hpa324 tests/tools/hpa324 \
  docs/superpowers/evidence/2026-08-09-hpa-324-reference-lane-audit.json \
  src/benchmark/taxonomy.py tests/benchmark/test_taxonomy.py
git commit -m "feat: audit benchmark reference lanes"
```

---

## Task 5: Publish the model-independent eligibility manifest

**Files:**
- Create: `src/benchmark/reference_set_manifest.py`
- Create: `tests/benchmark/test_reference_set_manifest.py`
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
```

Added row fields:

```text
schema_version = crux.benchmark-reference-manifest/v1
source_reference_timing_manifest_sha256
source_reference_timing_version
taxonomy_version
lane_map_version
reference_eligibility_status
reference_eligibility_reason_codes
reference_eligibility_warnings
mapped_event_count
common_scored_event_count
ignored_event_count
unmapped_event_count
duplicate_common_event_count
```

There is no mapped-event artifact path/count. The inherited HPA-323 `reference_events_cache_path` remains the source HPA-325 reads.

- [ ] **Step 1: Write upstream quarantine tests**

A timing-quarantined row becomes HPA-324 quarantined with `upstream_reference_unavailable`, zero mapping counts, and unchanged upstream timing reasons.

- [ ] **Step 2: Write artifact-integrity tests**

For timing-ready rows, quarantine with `reference_event_artifact_invalid` when the inherited event path is unsafe/missing/hash-mismatched or `read_reference_events` rejects its bytes.

These are row-local failures.

- [ ] **Step 3: Write mapping/status tests**

Cover:

1. mapped events, no warnings -> `eligible`;
2. mapped events plus reviewed ignored lanes -> `eligible` + deterministic warning;
3. exact common collapse -> `eligible` + duplicate warning;
4. unclassified lane -> `quarantined` with `unclassified_reference_lane`;
5. no mapped events after reviewed ignores -> `quarantined` with `no_scored_drum_events`.

Warnings:

```text
ignored_reference_lane:<LANE>:count=<N>
duplicate_common_projection:count=<N>
```

- [ ] **Step 4: Tie realistic non-drum acceptance to committed evidence**

Read `docs/superpowers/evidence/2026-08-09-hpa-324-reference-lane-audit.json` in the test. For every lane listed in `ignored_non_drum_lanes`, build a reference event on that lane and assert the row stays eligible (assuming at least one mapped drum event is also present) and reports the ignored-lane warning.

This prevents the implementation from committing evidence/policy that the eligibility path does not honor.

- [ ] **Step 5: Write accounting tests**

```python
assert eligible_count + quarantined_count == total_input_rows
```

No quarantines -> exit `0`; any quarantine -> exit `1`; fatal manifest/publication failure -> exit `2` with no manifest.

- [ ] **Step 6: Add the manifest schema golden**

Create one eligible-with-warnings row and one quarantined row. The eligible row still has `reference_eligibility_status="eligible"`.

Register only `crux.benchmark-reference-manifest/v1`; there is no mapped-event golden.

- [ ] **Step 7: Verify RED**

```bash
uv run pytest tests/benchmark/test_reference_set_manifest.py \
  tests/benchmark/test_schema_goldens.py -q
```

- [ ] **Step 8: Implement row evaluation**

For each loaded HPA-323 row:

```text
if timing_status != ready:
    quarantine upstream_reference_unavailable
else:
    verify/read inherited native reference artifact
    result = map_reference_events(native_events)
    if result.diagnostics.unmapped:
        quarantine unclassified_reference_lane
    elif not result.mapped_events:
        quarantine no_scored_drum_events
    else:
        eligible
        record deterministic ignored/collapse warnings + counts
```

Do not publish event bytes.

- [ ] **Step 9: Implement explicit row derivation**

Copy the validated HPA-323 source row, remove only top-level `corpus_version`, replace `schema_version`, add the HPA-324 fields above, then let `render_manifest` derive the new corpus version.

Record immediate lineage:

```text
source_reference_timing_manifest_sha256 = exact input bytes SHA-256
source_reference_timing_version = HPA-323 input corpus_version
```

Do not generalize `build_timing_row()` into a callback framework; its timing fields/invariants are domain-specific.

- [ ] **Step 10: Implement explicit outcome accounting**

Keep the 0/1/2 convention identical to HPA-323, but keep HPA-324's two-count invariant explicit rather than adding a generic invariant-list helper.

- [ ] **Step 11: Run focused validation**

```bash
uv run pytest tests/benchmark/test_reference_set_manifest.py \
  tests/benchmark/test_schema_goldens.py -q
uv run ruff check src/benchmark/reference_set_manifest.py \
  tests/benchmark/test_reference_set_manifest.py
```

- [ ] **Step 12: Commit**

```bash
git add src/benchmark/reference_set_manifest.py \
  tests/benchmark/test_reference_set_manifest.py \
  tests/benchmark/schema_goldens/crux.benchmark-reference-manifest-v1.jsonl \
  tests/benchmark/schema_goldens/manifest.json
git commit -m "feat: classify benchmark reference eligibility"
```

---

## Task 6: Add offline CLI and prove HPA-325 reconstruction

**Files:**
- Modify: `src/cli/benchmark.py`
- Modify: `tests/test_cli_benchmark.py`
- Create: `tests/benchmark/test_reference_set_acceptance.py`

**CLI:**

```bash
uv run crux benchmark build-reference-set \
  --manifest "$REFERENCE_TIMING_MANIFEST" \
  --output-dir artifacts/benchmark/reference-set
```

Machine-readable stdout:

```text
corpus_version
eligible_count
exit_code
manifest_path
manifest_sha256
quarantined_count
status
```

- [ ] **Step 1: Write CLI tests**

Cover required `--manifest`, default output dir, exits 0/1/2, one canonical JSON stdout object, and absence of cache/R2/model/tolerance/concurrency options.

- [ ] **Step 2: Write offline acceptance fixture**

Build a temporary HPA-323 manifest with:

- one ready kick/snare reference;
- one ready simultaneous high/low tom reference producing a duplicate warning;
- one upstream-quarantined reference.

Require:

```python
assert outcome.eligible_count == 2
assert outcome.quarantined_count == 1
assert outcome.exit_code == 1
```

- [ ] **Step 3: Prove no mapped artifact exists or is needed**

For an eligible published HPA-324 row:

1. read its inherited `reference_events_cache_path`;
2. call `read_reference_events`;
3. call `map_reference_events`;
4. call `project_common_reference_events`;
5. require event counts equal the manifest's `mapped_event_count` / `common_scored_event_count`.

This is the HPA-325 reconstruction contract.

- [ ] **Step 4: Prove remapping needs no inference**

Persist one native HPA-323 event artifact once. Map it under DTX map v1, then call `map_reference_events` with a test-only alternate lane map. Require unchanged native bytes/hash and changed mapped result without audio/model/transcription calls.

- [ ] **Step 5: Verify RED**

```bash
uv run pytest tests/test_cli_benchmark.py -k build_reference_set -q
uv run pytest tests/benchmark/test_reference_set_acceptance.py -q
```

- [ ] **Step 6: Implement CLI command/summary**

Follow `_emit_reference_timing_summary` style with lazy imports. No additional infrastructure.

- [ ] **Step 7: Run focused acceptance**

```bash
uv run pytest tests/benchmark/test_reference_set.py \
  tests/benchmark/test_reference_set_manifest.py \
  tests/benchmark/test_reference_set_acceptance.py -q
uv run pytest tests/test_cli_benchmark.py -q
```

- [ ] **Step 8: Commit**

```bash
git add src/cli/benchmark.py tests/test_cli_benchmark.py \
  tests/benchmark/test_reference_set_acceptance.py
git commit -m "feat: publish benchmark reference eligibility"
```

---

## Task 7: Integrate the HPA-423 prediction schema/map seam — BLOCKING DONE GATE

This task runs only after rebasing onto the merged HPA-423 implementation. Tasks 1-6 may be merged first, but HPA-324 stays In Progress until this task passes.

**Current seam files:**
- `src/benchmark/prediction_artifact.py`
- `src/benchmark/scorer_input.py`
- `tests/benchmark/test_prediction_artifact.py`
- `tests/benchmark/test_scorer_input.py`
- `tests/benchmark/test_taxonomy.py`

- [ ] **Step 1: Rebase onto merged HPA-423**

```bash
git fetch origin
git rebase origin/main
```

Inspect HPA-423's final persistence/mapping API. Do not restore files it intentionally deleted.

- [ ] **Step 2: Require prediction artifact v2**

The active schema must expose both `canonical_class` and `common_class`:

```text
PREDICTION_SCHEMA = crux.drum-prediction-events/v2
```

Mapped event invariant:

```text
prediction_map_version != null
common_class != null
canonical_class may be null
mapping_status = mapped
```

Unmapped event invariant:

```text
prediction_map_version != null
common_class = null
canonical_class = null
mapping_status = unmapped
```

No v1 compatibility reader is required.

- [ ] **Step 3: Write real OaF-shaped round-trip tests**

Use events retaining native identity, for example:

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

Add hihat, toms, ride_bell, and sticks examples with their real MIDI-style native IDs.

Require:

- group metadata drives `OAF_PREDICTION_MAP` lookup;
- native ID/bin/MIDI/confidence/velocity survive serialization;
- hihat round-trips with `canonical_class=None`, `common_class="hihat"`;
- toms round-trips with `canonical_class=None`, `common_class="tom"`;
- sticks round-trips as unmapped with both classes null;
- every attempted mapping carries `crux.prediction-map/oaf-egmd-8hit-v1`.

- [ ] **Step 4: Verify scorer input consumes persisted common class**

`read_scorer_events` (or HPA-423's merged replacement) must read the persisted common class without a model-specific OaF branch.

Do not make HPA-325 infer common class from `canonical_class`; hihat/toms intentionally have no detailed class.

- [ ] **Step 5: Run HPA-423/HPA-324 seam tests**

```bash
uv run pytest tests/benchmark/test_prediction_artifact.py \
  tests/benchmark/test_scorer_input.py tests/benchmark/test_taxonomy.py -q
```

Also run any HPA-423 mapper test file introduced by the merge.

- [ ] **Step 6: Commit only if HPA-423 did not already satisfy the contract**

If HPA-423 already implements v2/common-class persistence, this step is test-only. Otherwise make the smallest breaking schema update required; do not add compatibility code.

```bash
git add src/benchmark/prediction_artifact.py src/benchmark/scorer_input.py \
  tests/benchmark/test_prediction_artifact.py tests/benchmark/test_scorer_input.py \
  tests/benchmark/test_taxonomy.py
git commit -m "feat: persist common prediction classes"
```

---

## Task 8: Final verification and completion gate

- [ ] **Step 1: Run HPA-324 focused tests**

```bash
uv run pytest tests/benchmark/test_taxonomy.py tests/benchmark/test_mapping.py \
  tests/benchmark/test_midi_io.py tests/benchmark/test_reference_timing.py \
  tests/benchmark/test_reference_timing_manifest.py tests/benchmark/test_reference_set.py \
  tests/benchmark/test_reference_set_manifest.py \
  tests/benchmark/test_reference_set_acceptance.py \
  tests/tools/hpa324/test_analyze_reference_lanes.py \
  tests/test_cli_benchmark.py -q
```

- [ ] **Step 2: Run repository-defined gates**

```bash
uv run pytest
uv run ruff check src tests
uv run ruff format --check src tests
uv run pylint src/app src/cli
```

These are the repository gates from `CLAUDE.md`.

For HPA-324's new tool files additionally run:

```bash
uv run ruff check tools/hpa324 tests/tools/hpa324
uv run ruff format --check tools/hpa324 tests/tools/hpa324
```

A broader `uv run pylint src` may be diagnostic, but it is not the completion gate.

- [ ] **Step 3: Verify schema goldens**

```bash
uv run pytest tests/benchmark/test_schema_goldens.py -q
```

Only the benchmark-reference **manifest** golden is new. No mapped-reference-event golden should exist.

- [ ] **Step 4: Verify stale policy is gone**

```bash
git grep -n "DEFAULT_DTX_LANE_MAP\|DtxClassMapping\|\"low_tom\"\|\"mid_tom\"" -- src tests scripts
```

Resolve active policy duplicates. Historical docs may retain old terminology.

- [ ] **Step 5: Verify audit evidence and eligibility are consistent**

Require the committed lane-audit file exists and its prospective counts reconcile. Re-run the audit against the same source manifest if the source artifact is still available and compare the normalized JSON output to the committed evidence.

- [ ] **Step 6: Verify HPA-326 blocker**

Before moving HPA-324 to Done, confirm Task 7 passed: prediction artifact v2 persists `common_class`, OaF group mapping preserves native identity, and scorer input reads common classes.

If HPA-423 is still unmerged, **stop here and leave HPA-324 In Progress**.

- [ ] **Step 7: Scope check**

```bash
git diff --stat origin/main...HEAD
git diff --check origin/main...HEAD
```

Confirm the implementation did not add:

- mapped-reference event persistence;
- MuScriptor/IDM guessed tables;
- velocity scoring;
- DTX/audio/R2 reprocessing;
- scoring metrics/tolerances;
- plugin/config/key-selector frameworks;
- result-driven map changes.

---

## HPA-324 Completion Gate

HPA-324 is Done only when all are true:

1. detailed/common taxonomy and total projection are frozen in one module;
2. DTX lane map v1 and OaF group map v1 are versioned and immutable;
3. OaF map keys are test-bound to `OAF_GROUP_IDS` and backend ID to `OAF_BACKEND_ID`;
4. HPA-323 native events are consumed through thin validated readers;
5. reference mapping/common projection is pure and persisted mapped copies are not added;
6. real-corpus lane audit evidence is committed and reviewed before eligibility policy;
7. unknown/ignored/collision diagnostics are visible and every row is exactly eligible or quarantined;
8. HPA-325 can reconstruct common reference events from inherited native artifacts + version IDs;
9. active prediction artifacts persist `common_class` and OaF hihat/toms round-trip scoreably;
10. HPA-423 consumes the OaF group map without rewriting native MIDI identity;
11. HPA-395/HPA-396 retain ownership of their exact future maps;
12. HPA-326 has not started before items 9-10 pass.
