# HPA-324: Versioned Drum Taxonomy, Mapping, and Eligibility Design

## Context

HPA-324 is the next unstarted Crux benchmark task that is ready.

- HPA-322 is complete and publishes authoritative chart selection.
- HPA-323 is complete and publishes audio-relative native reference events plus `crux.reference-timing-manifest/v1`.
- HPA-324 is blocked only by HPA-322 and HPA-323, so both explicit blockers are satisfied.
- HPA-324 blocks HPA-325 scoring, HPA-326 OaF corpus inference, and HPA-327 reviewed-subset work.
- HPA-423 is already in progress and explicitly owns the `NativeEvent -> BenchmarkEvent` prediction-mapping mechanism; HPA-324 owns the taxonomy and mapping data that mechanism consumes.

The merged HPA-323 boundary is already sufficient:

- `reference_timing.py` publishes lossless `crux.dtx-reference-event/v1` JSONL with native DTX lane/note identity and audio-relative times;
- `reference_timing_manifest.py` classifies rows as timing `ready` or `quarantined` and preserves HPA-322/HPA-321 lineage;
- `mapping.py` still contains an unversioned legacy DTX/MIDI map;
- `prediction_artifact.py` reserves canonical mapping fields but persists OaF native identity separately from its 8-hit group identity;
- `scorer_input.py` remains intentionally blocked until HPA-423 supplies canonical prediction mapping.

HPA-324 builds on these seams. It does not add another corpus pipeline or model framework.

## Requirement Clarification: Benchmark Target

The Linear addendum is normative: the initial benchmark scores **onset time plus drum class**.

DTX `#WAVxx`, `#VOLUMExx`, sample choice, and chip identity are not trustworthy per-hit velocity ground truth. Therefore:

- predicted velocity/confidence remains diagnostic metadata only;
- DTX sample/note identity remains diagnostic metadata only;
- velocity is not part of reference eligibility or headline scoring;
- a future velocity metric requires a separately reviewed reference and metric version.

## Design Choice

Use a **small code-defined versioned taxonomy and mapping-data layer**, plus one derived reference/eligibility stage.

Do not add YAML/JSON configuration frameworks, plugin discovery, databases, a mapping DSL, or model-registration infrastructure. These mappings are small reviewed benchmark policy. Python dataclasses/constants plus stable version IDs are the smallest maintainable representation.

Three approaches were considered:

1. **Selected: typed mapping data in Python.** Smallest surface, direct tests, stable version IDs, no runtime parser.
2. External mapping JSON files. More editable, but adds schema/loading/publication machinery without a current need.
3. Keep ad-hoc dictionaries/functions in `mapping.py`. Small today, but cannot cleanly express stable per-model identity or detailed-vs-common projection.

## Key Modeling Decision: Detailed vs Common Class

One flat class label cannot honestly satisfy the benchmark requirements.

The authored DTX can distinguish open/closed hi-hat and high/low tom lanes. Frozen OaF does not: its current 8-hit grouping contains one `hihat` group and one `toms` group. Mapping those outputs directly to `closed_hihat` or `high_tom` would fabricate distinctions the model never produced.

HPA-324 therefore defines two explicit class levels.

### Detailed canonical class

```text
kick
snare
closed_hihat
open_hihat
crash
ride
high_tom
low_or_floor_tom
```

### Common comparison class

```text
kick
snare
hihat
crash
ride
tom
```

Detailed classes project through one frozen table:

```text
kick              -> kick
snare             -> snare
closed_hihat      -> hihat
open_hihat        -> hihat
crash             -> crash
ride              -> ride
high_tom          -> tom
low_or_floor_tom  -> tom
```

The projection is code, not prose-only policy:

```python
DETAILED_TO_COMMON: Mapping[DetailedDrumClass, CommonDrumClass]


def project_to_common(detailed: DetailedDrumClass) -> CommonDrumClass: ...
```

Every detailed class must appear exactly once in `DETAILED_TO_COMMON`. Any `ClassMapping` with a non-null detailed class must use the corresponding common class. This invariant is tested for every frozen DTX/OaF entry and is reused by later MuScriptor/IDM maps.

A model prediction may have `canonical_class=None` while still having a valid `common_class` when its native output cannot justify the detailed distinction. OaF `hihat` and `toms` are the primary examples.

## Versioned Mapping Contracts

Create `src/benchmark/taxonomy.py` as the single policy/data owner.

```python
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal

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
    model_id: str
    native_output_space_id: str
    classes: Mapping[str, ClassMapping]
```

Use immutable mappings (`MappingProxyType`) for the frozen tables. Changing mapping semantics requires a new ID. Git history is sufficient archival storage for this hobby project; no signed registry or mapping database is needed.

### DTX lane map v1

Freeze the current intended drum lanes, replacing the legacy `low_tom` spelling with the ticket terminology:

```text
11 -> closed_hihat / hihat
12 -> snare / snare
13 -> kick / kick
14 -> high_tom / tom
15 -> low_or_floor_tom / tom
16 -> crash / crash
17 -> low_or_floor_tom / tom
18 -> open_hihat / hihat
19 -> ride / ride
1A -> crash / crash
1B -> closed_hihat / hihat
1C -> kick / kick
```

`lane_id` and `note_id` remain preserved on every mapped reference event.

The taxonomy rename must update legacy reverse MIDI export in `midi_io.py` at the same time. `REFERENCE_CLASS_TO_MIDI` must accept `low_or_floor_tom`; no active code may silently skip the renamed tom class. The legacy `mid_tom` key should be removed unless an active producer still emits it after the taxonomy migration.

### OaF prediction map v1: preserve native identity

Do **not** redefine OaF `NativeEvent.native_class_id`.

The existing OaF contract intentionally uses identities such as `native_class_id="midi_36"`, while the frozen 8-hit semantic group is carried separately in:

```python
event.native_metadata["upstream_8hit_group_id"]
```

HPA-324 freezes `OAF_PREDICTION_MAP.classes` by the eight **group IDs**, not by `native_class_id`:

```text
kick      -> kick / kick
snare     -> snare / snare
hihat     -> None / hihat
toms      -> None / tom
ride      -> ride / ride
ride_bell -> ride / ride
crash     -> crash / crash
sticks    -> None / None
```

HPA-423 must look up OaF mapping data using the group metadata:

```python
key = event.native_metadata["upstream_8hit_group_id"]
mapping = OAF_PREDICTION_MAP.classes.get(key)
```

`native_class_id`, `model_output_bin`, `native_midi_note`, confidence, velocity, and all native metadata remain diagnostic identity. HPA-324 must not force HPA-423 to normalize `kick`, `hihat`, or `toms` into `native_class_id`.

`sticks` stays visible as unmapped native coverage; it is not silently converted to snare.

This is deliberately OaF-specific and small. HPA-324 does not invent a generic key-selector framework for future models.

### MuScriptor and IDM ownership

Do **not** guess exact MuScriptor or IDM native class tables inside HPA-324.

HPA-395 must observe the exact locked MuScriptor drum pitch vocabulary and freeze its map before scored corpus results. HPA-396 must read the exact loaded IDM `train_classes` ordering and freeze its map before its pilot.

HPA-324 defines the `PredictionMap`, class projection, and versioning contract they must obey. HPA-395/HPA-396 decide how their own adapters extract the lookup key from their concrete native event shapes. This avoids a premature generic selector layer.

## Reference Mapping Pipeline

HPA-323 already performed chart selection, DTX parsing, timing, source-audio resolution/decoding, hash validation, BGM anchoring, and bounds checks.

HPA-324 must **not repeat those checks**. A timing `ready` row is the upstream technical-integrity gate.

Expose only the missing read APIs over HPA-323's existing validators:

```python
def read_reference_events(content: bytes) -> tuple[NativeReferenceEvent, ...]: ...


@dataclass(frozen=True)
class ReferenceTimingRowView:
    ...
```

Then map each native reference event to:

```python
@dataclass(frozen=True)
class MappedReferenceEvent:
    native: NativeReferenceEvent
    canonical_class: DetailedDrumClass
    common_class: CommonDrumClass
```

Publish the detailed mapped events as content-addressed canonical JSONL under the stable schema identity:

```text
crux.benchmark-reference-event/v1
```

Like HPA-323's `crux.dtx-reference-event/v1`, **event rows do not carry a per-row `schema` field**. The schema identity is the validator/golden registration plus the manifest contract.

Each mapped row contains:

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
lane_map_version
taxonomy_version
```

The native HPA-323 event artifact remains authoritative and immutable; HPA-324's event artifact is a derived scoring view.

## Unknown/Non-Drum Lane Policy

Do not silently treat every unmapped pattern channel as irrelevant.

The implementation first adds a committed real-corpus diagnostic that enumerates:

- every observed `lane_id` and event count;
- mapped vs unmapped lane counts;
- simfiles containing each unmapped lane;
- simultaneous events that collide after common-class projection.

Use production HPA-323 readers and HPA-324 mapping functions only. Do not build a parallel throwaway parser.

Start:

```python
IGNORED_NON_DRUM_LANES = frozenset()
```

After reviewing the real corpus, add only observed lanes whose non-drum meaning is verified. Any other unmapped lane quarantines the row rather than disappearing silently. Corpus yield is not a classification criterion.

## Common Scoring Projection

Preserve all native/mapped event rows. Deduplicate only the **common scorer projection**.

This projection is one public pure function, owned by `reference_set.py` and reused by eligibility now and HPA-325 later:

```python
def project_common_reference_events(
    mapped_events: tuple[MappedReferenceEvent, ...],
) -> tuple[CommonReferenceEvent, ...]: ...
```

`map_reference_events()` may call this helper; it must not embed a second private collapse implementation.

### Exact identity uses canonical durable time

The common-scoring identity is:

```text
(canonical_audio_time, common_class)
```

where `canonical_audio_time` is the same decimal representation used for durable event rendering, e.g.:

```python
Decimal(str(event.native.audio_time_sec))
```

Do not key a durable exact identity directly on raw binary-float object identity. Do not round or bucket timestamps. HPA-325 owns tolerance matching.

If multiple mapped events collapse to the same identity, the scorer projection contains one common event and retains every source event for diagnostics.

Examples:

- simultaneous high/low toms -> one common `tom` event;
- simultaneous open/closed hi-hat -> one common `hihat` event;
- events even slightly apart in canonical time remain separate.

## Reference Eligibility

Publish one derived manifest:

```text
crux.benchmark-reference-manifest/v1
```

It consumes `crux.reference-timing-manifest/v1` and its referenced native event artifacts. All upstream fields pass through verbatim except the top-level derived `corpus_version`.

### Status

Use the same binary availability shape established by HPA-323:

```text
eligible
quarantined
```

Warnings are orthogonal data, not a third status. A row with warnings remains `eligible` and carries deterministic `reference_eligibility_warnings`.

### Closed reason codes

```python
EligibilityReasonCode = Literal[
    "upstream_reference_unavailable",
    "reference_event_artifact_invalid",
    "unclassified_reference_lane",
    "no_scored_drum_events",
]
```

Meaning:

- `upstream_reference_unavailable`: HPA-323 timing status was not `ready`;
- `reference_event_artifact_invalid`: referenced native JSONL is missing, hash-mismatched, non-canonical, or schema-invalid;
- `unclassified_reference_lane`: an event lane is neither mapped nor explicitly ignored;
- `no_scored_drum_events`: after explicit non-drum exclusions, no mapped drum event remains.

Do not duplicate HPA-323 timing reason codes into this list; the carried-through row already contains them.

Deterministic warning formats:

```text
ignored_reference_lane:<LANE>:count=<N>
duplicate_common_projection:count=<N>
```

### Accounting

The outcome enforces:

```text
inventoried = eligible + quarantined
```

Every `eligible` row publishes one mapped-event artifact whether or not warnings are present.

Low future model scores, simplified charts, and model-specific inference failures never change reference eligibility.

## Prediction Mapping Seam with HPA-423

HPA-423 owns prediction conversion mechanics. HPA-324 owns the policy data.

After HPA-423 lands, the seam verification must use **real OaF-shaped native events**, for example:

```python
NativeEvent(
    time_sec=...,
    native_class_id="midi_36",
    model_output_bin=15,
    native_midi_note=36,
    native_metadata={"upstream_8hit_group_id": "kick"},
    confidence=...,
    velocity_midi=...,
)
```

The test proves HPA-423 consumes `OAF_PREDICTION_MAP` by group metadata, stamps `prediction_map_version == OAF_PREDICTION_MAP.map_id`, preserves the MIDI/native identities, and exposes unmapped `sticks`.

Do not define a speculative HPA-423 production function signature in HPA-324. Rebase first, inspect the actual merged API, and add only the minimum seam adaptation needed. If HPA-423 has not landed, correct OaF map data plus taxonomy tests are sufficient for the HPA-324 planning boundary; seam verification waits.

## CLI

Add one derived-stage command:

```bash
uv run crux benchmark build-reference-set \
  --manifest artifacts/benchmark/reference-timing/manifests/<manifest-sha>.jsonl \
  --output-dir artifacts/benchmark/reference-set
```

`--manifest` is required. `--output-dir` defaults to `artifacts/benchmark/reference-set`.

No R2 credentials are needed. Every source/chart/audio decision was already frozen by HPA-321/322/323.

Exit codes follow the existing derived-manifest convention:

- `0`: manifest published with no quarantined rows;
- `1`: manifest published with one or more quarantined rows;
- `2`: fatal input/publication error before a valid derived manifest can be published.

## Output Layout

```text
artifacts/benchmark/reference-set/
  events/<sha256>.jsonl
  manifests/<sha256>.jsonl
  latest.json
```

Mapped reference events remain detailed and immutable. The common scorer projection is reproducibly re-derived through `project_common_reference_events`; no second common-event artifact is needed.

## Testing

Focused tests:

- detailed/common taxonomy and `DETAILED_TO_COMMON` are exact and total;
- every mapping with a detailed class agrees with `project_to_common`;
- DTX lane map produces detailed + common classes for every v1 lane;
- legacy reverse MIDI export supports `low_or_floor_tom` after the rename;
- OaF lookup uses `upstream_8hit_group_id` while preserving `native_class_id="midi_<note>"`;
- OaF hihat/toms do not fabricate detailed distinctions;
- OaF sticks remains explicitly unmapped;
- HPA-323 native event reader rejects non-canonical/hash-invalid content;
- reference mapping preserves every native identity field;
- mapped event rows mirror HPA-323 style and contain no per-row `schema` field;
- unknown lane handling is visible and deterministic;
- `project_common_reference_events` collapses only identical canonical decimal time + common class;
- eligibility is exactly `eligible|quarantined`, with warnings orthogonal to status;
- upstream HPA-323 quarantine is carried through without re-running timing/audio checks;
- mapped event and manifest schema goldens are byte-stable;
- CLI returns 0/1/2 consistently;
- persisted native events can be remapped under a new mapping version without inference.

No real model inference belongs in HPA-324 tests.

## Non-goals

- rerunning chart selection, DTX timing, BGM/audio resolution, or audio decoding;
- scoring precision/recall/F1 or choosing matching tolerances (HPA-325);
- corpus OaF inference/resume/runtime orchestration (HPA-326);
- implementing MuScriptor or IDM adapters (HPA-395/HPA-396);
- guessing MuScriptor pitches or IDM `train_classes` before exact locked vocabularies are observed;
- velocity scoring;
- manually judging musical fidelity or chart simplification;
- result-driven remapping after scores are inspected;
- a generic plugin/key-selector system, external mapping service, database, or configuration framework.

## Acceptance Interpretation

HPA-324 itself is complete when:

1. stable detailed/common taxonomy, total detailed-to-common projection, and DTX lane map v1 exist;
2. the OaF prediction map is frozen by `upstream_8hit_group_id` without rewriting native MIDI identity;
3. HPA-323 native events are deterministically remapped into immutable benchmark-reference events with no per-row schema field;
4. unknown/ignored lanes and duplicate common collapses are visible;
5. common scorer projection is one exported pure function keyed by canonical durable time;
6. every HPA-323 input row is exactly `eligible` or `quarantined`, with warnings orthogonal to status;
7. reference eligibility is model-independent and reconciles exactly;
8. a mapping version change can remap persisted native artifacts without rerunning inference;
9. after HPA-423 lands, it consumes HPA-324's OaF map data through group metadata and stamps map identity without redefining `native_class_id`;
10. HPA-395/HPA-396 remain responsible for freezing their exact MuScriptor/IDM maps before their scored runs.

The cross-model invariant that OaF, MuScriptor, and IDM each use distinct locked prediction maps is completed when HPA-395/HPA-396 make those model vocabularies concrete. HPA-324 must not invent future tables merely to make the present ticket look complete.
