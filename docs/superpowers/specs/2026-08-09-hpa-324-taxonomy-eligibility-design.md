# HPA-324: Versioned Drum Taxonomy, Mapping, and Eligibility Design

**Issue:** HPA-324  
**Status:** Revised after review  
**Date:** 2026-08-09

## Context

HPA-324 is the next unstarted Crux benchmark task that is ready:

- HPA-322 is complete and selects authoritative DTX charts.
- HPA-323 is complete and publishes audio-relative native DTX reference events plus `crux.reference-timing-manifest/v1`.
- HPA-324 is blocked only by HPA-322 and HPA-323, so its explicit blockers are satisfied.
- HPA-324 blocks HPA-325 scoring, HPA-326 OaF corpus inference, and HPA-327 reviewed-reference work.
- HPA-423 is already in progress and owns prediction conversion mechanics. HPA-324 owns the taxonomy and mapping policy/data that HPA-423 must consume.

The repository already has the important persistence seams:

- `reference_timing.py` persists native DTX events with lane/note identity and audio-relative time.
- `reference_timing_manifest.py` carries source/chart/audio lineage and `ready|quarantined` timing status.
- `prediction_artifact.py` persists OaF native identity, including `native_class_id="midi_<note>"` and `native_metadata["upstream_8hit_group_id"]`.
- `prediction_artifact.py` currently has `canonical_class`, `mapping_status`, and `prediction_map_version` fields, but **no common-class field**.
- `scorer_input.py` is intentionally blocked until prediction mapping is available.

HPA-324 should define the smallest stable taxonomy and reference-eligibility layer on top of those seams. It should not add another corpus pipeline or persist data that can be deterministically recomputed from already-persisted native events.

## Benchmark Target

The initial benchmark scores **onset time plus instrument class**.

DTX `#WAVxx`, `#VOLUMExx`, chip identity, and sample choice are not trustworthy per-hit performance-velocity ground truth. Therefore:

- predicted confidence and velocity remain diagnostic metadata only;
- DTX `note_id` and sample metadata remain diagnostic identity only;
- velocity is not part of reference eligibility or headline scoring;
- a future velocity metric requires its own reviewed reference and versioned metric.

## Design Choice

Use a **small code-defined taxonomy/mapping data layer** plus one derived eligibility manifest.

Do not add YAML/JSON mapping configuration, plugin discovery, a mapping database, a generic native-key selector, or a new service. These maps are small benchmark policy. Python constants, frozen dataclasses, and stable version IDs are enough.

The revised design is intentionally smaller than the first draft:

1. keep taxonomy and maps in memory;
2. read HPA-323 native reference artifacts directly;
3. map and project them through pure functions;
4. publish only the eligibility manifest and diagnostic counts;
5. let HPA-325 call the same pure functions instead of publishing a redundant mapped-event artifact.

## 1. Two Class Levels

A single class label is insufficient because authored DTX contains distinctions OaF does not emit.

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

The common class is the **headline scoring space**.

Freeze one total projection:

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

DETAILED_TO_COMMON: Mapping[DetailedDrumClass, CommonDrumClass] = MappingProxyType(
    {
        "kick": "kick",
        "snare": "snare",
        "closed_hihat": "hihat",
        "open_hihat": "hihat",
        "crash": "crash",
        "ride": "ride",
        "high_tom": "tom",
        "low_or_floor_tom": "tom",
    }
)


def project_to_common(detailed: DetailedDrumClass) -> CommonDrumClass:
    return DETAILED_TO_COMMON[detailed]
```

Every map entry with a non-null detailed class must satisfy:

```python
mapping.common_class == project_to_common(mapping.canonical_class)
```

This makes the detailed-to-common relationship code, not prose.

## 2. Versioned Mapping Data

Create `src/benchmark/taxonomy.py` as the single policy/data owner.

```python
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
```

Use `MappingProxyType` for frozen tables. Git history is sufficient archival storage for old mapping code in this hobby project. A map-semantics change requires a new version ID.

### DTX lane map v1

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

Native `lane_id` and `note_id` are never replaced.

The `low_or_floor_tom` rename must update the legacy reverse MIDI map in `midi_io.py` in the same change so `write_reference_midi` cannot silently drop the renamed class. `mid_tom` is removed if no active producer remains after the migration.

### OaF prediction map v1

Do **not** redefine OaF native identity.

Current OaF events intentionally use values such as:

```text
native_class_id = midi_36
native_midi_note = 36
model_output_bin = 15
native_metadata.upstream_8hit_group_id = kick
```

The 8-hit semantic lookup key is:

```python
key = event.native_metadata["upstream_8hit_group_id"]
```

Freeze:

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

`sticks` remains visible as unmapped coverage.

Tests must bind this table to the existing locked vocabulary rather than duplicate the vocabulary as an independent literal:

```python
assert set(OAF_PREDICTION_MAP.classes) == OAF_GROUP_IDS
assert OAF_PREDICTION_MAP.model_id == OAF_BACKEND_ID
```

These assertions can live in tests to avoid creating an undesirable production import cycle from `taxonomy.py` into `prediction_artifact.py`.

### MuScriptor and IDM

Do not guess their exact tables in HPA-324.

- HPA-395 freezes the exact observed MuScriptor drum vocabulary before its scored run.
- HPA-396 freezes the exact loaded IDM `train_classes` ordering before its pilot.

Both must use this taxonomy/projection contract and their own immutable prediction-map IDs.

## 3. Prediction Persistence Must Carry the Common Class

This is a benchmark correctness gate, not an optional HPA-423 implementation detail.

The current `crux.drum-prediction-events/v1` event row has `canonical_class` but no `common_class`. That is insufficient because valid OaF predictions such as `hihat` and `toms` intentionally have no detailed canonical class.

Before HPA-326 can persist corpus predictions, the active prediction artifact contract must become:

```text
crux.drum-prediction-events/v2
```

with both:

```text
canonical_class
common_class
```

Required event semantics:

```text
mapping_status = mapped:
  prediction_map_version is non-null
  common_class is non-null
  canonical_class may be null

mapping_status = unmapped:
  prediction_map_version is non-null
  canonical_class is null
  common_class is null
```

Native fields remain unchanged:

```text
native_class_id
model_output_bin
native_midi_note
native_metadata
confidence
velocity_midi
```

A mapped OaF hi-hat event therefore persists approximately:

```json
{
  "native_class_id": "midi_46",
  "native_metadata": {"upstream_8hit_group_id": "hihat"},
  "canonical_class": null,
  "common_class": "hihat",
  "mapping_status": "mapped",
  "prediction_map_version": "crux.prediction-map/oaf-egmd-8hit-v1"
}
```

The exact serializer implementation remains HPA-423-owned. HPA-324 defines this data requirement because HPA-324 defines the scoring taxonomy.

**HPA-326 must not start until a real OaF-shaped hi-hat/tom event round-trips through the active prediction artifact with `common_class` intact.** Otherwise the expensive corpus run would produce unscoreable predictions and require rerunning inference.

## 4. Read HPA-323 Artifacts; Do Not Persist a Derived Mapping Artifact

HPA-323 already persists every native reference identity needed for deterministic remapping.

Add only the missing read adapters:

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
```

`read_reference_events` reuses HPA-323's existing row/sequence validators and requires byte-identical rerendering.

`ReferenceTimingRowView` reuses `_validate_timing_status_shape` and exposes only the HPA-324 fields.

### No `crux.benchmark-reference-event/v1`

Do **not** publish a second mapped-event JSONL.

`canonical_class` and `common_class` are pure functions of `lane_id` plus the frozen taxonomy/map version. The source HPA-323 event bytes are already content-addressed and the HPA-324 manifest records the taxonomy/lane-map version.

Persisting a mapped copy would add:

- another renderer;
- another reader;
- another key-set validator;
- another schema golden;
- another immutable publication path;
- another artifact-path/count field;

without adding independent information.

If a future workflow genuinely needs frozen mapped bytes, it can add that one publication step later. HPA-324 does not prepay that complexity.

## 5. Pure Reference Mapping and Common Projection

Create `src/benchmark/reference_set.py` with pure in-memory functions only.

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


def map_reference_events(
    events: tuple[NativeReferenceEvent, ...],
) -> ReferenceMappingResult: ...


def project_common_reference_events(
    mapped_events: tuple[MappedReferenceEvent, ...],
) -> tuple[CommonReferenceEvent, ...]: ...
```

`project_common_reference_events` is the single scorer-facing projection HPA-325 must reuse.

### Exact duplicate identity

Use durable canonical time:

```python
Decimal(str(event.native.audio_time_sec))
```

Common scoring identity is:

```text
(canonical_audio_time, common_class)
```

No rounding, fuzzy bucket, or tolerance belongs here. HPA-325 owns matching tolerances.

Multiple detailed/native events at the same exact durable time/common class become one common scorer event while all source events remain available for diagnostics.

## 6. Unknown-Lane Policy Requires Real-Corpus Evidence

HPA-323 currently preserves pattern channels outside the explicitly handled control channels. Therefore an empty ignored-lane set plus "any unknown lane quarantines" could quarantine a large fraction of the real corpus while all accounting checks still pass.

The exact impact is not known from the repository alone. Do not guess.

Start with:

```python
IGNORED_NON_DRUM_LANES = frozenset()
```

Then run a committed real-corpus audit **before eligibility policy is allowed to freeze**.

### Hard audit gate

Add `tools/hpa324/analyze_reference_lanes.py`, using only production readers/mapping functions.

Commit its reviewed output to:

```text
docs/superpowers/evidence/2026-08-09-hpa-324-reference-lane-audit.json
```

The report records at minimum:

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
prospective_eligible_row_count
prospective_quarantined_row_count
```

Review every observed unmapped lane. Add a lane to `IGNORED_NON_DRUM_LANES` only when its non-drum meaning is verified from the DTX semantics/source data. Uncertain lanes remain unclassified and quarantine affected rows.

Do **not** replace this with a broad rule such as "ignore every channel outside the current drum map" or "ignore every non-1x channel". That would silently discard a future/extended playable drum lane.

### No escape hatch

If the real HPA-323 manifest/artifacts are unavailable, HPA-324 implementation stops at the audit gate. It does not ship eligibility policy with an unmeasured empty ignore set.

If the audit reports zero prospective eligible rows, stop and review lane disposition before implementing the eligibility manifest.

Acceptance coverage must include at least one observed, reviewed non-drum lane from the committed audit whenever the audit produces such a lane.

## 7. Reference Eligibility Manifest

Publish only:

```text
crux.benchmark-reference-manifest/v1
```

It consumes the HPA-323 timing manifest and native event artifacts and carries all upstream fields through verbatim except top-level `corpus_version`.

### Status

```text
eligible
quarantined
```

Warnings are orthogonal data on eligible rows.

### Closed reason codes

```python
EligibilityReasonCode = Literal[
    "upstream_reference_unavailable",
    "reference_event_artifact_invalid",
    "unclassified_reference_lane",
    "no_scored_drum_events",
]
```

### Added fields

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

Do **not** add `benchmark_reference_events_path` or a mapped-event artifact counter.

The inherited HPA-323 `reference_events_cache_path` remains the durable source event reference HPA-325 uses.

Deterministic warning formats:

```text
ignored_reference_lane:<LANE>:count=<N>
duplicate_common_projection:count=<N>
```

Accounting:

```text
inventoried = eligible + quarantined
```

Low model scores, model-specific failures, and chart simplification judgments never change reference eligibility.

## 8. Reuse the Existing Manifest Validation Core, Not a Generic Framework

The current `load_reference_chart_manifest()` is not directly reusable: it hardcodes the HPA-322 schema and `ReferenceChartRowView` source-identity fields.

However, HPA-324 should not create a third independent canonical-JSONL parser either.

The targeted refactor is:

1. extract the byte/canonical-JSONL/re-render core from `load_reference_chart_manifest()` into a small private helper inside `reference_timing_manifest.py`;
2. keep HPA-322-specific identity checks in `load_reference_chart_manifest()`;
3. add `load_reference_timing_manifest()` in the same source-owner module, reusing the private core plus `ReferenceTimingRowView` validation;
4. let `reference_set_manifest.py` consume that typed loader.

Do not generalize row remapping or outcome accounting into callback-heavy frameworks. `build_timing_row()` is timing-specific, and the HPA-324 row/outcome invariants are short enough to remain explicit.

## 9. CLI

Add:

```bash
uv run crux benchmark build-reference-set \
  --manifest artifacts/benchmark/reference-timing/manifests/<sha256>.jsonl \
  --output-dir artifacts/benchmark/reference-set
```

`--manifest` is required. `--output-dir` defaults to `artifacts/benchmark/reference-set`.

No cache/R2/model/tolerance/concurrency options belong here.

Exit codes follow the existing derived-manifest convention:

- `0`: manifest published and every row is eligible;
- `1`: manifest published with one or more quarantined rows;
- `2`: fatal input/publication failure before a valid derived manifest can be published.

Output layout:

```text
artifacts/benchmark/reference-set/
  manifests/<sha256>.jsonl
  latest.json
```

## Risks and Gates

### Risk 1: common prediction class discovered after inference

**Risk:** HPA-326 persists OaF hihat/tom events without a scorable common class and must rerun expensive inference.

**Gate:** `crux.drum-prediction-events/v2` with `common_class` must be implemented and a hihat/tom round-trip test must pass before HPA-326 starts.

### Risk 2: corpus-wide quarantine from unknown lanes

**Risk:** HPA-323 preserves many pattern channels, so an unreviewed empty ignore set can make technically valid references all quarantine.

**Gate:** committed real-corpus lane audit is mandatory before eligibility policy/Task 5.

### Risk 3: HPA-423 integration timing

HPA-324 Tasks covering taxonomy/reference mapping/eligibility may land before HPA-423 finishes, but **HPA-324 must remain In Progress** until the prediction persistence/mapping seam is verified against the merged HPA-423 implementation.

If HPA-423 is not ready, stop after the independent HPA-324 work. Do not mark HPA-324 Done and do not start HPA-326.

## Testing Strategy

Focused tests only:

- detailed/common taxonomies and total projection are exact;
- DTX map uses the projection invariant;
- OaF map keys equal existing `OAF_GROUP_IDS` and model ID equals `OAF_BACKEND_ID`;
- OaF hihat/toms do not fabricate detailed distinctions;
- reverse MIDI export handles `low_or_floor_tom`;
- HPA-323 event reader rejects non-canonical/invalid content and round-trips valid bytes;
- timing-row view/loader reuses existing validation;
- mapping preserves native identity;
- common collapse uses `Decimal(str(audio_time_sec))` and deduplicates only exact identity;
- hard lane-audit evidence is produced before eligibility policy freezes;
- observed reviewed non-drum lanes remain eligible with warnings rather than being silently dropped/quarantined;
- unknown lanes remain visible and quarantine;
- eligibility accounting is exact;
- HPA-325 can reload native events and reuse `project_common_reference_events` without a mapped-event artifact;
- prediction artifact v2 round-trips mapped OaF hihat/toms with `common_class` intact;
- CLI returns 0/1/2 consistently.

No real model inference belongs in HPA-324 unit tests.

## Validation Commands

Repository-defined validation remains the gate:

```bash
uv run pytest
uv run ruff check src tests
uv run ruff format --check src tests
uv run pylint src/app src/cli
```

HPA-324-specific tests may additionally lint/check `tools/hpa324`, but a broader `pylint src` run is diagnostic only, not the repository gate.

## Non-Goals

- rerunning chart selection, DTX timing, BGM/audio resolution, or audio decoding;
- publishing a redundant mapped-reference event artifact;
- scoring precision/recall/F1 or choosing tolerances;
- corpus OaF inference/resume/runtime orchestration;
- implementing MuScriptor or IDM adapters;
- guessing future model vocabularies;
- velocity scoring;
- manually judging musical fidelity/chart simplification;
- result-driven remapping after scores are inspected;
- generic loader/outcome frameworks, plugin systems, mapping services, databases, or configuration DSLs.

## Acceptance Criteria

HPA-324 can be moved to Done only when:

1. one code-owned detailed/common taxonomy and total projection are frozen;
2. DTX lane map v1 is frozen and legacy mapping/MIDI consumers use the new class names;
3. OaF prediction map v1 is keyed by `upstream_8hit_group_id` and bound by tests to the existing OaF vocabulary/model identity;
4. HPA-323 reference events are consumed through thin readers and can be remapped without rerunning inference;
5. `map_reference_events` and public `project_common_reference_events` are the single reference mapping/projection implementations;
6. the reviewed real-corpus lane audit is committed and the ignore/quarantine policy is frozen from that evidence;
7. every HPA-323 row is exactly `eligible` or `quarantined`, with deterministic warnings/counts;
8. HPA-325 can reconstruct scorer-ready common reference events from the inherited native artifact plus HPA-324 version IDs;
9. the active prediction artifact persists `common_class` and a real OaF-shaped hihat/tom event round-trips with its common class intact;
10. HPA-423 consumes the OaF group map without rewriting native MIDI identity and stamps the prediction-map ID;
11. HPA-395/HPA-396 remain responsible for their exact later model maps;
12. HPA-326 has not started before items 9-10 are verified.
