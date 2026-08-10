# HPA-324: Versioned Drum Taxonomy, Mapping, and Eligibility Design

## Context

HPA-324 is the next unstarted Crux benchmark task that is actually ready.

- HPA-322 is complete and publishes authoritative chart selection.
- HPA-323 is complete and publishes audio-relative native reference events plus `crux.reference-timing-manifest/v1`.
- HPA-324 is blocked only by HPA-322 and HPA-323, so both explicit blockers are now satisfied.
- HPA-324 blocks HPA-325 scoring, HPA-326 OaF corpus inference, and HPA-327 reviewed-subset work.
- HPA-423 is already in progress and explicitly says HPA-324 owns the vocabulary/data while HPA-423 owns the `NativeEvent -> BenchmarkEvent` prediction-mapping mechanism.

The main branch already contains the right upstream artifact boundary from HPA-323:

- `reference_timing.py` publishes lossless `crux.dtx-reference-event/v1` JSONL with DTX lane/note identity and audio-relative times;
- `reference_timing_manifest.py` classifies rows as timing `ready` or `quarantined` and preserves all HPA-322/HPA-321 lineage;
- `mapping.py` still contains an unversioned legacy collapsed DTX/MIDI map;
- `prediction_artifact.py` already reserves `canonical_class`, `mapping_status`, and `prediction_map_version`, but native events are currently serialized with mapping `not_applied`;
- `scorer_input.py` is intentionally blocked until canonical prediction mapping exists.

HPA-324 should build on those seams, not add another corpus pipeline or model framework.

## Requirement Clarification: Benchmark Target

The Linear addendum is normative: the initial benchmark scores **onset time plus drum class**.

DTX `#WAVxx`, `#VOLUMExx`, sample choice, and chip identity are not trustworthy per-hit velocity ground truth. Therefore:

- predicted velocity/confidence stays diagnostic metadata only;
- DTX sample/note identity stays diagnostic metadata only;
- velocity is not part of reference eligibility or headline scoring;
- adding a velocity metric later requires a separately reviewed reference and versioned metric.

## Design Choice

Use a **small code-defined versioned taxonomy and mapping data layer**, plus one derived reference/eligibility stage.

Do not add YAML/JSON configuration frameworks, plugin discovery, databases, or a generic mapping DSL. The mappings are small, reviewed benchmark policy. Python dataclasses/constants plus stable version IDs are easier to test, refactor, and change deliberately.

Three approaches were considered:

1. **Recommended: typed mapping data in Python.** Smallest surface, easy unit testing, stable version IDs, and no runtime config parser.
2. External mapping JSON files. More editable, but adds schema/loading/publication machinery without a current user need.
3. Keep ad-hoc dictionaries/functions in `mapping.py`. Small today, but it cannot express stable per-model identities, detailed-vs-common class projection, or future rescoring cleanly.

## Key Modeling Decision: Detailed Class vs Common Comparison Class

One flat class label cannot honestly satisfy the benchmark requirements.

The authored DTX can distinguish open/closed hi-hat and high/low tom lanes, but frozen OaF uses an 8-hit training/output grouping. In the current OaF runtime, `CALIBRATION_TRAINING_GROUPS` contains one `hihat` group and one `toms` group. Mapping those outputs directly to `closed_hihat` or `high_tom` would fabricate distinctions the model never produced.

HPA-324 therefore defines two explicit class levels:

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

This preserves the benchmark's intended DTX semantics when the source/model can support them.

### Common comparison class

```text
kick
snare
hihat
crash
ride
tom
```

This is the shared headline comparison space. Detailed classes project as:

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

A prediction may have `canonical_class=None` when its native output cannot justify the detailed distinction while still having a valid `common_class`. OaF `hihat` and `toms` are the important examples.

This preserves both goals:

- detailed/native coverage remains diagnosable;
- cross-model scores compare only distinctions that every participating model can actually represent.

## Versioned Mapping Contracts

Create `src/benchmark/taxonomy.py` as the single policy/data owner.

```python
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

@dataclass(frozen=True)
class ClassMapping:
    canonical_class: DetailedDrumClass | None
    common_class: CommonDrumClass | None

@dataclass(frozen=True)
class PredictionMap:
    map_id: str
    model_id: str
    native_output_space_id: str
    classes: dict[str, ClassMapping]
```

Every versioned map is immutable by convention and test: changing mapping semantics requires a new ID. Git history is sufficient archival storage for this hobby project; a separate mapping database or signed registry is unnecessary.

### DTX lane map v1

Freeze the current intended drum lanes, correcting the old `low_tom` name to the issue's canonical terminology:

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

### OaF prediction map v1

Use the frozen OaF native output identities, not a guessed General MIDI table. The current runtime has the eight training/output groups `kick`, `snare`, `toms`, `hihat`, `ride`, `ride_bell`, `crash`, and `sticks`.

The HPA-423 adapter/mechanism should normalize OaF native event IDs consistently and apply one explicit map ID, for example:

```text
crux.prediction-map/oaf-egmd-8hit-v1
```

Required semantics:

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

`sticks` remains visible as unmapped native coverage; it is not silently converted to snare.

### MuScriptor and IDM ownership

Do **not** guess their exact native class tables inside HPA-324.

The official MuScriptor API emits `NoteStartEvent` with `pitch`, `start_time`, and `instrument`; hard `instruments=["drums"]` conditioning masks non-drum tokens, and the released event stream does not preserve velocity. HPA-395 must observe the exact locked drum pitch vocabulary and freeze a map before scored corpus results.

The official IDM repository explicitly exposes `model.train_classes`; HPA-396 already requires reading the exact loaded `idm-44-train-kits` ordering before scoring. HPA-396 must freeze that exact class table before its pilot.

HPA-324 defines the `PredictionMap` contract and common taxonomy they must use. HPA-395/HPA-396 materialize their exact map data once their locked native vocabularies are available. This avoids a circular dependency and satisfies the requirement that mappings be frozen before scored runs without inventing unsupported classes today.

Primary references:

- MuScriptor: https://github.com/muscriptor/muscriptor
- Inverse Drum Machine: https://github.com/bernardo-torres/inverse-drum-machine

## Reference Mapping Pipeline

HPA-323 already did the expensive/fragile work: chart selection, DTX parsing, timing, source-audio resolution/decoding, hash validation, BGM anchoring, and bounds checks.

HPA-324 must **not repeat those checks**. A timing `ready` row is the upstream technical-integrity gate.

Add a typed reader for `crux.dtx-reference-event/v1`, then map each native reference event to a new lossless scored-reference row.

```python
@dataclass(frozen=True)
class MappedReferenceEvent:
    native: NativeReferenceEvent
    canonical_class: DetailedDrumClass
    common_class: CommonDrumClass
```

Publish mapped reference events as content-addressed JSONL using a new stable schema:

```text
crux.benchmark-reference-event/v1
```

Each row carries both canonical levels plus the original HPA-323 identity fields:

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

The native HPA-323 event artifact remains authoritative and immutable; HPA-324's artifact is a derived scoring view.

## Unknown/Non-Drum Lane Policy

Do not silently treat every non-mapped pattern channel as irrelevant.

HPA-323 intentionally preserved native pattern events. HPA-324 should first add a committed corpus diagnostic that enumerates:

- every observed `lane_id` and event count;
- mapped vs unmapped lane counts;
- simfiles containing each unmapped lane;
- simultaneous events that would collide after common-class projection.

Use the real HPA-323 event reader/mapping functions. Do not build a parallel throwaway parser.

After the real corpus diagnostic is reviewed, freeze a small explicit `IGNORED_NON_DRUM_LANES` set only for observed lanes that are known non-drum instrumentation/control data. Any other unmapped lane quarantines the row rather than disappearing silently.

This is intentionally conservative. It prevents a newly observed playable drum lane from being excluded merely because the initial map forgot it.

## Duplicate-After-Collapse Semantics

Preserve all native/mapped event rows. Deduplicate only the **common scoring projection**.

For one song, common-scoring identity is:

```text
(audio_time_sec, common_class)
```

If multiple detailed/native events collapse to the same identity, the scorer receives one event. Diagnostics record the collapsed multiplicity and native identities.

Examples:

- simultaneous high/low toms become one common `tom` scoring event;
- simultaneous open/closed hi-hat becomes one common `hihat` scoring event;
- native artifacts still retain both authored hits.

This avoids unfairly charging a model twice for distinctions its locked output space cannot represent, while keeping the lost distinctions visible in coverage diagnostics.

Do not deduplicate events at different timestamps and do not introduce a fuzzy time bucket. Tolerance belongs to HPA-325 matching, not mapping.

## Reference Eligibility

Publish one derived manifest:

```text
crux.benchmark-reference-manifest/v1
```

The stage consumes `crux.reference-timing-manifest/v1` and its referenced native event artifacts. It carries all upstream fields through verbatim except the top-level derived `corpus_version`, then adds mapping/eligibility fields.

### Status

```text
eligible
eligible_with_warnings
quarantined
```

### Closed reason codes

Start with the smallest stable set HPA-324 itself owns:

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

Do not duplicate HPA-323 timing reason codes into this list. They remain present in the carried-through upstream row.

Warnings are deterministic strings for non-fatal mapping diagnostics such as explicit ignored-lane counts and duplicate-after-collapse counts.

### Accounting

The outcome must enforce:

```text
inventoried = eligible + eligible_with_warnings + quarantined
```

`eligible_with_warnings` is still benchmarkable. Low future model scores, simplified charts, and model-specific inference failures never change reference eligibility.

## Prediction Mapping Seam with HPA-423

HPA-423 owns the mechanism. HPA-324 owns the data.

The shared seam should be one explicit call shape equivalent to:

```python
def map_native_events(
    events: tuple[NativeEvent, ...],
    *,
    prediction_map: PredictionMap,
) -> MappingResult: ...
```

HPA-423 is responsible for:

- converting native prediction events;
- stamping `prediction_map_version`/map ID;
- preserving native identity and confidence/velocity;
- surfacing unmapped events;
- unblocking `scorer_input.py`.

HPA-324 is responsible for:

- the canonical/common class definitions;
- DTX lane-map data;
- OaF prediction-map data;
- stable map IDs and map validation;
- reference-side mapping/eligibility artifacts.

Implementation should rebase onto the HPA-423 merge before touching the prediction seam. Do not independently rewrite `NativeEvent`, `prediction_artifact.py`, or `scorer_input.py` on the HPA-324 branch if HPA-423 has already changed them.

## CLI

Add one derived-stage command:

```bash
uv run crux benchmark build-reference-set \
  --manifest artifacts/benchmark/reference-timing/manifests/<sha256>.jsonl \
  --output-dir artifacts/benchmark/reference-set
```

`--manifest` is required. `--output-dir` defaults to `artifacts/benchmark/reference-set`.

No R2 credentials are needed. Every required source/chart/audio decision has already been frozen by HPA-321/322/323.

Exit codes follow the existing derived-manifest convention:

- `0`: no quarantined rows;
- `1`: manifest published with one or more quarantined rows;
- `2`: fatal input/publication error before a valid derived manifest can be published.

## Output Layout

```text
artifacts/benchmark/reference-set/
  events/<sha256>.jsonl
  manifests/<sha256>.jsonl
  latest.json
```

Mapped reference event artifacts are immutable/content-addressed. The manifest points to them by relative path and hash.

No database or mutable per-song cache is added.

## Testing

Focused tests:

- taxonomy version and allowed classes are exact;
- DTX lane map produces detailed + common classes for every v1 lane;
- no map version changes semantics without a test/golden update;
- OaF hihat/toms do not fabricate detailed distinctions;
- OaF sticks remains explicitly unmapped;
- HPA-323 native event reader rejects non-canonical/hash-invalid content;
- reference mapping preserves every native identity field;
- unknown lane handling is visible and deterministic;
- duplicate common projections dedupe only at exact same timestamp/class;
- eligibility status/reasons reconcile exactly;
- upstream HPA-323 quarantine is carried through without re-running timing/audio checks;
- mapped event and manifest schema goldens are byte-stable;
- CLI returns 0/1/2 consistently;
- acceptance fixture proves a mapping-version correction can remap persisted native events without inference.

No real model inference belongs in HPA-324 tests.

## Non-goals

- rerunning chart selection, DTX timing, BGM/audio resolution, or audio decoding;
- scoring precision/recall/F1 or choosing matching tolerances (HPA-325);
- corpus OaF inference/resume/runtime orchestration (HPA-326);
- implementing MuScriptor or IDM adapters (HPA-395/HPA-396);
- guessing MuScriptor pitches or IDM `train_classes` before their exact locked vocabularies are observed;
- velocity scoring;
- manually judging musical fidelity or chart simplification;
- result-driven remapping after scores are inspected;
- a generic plugin system, external mapping service, database, or configuration framework.

## Acceptance Interpretation

HPA-324 itself is complete when:

1. stable detailed/common taxonomy and DTX lane map v1 exist;
2. the OaF prediction map is frozen before HPA-326 scored inference;
3. HPA-423 can consume map data without owning taxonomy policy;
4. native HPA-323 events can be deterministically remapped into immutable benchmark-reference events;
5. reference eligibility is model-independent and reconciles exactly;
6. unmapped/ignored lanes and duplicate collapses are visible;
7. a mapping version change can rescore persisted native artifacts without rerunning inference;
8. MuScriptor/IDM have an explicit contract and ownership rule requiring their own frozen map IDs before their later scored runs.

The cross-epic invariant that OaF, MuScriptor, and IDM each use distinct locked prediction maps is enforced by HPA-395/HPA-396 when those model vocabularies become concrete; HPA-324 must not fake those future tables merely to make the present ticket look complete.
