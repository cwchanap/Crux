# HPA-395 MuScriptor Full-Mix Benchmark Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Freeze one official MuScriptor checkpoint and pitch map, run hard drum-only native-event inference over the same canonical full mixes/references as OaF, score broad and HPA-327 reviewed populations through HPA-325, and publish narrow identity-matched OaF deltas plus runtime/license evidence.

**Architecture:** Keep MuScriptor comparator-only. Reuse the existing `TranscriptionBackend`, prediction-v2, HPA-324 reference, HPA-325 scorer/report, HPA-326 full-mix, and HPA-327 subset seams. Add one optional MuScriptor adapter and one model-specific corpus runner. Extend descriptor/prediction validation with backend-specific invariants rather than weakening OaF rules or creating a new prediction schema. Promote the three model-neutral helpers that now have two real callers: canonical full-mix materialization, verified source-audio resolution/re-pin, and immutable prediction-path construction. Keep two concrete runners; do not add a runner base class.

**Tech Stack:** Python 3.12, `muscriptor==0.3.0`, PyTorch through an optional MuScriptor extra, Hugging Face Hub for one-time gated checkpoint freezing, existing librosa/soundfile canonicalization, Click, pytest, existing canonical JSON/JSONL helpers, and HPA-323/HPA-324/HPA-325/HPA-326/HPA-327 artifacts.

## Global constraints

- Freeze MuScriptor `v0.3.0` at release commit `d73147e75e5b9b0c0a79ebe154587db4fd603e0c`.
- Start with the published medium checkpoint. Switching once to small is allowed only for demonstrated hardware impracticality before benchmark scores are inspected.
- Scored runs load a local safetensors file pinned by exact Hugging Face repo revision, SHA-256, byte size, companion config hash/size, package version, device/dtype, and code/weight licenses in `runtime/muscriptor/model.json`.
- Never load scored weights with bare `"medium"`, `"small"`, or an unqualified `hf://...` URL.
- Freeze `use_sampling=False`, `temperature=1.0`, `cfg_coef=1.0`, `instruments=["drums"]`, `batch_size=1`, `no_eos_is_ok=True`, `beam_size=1`, and `prelude_forcing=True`. No scored CLI overrides.
- Persist raw `NoteStartEvent.start_time` plus native drum MIDI pitch. `NoteEndEvent` and `ProgressEvent` create no hit. Do not invent confidence or velocity and do not round-trip through MIDI.
- Freeze `MUSCRIPTOR_PREDICTION_MAP_ID = "crux.prediction-map/muscriptor-drums-v1"` and the exact v1 pitch table in Task 4 before any real inference or scores are inspected.
- Reuse the exact HPA-326 44.1 kHz mono PCM16 `soxr_hq` full-mix bytes so matching OaF/MuScriptor songs share the same historical `input_view_id = "crux.oaf-full-mix-mono44k1-pcm16/v1"` and `input_audio_sha256`. MuScriptor's own 16 kHz preprocessing remains internal and frozen as model semantics.
- Keep `crux.drum-prediction-events/v2`; add backend-specific header and event validation. Existing OaF v2 bytes and validation rules must remain unchanged.
- Keep HPA-323/HPA-324/HPA-327 identities, taxonomy, tolerances, modes, and HPA-325 scoring semantics frozen.
- Persist unsupported MuScriptor pitches as unmapped events rather than dropping them.
- Do not add MuScriptor to `default_backend_registry()` or production API selection.
- CI uses fake MuScriptor models/events. It never downloads gated weights or requires an accelerator.
- Do not add a generic experiment runner, plugin framework, database, queue, worker protocol, second scorer, tuning DSL, training, or stem separation.
- HPA-562 owns generalized multi-model reporting later; HPA-395 only needs OaF-vs-MuScriptor comparison.

---

## File structure

### New production files

- `src/benchmark/muscriptor_model.py` — frozen model-lock/smoke contracts and checkpoint verification.
- `src/benchmark/backends/muscriptor.py` — optional lazy MuScriptor adapter implementing `TranscriptionBackend`.
- `src/benchmark/muscriptor_corpus_run.py` — concrete persisted MuScriptor corpus runner.
- `src/benchmark/muscriptor_comparison.py` — narrow OaF-vs-MuScriptor persisted-result comparison.
- `scripts/freeze_muscriptor_model.py` — one-time authenticated checkpoint/revision/license freeze.

### Existing files with narrow changes

- `src/benchmark/input_view.py` — shared canonical full-mix materializer.
- `src/benchmark/corpus_cache.py` — shared verified `ResolvedSourceAudio` + resolution/re-pin helper.
- `src/benchmark/prediction_artifact.py` — shared `prediction_path()` plus MuScriptor header/event validation.
- `src/benchmark/backend_identity.py` — second frozen descriptor family.
- `src/benchmark/taxonomy.py` — frozen MuScriptor prediction-map identity/table.
- `src/benchmark/mapping.py` — persist-unmapped MuScriptor mapping.
- `src/benchmark/cohort_scoring.py` — model-neutral zero-hit map identity.
- `src/benchmark/oaf_corpus_run.py` — imports the promoted neutral helpers; no behavior change.
- `src/benchmark/reviewed_subset.py` — small model-neutral subset filter/score core.
- `src/cli/benchmark.py` — three thin MuScriptor benchmark commands.
- `tests/test_cli_benchmark.py` — all CLI surface coverage.

---

### Task 1: Freeze the optional package/checkpoint/config contract

**Files:**
- Modify: `pyproject.toml`
- Modify: `uv.lock`
- Create: `src/benchmark/muscriptor_model.py`
- Create: `scripts/freeze_muscriptor_model.py`
- Create during authenticated setup: `runtime/muscriptor/model.json`
- Create before first inference: `runtime/muscriptor/smoke.json`
- Create: `tests/benchmark/test_muscriptor_model.py`

**Interfaces:**

```python
MUSCRIPTOR_MODEL_SCHEMA = "crux.muscriptor-model/v1"
MUSCRIPTOR_SMOKE_SCHEMA = "crux.muscriptor-smoke/v1"

@dataclass(frozen=True)
class MuscriptorModelLock:
    backend_id: str
    model_id: str
    architecture_id: str
    package_name: str
    package_version: str
    upstream_source_commit: str
    code_license: str
    weight_license: str
    checkpoint_repo_id: str
    checkpoint_revision: str
    checkpoint_filename: str
    checkpoint_sha256: str
    checkpoint_byte_length: int
    checkpoint_config_filename: str
    checkpoint_config_sha256: str
    checkpoint_config_byte_length: int
    device: str
    dtype: str
    input_sample_rate_hz: int
    chunk_duration_sec: float
    use_sampling: bool
    temperature: float
    cfg_coef: float
    instruments: tuple[str, ...]
    batch_size: int
    no_eos_is_ok: bool
    beam_size: int
    prelude_forcing: bool
    native_output_space_id: str
    native_metadata_schema_id: str
    training_data_map_id: str


def load_muscriptor_model_lock(path: Path) -> MuscriptorModelLock: ...
def verify_muscriptor_checkpoint(lock: MuscriptorModelLock, checkpoint_dir: Path) -> Path: ...
def load_muscriptor_smoke(path: Path) -> tuple[MuscriptorSmokeItem, ...]: ...
```

- [ ] **Step 1: Write strict model-lock tests first**

Add tests for exact schema/key set, `muscriptor==0.3.0`, release commit, lowercase 40-character HF revision, lowercase SHA-256 values, positive sizes, separate nonempty code/weight license fields, exact drum-only decode settings, 16 kHz internal target, five-second chunk duration, and rejection of missing/edited checkpoint/config files.

```python
def test_model_lock_rejects_sampling() -> None:
    row = valid_lock_row()
    row["use_sampling"] = True
    with pytest.raises(ValueError, match="use_sampling"):
        parse_lock(row)
```

- [ ] **Step 2: Verify red**

```bash
uv run pytest tests/benchmark/test_muscriptor_model.py -q
```

Expected: FAIL because the model-lock module does not exist.

- [ ] **Step 3: Add the optional dependency and lock it**

```toml
[project.optional-dependencies]
muscriptor = ["muscriptor==0.3.0"]
```

```bash
uv lock
```

If this cannot coexist with the current Python 3.12 dependency set, stop and introduce a tiny dedicated MuScriptor runtime only then. Do not preemptively add a subprocess boundary.

- [ ] **Step 4: Implement the strict local lock/checkpoint verifier**

Use existing `strict_json_loads()`, `require_sha256()`, `sha256_hex()`, and regular-file helpers. The verifier hashes the exact local `model.safetensors` and companion `config.json`; a scored run receives only the verified local safetensors path.

The parser requires these fixed values:

```python
package_name == "muscriptor"
package_version == "0.3.0"
upstream_source_commit == "d73147e75e5b9b0c0a79ebe154587db4fd603e0c"
instruments == ("drums",)
use_sampling is False
temperature == 1.0
cfg_coef == 1.0
batch_size == 1
no_eos_is_ok is True
beam_size == 1
prelude_forcing is True
input_sample_rate_hz == 16000
chunk_duration_sec == 5.0
```

- [ ] **Step 5: Implement the one-time freeze script**

`scripts/freeze_muscriptor_model.py` must:

1. resolve `MuScriptor/muscriptor-medium` through Hugging Face Hub metadata;
2. capture the exact repository commit SHA;
3. download `model.safetensors` and `config.json` with `revision=<exact sha>`;
4. hash/size both local files;
5. verify official code/weight license evidence;
6. write canonical `runtime/muscriptor/model.json` with the selected device/dtype;
7. never import or invoke benchmark scoring.

A missing/contradictory weight license exits nonzero rather than inventing a value.

- [ ] **Step 6: Freeze the five pre-score smoke roles**

`runtime/muscriptor/smoke.json` contains exactly one unique positive simfile ID and nonempty pre-model reason for each role:

```text
short
long
dense
sparse
non_drum_heavy
```

Select IDs only from HPA-323/HPA-324/HPA-327 reference/audit/source-listening evidence, never model scores.

- [ ] **Step 7: Run focused tests**

```bash
uv run pytest tests/benchmark/test_muscriptor_model.py -q
```

Expected: PASS without network access.

- [ ] **Step 8: Commit**

```bash
git add pyproject.toml uv.lock src/benchmark/muscriptor_model.py scripts/freeze_muscriptor_model.py runtime/muscriptor tests/benchmark/test_muscriptor_model.py
git commit -m "feat: freeze MuScriptor benchmark model"
```

**Operational gate:** Actual checkpoint/revision/hash/license values must be frozen before Task 8. No synthetic digest may reach a scored run.

---

### Task 2: Promote the canonical full-mix materializer with byte-for-byte OaF characterization

**Files:**
- Modify: `src/benchmark/input_view.py`
- Modify: `src/benchmark/oaf_corpus_run.py`
- Modify: `tests/benchmark/test_input_view.py`
- Modify: `tests/benchmark/test_oaf_corpus_run.py`
- Verify unchanged: `tests/benchmark/test_oaf_corpus_run_acceptance.py`

**Interface:**

```python
FULL_MIX_INPUT_VIEW_ID = "crux.oaf-full-mix-mono44k1-pcm16/v1"
FULL_MIX_CANONICALIZATION_REVISION = "librosa-soxr-hq-mono44k1-soundfile-pcm16/v1"


def materialize_full_mix_audio(
    source: Path | bytes,
    output_path: Path,
    *,
    source_audio_id: str,
    source_audio_sha256: str,
    max_input_audio_frames: int | None,
) -> CanonicalAudio:
    ...
```

- [ ] **Step 1: Add a characterization test around current OaF materialization**

Use a deterministic source fixture and capture output WAV bytes, SHA-256, sample rate/channel/sample width/frame count, and `input_view_id`.

```python
def test_oaf_full_mix_characterization(tmp_path: Path) -> None:
    audio = current_materializer(...)
    assert audio.input_view_id == "crux.oaf-full-mix-mono44k1-pcm16/v1"
    assert sha256(audio.path.read_bytes()).hexdigest() == EXPECTED_SHA256
```

The test fixture determines `EXPECTED_SHA256` from the pre-refactor implementation and commits that literal as the characterization oracle.

- [ ] **Step 2: Verify green before refactor**

```bash
uv run pytest tests/benchmark/test_oaf_corpus_run.py -k materialize -q
```

Expected: PASS against the existing private implementation.

- [ ] **Step 3: Move exactly the model-neutral conversion**

Move the existing behavior to `input_view.py` without semantic edits:

```python
samples, _ = librosa.load(source_input, sr=44100, mono=True, res_type="soxr_hq")
soundfile.write(output_path, samples, 44100, format="WAV", subtype="PCM_16")
```

Then call existing `load_materialized_audio()` with the historical full-mix input-view ID.

- [ ] **Step 4: Point OaF at the helper and delete `_materialize_oaf_full_mix()`**

OaF passes `config.max_input_audio_frames`. Do not retain a compatibility alias.

- [ ] **Step 5: Prove unchanged OaF bytes/acceptance**

```bash
uv run pytest tests/benchmark/test_input_view.py tests/benchmark/test_oaf_corpus_run.py tests/benchmark/test_oaf_corpus_run_acceptance.py -q
```

Expected: PASS and the characterization bytes/hash stay unchanged.

- [ ] **Step 6: Commit**

```bash
git add src/benchmark/input_view.py src/benchmark/oaf_corpus_run.py tests/benchmark/test_input_view.py tests/benchmark/test_oaf_corpus_run.py
git commit -m "refactor: share canonical full mix input"
```

---

### Task 3: Extend frozen descriptor/prediction-v2/scorer identity for a second backend

**Files:**
- Modify: `src/benchmark/backend_identity.py`
- Modify: `src/benchmark/prediction_artifact.py`
- Modify: `src/benchmark/cohort_scoring.py`
- Modify: `tests/benchmark/test_backend_identity.py`
- Modify: `tests/benchmark/test_prediction_artifact.py`
- Modify: `tests/benchmark/test_cohort_scoring.py`
- Verify unchanged: `tests/benchmark/schema_goldens/crux.drum-prediction-events-v2.jsonl`
- Verify unchanged: `tests/benchmark/schema_goldens/manifest.json`
- Verify unchanged: `tests/benchmark/test_schema_goldens.py`

**MuScriptor descriptor family:**

```text
backend_id = muscriptor-v0.3.0-drums-v1
architecture_id = muscriptor-transformer-v0.3.0
native_output_space_id = muscriptor-drums-midi128-v1
native_metadata_schema_id = muscriptor-note-start-metadata-v1
prediction_schema = crux.drum-prediction-events/v2
```

`model_id` is derived from the exact frozen checkpoint variant/revision/hash. The runner later requires descriptor equality with the loaded lock.

- [ ] **Step 1: Add red descriptor tests**

`normalize_known_backend_descriptor()` accepts only the existing OaF family or the MuScriptor v0.3.0 family and rejects mixed fields, unknown backends, bad source commit, invalid model identity format, and wrong metadata/output-space identity.

- [ ] **Step 2: Add red MuScriptor prediction-v2 round-trip tests**

Use:

```python
NativeEvent(
    time_sec=1.25,
    native_class_id="drums:midi_38",
    model_output_bin=None,
    native_midi_note=38,
    native_metadata={"instrument_group": "drums"},
    confidence=None,
    velocity_midi=None,
)
```

Expect render -> read -> render byte identity. Also reject non-null output bin/confidence/velocity, missing/out-of-range MIDI pitch, non-drum metadata, and mismatched class ID. Existing OaF nullability/identity tests stay unchanged.

- [ ] **Step 3: Verify red**

```bash
uv run pytest tests/benchmark/test_backend_identity.py tests/benchmark/test_prediction_artifact.py -q
```

Expected: new MuScriptor cases FAIL while existing OaF cases pass.

- [ ] **Step 4: Extend header metadata validation explicitly**

Add:

```python
MUSCRIPTOR_METADATA_SCHEMA = "muscriptor-note-start-metadata-v1"
NATIVE_METADATA_SCHEMAS = {
    OAF_METADATA_SCHEMA: {"upstream_8hit_group_id": OAF_GROUP_IDS | {None}},
    MUSCRIPTOR_METADATA_SCHEMA: {"instrument_group": {"drums"}},
}
```

In `_build_header()`:

1. keep exact descriptor key-set/hash checks;
2. call `normalize_known_backend_descriptor(descriptor_payload)`;
3. remove the redundant `backend_id must be the OaF backend` check;
4. require `prediction_schema == PREDICTION_SCHEMA`;
5. require the descriptor's metadata schema to exist in `NATIVE_METADATA_SCHEMAS`.

- [ ] **Step 5: Dispatch native-event invariants by backend**

Change normalization to receive the validated backend ID:

```python
def _normalize_event(
    event: MappedPredictionEvent,
    backend_id: str,
    metadata_schema: str,
) -> _NormalizedEvent:
    ...
```

OaF branch preserves current checks exactly. MuScriptor branch requires:

```python
native.model_output_bin is None
native.native_midi_note is not None
native.confidence is None
native.velocity_midi is None
native.native_class_id == f"drums:midi_{native.native_midi_note}"
native.native_metadata == {"instrument_group": "drums"}
```

Do not weaken OaF nullability globally.

- [ ] **Step 6: Close the zero-hit cohort seam explicitly**

Change the internal artifact-identity helper to receive the expected cohort identity:

```python
def _artifact_identity_from_artifacts(
    identity: CohortIdentity,
    simfile_id: str,
    reference: ReferenceMappingResult,
    prediction: PredictionArtifact,
) -> CohortArtifactIdentity:
    ...
```

For an empty event artifact, use `identity.prediction_map_version` after descriptor/audio validation. Add a zero-hit MuScriptor success test.

- [ ] **Step 7: Keep the existing schema golden untouched**

Do **not** append MuScriptor to `crux.drum-prediction-events-v2.jsonl` and do not add a duplicate registry row. `test_prediction_artifact.py` owns MuScriptor canonical coverage; the existing golden proves unchanged OaF bytes.

- [ ] **Step 8: Run focused regression**

```bash
uv run pytest tests/benchmark/test_backend_identity.py tests/benchmark/test_prediction_artifact.py tests/benchmark/test_cohort_scoring.py tests/benchmark/test_cohort_scoring_acceptance.py tests/benchmark/test_schema_goldens.py -q
```

Expected: PASS; OaF golden bytes are unchanged.

- [ ] **Step 9: Commit**

```bash
git add src/benchmark/backend_identity.py src/benchmark/prediction_artifact.py src/benchmark/cohort_scoring.py tests/benchmark/test_backend_identity.py tests/benchmark/test_prediction_artifact.py tests/benchmark/test_cohort_scoring.py
git commit -m "feat: support MuScriptor prediction identity"
```

---

### Task 4: Add the direct MuScriptor adapter and freeze the exact pitch map

**Files:**
- Create: `src/benchmark/backends/muscriptor.py`
- Modify: `src/benchmark/taxonomy.py`
- Modify: `src/benchmark/mapping.py`
- Create: `tests/benchmark/test_muscriptor_backend.py`
- Modify: `tests/benchmark/test_mapping.py`

**Frozen mapping identity:**

```python
MUSCRIPTOR_PREDICTION_MAP_ID = "crux.prediction-map/muscriptor-drums-v1"
```

**Frozen v1 pitch table:**

```python
MUSCRIPTOR_MIDI_NOTE_MAP: Mapping[int, ClassMapping] = MappingProxyType(
    {
        35: ClassMapping("kick", "kick"),
        36: ClassMapping("kick", "kick"),
        38: ClassMapping("snare", "snare"),
        40: ClassMapping("snare", "snare"),
        41: ClassMapping("low_or_floor_tom", "tom"),
        43: ClassMapping("low_or_floor_tom", "tom"),
        45: ClassMapping("low_or_floor_tom", "tom"),
        47: ClassMapping("low_or_floor_tom", "tom"),
        48: ClassMapping("high_tom", "tom"),
        50: ClassMapping("high_tom", "tom"),
        42: ClassMapping("closed_hihat", "hihat"),
        44: ClassMapping("closed_hihat", "hihat"),
        46: ClassMapping("open_hihat", "hihat"),
        49: ClassMapping("crash", "crash"),
        57: ClassMapping("crash", "crash"),
        51: ClassMapping("ride", "ride"),
        53: ClassMapping("ride", "ride"),
        59: ClassMapping("ride", "ride"),
    }
)
```

This contains the existing `DEFAULT_MIDI_NOTE_MAP` notes plus clear GM aliases 35, 40, 43, 44, 48, 53, 57, and 59. All other pitches are unmapped in v1; do not silently add side-stick, clap, splash/Chinese cymbal, cowbell, tambourine, or other percussion.

- [ ] **Step 1: Write red table identity tests**

```python
def test_muscriptor_pitch_map_is_frozen_v1() -> None:
    assert MUSCRIPTOR_PREDICTION_MAP.map_id == "crux.prediction-map/muscriptor-drums-v1"
    assert set(MUSCRIPTOR_MIDI_NOTE_MAP) == {
        35, 36, 38, 40, 41, 43, 45, 47, 48, 50, 42, 44, 46, 49, 57, 51, 53, 59
    }
```

Also assert `37`, `39`, `52`, `55`, and `56` are not mapped.

- [ ] **Step 2: Write fake upstream event/model fixtures**

Model only public v0.3.0 fields needed by Crux: `NoteStartEvent`, `NoteEndEvent`, `ProgressEvent`, `load_model()`, and `transcribe()`.

- [ ] **Step 3: Add red adapter tests**

Assert one model is loaded from the verified local safetensors path and `transcribe()` is called with exactly the frozen drum-only arguments. Only `NoteStartEvent` creates a `NativeEvent`; pitch/onset are preserved; no confidence/velocity/bin is fabricated. Reject non-drum events, nonfinite/negative onsets, and pitch outside 0..127.

- [ ] **Step 4: Add persist-unmapped MuScriptor mapping**

Implement:

```python
def map_muscriptor_prediction(
    prediction: NativePrediction,
    prediction_map: PredictionMap = MUSCRIPTOR_PREDICTION_MAP,
) -> tuple[MappedPrediction, MappingDiagnostics]:
    ...
```

For each native event:

```python
key = str(native.native_midi_note)
class_mapping = prediction_map.classes.get(key)
is_mapped = class_mapping is not None and class_mapping.common_class is not None
```

Append every native event. Unsupported pitches produce `mapping_status="unmapped"`, null classes, and an unmapped count keyed by pitch. Do **not** reuse `map_midi_events()`, which drops unmapped notes.

- [ ] **Step 5: Implement the lazy optional backend**

Import MuScriptor only inside backend construction. Verify installed package version, model lock, local checkpoint, device/dtype, and descriptor before first song. `close()` only releases local model state; do not add a process solely for teardown.

- [ ] **Step 6: Run focused tests**

```bash
uv run pytest tests/benchmark/test_muscriptor_backend.py tests/benchmark/test_mapping.py tests/benchmark/test_prediction_artifact.py -q
```

Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add src/benchmark/backends/muscriptor.py src/benchmark/taxonomy.py src/benchmark/mapping.py tests/benchmark/test_muscriptor_backend.py tests/benchmark/test_mapping.py
git commit -m "feat: map MuScriptor drum events"
```

**Hard gate:** Do not run real MuScriptor inference until this exact map ID/table is committed.

---

### Task 5: Promote the remaining two-caller helpers and add the concrete MuScriptor runner

**Files:**
- Modify: `src/benchmark/corpus_cache.py`
- Modify: `src/benchmark/prediction_artifact.py`
- Modify: `src/benchmark/oaf_corpus_run.py`
- Create: `src/benchmark/muscriptor_corpus_run.py`
- Modify: `tests/benchmark/test_corpus_cache.py`
- Modify: `tests/benchmark/test_prediction_artifact.py`
- Modify: `tests/benchmark/test_oaf_corpus_run.py`
- Create: `tests/benchmark/test_muscriptor_corpus_run.py`
- Create: `tests/benchmark/test_muscriptor_corpus_run_acceptance.py`
- Verify unchanged: `tests/benchmark/test_oaf_corpus_run_acceptance.py`

**Contracts:**

```text
crux.muscriptor-corpus-run/v1
crux.muscriptor-inference-config/v1
```

**Required neutral interfaces:**

```python
# corpus_cache.py
@dataclass(frozen=True)
class ResolvedSourceAudio:
    path: Path
    source_audio_id: str
    source_audio_sha256: str
    duration_sec: float
    content: bytes | None = None


def resolve_source_audio(..., load_body: bool = True) -> ResolvedSourceAudio: ...

# prediction_artifact.py
def prediction_path(
    output_dir: Path,
    *,
    simfile_id: int,
    source_audio_sha256: str,
    backend_descriptor_sha256: str,
    inference_config_sha256: str,
) -> Path: ...
```

- [ ] **Step 1: Characterize the current source resolver and prediction path**

Before moving anything, add tests that pin:

- verified-cache resolution;
- fallback from manifest `cache_status != verified` through the cache index;
- exact source digest mismatch failure;
- `load_body=False` retaining path/duration without corpus bytes;
- exact immutable prediction path layout.

```python
def test_prediction_path_contract(tmp_path: Path) -> None:
    assert prediction_path(tmp_path, simfile_id=7, source_audio_sha256="a" * 64,
        backend_descriptor_sha256="b" * 64, inference_config_sha256="c" * 64) == (
        tmp_path / "predictions" / "7" / ("a" * 64) / ("b" * 64) / f"{'c' * 64}.jsonl"
    )
```

Run the OaF tests before the move to prove the existing behavior is green.

- [ ] **Step 2: Move source resolution to `corpus_cache.py`**

Move `ResolvedSourceAudio`, `_remote_from_source_mapping()`, `_source_audio_parts()`, and `_resolve_source_audio()` without semantic edits. Rename only the public entrypoint:

```python
resolve_source_audio(...)
```

OaF imports it from `corpus_cache.py`. No alias remains in `oaf_corpus_run.py`.

This is required: MuScriptor must not import `oaf_corpus_run.py` merely to resolve HPA-321 cache bytes, because that module imports the OaF runtime at module load.

- [ ] **Step 3: Move `prediction_path()` to `prediction_artifact.py`**

Move the existing function unchanged. OaF and MuScriptor import one path convention from the artifact module. Delete the OaF-local copy; no compatibility alias.

- [ ] **Step 4: Run extraction regressions before writing the new runner**

```bash
uv run pytest tests/benchmark/test_corpus_cache.py tests/benchmark/test_prediction_artifact.py tests/benchmark/test_oaf_corpus_run.py tests/benchmark/test_oaf_corpus_run_acceptance.py -q
```

Expected: PASS with unchanged OaF path/cache behavior.

- [ ] **Step 5: Add red MuScriptor run-ID/config/snapshot tests**

Bind reference/timing identities, model-lock SHA, checkpoint revision/SHA, descriptor/config hashes, adapter revision, `MUSCRIPTOR_PREDICTION_MAP_ID`, shared input-view ID, decode settings, device/dtype, and include/exclude scope. Complete snapshots require reconciled item dispositions/counts.

- [ ] **Step 6: Add red execution/resume tests with an injected fake backend**

Prove:

- lineage/scope/model/checkpoint/map checks happen before backend creation;
- source bytes are resolved in the first pass and re-pinned immediately before materialization;
- `materialize_full_mix_audio()` is used;
- one backend instance is lazily created and reused;
- prediction path comes from `prediction_artifact.prediction_path()`;
- immutable prediction artifacts persist;
- item-local inference failures continue;
- model/descriptor/config failures stop;
- resume reuses exact artifacts and rejects edited persisted evidence.

- [ ] **Step 7: Implement the concrete runner**

Follow HPA-326's semantic order without extracting a generic base class:

```text
preflight
-> durable snapshot
-> verified source pass
-> per-item re-pin/materialize/resume-or-infer/map/publish/checkpoint
-> close
-> runtime evidence
-> HPA-325 broad scoring/reports
```

Do not copy the promoted source resolver or prediction-path code back into the new runner.

- [ ] **Step 8: Record cheap runtime evidence**

Persist device, dtype, measured wall time/audio duration, aggregate RTF, projected wall time when duration coverage is complete, and peak process RSS where cheaply available. Accelerator-specific peak memory is optional evidence when directly available; no telemetry subsystem.

- [ ] **Step 9: Run MuScriptor + OaF regression tests**

```bash
uv run pytest tests/benchmark/test_muscriptor_corpus_run.py tests/benchmark/test_muscriptor_corpus_run_acceptance.py tests/benchmark/test_corpus_cache.py tests/benchmark/test_prediction_artifact.py tests/benchmark/test_oaf_corpus_run.py tests/benchmark/test_oaf_corpus_run_acceptance.py -q
```

Expected: PASS.

- [ ] **Step 10: Commit**

```bash
git add src/benchmark/corpus_cache.py src/benchmark/prediction_artifact.py src/benchmark/oaf_corpus_run.py src/benchmark/muscriptor_corpus_run.py tests/benchmark/test_corpus_cache.py tests/benchmark/test_prediction_artifact.py tests/benchmark/test_oaf_corpus_run.py tests/benchmark/test_muscriptor_corpus_run.py tests/benchmark/test_muscriptor_corpus_run_acceptance.py
git commit -m "feat: run persisted MuScriptor corpus benchmark"
```

---

### Task 6: Share reviewed-subset filtering and add MuScriptor CLI

**Files:**
- Modify: `src/benchmark/reviewed_subset.py`
- Modify: `src/cli/benchmark.py`
- Modify: `tests/benchmark/test_reviewed_subset.py`
- Modify: `tests/benchmark/test_reviewed_subset_acceptance.py`
- Modify: `tests/test_cli_benchmark.py`

- [ ] **Step 1: Characterize existing `score_oaf_reviewed_subset()`**

Capture selected IDs, subset cohort ID, diagnostics IDs, exit codes, and report behavior for success plus failed selected rows.

- [ ] **Step 2: Extract only the model-neutral selection/scoring core**

```python
def score_reviewed_subset_cohort(
    parent_identity: CohortIdentity,
    parent_items: tuple[CohortItem, ...],
    subset: LoadedReviewedSubsetManifest,
    *,
    output_dir: Path,
) -> ScoreReviewedSubsetOutcome:
    ...
```

It filters exact HPA-327 membership, rejects missing members, derives subset cohort ID from parent cohort + subset manifest hash, and calls HPA-325 with diagnostics for successful selected IDs. OaF keeps its existing run parsing/lineage checks and delegates to this core.

- [ ] **Step 3: Add `score_muscriptor_reviewed_subset()`**

Parse the MuScriptor run, validate the same HPA-323/HPA-324/HPA-327 lineage, reconstruct persisted MuScriptor `CohortItem`s, then delegate to the same core. No inference.

- [ ] **Step 4: Add CLI tests before commands**

In `tests/test_cli_benchmark.py`, assert only these new commands:

```text
crux benchmark run-muscriptor-corpus
crux benchmark score-muscriptor-reviewed-subset
```

`run-muscriptor-corpus` mirrors HPA-326 manifest/timing/cache/output/include/exclude/resume arguments. Assert the help output has no model-size, temperature, beam, instrument, mapping-version, device, or dtype flags.

- [ ] **Step 5: Implement lazy-import CLI commands**

Keep imports inside command functions so normal CLI import does not require MuScriptor/PyTorch.

- [ ] **Step 6: Run focused tests**

```bash
uv run pytest tests/benchmark/test_reviewed_subset.py tests/benchmark/test_reviewed_subset_acceptance.py tests/benchmark/test_muscriptor_corpus_run.py tests/test_cli_benchmark.py -q
```

Expected: PASS; existing OaF CLI output is unchanged.

- [ ] **Step 7: Commit**

```bash
git add src/benchmark/reviewed_subset.py src/cli/benchmark.py tests/benchmark/test_reviewed_subset.py tests/benchmark/test_reviewed_subset_acceptance.py tests/test_cli_benchmark.py
git commit -m "feat: score MuScriptor reviewed subset"
```

---

### Task 7: Add narrow identity-matched OaF-vs-MuScriptor deltas

**Files:**
- Create: `src/benchmark/muscriptor_comparison.py`
- Modify: `src/cli/benchmark.py`
- Create: `tests/benchmark/test_muscriptor_comparison.py`
- Modify: `tests/test_cli_benchmark.py`

**Outputs:**

```text
paired_per_song.csv
summary.json
summary.md
```

- [ ] **Step 1: Add red exact-pairing tests**

A song is pairable only when both persisted items succeed and match:

```text
simfile_id
HPA-324 reference manifest identity
HPA-323 timing identity
source_audio_sha256
input_view_id
input_audio_sha256
```

If the same simfile/source-audio identity has a different `input_view_id` or `input_audio_sha256`, raise a comparison integrity error rather than silently excluding it. This closes the ticket's byte-identical-input claim.

- [ ] **Step 2: Add red delta-output tests**

Rebuild both HPA-325 cohorts from persisted runs and emit deterministic rows:

```text
simfile_id,tolerance_ms,mode,oaf_f1,muscriptor_f1,delta_f1
```

with:

```python
delta_f1 = muscriptor_f1 - oaf_f1
```

`summary.json` contains paired count, excluded/failing population counts for both models, and mean/median delta per tolerance/mode. Optional reviewed-subset filtering uses the exact HPA-327 manifest.

No significance tests, rankings, bootstrap intervals, or generic N-model structures.

- [ ] **Step 3: Add `compare-oaf-muscriptor` CLI test and command**

`tests/test_cli_benchmark.py` pins:

```text
--oaf-run
--muscriptor-run
--manifest
--timing-manifest
--output-dir
--subset-manifest (optional)
```

The command never invokes inference.

- [ ] **Step 4: Run focused tests**

```bash
uv run pytest tests/benchmark/test_muscriptor_comparison.py tests/benchmark/test_cohort_scoring.py tests/test_cli_benchmark.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/benchmark/muscriptor_comparison.py src/cli/benchmark.py tests/benchmark/test_muscriptor_comparison.py tests/test_cli_benchmark.py
git commit -m "feat: compare OaF and MuScriptor cohorts"
```

---

### Task 8: Run the frozen smoke gate, broad corpus, reviewed subset, and final regression

**Operational evidence:** local MuScriptor run/predictions/reports and paired reports; large model/artifact bytes remain outside Git.

- [ ] **Step 1: Install/authenticate and verify the exact frozen local checkpoint**

```bash
uv sync --extra muscriptor
uvx hf auth whoami
```

Run the freeze/checkpoint verifier. If gating/license acceptance, package identity, revision, config, or exact hash verification fails, stop rather than bypassing the official source.

- [ ] **Step 2: Assert the pitch map is frozen before first real inference**

Run:

```bash
uv run pytest tests/benchmark/test_mapping.py -k muscriptor -q
```

Require the exact v1 map ID/table from Task 4. Do not run the real model against an edited or unversioned table.

- [ ] **Step 3: Run only the five committed smoke IDs**

Verify drum-only output, native onset/pitch validity, repeatability on at least one repeated song, sane behavior across five-second chunk boundaries on the long song, resume identity, wall time/RTF, and memory evidence.

Do not use smoke F1 to tune decoding or mapping.

- [ ] **Step 4: Freeze medium vs small from feasibility only**

If medium is stable/practical, keep it. If not, regenerate the lock once for small and rerun smoke from scratch. Do not inspect reference-score deltas to choose.

- [ ] **Step 5: Run the full technically eligible HPA-324 population**

Use the exact immutable HPA-324 manifest, matching HPA-323 timing manifest, and verified HPA-321 cache with `run-muscriptor-corpus`. Do not curate by MuScriptor outcome.

- [ ] **Step 6: Verify canonical-input equality against the persisted OaF run before publishing comparisons**

For every overlapping successful same-source song, compare persisted rows and require:

```python
muscriptor_row["source_audio_sha256"] == oaf_row["source_audio_sha256"]
muscriptor_row["input_view_id"] == oaf_row["input_view_id"]
muscriptor_row["input_audio_sha256"] == oaf_row["input_audio_sha256"]
```

Any input-view/hash mismatch is fatal evidence that the shared materialization contract failed. Do not publish paired deltas until the mismatch count is exactly zero.

- [ ] **Step 7: Score the accepted HPA-327 subset from persisted predictions**

Use `score-muscriptor-reviewed-subset`; do not rerun inference.

- [ ] **Step 8: Produce broad and reviewed OaF pair deltas**

Run `compare-oaf-muscriptor` against the matching persisted HPA-326 OaF run once for the broad population and once with the HPA-327 subset filter.

- [ ] **Step 9: Verify acceptance evidence**

Confirm:

- 30/50/100 ms event-micro, song-macro, class-macro, per-class results;
- complete MuScriptor success/failure population;
- native mapped/unmapped pitch coverage;
- exact v1 map ID in persisted prediction identity;
- reviewed event diagnostics;
- paired deltas and intersection count;
- zero same-source canonical-input mismatches;
- runtime/device/dtype/memory;
- exact model/package/checkpoint/config hashes;
- separate code/weight licenses.

- [ ] **Step 10: Run full regression/static checks**

```bash
uv run pytest tests/benchmark -q
uv run pytest -q
uv run ruff check .
uv run ruff format --check src tests scripts
uv run pylint --errors-only --disable=E1120,E0401 --jobs=1 src

git diff --check main...HEAD
```

Re-run unchanged OaF prediction schema golden plus HPA-325/HPA-326 acceptance tests before claiming completion.

- [ ] **Step 11: Commit operational reports/config evidence that belongs in Git**

Commit only small lock/smoke/config/report metadata explicitly required by the ticket. Keep checkpoint bytes, source audio, large predictions, and large generated reports outside Git.

---

## Stop conditions

Do not publish scored results while any of these is unresolved:

- checkpoint/license/revision/hash is not exact;
- medium/small choice is not frozen from pre-score feasibility evidence;
- hard `instruments=["drums"]` is unavailable in pinned v0.3.0;
- `MUSCRIPTOR_PREDICTION_MAP_ID` or the exact v1 pitch table is not committed/frozen;
- a real run used a pitch table different from `crux.prediction-map/muscriptor-drums-v1`;
- shared canonical full-mix bytes differ from the current OaF output for the same source;
- a same-source OaF/MuScriptor row has a different input-view or input-audio hash;
- existing OaF prediction-v2 golden changes;
- HPA-323/HPA-324/HPA-327 lineage does not match;
- MuScriptor imports source-resolution/path helpers through `oaf_corpus_run.py` instead of the promoted neutral modules;
- source resolver or prediction-path logic is copied into the new runner;
- a proposed abstraction exists only for hypothetical future models rather than two current callers.

## Expected long-lived additions

Keep new production modules to the concrete ticket needs:

```text
src/benchmark/muscriptor_model.py
src/benchmark/backends/muscriptor.py
src/benchmark/muscriptor_corpus_run.py
src/benchmark/muscriptor_comparison.py
```

Existing modules receive narrow extensions/extractions only. If implementation starts creating a generic model registry, experiment object model, shared runner base class, or second report stack, stop and simplify.
