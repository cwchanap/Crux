# HPA-395 MuScriptor Full-Mix Benchmark Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Freeze one official MuScriptor checkpoint and execution contract, run hard drum-only native-event inference over the same canonical full mixes/references as OaF, score broad and HPA-327 reviewed populations through HPA-325, then publish exact-identity per-song and per-class OaF deltas from the already-published HPA-325 reports.

**Architecture:** Keep MuScriptor comparator-only. Add one optional adapter and one concrete MuScriptor corpus runner. Promote only three existing model-neutral functions that now have two real callers: verified source resolution, full-mix materialization, and immutable prediction-path construction. Extend prediction-v2 write/read validation for a second frozen backend family without weakening OaF. Reuse HPA-325 report artifacts for comparison instead of introducing a second scoring path.

**Tech Stack:** Python 3.12, `muscriptor==0.3.0`, optional PyTorch/MuScriptor dependency, Hugging Face Hub for authenticated freeze, existing librosa/soundfile canonicalization, Click, pytest, Crux canonical JSON/JSONL helpers, HPA-323/HPA-324/HPA-325/HPA-326/HPA-327 artifacts.

## Global constraints

- Freeze MuScriptor `v0.3.0` at release commit `d73147e75e5b9b0c0a79ebe154587db4fd603e0c`.
- Start with `medium`; switch once to `small` only for pre-score hardware feasibility, never F1.
- Scored weights are an exact verified local `model.safetensors`, never bare `"medium"`, `"small"`, or moving `hf://` aliases.
- The `crux.muscriptor-model/v1` lock retains the deterministic settings required by HPA-395: device/dtype, preprocessing/chunk facts, hard drum constraint, sampling/temperature/CFG/beam/batch/prelude settings, and code/weight licenses.
- Model ID grammar is `muscriptor-(medium|small)-<revision12>-<checkpoint12>` with lowercase hex fragments derived from the frozen lock.
- Consume raw `NoteStartEvent.start_time` + native MIDI pitch; `NoteEndEvent` and `ProgressEvent` create no hit.
- Reuse the historical `crux.oaf-full-mix-mono44k1-pcm16/v1` input-view ID and exact OaF canonical WAV bytes.
- Preserve the current materializer's output-beneath-`input_root` guard.
- Keep `crux.drum-prediction-events/v2`; both `_build_header()` and `_validate_header()` dispatch through the frozen backend family. OaF invariants and golden stay unchanged.
- Zero-hit prediction-map identity is derived independently by `backend_id -> prediction_map_id`, not copied from expected cohort identity.
- Freeze one string-keyed `MUSCRIPTOR_PREDICTION_MAP`; unsupported pitches remain persisted as unmapped events.
- Do not add MuScriptor to `default_backend_registry()`.
- CI remains offline with fake MuScriptor objects.
- Keep two concrete runners; no generic runner/base class, queue, worker protocol, DB, mapping DSL, experiment framework, second scorer, training, or stem separation.
- Comparison reads HPA-325 `per_song.csv` and `per_class.csv`; it does not re-score persisted predictions.
- `input_view_id` is run-level identity. `source_audio_sha256` and `input_audio_sha256` are successful item-level identity.
- Extend `tests/test_cli_benchmark.py`; no second CLI harness.

---

### Task 1: Freeze the optional package, checkpoint, model ID, and smoke contract

**Files:**
- Modify: `pyproject.toml`
- Modify: `uv.lock`
- Create: `src/benchmark/muscriptor_model.py`
- Create: `scripts/freeze_muscriptor_model.py`
- Create during authenticated preflight: `runtime/muscriptor/model.json`
- Create before first real inference: `runtime/muscriptor/smoke.json`
- Create: `tests/benchmark/test_muscriptor_model.py`

**Interfaces:**

```python
MUSCRIPTOR_MODEL_SCHEMA = "crux.muscriptor-model/v1"
MUSCRIPTOR_RELEASE_COMMIT = "d73147e75e5b9b0c0a79ebe154587db4fd603e0c"
MUSCRIPTOR_MODEL_ID_RE = re.compile(
    r"muscriptor-(medium|small)-[0-9a-f]{12}-[0-9a-f]{12}\Z"
)

@dataclass(frozen=True)
class MuscriptorModelLock:
    package_name: str
    package_version: str
    upstream_source_commit: str
    code_license: str
    weight_license: str
    checkpoint_variant: Literal["medium", "small"]
    checkpoint_repo_id: str
    checkpoint_revision: str
    checkpoint_filename: str
    checkpoint_sha256: str
    checkpoint_byte_length: int
    checkpoint_config_filename: str
    checkpoint_config_sha256: str
    checkpoint_config_byte_length: int
    model_id: str
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
def derive_muscriptor_model_id(lock: MuscriptorModelLock) -> str: ...
def verify_muscriptor_checkpoint(lock: MuscriptorModelLock, checkpoint_dir: Path) -> Path: ...
```

- [ ] **Step 1: Write strict model-lock tests first**

Cover the exact schema/key set and these invariants:

```python
assert lock.package_name == "muscriptor"
assert lock.package_version == "0.3.0"
assert lock.upstream_source_commit == MUSCRIPTOR_RELEASE_COMMIT
assert lock.checkpoint_variant in {"medium", "small"}
assert re.fullmatch(r"[0-9a-f]{40}", lock.checkpoint_revision)
assert derive_muscriptor_model_id(lock) == (
    f"muscriptor-{lock.checkpoint_variant}-"
    f"{lock.checkpoint_revision[:12]}-{lock.checkpoint_sha256[:12]}"
)
assert lock.model_id == derive_muscriptor_model_id(lock)
assert lock.instruments == ("drums",)
assert lock.use_sampling is False
assert lock.temperature == 1.0
assert lock.cfg_coef == 1.0
assert lock.batch_size == 1
assert lock.no_eos_is_ok is True
assert lock.beam_size == 1
assert lock.prelude_forcing is True
assert lock.input_sample_rate_hz == 16000
assert lock.chunk_duration_sec == 5.0
```

Also reject edited/missing safetensors/config files, invalid hashes/sizes/licenses/device/dtype, and a model ID whose digest fragments do not match the lock.

- [ ] **Step 2: Verify red**

```bash
uv run pytest tests/benchmark/test_muscriptor_model.py -q
```

Expected: FAIL because the module does not exist.

- [ ] **Step 3: Add the optional dependency**

```toml
[project.optional-dependencies]
muscriptor = ["muscriptor==0.3.0"]
```

Then:

```bash
uv lock
```

If lock resolution fails because MuScriptor cannot coexist with Python 3.12, stop this path and design the smallest subprocess runtime. Do not add it preemptively.

- [ ] **Step 4: Implement the strict lock/parser/verifier**

Use existing `strict_json_loads()`, `require_sha256()`, canonical JSON helpers, and safe regular-file readers. Unknown fields fail closed.

The lock must retain the deterministic settings listed in `Global constraints`; do not replace them with code-only constants because HPA-395 explicitly requires them as lock evidence.

- [ ] **Step 5: Implement the authenticated freeze script**

The script must:

1. resolve `MuScriptor/muscriptor-<variant>` to an exact Hugging Face revision;
2. download `model.safetensors` and `config.json` with `revision=<exact_sha>`;
3. hash/size both files;
4. verify official code and weight licenses;
5. resolve `--device auto` / `--dtype auto` to exact persisted values;
6. derive the exact `model_id` grammar above;
7. write canonical `runtime/muscriptor/model.json`.

The script never invokes HPA-325 scoring.

- [ ] **Step 6: Add strict smoke-manifest tests and file**

`runtime/muscriptor/smoke.json` contains exactly five unique positive simfile IDs with pre-model reasons:

```text
short
long
dense
sparse
non_drum_heavy
```

Selection may use HPA-323/HPA-324/HPA-327 evidence and source listening only.

- [ ] **Step 7: Run offline tests**

```bash
uv run pytest tests/benchmark/test_muscriptor_model.py -q
```

Expected: PASS without network access.

- [ ] **Step 8: Exercise one real authenticated freeze before relying on upstream assumptions**

```bash
uv run --extra muscriptor python scripts/freeze_muscriptor_model.py \
  --variant medium \
  --device auto \
  --dtype auto \
  --checkpoint-dir artifacts/muscriptor/checkpoint \
  --output runtime/muscriptor/model.json
```

Then verify the generated lock and local bytes:

```bash
uv run pytest tests/benchmark/test_muscriptor_model.py -q
```

If Hugging Face authentication/license acceptance is unavailable, record the operational block. Offline implementation may continue, but no real inference/scoring is allowed until this step succeeds.

- [ ] **Step 9: Commit**

```bash
git add pyproject.toml uv.lock src/benchmark/muscriptor_model.py scripts/freeze_muscriptor_model.py runtime/muscriptor tests/benchmark/test_muscriptor_model.py
git commit -m "feat: freeze MuScriptor benchmark identity"
```

---

### Task 2: Promote the three model-neutral execution seams with OaF characterization

**Files:**
- Modify: `src/benchmark/corpus_cache.py`
- Modify: `src/benchmark/input_view.py`
- Modify: `src/benchmark/prediction_artifact.py`
- Modify: `src/benchmark/oaf_corpus_run.py`
- Modify: `tests/benchmark/test_corpus_cache.py`
- Modify: `tests/benchmark/test_input_view.py`
- Modify: `tests/benchmark/test_prediction_artifact.py`
- Modify: `tests/benchmark/test_oaf_corpus_run.py`
- Modify: `tests/benchmark/test_oaf_corpus_run_branches.py`
- Verify unchanged: `tests/benchmark/test_oaf_corpus_run_acceptance.py`

**Interfaces:**

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

# input_view.py
def materialize_full_mix_audio(
    source_audio: ResolvedSourceAudio,
    output_path: Path,
    *,
    input_root: Path,
    input_view_id: str,
    max_input_audio_frames: int | None,
) -> CanonicalAudio: ...

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

- [ ] **Step 1: Characterize all three current seams before moving them**

Add/retain tests for:

- verified source resolution;
- cache-index fallback for non-verified manifest rows;
- exact source digest mismatch;
- `load_body=False` path/duration behavior;
- exact prediction path layout;
- exact canonical WAV bytes/hash/metadata;
- source type rejection;
- output/input-root `Path` type rejection;
- output escaping `input_root` rejection.

Keep the existing OaF config type test at the OaF call-site boundary; the neutral helper receives only `max_input_audio_frames`.

- [ ] **Step 2: Run the exact materializer branch tests before refactor**

Do not use a loose `-k` selector. Run the named tests:

```bash
uv run pytest \
  tests/benchmark/test_oaf_corpus_run_branches.py::test_materialize_rejects_non_resolved_source \
  tests/benchmark/test_oaf_corpus_run_branches.py::test_materialize_rejects_non_path_args \
  tests/benchmark/test_oaf_corpus_run_branches.py::test_materialize_rejects_non_model_config \
  tests/benchmark/test_oaf_corpus_run_branches.py::test_materialize_rejects_output_outside_input_root \
  -q
```

Expected: PASS before the move.

- [ ] **Step 3: Move verified source resolution to `corpus_cache.py`**

Move without semantic changes:

```text
ResolvedSourceAudio
_remote_from_source_mapping()
_source_audio_parts()
_resolve_source_audio()
```

Rename only the public entrypoint to `resolve_source_audio()`. OaF imports it from `corpus_cache.py`; no compatibility alias stays in `oaf_corpus_run.py`.

- [ ] **Step 4: Move prediction-path construction to `prediction_artifact.py`**

Move `prediction_path()` unchanged and update OaF imports. No path strategy object.

- [ ] **Step 5: Promote canonical materialization with `input_root` intact**

Move the neutral source/path/type/containment and conversion logic to `input_view.materialize_full_mix_audio()`.

The implementation must still contain the equivalent of:

```python
root = input_root.resolve()
destination = output_path.resolve()
try:
    destination.relative_to(root)
except ValueError:
    raise ValueError("canonical input must be beneath input_root") from None
```

and the exact conversion:

```python
samples, _ = librosa.load(source_input, sr=44100, mono=True, res_type="soxr_hq")
soundfile.write(output_path, samples, 44100, format="WAV", subtype="PCM_16")
```

OaF still validates `OafModelConfig` at its own boundary, then passes `config.max_input_audio_frames`.

- [ ] **Step 6: Update branch tests to import the promoted public seams**

`tests/benchmark/test_oaf_corpus_run_branches.py` currently imports all moved privates directly. Update those imports/tests as part of the extraction rather than discovering them later.

- [ ] **Step 7: Run extraction regressions**

```bash
uv run pytest \
  tests/benchmark/test_corpus_cache.py \
  tests/benchmark/test_input_view.py \
  tests/benchmark/test_prediction_artifact.py \
  tests/benchmark/test_oaf_corpus_run.py \
  tests/benchmark/test_oaf_corpus_run_branches.py \
  tests/benchmark/test_oaf_corpus_run_acceptance.py \
  -q
```

Expected: PASS with unchanged OaF canonical bytes, cache semantics, prediction path, and containment behavior.

- [ ] **Step 8: Commit**

```bash
git add src/benchmark/corpus_cache.py src/benchmark/input_view.py src/benchmark/prediction_artifact.py src/benchmark/oaf_corpus_run.py tests/benchmark/test_corpus_cache.py tests/benchmark/test_input_view.py tests/benchmark/test_prediction_artifact.py tests/benchmark/test_oaf_corpus_run.py tests/benchmark/test_oaf_corpus_run_branches.py
git commit -m "refactor: share benchmark input artifact rails"
```

---

### Task 3: Extend descriptor, prediction-v2 write/read validation, and zero-hit identity

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

**Frozen family constants:**

```python
MUSCRIPTOR_BACKEND_ID = "muscriptor-v0.3.0-drums-v1"
MUSCRIPTOR_DESCRIPTOR_SCHEMA = "crux.transcription-backend-descriptor/v2"
MUSCRIPTOR_MODEL_ID_RE = re.compile(
    r"muscriptor-(medium|small)-[0-9a-f]{12}-[0-9a-f]{12}\Z"
)
```

- [ ] **Step 1: Add red descriptor-family tests**

Prove:

- existing OaF descriptors still require exact constants;
- MuScriptor accepts only the exact key set/constants plus patterned `model_id`;
- mixed OaF/MuScriptor identities reject;
- invalid model-ID variant/hex lengths reject;
- the runner-facing helper can derive the exact expected model ID from a lock.

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

Assert render -> read -> render byte identity.

Reject:

```text
non-null model_output_bin
missing/out-of-range MIDI pitch
non-null confidence
non-null velocity
non-drum metadata
native_class_id/pitch mismatch
```

Existing OaF nullability/identity tests must remain unchanged.

- [ ] **Step 3: Verify red**

```bash
uv run pytest tests/benchmark/test_backend_identity.py tests/benchmark/test_prediction_artifact.py -q
```

Expected: new MuScriptor cases fail while OaF cases remain green.

- [ ] **Step 4: Implement explicit per-family descriptor validation**

Use one small policy lookup, not a registry framework:

```python
# conceptual shape
backend_id -> exact_keys + exact_identities + pattern_fields
```

OaF has no patterned fields. MuScriptor pattern-checks only `model_id`; later runner preflight checks exact lock-derived equality.

- [ ] **Step 5: Fix both prediction-v2 header paths**

Add:

```python
MUSCRIPTOR_METADATA_SCHEMA = "muscriptor-note-start-metadata-v1"
NATIVE_METADATA_SCHEMAS[MUSCRIPTOR_METADATA_SCHEMA] = {
    "instrument_group": {"drums"},
}
```

In `_build_header()`:

- keep `normalize_known_backend_descriptor()`;
- remove the redundant `backend_id must be OaF` check;
- keep prediction-schema and metadata-schema validation.

In `_validate_header()`:

- keep `normalize_known_backend_descriptor()`;
- remove the second redundant `backend_id must be OaF` read-path check;
- **do not change** descriptor↔header equality checks for architecture/model/metadata/output/training/source fields.

- [ ] **Step 6: Dispatch native event invariants by backend**

OaF keeps every current rule.

MuScriptor requires:

```python
native.model_output_bin is None
native.native_midi_note is not None
native.confidence is None
native.velocity_midi is None
native.native_metadata == {"instrument_group": "drums"}
native.native_class_id == f"drums:midi_{native.native_midi_note}"
```

Mapping-status/common-class validation remains common.

- [ ] **Step 7: Preserve independent zero-hit map validation**

Do **not** change `_artifact_identity_from_artifacts()` to accept `CohortIdentity`.

Instead add:

```python
ZERO_HIT_PREDICTION_MAPS = {
    OAF_BACKEND_ID: OAF_PREDICTION_MAP_ID,
    MUSCRIPTOR_BACKEND_ID: MUSCRIPTOR_PREDICTION_MAP_ID,
}
```

Use it only when the event-map set is empty. Unknown backends keep the current loud error.

Add tests where an empty MuScriptor artifact succeeds only with `MUSCRIPTOR_PREDICTION_MAP_ID`, and a deliberately wrong cohort map still fails `_validate_artifact_identity()`.

- [ ] **Step 8: Keep the schema golden untouched**

No new manifest row and no mixed golden. MuScriptor coverage stays in `test_prediction_artifact.py`.

- [ ] **Step 9: Run regression tests**

```bash
uv run pytest \
  tests/benchmark/test_backend_identity.py \
  tests/benchmark/test_prediction_artifact.py \
  tests/benchmark/test_cohort_scoring.py \
  tests/benchmark/test_cohort_scoring_acceptance.py \
  tests/benchmark/test_schema_goldens.py \
  -q
```

Expected: PASS and existing OaF golden bytes unchanged.

- [ ] **Step 10: Commit**

```bash
git add src/benchmark/backend_identity.py src/benchmark/prediction_artifact.py src/benchmark/cohort_scoring.py tests/benchmark/test_backend_identity.py tests/benchmark/test_prediction_artifact.py tests/benchmark/test_cohort_scoring.py
git commit -m "feat: support MuScriptor prediction identity"
```

---

### Task 4: Add the direct MuScriptor adapter and one frozen string-keyed pitch map

**Files:**
- Create: `src/benchmark/backends/muscriptor.py`
- Modify: `src/benchmark/taxonomy.py`
- Modify: `src/benchmark/mapping.py`
- Create: `tests/benchmark/test_muscriptor_backend.py`
- Modify: `tests/benchmark/test_mapping.py`

**Map identity:**

```python
MUSCRIPTOR_PREDICTION_MAP_ID = "crux.prediction-map/muscriptor-drums-v1"
```

- [ ] **Step 1: Write fake upstream model/event fixtures**

Model only the public v0.3.0 seam Crux needs: `load_model()`, `transcribe()`, `NoteStartEvent`, `NoteEndEvent`, and `ProgressEvent`.

- [ ] **Step 2: Add red adapter tests**

Prove:

- one model loads from the verified local safetensors path;
- exact frozen transcribe arguments are used;
- only `NoteStartEvent` creates a hit;
- onset/pitch are preserved exactly;
- no confidence/velocity/bin is invented;
- non-drum events, nonfinite/negative onset, and MIDI outside `0..127` reject.

- [ ] **Step 3: Define exactly one `PredictionMap`**

Use string keys only:

```python
MUSCRIPTOR_PREDICTION_MAP = PredictionMap(
    map_id=MUSCRIPTOR_PREDICTION_MAP_ID,
    backend_id=MUSCRIPTOR_BACKEND_ID,
    native_output_space_id="muscriptor-drums-midi128-v1",
    classes=MappingProxyType({
        "35": ClassMapping("kick", "kick"),
        "36": ClassMapping("kick", "kick"),
        "38": ClassMapping("snare", "snare"),
        "40": ClassMapping("snare", "snare"),
        "41": ClassMapping("low_or_floor_tom", "tom"),
        "43": ClassMapping("low_or_floor_tom", "tom"),
        "45": ClassMapping("low_or_floor_tom", "tom"),
        "47": ClassMapping("low_or_floor_tom", "tom"),
        "48": ClassMapping("high_tom", "tom"),
        "50": ClassMapping("high_tom", "tom"),
        "42": ClassMapping("closed_hihat", "hihat"),
        "44": ClassMapping("closed_hihat", "hihat"),
        "46": ClassMapping("open_hihat", "hihat"),
        "49": ClassMapping("crash", "crash"),
        "57": ClassMapping("crash", "crash"),
        "51": ClassMapping("ride", "ride"),
        "53": ClassMapping("ride", "ride"),
        "59": ClassMapping("ride", "ride"),
    }),
)
```

Do not also create `MUSCRIPTOR_MIDI_NOTE_MAP` with integer keys.

- [ ] **Step 4: Add `map_muscriptor_prediction()` with the same identity guards as OaF**

Begin with:

```python
if prediction.descriptor.payload.get("backend_id") != prediction_map.backend_id:
    raise ValueError("prediction backend_id does not match prediction map")
if prediction.descriptor.payload.get("native_output_space_id") != (
    prediction_map.native_output_space_id
):
    raise ValueError("prediction native_output_space_id does not match prediction map")
```

Then:

```python
key = str(native.native_midi_note)
class_mapping = prediction_map.classes.get(key)
```

Always append an event. Missing mappings become `mapping_status="unmapped"` with null canonical/common classes.

- [ ] **Step 5: Implement lazy optional backend construction**

Import MuScriptor only inside construction. Verify package, lock, checkpoint, device/dtype, and exact lock-derived descriptor identity before model load.

- [ ] **Step 6: Test mapped aliases and unmapped coverage**

Include at least:

```text
35 -> kick
40 -> snare
44 -> closed_hihat/hihat
53 -> ride
57 -> crash
37 -> unmapped
55 -> unmapped
```

Confirm unmapped hits survive render/read and contribute native/unmapped coverage.

- [ ] **Step 7: Run focused tests**

```bash
uv run pytest tests/benchmark/test_muscriptor_backend.py tests/benchmark/test_mapping.py tests/benchmark/test_prediction_artifact.py -q
```

Expected: PASS.

- [ ] **Step 8: Commit**

```bash
git add src/benchmark/backends/muscriptor.py src/benchmark/taxonomy.py src/benchmark/mapping.py tests/benchmark/test_muscriptor_backend.py tests/benchmark/test_mapping.py
git commit -m "feat: adapt MuScriptor drum events"
```

---

### Task 5: Add the concrete persisted MuScriptor corpus runner

**Files:**
- Create: `src/benchmark/muscriptor_corpus_run.py`
- Create: `tests/benchmark/test_muscriptor_corpus_run.py`
- Create: `tests/benchmark/test_muscriptor_corpus_run_acceptance.py`
- Verify unchanged: `tests/benchmark/test_oaf_corpus_run.py`
- Verify unchanged: `tests/benchmark/test_oaf_corpus_run_branches.py`
- Verify unchanged: `tests/benchmark/test_oaf_corpus_run_acceptance.py`

**Contracts:**

```text
crux.muscriptor-corpus-run/v1
crux.muscriptor-inference-config/v1
```

- [ ] **Step 1: Add red run-config/run-ID/snapshot tests**

Bind:

```text
reference manifest hash/version
timing manifest hash/version
model-lock SHA
checkpoint revision/SHA
backend descriptor SHA
inference-config SHA
MUSCRIPTOR_PREDICTION_MAP_ID
shared input-view ID
adapter revision
Crux commit
include/exclude scope
```

Complete snapshots require unique sorted items and reconciled success/failure/skip/quarantine counts.

- [ ] **Step 2: Add red fake-backend execution tests**

Prove all fatal preflight happens before backend construction; source resolution happens in a first pass; each eligible item is re-pinned immediately before materialization; one backend is lazily constructed/reused; prediction artifacts are immutable; item-local failures continue; poisoned model state stops; and resume rejects edited evidence.

- [ ] **Step 3: Require the promoted seams**

The runner must import:

```python
from src.benchmark.corpus_cache import resolve_source_audio
from src.benchmark.input_view import materialize_full_mix_audio
from src.benchmark.prediction_artifact import prediction_path
```

Do not copy their implementations into the new runner.

- [ ] **Step 4: Implement the concrete sequential runner**

Follow:

```text
preflight
-> durable initial snapshot
-> source pass
-> per-item re-pin/materialize
-> resume-or-infer
-> map/publish/checkpoint
-> close
-> runtime evidence
-> HPA-325 broad score/report
```

No generic runner class or shared lifecycle engine.

- [ ] **Step 5: Preserve paired input evidence in run rows**

Successful/inferred/resumed rows persist at least:

```text
source_audio_id
source_audio_sha256
source_duration_sec
input_audio_sha256
prediction_path
prediction_artifact_sha256
wall_time_sec
rtf
```

`input_view_id` remains run-level/inference-config identity, matching HPA-326.

- [ ] **Step 6: Record cheap runtime evidence**

Persist device/dtype, measured wall/audio duration, aggregate RTF, projected wall time when duration coverage is complete, process peak RSS when cheap, and device peak memory only when directly available.

- [ ] **Step 7: Run MuScriptor + OaF regression tests**

```bash
uv run pytest \
  tests/benchmark/test_muscriptor_corpus_run.py \
  tests/benchmark/test_muscriptor_corpus_run_acceptance.py \
  tests/benchmark/test_oaf_corpus_run.py \
  tests/benchmark/test_oaf_corpus_run_branches.py \
  tests/benchmark/test_oaf_corpus_run_acceptance.py \
  -q
```

Expected: PASS.

- [ ] **Step 8: Commit**

```bash
git add src/benchmark/muscriptor_corpus_run.py tests/benchmark/test_muscriptor_corpus_run.py tests/benchmark/test_muscriptor_corpus_run_acceptance.py
git commit -m "feat: run persisted MuScriptor corpus benchmark"
```

---

### Task 6: Move subset lineage into the shared core and add MuScriptor CLI

**Files:**
- Modify: `src/benchmark/reviewed_subset.py`
- Modify: `src/cli/benchmark.py`
- Modify: `tests/benchmark/test_reviewed_subset.py`
- Modify: `tests/benchmark/test_reviewed_subset_acceptance.py`
- Modify: `tests/test_cli_benchmark.py`

**Core interface:**

```python
def score_reviewed_subset_cohort(
    parent_identity: CohortIdentity,
    parent_items: tuple[CohortItem, ...],
    reference: LoadedReferenceSetManifest,
    timing: LoadedReferenceTimingManifest,
    subset: LoadedReviewedSubsetManifest,
    *,
    output_dir: Path,
) -> ScoreReviewedSubsetOutcome: ...
```

- [ ] **Step 1: Characterize existing OaF subset scoring**

Pin current selected IDs, subset cohort ID, diagnostics IDs, success/failure exit behavior, and reports.

- [ ] **Step 2: Add red tests for model-neutral subset lineage**

The core rejects:

```python
subset.source_reference_manifest_sha256 != reference.manifest_sha256
subset.source_reference_manifest_version != reference.corpus_version
subset.source_timing_manifest_sha256 != timing.manifest_sha256
subset.source_timing_manifest_version != timing.corpus_version
```

and rejects missing subset members from `parent_items`.

- [ ] **Step 3: Extract lineage + membership + HPA-325 scoring into the core**

The core:

1. validates subset↔reference/timing lineage;
2. selects exact subset IDs;
3. rejects missing parent members;
4. derives subset cohort ID from parent cohort ID + subset manifest SHA;
5. calls `score_cohort()` with diagnostics only for successful selected IDs;
6. calls `write_cohort_reports()`.

OaF wrapper retains only run parsing, run↔manifest checks, broad-output alias protection, persisted cohort reconstruction, and delegation.

- [ ] **Step 4: Add `score_muscriptor_reviewed_subset()`**

MuScriptor wrapper performs the analogous MuScriptor run parsing/run↔manifest checks/cohort reconstruction, then delegates to the same core. No inference.

- [ ] **Step 5: Add CLI tests before commands**

In `tests/test_cli_benchmark.py`, add:

```text
crux benchmark run-muscriptor-corpus
crux benchmark score-muscriptor-reviewed-subset
```

Assert runner help has no model-size/device/dtype/temperature/beam/instrument/map-version flags.

- [ ] **Step 6: Implement lazy-import CLI commands**

Keep MuScriptor imports inside the commands so normal CLI import remains optional-dependency safe.

- [ ] **Step 7: Run focused tests**

```bash
uv run pytest \
  tests/benchmark/test_reviewed_subset.py \
  tests/benchmark/test_reviewed_subset_acceptance.py \
  tests/benchmark/test_muscriptor_corpus_run.py \
  tests/test_cli_benchmark.py \
  -q
```

Expected: PASS; existing OaF command output remains unchanged.

- [ ] **Step 8: Commit**

```bash
git add src/benchmark/reviewed_subset.py src/cli/benchmark.py tests/benchmark/test_reviewed_subset.py tests/benchmark/test_reviewed_subset_acceptance.py tests/test_cli_benchmark.py
git commit -m "feat: score MuScriptor reviewed subset"
```

---

### Task 7: Compare published OaF/MuScriptor score reports without re-scoring

**Files:**
- Create: `src/benchmark/muscriptor_comparison.py`
- Modify: `src/cli/benchmark.py`
- Create: `tests/benchmark/test_muscriptor_comparison.py`
- Modify: `tests/test_cli_benchmark.py`

**Outputs:**

```text
paired_per_song.csv
paired_per_class.csv
summary.json
summary.md
```

- [ ] **Step 1: Add red report-parser/identity tests**

Load each run snapshot plus its sibling HPA-325 reports:

```text
run.json
reports/per_song.csv
reports/per_class.csv
reports/items.csv
```

Validate report identity columns against the run/cohort identity:

```text
cohort_id
model_id
model_lock_sha256
prediction_map_version
input_view_id
scoring_version
```

Reject duplicate score keys, malformed numeric fields, or report identity that does not match the run.

- [ ] **Step 2: Add red canonical-input pairing tests at the correct identity levels**

Run-level:

```text
reference_manifest_sha256
reference_timing_manifest_sha256/reference_timing_version
input_view_id
```

Successful item-level:

```text
simfile_id
source_audio_sha256
input_audio_sha256
```

Do **not** read `input_view_id` from item rows.

For the same `simfile_id` and source-audio hash, mismatched run input-view or item input-audio hash raises a comparison integrity error.

- [ ] **Step 3: Join published per-song rows**

Join on:

```text
(simfile_id, tolerance_ms, mode)
```

Emit:

```text
simfile_id,tolerance_ms,mode,oaf_precision,muscriptor_precision,delta_precision,oaf_recall,muscriptor_recall,delta_recall,oaf_f1,muscriptor_f1,delta_f1
```

with MuScriptor-minus-OaF deltas.

- [ ] **Step 4: Join published per-class rows**

Join on:

```text
(simfile_id, tolerance_ms, mode, common_class)
```

Emit equivalent precision/recall/F1 deltas plus reference/prediction supports. Do not synthesize missing class rows.

- [ ] **Step 5: Publish population/intersection summary**

Use `items.csv`/run evidence to report each model's full eligible success/failure/skip/quarantine population, pairable success intersection, exclusions, and aggregate mean/median deltas by tolerance/mode for song rows and class rows.

Optional `--subset-manifest` filters exact HPA-327 IDs before joining; it does not invoke subset scoring again.

- [ ] **Step 6: Add `compare-oaf-muscriptor` CLI**

```text
--oaf-run
--muscriptor-run
--manifest
--timing-manifest
--output-dir
--subset-manifest (optional)
```

The command derives each broad reports directory from `run_path.parent / "reports"` and never calls `score_cohort()`.

- [ ] **Step 7: Run focused tests**

```bash
uv run pytest tests/benchmark/test_muscriptor_comparison.py tests/test_cli_benchmark.py -q
```

Expected: PASS.

- [ ] **Step 8: Commit**

```bash
git add src/benchmark/muscriptor_comparison.py src/cli/benchmark.py tests/benchmark/test_muscriptor_comparison.py tests/test_cli_benchmark.py
git commit -m "feat: compare published OaF and MuScriptor reports"
```

---

### Task 8: Run the frozen smoke gate, broad corpus, reviewed subset, pairing, and full regression

**Operational evidence:** real local checkpoint/run/prediction/report artifacts. Large model/prediction bytes stay outside Git.

- [ ] **Step 1: Re-verify the authenticated lock and local checkpoint**

```bash
uv sync --extra muscriptor
uvx hf auth whoami
uv run pytest tests/benchmark/test_muscriptor_model.py -q
```

If auth/license/revision/hash verification fails, stop. Do not use the hosted demo or another checkpoint.

- [ ] **Step 2: Verify the exact frozen pitch map before real inference**

```bash
uv run pytest tests/benchmark/test_mapping.py -k muscriptor -q
```

Require exactly `crux.prediction-map/muscriptor-drums-v1` and the frozen string-keyed table from Task 4.

- [ ] **Step 3: Run only the five committed smoke IDs**

Verify:

```text
hard drum-only output
native onset/pitch validity
repeatability on at least one song
five-second chunk-boundary behavior on long audio
resume identity
wall time / RTF
memory evidence where available
```

Do not inspect F1 to tune checkpoint/decoding/mapping.

- [ ] **Step 4: Freeze medium vs small from feasibility only**

Keep medium if practical. If not, rerun the Task 1 freeze for small, regenerate exact lock/model ID, and rerun smoke from scratch before any scored result is inspected.

- [ ] **Step 5: Run the full technically eligible HPA-324 population**

Use the exact immutable HPA-324 reference manifest, matching HPA-323 timing manifest, verified HPA-321 cache, and `run-muscriptor-corpus`. Do not filter by MuScriptor outcome.

- [ ] **Step 6: Verify canonical-input equality against persisted OaF evidence**

For overlapping successful same-source songs require:

```python
muscriptor_snapshot["input_view_id"] == oaf_snapshot["input_view_id"]
muscriptor_row["source_audio_sha256"] == oaf_row["source_audio_sha256"]
muscriptor_row["input_audio_sha256"] == oaf_row["input_audio_sha256"]
```

The mismatch count must be exactly zero before paired findings are published.

- [ ] **Step 7: Score the accepted HPA-327 subset from persisted MuScriptor predictions**

Run `score-muscriptor-reviewed-subset`; do not rerun inference.

- [ ] **Step 8: Produce broad and reviewed report joins**

Run `compare-oaf-muscriptor` once on broad reports and once with the exact HPA-327 subset manifest. Confirm both `paired_per_song.csv` and `paired_per_class.csv` are produced from existing HPA-325 rows.

- [ ] **Step 9: Verify HPA-395 acceptance evidence**

Confirm:

```text
30/50/100 ms raw + aligned metrics
per-song and per-class deltas
complete model populations and pair intersections
native mapped/unmapped pitch coverage
reviewed event diagnostics
runtime/device/dtype/memory evidence
exact package/checkpoint/config/model-lock hashes
separate code/weight licenses
zero canonical-input mismatches
```

- [ ] **Step 10: Run full regression/static checks**

```bash
uv run pytest tests/benchmark -q
uv run pytest -q
uv run ruff check .
uv run ruff format --check src tests
uv run pylint --errors-only --disable=E1120,E0401 --jobs=1 src
```

Re-run the unchanged OaF prediction-v2 schema golden plus HPA-325/HPA-326 acceptance tests before claiming completion.

- [ ] **Step 11: Commit operationally retainable evidence only**

Commit code/config/small reviewed evidence documents required by the ticket. Do not commit gated model weights, corpus audio, or corpus-sized prediction/report artifacts unless the repository already treats that artifact class as source-controlled.

---

## Stop conditions

Do not publish scored MuScriptor findings while any of these is unresolved:

- package/checkpoint/license/revision/hash/model ID is not exact;
- medium/small choice is not frozen from pre-score feasibility evidence;
- hard `instruments=["drums"]` is unavailable under pinned v0.3.0;
- exact pitch map ID/table is not committed;
- prediction-v2 write **or read** path still assumes OaF-only backend identity;
- zero-hit map identity is copied from expected cohort identity rather than independently derived;
- existing OaF prediction-v2 golden changes;
- source resolver/prediction path/materializer diverge between OaF and MuScriptor;
- materializer no longer enforces `output_path` beneath `input_root`;
- HPA-323/HPA-324/HPA-327 lineage differs;
- overlapping same-source canonical input hashes differ;
- comparison is re-scoring predictions instead of joining the published HPA-325 reports;
- a proposed abstraction exists only for hypothetical future models rather than two current callers.

## Expected long-lived additions

```text
src/benchmark/muscriptor_model.py
src/benchmark/backends/muscriptor.py
src/benchmark/muscriptor_corpus_run.py
src/benchmark/muscriptor_comparison.py
```

Existing modules receive narrow two-caller promotions/extensions only. If implementation starts creating a generic backend registry extension, runner base class, experiment object model, mapping DSL, or second report/scoring stack, stop and simplify.
