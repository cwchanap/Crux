# HPA-395 MuScriptor Full-Mix Benchmark Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Freeze one official MuScriptor checkpoint, run hard drum-only native-event inference over the same canonical full mixes/references as OaF, score broad and HPA-327 reviewed populations through HPA-325, and publish narrow identity-matched OaF deltas plus runtime/license evidence.

**Architecture:** Keep MuScriptor comparator-only. Reuse the existing `TranscriptionBackend`, prediction-v2, HPA-324 reference, HPA-325 scorer/report, HPA-326 full-mix, and HPA-327 subset seams. Add one optional MuScriptor adapter and one model-specific corpus runner. Extend descriptor/prediction validation with backend-specific invariants rather than weakening OaF rules or inventing a schema v3. Extract only two genuinely model-neutral seams: canonical full-mix materialization and reviewed-subset filtering/scoring.

**Tech Stack:** Python 3.12, `muscriptor==0.3.0`, PyTorch transitively through the optional MuScriptor extra, Hugging Face Hub for the one-time gated checkpoint freeze, librosa/soundfile through the existing Crux canonical input path, Click, pytest, existing canonical JSON/JSONL helpers, HPA-323/HPA-324/HPA-325/HPA-326/HPA-327 artifacts.

## Global Constraints

- Freeze official MuScriptor `v0.3.0`, release commit `d73147e75e5b9b0c0a79ebe154587db4fd603e0c`.
- Start with the published medium checkpoint. Switching to small is allowed once only for demonstrated hardware impracticality before benchmark scores are inspected.
- Scored runs load an exact local safetensors file whose Hugging Face repo revision, SHA-256, size, config SHA-256, package version, device, dtype, and licenses are frozen in `runtime/muscriptor/model.json`.
- Never load scored weights with bare `"medium"`, `"small"`, or unqualified `hf://...`.
- Freeze `use_sampling=False`, `temperature=1.0`, `cfg_coef=1.0`, `instruments=["drums"]`, `batch_size=1`, `no_eos_is_ok=True`, `beam_size=1`, `prelude_forcing=True` in the lock/config. No scored CLI overrides.
- Consume MuScriptor `NoteStartEvent` directly. Do not round-trip through MIDI and do not apply `transcribe_to_midi()` beat/downbeat timing correction.
- Persist native drum MIDI pitch and raw `start_time`. `NoteEndEvent` does not create a hit. Do not invent confidence or velocity.
- Keep `crux.drum-prediction-events/v2`; dispatch backend-specific validation. Existing OaF v2 canonical bytes and invariants must remain unchanged.
- Reuse the exact HPA-326 44.1 kHz mono PCM16 `soxr_hq` canonical full-mix bytes so matching OaF/MuScriptor songs share `input_view_id` and `input_audio_sha256`.
- Keep the HPA-324 reference manifest, HPA-323 timing manifest, HPA-327 subset membership, taxonomy version, HPA-325 tolerances/modes, and scorer logic frozen.
- Persist unmapped MuScriptor drum pitches as unmapped events; never drop them before artifact publication.
- MuScriptor is not added to `default_backend_registry()` and is not exposed as a production API backend.
- CI stays offline with fake MuScriptor models/events. No gated checkpoint or accelerator is required in CI.
- Do not add a generic experiment runner, backend plugin framework, DB, queue, worker protocol, model-selection DSL, second scorer, statistics package, training, or stem separation.
- HPA-562 owns later generalized multi-model comparison. HPA-395 only emits narrow OaF-vs-MuScriptor deltas.

---

### Task 1: Add the optional MuScriptor dependency and strict frozen model-lock rail

**Files:**
- Modify: `pyproject.toml`
- Modify: `uv.lock`
- Create: `src/benchmark/muscriptor_model.py`
- Create: `scripts/freeze_muscriptor_model.py`
- Create during authenticated preflight: `runtime/muscriptor/model.json`
- Create during pre-score setup: `runtime/muscriptor/smoke.json`
- Create: `tests/benchmark/test_muscriptor_model.py`

**Interfaces:**

```python
MUSCRIPTOR_MODEL_SCHEMA = "crux.muscriptor-model/v1"

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
```

- [ ] **Step 1: Add red strict-loader tests**

Cover:

- exact schema and key set;
- package must be `muscriptor`, version exactly `0.3.0`;
- upstream source commit exactly `d73147e75e5b9b0c0a79ebe154587db4fd603e0c`;
- checkpoint revision is lowercase 40-character Git/HF identity;
- SHA-256 fields are lowercase 64-character digests;
- checkpoint/config sizes are positive integers;
- code license and weight license are nonempty and distinct fields;
- `instruments == ("drums",)`;
- deterministic decode settings equal the frozen values;
- sample rate is 16000 and chunk duration is 5.0;
- checkpoint verifier rejects missing/edited files.

- [ ] **Step 2: Verify red**

```bash
uv run pytest tests/benchmark/test_muscriptor_model.py -q
```

Expected: FAIL because the module does not exist.

- [ ] **Step 3: Add the optional dependency**

Add:

```toml
muscriptor = ["muscriptor==0.3.0"]
```

under `[project.optional-dependencies]`, then refresh the lock:

```bash
uv lock
```

Do not add MuScriptor to the base dependency list.

- [ ] **Step 4: Implement the strict model-lock parser/checkpoint verifier**

Use existing `strict_json_loads()`, `require_sha256()`, and `read_regular_file_no_follow()` where applicable. Keep the file contract closed; unknown fields fail.

The loader must validate configuration values, not merely deserialize them. The checkpoint verifier reads the exact local `model.safetensors` and `config.json`, checks byte lengths and SHA-256, and returns the safetensors path only on exact match.

- [ ] **Step 5: Add the one-time authenticated freeze script**

`scripts/freeze_muscriptor_model.py` must:

1. resolve `MuScriptor/muscriptor-medium` with Hugging Face Hub metadata;
2. capture the exact repo commit SHA;
3. download `model.safetensors` and `config.json` with `revision=<exact sha>`;
4. hash/size both local files;
5. write a proposed canonical `runtime/muscriptor/model.json` with the release/package/code identity and an explicit target device/dtype supplied to the script;
6. default the candidate size to medium;
7. never call benchmark scoring.

The generated lock carries `code_license = "MIT"` and the exact weight-license string returned/verified from the official model metadata/documentation. If the official source does not expose or agree on the expected CC BY-NC 4.0 license, exit nonzero instead of inventing a value.

- [ ] **Step 6: Freeze the pre-score smoke membership**

Create strict `runtime/muscriptor/smoke.json` with exactly five role records:

```text
short
long
dense
sparse
non_drum_heavy
```

Each record contains a positive `simfile_id` and nonempty pre-model `reason`. Choose the IDs using only HPA-323/HPA-324/HPA-327 reference/audit data and source listening. Do not run MuScriptor before the file is committed.

Add the smoke loader validation to `muscriptor_model.py`; five unique simfile IDs are required.

- [ ] **Step 7: Run focused tests**

```bash
uv run pytest tests/benchmark/test_muscriptor_model.py -q
```

Expected: PASS without Hugging Face access.

- [ ] **Step 8: Commit**

```bash
git add pyproject.toml uv.lock src/benchmark/muscriptor_model.py scripts/freeze_muscriptor_model.py runtime/muscriptor tests/benchmark/test_muscriptor_model.py
git commit -m "feat: freeze MuScriptor benchmark model"
```

**Operational gate:** The real `runtime/muscriptor/model.json` and `smoke.json` must contain actual frozen values before Task 8 broad scoring. No synthetic digest may reach the scored run.

---

### Task 2: Promote the canonical full-mix materializer without changing OaF bytes

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

Keep the existing input-view string exactly. The OaF-prefixed historical identifier is already persisted evidence and is now the shared paired-input identity; renaming it would break the intended exact OaF/MuScriptor match.

- [ ] **Step 1: Add a characterization test around current OaF materialization**

Use a deterministic small source fixture and capture:

- output WAV bytes;
- SHA-256;
- sample rate/channel/sample width/frame count;
- `CanonicalAudio.input_view_id`.

- [ ] **Step 2: Verify green before refactor**

```bash
uv run pytest tests/benchmark/test_oaf_corpus_run.py -k materialize -q
```

Expected: PASS against the current private implementation.

- [ ] **Step 3: Add the shared helper tests and move the implementation**

Move exactly the librosa/soundfile behavior:

```python
samples, _ = librosa.load(source_input, sr=44100, mono=True, res_type="soxr_hq")
soundfile.write(output_path, samples, 44100, format="WAV", subtype="PCM_16")
```

then call existing `load_materialized_audio()`.

The helper owns no backend selection and no model configuration. `max_input_audio_frames=None` means no model-specific length guard.

- [ ] **Step 4: Replace `_materialize_oaf_full_mix()` with the shared call**

OaF passes `config.max_input_audio_frames`. Delete the private materializer after characterization passes; no compatibility alias.

- [ ] **Step 5: Prove byte identity**

```bash
uv run pytest tests/benchmark/test_input_view.py tests/benchmark/test_oaf_corpus_run.py tests/benchmark/test_oaf_corpus_run_acceptance.py -q
```

Expected: PASS; the characterization hash/bytes remain unchanged.

- [ ] **Step 6: Commit**

```bash
git add src/benchmark/input_view.py src/benchmark/oaf_corpus_run.py tests/benchmark/test_input_view.py tests/benchmark/test_oaf_corpus_run.py
git commit -m "refactor: share canonical full mix input"
```

---

### Task 3: Extend frozen backend identity and prediction-v2 validation for MuScriptor

**Files:**
- Modify: `src/benchmark/backend_identity.py`
- Modify: `src/benchmark/prediction_artifact.py`
- Modify: `src/benchmark/cohort_scoring.py`
- Modify: `tests/benchmark/test_backend_identity.py`
- Modify: `tests/benchmark/test_prediction_artifact.py`
- Modify: `tests/benchmark/test_cohort_scoring.py`
- Modify: `tests/benchmark/schema_goldens/crux.drum-prediction-events-v2.jsonl`
- Modify: `tests/benchmark/schema_goldens/manifest.json`

**Frozen identities:**

```text
backend_id = muscriptor-v0.3.0-drums-v1
architecture_id = muscriptor-transformer-v0.3.0
native_output_space_id = muscriptor-drums-midi128-v1
native_metadata_schema_id = muscriptor-note-start-metadata-v1
prediction_schema = crux.drum-prediction-events/v2
```

`model_id` binds the selected checkpoint size/revision identity derived from `runtime/muscriptor/model.json`.

- [ ] **Step 1: Add red descriptor tests**

Prove `normalize_known_backend_descriptor()` accepts the exact frozen MuScriptor identity and rejects mixed OaF/MuScriptor fields, unknown backends, edited source commit, and edited model identity.

Existing OaF descriptor tests must remain unchanged and green.

- [ ] **Step 2: Add red MuScriptor prediction-v2 round-trip tests**

Build a mapped prediction containing:

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

Expect render -> read -> render byte identity.

Also test rejection of:

- non-null model output bin;
- missing/native MIDI out of range;
- non-null confidence or velocity if v1 adapter does not produce them;
- non-drum metadata;
- mismatched `native_class_id`;
- OaF event using MuScriptor nullability.

- [ ] **Step 3: Verify red**

```bash
uv run pytest tests/benchmark/test_backend_identity.py tests/benchmark/test_prediction_artifact.py -q
```

Expected: MuScriptor cases FAIL while existing OaF cases pass.

- [ ] **Step 4: Dispatch descriptor/event validation by frozen backend**

Keep descriptor schema v2. Replace the single OaF-only guard with an explicit two-branch validator. In `prediction_artifact.py`, determine the backend only from the already-validated descriptor, then call an OaF or MuScriptor native-event validator.

Do not turn required OaF fields optional globally.

- [ ] **Step 5: Close the empty-prediction scorer seam**

In `cohort_scoring._artifact_identity_from_artifacts()`, remove the OaF-only empty-event fallback. For an empty but otherwise validated prediction artifact, use the expected cohort/descriptor prediction-map identity supplied by the caller's frozen identity path rather than rejecting every non-OaF backend.

Add a successful zero-hit MuScriptor song test that still carries correct native/mapped coverage counts.

- [ ] **Step 6: Extend the prediction-v2 golden**

Keep the existing OaF golden record and add a canonical MuScriptor record to the same schema golden. Update `manifest.json` digest/size through the repository's existing golden-update procedure.

- [ ] **Step 7: Run focused + OaF regression tests**

```bash
uv run pytest tests/benchmark/test_backend_identity.py tests/benchmark/test_prediction_artifact.py tests/benchmark/test_cohort_scoring.py tests/benchmark/test_cohort_scoring_acceptance.py -q
```

Expected: PASS with no OaF behavior change.

- [ ] **Step 8: Commit**

```bash
git add src/benchmark/backend_identity.py src/benchmark/prediction_artifact.py src/benchmark/cohort_scoring.py tests/benchmark/test_backend_identity.py tests/benchmark/test_prediction_artifact.py tests/benchmark/test_cohort_scoring.py tests/benchmark/schema_goldens
git commit -m "feat: support MuScriptor prediction artifacts"
```

---

### Task 4: Add the direct MuScriptor adapter and explicit drum-pitch mapping

**Files:**
- Create: `src/benchmark/backends/muscriptor.py`
- Modify: `src/benchmark/taxonomy.py`
- Modify: `src/benchmark/mapping.py`
- Create: `tests/benchmark/test_muscriptor_backend.py`
- Modify: `tests/benchmark/test_mapping.py`

**Interfaces:**

```python
MUSCRIPTOR_ADAPTER_REVISION = "crux-muscriptor-adapter/v1"
MUSCRIPTOR_PREDICTION_MAP_ID = "crux.prediction-map/muscriptor-drums-v1"

class MuscriptorBackend:
    def descriptor(self) -> BackendDescriptor: ...
    def transcribe(self, audio: CanonicalAudio) -> NativePrediction: ...
    def close(self) -> None: ...


def create_backend(
    *,
    model_lock: MuscriptorModelLock,
    checkpoint_dir: Path,
    model_factory: Callable[..., object] | None = None,
) -> MuscriptorBackend: ...


def map_muscriptor_prediction(
    prediction: NativePrediction,
    prediction_map: PredictionMap = MUSCRIPTOR_PREDICTION_MAP,
) -> tuple[MappedPrediction, MappingDiagnostics]: ...
```

- [ ] **Step 1: Write fake upstream event/model fixtures**

Use small local dataclasses or simple namespace objects matching only the public fields required by v0.3.0:

```text
NoteStartEvent: pitch, start_time, index, instrument
NoteEndEvent: end_time, start_event
ProgressEvent: completed, total
```

No import of real `muscriptor` is required for the unit test.

- [ ] **Step 2: Add red adapter tests**

Assert the adapter calls `transcribe()` exactly once with the frozen arguments and canonical WAV path. Assert:

- only start events become hits;
- onset/pitch/instrument metadata are preserved;
- no confidence/velocity/output-bin is fabricated;
- progress/end events do not create hits;
- non-drum start event is rejected;
- negative/nonfinite onset is rejected;
- pitch outside 0..127 is rejected;
- descriptor matches the frozen model lock;
- `close()` is idempotent.

- [ ] **Step 3: Verify red**

```bash
uv run pytest tests/benchmark/test_muscriptor_backend.py -q
```

Expected: FAIL because the adapter does not exist.

- [ ] **Step 4: Implement lazy optional import and one-model lifecycle**

The real model factory is imported inside `create_backend()`. Verify installed package version before model construction. Call:

```python
TranscriptionModel.load_model(
    weights_path=verified_checkpoint_path,
    device=lock.device,
    dtype=lock.dtype,
)
```

Never pass a remote checkpoint locator.

- [ ] **Step 5: Add the frozen explicit pitch map**

Create an explicit dictionary for General MIDI drum pitches that unambiguously project into the existing detailed/common taxonomy. Preserve all native pitches in artifacts. Unknown pitches become `MappingDiagnostics.unmapped` entries and `MappedPredictionEvent(mapping_status="unmapped", ...)`.

Do not silently reuse `DEFAULT_MIDI_NOTE_MAP` if it omits released MuScriptor drum-family pitches. The new map should be reviewed as a complete v1 table for the supported common classes.

- [ ] **Step 6: Add mapping tests**

Cover at least:

- two kick pitches collapse to common `kick` but retain distinct native MIDI notes;
- multiple tom pitches collapse to `tom`;
- closed/open hihat distinction maps to detailed classes when unambiguous and common `hihat`;
- crash/ride families;
- one unsupported native pitch remains persisted/unmapped;
- backend/native-output-space mismatch fails.

- [ ] **Step 7: Run focused tests**

```bash
uv run pytest tests/benchmark/test_muscriptor_backend.py tests/benchmark/test_mapping.py tests/benchmark/test_prediction_artifact.py -q
```

Expected: PASS.

- [ ] **Step 8: Commit**

```bash
git add src/benchmark/backends/muscriptor.py src/benchmark/taxonomy.py src/benchmark/mapping.py tests/benchmark/test_muscriptor_backend.py tests/benchmark/test_mapping.py
git commit -m "feat: add MuScriptor drum adapter"
```

---

### Task 5: Add the MuScriptor corpus-run persistence/resume contract

**Files:**
- Create: `src/benchmark/muscriptor_corpus_run.py`
- Modify narrowly if extraction is smaller: `src/benchmark/oaf_corpus_run.py`
- Modify narrowly if extraction is smaller: `src/benchmark/corpus_cache.py`
- Create: `tests/benchmark/test_muscriptor_corpus_run.py`
- Create: `tests/benchmark/test_muscriptor_corpus_run_acceptance.py`

**Interfaces:**

```python
MUSCRIPTOR_CORPUS_RUN_SCHEMA = "crux.muscriptor-corpus-run/v1"
MUSCRIPTOR_INFERENCE_CONFIG_SCHEMA = "crux.muscriptor-inference-config/v1"

@dataclass(frozen=True)
class MuscriptorCorpusRunRequest:
    reference_manifest_path: Path
    timing_manifest_path: Path
    cache_dir: Path
    output_dir: Path
    include_simfile_ids: tuple[int, ...] = ()
    exclude_simfile_ids: tuple[int, ...] = ()
    resume: bool = False
    crux_commit: str | None = None

@dataclass(frozen=True)
class MuscriptorCorpusRunOutcome:
    overall_status: Literal["complete", "partial", "failed"]
    exit_code: Literal[0, 1, 2]
    run_id: str | None
    run_path: Path | None
    reports_path: Path | None
    success_count: int
    failed_count: int
    skipped_count: int
    quarantined_count: int
    aggregate_rtf: float | None
    projected_full_wall_time_sec: float | None
    peak_memory_bytes: int | None


def run_muscriptor_corpus(...): ...
def parse_muscriptor_corpus_run(...): ...
def build_muscriptor_cohort_from_snapshot(...): ...
```

- [ ] **Step 1: Add red pure identity/config tests**

Test deterministic inference-config hash and run ID. Every inference-semantic value from the model lock is included. Scope normalization follows HPA-326 semantics: positive IDs, sorted/deduplicated, no include/exclude overlap, unknown IDs fail before backend construction.

- [ ] **Step 2: Add red snapshot canonicalization tests**

Mirror the useful HPA-326 contract without copying TF1-specific fields. Complete snapshots require reconciled item dispositions/counts. Persist each inferred/resumed row's:

```text
source_audio_id
source_audio_sha256
input_view_id
input_audio_sha256
prediction_path
prediction_artifact_sha256
wall_time_sec
rtf
```

Header persists model-lock SHA, checkpoint SHA/revision, descriptor/config hashes, device/dtype, licenses, adapter revision, reference identities, and input-view identity.

- [ ] **Step 3: Add red execution tests with injected fake backend**

Prove:

- all lineage/scope/model-lock/checkpoint checks happen before backend creation;
- source cache is re-pinned immediately before materialization;
- shared `materialize_full_mix_audio()` is used;
- backend is lazily created once and reused sequentially;
- mapped prediction-v2 artifact is immutable and source/config keyed;
- per-item inference failure continues to the next item;
- descriptor/config integrity failure stops the run;
- excluded/quarantined rows never construct the backend by themselves;
- resume accepts an exact persisted artifact without inference;
- resume rejects edited artifact bytes or row identity;
- existing artifact without `--resume` is a conflict;
- final snapshot and HPA-325 reports reconcile counts.

- [ ] **Step 4: Verify red**

```bash
uv run pytest tests/benchmark/test_muscriptor_corpus_run.py tests/benchmark/test_muscriptor_corpus_run_acceptance.py -q
```

Expected: FAIL because the runner does not exist.

- [ ] **Step 5: Reuse source-resolution logic with the smallest safe extraction**

HPA-326 currently owns `_resolve_source_audio()` and `ResolvedSourceAudio`. If copying that cache-integrity code would duplicate more than a thin wrapper, promote only those two model-neutral pieces into a small existing benchmark module such as `input_view.py` or `corpus_cache.py`, then point OaF and MuScriptor at it with characterization tests.

Do **not** extract OaF snapshot, worker, failure-classifier, or backend lifecycle code.

- [ ] **Step 6: Implement run snapshot + execution**

Follow HPA-326 ordering:

```text
reference/timing/scope/model preflight
-> create initial durable snapshot
-> source-resolution pass
-> per-item re-pin/materialize/resume-or-infer/map/publish/checkpoint
-> close model
-> finalize runtime projection
-> build CohortItems
-> score_cohort(..., diagnostics_for=())
-> write_cohort_reports()
```

Use `prediction_path()` if its existing descriptor/config/source keyed location is model-neutral; otherwise move that exact helper to a neutral module and keep path bytes identical for OaF.

- [ ] **Step 7: Add practical memory measurement**

Record process peak RSS through the simplest stdlib/platform-supported path available to the benchmark host. For CUDA/MPS accelerator memory, record a value only when the selected PyTorch device exposes a direct API. Missing accelerator-specific memory is `None`, not a failure.

Do not add a telemetry abstraction or sampling thread.

- [ ] **Step 8: Run focused + OaF regression tests**

```bash
uv run pytest tests/benchmark/test_muscriptor_corpus_run.py tests/benchmark/test_muscriptor_corpus_run_acceptance.py tests/benchmark/test_oaf_corpus_run.py tests/benchmark/test_oaf_corpus_run_acceptance.py -q
```

Expected: PASS.

- [ ] **Step 9: Commit**

```bash
git add src/benchmark/muscriptor_corpus_run.py src/benchmark/oaf_corpus_run.py src/benchmark/input_view.py src/benchmark/corpus_cache.py tests/benchmark/test_muscriptor_corpus_run.py tests/benchmark/test_muscriptor_corpus_run_acceptance.py tests/benchmark/test_oaf_corpus_run.py
git commit -m "feat: run MuScriptor benchmark corpus"
```

Only add files that actually changed; do not touch neutral modules merely to match the command above.

---

### Task 6: Add the MuScriptor CLI and shared reviewed-subset scoring core

**Files:**
- Modify: `src/benchmark/reviewed_subset.py`
- Modify: `src/cli/benchmark.py`
- Modify: `tests/benchmark/test_reviewed_subset.py`
- Modify: `tests/benchmark/test_reviewed_subset_acceptance.py`
- Modify or create the existing benchmark CLI test file used by this repo

**Shared interface:**

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

The helper validates source reference/timing lineage passed by the wrapper, exact subset membership, derives the subset cohort ID from parent cohort + subset manifest hash, and calls:

```python
score_cohort(
    subset_identity,
    selected_items,
    diagnostics_for=tuple(sorted(successful_selected_ids)),
)
```

- [ ] **Step 1: Add characterization around existing OaF reviewed-subset outcome**

Capture existing selected items, derived cohort ID, diagnostics IDs, exit-code semantics, and report output for representative success + failed rows.

- [ ] **Step 2: Extract the model-neutral core and keep OaF behavior identical**

`score_oaf_reviewed_subset()` still owns parsing the OaF run and HPA-323/HPA-324 lineage checks, then delegates after `build_oaf_cohort_from_snapshot()`.

Do not introduce callback registries or backend string switches into `reviewed_subset.py`.

- [ ] **Step 3: Add `score_muscriptor_reviewed_subset()` wrapper**

It parses the MuScriptor run, validates the exact same supplied HPA-323/HPA-324 and HPA-327 lineage, builds persisted MuScriptor CohortItems, and delegates to the shared core.

- [ ] **Step 4: Add CLI tests first**

Add:

```text
crux benchmark run-muscriptor-corpus
crux benchmark score-muscriptor-reviewed-subset
```

`run-muscriptor-corpus` mirrors HPA-326 arguments and canonical JSON summary. No model/config tuning flags.

- [ ] **Step 5: Implement CLI commands**

Lazy-import the optional MuScriptor runner inside the command function so unrelated CLI startup does not import PyTorch.

- [ ] **Step 6: Run focused tests**

```bash
uv run pytest tests/benchmark/test_reviewed_subset.py tests/benchmark/test_reviewed_subset_acceptance.py tests/benchmark/test_muscriptor_corpus_run.py -q
```

Also run the repo's benchmark CLI test module for the new command cases.

Expected: PASS; existing `score-oaf-reviewed-subset` outputs remain unchanged.

- [ ] **Step 7: Commit**

```bash
git add src/benchmark/reviewed_subset.py src/cli/benchmark.py tests/benchmark/test_reviewed_subset.py tests/benchmark/test_reviewed_subset_acceptance.py
git commit -m "feat: score MuScriptor reviewed subset"
```

---

### Task 7: Add narrow identity-matched OaF-vs-MuScriptor deltas

**Files:**
- Create: `src/benchmark/muscriptor_comparison.py`
- Modify: `src/cli/benchmark.py`
- Create: `tests/benchmark/test_muscriptor_comparison.py`
- Modify or create the existing benchmark CLI test file used by this repo

**Interface:**

```python
@dataclass(frozen=True)
class PairedSongDelta:
    simfile_id: str
    tolerance_ms: int
    mode: Literal["raw", "aligned"]
    oaf_f1: float
    muscriptor_f1: float
    delta_f1: float


def compare_oaf_muscriptor_runs(
    *,
    oaf_run_path: Path,
    muscriptor_run_path: Path,
    reference_manifest_path: Path,
    timing_manifest_path: Path,
    output_dir: Path,
    subset_manifest_path: Path | None = None,
) -> Path:
    ...
```

- [ ] **Step 1: Add red pairing tests**

Build persisted synthetic run/cohort evidence and prove a pair is included only when both items succeed and all are equal:

```text
simfile_id
reference manifest identity
timing identity
source_audio_sha256
input_view_id
input_audio_sha256
```

Reject/exclude edited source hash, input hash/view, missing song, non-success item, or mismatched reference lineage.

- [ ] **Step 2: Add red score-delta tests**

For every HPA-325 `30/50/100 ms × raw/aligned` SongScore pair, assert:

```python
delta_f1 == muscriptor_f1 - oaf_f1
```

Output deterministic CSV rows sorted `(simfile_id, tolerance_ms, mode)` and a small canonical JSON summary containing pair count plus mean/median delta per tolerance/mode.

Do not add p-values, bootstrap intervals, ranking, or generic N-model structures.

- [ ] **Step 3: Verify red**

```bash
uv run pytest tests/benchmark/test_muscriptor_comparison.py -q
```

Expected: FAIL because the helper does not exist.

- [ ] **Step 4: Implement by reconstructing persisted cohorts and reusing HPA-325**

Do not parse arbitrary report CSV as authority. Parse both run snapshots, rebuild their persisted `CohortItem` values, call `score_cohort()` with the same fixed HPA-325 settings, then join the resulting song scores after artifact identity checks.

If `subset_manifest_path` is supplied, validate and filter to the exact HPA-327 IDs before scoring/joining.

- [ ] **Step 5: Add `compare-oaf-muscriptor` CLI**

Arguments:

```text
--oaf-run
--muscriptor-run
--manifest
--timing-manifest
--output-dir
--subset-manifest (optional)
```

The command never reruns inference.

- [ ] **Step 6: Run focused tests**

```bash
uv run pytest tests/benchmark/test_muscriptor_comparison.py tests/benchmark/test_cohort_scoring.py -q
```

Also run the benchmark CLI test module.

Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add src/benchmark/muscriptor_comparison.py src/cli/benchmark.py tests/benchmark/test_muscriptor_comparison.py
git commit -m "feat: compare OaF and MuScriptor scores"
```

---

### Task 8: Execute the frozen smoke gate, then the broad and reviewed benchmark

**Files/evidence produced:**
- Verify/update before first inference: `runtime/muscriptor/model.json`
- Verify: `runtime/muscriptor/smoke.json`
- Produce under local artifact root: MuScriptor run snapshot, prediction JSONL, broad reports, reviewed-subset reports, paired comparison reports
- Add a short checked-in evidence note only if the repository's current benchmark evidence convention requires one; otherwise keep large artifacts local and reference their immutable hashes in the Linear issue/PR.

This task is operational and uses real gated weights. It must occur only after Tasks 1–7 are green.

- [ ] **Step 1: Create/install the optional environment and authenticate**

```bash
uv sync --extra muscriptor
uvx hf auth whoami
```

If authentication/license acceptance is missing, stop. Do not bypass gating or copy weights from an unverifiable source.

- [ ] **Step 2: Freeze/verify the exact candidate medium checkpoint**

Run the freeze script for the target benchmark device/dtype, inspect the generated lock, and verify:

```bash
uv run python -c 'from pathlib import Path; from src.benchmark.muscriptor_model import load_muscriptor_model_lock, verify_muscriptor_checkpoint; lock=load_muscriptor_model_lock(Path("runtime/muscriptor/model.json")); print(verify_muscriptor_checkpoint(lock, Path.home()/".cache"/"muscriptor-benchmark"))'
```

Use the actual checkpoint directory selected by the freeze script. Do not edit hashes manually.

- [ ] **Step 3: Run only the fixed smoke IDs first**

Invoke `run-muscriptor-corpus` with the five IDs from `runtime/muscriptor/smoke.json` as repeatable includes into a dedicated smoke output root.

Verify:

- all outputs are drum-only native events;
- one chosen song repeated in a clean smoke run yields byte-identical prediction artifacts;
- the long example has sane events around five-second boundaries without duplicate/drop artifacts introduced by the adapter;
- resume performs zero new inference for exact artifacts;
- edited artifact/hash evidence is rejected in the automated fixture tests;
- wall time/RTF/peak-memory are recorded.

- [ ] **Step 4: Decide medium feasibility before inspecting benchmark F1**

Use only runtime/memory/device stability evidence. If medium is practical, freeze it permanently. If it is impractical, regenerate the model lock once for small and rerun the smoke gate from scratch. Record the reason. Do not compare F1 to decide.

After this point the model lock/config is frozen for HPA-395.

- [ ] **Step 5: Run the full technically eligible broad corpus**

```bash
uv run crux benchmark run-muscriptor-corpus \
  --manifest <immutable HPA-324 manifest> \
  --timing-manifest <matching HPA-323 manifest> \
  --cache-dir <verified HPA-321 cache> \
  --output-dir <HPA-395 artifact root>
```

Use the actual local immutable artifact paths. No include/exclude flags for the broad population unless the HPA-324 technical eligibility contract already requires an explicit frozen scope; do not curate by model outcome.

- [ ] **Step 6: Score the exact HPA-327 reviewed subset from persisted predictions**

```bash
uv run crux benchmark score-muscriptor-reviewed-subset \
  --run <MuScriptor run.json> \
  --manifest <same HPA-324 manifest> \
  --timing-manifest <same HPA-323 manifest> \
  --subset-manifest <accepted HPA-327 manifest> \
  --output-dir <HPA-395 reviewed report dir>
```

No inference rerun is allowed here.

- [ ] **Step 7: Produce broad and reviewed paired OaF deltas**

Run `compare-oaf-muscriptor` once without the subset filter and once with the HPA-327 manifest, using the matching persisted HPA-326 OaF run.

- [ ] **Step 8: Inspect required evidence before claiming acceptance**

Confirm reports contain:

- event-micro P/R/F1 at 30/50/100 ms;
- song-macro and class-macro F1;
- per-class counts/scores;
- population failures/skips/quarantines;
- MuScriptor native mapped/unmapped pitch coverage;
- reviewed event diagnostics;
- broad/reviewed paired OaF deltas;
- wall time/RTF/device/dtype/memory;
- model/package/checkpoint hashes and code/weight licenses.

Do not interpret helped/harmed examples until the fixed scoring outputs are complete.

---

### Task 9: Final regression, static checks, and acceptance documentation

**Files:**
- Modify only if needed for operator docs: `README.md` or the repository's existing benchmark runbook
- Verify: all files changed by Tasks 1–8

- [ ] **Step 1: Run all benchmark tests**

```bash
uv run pytest tests/benchmark -q
```

Expected: PASS.

- [ ] **Step 2: Run the full project test suite**

```bash
uv run pytest -q
```

Expected: PASS.

- [ ] **Step 3: Run configured lint/type/static checks**

Use the repository's existing commands; at minimum run the configured Ruff/Pylint/type checks against the changed Python files. Do not add a new lint tool for HPA-395.

- [ ] **Step 4: Verify no accidental production exposure**

Assert:

- `default_backend_registry()` still defaults to and exposes only the intended production OaF selection;
- importing normal CLI/API modules without the MuScriptor extra succeeds;
- no test/fixture downloads gated weights;
- no benchmark command exposes model-size, temperature, beam, device, dtype, or instrument tuning flags.

- [ ] **Step 5: Verify immutable OaF regression evidence**

Re-run the existing OaF prediction-v2 golden and HPA-325/HPA-326 acceptance tests. The same OaF artifact bytes must still load and score.

- [ ] **Step 6: Self-review the diff against HPA-395 non-goals**

Delete any abstraction added only for hypothetical third/fourth models. Keep only the shared full-mix and reviewed-subset seams that have two real callers now.

- [ ] **Step 7: Commit final documentation/cleanup**

```bash
git add -A
git commit -m "docs: record HPA-395 benchmark evidence"
```

Skip the commit if no documentation/cleanup changes are needed.

## Implementation order and stop conditions

```text
Task 1 model/package/checkpoint freeze rail
  -> Task 2 exact shared full-mix input
  -> Task 3 common persisted artifact/scorer support
  -> Task 4 native adapter + mapping
  -> Task 5 corpus runner
  -> Task 6 reviewed-subset + CLI
  -> Task 7 paired deltas
  -> Task 8 real smoke/broad/reviewed execution
  -> Task 9 regression/acceptance
```

Stop before scored inference if any of these remain unresolved:

- gated checkpoint/license cannot be verified;
- model-lock revision/hash is not exact;
- medium/small selection has not been frozen from pre-score feasibility evidence;
- hard `instruments=["drums"]` behavior is unavailable in the pinned release;
- canonical input bytes differ from HPA-326 for the same source;
- OaF prediction-v2 regression tests change;
- HPA-324/HPA-323/HPA-327 lineage does not match.

## Expected implementation footprint

New long-lived modules should stay limited to:

```text
src/benchmark/muscriptor_model.py
src/benchmark/backends/muscriptor.py
src/benchmark/muscriptor_corpus_run.py
src/benchmark/muscriptor_comparison.py
```

plus strict runtime lock/smoke JSON and focused tests.

Existing modules should receive narrow extensions/extractions only:

```text
src/benchmark/input_view.py
src/benchmark/backend_identity.py
src/benchmark/prediction_artifact.py
src/benchmark/taxonomy.py
src/benchmark/mapping.py
src/benchmark/cohort_scoring.py
src/benchmark/reviewed_subset.py
src/cli/benchmark.py
```

If implementation starts creating a model registry, generic experiment object model, second report stack, or shared runner base class, stop and simplify back to the concrete HPA-395 seams above.