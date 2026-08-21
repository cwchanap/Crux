# HPA-396 Frozen IDM Stem Pilot Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Freeze one official Inverse Drum Machine runtime/model/extraction contract, run its native transcription head over the exact HPA-328 HTDemucs canonical-input population, score IDM and retained OaF predictions through unchanged HPA-325 scoring, and publish exact-input paired OaF→IDM evidence plus a separately frozen five-song full-mix diagnostic.

**Architecture:** Keep IDM comparator-only. Run the exact upstream package in one isolated Python 3.11 `uv` environment behind the existing `WorkerProcess`. Load the SHA-verified model config/checkpoint directly rather than using upstream `idm.inference` discovery. Extend the closed descriptor/prediction-v2/taxonomy/metadata/zero-hit contracts for one third family. Consume the immutable HPA-328 handoff rather than rerunning separators/OaF. Keep one concrete `idm_pilot_run.py` and one concrete `idm_comparison.py`; only extract proven model-neutral duplicates that acquire a third caller.

**Tech Stack:** Main Crux Python 3.12; isolated IDM Python 3.11; `uv`; pinned PyTorch/torchaudio/librosa through upstream IDM; `soundfile` for no-resample canonical WAV decode; official source commit `456656868538205ef756912c7cf5b0fd936de8af`; Click; pytest; Crux canonical JSON/JSONL helpers; HPA-324 taxonomy; HPA-325 scorer/reports; HPA-327 references; HPA-328 immutable separation handoff.

**Spec:** `docs/superpowers/specs/2026-08-20-hpa-396-idm-stem-pilot-design.md`

## PR constraint

This draft is the **single HPA-396 PR**. Continue implementation on this branch/PR after planning review. Do not open a second HPA-396 implementation PR.

Use small per-task commits on this PR. Keep the runtime-lock feasibility commit separate from later scoring work so upstream-runtime changes are reviewable independently.

## Global constraints

- Official upstream only: `https://github.com/bernardo-torres/inverse-drum-machine` at `456656868538205ef756912c7cf5b0fd936de8af`.
- Primary model is exactly `idm-44-train-kits`.
- Isolated runtime is Python `3.11.*`; if the pinned dependency set cannot resolve/run there without patching upstream, stop and report the blocker.
- **Never import `idm.inference` in the scored worker.** Its rootutils/checkpoint-discovery path is not the frozen boundary.
- Direct model load is exact verified `model.yaml` + exact verified checkpoint + strict `load_state_dict()`.
- Primary population is exactly the production HPA-328 handoff population. No seed/count/include/exclude controls.
- The merged HPA-328 PR currently records that production Task 11 was not run. Real HPA-396 execution is blocked until its immutable production handoff exists.
- Primary input bytes are retained HPA-328 `htdemucs.input` WAV bytes; do not rerun HTDemucs or recanonicalize them.
- Retained HPA-328 OaF-on-HTDemucs predictions are the comparator; do not rerun OaF.
- Preserve `crux.oaf-htdemucs-drums-mono44k1-pcm16/v1` as the primary input-view ID.
- Worker WAV decode is `soundfile-preserve-wav/v1`: WAV, 44.1 kHz, mono, PCM16; no librosa loader/resample/mixdown/normalization/rewrite.
- Librosa remains runtime-significant because upstream `PeakPicking` calls `librosa.util.peak_pick()` and `samples_to_time()`; the isolated `uv.lock` freezes that behavior.
- Freeze `IDM_REQUEST_TIMEOUT_SECONDS = 1800.0` in `crux.idm-inference-config/v1`; changing it changes run identity.
- Do **not** add an IDM close-timeout constant. Reuse `WorkerProcess`'s existing default.
- Launch `WorkerProcess.start()` with a `Sequence[str]` beginning with the isolated runtime Python; never pass a lone worker path/string.
- `IdmBackend` has no OaF-style `input_root`; runner-side retained-artifact validation owns path containment.
- Default ignored model root is `artifacts/idm/pretrained/idm-44-train-kits`.
- Keep `crux.drum-prediction-events/v2`; add IDM as the third closed family without weakening OaF/MuScriptor.
- Add `ZERO_HIT_PREDICTION_MAPS[IDM_BACKEND_ID] = IDM_PREDICTION_MAP_ID` unconditionally.
- Persist IDM `native_metadata` with exactly `frame_index` and pre-MIDI `native_velocity` strings.
- Tiny valid velocity may quantize to `"0"`; `"0"` is valid metadata and must not drop a song.
- Precise expected native-velocity band is a snapshot diagnostic, not an onset-scoring gate.
- `map_idm_prediction()` keys on `native_class_id`.
- `TT_LMT -> ClassMapping(None, "tom")`.
- Use unchanged `score_cohort()` and `write_cohort_reports()`.
- Paired stem comparison calls `pairable_success_ids(..., require_identical_input_hash=True)`.
- Keep native failure codes in the HPA-396 run snapshot/histogram; map to existing HPA-325 reasons without extending the scorer enum.
- `upstream_stem_unavailable -> inference_failed`.
- `source_cache_dir` belongs only to the separate full-mix smoke request.
- No production registry/API integration, prediction v3, generic runner, generic comparison registry, database, queue, score-driven tuning, full-corpus path, training, or fine-tuning.

## Risks / hard gates

- **Production HPA-328 handoff absent:** offline implementation may proceed; real smoke/pilot execution may not.
- **Upstream load wrapper:** importing `idm.inference` from the installed venv is not the benchmark gate; direct verified config/checkpoint construction is.
- **PeakPicking compatibility:** Task 0 executes and introspects the exact pinned validation picker before lock publication.
- **Librosa drift:** soundfile owns WAV decode, but librosa owns upstream peak picking; runtime-lock changes are benchmark identity changes.
- **Timeout-dependent population:** request timeout is frozen before scores.
- **Zero-hit artifacts:** mandatory zero-hit map test lands before the runner.
- **Diagnostic velocity:** raw expected-band anomalies are counted, not used to drop otherwise valid onsets; quantized zero is valid.
- **Metadata recovery:** frame index/native velocity land before real inference.
- **Training-data provenance:** record StemGMD provenance context for HPA-329; pinned repository/checkpoint license basis itself is recorded as Apache-2.0 with no separate weight notice found at the pinned revision.

---

## Task 0: Prove the exact installed runtime, direct model load, demo WAV, and frozen PeakPicking/velocity facts

**Purpose:** Fail on a real upstream incompatibility, not on `idm.inference`'s source-tree discovery assumptions.

**Files:**
- Create: `runtime/idm/pyproject.toml`
- Create: `runtime/idm/uv.lock`
- Temporary/local: `runtime/idm/.venv/`
- Temporary/local model root: `artifacts/idm/pretrained/idm-44-train-kits/`
- Verify pinned upstream: `demo/mix.wav`
- Verify pinned upstream: `pretrained/idm-44-train-kits/checkpoints/model.yaml`
- Verify pinned upstream: `pretrained/idm-44-train-kits/checkpoints/val-epoch=518-global_step=0.ckpt`

### Step 1: Create the isolated runtime project

- [ ] Create exactly:

```toml
[project]
name = "crux-idm-runtime"
version = "0.0.0"
requires-python = "==3.11.*"
dependencies = [
  "inverse-drum-machine @ git+https://github.com/bernardo-torres/inverse-drum-machine.git@456656868538205ef756912c7cf5b0fd936de8af",
]
```

Do not modify root `pyproject.toml` or root `uv.lock`.

### Step 2: Resolve/freeze the environment

- [ ] Run:

```bash
uv lock --project runtime/idm
uv sync --project runtime/idm --frozen
```

Expected: both exit 0 and `runtime/idm/uv.lock` binds the exact source revision and upstream-compatible dependency set, including librosa 0.9.x behavior used by `PeakPicking`.

If resolution requires patching/upgrading upstream semantics, stop HPA-396 instead.

### Step 3: Record pinned source/model/license evidence

- [ ] Verify from the pinned repository:

```text
package version = 0.1.0
root LICENSE = Apache-2.0
no separate checkpoint/weights LICENSE or NOTICE overrides root license
model.yaml exists
checkpoint exists
training data config names StemGMD
```

Record in the implementation evidence:

```text
checkpoint license basis = repository Apache-2.0; no separate weight notice found
training data provenance = StemGMD; carry forward for HPA-329 operational/license discussion
```

If a contradictory checkpoint-specific notice is discovered, stop before scored work.

### Step 4: Acquire the exact demo/model bytes into explicit local paths

- [ ] Materialize the pinned repository files into the ignored local model/demo area without using checkpoint discovery.

Required paths:

```text
artifacts/idm/pretrained/idm-44-train-kits/checkpoints/model.yaml
artifacts/idm/pretrained/idm-44-train-kits/checkpoints/val-epoch=518-global_step=0.ckpt
```

Record SHA-256 + byte lengths for Task 1. No recursive `find_checkpoint()` or alternate-model fallback.

### Step 5: Verify `demo/mix.wav`; do not convert it

- [ ] Using isolated `soundfile`:

```python
import soundfile as sf

info = sf.info(demo_path)
assert info.format == "WAV"
assert info.subtype == "PCM_16"
assert info.samplerate == 44100
assert info.channels == 1
samples, sr = sf.read(demo_path, dtype="float32", always_2d=True)
assert sr == 44100
assert samples.shape[1] == 1
```

No conversion, resample, rewrite, normalization, or librosa loader.

### Step 6: Direct-load the verified model — no `idm.inference`

- [ ] Run in isolated Python:

```python
from omegaconf import OmegaConf
import hydra
import torch

cfg = OmegaConf.load(model_root / "checkpoints" / "model.yaml")
model = hydra.utils.instantiate(cfg)
ckpt = torch.load(
    model_root / "checkpoints" / "val-epoch=518-global_step=0.ckpt",
    map_location="cpu",
)
model.load_state_dict(ckpt.get("state_dict", ckpt), strict=True)
model = model.to(device).eval()
```

This exact construction is the future worker load path.

### Step 7: Assert every model fact Task 1 will freeze

- [ ] Assert:

```python
assert tuple(model.train_classes) == (
    "CY_CR", "CY_RD", "HH_CHH", "HH_OHH", "KD", "SD", "TT_HFT", "TT_HMT", "TT_LMT"
)
assert model.encoder.sampling_rate == 44100
assert model.encoder.transform.hop_length == 256
assert model.encoder.frame_rate == 44100 / 256

picker = model.decoder.peak_picking_val
assert picker.div_max == 20
assert picker.div_avg == 10
assert picker.div_wait == 16
assert picker.div_thre == 5
assert picker.normalize is False

velocity_act = model.encoder.transcription_head.velocity_act
assert velocity_act.keywords["exponent"] == 10.0
assert velocity_act.keywords["max_value"] == 2
assert velocity_act.keywords["threshold"] == 1e-7
```

If the upstream object representation differs but semantics are inspectable, assert the same values through the actual pinned object shape. Do not silently omit them.

### Step 8: Prove the no-resample encoder + PeakPicking path

- [ ] Run:

```python
audio = torch.from_numpy(samples[:, 0]).to(device).unsqueeze(0)
outputs = model.encoder(audio)
assert outputs["activations"]["onset"].shape[1] == 9
assert outputs["activations"]["velocity"].shape == outputs["activations"]["onset"].shape
assert outputs["activation_rate"] == 44100 / 256
assert torch.isfinite(outputs["activations"]["onset"]).all()
assert torch.isfinite(outputs["activations"]["velocity"]).all()

onset_scores = model.decoder.activation(outputs["activations"]["onset"])
onsets = model.decoder.peak_picking_val(
    onset_scores,
    activation_rate=outputs["activation_rate"],
)
assert onsets.shape == onset_scores.shape
assert torch.isfinite(onset_scores).all()
```

Expected: PASS without cwd/rootutils tricks and without importing `idm.inference`.

### Step 9: Commit the proven runtime lock

- [ ] Commit only after Steps 1–8 pass:

```bash
git add runtime/idm/pyproject.toml runtime/idm/uv.lock
git commit -m "build: freeze IDM benchmark runtime"
```

---

## Task 1: Freeze direct model bytes, loader/extraction semantics, and thin inference identity

**Files:**
- Create: `src/benchmark/idm_model.py`
- Create: `scripts/freeze_idm_model.py`
- Create during real freeze: `runtime/idm/model.json`
- Create: `tests/benchmark/test_idm_model.py`

**Interfaces:**

```python
IDM_MODEL_SCHEMA = "crux.idm-model/v1"
IDM_RELEASE_COMMIT = "456656868538205ef756912c7cf5b0fd936de8af"
IDM_MODEL_ID_RE = re.compile(r"idm-44-train-kits-[0-9a-f]{12}-[0-9a-f]{12}\Z")
IDM_REQUEST_TIMEOUT_SECONDS = 1800.0
IDM_AUDIO_LOADER_REVISION = "soundfile-preserve-wav/v1"
IDM_VELOCITY_TO_MIDI_REVISION = "clamp-half-round-midi127/v1"
IDM_NATIVE_VELOCITY_PERSISTENCE_REVISION = "quantize-six-canonical-string/v1"
DEFAULT_IDM_MODEL_ROOT = Path("artifacts/idm/pretrained/idm-44-train-kits")

IDM_TRAIN_CLASSES = (
    "CY_CR", "CY_RD", "HH_CHH", "HH_OHH", "KD", "SD", "TT_HFT", "TT_HMT", "TT_LMT"
)

@dataclass(frozen=True)
class IdmModelLock:
    repository_url: str
    repository_revision: str
    package_name: str
    package_version: str
    code_license: str
    weight_license: str
    weight_license_basis: str
    runtime_lock_sha256: str
    python_version: str
    model_name: str
    model_config_relative_path: str
    model_config_sha256: str
    model_config_byte_length: int
    checkpoint_relative_path: str
    checkpoint_sha256: str
    checkpoint_byte_length: int
    model_id: str
    device: str
    dtype: str
    sample_rate_hz: int
    input_channel_count: int
    input_container: str
    input_subtype: str
    audio_loader_revision: str
    resampling: str
    mixdown: str
    normalization: str
    mel_n_fft: int
    mel_hop_length: int
    mel_n_mels: int
    activation_rate_hz: float
    train_classes: tuple[str, ...]
    onset_activation: str
    peak_pick_div_max: int
    peak_pick_div_avg: int
    peak_pick_div_wait: int
    peak_pick_div_threshold: int
    peak_pick_normalize: bool
    velocity_activation: str
    velocity_exponent: float
    velocity_max_value: float
    velocity_threshold: float
    velocity_to_midi_revision: str
    native_velocity_persistence_revision: str
    manual_onset_override: bool
    reconstructed_stems: bool
    masking: str
    chunking_mode: str
    native_output_space_id: str
    native_metadata_schema_id: str
    training_data_map_id: str


def load_idm_model_lock(path: Path) -> IdmModelLock: ...
def derive_idm_model_id(lock: IdmModelLock) -> str: ...
def verify_idm_model_files(lock: IdmModelLock, model_root: Path) -> tuple[Path, Path]: ...
def idm_inference_config(lock_sha256: str, descriptor_sha256: str, input_view_id: str) -> dict[str, object]: ...
def idm_inference_config_sha256(config: Mapping[str, object]) -> str: ...
```

`crux.idm-inference-config/v1` contains exactly:

```text
schema
backend_descriptor_sha256
model_lock_sha256
adapter_revision
prediction_map_version
input_view_id
request_timeout_seconds
```

### Step 1: Write strict model-lock/config red tests

- [ ] Require all Task 0 facts, including:

```python
assert lock.repository_revision == IDM_RELEASE_COMMIT
assert lock.python_version.startswith("3.11.")
assert lock.code_license == "Apache-2.0"
assert lock.weight_license == "Apache-2.0"
assert lock.weight_license_basis == "repository-license-no-separate-weight-notice/v1"
assert lock.sample_rate_hz == 44100
assert lock.input_channel_count == 1
assert lock.input_container == "WAV"
assert lock.input_subtype == "PCM_16"
assert lock.audio_loader_revision == IDM_AUDIO_LOADER_REVISION
assert lock.resampling == "forbidden"
assert lock.mixdown == "forbidden"
assert lock.normalization == "forbidden"
assert lock.mel_hop_length == 256
assert lock.activation_rate_hz == 44100 / 256
assert lock.train_classes == IDM_TRAIN_CLASSES
assert lock.peak_pick_div_max == 20
assert lock.peak_pick_div_avg == 10
assert lock.peak_pick_div_wait == 16
assert lock.peak_pick_div_threshold == 5
assert lock.peak_pick_normalize is False
assert lock.velocity_activation == "exp_sigmoid"
assert lock.velocity_exponent == 10
assert lock.velocity_max_value == 2
assert lock.velocity_threshold == 1e-7
assert lock.manual_onset_override is False
assert lock.reconstructed_stems is False
assert lock.masking == "none"
assert lock.chunking_mode == "none"
assert lock.model_id == derive_idm_model_id(lock)
```

Also test inference config:

```python
config = idm_inference_config(lock_sha, descriptor_sha, IDM_STEM_INPUT_VIEW_ID)
assert config["request_timeout_seconds"] == 1800
```

Reject changed/missing runtime lock/model/config bytes, invalid license fields, reordered classes, changed picker/velocity semantics, invalid device/dtype, invalid timeout, and noncanonical JSON.

### Step 2: Verify red

- [ ] Run:

```bash
uv run pytest tests/benchmark/test_idm_model.py -q
```

Expected: FAIL because module does not exist.

### Step 3: Implement strict lock/config parser

- [ ] Reuse only existing canonical/hash/no-follow helpers. Keep `idm_model.py` free of PyTorch/IDM imports.

### Step 4: Implement the model freezer

- [ ] `scripts/freeze_idm_model.py` must:

1. hash the proven `runtime/idm/uv.lock`;
2. require the exact local model config/checkpoint paths — no scanning;
3. hash/size both files;
4. verify the Task 0 model YAML facts;
5. freeze code/weight license basis;
6. freeze selected device/dtype once for feasibility;
7. derive `model_id` from source/checkpoint digests;
8. write canonical `runtime/idm/model.json`;
9. round-trip through `load_idm_model_lock()` and `verify_idm_model_files()`.

No score/F1 participates.

### Step 5: Run focused tests

- [ ] Run:

```bash
uv run pytest tests/benchmark/test_idm_model.py -q
```

Expected: PASS with synthetic fixtures.

### Step 6: Perform real freeze

- [ ] Run against `DEFAULT_IDM_MODEL_ROOT` after Task 0 passes and commit only the small lock:

```bash
git add src/benchmark/idm_model.py scripts/freeze_idm_model.py runtime/idm/model.json tests/benchmark/test_idm_model.py
git commit -m "feat: freeze IDM benchmark identity"
```

---

## Task 2: Refactor proven prediction mapping duplicate, then add IDM closed identity/metadata/zero-hit contracts

**Files:**
- Modify: `src/benchmark/backend_identity.py`
- Modify: `src/benchmark/prediction_artifact.py`
- Modify: `src/benchmark/taxonomy.py`
- Modify: `src/benchmark/mapping.py`
- Modify: `src/benchmark/cohort_scoring.py`
- Modify: `tests/benchmark/test_backend_identity.py`
- Modify: `tests/benchmark/test_prediction_artifact.py`
- Modify: `tests/benchmark/test_prediction_artifact_coverage.py`
- Modify: `tests/benchmark/test_mapping.py`
- Modify: `tests/benchmark/test_cohort_scoring.py`

### Step 1: Characterize existing OaF/MuScriptor mapping before refactor

- [ ] Add/retain tests pinning current mapped/unmapped events, descriptor checks, diagnostics, and empty/nonempty behavior for both existing backends.

- [ ] Run:

```bash
uv run pytest tests/benchmark/test_mapping.py -q
```

Expected: PASS before refactor.

### Step 2: Extract the private shared mapping body

- [ ] Implement only:

```python
def _map_prediction(
    prediction: NativePrediction,
    prediction_map: PredictionMap,
    key_of: Callable[[NativeEvent], str],
) -> tuple[MappedPrediction, MappingDiagnostics]: ...
```

Thin wrappers:

```python
def map_muscriptor_prediction(...):
    return _map_prediction(prediction, prediction_map, lambda e: str(e.native_midi_note))


def map_oaf_prediction(...):
    return _map_prediction(
        prediction,
        prediction_map,
        lambda e: "null" if e.native_metadata["upstream_8hit_group_id"] is None
        else str(e.native_metadata["upstream_8hit_group_id"]),
    )
```

- [ ] Re-run `tests/benchmark/test_mapping.py`; bytes/domain outputs must be unchanged.

### Step 3: Write IDM descriptor/taxonomy/mapping red tests

- [ ] Freeze:

```python
IDM_BACKEND_ID = "idm-44-train-kits-v1"
IDM_PREDICTION_MAP_ID = "crux.prediction-map/idm-44-train-kits-v1"
IDM_NATIVE_METADATA_SCHEMA_ID = "idm-peak-event-metadata-v1"
```

Require all nine mappings, especially:

```python
assert IDM_PREDICTION_MAP.classes["TT_LMT"] == ClassMapping(None, "tom")
```

Require `map_idm_prediction()` to key on `native_class_id`.

### Step 4: Write IDM metadata-policy red tests

- [ ] Current metadata policy values are allowed-value sets. Add tests requiring the policy to support exact key sets plus value predicates.

Valid examples:

```python
{"frame_index": "0", "native_velocity": "0"}
{"frame_index": "215", "native_velocity": "1.337421"}
{"frame_index": "99999", "native_velocity": "2"}
```

Reject:

```text
negative/noncanonical frame string
extra/missing key
noncanonical/nonfinite decimal string
native_velocity < 0
native_velocity > 2.000001
```

Do not reject `"0"`.

### Step 5: Write native IDM prediction-v2 red tests

- [ ] Use:

```python
NativeEvent(
    time_sec=1.25,
    native_class_id="KD",
    model_output_bin=4,
    native_midi_note=None,
    native_metadata={"frame_index": "215", "native_velocity": "1.337421"},
    confidence=0.83,
    velocity_midi=85,
)
```

Reject class-index mismatch, bin outside `0..8`, MIDI present, invalid confidence, invalid MIDI velocity, malformed metadata, or wrong descriptor family. Existing OaF/MuScriptor behavior remains unchanged.

### Step 6: Write mandatory zero-hit red test

- [ ] Build a valid empty IDM prediction and pass it through `cohort_item_from_validated_prediction_artifact()`.

Expected after implementation: successful item with zero prediction events and `IDM_PREDICTION_MAP_ID`.

### Step 7: Verify new cases red

- [ ] Run:

```bash
uv run pytest \
  tests/benchmark/test_backend_identity.py \
  tests/benchmark/test_prediction_artifact.py \
  tests/benchmark/test_mapping.py \
  tests/benchmark/test_cohort_scoring.py \
  -q
```

Expected: new IDM/predicate cases FAIL.

### Step 8: Implement the closed descriptor/metadata policy changes

- [ ] Add IDM to `_DESCRIPTOR_POLICIES`.

- [ ] Widen metadata value policy explicitly:

```python
MetadataValuePolicy = frozenset[str | None] | Callable[[str | None], bool]
```

Add two small IDM predicates and make `_validate_metadata()` dispatch membership vs predicate. No generic plugin registry.

### Step 9: Add IDM map wrapper

- [ ] Add only:

```python
def map_idm_prediction(...):
    return _map_prediction(prediction, prediction_map, lambda e: e.native_class_id)
```

### Step 10: Add mandatory zero-hit entry

- [ ] In `cohort_scoring.py`:

```python
ZERO_HIT_PREDICTION_MAPS = {
    OAF_BACKEND_ID: OAF_PREDICTION_MAP_ID,
    MUSCRIPTOR_BACKEND_ID: MUSCRIPTOR_PREDICTION_MAP_ID,
    IDM_BACKEND_ID: IDM_PREDICTION_MAP_ID,
}
```

### Step 11: Run focused regressions

- [ ] Run:

```bash
uv run pytest \
  tests/benchmark/test_backend_identity.py \
  tests/benchmark/test_prediction_artifact.py \
  tests/benchmark/test_prediction_artifact_coverage.py \
  tests/benchmark/test_mapping.py \
  tests/benchmark/test_cohort_scoring.py \
  -q
```

Expected: PASS.

### Step 12: Commit

- [ ] Commit mechanical refactor + third-family contract together because the third caller is what justifies the extraction:

```bash
git add src/benchmark/backend_identity.py src/benchmark/prediction_artifact.py src/benchmark/taxonomy.py src/benchmark/mapping.py src/benchmark/cohort_scoring.py tests/benchmark/test_backend_identity.py tests/benchmark/test_prediction_artifact.py tests/benchmark/test_prediction_artifact_coverage.py tests/benchmark/test_mapping.py tests/benchmark/test_cohort_scoring.py
git commit -m "feat: add IDM prediction contract"
```

---

## Task 3: Add the isolated persistent IDM worker/backend with direct model load and diagnostic velocity semantics

**Files:**
- Create: `runtime/idm/worker.py`
- Create: `src/benchmark/backends/idm.py`
- Create: `tests/benchmark/test_idm_backend.py`
- Reuse unchanged: `src/benchmark/worker_process.py`

**Interfaces:**

```python
IDM_ADAPTER_REVISION = "crux.idm-adapter/v1"

class IdmBackendError(RuntimeError):
    code: str

class IdmBackend:
    def __init__(
        self,
        runtime_python: Path,
        model_lock_path: Path,
        model_root: Path = DEFAULT_IDM_MODEL_ROOT,
        *,
        process_factory=WorkerProcess.start,
        timeout_seconds: float = IDM_REQUEST_TIMEOUT_SECONDS,
        descriptor: BackendDescriptor | None = None,
    ) -> None: ...

    def descriptor(self) -> BackendDescriptor: ...
    def transcribe(self, audio: CanonicalAudio) -> NativePrediction: ...
    def close(self) -> None: ...
```

No `input_root`; no IDM close-timeout option.

### Step 1: Write backend lifecycle red tests

- [ ] Cover:

- exact argv begins with `runtime_python` and names explicit model root;
- request timeout forwarded as 1800 by default;
- no `close_timeout_seconds` forwarding/configuration;
- valid ready accepted;
- wrong model/classes/sample rate/frame rate rejected before requests;
- request ID/event decoding;
- worker protocol error -> `worker_protocol_failed` + poison;
- worker start error -> `worker_start_failed`;
- malformed structural event -> `native_event_invalid`;
- close called once; close failure stays run-level finalization evidence.

### Step 2: Write raw-event conversion/diagnostic red tests

- [ ] Given:

```python
{
  "class_index": 4,
  "native_class_id": "KD",
  "frame_index": 215,
  "time_sec": 1.248526,
  "onset_score": 0.83,
  "native_velocity": 1.337421,
}
```

require the expected `NativeEvent`.

Also cover raw valid edges:

```text
native_velocity = 1e-7
native_velocity = 2.0000001
native_velocity = 1.1e-7 -> persisted metadata may become "0"
```

Raw decode rejects nonfinite velocity, but **does not reject the precise expected band**. Instead expose/return enough per-request diagnostics for the runner to count values outside `[1e-7, 2.0000001]`.

`velocity_midi` uses the raw float with clamp.

### Step 3: Verify red

- [ ] Run:

```bash
uv run pytest tests/benchmark/test_idm_backend.py -q
```

Expected: FAIL because backend does not exist.

### Step 4: Implement standalone worker startup — direct model load only

- [ ] `runtime/idm/worker.py` contains no Crux imports and never imports `idm.inference`.

Require explicit `--model-root`, then:

```python
config_path = model_root / "checkpoints" / "model.yaml"
checkpoint_path = model_root / "checkpoints" / "val-epoch=518-global_step=0.ckpt"
cfg = OmegaConf.load(config_path)
model = hydra.utils.instantiate(cfg)
ckpt = torch.load(checkpoint_path, map_location="cpu")
model.load_state_dict(ckpt.get("state_dict", ckpt), strict=True)
model = model.to(device).eval()
```

The host has already SHA-verified both files. Worker startup additionally verifies observed model/class/frame-rate/picker facts against arguments/lock-derived expected values before `ready`.

### Step 5: Implement strict soundfile request decode

- [ ] Require `WAV / PCM_16 / 44100 / mono` and read `float32` without conversion/resampling/mixdown.

### Step 6: Implement transcription-only inference

- [ ] Run encoder → decoder sigmoid → `peak_picking_val(..., activation_rate=...)` only.

For each picked activation capture:

```text
class_index
native_class_id
frame_index
time_sec
onset_score
native_velocity at that exact frame
```

No synthesis/sequencer/masking/manual override.

### Step 7: Implement host-side event conversion

- [ ] Persist:

```python
native_metadata={
    "frame_index": str(frame_index),
    "native_velocity": canonical_six_decimal_string(raw_native_velocity),
}
```

Allow canonical `"0"` after six-decimal quantization.

Count raw values outside `[1e-7, 2.0000001]` as request/run diagnostics; do not classify a valid onset as `native_event_invalid` solely for that expected-band diagnostic.

### Step 8: Run focused tests

- [ ] Run:

```bash
uv run pytest tests/benchmark/test_idm_backend.py tests/benchmark/test_worker_process.py -q
```

Expected: PASS.

### Step 9: Run one real worker smoke on the upstream demo WAV

- [ ] Verify `demo/mix.wav` as canonical; do **not** convert it.

Invoke `IdmBackend` with the isolated Python + frozen model root. Confirm ready identity and structurally valid prediction response. Do not score/tune.

### Step 10: Commit

- [ ] Commit:

```bash
git add runtime/idm/worker.py src/benchmark/backends/idm.py tests/benchmark/test_idm_backend.py
git commit -m "feat: add persistent IDM benchmark backend"
```

---

## Task 4: Build the concrete HPA-328 handoff consumer, one-run/two-view ledger, failure mapping, resume, and HPA-325 reports

**Files:**
- Create: `src/benchmark/idm_pilot_run.py`
- Create: `tests/benchmark/idm_pilot_fixtures.py`
- Create: `tests/benchmark/test_idm_pilot_run.py`
- Create: `tests/benchmark/test_idm_pilot_run_acceptance.py`
- Reuse: `src/benchmark/separation_handoff.py`
- Reuse: `src/benchmark/reference_set_manifest.py`
- Reuse: `src/benchmark/reference_timing_manifest.py`
- Reuse: `src/benchmark/cohort_scoring.py`
- Reuse: `src/benchmark/reports.py`
- Reuse: `src/benchmark/prediction_artifact.py`

**Interfaces:**

```python
IDM_PILOT_RUN_SCHEMA = "crux.idm-stem-pilot-run/v1"
IDM_STEM_INPUT_VIEW_ID = "crux.oaf-htdemucs-drums-mono44k1-pcm16/v1"

IDM_FAILURE_TO_COHORT_REASON = {
    "worker_start_failed": "backend_unavailable",
    "worker_protocol_failed": "backend_unavailable",
    "inference_failed": "inference_failed",
    "native_event_invalid": "inference_failed",
    "upstream_stem_unavailable": "inference_failed",
    "retained_input_invalid": "prediction_artifact_invalid",
    "retained_oaf_prediction_invalid": "prediction_artifact_invalid",
    "prediction_artifact_invalid": "prediction_artifact_invalid",
    "prediction_output_conflict": "prediction_artifact_invalid",
    "prediction_publish_failed": "prediction_artifact_invalid",
}

@dataclass(frozen=True)
class IdmPilotRunRequest:
    separation_handoff_path: Path
    reference_manifest_path: Path
    timing_manifest_path: Path
    separation_artifact_root: Path
    stem_cache_root: Path
    output_dir: Path
    model_lock_path: Path
    runtime_python: Path
    model_root: Path = DEFAULT_IDM_MODEL_ROOT
    resume: bool = False
    crux_commit: str | None = None
```

### Step 1: Write failure-table/request-shape tests

- [ ] Assert mapping values are in `COHORT_FAILURE_REASONS`, scorer enum unchanged, `upstream_stem_unavailable == "inference_failed"`, and no `source_cache_dir` exists on primary request.

### Step 2: Write strict handoff/reference/artifact lineage red tests

- [ ] Require:

- exact production handoff membership;
- explicit reference/timing SHA/version match handoff identity;
- only explicit artifact-owner roots resolve retained paths;
- no-follow regular-file reads;
- retained HTDemucs input hash/format;
- retained OaF prediction bytes/header/source/input identity;
- no directory scan/fallback repair/rerun.

### Step 3: Write snapshot-shape tests

- [ ] One run snapshot has fixed rows with:

```text
items[].oaf
items[].idm
```

Each view carries status/disposition, source/input hashes, prediction evidence, native failure code when applicable, and runtime evidence.

Run identity binds:

```text
schema
handoff SHA/version
reference SHA/version
timing SHA/version
IDM descriptor SHA
IDM model-lock SHA
IDM inference-config SHA
input-view ID
Crux commit
```

### Step 4: Write resume tests

- [ ] Exact identity reuses valid IDM artifacts; drifted/missing retained upstream evidence becomes an explicit failure and never triggers HTDemucs/OaF rerun.

Changing request timeout must change inference-config SHA/run ID.

### Step 5: Write complete-population scoring tests

- [ ] Reconstruct both OaF and IDM cohort items over the same fixed population/reference mapping.

Non-success rows remain in population accounting with fixed failure mapping.

Require only existing:

```python
score_cohort()
write_cohort_reports()
```

### Step 6: Write diagnostic-velocity non-gating tests

- [ ] A valid prediction whose raw native velocity quantizes to `"0"` remains a successful cohort item.

A raw value outside exact expected `[1e-7, 2.0000001]` but still structurally representable increments `native_velocity_out_of_expected_band_count` without changing item/scorer status.

### Step 7: Verify red

- [ ] Run:

```bash
uv run pytest tests/benchmark/test_idm_pilot_run.py tests/benchmark/test_idm_pilot_run_acceptance.py -q
```

Expected: FAIL because runner does not exist.

### Step 8: Implement preflight/snapshot parsing

- [ ] Load immutable handoff + explicit reference/timing manifests + model lock/descriptor/config. Validate roots and identity before worker startup.

### Step 9: Implement sequential persisted execution

- [ ] For each fixed row:

1. verify retained HTDemucs input;
2. reconstruct retained OaF evidence into `oaf` view;
3. resume valid IDM artifact if exact identity matches;
4. otherwise call one persistent `IdmBackend`;
5. map with `map_idm_prediction()`;
6. publish immutable prediction-v2;
7. checkpoint mutable snapshot;
8. preserve native failure code and velocity diagnostic counters.

Do not auto-restart a poisoned worker in this ticket.

### Step 10: Rebuild both HPA-325 report sets

- [ ] Write separate OaF/IDM report directories under the HPA-396 run using unchanged scorer/writer.

### Step 11: Run focused tests

- [ ] Run:

```bash
uv run pytest \
  tests/benchmark/test_idm_pilot_run.py \
  tests/benchmark/test_idm_pilot_run_acceptance.py \
  tests/benchmark/test_cohort_scoring.py \
  tests/benchmark/test_reports.py \
  -q
```

Expected: PASS.

### Step 12: Commit

- [ ] Commit:

```bash
git add src/benchmark/idm_pilot_run.py tests/benchmark/idm_pilot_fixtures.py tests/benchmark/test_idm_pilot_run.py tests/benchmark/test_idm_pilot_run_acceptance.py
git commit -m "feat: run fixed IDM stem pilot"
```

---

## Task 5: Promote existing one-snapshot comparison helpers, then add strict OaF↔IDM comparison

**Files:**
- Modify: `src/benchmark/published_comparison.py`
- Modify: `src/benchmark/separation_comparison.py`
- Modify: `tests/benchmark/test_published_comparison.py`
- Modify: `tests/benchmark/test_separation_comparison.py`
- Create: `src/benchmark/idm_comparison.py`
- Create: `tests/benchmark/test_idm_comparison.py`

**Interfaces:**

```python
def published_status(value: object) -> str: ...

def view_evidence(
    snapshot: Mapping[str, object],
    reports: PublishedCohortReports,
    view_name: str,
) -> PublishedRunEvidence: ...

def failure_histogram(
    snapshot: Mapping[str, object],
    view_name: str,
) -> dict[str, int]: ...

@dataclass(frozen=True)
class IdmComparisonRequest:
    run_path: Path
    output_dir: Path


def compare_oaf_idm(request: IdmComparisonRequest) -> Path: ...
```

### Step 1: Characterize current separation helpers

- [ ] Add/retain tests pinning current `_status`, `_view_evidence`, `_failure_histogram` behavior through `separation_comparison` outputs.

- [ ] Run:

```bash
uv run pytest tests/benchmark/test_separation_comparison.py tests/benchmark/test_published_comparison.py -q
```

Expected: PASS.

### Step 2: Move/rename the three model-neutral helpers

- [ ] Move the bodies to `published_comparison.py` as public label-keyed helpers. Update `separation_comparison.py` to import them.

No new behavior, no callback framework, no registry.

- [ ] Re-run the same tests; existing separation comparison output must remain unchanged.

### Step 3: Write IDM one-snapshot comparison red tests

- [ ] Build `PublishedRunEvidence` for `oaf` and `idm` from one HPA-396 snapshot using `view_evidence()`.

Require:

```python
pairable_ids, exclusions = pairable_success_ids(
    oaf,
    idm,
    selected_ids,
    require_identical_input_hash=True,
    left_label="oaf",
    right_label="idm",
)
```

### Step 4: Pin identity checks from recorded cohort/report evidence

- [ ] Before pairing, require OaF/IDM reports to agree on:

```text
reference_manifest_sha256
reference_timing_version
taxonomy_version
lane_map_version
scoring_version
input_view_id
```

Per pair, source/input SHA must match exactly.

Do not add reference/timing paths to `IdmComparisonRequest`; Task 4 already bound those source manifests to the run. Comparison validates the two recorded identities against each other.

### Step 5: Write output red tests

- [ ] Require deterministic:

```text
summary.json
summary.md
paired_per_song.csv
paired_per_class.csv
```

Summary includes full populations, intersection/exclusions, failure histogram, class coverage, runtime/RTF/memory when available, velocity distribution, and native-velocity expected-band counter.

### Step 6: Verify red

- [ ] Run:

```bash
uv run pytest tests/benchmark/test_idm_comparison.py -q
```

Expected: FAIL because module does not exist.

### Step 7: Implement concrete IDM comparison

- [ ] Follow `separation_comparison.py`'s one-snapshot/multiple-view driver topology. Use only public `published_comparison` helpers.

Do not import `muscriptor_comparison.py` private parsers and do not unify model comparisons in this task.

### Step 8: Run focused tests

- [ ] Run:

```bash
uv run pytest \
  tests/benchmark/test_published_comparison.py \
  tests/benchmark/test_separation_comparison.py \
  tests/benchmark/test_idm_comparison.py \
  -q
```

Expected: PASS.

### Step 9: Commit

- [ ] Commit:

```bash
git add src/benchmark/published_comparison.py src/benchmark/separation_comparison.py src/benchmark/idm_comparison.py tests/benchmark/test_published_comparison.py tests/benchmark/test_separation_comparison.py tests/benchmark/test_idm_comparison.py
git commit -m "feat: compare OaF and IDM stem reports"
```

---

## Task 6: Implement/test the pure smoke selector and freeze script, then run the separate full-mix diagnostic

**Operational prerequisite for the real file:** production HPA-328 handoff exists. If absent, selector/tests/script can land, but `runtime/idm/smoke.json` cannot be generated from fixture IDs.

**Files:**
- Create: `src/benchmark/idm_smoke.py`
- Create: `scripts/freeze_idm_smoke.py`
- Create: `tests/benchmark/test_idm_smoke.py`
- Create after real handoff exists: `runtime/idm/smoke.json`
- Modify: `src/benchmark/idm_pilot_run.py`
- Modify: `tests/benchmark/test_idm_pilot_run.py`
- Reuse: `src/benchmark/corpus_cache.py`
- Reuse: `src/benchmark/input_view.py`

**Interfaces:**

```python
@dataclass(frozen=True)
class IdmSmokeCandidate:
    simfile_id: int
    source_duration_sec: Decimal
    common_reference_event_count: int

@dataclass(frozen=True)
class IdmSmokeCase:
    reason: Literal["short", "long", "sparse", "dense", "median_duration"]
    simfile_id: int


def select_idm_smoke_cases(
    candidates: Sequence[IdmSmokeCandidate],
) -> tuple[IdmSmokeCase, ...]: ...

@dataclass(frozen=True)
class IdmFullMixSmokeRequest:
    separation_handoff_path: Path
    reference_manifest_path: Path
    timing_manifest_path: Path
    smoke_manifest_path: Path
    source_cache_dir: Path
    output_dir: Path
    model_lock_path: Path
    runtime_python: Path
    model_root: Path = DEFAULT_IDM_MODEL_ROOT
    crux_commit: str | None = None
```

### Step 0: Write selector tests before any real smoke membership is chosen

- [ ] Test exact algorithm over synthetic candidates:

1. short = minimum `(duration, simfile_id)`;
2. long = maximum duration with lowest-ID tie break;
3. sparse = minimum `(event_count, simfile_id)` from remaining;
4. dense = maximum event count with lowest-ID tie break from remaining;
5. median = median of durations in remaining pool after first four; even count uses arithmetic mean of middle two; select minimum `(abs(duration-target), duration, simfile_id)`.

Cover:

```text
duration tie
event-count tie
selection without replacement
odd remaining-count median
even remaining-count median
<5 candidates rejection
input ordering does not change output
```

### Step 1: Verify selector red

- [ ] Run:

```bash
uv run pytest tests/benchmark/test_idm_smoke.py -q
```

Expected: FAIL because module does not exist.

### Step 2: Implement pure selector

- [ ] No model/OaF scores, filesystem, random seed, or CLI policy inside selector.

- [ ] Re-run tests; expected PASS.

### Step 3: Implement `freeze_idm_smoke.py`

- [ ] Script loads:

- production HPA-328 handoff;
- explicit reference/timing evidence needed to compute common reference count;
- successful/resumed HTDemucs rows only.

It constructs `IdmSmokeCandidate`s, calls only `select_idm_smoke_cases()`, and writes canonical `crux.idm-smoke/v1`.

No manual ID selection.

### Step 4: Write manifest/script tests

- [ ] Require exactly five unique positive production-member IDs and one each of:

```text
short
long
sparse
dense
median_duration
```

Test that the emitted file equals direct selector output for the same fixture evidence.

### Step 5: Generate/commit the real smoke manifest when production handoff exists

- [ ] Run script on real evidence. Never copy MuScriptor IDs `1..5` or fixture IDs.

Commit `runtime/idm/smoke.json` **before any IDM request against production HPA-328 inputs**.

### Step 6: Keep smoke inputs separate from primary request

- [ ] Assert `IdmPilotRunRequest` has no `source_cache_dir`; only `IdmFullMixSmokeRequest` does.

### Step 7: Add separate full-mix materialization/inference

- [ ] Resolve full mix for exactly smoke IDs with existing source/cache rails. Materialize via historical OaF full-mix canonicalization and run same frozen IDM backend under a distinct full-mix input-view/config identity.

Do not alter primary stem run membership/config.

### Step 8: Score/report smoke separately

- [ ] Write separate five-song reports. Never feed smoke F1 into peak/model/config choices.

### Step 9: Run focused tests

- [ ] Run:

```bash
uv run pytest tests/benchmark/test_idm_smoke.py tests/benchmark/test_idm_pilot_run.py -q
```

Expected: PASS.

### Step 10: Commit

- [ ] Commit selector/script/tests and, only when real, the production smoke manifest:

```bash
git add src/benchmark/idm_smoke.py scripts/freeze_idm_smoke.py tests/benchmark/test_idm_smoke.py src/benchmark/idm_pilot_run.py tests/benchmark/test_idm_pilot_run.py
git add runtime/idm/smoke.json  # only after generated from production evidence
git commit -m "feat: freeze IDM smoke diagnostic"
```

---

## Task 7: Wire one CLI command and verify offline acceptance

**Files:**
- Modify: `src/cli/benchmark.py`
- Modify: `tests/test_cli_benchmark.py`
- Modify: `tests/test_cli_benchmark_coverage.py`

**Interface:**

```text
crux benchmark run-idm-pilot
```

Primary model-root default is `DEFAULT_IDM_MODEL_ROOT`; callers may override explicitly. Full-mix smoke options are separate and only valid with a frozen smoke manifest + source cache.

### Step 1: Write CLI red tests

- [ ] Cover:

- required handoff/reference/timing/artifact roots;
- runtime Python/model lock;
- model-root default/override;
- resume;
- no include/exclude/count/seed/tuning flags;
- canonical JSON outcome;
- 0 complete / 1 partial / 2 fatal convention;
- smoke options accepted/rejected as one consistent set;
- no main-environment IDM import;
- no close-timeout CLI/config flag.

### Step 2: Verify red

- [ ] Run:

```bash
uv run pytest tests/test_cli_benchmark.py tests/test_cli_benchmark_coverage.py -q
```

Expected: new IDM cases FAIL.

### Step 3: Implement thin wiring

- [ ] Construct concrete request values, call runner/comparison/smoke functions, print canonical outcome JSON, propagate exit code. No benchmark policy in Click callbacks.

### Step 4: Run focused HPA-396/CLI tests

- [ ] Run:

```bash
uv run pytest \
  tests/benchmark/test_idm_model.py \
  tests/benchmark/test_idm_backend.py \
  tests/benchmark/test_idm_smoke.py \
  tests/benchmark/test_idm_pilot_run.py \
  tests/benchmark/test_idm_pilot_run_acceptance.py \
  tests/benchmark/test_idm_comparison.py \
  tests/test_cli_benchmark.py \
  tests/test_cli_benchmark_coverage.py \
  -q
```

Expected: PASS.

### Step 5: Commit

- [ ] Commit:

```bash
git add src/cli/benchmark.py tests/test_cli_benchmark.py tests/test_cli_benchmark_coverage.py
git commit -m "feat: expose fixed IDM benchmark pilot"
```

---

## Task 8: Execute the real frozen pilot when production HPA-328 evidence exists, then run repository gates

**Operational prerequisite:** production HPA-328 immutable handoff + retained HTDemucs inputs/OaF predictions + reference/timing manifests are available. If not, leave HPA-396 In Progress with an operational block; do not manufacture acceptance evidence.

### Step 1: Re-run runtime/model verification

- [ ] Run:

```bash
uv sync --project runtime/idm --frozen
uv run pytest tests/benchmark/test_idm_model.py tests/benchmark/test_idm_backend.py -q
```

Expected: PASS.

### Step 2: Verify/freeze production smoke membership before any IDM production inference

- [ ] Load real handoff. If `runtime/idm/smoke.json` is absent, run `scripts/freeze_idm_smoke.py`, verify it against real membership, and commit it **before** stem/full-mix IDM execution.

### Step 3: Run fixed stem pilot with no score-driven changes

- [ ] Execute `crux benchmark run-idm-pilot` using exact production handoff/reference/timing/artifact paths + frozen runtime/model/config.

Do not change:

```text
source revision
checkpoint
PeakPicking params
mapping
timeout
device/dtype
smoke membership
```

after inspecting scores.

Expected: every fixed row has a valid IDM prediction or explicit native failure code; diagnostic velocity anomalies do not change scorer status by themselves.

### Step 4: Publish strict OaF↔IDM comparison + separate full-mix smoke

- [ ] Require identical source/input SHA for headline pairs. Require recorded reference/timing/taxonomy/lane-map/scoring/input-view identity equality. Keep full-mix smoke separate.

### Step 5: Run full repository verification

- [ ] Fresh run:

```bash
uv run pytest
uv run ruff check .
uv run ruff format --check src tests
uv run pylint --errors-only --disable=E1120,E0401 --jobs=1 src
git diff --check main...HEAD
```

Expected: all commands exit 0.

### Step 6: Final scope review

- [ ] Confirm no production registry entry, generic runner/comparison framework, prediction v3, scorer enum change, full-corpus path, required IDM synthesis output, manual override, score-driven tuning, training, or fine-tuning.

### Step 7: Commit only final small evidence/docs if required

- [ ] Keep audio/model weights/generated prediction/report bulk out of Git. Only existing small canonical identity/handoff rails may be committed.
