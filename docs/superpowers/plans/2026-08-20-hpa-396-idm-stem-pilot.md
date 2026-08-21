# HPA-396 Frozen IDM Stem Pilot Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Freeze one official Inverse Drum Machine runtime/model/extraction contract, run its native transcription head over the exact HPA-328 HTDemucs canonical input population, score IDM and retained OaF predictions through the unchanged HPA-325 scorer, and publish exact-input paired OaF→IDM evidence plus a separate fixed five-song full-mix compatibility diagnostic.

**Architecture:** Keep IDM comparator-only. Run the exact upstream package in one isolated Python 3.11 `uv` environment behind the existing persistent `WorkerProcess`. Extend the closed backend descriptor/prediction-v2/taxonomy/zero-hit tables for one third family. Consume the immutable HPA-328 handoff rather than rerunning separators/OaF, keep every fixed row in population accounting, and implement one concrete `idm_pilot_run.py` plus one concrete `idm_comparison.py` around existing HPA-325/public comparison rails.

**Tech Stack:** Main Crux Python 3.12; isolated IDM Python 3.11; `uv`; PyTorch/torchaudio through the pinned upstream package; `soundfile` for exact 44.1 kHz mono PCM16 decode; official `inverse-drum-machine` commit `456656868538205ef756912c7cf5b0fd936de8af`; Click; pytest; Crux canonical JSON/JSONL helpers; HPA-324 taxonomy; HPA-325 scorer/reports; HPA-327 references; HPA-328 immutable separation handoff.

**Spec:** `docs/superpowers/specs/2026-08-20-hpa-396-idm-stem-pilot-design.md`

## Global Constraints

- This is the **single HPA-396 PR**. Continue implementation on this branch/PR after planning review; do not open a second implementation PR.
- Official upstream only: `https://github.com/bernardo-torres/inverse-drum-machine` at `456656868538205ef756912c7cf5b0fd936de8af`.
- Primary model is exactly `idm-44-train-kits`.
- Isolated runtime is Python `3.11.*`; if the pinned dependency set cannot resolve/run there without patching upstream, stop and report the blocker rather than broadening runtime policy.
- Primary population is exactly the production HPA-328 handoff population. No seed/count/include/exclude controls.
- The merged HPA-328 PR currently records that production Task 11 was not run. Real HPA-396 corpus/smoke inference remains blocked until the immutable production handoff exists.
- Primary input bytes are the retained HPA-328 `htdemucs.input` canonical WAV bytes. Do not rerun HTDemucs or recanonicalize successful rows.
- Retained HPA-328 OaF-on-HTDemucs predictions are the comparator. Do not rerun OaF.
- Preserve `crux.oaf-htdemucs-drums-mono44k1-pcm16/v1` as the primary input-view ID.
- Worker WAV decode is `soundfile-preserve-wav/v1`: WAV, 44.1 kHz, mono, PCM16; no librosa load, resample, mixdown, normalization, or rewrite.
- Freeze `IDM_REQUEST_TIMEOUT_SECONDS = 1800.0`; include it in `crux.idm-inference-config/v1` run identity. Freeze `IDM_WORKER_CLOSE_TIMEOUT_SECONDS = 30.0` as lifecycle policy.
- Launch `WorkerProcess.start()` with a `Sequence[str]` beginning with the isolated `runtime_python`; never pass a lone worker `Path`/string.
- Keep `crux.drum-prediction-events/v2`; add IDM as the third closed family without weakening OaF/MuScriptor.
- Add `ZERO_HIT_PREDICTION_MAPS[IDM_BACKEND_ID] = IDM_PREDICTION_MAP_ID` in the same schema-family task. This is mandatory.
- Persist IDM `native_metadata` with exactly `frame_index` and pre-MIDI `native_velocity` as canonical strings. Keep `confidence` and `velocity_midi` diagnostic-only.
- `map_idm_prediction()` keys on `native_class_id`.
- `TT_LMT` maps to `ClassMapping(None, "tom")`.
- Use unchanged `score_cohort()` and `write_cohort_reports()`.
- Paired stem comparison must call `pairable_success_ids(..., require_identical_input_hash=True)`.
- Keep HPA-396 native failure codes in the run snapshot/histogram; map them to the existing HPA-325 failure enum using the fixed table in the spec. Do not extend the scorer enum.
- `upstream_stem_unavailable` maps to HPA-325 `inference_failed`, matching existing source-unavailable runner semantics; it does not map to `prediction_missing`.
- No manual onset override, score-driven tuning, required IDM reconstructed stems, Wiener masking, full-corpus path, production registry entry, generic runner/comparison framework, database, queue, training, or fine-tuning.
- `source_cache_dir` belongs only to the separate full-mix smoke request, never the primary stem request.
- Per-task commits stay on this PR. Keep Task 0/runtime lock separate from later scoring commits.

## Risks / Hard Gates

- **Production HPA-328 handoff absent:** do not fabricate smoke/pilot membership from tests. Offline implementation may proceed; real Task 6/8 execution cannot.
- **Weight license:** missing or contradictory checkpoint-license evidence stops scored work.
- **PeakPicking compatibility:** Task 0 must execute the exact pinned decoder sigmoid + `peak_picking_val` path. Do not replace it with a locally tuned picker if it fails.
- **librosa resampling:** any path that calls default `librosa.load(path)` is a contract violation. The worker uses `soundfile` and rejects noncanonical WAV.
- **Timeout-dependent population:** request timeout changes success/pairable population. The fixed `1800`-second value is hashed into inference identity before scores.
- **Zero-hit artifacts:** valid silence/no-peak output must score through the IDM zero-hit map entry.
- **Metadata recovery cost:** `frame_index` and pre-MIDI native velocity must be persisted before real inference; do not plan a later rerun merely to recover them.

---

### Task 0: Prove the exact upstream runtime, demo, WAV tensor path, and PeakPicking contract

**Files:**
- Create: `runtime/idm/pyproject.toml`
- Create: `runtime/idm/uv.lock`
- Temporary/local only: `runtime/idm/.venv/`
- Verify upstream: `demo/mix.wav`
- Verify upstream: `pretrained/idm-44-train-kits/checkpoints/model.yaml`
- Verify upstream: `pretrained/idm-44-train-kits/checkpoints/val-epoch=518-global_step=0.ckpt`

**Interfaces:**

```toml
[project]
name = "crux-idm-runtime"
version = "0.0.0"
requires-python = "==3.11.*"
dependencies = [
  "inverse-drum-machine @ git+https://github.com/bernardo-torres/inverse-drum-machine.git@456656868538205ef756912c7cf5b0fd936de8af",
]
```

- [ ] **Step 1: Create the isolated runtime project**

Write `runtime/idm/pyproject.toml` exactly with Python 3.11 and the pinned Git dependency above. Do not touch root `pyproject.toml` or root `uv.lock`.

- [ ] **Step 2: Resolve the runtime**

Run:

```bash
uv lock --project runtime/idm
uv sync --project runtime/idm --frozen
```

Expected: both commands exit 0; `runtime/idm/uv.lock` resolves the exact upstream commit. If not, stop HPA-396 rather than patching/upgrading upstream.

- [ ] **Step 3: Verify source/model/license facts before integration**

Using the pinned checkout/runtime, assert:

```text
package version = 0.1.0
root license = Apache-2.0
model.yaml exists
val-epoch=518-global_step=0.ckpt exists
train_classes = CY_CR,CY_RD,HH_CHH,HH_OHH,KD,SD,TT_HFT,TT_HMT,TT_LMT
sample rate = 44100
mel n_fft = 1024
mel hop = 256
mel bins = 128
```

Inspect the pinned repository for checkpoint-specific license/notice text. If the checkpoint licensing basis cannot be established, stop before scored execution.

- [ ] **Step 4: Reproduce the official demo model load**

Use upstream `demo/mix.wav` and the isolated Python:

```python
from pathlib import Path
import torch
from idm.inference import load_model

model, name = load_model(
    "idm-44-train-kits",
    torch.device("cpu"),
    log_dir=Path("pretrained"),
)
assert name == "idm-44-train-kits"
assert tuple(model.train_classes) == (
    "CY_CR", "CY_RD", "HH_CHH", "HH_OHH", "KD", "SD", "TT_HFT", "TT_HMT", "TT_LMT"
)
assert model.encoder.sampling_rate == 44100
assert model.encoder.frame_rate == 44100 / 256
```

Use the pinned upstream checkout as working directory so `pretrained/` resolves exactly.

- [ ] **Step 5: Prove the no-resample soundfile tensor path**

Run a small script in the isolated runtime:

```python
import soundfile as sf
import torch

info = sf.info("demo/mix.wav")
assert info.format == "WAV"
assert info.subtype == "PCM_16"
assert info.samplerate == 44100
assert info.channels == 1
samples, sr = sf.read("demo/mix.wav", dtype="float32", always_2d=True)
assert sr == 44100
assert samples.shape[1] == 1
audio = torch.from_numpy(samples[:, 0]).unsqueeze(0)
outputs = model.encoder(audio)
assert outputs["activations"]["onset"].shape[1] == 9
assert outputs["activations"]["velocity"].shape == outputs["activations"]["onset"].shape
assert outputs["activation_rate"] == 44100 / 256
assert torch.isfinite(outputs["activations"]["onset"]).all()
assert torch.isfinite(outputs["activations"]["velocity"]).all()
```

Do not use `librosa.load()` in this probe.

- [ ] **Step 6: Prove the exact evaluation PeakPicking path**

Continue the same probe:

```python
onset_logits = outputs["activations"]["onset"]
onset_scores = model.decoder.activation(onset_logits)
onsets = model.decoder.peak_picking_val(
    onset_scores,
    activation_rate=outputs["activation_rate"],
)
assert onsets.shape == onset_scores.shape
assert torch.isfinite(onset_scores).all()
```

If this pinned API/path is unusable, stop. Do not substitute custom thresholds or a second picker.

- [ ] **Step 7: Commit the proven isolated runtime lock**

```bash
git add runtime/idm/pyproject.toml runtime/idm/uv.lock
git commit -m "build: freeze IDM benchmark runtime"
```

---

### Task 1: Freeze the model, loader, extraction, and thin inference-config identity

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
IDM_WORKER_CLOSE_TIMEOUT_SECONDS = 30.0
IDM_AUDIO_LOADER_REVISION = "soundfile-preserve-wav/v1"
IDM_VELOCITY_TO_MIDI_REVISION = "clamp-half-round-midi127/v1"
IDM_NATIVE_VELOCITY_PERSISTENCE_REVISION = "quantize-six-canonical-string/v1"
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
    velocity_max_value: float
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

`crux.idm-inference-config/v1` contains only:

```text
schema
backend_descriptor_sha256
model_lock_sha256
adapter_revision
prediction_map_version
input_view_id
request_timeout_seconds
```

- [ ] **Step 1: Write model-lock/inference-config tests first**

Tests must require:

```python
assert lock.repository_revision == IDM_RELEASE_COMMIT
assert lock.python_version.startswith("3.11.")
assert lock.model_name == "idm-44-train-kits"
assert lock.sample_rate_hz == 44100
assert lock.input_channel_count == 1
assert lock.input_container == "WAV"
assert lock.input_subtype == "PCM_16"
assert lock.audio_loader_revision == IDM_AUDIO_LOADER_REVISION
assert lock.resampling == "forbidden"
assert lock.mixdown == "forbidden"
assert lock.train_classes == IDM_TRAIN_CLASSES
assert lock.peak_pick_div_max == 20
assert lock.peak_pick_div_avg == 10
assert lock.peak_pick_div_wait == 16
assert lock.peak_pick_div_threshold == 5
assert lock.peak_pick_normalize is False
assert lock.velocity_to_midi_revision == IDM_VELOCITY_TO_MIDI_REVISION
assert lock.native_velocity_persistence_revision == IDM_NATIVE_VELOCITY_PERSISTENCE_REVISION
assert lock.manual_onset_override is False
assert lock.reconstructed_stems is False
assert lock.masking == "none"
assert lock.chunking_mode == "none"
assert lock.model_id == derive_idm_model_id(lock)

config = idm_inference_config(lock_sha, descriptor_sha, "crux.oaf-htdemucs-drums-mono44k1-pcm16/v1")
assert config["request_timeout_seconds"] == 1800
```

Also reject edited/missing runtime lock/model/config bytes, invalid licenses, wrong loader facts, reordered classes, changed peak settings, invalid device/dtype, invalid timeout, and noncanonical JSON.

- [ ] **Step 2: Verify red**

```bash
uv run pytest tests/benchmark/test_idm_model.py -q
```

Expected: FAIL because the module does not exist.

- [ ] **Step 3: Implement strict lock/config parsing**

Reuse `strict_json_loads()`, `canonical_json_bytes()`, `require_sha256()`, and `read_regular_file_no_follow()`. Keep the module free of IDM/PyTorch imports.

- [ ] **Step 4: Implement the freeze script**

The script must hash the proven `runtime/idm/uv.lock`, fetch/copy only the exact upstream `model.yaml` and checkpoint from the pinned revision, verify fixed YAML facts, establish code/weight license evidence, choose device/dtype once for feasibility, derive `model_id`, write `runtime/idm/model.json`, and round-trip through the strict loader/verifier.

Do not put request timeout or duplicate peak settings into a second model structure; timeout lives in the thin inference config while peak/loader semantics live in the model lock.

- [ ] **Step 5: Run focused tests**

```bash
uv run pytest tests/benchmark/test_idm_model.py -q
```

Expected: PASS with synthetic lock/model fixtures.

- [ ] **Step 6: Perform the real model freeze after Task 0 passes**

Run `scripts/freeze_idm_model.py` against the pinned public model bytes and commit only `runtime/idm/model.json`; checkpoint/config bytes stay under ignored local artifacts.

- [ ] **Step 7: Commit**

```bash
git add src/benchmark/idm_model.py scripts/freeze_idm_model.py runtime/idm/model.json tests/benchmark/test_idm_model.py
git commit -m "feat: freeze IDM benchmark identity"
```

---

### Task 2: Add IDM to every required closed identity table, including zero-hit and peak metadata

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

**Interfaces:**

```python
IDM_BACKEND_ID = "idm-44-train-kits-v1"
IDM_DESCRIPTOR_SCHEMA = "crux.transcription-backend-descriptor/v2"
IDM_PREDICTION_MAP_ID = "crux.prediction-map/idm-44-train-kits-v1"
IDM_NATIVE_METADATA_SCHEMA_ID = "idm-peak-event-metadata-v1"

IDM_PREDICTION_MAP = PredictionMap(
    map_id=IDM_PREDICTION_MAP_ID,
    backend_id=IDM_BACKEND_ID,
    native_output_space_id="idm-44-train-kits-9class-v1",
    classes=MappingProxyType({
        "KD": ClassMapping("kick", "kick"),
        "SD": ClassMapping("snare", "snare"),
        "HH_CHH": ClassMapping("closed_hihat", "hihat"),
        "HH_OHH": ClassMapping("open_hihat", "hihat"),
        "CY_CR": ClassMapping("crash", "crash"),
        "CY_RD": ClassMapping("ride", "ride"),
        "TT_HMT": ClassMapping("high_tom", "tom"),
        "TT_LMT": ClassMapping(None, "tom"),
        "TT_HFT": ClassMapping("low_or_floor_tom", "tom"),
    }),
)
```

- [ ] **Step 1: Write red tests for the third descriptor family**

Require fixed IDM identities, patterned model ID only, exact upstream commit, and unknown-family rejection. Keep existing OaF/MuScriptor tests unchanged.

- [ ] **Step 2: Write red prediction-v2 tests with preserved peak metadata**

Use:

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

Reject:

- bin outside `0..8`;
- bin/class mismatch;
- MIDI note present;
- null/out-of-range confidence or velocity;
- missing/extra metadata keys;
- noncanonical/negative `frame_index` string;
- noncanonical, nonfinite, `<=0`, or `>2` `native_velocity` string;
- changed OaF/MuScriptor semantics.

- [ ] **Step 3: Write zero-hit red test**

Create a valid empty IDM prediction artifact and pass it through `cohort_item_from_validated_prediction_artifact()`. Expected after implementation: success with `prediction_map_version == IDM_PREDICTION_MAP_ID` and zero prediction events.

- [ ] **Step 4: Write mapping red tests**

Assert `map_idm_prediction()` uses `native_class_id` and maps all nine fixed classes exactly, including:

```python
assert IDM_PREDICTION_MAP.classes["TT_LMT"] == ClassMapping(None, "tom")
```

- [ ] **Step 5: Verify red**

```bash
uv run pytest \
  tests/benchmark/test_backend_identity.py \
  tests/benchmark/test_prediction_artifact.py \
  tests/benchmark/test_mapping.py \
  tests/benchmark/test_cohort_scoring.py \
  -q
```

Expected: new IDM cases FAIL.

- [ ] **Step 6: Extend only the closed policy tables/branches**

Add the IDM descriptor family to `_DESCRIPTOR_POLICIES`. Add exact IDM metadata key/value-format validation without introducing a generic metadata plugin. Extend native-event validation with an IDM branch while leaving OaF/MuScriptor invariants unchanged.

- [ ] **Step 7: Add the mandatory zero-hit table entry**

In `cohort_scoring.py`:

```python
ZERO_HIT_PREDICTION_MAPS = {
    OAF_BACKEND_ID: OAF_PREDICTION_MAP_ID,
    MUSCRIPTOR_BACKEND_ID: MUSCRIPTOR_PREDICTION_MAP_ID,
    IDM_BACKEND_ID: IDM_PREDICTION_MAP_ID,
}
```

No conditional/"only if required" wording remains.

- [ ] **Step 8: Implement the native-class mapping**

`map_idm_prediction()` uses:

```python
key = native.native_class_id
class_mapping = prediction_map.classes.get(key)
```

Persist unmapped unexpected classes as evidence.

- [ ] **Step 9: Run focused regressions**

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

- [ ] **Step 10: Commit**

```bash
git add src/benchmark/backend_identity.py src/benchmark/prediction_artifact.py src/benchmark/taxonomy.py src/benchmark/mapping.py src/benchmark/cohort_scoring.py tests/benchmark/test_backend_identity.py tests/benchmark/test_prediction_artifact.py tests/benchmark/test_prediction_artifact_coverage.py tests/benchmark/test_mapping.py tests/benchmark/test_cohort_scoring.py
git commit -m "feat: add IDM prediction contract"
```

---

### Task 3: Add the isolated persistent worker backend with frozen decode/timeout semantics

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
        model_root: Path,
        input_root: Path,
        *,
        process_factory=WorkerProcess.start,
        timeout_seconds: float = IDM_REQUEST_TIMEOUT_SECONDS,
        close_timeout_seconds: float = IDM_WORKER_CLOSE_TIMEOUT_SECONDS,
        descriptor: BackendDescriptor | None = None,
    ) -> None: ...

    def descriptor(self) -> BackendDescriptor: ...
    def transcribe(self, audio: CanonicalAudio) -> NativePrediction: ...
    def close(self) -> None: ...
```

- [ ] **Step 1: Write host-backend lifecycle red tests**

Using fake `process_factory`, cover:

- exact argv sequence begins with `runtime_python`;
- exact timeout values forwarded;
- valid ready payload accepted;
- wrong model/classes/sample rate/frame rate rejected before requests;
- request ID/event decoding;
- worker protocol error -> `worker_protocol_failed` and poisoned process;
- worker start error -> `worker_start_failed`;
- malformed raw event -> `native_event_invalid`;
- close called once; close failure remains run-level finalization evidence.

- [ ] **Step 2: Write raw-event conversion tests**

Given:

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

Require:

```python
event.native_class_id == "KD"
event.model_output_bin == 4
event.native_midi_note is None
event.native_metadata == {"frame_index": "215", "native_velocity": "1.337421"}
event.confidence == 0.83
event.velocity_midi == round((1.337421 / 2.0) * 127)
```

Reject inconsistent class index/name, negative/out-of-range frame, nonfinite values, onset score outside `0..1`, velocity outside `(0,2]`, and time inconsistent with the selected frame beyond the frozen activation-rate tolerance.

- [ ] **Step 3: Verify red**

```bash
uv run pytest tests/benchmark/test_idm_backend.py -q
```

Expected: FAIL because backend does not exist.

- [ ] **Step 4: Implement the standalone worker WAV loader**

`runtime/idm/worker.py` contains no Crux imports. For each request:

```python
info = soundfile.info(path)
if info.format != "WAV" or info.subtype != "PCM_16" or info.samplerate != 44100 or info.channels != 1:
    return error("invalid_request")
samples, sr = soundfile.read(path, dtype="float32", always_2d=True)
if sr != 44100 or samples.shape[1] != 1:
    return error("invalid_request")
audio = torch.from_numpy(samples[:, 0]).to(device).unsqueeze(0)
```

No librosa call is allowed.

- [ ] **Step 5: Implement transcription-only inference**

Load the model once. Run encoder → decoder sigmoid → decoder `peak_picking_val`; extract class index/name, exact picked frame index, time, score, and native velocity. Do not synthesize stems or use manual override.

- [ ] **Step 6: Implement `IdmBackend` by mirroring OaF host lifecycle**

Use process-factory injection, ready validation, exact isolated-python argv sequence, request decoding, poison semantics, and bounded close. Do not introduce a second process abstraction.

- [ ] **Step 7: Run focused tests**

```bash
uv run pytest tests/benchmark/test_idm_backend.py tests/benchmark/test_worker_process.py -q
```

Expected: PASS.

- [ ] **Step 8: Run one real worker smoke in the isolated runtime**

After `runtime/idm/model.json` exists, invoke the worker through `IdmBackend` on upstream `demo/mix.wav` converted/verified as the frozen canonical WAV contract. Confirm ready identity and at least a structurally valid prediction response; do not score/tune it.

- [ ] **Step 9: Commit**

```bash
git add runtime/idm/worker.py src/benchmark/backends/idm.py tests/benchmark/test_idm_backend.py
git commit -m "feat: add persistent IDM benchmark backend"
```

---

### Task 4: Build the concrete HPA-328 handoff consumer, fixed failure mapping, resume ledger, and HPA-325 reports

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
    model_root: Path
    runtime_python: Path
    resume: bool = False
    crux_commit: str | None = None
```

- [ ] **Step 1: Write failure-table and request-shape tests first**

Assert every mapping value is in `COHORT_FAILURE_REASONS`, `upstream_stem_unavailable == "inference_failed"`, and no `source_cache_dir` field exists on `IdmPilotRunRequest`.

- [ ] **Step 2: Write handoff/lineage red tests**

Use synthetic immutable handoff fixtures to require:

- exact 20–30 membership;
- reference/timing file SHA/version match handoff run-level identity;
- only explicit `separation_artifact_root` / `stem_cache_root` resolve retained paths;
- path stays under owner root and is read no-follow;
- retained HTDemucs input bytes hash matches handoff;
- retained OaF prediction bytes/header/source/input identities match handoff;
- no directory scan or fallback repair.

- [ ] **Step 3: Write run identity/resume red tests**

Run identity includes:

```text
schema
handoff manifest SHA/version
reference manifest SHA/version
timing manifest SHA/version
IDM backend descriptor SHA
IDM model lock SHA
IDM inference config SHA
input view ID
Crux commit
```

Changing timeout changes inference-config SHA/run ID. Exact identity resume reuses valid IDM prediction artifacts; changed/missing retained upstream evidence becomes explicit failure and never triggers HTDemucs/OaF rerun.

- [ ] **Step 4: Write complete-population scoring red tests**

For successful HTDemucs rows, create/persist IDM predictions and reconstruct both OaF and IDM `CohortItem`s against the same reference mapping. For non-success rows, preserve the row in population accounting and map failure through the fixed table.

Require `score_cohort()`/`write_cohort_reports()` as the only scorer/writer.

- [ ] **Step 5: Verify red**

```bash
uv run pytest tests/benchmark/test_idm_pilot_run.py tests/benchmark/test_idm_pilot_run_acceptance.py -q
```

Expected: FAIL because runner does not exist.

- [ ] **Step 6: Implement strict preflight and snapshot parsing**

Load `load_separation_pilot_manifest()`, `load_reference_set_manifest()`, and `load_reference_timing_manifest()`. Bind exact hashes/versions. Load model lock/descriptor/inference config before constructing worker. Validate output roots do not alias inputs.

- [ ] **Step 7: Implement sequential persisted execution**

For each fixed row:

1. resolve/re-read retained HTDemucs canonical input;
2. verify bytes/hash/format;
3. if exact valid IDM prediction already exists on resume, reuse it;
4. otherwise call one persistent `IdmBackend`;
5. map with `map_idm_prediction()`;
6. publish immutable prediction-v2;
7. checkpoint mutable HPA-396 snapshot after each row;
8. retain native failure code on failure.

Stop/poison behavior follows the backend disposition; do not auto-restart a poisoned worker in this ticket.

- [ ] **Step 8: Rebuild OaF + IDM HPA-325 reports**

Re-read retained OaF prediction artifacts and new IDM artifacts. Build both cohorts over the same fixed population and write separate report directories with unchanged HPA-325 scoring.

- [ ] **Step 9: Run focused tests**

```bash
uv run pytest \
  tests/benchmark/test_idm_pilot_run.py \
  tests/benchmark/test_idm_pilot_run_acceptance.py \
  tests/benchmark/test_cohort_scoring.py \
  tests/benchmark/test_reports.py \
  -q
```

Expected: PASS.

- [ ] **Step 10: Commit**

```bash
git add src/benchmark/idm_pilot_run.py tests/benchmark/idm_pilot_fixtures.py tests/benchmark/test_idm_pilot_run.py tests/benchmark/test_idm_pilot_run_acceptance.py
git commit -m "feat: run fixed IDM stem pilot"
```

---

### Task 5: Publish strict OaF↔IDM paired comparison from public report helpers

**Files:**
- Create: `src/benchmark/idm_comparison.py`
- Create: `tests/benchmark/test_idm_comparison.py`
- Reuse unchanged: `src/benchmark/published_comparison.py`
- Reuse: `src/benchmark/reports.py`

**Interfaces:**

```python
@dataclass(frozen=True)
class IdmComparisonRequest:
    run_path: Path
    output_dir: Path


def compare_oaf_idm(request: IdmComparisonRequest) -> Path: ...
```

- [ ] **Step 1: Write paired-identity red tests**

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

Reject source hash, input hash, reference/timing/taxonomy/scoring identity mismatch. Report full eligible/success/failure populations before the intersection.

- [ ] **Step 2: Write output red tests**

Require deterministic:

```text
summary.json
summary.md
paired_per_song.csv
paired_per_class.csv
```

Summary includes intersection/exclusion counts, HPA-396 native failure histogram, IDM mapped/unmapped/native-class coverage, runtime/RTF/peak memory when present, and velocity availability/distribution as diagnostics only.

- [ ] **Step 3: Verify red**

```bash
uv run pytest tests/benchmark/test_idm_comparison.py -q
```

Expected: FAIL because module does not exist.

- [ ] **Step 4: Implement the concrete driver**

Mirror `muscriptor_comparison.py`'s driver shape but call only public `published_comparison.py` helpers. Do not import private MuScriptor helpers and do not create a generic comparison registry.

- [ ] **Step 5: Run focused tests**

```bash
uv run pytest tests/benchmark/test_idm_comparison.py tests/benchmark/test_published_comparison.py -q
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/benchmark/idm_comparison.py tests/benchmark/test_idm_comparison.py
git commit -m "feat: compare OaF and IDM stem reports"
```

---

### Task 6: Freeze real five-song smoke membership, then add the separate full-mix diagnostic

**Operational prerequisite:** A production HPA-328 handoff must exist. PR #26 did not produce one. If it is still absent, stop this task without inventing IDs.

**Files:**
- Create after production handoff exists and before any IDM HPA-328/full-mix inference: `runtime/idm/smoke.json`
- Modify: `src/benchmark/idm_pilot_run.py`
- Modify: `tests/benchmark/test_idm_pilot_run.py`
- Reuse: `src/benchmark/corpus_cache.py`
- Reuse: `src/benchmark/input_view.py`

**Interface:**

```python
@dataclass(frozen=True)
class IdmFullMixSmokeRequest:
    separation_handoff_path: Path
    reference_manifest_path: Path
    timing_manifest_path: Path
    smoke_manifest_path: Path
    source_cache_dir: Path
    output_dir: Path
    model_lock_path: Path
    model_root: Path
    runtime_python: Path
    crux_commit: str | None = None
```

- [ ] **Step 1: Freeze membership from the real handoff before inference**

Use only real handoff membership and pre-IDM evidence. Do not copy MuScriptor IDs `1..5`.

Select five unique HPA-328 members before any IDM corpus/full-mix request, using duration/reference-event evidence and no model scores. Record the exact IDs in:

```json
{"cases":[{"reason":"short","simfile_id":0},{"reason":"long","simfile_id":0},{"reason":"dense","simfile_id":0},{"reason":"sparse","simfile_id":0},{"reason":"median_duration","simfile_id":0}],"schema":"crux.idm-smoke/v1"}
```

The zero values above document file shape only; the file must never be committed with zero or fixture IDs. The committed file must contain the five actual production handoff IDs.

A deterministic selection order is preferred when ties occur: lowest `simfile_id` wins after the primary duration/event-count sort. No F1 or model output participates.

- [ ] **Step 2: Write/execute smoke-manifest validation before any IDM corpus inference**

Tests require exactly five unique positive IDs, all present in the loaded production handoff, and exactly one reason from:

```text
short
long
dense
sparse
median_duration
```

Commit `runtime/idm/smoke.json` immediately after these tests pass and before invoking IDM on HPA-328 stem/full-mix inputs.

- [ ] **Step 3: Keep smoke roots out of the primary request**

Assert `IdmPilotRunRequest` still has no `source_cache_dir`. Full-mix source resolution exists only through `IdmFullMixSmokeRequest`.

- [ ] **Step 4: Add separate full-mix materialization/inference**

Resolve authoritative full mix for exactly the five smoke IDs with existing cache/source rails, materialize using the historical OaF full-mix canonicalization, and run the same frozen IDM backend. Use a separate full-mix input-view/config identity.

Do not change stem-pilot membership/config and do not feed smoke results into parameter selection.

- [ ] **Step 5: Score/report smoke separately**

Write a separate five-song report directory. Keep it out of `idm_comparison.py` headline stem files.

- [ ] **Step 6: Run focused tests**

```bash
uv run pytest tests/benchmark/test_idm_pilot_run.py -q
```

Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add runtime/idm/smoke.json src/benchmark/idm_pilot_run.py tests/benchmark/test_idm_pilot_run.py
git commit -m "feat: add frozen IDM full-mix smoke"
```

---

### Task 7: Wire one CLI command and verify offline acceptance

**Files:**
- Modify: `src/cli/benchmark.py`
- Modify: `tests/test_cli_benchmark.py`
- Modify: `tests/test_cli_benchmark_coverage.py`

**Interface:**

```text
crux benchmark run-idm-pilot
```

Primary required inputs mirror `IdmPilotRunRequest`. Full-mix smoke inputs are optional only through explicit smoke options after `runtime/idm/smoke.json` exists; they do not become primary stem-run identity.

- [ ] **Step 1: Write CLI red tests**

Cover:

- required production handoff/reference/timing/artifact roots;
- runtime Python/model lock/model root;
- resume;
- no include/exclude/count/seed/tuning flags;
- canonical JSON outcome;
- 0 complete / 1 partial / 2 fatal convention;
- smoke options rejected unless the smoke manifest/source cache are supplied together;
- no implicit main-environment IDM import.

- [ ] **Step 2: Verify red**

```bash
uv run pytest tests/test_cli_benchmark.py tests/test_cli_benchmark_coverage.py -q
```

Expected: new IDM CLI cases FAIL.

- [ ] **Step 3: Implement thin CLI wiring**

Parse paths/options, construct request values, call the concrete runner/comparison/smoke functions, print canonical outcome JSON, and propagate exit code. No business logic in Click callbacks.

- [ ] **Step 4: Run focused CLI + full HPA-396 tests**

```bash
uv run pytest \
  tests/benchmark/test_idm_model.py \
  tests/benchmark/test_idm_backend.py \
  tests/benchmark/test_idm_pilot_run.py \
  tests/benchmark/test_idm_pilot_run_acceptance.py \
  tests/benchmark/test_idm_comparison.py \
  tests/test_cli_benchmark.py \
  tests/test_cli_benchmark_coverage.py \
  -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/cli/benchmark.py tests/test_cli_benchmark.py tests/test_cli_benchmark_coverage.py
git commit -m "feat: expose fixed IDM benchmark pilot"
```

---

### Task 8: Execute the real frozen pilot when HPA-328 Task 11 evidence exists, then run repository gates

**Operational prerequisite:** production HPA-328 immutable handoff + retained HTDemucs inputs/OaF predictions + reference/timing manifests are available. If not, leave HPA-396 In Progress and record that operational block; do not manufacture a synthetic acceptance run.

**Files:**
- Generated local/published benchmark outputs only; do not commit audio/model weights.
- Commit only small canonical runtime/smoke locks when produced by earlier tasks.

- [ ] **Step 1: Re-run Task 0/1 runtime/model verification against the exact committed locks**

```bash
uv sync --project runtime/idm --frozen
uv run pytest tests/benchmark/test_idm_model.py tests/benchmark/test_idm_backend.py -q
```

Expected: PASS.

- [ ] **Step 2: Verify the production handoff and smoke membership before inference**

Load `load_separation_pilot_manifest()` against the production handoff and verify `runtime/idm/smoke.json` IDs are all members. If the file does not yet contain real production IDs, complete Task 6 Step 1 and commit them first.

- [ ] **Step 3: Run the fixed stem pilot with no score-driven changes**

Execute `crux benchmark run-idm-pilot` using the exact production HPA-328/reference/timing/artifact paths. Do not change model, peak settings, mapping, timeout, device/dtype, or smoke membership after inspecting the first scores.

Expected: every fixed population row has either a validated IDM prediction artifact or an explicit native failure code.

- [ ] **Step 4: Publish OaF↔IDM comparison and five-song smoke**

Require identical source/input SHA for every headline pair. Keep full-mix smoke under its separate view/report path.

- [ ] **Step 5: Run full repository verification**

```bash
uv run pytest
uv run ruff check .
uv run ruff format --check src tests
uv run pylint --errors-only --disable=E1120,E0401 --jobs=1 src
git diff --check main...HEAD
```

Expected: all commands exit 0.

- [ ] **Step 6: Review final scope**

Confirm no production registry entry, generic runner/comparison framework, prediction v3, scorer change, full-corpus path, required IDM synthesis output, manual override, tuning, training, or fine-tuning was added.

- [ ] **Step 7: Commit only final documentation/evidence updates if needed**

Keep generated weights/audio/prediction outputs out of Git unless an existing benchmark-manifest rail explicitly requires a small identity artifact.
