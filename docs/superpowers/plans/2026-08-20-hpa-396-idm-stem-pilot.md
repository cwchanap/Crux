# HPA-396 Frozen IDM Stem Pilot Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Freeze one official Inverse Drum Machine runtime/model contract, run its native transcription head over the exact HPA-328 HTDemucs canonical input population, score IDM and retained OaF predictions through the unchanged HPA-325 scorer, and publish exact-input paired OaF→IDM evidence plus a separate five-song full-mix compatibility smoke.

**Architecture:** Keep IDM comparator-only. Run the exact upstream package in one isolated `uv` environment behind the existing persistent `WorkerProcess`. Extend the closed backend descriptor/prediction-v2/taxonomy unions for one third family. Consume the immutable HPA-328 handoff rather than rerunning separators/OaF, keep every fixed row in the population, and build one concrete HPA-396 runner plus one small comparison driver around existing scoring/comparison primitives.

**Tech Stack:** Main Crux Python 3.12, isolated IDM Python selected/frozen during Task 0 (prefer 3.11 if required by upstream dependencies), `uv`, PyTorch/torchaudio, official `inverse-drum-machine` commit `456656868538205ef756912c7cf5b0fd936de8af`, Click, pytest, Crux canonical JSON/JSONL helpers, HPA-324 taxonomy, HPA-325 scorer/reports, HPA-327 subset, HPA-328 separation handoff.

## PR constraint

This draft planning PR is the **single HPA-396 PR**. After planning review, implementation commits must continue on this same branch/PR. Do not open a second implementation PR for HPA-396.

## Global constraints

- Official upstream repository only: `https://github.com/bernardo-torres/inverse-drum-machine`.
- Freeze source commit `456656868538205ef756912c7cf5b0fd936de8af` unless Task 0 proves that exact revision cannot load its committed checkpoint. Do not silently move to a newer commit.
- Primary model is exactly `idm-44-train-kits`.
- Primary population is exactly the finalized HPA-328 handoff population. No seed/count/include/exclude controls.
- Primary input bytes are the retained HPA-328 `htdemucs.input` canonical WAV bytes; do not rerun HTDemucs or re-canonicalize successful rows.
- Retained HPA-328 OaF predictions are the comparator. Do not rerun OaF.
- Use the existing `crux.oaf-htdemucs-drums-mono44k1-pcm16/v1` input-view ID because it identifies the frozen bytes, not the consumer model.
- Use the existing HPA-325 tolerance/alignment/scoring implementation unchanged.
- Keep `crux.drum-prediction-events/v2`; add IDM as the third closed family without weakening OaF/MuScriptor invariants.
- Persist the post-sigmoid onset activation as `confidence`; do not add a Crux confidence threshold.
- Project native IDM velocity `(0, 2]` to the existing `velocity_midi` field with one frozen `round(clamp(v/2, 0, 1) * 127)` transform. Velocity is diagnostic only and never part of onset F1.
- Use IDM's own evaluation sigmoid + `PeakPicking` semantics. No manual onset override and no hand-edited events.
- Do not require IDM synthesized/reconstructed stems. No Wiener masking path is needed for the scored transcription pilot.
- Full-mix work is exactly five predeclared smoke songs and stays separate from the headline stem results.
- No production registry/API integration.
- No full-corpus IDM runner, generic runner/base class, model runtime framework, mapping DSL, experiment framework, database, queue, second scorer, tuning loop, training, or fine-tuning.
- CI remains offline and does not install/download the isolated IDM runtime/model.

---

### Task 0: Prove the exact upstream runtime/model path before building Crux integration

**Purpose:** HPA-396 is experimental integration work. Prove the scientific target is runnable first so implementation does not accumulate around a false upstream assumption.

**Files:**
- Create: `runtime/idm/pyproject.toml`
- Create: `runtime/idm/uv.lock`
- Temporary/local only: isolated `.venv` and downloaded model cache
- Verify upstream: `pretrained/idm-44-train-kits/checkpoints/model.yaml`
- Verify upstream: `pretrained/idm-44-train-kits/checkpoints/val-epoch=518-global_step=0.ckpt`

- [ ] **Step 1: Create the smallest isolated runtime project**

Use one exact git dependency:

```toml
[project]
name = "crux-idm-runtime"
version = "0.0.0"
requires-python = "==<selected-minor>.*"
dependencies = [
  "inverse-drum-machine @ git+https://github.com/bernardo-torres/inverse-drum-machine.git@456656868538205ef756912c7cf5b0fd936de8af",
]
```

Start with Python 3.11 because upstream's dependency set was authored around Python 3.10/3.11-era PyTorch/librosa. If `uv lock` proves 3.11 invalid but another upstream-supported minor works, select that minor once and document the evidence. Do not change Crux's main Python requirement.

- [ ] **Step 2: Resolve and freeze the isolated environment**

```bash
uv lock --project runtime/idm
uv sync --project runtime/idm --frozen
```

Expected: exact git revision appears in `runtime/idm/uv.lock` and the isolated environment imports `idm`.

If dependency resolution cannot produce an upstream-compatible environment without patching the package, stop and record HPA-396 blocked rather than modifying upstream semantics.

- [ ] **Step 3: Verify source/model/license evidence at the pinned revision**

Confirm:

```text
package version: 0.1.0
root code license: Apache-2.0
model config path exists
checkpoint path exists
train_classes are exactly the frozen nine classes
model config sample rate/hop/activation facts match the design
```

Check for a checkpoint-specific license/notice. If the weight-license basis is ambiguous or contradicts the repository license, stop before scored inference and record the blocker.

- [ ] **Step 4: Run one real official model load outside Crux integration**

Using the isolated environment and local copy of the two official model files, execute the upstream load path and inspect:

```python
model, name = load_model("idm-44-train-kits", device, log_dir=<local-pretrained-root>)
assert name == "idm-44-train-kits"
assert model.encoder.sampling_rate == 44100
assert model.encoder.frame_rate == 44100 / 256
```

Run one known-good WAV through `model.encoder()` and assert:

```text
activations["onset"] shape == (1, 9, frames)
activations["velocity"] shape == (1, 9, frames)
all values finite
activation_rate == 44100 / 256
```

Then execute the decoder evaluation peak-picking path without manual override and prove at least the protocol shape is usable.

- [ ] **Step 5: Record the feasibility result in the PR before continuing**

If the exact official path works, continue. If not, update HPA-396/this PR with the concrete blocker; do not substitute another model based on convenience or F1.

- [ ] **Step 6: Commit the isolated runtime lock only after it is proven**

```bash
git add runtime/idm/pyproject.toml runtime/idm/uv.lock
git commit -m "build: freeze IDM benchmark runtime"
```

---

### Task 1: Freeze the IDM model/extraction identity and local model cache

**Files:**
- Create: `src/benchmark/idm_model.py`
- Create: `scripts/freeze_idm_model.py`
- Create during real freeze: `runtime/idm/model.json`
- Create: `tests/benchmark/test_idm_model.py`

**Interfaces:**

```python
IDM_MODEL_SCHEMA = "crux.idm-model/v1"
IDM_RELEASE_COMMIT = "456656868538205ef756912c7cf5b0fd936de8af"
IDM_MODEL_ID_RE = re.compile(
    r"idm-44-train-kits-[0-9a-f]{12}-[0-9a-f]{12}\Z"
)
IDM_TRAIN_CLASSES = (
    "CY_CR",
    "CY_RD",
    "HH_CHH",
    "HH_OHH",
    "KD",
    "SD",
    "TT_HFT",
    "TT_HMT",
    "TT_LMT",
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
```

- [ ] **Step 1: Write strict model-lock tests first**

Cover exact keys/schema and frozen facts:

```python
assert lock.repository_revision == IDM_RELEASE_COMMIT
assert lock.package_name == "inverse-drum-machine"
assert lock.package_version == "0.1.0"
assert lock.model_name == "idm-44-train-kits"
assert lock.sample_rate_hz == 44100
assert lock.mel_n_fft == 1024
assert lock.mel_hop_length == 256
assert lock.mel_n_mels == 128
assert lock.activation_rate_hz == 44100 / 256
assert lock.train_classes == IDM_TRAIN_CLASSES
assert lock.peak_pick_div_max == 20
assert lock.peak_pick_div_avg == 10
assert lock.peak_pick_div_wait == 16
assert lock.peak_pick_div_threshold == 5
assert lock.peak_pick_normalize is False
assert lock.manual_onset_override is False
assert lock.reconstructed_stems is False
assert lock.masking == "none"
assert lock.chunking_mode == "none"
assert lock.model_id == derive_idm_model_id(lock)
```

Also reject edited/missing runtime lock, model config, checkpoint, licenses, revision, class order, extraction settings, device/dtype, digest fragments, sizes, and non-canonical JSON.

- [ ] **Step 2: Verify red**

```bash
uv run pytest tests/benchmark/test_idm_model.py -q
```

Expected: FAIL because `idm_model.py` does not exist.

- [ ] **Step 3: Implement the strict lock/parser/verifier**

Reuse:

```text
strict_json_loads()
canonical_json_bytes()
require_sha256()
read_regular_file_no_follow()
```

Do not import IDM/PyTorch here. This module is pure Crux identity verification and remains testable in the main environment.

The verifier checks only the two model files named by the closed lock beneath the caller-supplied model root; no directory discovery or generic model-cache framework.

- [ ] **Step 4: Implement the one-time freeze script**

The freeze script must:

1. require the proven `runtime/idm/uv.lock` and hash its canonical bytes;
2. acquire/read the two exact model files from the pinned public upstream revision into a caller-selected cache;
3. hash/size both;
4. verify model YAML facts and exact ordered `train_classes`;
5. establish code/weight license evidence;
6. resolve `--device auto` once to an exact supported device and freeze `dtype`;
7. derive `model_id`;
8. write canonical `runtime/idm/model.json`;
9. round-trip it through `load_idm_model_lock()` and `verify_idm_model_files()`.

The freeze script does not run HPA-325 scoring and does not choose a model/config from F1.

- [ ] **Step 5: Run focused tests**

```bash
uv run pytest tests/benchmark/test_idm_model.py -q
```

Expected: PASS offline with synthetic fixture files.

- [ ] **Step 6: Perform the real freeze**

Use an explicit model cache/output, for example:

```bash
uv run python scripts/freeze_idm_model.py \
  --runtime-lock runtime/idm/uv.lock \
  --model-root artifacts/idm/pretrained \
  --device auto \
  --dtype float32 \
  --output runtime/idm/model.json
```

Then rerun the model-lock tests against the committed lock/local cache.

- [ ] **Step 7: Commit**

```bash
git add src/benchmark/idm_model.py scripts/freeze_idm_model.py runtime/idm/model.json tests/benchmark/test_idm_model.py
git commit -m "feat: freeze IDM benchmark identity"
```

---

### Task 2: Add IDM as the third closed descriptor/prediction/taxonomy family

**Files:**
- Modify: `src/benchmark/backend_identity.py`
- Modify: `src/benchmark/prediction_artifact.py`
- Modify: `src/benchmark/taxonomy.py`
- Modify: `src/benchmark/mapping.py`
- Modify: `src/benchmark/cohort_scoring.py` only if its zero-hit closed map table still needs the third family
- Modify: `tests/benchmark/test_backend_identity.py`
- Modify: `tests/benchmark/test_prediction_artifact.py`
- Modify: `tests/benchmark/test_prediction_artifact_coverage.py`
- Modify: `tests/benchmark/test_mapping.py`
- Modify existing taxonomy tests or create `tests/benchmark/test_taxonomy.py` only if that file already fits repository conventions

**Interfaces:**

```python
IDM_BACKEND_ID = "idm-44-train-kits-v1"
IDM_DESCRIPTOR_SCHEMA = "crux.transcription-backend-descriptor/v2"
IDM_PREDICTION_MAP_ID = "crux.prediction-map/idm-44-train-kits-v1"

IDM_PREDICTION_MAP = PredictionMap(
    map_id=IDM_PREDICTION_MAP_ID,
    backend_id=IDM_BACKEND_ID,
    native_output_space_id="idm-44-train-kits-9class-v1",
    classes=...,
)


def map_idm_prediction(
    prediction: NativePrediction,
    prediction_map: PredictionMap = IDM_PREDICTION_MAP,
) -> tuple[MappedPrediction, MappingDiagnostics]: ...
```

- [ ] **Step 1: Write descriptor and prediction-v2 red tests**

Add one valid IDM descriptor shape and failures for:

- wrong fixed identities;
- invalid source commit;
- invalid model ID grammar;
- wrong native output/metadata schema;
- unknown backend still rejected.

Add canonical IDM prediction round-trip tests with:

```python
NativeEvent(
    time_sec=1.25,
    native_class_id="KD",
    model_output_bin=4,
    native_midi_note=None,
    native_metadata={},
    confidence=0.83,
    velocity_midi=97,
)
```

Reject:

- bin outside `0..8`;
- bin/class mismatch;
- MIDI note present;
- null/out-of-range confidence;
- null/out-of-range velocity;
- nonempty metadata;
- malformed native class;
- changed OaF/MuScriptor behavior.

- [ ] **Step 2: Verify red**

```bash
uv run pytest \
  tests/benchmark/test_backend_identity.py \
  tests/benchmark/test_prediction_artifact.py \
  tests/benchmark/test_mapping.py \
  -q
```

Expected: new IDM cases FAIL.

- [ ] **Step 3: Extend the descriptor policy table only**

Add the IDM family to the existing `_DESCRIPTOR_POLICIES`. Do not replace the closed table with a plugin/registry abstraction.

`model_id` follows:

```text
idm-44-train-kits-<revision12>-<checkpoint12>
```

The backend later checks exact lock equality.

- [ ] **Step 4: Extend prediction-v2 native event dispatch**

Add:

```python
IDM_METADATA_SCHEMA = "idm-peak-event-metadata-v1"
NATIVE_METADATA_SCHEMAS[IDM_METADATA_SCHEMA] = {}
```

Dispatch IDM event invariants after descriptor normalization. Keep OaF/MuScriptor branches unchanged.

Keep the existing prediction-v2 golden unchanged. Add focused IDM render/read/render coverage rather than changing the golden's backend family.

- [ ] **Step 5: Freeze the IDM mapping table**

Use:

```text
KD       -> kick / kick
SD       -> snare / snare
HH_CHH   -> closed_hihat / hihat
HH_OHH   -> open_hihat / hihat
CY_CR    -> crash / crash
CY_RD    -> ride / ride
TT_HMT   -> high_tom / tom
TT_LMT   -> null / tom
TT_HFT   -> low_or_floor_tom / tom
```

The `TT_LMT` detailed class stays null because Crux has no mid-tom detailed category. Do not distort it into high/low merely to fill a field.

- [ ] **Step 6: Add `map_idm_prediction()`**

Mirror the small existing OaF/MuScriptor mapping functions. Persist all native events and count unexpected classes as unmapped.

- [ ] **Step 7: Extend zero-hit prediction map identity only if required**

If `cohort_scoring.py` still uses a closed backend→map table for empty artifacts, add IDM there. Do not derive the expected map from cohort identity.

- [ ] **Step 8: Run regression tests**

```bash
uv run pytest \
  tests/benchmark/test_backend_identity.py \
  tests/benchmark/test_prediction_artifact.py \
  tests/benchmark/test_prediction_artifact_coverage.py \
  tests/benchmark/test_mapping.py \
  tests/benchmark/test_cohort_scoring.py \
  -q
```

Expected: PASS with OaF/MuScriptor behavior unchanged.

- [ ] **Step 9: Commit**

```bash
git add src/benchmark/backend_identity.py src/benchmark/prediction_artifact.py src/benchmark/taxonomy.py src/benchmark/mapping.py src/benchmark/cohort_scoring.py tests/benchmark
git commit -m "feat: add frozen IDM prediction family"
```

---

### Task 3: Implement the persistent IDM worker and host backend adapter

**Files:**
- Create: `runtime/idm/worker.py`
- Create: `src/benchmark/backends/idm.py`
- Modify narrowly: `src/benchmark/worker_process.py` only to neutralize OaF-specific wording if necessary; do not change the protocol
- Create: `tests/benchmark/test_idm_backend.py`
- Modify: `tests/benchmark/test_worker_process.py` only if second-caller characterization is needed

**Interfaces:**

```python
IDM_ADAPTER_REVISION = "crux.idm-adapter/v1"

class IdmBackendError(RuntimeError):
    code: str

class IdmBackend:
    def __init__(
        self,
        model_lock_path: Path | IdmModelLock,
        model_root: Path,
        runtime_python: Path,
        *,
        process_factory=WorkerProcess.start,
        timeout_seconds: float = ...,
        close_timeout_seconds: float = ...,
        descriptor: BackendDescriptor | None = None,
    ) -> None: ...

    def descriptor(self) -> BackendDescriptor: ...
    def transcribe(self, audio: CanonicalAudio) -> NativePrediction: ...
    def close(self) -> None: ...
```

- [ ] **Step 1: Write host-adapter tests with a fake process**

Cover:

- exact command construction with isolated Python + worker + verified model root;
- ready payload must match model ID, source revision, sample rate, activation rate, and exact ordered classes;
- one model process reused across multiple `transcribe()` calls;
- source audio must be `CanonicalAudio` at 44.1 kHz mono PCM16;
- worker error mapping;
- process poisoning/close behavior;
- descriptor equality with the model lock;
- raw event validation;
- velocity conversion boundaries `0 -> 0`, `1 -> ~64`, `2 -> 127`, clamping only at the frozen projection boundary;
- non-finite time/onset/velocity rejection;
- bin/native-class mismatch rejection.

- [ ] **Step 2: Verify red**

```bash
uv run pytest tests/benchmark/test_idm_backend.py -q
```

Expected: FAIL because the backend does not exist.

- [ ] **Step 3: Implement `runtime/idm/worker.py`**

The worker has no Crux imports. On startup:

```python
model, name = load_model("idm-44-train-kits", device, log_dir=model_root)
model.eval()
```

Validate observed model facts and emit one ready record.

For each request:

1. read the canonical WAV;
2. require 44.1 kHz mono finite audio;
3. run `model.encoder()`;
4. require `onset`/`velocity` tensors with `(1, 9, frames)` and finite values;
5. compute `onset_scores = model.decoder.activation(onset_logits)`;
6. compute sparse peaks with `model.decoder.peak_picking_val(..., activation_rate=...)`;
7. return raw event dictionaries containing bin/class/frame/time/onset score/native velocity.

Do not call `separate()` because it always continues into synthesis/masking work. Do not call or expose manual onset override.

- [ ] **Step 4: Implement `IdmBackend`**

Host responsibilities:

- load/verify model lock/model files;
- build/validate exact descriptor;
- launch worker using the isolated interpreter;
- verify ready payload;
- validate every raw event;
- map `onset_score -> NativeEvent.confidence`;
- map native velocity to frozen `velocity_midi`;
- leave native MIDI null;
- close the persistent process.

No model/PyTorch import enters the main Crux environment.

- [ ] **Step 5: Run focused tests**

```bash
uv run pytest \
  tests/benchmark/test_idm_backend.py \
  tests/benchmark/test_worker_process.py \
  -q
```

Expected: PASS offline with fake process responses.

- [ ] **Step 6: Run one real worker smoke before runner work**

Use the frozen local model root and one known-good canonical WAV. Verify:

```text
ready succeeds
at least protocol-valid response returns
all event bins/classes match frozen order
confidence is in 0..1
velocity projection is valid
worker is reused for a second request
```

This is a runtime contract check, not a score gate.

- [ ] **Step 7: Commit**

```bash
git add runtime/idm/worker.py src/benchmark/backends/idm.py src/benchmark/worker_process.py tests/benchmark/test_idm_backend.py tests/benchmark/test_worker_process.py
git commit -m "feat: add IDM benchmark backend"
```

---

### Task 4: Build the fixed HPA-328 handoff consumer and resumable IDM stem runner

**Files:**
- Create: `src/benchmark/idm_pilot_run.py`
- Create: `tests/benchmark/idm_pilot_fixtures.py` only if fixtures cannot stay local to the test module
- Create: `tests/benchmark/test_idm_pilot_run.py`
- Create: `tests/benchmark/test_idm_pilot_run_coverage.py` only if branch coverage convention requires it
- Reuse unchanged: `src/benchmark/separation_handoff.py`
- Reuse unchanged: `src/benchmark/prediction_artifact.py`
- Reuse unchanged: `src/benchmark/artifact_io.py`

**Interfaces:**

```python
IDM_PILOT_RUN_SCHEMA = "crux.idm-stem-pilot-run/v1"
IDM_STEM_INPUT_VIEW_ID = "crux.oaf-htdemucs-drums-mono44k1-pcm16/v1"
IDM_INFERENCE_CONFIG_SCHEMA = "crux.idm-inference-config/v1"

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
    source_cache_dir: Path | None = None  # only needed by Task 6 full-mix smoke
    resume: bool = False
    crux_commit: str | None = None

@dataclass(frozen=True)
class IdmPilotRunOutcome:
    overall_status: Literal["complete", "partial", "failed"]
    exit_code: Literal[0, 1, 2]
    run_id: str | None
    run_path: Path | None
    reports_path: Path | None
    success_count: int
    failed_count: int
    upstream_unavailable_count: int
    aggregate_rtf: float | None
    worker_startup_wall_time_sec: float | None
    fatal_reason: str | None = None


def run_idm_pilot(request: IdmPilotRunRequest) -> IdmPilotRunOutcome: ...
def parse_idm_pilot_run(content: bytes) -> dict[str, JsonValue]: ...
def render_idm_pilot_run(snapshot: Mapping[str, object]) -> bytes: ...
```

- [ ] **Step 1: Write strict run-identity/population tests first**

Build synthetic finalized HPA-328 handoff rows and cover:

- `load_separation_pilot_manifest()` is the only membership source;
- exactly the handoff's 20–30 sorted unique rows appear in the IDM snapshot;
- no include/exclude/sample controls exist;
- run ID changes with handoff/model/runtime/inference/scoring/Crux identity;
- run snapshot rejects mixed source/reference/timing/input/model identity;
- output roots may not alias protected inputs;
- runtime/model preflight failure produces fatal exit 2 before mutable run publication.

- [ ] **Step 2: Write retained-artifact binding tests**

For successful HTDemucs rows require and verify:

```text
htdemucs.stem path/hash/source/separator-lock identity
htdemucs.input path/input_view/input SHA/source identity
htdemucs.prediction path/artifact SHA/source/input identity
```

Resolution rules:

- only explicit `separation_artifact_root` / `stem_cache_root` are owners;
- paths must remain beneath their owner root;
- no directory scan or fallback discovery;
- use no-follow regular-file reads;
- hash bytes before use.

The IDM audio passed to the backend is the exact retained `htdemucs.input` file. Do not rematerialize it.

- [ ] **Step 3: Write upstream-failure preservation tests**

Rows with non-successful HTDemucs evidence remain in the fixed HPA-396 population with native failure `upstream_stem_unavailable` and a mapped HPA-325 failure reason. They are not handed to IDM and are not dropped.

- [ ] **Step 4: Verify red**

```bash
uv run pytest tests/benchmark/test_idm_pilot_run.py -q
```

Expected: FAIL because the runner does not exist.

- [ ] **Step 5: Implement preflight and immutable run identity**

Preflight:

1. load reference/timing manifests;
2. load finalized HPA-328 handoff;
3. verify cross-lineage equality;
4. load/verify IDM model + runtime lock;
5. build exact descriptor/inference config;
6. validate artifact owner roots;
7. start backend once and verify ready payload;
8. only then create mutable HPA-396 run state.

No full-mix smoke or scored execution occurs before identity is frozen.

- [ ] **Step 6: Implement sequential primary execution**

For each fixed row in sorted order:

- preserve upstream-unavailable rows as failure;
- validate retained HTDemucs input + OaF prediction;
- construct `CanonicalAudio` from the exact retained input metadata/bytes;
- transcribe with the persistent IDM backend;
- `map_idm_prediction()`;
- publish through `prediction_path()` + `publish_prediction_artifact()`;
- checkpoint the run snapshot after each terminal item;
- record item wall time/RTF only around IDM request/inference, excluding worker startup;
- on item-local failure, persist one stable failure code and continue.

- [ ] **Step 7: Implement resume/conflict semantics**

On resume:

- parse exact existing run identity;
- revalidate retained HPA-328 input/OaF evidence;
- read existing IDM prediction bytes;
- require artifact/audio/descriptor/map identity match;
- reuse only exact matches;
- immutable output conflict is failure, not overwrite.

Do not add a generic run-snapshot helper merely because OaF/MuScriptor/HPA-328 have their own snapshots.

- [ ] **Step 8: Run focused runner tests**

```bash
uv run pytest \
  tests/benchmark/test_idm_pilot_run.py \
  tests/benchmark/test_idm_pilot_run_coverage.py \
  -q
```

Omit the coverage file command if no separate file was needed.

- [ ] **Step 9: Commit**

```bash
git add src/benchmark/idm_pilot_run.py tests/benchmark/test_idm_pilot_run.py tests/benchmark/test_idm_pilot_run_coverage.py tests/benchmark/idm_pilot_fixtures.py
git commit -m "feat: run fixed IDM stem pilot"
```

Only add paths that actually exist.

---

### Task 5: Rebuild HPA-325 OaF/IDM cohorts and publish exact-input paired deltas

**Files:**
- Modify: `src/benchmark/idm_pilot_run.py`
- Create: `src/benchmark/idm_comparison.py`
- Create: `tests/benchmark/test_idm_comparison.py`
- Create: `tests/benchmark/test_idm_comparison_coverage.py` only if needed
- Modify: `tests/benchmark/test_idm_pilot_run.py`
- Reuse unchanged: `src/benchmark/cohort_scoring.py`
- Reuse unchanged: `src/benchmark/reports.py`
- Reuse primitives: `src/benchmark/published_comparison.py`

**Interfaces:**

```python
IDM_COMPARISON_SCHEMA = "crux.oaf-idm-comparison/v1"

@dataclass(frozen=True)
class IdmComparisonRequest:
    oaf_evidence: PublishedRunEvidence
    idm_evidence: PublishedRunEvidence
    output_dir: Path


def compare_oaf_idm(request: IdmComparisonRequest) -> None: ...
```

- [ ] **Step 1: Write cohort reconstruction tests**

For every HPA-396 row:

- build one OaF cohort item from the retained HPA-328 OaF prediction artifact when valid;
- build one IDM cohort item from the HPA-396 artifact when valid;
- use the same preflighted reference mapping object;
- represent every failure with `cohort_item_without_prediction()`;
- preserve exact input-view/source/input hash identity.

Assert that no OaF inference function is called.

- [ ] **Step 2: Write exact pairable-set tests**

Call:

```python
pairable_success_ids(
    oaf,
    idm,
    selected_ids=None,
    require_identical_input_hash=True,
    left_label="oaf",
    right_label="idm",
)
```

Cover:

- source mismatch excludes pair;
- input mismatch is an integrity error;
- OaF-only/IDM-only success counts are visible;
- failed upstream stem rows remain in population counts but not paired rows;
- per-song/per-class score grids must match exactly.

- [ ] **Step 3: Verify red**

```bash
uv run pytest \
  tests/benchmark/test_idm_pilot_run.py \
  tests/benchmark/test_idm_comparison.py \
  -q
```

Expected: new scoring/comparison cases FAIL.

- [ ] **Step 4: Publish HPA-325 reports for both retained OaF and IDM**

Use exactly:

```python
score_cohort(...)
write_cohort_reports(...)
```

for the same fixed HTDemucs population. Do not copy cells from HPA-328 comparison CSVs; rebuild from the retained prediction artifacts so OaF↔IDM pairing is independent of HPA-328 full-mix success.

Suggested layout:

```text
<run>/views/htdemucs/oaf/reports/
<run>/views/htdemucs/idm/reports/
```

- [ ] **Step 5: Implement the small IDM comparison driver**

Reuse model-neutral helpers from `published_comparison.py` for paired song/class rows, deltas, aggregate deltas, population, and runtime fields. Keep only schema/title/summary assembly IDM-specific.

Publish:

```text
<run>/comparison/summary.json
<run>/comparison/summary.md
<run>/comparison/paired_per_song.csv
<run>/comparison/paired_per_class.csv
```

- [ ] **Step 6: Add diagnostics without changing scorer semantics**

Include:

- mapped/unmapped native class counts;
- confidence/onset activation distribution summaries;
- MIDI-projected velocity distribution summaries;
- failure-code histogram;
- worker startup wall time;
- per-item/aggregate RTF;
- peak memory only if already available cheaply from process/runtime evidence.

Do not add confidence/velocity weighting to HPA-325.

- [ ] **Step 7: Add representative error rows mechanically**

Derive a small displayed list from the complete paired CSV, e.g. largest F1 regressions/improvements under one frozen tolerance/mode used only for presentation. Do not make a hidden hand-curated subset and do not change aggregate calculations.

- [ ] **Step 8: Run focused tests**

```bash
uv run pytest \
  tests/benchmark/test_idm_pilot_run.py \
  tests/benchmark/test_idm_comparison.py \
  tests/benchmark/test_idm_comparison_coverage.py \
  -q
```

Omit nonexistent coverage file.

- [ ] **Step 9: Commit**

```bash
git add src/benchmark/idm_pilot_run.py src/benchmark/idm_comparison.py tests/benchmark/test_idm_pilot_run.py tests/benchmark/test_idm_comparison.py tests/benchmark/test_idm_comparison_coverage.py
git commit -m "feat: compare IDM with OaF on fixed stems"
```

---

### Task 6: Add the fixed five-song full-mix compatibility smoke without contaminating the primary result

**Files:**
- Create before first score inspection: `runtime/idm/smoke.json`
- Modify: `src/benchmark/idm_pilot_run.py`
- Modify: `tests/benchmark/test_idm_pilot_run.py`
- Reuse: `src/benchmark/corpus_cache.py`
- Reuse: `src/benchmark/input_view.py`

**Smoke contract:**

```text
exactly 5 unique positive HPA-327 simfile IDs
one pre-model reason per ID
membership frozen before IDM score inspection
```

Suggested reason vocabulary:

```text
short
long
dense
sparse
non_drum_heavy
```

- [ ] **Step 1: Select/freeze smoke membership from existing reference/source evidence**

Selection may use durations/reference hit counts and source listening needed to classify the reason. It must not use IDM/OaF F1 to select songs.

Write and commit `runtime/idm/smoke.json` before running the five scores.

- [ ] **Step 2: Write smoke isolation tests**

Assert:

- exactly five IDs;
- every ID belongs to the HPA-328 handoff population;
- no duplicate reasons/IDs if the chosen contract requires one each;
- smoke results do not enter primary HTDemucs cohort rows;
- smoke results do not alter primary inference config/model identity;
- no F1 pass/fail threshold exists.

- [ ] **Step 3: Materialize/reuse exact canonical full mix**

Use retained full-mix canonical input if the handoff retains a usable path. Otherwise use the authoritative source hash from the handoff/reference row and the existing:

```python
resolve_source_audio(...)
materialize_full_mix_audio(...)
```

with historical input view:

```text
crux.oaf-full-mix-mono44k1-pcm16/v1
```

Do not invent an IDM-specific full-mix canonicalization.

- [ ] **Step 4: Run IDM through the same frozen backend/config**

Persist smoke prediction artifacts separately, score/report separately, and include a `full_mix_smoke` section in `summary.json`/`summary.md`.

If one or more smoke songs fail, report failures. Primary stem execution remains valid if the frozen IDM runtime itself works on the required HTDemucs input path.

- [ ] **Step 5: Run focused tests**

```bash
uv run pytest tests/benchmark/test_idm_pilot_run.py -q
```

- [ ] **Step 6: Commit**

```bash
git add runtime/idm/smoke.json src/benchmark/idm_pilot_run.py tests/benchmark/test_idm_pilot_run.py
git commit -m "feat: add IDM full-mix smoke diagnostic"
```

---

### Task 7: Expose one HPA-396 CLI path and freeze the command contract

**Files:**
- Modify: `src/cli/benchmark.py`
- Modify: `tests/test_cli_benchmark.py`
- Modify: `tests/test_cli_benchmark_coverage.py`

**Command:**

```text
crux benchmark run-idm-pilot
```

No generic `--model`, `--separator`, `--sample-count`, threshold, masking, manual-onset, tuning, or policy flags.

Required paths should expose only the fixed external evidence/runtime roots needed by `IdmPilotRunRequest`, for example:

```text
--separation-handoff
--reference-manifest
--timing-manifest
--separation-artifact-root
--stem-cache-root
--output-dir
--idm-model-lock
--idm-model-root
--idm-python
--source-cache-dir   # only for full-mix smoke fallback
--resume
```

- [ ] **Step 1: Write CLI red tests**

Cover:

- exact required options;
- `--resume` only behavioral flag;
- typed path conversion;
- request construction;
- exit 0 complete / exit 1 partial / exit 2 fatal;
- canonical result JSON includes run/report paths/counts/RTF/startup time/fatal reason only as intended;
- no forbidden sampling/tuning/manual-onset flags.

- [ ] **Step 2: Verify red**

```bash
uv run pytest tests/test_cli_benchmark.py tests/test_cli_benchmark_coverage.py -q
```

- [ ] **Step 3: Implement the thin Click wrapper**

CLI owns parsing/result presentation only. `run_idm_pilot()` owns validation/execution/reporting.

Do not create a second CLI module unless `benchmark.py` structure forces it.

- [ ] **Step 4: Run focused CLI + runner tests**

```bash
uv run pytest \
  tests/test_cli_benchmark.py \
  tests/test_cli_benchmark_coverage.py \
  tests/benchmark/test_idm_pilot_run.py \
  tests/benchmark/test_idm_comparison.py \
  -q
```

- [ ] **Step 5: Commit**

```bash
git add src/cli/benchmark.py tests/test_cli_benchmark.py tests/test_cli_benchmark_coverage.py
git commit -m "feat: expose fixed IDM pilot command"
```

---

### Task 8: Run the real frozen smoke + 20–30-song pilot and close HPA-396 without post-score tuning

**Files:**
- No required production changes unless a test-first defect is found
- Real local artifacts under caller-selected artifact directories
- Update this PR description / HPA-396 evidence with command outcomes

- [ ] **Step 1: Verify the committed runtime/model locks against the real local cache**

```bash
uv sync --project runtime/idm --frozen
uv run pytest tests/benchmark/test_idm_model.py -q
```

Run the model verifier against the real model root as part of the CLI preflight.

- [ ] **Step 2: Re-run one real worker smoke after final code changes**

Confirm the ready payload and one real retained HTDemucs input still match the committed lock/adapter assumptions.

- [ ] **Step 3: Execute the fixed five-song full-mix smoke**

Do not change model/threshold/map membership based on the scores. Record compatibility/failure/runtime evidence.

- [ ] **Step 4: Execute the exact HPA-328 handoff population**

Example shape:

```bash
uv run crux benchmark run-idm-pilot \
  --separation-handoff <hpa328-final-manifest.jsonl> \
  --reference-manifest <reference-manifest.jsonl> \
  --timing-manifest <timing-manifest.jsonl> \
  --separation-artifact-root <hpa328-artifact-root> \
  --stem-cache-root <hpa328-stem-cache-root> \
  --output-dir <idm-pilot-output> \
  --idm-model-lock runtime/idm/model.json \
  --idm-model-root <local-idm-pretrained-root> \
  --idm-python runtime/idm/.venv/bin/python \
  --source-cache-dir <source-cache>
```

Expected: complete or explicit partial outcome over the entire fixed population; never an implicitly smaller cohort.

- [ ] **Step 5: Inspect the published evidence, not hidden scratch calculations**

Verify:

```text
OaF + IDM HPA-325 report grids exist
paired per-song/per-class CSVs use exact source+input hashes
all HPA-328 rows appear in population counts
failure histogram reconciles with run items
IDM native class coverage reconciles with prediction artifacts
velocity/confidence diagnostics reconcile with prediction artifacts
full-mix smoke is separate
runtime/RTF counts reconcile with executed rows
```

- [ ] **Step 6: Do not tune after inspection**

Unexpected scientific performance is a result, not an implementation bug. Only fix a concrete contract/implementation defect with a failing test. Do not change peak-picking thresholds, map policy, selected model, device/dtype, smoke membership, or primary population because F1 is disappointing.

- [ ] **Step 7: Run full repository verification**

Use the repository's current documented gates. At minimum:

```bash
uv run pytest -q
uv run ruff check .
uv run ruff format --check src tests
uv run pylint --errors-only --disable=E1120,E0401 src
git diff --check main...HEAD
```

If the current `CLAUDE.md` documents different authoritative commands at implementation time, follow those commands instead of stale plan text.

- [ ] **Step 8: Final diff review**

Confirm:

```text
one HPA-396 PR only
no production registry/API IDM integration
no generic runtime/runner framework
no full-corpus IDM path
no required synthesized IDM stems
no HPA-325 semantic changes
no HPA-327/HPA-328 membership changes
no score-driven tuning
```

- [ ] **Step 9: Update the same draft PR with implementation evidence and mark ready only when verified**

Do not create a second HPA-396 implementation PR.

## Final acceptance checklist

- [ ] Exact official IDM source commit frozen.
- [ ] Exact runtime lock, model config, checkpoint, and license evidence frozen.
- [ ] Exact observed sample/frame/class/extraction contract validated.
- [ ] Persistent isolated worker reuses one model load.
- [ ] No manual onset override or required synthesis/masking path.
- [ ] Prediction-v2 supports IDM while preserving OaF/MuScriptor behavior.
- [ ] IDM native mapping is frozen, including `TT_LMT -> common tom / detailed null`.
- [ ] Exact HPA-328 handoff population is preserved.
- [ ] Primary IDM inference consumes byte-identical HPA-328 HTDemucs canonical inputs.
- [ ] Retained HPA-328 OaF predictions are validated and not rerun.
- [ ] OaF and IDM use the unchanged HPA-325 scorer.
- [ ] Paired deltas require identical source and input SHA-256.
- [ ] Failure population is explicit; no silent row dropping.
- [ ] Onset activation + velocity diagnostics are persisted without entering F1.
- [ ] Five-song full-mix smoke is frozen before score inspection and reported separately.
- [ ] Runtime/RTF/failure/class evidence reconciles with run rows.
- [ ] No tuning/training/full-corpus/production/generic-framework scope creep.
- [ ] Full repository verification passes.
- [ ] HPA-396 implementation remains in this single PR.
