# HPA-328 OaF Drum-Stem Separation Ablation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run one fixed 20–30-song OaF input ablation comparing the persisted full-mix baseline with Spleeter 4-stem drums and standard HTDemucs drums, then publish paired HPA-325 evidence and an immutable HTDemucs handoff for HPA-396.

**Architecture:** Start from current `main` (`d9a124835a020b630db088112fa733b33baee619` or newer), where HPA-395 has already landed the neutral source/materialization/prediction/subset-scoring seams. Keep HPA-328 concrete: two isolated separator commands, one fixed-subset pilot runner, normal OaF prediction-v2 artifacts, HPA-325 scoring, one HPA-328 comparison, and one downstream handoff. Promote only two shared areas now justified by a third consumer: the scorer-local persisted-artifact bridge duplicated by OaF/MuScriptor, and the HPA-325 report reader currently embedded in `muscriptor_comparison.py`.

**Tech Stack:** Python 3.12 Crux host, Click, existing canonical JSON/JSONL/CSV helpers, NumPy/librosa/soundfile, isolated Spleeter runtime, isolated Demucs runtime, existing OaF TF1 worker, pytest/Ruff/Pylint.

## Global Constraints

- Current `main` already contains HPA-395 PR #24. Do not wait for or recreate its promoted seams.
- Pilot membership is exactly the supplied HPA-327 `crux.reviewed-reference-subset/v1` population. No second selector, seed, replacement, or score-aware filter.
- Full-mix OaF is never rerun by HPA-328.
- Required separators are exactly official Spleeter 4-stem drums and standard `htdemucs` drums. No third separator in v1.
- Spleeter and Demucs stay out of Crux's main Python dependency graph.
- Freeze exact separator package/repository/model bytes and command arguments before scored results are inspected.
- Fixed QC constants: duration difference `<= max(0.5s, source_duration * 0.005)`, near-silent failure at RMS `<= -80 dBFS`, clipping evidence at absolute peak `>= 0.9999`. None are CLI flags.
- Retain exact separator-produced WAV bytes through HPA-396.
- Derived OaF view IDs are `crux.oaf-spleeter4-drums-mono44k1-pcm16/v1` and `crux.oaf-htdemucs-drums-mono44k1-pcm16/v1`.
- OaF descriptor/model lock/checkpoint/adapter/prediction map and HPA-325 scoring configuration remain unchanged across views.
- Persisted prediction artifacts keep authoritative source-object identity; scorer-local artifacts may normalize only `source_audio_id` to `simfile_id` after persisted bytes are validated.
- Per-view failures remain in population accounting. Paired deltas use matching successful songs only and report independent failure counts separately.
- FP/FN-per-minute is derived only as `fp_or_fn / (source_duration_sec / 60)` from persisted HPA-325 counts and authoritative source duration. It is not a new scorer.
- No full-corpus separator mode, generic experiment runner, plugin registry, queue/worker pool, database, retry framework, SDR scorer, model tuning, or compatibility layer.

---

## File Structure

**Shared seams modified narrowly**

- `src/benchmark/cohort_scoring.py` — scorer-local prediction-source normalization and model-neutral non-success item construction.
- `src/benchmark/oaf_corpus_run.py` — retarget scorer bridge; parameterize OaF inference-config `input_view_id` while preserving full-mix defaults.
- `src/benchmark/muscriptor_corpus_run.py` — retarget only the shared scorer bridge; no MuScriptor behavior change.
- `src/benchmark/input_view.py` — derived-stem canonicalization while preserving authoritative source identity.
- `src/benchmark/reports.py` — shared typed reader for already-published HPA-325 report directories.
- `src/benchmark/muscriptor_comparison.py` — use shared report reader; keep model-family/pairing logic local and output unchanged.
- `src/cli/benchmark.py` — two thin HPA-328 commands.

**New HPA-328 modules**

- `src/benchmark/separators.py` — separator lock validation, concrete Spleeter/HTDemucs execution, exact stem cache, and QC.
- `src/benchmark/separation_pilot.py` — fixed-subset preflight, run snapshot/resume, derived OaF inference, and HPA-325 cohort publication.
- `src/benchmark/separation_comparison.py` — fixed full-mix↔separator joins, FP/FN-per-minute, runtime/storage/cost summary.
- `src/benchmark/separation_handoff.py` — canonical `crux.oaf-separation-pilot/v1` finalization/loader for HPA-396.
- `scripts/freeze_separator_runtime.py` — mechanical lock generator for reproduced isolated runtimes.
- `runtime/separators/spleeter/model.json`
- `runtime/separators/htdemucs/model.json`

**New tests**

- `tests/benchmark/test_separators.py`
- `tests/benchmark/test_separation_pilot.py`
- `tests/benchmark/test_separation_pilot_acceptance.py`
- `tests/benchmark/test_separation_comparison.py`
- `tests/benchmark/test_separation_handoff.py`
- `tests/benchmark/schema_goldens/oaf-separation-pilot-v1.jsonl`

Existing OaF/MuScriptor/scoring/report/CLI tests change only where the public seam is deliberately promoted.

---

### Task 1: Promote the scorer-local persisted-artifact bridge

**Files:**
- Modify: `src/benchmark/cohort_scoring.py`
- Modify: `src/benchmark/oaf_corpus_run.py`
- Modify: `src/benchmark/muscriptor_corpus_run.py`
- Test: `tests/benchmark/test_cohort_scoring.py`
- Test: `tests/benchmark/test_oaf_corpus_run.py`
- Test: `tests/benchmark/test_muscriptor_corpus_run.py`

**Interfaces:**

```python
def cohort_item_from_validated_prediction_artifact(
    identity: CohortIdentity,
    simfile_id: str,
    reference: ReferenceMappingResult,
    prediction: PredictionArtifact,
    *,
    warnings: tuple[str, ...] = (),
) -> CohortItem: ...


def cohort_item_without_prediction(
    identity: CohortIdentity,
    simfile_id: str,
    reference: ReferenceMappingResult | None,
    *,
    status: Literal["failed", "skipped", "quarantined"],
    failure_reason: CohortFailureReason,
    warnings: tuple[str, ...] = (),
) -> CohortItem: ...
```

`cohort_item_from_validated_prediction_artifact()` assumes the caller has already validated persisted bytes against its run row. It changes only the scorer-local `CanonicalAudio.source_audio_id` when necessary; it does not write or replace the persisted artifact.

- [ ] **Step 1: Characterize the current OaF and MuScriptor scorer adaptation**

Run current tests before edits:

```bash
uv run pytest -q \
  tests/benchmark/test_cohort_scoring.py \
  tests/benchmark/test_oaf_corpus_run.py \
  tests/benchmark/test_muscriptor_corpus_run.py
```

Expected: PASS.

- [ ] **Step 2: Add failing helper tests**

Add a test with an artifact whose persisted `source_audio_id="songs/42/audio.ogg"` and `simfile_id="42"`:

```python
item = cohort_item_from_validated_prediction_artifact(
    identity,
    "42",
    reference,
    persisted_artifact,
)
assert item.status == "success"
assert item.prediction_artifact is not None
assert item.prediction_artifact.prediction.audio.source_audio_id == "42"
assert persisted_artifact.prediction.audio.source_audio_id == "songs/42/audio.ogg"
assert item.prediction_artifact.prediction.audio.source_audio_sha256 == (
    persisted_artifact.prediction.audio.source_audio_sha256
)
assert item.prediction_artifact.prediction.audio.input_audio_sha256 == (
    persisted_artifact.prediction.audio.input_audio_sha256
)
```

Also test `cohort_item_without_prediction()` for `failed/inference_failed`, `skipped/explicitly_skipped`, and `quarantined/reference_quarantined` with existing HPA-325 validation.

- [ ] **Step 3: Implement the minimal bridge**

The source-ID adaptation is exactly:

```python
scorer_artifact = prediction
if prediction.prediction.audio.source_audio_id != simfile_id:
    scorer_audio = replace(prediction.prediction.audio, source_audio_id=simfile_id)
    scorer_prediction = replace(prediction.prediction, audio=scorer_audio)
    scorer_artifact = read_prediction_artifact(render_prediction_artifact(scorer_prediction))
return cohort_item_from_artifacts(
    identity,
    simfile_id,
    reference,
    scorer_artifact,
    warnings=warnings,
)
```

Do not relax `_artifact_identity_from_artifacts()` or change persisted source semantics.

- [ ] **Step 4: Retarget OaF and MuScriptor only after run-row validation**

Replace each runner's duplicated scorer-local `replace(... source_audio_id=simfile_id)` block with `cohort_item_from_validated_prediction_artifact()`. Replace duplicated `_cohort_item_without_prediction()` bodies with the shared helper.

Keep each runner's persisted-artifact-vs-run-row checks exactly where they are.

- [ ] **Step 5: Verify no behavior change**

```bash
uv run pytest -q \
  tests/benchmark/test_cohort_scoring.py \
  tests/benchmark/test_oaf_corpus_run.py \
  tests/benchmark/test_oaf_corpus_run_acceptance.py \
  tests/benchmark/test_muscriptor_corpus_run.py \
  tests/benchmark/test_muscriptor_corpus_run_acceptance.py
```

- [ ] **Step 6: Commit**

```bash
git add src/benchmark/cohort_scoring.py src/benchmark/oaf_corpus_run.py \
  src/benchmark/muscriptor_corpus_run.py tests/benchmark
git commit -m "refactor: share persisted cohort scoring bridge"
```

---

### Task 2: Support derived OaF input views without changing full-mix identity

**Files:**
- Modify: `src/benchmark/input_view.py`
- Modify: `src/benchmark/oaf_corpus_run.py`
- Test: `tests/benchmark/test_input_view.py`
- Test: `tests/benchmark/test_oaf_corpus_run.py`
- Test: `tests/benchmark/test_oaf_corpus_run_acceptance.py`
- Test: `tests/benchmark/test_oaf_corpus_run_branches.py`

**Interfaces:**

```python
def materialize_derived_audio(
    source_audio: ResolvedSourceAudio,
    derived_audio_path: Path,
    output_path: Path,
    *,
    input_root: Path,
    input_view_id: str,
    max_input_audio_frames: int | None,
) -> CanonicalAudio: ...


def build_inference_config(
    config: OafModelConfig,
    descriptor: BackendDescriptor,
    model_lock_sha256: str,
    *,
    input_view_id: str = OAF_FULL_MIX_INPUT_VIEW_ID,
) -> dict[str, str]: ...
```

- [ ] **Step 1: Characterize current full-mix bytes and config identity**

Keep/add assertions:

```python
payload = build_inference_config(config, descriptor, "a" * 64)
assert payload["input_view_id"] == OAF_FULL_MIX_INPUT_VIEW_ID
assert payload["canonicalization_revision"] == OAF_CANONICALIZATION_REVISION
```

Run:

```bash
uv run pytest -q tests/benchmark/test_input_view.py tests/benchmark/test_oaf_corpus_run.py
```

Expected: PASS.

- [ ] **Step 2: Add failing derived-view tests**

```python
stem = build_inference_config(
    config,
    descriptor,
    "a" * 64,
    input_view_id="crux.oaf-htdemucs-drums-mono44k1-pcm16/v1",
)
full = build_inference_config(config, descriptor, "a" * 64)
assert stem["input_view_id"] != full["input_view_id"]
assert {k: v for k, v in stem.items() if k != "input_view_id"} == {
    k: v for k, v in full.items() if k != "input_view_id"
}
```

Add a stereo/non-44.1-kHz stem fixture and assert `materialize_derived_audio()` outputs 44.1-kHz mono PCM16, preserves the authoritative source ID/SHA, and hashes the canonical stem bytes as `input_audio_sha256`.

- [ ] **Step 3: Factor only the shared canonicalization body**

Keep `materialize_full_mix_audio()` byte-identical. Both wrappers may call one private helper that performs the existing `librosa.load(... sr=44100, mono=True, res_type="soxr_hq")` + `soundfile.write(... subtype="PCM_16")` sequence.

`materialize_derived_audio()` reads `derived_audio_path`, not `source_audio.path`; `source_audio` supplies only authoritative identity and limits.

- [ ] **Step 4: Parameterize only `input_view_id` in OaF inference config**

Reject empty/non-string values. `build_run_id()` remains full-mix-specific and unchanged.

- [ ] **Step 5: Verify HPA-326 regressions**

```bash
uv run pytest -q \
  tests/benchmark/test_input_view.py \
  tests/benchmark/test_oaf_corpus_run.py \
  tests/benchmark/test_oaf_corpus_run_acceptance.py \
  tests/benchmark/test_oaf_corpus_run_branches.py
```

- [ ] **Step 6: Commit**

```bash
git add src/benchmark/input_view.py src/benchmark/oaf_corpus_run.py tests/benchmark
git commit -m "refactor: support derived OaF input views"
```

---

### Task 3: Freeze the two external separator identities

**Files:**
- Create: `src/benchmark/separators.py`
- Create: `scripts/freeze_separator_runtime.py`
- Create: `runtime/separators/spleeter/model.json`
- Create: `runtime/separators/htdemucs/model.json`
- Test: `tests/benchmark/test_separators.py`

**Interfaces:**

```python
SEPARATOR_LOCK_SCHEMA = "crux.separator-lock/v1"
SPLEETER_SEPARATOR_ID = "spleeter4-drums-v1"
HTDEMUCS_SEPARATOR_ID = "htdemucs-drums-v1"

@dataclass(frozen=True)
class SeparatorModelFile:
    name: str
    sha256: str

@dataclass(frozen=True)
class SeparatorLock:
    separator_id: str
    repository_url: str
    repository_revision: str
    package_name: str
    package_version: str
    model_id: str
    model_files: tuple[SeparatorModelFile, ...]
    code_license: str
    model_license: str
    argv: tuple[str, ...]
    expected_drum_stem_relative_path: str
    output_container: str
    sha256: str


def load_separator_lock(path: Path) -> SeparatorLock: ...
```

- [ ] **Step 1: Write lock parser/hash tests**

Reject unknown/missing keys, noncanonical JSON, duplicate/absolute model filenames, malformed hashes, unsupported separator IDs, and command/model mismatches. Require `spleeter:4stems` for Spleeter and `-n htdemucs` for Demucs.

- [ ] **Step 2: Implement the closed canonical lock**

Reuse `strict_json_loads()`, `canonical_json_bytes()`, and `require_sha256()`. `SeparatorLock.sha256` is the SHA-256 of canonical lock bytes.

- [ ] **Step 3: Implement mechanical runtime freezing**

`freeze_separator_runtime.py` receives an isolated interpreter, repository URL/revision, model ID, explicit downloaded model files, and licenses. It queries package version through that interpreter, hashes supplied model files, strips machine-local directories to basenames, renders canonical JSON, and round-trips through `load_separator_lock()`.

Package probe:

```python
subprocess.run(
    [python, "-c", f"import importlib.metadata as m; print(m.version({package!r}))"],
    check=True,
    capture_output=True,
    text=True,
)
```

- [ ] **Step 4: Reproduce both upstream smoke commands before freezing**

Use a fixed staged input basename `input.wav` so expected output paths are deterministic.

```text
python -m spleeter separate -p spleeter:4stems -o <output> input.wav
python -m demucs -n htdemucs -o <output> input.wav
```

Require decodable `drums.wav` output from each. Capture the exact package/revision/model files actually used. Do not inspect benchmark scores.

- [ ] **Step 5: Generate real checked-in locks and verify**

```bash
uv run pytest -q tests/benchmark/test_separators.py
uv run ruff check src/benchmark/separators.py scripts/freeze_separator_runtime.py \
  tests/benchmark/test_separators.py
```

- [ ] **Step 6: Commit**

```bash
git add src/benchmark/separators.py scripts/freeze_separator_runtime.py \
  runtime/separators tests/benchmark/test_separators.py
git commit -m "feat: freeze separation runtimes"
```

---

### Task 4: Cache exact stems and apply practical fixed QC

**Files:**
- Modify: `src/benchmark/separators.py`
- Test: `tests/benchmark/test_separators.py`

**Interfaces:**

```python
STEM_NEAR_SILENT_DBFS = -80.0
STEM_CLIP_ABS = 0.9999
STEM_MAX_DURATION_DELTA_SECONDS = 0.5
STEM_MAX_DURATION_DELTA_RATIO = 0.005

@dataclass(frozen=True)
class StemQc:
    sample_rate: int
    channel_count: int
    duration_sec: float
    rms_dbfs: float
    peak_abs: float
    clipped_sample_fraction: float
    warnings: tuple[str, ...]

@dataclass(frozen=True)
class SeparatedStem:
    separator_id: str
    separator_config_sha256: str
    source_audio_sha256: str
    path: Path
    sha256: str
    byte_length: int
    separation_wall_time_sec: float
    qc: StemQc


def stem_cache_path(output_root: Path, *, source_audio_sha256: str, lock: SeparatorLock) -> Path: ...
def run_spleeter_drums(...) -> SeparatedStem: ...
def run_htdemucs_drums(...) -> SeparatedStem: ...
```

- [ ] **Step 1: Add failing cache/QC/process tests**

Cover exact cache identity, matching reuse, conflicting existing bytes, nonzero process exit, missing stem, decode failure, zero/near-silent stem, large duration mismatch, >2 channels, sample-rate/channel warnings, clipping evidence, and finite-sample validation.

- [ ] **Step 2: Implement deterministic cache path**

```python
output_root / "derived" / "stems" / lock.separator_id / source_sha / lock.sha256 / "drums.wav"
```

Run each separator in a private temporary directory with staged `input.wav`, read exact output bytes, and publish through existing `publish_immutable_file()`.

- [ ] **Step 3: Implement fixed QC**

```python
samples, sample_rate = soundfile.read(path, always_2d=True, dtype="float32")
channel_count = samples.shape[1]
duration_sec = samples.shape[0] / sample_rate
rms = float(np.sqrt(np.mean(np.square(samples), dtype=np.float64)))
rms_dbfs = float("-inf") if rms == 0.0 else 20.0 * math.log10(rms)
abs_samples = np.abs(samples)
peak_abs = float(np.max(abs_samples))
clipped_fraction = float(np.mean(abs_samples >= STEM_CLIP_ABS))
```

Fail for channel count outside `{1,2}`, nonfinite samples, `rms_dbfs <= -80`, or duration mismatch above `max(0.5, source_duration * 0.005)`. Warn for non-44.1-kHz output, mono output, or peak `>= 0.9999`.

- [ ] **Step 4: Implement exactly two command renderers**

Only source/output paths vary at runtime. No model/shift/overlap/clip-mode flags are exposed.

- [ ] **Step 5: Prove cache hit bypasses subprocess**

Seed matching immutable stem bytes, inject a runner that raises if invoked, and assert the cache/QC path returns the existing stem.

- [ ] **Step 6: Verify and commit**

```bash
uv run pytest -q tests/benchmark/test_separators.py
git add src/benchmark/separators.py tests/benchmark/test_separators.py
git commit -m "feat: cache and validate drum stems"
```

---

### Task 5: Define fixed-subset preflight, run identity, and snapshot

**Files:**
- Create: `src/benchmark/separation_pilot.py`
- Create: `tests/benchmark/test_separation_pilot.py`
- Reuse: `tests/benchmark/reviewed_subset_fixtures.py`

**Interfaces:**

```python
SEPARATION_RUN_SCHEMA = "crux.oaf-separation-run/v1"
SPLEETER_INPUT_VIEW_ID = "crux.oaf-spleeter4-drums-mono44k1-pcm16/v1"
HTDEMUCS_INPUT_VIEW_ID = "crux.oaf-htdemucs-drums-mono44k1-pcm16/v1"

@dataclass(frozen=True)
class OafSeparationPilotRequest:
    reference_manifest_path: Path
    timing_manifest_path: Path
    subset_manifest_path: Path
    oaf_run_path: Path
    cache_dir: Path
    output_dir: Path
    spleeter_python: Path
    demucs_python: Path
    resume: bool = False
    crux_commit: str | None = None

@dataclass(frozen=True)
class OafSeparationPilotOutcome:
    overall_status: Literal["complete", "partial", "failed"]
    exit_code: Literal[0, 1, 2]
    run_id: str | None
    run_path: Path | None
    full_mix_reports_path: Path | None
    spleeter_reports_path: Path | None
    htdemucs_reports_path: Path | None
    comparison_path: Path | None
```

- [ ] **Step 1: Add fatal-preflight tests with process/backend sentinels**

Neither separator nor OaF may be touched for subset/reference mismatch, subset/timing mismatch, parent HPA-326 mismatch, subset member absent from parent population, invalid separator lock, mixed OaF identity, or protected-output aliasing.

- [ ] **Step 2: Add exact-membership test**

A canonical 20-row reviewed subset produces exactly those 20 sorted run rows. There is no selector/seed/count field on the request.

- [ ] **Step 3: Bind deterministic run identity**

Hash exactly:

```python
{
    "schema": SEPARATION_RUN_SCHEMA,
    "reviewed_subset_manifest_sha256": subset.manifest_sha256,
    "reference_manifest_sha256": reference.manifest_sha256,
    "reference_timing_manifest_sha256": timing.manifest_sha256,
    "parent_oaf_run_id": parent_snapshot["run_id"],
    "oaf_backend_descriptor_sha256": parent_snapshot["backend_descriptor_sha256"],
    "oaf_model_lock_sha256": parent_snapshot["model_lock_sha256"],
    "oaf_checkpoint_archive_sha256": parent_snapshot["checkpoint_archive_sha256"],
    "spleeter_lock_sha256": spleeter_lock.sha256,
    "htdemucs_lock_sha256": htdemucs_lock.sha256,
    "spleeter_input_view_id": SPLEETER_INPUT_VIEW_ID,
    "htdemucs_input_view_id": HTDEMUCS_INPUT_VIEW_ID,
    "scoring_version": SCORING_VERSION,
    "crux_commit": request.crux_commit,
}
```

Run ID: `"oaf-separation-" + sha256(canonical_json_bytes(payload)).hexdigest()[:16]`.

- [ ] **Step 4: Persist closed per-view state**

Allowed derived statuses:

```text
pending
separation_failed
stem_invalid
inference_failed
prediction_invalid
success
resumed
```

Each row also stores `source_row_sha256`, authoritative source ID/SHA/duration, parent full-mix status/prediction identity, and nested Spleeter/HTDemucs lock/stem/QC/runtime/canonical-input/prediction evidence.

- [ ] **Step 5: Implement strict snapshot parse/render/write**

Follow HPA-326 canonical JSON + six-place numeric normalization + `atomic_replace_bytes()` + sorted items + exact resume identity. Keep this HPA-328-specific; do not extract a run framework.

- [ ] **Step 6: Reconstruct full-mix control reports without backend execution**

Use `score_reviewed_subset_cohort()` on the persisted HPA-326 parent cohort and write:

```text
<run>/views/full_mix/reports/
```

A fake backend that raises must never be called.

- [ ] **Step 7: Verify and commit**

```bash
uv run pytest -q tests/benchmark/test_separation_pilot.py
git add src/benchmark/separation_pilot.py tests/benchmark/test_separation_pilot.py
git commit -m "feat: define OaF separation pilot run"
```

---

### Task 6: Execute and resume the two derived views through frozen OaF

**Files:**
- Modify: `src/benchmark/separation_pilot.py`
- Modify: `tests/benchmark/test_separation_pilot.py`
- Create: `tests/benchmark/test_separation_pilot_acceptance.py`

**Interfaces:**

```python
def run_oaf_separation_pilot(
    request: OafSeparationPilotRequest,
    *,
    backend_factory: Callable[..., OafBackend] = create_backend,
    spleeter_runner: SeparatorRunner = run_spleeter_drums,
    htdemucs_runner: SeparatorRunner = run_htdemucs_drums,
    perf_counter: Callable[[], float] = time.perf_counter,
    clock: Callable[[], datetime] = _utc_now,
) -> OafSeparationPilotOutcome: ...
```

- [ ] **Step 1: Prove full mix is never inferred**

Fake backend records `CanonicalAudio.input_view_id`; calls contain only the two derived IDs and never `OAF_FULL_MIX_INPUT_VIEW_ID`.

- [ ] **Step 2: Add resume matrix tests**

Cover exact stem+prediction hit (`resumed`, no process/backend), valid stem+missing prediction (OaF only), wrong stem hash (regenerate/fail according to immutable identity), prediction input/config/descriptor mismatch (no reuse), and conflicting immutable prediction (`prediction_invalid`, no overwrite).

- [ ] **Step 3: Resolve all authoritative sources before expensive work**

Use `resolve_source_audio(..., load_body=False)` and persist source ID/SHA/duration. Failure for a fixed member is fatal preflight; never shrink membership.

- [ ] **Step 4: Build two OaF inference configs differing only by view**

```python
spleeter_config = build_inference_config(
    oaf_config,
    descriptor,
    model_lock_sha,
    input_view_id=SPLEETER_INPUT_VIEW_ID,
)
htdemucs_config = build_inference_config(
    oaf_config,
    descriptor,
    model_lock_sha,
    input_view_id=HTDEMUCS_INPUT_VIEW_ID,
)
```

Assert every other field equals the frozen full-mix config.

- [ ] **Step 5: Checkpoint each durable boundary**

Per view:

1. run or validate cached exact stem;
2. checkpoint stem/QC/separation runtime;
3. materialize temporary canonical OaF WAV under `<run>/inputs/<simfile_id>/`;
4. run or validate immutable prediction-v2 artifact via `prediction_path()`;
5. checkpoint input hash, prediction path/hash, OaF runtime/status;
6. delete only temporary canonical WAV in `finally`.

Never delete retained stem.

- [ ] **Step 6: Use one persistent OaF backend for all misses**

Construct lazily. Reuse HPA-326 error classification. Item-local errors fail current view; poison checkpoints and stops later OaF inference in the invocation; no automatic restart.

- [ ] **Step 7: Keep separator failures independent**

Spleeter failure must not prevent HTDemucs for the same song and vice versa.

- [ ] **Step 8: Verify and commit**

```bash
uv run pytest -q tests/benchmark/test_separation_pilot.py \
  tests/benchmark/test_separation_pilot_acceptance.py
git add src/benchmark/separation_pilot.py tests/benchmark/test_separation_pilot.py \
  tests/benchmark/test_separation_pilot_acceptance.py
git commit -m "feat: run OaF on separated drum views"
```

---

### Task 7: Score both derived cohorts through HPA-325 unchanged

**Files:**
- Modify: `src/benchmark/separation_pilot.py`
- Modify: `tests/benchmark/test_separation_pilot_acceptance.py`
- Test: `tests/benchmark/test_cohort_scoring.py`

**Consumes:** `cohort_item_from_validated_prediction_artifact()`, `cohort_item_without_prediction()`, `score_cohort()`, `write_cohort_reports()`.

**Produces:**

```text
<run>/views/spleeter4_drums/reports/
<run>/views/htdemucs_drums/reports/
```

- [ ] **Step 1: Add mixed-success population test**

Use three songs: both-success, Spleeter-fail/HTDemucs-success, Spleeter-success/HTDemucs-fail. Each view's `items.csv` contains all three with correct failure counts.

- [ ] **Step 2: Build view cohort identity from frozen OaF identity**

Only `cohort_id` and `input_view_id` differ. Taxonomy/lane/prediction-map/model/model-lock/descriptor/scoring version stay fixed.

- [ ] **Step 3: Validate persisted prediction before scorer adaptation**

For success/resume, require run-row artifact SHA/source SHA/input-view/input SHA/descriptor/config identity to match the immutable prediction bytes. Then call:

```python
item = cohort_item_from_validated_prediction_artifact(
    identity,
    str(simfile_id),
    mapping,
    persisted_prediction,
    warnings=warnings,
)
```

Assert persisted source object ID remains unchanged while `item.prediction_artifact.prediction.audio.source_audio_id == str(simfile_id)`.

- [ ] **Step 4: Build non-success rows through the shared constructor**

Map separator/materialization/OaF/prediction failures to existing HPA-325 closed reasons. Detailed separator codes/messages remain only in HPA-328 `run.json`.

- [ ] **Step 5: Score/write with no new knobs**

```python
result = score_cohort(identity, cohort_items, diagnostics_for=successful_subset_ids)
write_cohort_reports(result, reports_path)
```

- [ ] **Step 6: Verify and commit**

```bash
uv run pytest -q tests/benchmark/test_separation_pilot_acceptance.py \
  tests/benchmark/test_cohort_scoring.py
git add src/benchmark/separation_pilot.py tests/benchmark/test_separation_pilot_acceptance.py
git commit -m "feat: score separated OaF cohorts"
```

---

### Task 8: Share HPA-325 report reading and publish the fixed comparison

**Files:**
- Modify: `src/benchmark/reports.py`
- Modify: `src/benchmark/muscriptor_comparison.py`
- Test: `tests/benchmark/test_muscriptor_comparison.py`
- Test: `tests/benchmark/test_muscriptor_comparison_coverage.py`
- Create: `src/benchmark/separation_comparison.py`
- Create: `tests/benchmark/test_separation_comparison.py`
- Modify: `src/benchmark/separation_pilot.py`

**Shared report-reader interfaces:**

```python
@dataclass(frozen=True)
class PublishedItemRow:
    simfile_id: str
    status: CohortExecutionStatus

@dataclass(frozen=True)
class PublishedSongRow:
    simfile_id: str
    tolerance_ms: int
    mode: ScoreMode
    tp: int
    fp: int
    fn: int
    precision: Decimal | None
    recall: Decimal | None
    f1: Decimal | None
    prediction_to_reference_ratio: Decimal
    median_abs_error_ms: Decimal | None
    p95_abs_error_ms: Decimal | None
    offset_ms: Decimal | None

@dataclass(frozen=True)
class PublishedClassRow:
    simfile_id: str
    tolerance_ms: int
    mode: ScoreMode
    common_class: str
    reference_support: int
    prediction_support: int
    precision: Decimal | None
    recall: Decimal | None
    f1: Decimal | None

@dataclass(frozen=True)
class PublishedCohortReports:
    identity: CohortIdentity
    items: tuple[PublishedItemRow, ...]
    songs: tuple[PublishedSongRow, ...]
    classes: tuple[PublishedClassRow, ...]


def read_cohort_reports(
    report_dir: Path,
    *,
    expected_identity: CohortIdentity,
) -> PublishedCohortReports: ...
```

**HPA-328 interface:**

```python
@dataclass(frozen=True)
class CostRates:
    compute_cost_per_hour: Decimal = Decimal("0")
    storage_cost_per_gb_month: Decimal = Decimal("0")

@dataclass(frozen=True)
class SeparationComparisonArtifacts:
    summary_path: Path
    per_song_delta_path: Path
    per_class_delta_path: Path
    markdown_path: Path


def write_separation_comparison(
    *,
    full_mix_reports: Path,
    spleeter_reports: Path,
    htdemucs_reports: Path,
    full_mix_identity: CohortIdentity,
    spleeter_identity: CohortIdentity,
    htdemucs_identity: CohortIdentity,
    run_snapshot: Mapping[str, object],
    output_dir: Path,
    cost_rates: CostRates = CostRates(),
) -> SeparationComparisonArtifacts: ...
```

- [ ] **Step 1: Run existing MuScriptor comparison tests before refactor**

```bash
uv run pytest -q tests/benchmark/test_muscriptor_comparison.py \
  tests/benchmark/test_muscriptor_comparison_coverage.py
```

Expected: PASS.

- [ ] **Step 2: Add failing `read_cohort_reports()` tests**

Reject wrong/missing columns, malformed canonical numeric tokens, duplicate item/song/class keys, score rows for non-success items, summary identity mismatch, and row identity mismatch. Accept the current HPA-325 report fixture without changing report schema.

- [ ] **Step 3: Move only report parsing into `reports.py`**

Use the existing `_ITEM_FIELDNAMES`, `_PER_SONG_FIELDNAMES`, and `_PER_CLASS_FIELDNAMES` as the single schema definitions. Do not move run parsing, backend-family checks, pairable-intersection rules, or MuScriptor labels out of `muscriptor_comparison.py`.

- [ ] **Step 4: Retarget MuScriptor comparison with unchanged outputs**

`muscriptor_comparison.py` still validates its OaF/MuScriptor run snapshots and exact source/input pairing, then passes expected cohort identities to `read_cohort_reports()`. Run the full existing comparison tests and require byte-identical fixture outputs where existing tests pin them.

- [ ] **Step 5: Add HPA-328 report-identity tests**

Reject reference/timing/taxonomy/lane/prediction-map/OaF/scoring identity mismatches. The three expected input-view IDs are different by design; all other frozen identity fields must match.

- [ ] **Step 6: Join published HPA-325 rows without rescoring**

Produce full-mix↔Spleeter and full-mix↔HTDemucs intersections. Emit source values plus `derived - full_mix` precision/recall/F1 deltas for matching tolerance/mode/song/class keys. Independent failure populations remain in summary counts.

- [ ] **Step 7: Derive FP/FN per minute**

For each per-song score row, require positive finite `source_duration_sec` from the HPA-328 snapshot:

```python
minutes = Decimal(str(source_duration_sec)) / Decimal("60")
fp_per_min = Decimal(fp) / minutes
fn_per_min = Decimal(fn) / minutes
```

Render six-place canonical tokens and derived-minus-full-mix deltas.

- [ ] **Step 8: Carry timing diagnostics and bounded helped/harmed cases**

Carry `median_abs_error_ms`, `p95_abs_error_ms`, and `offset_ms`. For each view/tolerance/mode, list at most five most helped and five most harmed songs sorted by F1 delta then `simfile_id`; include per-class direction without statistical-significance machinery.

- [ ] **Step 9: Add runtime/storage/cost arithmetic**

```python
compute_cost = Decimal(str(total_wall_time_sec)) / Decimal("3600") * compute_rate
storage_gb = Decimal(total_bytes) / Decimal(1024 ** 3)
storage_monthly_cost = storage_gb * storage_rate
```

Rates default to zero and affect display only.

- [ ] **Step 10: Render deterministic artifacts**

```text
comparison/summary.json
comparison/per_song_delta.csv
comparison/per_class_delta.csv
comparison/summary.md
```

- [ ] **Step 11: Verify and commit**

```bash
uv run pytest -q \
  tests/benchmark/test_muscriptor_comparison.py \
  tests/benchmark/test_muscriptor_comparison_coverage.py \
  tests/benchmark/test_separation_comparison.py \
  tests/benchmark/test_separation_pilot_acceptance.py
git add src/benchmark/reports.py src/benchmark/muscriptor_comparison.py \
  src/benchmark/separation_comparison.py src/benchmark/separation_pilot.py \
  tests/benchmark
git commit -m "refactor: share published benchmark report reading"
```

---

### Task 9: Publish immutable HPA-396 handoff and operator decision

**Files:**
- Create: `src/benchmark/separation_handoff.py`
- Create: `tests/benchmark/test_separation_handoff.py`
- Create: `tests/benchmark/schema_goldens/oaf-separation-pilot-v1.jsonl`

**Interfaces:**

```python
SEPARATION_PILOT_SCHEMA = "crux.oaf-separation-pilot/v1"
SeparationDecision = Literal[
    "keep_full_mix",
    "use_spleeter",
    "use_htdemucs",
    "gather_more_evidence",
    "prioritize_another_model",
]

@dataclass(frozen=True)
class FinalizeSeparationPilotRequest:
    run_path: Path
    subset_manifest_path: Path
    output_manifest: Path
    decision: SeparationDecision
    rationale: str
    compute_cost_per_hour: Decimal = Decimal("0")
    storage_cost_per_gb_month: Decimal = Decimal("0")


def finalize_separation_pilot(request: FinalizeSeparationPilotRequest) -> PublishedManifest: ...
def load_separation_pilot_manifest(path: Path) -> LoadedSeparationPilotManifest: ...
```

- [ ] **Step 1: Add schema golden/parser tests**

Golden validates exact one-row shape/canonical bytes. Production loader/finalizer validates exact HPA-327 20–30 membership.

- [ ] **Step 2: Freeze row identity and nullability**

Every row includes:

```text
simfile_id
source_row_sha256
reviewed_subset_manifest_sha256
reference manifest SHA/version
reference timing SHA/version
taxonomy/lane/prediction-map/scoring versions
source audio ID/SHA
OaF backend/model/model-lock/checkpoint/adapter identities
full-mix status/input/prediction path+SHA
Spleeter status/lock/stem/input/prediction identities
HTDemucs status/lock/stem/input/prediction identities
comparison artifact SHA values
decision/rationale/cost rates
Crux commit
```

`source_row_sha256` must equal the exact HPA-327 row's persisted source-row hash.

- [ ] **Step 3: Re-read/hash artifacts before publication**

Verify retained stems, prediction artifacts, and comparison files against persisted paths/hashes. Missing or edited HTDemucs bytes prevent publication of a successful HTDemucs row.

- [ ] **Step 4: Publish through existing canonical manifest rails**

Reuse `render_manifest()`/`publish_manifest()` style helpers. Repeat shared fields across 20–30 rows rather than creating a header/database protocol.

- [ ] **Step 5: Add HPA-396 consumer test**

For each successful HTDemucs row, downstream code obtains without mutable `run.json`:

```python
(
    row.simfile_id,
    row.source_row_sha256,
    row.source_audio_sha256,
    row.htdemucs_stem_path,
    row.htdemucs_stem_sha256,
    row.htdemucs_input_audio_sha256,
    row.htdemucs_prediction_sha256,
)
```

- [ ] **Step 6: Verify and commit**

```bash
uv run pytest -q tests/benchmark/test_separation_handoff.py
git add src/benchmark/separation_handoff.py tests/benchmark/test_separation_handoff.py \
  tests/benchmark/schema_goldens/oaf-separation-pilot-v1.jsonl
git commit -m "feat: publish separation pilot handoff"
```

---

### Task 10: Add two thin CLI commands

**Files:**
- Modify: `src/cli/benchmark.py`
- Modify: `tests/test_cli_benchmark.py`
- Modify if required: `tests/test_cli_benchmark_coverage.py`

- [ ] **Step 1: Add exact Click signature tests**

Run command exposes only:

```text
--manifest
--timing-manifest
--subset-manifest
--oaf-run
--cache-dir
--output-dir
--spleeter-python
--demucs-python
--resume
```

Finalize command exposes only:

```text
--run
--subset-manifest
--output-manifest
--decision
--rationale
--compute-cost-per-hour
--storage-cost-per-gb-month
```

Assert no model/shift/overlap/QC/tolerance/full-corpus/third-separator flags.

- [ ] **Step 2: Add lazy imports**

Keep external/heavy benchmark modules behind the Click command boundary.

- [ ] **Step 3: Emit canonical summaries and stable exit codes**

Run uses 0 complete / 1 partial / 2 fatal. Finalize exits 0 only after immutable publication. `--decision` is one of the five frozen values; rationale is nonempty; rates are nonnegative.

- [ ] **Step 4: Verify and commit**

```bash
uv run pytest -q tests/test_cli_benchmark.py tests/test_cli_benchmark_coverage.py
git add src/cli/benchmark.py tests/test_cli_benchmark.py tests/test_cli_benchmark_coverage.py
git commit -m "feat: add separation pilot commands"
```

---

### Task 11: Run the real frozen pilot and close verification

**Files:**
- Generated benchmark artifacts stay outside git.

- [ ] **Step 1: Freeze execution evidence before scoring**

```bash
git rev-parse HEAD
sha256sum runtime/separators/spleeter/model.json runtime/separators/htdemucs/model.json
```

Confirm no separator model/QC/scoring tuning flags exist.

- [ ] **Step 2: Run exact reviewed subset**

Resolve the existing HPA-324/HPA-323/HPA-327/HPA-326/HPA-321 artifact paths and isolated interpreters once, then run:

```bash
uv run crux benchmark run-oaf-separation-pilot \
  --manifest "$REFERENCE_MANIFEST" \
  --timing-manifest "$TIMING_MANIFEST" \
  --subset-manifest "$REVIEWED_SUBSET_MANIFEST" \
  --oaf-run "$OAF_RUN" \
  --cache-dir "$CORPUS_CACHE" \
  --output-dir "$SEPARATION_OUTPUT" \
  --spleeter-python "$SPLEETER_PYTHON" \
  --demucs-python "$DEMUCS_PYTHON"
```

If interrupted, repeat the exact command with `--resume`; do not alter locks/config after partial scores are visible.

- [ ] **Step 3: Review required evidence**

Require all HPA-327 members in every population ledger; full-mix/Spleeter/HTDemucs HPA-325 reports; exact stem/prediction identities; 30/50/100-ms per-song/per-class deltas; FP/FN per minute; timing/offset diagnostics; separator/OaF runtime; retained bytes; cost display; bounded helped/harmed cases.

- [ ] **Step 4: Finalize one closed decision**

```bash
uv run crux benchmark finalize-oaf-separation-pilot \
  --run "$SEPARATION_RUN" \
  --subset-manifest "$REVIEWED_SUBSET_MANIFEST" \
  --output-manifest "$SEPARATION_HANDOFF" \
  --decision "$DECISION" \
  --rationale "$RATIONALE" \
  --compute-cost-per-hour "$COMPUTE_RATE" \
  --storage-cost-per-gb-month "$STORAGE_RATE"
```

Use zero rates for local sunk hardware/storage; otherwise use actual marginal rates. Handoff must expose exact HTDemucs stem hashes/paths and OaF-on-HTDemucs prediction identities.

- [ ] **Step 5: Run repository-wide verification**

```bash
uv run pytest -q
uv run ruff check .
uv run ruff format --check src tests scripts
uv run pylint --errors-only --disable=E1120,E0401 --jobs=1 src
git diff --check
git status --short
```

Real separator execution is operational acceptance, not a CI dependency.

- [ ] **Step 6: Scope audit**

```bash
git diff --stat origin/main...HEAD
```

Confirm no full-corpus separator path, generic experiment/plugin framework, bundled model weights, or committed audio/stem/result artifact.

---

## Plan Self-Review

**Latest-main coverage:** Task 1 explicitly closes the scorer-source mismatch found on current `main`; Task 8 reuses the report parsing introduced by HPA-395 rather than duplicating its CSV contracts. Task 2 consumes the final landed neutral materialization/prediction seams without recreating them.

**Spec coverage:** Tasks 5–6 freeze exact membership and reuse the persisted control; Tasks 3–4 freeze/cache/QC both separators; Task 7 reuses HPA-325; Task 8 covers paired accuracy, FP/FN per minute, timing, runtime, storage, and cost; Task 9 provides row-level HPA-327 lineage plus exact HTDemucs/OaF handoff; Task 10 keeps operator surfaces narrow; Task 11 performs the real frozen gate.

**Scope:** The two shared extractions are justified by current duplication and a third caller. OaF/MuScriptor/HPA-328 runners stay concrete; pairing policies stay model/experiment-specific.

**Stable names:** `cohort_item_from_validated_prediction_artifact`, `cohort_item_without_prediction`, `materialize_derived_audio`, `read_cohort_reports`, `SeparatorLock`, `SeparatedStem`, `StemQc`, `OafSeparationPilotRequest`, `OafSeparationPilotOutcome`, `run_oaf_separation_pilot`, `write_separation_comparison`, `FinalizeSeparationPilotRequest`, `finalize_separation_pilot`, `SPLEETER_INPUT_VIEW_ID`, and `HTDEMUCS_INPUT_VIEW_ID` are the shared names used across tasks, CLI, and tests.