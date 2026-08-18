# HPA-328 OaF Drum-Stem Separation Ablation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run one fixed 20–30-song OaF input ablation comparing the persisted full-mix baseline with Spleeter 4-stem drums and standard HTDemucs drums, then publish paired HPA-325 evidence and an immutable HTDemucs handoff for HPA-396.

**Architecture:** Start from current `main` (`d9a124835a020b630db088112fa733b33baee619` or newer). Keep HPA-328 concrete: two isolated separator commands, one fixed-subset pilot runner, the existing OaF transcribe→map→publish path, HPA-325 scoring, one fixed comparison, and one downstream handoff. Promote only two real duplicated seams: the scorer-local persisted-artifact bridge used by OaF/MuScriptor/HPA-328, and a model-neutral reader for HPA-325 reports whose schemas are already owned by `reports.py`.

**Tech Stack:** Python 3.12 Crux host, Click, existing canonical JSON/JSONL/CSV helpers, NumPy/librosa/soundfile, isolated Spleeter runtime, isolated Demucs runtime, existing OaF TF1 worker, pytest/Ruff/Pylint.

## Global Constraints

- Pilot membership is exactly the supplied HPA-327 `crux.reviewed-reference-subset/v1` population. No second selector, seed, replacement, or score-aware filter.
- Full-mix OaF is never rerun by HPA-328.
- Full-mix reviewed reports are produced through `score_oaf_reviewed_subset()`, not by restitching its lower-level helpers.
- Required separators are exactly official Spleeter 4-stem drums and standard `htdemucs` drums. No third separator in v1.
- Spleeter and Demucs stay out of Crux's main Python dependency graph.
- Real separator package/repository/model bytes are frozen before scored pilot results are inspected, but live freezing is deferred to Task 11; unit tasks use fixture locks.
- Fixed QC constants: duration difference `<= max(0.5s, source_duration * 0.005)`, near-silent failure at RMS `<= -80 dBFS`, clipping evidence at absolute peak `>= 0.9999`. None are CLI flags.
- Retain exact separator-produced WAV bytes through HPA-396. Canonical OaF WAVs are temporary.
- Derived OaF view IDs are `crux.oaf-spleeter4-drums-mono44k1-pcm16/v1` and `crux.oaf-htdemucs-drums-mono44k1-pcm16/v1`.
- OaF descriptor/model lock/checkpoint/adapter/prediction map and HPA-325 scoring configuration remain unchanged across views.
- Derived prediction resume must never use the `OAF_FULL_MIX_INPUT_VIEW_ID` equality inside `_prediction_artifact_matches()`.
- Persisted prediction artifacts keep authoritative source-object identity; scorer-local artifacts may normalize only `source_audio_id` to `simfile_id` after persisted bytes are validated.
- Do not extend `CohortFailureReason`; use the frozen HPA-328 failure-code mapping in Task 7.
- Independent view populations remain visible. Headline paired event-micro metrics are recomputed from published per-song TP/FP/FN counts on the exact paired-success intersection.
- FP/FN-per-minute is derived only from persisted HPA-325 counts and authoritative source duration. It is not a new scorer.
- No full-corpus separator mode, generic experiment runner, plugin registry, RPC, queue/worker pool, database, retry framework, SDR scorer, bootstrap/significance layer, model tuning, or compatibility layer.

---

## File Structure

**Shared seams modified narrowly**

- `src/benchmark/cohort_scoring.py` — scorer-local prediction-source normalization and model-neutral non-success item construction.
- `src/benchmark/oaf_corpus_run.py` — retarget scorer bridge; parameterize OaF inference-config `input_view_id`; keep full-mix run ID and full-mix matcher unchanged.
- `src/benchmark/muscriptor_corpus_run.py` — retarget only the shared scorer bridge; no MuScriptor behavior change.
- `src/benchmark/input_view.py` — derived-stem canonicalization while preserving authoritative source identity.
- `src/benchmark/reports.py` — shared typed reader for published HPA-325 `summary.json` + CSV artifacts.
- `src/benchmark/muscriptor_comparison.py` — consume shared report reader; pairing behavior/output unchanged.
- `src/cli/benchmark.py` — two thin HPA-328 commands.

**New HPA-328 modules**

- `src/benchmark/separators.py` — separator lock validation, concrete Spleeter/HTDemucs execution, exact stem cache, and QC.
- `src/benchmark/separation_pilot.py` — fixed-subset preflight, run snapshot/resume, exact OaF derived inference, and derived HPA-325 cohort publication.
- `src/benchmark/separation_comparison.py` — fixed full-mix↔separator joins, paired-success event-micro aggregation, FP/FN-per-minute, runtime/storage/cost summary.
- `src/benchmark/separation_handoff.py` — canonical `crux.oaf-separation-pilot/v1` finalization/loader for HPA-396.
- `scripts/freeze_separator_runtime.py` — mechanical real-runtime lock generator.

**Fixture-only lock files created before Task 11**

- `tests/fixtures/separators/spleeter-model.json`
- `tests/fixtures/separators/htdemucs-model.json`

**Real lock files created and committed in Task 11 before scored execution**

- `runtime/separators/spleeter/model.json`
- `runtime/separators/htdemucs/model.json`

**New tests**

- `tests/benchmark/test_separators.py`
- `tests/benchmark/test_separation_pilot.py`
- `tests/benchmark/test_separation_pilot_acceptance.py`
- `tests/benchmark/test_separation_comparison.py`
- `tests/benchmark/test_separation_handoff.py`
- `tests/benchmark/schema_goldens/oaf-separation-pilot-v1.jsonl`

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

`cohort_item_from_validated_prediction_artifact()` assumes the caller has already validated raw persisted bytes against its run-row evidence. It changes only scorer-local `CanonicalAudio.source_audio_id` when required.

- [ ] **Step 1: Characterize current OaF and MuScriptor scorer adaptation**

```bash
uv run pytest -q \
  tests/benchmark/test_cohort_scoring.py \
  tests/benchmark/test_oaf_corpus_run.py \
  tests/benchmark/test_muscriptor_corpus_run.py
```

Expected: PASS before refactor.

- [ ] **Step 2: Add failing source-ID adaptation test**

Use a persisted artifact with `source_audio_id="songs/42/audio.ogg"` and `simfile_id="42"`:

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

- [ ] **Step 3: Add non-success constructor tests**

Cover only existing legal HPA-325 combinations:

```python
("failed", "inference_failed")
("failed", "prediction_artifact_invalid")
("failed", "prediction_missing")
("failed", "backend_unavailable")
("skipped", "explicitly_skipped")
("quarantined", "reference_quarantined")
```

- [ ] **Step 4: Implement the minimal scorer-local bridge**

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

- [ ] **Step 5: Retarget OaF and MuScriptor after their existing raw-artifact checks**

Replace only the duplicated scorer-local `replace(... source_audio_id=simfile_id)` block and duplicated non-success constructor. Keep each runner's persisted-artifact-vs-run-row validation unchanged and before the shared scorer helper.

- [ ] **Step 6: Verify no behavior change**

```bash
uv run pytest -q \
  tests/benchmark/test_cohort_scoring.py \
  tests/benchmark/test_oaf_corpus_run.py \
  tests/benchmark/test_oaf_corpus_run_acceptance.py \
  tests/benchmark/test_muscriptor_corpus_run.py \
  tests/benchmark/test_muscriptor_corpus_run_acceptance.py
```

- [ ] **Step 7: Commit**

```bash
git add src/benchmark/cohort_scoring.py src/benchmark/oaf_corpus_run.py \
  src/benchmark/muscriptor_corpus_run.py tests/benchmark
git commit -m "refactor: share persisted cohort scoring bridge"
```

---

### Task 2: Support derived OaF inputs without changing full-mix identity

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

- [ ] **Step 1: Characterize existing full-mix bytes and config**

```python
full = build_inference_config(config, descriptor, "a" * 64)
assert full["input_view_id"] == OAF_FULL_MIX_INPUT_VIEW_ID
assert full["canonicalization_revision"] == OAF_CANONICALIZATION_REVISION
```

```bash
uv run pytest -q tests/benchmark/test_input_view.py tests/benchmark/test_oaf_corpus_run.py
```

- [ ] **Step 2: Add derived-view config test**

```python
stem = build_inference_config(
    config,
    descriptor,
    "a" * 64,
    input_view_id="crux.oaf-htdemucs-drums-mono44k1-pcm16/v1",
)
assert stem["input_view_id"] != full["input_view_id"]
assert {k: v for k, v in stem.items() if k != "input_view_id"} == {
    k: v for k, v in full.items() if k != "input_view_id"
}
```

- [ ] **Step 3: Add derived materialization test**

Use a stereo/non-44.1-kHz stem. Assert 44.1-kHz mono PCM16 output, authoritative source ID/SHA preserved, derived view ID preserved, and canonical stem bytes hashed into `input_audio_sha256`.

- [ ] **Step 4: Factor only the shared canonicalization body**

Both public materializers may call a private helper implementing the existing:

```python
samples, _ = librosa.load(source, sr=44100, mono=True, res_type="soxr_hq")
soundfile.write(output_path, samples, 44100, format="WAV", subtype="PCM_16")
```

Keep full-mix bytes unchanged. `materialize_derived_audio()` reads `derived_audio_path`; `source_audio` supplies authoritative identity.

- [ ] **Step 5: Parameterize only `input_view_id` in `build_inference_config()`**

Reject empty/non-string IDs. Do not change `build_run_id()`; it remains full-mix-specific.

- [ ] **Step 6: Verify HPA-326 regressions**

```bash
uv run pytest -q \
  tests/benchmark/test_input_view.py \
  tests/benchmark/test_oaf_corpus_run.py \
  tests/benchmark/test_oaf_corpus_run_acceptance.py \
  tests/benchmark/test_oaf_corpus_run_branches.py
```

- [ ] **Step 7: Commit**

```bash
git add src/benchmark/input_view.py src/benchmark/oaf_corpus_run.py tests/benchmark
git commit -m "refactor: support derived OaF input views"
```

---

### Task 3: Define separator lock schema, fixture locks, and freeze script

**Files:**
- Create: `src/benchmark/separators.py`
- Create: `scripts/freeze_separator_runtime.py`
- Create: `tests/fixtures/separators/spleeter-model.json`
- Create: `tests/fixtures/separators/htdemucs-model.json`
- Create: `tests/benchmark/test_separators.py`

**Do not create real `runtime/separators/*/model.json` yet.** Those require reproduced real environments and are Task 11 evidence.

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

Reject unknown/missing keys, noncanonical JSON, duplicate/absolute model filenames, malformed hashes, unsupported separator IDs, and command/model mismatch. Require `spleeter:4stems` for Spleeter and `-n htdemucs` for Demucs.

- [ ] **Step 2: Implement closed canonical lock parsing**

Reuse `strict_json_loads()`, `canonical_json_bytes()`, and `require_sha256()`. `SeparatorLock.sha256` is the SHA-256 of exact canonical lock bytes.

- [ ] **Step 3: Add canonical fixture locks with synthetic model hashes**

Fixtures must pass the same loader as future production locks but are explicitly test data. Example model hashes may use repeated hex tokens; they must never be copied into `runtime/separators/`.

- [ ] **Step 4: Implement mechanical freeze script**

The script accepts an isolated interpreter, repository URL/revision, model ID, explicit downloaded model files, and licenses. It queries package version through that interpreter, hashes supplied model files, strips directories to basenames, renders canonical JSON, and round-trips through `load_separator_lock()`.

```python
subprocess.run(
    [python, "-c", f"import importlib.metadata as m; print(m.version({package!r}))"],
    check=True,
    capture_output=True,
    text=True,
)
```

No separator inference or benchmark scoring occurs in this task.

- [ ] **Step 5: Verify**

```bash
uv run pytest -q tests/benchmark/test_separators.py
uv run ruff check src/benchmark/separators.py scripts/freeze_separator_runtime.py \
  tests/benchmark/test_separators.py
```

- [ ] **Step 6: Commit**

```bash
git add src/benchmark/separators.py scripts/freeze_separator_runtime.py \
  tests/fixtures/separators tests/benchmark/test_separators.py
git commit -m "feat: define frozen separator identities"
```

---

### Task 4: Cache exact stems and apply fixed practical QC

**Files:**
- Modify: `src/benchmark/separators.py`
- Modify: `tests/benchmark/test_separators.py`

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

- [ ] **Step 1: Add cache/QC/process tests**

Cover cache identity, matching reuse, conflicting bytes, process nonzero exit, missing output, decode failure, near-silent stem, duration mismatch, >2 channels, nonfinite samples, sample-rate/channel warnings, and clipping evidence.

- [ ] **Step 2: Implement deterministic cache path**

```python
output_root / "derived" / "stems" / lock.separator_id / source_sha / lock.sha256 / "drums.wav"
```

Use existing `publish_immutable_file()` for exact stem bytes.

- [ ] **Step 3: Stage deterministic separator input**

Each process receives a private temporary directory with the authoritative source staged as `input.wav`. Only source/output paths vary; model/config arguments come from the validated lock.

- [ ] **Step 4: Implement fixed QC**

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

Fail for channels outside `{1, 2}`, nonfinite samples, `rms_dbfs <= -80`, or duration delta above `max(0.5, source_duration * 0.005)`. Warn for non-44.1-kHz, mono, or peak `>= 0.9999`.

- [ ] **Step 5: Prove cache hit bypasses subprocess**

Seed matching immutable bytes, inject a runner that raises if invoked, and assert QC/reuse succeeds without execution.

- [ ] **Step 6: Verify and commit**

```bash
uv run pytest -q tests/benchmark/test_separators.py
git add src/benchmark/separators.py tests/benchmark/test_separators.py
git commit -m "feat: cache and validate drum stems"
```

---

### Task 5: Define fixed-subset preflight, run identity, snapshot, and full-mix control

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
```

- [ ] **Step 1: Add fatal-preflight tests with process/backend sentinels**

Neither separator nor OaF may be touched for subset/reference mismatch, subset/timing mismatch, parent HPA-326 mismatch, subset member absent from parent population, invalid fixture lock, mixed OaF identity, or protected output alias.

- [ ] **Step 2: Add exact-membership test**

A canonical 20-row reviewed subset produces exactly those 20 sorted run rows. No sampler field exists on the request.

- [ ] **Step 3: Bind deterministic HPA-328 run identity**

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

Statuses:

```text
pending
separation_failed
stem_invalid
inference_failed
prediction_invalid
success
resumed
```

Detailed failure codes may additionally record `canonical_input_failed`, `prediction_output_conflict`, and `prediction_publish_failed`.

Each row stores `source_row_sha256`, authoritative source ID/SHA/duration, full-mix parent status/input/prediction identity, and nested Spleeter/HTDemucs lock/stem/QC/runtime/input/prediction evidence.

- [ ] **Step 5: Implement strict canonical snapshot parse/render/write**

Follow HPA-326 canonical JSON, six-place normalization, `atomic_replace_bytes()`, sorted items, exact resume identity, and status/nullability checks. Do not extract a run framework.

- [ ] **Step 6: Produce full-mix reports through the existing public wrapper**

Construct:

```python
score_request = ScoreReviewedSubsetRequest(
    reference_manifest_path=request.reference_manifest_path,
    timing_manifest_path=request.timing_manifest_path,
    subset_manifest_path=request.subset_manifest_path,
    run_path=request.oaf_run_path,
    output_dir=run_dir / "views" / "full_mix" / "reports",
)
outcome = score_oaf_reviewed_subset(score_request)
```

Require `outcome.exit_code != 2`. A fake backend that raises must never be called.

Still parse the HPA-326 parent snapshot separately for HPA-328 run/header/item identities; do not use `score_reviewed_subset_cohort()` directly for full mix.

- [ ] **Step 7: Verify and commit**

```bash
uv run pytest -q tests/benchmark/test_separation_pilot.py tests/benchmark/test_reviewed_subset.py
git add src/benchmark/separation_pilot.py tests/benchmark/test_separation_pilot.py
git commit -m "feat: define OaF separation pilot run"
```

---

### Task 6: Execute and resume derived views through the exact OaF write path

**Files:**
- Modify: `src/benchmark/separation_pilot.py`
- Modify: `tests/benchmark/test_separation_pilot.py`
- Create: `tests/benchmark/test_separation_pilot_acceptance.py`

**Interface:**

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

- [ ] **Step 2: Add the derived resume matrix**

Cover:

```text
valid stem + valid derived prediction -> resumed, no separator/backend
valid stem + missing prediction       -> OaF only
wrong stem hash                       -> no reuse
run-row prediction evidence mismatch  -> prediction_invalid
materialized input SHA mismatch       -> prediction_invalid
wrong descriptor/model                -> prediction_invalid
wrong derived input_view_id           -> prediction_invalid
conflicting immutable prediction      -> prediction_invalid, never overwrite
```

Add a regression whose artifact uses `HTDEMUCS_INPUT_VIEW_ID`; it must resume successfully even though `_prediction_artifact_matches()` would reject it.

- [ ] **Step 3: Resolve all authoritative sources before expensive work**

Use `resolve_source_audio(..., load_body=False)` and persist source ID/SHA/duration. Failure for a fixed member is fatal; never shrink membership.

- [ ] **Step 4: Build two OaF configs differing only by view**

```python
spleeter_config = build_inference_config(
    oaf_config, descriptor, model_lock_sha, input_view_id=SPLEETER_INPUT_VIEW_ID
)
htdemucs_config = build_inference_config(
    oaf_config, descriptor, model_lock_sha, input_view_id=HTDEMUCS_INPUT_VIEW_ID
)
```

- [ ] **Step 5: Implement view-aware prediction reuse**

Read candidate with `read_prediction_artifact()`.

When prior persisted prediction evidence exists, call the existing view-agnostic:

```python
if not _prediction_artifact_matches_run_row(artifact, prior_row):
    fail("prediction_invalid")
```

Then compare against current materialized audio explicitly:

```python
prediction = artifact.prediction
matches_current_audio = (
    prediction.descriptor.sha256 == descriptor.sha256
    and dict(prediction.descriptor.payload) == dict(descriptor.payload)
    and prediction.audio.source_audio_id == source.source_audio_id
    and prediction.audio.source_audio_sha256 == source.source_audio_sha256
    and prediction.audio.input_view_id == audio.input_view_id
    and prediction.audio.input_audio_sha256 == audio.input_audio_sha256
    and all(
        event.prediction_map_version == OAF_PREDICTION_MAP_ID
        for event in prediction.events
    )
)
```

Do **not** call or copy `_prediction_artifact_matches()`'s full-mix constant check.

- [ ] **Step 6: Use the existing OaF inference/write functions exactly**

On a prediction miss:

```python
started = perf_counter()
try:
    native = backend.transcribe(audio)
finally:
    elapsed = max(0.0, perf_counter() - started)

if not isinstance(native, NativePrediction):
    raise OafBackendError("native prediction is invalid", code="native_event_invalid")

mapped, _ = map_oaf_prediction(native)
published = publish_prediction_artifact(prediction_target, mapped)
```

Catch `OafBackendError` and call `classify_oaf_backend_error(error.code)`. Catch `ArtifactPublicationError` / `PredictionArtifactError` as `prediction_publish_failed`. Do not add another prediction publisher or mapper.

- [ ] **Step 7: Checkpoint every durable boundary**

Per view:

1. run/validate cached stem;
2. checkpoint stem/QC/separation runtime;
3. materialize temporary canonical OaF WAV;
4. reuse or create prediction through Steps 5–6;
5. checkpoint input hash, prediction path/hash, OaF runtime/status;
6. delete only temporary canonical WAV in `finally`.

- [ ] **Step 8: Keep separator failures independent and OaF poison behavior unchanged**

Spleeter failure must not prevent HTDemucs for the same song. One persistent OaF backend handles misses; poison stops later inference in the invocation and leaves resume state. No automatic restart.

- [ ] **Step 9: Verify and commit**

```bash
uv run pytest -q tests/benchmark/test_separation_pilot.py \
  tests/benchmark/test_separation_pilot_acceptance.py \
  tests/benchmark/test_oaf_corpus_run.py

git add src/benchmark/separation_pilot.py tests/benchmark/test_separation_pilot.py \
  tests/benchmark/test_separation_pilot_acceptance.py
git commit -m "feat: run OaF on separated drum views"
```

---

### Task 7: Score derived cohorts with the frozen HPA-325 reason map

**Files:**
- Modify: `src/benchmark/separation_pilot.py`
- Modify: `tests/benchmark/test_separation_pilot_acceptance.py`
- Test: `tests/benchmark/test_cohort_scoring.py`

**Interfaces:**

```python
SEPARATION_FAILURE_TO_COHORT_REASON: dict[str, CohortFailureReason] = {
    "separation_failed": "inference_failed",
    "stem_invalid": "inference_failed",
    "canonical_input_failed": "inference_failed",
    "inference_failed": "inference_failed",
    "prediction_invalid": "prediction_artifact_invalid",
    "prediction_output_conflict": "prediction_artifact_invalid",
    "prediction_publish_failed": "prediction_artifact_invalid",
}
```

Produces:

```text
<run>/views/spleeter4_drums/reports/
<run>/views/htdemucs_drums/reports/
```

- [ ] **Step 1: Pin the exact mapping table in a test**

```python
assert SEPARATION_FAILURE_TO_COHORT_REASON == {
    "separation_failed": "inference_failed",
    "stem_invalid": "inference_failed",
    "canonical_input_failed": "inference_failed",
    "inference_failed": "inference_failed",
    "prediction_invalid": "prediction_artifact_invalid",
    "prediction_output_conflict": "prediction_artifact_invalid",
    "prediction_publish_failed": "prediction_artifact_invalid",
}
```

Assert the table values are a subset of `COHORT_FAILURE_REASONS`. Do not modify `CohortFailureReason`.

- [ ] **Step 2: Add mixed-success population test**

Use three songs: both-success, Spleeter-fail/HTDemucs-success, Spleeter-success/HTDemucs-fail. Each view `items.csv` contains all three.

- [ ] **Step 3: Build view `CohortIdentity` from frozen OaF identity**

Only `cohort_id` and `input_view_id` differ. Taxonomy/lane/prediction-map/model/model-lock/descriptor/scoring version stay fixed.

- [ ] **Step 4: Build success/resume rows through the shared scorer bridge**

After Task 6 raw persisted validation:

```python
item = cohort_item_from_validated_prediction_artifact(
    identity,
    str(simfile_id),
    mapping,
    persisted_prediction,
    warnings=warnings,
)
```

- [ ] **Step 5: Build failure rows through `cohort_item_without_prediction()`**

Translate only through `SEPARATION_FAILURE_TO_COHORT_REASON`. Detailed separator codes/messages remain in HPA-328 `run.json`.

- [ ] **Step 6: Score/write with no new knobs**

```python
result = score_cohort(identity, cohort_items, diagnostics_for=successful_subset_ids)
write_cohort_reports(result, reports_path)
```

- [ ] **Step 7: Verify and commit**

```bash
uv run pytest -q tests/benchmark/test_separation_pilot_acceptance.py \
  tests/benchmark/test_cohort_scoring.py

git add src/benchmark/separation_pilot.py tests/benchmark/test_separation_pilot_acceptance.py
git commit -m "feat: score separated OaF cohorts"
```

---

### Task 8A: Promote model-neutral HPA-325 report reading

**Files:**
- Modify: `src/benchmark/reports.py`
- Modify: `src/benchmark/muscriptor_comparison.py`
- Modify/Create reader-focused tests in `tests/benchmark/test_reports.py`
- Test: `tests/benchmark/test_muscriptor_comparison.py`
- Test: `tests/benchmark/test_muscriptor_comparison_coverage.py`

**Interfaces:**

```python
@dataclass(frozen=True)
class PublishedEventMicro:
    tp: int
    fp: int
    fn: int
    precision: Decimal | None
    recall: Decimal | None
    f1: Decimal | None

@dataclass(frozen=True)
class PublishedAggregateRow:
    tolerance_ms: int
    mode: ScoreMode
    event_micro: PublishedEventMicro
    song_macro_f1: Decimal | None
    class_macro_f1: Decimal | None
    successful_song_count: int

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
    prediction_to_reference_ratio: Decimal | None
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
    population: PopulationSummary
    aggregates: tuple[PublishedAggregateRow, ...]
    items: tuple[PublishedItemRow, ...]
    songs: tuple[PublishedSongRow, ...]
    classes: tuple[PublishedClassRow, ...]


def read_cohort_reports(
    report_dir: Path,
    *,
    expected_identity: CohortIdentity,
) -> PublishedCohortReports: ...
```

- [ ] **Step 1: Characterize MuScriptor comparison before any refactor**

```bash
uv run pytest -q tests/benchmark/test_muscriptor_comparison.py \
  tests/benchmark/test_muscriptor_comparison_coverage.py
```

Capture any golden/fixture output bytes already pinned by those tests.

- [ ] **Step 2: Move field-name ownership to `reports.py` only**

Keep `_ITEM_FIELDNAMES`, `_PER_SONG_FIELDNAMES`, and `_PER_CLASS_FIELDNAMES` as the writer/reader single source. Remove the copies from `muscriptor_comparison.py` after the reader lands.

- [ ] **Step 3: Add strict summary reader tests**

Read canonical `summary.json`; require `REPORT_SCHEMA`, exact expected identity, tolerance set, population fields, and aggregate rows. For every aggregate require event-micro `tp/fp/fn/precision/recall/f1`, macro values, and successful-song count.

- [ ] **Step 4: Add strict CSV reader tests**

Reject wrong/missing columns, malformed numeric tokens, duplicate item/song/class keys, score rows for non-success items, and row identity mismatches.

- [ ] **Step 5: Implement `read_cohort_reports()`**

It reads only HPA-325 published report files. It does not parse run snapshots, choose pairable IDs, know backend families, or write comparison output.

- [ ] **Step 6: Retarget MuScriptor comparison**

Keep its existing run parsing, backend-family validation, source/input pairability, subset handling, output names, labels, and aggregate policy. Replace only private CSV/report parsers with `read_cohort_reports()`.

- [ ] **Step 7: Require unchanged MuScriptor outputs**

```bash
uv run pytest -q tests/benchmark/test_reports.py \
  tests/benchmark/test_muscriptor_comparison.py \
  tests/benchmark/test_muscriptor_comparison_coverage.py
```

- [ ] **Step 8: Commit**

```bash
git add src/benchmark/reports.py src/benchmark/muscriptor_comparison.py \
  tests/benchmark/test_reports.py tests/benchmark/test_muscriptor_comparison.py \
  tests/benchmark/test_muscriptor_comparison_coverage.py
git commit -m "refactor: share published cohort report reading"
```

---

### Task 8B: Write the fixed HPA-328 paired comparison

**Files:**
- Create: `src/benchmark/separation_comparison.py`
- Create: `tests/benchmark/test_separation_comparison.py`
- Modify: `src/benchmark/separation_pilot.py`

**Interfaces:**

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
    full_mix_reports: PublishedCohortReports,
    spleeter_reports: PublishedCohortReports,
    htdemucs_reports: PublishedCohortReports,
    run_snapshot: Mapping[str, object],
    output_dir: Path,
    cost_rates: CostRates = CostRates(),
) -> SeparationComparisonArtifacts: ...
```

- [ ] **Step 1: Add identity tests**

Require reference/timing/taxonomy/lane/prediction-map/OaF/scoring identity equality. The three `input_view_id` values intentionally differ.

- [ ] **Step 2: Add independent-population test**

`summary.json` must retain each view's independent HPA-325 population and independent summary aggregates, even when success memberships differ.

- [ ] **Step 3: Add pairable-success intersection test**

For each pair, pairable IDs are exactly:

```python
full_success_ids & derived_success_ids
```

Delta CSV rows contain only those IDs and require identical score key grids on the paired set.

- [ ] **Step 4: Recompute paired event-micro from published per-song counts**

For every tolerance/mode and pairable set:

```python
def paired_event_micro(rows: Iterable[PublishedSongRow]) -> ScoreSummary:
    tp = sum(row.tp for row in rows)
    fp = sum(row.fp for row in rows)
    fn = sum(row.fn for row in rows)
    return ScoreSummary(tp, fp, fn)
```

Build separate `ScoreSummary` values for full mix and derived view using the same paired IDs, then persist TP/FP/FN/precision/recall/F1 and `derived - full_mix` F1.

Do not subtract the independent `PublishedAggregateRow.event_micro.f1` values as the headline when memberships differ.

- [ ] **Step 5: Join per-song and per-class published rows**

Emit source values and `derived - full_mix` precision/recall/F1 deltas for matching tolerance/mode/song/class keys.

- [ ] **Step 6: Derive FP/FN per minute**

For each paired per-song row require positive finite `source_duration_sec` from HPA-328 run evidence:

```python
minutes = Decimal(str(source_duration_sec)) / Decimal("60")
fp_per_min = Decimal(row.fp) / minutes
fn_per_min = Decimal(row.fn) / minutes
```

Persist absolute rates plus derived-minus-full-mix deltas using six-place canonical tokens.

- [ ] **Step 7: Carry timing diagnostics and bounded helped/harmed cases**

Carry `median_abs_error_ms`, `p95_abs_error_ms`, `offset_ms`. Per view/tolerance/mode list at most five most helped and five most harmed songs sorted by F1 delta then `simfile_id`; include per-class direction.

- [ ] **Step 8: Add runtime/storage/cost evidence**

```python
compute_cost = Decimal(str(total_wall_time_sec)) / Decimal("3600") * compute_rate
storage_gb = Decimal(total_bytes) / Decimal(1024 ** 3)
storage_monthly_cost = storage_gb * storage_rate
```

Rates default to zero and affect display only.

- [ ] **Step 9: Render deterministic comparison artifacts**

```text
comparison/summary.json
comparison/per_song_delta.csv
comparison/per_class_delta.csv
comparison/summary.md
```

- [ ] **Step 10: Verify and commit**

```bash
uv run pytest -q tests/benchmark/test_separation_comparison.py \
  tests/benchmark/test_separation_pilot_acceptance.py

git add src/benchmark/separation_comparison.py src/benchmark/separation_pilot.py \
  tests/benchmark/test_separation_comparison.py
git commit -m "feat: compare OaF separation views"
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
```

- [ ] **Step 1: Add one-row schema golden and production population tests**

Golden validates exact row shape/canonical bytes. Production finalization requires exact HPA-327 20–30 membership.

- [ ] **Step 2: Freeze row identity/nullability**

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

- [ ] **Step 3: Re-read/hash durable artifacts before publication**

Verify stems, predictions, and comparison files against run evidence. Missing/edited successful HTDemucs evidence prevents finalization.

- [ ] **Step 4: Publish through existing manifest rails**

Reuse `render_manifest()` / `publish_manifest()`. Do not create a header/database protocol.

- [ ] **Step 5: Add HPA-396 consumer test**

For every successful HTDemucs row downstream obtains without mutable `run.json`:

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

No input-view manifest is required; HPA-396 can rematerialize from retained native stems.

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

- [ ] **Step 2: Keep imports lazy**

Separator/pilot/finalization modules import behind the Click command boundary.

- [ ] **Step 3: Emit canonical summaries and stable exits**

Run uses 0 complete / 1 partial / 2 fatal. Finalize exits 0 only after immutable publication. Decision is one of five frozen values; rationale nonempty; rates nonnegative.

- [ ] **Step 4: Verify and commit**

```bash
uv run pytest -q tests/test_cli_benchmark.py tests/test_cli_benchmark_coverage.py
git add src/cli/benchmark.py tests/test_cli_benchmark.py tests/test_cli_benchmark_coverage.py
git commit -m "feat: add separation pilot commands"
```

---

### Task 11: Freeze real separator runtimes, run the pilot, and close verification

**Files:**
- Create and commit before scoring: `runtime/separators/spleeter/model.json`
- Create and commit before scoring: `runtime/separators/htdemucs/model.json`
- Generated stems, predictions, reports, and handoff outputs stay outside git.

- [ ] **Step 1: Reproduce real Spleeter known-audio smoke**

Using the isolated intended interpreter, stage a known non-benchmark audio file as `input.wav` and run:

```bash
"$SPLEETER_PYTHON" -m spleeter separate -p spleeter:4stems \
  -o "$SPLEETER_SMOKE_OUT" "$SMOKE_DIR/input.wav"
```

Require a decodable drum stem. Record exact package version, repository revision, model files, licenses, and command semantics. Do not inspect HPA-328 scores.

- [ ] **Step 2: Reproduce real HTDemucs known-audio smoke**

```bash
"$DEMUCS_PYTHON" -m demucs -n htdemucs \
  -o "$HTDEMUCS_SMOKE_OUT" "$SMOKE_DIR/input.wav"
```

Require decodable `drums.wav` and record exact package/repository/model files and licenses.

- [ ] **Step 3: Generate canonical production locks**

Run `scripts/freeze_separator_runtime.py` with the exact reproduced interpreters/revisions/model files. Round-trip both results through `load_separator_lock()`.

```bash
uv run python scripts/freeze_separator_runtime.py ... \
  --output runtime/separators/spleeter/model.json
uv run python scripts/freeze_separator_runtime.py ... \
  --output runtime/separators/htdemucs/model.json
```

- [ ] **Step 4: Commit real locks before any reference-score execution**

```bash
uv run pytest -q tests/benchmark/test_separators.py
git add runtime/separators/spleeter/model.json runtime/separators/htdemucs/model.json
git commit -m "chore: freeze separator benchmark runtimes"
```

Record:

```bash
git rev-parse HEAD
sha256sum runtime/separators/spleeter/model.json runtime/separators/htdemucs/model.json
```

After this point do not alter separator model/config/QC/scoring semantics based on pilot scores.

- [ ] **Step 5: Run exact HPA-327 pilot**

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

If interrupted, repeat exactly with `--resume`.

- [ ] **Step 6: Review required evidence only**

Require:

- all HPA-327 members in all three population ledgers;
- full-mix/Spleeter/HTDemucs HPA-325 reports;
- exact stem/input/prediction identities;
- independent population and summary aggregates;
- paired-success per-song/per-class deltas;
- paired-success event-micro TP/FP/FN/precision/recall/F1;
- FP/FN per minute;
- timing/offset diagnostics;
- separator + derived OaF runtime/RTF;
- retained bytes and cost display;
- bounded helped/harmed examples.

- [ ] **Step 7: Finalize one closed decision**

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

Use zero rates for local sunk hardware/storage or actual marginal rates otherwise.

- [ ] **Step 8: Run repository-wide verification**

```bash
uv run pytest -q
uv run ruff check .
uv run ruff format --check src tests scripts
uv run pylint --errors-only --disable=E1120,E0401 --jobs=1 src
git diff --check
git status --short
```

- [ ] **Step 9: Scope audit**

```bash
git diff --stat origin/main...HEAD
```

Confirm no full-corpus separator path, generic experiment/plugin framework, bundled model weights, committed audio/stems/results, new scorer, or new HPA-325 failure enum.

---

## Plan Self-Review

**Review finding 1 — derived resume:** Task 6 explicitly rejects `_prediction_artifact_matches()` for derived views, reuses `_prediction_artifact_matches_run_row()`, and binds the view to the current materialized `CanonicalAudio.input_view_id`.

**Review finding 2 — OaF write path:** Task 6 explicitly calls `backend.transcribe()` → `map_oaf_prediction()` → `publish_prediction_artifact()` and routes backend failures through `classify_oaf_backend_error()`.

**Review finding 3 — full-mix control:** Task 5 calls `score_oaf_reviewed_subset()` with `ScoreReviewedSubsetRequest.output_dir=<run>/views/full_mix/reports`; parent snapshot parsing remains identity evidence only.

**Review finding 4 — failure vocabulary:** Task 7 pins the exact HPA-328 failure-code → existing `CohortFailureReason` table and prohibits enum growth.

**Review finding 5 — paired F1:** Task 8A reads independent HPA-325 summary aggregates; Task 8B recomputes headline paired event-micro metrics by summing published per-song counts on the successful-song intersection.

**Review finding 6 — sequencing:** Task 3 is schema/freeze-script/fixture-only. Task 8 is split into independent 8A reader extraction and 8B comparison implementation. Task 11 reproduces/freeze real separator runtimes before scored pilot execution.

**Scope:** Same experiment and same non-goals. Shared extractions stop at real third-caller duplication; runners, resume policy, and pair-specific comparison logic remain concrete.
