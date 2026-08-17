# HPA-328 OaF Drum-Stem Separation Ablation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run one fixed 20–30-song OaF input ablation comparing the persisted full-mix baseline with Spleeter 4-stem drums and standard HTDemucs drums, then publish paired HPA-325 reports and an immutable HTDemucs handoff for HPA-396.

**Architecture:** Reuse the HPA-327 reviewed subset as the entire pilot and the persisted HPA-326 full-mix run as the control. Execute Spleeter and Demucs as two pinned external commands, cache their exact drum WAVs by source/config identity, canonicalize those stems through the existing OaF audio boundary, persist normal prediction-v2 artifacts, and score through HPA-325. Keep experiment orchestration, comparison reporting, and the final handoff as small HPA-328-specific modules rather than introducing a generic experiment/separator framework.

**Tech Stack:** Python 3.12 Crux host, Click, existing benchmark canonical JSON/JSONL/CSV helpers, librosa/soundfile/NumPy already present in Crux, external Spleeter isolated Python environment, external Demucs isolated Python environment, existing OaF TF1 worker, pytest/Ruff/Pylint.

## Global Constraints

- Implementation starts from refreshed `main` after HPA-395 PR #24 lands so HPA-328 reuses the promoted neutral source/materialization/prediction/subset seams instead of duplicating them.
- Pilot membership is exactly the supplied canonical HPA-327 `crux.reviewed-reference-subset/v1` population; no second selector or score-aware replacement.
- Full-mix OaF is never rerun by HPA-328; reconstruct it from the supplied HPA-326 persisted run and immutable predictions.
- Required separators are exactly official Spleeter 4-stem drums and standard `htdemucs` drums; no third separator in v1.
- Spleeter and Demucs stay outside Crux's main Python dependency graph.
- Freeze exact separator package/repository/model bytes and command arguments before scored pilot results are inspected.
- Separator QC constants are fixed in code: duration difference `<= max(0.5s, 0.5%)`, near-silent failure at RMS `<= -80 dBFS`, clipping evidence at absolute peak `>= 0.9999`; none are CLI flags.
- Retain exact separator-produced WAV bytes through HPA-396; do not add a pilot-time archive transcode.
- Derived OaF views are `crux.oaf-spleeter4-drums-mono44k1-pcm16/v1` and `crux.oaf-htdemucs-drums-mono44k1-pcm16/v1`.
- OaF descriptor/model lock/checkpoint/adapter/prediction map and HPA-325 scoring semantics remain unchanged across views.
- A per-view failure remains in population accounting; paired metric deltas use only matching successful songs and report the independent failure counts separately.
- No full-corpus separator mode, generic experiment runner, plugin registry, queue/worker pool, database, retry framework, SDR scorer, training, tuning, or backward-compatibility layer.

---

## File map

### Shared seams modified narrowly

- `src/benchmark/input_view.py` — add derived-stem materialization while preserving the authoritative source identity.
- `src/benchmark/oaf_corpus_run.py` — allow the existing OaF inference-config builder to receive an explicit input-view ID with a full-mix default.
- `src/cli/benchmark.py` — add the two HPA-328 commands only after the model layer is complete.

### New HPA-328 modules

- `src/benchmark/separators.py` — separator lock parsing/hashing, concrete Spleeter/HTDemucs command construction, process result, exact stem cache publication, and stem QC.
- `src/benchmark/separation_pilot.py` — fixed-subset preflight, run identity/snapshot, derived-view separation/OaF execution/resume, and HPA-325 cohort publication.
- `src/benchmark/separation_comparison.py` — narrow full-mix↔separator report joins and runtime/storage/cost summary.
- `src/benchmark/separation_handoff.py` — canonical `crux.oaf-separation-pilot/v1` JSONL loader/golden/finalizer for HPA-396.
- `scripts/freeze_separator_runtime.py` — mechanically build the two checked-in separator locks from an isolated interpreter plus explicit downloaded model files.
- `runtime/separators/spleeter/model.json` — frozen official Spleeter 4-stem runtime/model evidence.
- `runtime/separators/htdemucs/model.json` — frozen standard HTDemucs runtime/model evidence.

### Tests

- `tests/benchmark/test_separators.py`
- `tests/benchmark/test_separation_pilot.py`
- `tests/benchmark/test_separation_pilot_acceptance.py`
- `tests/benchmark/test_separation_comparison.py`
- `tests/benchmark/test_separation_handoff.py`
- `tests/benchmark/schema_goldens/oaf-separation-pilot-v1.jsonl`
- existing `tests/benchmark/test_input_view.py`
- existing `tests/benchmark/test_oaf_corpus_run.py`
- existing `tests/test_cli_benchmark.py`

---

### Task 1: Make the existing OaF audio/config seam support derived input views

**Files:**
- Modify: `src/benchmark/input_view.py`
- Modify: `src/benchmark/oaf_corpus_run.py`
- Test: `tests/benchmark/test_input_view.py`
- Test: `tests/benchmark/test_oaf_corpus_run.py`
- Test: `tests/benchmark/test_oaf_corpus_run_acceptance.py`

**Interfaces:**
- Consumes after #24: `ResolvedSourceAudio`, `materialize_full_mix_audio()`, `load_materialized_audio()`, `build_inference_config()`.
- Produces:

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

- [ ] **Step 1: Rebase the implementation branch onto the mainline that contains HPA-395's neutral seam extractions**

Run:

```bash
git fetch origin
git rebase origin/main
```

Verify the mainline definitions exist before editing:

```bash
rg 'def resolve_source_audio|def materialize_full_mix_audio|def prediction_path|def score_reviewed_subset_cohort' src/benchmark
```

Expected: all four promoted seams are present exactly once. If #24 renamed one during review, update this plan/spec reference during the rebase rather than adding an alias.

- [ ] **Step 2: Add characterization tests proving the full-mix config and materialization remain unchanged**

Add a test around the current default inference payload:

```python
def test_build_inference_config_default_keeps_full_mix_identity(oaf_config, oaf_descriptor):
    payload = build_inference_config(
        oaf_config,
        oaf_descriptor,
        "a" * 64,
    )
    assert payload["input_view_id"] == OAF_FULL_MIX_INPUT_VIEW_ID
    assert payload["canonicalization_revision"] == OAF_CANONICALIZATION_REVISION
```

Add/retain the existing full-mix materialization byte/hash assertion so extracting a private audio-writing helper cannot alter HPA-326 output.

Run:

```bash
uv run pytest -q tests/benchmark/test_input_view.py tests/benchmark/test_oaf_corpus_run.py
```

Expected: PASS before semantic change.

- [ ] **Step 3: Write the failing derived-view tests**

Add:

```python
def test_build_inference_config_accepts_explicit_input_view(oaf_config, oaf_descriptor):
    payload = build_inference_config(
        oaf_config,
        oaf_descriptor,
        "a" * 64,
        input_view_id="crux.oaf-htdemucs-drums-mono44k1-pcm16/v1",
    )
    assert payload["input_view_id"] == "crux.oaf-htdemucs-drums-mono44k1-pcm16/v1"


def test_materialize_derived_audio_preserves_source_and_hashes_input(tmp_path, source_audio):
    stem = tmp_path / "stems" / "drums.wav"
    write_test_wav(stem, sample_rate=48000, channels=2)
    audio = materialize_derived_audio(
        source_audio,
        stem,
        tmp_path / "inputs" / "stem.wav",
        input_root=tmp_path / "inputs",
        input_view_id="crux.oaf-htdemucs-drums-mono44k1-pcm16/v1",
        max_input_audio_frames=None,
    )
    assert audio.source_audio_id == source_audio.source_audio_id
    assert audio.source_audio_sha256 == source_audio.source_audio_sha256
    assert audio.input_view_id == "crux.oaf-htdemucs-drums-mono44k1-pcm16/v1"
    assert audio.input_audio_sha256 != source_audio.source_audio_sha256
    assert audio.sample_rate == 44100
    assert audio.channel_count == 1
    assert audio.sample_width_bytes == 2
```

Also assert the derived destination cannot escape `input_root`.

Run the two new tests and confirm they fail because the keyword argument/function does not exist yet.

- [ ] **Step 4: Implement the smallest shared canonicalization core**

In `input_view.py`, keep `materialize_full_mix_audio()` public and extract only the duplicated decode/write body into a private helper. The derived public function must read the retained stem but pass the authoritative source identity to `load_materialized_audio()`:

```python
def materialize_derived_audio(
    source_audio: ResolvedSourceAudio,
    derived_audio_path: Path,
    output_path: Path,
    *,
    input_root: Path,
    input_view_id: str,
    max_input_audio_frames: int | None,
) -> CanonicalAudio:
    _require_output_beneath_root(output_path, input_root)
    _materialize_pcm16_mono44k1(derived_audio_path, output_path)
    return load_materialized_audio(
        path=output_path,
        source_audio_id=source_audio.source_audio_id,
        source_audio_sha256=source_audio.source_audio_sha256,
        input_view_id=input_view_id,
        max_input_audio_frames=max_input_audio_frames,
    )
```

Do not add an `InputView` class hierarchy.

- [ ] **Step 5: Parameterize only the existing OaF inference config's input-view field**

Change the builder to:

```python
def build_inference_config(
    config: OafModelConfig,
    descriptor: BackendDescriptor,
    model_lock_sha256: str,
    *,
    input_view_id: str = OAF_FULL_MIX_INPUT_VIEW_ID,
) -> dict[str, str]:
    if not isinstance(input_view_id, str) or not input_view_id:
        raise ValueError("input_view_id must be a nonempty string")
    ...
    return {
        ...,
        "input_view_id": input_view_id,
        "canonicalization_revision": OAF_CANONICALIZATION_REVISION,
    }
```

Do not parameterize checkpoint, map, adapter, thresholds, or canonicalization revision.

- [ ] **Step 6: Run focused and HPA-326 regression tests**

```bash
uv run pytest -q \
  tests/benchmark/test_input_view.py \
  tests/benchmark/test_oaf_corpus_run.py \
  tests/benchmark/test_oaf_corpus_run_acceptance.py \
  tests/benchmark/test_oaf_corpus_run_branches.py
```

Expected: all pass; existing full-mix expected bytes/hashes remain unchanged.

- [ ] **Step 7: Commit**

```bash
git add src/benchmark/input_view.py src/benchmark/oaf_corpus_run.py \
  tests/benchmark/test_input_view.py tests/benchmark/test_oaf_corpus_run.py \
  tests/benchmark/test_oaf_corpus_run_acceptance.py
git commit -m "refactor: support derived OaF input views"
```

---

### Task 2: Freeze and validate the two external separator runtimes

**Files:**
- Create: `src/benchmark/separators.py`
- Create: `scripts/freeze_separator_runtime.py`
- Create: `runtime/separators/spleeter/model.json`
- Create: `runtime/separators/htdemucs/model.json`
- Test: `tests/benchmark/test_separators.py`

**Interfaces:**
- Produces:

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
def render_separator_lock(payload: Mapping[str, object]) -> bytes: ...
```

- [ ] **Step 1: Write lock parser/hash tests first**

Use fixed synthetic hashes rather than real model bytes:

```python
def test_separator_lock_sha_binds_model_and_command(tmp_path):
    lock = write_lock(
        tmp_path,
        separator_id=SPLEETER_SEPARATOR_ID,
        model_files=[("model.data", "a" * 64)],
        argv=["-m", "spleeter", "separate", "-p", "spleeter:4stems"],
    )
    loaded = load_separator_lock(lock)
    assert loaded.separator_id == SPLEETER_SEPARATOR_ID
    assert len(loaded.sha256) == 64
```

Add rejection tests for:

- unknown/missing keys;
- noncanonical JSON;
- duplicate model filenames;
- invalid/lowercase SHA-256;
- empty package/repository/model fields;
- absolute model filenames;
- unsupported separator IDs;
- model/config arguments inconsistent with the selected separator (`spleeter:4stems` for Spleeter, `-n htdemucs` for Demucs).

Run:

```bash
uv run pytest -q tests/benchmark/test_separators.py
```

Expected: FAIL because the module does not exist.

- [ ] **Step 2: Implement the closed lock schema and canonical hash**

Use the repository's `strict_json_loads()`, `canonical_json_bytes()`, and `require_sha256()`; do not write another JSON utility.

Keep the schema's exact keys closed. Compute `SeparatorLock.sha256` from the canonical lock bytes themselves, not from a second manually maintained ID.

- [ ] **Step 3: Implement a mechanical freeze script**

The script receives explicit evidence rather than guessing model cache locations:

```text
python scripts/freeze_separator_runtime.py \
  --separator spleeter \
  --python /path/to/spleeter-python \
  --repository-url https://github.com/deezer/spleeter \
  --repository-revision <40-hex revision> \
  --model-id spleeter:4stems \
  --model-file /path/to/downloaded/model/file ... \
  --code-license MIT \
  --model-license <published-model-license> \
  --output runtime/separators/spleeter/model.json
```

For Demucs the fixed semantic command is standard `htdemucs` and the model files are the exact downloaded checkpoint files used by the isolated interpreter.

The script queries package version mechanically through the chosen interpreter:

```python
subprocess.run(
    [python, "-c", "import importlib.metadata as m; print(m.version(PACKAGE))"],
    check=True,
    capture_output=True,
    text=True,
)
```

It hashes every `--model-file`, writes only its basename + SHA-256, and emits canonical JSON. It must never copy absolute local paths into the lock.

- [ ] **Step 4: Reproduce each official upstream smoke before freezing**

Spleeter isolated environment:

```bash
/path/to/spleeter-python -m spleeter separate -p spleeter:4stems -o /tmp/crux-spleeter-smoke <known-test-audio>
```

Require a decodable `drums.wav` and record the exact package/repository/model files used.

Demucs isolated environment:

```bash
/path/to/demucs-python -m demucs -n htdemucs -o /tmp/crux-demucs-smoke <known-test-audio>
```

Require `htdemucs/<track-name>/drums.wav` and record the exact checkpoint files used.

Do not run benchmark references or inspect OaF scores during this freeze.

- [ ] **Step 5: Generate and commit the real lock files**

Run `freeze_separator_runtime.py` twice with the reproduced environments/model files, then validate them through `load_separator_lock()` in a focused test or one-off command.

The task is complete only when both checked-in lock files contain real package/repository/model hashes from a successfully reproduced upstream command; do not commit synthetic hashes in `runtime/`.

- [ ] **Step 6: Run focused tests and lint**

```bash
uv run pytest -q tests/benchmark/test_separators.py
uv run ruff check src/benchmark/separators.py scripts/freeze_separator_runtime.py tests/benchmark/test_separators.py
```

- [ ] **Step 7: Commit**

```bash
git add src/benchmark/separators.py scripts/freeze_separator_runtime.py \
  runtime/separators/spleeter/model.json runtime/separators/htdemucs/model.json \
  tests/benchmark/test_separators.py
git commit -m "feat: freeze separation runtimes"
```

---

### Task 3: Add exact stem publication, cache identity, and QC

**Files:**
- Modify: `src/benchmark/separators.py`
- Test: `tests/benchmark/test_separators.py`

**Interfaces:**
- Consumes: `SeparatorLock`, `publish_immutable_file()`, `ResolvedSourceAudio`.
- Produces:

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


def stem_cache_path(
    output_root: Path,
    *,
    source_audio_sha256: str,
    lock: SeparatorLock,
) -> Path: ...


def run_spleeter_drums(..., command_runner: CommandRunner = subprocess.run) -> SeparatedStem: ...
def run_htdemucs_drums(..., command_runner: CommandRunner = subprocess.run) -> SeparatedStem: ...
```

- [ ] **Step 1: Write cache/QC failures with tiny WAV fixtures**

Cover:

```python
def test_stem_cache_path_binds_source_and_separator_lock(...): ...
def test_publish_reuses_matching_exact_stem(...): ...
def test_publish_rejects_conflicting_existing_stem(...): ...
def test_qc_rejects_zero_and_near_silent(...): ...
def test_qc_rejects_large_duration_mismatch(...): ...
def test_qc_records_clipping_without_rejecting(...): ...
def test_qc_warns_on_non_44100_or_mono_output(...): ...
def test_qc_rejects_more_than_two_channels(...): ...
def test_process_nonzero_is_item_local_separator_error(...): ...
def test_expected_drum_output_must_exist(...): ...
```

Use `soundfile.write()` to create deterministic short fixtures. Keep QC independent of DTX/reference data.

- [ ] **Step 2: Implement deterministic cache path and exact immutable publication**

Path:

```python
return (
    output_root
    / "derived"
    / "stems"
    / lock.separator_id
    / source_audio_sha256
    / lock.sha256
    / "drums.wav"
)
```

After the external process writes to a private temporary work directory, read the exact drum WAV bytes and publish them through `publish_immutable_file()`. Never move/overwrite directly into the identity path.

- [ ] **Step 3: Implement fixed stem QC**

Decode with `soundfile` as float data. Derive:

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

Fail when:

```python
channel_count not in {1, 2}
not np.isfinite(samples).all()
rms_dbfs <= STEM_NEAR_SILENT_DBFS
abs(duration_sec - source_duration_sec) > max(
    STEM_MAX_DURATION_DELTA_SECONDS,
    source_duration_sec * STEM_MAX_DURATION_DELTA_RATIO,
)
```

Warn for sample rate != 44100, channel count != 2, or `peak_abs >= STEM_CLIP_ABS`.

- [ ] **Step 4: Implement exactly two command renderers**

Spleeter semantic argv after interpreter:

```text
-m spleeter separate -p spleeter:4stems -o <temporary-output-dir> <source-path>
```

Demucs semantic argv after interpreter:

```text
-m demucs -n htdemucs -o <temporary-output-dir> <source-path>
```

The lock parser verifies these score-relevant model/config tokens. The only runtime substitution is source path and temporary output directory.

- [ ] **Step 5: Test cache hits avoid process execution**

Use an injected runner that raises if called, seed a persisted exact stem + expected identity evidence, and assert a resumed cache hit validates bytes/QC without invoking the process.

- [ ] **Step 6: Run tests**

```bash
uv run pytest -q tests/benchmark/test_separators.py
```

- [ ] **Step 7: Commit**

```bash
git add src/benchmark/separators.py tests/benchmark/test_separators.py
git commit -m "feat: cache and validate drum stems"
```

---

### Task 4: Build the fixed-subset HPA-328 run identity and snapshot preflight

**Files:**
- Create: `src/benchmark/separation_pilot.py`
- Create: `tests/benchmark/test_separation_pilot.py`
- Reuse: `tests/benchmark/reviewed_subset_fixtures.py`

**Interfaces:**
- Consumes: HPA-323/HPA-324/HPA-327 loaders, `preflight_reference_mappings()`, HPA-326 run parser/cohort reconstruction, two separator locks.
- Produces:

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

- [ ] **Step 1: Write fatal-preflight tests before any separator/backend can be constructed**

Inject factories/runners that raise on invocation and prove they are untouched for:

- reviewed-subset/reference mismatch;
- reviewed-subset/timing mismatch;
- parent HPA-326 run reference/timing mismatch;
- missing subset member in parent HPA-326 population;
- invalid/missing separator lock;
- mixed OaF descriptor/model lock/checkpoint identity;
- run output aliasing the parent HPA-326 reports directory.

- [ ] **Step 2: Write the fixed-membership test**

Build a 20-row canonical reviewed subset fixture and assert the initial run snapshot contains exactly those 20 `simfile_id` values, sorted numerically, with no selector/seed/count option in the request.

- [ ] **Step 3: Define deterministic run identity**

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

Use `"oaf-separation-" + sha256(...).hexdigest()[:16]` as the directory ID.

- [ ] **Step 4: Define one closed per-view snapshot shape**

Each item has authoritative source/full-mix evidence and two nested view objects. A pending view begins as:

```python
{
    "status": "pending",
    "separator_config_sha256": lock.sha256,
    "stem_path": None,
    "stem_sha256": None,
    "stem_byte_length": None,
    "separation_wall_time_sec": None,
    "stem_qc": None,
    "input_view_id": input_view_id,
    "input_audio_sha256": None,
    "prediction_path": None,
    "prediction_sha256": None,
    "oaf_wall_time_sec": None,
    "failure_code": None,
    "failure_detail": None,
}
```

Allowed status values are exactly `pending`, `separation_failed`, `stem_invalid`, `inference_failed`, `prediction_invalid`, `success`, `resumed`.

- [ ] **Step 5: Implement canonical snapshot parse/render/write**

Follow the HPA-326 run pattern: `canonical_json_bytes()`, six-place float normalization, `atomic_replace_bytes()`, strict exact identity on resume, sorted items, and closed status/nullability checks.

Do not extract a generic run-snapshot framework.

- [ ] **Step 6: Reconstruct and publish the full-mix control reports without backend execution**

Use the promoted shared reviewed-subset scorer against the parent HPA-326 cohort and write under:

```text
<run_dir>/views/full_mix/reports/
```

Add a test where the OaF backend factory raises unconditionally and baseline scoring still succeeds.

- [ ] **Step 7: Run focused tests**

```bash
uv run pytest -q tests/benchmark/test_separation_pilot.py
```

- [ ] **Step 8: Commit**

```bash
git add src/benchmark/separation_pilot.py tests/benchmark/test_separation_pilot.py
git commit -m "feat: define OaF separation pilot run"
```

---

### Task 5: Execute/resume the two separator views through one frozen OaF backend

**Files:**
- Modify: `src/benchmark/separation_pilot.py`
- Modify: `tests/benchmark/test_separation_pilot.py`
- Create: `tests/benchmark/test_separation_pilot_acceptance.py`

**Interfaces:**
- Consumes: `resolve_source_audio()`, `run_spleeter_drums()`, `run_htdemucs_drums()`, `materialize_derived_audio()`, OaF `create_backend()`, `map_oaf_prediction()`, `prediction_path()`, `publish_prediction_artifact()`, existing HPA-326 backend error policy.
- Produces:

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

- [ ] **Step 1: Write an acceptance test proving the full mix is not inferred**

Use a fake backend recording every `CanonicalAudio.input_view_id`. Run a two-song synthetic pilot and assert:

```python
assert called_views == [
    SPLEETER_INPUT_VIEW_ID,
    HTDEMUCS_INPUT_VIEW_ID,
    SPLEETER_INPUT_VIEW_ID,
    HTDEMUCS_INPUT_VIEW_ID,
]
assert OAF_FULL_MIX_INPUT_VIEW_ID not in called_views
```

Order may be grouped by song or view, but freeze one deterministic order and assert it.

- [ ] **Step 2: Write resume tests before implementation**

Cover:

- exact stem + prediction hit becomes `resumed` with no separator/backend call;
- valid stem but missing prediction skips separator and reruns OaF only;
- invalid stem hash never resumes;
- prediction descriptor/config/input hash mismatch never resumes;
- existing different immutable prediction at the expected path becomes `prediction_invalid`, not overwrite;
- completed full-mix control is never regenerated by resume.

- [ ] **Step 3: Resolve all source paths before separator/backend work**

For each HPA-327 row, use `resolve_source_audio(..., load_body=False)` against the verified HPA-321 cache and persist:

```text
source_audio_id
source_audio_sha256
source_duration_sec
```

A source-resolution failure is fatal for this fixed pilot because neither separator can make a valid paired observation for that song. Do not silently shrink membership.

- [ ] **Step 4: Build the two derived OaF inference configs from the same frozen OaF model evidence**

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

Compute separate config SHA-256 values. Assert every field except `input_view_id` is identical between them and to the full-mix semantic config.

- [ ] **Step 5: Execute one song/view with checkpointing at each durable boundary**

For each separator view:

1. run/reuse exact stem;
2. persist stem/QC/runtime row and checkpoint `run.json`;
3. materialize temporary canonical WAV beneath `<run_dir>/inputs/<simfile_id>/`;
4. run/reuse normal OaF prediction v2;
5. persist canonical input hash, prediction path/hash, OaF wall time and checkpoint;
6. delete temporary canonical WAV in a `finally` block constrained beneath the input root.

The exact retained stem is never deleted.

- [ ] **Step 6: Keep one persistent OaF backend for all required derived inference**

Construct it lazily before the first prediction miss. Reuse the HPA-326 backend error classifier:

- item-local errors mark only the current view `inference_failed`;
- poison errors mark the current view, stop later OaF inference for this invocation, checkpoint, and produce partial exit 1;
- close failure is run-level evidence and produces partial exit 1 after reports are finalized where possible.

Do not add automatic worker restart.

- [ ] **Step 7: Keep separator failures independent**

If Spleeter fails/QC-rejects a song, still attempt HTDemucs for that song. If HTDemucs fails, preserve Spleeter. Do not remove the song from the fixed run snapshot.

- [ ] **Step 8: Run execution tests**

```bash
uv run pytest -q \
  tests/benchmark/test_separation_pilot.py \
  tests/benchmark/test_separation_pilot_acceptance.py
```

- [ ] **Step 9: Commit**

```bash
git add src/benchmark/separation_pilot.py \
  tests/benchmark/test_separation_pilot.py \
  tests/benchmark/test_separation_pilot_acceptance.py
git commit -m "feat: run OaF on separated drum views"
```

---

### Task 6: Assemble and score the two derived HPA-325 cohorts

**Files:**
- Modify: `src/benchmark/separation_pilot.py`
- Modify: `tests/benchmark/test_separation_pilot_acceptance.py`
- Test: existing `tests/benchmark/test_cohort_scoring.py`

**Interfaces:**
- Consumes: preflight reference mappings, persisted per-view prediction artifacts, `cohort_item_from_artifacts()`, `score_cohort()`, `write_cohort_reports()`.
- Produces reports at:

```text
<run_dir>/views/spleeter4_drums/reports/
<run_dir>/views/htdemucs_drums/reports/
```

- [ ] **Step 1: Write a mixed-success population test**

Construct three songs where:

- song 1 succeeds in both views;
- song 2 Spleeter fails but HTDemucs succeeds;
- song 3 Spleeter succeeds but HTDemucs fails.

Assert each view's `items.csv` has all three songs and exact success/failed counts. No view may contain only the paired-success intersection.

- [ ] **Step 2: Define one view-cohort identity helper**

For each view, construct `CohortIdentity` from the same HPA-326/OaF identities and only change:

```text
cohort_id
input_view_id
```

`prediction_map_version`, taxonomy/lane map, descriptor/model/model-lock identities remain the existing OaF values.

The cohort ID binds the HPA-328 run ID + view ID:

```python
cohort_id = sha256(
    canonical_json_bytes(
        {"separation_run_id": run_id, "input_view_id": input_view_id}
    )
).hexdigest()
```

- [ ] **Step 3: Convert snapshot rows into HPA-325 items**

For `success`/`resumed`, read the immutable prediction artifact and call `cohort_item_from_artifacts()` with the matching reference mapping.

For a non-success view, construct a failed `CohortItem` with the mapped reference population retained and a stable failure family (`inference_failed`, `prediction_artifact_invalid`, or `backend_unavailable` as appropriate). Keep detailed separator/QC text only in `run.json`, not as arbitrary grouped HPA-325 failure strings.

- [ ] **Step 4: Score/report unchanged**

```python
result = score_cohort(identity, cohort_items, diagnostics_for=successful_subset_ids)
write_cohort_reports(result, reports_path)
```

Retain event diagnostics for successful reviewed songs because HPA-328/HPA-329 need representative helped/harmed FP/FN examples.

- [ ] **Step 5: Assert 30/50/100 ms raw/aligned semantics are inherited**

Do not add tolerance arguments to the HPA-328 request/CLI. Tests assert HPA-325 produces its existing tolerance/mode rows for both views.

- [ ] **Step 6: Run tests**

```bash
uv run pytest -q \
  tests/benchmark/test_separation_pilot_acceptance.py \
  tests/benchmark/test_cohort_scoring.py
```

- [ ] **Step 7: Commit**

```bash
git add src/benchmark/separation_pilot.py tests/benchmark/test_separation_pilot_acceptance.py
git commit -m "feat: score separated OaF cohorts"
```

---

### Task 7: Produce the narrow paired comparison and resource report

**Files:**
- Create: `src/benchmark/separation_comparison.py`
- Create: `tests/benchmark/test_separation_comparison.py`
- Modify: `src/benchmark/separation_pilot.py`

**Interfaces:**
- Produces:

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
    run_snapshot: Mapping[str, object],
    output_dir: Path,
    cost_rates: CostRates = CostRates(),
) -> SeparationComparisonArtifacts: ...
```

- [ ] **Step 1: Write identity-join failures first**

Reject comparison when any report differs in:

- reference manifest SHA;
- timing version;
- taxonomy/lane map/prediction map;
- OaF backend/model/model-lock/descriptor;
- scoring version;
- expected input-view ID.

Also reject duplicate per-song/per-class join keys.

- [ ] **Step 2: Write the paired-intersection test**

Use synthetic HPA-325 CSVs where each view has a different failure. Assert:

```python
assert summary["spleeter"]["independent_success_count"] == expected
assert summary["spleeter"]["paired_success_count"] == intersection_expected
assert summary["htdemucs"]["independent_success_count"] == expected
```

Only rows successful in both full mix and the compared derived view appear in that pair's delta calculations.

- [ ] **Step 3: Implement CSV joins without rescoring**

Read the published HPA-325 files. For each matching score row emit the original metric values plus derived-minus-full-mix deltas. Preserve the complete key dimensions already in the score rows, including tolerance and scoring mode.

Do not reconstruct events or call `score_cohort()` from this module.

- [ ] **Step 4: Add helped/harmed summary**

For each derived view and tolerance/mode, sort by F1 delta then `simfile_id`. The Markdown summary lists a bounded five most helped and five most harmed songs plus per-class direction. Do not add a configurable ranking policy.

- [ ] **Step 5: Add measured runtime/storage and simple cost arithmetic**

From `run.json` sum:

```text
separator wall time
OaF derived-view wall time
retained stem bytes
prediction bytes
report bytes
```

Use decimal arithmetic:

```python
compute_cost = Decimal(str(total_wall_time_sec)) / Decimal("3600") * rate
storage_gb = Decimal(total_bytes) / Decimal(1024 ** 3)
storage_monthly_cost = storage_gb * storage_rate
```

Rates are evidence/display only and default to zero for local-hardware marginal-cost reporting.

- [ ] **Step 6: Render deterministic artifacts**

```text
comparison/summary.json
comparison/per_song_delta.csv
comparison/per_class_delta.csv
comparison/summary.md
```

Use existing canonical numeric conventions: six-place Decimal tokens for float-derived values and stable row ordering.

- [ ] **Step 7: Wire comparison publication into completed/partial pilot finalization**

The run command writes comparison artifacts after all three HPA-325 report sets exist. A partial run can still publish comparison evidence for successful pairs while retaining failures in `summary.json`.

- [ ] **Step 8: Run tests**

```bash
uv run pytest -q tests/benchmark/test_separation_comparison.py tests/benchmark/test_separation_pilot_acceptance.py
```

- [ ] **Step 9: Commit**

```bash
git add src/benchmark/separation_comparison.py src/benchmark/separation_pilot.py \
  tests/benchmark/test_separation_comparison.py tests/benchmark/test_separation_pilot_acceptance.py
git commit -m "feat: compare OaF separation views"
```

---

### Task 8: Publish the immutable HPA-396 handoff manifest and operator decision

**Files:**
- Create: `src/benchmark/separation_handoff.py`
- Create: `tests/benchmark/test_separation_handoff.py`
- Create: `tests/benchmark/schema_goldens/oaf-separation-pilot-v1.jsonl`

**Interfaces:**
- Consumes: completed/partial HPA-328 run snapshot, comparison artifact hashes, HPA-327 subset.
- Produces:

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
def validate_schema_golden(content: bytes) -> None: ...
```

- [ ] **Step 1: Write the one-row schema golden and parser tests**

The golden exercises exact key shape but does not enforce the production 20–30 population in the golden validator, matching the existing reviewed-subset pattern.

Normal `load_separation_pilot_manifest()` enforces exact equality with the linked reviewed-subset membership when used by finalization/downstream HPA-396.

- [ ] **Step 2: Freeze exact row fields and status/nullability**

Use the design's flat row fields. For each of the three view groups:

- `success`/`resumed` requires input hash + prediction path/hash;
- a failed view requires those unavailable fields to be null;
- Spleeter/HTDemucs success additionally requires separator config + stem path/hash;
- HTDemucs success always carries the exact retained stem bytes identity HPA-396 will consume.

Do not let generated row fields be edited during finalization.

- [ ] **Step 3: Bind all shared identity in every row**

Require one value across all rows for:

```text
reviewed subset SHA
reference/timing identities
taxonomy/lane/prediction-map/scoring versions
OaF backend/model/model-lock/checkpoint/adapter identities
Spleeter lock SHA
HTDemucs lock SHA
comparison artifact SHA values
decision/rationale/cost rates
Crux commit
```

Repeating these across 20–30 rows is intentional; do not create a header/database just to deduplicate them.

- [ ] **Step 4: Resolve and hash every handoff artifact before publication**

Finalization re-reads:

- exact retained stems;
- prediction artifacts;
- comparison artifacts;

and verifies each persisted SHA/path before rendering the manifest. A missing/edited HTDemucs stem prevents publishing a successful HTDemucs row.

- [ ] **Step 5: Publish through existing canonical manifest rails**

Reuse `render_manifest()`/`publish_manifest()` style helpers already used by benchmark manifests. The final manifest is immutable/content-derived and rerender deterministic.

- [ ] **Step 6: Add a downstream-oriented loader test**

Demonstrate HPA-396 can obtain, for every successful HTDemucs row:

```python
(
    row.simfile_id,
    row.source_audio_sha256,
    row.htdemucs_stem_path,
    row.htdemucs_stem_sha256,
    row.htdemucs_input_audio_sha256,
    row.htdemucs_prediction_sha256,
)
```

without reading HPA-328's mutable `run.json`.

- [ ] **Step 7: Run tests**

```bash
uv run pytest -q tests/benchmark/test_separation_handoff.py
```

- [ ] **Step 8: Commit**

```bash
git add src/benchmark/separation_handoff.py tests/benchmark/test_separation_handoff.py \
  tests/benchmark/schema_goldens/oaf-separation-pilot-v1.jsonl
git commit -m "feat: publish separation pilot handoff"
```

---

### Task 9: Add two thin CLI commands

**Files:**
- Modify: `src/cli/benchmark.py`
- Modify: `tests/test_cli_benchmark.py`
- Modify if coverage style requires: `tests/test_cli_benchmark_coverage.py`

**Interfaces:**
- Consumes `run_oaf_separation_pilot()` and `finalize_separation_pilot()`.

- [ ] **Step 1: Write Click signature tests**

`run-oaf-separation-pilot` must expose exactly the design inputs:

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

Assert it does **not** expose model name, shift, overlap, QC threshold, tolerance, score threshold, full-corpus, or third-separator flags.

`finalize-oaf-separation-pilot` exposes:

```text
--run
--subset-manifest
--output-manifest
--decision
--rationale
--compute-cost-per-hour
--storage-cost-per-gb-month
```

- [ ] **Step 2: Add lazy imports inside each command**

Keep Spleeter/Demucs/OaF heavy modules out of CLI import-time startup, following the file's existing lazy-boundary pattern.

- [ ] **Step 3: Map outcome to canonical JSON and exit codes**

Run command emits:

```python
{
    "exit_code": outcome.exit_code,
    "overall_status": outcome.overall_status,
    "run_id": outcome.run_id,
    "run_path": str(outcome.run_path) if outcome.run_path else None,
    "full_mix_reports_path": ...,
    "spleeter_reports_path": ...,
    "htdemucs_reports_path": ...,
    "comparison_path": ...,
}
```

Use `canonical_json_bytes(..., trailing_newline=True)` and `ctx.exit(outcome.exit_code)`.

Finalize command emits manifest path/version/hash + decision and exits 0 only after immutable publication.

- [ ] **Step 4: Add UsageError tests for invalid negative cost rates and missing rationale**

The decision is a `click.Choice` over the five frozen values. Rationale must be nonempty after stripping whitespace.

- [ ] **Step 5: Run CLI tests**

```bash
uv run pytest -q tests/test_cli_benchmark.py tests/test_cli_benchmark_coverage.py
```

- [ ] **Step 6: Commit**

```bash
git add src/cli/benchmark.py tests/test_cli_benchmark.py tests/test_cli_benchmark_coverage.py
git commit -m "feat: add separation pilot commands"
```

---

### Task 10: Run the real frozen pilot, finalize the decision, and close verification

**Files:**
- Runtime artifacts under the chosen benchmark output directory; do not commit source audio, stems, model weights, or prediction/result artifacts to git.
- Update documentation only if actual operator commands require a correction discovered before scored results are inspected.

**Interfaces:**
- Consumes the real HPA-323/HPA-324/HPA-327 manifests, HPA-321 cache, HPA-326 run, two frozen separator environments/locks, OaF checkpoint/runtime.
- Produces the final `crux.oaf-separation-pilot/v1` manifest used by HPA-396.

- [ ] **Step 1: Verify the implementation branch is score-frozen**

Before running the pilot, record:

```bash
git rev-parse HEAD
sha256sum runtime/separators/spleeter/model.json runtime/separators/htdemucs/model.json
```

Confirm no separator model/config/QC/tolerance flags exist beyond the frozen design.

- [ ] **Step 2: Run the complete HPA-327 membership once**

```bash
uv run crux benchmark run-oaf-separation-pilot \
  --manifest <HPA-324-manifest> \
  --timing-manifest <HPA-323-timing-manifest> \
  --subset-manifest <HPA-327-reviewed-subset> \
  --oaf-run <HPA-326-run.json> \
  --cache-dir <HPA-321-cache> \
  --output-dir <HPA-328-output> \
  --spleeter-python <isolated-spleeter-python> \
  --demucs-python <isolated-demucs-python>
```

If execution is interrupted, rerun the exact command with `--resume`. Do not alter separator locks/config based on partial scores.

- [ ] **Step 3: Inspect required evidence, not tune it**

Require:

- full-mix/Spleeter/HTDemucs HPA-325 reports;
- every HPA-327 member present in every view population ledger;
- exact Spleeter/HTDemucs stem status/hash evidence;
- paired-success counts and failure counts;
- 30/50/100 ms per-song and per-class deltas;
- separation/OaF runtime and retained bytes;
- representative helped/harmed diagnostics.

Do not rerun alternative model variants because one score looks disappointing.

- [ ] **Step 4: Record the operator decision and publish handoff**

Choose exactly one closed decision based on the frozen evidence:

```bash
uv run crux benchmark finalize-oaf-separation-pilot \
  --run <HPA-328-output>/runs/<run-id>/run.json \
  --subset-manifest <HPA-327-reviewed-subset> \
  --output-manifest <HPA-328-output>/handoff/manifest.jsonl \
  --decision <one-closed-decision> \
  --rationale "<short evidence-based rationale>" \
  --compute-cost-per-hour <actual marginal rate, 0 for local sunk hardware> \
  --storage-cost-per-gb-month <actual marginal rate, 0 for local sunk storage>
```

The published manifest must expose exact HTDemucs stem hashes/paths and OaF-on-HTDemucs prediction identities for HPA-396.

- [ ] **Step 5: Run repository-wide verification**

```bash
uv run pytest -q
uv run ruff check .
uv run ruff format --check src tests scripts
uv run pylint --errors-only --disable=E1120,E0401 --jobs=1 src
git diff --check
```

Expected: all pass. Real Spleeter/Demucs execution remains an operational acceptance run, not a CI dependency.

- [ ] **Step 6: Check scope**

```bash
git diff --stat origin/main...HEAD
git status --short
```

Confirm there is no full-corpus separator path, generic experiment/plugin framework, bundled model weight, or committed audio/stem/result artifact.

- [ ] **Step 7: Commit any final test/document-only corrections made before completion**

If verification required source/test corrections, commit those intentionally. Do not commit generated benchmark artifacts.

---

## Plan self-review

### Spec coverage

- Fixed score-blind subset: Tasks 4–5.
- Persisted full-mix control/no rerun: Tasks 4–5.
- Exact Spleeter 4-stem + HTDemucs locks: Tasks 2–3.
- Source/config cache identity and practical QC: Task 3.
- Same OaF with distinct derived input views: Tasks 1 and 5.
- HPA-325 scoring with complete population: Task 6.
- Paired per-song/per-class accuracy plus runtime/storage/cost: Task 7.
- Immutable HPA-396 HTDemucs handoff: Task 8.
- Thin operator workflow and closed recommendation decision: Task 9.
- Real frozen evidence and repository gates: Task 10.

### Scope check

The plan adds four focused HPA-328 modules because process execution, orchestration, report joining, and downstream manifest validation are independently testable responsibilities. It deliberately does not extract generic separator/experiment abstractions from them.

### Type/name consistency

The plan consistently uses:

```text
SeparatorLock / SeparatedStem / StemQc
OafSeparationPilotRequest / OafSeparationPilotOutcome
run_oaf_separation_pilot
write_separation_comparison
FinalizeSeparationPilotRequest / finalize_separation_pilot
SPLEETER_INPUT_VIEW_ID / HTDEMUCS_INPUT_VIEW_ID
```

The same exact names should be used by CLI and tests.
