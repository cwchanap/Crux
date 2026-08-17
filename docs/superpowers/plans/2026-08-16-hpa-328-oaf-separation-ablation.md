# HPA-328 OaF Drum-Stem Separation Ablation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run one fixed 20–30-song OaF input ablation comparing the persisted full-mix baseline with Spleeter 4-stem drums and standard HTDemucs drums, then publish paired HPA-325 reports and an immutable HTDemucs handoff for HPA-396.

**Architecture:** Reuse the HPA-327 reviewed subset as the whole pilot and reconstruct the HPA-326 full-mix control without rerunning OaF. Invoke Spleeter and Demucs as two pinned external commands, cache their exact drum WAVs by source/config identity, canonicalize those stems through the existing OaF audio boundary, persist normal prediction-v2 artifacts, and score through HPA-325. Keep process execution, pilot orchestration, comparison reporting, and final handoff as four focused HPA-328 modules; do not build a generic experiment/separator framework.

**Tech Stack:** Python 3.12 Crux host, Click, existing canonical JSON/JSONL/CSV helpers, NumPy/librosa/soundfile, isolated external Spleeter environment, isolated external Demucs environment, existing OaF TF1 worker, pytest/Ruff/Pylint.

## Global Constraints

- Start implementation from refreshed `main` after HPA-395 PR #24 lands; reuse its promoted `resolve_source_audio()`, `materialize_full_mix_audio()`, `prediction_path()`, and `score_reviewed_subset_cohort()` seams rather than duplicating them.
- Pilot membership is exactly the supplied HPA-327 `crux.reviewed-reference-subset/v1` population. No second selector, seed, replacement, or score-aware filter.
- Full-mix OaF is never rerun by HPA-328.
- Required separators are exactly official Spleeter 4-stem drums and standard `htdemucs` drums. No third separator in v1.
- Spleeter and Demucs stay out of Crux's main Python dependency graph.
- Freeze exact separator package/repository/model bytes and command arguments before scored results are inspected.
- Fixed QC constants: duration difference `<= max(0.5s, source_duration * 0.005)`, near-silent failure at RMS `<= -80 dBFS`, clipping evidence at absolute peak `>= 0.9999`. None are CLI flags.
- Retain exact separator-produced WAV bytes through HPA-396.
- Derived OaF view IDs are `crux.oaf-spleeter4-drums-mono44k1-pcm16/v1` and `crux.oaf-htdemucs-drums-mono44k1-pcm16/v1`.
- OaF descriptor/model lock/checkpoint/adapter/prediction map and HPA-325 scoring configuration remain unchanged across views.
- Per-view failures remain in population accounting. Paired deltas use matching successful songs only and report independent failure counts separately.
- FP/FN-per-minute is derived only as `fp_or_fn / (source_duration_sec / 60)` from the persisted HPA-325 counts and authoritative source duration. It is not a new scorer.
- No full-corpus separator mode, generic experiment runner, plugin registry, queue/worker pool, database, retry framework, SDR scorer, model tuning, or compatibility layer.

---

## File Structure

**Shared seams modified narrowly**

- `src/benchmark/input_view.py` — derived-stem canonicalization while preserving authoritative source identity.
- `src/benchmark/oaf_corpus_run.py` — explicit input-view argument on the existing OaF inference-config builder, defaulting to full mix.
- `src/cli/benchmark.py` — two thin HPA-328 commands.

**New HPA-328 modules**

- `src/benchmark/separators.py` — separator lock validation, concrete Spleeter/HTDemucs command execution, exact stem publication/cache, and QC.
- `src/benchmark/separation_pilot.py` — fixed-subset preflight, run snapshot/resume, derived OaF inference, and HPA-325 cohort publication.
- `src/benchmark/separation_comparison.py` — narrow full-mix↔separator joins, FP/FN-per-minute, runtime/storage/cost summary.
- `src/benchmark/separation_handoff.py` — canonical `crux.oaf-separation-pilot/v1` JSONL finalization/loader for HPA-396.
- `scripts/freeze_separator_runtime.py` — mechanically generate canonical separator locks from reproduced isolated environments and explicit model files.
- `runtime/separators/spleeter/model.json`
- `runtime/separators/htdemucs/model.json`

**Tests**

- `tests/benchmark/test_separators.py`
- `tests/benchmark/test_separation_pilot.py`
- `tests/benchmark/test_separation_pilot_acceptance.py`
- `tests/benchmark/test_separation_comparison.py`
- `tests/benchmark/test_separation_handoff.py`
- `tests/benchmark/schema_goldens/oaf-separation-pilot-v1.jsonl`
- existing input-view/OaF/CLI test files touched only where their public seam changes.

---

### Task 1: Support derived OaF input views without changing the full-mix baseline

**Files:**
- Modify: `src/benchmark/input_view.py`
- Modify: `src/benchmark/oaf_corpus_run.py`
- Test: `tests/benchmark/test_input_view.py`
- Test: `tests/benchmark/test_oaf_corpus_run.py`
- Test: `tests/benchmark/test_oaf_corpus_run_acceptance.py`

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

- [ ] **Step 1: Rebase and confirm HPA-395 neutral seams**

```bash
git fetch origin
git rebase origin/main
rg 'def resolve_source_audio|def materialize_full_mix_audio|def prediction_path|def score_reviewed_subset_cohort' src/benchmark
```

Expected: all four seams exist exactly once. If #24 changed a name during review, update this plan/spec reference during the rebase instead of preserving a stale alias.

- [ ] **Step 2: Characterize current full-mix bytes/identity before changing shared code**

Add/retain tests asserting:

```python
payload = build_inference_config(config, descriptor, "a" * 64)
assert payload["input_view_id"] == OAF_FULL_MIX_INPUT_VIEW_ID
assert payload["canonicalization_revision"] == OAF_CANONICALIZATION_REVISION
```

Keep the existing full-mix canonical WAV/hash expectation. Run:

```bash
uv run pytest -q tests/benchmark/test_input_view.py tests/benchmark/test_oaf_corpus_run.py
```

Expected: PASS.

- [ ] **Step 3: Write failing derived-view tests**

```python
def test_explicit_oaf_input_view_only_changes_view(config, descriptor):
    full = build_inference_config(config, descriptor, "a" * 64)
    stem = build_inference_config(
        config,
        descriptor,
        "a" * 64,
        input_view_id="crux.oaf-htdemucs-drums-mono44k1-pcm16/v1",
    )
    assert stem["input_view_id"] == "crux.oaf-htdemucs-drums-mono44k1-pcm16/v1"
    assert {k: v for k, v in stem.items() if k != "input_view_id"} == {
        k: v for k, v in full.items() if k != "input_view_id"
    }
```

Add a `materialize_derived_audio()` fixture using a stereo/non-44.1 kHz stem and assert output is 44.1 kHz mono PCM16, `source_audio_*` remains the authoritative full mix, and `input_audio_sha256` hashes the canonical stem input.

- [ ] **Step 4: Implement one private canonicalization helper plus the derived public wrapper**

Keep `materialize_full_mix_audio()` behavior byte-identical. The new wrapper must effectively do:

```python
_materialize_pcm16_mono44k1(derived_audio_path, output_path)
return load_materialized_audio(
    path=output_path,
    source_audio_id=source_audio.source_audio_id,
    source_audio_sha256=source_audio.source_audio_sha256,
    input_view_id=input_view_id,
    max_input_audio_frames=max_input_audio_frames,
)
```

Do not introduce an input-view class hierarchy.

- [ ] **Step 5: Parameterize only `input_view_id` in `build_inference_config()`**

Reject empty/non-string IDs; keep all other fields unchanged.

- [ ] **Step 6: Run focused HPA-326 regressions**

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

### Task 2: Freeze the two external separator identities

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

Use synthetic model hashes in tests. Reject unknown/missing keys, noncanonical JSON, duplicate/absolute model filenames, malformed hashes, unsupported separator IDs, and command/model mismatches. The Spleeter lock must contain `spleeter:4stems`; the Demucs lock must contain `-n htdemucs`.

- [ ] **Step 2: Implement the closed canonical lock schema**

Reuse `strict_json_loads()`, `canonical_json_bytes()`, and `require_sha256()`. `SeparatorLock.sha256` is the SHA-256 of the canonical lock bytes.

- [ ] **Step 3: Implement `freeze_separator_runtime.py` mechanically**

The script receives an isolated Python executable, repository URL/revision, model ID, explicit downloaded model files, and licenses. It queries package version through that interpreter, hashes every supplied model file, strips machine-local directories to basenames, renders canonical JSON, then round-trips through `load_separator_lock()`.

Package-version probe:

```python
subprocess.run(
    [python, "-c", f"import importlib.metadata as m; print(m.version({package!r}))"],
    check=True,
    capture_output=True,
    text=True,
)
```

- [ ] **Step 4: Reproduce both upstream smoke commands before freezing**

Spleeter semantic command:

```text
python -m spleeter separate -p spleeter:4stems -o <isolated-output> <known-audio>
```

Demucs semantic command:

```text
python -m demucs -n htdemucs -o <isolated-output> <known-audio>
```

Require decodable `drums.wav` output from each. Capture the exact package, upstream revision, and downloaded model files actually used. Do not inspect benchmark scores during this step.

- [ ] **Step 5: Generate real checked-in locks**

Run the freeze script twice using the reproduced environments and real model files. The runtime JSONs must contain real hashes; synthetic fixture hashes stay only in tests.

- [ ] **Step 6: Verify and commit**

```bash
uv run pytest -q tests/benchmark/test_separators.py
uv run ruff check src/benchmark/separators.py scripts/freeze_separator_runtime.py tests/benchmark/test_separators.py
git add src/benchmark/separators.py scripts/freeze_separator_runtime.py runtime/separators tests/benchmark/test_separators.py
git commit -m "feat: freeze separation runtimes"
```

---

### Task 3: Publish/cache exact stems and run practical QC

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

- [ ] **Step 1: Write failing cache/QC/process tests**

Cover exact cache identity, matching reuse, conflicting existing bytes, nonzero process exit, missing stem, decode failure, zero/near-silent stem, large duration mismatch, >2 channels, sample-rate/channel warnings, clipping evidence, and finite-sample validation.

- [ ] **Step 2: Implement deterministic cache path**

```python
output_root / "derived" / "stems" / lock.separator_id / source_sha / lock.sha256 / "drums.wav"
```

Run each separator in a private temporary directory, read the exact produced WAV bytes, and publish them with existing `publish_immutable_file()`. Never overwrite a conflicting identity path.

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

Fail if channels not in `{1, 2}`, any sample is nonfinite, `rms_dbfs <= -80`, or duration mismatch exceeds `max(0.5, source_duration * 0.005)`. Warn, but do not reject solely, for non-44.1 kHz output, mono output, or peak `>= 0.9999`.

- [ ] **Step 4: Implement exactly two concrete command renderers**

Spleeter arguments after interpreter:

```text
-m spleeter separate -p spleeter:4stems -o <temporary-output> <source>
```

Demucs arguments after interpreter:

```text
-m demucs -n htdemucs -o <temporary-output> <source>
```

Only source/output paths are substituted at runtime. No model/shift/overlap/clip-mode flags are exposed.

- [ ] **Step 5: Prove cache hits bypass subprocess execution**

Seed a matching immutable stem, inject a runner that raises if called, and assert resume validates bytes/QC without spawning the separator.

- [ ] **Step 6: Verify and commit**

```bash
uv run pytest -q tests/benchmark/test_separators.py
git add src/benchmark/separators.py tests/benchmark/test_separators.py
git commit -m "feat: cache and validate drum stems"
```

---

### Task 4: Define fixed-subset pilot preflight, run identity, and snapshot

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

- [ ] **Step 1: Write fatal-preflight tests with separator/backend sentinels**

Prove neither separator nor OaF backend is touched for subset/reference mismatch, subset/timing mismatch, parent HPA-326 mismatch, subset member absent from the parent population, invalid lock, mixed OaF identity, or protected-output aliasing.

- [ ] **Step 2: Write exact-membership test**

A canonical 20-row reviewed subset produces exactly those 20 sorted item rows. There is no selector/seed/count field on the request.

- [ ] **Step 3: Bind deterministic run identity**

Hash this exact payload:

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

- [ ] **Step 4: Persist one closed per-view state**

Allowed status values:

```text
pending
separation_failed
stem_invalid
inference_failed
prediction_invalid
success
resumed
```

Each row also persists `source_row_sha256` copied from HPA-327, authoritative source audio ID/SHA/duration, the parent full-mix prediction identity/status, and Spleeter/HTDemucs nested state containing lock hash, stem/QC/runtime, canonical input hash, prediction path/hash/runtime, and bounded failure evidence.

- [ ] **Step 5: Implement strict canonical snapshot parse/render/write**

Follow HPA-326's pattern: canonical JSON, six-place numeric normalization, `atomic_replace_bytes()`, sorted items, exact resume identity, and status/nullability validation. Do not extract a generic run-snapshot framework.

- [ ] **Step 6: Reconstruct full-mix control reports without backend execution**

Use the promoted reviewed-subset scorer against HPA-326 persisted artifacts and write:

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

### Task 5: Execute/resume the two derived views through frozen OaF

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

- [ ] **Step 1: Write acceptance test proving full mix is not inferred**

With a fake backend recording `CanonicalAudio.input_view_id`, assert calls contain only the two derived view IDs and never `OAF_FULL_MIX_INPUT_VIEW_ID`.

- [ ] **Step 2: Write resume matrix tests**

Cover exact stem+prediction hit (`resumed`, no process/backend), valid stem+missing prediction (OaF only), wrong stem hash (no resume), prediction input/config/descriptor mismatch (no resume), and conflicting immutable prediction (`prediction_invalid`, no overwrite).

- [ ] **Step 3: Resolve every authoritative source before expensive work**

Use `resolve_source_audio(..., load_body=False)` from the verified HPA-321 cache and persist source ID/SHA/duration. Source resolution for a fixed pilot member is fatal; do not silently shrink membership.

- [ ] **Step 4: Build two inference configs differing only by input view**

```python
spleeter_config = build_inference_config(
    oaf_config, descriptor, model_lock_sha, input_view_id=SPLEETER_INPUT_VIEW_ID
)
htdemucs_config = build_inference_config(
    oaf_config, descriptor, model_lock_sha, input_view_id=HTDEMUCS_INPUT_VIEW_ID
)
```

Assert in code/tests that all other fields equal the frozen full-mix OaF config.

- [ ] **Step 5: Execute/checkpoint one song/view at each durable boundary**

Order per view:

1. run or validate cached exact stem;
2. checkpoint stem/QC/separation runtime;
3. materialize temporary canonical OaF WAV under `<run>/inputs/<simfile_id>/`;
4. run or validate immutable prediction-v2 artifact;
5. checkpoint input hash, prediction path/hash, OaF runtime/status;
6. delete only the temporary canonical WAV in `finally`.

Never delete the retained exact stem.

- [ ] **Step 6: Use one persistent OaF backend for all prediction misses**

Construct lazily. Reuse HPA-326's error classifier. Item-local backend errors fail the current view; poison errors checkpoint and stop later OaF inference for the invocation; no automatic worker restart.

- [ ] **Step 7: Keep separator failures independent**

Spleeter failure must not prevent HTDemucs for the same song and vice versa.

- [ ] **Step 8: Verify and commit**

```bash
uv run pytest -q tests/benchmark/test_separation_pilot.py tests/benchmark/test_separation_pilot_acceptance.py
git add src/benchmark/separation_pilot.py tests/benchmark/test_separation_pilot.py tests/benchmark/test_separation_pilot_acceptance.py
git commit -m "feat: run OaF on separated drum views"
```

---

### Task 6: Score both derived cohorts through HPA-325 unchanged

**Files:**
- Modify: `src/benchmark/separation_pilot.py`
- Modify: `tests/benchmark/test_separation_pilot_acceptance.py`

**Interfaces:**
- Consumes: reference mappings, immutable prediction artifacts, `cohort_item_from_artifacts()`, `score_cohort()`, `write_cohort_reports()`.
- Produces:

```text
<run>/views/spleeter4_drums/reports/
<run>/views/htdemucs_drums/reports/
```

- [ ] **Step 1: Write mixed-success population test**

Use three songs: both-success, Spleeter-fail/HTDemucs-success, Spleeter-success/HTDemucs-fail. Each view's `items.csv` must still contain all three songs with correct failure counts.

- [ ] **Step 2: Build view cohort identity from frozen OaF identity**

Only `cohort_id` and `input_view_id` differ. `scoring_version` stays `SCORING_VERSION`; taxonomy/lane/prediction map/model/model-lock/descriptor stay fixed.

- [ ] **Step 3: Reconstruct HPA-325 items**

Success/resume reads the normal prediction artifact and calls `cohort_item_from_artifacts()`. Non-success retains mapped reference coverage and uses an existing closed HPA-325 failure family; detailed separator errors remain only in `run.json`.

- [ ] **Step 4: Score/write without new knobs**

```python
result = score_cohort(identity, cohort_items, diagnostics_for=successful_subset_ids)
write_cohort_reports(result, reports_path)
```

HPA-325 continues to own 30/50/100 ms raw/aligned behavior.

- [ ] **Step 5: Verify and commit**

```bash
uv run pytest -q tests/benchmark/test_separation_pilot_acceptance.py tests/benchmark/test_cohort_scoring.py
git add src/benchmark/separation_pilot.py tests/benchmark/test_separation_pilot_acceptance.py
git commit -m "feat: score separated OaF cohorts"
```

---

### Task 7: Produce paired metric/resource comparison, including FP/FN per minute

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
    full_mix_reports: Path,
    spleeter_reports: Path,
    htdemucs_reports: Path,
    run_snapshot: Mapping[str, object],
    output_dir: Path,
    cost_rates: CostRates = CostRates(),
) -> SeparationComparisonArtifacts: ...
```

- [ ] **Step 1: Write report-identity rejection tests**

Read each view's `summary.json` for full identity. Reject reference/timing/taxonomy/lane/prediction-map/OaF/scoring/input-view mismatches and duplicate CSV join keys.

- [ ] **Step 2: Write paired-intersection/failure-accounting test**

The summary reports independent success/failure counts per view and a separate paired-success count for full-mix↔Spleeter and full-mix↔HTDemucs. Delta rows use only the matching success intersection.

- [ ] **Step 3: Join published HPA-325 rows without rescoring**

Read `per_song.csv`, `per_class.csv`, `items.csv`, and `summary.json`. Emit source values plus `derived - full_mix` precision/recall/F1 deltas for every matching tolerance/mode/key. Never reconstruct events in this module.

- [ ] **Step 4: Derive FP/FN per minute from persisted duration**

For every per-song score row, look up the same `simfile_id` in `run_snapshot["items"]` and require positive finite `source_duration_sec`.

```python
minutes = Decimal(str(source_duration_sec)) / Decimal("60")
fp_per_min = Decimal(fp) / minutes
fn_per_min = Decimal(fn) / minutes
```

Render canonical six-place tokens. Emit both absolute values and derived-minus-full-mix deltas. This directly closes HPA-328's FP/FN-per-minute requirement without changing HPA-325.

- [ ] **Step 5: Preserve timing diagnostics and bounded helped/harmed lists**

Carry `median_abs_error_ms`, `p95_abs_error_ms`, and `offset_ms` from HPA-325 per-song rows. For each view/tolerance/mode, list at most five most helped and five most harmed songs sorted by F1 delta then `simfile_id`, plus per-class delta direction.

- [ ] **Step 6: Add runtime/storage/cost arithmetic**

Sum separator wall time, derived OaF wall time, retained stem bytes, prediction bytes, and report bytes. Cost uses Decimal only:

```python
compute_cost = Decimal(str(total_wall_time_sec)) / Decimal("3600") * compute_rate
storage_gb = Decimal(total_bytes) / Decimal(1024 ** 3)
storage_monthly_cost = storage_gb * storage_rate
```

Rates default to zero for local sunk hardware/storage and affect display only.

- [ ] **Step 7: Render deterministic comparison artifacts**

```text
comparison/summary.json
comparison/per_song_delta.csv
comparison/per_class_delta.csv
comparison/summary.md
```

Partial runs may still publish paired evidence; failures stay explicit in summary counts.

- [ ] **Step 8: Verify and commit**

```bash
uv run pytest -q tests/benchmark/test_separation_comparison.py tests/benchmark/test_separation_pilot_acceptance.py
git add src/benchmark/separation_comparison.py src/benchmark/separation_pilot.py tests/benchmark/test_separation_comparison.py
git commit -m "feat: compare OaF separation views"
```

---

### Task 8: Publish immutable HPA-396 handoff and operator decision

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

The golden validates exact one-row shape/canonical bytes. Production loader/finalizer validates exact HPA-327 20–30 membership.

- [ ] **Step 2: Freeze row identity and status/nullability**

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

`source_row_sha256` must equal the exact HPA-327 row's persisted source-row hash, closing the row-level reference/mapping lineage for HPA-396.

Success/resume requires its input/prediction fields. Separator success additionally requires exact stem path/hash. Failed views require unavailable artifact fields to be null.

- [ ] **Step 3: Re-read/hash handoff artifacts before publication**

Verify retained stems, prediction artifacts, and comparison files against persisted paths/hashes. Missing/edited HTDemucs bytes prevent publishing a successful HTDemucs row.

- [ ] **Step 4: Publish through existing canonical manifest rails**

Reuse `render_manifest()`/`publish_manifest()` style helpers. Repeat shared fields across 20–30 rows rather than creating a header/database protocol.

- [ ] **Step 5: Add downstream HPA-396 loader test**

For each successful HTDemucs row, HPA-396 must obtain without reading mutable `run.json`:

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
git add src/benchmark/separation_handoff.py tests/benchmark/test_separation_handoff.py tests/benchmark/schema_goldens/oaf-separation-pilot-v1.jsonl
git commit -m "feat: publish separation pilot handoff"
```

---

### Task 9: Add two thin CLI commands

**Files:**
- Modify: `src/cli/benchmark.py`
- Modify: `tests/test_cli_benchmark.py`
- Modify if required by existing coverage organization: `tests/test_cli_benchmark_coverage.py`

- [ ] **Step 1: Write exact Click signature tests**

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

Assert no separator model/shift/overlap/QC/tolerance/full-corpus/third-separator flags.

Finalize command exposes:

```text
--run
--subset-manifest
--output-manifest
--decision
--rationale
--compute-cost-per-hour
--storage-cost-per-gb-month
```

- [ ] **Step 2: Add lazy command imports**

Keep external/heavy benchmark modules behind the Click command boundary.

- [ ] **Step 3: Emit canonical run/finalize summaries and stable exit codes**

Run uses 0 complete / 1 partial / 2 fatal. Finalize exits 0 only after immutable manifest publication. `--decision` is a Click choice over the five frozen values; rationale must be nonempty; cost rates must be nonnegative.

- [ ] **Step 4: Verify and commit**

```bash
uv run pytest -q tests/test_cli_benchmark.py tests/test_cli_benchmark_coverage.py
git add src/cli/benchmark.py tests/test_cli_benchmark.py tests/test_cli_benchmark_coverage.py
git commit -m "feat: add separation pilot commands"
```

---

### Task 10: Run the real frozen pilot and close verification

**Files:**
- Generated benchmark artifacts remain outside git.

- [ ] **Step 1: Freeze execution evidence before scoring**

Record the implementation commit and lock hashes:

```bash
git rev-parse HEAD
sha256sum runtime/separators/spleeter/model.json runtime/separators/htdemucs/model.json
```

Confirm no separator model/QC/scoring tuning flags exist.

- [ ] **Step 2: Resolve actual artifact paths once and run the exact reviewed subset**

Set shell variables to the real already-published local artifacts for HPA-324/HPA-323/HPA-327/HPA-326/HPA-321 and the two isolated interpreters, then run:

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

If interrupted, repeat the exact command with `--resume`. Do not alter locks/config after partial scores are visible.

- [ ] **Step 3: Review required evidence only**

Require all HPA-327 members in every population ledger; full-mix/Spleeter/HTDemucs HPA-325 reports; exact stem/prediction identities; 30/50/100 ms per-song/per-class deltas; FP/FN per minute; timing/offset diagnostics; separation/OaF runtime; retained bytes; cost display; bounded helped/harmed cases.

- [ ] **Step 4: Finalize one closed decision**

Set `DECISION` to one of the five frozen enum values and `RATIONALE` to the short evidence-based conclusion, then run:

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

Use zero rates for local sunk hardware/storage; otherwise use the actual marginal rates. The handoff must expose exact HTDemucs stem hashes/paths and OaF-on-HTDemucs prediction identities for HPA-396.

- [ ] **Step 5: Run repository-wide verification**

```bash
uv run pytest -q
uv run ruff check .
uv run ruff format --check src tests scripts
uv run pylint --errors-only --disable=E1120,E0401 --jobs=1 src
git diff --check
git status --short
```

Real separator execution is an operational acceptance run, not a CI dependency.

- [ ] **Step 6: Scope audit**

```bash
git diff --stat origin/main...HEAD
```

Confirm no full-corpus separator path, generic experiment/plugin framework, bundled model weights, or committed audio/stem/result artifact.

---

## Plan Self-Review

**Spec coverage:** Tasks 4–5 freeze exact membership and reuse the persisted control; Tasks 2–3 freeze/cache/QC both separators; Tasks 1/5 keep OaF unchanged except input view; Task 6 reuses HPA-325; Task 7 covers paired accuracy, FP/FN per minute, timing, runtime, storage, and cost; Task 8 provides row-level HPA-327 lineage plus exact HTDemucs/OaF handoff; Task 9 keeps operator surfaces narrow; Task 10 performs the real frozen gate.

**Scope:** Four HPA-328 modules are justified by independently testable responsibilities—external process/artifact handling, run orchestration, report joining, and downstream manifest validation. None is a generic framework.

**Stable names:** `SeparatorLock`, `SeparatedStem`, `StemQc`, `OafSeparationPilotRequest`, `OafSeparationPilotOutcome`, `run_oaf_separation_pilot`, `write_separation_comparison`, `FinalizeSeparationPilotRequest`, `finalize_separation_pilot`, `SPLEETER_INPUT_VIEW_ID`, and `HTDEMUCS_INPUT_VIEW_ID` are the shared names used across tasks, CLI, and tests.
