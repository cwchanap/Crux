# HPA-562 Paired Benchmark Publication Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Publish one deterministic HPA-562 bundle that composes the existing OaF/MuScriptor, OaF/separation, and OaF/IDM pairwise comparisons and exposes a clearly scoped model × input-view population matrix.

**Architecture:** Keep the three existing pairwise comparison drivers authoritative. Add one concrete `cross_comparison.py` coordinator that stages those outputs, validates shared reference/scoring identity plus the frozen OaF model lock, and publishes only a top-level index/matrix. The coordinator always keeps MuScriptor broad (`subset_manifest_path=None`) and uses a closed matrix source table so same-named `oaf` entries from different comparison topologies cannot be confused.

**Tech Stack:** Python 3.12, dataclasses, `pathlib`, existing strict/canonical JSON and SHA-256 helpers, Click, pytest.

**Spec:** `docs/superpowers/specs/2026-08-22-hpa-562-paired-comparisons-design.md`

## Global Constraints

- Keep all HPA-562 planning and implementation on the existing `agent/hpa-562-paired-comparisons` branch and the same draft PR; do not open a second HPA-562 PR.
- Reuse `compare_oaf_muscriptor()`, `compare_oaf_separation()`, and `compare_oaf_idm()`; do not rebuild pair joins or rescore events.
- MuScriptor HPA-562 comparison is always broad full mix: pass `subset_manifest_path=None` even though the HPA-562 request also carries the pilot subset for HPA-328.
- Same-input model comparisons keep exact input-audio hash matching in their existing drivers; cross-input OaF separation keeps HPA-328's source-hash + fixed-view contract.
- Full-corpus and reviewed-pilot populations remain explicitly distinct.
- Headline rows come from one closed `(scope, model, expected view, comparison id, models-key)` table; never select a row by a generic `"oaf"` lookup.
- Require the OaF `model_lock_sha256` to match across MuScriptor, separation, and IDM comparison summaries.
- Require separation HTDemucs and IDM's OaF peer to name the same HTDemucs stem input view.
- Persist only relative paths inside the HPA-562 bundle and use existing `sha256_hex()` for hashes.
- Do not add backward-compatibility branches for old comparison summaries.
- Do not add a generic comparison registry, experiment runner, leaderboard service, database, UI, significance layer, or automatic winner policy.

---

## File structure

```text
Create
  src/benchmark/cross_comparison.py
  tests/benchmark/test_cross_comparison.py

Modify
  src/benchmark/muscriptor_comparison.py
  src/benchmark/separation_comparison.py
  tests/benchmark/test_muscriptor_comparison.py
  tests/benchmark/test_separation_comparison.py
  src/cli/benchmark.py
  tests/test_cli_benchmark.py
```

`cross_comparison.py` owns only bundle orchestration, cross-summary validation, the closed matrix source table, index rendering, and final publication. Existing comparison modules continue to own all pair construction and domain-specific integrity checks.

---

### Task 1: Complete the published pairwise identity needed by HPA-562

**Files:**
- Modify: `src/benchmark/muscriptor_comparison.py`
- Modify: `src/benchmark/separation_comparison.py`
- Test: `tests/benchmark/test_muscriptor_comparison.py`
- Test: `tests/benchmark/test_separation_comparison.py`

**Interfaces:**
- Consumes: `TAXONOMY_VERSION`, `DTX_LANE_MAP_VERSION`, `SCORING_VERSION`.
- Produces: MuScriptor and separation comparison `summary.json` identity objects that expose `taxonomy_version`, `lane_map_version`, and `scoring_version` using the same key spellings as IDM.

- [ ] **Step 1: Add failing assertions for MuScriptor comparison identity**

Extend the existing successful comparison test so its parsed `summary.json` must include:

```python
assert summary["identity"]["taxonomy_version"] == TAXONOMY_VERSION
assert summary["identity"]["lane_map_version"] == DTX_LANE_MAP_VERSION
assert summary["identity"]["scoring_version"] == SCORING_VERSION
```

- [ ] **Step 2: Run the focused MuScriptor comparison test and verify it fails**

```bash
uv run pytest tests/benchmark/test_muscriptor_comparison.py -q
```

Expected: the new taxonomy/lane assertions fail because the current default `comparison_summary()` identity contains reference/timing/input-view identity but omits taxonomy/lane.

- [ ] **Step 3: Persist the complete MuScriptor comparison identity**

In the final `comparison_summary(...)` call inside `compare_oaf_muscriptor()`, pass the existing `identity=` hook explicitly:

```python
identity={
    "reference_manifest_sha256": reference_manifest.manifest_sha256,
    "reference_manifest_version": reference_manifest.corpus_version,
    "reference_timing_manifest_sha256": timing_manifest.manifest_sha256,
    "reference_timing_version": timing_manifest.corpus_version,
    "taxonomy_version": TAXONOMY_VERSION,
    "lane_map_version": DTX_LANE_MAP_VERSION,
    "input_view_id": oaf.identity.input_view_id,
    "scoring_version": SCORING_VERSION,
}
```

Do not change pairing, report filenames, model entries, or schemas.

- [ ] **Step 4: Add failing assertions for separation comparison identity**

In `tests/benchmark/test_separation_comparison.py`, extend the successful publication assertion:

```python
assert summary["identity"]["taxonomy_version"] == TAXONOMY_VERSION
assert summary["identity"]["lane_map_version"] == DTX_LANE_MAP_VERSION
assert summary["identity"]["scoring_version"] == SCORING_VERSION
```

- [ ] **Step 5: Run the focused separation test and verify it fails**

```bash
uv run pytest tests/benchmark/test_separation_comparison.py -q
```

Expected: the new taxonomy/lane assertions fail against the current `_comparison_identity()` output.

- [ ] **Step 6: Complete `_comparison_identity()`**

Extend the existing returned mapping without changing HPA-328 comparison semantics:

```python
return {
    # keep all existing run/reference/timing/subset/input-view fields
    "taxonomy_version": TAXONOMY_VERSION,
    "lane_map_version": DTX_LANE_MAP_VERSION,
    "scoring_version": SCORING_VERSION,
}
```

- [ ] **Step 7: Run both focused suites**

```bash
uv run pytest \
  tests/benchmark/test_muscriptor_comparison.py \
  tests/benchmark/test_separation_comparison.py -q
```

Expected: PASS.

- [ ] **Step 8: Commit on the existing HPA-562 branch**

```bash
git add \
  src/benchmark/muscriptor_comparison.py \
  src/benchmark/separation_comparison.py \
  tests/benchmark/test_muscriptor_comparison.py \
  tests/benchmark/test_separation_comparison.py
git commit -m "refactor: expose complete comparison identity"
```

---

### Task 2: Add the concrete coordinator, scope routing, and cross-summary validation

**Files:**
- Create: `src/benchmark/cross_comparison.py`
- Create: `tests/benchmark/test_cross_comparison.py`

**Interfaces:**
- Consumes:
  - `compare_oaf_muscriptor(ComparisonRequest) -> ComparisonOutcome`
  - `compare_oaf_separation(SeparationComparisonRequest) -> SeparationComparisonOutcome`
  - `compare_oaf_idm(IdmComparisonRequest) -> Path`
  - `strict_json_loads(..., require_canonical=True)`
  - `ComparisonIntegrityError`
- Produces:

```python
PAIRED_BENCHMARK_PUBLICATION_SCHEMA = "crux.paired-benchmark-publication/v1"

@dataclass(frozen=True)
class CrossComparisonRequest:
    oaf_run_path: Path
    muscriptor_run_path: Path
    separation_run_path: Path
    idm_run_path: Path
    reference_manifest_path: Path
    timing_manifest_path: Path
    subset_manifest_path: Path
    output_dir: Path
    separation_cache_dir: Path | None = None

@dataclass(frozen=True)
class CrossComparisonOutcome:
    output_dir: Path
    headline_matrix_path: Path
    comparison_paths: dict[str, Path]
    pairable_success_counts: dict[str, int]


def publish_cross_comparisons(request: CrossComparisonRequest) -> CrossComparisonOutcome:
    ...
```

- [ ] **Step 1: Write request/outcome validation tests**

Pin Path-only inputs, optional `separation_cache_dir`, non-empty comparison maps, and non-negative pair counts:

```python
def test_cross_comparison_request_requires_paths(tmp_path: Path) -> None:
    with pytest.raises(TypeError, match="oaf_run_path"):
        CrossComparisonRequest(
            oaf_run_path="run.json",  # type: ignore[arg-type]
            muscriptor_run_path=tmp_path / "muscriptor.json",
            separation_run_path=tmp_path / "separation.json",
            idm_run_path=tmp_path / "idm.json",
            reference_manifest_path=tmp_path / "reference.jsonl",
            timing_manifest_path=tmp_path / "timing.jsonl",
            subset_manifest_path=tmp_path / "subset.jsonl",
            output_dir=tmp_path / "out",
        )
```

- [ ] **Step 2: Run the new test and verify import failure**

```bash
uv run pytest tests/benchmark/test_cross_comparison.py -q
```

Expected: FAIL because `src.benchmark.cross_comparison` does not exist.

- [ ] **Step 3: Implement fixed comparison IDs and the exact request routing**

Use:

```python
_COMPARISON_DIRS = {
    "oaf_muscriptor_full_mix": Path("comparisons/oaf-muscriptor"),
    "oaf_separation_pilot": Path("comparisons/oaf-separation"),
    "oaf_idm_htdemucs": Path("comparisons/oaf-idm"),
}
```

Construct requests exactly as follows:

```python
ComparisonRequest(
    oaf_run_path=request.oaf_run_path,
    muscriptor_run_path=request.muscriptor_run_path,
    reference_manifest_path=request.reference_manifest_path,
    timing_manifest_path=request.timing_manifest_path,
    output_dir=stage / _COMPARISON_DIRS["oaf_muscriptor_full_mix"],
    subset_manifest_path=None,
)

SeparationComparisonRequest(
    run_path=request.separation_run_path,
    reference_manifest_path=request.reference_manifest_path,
    timing_manifest_path=request.timing_manifest_path,
    subset_manifest_path=request.subset_manifest_path,
    output_dir=stage / _COMPARISON_DIRS["oaf_separation_pilot"],
    cache_dir=request.separation_cache_dir,
)

IdmComparisonRequest(
    run_path=request.idm_run_path,
    output_dir=stage / _COMPARISON_DIRS["oaf_idm_htdemucs"],
)
```

Reject an existing final `output_dir`, use one sibling staging directory, and do not parse prediction artifacts or call `score_cohort()`.

- [ ] **Step 4: Add delegation tests that pin `subset_manifest_path=None` for MuScriptor**

Monkeypatch the three driver call sites with fakes that record their request objects and write canonical minimal summaries. Assert:

```python
assert muscriptor_request.subset_manifest_path is None
assert separation_request.subset_manifest_path == request.subset_manifest_path
assert separation_request.cache_dir == request.separation_cache_dir
assert idm_request.run_path == request.idm_run_path
```

Each fake must be called exactly once.

- [ ] **Step 5: Implement canonical summary loading**

Add:

```python
def _read_summary(path: Path, *, expected_schema: str) -> Mapping[str, object]:
    content = read_regular_file_no_follow(path)
    value = strict_json_loads(content, require_canonical=True)
    if not isinstance(value, Mapping) or value.get("schema") != expected_schema:
        raise ComparisonIntegrityError("comparison summary schema mismatch")
    return value
```

Use the three existing comparison schema constants; do not accept arbitrary summaries.

- [ ] **Step 6: Implement shared reference/scoring identity validation**

Compare these exact fields across all three nested summary `identity` objects:

```python
_SHARED_FIELDS = (
    "reference_manifest_sha256",
    "reference_manifest_version",
    "reference_timing_manifest_sha256",
    "reference_timing_version",
    "taxonomy_version",
    "lane_map_version",
    "scoring_version",
)
```

A missing or unequal field raises `ComparisonIntegrityError` naming that field.

- [ ] **Step 7: Add the frozen OaF model-lock and HTDemucs-view checks**

Read these exact model entries:

```python
muscriptor_oaf = muscriptor_summary["models"]["oaf"]
separation_full_mix = separation_summary["models"]["full_mix"]
idm_oaf = idm_summary["models"]["oaf"]
```

Require:

```python
locks = {
    muscriptor_oaf["model_lock_sha256"],
    separation_full_mix["model_lock_sha256"],
    idm_oaf["model_lock_sha256"],
}
if len(locks) != 1:
    raise ComparisonIntegrityError("model_lock_sha256 mismatch for OaF comparisons")
```

Then require:

```python
separation_view = separation_summary["models"]["htdemucs"]["input_view_id"]
idm_view = idm_oaf["input_view_id"]
if (
    separation_view != HTDEMUCS_INPUT_VIEW_ID
    or idm_view != IDM_STEM_INPUT_VIEW_ID
    or separation_view != idm_view
):
    raise ComparisonIntegrityError("HTDemucs input_view_id mismatch")
```

Use exceptions rather than Python `assert` in production code.

- [ ] **Step 8: Add mismatch tests**

Parametrize shared identity mismatches:

```python
@pytest.mark.parametrize(
    "field",
    (
        "reference_manifest_sha256",
        "reference_timing_manifest_sha256",
        "taxonomy_version",
        "lane_map_version",
        "scoring_version",
    ),
)
def test_cross_publication_rejects_shared_identity_mismatch(..., field: str) -> None:
    with pytest.raises(ComparisonIntegrityError, match=field):
        publish_cross_comparisons(request)
    assert not request.output_dir.exists()
```

Add separate tests for OaF `model_lock_sha256` mismatch and HTDemucs `input_view_id` mismatch. Both must leave no final output directory.

- [ ] **Step 9: Run the focused suite**

```bash
uv run pytest tests/benchmark/test_cross_comparison.py -q
```

Expected: PASS.

- [ ] **Step 10: Commit**

```bash
git add src/benchmark/cross_comparison.py tests/benchmark/test_cross_comparison.py
git commit -m "feat: compose paired benchmark comparisons"
```

---

### Task 3: Publish the closed-source headline matrix and top-level index

**Files:**
- Modify: `src/benchmark/cross_comparison.py`
- Modify: `tests/benchmark/test_cross_comparison.py`

**Interfaces:**
- Consumes: validated nested summary mappings from Task 2.
- Produces:
  - `headline_matrix.csv`
  - `summary.json`
  - `summary.md`
  - relative-path/hash index of the three nested comparison trees.

- [ ] **Step 1: Freeze the matrix source table before writing rendering code**

Use one closed tuple; do not infer model keys from labels:

```python
_HEADLINE_SOURCES = (
    (
        "broad_full_mix",
        "oaf",
        OAF_FULL_MIX_INPUT_VIEW_ID,
        "oaf_muscriptor_full_mix",
        "oaf",
    ),
    (
        "broad_full_mix",
        "muscriptor",
        MUSCRIPTOR_FULL_MIX_INPUT_VIEW_ID,
        "oaf_muscriptor_full_mix",
        "muscriptor",
    ),
    (
        "reviewed_pilot",
        "oaf",
        OAF_FULL_MIX_INPUT_VIEW_ID,
        "oaf_separation_pilot",
        "full_mix",
    ),
    (
        "reviewed_pilot",
        "oaf",
        SPLEETER_INPUT_VIEW_ID,
        "oaf_separation_pilot",
        "spleeter",
    ),
    (
        "reviewed_pilot",
        "oaf",
        HTDEMUCS_INPUT_VIEW_ID,
        "oaf_separation_pilot",
        "htdemucs",
    ),
    (
        "reviewed_pilot",
        "idm",
        IDM_STEM_INPUT_VIEW_ID,
        "oaf_idm_htdemucs",
        "idm",
    ),
)
```

The final row must never source IDM `models["oaf"]`.

- [ ] **Step 2: Write a matrix test with distinct source populations**

Make every `models[...]` source population use a unique `total_count`, for example:

```python
muscriptor models["oaf"]          -> total_count 101
muscriptor models["muscriptor"]   -> total_count 102
separation models["full_mix"]     -> total_count 201
separation models["spleeter"]     -> total_count 202
separation models["htdemucs"]     -> total_count 203
idm models["idm"]                 -> total_count 301
idm models["oaf"]                 -> total_count 999  # sentinel; must never be a matrix source
```

Assert:

```python
assert [(row["scope"], row["model"], int(row["total_count"])) for row in rows] == [
    ("broad_full_mix", "oaf", 101),
    ("broad_full_mix", "muscriptor", 102),
    ("reviewed_pilot", "oaf", 201),
    ("reviewed_pilot", "oaf", 202),
    ("reviewed_pilot", "oaf", 203),
    ("reviewed_pilot", "idm", 301),
]
```

This catches both accidental broad/pilot merging and accidental use of IDM `models["oaf"]`.

- [ ] **Step 3: Run the matrix test and verify it fails**

```bash
uv run pytest tests/benchmark/test_cross_comparison.py -q
```

Expected: FAIL because no headline matrix is written yet.

- [ ] **Step 4: Implement deterministic matrix rendering with existing writer**

Use fixed columns:

```python
_HEADLINE_FIELDS = (
    "scope",
    "model",
    "input_view_id",
    "total_count",
    "eligible_count",
    "success_count",
    "failed_count",
    "skipped_count",
    "quarantined_count",
    "comparison_ids",
)
```

For each `_HEADLINE_SOURCES` row, read only `summaries[comparison_id]["models"][model_key]["population"]`, validate the model entry's `input_view_id` against the expected view, and render using `published_comparison.write_csv()`.

- [ ] **Step 5: Add top-level summary/hash assertions**

Require:

```python
assert summary["schema"] == PAIRED_BENCHMARK_PUBLICATION_SCHEMA
assert summary["identity"]["oaf_model_lock_sha256"] == EXPECTED_OAF_LOCK
assert summary["headline_matrix"]["path"] == "headline_matrix.csv"
assert summary["headline_matrix"]["sha256"] == sha256_hex(
    (out / "headline_matrix.csv").read_bytes()
)
assert summary["comparisons"]["oaf_muscriptor_full_mix"]["path"] == (
    "comparisons/oaf-muscriptor"
)
```

For each nested comparison, index the canonical `summary.json` plus every nested `paired_per_song.csv` / `paired_per_class.csv` file under that comparison root. Store only relative paths and `sha256_hex(file_bytes)` values.

- [ ] **Step 6: Implement top-level JSON and Markdown writers**

`summary.json` includes:

```text
schema
identity
  reference_manifest_sha256
  reference_manifest_version
  reference_timing_manifest_sha256
  reference_timing_version
  taxonomy_version
  lane_map_version
  scoring_version
  oaf_model_lock_sha256
reviewed_subset
comparisons
headline_matrix
```

Copy reviewed-subset identity from the validated separation summary; do not derive it from a path.

`summary.md` renders the six matrix rows and includes this exact meaning:

```text
Broad full-mix and reviewed-pilot rows have different populations and must not be ranked as one leaderboard.
```

- [ ] **Step 7: Add output-root independence test**

Publish identical fake evidence twice to two different non-existing output directories and assert:

```python
for name in ("summary.json", "summary.md", "headline_matrix.csv"):
    assert (left / name).read_bytes() == (right / name).read_bytes()
```

- [ ] **Step 8: Add one real-driver summary-shape integration test**

Add:

```python
def test_publish_cross_comparisons_accepts_real_driver_summary_shapes(...):
    request = _real_driver_request(...)
    outcome = publish_cross_comparisons(request)
    assert outcome.output_dir == request.output_dir
    assert (request.output_dir / "summary.json").is_file()
```

`_real_driver_request()` must write one small internally consistent raw-evidence set using the fixture builders/evidence conventions already present in:

```text
tests/benchmark/test_muscriptor_comparison.py
tests/benchmark/test_separation_comparison.py
tests/benchmark/test_idm_comparison.py
tests/benchmark/idm_pilot_fixtures.py
tests/benchmark/reviewed_subset_fixtures.py
```

Do **not** monkeypatch `compare_oaf_muscriptor`, `compare_oaf_separation`, or `compare_oaf_idm` in this test. The point is to make `publish_cross_comparisons()` execute the three real drivers and validate their real `summary.json` shapes together. Keep the helper local to `test_cross_comparison.py`; do not create a production fixture abstraction.

The fixture must use one shared reference/timing identity and one shared OaF model-lock hash across all three raw run snapshots. If the real summary shapes disagree, the test must fail with `ComparisonIntegrityError` naming the concrete field rather than being replaced by fake `_identity()` dictionaries.

- [ ] **Step 9: Run the focused suite**

```bash
uv run pytest tests/benchmark/test_cross_comparison.py -q
```

Expected: PASS, including the real-driver integration test.

- [ ] **Step 10: Commit**

```bash
git add src/benchmark/cross_comparison.py tests/benchmark/test_cross_comparison.py
git commit -m "feat: publish benchmark comparison index"
```

---

### Task 4: Wire one thin CLI command

**Files:**
- Modify: `src/cli/benchmark.py`
- Modify: `tests/test_cli_benchmark.py`

**Interfaces:**
- Consumes: `CrossComparisonRequest`, `publish_cross_comparisons()`.
- Produces: `crux benchmark publish-paired-comparisons` with exit `0` on complete publication and exit `2` on malformed/mismatched evidence or publication failure.

- [ ] **Step 1: Write the CLI success test**

Use `CliRunner` and monkeypatch only the coordinator boundary. Pin the command name and canonical output fields:

```python
assert result.exit_code == 0
payload = json.loads(result.output)
assert payload == {
    "comparison_paths": {
        "oaf_idm_htdemucs": str(output_dir / "comparisons/oaf-idm"),
        "oaf_muscriptor_full_mix": str(output_dir / "comparisons/oaf-muscriptor"),
        "oaf_separation_pilot": str(output_dir / "comparisons/oaf-separation"),
    },
    "exit_code": 0,
    "headline_matrix_path": str(output_dir / "headline_matrix.csv"),
    "output_dir": str(output_dir),
    "pairable_success_counts": expected_counts,
}
```

- [ ] **Step 2: Run the CLI test and verify command absence**

```bash
uv run pytest tests/test_cli_benchmark.py -q -k publish_paired_comparisons
```

Expected: FAIL because the command is not registered.

- [ ] **Step 3: Add `publish-paired-comparisons`**

Wire these options directly to `CrossComparisonRequest`:

```text
--oaf-run
--muscriptor-run
--separation-run
--idm-run
--manifest
--timing-manifest
--subset-manifest
--separation-cache-dir   (optional)
--output-dir
```

Keep the implementation import lazy inside the Click command. The subset remains a required CLI input because HPA-328 needs it; the coordinator, not the CLI, owns the rule that MuScriptor receives `None`.

Catch `ComparisonIntegrityError`, `OSError`, `TypeError`, and `ValueError`; write the message to stderr, emit one canonical failure JSON object with `exit_code: 2`, then exit `2`.

- [ ] **Step 4: Add the fatal CLI test**

```python
def test_publish_paired_comparisons_returns_exit_2_for_integrity_failure(...):
    # coordinator fake raises ComparisonIntegrityError("model_lock_sha256 mismatch")
    result = runner.invoke(...)
    assert result.exit_code == 2
    assert "model_lock_sha256 mismatch" in result.stderr
    assert json.loads(result.stdout)["exit_code"] == 2
```

- [ ] **Step 5: Run CLI and coordinator tests**

```bash
uv run pytest \
  tests/benchmark/test_cross_comparison.py \
  tests/test_cli_benchmark.py -q
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/cli/benchmark.py tests/test_cli_benchmark.py
git commit -m "feat: add paired comparison publication CLI"
```

---

### Task 5: Prove acceptance, then publish real evidence when operational inputs exist

**Files:**
- Modify only if a failing acceptance test reveals a concrete defect in the files already listed above.

**Interfaces:**
- Consumes: complete HPA-562 implementation from Tasks 1–4.
- Produces: repository-wide verification plus, when real upstream evidence is available, one production HPA-562 publication bundle.

- [ ] **Step 1: Run all pairwise and HPA-562 comparison suites together**

```bash
uv run pytest \
  tests/benchmark/test_muscriptor_comparison.py \
  tests/benchmark/test_separation_comparison.py \
  tests/benchmark/test_idm_comparison.py \
  tests/benchmark/test_cross_comparison.py \
  tests/test_cli_benchmark.py -q
```

Expected: PASS. This includes the real-driver cross-summary integration test and does not depend on production Hugging Face access.

- [ ] **Step 2: Run the repository regression suite**

```bash
uv run pytest -q
```

Expected: PASS.

- [ ] **Step 3: Run CI-equivalent static verification**

```bash
uv run ruff check .
uv run ruff format --check src tests
uv run pylint --errors-only --disable=E1120,E0401 --jobs=1 src
git diff --check origin/main...HEAD
```

Expected: all commands succeed.

- [ ] **Step 4: Execute the production publication only when the real evidence paths are present**

```bash
uv run crux benchmark publish-paired-comparisons \
  --oaf-run "$OAF_RUN" \
  --muscriptor-run "$MUSCRIPTOR_RUN" \
  --separation-run "$SEPARATION_RUN" \
  --idm-run "$IDM_RUN" \
  --manifest "$REFERENCE_MANIFEST" \
  --timing-manifest "$TIMING_MANIFEST" \
  --subset-manifest "$REVIEWED_SUBSET_MANIFEST" \
  --separation-cache-dir "$SEPARATION_CACHE_DIR" \
  --output-dir "$HPA562_OUTPUT_DIR"
```

Expected when all real HPA-627/HPA-328/HPA-396 evidence exists: exit `0` and a complete `crux.paired-benchmark-publication/v1` bundle.

If any real input is unavailable, do not substitute fixture/synthetic evidence and do not block merging the implementation solely on Hugging Face access. Leave HPA-562 In Progress and record the exact missing production artifact in the existing PR/Linear issue.

- [ ] **Step 5: Verify the production bundle before closing HPA-562**

```bash
python - <<'PY'
import csv
import json
import os
from pathlib import Path

root = Path(os.environ["HPA562_OUTPUT_DIR"])
summary = json.loads((root / "summary.json").read_text())
assert summary["schema"] == "crux.paired-benchmark-publication/v1"
assert summary["identity"]["oaf_model_lock_sha256"]
with (root / "headline_matrix.csv").open(newline="", encoding="utf-8") as handle:
    rows = list(csv.DictReader(handle))
assert len(rows) == 6
assert {row["scope"] for row in rows} == {"broad_full_mix", "reviewed_pilot"}
for rel in (
    "comparisons/oaf-muscriptor/summary.json",
    "comparisons/oaf-separation/summary.json",
    "comparisons/oaf-idm/summary.json",
):
    assert (root / rel).is_file(), rel
print(root)
PY
```

Expected: prints the production bundle root and exits `0`.

- [ ] **Step 6: Commit any final acceptance-only corrections on the same PR**

If Steps 1–5 require no code correction, do not create an empty commit. Otherwise stage only the concrete fixes and tests, then:

```bash
git commit -m "fix: close paired comparison acceptance gaps"
```

---

## Self-review

- Spec coverage: all HPA-562 pair types stay owned by the existing three comparison drivers; HPA-562 adds only cross-summary validation, the scoped matrix, and the top-level publication index.
- Scope routing: MuScriptor is always broad (`subset_manifest_path=None`); the reviewed subset is supplied only to separation, while IDM carries its pilot lineage in its own run.
- Matrix mapping: all six rows use an explicit summary/model-key source table; IDM `models["oaf"]` is never a headline source.
- Identity: reference/timing/taxonomy/lane/scoring and the OaF model lock are checked across publications; HTDemucs view IDs are cross-checked; same-input hashes remain enforced inside the existing same-view comparators.
- Real-shape evidence: one HPA-562 integration fixture executes all three actual compare drivers rather than validating only fake summary dictionaries.
- Population labeling: broad full-mix and reviewed-pilot rows are distinct by construction.
- Determinism: top-level persisted paths are relative, existing SHA helpers are reused, and duplicate publications in different roots must be byte-identical.
- Operational truthfulness: synthetic fixtures prove implementation behavior but cannot complete the production benchmark gate; implementation is not held hostage to gated Hugging Face access.
- PR shape: this planning update and all implementation tasks remain on the same HPA-562 draft PR.