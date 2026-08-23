# HPA-562 Paired Benchmark Publication Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Publish one deterministic HPA-562 bundle that composes the existing OaF/MuScriptor, OaF/separation, and OaF/IDM pairwise comparisons and exposes a clearly scoped model × input-view population matrix.

**Architecture:** Keep the three existing pairwise comparison drivers authoritative. Add one concrete `cross_comparison.py` coordinator that stages those outputs, validates their shared identities, and publishes only a top-level index/matrix. No new scorer, generic comparison engine, database, or experiment framework.

**Tech Stack:** Python 3.12, dataclasses, `pathlib`, existing strict/canonical JSON helpers, Click, pytest.

**Spec:** `docs/superpowers/specs/2026-08-22-hpa-562-paired-comparisons-design.md`

## Global Constraints

- Keep all HPA-562 planning and implementation on the existing `agent/hpa-562-paired-comparisons` branch and the same draft PR; do not open a second HPA-562 PR.
- Reuse `compare_oaf_muscriptor()`, `compare_oaf_separation()`, and `compare_oaf_idm()`; do not rebuild pair joins or rescore events.
- Same-input model comparisons keep exact input-audio hash matching in their existing drivers; cross-input OaF separation keeps HPA-328's source-hash + fixed-view contract.
- Full-corpus and reviewed-pilot populations remain explicitly distinct.
- Persist only relative paths inside the HPA-562 bundle.
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

`cross_comparison.py` owns only bundle orchestration, cross-summary validation, matrix/index rendering, and final publication. Existing comparison modules continue to own all pair construction and domain-specific integrity checks.

---

### Task 1: Complete the published pairwise identity needed by HPA-562

**Files:**
- Modify: `src/benchmark/muscriptor_comparison.py`
- Modify: `src/benchmark/separation_comparison.py`
- Test: `tests/benchmark/test_muscriptor_comparison.py`
- Test: `tests/benchmark/test_separation_comparison.py`

**Interfaces:**
- Consumes: `TAXONOMY_VERSION`, `DTX_LANE_MAP_VERSION`, `SCORING_VERSION`.
- Produces: nested comparison `summary.json` identity objects that all expose `taxonomy_version`, `lane_map_version`, and `scoring_version` in addition to their existing reference/timing/input identity.

- [ ] **Step 1: Add failing assertions for MuScriptor comparison identity**

Extend the existing successful comparison test so its parsed `summary.json` must include:

```python
assert summary["identity"]["taxonomy_version"] == TAXONOMY_VERSION
assert summary["identity"]["lane_map_version"] == DTX_LANE_MAP_VERSION
assert summary["identity"]["scoring_version"] == SCORING_VERSION
```

- [ ] **Step 2: Run the focused MuScriptor comparison test and verify it fails**

Run:

```bash
uv run pytest tests/benchmark/test_muscriptor_comparison.py -q
```

Expected: at least the new identity assertion fails because taxonomy/lane-map identity is not currently persisted by the comparison summary.

- [ ] **Step 3: Persist the complete MuScriptor comparison identity**

In the final `comparison_summary(...)` call inside `compare_oaf_muscriptor()`, pass an explicit identity mapping using the already validated manifests/run identity:

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

Keep all existing pair validation and report filenames unchanged.

- [ ] **Step 4: Add failing assertions for separation comparison identity**

In `tests/benchmark/test_separation_comparison.py`, extend the successful publication assertion:

```python
assert summary["identity"]["taxonomy_version"] == TAXONOMY_VERSION
assert summary["identity"]["lane_map_version"] == DTX_LANE_MAP_VERSION
assert summary["identity"]["scoring_version"] == SCORING_VERSION
```

- [ ] **Step 5: Run the focused separation test and verify it fails**

Run:

```bash
uv run pytest tests/benchmark/test_separation_comparison.py -q
```

Expected: the new taxonomy/lane assertions fail against the current `_comparison_identity()` output.

- [ ] **Step 6: Complete `_comparison_identity()`**

Add the two constants without changing the HPA-328 comparison topology:

```python
return {
    # existing reference/timing/subset/input-view fields...
    "taxonomy_version": TAXONOMY_VERSION,
    "lane_map_version": DTX_LANE_MAP_VERSION,
    "scoring_version": SCORING_VERSION,
}
```

- [ ] **Step 7: Run both focused suites**

Run:

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

### Task 2: Add the concrete HPA-562 coordinator and cross-summary validation

**Files:**
- Create: `src/benchmark/cross_comparison.py`
- Create: `tests/benchmark/test_cross_comparison.py`

**Interfaces:**
- Consumes:
  - `compare_oaf_muscriptor(ComparisonRequest) -> ComparisonOutcome`
  - `compare_oaf_separation(SeparationComparisonRequest) -> SeparationComparisonOutcome`
  - `compare_oaf_idm(IdmComparisonRequest) -> Path`
  - `strict_json_loads(..., require_canonical=True)`
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

Add tests that pin Path-only inputs, optional `separation_cache_dir`, non-empty comparison maps, and non-negative pair counts. Example:

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

- [ ] **Step 3: Implement the public types and staged pairwise calls**

Use fixed comparison IDs and directories:

```python
_COMPARISON_DIRS = {
    "oaf_muscriptor_full_mix": Path("comparisons/oaf-muscriptor"),
    "oaf_separation_pilot": Path("comparisons/oaf-separation"),
    "oaf_idm_htdemucs": Path("comparisons/oaf-idm"),
}
```

`publish_cross_comparisons()` must reject an existing final output directory, create one sibling temporary staging root, and invoke the three existing drivers using the request's raw run/manifest evidence. Do not parse prediction artifacts or call `score_cohort()`.

- [ ] **Step 4: Add tests proving the coordinator delegates instead of reimplementing**

Monkeypatch the three driver call sites in `cross_comparison.py` with fakes that record their request objects and write canonical minimal `summary.json` fixtures. Assert each fake is called exactly once and receives the supplied run/manifest paths.

Use a helper with explicit shared identity:

```python
def _identity() -> dict[str, object]:
    return {
        "reference_manifest_sha256": "a" * 64,
        "reference_manifest_version": "hpa324-v1",
        "reference_timing_manifest_sha256": "b" * 64,
        "reference_timing_version": "hpa323-v1",
        "taxonomy_version": TAXONOMY_VERSION,
        "lane_map_version": DTX_LANE_MAP_VERSION,
        "scoring_version": SCORING_VERSION,
    }
```

- [ ] **Step 5: Implement canonical summary loading and shared identity validation**

Add private helpers:

```python
def _read_summary(path: Path, *, expected_schema: str) -> Mapping[str, object]:
    ...


def _shared_identity(summaries: Mapping[str, Mapping[str, object]]) -> dict[str, object]:
    ...
```

`_shared_identity()` compares exactly:

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

Any missing or unequal field raises `ComparisonIntegrityError` naming the field.

- [ ] **Step 6: Add one parametrized mismatch test**

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
    ...
    with pytest.raises(ComparisonIntegrityError, match=field):
        publish_cross_comparisons(request)
    assert not request.output_dir.exists()
```

- [ ] **Step 7: Run the new focused suite**

```bash
uv run pytest tests/benchmark/test_cross_comparison.py -q
```

Expected: PASS.

- [ ] **Step 8: Commit**

```bash
git add src/benchmark/cross_comparison.py tests/benchmark/test_cross_comparison.py
git commit -m "feat: compose paired benchmark comparisons"
```

---

### Task 3: Publish the scoped headline matrix and top-level index

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

- [ ] **Step 1: Write the six-row matrix test**

Pin exact row order and scopes:

```python
assert [(row["scope"], row["model"]) for row in rows] == [
    ("broad_full_mix", "oaf"),
    ("broad_full_mix", "muscriptor"),
    ("reviewed_pilot", "oaf"),
    ("reviewed_pilot", "oaf"),
    ("reviewed_pilot", "oaf"),
    ("reviewed_pilot", "idm"),
]
assert [row["input_view_id"] for row in rows[2:]] == [
    OAF_FULL_MIX_INPUT_VIEW_ID,
    SPLEETER_INPUT_VIEW_ID,
    HTDEMUCS_INPUT_VIEW_ID,
    IDM_STEM_INPUT_VIEW_ID,
]
```

The fixture populations must use different counts for broad and pilot OaF so the test detects accidental merging.

- [ ] **Step 2: Run the matrix test and verify it fails**

```bash
uv run pytest tests/benchmark/test_cross_comparison.py -q
```

Expected: FAIL because no headline matrix is written yet.

- [ ] **Step 3: Implement deterministic matrix rendering**

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

Build rows only from validated nested `models[*].population` objects. `comparison_ids` is a stable comma-separated token, not a filesystem path.

- [ ] **Step 4: Add top-level summary/hash assertions**

Require:

```python
assert summary["schema"] == PAIRED_BENCHMARK_PUBLICATION_SCHEMA
assert summary["headline_matrix"]["path"] == "headline_matrix.csv"
assert summary["headline_matrix"]["sha256"] == sha256_hex(
    (out / "headline_matrix.csv").read_bytes()
)
assert summary["comparisons"]["oaf_muscriptor_full_mix"]["path"] == (
    "comparisons/oaf-muscriptor"
)
```

For each nested comparison, hash the canonical `summary.json` plus every nested `paired_per_song.csv` / `paired_per_class.csv` file that exists under that comparison root. Store only relative paths and SHA-256 values.

- [ ] **Step 5: Implement top-level JSON and Markdown writers**

`summary.json` must include the shared identity, supplied reviewed-subset identity copied from the validated separation summary, comparison index, and headline matrix hash.

`summary.md` must render the matrix and include this exact meaning in prose:

```text
Broad full-mix and reviewed-pilot rows have different populations and must not be ranked as one leaderboard.
```

- [ ] **Step 6: Add output-root independence test**

Publish identical fake evidence twice to two different non-existing output directories and assert:

```python
for name in ("summary.json", "summary.md", "headline_matrix.csv"):
    assert (left / name).read_bytes() == (right / name).read_bytes()
```

- [ ] **Step 7: Run the focused suite**

```bash
uv run pytest tests/benchmark/test_cross_comparison.py -q
```

Expected: PASS.

- [ ] **Step 8: Commit**

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

Use `CliRunner` and monkeypatch the coordinator. Pin the command name and canonical output fields:

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

Keep the implementation import lazy inside the Click command, matching the existing benchmark CLI pattern.

Catch `ComparisonIntegrityError`, `OSError`, `TypeError`, and `ValueError`; write the message to stderr, emit one canonical failure JSON object with `exit_code: 2`, then exit `2`.

- [ ] **Step 4: Add the fatal CLI test**

```python
def test_publish_paired_comparisons_returns_exit_2_for_integrity_failure(...):
    monkeypatch.setattr(..., side_effect=ComparisonIntegrityError("taxonomy_version mismatch"))
    result = runner.invoke(...)
    assert result.exit_code == 2
    assert "taxonomy_version mismatch" in result.stderr
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

Expected: PASS.

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

Use explicit environment variables so the command stays copyable without inventing repository paths:

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

If any real input is unavailable, do not substitute fixture/synthetic evidence. Leave HPA-562 In Progress and record the exact missing upstream artifact in the existing draft PR/Linear issue.

- [ ] **Step 5: Verify the production bundle before closing HPA-562**

```bash
python - <<'PY'
import json, os
from pathlib import Path
root = Path(os.environ["HPA562_OUTPUT_DIR"])
summary = json.loads((root / "summary.json").read_text())
assert summary["schema"] == "crux.paired-benchmark-publication/v1"
assert (root / "headline_matrix.csv").is_file()
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

If Steps 1–5 require no code correction, do not create an empty commit. Otherwise commit only the concrete fixes and their regression tests with:

```bash
git commit -am "fix: close paired comparison acceptance gaps"
```

---

## Self-review

- Spec coverage: all HPA-562 pair types stay owned by the existing three comparison drivers; HPA-562 adds only cross-summary identity validation, the scoped matrix, and the top-level publication index.
- Scope: one implementation coordinator and one CLI command; no generic framework or new scorer.
- Identity: reference/timing/taxonomy/lane/scoring are checked across publications; same-input hashes remain enforced inside the existing same-view comparators.
- Population labeling: broad full-mix and reviewed-pilot rows are distinct by construction.
- Determinism: top-level persisted paths are relative and duplicate publications in different roots must be byte-identical.
- Operational truthfulness: synthetic fixtures can prove code behavior but cannot complete the production benchmark gate.
- PR shape: this planning commit and all implementation tasks remain on the same HPA-562 draft PR.
