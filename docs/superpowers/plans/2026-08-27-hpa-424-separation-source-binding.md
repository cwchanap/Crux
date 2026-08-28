# HPA-424 Separation Pilot Authoritative Source Binding Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the HPA-328 separation pilot resolve reviewed-subset audio from the already-validated authoritative HPA-324 source rows instead of the slim HPA-327 rows, with deterministic coverage of the source-inventory contract that originally failed.

**Architecture:** Reuse `_validate_subset_population()` as the single HPA-327 -> HPA-324 authority boundary and return only the HPA-324 rows that passed its existing identity checks. Thread that in-memory binding into `_resolve_pilot_sources()`, align the flow with the existing IDM full-mix source lookup, and leave `corpus_cache.py`, schemas, execution order, resolver error propagation, and persisted pilot semantics unchanged.

**Tech Stack:** Python 3.13, existing benchmark manifest loaders/cache resolver, pytest, Ruff, Pylint.

**Spec:** `docs/superpowers/specs/2026-08-27-hpa-424-separation-source-binding-design.md`

## Global Constraints

- Keep HPA-424 in one PR; planning docs, fixture correction, tests, and production fix stay on the same branch/PR.
- Do not implement until the production precondition proves the real HPA-324 rows used by HPA-328 actually contain and can resolve each reviewed member's `source_audio_key`.
- If that precondition fails because the authoritative inventory is chart-scoped or otherwise lacks the audio remote, stop HPA-424 and move the defect upstream; do not compensate in HPA-327 or `corpus_cache.py`.
- `_validate_subset_population()` remains the only HPA-327 -> HPA-324 membership/identity validator.
- Reuse its existing `simfile_id`, selected-chart/audio identity, and `source_row_sha256` checks verbatim; do not duplicate them in `_resolve_pilot_sources()`.
- Return only validated subset members, not the complete HPA-324 population.
- `resolve_source_audio()` must receive the authoritative HPA-324 source row and `_source_audio_kwargs()` derived from that same row.
- Preserve `_resolve_pilot_sources()` result-type validation, `source_audio_id` / `source_audio_sha256` comparison, error strings, row updates, and resolver `ValueError` propagation.
- Keep source resolution before separator attestation and OaF execution.
- Do not modify HPA-324/HPA-327 schemas, `src/benchmark/corpus_cache.py`, CLI contracts, separator locks, scoring, comparison, reports, or handoff schemas.
- Add no dependency, reusable manifest-join framework, compatibility path, network-backed automated test, or second verified-cache harness.
- Make the synthetic reviewed-subset reference inventory faithful enough to contain the selected source audio object; cover the original mapping failure directly with `_remote_from_source_mapping()`.
- `_task6_seams()` is shared by six suites; strengthening it is intentionally a cross-suite regression tripwire.
- Use `ruff format --check`, matching the repository pre-commit hook; do not rely on Black as the formatting gate.
- Run the real HPA-328 pilot once on the final committed implementation tree as corroborating production evidence. Do not rerun an unchanged tree merely because tracking metadata changed.

## File Structure

| File | Responsibility in HPA-424 |
| --- | --- |
| `src/benchmark/separation_pilot.py` | Return the validated subset-only HPA-324 binding and feed it into authoritative source resolution. |
| `tests/benchmark/reviewed_subset_fixtures.py` | Make synthetic HPA-323/HPA-324 source inventories include both selected chart and selected source audio remotes. |
| `tests/benchmark/test_separation_pilot.py` | Prove a loaded HPA-324 row resolves its audio remote and make the shared Task 6 fake require real audio membership. |

No new files are required.

---

### Task 0: Prove the real HPA-324 inventory assumption before touching code

**Files:** read workstation-local production artifacts only; no repository change.

**Interfaces:**
- `load_reference_set_manifest(Path) -> LoadedReferenceSetManifest`
- `load_reviewed_subset_manifest(Path) -> LoadedReviewedSubsetManifest`
- `CacheIndexStore.load(Path) -> CacheIndexStore`
- `resolve_source_audio(..., load_body=False) -> ResolvedSourceAudio`
- `_source_audio_kwargs(source_row) -> dict[str, str | None]`

- [ ] **Step 1: Resolve the exact production paths from the failed HPA-328 workspace**

Set:

```bash
export REFERENCE_MANIFEST=...   # published HPA-324 manifest used by the failed pilot
export SUBSET_MANIFEST=...      # published 30-row HPA-327 manifest
export CACHE_DIR=...            # verified HPA-321 cache used by the failed pilot
```

The known parent OaF run is `oaf-149faa97328e20eb`; this preflight does not run OaF or either separator.

- [ ] **Step 2: Resolve every reviewed member from its authoritative row**

Run one read-only Python preflight:

```bash
uv run python - <<'PY'
import os
from pathlib import Path

from src.benchmark.corpus_cache import CacheIndexStore, resolve_source_audio
from src.benchmark.reference_set_manifest import load_reference_set_manifest
from src.benchmark.reviewed_subset import load_reviewed_subset_manifest
from src.benchmark.separation_pilot import _source_audio_kwargs

reference = load_reference_set_manifest(Path(os.environ["REFERENCE_MANIFEST"]))
subset = load_reviewed_subset_manifest(Path(os.environ["SUBSET_MANIFEST"]))
cache_dir = Path(os.environ["CACHE_DIR"])
cache_index = CacheIndexStore.load(cache_dir)
reference_rows = {row.view.simfile_id: row.source_row for row in reference.rows}

for reviewed in subset.rows:
    source_row = reference_rows[reviewed.view.simfile_id]
    key = source_row["source_audio_key"]
    objects = source_row["objects"]
    assert isinstance(key, str) and key
    assert isinstance(objects, list)
    assert any(isinstance(obj, dict) and obj.get("key") == key for obj in objects), (
        reviewed.view.simfile_id,
        key,
    )
    resolved = resolve_source_audio(
        source_row,
        cache_dir,
        cache_index,
        **_source_audio_kwargs(source_row),
        load_body=False,
    )
    assert resolved.source_audio_id == key
    assert resolved.source_audio_sha256 == source_row["source_audio_content_hash"]

print(f"validated {len(subset.rows)} reviewed source rows")
PY
```

Expected: `validated 30 reviewed source rows`.

- [ ] **Step 3: Gate the design on the result**

If any row lacks `source_audio_key` in `objects`, or a same-row remote cannot be resolved from the existing cache/index, stop. Record the exact failing `simfile_id`/error on HPA-424 and PR #31 and re-scope the defect to the HPA-323/HPA-324 lineage. Do not proceed to Task 1.

If all reviewed members pass, record the preflight result on HPA-424/PR #31 and continue. This retires the plan's highest-risk assumption before code changes.

---

### Task 1: Make the synthetic HPA-324 fixture prove the selected-audio mapping contract

**Files:**
- Modify: `tests/benchmark/reviewed_subset_fixtures.py:95-135`
- Modify: `tests/benchmark/test_separation_pilot.py:1-40` and add one focused test near fixture/preflight tests.

**Interfaces:**
- Consumes: `build_reviewed_subset_reference_fixture()`, `load_reference_set_manifest()`, and existing `src.benchmark.corpus_cache._remote_from_source_mapping()`.
- Produces: synthetic HPA-324 rows whose `objects` include their `source_audio_key`, plus one deterministic test for the exact mapping path that raised in production.

- [ ] **Step 1: Extend the fixture inventory with a valid source-audio remote**

The current fixture rewrites every golden remote to `selected_chart_key`, leaving `source_audio_key` absent from `objects`. Replace that chart-only construction with two remotes per row.

Use the existing golden object as the field template and construct the audio remote with the row's source-audio identity:

```python
source_audio_key = f"{simfile_id}/bgm.wav"
source_audio_hash = str(ready_row["source_audio_content_hash"])
objects = ready_row["objects"]
assert isinstance(objects, list) and objects
remote_template = objects[0]
assert isinstance(remote_template, dict)

chart_remote = {**remote_template, "key": selected_chart_key}
audio_remote = {
    **remote_template,
    "key": source_audio_key,
    "content_type": "audio/wav",
    "etag": f'"audio-{simfile_id}"',
    "size": 88244,
    "sha256": source_audio_hash,
    "cache_path": f"sha256/{source_audio_hash[:2]}/{source_audio_hash}",
}

row["objects"] = [chart_remote, audio_remote]
row["source_audio_key"] = source_audio_key
```

Keep `source_audio_content_hash` unchanged so the generated native reference events and HPA-324 row continue sharing one source identity.

- [ ] **Step 2: Add direct coverage over a loaded HPA-324 row**

Import the existing mapping helper and lighter reference fixture:

```python
from src.benchmark.corpus_cache import _remote_from_source_mapping
from tests.benchmark.reviewed_subset_fixtures import (
    build_reviewed_subset_oaf_fixture,
    build_reviewed_subset_reference_fixture,
)
```

Add:

```python
def test_reviewed_subset_reference_fixture_contains_resolvable_source_audio_remote(
    tmp_path: Path,
) -> None:
    fixture = build_reviewed_subset_reference_fixture(tmp_path, eligible_count=20)
    reference = load_reference_set_manifest(fixture.reference_manifest_path)
    source_row = reference.rows[0].source_row
    source_audio_key = source_row["source_audio_key"]
    assert isinstance(source_audio_key, str)

    remote = _remote_from_source_mapping(
        source_row,
        source_audio_key=source_audio_key,
    )

    assert remote.key == source_audio_key
    assert remote.sha256 == source_row["source_audio_content_hash"]
```

This test intentionally targets `_remote_from_source_mapping()` rather than recreating a second cache-body fixture. `test_oaf_corpus_run.py` already proves verified body resolution once a valid remote exists.

- [ ] **Step 3: Run the mapping-contract test**

Run:

```bash
uv run pytest \
  tests/benchmark/test_separation_pilot.py::test_reviewed_subset_reference_fixture_contains_resolvable_source_audio_remote \
  -q
```

Expected: PASS with only the fixture/test change. The loaded HPA-324 row now models the source-audio inventory contract used by production callers.

---

### Task 2: Turn the shared fake into a red tripwire, then fix the caller wiring

**Files:**
- Modify: `tests/benchmark/test_separation_pilot.py:128-176, 334-385`
- Modify: `src/benchmark/separation_pilot.py:697-716, 1245-1278, 1955-1980`
- Verify shared seam consumers:
  - `tests/benchmark/test_separation_pilot_acceptance.py`
  - `tests/benchmark/test_separation_comparison.py`
  - `tests/benchmark/test_cross_comparison.py`
  - `tests/benchmark/test_separation_handoff.py`
  - `tests/test_cli_benchmark.py`

**Interfaces:**

```python
def _validate_subset_population(
    subset: LoadedReviewedSubsetManifest,
    reference: LoadedReferenceSetManifest,
) -> dict[int, Mapping[str, object]]:
    ...


def _resolve_pilot_sources(
    request: OafSeparationPilotRequest,
    reference_rows: Mapping[int, Mapping[str, object]],
    rows: tuple[dict[str, object], ...],
) -> dict[int, ResolvedSourceAudio]:
    ...
```

- Consumes: the already-loaded HPA-327/HPA-324 manifests, `_source_row_sha256()`, `_source_audio_kwargs()`, `CacheIndexStore`, and `resolve_source_audio()`.
- Produces: the same resolved-source map and persisted pilot behavior as today; only the owner of the mapping passed into source resolution changes.

- [ ] **Step 1: Strengthen `_task6_seams()` with the actual audio-membership contract**

Add a call bucket for the source mappings, then require the mapping to contain the selected source audio object:

```python
calls: dict[str, list[object]] = {
    "resolve": [],
    "resolve_source_rows": [],
    "separate": [],
    "materialize": [],
    "backend": [],
    "close": [],
    "transcribe": [],
    "score": [],
    "attest": [],
    "revalidate": [],
    "events": [],
}
```

Inside fake `resolve()`:

```python
calls["resolve"].append(kwargs.get("load_body"))
assert isinstance(source, Mapping)
calls["resolve_source_rows"].append(source)

objects = source.get("objects")
assert isinstance(objects, list)
source_audio_key = source["source_audio_key"]
assert isinstance(source_audio_key, str)
assert any(
    isinstance(obj, Mapping) and obj.get("key") == source_audio_key
    for obj in objects
)
assert isinstance(source.get("source_endpoint_sha256"), str)
assert isinstance(source.get("source_bucket"), str)
```

Keep the existing returned `ResolvedSourceAudio` unchanged.

In `test_task6_infers_only_the_two_derived_views_after_resolving_membership()`, add:

```python
assert len(calls["resolve_source_rows"]) == 20
assert all(
    isinstance(source, Mapping)
    and isinstance(source.get("objects"), list)
    and isinstance(source.get("source_audio_key"), str)
    and any(
        isinstance(obj, Mapping) and obj.get("key") == source["source_audio_key"]
        for obj in source["objects"]
    )
    for source in calls["resolve_source_rows"]
)
```

Do not let any importer accept the old HPA-327 shape.

- [ ] **Step 2: Prove all six importer suites are red before the production fix**

Run:

```bash
uv run pytest \
  tests/benchmark/test_separation_pilot.py \
  tests/benchmark/test_separation_pilot_acceptance.py \
  tests/benchmark/test_separation_comparison.py \
  tests/benchmark/test_cross_comparison.py \
  tests/benchmark/test_separation_handoff.py \
  tests/test_cli_benchmark.py \
  -q
```

Expected before changing production wiring: failures reach the shared fake because `_resolve_pilot_sources()` still passes the HPA-327 reviewed row, which has no authoritative `objects` inventory.

- [ ] **Step 3: Return only the HPA-324 rows that passed existing subset validation**

Preserve the current validation and error strings. Add only the validated result collection:

```python
def _validate_subset_population(
    subset: LoadedReviewedSubsetManifest,
    reference: LoadedReferenceSetManifest,
) -> dict[int, Mapping[str, object]]:
    """Bind every HPA-327 row to the corresponding HPA-324 reference row."""
    reference_rows = {loaded.view.simfile_id: loaded.source_row for loaded in reference.rows}
    validated: dict[int, Mapping[str, object]] = {}
    for loaded in subset.rows:
        reference_row = reference_rows.get(loaded.view.simfile_id)
        if reference_row is None:
            raise SeparationRunError("reviewed subset member is absent from reference manifest")
        for field in (
            "selected_chart_key",
            "selected_chart_content_hash",
            "source_audio_key",
            "source_audio_content_hash",
        ):
            if loaded.source_row[field] != reference_row[field]:
                raise SeparationRunError("reviewed subset member reference identity is invalid")
        if loaded.source_row["source_row_sha256"] != _source_row_sha256(reference_row):
            raise SeparationRunError("reviewed subset source row identity is invalid")
        validated[loaded.view.simfile_id] = reference_row
    return validated
```

At the existing preflight call site:

```python
reference_rows = _validate_subset_population(subset, reference)
```

This map contains only fixed reviewed members whose HPA-324 identity has already passed validation.

- [ ] **Step 4: Change only the source-row owner used by `_resolve_pilot_sources()`**

Replace the `subset` argument and `loaded_by_id` lookup with the validated HPA-324 map. Preserve every other current check, error string, and side effect:

```python
def _resolve_pilot_sources(
    request: OafSeparationPilotRequest,
    reference_rows: Mapping[int, Mapping[str, object]],
    rows: tuple[dict[str, object], ...],
) -> dict[int, ResolvedSourceAudio]:
    """Resolve every fixed member's authoritative source before execution."""
    cache_index = CacheIndexStore.load(request.cache_dir)
    sources: dict[int, ResolvedSourceAudio] = {}
    for row in rows:
        simfile_id = row["simfile_id"]
        if not isinstance(simfile_id, int):
            raise SeparationRunError("pilot row simfile_id is invalid")
        source_row = reference_rows.get(simfile_id)
        if source_row is None:
            raise SeparationRunError("pilot row source member is unavailable")
        source = resolve_source_audio(
            source_row,
            request.cache_dir,
            cache_index,
            **_source_audio_kwargs(source_row),
            load_body=False,
        )
        if not isinstance(source, ResolvedSourceAudio):
            raise SeparationRunError("source resolver returned an invalid result")
        if source.source_audio_id != row.get(
            "source_audio_id"
        ) or source.source_audio_sha256 != row.get("source_audio_sha256"):
            raise SeparationRunError("resolved source identity does not match fixed membership")
        row["source_audio_id"] = source.source_audio_id
        row["source_audio_sha256"] = source.source_audio_sha256
        row["source_duration_sec"] = source.duration_sec
        sources[simfile_id] = source
    return sources
```

Update only the call argument:

```python
sources = _resolve_pilot_sources(request, reference_rows, rows)
```

Do not add exception wrapping, a second hash check, new failure codes, stricter typing, different membership fields, or different row mutation. This matches the existing authoritative-row pattern in `src/benchmark/idm_pilot_run.py`.

- [ ] **Step 5: Prove all six importer suites are green**

Run the same importer set:

```bash
uv run pytest \
  tests/benchmark/test_separation_pilot.py \
  tests/benchmark/test_separation_pilot_acceptance.py \
  tests/benchmark/test_separation_comparison.py \
  tests/benchmark/test_cross_comparison.py \
  tests/benchmark/test_separation_handoff.py \
  tests/test_cli_benchmark.py \
  -q
```

Expected: PASS. The fake sees an HPA-324 row whose `objects` actually contains `source_audio_key`, while persisted pilot/comparison/handoff/CLI semantics remain unchanged.

---

### Task 3: Verify, commit, and corroborate once on the final tree

**Files:** all three implementation/test files above; no additional production files.

- [ ] **Step 1: Run deterministic mapping and quality gates**

Run:

```bash
uv run pytest \
  tests/benchmark/test_separation_pilot.py::test_reviewed_subset_reference_fixture_contains_resolvable_source_audio_remote \
  -q

uv run ruff check \
  src/benchmark/separation_pilot.py \
  tests/benchmark/reviewed_subset_fixtures.py \
  tests/benchmark/test_separation_pilot.py

uv run ruff format --check \
  src/benchmark/separation_pilot.py \
  tests/benchmark/reviewed_subset_fixtures.py \
  tests/benchmark/test_separation_pilot.py

uv run pylint --errors-only --disable=E1120 \
  src/benchmark/separation_pilot.py \
  tests/benchmark/reviewed_subset_fixtures.py \
  tests/benchmark/test_separation_pilot.py

git diff --check main...HEAD
```

Expected: PASS.

- [ ] **Step 2: Verify the implementation diff stays narrow**

Run:

```bash
git diff --name-only main...HEAD
```

Expected PR files:

```text
docs/superpowers/specs/2026-08-27-hpa-424-separation-source-binding-design.md
docs/superpowers/plans/2026-08-27-hpa-424-separation-source-binding.md
src/benchmark/separation_pilot.py
tests/benchmark/reviewed_subset_fixtures.py
tests/benchmark/test_separation_pilot.py
```

There must be no change to `src/benchmark/corpus_cache.py`, manifest schema goldens, CLI contracts, separator fixtures, or persisted run schemas.

- [ ] **Step 3: Commit the implementation on the existing HPA-424 PR**

Run:

```bash
git add \
  src/benchmark/separation_pilot.py \
  tests/benchmark/reviewed_subset_fixtures.py \
  tests/benchmark/test_separation_pilot.py

git commit -m "fix: bind separation sources to reference rows"
```

No second HPA-424 PR.

- [ ] **Step 4: Run the real HPA-328 pilot once on that committed tree**

On the workstation that produced the failed HPA-328 evidence, rerun the same `run-oaf-separation-pilot` invocation against:

```text
parent OaF run: oaf-149faa97328e20eb
reviewed subset: published 30-row HPA-327 manifest
reference source: matching published HPA-324 manifest
cache: the existing verified local HPA-321 cache used by Task 0
```

Required HPA-424 corroboration:

- source resolution completes for the fixed population;
- neither `source manifest does not contain an object inventory` nor `source audio key is absent from the source inventory` reappears.

If a later separator/runtime/inference problem occurs, record it separately; HPA-424 owns only the source-binding defect.

Record the exact final-tree commit, command, and source-resolution result on HPA-424 and PR #31.

Do not rerun the 30-item pilot again unless the implementation tree changes after this evidence. PR-body or Linear-comment edits do not invalidate it.

- [ ] **Step 5: Move the same PR to review**

After the committed deterministic gates and the one real rerun are recorded, update PR #31 with the evidence and mark the same PR ready for review. Keep HPA-424 open until merge.

## Self-review

- Spec coverage: Task 0 retires the production inventory assumption before implementation; Task 1 makes the synthetic HPA-324 inventory faithful and directly covers the original mapping helper; Task 2 creates the cross-suite red/green regression and applies the minimal caller fix; Task 3 matches repository formatter hooks, commits once, and runs the production corroboration once on the final tree.
- Placeholder scan: no implementation behavior is left unspecified. The only environment-specific values are the existing production artifact paths that must be resolved from the workstation which already produced the failed HPA-328 run.
- Type consistency: `_validate_subset_population()` returns `dict[int, Mapping[str, object]]` containing only validated subset members; `_resolve_pilot_sources()` consumes `Mapping[int, Mapping[str, object]]`; `run_oaf_separation_pilot()` threads that value directly between them.
- Scope check: one production helper return, one caller argument swap, one existing fixture correction, and regression coverage; no schema/cache/runtime framework change.
