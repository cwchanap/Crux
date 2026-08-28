# HPA-424 Separation Pilot Authoritative Source Binding Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the HPA-328 separation pilot resolve reviewed-subset audio from the already-validated authoritative HPA-324 source rows instead of the slim HPA-327 rows.

**Architecture:** Reuse `_validate_subset_population()` as the single HPA-327 -> HPA-324 authority boundary and return its validated source-row mapping. Thread that in-memory mapping into `_resolve_pilot_sources()` while preserving the resolver call ordering, result validation, fixed-membership checks, row mutation, manifest schemas, and persisted pilot semantics.

**Tech Stack:** Python 3.13, existing benchmark manifest loaders/cache resolver, pytest, Ruff, Black, Pylint.

**Spec:** `docs/superpowers/specs/2026-08-27-hpa-424-separation-source-binding-design.md`

## Global Constraints

- Keep HPA-424 in one PR; the planning docs and implementation stay on the same branch/PR.
- `_validate_subset_population()` remains the only HPA-327 -> HPA-324 membership/identity validator.
- Reuse its existing `simfile_id`, selected-chart/audio identity, and `source_row_sha256` checks verbatim; do not duplicate them in `_resolve_pilot_sources()`.
- `resolve_source_audio()` must receive the authoritative HPA-324 source row and `_source_audio_kwargs()` derived from that same row.
- Keep source resolution before separator attestation and OaF execution.
- Preserve `_resolve_pilot_sources()` result-type validation, `source_audio_id` / `source_audio_sha256` comparison, error strings, and row updates.
- Do not wrap `resolve_source_audio()` `ValueError`; `run_oaf_separation_pilot()` already owns fatal error mapping.
- Do not modify HPA-324/HPA-327 schemas, `src/benchmark/corpus_cache.py`, CLI contracts, separator locks, scoring, comparison, reports, or handoff schemas.
- Add no dependency, reusable manifest-join framework, compatibility path, or network-backed automated test.
- The strengthened `_task6_seams()` resolver is a shared cross-suite contract. Every suite importing it must be green before the implementation is considered verified.
- The real HPA-328 rerun is the production-error gate because the synthetic fixture cannot successfully exercise `_remote_from_source_mapping()` with its real cache inventory. Do not mark HPA-424 ready/merged/done on unit coverage alone.

## File Structure

| File | Responsibility in HPA-424 |
| --- | --- |
| `src/benchmark/separation_pilot.py` | Preserve the existing subset/reference validation, return its authoritative source-row binding, and feed that binding into source resolution. |
| `tests/benchmark/test_separation_pilot.py` | Turn the existing shared Task 6 fake resolver into a regression guard that rejects the slim HPA-327 row shape. |

No new implementation files are required.

---

### Task 1: Carry the validated HPA-324 row into source resolution

**Files:**
- Modify: `tests/benchmark/test_separation_pilot.py:128-176, 334-385`
- Modify: `src/benchmark/separation_pilot.py:686-716, 1245-1278, 1955-1980`
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

- Consumes: the already-loaded `LoadedReviewedSubsetManifest`, `LoadedReferenceSetManifest`, `_source_row_sha256()`, `_source_audio_kwargs()`, `CacheIndexStore`, and `resolve_source_audio()`.
- Produces: the same `dict[int, ResolvedSourceAudio]` currently consumed by the pilot; no persisted interface changes.

- [ ] **Step 1: Strengthen the existing shared Task 6 fake resolver so current `main` is red**

In `_task6_seams()`, add one call bucket for the source mappings passed into the resolver:

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

Strengthen the fake `resolve()` without changing its returned `ResolvedSourceAudio`:

```python
def resolve(source: object, *_args: object, **kwargs: object) -> ResolvedSourceAudio:
    calls["resolve"].append(kwargs.get("load_body"))
    assert isinstance(source, Mapping)
    calls["resolve_source_rows"].append(source)

    assert isinstance(source.get("objects"), list)
    assert isinstance(source.get("source_endpoint_sha256"), str)
    assert isinstance(source.get("source_bucket"), str)

    source_id = source["source_audio_key"]
    source_sha = source["source_audio_content_hash"]
    return ResolvedSourceAudio(
        path=tmp_path / "authoritative-source.wav",
        source_audio_id=source_id,
        source_audio_sha256=source_sha,
        duration_sec=1.0,
    )
```

Keep `test_task6_infers_only_the_two_derived_views_after_resolving_membership()` as the direct integration regression and add:

```python
assert len(calls["resolve_source_rows"]) == 20
assert all(
    isinstance(source, Mapping)
    and isinstance(source.get("objects"), list)
    and isinstance(source.get("source_endpoint_sha256"), str)
    and isinstance(source.get("source_bucket"), str)
    for source in calls["resolve_source_rows"]
)
```

Do not make the fake accept both HPA-327 and HPA-324 shapes; `_task6_seams()` is shared intentionally, so all importers should trip on the old wiring.

- [ ] **Step 2: Verify the strengthened shared seam is red on the current wiring**

Run every suite that imports `_task6_seams()`:

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

Expected before the production fix: failures reach the shared fake resolver because current `_resolve_pilot_sources()` passes the reviewed-subset row and `source.get("objects")` is not a list. Do not weaken the fake to make individual importer suites pass.

- [ ] **Step 3: Return the existing validated HPA-324 source-row binding**

Change only the return contract of `_validate_subset_population()`; preserve its existing loop and error messages exactly:

```python
def _validate_subset_population(
    subset: LoadedReviewedSubsetManifest,
    reference: LoadedReferenceSetManifest,
) -> dict[int, Mapping[str, object]]:
    """Bind every HPA-327 row to the corresponding HPA-324 reference row."""
    reference_rows = {loaded.view.simfile_id: loaded.source_row for loaded in reference.rows}
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
    return reference_rows
```

At the existing preflight call site, retain the returned mapping:

```python
reference_rows = _validate_subset_population(subset, reference)
```

Returning the complete HPA-324 map is intentional. `_subset_parent_rows()` constructs execution rows only by iterating `subset.rows`, so the current path cannot introduce an unbound reference ID.

Do not add a second membership/hash check at the resolver boundary.

- [ ] **Step 4: Change only the source-row owner used by `_resolve_pilot_sources()`**

Replace the `subset` argument and its `loaded_by_id` map with the validated HPA-324 mapping. Preserve every other current check, error string, and side effect:

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

Do not add exception wrapping, new failure codes, stricter `simfile_id` typing, different fixed-membership fields, or different row mutation. `run_oaf_separation_pilot()` already turns source-resolution `ValueError` into the existing fatal exit-2 behavior.

- [ ] **Step 5: Run every shared-seam importer after the fix**

Run the same complete importer set used for the red gate:

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

Expected after the fix: PASS. Every resolver source mapping reaching the shared fake has the HPA-324 `objects`, `source_endpoint_sha256`, and `source_bucket` fields, and all comparison/handoff/CLI consumers remain compatible with the unchanged persisted pilot contract.

- [ ] **Step 6: Run focused static checks and diff hygiene**

Run:

```bash
uv run ruff check src/benchmark/separation_pilot.py tests/benchmark/test_separation_pilot.py
uv run black --check src/benchmark/separation_pilot.py tests/benchmark/test_separation_pilot.py
uv run pylint src/benchmark/separation_pilot.py tests/benchmark/test_separation_pilot.py
git diff --check main...HEAD
git diff --name-only main...HEAD
```

Expected implementation surface in addition to this PR's two planning docs:

```text
src/benchmark/separation_pilot.py
tests/benchmark/test_separation_pilot.py
```

There must be no change to `src/benchmark/corpus_cache.py`, manifest schema goldens, CLI contracts, separator fixtures, or persisted run schemas.

- [ ] **Step 7: Pass the real production-error gate before review/merge/closure**

Automated tests intentionally keep the fake resolver. The synthetic HPA-324 fixture's `objects` inventory does not contain the source-audio key shape needed for a successful real `resolve_source_audio()` call, so these tests do not execute the production `_remote_from_source_mapping()` path that originally raised `source manifest does not contain an object inventory`.

On the workstation that produced the HPA-328 evidence, rerun the same real `run-oaf-separation-pilot` invocation against:

```text
parent OaF run: oaf-149faa97328e20eb
reviewed subset: published 30-row HPA-327 manifest
reference source: matching published HPA-324 manifest
cache: the existing verified local HPA-321 cache used by the failed run
```

Required result: the run proceeds past authoritative source resolution without:

```text
source manifest does not contain an object inventory
```

Any later separator/runtime/inference failure is separate evidence; record it without hiding or reclassifying it as HPA-424 success/failure.

Record the exact rerun command/result on HPA-424 and PR #31. If the workstation-local evidence is unavailable, state that explicitly on PR #31, keep HPA-424 open, and do not mark the PR ready, merge it, or close the ticket based on unit tests alone.

- [ ] **Step 8: Commit implementation and finish the same PR**

After Steps 5-7 pass, commit the implementation on the existing HPA-424 branch:

```bash
git add src/benchmark/separation_pilot.py tests/benchmark/test_separation_pilot.py
git commit -m "fix: bind separation sources to reference rows"
```

Re-run Steps 5-7 from the committed revision, update PR #31 with the verification evidence, then mark the same PR ready for review. Do not open a second implementation PR.

## Self-review

- Spec coverage: the single task covers validated binding reuse, authoritative row resolution, preserved result/fixed-membership semantics, every importer of the shared Task 6 seam, static hygiene, and the real production-error gate.
- Placeholder scan: there are no implementation placeholders or unspecified new types/functions. The environment-dependent production gate references the exact already-recorded run identity and requires the original failure mode to disappear before review/merge/closure.
- Type consistency: `_validate_subset_population()` returns `dict[int, Mapping[str, object]]`; `_resolve_pilot_sources()` consumes `Mapping[int, Mapping[str, object]]`; `run_oaf_separation_pilot()` threads that value directly between them.
