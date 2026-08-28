# HPA-424 Separation Pilot Authoritative Source Binding Design

## Status

Proposed for HPA-424. This is a bounded HPA-328 bug fix and should remain one implementation PR.

## Problem

A real `run-oaf-separation-pilot` execution fails before separator or OaF work begins:

```text
_resolve_pilot_sources
  -> resolve_source_audio(loaded.source_row, ...)
  -> _remote_from_source_mapping
  -> ValueError: source manifest does not contain an object inventory
```

The pilot is passing the wrong row shape to the source cache resolver.

- HPA-327 reviewed-subset rows are intentionally compact. They carry selected chart/audio identity plus `source_row_sha256`, but not the HPA-321 R2 object inventory, `source_endpoint_sha256`, or `source_bucket`.
- `resolve_source_audio()` accepts a source mapping only when that mapping contains the authoritative object inventory and endpoint/bucket identity needed to resolve the selected audio object.
- HPA-324 reference rows inherit the upstream source row through HPA-322/HPA-323, but HPA-424 must verify the real production manifest actually contains each selected `source_audio_key` before relying on that lineage.

The required HPA-324 manifest is already loaded by `run_oaf_separation_pilot()`. If its real rows contain the selected audio object, the defect is a caller-wiring bug. If they do not, the defect is upstream in the HPA-323/HPA-324 source lineage and HPA-424 must stop rather than paper over it in the pilot.

## Production precondition

Before implementation, inspect the already-published production HPA-324 manifest and HPA-321 cache used by the failed HPA-328 run.

For every HPA-327 pilot member, resolve its same-`simfile_id` HPA-324 row and require:

1. `objects` is a list;
2. one object has `key == source_audio_key`;
3. that object has a valid remote identity; and
4. `resolve_source_audio(..., load_body=False)` can resolve it through the existing verified cache/index.

This is a cheap preflight, not separator inference. If any member instead fails with `source audio key is absent from the source inventory` or an equivalent authoritative-row problem, stop HPA-424 and move the fix upstream. Do not extend HPA-327 or weaken `corpus_cache.py` to compensate.

## Existing rails to reuse

### `_validate_subset_population()` already owns the join

`src/benchmark/separation_pilot.py::_validate_subset_population()` already builds the HPA-324 source-row lookup by `simfile_id` and rejects a reviewed member when:

- the `simfile_id` is absent from the HPA-324 reference manifest;
- selected chart key/hash differs;
- source audio key/hash differs; or
- `source_row_sha256` differs from `_source_row_sha256(reference_row)`.

This is already the correct HPA-327 -> HPA-324 authority check. HPA-424 must not implement a second join or row-identity validator.

The helper should return only rows that passed these checks:

```python
validated: dict[int, Mapping[str, object]] = {}
for loaded in subset.rows:
    reference_row = reference_rows.get(loaded.view.simfile_id)
    # Existing validation remains unchanged.
    ...
    validated[loaded.view.simfile_id] = reference_row
return validated
```

This makes the returned value exactly the validated pilot binding rather than the entire HPA-324 population.

### Existing IDM code is the precedent

`src/benchmark/idm_pilot_run.py` already uses the same runtime shape for full-mix smoke inference:

```text
reference rows keyed by simfile_id
  -> select the authoritative source_row
  -> resolve_source_audio(source_row, ...)
  -> compare resolved source_audio_id/source_audio_sha256 with fixed evidence
```

HPA-424 aligns the OaF separation pilot with that existing convention rather than inventing a new source-binding pattern.

`idm_pilot_run.py::_smoke_source_kwargs()` is near-duplicate logic to `_source_audio_kwargs()`, but unifying those helpers is unrelated cleanup and remains out of scope.

### `_source_audio_kwargs()` already extracts the resolver identity

`_source_audio_kwargs()` already extracts:

- `source_audio_key`;
- `source_audio_content_hash`;
- `source_endpoint_sha256`; and
- `source_bucket`.

It works for an authoritative HPA-324 row. No new request object or resolver adapter is required.

### `resolve_source_audio()` has the correct contract

`src/benchmark/corpus_cache.py::resolve_source_audio()` and `_remote_from_source_mapping()` legitimately require the source mapping to expose the selected audio object plus endpoint/bucket identity. HPA-424 leaves this generic cache boundary unchanged.

The repo already has direct `_remote_from_source_mapping()` branch tests and end-to-end verified-cache tests for `resolve_source_audio()`. HPA-424 should reuse those rails rather than adding another cache framework.

## Design

### 1. Carry only the validated HPA-324 binding forward

Change `_validate_subset_population()` from validate-and-discard to validate-and-return:

```python
def _validate_subset_population(
    subset: LoadedReviewedSubsetManifest,
    reference: LoadedReferenceSetManifest,
) -> dict[int, Mapping[str, object]]:
    reference_rows = {
        loaded.view.simfile_id: loaded.source_row
        for loaded in reference.rows
    }
    validated: dict[int, Mapping[str, object]] = {}
    for loaded in subset.rows:
        reference_row = reference_rows.get(loaded.view.simfile_id)
        # Existing validation remains unchanged.
        ...
        validated[loaded.view.simfile_id] = reference_row
    return validated
```

The returned mapping is not a persisted model. It is only the in-memory result of validation the pilot already performs.

`run_oaf_separation_pilot()` stores that result:

```python
reference_rows = _validate_subset_population(subset, reference)
```

### 2. Resolve from the authoritative row

Change `_resolve_pilot_sources()` to consume the validated HPA-324 row mapping instead of the HPA-327 manifest:

```python
def _resolve_pilot_sources(
    request: OafSeparationPilotRequest,
    reference_rows: Mapping[int, Mapping[str, object]],
    rows: tuple[dict[str, object], ...],
) -> dict[int, ResolvedSourceAudio]:
    ...
```

For each fixed pilot row:

1. read `simfile_id`;
2. look up the already-validated HPA-324 `source_row`;
3. pass that row to `resolve_source_audio()`;
4. derive `_source_audio_kwargs()` from that same row; and
5. keep the existing result-type validation and resolved `source_audio_id` / `source_audio_sha256` comparison against fixed membership.

Keep the existing missing-map guard inside `_resolve_pilot_sources()`. With a validated-members-only map it remains a useful defensive invariant rather than relying on a full-reference-map argument.

Do not wrap resolver `ValueError`; `run_oaf_separation_pilot()` already owns fatal exit-2 mapping.

### 3. Keep execution order unchanged

The order remains:

```text
load HPA-324 + timing + HPA-327
  -> validate global manifest lineage
  -> validate/bind each HPA-327 member to HPA-324
  -> validate parent OaF run
  -> construct fixed rows
  -> resolve all authoritative source audio
  -> attest separators
  -> score/control/derived execution
```

Source resolution still completes before separator attestation or expensive inference. HPA-424 does not move the mutable snapshot boundary or change resume semantics.

## Test strategy

The current Task 6 fake hides the production failure because it reads only source key/hash from any mapping. HPA-424 closes that hole at two levels.

### Faithful HPA-324 fixture contract

`tests/benchmark/reviewed_subset_fixtures.py` currently rewrites every `objects` entry to the selected chart key, so its HPA-324 rows are chart-only even though `source_audio_key` points at `bgm.wav`.

Extend that fixture so each source row contains both:

- the selected chart remote; and
- a valid source-audio remote whose `key` and `sha256` match `source_audio_key` / `source_audio_content_hash`.

Then add one deterministic test over a loaded HPA-324 row that calls the existing `_remote_from_source_mapping()` directly and asserts it returns the selected audio remote. This exercises the exact mapping path that produced the production error without adding network access or a second cache harness.

Existing corpus-cache tests already prove that a valid `RemoteObject` plus verified cache body resolves end to end, so duplicating that machinery here is unnecessary.

### Shared Task 6 seam

Strengthen the existing `_task6_seams()` fake resolver to require not only an `objects` list but actual membership of the source audio key:

```python
objects = source.get("objects")
assert isinstance(objects, list)
assert any(
    isinstance(obj, Mapping) and obj.get("key") == source["source_audio_key"]
    for obj in objects
)
assert isinstance(source.get("source_endpoint_sha256"), str)
assert isinstance(source.get("source_bucket"), str)
```

The fake is shared by the pilot, acceptance, separation-comparison, cross-comparison, handoff, and CLI suites. All six are intentionally regression tripwires:

- on current wiring, they fail because the HPA-327 row has no authoritative inventory;
- after the fix, they pass because source resolution receives the validated HPA-324 row;
- the membership assertion prevents a chart-only list from masquerading as a valid source contract.

## Alternatives considered

### Expand HPA-327 rows with the R2 object inventory

Rejected. It duplicates upstream source authority into a reviewed-subset schema and creates schema churn for a caller-wiring bug.

### Teach `resolve_source_audio()` to understand HPA-327 manifests

Rejected. The generic cache resolver should not know how to join a benchmark review artifact back to HPA-324.

### Re-run the join independently inside `_resolve_pilot_sources()`

Rejected. `_validate_subset_population()` already owns the join and `source_row_sha256` guard. Repeating it would create two places that must stay semantically identical.

### Add a reusable manifest-join abstraction

Rejected under YAGNI. This bug has one concrete owner and the existing helper already performs the required binding.

### Add another full verified-cache fixture

Rejected. The original defect occurs in `_remote_from_source_mapping()`, which can be covered directly with a faithful loaded HPA-324 row. Existing corpus-cache tests already cover verified body resolution.

## Files changed by implementation

- `src/benchmark/separation_pilot.py`
  - return only validated HPA-324 source rows from `_validate_subset_population()`;
  - pass that mapping through `run_oaf_separation_pilot()`;
  - use it in `_resolve_pilot_sources()`.
- `tests/benchmark/reviewed_subset_fixtures.py`
  - make synthetic HPA-324 inventories include the selected source audio object.
- `tests/benchmark/test_separation_pilot.py`
  - add the direct HPA-324 `_remote_from_source_mapping()` contract test;
  - strengthen the shared Task 6 resolver seam with source-audio membership.

No new runtime files, schema files, dependencies, or CLI flags are needed.

## Non-goals

- HPA-324 or HPA-327 schema changes.
- `corpus_cache.py` changes.
- New cache/download/retry behavior.
- A reusable source/join framework.
- Unifying IDM and separation-pilot source-kwargs helpers.
- New separator locks or separator execution behavior.
- OaF inference, scoring, comparison, report, or handoff changes.
- HPA-329 result publication.
- HPA-627 MuScriptor gated-model work.

## Acceptance

HPA-424 is complete when:

1. the production precondition confirms the real HPA-324 rows used by the pilot contain and can resolve their `source_audio_key` objects; otherwise HPA-424 stops and the defect is moved upstream;
2. a deterministic fixture test proves a loaded synthetic HPA-324 row resolves its selected audio object through `_remote_from_source_mapping()`;
3. the shared Task 6 regression proves `resolve_source_audio()` receives an HPA-324 row whose inventory actually contains `source_audio_key`;
4. existing HPA-327 -> HPA-324 identity validation remains the sole binding authority and retains its current failure semantics;
5. existing post-resolution source-audio identity checks remain unchanged;
6. all six `_task6_seams()` consumer suites and quality checks pass; and
7. one real HPA-328 rerun on the final committed tree corroborates that the original source-resolution failure is gone before the PR is marked ready.
