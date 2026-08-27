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

- HPA-327 reviewed-subset rows are intentionally compact. They carry the selected chart/audio identity plus `source_row_sha256`, but they do not carry the HPA-321 R2 object inventory, `source_endpoint_sha256`, or `source_bucket`.
- `resolve_source_audio()` accepts a source mapping only when that mapping contains the authoritative object inventory and endpoint/bucket identity needed to resolve a verified cache object.
- HPA-324 reference rows already contain those fields because the HPA-324 manifest is derived from the HPA-323/HPA-321 source lineage.

The required HPA-324 manifest is already loaded by `run_oaf_separation_pilot()`. The defect is therefore a caller-wiring bug, not a missing schema or a cache-resolver capability gap.

## Existing rails to reuse

### `_validate_subset_population()` already owns the join

`src/benchmark/separation_pilot.py::_validate_subset_population()` already builds the HPA-324 source-row map by `simfile_id` and rejects a reviewed member when:

- the `simfile_id` is absent from the HPA-324 reference manifest;
- selected chart key/hash differs;
- source audio key/hash differs; or
- `source_row_sha256` differs from `_source_row_sha256(reference_row)`.

This is already the correct HPA-327 -> HPA-324 authority check. HPA-424 must not implement a second join or row-identity validator.

### `_source_audio_kwargs()` already extracts the resolver identity

`_source_audio_kwargs()` already extracts:

- `source_audio_key`;
- `source_audio_content_hash`;
- `source_endpoint_sha256`; and
- `source_bucket`.

It works for the HPA-324 row. No new request object or resolver adapter is required.

### `resolve_source_audio()` has the correct contract

`src/benchmark/corpus_cache.py::resolve_source_audio()` legitimately requires the source mapping to expose its object inventory plus endpoint/bucket identity. HPA-424 must leave this generic cache boundary unchanged.

## Design

### 1. Carry the validated HPA-324 binding forward

Change `_validate_subset_population()` from a validate-and-discard helper into a validate-and-return helper:

```python
def _validate_subset_population(
    subset: LoadedReviewedSubsetManifest,
    reference: LoadedReferenceSetManifest,
) -> dict[int, Mapping[str, object]]:
    reference_rows = {
        loaded.view.simfile_id: loaded.source_row
        for loaded in reference.rows
    }
    for loaded in subset.rows:
        reference_row = reference_rows.get(loaded.view.simfile_id)
        # Existing validation remains unchanged.
        ...
    return reference_rows
```

The returned mapping is not a new persisted model. It is only the in-memory result of the validation the pilot already performs.

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
4. derive `_source_audio_kwargs()` from that same HPA-324 row; and
5. keep the existing resolved `source_audio_id` / `source_audio_sha256` comparison against fixed membership.

The defensive missing-map error inside `_resolve_pilot_sources()` may remain, but it is not a second authority check. A normal call has already passed `_validate_subset_population()`.

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

The current Task 6 test seam hides the production failure because its fake `resolve_source_audio()` reads only `source_audio_key` and `source_audio_content_hash` from whatever mapping it receives.

Strengthen the existing Task 6 seam rather than building a second end-to-end cache fixture. The fake resolver should record the mapping it receives and require the fields that distinguish the authoritative HPA-324 row:

```python
assert isinstance(source.get("objects"), list)
assert isinstance(source.get("source_endpoint_sha256"), str)
assert isinstance(source.get("source_bucket"), str)
```

The existing `test_task6_infers_only_the_two_derived_views_after_resolving_membership` then becomes the regression test:

- on current `main`, it fails before the fake returns because the HPA-327 row has no `objects`;
- after the fix, it passes because `_resolve_pilot_sources()` receives the validated HPA-324 row;
- all existing assertions still prove there are 20 source resolutions and no full-mix retranscription.

Do not add network calls, real separator execution, or a generalized manifest-join fixture to automated tests.

## Alternatives considered

### Expand HPA-327 rows with the R2 object inventory

Rejected. It duplicates upstream source authority into a reviewed-subset schema and creates schema churn for a caller-wiring bug.

### Teach `resolve_source_audio()` to understand HPA-327 manifests

Rejected. The generic cache resolver should not know how to join a benchmark review artifact back to HPA-324.

### Re-run the join independently inside `_resolve_pilot_sources()`

Rejected. `_validate_subset_population()` already owns the join and `source_row_sha256` guard. Repeating it would create two places that must stay semantically identical.

### Add a reusable manifest-join abstraction

Rejected under YAGNI. This bug has one concrete owner and the existing helper already performs the required binding.

## Files changed by implementation

- `src/benchmark/separation_pilot.py`
  - return the validated HPA-324 source-row map from `_validate_subset_population()`;
  - pass it through `run_oaf_separation_pilot()`;
  - use it in `_resolve_pilot_sources()`.
- `tests/benchmark/test_separation_pilot.py`
  - strengthen the existing Task 6 resolver seam to assert the authoritative HPA-324 row shape.

No new runtime files, schema files, fixtures, dependencies, or CLI flags are needed.

## Non-goals

- HPA-324 or HPA-327 schema changes.
- `corpus_cache.py` changes.
- New cache/download/retry behavior.
- New separator locks or separator execution behavior.
- OaF inference, scoring, comparison, report, or handoff changes.
- HPA-329 result publication.
- HPA-627 MuScriptor gated-model work.

## Acceptance

HPA-424 is complete when:

1. the Task 6 regression proves `resolve_source_audio()` is called with the authoritative HPA-324 source row;
2. existing HPA-327 -> HPA-324 identity validation remains the sole binding authority and retains its current failure semantics;
3. existing post-resolution source-audio identity checks remain unchanged;
4. focused HPA-328 tests and quality checks pass; and
5. when the existing local real-run evidence is available, rerunning the pilot proceeds past the previously observed `source manifest does not contain an object inventory` failure.
