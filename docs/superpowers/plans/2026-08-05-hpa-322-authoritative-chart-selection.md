# HPA-322 Authoritative Chart Selection Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Consume an immutable HPA-321 manifest/cache, select one authoritative `.dtx` or `.txt` chart per simfile, and publish a closed deterministic reference-chart manifest for HPA-323.

**Architecture:** Establish shared row-read, verified-cache, and pure object-key contracts first. Then add DTXMania decoding, `set.def` parsing, a pure inventory-backed selector, and one offline publication/CLI stage.

**Tech Stack:** Python 3.12, dataclasses, datetime, pathlib/PurePosixPath, JSONL, Click, pytest, Ruff, Pylint.

## Global Constraints

- R2 objects, `set.def`, and selected chart contents are the only authorities.
- Use existing `RemoteObject`, `SimfileInventory`, `SyncError`, and `ProvenanceRecord`.
- Reuse `validate_cached_body`; do not add a second filesystem verifier.
- Use one shared `resolve_inventory_object_key` helper in HPA-322 and HPA-323.
- Evaluate authored slots L5, L4, L3, L2, L1.
- Treat `.dtx` and `.txt` as chart suffixes, case-insensitively.
- Preserve HPA-321 artifacts and publish a new immutable manifest.
- Work offline and sequentially.
- Quarantine ambiguity instead of guessing.
- Freeze the exact output row schema before HPA-323 implementation.
- Use `pytest`, `ruff check`, `ruff format --check`, and the enabled CI Pylint command.

---

## File Map

### Create

- `src/benchmark/dtx_text.py`
- `src/benchmark/inventory_object_keys.py`
- `src/benchmark/set_def_parser.py`
- `src/benchmark/reference_chart_selection.py`
- `src/benchmark/reference_chart_manifest.py`
- `tests/benchmark/test_dtx_text.py`
- `tests/benchmark/test_inventory_object_keys.py`
- `tests/benchmark/test_set_def_parser.py`
- `tests/benchmark/test_reference_chart_selection.py`
- `tests/benchmark/test_reference_chart_manifest.py`
- `tests/benchmark/test_reference_chart_acceptance.py`
- `tests/benchmark/schema_goldens/reference-chart-manifest.jsonl`
- `config/benchmark-reference-chart-overrides.json`

### Modify

- `src/benchmark/r2_corpus_models.py`
- `src/benchmark/corpus_manifest.py`
- `src/benchmark/corpus_cache.py`
- `src/benchmark/dtx_parser.py`
- `src/cli/benchmark.py`
- `tests/benchmark/test_r2_corpus_models.py`
- `tests/benchmark/test_corpus_manifest.py`
- `tests/benchmark/test_corpus_cache.py`
- `tests/benchmark/test_dtx_parser.py`
- `tests/benchmark/schema_goldens/manifest.json`
- `tests/benchmark/test_schema_goldens.py`
- `tests/test_cli_benchmark.py`

---

### Task 1: Shared HPA-321 row, cache-body, and object-key contracts

**Files:**
- Modify: `src/benchmark/r2_corpus_models.py`
- Modify: `src/benchmark/corpus_manifest.py`
- Modify: `src/benchmark/corpus_cache.py`
- Create: `src/benchmark/inventory_object_keys.py`
- Modify: `tests/benchmark/test_r2_corpus_models.py`
- Modify: `tests/benchmark/test_corpus_manifest.py`
- Modify: `tests/benchmark/test_corpus_cache.py`
- Create: `tests/benchmark/test_inventory_object_keys.py`

**Interfaces:**

```python
def parse_manifest_timestamp(value: object) -> datetime

@dataclass(frozen=True)
class ManifestRowView:
    inventory: SimfileInventory
    provenance: ProvenanceRecord
    corpus_version: str
    cache_profile: str
    source_endpoint_sha256: str
    source_bucket: str
    source_discovery_method: str

def manifest_row_view_from_row(row: Mapping[str, object]) -> ManifestRowView

def inventory_from_manifest_row(row: Mapping[str, object]) -> SimfileInventory

def resolve_verified_cache_body(
    cache_dir: Path,
    remote: RemoteObject,
    *,
    source_endpoint_sha256: str,
    bucket: str,
    expected_sha256: str | None = None,
) -> Path

ObjectKeyResolutionStatus = Literal[
    "exact",
    "casefold",
    "missing",
    "ambiguous",
    "invalid_path",
]

@dataclass(frozen=True)
class ResolvedObjectKey:
    status: ObjectKeyResolutionStatus
    normalized_key: str | None
    remote: RemoteObject | None

def resolve_inventory_object_key(
    relative_path: str,
    *,
    base_object_key_dir: str,
    object_prefix: str,
    objects: tuple[RemoteObject, ...],
) -> ResolvedObjectKey
```

- [ ] **Step 1: Write timestamp inverse tests**

```python
@pytest.mark.parametrize(
    "value",
    [
        datetime(2026, 1, 2, 3, 4, 5, tzinfo=timezone.utc),
        datetime(2026, 1, 2, 3, 4, 5, 123456, tzinfo=timezone.utc),
    ],
)
def test_manifest_timestamp_round_trip(value: datetime) -> None:
    assert parse_manifest_timestamp(format_manifest_timestamp(value)) == value
```

Also reject non-strings, offsets other than `Z`, missing timezone, invalid calendar values, and surrounding whitespace.

- [ ] **Step 2: Verify timestamp tests fail**

```bash
uv run pytest tests/benchmark/test_r2_corpus_models.py -k manifest_timestamp -q
```

Expected: failure because `parse_manifest_timestamp` does not exist.

- [ ] **Step 3: Implement the exact timestamp inverse**

Place it beside `format_manifest_timestamp`. Accept `YYYY-MM-DDTHH:MM:SSZ` and the formatter's optional 1–6 fractional digits only.

- [ ] **Step 4: Write truthful manifest-row tests**

Build a row with `build_manifest_rows`, add the rendered `corpus_version`, and assert:

```python
view = manifest_row_view_from_row(row)

assert view.inventory.simfile_id == source.simfile_id
assert view.inventory.object_prefix == source.object_prefix
assert view.inventory.sync_status == source.sync_status
assert view.inventory.sync_errors == source.sync_errors
assert all(remote.errors == () for remote in view.inventory.objects)
assert view.provenance == provenance
assert view.source_endpoint_sha256 == endpoint_hash
assert view.source_bucket == "simfile-dtx"
assert inventory_from_manifest_row(row) == view.inventory
```

Rebuild the base row and compare the complete HPA-321 subset:

```python
rebuilt = build_manifest_rows(
    (view.inventory,),
    {view.inventory.simfile_id: view.provenance},
    view.source_endpoint_sha256,
    view.source_bucket,
)[0]
expected = {key: value for key, value in row.items() if key != "corpus_version"}

assert rebuilt == expected
```

Add rejection tests for exact row/object key sets, duplicate object keys, invalid enums, malformed timestamps, non-integer IDs, invalid hashes, and an error `object_key` absent from the row.

- [ ] **Step 5: Verify row tests fail**

```bash
uv run pytest tests/benchmark/test_corpus_manifest.py -k "manifest_row_view or inventory_from_manifest_row" -q
```

- [ ] **Step 6: Implement `ManifestRowView` parsing**

Parse every serialized error into `SimfileInventory.sync_errors`. Construct each `RemoteObject` with `errors=()`, because HPA-321 does not serialize object-level error arrays. Make `inventory_from_manifest_row` delegate to the row-view function.

- [ ] **Step 7: Write exact cache-adapter tests**

```python
expected = cache_dir / "sha256" / digest[:2] / digest
assert resolve_verified_cache_body(
    cache_dir,
    remote,
    source_endpoint_sha256="f" * 64,
    bucket="simfile-dtx",
) == expected
```

Cover non-verified status, missing/malformed digest, missing cache path, non-canonical cache path, expected-hash mismatch, and missing/unreadable/size-mismatched/digest-mismatched content. Every failure must be exactly `ValueError("verified cache body unavailable")`.

- [ ] **Step 8: Verify cache tests fail**

```bash
uv run pytest tests/benchmark/test_corpus_cache.py -k resolve_verified_cache_body -q
```

- [ ] **Step 9: Implement the adapter over `validate_cached_body`**

Construct exactly:

```python
canonical_cache_path = f"sha256/{digest[:2]}/{digest}"
entry = CacheIndexEntry(
    source_endpoint_sha256=source_endpoint_sha256,
    bucket=bucket,
    key=remote.key,
    etag=remote.etag,
    etag_is_weak=remote.etag_is_weak,
    size=remote.size,
    last_modified=format_manifest_timestamp(remote.last_modified),
    sha256=digest,
    cache_path=canonical_cache_path,
)
```

Require `remote.cache_path == canonical_cache_path`, call `validate_cached_body(cache_dir, entry)`, and return `cache_dir / canonical_cache_path`. Do not call `Path.resolve`, `stat`, or another hashing loop.

- [ ] **Step 10: Write pure object-key resolver tests**

Cover slash normalization, `.` and contained `..`, exact match, unique `str.casefold()` match, missing, duplicate casefold ambiguity, empty/NUL/absolute/UNC/drive-prefixed paths, prefix escape, and the `42/` versus `420/` boundary.

- [ ] **Step 11: Verify resolver tests fail**

```bash
uv run pytest tests/benchmark/test_inventory_object_keys.py -q
```

- [ ] **Step 12: Implement the resolver**

Use POSIX components plus an explicit stack to normalize `.` and `..`. Return statuses; never throw for row-level path outcomes and never access the filesystem.

- [ ] **Step 13: Validate Task 1**

```bash
uv run pytest \
  tests/benchmark/test_r2_corpus_models.py \
  tests/benchmark/test_corpus_manifest.py \
  tests/benchmark/test_corpus_cache.py \
  tests/benchmark/test_inventory_object_keys.py -q
uv run ruff check \
  src/benchmark/r2_corpus_models.py \
  src/benchmark/corpus_manifest.py \
  src/benchmark/corpus_cache.py \
  src/benchmark/inventory_object_keys.py \
  tests/benchmark/test_r2_corpus_models.py \
  tests/benchmark/test_corpus_manifest.py \
  tests/benchmark/test_corpus_cache.py \
  tests/benchmark/test_inventory_object_keys.py
uv run ruff format --check \
  src/benchmark/r2_corpus_models.py \
  src/benchmark/corpus_manifest.py \
  src/benchmark/corpus_cache.py \
  src/benchmark/inventory_object_keys.py \
  tests/benchmark/test_r2_corpus_models.py \
  tests/benchmark/test_corpus_manifest.py \
  tests/benchmark/test_corpus_cache.py \
  tests/benchmark/test_inventory_object_keys.py
```

- [ ] **Step 14: Commit Task 1**

```bash
git add \
  src/benchmark/r2_corpus_models.py \
  src/benchmark/corpus_manifest.py \
  src/benchmark/corpus_cache.py \
  src/benchmark/inventory_object_keys.py \
  tests/benchmark/test_r2_corpus_models.py \
  tests/benchmark/test_corpus_manifest.py \
  tests/benchmark/test_corpus_cache.py \
  tests/benchmark/test_inventory_object_keys.py
git commit -m "feat: read verified corpus objects"
```

---

### Task 2: Shared DTXMania decoding and DLEVEL metadata

**Files:**
- Create: `src/benchmark/dtx_text.py`
- Create: `tests/benchmark/test_dtx_text.py`
- Modify: `src/benchmark/dtx_parser.py`
- Modify: `tests/benchmark/test_dtx_parser.py`

**Interfaces:**

```python
DtxTextKind = Literal["dtx", "set_def"]

def decode_dtxmania_text(
    raw: bytes,
    *,
    source_name: str,
    kind: DtxTextKind,
) -> str

ParsedDtxChart.dlevel_raw: str | None
ParsedDtxChart.dlevel_normalized: int | None
```

- [ ] **Step 1: Write decoder tests**

Cover UTF-8 BOM, UTF-16 LE/BE BOM, plain UTF-8, CP932-only content, Shift-JIS, binary content, and separate acceptance predicates:

```python
decode_dtxmania_text(b"#BPM: 120\n", source_name="a.dtx", kind="dtx")
decode_dtxmania_text(b"#L5FILE: real.dtx\n", source_name="set.def", kind="set_def")
```

A generic `#COMMENT` must not be sufficient for `kind="set_def"`.

- [ ] **Step 2: Verify decoder tests fail**

```bash
uv run pytest tests/benchmark/test_dtx_text.py -q
```

- [ ] **Step 3: Implement locked decoding behavior**

Try BOM-declared UTF-8/UTF-16 first, then plain UTF-8, CP932, and Shift-JIS. For `dtx`, require the current `^[#*]\s*[0-9A-Za-z]` directive shape. For `set_def`, require an L1–L5 LABEL/FILE directive. Never include body contents in errors.

- [ ] **Step 4: Move `parse_dtx_file` to the shared decoder**

Call `decode_dtxmania_text(..., kind="dtx")`. Preserve every existing decoder test outcome before adding new CP932 coverage.

- [ ] **Step 5: Write DLEVEL tests**

```python
def test_parse_dtx_retains_numeric_dlevel() -> None:
    chart = parse_dtx_text("#DLEVEL: 87\n#BPM: 120\n", "song")

    assert chart.dlevel_raw == "87"
    assert chart.dlevel_normalized == 87
```

Cover duplicate directives (last wins), decimals, non-ASCII digits, negatives, and values above 100 as warnings with `dlevel_normalized is None`.

- [ ] **Step 6: Implement DLEVEL metadata**

Add fields to `ParsedDtxChart`, retain the last raw value, and normalize ASCII decimal values in `0..100`.

- [ ] **Step 7: Validate Task 2**

```bash
uv run pytest tests/benchmark/test_dtx_text.py tests/benchmark/test_dtx_parser.py -q
uv run ruff check \
  src/benchmark/dtx_text.py src/benchmark/dtx_parser.py \
  tests/benchmark/test_dtx_text.py tests/benchmark/test_dtx_parser.py
uv run ruff format --check \
  src/benchmark/dtx_text.py src/benchmark/dtx_parser.py \
  tests/benchmark/test_dtx_text.py tests/benchmark/test_dtx_parser.py
```

- [ ] **Step 8: Commit Task 2**

```bash
git add src/benchmark/dtx_text.py src/benchmark/dtx_parser.py \
  tests/benchmark/test_dtx_text.py tests/benchmark/test_dtx_parser.py
git commit -m "feat: share DTXMania text decoding"
```

---

### Task 3: Parse authored `set.def` slots

**Files:**
- Create: `src/benchmark/set_def_parser.py`
- Create: `tests/benchmark/test_set_def_parser.py`

**Interfaces:**

```python
@dataclass(frozen=True)
class SetDefSlot:
    level: int
    label: str | None
    file: str | None

@dataclass(frozen=True)
class ParsedSetDef:
    slots: tuple[SetDefSlot, ...]
    warnings: tuple[str, ...]

def parse_set_def_text(text: str) -> ParsedSetDef

def parse_set_def_bytes(raw: bytes, *, source_name: str) -> ParsedSetDef
```

- [ ] **Step 1: Write grammar tests**

Cover `#`/`*`, colon/whitespace forms, optional spaces, lower-case directives, quoted values, CP932 labels, duplicate fields, empty slots, custom filenames, and `.txt` FILE values.

- [ ] **Step 2: Verify grammar tests fail**

```bash
uv run pytest tests/benchmark/test_set_def_parser.py -q
```

- [ ] **Step 3: Implement one focused parser**

Decode with `kind="set_def"`. Return exactly five ordered slots L1–L5. Use the last duplicate field and append `duplicate LxFIELD; last value wins`. Ignore unrelated directives and semicolon comments.

- [ ] **Step 4: Validate Task 3**

```bash
uv run pytest tests/benchmark/test_set_def_parser.py -q
uv run ruff check src/benchmark/set_def_parser.py tests/benchmark/test_set_def_parser.py
uv run ruff format --check src/benchmark/set_def_parser.py tests/benchmark/test_set_def_parser.py
```

- [ ] **Step 5: Commit Task 3**

```bash
git add src/benchmark/set_def_parser.py tests/benchmark/test_set_def_parser.py
git commit -m "feat: parse authored set.def slots"
```

---

### Task 4: Select one authoritative chart

**Files:**
- Create: `src/benchmark/reference_chart_selection.py`
- Create: `tests/benchmark/test_reference_chart_selection.py`
- Create: `config/benchmark-reference-chart-overrides.json`

**Interfaces:**

```python
SelectionStatus = Literal["selected", "quarantined"]

SelectionMethod = Literal[
    "override",
    "set_def_slot",
    "single_candidate_fallback",
    "dlevel_fallback",
    "filename_tiebreak_fallback",
]

SelectionReasonCode = Literal[
    "source_inventory_unusable",
    "cached_body_unavailable",
    "no_verified_chart",
    "ambiguous_set_def",
    "invalid_set_def",
    "referenced_chart_missing",
    "invalid_chart_reference",
    "ambiguous_chart_key",
    "selected_chart_parse_failed",
    "override_invalid",
    "ambiguous_fallback",
]

@dataclass(frozen=True)
class SelectionOverride:
    chart_key: str
    reason: str

@dataclass(frozen=True)
class LoadedOverrides:
    document_sha256: str
    by_simfile_id: Mapping[int, SelectionOverride]

@dataclass(frozen=True)
class ChartSelection:
    status: SelectionStatus
    method: SelectionMethod | None
    reason_codes: tuple[SelectionReasonCode, ...]
    warnings: tuple[str, ...]
    set_def: RemoteObject | None
    selected_chart: RemoteObject | None
    selected_level_slot: str | None
    selected_level_label: str | None
    dlevel_raw: str | None
    dlevel_normalized: int | None
    title: str | None
    artist: str | None
    override: SelectionOverride | None

def select_reference_chart(
    row: ManifestRowView,
    *,
    cache_dir: Path,
    overrides: LoadedOverrides,
) -> ChartSelection
```

- [ ] **Step 1: Write override loader tests**

Commit this exact empty document:

```json
{"overrides":{},"schema_version":"crux.reference-chart-overrides/v1"}
```

Require canonical decimal IDs, exact keys `chart_key` and `reason`, non-empty strings, unique normalized IDs, and exact-byte SHA-256.

- [ ] **Step 2: Write canonical `set.def` discovery tests**

Cover one root copy, root plus nested copies, one nested copy, exact lowercase preference, same-depth ambiguity, and a canonical candidate whose cache/parse failure quarantines instead of trying another copy.

- [ ] **Step 3: Implement discovery through verified cache bodies**

Operate only on `row.inventory.objects`. Open every body through `resolve_verified_cache_body` using source identity from `ManifestRowView`.

- [ ] **Step 4: Write authored-slot tests**

Cover L5 over L4, custom names, `.txt` references, nested relative paths, exact match, unique casefold, contained `..`, simfile-root fallback after a relative miss, missing L5 continuing to L4, invalid/ambiguous paths, and an existing invalid chart quarantining without downgrade.

- [ ] **Step 5: Implement authored-slot selection with the shared resolver**

Call `resolve_inventory_object_key` relative to the `set.def` directory. On `missing` for nested `set.def`, call the same helper with the simfile-root directory and add warning `set_def_root_fallback` when it resolves. Require `.dtx` or `.txt`.

- [ ] **Step 6: Write override application tests**

Require the override key to be exact, verified, parseable, and end with `.dtx` or `.txt`, case-insensitively. Overrides run before `set.def`. Invalid row entries quarantine without fallback and remain visible in `ChartSelection.override`.

- [ ] **Step 7: Write fallback tests**

Candidates are verified objects matching:

```python
remote.key.casefold().endswith((".dtx", ".txt"))
```

Cover one candidate, unique highest DLEVEL, equal highest level with `real.txt > full.dtx > mas.txt`, higher DLEVEL outranking filename, duplicate recognized basenames remaining ambiguous, no parseable candidate, and no alphabetical tie-break.

- [ ] **Step 8: Implement fallback and dataclass invariants**

Selected results require a method, selected chart, empty reason codes, and string title/artist. Quarantined results require `method=None`, no selected chart, and non-empty reason codes. Keep declared reason order and source warning order.

- [ ] **Step 9: Validate Task 4**

```bash
uv run pytest tests/benchmark/test_reference_chart_selection.py -q
uv run ruff check \
  src/benchmark/reference_chart_selection.py \
  tests/benchmark/test_reference_chart_selection.py
uv run ruff format --check \
  src/benchmark/reference_chart_selection.py \
  tests/benchmark/test_reference_chart_selection.py
```

- [ ] **Step 10: Commit Task 4**

```bash
git add \
  src/benchmark/reference_chart_selection.py \
  tests/benchmark/test_reference_chart_selection.py \
  config/benchmark-reference-chart-overrides.json
git commit -m "feat: select authoritative cached charts"
```

---

### Task 5: Freeze and publish the derived manifest

**Files:**
- Create: `src/benchmark/reference_chart_manifest.py`
- Create: `tests/benchmark/test_reference_chart_manifest.py`
- Create: `tests/benchmark/test_reference_chart_acceptance.py`
- Create: `tests/benchmark/schema_goldens/reference-chart-manifest.jsonl`
- Modify: `tests/benchmark/schema_goldens/manifest.json`
- Modify: `tests/benchmark/test_schema_goldens.py`
- Modify: `src/cli/benchmark.py`
- Modify: `tests/test_cli_benchmark.py`

**Interfaces:**

```python
REFERENCE_CHART_MANIFEST_SCHEMA = "crux.reference-chart-manifest/v1"

@dataclass(frozen=True)
class SelectionRequest:
    manifest_path: Path
    cache_dir: Path
    overrides_file: Path | None
    output_dir: Path
    default_overrides_missing_ok: bool = False

@dataclass(frozen=True)
class SelectionOutcome:
    status: Literal["complete", "partial", "failed"]
    exit_code: Literal[0, 1, 2]
    report_path: Path | None
    manifest: PublishedManifest | None
    selected_count: int
    quarantined_count: int

def select_reference_manifest(request: SelectionRequest) -> SelectionOutcome

def validate_schema_golden(schema: str, content: bytes) -> None
```

- [ ] **Step 1: Write input-validation tests**

Read input bytes exactly once. Reject zero records, malformed/non-canonical JSONL, wrong schema, mixed corpus versions/endpoints/buckets/cache profiles/discovery methods, duplicate IDs, and malformed row views. Record exact `source_manifest_sha256`.

- [ ] **Step 2: Implement typed loading**

Use `manifest_row_view_from_row` once per row. Do not separately parse provenance or source identity in orchestration.

- [ ] **Step 3: Write exact row-construction tests**

Require the complete HPA-321 field set plus:

```python
SELECTION_ROW_KEYS = {
    "source_manifest_sha256",
    "source_corpus_version",
    "selection_status",
    "selection_method",
    "selection_reason_codes",
    "selection_warnings",
    "set_def_key",
    "set_def_content_hash",
    "selected_chart_key",
    "selected_chart_content_hash",
    "selected_chart_cache_path",
    "selected_level_slot",
    "selected_level_label",
    "dlevel_raw",
    "dlevel_normalized",
    "title",
    "artist",
    "override_document_sha256",
    "selection_override",
}
```

Selected rows require all selected identities. Quarantined rows require chart/slot/DLEVEL/title/artist fields to be `None`, `selection_method is None`, and non-empty reason codes. `set_def_*` may remain populated when discovery succeeded before a later failure.

- [ ] **Step 4: Implement row construction**

Rebuild the HPA-321 base fields from `ManifestRowView`, replace only `schema_version`, let `render_manifest` add the derived `corpus_version`, and append the closed selection fields.

- [ ] **Step 5: Add a two-row schema golden**

Create canonical JSONL with one selected and one quarantined row. Register:

```json
{"golden_path":"tests/benchmark/schema_goldens/reference-chart-manifest.jsonl","schema":"crux.reference-chart-manifest/v1","validator_modules":["src.benchmark.reference_chart_manifest"]}
```

in the canonically sorted golden manifest, and add the schema ID to the expected subset in `test_schema_goldens.py`.

- [ ] **Step 6: Implement the strict golden validator**

Require exactly two canonical rows, one selected and one quarantined; exact row/object/error/override key sets; exact enums/nullability; valid hashes/timestamps/cache paths/source identities; one consistent derived `corpus_version`; and a recomputed derived version equal to the rows. Existing mutation tests must reject removed, added, duplicate, and mistyped fields.

- [ ] **Step 7: Write deterministic publication tests**

Assert identical normalized rows produce identical bytes and manifest hash; partial output publishes; selected plus quarantined equals input; `override_document_sha256` hashes exact bytes; and changing only override bytes changes row bytes, derived corpus version, and manifest SHA-256.

- [ ] **Step 8: Implement immutable publication and exit rules**

Use `render_manifest`, `publish_manifest`, and `publish_latest_manifest`. Fatal load/publication returns exit `2` without a manifest. A non-empty publication returns:

```python
exit_code = 0 if quarantined_count == 0 else 1
```

All-quarantined is exit `1`.

- [ ] **Step 9: Wire `select-reference-charts`**

Use lazy imports. Default cache to `manifest.parent.parent / "cache"`. Emit one sorted JSON summary containing status, exit code, manifest path/hash/version, selected/quarantined counts, and report path.

- [ ] **Step 10: Build the offline acceptance fixture**

Include authored L5 `real.dtx`, custom `.txt`, nested path, unique casefold, simfile-root fallback, explicit override, and one quarantined ambiguity. Use no R2 dependency or network.

Run twice and assert:

```python
assert second_manifest_bytes == first_manifest_bytes
assert second_manifest_sha256 == first_manifest_sha256
assert second_corpus_version == first_corpus_version
```

Run again with changed override bytes and assert all three identities change.

- [ ] **Step 11: Write CLI exit tests**

Cover all selected -> `0`; mixed -> `1`; all quarantined -> `1` with a published manifest; empty/malformed input -> `2`; publication failure -> `2`.

- [ ] **Step 12: Run full validation**

```bash
uv run pytest -q
uv run ruff check .
uv run ruff format --check src tests
```

Run the exact Pylint command from `.github/workflows/ci.yml` if enabled.

- [ ] **Step 13: Commit Task 5**

```bash
git add \
  src/benchmark/reference_chart_manifest.py \
  src/cli/benchmark.py \
  tests/benchmark/test_reference_chart_manifest.py \
  tests/benchmark/test_reference_chart_acceptance.py \
  tests/benchmark/schema_goldens/reference-chart-manifest.jsonl \
  tests/benchmark/schema_goldens/manifest.json \
  tests/benchmark/test_schema_goldens.py \
  tests/test_cli_benchmark.py
git commit -m "feat: publish reference chart manifest"
```

---

## Risk Verification Matrix

| Risk | Required proof |
|---|---|
| Wrong chart chosen quietly | Authored-slot, containment, casefold, root-fallback, override, and fallback tests |
| Decoder extraction changes chart identity | Existing decoder suite plus CP932-only and set.def predicate tests |
| HPA-323 forks row/key semantics | HPA-323 imports the row view, cache verifier, and object-key resolver |
| Derived schema drifts | Closed key set, two-row golden, and mutation tests |
| Rerun changes identity | Byte-identical second-run acceptance assertion |
| Override changes disappear | Exact override-byte hash and changed-identity assertion |

## Final Review Checklist

- [ ] Manifest timestamps round-trip exactly.
- [ ] `ManifestRowView` preserves inventory, provenance, and source identity.
- [ ] Reconstructed `RemoteObject.errors` are empty because the wire format omits them.
- [ ] Cache-body resolution constructs the exact canonical `CacheIndexEntry`.
- [ ] HPA-322 and HPA-323 share one object-key resolver.
- [ ] DTX and `set.def` decode acceptance predicates are distinct and regression-tested.
- [ ] Authored and fallback candidates include `.dtx` and `.txt`.
- [ ] Selection follows authored order and quarantines ambiguity.
- [ ] The output row schema has a strict selected/quarantined golden.
- [ ] All-quarantined publishes and exits `1`.
- [ ] A second identical run is byte-identical.
- [ ] No network, database, service, concurrency, compatibility, or unrelated timing scope was added.
- [ ] Full tests, Ruff, formatter, and enabled Pylint gates pass.
