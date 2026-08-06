# HPA-322 Authoritative Chart Selection Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Consume an immutable HPA-321 manifest/cache, select one authoritative `.dtx` or `.txt` chart per simfile, and publish a closed deterministic reference-chart manifest for HPA-323.

**Architecture:** Establish shared row-read, verified-cache, cache-profile key, and pure object-key contracts first. Then add DTXMania decoding, `set.def` parsing, one inventory-backed selector, deterministic publication, and a thin offline CLI/acceptance stage.

**Tech Stack:** Python 3.12, dataclasses, datetime, pathlib/PurePosixPath, JSONL, Click, pytest, Ruff, Pylint.

## Global Constraints

- R2 objects, `set.def`, and selected chart contents are the only authorities.
- Use existing `RemoteObject`, `SimfileInventory`, `SyncError`, and `ProvenanceRecord`.
- Reuse `validate_cached_body`; do not add a second filesystem verifier.
- Use one shared `resolve_inventory_object_key` helper in HPA-322 and HPA-323.
- Evaluate authored slots L5, L4, L3, L2, L1.
- Derive `set.def` and chart candidates from shared cache-profile predicates.
- Use one shared filename rank `real, full, mas, ext, adv, bas` in legacy prepare and HPA-322 fallback.
- Preserve HPA-321 artifacts and publish a new immutable manifest.
- Work offline and sequentially.
- Quarantine ambiguity instead of guessing.
- Freeze the exact output row schema before HPA-323 implementation.
- Use `pytest`, `ruff check`, `ruff format --check`, and the enabled CI Pylint command.

---

## File Map

### Create

- `src/benchmark/chart_names.py`
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
- `tests/benchmark/schema_goldens/crux.reference-chart-manifest-v1.jsonl`
- `config/benchmark-reference-chart-overrides.json`

### Modify

- `src/benchmark/r2_corpus_models.py`
- `src/benchmark/corpus_manifest.py`
- `src/benchmark/corpus_cache.py`
- `src/benchmark/dtx_parser.py`
- `src/benchmark/prepare.py`
- `src/cli/benchmark.py`
- `tests/benchmark/test_r2_corpus_models.py`
- `tests/benchmark/test_corpus_manifest.py`
- `tests/benchmark/test_corpus_cache.py`
- `tests/benchmark/test_dtx_parser.py`
- `tests/benchmark/test_prepare.py`
- `tests/benchmark/schema_goldens/manifest.json`
- `tests/benchmark/test_schema_goldens.py`
- `tests/test_cli_benchmark.py`

---

### Task 1: Shared HPA-321 row, cache, profile-key, and object-key contracts

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

def manifest_row_view_from_row(
    row: Mapping[str, object],
) -> ManifestRowView

def inventory_from_manifest_row(
    row: Mapping[str, object],
) -> SimfileInventory

def resolve_verified_cache_body(
    cache_dir: Path,
    remote: RemoteObject,
    *,
    source_endpoint_sha256: str,
    bucket: str,
    expected_sha256: str | None = None,
) -> Path

def is_set_def_key(key: str) -> bool
def is_chart_key(key: str) -> bool
def is_selected(key: str) -> bool

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

Also reject non-strings, offsets other than `Z`, missing timezone, invalid calendar
values, and surrounding whitespace.

- [ ] **Step 2: Run timestamp tests and confirm failure**

```bash
uv run pytest tests/benchmark/test_r2_corpus_models.py -k manifest_timestamp -q
```

Expected: failure because `parse_manifest_timestamp` does not exist.

- [ ] **Step 3: Implement the exact timestamp inverse**

Place it beside `format_manifest_timestamp`. Accept
`YYYY-MM-DDTHH:MM:SSZ` and the formatter's optional 1–6 fractional digits only.
Rewrite `_is_canonical_utc_timestamp` as a `try`/`except` wrapper around this parser,
and retain the existing cache-index timestamp validation tests.

- [ ] **Step 4: Write truthful manifest-row tests**

Build a real row with `build_manifest_rows`, add the rendered `corpus_version`, and
assert:

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

Rebuild the base row:

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

Add rejection tests for exact row/object key sets, duplicate object keys, invalid enums,
malformed timestamps, non-integer IDs, invalid hashes, and an error `object_key` that is
absent from the row.

- [ ] **Step 5: Run row tests and confirm failure**

```bash
uv run pytest tests/benchmark/test_corpus_manifest.py -k "manifest_row_view or inventory_from_manifest_row" -q
```

- [ ] **Step 6: Implement `ManifestRowView` parsing**

Parse all serialized errors into `SimfileInventory.sync_errors`. Construct every
`RemoteObject` with `errors=()`, because HPA-321 does not serialize object-level error
arrays. Make `inventory_from_manifest_row` delegate to the row-view function.

- [ ] **Step 7: Write exact cache-adapter tests**

Use the existing content-addressed cache fixture and assert a verified row returns:

```python
expected = cache_dir / "sha256" / digest[:2] / digest
assert resolve_verified_cache_body(
    cache_dir,
    remote,
    source_endpoint_sha256="f" * 64,
    bucket="simfile-dtx",
) == expected
```

Cover:

- `cache_status` other than `verified`;
- missing or malformed digest;
- `remote.cache_path is None`;
- cache path rejected by `_validate_relative_cache_path`;
- expected-hash mismatch;
- missing, unreadable, size-mismatched, and digest-mismatched content.

Every failure must be exactly `ValueError("verified cache body unavailable")`.

- [ ] **Step 8: Run cache tests and confirm failure**

```bash
uv run pytest tests/benchmark/test_corpus_cache.py -k resolve_verified_cache_body -q
```

- [ ] **Step 9: Implement the adapter over `validate_cached_body`**

Construct:

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

Call `_validate_relative_cache_path(remote.cache_path, digest)` before constructing
the entry. Then call `validate_cached_body(cache_dir, entry)` and return
`cache_dir / entry.cache_path`. Do not call `Path.resolve`, `stat`, or another hashing
loop.

- [ ] **Step 10: Write cache-profile predicate tests**

Lock the existing profile through:

```python
assert is_set_def_key("42/SET.DEF")
assert is_chart_key("42/mas.DTX")
assert is_chart_key("42/readme.TXT")
assert is_selected("42/set.def")
assert is_selected("42/mas.dtx")
assert not is_selected("42/bgm.ogg")
```

- [ ] **Step 11: Run predicate tests and confirm failure**

```bash
uv run pytest tests/benchmark/test_corpus_cache.py -k "is_set_def_key or is_chart_key" -q
```

- [ ] **Step 12: Split the existing cache-profile predicate**

Make `is_selected` compose `is_set_def_key` and `is_chart_key` without changing
`setdef_dtx_txt_v1`.

- [ ] **Step 13: Write pure object-key resolver tests**

Create a minimal tuple of `RemoteObject` values and cover:

```python
result = resolve_inventory_object_key(
    r"..\Charts\REAL.DTX",
    base_object_key_dir="42/config",
    object_prefix="42/",
    objects=objects,
)
```

Required cases:

- slash normalization;
- `.` and `..` that remain inside `42/`;
- exact key winning over casefold;
- one unique `str.casefold()` match;
- missing;
- duplicate casefold ambiguity;
- empty, NUL, `/absolute`, `//unc`, `C:\drive`, and prefix escape;
- a sibling prefix such as `420/` never satisfying prefix `42/`.

- [ ] **Step 14: Run resolver tests and confirm failure**

```bash
uv run pytest tests/benchmark/test_inventory_object_keys.py -q
```

- [ ] **Step 15: Implement the resolver**

Use `PurePosixPath` components plus an explicit stack to normalize `.` and `..`.
Return statuses; never throw for row-level path outcomes and never access the
filesystem.

- [ ] **Step 16: Validate Task 1**

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

- [ ] **Step 17: Commit Task 1**

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

- [ ] **Step 1: Write codec and acceptance-predicate tests**

Cover:

- UTF-8 BOM;
- UTF-16 LE and BE BOM;
- BOM-less UTF-16LE and UTF-16BE;
- plain UTF-8;
- existing Shift-JIS chart fixture;
- CP932-only chart filename or title;
- DTX data/header directives accepted for `kind="dtx"`;
- only `L1..L5 LABEL/FILE` accepted for `kind="set_def"`;
- binary/gibberish rejected even when a codec decodes it;
- errors contain `source_name` but never source bytes.

- [ ] **Step 2: Run decoder tests and confirm failure**

```bash
uv run pytest tests/benchmark/test_dtx_text.py -q
```

- [ ] **Step 3: Implement the focused decoder**

Try BOM-declared codecs first, then `utf-8`, `cp932`, `shift-jis`, BOM-less
`utf-16le`, and BOM-less `utf-16be`. Use separate compiled predicates for DTX and
`set.def`. Return text with a leading BOM removed.

- [ ] **Step 4: Write parser regression and DLEVEL tests**

Retain every current `test_dtx_parser.py` decoder outcome, including BOM-less UTF-16LE.
Add BOM-less UTF-16BE. Add:

```python
def test_parse_dtx_retains_numeric_dlevel() -> None:
    chart = parse_dtx_text("#DLEVEL: 87\n#BPM: 120\n", "song")

    assert chart.dlevel_raw == "87"
    assert chart.dlevel_normalized == 87
```

Cover duplicate directives (last wins), decimals, non-ASCII digits, negatives, and
values above 100 as warnings with `dlevel_normalized is None`.

- [ ] **Step 5: Run parser regression tests and confirm failure**

```bash
uv run pytest tests/benchmark/test_dtx_parser.py -k "decode or encoding or dlevel" -q
```

- [ ] **Step 6: Integrate the decoder and DLEVEL metadata**

Call `decode_dtxmania_text(..., kind="dtx")` from `parse_dtx_file`. Add fields to
`ParsedDtxChart`, retain the last raw DLEVEL, and normalize ASCII decimal values in
`0..100`.

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

Cover `#`/`*`, colon/whitespace forms, optional spaces, lower-case directives, quoted
values, CP932 labels, duplicate fields, empty slots, custom filenames, and `.txt` FILE
values.

- [ ] **Step 2: Run grammar tests and confirm failure**

```bash
uv run pytest tests/benchmark/test_set_def_parser.py -q
```

- [ ] **Step 3: Implement one focused parser**

Decode with `kind="set_def"`. Return exactly five ordered slots L1–L5. Use the last
duplicate field and append `duplicate LxFIELD; last value wins`. Ignore unrelated
directives and semicolon comments.

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
- Create: `src/benchmark/chart_names.py`
- Create: `src/benchmark/reference_chart_selection.py`
- Create: `tests/benchmark/test_reference_chart_selection.py`
- Create: `config/benchmark-reference-chart-overrides.json`
- Modify: `src/benchmark/prepare.py`
- Modify: `tests/benchmark/test_prepare.py`

**Interfaces:**

```python
CHART_FILENAME_PRIORITY = ("real", "full", "mas", "ext", "adv", "bas")

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

def load_selection_overrides(
    path: Path | None,
    *,
    missing_ok: bool,
) -> LoadedOverrides

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

- [ ] **Step 1: Write shared filename-rank tests**

Assert the exact total order:

```python
assert CHART_FILENAME_PRIORITY == ("real", "full", "mas", "ext", "adv", "bas")
```

In `test_prepare.py`, create same-folder `real.dtx`, `full.dtx`, `mas.dtx`, `ext.dtx`,
`adv.dtx`, and `bas.dtx` fixtures and assert the first available rank wins. Add a
selector test proving the same constant decides only equal-DLEVEL fallback; authored
`set.def` order still wins.

- [ ] **Step 2: Run rank tests and confirm failure**

```bash
uv run pytest tests/benchmark/test_prepare.py tests/benchmark/test_reference_chart_selection.py -k "filename_priority or shared_rank" -q
```

- [ ] **Step 3: Implement the shared rank**

Create `chart_names.py`, import the constant from `prepare.py`, and import it from the
reference selector. Do not port `set.def` logic into the legacy folder path.

- [ ] **Step 4: Write override-loader tests**

Commit this exact empty document:

```json
{"overrides":{},"schema_version":"crux.reference-chart-overrides/v1"}
```

Test `load_selection_overrides` with:

- canonical decimal IDs;
- exact entry keys `chart_key` and `reason`;
- non-empty strings;
- duplicate IDs after numeric normalization;
- canonical JSON bytes through `strict_json_loads(..., require_canonical=True)`;
- exact-byte SHA-256;
- `path=None, missing_ok=True` using the canonical empty bytes;
- missing or `None` with `missing_ok=False` failing.

- [ ] **Step 5: Run override-loader tests and confirm failure**

```bash
uv run pytest tests/benchmark/test_reference_chart_selection.py -k load_selection_overrides -q
```

- [ ] **Step 6: Implement `load_selection_overrides`**

The loader owns file/default selection, strict parsing, typed validation, and exact-byte
hashing. It returns `LoadedOverrides`; `select_reference_chart` never reads the file.

- [ ] **Step 7: Write inventory-gate and `set.def` discovery tests**

Cover:

- `sync_status == "empty"` -> `source_inventory_unusable`;
- zero verified objects -> `source_inventory_unusable`;
- partial/failed status with a usable verified chart -> warning `partial_inventory`;
- unrelated `sync_errors` do not block a verified authored chart;
- one root `set.def`, root plus nested, one nested, exact lowercase preference;
- same-depth ambiguity;
- a canonical candidate whose cache/parse failure quarantines instead of trying another
  copy.

- [ ] **Step 8: Run inventory/discovery tests and confirm failure**

```bash
uv run pytest tests/benchmark/test_reference_chart_selection.py -k "source_inventory or set_def" -q
```

- [ ] **Step 9: Implement the inventory gate and discovery**

Use `is_set_def_key`; do not restate the basename rule. Apply the exact
`source_inventory_unusable` trigger and preserve `partial_inventory` warning order.

- [ ] **Step 10: Write authored-slot and override-application tests**

Cover L5 over L4, custom names, `.txt`, nested relative paths, exact match, unique
casefold, contained `..`, simfile-root fallback, missing L5 continuing to L4,
invalid/ambiguous paths, and an existing invalid chart quarantining without downgrade.

Overrides run before `set.def`. Require an exact verified object accepted by
`is_chart_key`; invalid row entries quarantine without fallback and remain visible in
`ChartSelection.override`.

- [ ] **Step 11: Run authored/override tests and confirm failure**

```bash
uv run pytest tests/benchmark/test_reference_chart_selection.py -k "authored or override_application" -q
```

- [ ] **Step 12: Implement authored and override selection**

Use `resolve_inventory_object_key`, `is_chart_key`, and
`resolve_verified_cache_body`. Do not inline suffix, containment, casefold, or cache
rules.

- [ ] **Step 13: Write evidence-bearing fallback tests**

Candidates come only from verified objects accepted by `is_chart_key`.

Cover:

- a header-only `.txt` is not a fallback candidate;
- BPM/measure-length/BGM-only files are not fallback candidates;
- a parsed event with `lane_id != "01"` supplies fallback evidence;
- one evidence-bearing candidate;
- unique highest DLEVEL;
- equal highest DLEVEL using the shared rank across `real`, `full`, `mas`, `ext`,
  `adv`, and `bas`;
- higher DLEVEL outranking filename;
- duplicate recognized basenames remaining ambiguous;
- no evidence-bearing candidate;
- no alphabetical tie-break.

The evidence gate applies only to unauthored fallback. It does not override `set.def`
or a manual override.

- [ ] **Step 14: Run fallback tests and confirm failure**

```bash
uv run pytest tests/benchmark/test_reference_chart_selection.py -k fallback -q
```

- [ ] **Step 15: Implement fallback and dataclass invariants**

Use:

```python
has_note_evidence = any(event.lane_id != "01" for event in chart.events)
```

Selected results require a method, selected chart, empty reason codes, and string
title/artist. Quarantined results require `method=None`, no selected chart, and
non-empty reason codes.

- [ ] **Step 16: Validate Task 4**

```bash
uv run pytest tests/benchmark/test_prepare.py tests/benchmark/test_reference_chart_selection.py -q
uv run ruff check \
  src/benchmark/chart_names.py \
  src/benchmark/prepare.py \
  src/benchmark/reference_chart_selection.py \
  tests/benchmark/test_prepare.py \
  tests/benchmark/test_reference_chart_selection.py
uv run ruff format --check \
  src/benchmark/chart_names.py \
  src/benchmark/prepare.py \
  src/benchmark/reference_chart_selection.py \
  tests/benchmark/test_prepare.py \
  tests/benchmark/test_reference_chart_selection.py
```

- [ ] **Step 17: Commit Task 4**

```bash
git add \
  src/benchmark/chart_names.py \
  src/benchmark/prepare.py \
  src/benchmark/reference_chart_selection.py \
  tests/benchmark/test_prepare.py \
  tests/benchmark/test_reference_chart_selection.py \
  config/benchmark-reference-chart-overrides.json
git commit -m "feat: select authoritative cached charts"
```

---

### Task 5: Freeze and publish the derived manifest

**Files:**
- Create: `src/benchmark/reference_chart_manifest.py`
- Create: `tests/benchmark/test_reference_chart_manifest.py`
- Create: `tests/benchmark/schema_goldens/crux.reference-chart-manifest-v1.jsonl`
- Modify: `tests/benchmark/schema_goldens/manifest.json`
- Modify: `tests/benchmark/test_schema_goldens.py`

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
    manifest: PublishedManifest | None
    selected_count: int
    quarantined_count: int

def select_reference_manifest(request: SelectionRequest) -> SelectionOutcome

def validate_schema_golden(schema: str, content: bytes) -> None
```

- [ ] **Step 1: Write strict input-loading tests**

Read input bytes exactly once. Require a final newline and no blank/partial physical
lines. Parse every line with:

```python
strict_json_loads(line[:-1], require_canonical=True)
```

Reject zero records, non-object rows, wrong schema, mixed corpus
versions/endpoints/buckets/cache profiles/discovery methods, duplicate IDs, and
malformed row views. Record exact `source_manifest_sha256`.

- [ ] **Step 2: Run input tests and confirm failure**

```bash
uv run pytest tests/benchmark/test_reference_chart_manifest.py -k input -q
```

- [ ] **Step 3: Implement strict loading**

Retain each validated source row mapping beside its `ManifestRowView`. Do not separately
parse provenance or source identity.

- [ ] **Step 4: Write exact row-construction tests**

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

Selected rows require all selected identities. Quarantined rows require
chart/slot/DLEVEL/title/artist fields to be `None`, `selection_method is None`, and
non-empty reason codes. `set_def_*` may remain populated after later failure.

Assert the validated source mapping, excluding only `corpus_version`, equals the
`build_manifest_rows` reconstruction. This proves the typed reader but does not use the
reconstruction for output.

- [ ] **Step 5: Run row-construction tests and confirm failure**

```bash
uv run pytest tests/benchmark/test_reference_chart_manifest.py -k row -q
```

- [ ] **Step 6: Implement verbatim base-row pass-through**

Start from `dict(validated_source_row)`, remove its source `corpus_version`, replace
`schema_version`, and append the closed selection fields. Let `render_manifest` add the
derived `corpus_version`.

- [ ] **Step 7: Add the two-row schema golden**

Create one selected and one quarantined canonical row at:

```text
tests/benchmark/schema_goldens/crux.reference-chart-manifest-v1.jsonl
```

Register:

```json
{"golden_path":"tests/benchmark/schema_goldens/crux.reference-chart-manifest-v1.jsonl","schema":"crux.reference-chart-manifest/v1","validator_modules":["src.benchmark.reference_chart_manifest"]}
```

and add the schema ID to the expected subset in `test_schema_goldens.py`.

- [ ] **Step 8: Run golden tests and confirm failure**

```bash
uv run pytest tests/benchmark/test_schema_goldens.py -k reference_chart -q
```

- [ ] **Step 9: Implement the strict golden validator**

Require exactly two canonical rows, one selected and one quarantined; exact
row/object/error/override key sets; exact enums/nullability; valid hashes/timestamps/
cache paths/source identities; one consistent derived `corpus_version`; and a
recomputed derived version equal to the rows. Existing mutation tests must reject
removed, added, duplicate, and mistyped fields.

- [ ] **Step 10: Write deterministic publication tests**

Assert:

- identical normalized rows produce identical bytes, corpus version, and manifest hash;
- partial and all-quarantined output publish;
- selected plus quarantined equals input;
- empty simfile rows quarantine as `source_inventory_unusable`;
- `override_document_sha256` hashes exact bytes;
- changing only override bytes changes row bytes, corpus version, and manifest SHA-256;
- fatal loading/publication returns exit `2` without a manifest.

- [ ] **Step 11: Run publication tests and confirm failure**

```bash
uv run pytest tests/benchmark/test_reference_chart_manifest.py -k publication -q
```

- [ ] **Step 12: Implement immutable publication and exit rules**

Use `render_manifest`, `publish_manifest`, and `publish_latest_manifest`. A published
manifest returns:

```python
exit_code = 0 if quarantined_count == 0 else 1
```

Zero source records or fatal load/publication returns exit `2`. Do not create a separate
report artifact; row reasons and counts are already in the manifest/outcome.

- [ ] **Step 13: Validate Task 5**

```bash
uv run pytest \
  tests/benchmark/test_reference_chart_manifest.py \
  tests/benchmark/test_schema_goldens.py -q
uv run ruff check \
  src/benchmark/reference_chart_manifest.py \
  tests/benchmark/test_reference_chart_manifest.py \
  tests/benchmark/test_schema_goldens.py
uv run ruff format --check \
  src/benchmark/reference_chart_manifest.py \
  tests/benchmark/test_reference_chart_manifest.py \
  tests/benchmark/test_schema_goldens.py
```

- [ ] **Step 14: Commit Task 5**

```bash
git add \
  src/benchmark/reference_chart_manifest.py \
  tests/benchmark/test_reference_chart_manifest.py \
  tests/benchmark/schema_goldens/crux.reference-chart-manifest-v1.jsonl \
  tests/benchmark/schema_goldens/manifest.json \
  tests/benchmark/test_schema_goldens.py
git commit -m "feat: publish reference chart manifest"
```

---

### Task 6: Wire the CLI and offline acceptance path

**Files:**
- Create: `tests/benchmark/test_reference_chart_acceptance.py`
- Modify: `src/cli/benchmark.py`
- Modify: `tests/test_cli_benchmark.py`

- [ ] **Step 1: Write CLI and acceptance tests**

Cover:

- summary fields: status, exit code, manifest path/hash/version, selected count, and
  quarantined count;
- no `report_path` field;
- all selected -> `0`;
- mixed -> `1`;
- all quarantined -> `1` with a manifest;
- one HPA-321 `sync_status="empty"` row -> `1` with
  `source_inventory_unusable`;
- zero/malformed source rows -> `2`;
- publication failure -> `2`;
- no R2 dependency or network access.

The offline corpus fixture includes authored L5 `real.dtx`, custom `.txt`, nested path,
unique casefold, simfile-root fallback, explicit override, evidence-bearing fallback,
header-only `.txt` rejection, and one quarantined ambiguity.

- [ ] **Step 2: Run CLI/acceptance tests and confirm failure**

```bash
uv run pytest \
  tests/test_cli_benchmark.py \
  tests/benchmark/test_reference_chart_acceptance.py \
  -k "select_reference or reference_chart" -q
```

- [ ] **Step 3: Implement `select-reference-charts`**

Use lazy imports. Default cache to `manifest.parent.parent / "cache"`. Call
`select_reference_manifest` and emit one sorted canonical JSON summary with no report
artifact.

- [ ] **Step 4: Add deterministic rerun assertions**

Run the offline selection twice:

```python
assert second_manifest_bytes == first_manifest_bytes
assert second_manifest_sha256 == first_manifest_sha256
assert second_corpus_version == first_corpus_version
```

Run again with changed override bytes and assert all three identities change.

- [ ] **Step 5: Run focused validation**

```bash
uv run pytest \
  tests/test_cli_benchmark.py \
  tests/benchmark/test_reference_chart_acceptance.py -q
```

- [ ] **Step 6: Run full validation**

```bash
uv run pytest -q
uv run ruff check .
uv run ruff format --check src tests
```

Run the exact Pylint command from `.github/workflows/ci.yml` if enabled.

- [ ] **Step 7: Commit Task 6**

```bash
git add \
  src/cli/benchmark.py \
  tests/test_cli_benchmark.py \
  tests/benchmark/test_reference_chart_acceptance.py
git commit -m "feat: expose reference chart selection"
```

---

## Risk Verification Matrix

| Risk | Required proof |
|---|---|
| Wrong chart chosen quietly | Authored-slot, containment, casefold, evidence, root-fallback, override, and fallback tests |
| Legacy and R2 fallback ranks diverge | Shared rank constant plus prepare/selector parity tests |
| Decoder extraction changes chart identity | Existing decoder suite plus CP932, BOM-less UTF-16, and set.def predicate tests |
| HPA-323 forks row/key semantics | HPA-323 imports the row view, cache verifier, and object-key resolver |
| Derived schema drifts | Closed key set, convention-named two-row golden, and mutation tests |
| Upstream data changes during derivation | Verbatim validated source-row pass-through and reconstruction assertion |
| Rerun changes identity | Byte-identical second-run acceptance assertion |
| Override changes disappear | Exact override-byte hash and changed-identity assertion |

## Final Review Checklist

- [ ] Manifest timestamps round-trip exactly and cache-index timestamp validation reuses the parser.
- [ ] `ManifestRowView` preserves inventory, provenance, and source identity.
- [ ] Reconstructed `RemoteObject.errors` are empty because the wire format omits them.
- [ ] Cache-body resolution reuses `_validate_relative_cache_path` and `validate_cached_body`.
- [ ] `is_selected` composes shared `is_set_def_key` and `is_chart_key`.
- [ ] HPA-322 and HPA-323 share one object-key resolver.
- [ ] DTX decoding retains BOM-less UTF-16LE/BE support.
- [ ] Legacy prepare and HPA-322 fallback share one filename rank.
- [ ] Override loading has one explicit owner and exact-byte identity.
- [ ] Empty simfile rows quarantine and publish with exit `1`.
- [ ] Unauthored fallback requires non-control note evidence.
- [ ] Derived rows pass validated upstream fields through verbatim.
- [ ] The output row schema has a strict convention-named selected/quarantined golden.
- [ ] All-quarantined publishes and exits `1`.
- [ ] A second identical run is byte-identical.
- [ ] No network, database, service, concurrency, compatibility, or unrelated timing scope was added.
- [ ] Full tests, Ruff, formatter, and enabled Pylint gates pass.
