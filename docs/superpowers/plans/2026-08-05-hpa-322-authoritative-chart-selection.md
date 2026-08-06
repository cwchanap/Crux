# HPA-322 Authoritative Chart Selection Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Consume an immutable HPA-321 manifest/cache, select one authoritative DTX chart per simfile, and publish a deterministic reference-chart manifest for HPA-323.

**Architecture:** Establish shared manifest-read and hardened cache-body contracts first, then add DTXMania decoding, `set.def` parsing, a pure inventory-backed selector, and one offline manifest/CLI stage.

**Tech Stack:** Python 3.12, dataclasses, datetime, pathlib/PurePosixPath, JSONL, Click, pytest, Ruff.

## Global Constraints

- R2 objects, `set.def`, and selected DTX contents are the only authorities.
- Use existing `RemoteObject`, `SimfileInventory`, and `SyncError` types.
- Reuse `validate_cached_body`; do not add a second filesystem verifier.
- Evaluate authored slots L5, L4, L3, L2, L1.
- Prefer exact object keys, then one unique case-insensitive match.
- Preserve HPA-321 artifacts and publish a new immutable manifest.
- Work offline and sequentially.
- Quarantine ambiguity instead of guessing.
- Use `pytest`, `ruff check`, and `ruff format --check`.

---

## File Map

### Create

- `src/benchmark/dtx_text.py`
- `src/benchmark/set_def_parser.py`
- `src/benchmark/reference_chart_selection.py`
- `src/benchmark/reference_chart_manifest.py`
- `tests/benchmark/test_dtx_text.py`
- `tests/benchmark/test_set_def_parser.py`
- `tests/benchmark/test_reference_chart_selection.py`
- `tests/benchmark/test_reference_chart_manifest.py`
- `tests/benchmark/test_reference_chart_acceptance.py`
- `config/benchmark-reference-chart-overrides.json`

### Modify

- `src/benchmark/r2_corpus_models.py`
- `src/benchmark/corpus_manifest.py`
- `src/benchmark/corpus_cache.py`
- `src/benchmark/dtx_parser.py`
- `src/cli/benchmark.py`
- corresponding existing tests

---

### Task 1: Shared manifest reader and verified cache-body adapter

**Files:**
- Modify: `src/benchmark/r2_corpus_models.py`
- Modify: `src/benchmark/corpus_manifest.py`
- Modify: `src/benchmark/corpus_cache.py`
- Modify: `tests/benchmark/test_r2_corpus_models.py`
- Modify: `tests/benchmark/test_corpus_manifest.py`
- Modify: `tests/benchmark/test_corpus_cache.py`

**Interfaces:**

```python
def parse_manifest_timestamp(value: object) -> datetime

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
```

- [ ] **Step 1: Write timestamp round-trip tests**

```python
@pytest.mark.parametrize(
    "value",
    [
        datetime(2026, 1, 2, 3, 4, 5, tzinfo=timezone.utc),
        datetime(2026, 1, 2, 3, 4, 5, 123456, tzinfo=timezone.utc),
    ],
)
def test_manifest_timestamp_round_trip(value: datetime) -> None:
    rendered = format_manifest_timestamp(value)

    assert parse_manifest_timestamp(rendered) == value
```

Also reject non-strings, offsets other than `Z`, missing timezone, invalid calendar
values, and extra whitespace.

- [ ] **Step 2: Verify timestamp tests fail**

```bash
uv run pytest tests/benchmark/test_r2_corpus_models.py -k manifest_timestamp -q
```

- [ ] **Step 3: Implement the exact inverse**

Implement `parse_manifest_timestamp` beside `format_manifest_timestamp`. Accept only
the emitted UTC forms with optional fractional seconds and return an aware UTC value.

- [ ] **Step 4: Write manifest-row reconstruction tests**

Build a real HPA-321 row through `build_manifest_rows`, then assert:

```python
inventory = inventory_from_manifest_row(row)

assert inventory.simfile_id == source.simfile_id
assert inventory.object_prefix == source.object_prefix
assert inventory.objects == source.objects
assert inventory.sync_status == source.sync_status
assert inventory.sync_errors == source.sync_errors
```

Add failures for duplicate object keys, malformed arrays, invalid enum values,
malformed timestamps, non-integer IDs, and an object-scoped error referencing an absent
key.

- [ ] **Step 5: Verify reconstruction tests fail**

```bash
uv run pytest tests/benchmark/test_corpus_manifest.py -k inventory_from_manifest_row -q
```

- [ ] **Step 6: Implement reconstruction beside `_build_row`**

Parse the existing manifest fields into `RemoteObject`, `SyncError`, and
`SimfileInventory`. Preserve object order. Do not create a new inventory class.

- [ ] **Step 7: Write hardened body-resolution tests**

Use the existing content-addressed cache fixture. Cover:

- verified body returns `cache/sha256/<shard>/<digest>`;
- `not_selected` and `failed` statuses;
- missing digest;
- size mismatch;
- digest mismatch;
- missing/unreadable body;
- expected-hash mismatch.

- [ ] **Step 8: Verify body-resolution tests fail**

```bash
uv run pytest tests/benchmark/test_corpus_cache.py -k resolve_verified_cache_body -q
```

- [ ] **Step 9: Implement as an adapter over `validate_cached_body`**

Construct a `CacheIndexEntry` from the supplied source identity and `RemoteObject`.
Call `validate_cached_body(cache_dir, entry)`. On any state other than `verified`, raise
`ValueError("verified cache body unavailable")`. Return the canonical digest path.
Do not use `Path.resolve`, `stat`, or a second hashing loop.

- [ ] **Step 10: Validate and commit**

```bash
uv run pytest \
  tests/benchmark/test_r2_corpus_models.py \
  tests/benchmark/test_corpus_manifest.py \
  tests/benchmark/test_corpus_cache.py -q
uv run ruff check \
  src/benchmark/r2_corpus_models.py \
  src/benchmark/corpus_manifest.py \
  src/benchmark/corpus_cache.py \
  tests/benchmark/test_r2_corpus_models.py \
  tests/benchmark/test_corpus_manifest.py \
  tests/benchmark/test_corpus_cache.py
uv run ruff format --check \
  src/benchmark/r2_corpus_models.py \
  src/benchmark/corpus_manifest.py \
  src/benchmark/corpus_cache.py \
  tests/benchmark/test_r2_corpus_models.py \
  tests/benchmark/test_corpus_manifest.py \
  tests/benchmark/test_corpus_cache.py
git add src/benchmark/r2_corpus_models.py src/benchmark/corpus_manifest.py src/benchmark/corpus_cache.py tests/benchmark/test_r2_corpus_models.py tests/benchmark/test_corpus_manifest.py tests/benchmark/test_corpus_cache.py
git commit -m "feat: read verified corpus manifest objects"
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
def decode_dtxmania_text(raw: bytes, *, source_name: str) -> str

ParsedDtxChart.dlevel_raw: str | None
ParsedDtxChart.dlevel_normalized: int | None
```

- [ ] **Step 1: Write decoder tests**

Cover UTF-8 BOM, UTF-16 LE/BE BOM, plain UTF-8, CP932, Shift-JIS, and binary content
without recognizable directives.

- [ ] **Step 2: Verify failure**

```bash
uv run pytest tests/benchmark/test_dtx_text.py -q
```

- [ ] **Step 3: Implement the focused decoder**

Try BOM-declared encodings first, then `utf-8`, `cp932`, and `shift-jis`. Accept a
candidate only when a non-empty line begins with `#` or `*` after leading whitespace.
Do not include file content in errors.

- [ ] **Step 4: Write DLEVEL tests**

```python
def test_parse_dtx_retains_numeric_dlevel() -> None:
    chart = parse_dtx_text("#DLEVEL: 87\n#BPM: 120\n", "song")

    assert chart.dlevel_raw == "87"
    assert chart.dlevel_normalized == 87
```

Cover decimals, non-ASCII digits, negatives, and values above `100` as warnings with
`dlevel_normalized is None`.

- [ ] **Step 5: Implement parser integration**

Use `decode_dtxmania_text` in `parse_dtx_file`. Retain the last `#DLEVEL` directive and
normalize ASCII decimal values in `0..100`.

- [ ] **Step 6: Validate and commit**

```bash
uv run pytest tests/benchmark/test_dtx_text.py tests/benchmark/test_dtx_parser.py -q
uv run ruff check src/benchmark/dtx_text.py src/benchmark/dtx_parser.py tests/benchmark/test_dtx_text.py tests/benchmark/test_dtx_parser.py
uv run ruff format --check src/benchmark/dtx_text.py src/benchmark/dtx_parser.py tests/benchmark/test_dtx_text.py tests/benchmark/test_dtx_parser.py
git add src/benchmark/dtx_text.py src/benchmark/dtx_parser.py tests/benchmark/test_dtx_text.py tests/benchmark/test_dtx_parser.py
git commit -m "feat: share DTXMania decoding metadata"
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
    label: str | None = None
    file: str | None = None

@dataclass(frozen=True)
class ParsedSetDef:
    slots: tuple[SetDefSlot, ...]
    warnings: tuple[str, ...] = ()

def parse_set_def_text(text: str) -> ParsedSetDef
def parse_set_def_bytes(raw: bytes, *, source_name: str) -> ParsedSetDef
```

- [ ] **Step 1: Write grammar tests**

Cover `#`/`*`, colon/whitespace forms, optional spaces, lower-case directives, quoted
values, CP932 labels, duplicate fields, and empty slots.

- [ ] **Step 2: Verify failure**

```bash
uv run pytest tests/benchmark/test_set_def_parser.py -q
```

- [ ] **Step 3: Implement one regex and explicit five-slot result**

Use the last duplicate field and append `duplicate LxFIELD; last value wins`. Ignore
unrelated directives and semicolon comments.

- [ ] **Step 4: Validate and commit**

```bash
uv run pytest tests/benchmark/test_set_def_parser.py -q
uv run ruff check src/benchmark/set_def_parser.py tests/benchmark/test_set_def_parser.py
uv run ruff format --check src/benchmark/set_def_parser.py tests/benchmark/test_set_def_parser.py
git add src/benchmark/set_def_parser.py tests/benchmark/test_set_def_parser.py
git commit -m "feat: parse authored set.def slots"
```

---

### Task 4: Select one authoritative chart from reconstructed inventory

**Files:**
- Create: `src/benchmark/reference_chart_selection.py`
- Create: `tests/benchmark/test_reference_chart_selection.py`
- Create: `config/benchmark-reference-chart-overrides.json`

**Interfaces:**

```python
def select_reference_chart(
    inventory: SimfileInventory,
    *,
    cache_dir: Path,
    source_endpoint_sha256: str,
    source_bucket: str,
    overrides: LoadedOverrides,
) -> ChartSelection
```

- [ ] **Step 1: Define selection records and empty override document**

Use frozen dataclasses for `SelectionOverride`, `LoadedOverrides`, and
`ChartSelection`. Commit:

```json
{
  "schema_version": "crux.reference-chart-overrides/v1",
  "overrides": {}
}
```

- [ ] **Step 2: Write canonical `set.def` discovery tests**

Cover one root copy, root plus nested copies, one nested copy, exact lowercase
preference, and ambiguous same-depth copies.

- [ ] **Step 3: Implement discovery against `inventory.objects`**

Selection policy operates on `RemoteObject` metadata. Open every candidate body through
`resolve_verified_cache_body`.

- [ ] **Step 4: Write referenced-key tests**

Cover nested relative paths, `..` that remains inside the simfile prefix, backslashes,
absolute/traversal rejection, exact match, unique casefold fallback, and ambiguous
casefold matches.

- [ ] **Step 5: Implement L5-to-L1 selection**

For each populated slot:

- missing object: warn and continue;
- ambiguous object: quarantine;
- existing verified/parseable DTX: select;
- existing but invalid body or DTX: quarantine without downgrading.

- [ ] **Step 6: Implement override validation**

Require canonical decimal simfile keys, an exact verified `.dtx` object key, and a
non-empty reason. Invalid row-specific overrides produce `override_invalid`.

- [ ] **Step 7: Implement conservative fallback**

Test one candidate, unique highest DLEVEL, equal DLEVEL filename tie-break, higher
DLEVEL outranking filename, and unresolved ties. Never use alphabetical order.

- [ ] **Step 8: Validate and commit**

```bash
uv run pytest tests/benchmark/test_reference_chart_selection.py -q
uv run ruff check src/benchmark/reference_chart_selection.py tests/benchmark/test_reference_chart_selection.py
uv run ruff format --check src/benchmark/reference_chart_selection.py tests/benchmark/test_reference_chart_selection.py
git add src/benchmark/reference_chart_selection.py tests/benchmark/test_reference_chart_selection.py config/benchmark-reference-chart-overrides.json
git commit -m "feat: select authoritative cached charts"
```

---

### Task 5: Publish the derived manifest and CLI

**Files:**
- Create: `src/benchmark/reference_chart_manifest.py`
- Create: `tests/benchmark/test_reference_chart_manifest.py`
- Create: `tests/benchmark/test_reference_chart_acceptance.py`
- Modify: `src/cli/benchmark.py`
- Modify: `tests/test_cli_benchmark.py`

**Interfaces:**

```python
@dataclass(frozen=True)
class SelectionRequest:
    manifest_path: Path
    cache_dir: Path
    overrides_file: Path | None
    output_dir: Path
    default_overrides_missing_ok: bool = False

def select_reference_manifest(request: SelectionRequest) -> SelectionOutcome
```

- [ ] **Step 1: Write input-validation tests**

Cover empty/invalid JSONL, wrong schema, mixed corpus versions, duplicate IDs, and
malformed source rows. Read exact bytes once and retain their SHA-256.

- [ ] **Step 2: Implement loading through `inventory_from_manifest_row`**

Pass each inventory plus row source identity into `select_reference_chart`. Preserve all
HPA-321 values except replaced top-level schema/corpus identity.

- [ ] **Step 3: Write deterministic publication tests**

Assert identical inputs reuse identical manifest bytes, override changes alter derived
identity, selected plus quarantined equals input rows, and partial output still publishes.

- [ ] **Step 4: Implement immutable publication**

Use existing `render_manifest`, `publish_manifest`, and `publish_latest_manifest`.
All selected => exit `0`; any quarantine => exit `1`; fatal input/publication => exit
`2` at the CLI boundary.

- [ ] **Step 5: Wire `select-reference-charts`**

Use lazy imports. Default cache to `manifest.parent.parent / "cache"`. Emit one sorted
JSON summary to stdout.

- [ ] **Step 6: Build the offline acceptance fixture**

Include authored L5 selection, custom filename, nested paths, case fallback, explicit
override, and one quarantined ambiguity. Use no R2 dependency or network.

- [ ] **Step 7: Run full validation**

```bash
uv run pytest -q
uv run ruff check .
uv run ruff format --check src tests
```

Run the exact Pylint command from `.github/workflows/ci.yml` if it remains enabled.

- [ ] **Step 8: Commit**

```bash
git add src/benchmark/reference_chart_manifest.py src/cli/benchmark.py tests/benchmark/test_reference_chart_manifest.py tests/benchmark/test_reference_chart_acceptance.py tests/test_cli_benchmark.py
git commit -m "feat: publish reference chart manifest"
```

---

## Final Review Checklist

- [ ] Manifest timestamps round-trip exactly.
- [ ] Manifest rows reconstruct existing HPA-321 domain types.
- [ ] HPA-322 uses `validate_cached_body` through one thin adapter.
- [ ] HPA-323 can import the reader and verifier without refactoring HPA-322.
- [ ] Selection follows `set.def` authored order and quarantines ambiguity.
- [ ] No network, database, service, or concurrency scope was added.
- [ ] Full tests and Ruff CI gates pass.
