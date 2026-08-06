# HPA-322: Authoritative Benchmark Chart Selection Design

## Context

HPA-321 inventories the authoritative `simfile-dtx` R2 corpus, caches `set.def`,
`.dtx`, and `.txt` bodies in a content-addressed store, and publishes an immutable
JSONL manifest. HPA-322 must consume that exact artifact and select the intended
high-fidelity reference chart for every simfile.

The source-of-truth order is:

1. the HPA-321 manifest and content-addressed cache identify exact R2 objects;
2. the canonical `set.def` defines authored level slots and referenced DTX files;
3. the selected DTX defines chart content and metadata;
4. D1 or GraphQL may enrich display metadata but cannot override those files.

HPA-322 is also the first downstream stage that reads an HPA-321 manifest back into
domain objects. It must establish the small shared read/verification contract that
HPA-323 will reuse, rather than implementing a private cache verifier that is later
refactored.

## Goals

- Load HPA-321 rows into the existing `RemoteObject`, `SimfileInventory`, and
  `SyncError` types.
- Add the exact inverse of `format_manifest_timestamp` beside that formatter.
- Reuse `validate_cached_body` and the hardened content-addressed cache access path;
  do not hand-roll a second path/stat/hash verifier.
- Select the highest populated and resolvable `set.def` slot, evaluating L5 through
  L1.
- Support root and nested `set.def` files and relative chart references.
- Prefer exact object-key matches, then one unique case-insensitive match.
- Parse common DTXMania encodings and retain `#DLEVEL`, title, and artist.
- Support explicit version-controlled overrides for exceptional songs.
- Quarantine ambiguity or unusable source files instead of guessing.
- Publish a deterministic immutable `crux.reference-chart-manifest/v1` artifact.
- Keep all selection work offline, sequential, and reusable by HPA-323.

## Non-goals

- Timing, BPM integration, channel `01` BGM alignment, or playable-lane mapping.
- Model inference, source separation, scoring, or musical-quality judgment.
- Downloading missing files or contacting R2.
- Repairing malformed authored files.
- Adding a database, service, workflow framework, plugin system, or concurrency.
- Replacing HPA-321 cache durability or security semantics.

## Operator Interface

```bash
uv run crux benchmark select-reference-charts \
  --manifest artifacts/benchmark/r2-corpus/manifests/<sha256>.jsonl \
  --cache-dir artifacts/benchmark/r2-corpus/cache \
  --overrides-file config/benchmark-reference-chart-overrides.json \
  --output-dir artifacts/benchmark/reference-charts
```

- `--manifest PATH` is required.
- `--cache-dir PATH` defaults to `manifest.parent.parent / "cache"`.
- `--overrides-file PATH` defaults to
  `config/benchmark-reference-chart-overrides.json`. An omitted missing default is
  treated as the canonical empty override document; an explicitly supplied missing
  file is fatal.
- `--output-dir PATH` defaults to `artifacts/benchmark/reference-charts`.

The command performs no network access and emits one JSON summary. Exit `0` means all
rows selected, exit `1` means a partial manifest with quarantines, and exit `2` means
no usable manifest was produced.

## Shared Manifest and Cache Contracts

### Timestamp parsing

Add the exact inverse of `format_manifest_timestamp` to
`src/benchmark/r2_corpus_models.py`:

```python
def parse_manifest_timestamp(value: object) -> datetime: ...
```

It accepts only the UTC format emitted by HPA-321 and returns a timezone-aware UTC
`datetime`. Keeping both directions together makes the serialized contract explicit.

### Manifest row reconstruction

Add to `src/benchmark/corpus_manifest.py` beside `_build_row`:

```python
def inventory_from_manifest_row(
    row: Mapping[str, object],
) -> SimfileInventory: ...
```

The loader validates and reconstructs:

- `RemoteObject` entries in manifest order;
- row and object `SyncError` values;
- timezone-aware `last_modified` values;
- `cache_status`, SHA-256, ETag, size, and content type;
- one `SimfileInventory` with exact `simfile_id`, prefix, and sync status.

Duplicate object keys, malformed arrays, invalid enum values, timestamps, or error
references are rejected as malformed input. This is the precise inverse seam of
HPA-321 manifest construction, not a new parallel inventory model.

### Verified cache-body resolution

Add a thin adapter to `src/benchmark/corpus_cache.py`:

```python
def resolve_verified_cache_body(
    cache_dir: Path,
    remote: RemoteObject,
    *,
    source_endpoint_sha256: str,
    bucket: str,
    expected_sha256: str | None = None,
) -> Path: ...
```

The adapter:

1. requires `remote.cache_status == "verified"` and a valid digest;
2. builds a `CacheIndexEntry` from manifest/domain fields;
3. delegates verification to existing `validate_cached_body`;
4. maps every non-verified state to a safe `ValueError`;
5. optionally requires equality with `expected_sha256`;
6. returns the canonical content-addressed path
   `cache_dir / "sha256" / digest[:2] / digest`.

It does not reimplement path containment, symlink handling, file opening, binding
checks, size checks, or hashing. HPA-322 and HPA-323 must use this same adapter.

## DTXMania Text and Metadata

`src/benchmark/dtx_text.py` owns BOM-aware decoding shared by DTX and `set.def`:
UTF-8/UTF-16 BOMs first, then UTF-8, CP932, and Shift-JIS without a BOM. A decode is
accepted only when recognizable `#` or `*` directives exist.

`ParsedDtxChart` gains:

```text
dlevel_raw
dlevel_normalized
```

Normalization accepts only ASCII decimal integers in `0..100`; invalid values remain
visible as parser warnings.

## `set.def` Parsing

`src/benchmark/set_def_parser.py` parses only `L1..L5` `LABEL` and `FILE`
directives. It accepts `#` or `*`, colon or whitespace forms, optional spaces, quoted
values, and common DTXMania encodings. Duplicate fields use the last source occurrence
and emit a warning.

## Selection Engine

`src/benchmark/reference_chart_selection.py` receives one reconstructed
`SimfileInventory`, the cache root, source identity, and loaded overrides. It owns only
selection policy:

- canonical `set.def` discovery;
- safe R2 object-key resolution;
- L5-to-L1 authored-slot evaluation;
- explicit override application;
- conservative fallback when `set.def` is unusable;
- row-level selected/quarantined results.

Every source body is opened through `resolve_verified_cache_body`.

### Canonical `set.def`

1. Prefer a root-level `set.def`.
2. If multiple root candidates exist, prefer one exact lowercase basename only when
   unique; otherwise quarantine.
3. Without a root candidate, choose the unique shallowest nested copy.
4. Ties at the same depth quarantine as `ambiguous_set_def`.

### Authored slot resolution

Evaluate L5, L4, L3, L2, L1. For each populated file value:

1. normalize backslashes to `/`;
2. resolve relative to the selected `set.def` directory;
3. reject absolute or escaping paths;
4. prefer an exact object key;
5. otherwise allow one unique case-insensitive match;
6. continue to the next slot only when the referenced object is absent;
7. quarantine if an existing selected file cannot be verified, decoded, or parsed.

Filenames such as `mas.dtx`, `real.dtx`, and `full.dtx` do not define authored order.

### Overrides

The versioned override document maps a decimal `simfile_id` to an exact chart key and
required reason. Invalid overrides quarantine the row and never silently fall back.
The exact override bytes are hashed into every output row.

### Fallback

When no usable `set.def` exists:

1. one valid DTX candidate wins;
2. otherwise the unique highest numeric `#DLEVEL` wins;
3. equal highest levels may use the narrow rank `real.dtx > full.dtx > mas.dtx` only
   when it yields one recognized winner;
4. all remaining ties quarantine.

Alphabetical order is never a selection rule.

## Derived Manifest

`src/benchmark/reference_chart_manifest.py` reads exact source bytes once, calculates
`source_manifest_sha256`, validates one HPA-321 schema/corpus version and unique IDs,
reconstructs inventories, applies the selector, and publishes through existing
`render_manifest`, `publish_manifest`, and `publish_latest_manifest` helpers.

Output schema: `crux.reference-chart-manifest/v1`.

Each row preserves HPA-321 inventory fields and adds selection status, method, reason
codes, warnings, source lineage, selected chart identity/hash, authored slot/label,
DTX metadata, and override identity. Quarantined rows keep selected fields `null`.

## Error Handling

Fatal errors:

- malformed/unsupported input manifest;
- duplicate simfile IDs or mixed corpus versions;
- malformed override document;
- immutable publication failure.

Row quarantines include:

- `source_inventory_unusable`;
- `cached_body_unavailable`;
- `no_verified_dtx`;
- `ambiguous_set_def`;
- `referenced_chart_missing`;
- `ambiguous_chart_key`;
- `selected_chart_invalid`;
- `override_invalid`;
- `ambiguous_fallback`.

One bad simfile never discards successful rows.

## Testing

Tests cover:

- timestamp format/parse round trips;
- `build_manifest_rows` -> JSON row -> `inventory_from_manifest_row` symmetry;
- hardened cache verification through `validate_cached_body`, including missing,
  corrupt, mismatched, and expected-hash cases;
- DTX/set.def encoding and grammar;
- authored slot, path, case, override, fallback, and quarantine policies;
- deterministic immutable manifest publication;
- CLI exit `0`, `1`, and `2`;
- an offline acceptance fixture using a real HPA-321-shaped manifest/cache.

CI validation uses `pytest`, `ruff check`, and `ruff format --check`.

## Delivery Sequence

1. Establish shared timestamp, manifest-reader, and cache-body contracts.
2. Extract shared DTXMania decoding and retain DLEVEL metadata.
3. Add the focused `set.def` parser.
4. Implement selection and overrides against reconstructed inventories.
5. Publish the derived manifest and wire the CLI acceptance path.

HPA-323 consumes these merged contracts and must not refactor HPA-322 to obtain them.
