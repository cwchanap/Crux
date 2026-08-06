# HPA-322: Authoritative Benchmark Chart Selection Design

## Context

HPA-321 inventories the authoritative `simfile-dtx` R2 corpus, caches `set.def`,
`.dtx`, and `.txt` bodies in a content-addressed store, and publishes an immutable
JSONL manifest. HPA-322 consumes that exact artifact and selects the intended
high-fidelity reference chart for every simfile.

The source-of-truth order is:

1. the HPA-321 manifest and verified cache identify exact R2 objects;
2. the canonical `set.def` defines authored level slots and referenced chart files;
3. the selected `.dtx` or `.txt` file defines chart content and metadata;
4. D1 or GraphQL may enrich display metadata but cannot override those files.

HPA-322 is also the first downstream stage that reads HPA-321 rows back into domain
objects. It therefore owns the small manifest, cache-body, and object-key contracts
that HPA-323 immediately reuses.

## Goals

- Read HPA-321 rows without inventing a parallel inventory model.
- Preserve provenance and source identity while reconstructing `SimfileInventory`.
- Reuse `validate_cached_body`; do not add a second filesystem verifier.
- Resolve inventory object keys through one shared pure helper used by HPA-322 and
  HPA-323.
- Select the highest populated and resolvable authored slot, evaluating L5 through L1.
- Support root and nested `set.def` files, relative references, and a documented
  simfile-root compatibility fallback.
- Treat `.dtx` and `.txt` as chart candidates consistently with HPA-321.
- Parse common DTXMania encodings and retain `#DLEVEL`, title, and artist.
- Support explicit version-controlled overrides.
- Quarantine ambiguity or unusable source files instead of guessing.
- Publish a closed, deterministic `crux.reference-chart-manifest/v1` contract.
- Keep the stage offline and sequential.

## Non-goals

- Timing, BPM integration, channel `01` BGM alignment, or playable-lane mapping.
- Model inference, source separation, scoring, or musical-quality judgment.
- Downloading missing files or contacting R2.
- Repairing malformed authored files.
- Adding a database, service, workflow framework, plugin system, or concurrency.
- Unifying the legacy local-folder `_select_chart` path.
- Backward-compatibility readers or schema migrations.

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
  `config/benchmark-reference-chart-overrides.json`.
- If the default override path is omitted and missing, use the exact canonical bytes
  `{"overrides":{},"schema_version":"crux.reference-chart-overrides/v1"}\n`.
- An explicitly supplied missing override file is fatal.
- `--output-dir PATH` defaults to `artifacts/benchmark/reference-charts`.

The command performs no network access and emits one canonical JSON summary.

Exit codes are exact:

- `0`: a manifest was published and every non-empty input row is selected;
- `1`: a manifest was published with one or more quarantined rows, including when all
  input rows are quarantined;
- `2`: the input has zero rows, loading/schema validation fails, or publication fails.

## Shared Manifest, Cache, and Key Contracts

### Timestamp parsing

Add beside `format_manifest_timestamp` in `src/benchmark/r2_corpus_models.py`:

```python
def parse_manifest_timestamp(value: object) -> datetime: ...
```

It accepts only the UTC forms emitted by HPA-321 and returns an aware UTC `datetime`.

### Typed HPA-321 row view

HPA-321 serializes inventory, provenance, and source identity in one row. Only the
inventory portion maps to `SimfileInventory`, so the reader exposes both a complete row
view and the smaller compatibility helper:

```python
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
) -> ManifestRowView: ...

def inventory_from_manifest_row(
    row: Mapping[str, object],
) -> SimfileInventory: ...
```

`inventory_from_manifest_row` delegates to `manifest_row_view_from_row(...).inventory`.

The reader validates the exact HPA-321 row and object key sets. It reconstructs:

- `RemoteObject` entries in manifest order;
- timezone-aware `last_modified` values;
- `cache_status`, digest, ETag, size, content type, and cache path;
- all serialized `sync_errors` on `SimfileInventory.sync_errors`;
- provenance and source identity on `ManifestRowView`.

HPA-321 does not serialize `RemoteObject.errors`. Reconstructed objects therefore have
`errors=()`. Selection must use `cache_status`,
`SimfileInventory.sync_errors`, and `resolve_verified_cache_body`, never reconstructed
object errors.

A round-trip test rebuilds the HPA-321 base row with `build_manifest_rows` and compares
it with the validated source row after removing only `corpus_version`. Provenance is
asserted separately through `ManifestRowView`.

### Verified cache-body resolution

Add to `src/benchmark/corpus_cache.py`:

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

The adapter requires:

- `remote.cache_status == "verified"`;
- a lowercase 64-character SHA-256;
- `remote.cache_path == f"sha256/{digest[:2]}/{digest}"`;
- equality with `expected_sha256` when supplied.

It constructs:

```python
CacheIndexEntry(
    source_endpoint_sha256=source_endpoint_sha256,
    bucket=bucket,
    key=remote.key,
    etag=remote.etag,
    etag_is_weak=remote.etag_is_weak,
    size=remote.size,
    last_modified=format_manifest_timestamp(remote.last_modified),
    sha256=digest,
    cache_path=f"sha256/{digest[:2]}/{digest}",
)
```

It then calls `validate_cached_body(cache_dir, entry)`. Every state other than
`verified`, plus any missing/mismatched manifest field, raises
`ValueError("verified cache body unavailable")`. The function returns
`cache_dir / entry.cache_path` and never calls `Path.resolve` or implements another
hashing path.

### Shared inventory object-key resolution

Create `src/benchmark/inventory_object_keys.py`:

```python
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
) -> ResolvedObjectKey: ...
```

The helper performs no filesystem I/O. It:

1. converts `\` to `/`;
2. rejects empty values, NUL, absolute paths, UNC paths, and drive prefixes;
3. resolves `.` and `..` against the supplied POSIX object-key directory;
4. rejects a result outside `object_prefix`;
5. returns an exact key match when present;
6. otherwise returns one unique `str.casefold()` match;
7. distinguishes missing, ambiguous, and invalid paths.

HPA-322 calls it for `set.def` chart references. HPA-323 calls the same helper for
`#WAVxx` values and may perform a second call against the simfile root only when its
measured policy retains root fallback.

## DTXMania Text and Metadata

Create `src/benchmark/dtx_text.py`:

```python
DtxTextKind = Literal["dtx", "set_def"]

def decode_dtxmania_text(
    raw: bytes,
    *,
    source_name: str,
    kind: DtxTextKind,
) -> str: ...
```

Decoding order is intentional:

1. UTF-8 BOM;
2. UTF-16 LE or BE BOM;
3. plain UTF-8;
4. CP932;
5. Shift-JIS.

Acceptance is not merely “a line begins with `#` or `*`”:

- `kind="dtx"` requires at least one line matching the current DTX directive shape
  `^[#*]\s*[0-9A-Za-z]`;
- `kind="set_def"` requires at least one case-insensitive
  `L1..L5` `LABEL` or `FILE` directive.

Existing UTF-8, Shift-JIS, UTF-16, star-prefix, and malformed-file parser tests must
retain their current outcomes. Add a CP932-only filename fixture so the new capability
is deliberate rather than an untested codec-order change. Exceptions identify the
source name but never include body contents.

`ParsedDtxChart` gains:

```text
dlevel_raw: str | None
dlevel_normalized: int | None
```

Normalization accepts only ASCII decimal integers in `0..100`; invalid values remain
visible as parser warnings.

## `set.def` Parsing

`src/benchmark/set_def_parser.py` parses only `L1..L5` `LABEL` and `FILE`
directives. It accepts `#` or `*`, colon or whitespace forms, optional spaces, quoted
values, and common DTXMania encodings. Duplicate fields use the last source occurrence
and emit a warning.

The result always contains exactly five slots:

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
```

`FILE` values may reference `.dtx` or `.txt` files, case-insensitively.

## Selection Contracts

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
```

Invariants:

- `selected` has a method, selected chart, no reason codes, and non-null string title
  and artist (empty strings are allowed).
- `quarantined` has `method=None`, no selected chart, and at least one reason code.
- `set_def` may be retained on a quarantined result when it was verified and parsed
  before a later failure.
- `override` is populated whenever the row had a structurally valid override entry,
  even if the referenced chart makes the row `override_invalid`.

## Selection Engine

`src/benchmark/reference_chart_selection.py` receives one `ManifestRowView`, the cache
root, and `LoadedOverrides`. It owns only selection policy. Every source body is opened
through `resolve_verified_cache_body`.

### Canonical `set.def`

1. Prefer a root-level `set.def`.
2. If multiple root candidates exist, prefer one exact lowercase basename only when
   unique; otherwise quarantine.
3. Without a root candidate, choose the unique shallowest nested copy.
4. Ties at the same depth quarantine as `ambiguous_set_def`.
5. An existing canonical candidate that cannot be verified or parsed quarantines; it
   does not silently fall through to another copy.

### Authored slot resolution

Evaluate L5, L4, L3, L2, L1. For each populated `FILE` value:

1. call `resolve_inventory_object_key` relative to the selected `set.def` directory;
2. if the result is `missing` and the `set.def` is nested, call the same helper against
   the simfile root and record warning `set_def_root_fallback` when it resolves;
3. reject `invalid_path`;
4. quarantine `ambiguous`;
5. continue to the next slot only when both allowed lookups are `missing`;
6. require a `.dtx` or `.txt` suffix, case-insensitively;
7. verify and parse the existing chart;
8. quarantine an existing chart that cannot be verified, decoded, or parsed.

Filenames such as `mas`, `real`, and `full` never define authored slot order.

### Overrides

The versioned override document maps a canonical decimal `simfile_id` to an exact chart
key and non-empty reason. The exact key must identify a verified `.dtx` or `.txt`
object. Invalid row-specific overrides quarantine and never fall back. The exact
override bytes are hashed into every output row.

### Fallback

When no usable `set.def` exists, candidates are verified objects whose key ends with
`.dtx` or `.txt`, case-insensitively.

1. One parseable candidate wins as `single_candidate_fallback`.
2. Otherwise the unique highest numeric `#DLEVEL` wins as `dlevel_fallback`.
3. At the same highest level, compare case-insensitive basenames without suffix using
   the narrow rank `real > full > mas`.
4. Use `filename_tiebreak_fallback` only when that rank yields exactly one recognized
   winner.
5. All remaining ties quarantine as `ambiguous_fallback`.

Alphabetical order is never a selection rule.

## Derived Manifest Contract

`src/benchmark/reference_chart_manifest.py` reads exact source bytes once, calculates
`source_manifest_sha256`, requires one non-empty HPA-321 schema/corpus version, rejects
duplicate IDs, loads `ManifestRowView` values, applies selection, and publishes through
`render_manifest`, `publish_manifest`, and `publish_latest_manifest`.

Output schema: `crux.reference-chart-manifest/v1`.

Every row contains the complete HPA-321 field set:

```text
schema_version
corpus_version
cache_profile
simfile_id
object_prefix
source_endpoint_sha256
source_bucket
source_discovery_method
objects
sync_status
sync_errors
source_origin
source_author_or_pack
source_reference
rights_status
redistribution_allowed
provenance_notes
```

Every row also contains this exact selection field set:

| Field | Type and nullability |
|---|---|
| `source_manifest_sha256` | lowercase 64-character string |
| `source_corpus_version` | `sha256:<64 lowercase hex>` |
| `selection_status` | `selected` or `quarantined` |
| `selection_method` | `SelectionMethod` or `null` |
| `selection_reason_codes` | array of `SelectionReasonCode`; empty only when selected |
| `selection_warnings` | string array |
| `set_def_key` | string or `null` |
| `set_def_content_hash` | lowercase SHA-256 or `null` |
| `selected_chart_key` | string or `null` |
| `selected_chart_content_hash` | lowercase SHA-256 or `null` |
| `selected_chart_cache_path` | canonical relative cache path or `null` |
| `selected_level_slot` | `L1` through `L5`, or `null` |
| `selected_level_label` | string or `null` |
| `dlevel_raw` | string or `null` |
| `dlevel_normalized` | integer `0..100` or `null` |
| `title` | string or `null`; selected rows use a string |
| `artist` | string or `null`; selected rows use a string |
| `override_document_sha256` | lowercase 64-character string |
| `selection_override` | `{"chart_key": string, "reason": string}` or `null` |

Selected example:

```json
{"artist":"Example Artist","cache_profile":"setdef_dtx_txt_v1","corpus_version":"sha256:5cfad8e9a015fbfa0dd3e89b8a54f0ee42fd17a5bfa212f5bbcf9d98f5b126cc","dlevel_normalized":99,"dlevel_raw":"99","object_prefix":"42/","objects":[{"cache_path":"sha256/bb/bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb","cache_status":"verified","content_type":"text/plain","etag":"setdef-etag","etag_is_weak":false,"key":"42/set.def","last_modified":"2026-08-05T00:00:00Z","sha256":"bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb","size":80,"version":null},{"cache_path":"sha256/aa/aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa","cache_status":"verified","content_type":"text/plain","etag":"chart-etag","etag_is_weak":false,"key":"42/real.dtx","last_modified":"2026-08-05T00:00:01Z","sha256":"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa","size":123,"version":null}],"override_document_sha256":"eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee","provenance_notes":null,"redistribution_allowed":null,"rights_status":"unknown","schema_version":"crux.reference-chart-manifest/v1","selected_chart_cache_path":"sha256/aa/aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa","selected_chart_content_hash":"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa","selected_chart_key":"42/real.dtx","selected_level_label":"REAL","selected_level_slot":"L5","selection_method":"set_def_slot","selection_override":null,"selection_reason_codes":[],"selection_status":"selected","selection_warnings":[],"set_def_content_hash":"bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb","set_def_key":"42/set.def","simfile_id":42,"source_author_or_pack":null,"source_bucket":"simfile-dtx","source_corpus_version":"sha256:cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc","source_discovery_method":"r2_list_objects_v2","source_endpoint_sha256":"ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff","source_manifest_sha256":"9999999999999999999999999999999999999999999999999999999999999999","source_origin":null,"source_reference":null,"sync_errors":[],"sync_status":"complete","title":"Example Song"}
```

Quarantined example:

```json
{"artist":null,"cache_profile":"setdef_dtx_txt_v1","corpus_version":"sha256:5cfad8e9a015fbfa0dd3e89b8a54f0ee42fd17a5bfa212f5bbcf9d98f5b126cc","dlevel_normalized":null,"dlevel_raw":null,"object_prefix":"43/","objects":[{"cache_path":"sha256/11/1111111111111111111111111111111111111111111111111111111111111111","cache_status":"verified","content_type":"text/plain","etag":"root-etag","etag_is_weak":false,"key":"43/set.def","last_modified":"2026-08-05T00:00:00Z","sha256":"1111111111111111111111111111111111111111111111111111111111111111","size":70,"version":null},{"cache_path":"sha256/22/2222222222222222222222222222222222222222222222222222222222222222","cache_status":"verified","content_type":"text/plain","etag":"other-etag","etag_is_weak":false,"key":"43/SET.DEF","last_modified":"2026-08-05T00:00:01Z","sha256":"2222222222222222222222222222222222222222222222222222222222222222","size":75,"version":null}],"override_document_sha256":"eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee","provenance_notes":null,"redistribution_allowed":null,"rights_status":"unknown","schema_version":"crux.reference-chart-manifest/v1","selected_chart_cache_path":null,"selected_chart_content_hash":null,"selected_chart_key":null,"selected_level_label":null,"selected_level_slot":null,"selection_method":null,"selection_override":null,"selection_reason_codes":["ambiguous_set_def"],"selection_status":"quarantined","selection_warnings":[],"set_def_content_hash":null,"set_def_key":null,"simfile_id":43,"source_author_or_pack":null,"source_bucket":"simfile-dtx","source_corpus_version":"sha256:cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc","source_discovery_method":"r2_list_objects_v2","source_endpoint_sha256":"ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff","source_manifest_sha256":"9999999999999999999999999999999999999999999999999999999999999999","source_origin":null,"source_reference":null,"sync_errors":[],"sync_status":"complete","title":null}
```

Add one two-row canonical JSONL golden at
`tests/benchmark/schema_goldens/reference-chart-manifest.jsonl`, register
`crux.reference-chart-manifest/v1` in the existing golden manifest, and expose
`validate_schema_golden` from `reference_chart_manifest.py`. Structural mutation tests
must reject removed, added, duplicate, and mistyped fields.

## Error Handling

Fatal errors:

- empty, malformed, or unsupported input manifest;
- duplicate simfile IDs or mixed corpus versions/source identities;
- malformed override document;
- immutable publication failure.

Row quarantine uses the closed `SelectionReasonCode` set. One bad simfile never
discards successful rows.

## Testing

Tests cover:

- timestamp format/parse round trips;
- HPA-321 row-view reconstruction and truthful object-error behavior;
- hardened cache verification and exact `CacheIndexEntry` construction;
- shared object-key normalization, containment, exact, casefold, missing, and ambiguity;
- DTX/set.def codec predicates and existing parser regression fixtures;
- `.dtx` and `.txt` authored slots, overrides, and fallback;
- closed selected/quarantined schema goldens;
- deterministic immutable publication;
- byte-identical output and identical derived identity on a second run;
- override-byte changes altering `override_document_sha256`, row bytes, and derived
  identity;
- CLI exit `0`, `1`, and `2`, including all-quarantined exit `1`;
- an offline acceptance fixture using a real HPA-321-shaped manifest/cache.

CI validation uses `pytest`, `ruff check`, `ruff format --check`, and the repository's
enabled Pylint command.

## Risk Verification Matrix

| Risk | Required proof |
|---|---|
| A wrong chart is selected silently | Authored-slot, containment, casefold, fallback, and quarantine fixtures |
| Decoder extraction changes chart identity | Existing decoder suite plus CP932-only and set.def predicate fixtures |
| HPA-323 forks the row or key contract | HPA-323 imports the HPA-322 row view, verifier, and object-key resolver |
| Derived schema drifts between stages | Closed field set, two-row schema golden, and structural mutation tests |
| A rerun changes identity without input changes | Byte-identical two-run acceptance assertion |
| Override changes are not represented | Exact override-byte hash and changed-identity assertion |

## Delivery Sequence

1. Establish shared timestamp, typed row-view, cache-body, and object-key contracts.
2. Extract shared DTXMania decoding and retain DLEVEL metadata.
3. Add the focused five-slot `set.def` parser.
4. Implement selection and overrides against reconstructed inventories.
5. Freeze the row schema, publish the derived manifest, and wire the CLI acceptance
   path.

HPA-323 consumes these merged contracts and does not reimplement them.
