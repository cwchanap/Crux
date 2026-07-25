# HPA-321: R2 Corpus Inventory, Cache, and Manifest Design

## Context

HPA-321 connects Crux to the authoritative `simfile-dtx` Cloudflare R2 bucket and
produces a reproducible base inventory for later benchmark stages. R2 object contents
are first-hand source data. D1 or GraphQL may help other stages discover descriptive
metadata, but neither may determine chart truth or override R2 files.

The current bucket convention is one numeric simfile prefix per song:

```text
<simfile_id>/
```

Objects may be nested beneath that prefix, and keys may contain non-ASCII characters,
spaces, or other special characters. The initial corpus is approximately 400 songs.

This design also incorporates HPA-321's provenance and rights addendum. Current private
or personally authorized source material may be inventoried immediately. Future
community additions must retain their source and rights context rather than being
merged anonymously.

## Goals

- Inventory every object under each valid simfile prefix, including nested keys.
- Preserve exact R2 keys and collect size, ETag or version signal, modification time,
  content type, and available remote checksums.
- Cache only bodies needed by the current stage. The initial profile caches
  case-insensitive `set.def`, `.dtx`, and `.txt` files.
- Verify cached bodies with locally computed SHA-256 digests.
- Skip unchanged verified cache entries on repeated synchronization.
- Produce deterministic, content-addressed JSONL manifests without overwriting
  historical manifests.
- Produce a machine-readable report for successful, partial, dry-run, and failed
  synchronization attempts.
- Support include and exclude filters for small pilots.
- Keep credentials, signed request details, and copyrighted audio bodies out of
  manifests and logs.
- Retain optional provenance and rights metadata in a version-controlled mapping.

## Non-goals

- Selecting the authoritative chart from `set.def`.
- Parsing DTX timing or determining the benchmark audio source.
- Running transcription, scoring, or stem separation.
- Uploading derived artifacts to R2.
- Mirroring every R2 object body during this stage.
- Using D1 or GraphQL as an authority for simfile contents.
- Publishing or embedding copyrighted song audio in reports.

## Operator Interface

Crux gains one transactional command:

```bash
uv run crux benchmark sync-r2-corpus \
  --provenance-file config/corpus-provenance.json \
  --include-simfile-id 42 \
  --exclude-simfile-id 99 \
  --dry-run
```

The command accepts:

- repeated `--include-simfile-id INTEGER` filters;
- repeated `--exclude-simfile-id INTEGER` filters;
- `--output-dir PATH`, defaulting to `artifacts/benchmark/r2-corpus/`;
- `--cache-dir PATH`, defaulting to `<output-dir>/cache/`;
- optional `--provenance-file PATH`;
- `--dry-run` for list-only planning.

An ID present in both filters is excluded. Invalid or negative IDs are rejected before
network access. An explicitly included ID whose prefix has no objects receives an
`empty` manifest row in a real sync and an empty-prefix entry in a dry-run report.

The HPA-321 selection profile is the fixed identifier `setdef_dtx_txt_v1`. There is no
`--cache-profile` option in v1. The identifier is recorded in manifests and reports so
later issues can introduce different profiles without silently changing this contract.

Configuration uses:

- `CRUX_R2_ENDPOINT_URL` for the S3-compatible R2 endpoint;
- `CRUX_R2_BUCKET`, defaulting to `simfile-dtx`;
- `AWS_ACCESS_KEY_ID`;
- `AWS_SECRET_ACCESS_KEY`;
- optional `AWS_SESSION_TOKEN`;
- `CRUX_R2_HEAD_CONCURRENCY`, default `8`, constrained to `1..32`;
- `CRUX_R2_DOWNLOAD_CONCURRENCY`, default `4`, constrained to `1..16`;
- `CRUX_R2_CONNECT_TIMEOUT_SECONDS`, default `10`;
- `CRUX_R2_READ_TIMEOUT_SECONDS`, default `60`;
- `CRUX_R2_MAX_ATTEMPTS`, default `5`, counting the initial request;
- region `auto`.

Crux does not expose credential-valued command-line options. The implementation uses
the standard AWS credential provider chain so temporary credentials and other secure
provider-chain sources remain possible. Configuration validation may name a missing
variable, but it must never print its value.

Crux accepts an HTTPS origin with no user information, query, or fragment. It
lowercases the scheme and hostname, removes the default port and trailing slash, and
computes `source_endpoint_sha256` from the normalized ASCII origin. The hash
distinguishes same-named buckets in different R2 accounts without storing or logging
the account-bearing endpoint itself.

Command exit codes are:

- `0` for a complete real sync or `dry_run_complete`;
- `1` for a published partial manifest or `dry_run_partial`;
- `2` for a fatal failure that publishes no manifest.

The report's `overall_status` remains the machine-readable result; the exit code is its
coarse process-level projection.

The default artifact layout is:

```text
artifacts/benchmark/r2-corpus/
├── cache/
│   ├── .index-v1.lock
│   ├── index-v1.json
│   └── sha256/<first-two-hex>/<full-sha256>
├── manifests/<manifest-sha256>.jsonl
├── reports/<UTC-YYYYMMDDTHHMMSS.ffffffZ>-<run-id>.json
├── latest.json
└── latest-report.json
```

## Architecture

The command coordinates four focused components.

### R2 inventory

`src/benchmark/r2_inventory.py` owns an R2 client protocol and the optional `boto3`
implementation. It:

- paginates `ListObjectsV2` without a delimiter so nested objects are returned;
- preserves object keys exactly as returned;
- groups digit-only top-level path segments into simfile prefixes;
- detects malformed root keys and ambiguous numeric aliases;
- performs bounded-concurrency `HeadObject` requests to obtain content type and other
  metadata unavailable from the listing;
- returns domain records rather than leaking raw SDK response objects.

The default HEAD concurrency is `8`; selected-body downloads use a separate default
concurrency of `4`. The botocore connection pool is at least `16` and never smaller
than the sum of both concurrency settings.

The SDK client and its imports are loaded only when this command runs. If the `r2`
optional dependency is absent, the command fails before configuration or network
access with `missing_optional_dependency` and a sanitized installation hint. The R2
protocol, domain records, canonicalization code, and fakes remain importable without
`boto3`. Tests use a fake implementation of the protocol and do not require
credentials or network access.

### Network policy

The botocore client uses:

- a 10-second connect timeout;
- a 60-second read timeout;
- TCP keepalive;
- `standard` retry mode;
- five total attempts per request, including the initial attempt;
- botocore's standard exponential backoff for retryable throttling, connection, and
  transient service failures.

The explicit standard mode is preferred over adaptive mode because its retry behavior
does not add client-wide rate-throttling state that can make bounded worker timing
surprising. Crux does not wrap SDK calls in another general retry loop, which avoids
multiplying the configured attempt count. After SDK retries are exhausted, root-list
failures are fatal while per-object HEAD or GET failures follow the partial-failure
policy. Conditional `412 Precondition Failed` responses are not retried because they
mean the inventory snapshot changed.

### Corpus cache

`src/benchmark/corpus_cache.py` owns download selection, the local cache index, body
verification, and atomic installation. It has no chart-selection logic. The fixed
`setdef_dtx_txt_v1` selection rule is:

- basename equals `set.def`, case-insensitively; or
- key suffix is `.dtx` or `.txt`, case-insensitively.

Future stages may add explicit cache profiles without changing the inventory format or
invalidating already verified content-addressed bodies.

### Corpus provenance

`src/benchmark/corpus_provenance.py` validates and loads an optional
version-controlled JSON mapping keyed by decimal simfile ID. It does not read D1 or
GraphQL.

Missing mapping entries are allowed and receive explicit unknown values. Invalid field
types, duplicate IDs after numeric normalization, malformed JSON, or any
`schema_version` other than the exact supported `crux.corpus-provenance/v1` value are
fatal configuration errors.

### Corpus manifest

`src/benchmark/corpus_manifest.py` converts domain records to canonical JSONL,
computes deterministic identities, publishes immutable manifests, writes sync reports,
and atomically updates the convenience pointer. `src/cli/benchmark.py` owns only
command wiring and presentation.

Existing `prepare-corpus` behavior remains unchanged. HPA-322 may later consume the
new manifest and cache, but no chart selection or DTX parsing belongs in these
components.

## Inventory and Prefix Rules

The authoritative discovery method is a complete R2 root listing.

For a key such as:

```text
42/assets/snare 01.ogg
```

the top-level segment `42` identifies simfile `42`, and the exact object prefix is
`42/`.

Rules:

1. A valid simfile prefix has a non-negative, digit-only first segment followed by `/`.
2. The manifest stores `simfile_id` as an integer and `object_prefix` as the exact
   source string.
3. A zero-byte folder marker whose key equals the prefix is inventory metadata but is
   not selected for caching. A prefix containing only that marker is `empty`, retains
   the marker as a `not_selected` object, makes the overall attempt `partial`, and
   produces exit `1`.
4. If distinct prefixes normalize to the same integer, such as `1/` and `01/`, both
   are quarantined as an ambiguous prefix rather than silently merged. Quarantine wins
   even when the operator explicitly passes `--include-simfile-id 1`; the report uses
   `ambiguous_simfile_prefix`, names the requested ID and conflicting prefixes, and no
   simfile row is emitted.
5. Root objects without a slash and non-numeric top-level segments are reported as
   malformed root keys. They do not disappear and do not become simfile rows.
6. Include and exclude filters are applied after authoritative root discovery. HPA-321
   does not substitute direct prefix listing because the complete root listing is what
   makes malformed and ambiguous prefixes observable.

Listing initially supplies `key`, `size`, `etag`, and `last_modified`. A successful
`HeadObject` response becomes authoritative for `size`, `etag`, and `last_modified`
when those fields are present, and supplements them with `content_type`, checksum
values, and checksum type. The normalized post-HEAD-merge values are the only values
used by both the cache index and manifest. If HEAD omits one of the identity fields,
the listing value is retained. Listing metadata is retained when `HeadObject` fails;
the affected object receives `object_head_failed`, cannot be selected for download,
and its simfile status becomes `partial`.

HEAD fan-out is intentional for this corpus scale: root listing alone does not return
the content type or checksum values required by the manifest. The bounded concurrency
and explicit timeouts cap its operational cost.

## Cache Identity and Synchronization

An R2 ETag is a remote change signal, not a universal content hash. Multipart ETags
encode part hashes and part count rather than the complete object's ordinary MD5.
Crux parses the returned ETag as an HTTP entity tag. It removes surrounding quotes,
stores the opaque value as `etag`, and stores weakness separately as
`etag_is_weak: true|false`; it never strips a leading `W/` without retaining that
flag. A malformed entity tag is an object metadata error.

The cache index records, per exact source endpoint, bucket, and key:

- source endpoint SHA-256;
- bucket name;
- exact object key;
- normalized ETag;
- ETag weakness;
- size;
- last-modified time;
- locally verified SHA-256;
- relative content-addressed cache path.

Raw endpoint URLs and credentials are not stored.

The index is canonical UTF-8 JSON at:

```text
<cache-dir>/index-v1.json
```

It has `schema_version: "crux.r2-cache-index/v1"` and an `entries` array sorted by
source endpoint SHA-256, bucket, and exact key. Each entry contains the fields listed
above; an array avoids inventing an escaping scheme for composite JSON object keys.
The index is intentionally profile-independent: it proves an exact remote object body
is present at a content-addressed path. The active cache profile is instead recorded
in the manifest and sync report, where selection policy affects corpus identity.

A real sync holds an exclusive advisory lock on `<cache-dir>/.index-v1.lock` while
reading and publishing index state. Index publication serializes the complete new
document to a uniquely named sibling temporary file, flushes and `fsync`s the file,
uses `os.replace` to install `index-v1.json`, and `fsync`s the cache directory where
the platform supports directory synchronization. Dry runs read the index without
rewriting it. The operating-system lock is released automatically when the process
exits; the lock file itself may remain harmlessly. A live lock conflict fails fast
with a sanitized `cache_locked` error. Invalid JSON or an unsupported cache-index
schema version is fatal rather than being treated as a cache miss.

The lock implementation is a non-blocking POSIX `fcntl` exclusive lock and supports
the macOS and Linux environments targeted by this CLI. A platform without compatible
`fcntl` locking fails before cache mutation with `unsupported_platform`; it must not
continue with an unlocked index.

For each selected object:

1. Compare key, ETag, size, and modification time with the index.
2. If they match, stream-hash the referenced local file.
3. Reuse the entry only when the computed SHA-256 and byte count match the index.
4. For a strong ETag, download to a temporary file with `If-Match` using the exact
   quoted entity tag reconstructed from `etag`.
5. For a weak ETag, do not send `If-Match`, because HTTP preconditions use strong
   comparison for that header. Perform an unconditional GET and require its normalized
   ETag, content length, and last-modified response metadata to equal the post-HEAD
   inventory identity. Missing comparison metadata produces `weak_etag_unverifiable`;
   changed metadata produces `source_changed_during_sync`.
6. Stream-compute SHA-256 and byte count during either download path.
7. Reject a byte-count mismatch, strong conditional-request failure, or weak-path
   response-metadata mismatch.
8. Atomically move valid content to:

   ```text
   sha256/<first-two-hex>/<full-sha256>
   ```

9. Atomically checkpoint the cache index after each successfully installed selected
   body and only after the content file is durable. Concurrent download workers
   serialize these short index-publication sections.

The cache is content-addressed and extensionless. Manifest metadata retains the source
key and media type, so the cache filename does not need to reproduce a potentially
unsafe object basename.

If the same key changes, its new content receives a new SHA-256 path. The old cache
file and every previously published manifest remain untouched. Orphaned cache cleanup
is outside HPA-321.

Resumability means safe re-execution, not a separate remote checkpoint protocol.
Successfully checkpointed cache entries survive interruption and become verified cache
hits on the next run. If a process stops after installing a content file but before
publishing its index entry, the next run may download that object again, then safely
deduplicate it to the same SHA-256 path.

## Provenance Mapping

The optional mapping uses this shape:

```json
{
  "schema_version": "crux.corpus-provenance/v1",
  "simfiles": {
    "42": {
      "source_origin": "personal",
      "source_author_or_pack": "Example Pack",
      "source_reference": "private archive",
      "rights_status": "privately_authorized",
      "redistribution_allowed": false,
      "provenance_notes": "Authorized for local benchmark use."
    }
  }
}
```

Manifest rows always contain:

- `source_origin`;
- `source_author_or_pack`;
- `source_reference`;
- `rights_status`;
- `redistribution_allowed`;
- `provenance_notes`.

For a missing entry, string fields are `null`, `rights_status` is `unknown`, and
`redistribution_allowed` is `null`. Unknown provenance is not a synchronization error
for the current corpus, but it remains visible to downstream policy checks.

`source_origin` and `rights_status` are descriptive free-text strings in v1 rather
than closed enums. Consumers must treat missing, `unknown`, or unrecognized values as
non-redistributable and may permit redistribution only when
`redistribution_allowed` is exactly `true`.

The mapping contains descriptions and rights assertions only. It must not contain
credentials, signed URLs, or source file bodies.

## Manifest Contract

The manifest is UTF-8 JSONL with one record per simfile. A representative record is:

```json
{"schema_version":"crux.r2-corpus-manifest/v1","corpus_version":"sha256:…","cache_profile":"setdef_dtx_txt_v1","simfile_id":42,"object_prefix":"42/","source_endpoint_sha256":"…","source_bucket":"simfile-dtx","source_discovery_method":"r2_list_objects_v2","objects":[{"key":"42/SET.DEF","size":1234,"etag":"…","etag_is_weak":false,"version":null,"last_modified":"2026-07-24T12:34:56Z","content_type":"text/plain","checksum_algorithms":["sha256"],"checksum_type":"full_object","remote_checksums":{"sha256":"base64-value"},"cache_status":"verified","sha256":"…","cache_path":"sha256/ab/ab…"}],"sync_status":"complete","sync_errors":[],"source_origin":null,"source_author_or_pack":null,"source_reference":null,"rights_status":"unknown","redistribution_allowed":null,"provenance_notes":null}
```

Each row includes `source_endpoint_sha256` and `source_bucket` so its remote source is
unambiguous without exposing the raw endpoint. Each row also includes the fixed
`cache_profile: "setdef_dtx_txt_v1"`, making the selection policy part of corpus
identity.

Each object contains:

- `key`;
- `size`;
- `etag`;
- `etag_is_weak`;
- `version`, reserved and always `null` in the HPA-321 S3 adapter because R2's S3
  compatibility does not expose bucket version-history semantics; supporting object
  versions later requires a new adapter/schema version rather than silently filling
  this v1 field;
- `last_modified` as UTC ISO 8601;
- `content_type`, nullable;
- `checksum_algorithms`, a sorted list of algorithms advertised by `ListObjectsV2` or
  inferred from checksum values returned by `HeadObject`;
- `checksum_type`, normalized to `full_object`, `composite`, or `null`;
- `remote_checksums`, an object containing only checksum values actually returned by
  `HeadObject`;
- `cache_status`;
- `sha256`, nullable for bodies not cached or not successfully downloaded;
- `cache_path`, nullable and relative to the configured cache root.

Checksum algorithm names use the closed lowercase set `crc32`, `crc32c`, `crc64nvme`,
`sha1`, and `sha256`. Remote checksum values remain in the base64 representation
returned by the SDK. The adapter maps `ChecksumCRC32`, `ChecksumCRC32C`,
`ChecksumCRC64NVME`, `ChecksumSHA1`, and `ChecksumSHA256` to those keys and never
places ETag values in `remote_checksums`. `checksum_type` uses the HEAD value when
present and otherwise the listing value; contradictory non-null values are an object
metadata error.

Before launching concurrent HEAD requests, the adapter probes one selected listed
object with `ChecksumMode="ENABLED"`. If R2 explicitly reports that checksum mode is
unsupported, the adapter retries that object without checksum mode, disables
checksum-value retrieval for the remainder of the run, and adds one non-fatal
capability note to the sync report. Other probe failures follow normal per-object
failure semantics.

The cache path is a logical content-addressed path such as
`sha256/ab/<full-sha256>`. It is independent of the operator's absolute output and
cache directories, may not contain `..`, and therefore does not make manifest identity
machine-specific.

Allowed manifest object cache statuses describe reproducible content state rather than
the action taken by the current invocation:

- `not_selected`;
- `verified`, used after either a valid cache hit or a successful download;
- `failed`.

The sync report separately records invocation actions: `planned`, `cache_hit`,
`downloaded`, and `failed`. Keeping those actions out of manifest rows ensures that
an initial download and an unchanged cache-hit rerun produce identical manifest bytes.

Real manifests use simfile statuses:

- `complete`;
- `partial`;
- `failed`;
- `empty`.

Errors use a stable structure:

```json
{
  "scope": "object",
  "code": "source_changed_during_sync",
  "object_key": "42/SET.DEF",
  "message": "Object metadata changed after inventory."
}
```

The message is safe, deterministic when practical, and does not include provider
request URLs, request headers, signatures, credential values, or raw exception
representations.

Serialized diagnostics use this closed code set:

| Code | Meaning |
| --- | --- |
| `invalid_config` | Invalid CLI or environment configuration |
| `missing_optional_dependency` | The `r2` optional dependency is not installed |
| `missing_credentials` | No usable credentials were resolved |
| `auth_failed` | R2 rejected the resolved credentials |
| `bucket_inaccessible` | The configured bucket cannot be accessed |
| `root_list_failed` | Authoritative root pagination did not complete |
| `cache_locked` | Another live writer holds the cache-index lock |
| `cache_index_invalid` | The cache index is malformed or uses an unsupported schema |
| `unsupported_platform` | Required POSIX locking or durability semantics are unavailable |
| `provenance_invalid` | The provenance document is malformed or unsupported |
| `artifact_write_failed` | A required report, cache, manifest, or pointer write failed |
| `object_head_failed` | Object metadata inspection failed after SDK retries |
| `object_get_failed` | A selected body read failed after SDK retries |
| `source_changed_during_sync` | Conditional or response metadata no longer matches inventory |
| `weak_etag_unverifiable` | A weak-ETag GET lacks metadata needed to verify the snapshot |
| `byte_count_mismatch` | Streamed bytes do not equal the authoritative object size |
| `cache_corrupt` | A referenced local cache body failed integrity and could not be repaired |
| `object_metadata_invalid` | Required object metadata is malformed or contradictory |
| `ambiguous_simfile_prefix` | Multiple exact prefixes normalize to one numeric ID |
| `malformed_root_key` | A root key cannot be assigned to a valid simfile prefix |
| `empty_prefix` | A requested prefix has no objects or a discovered prefix has only its folder marker |
| `internal_error` | An unforeseen failure was caught at the command boundary |
| `checksum_mode_unsupported` | Non-fatal report note: R2 rejected checksum-mode retrieval |

`checksum_mode_unsupported` appears only in the report's `notes` array and does not by
itself make a run partial. Every other code appears in `errors` when its condition is
surfaced. Unexpected exceptions map to the sanitized `internal_error`; SDK exception
names, bodies, and raw strings are never serialized as substitute codes or messages.

## Canonicalization and Immutability

Canonicalization rules are:

- simfile records sorted by numeric `simfile_id`, then exact `object_prefix`;
- objects sorted by exact key using Python's deterministic Unicode code-point order;
- mapping keys serialized in sorted order;
- UTF-8 output with `ensure_ascii=False`;
- compact JSON separators;
- one `\n` after every record, including the final record;
- UTC timestamps rendered from a UTC-aware value as
  `YYYY-MM-DDTHH:MM:SS.ffffffZ`, then trimmed by removing trailing zeroes from the
  fractional part and removing the decimal point when no fractional digits remain;
  the terminal `Z` is always retained;
- ETag values stored without surrounding HTTP quotes or the `W/` marker, with weakness
  represented only by `etag_is_weak`;
- no invocation timestamps, local absolute paths, or report-only counters in the
  manifest.

To avoid a self-referential hash:

1. Serialize the normalized corpus payload without `corpus_version`.
2. Compute SHA-256 over those canonical payload bytes.
3. Set every record's `corpus_version` to `sha256:<payload-hash>`.
4. Serialize the final JSONL.
5. Compute the final manifest SHA-256.

`corpus_version` intentionally appears on every simfile row. This lets a JSONL row
remain self-identifying when streamed, extracted, or sharded without introducing a
special header-record shape that every consumer must branch around. At the expected
corpus size, the duplicated value is negligible.

Provenance is part of the normalized corpus payload. Editing any provenance or rights
field therefore creates a new corpus version and manifest identity even when the R2
inventory is unchanged; provenance edits are not metadata-only mutations of an
existing manifest.

The final manifest is published as:

```text
manifests/<manifest-sha256>.jsonl
```

Publication writes a temporary file, verifies its hash, and installs it atomically. If
the destination already exists, Crux verifies that its bytes match and treats the run
as idempotent. It never replaces mismatched bytes at an existing hash path.

After publication, Crux atomically updates:

```text
latest.json
```

The pointer contains the corpus version, manifest SHA-256, relative path, publication
time, and `overall_status` (`complete` or `partial`). Fatal attempts publish no
manifest and do not update it. The pointer is operational convenience only, and its
presence is not a fitness signal. Benchmark runs must require `overall_status:
"complete"`, record the concrete hash path, and never use `latest.json` as immutable
input.

## Sync Report

Every attempt writes a machine-readable report outside the immutable manifest
identity. It contains:

- report schema version;
- start and end timestamps;
- dry-run flag;
- source endpoint SHA-256 and bucket name, excluding the raw endpoint URL;
- include and exclude filters;
- `cache_profile: "setdef_dtx_txt_v1"`;
- effective concurrency, timeout, and retry settings;
- simfile and object counts, including `simfiles_excluded_by_filter`;
- cache hits, planned downloads, completed downloads, failed downloads, and bytes;
- malformed root keys and ambiguous prefixes;
- sanitized errors;
- non-fatal capability notes;
- overall status: `complete`, `partial`, `failed`, `dry_run_complete`, or
  `dry_run_partial`;
- manifest corpus version, hash, and relative path when published.

At command start, Crux creates a UUID4 `run_id`. Reports use:

```text
reports/<UTC-YYYYMMDDTHHMMSS.ffffffZ>-<run-id>.json
```

The high-resolution timestamp and random run ID prevent collisions across rapid or
concurrent invocations. The report includes the run ID, and `latest-report.json`
points to the latest attempt. Reports do not contain credentials, signed URLs, request
headers, or audio bodies.

### Operator progress

The command emits sanitized human progress to stderr at phase boundaries and after
each 100 completed objects or five seconds of observed completions, whichever comes
first. Updates contain phase names, counts, and bytes only; they do not print endpoint
URLs, signatures, credentials, or object keys. Stdout is reserved for the final
summary and report or manifest paths. Progress output is transient and never used as
machine-readable state; the sync report remains authoritative.

## Dry-run Behavior

`--dry-run`:

- validates configuration;
- inventories R2;
- performs metadata inspection;
- applies include and exclude filters;
- validates provenance;
- compares remote identities with the local cache index;
- verifies existing selected cache entries;
- reports planned downloads and bytes.

It does not call `GetObject`, modify the cache or cache index, publish a manifest, or
change `latest.json`. Writing the timestamped sync report and updating
`latest-report.json` are the only allowed filesystem mutations.

## Failure Semantics

Fatal failures publish no manifest and exit `2`. They write a report when the
report directory is usable:

- invalid configuration;
- missing `r2` optional dependency;
- missing credentials;
- authentication failure;
- inaccessible bucket;
- root-list pagination failure;
- cache lock, index, or unsupported-platform failure;
- invalid provenance mapping;
- inability to write required local report or artifact directories.

If the report itself cannot be written, the command emits only a sanitized stderr
summary and exits `2`.

Per-object and per-simfile failures are isolated:

- listing metadata remains present where available;
- the object or simfile receives a structured error;
- unrelated simfiles continue;
- the reproducible partial manifest is published;
- the command exits `1` after publication.

Malformed root keys and ambiguous aliases have no trustworthy simfile row and therefore
live in the sync report. Their presence makes the attempt `partial` and produces a
`1` exit after publishing the valid rows. Explicitly included empty prefixes
receive an `empty` simfile row with the canonical requested prefix `<id>/` in a real
manifest.

Downstream benchmark stages should reject any row whose status is not `complete`
unless they explicitly implement an audit-only mode. That enforcement belongs to the
consumer issue, not HPA-321.

The operator policy is intentionally strict:

- headline benchmark input requires a full-corpus manifest whose
  `latest.json.overall_status` is `complete`;
- pilot runs should use `--include-simfile-id` for a known-good subset;
- empty and marker-only prefixes are inventory debt to repair at the source, not soft
  warnings to ignore.

A future consumer may define an explicit audit or allow-empty policy, but HPA-321 does
not weaken synchronization status or silently omit empty prefixes.

## Security and Rights Controls

- Use a bucket-scoped, read-only R2 credential.
- Never accept secrets in command-line flags.
- Never log the configured endpoint URL because it contains the Cloudflare account
  identifier and may be embedded in SDK error context.
- Convert SDK failures to allowlisted error codes and safe messages.
- Do not serialize SDK request or response metadata wholesale.
- Store logical cache paths relative to the configured cache root; never store
  absolute paths or `..` segments.
- Do not place derived artifacts beneath cached source-key namespaces.
- Do not copy audio bodies into manifests or reports.
- Published benchmark reports may include identifiers, hashes, metrics, and small
  non-audio diagnostics only unless rights are explicitly established elsewhere.

## Verification Strategy

Unit and CLI tests use fake R2 clients and temporary directories. They cover:

- paginated listings, including more than 1,000 objects;
- nested paths;
- non-ASCII, space-containing, and special-character keys;
- exact-key preservation and deterministic ordering;
- folder markers;
- marker-only prefixes becoming `empty`;
- malformed root objects;
- ambiguous numeric prefixes;
- explicit includes not overriding ambiguous-prefix quarantine;
- explicitly included empty prefixes;
- include and exclude precedence;
- `simfiles_excluded_by_filter` accounting;
- metadata HEAD failures;
- HEAD-over-listing metadata authority and one normalized mtime in index and manifest;
- single-part and multipart-style ETags;
- strong ETag conditional serialization;
- weak ETag unconditional GETs with exact response-metadata verification;
- weak ETag GETs with missing comparison metadata failing closed;
- remote checksum preservation without treating ETags as checksums;
- checksum key normalization and unsupported checksum-mode fallback;
- configured concurrency, connection-pool, timeout, retry, and exhausted-retry
  behavior;
- initial downloads;
- verified cache hits causing no body request;
- identical manifest bytes after the first run changes from a download action to a
  cache-hit action;
- locally corrupted cache entries being redownloaded;
- byte-count mismatches;
- `If-Match` failures when remote content changes after listing;
- temporary-file cleanup after failed downloads;
- atomic cache-index publication, POSIX lock conflicts, unsupported-platform failure,
  and restart cache hits after an interrupted run;
- unchanged reruns producing identical corpus and manifest identities;
- object changes producing new identities while preserving old manifests and cache
  files;
- provenance changes producing new identities;
- missing provenance producing explicit unknown values;
- malformed provenance failing before network access;
- unsupported provenance schema versions failing before network access;
- partial failures remaining visible per simfile;
- `latest.json` carrying complete or partial status;
- collision-resistant report filenames;
- sanitized periodic progress on stderr;
- dry runs issuing no body reads and changing neither cache nor manifests;
- atomic, idempotent manifest publication;
- secret, signed-query, endpoint, and header redaction;
- importability without `boto3`, the missing-extra installation hint, and lazy adapter
  loading;
- command defaults, help text, summary output, and exact `0`/`1`/`2` exit codes.

The credential-gated production smoke test is a manual acceptance gate, not a routine
CI job:

1. Dry-run a small, technically diverse include set.
2. Inspect nested and non-ASCII keys in the report.
3. Perform a real sync of the same include set.
4. Verify every cached selected body against its manifest SHA-256.
5. Repeat the sync and confirm zero body downloads and verified cache hits.
6. Inspect the immutable manifest and `latest.json`.
7. Run a full-corpus dry-run before the first complete production synchronization.
8. Record the ETag forms returned by real R2 objects. If a credentialed weak-ETag
   fixture is available, exercise the unconditional GET path and verify response
   metadata and local SHA-256; otherwise retain the fake-client weak-path test as the
   deterministic acceptance surface.

The current development shell has no R2 credentials configured. Deterministic tests
can run without them, but the production smoke test is required before HPA-321 can be
considered fully accepted.

## Dependencies

- Add an `r2` optional dependency extra containing `boto3>=1.42,<2`; do not add
  `boto3` to the base API/runtime dependency list.
- Operators install the command dependency with `uv pip install -e '.[r2]'`.
- Accept the extra's transitive `botocore`, `s3transfer`, `jmespath`, and `urllib3`
  installation and lockfile footprint only in environments that install the extra.
- Keep all `boto3` and `botocore` imports inside the adapter factory so the base CLI,
  API service, domain protocol, and tests remain importable without the extra.
- Do not add Parquet or database dependencies. JSONL is sufficient for the base
  inventory and keeps canonicalization directly inspectable.
- Do not introduce SQLite as an intermediate authority.

## Reference Behavior

- [Cloudflare R2 S3 compatibility](https://developers.cloudflare.com/r2/api/s3/api/)
  defines supported listing, HEAD, GET, conditional, and checksum-type behavior.
- [Cloudflare R2 upload behavior](https://developers.cloudflare.com/r2/objects/upload-objects/)
  documents multipart ETag construction.
- [Botocore client configuration](https://docs.aws.amazon.com/botocore/latest/reference/config.html)
  defines timeout, connection-pool, and retry settings.
- [Boto3 `HeadObject`](https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/s3/client/head_object.html)
  defines checksum-mode request and response fields.
- [RFC 9110 `If-Match`](https://www.rfc-editor.org/rfc/rfc9110.html#name-if-match)
  requires strong entity-tag comparison and therefore motivates the separate
  metadata-verified weak-ETag download path.

## Acceptance Mapping

- Production inventory: complete root listing and credential-gated smoke/full runs.
- Nested and special keys: delimiter-free pagination plus exact-key tests.
- Idempotent reruns: remote fingerprint comparison and local SHA-256 verification.
- Changed-object history: content-addressed cache and immutable manifest paths.
- Dry run: metadata and cache planning with no body reads or manifest/cache mutation.
- Secret safety: credential-provider-chain configuration and allowlisted errors.
- Partial failures: structured per-object/per-simfile errors and exit `1`.
- Provenance and rights: optional version-controlled mapping with explicit unknowns.
