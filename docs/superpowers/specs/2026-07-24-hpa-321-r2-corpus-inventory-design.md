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

Configuration uses:

- `CRUX_R2_ENDPOINT_URL` for the S3-compatible R2 endpoint;
- `CRUX_R2_BUCKET`, defaulting to `simfile-dtx`;
- `AWS_ACCESS_KEY_ID`;
- `AWS_SECRET_ACCESS_KEY`;
- optional `AWS_SESSION_TOKEN`;
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

## Architecture

The command coordinates four focused components.

### R2 inventory

`r2_inventory` owns an R2 client protocol and the `boto3` implementation. It:

- paginates `ListObjectsV2` without a delimiter so nested objects are returned;
- preserves object keys exactly as returned;
- groups digit-only top-level path segments into simfile prefixes;
- detects malformed root keys and ambiguous numeric aliases;
- performs bounded-concurrency `HeadObject` requests to obtain content type and other
  metadata unavailable from the listing;
- returns domain records rather than leaking raw SDK response objects.

The SDK client is created only when the command runs. Tests use a fake implementation
of the R2 protocol and do not require credentials or network access.

### Corpus cache

`corpus_cache` owns download selection, the local cache index, body verification, and
atomic installation. It has no chart-selection logic. The initial selection rule is:

- basename equals `set.def`, case-insensitively; or
- key suffix is `.dtx` or `.txt`, case-insensitively.

Future stages may add explicit cache profiles without changing the inventory format.

### Corpus provenance

`corpus_provenance` validates and loads an optional version-controlled JSON mapping
keyed by decimal simfile ID. It does not read D1 or GraphQL.

Missing mapping entries are allowed and receive explicit unknown values. Invalid field
types, duplicate IDs after numeric normalization, or malformed JSON are fatal
configuration errors.

### Corpus manifest

`corpus_manifest` converts domain records to canonical JSONL, computes deterministic
identities, publishes immutable manifests, writes sync reports, and atomically updates
the convenience pointer.

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
   not selected for caching.
4. If distinct prefixes normalize to the same integer, such as `1/` and `01/`, both
   are quarantined as an ambiguous prefix rather than silently merged.
5. Root objects without a slash and non-numeric top-level segments are reported as
   malformed root keys. They do not disappear and do not become simfile rows.
6. Include and exclude filters are applied after authoritative root discovery. HPA-321
   does not substitute direct prefix listing because the complete root listing is what
   makes malformed and ambiguous prefixes observable.

Listing metadata is retained even when `HeadObject` fails. The affected object receives
a structured error and its simfile status becomes `partial`.

## Cache Identity and Synchronization

An R2 ETag is a remote change signal, not a universal content hash. Multipart ETags
encode part hashes and part count rather than the complete object's ordinary MD5.

The cache index records, per exact source endpoint, bucket, and key:

- source endpoint SHA-256;
- bucket name;
- exact object key;
- normalized ETag;
- size;
- last-modified time;
- locally verified SHA-256;
- relative content-addressed cache path.

Raw endpoint URLs and credentials are not stored.

For each selected object:

1. Compare key, ETag, size, and modification time with the index.
2. If they match, stream-hash the referenced local file.
3. Reuse the entry only when the computed SHA-256 and byte count match the index.
4. Otherwise download to a temporary file with `If-Match` using the listed ETag.
5. Stream-compute SHA-256 and byte count during download.
6. Reject a byte-count mismatch or conditional-request failure.
7. Atomically move valid content to:

   ```text
   sha256/<first-two-hex>/<full-sha256>
   ```

8. Atomically update the cache index only after the content file is durable.

The cache is content-addressed and extensionless. Manifest metadata retains the source
key and media type, so the cache filename does not need to reproduce a potentially
unsafe object basename.

If the same key changes, its new content receives a new SHA-256 path. The old cache
file and every previously published manifest remain untouched. Orphaned cache cleanup
is outside HPA-321.

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

The mapping contains descriptions and rights assertions only. It must not contain
credentials, signed URLs, or source file bodies.

## Manifest Contract

The manifest is UTF-8 JSONL with one record per simfile. A representative record is:

```json
{"schema_version":"crux.r2-corpus-manifest/v1","corpus_version":"sha256:…","simfile_id":42,"object_prefix":"42/","source_endpoint_sha256":"…","source_bucket":"simfile-dtx","source_discovery_method":"r2_list_objects_v2","objects":[{"key":"42/SET.DEF","size":1234,"etag":"…","version":null,"last_modified":"2026-07-24T12:34:56Z","content_type":"text/plain","remote_checksums":{},"cache_status":"verified","sha256":"…","cache_path":"sha256/ab/ab…"}],"sync_status":"complete","sync_errors":[],"source_origin":null,"source_author_or_pack":null,"source_reference":null,"rights_status":"unknown","redistribution_allowed":null,"provenance_notes":null}
```

Each row includes `source_endpoint_sha256` and `source_bucket` so its remote source is
unambiguous without exposing the raw endpoint.

Each object contains:

- `key`;
- `size`;
- `etag`;
- `version`, nullable because R2's S3 compatibility does not expose bucket
  version-history semantics;
- `last_modified` as UTC ISO 8601;
- `content_type`, nullable;
- `remote_checksums`, an object containing only checksums actually returned;
- `cache_status`;
- `sha256`, nullable for bodies not cached or not successfully downloaded;
- `cache_path`, nullable and relative to the configured cache root.

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

## Canonicalization and Immutability

Canonicalization rules are:

- simfile records sorted by numeric `simfile_id`, then exact `object_prefix`;
- objects sorted by exact key using Python's deterministic Unicode code-point order;
- mapping keys serialized in sorted order;
- UTF-8 output with `ensure_ascii=False`;
- compact JSON separators;
- one `\n` after every record, including the final record;
- UTC timestamps normalized to `YYYY-MM-DDTHH:MM:SS.ffffffZ`, with trailing fractional
  zeroes removed consistently;
- ETags stored without surrounding HTTP quotes;
- no invocation timestamps, local absolute paths, or report-only counters in the
  manifest.

To avoid a self-referential hash:

1. Serialize the normalized corpus payload without `corpus_version`.
2. Compute SHA-256 over those canonical payload bytes.
3. Set every record's `corpus_version` to `sha256:<payload-hash>`.
4. Serialize the final JSONL.
5. Compute the final manifest SHA-256.

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

The pointer contains the corpus version, manifest SHA-256, relative path, and
publication time. It is operational convenience only. Benchmark runs must record and
open the concrete hash path, never rely on `latest.json` as immutable input.

## Sync Report

Every attempt writes a machine-readable report outside the immutable manifest
identity. It contains:

- report schema version;
- start and end timestamps;
- dry-run flag;
- source endpoint SHA-256 and bucket name, excluding the raw endpoint URL;
- include and exclude filters;
- selected cache profile;
- simfile and object counts;
- cache hits, planned downloads, completed downloads, failed downloads, and bytes;
- malformed root keys and ambiguous prefixes;
- sanitized errors;
- overall status;
- manifest corpus version, hash, and relative path when published.

Reports may use a timestamped filename beneath `reports/`; a convenience
`latest-report.json` may point to the latest attempt. Reports do not contain
credentials, signed URLs, request headers, or audio bodies.

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

Fatal failures publish no manifest and exit nonzero. They write a report when the
report directory is usable:

- invalid configuration;
- missing credentials;
- authentication failure;
- inaccessible bucket;
- root-list pagination failure;
- invalid provenance mapping;
- inability to write required local report or artifact directories.

If the report itself cannot be written, the command emits only a sanitized stderr
summary and exits nonzero.

Per-object and per-simfile failures are isolated:

- listing metadata remains present where available;
- the object or simfile receives a structured error;
- unrelated simfiles continue;
- the reproducible partial manifest is published;
- the command exits nonzero after publication.

Malformed root keys and ambiguous aliases have no trustworthy simfile row and therefore
live in the sync report. Their presence makes the attempt `partial` and produces a
nonzero exit after publishing the valid rows. Explicitly included empty prefixes
receive an `empty` simfile row with the canonical requested prefix `<id>/` in a real
manifest.

Downstream benchmark stages should reject any row whose status is not `complete`
unless they explicitly implement an audit-only mode. That enforcement belongs to the
consumer issue, not HPA-321.

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
- malformed root objects;
- ambiguous numeric prefixes;
- explicitly included empty prefixes;
- include and exclude precedence;
- metadata HEAD failures;
- single-part and multipart-style ETags;
- remote checksum preservation without treating ETags as checksums;
- initial downloads;
- verified cache hits causing no body request;
- identical manifest bytes after the first run changes from a download action to a
  cache-hit action;
- locally corrupted cache entries being redownloaded;
- byte-count mismatches;
- `If-Match` failures when remote content changes after listing;
- temporary-file cleanup after failed downloads;
- unchanged reruns producing identical corpus and manifest identities;
- object changes producing new identities while preserving old manifests and cache
  files;
- provenance changes producing new identities;
- missing provenance producing explicit unknown values;
- malformed provenance failing before network access;
- partial failures remaining visible per simfile;
- dry runs issuing no body reads and changing neither cache nor manifests;
- atomic, idempotent manifest publication;
- secret, signed-query, endpoint, and header redaction;
- command defaults, help text, summary output, and exit codes.

The credential-gated production smoke test is:

1. Dry-run a small, technically diverse include set.
2. Inspect nested and non-ASCII keys in the report.
3. Perform a real sync of the same include set.
4. Verify every cached selected body against its manifest SHA-256.
5. Repeat the sync and confirm zero body downloads and verified cache hits.
6. Inspect the immutable manifest and `latest.json`.
7. Run a full-corpus dry-run before the first complete production synchronization.

The current development shell has no R2 credentials configured. Deterministic tests
can run without them, but the production smoke test is required before HPA-321 can be
considered fully accepted.

## Dependencies

- Add `boto3` as a runtime dependency.
- Do not add Parquet or database dependencies. JSONL is sufficient for the base
  inventory and keeps canonicalization directly inspectable.
- Do not introduce SQLite as an intermediate authority.

## Acceptance Mapping

- Production inventory: complete root listing and credential-gated smoke/full runs.
- Nested and special keys: delimiter-free pagination plus exact-key tests.
- Idempotent reruns: remote fingerprint comparison and local SHA-256 verification.
- Changed-object history: content-addressed cache and immutable manifest paths.
- Dry run: metadata and cache planning with no body reads or manifest/cache mutation.
- Secret safety: credential-provider-chain configuration and allowlisted errors.
- Partial failures: structured per-object/per-simfile errors and nonzero partial exit.
- Provenance and rights: optional version-controlled mapping with explicit unknowns.
