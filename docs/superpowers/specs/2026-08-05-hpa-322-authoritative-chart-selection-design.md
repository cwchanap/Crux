# HPA-322: Authoritative Benchmark Chart Selection Design

## Context

HPA-321 already inventories the authoritative `simfile-dtx` R2 corpus, caches
`set.def`, `.dtx`, and `.txt` bodies by verified SHA-256, and publishes an immutable
JSONL manifest. HPA-322 must consume that result and select the intended high-fidelity
reference chart for every simfile.

The selection must use first-hand files only:

1. the HPA-321 manifest describes exact R2 objects and verified cache bodies;
2. the canonical `set.def` defines the authored level ordering and referenced DTX files;
3. the selected DTX supplies chart metadata and later ground-truth events;
4. D1 or GraphQL metadata may not override any of those files.

This issue is the next ready task in the Crux benchmark dependency chain. HPA-321 is
complete, while HPA-323, HPA-324, HPA-327, and eventually HPA-326 depend on a stable,
reproducible chart-selection result.

## Goals

- Select the highest populated and resolvable `set.def` slot, evaluating L5 through L1.
- Support root and nested `set.def` files without assuming `mas.dtx` is authoritative.
- Resolve referenced chart paths relative to the selected `set.def` directory.
- Prefer exact-case object-key matches, then a deterministic unambiguous
  case-insensitive match.
- Parse common DTXMania encodings with BOM-aware decoding.
- Preserve selected chart identity, level slot and label, `#DLEVEL`, title, artist,
  source hashes, warnings, and the selection method.
- Support explicit version-controlled overrides for known exceptional songs.
- Quarantine ambiguous or unusable simfiles instead of silently choosing an arbitrary
  DTX file.
- Publish a new immutable manifest whose identity changes whenever the input manifest,
  cached source files, override file, or selection result changes.
- Keep the implementation small, offline, deterministic, and reusable by HPA-323.

## Non-goals

- Parsing event timing, BPM changes, BGM alignment, or playable drum lanes.
- Deciding whether the authored chart is musically accurate.
- Reading D1 difficulty rows as selection authority.
- Downloading missing files or contacting R2.
- Repairing malformed `set.def` or DTX files automatically.
- Building a generic workflow engine, plugin framework, database, or review UI.
- Adding concurrency; the initial corpus is small enough for sequential local parsing.
- Race-resistant file-descriptor pinning beyond ordinary cache-path and hash checks.

## Considered Approaches

### 1. Offline manifest-enrichment command — recommended

Add a dedicated command that reads one immutable HPA-321 manifest and its verified
cache, performs chart selection, and writes a derived immutable manifest.

Advantages:

- selection can be rerun instantly without R2 requests;
- HPA-321 remains focused on inventory and caching;
- selection-policy changes create a new derived identity without rewriting source data;
- HPA-323 can consume the enriched rows directly;
- tests need only local files.

The extra command and manifest stage are small and map directly to the existing issue
boundaries.

### 2. Fold chart selection into `sync-r2-corpus`

This would avoid one command, but every selection change would require running the R2
synchronization path. It would also mix remote inventory, cache mutation, authored chart
policy, and downstream enrichment in one transaction. That coupling makes HPA-321
harder to maintain and is rejected.

### 3. Reuse legacy `prepare-corpus`

The legacy command assumes local song folders and currently chooses chart files using
older conventions. It does not consume the immutable R2 manifest or preserve the
required provenance. Adapting it would blur the new benchmark pipeline with the old
MIDI-scoring workflow and is rejected.

## Operator Interface

Add one offline command:

```bash
uv run crux benchmark select-reference-charts \
  --manifest artifacts/benchmark/r2-corpus/manifests/<sha256>.jsonl \
  --cache-dir artifacts/benchmark/r2-corpus/cache \
  --overrides-file config/benchmark-reference-chart-overrides.json \
  --output-dir artifacts/benchmark/reference-charts
```

Options:

- `--manifest PATH` is required and identifies the immutable HPA-321 JSONL input.
- `--cache-dir PATH` defaults to `<manifest-parent>/../cache` and may be supplied when
  HPA-321 used a custom cache directory.
- `--overrides-file PATH` defaults to
  `config/benchmark-reference-chart-overrides.json`. A missing default file is treated
  as an empty override set; an explicitly supplied missing file is a fatal usage error.
- `--output-dir PATH` defaults to `artifacts/benchmark/reference-charts`.

The command performs no network access. It emits progress and warnings to stderr and
one JSON summary to stdout:

```json
{
  "corpus_version": "sha256:...",
  "exit_code": 1,
  "manifest_path": "artifacts/benchmark/reference-charts/manifests/<sha>.jsonl",
  "quarantined": 3,
  "selected": 397,
  "status": "partial"
}
```

Exit codes follow the existing corpus command convention:

- `0`: every input row produced a selected chart;
- `1`: a manifest was published, but one or more rows were quarantined;
- `2`: invalid input, invalid override configuration, or publication failure prevented
  a usable output manifest.

## Architecture

The command adds four focused boundaries.

### DTXMania text decoding

`src/benchmark/dtx_text.py` owns BOM-aware decoding shared by DTX and `set.def` parsing.
It detects UTF-8, UTF-16 LE, and UTF-16 BE BOMs first. Without a BOM it tries UTF-8,
CP932, and Shift-JIS, accepting only text containing recognizable `#` or `*`
directives. This replaces the private encoding loop currently embedded in
`dtx_parser.py`.

The decoder returns Unicode text or raises a direct `ValueError`. It does not guess from
locale settings or silently return an empty document.

### `set.def` parsing

`src/benchmark/set_def_parser.py` parses only the selection-relevant directives:

- `#L1LABEL` through `#L5LABEL`;
- `#L1FILE` through `#L5FILE`.

Both colon and whitespace forms are accepted, with optional spaces and either `#` or
`*` prefixes. Directive names are case-insensitive. The parser preserves label and file
strings after trimming surrounding whitespace and quotes.

The result contains five explicit slots. Duplicate directives for the same slot and
field use the last source occurrence and emit a warning, matching authored file order
rather than inventing a merge rule.

### Selection engine

`src/benchmark/reference_chart_selection.py` owns:

- source-manifest row validation;
- safe cache-body resolution and SHA-256 verification;
- canonical `set.def` discovery;
- relative referenced-path resolution;
- exact and case-insensitive R2-key matching;
- override application;
- DTX header parsing and fallback comparison;
- row-level selected or quarantined outcomes.

It returns domain records and does not write files or depend on Click.

### Derived manifest publication

`src/benchmark/reference_chart_manifest.py` reads the HPA-321 JSONL, verifies that rows
share one `corpus_version` and the expected schema, calculates the raw source-manifest
SHA-256, applies the selector, and publishes the enriched rows using the existing
canonical JSONL and immutable manifest helpers in `corpus_manifest.py`.

The new row schema is `crux.reference-chart-manifest/v1`. The original inventory fields
remain present so later stages need only one manifest. The derived identity is computed
from the complete enriched rows; the source identity is retained separately.

`src/cli/benchmark.py` owns only Click option parsing, progress presentation, the JSON
summary, and exit-code mapping.

## Input Validation and Cache Access

The input manifest must:

- be UTF-8 JSONL;
- contain at least one row;
- contain one JSON object per non-empty line;
- use `schema_version: "crux.r2-corpus-manifest/v1"`;
- use one shared `corpus_version` value;
- contain unique integer `simfile_id` values;
- contain an `objects` array with exact object keys.

The loader calculates `source_manifest_sha256` over the exact input bytes. It does not
recompute HPA-321's internal `corpus_version`; that value remains the inventory-stage
identity, while the exact file digest identifies the consumed artifact.

Rows with `sync_status: "failed"` or `"empty"` are quarantined. A `partial` row may
still be selected when every file required by the decision is verified; the result then
retains a `source_inventory_partial` warning.

A candidate body is usable only when its object record has:

- `cache_status: "verified"`;
- a 64-character lowercase `sha256`;
- a non-empty relative `cache_path`;
- a regular file below `--cache-dir` whose size and SHA-256 match the manifest.

Missing or corrupt candidate bodies quarantine only the affected simfile. Absolute
paths and paths containing `..` are rejected. Complex symlink-race protection is outside
scope; normal resolved-path containment plus final content hashing is sufficient for
this local hobby workflow.

## Canonical `set.def` Discovery

Candidate objects are verified cached files whose basename equals `set.def`
case-insensitively.

Selection order:

1. Prefer root candidates whose key relative to `object_prefix` has one path segment.
2. If one root candidate exists, use it.
3. If multiple root candidates exist, use the unique exact basename `set.def`; otherwise
   quarantine as `ambiguous_set_def`.
4. If no root candidate exists, find the minimum nested depth.
5. Use the unique candidate at that depth; a tie is `ambiguous_set_def`.
6. Deeper copies are retained as warnings but cannot override the selected shallower
   file.

A selected `set.def` that cannot be decoded or parsed is considered unusable and allows
the documented DTX fallback. Its failure remains visible in `selection_warnings`; it
becomes a quarantine only when no deterministic fallback succeeds.

## Slot Resolution

Slots are evaluated L5, L4, L3, L2, then L1.

For each populated `FILE` value:

1. normalize backslashes to `/`;
2. resolve it relative to the selected `set.def` directory;
3. reject absolute paths and traversal outside the simfile prefix;
4. attempt an exact object-key match;
5. otherwise attempt a case-insensitive match;
6. if exactly one case-insensitive match exists, use it and emit a warning;
7. if multiple matches exist, quarantine as `ambiguous_chart_key`;
8. if no match exists, record `referenced_chart_missing` and continue to the next slot.

The first existing referenced DTX is authoritative. If that file exists but cannot be
decoded or parsed, the row is quarantined as `selected_chart_invalid`; the selector does
not silently downgrade to a lower authored slot.

The selected slot records both its number and optional authored label. Filename names
such as `real.dtx`, `full.dtx`, and `mas.dtx` have no influence while `set.def` is usable.

## Overrides

The version-controlled override file uses:

```json
{
  "schema_version": "crux.reference-chart-overrides/v1",
  "overrides": {
    "42": {
      "chart_key": "42/full.dtx",
      "reason": "L5 is a short challenge edit; use the authored full chart"
    }
  }
}
```

Rules:

- decimal keys normalize to `simfile_id`; aliases such as `1` and `01` are rejected;
- `chart_key` must be the exact R2 object key for a verified `.dtx` object under that
  simfile prefix;
- `reason` is required and non-empty;
- an override is applied before `set.def` selection;
- an invalid override quarantines the row and does not fall back silently;
- the complete override file SHA-256 is recorded in every output row;
- rows without an override store `selection_override: null`.

This keeps exceptions explicit, reviewable, and reproducible.

## Fallback Without a Usable `set.def`

Fallback considers every verified cached `.dtx` object under the simfile prefix.
Candidates that cannot be decoded or parsed are excluded with warnings.

Selection rules:

1. If exactly one valid DTX candidate exists, select it with
   `selection_method: "fallback_single"`.
2. Otherwise compare numeric `#DLEVEL` values in `0..100`.
3. If one candidate has the unique highest value, select it with
   `selection_method: "fallback_dlevel"`.
4. If highest values tie, apply the narrow filename tie-break rank
   `real.dtx > full.dtx > mas.dtx > other`.
5. Use the tie-break only when it produces one unique recognized winner; otherwise
   quarantine as `ambiguous_fallback`.
6. Multiple candidates without comparable numeric `#DLEVEL` values are quarantined.

A filename never outranks a higher `#DLEVEL`, and alphabetical order is never used as a
selection rule.

## DTX Metadata

Extend `ParsedDtxChart` with:

- `dlevel_raw: str | None`;
- `dlevel_normalized: int | None`.

The parser retains the stripped raw `#DLEVEL` value. Normalization succeeds only for an
ASCII decimal integer in `0..100`; otherwise it returns `None` and appends a warning.
Selection needs only headers, but using the existing parser avoids a second DTX grammar
and prepares the chosen chart for HPA-323.

## Derived Manifest Fields

Each output row preserves the HPA-321 inventory and adds:

```text
schema_version = crux.reference-chart-manifest/v1
source_manifest_sha256
source_corpus_version
selection_policy_version = crux.authoritative-chart-selection/v1
override_file_sha256
selection_status = selected | quarantined
selection_method = override | set_def | fallback_single | fallback_dlevel |
                   fallback_dlevel_filename_tiebreak | null
selection_reason_codes[]
set_def_key
set_def_content_hash
selected_chart_key
selected_chart_content_hash
selected_level_slot
selected_level_label
dlevel_raw
dlevel_normalized
selected_chart_title
selected_chart_artist
selection_override
selection_warnings[]
```

`selected_*` fields are `null` for quarantined rows. Reason codes are stable
machine-readable identifiers; warnings are concise human-readable diagnostics.

The derived `corpus_version` is computed by the existing canonical manifest renderer
after all enrichment fields are present. HPA-323 must consume this new manifest and
publish another derived identity rather than mutating it.

## Error Handling

Fatal command errors:

- malformed or unsupported input manifest;
- duplicate simfile IDs;
- malformed override document;
- output publication failure.

Row-level quarantine reasons:

- `source_inventory_unusable`;
- `no_verified_dtx`;
- `ambiguous_set_def`;
- `referenced_chart_missing` when no authored slot resolves;
- `ambiguous_chart_key`;
- `selected_chart_invalid`;
- `override_invalid`;
- `ambiguous_fallback`;
- `cached_body_unavailable`.

One bad simfile never discards successful selections for other rows.

## Testing

Focused unit tests cover:

- BOM-aware UTF-8 and UTF-16 plus UTF-8, CP932, and Shift-JIS without a BOM;
- `set.def` colon and whitespace syntax, `#` and `*`, labels, files, and duplicates;
- root versus nested `set.def` selection and ambiguous copies;
- L5 `real.dtx`, L5 `full.dtx`, custom L5 filenames, and missing L5 falling to L4;
- exact-case and unique case-insensitive object-key matches;
- relative nested paths, backslashes, and traversal rejection;
- explicit overrides and invalid overrides;
- fallback by unique file, unique highest `#DLEVEL`, filename tie-break, and ambiguity;
- raw and normalized DLEVEL metadata;
- cache-body size and SHA verification;
- deterministic manifest output and source-manifest lineage;
- CLI exit `0`, `1`, and `2` behavior.

One local acceptance fixture builds a minimal HPA-321 manifest and content-addressed
cache with several simfiles, runs the real CLI, and verifies selected and quarantined
rows. It uses no R2 credentials or network access.

## Delivery Sequence

1. Extract shared DTXMania decoding and add DLEVEL metadata to the existing DTX parser.
2. Add the focused `set.def` parser.
3. Implement cache-backed selection, overrides, authored-slot rules, and fallback.
4. Add input loading and immutable derived-manifest publication.
5. Wire the CLI and acceptance fixture.
6. Run focused tests, then the full repository test and lint stack.

No implementation of HPA-323 timing, HPA-324 taxonomy, or HPA-326 inference belongs in
this PR.
