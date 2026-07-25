# Task 4 Report: Strict Provenance Loading

## Files

- Added `src/benchmark/corpus_provenance.py` with strict JSON provenance loading,
  numeric-ID normalization, fixed sanitized validation errors, and explicit unknown defaults.
- Added `tests/benchmark/test_corpus_provenance.py` covering valid records, optional
  mappings, schemas, malformed JSON, invalid document/record types, invalid IDs,
  duplicate normalized IDs, unsupported fields, and strict value types.
- Added `config/corpus-provenance.json`, the empty checked-in v1 mapping.

## Verification

- `rtk uv run pytest tests/benchmark/test_corpus_provenance.py -q` — 21 passed.
- `rtk uv run ruff check src/benchmark/corpus_provenance.py tests/benchmark/test_corpus_provenance.py` — passed.
- `rtk uv run black --check src/benchmark/corpus_provenance.py tests/benchmark/test_corpus_provenance.py` — passed.
- `rtk uv run pytest` — 276 passed.

## Commit

- `adc6183 feat: add corpus provenance mapping`
- `6ee22e1 fix: harden corpus provenance parsing`

## Fix Round 1

- Enforced the closed root schema, preserving fixed sanitized diagnostics for undeclared
  fields and duplicate root members.
- Preserved raw object pairs through validation so duplicate root, record, and simfile
  member names cannot collapse silently.
- Normalized IDs lexically before conversion, allowing arbitrarily long zero-padded
  in-range IDs without exposing Python's integer conversion-limit error.
- Added regression coverage for the checked-in mapping, root security-bearing field
  names, duplicate raw JSON members, inclusive bounds, huge out-of-range IDs, and huge
  zero-padded IDs.

## Verification: Fix Round 1

- `rtk uv run pytest tests/benchmark/test_corpus_provenance.py -q` — 35 passed.
- `rtk uv run ruff check src/benchmark/corpus_provenance.py tests/benchmark/test_corpus_provenance.py` — passed.
- `rtk uv run black --check src/benchmark/corpus_provenance.py tests/benchmark/test_corpus_provenance.py` — passed.
- `rtk uv run pytest` — 290 passed.

## Concerns and Design Note

- Malformed and unsupported provenance remains a fatal validation boundary for the later
  orchestration layer to map into its `provenance_invalid` diagnostic.
- Allowed source and rights fields remain opaque free text. Per the controller
  resolution, this task deliberately does not add heuristic scanning for embedded
  credentials or signed URLs: the closed schema rejects undeclared security-bearing
  fields, and every validation diagnostic avoids untrusted keys and values.
