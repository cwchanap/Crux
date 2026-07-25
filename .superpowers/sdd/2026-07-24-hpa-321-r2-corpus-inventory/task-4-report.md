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

## Concerns

- None. Malformed and unsupported provenance remains a fatal validation boundary for the
  later orchestration layer to map into its `provenance_invalid` diagnostic.
