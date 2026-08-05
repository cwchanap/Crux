# HPA-322 Authoritative Chart Selection Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Consume an immutable HPA-321 corpus manifest and verified cache, select one authoritative DTX chart per simfile, and publish a deterministic enriched manifest for HPA-323 and later benchmark stages.

**Architecture:** Add a shared DTXMania text decoder, a focused `set.def` parser, a pure cache-backed chart selector, and a small manifest/CLI orchestration layer. Selection remains an offline stage: it never contacts R2 and never changes HPA-321 artifacts.

**Tech Stack:** Python 3.12, dataclasses, pathlib/PurePosixPath, hashlib, JSONL, Click, pytest.

## Global Constraints

- R2 object contents, `set.def`, and selected DTX files are the only selection authorities.
- Evaluate authored slots in the exact order L5, L4, L3, L2, L1.
- Prefer exact-case key matches; use case-insensitive matching only when it is unique.
- Do not assume `mas.dtx`, `real.dtx`, or `full.dtx` determines difficulty.
- Preserve the HPA-321 manifest and cache unchanged; publish a new immutable manifest.
- Process sequentially and offline; do not add concurrency, databases, services, or workflow frameworks.
- Quarantine ambiguous rows instead of choosing alphabetically or silently falling back.
- Python modules must remain importable without R2 optional dependencies.
- Use TDD, focused commits, and the repository's `uv run` commands.

---

## File Map

### Create

- `src/benchmark/dtx_text.py` — BOM-aware DTXMania text decoding shared by DTX and `set.def`.
- `src/benchmark/set_def_parser.py` — parse L1-L5 label/file directives into explicit slots.
- `src/benchmark/reference_chart_selection.py` — validate cached candidates, load overrides, discover `set.def`, resolve authored paths, and choose or quarantine a chart.
- `src/benchmark/reference_chart_manifest.py` — load the HPA-321 JSONL, enrich rows, and publish the immutable derived manifest.
- `tests/benchmark/test_dtx_text.py` — decoding tests.
- `tests/benchmark/test_set_def_parser.py` — `set.def` grammar tests.
- `tests/benchmark/test_reference_chart_selection.py` — selection-policy tests.
- `tests/benchmark/test_reference_chart_manifest.py` — input and deterministic publication tests.
- `tests/benchmark/test_reference_chart_acceptance.py` — offline end-to-end fixture.
- `config/benchmark-reference-chart-overrides.json` — empty versioned override document.

### Modify

- `src/benchmark/dtx_parser.py` — use shared decoding and retain `#DLEVEL` metadata.
- `tests/benchmark/test_dtx_parser.py` — cover DLEVEL normalization and shared decoding behavior.
- `src/cli/benchmark.py` — add `select-reference-charts` command and JSON summary.
- `tests/test_cli_benchmark.py` — verify options, request wiring, summaries, and exit codes.

---

### Task 1: Extract DTXMania decoding and retain DLEVEL metadata

**Files:**
- Create: `src/benchmark/dtx_text.py`
- Create: `tests/benchmark/test_dtx_text.py`
- Modify: `src/benchmark/dtx_parser.py`
- Modify: `tests/benchmark/test_dtx_parser.py`

**Interfaces:**
- Produces: `decode_dtxmania_text(raw: bytes, *, source_name: str) -> str`
- Produces: `ParsedDtxChart.dlevel_raw: str | None`
- Produces: `ParsedDtxChart.dlevel_normalized: int | None`
- Consumed by: Tasks 2 and 3.

- [ ] **Step 1: Write failing decoder tests**

```python
from src.benchmark.dtx_text import decode_dtxmania_text


def test_decode_dtxmania_text_honors_utf16le_bom() -> None:
    raw = "#L5FILE: real.dtx\r\n".encode("utf-16")

    assert decode_dtxmania_text(raw, source_name="set.def") == "#L5FILE: real.dtx\r\n"


def test_decode_dtxmania_text_accepts_cp932_without_bom() -> None:
    raw = "#TITLE: 蒼穹への招歌\r\n".encode("cp932")

    assert "蒼穹への招歌" in decode_dtxmania_text(raw, source_name="song.dtx")


def test_decode_dtxmania_text_rejects_non_directive_binary() -> None:
    with pytest.raises(ValueError, match="could not decode"):
        decode_dtxmania_text(bytes(range(256)), source_name="bad.dtx")
```

- [ ] **Step 2: Run the decoder tests and verify failure**

Run:

```bash
uv run pytest tests/benchmark/test_dtx_text.py -q
```

Expected: collection fails because `src.benchmark.dtx_text` does not exist.

- [ ] **Step 3: Implement the minimal shared decoder**

Create `src/benchmark/dtx_text.py` with this public shape:

```python
from __future__ import annotations

_DIRECTIVE_PREFIXES = ("#", "*")
_BOM_ENCODINGS = (
    (b"\xef\xbb\xbf", "utf-8-sig"),
    (b"\xff\xfe", "utf-16"),
    (b"\xfe\xff", "utf-16"),
)
_FALLBACK_ENCODINGS = ("utf-8", "cp932", "shift-jis")


def decode_dtxmania_text(raw: bytes, *, source_name: str) -> str:
    for bom, encoding in _BOM_ENCODINGS:
        if raw.startswith(bom):
            text = raw.decode(encoding)
            if _has_directive(text):
                return text.removeprefix("\ufeff")
            raise ValueError(f"could not decode DTXMania directives from {source_name}")

    for encoding in _FALLBACK_ENCODINGS:
        try:
            text = raw.decode(encoding)
        except UnicodeDecodeError:
            continue
        if _has_directive(text):
            return text
    raise ValueError(f"could not decode DTXMania directives from {source_name}")


def _has_directive(text: str) -> bool:
    return any(line.lstrip().startswith(_DIRECTIVE_PREFIXES) for line in text.splitlines())
```

Keep error messages free of file contents.

- [ ] **Step 4: Run decoder tests**

Run:

```bash
uv run pytest tests/benchmark/test_dtx_text.py -q
```

Expected: all decoder tests pass.

- [ ] **Step 5: Write failing DLEVEL tests**

Append to `tests/benchmark/test_dtx_parser.py`:

```python
def test_parse_dtx_retains_numeric_dlevel() -> None:
    chart = parse_dtx_text("#DLEVEL: 87\n#BPM: 120\n", chart_id="level")

    assert chart.dlevel_raw == "87"
    assert chart.dlevel_normalized == 87
    assert chart.warnings == []


def test_parse_dtx_retains_invalid_dlevel_as_warning() -> None:
    chart = parse_dtx_text("#DLEVEL: 8.7\n#BPM: 120\n", chart_id="level")

    assert chart.dlevel_raw == "8.7"
    assert chart.dlevel_normalized is None
    assert chart.warnings == ["ignoring non-integer DLEVEL value: '8.7'"]
```

- [ ] **Step 6: Run focused parser tests and verify failure**

Run:

```bash
uv run pytest tests/benchmark/test_dtx_parser.py -q
```

Expected: the new assertions fail because `ParsedDtxChart` has no DLEVEL fields.

- [ ] **Step 7: Refactor `dtx_parser.py` to use the decoder and parse DLEVEL**

Make these concrete changes:

```python
from src.benchmark.dtx_text import decode_dtxmania_text


@dataclass(frozen=True)
class ParsedDtxChart:
    chart_id: str
    title: str = ""
    artist: str = ""
    dlevel_raw: str | None = None
    dlevel_normalized: int | None = None
    # existing fields remain unchanged


def parse_dtx_file(path: Path, chart_id: str | None = None) -> ParsedDtxChart:
    text = decode_dtxmania_text(path.read_bytes(), source_name=str(path))
    return parse_dtx_text(text, chart_id=chart_id or path.stem)
```

Inside `parse_dtx_text`, retain the last `#DLEVEL` directive and normalize only decimal
integers in `0..100`:

```python
def _normalize_dlevel(raw: str, warnings: list[str]) -> int | None:
    if not raw.isascii() or not raw.isdecimal():
        warnings.append(f"ignoring non-integer DLEVEL value: {raw!r}")
        return None
    value = int(raw)
    if not 0 <= value <= 100:
        warnings.append(f"ignoring out-of-range DLEVEL value: {raw!r}")
        return None
    return value
```

Remove the old private encoding tuple and decode loop after existing decoding tests pass
through the shared helper.

- [ ] **Step 8: Run focused tests and formatting**

Run:

```bash
uv run pytest tests/benchmark/test_dtx_text.py tests/benchmark/test_dtx_parser.py -q
uv run ruff check src/benchmark/dtx_text.py src/benchmark/dtx_parser.py tests/benchmark/test_dtx_text.py tests/benchmark/test_dtx_parser.py
uv run black --check src/benchmark/dtx_text.py src/benchmark/dtx_parser.py tests/benchmark/test_dtx_text.py tests/benchmark/test_dtx_parser.py
```

Expected: all commands pass.

- [ ] **Step 9: Commit Task 1**

```bash
git add src/benchmark/dtx_text.py src/benchmark/dtx_parser.py tests/benchmark/test_dtx_text.py tests/benchmark/test_dtx_parser.py
git commit -m "feat: share DTXMania decoding metadata"
```

---

### Task 2: Parse authored `set.def` slots

**Files:**
- Create: `src/benchmark/set_def_parser.py`
- Create: `tests/benchmark/test_set_def_parser.py`

**Interfaces:**
- Consumes: `decode_dtxmania_text(raw, source_name=...)` from Task 1.
- Produces: `SetDefSlot(level: int, label: str | None, file: str | None)`.
- Produces: `ParsedSetDef(slots: tuple[SetDefSlot, ...], warnings: tuple[str, ...])`.
- Produces: `parse_set_def_text(text: str) -> ParsedSetDef`.
- Produces: `parse_set_def_bytes(raw: bytes, *, source_name: str) -> ParsedSetDef`.
- Consumed by: Task 3.

- [ ] **Step 1: Write failing grammar tests**

```python
from src.benchmark.set_def_parser import parse_set_def_text


def test_parse_set_def_accepts_colon_and_whitespace_forms() -> None:
    parsed = parse_set_def_text(
        "\n".join(
            [
                "#L4LABEL: MASTER",
                "#L4FILE mas.dtx",
                "*L5LABEL REAL",
                "*L5FILE: charts/real.dtx",
            ]
        )
    )

    assert parsed.slot(5).label == "REAL"
    assert parsed.slot(5).file == "charts/real.dtx"
    assert parsed.slot(4).file == "mas.dtx"


def test_parse_set_def_last_duplicate_wins_with_warning() -> None:
    parsed = parse_set_def_text("#L5FILE: old.dtx\n#L5FILE: real.dtx\n")

    assert parsed.slot(5).file == "real.dtx"
    assert parsed.warnings == ("duplicate L5FILE; last value wins",)
```

- [ ] **Step 2: Run tests and verify failure**

```bash
uv run pytest tests/benchmark/test_set_def_parser.py -q
```

Expected: collection fails because the parser module does not exist.

- [ ] **Step 3: Implement the focused parser**

Use explicit dataclasses and one regex:

```python
_DIRECTIVE_RE = re.compile(
    r"^[#*]\s*L(?P<level>[1-5])(?P<field>LABEL|FILE)\s*:?[ \t]*(?P<value>.*)$",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class SetDefSlot:
    level: int
    label: str | None = None
    file: str | None = None


@dataclass(frozen=True)
class ParsedSetDef:
    slots: tuple[SetDefSlot, ...]
    warnings: tuple[str, ...] = ()

    def slot(self, level: int) -> SetDefSlot:
        return self.slots[level - 1]
```

`parse_set_def_text` must:

- strip a leading Unicode BOM;
- ignore blank and semicolon-comment lines;
- strip surrounding single or double quotes from values;
- preserve internal spaces and non-ASCII text;
- initialize all five slots so callers never handle missing level objects;
- record one warning per duplicate field and keep the last value.

`parse_set_def_bytes` decodes through Task 1 and delegates to the text parser.

- [ ] **Step 4: Add malformed and encoding tests**

Cover empty values, lower-case directives, CP932 labels, unrelated directives, and files
containing spaces. Do not reject a document merely because one slot is empty.

- [ ] **Step 5: Run focused checks**

```bash
uv run pytest tests/benchmark/test_set_def_parser.py -q
uv run ruff check src/benchmark/set_def_parser.py tests/benchmark/test_set_def_parser.py
uv run black --check src/benchmark/set_def_parser.py tests/benchmark/test_set_def_parser.py
```

Expected: all pass.

- [ ] **Step 6: Commit Task 2**

```bash
git add src/benchmark/set_def_parser.py tests/benchmark/test_set_def_parser.py
git commit -m "feat: parse authored set.def slots"
```

---

### Task 3: Implement deterministic cache-backed chart selection

**Files:**
- Create: `src/benchmark/reference_chart_selection.py`
- Create: `tests/benchmark/test_reference_chart_selection.py`
- Create: `config/benchmark-reference-chart-overrides.json`

**Interfaces:**
- Consumes: HPA-321 row dictionaries, cache root, `ParsedSetDef`, and `parse_dtx_file`.
- Produces: `SelectionOverride(chart_key: str, reason: str)`.
- Produces: `ChartSelection` with selected/quarantined status and all enrichment fields.
- Produces: `load_selection_overrides(path: Path | None, *, default_missing_ok: bool) -> LoadedOverrides`.
- Produces: `select_reference_chart(row: Mapping[str, object], cache_dir: Path, overrides: LoadedOverrides) -> ChartSelection`.
- Consumed by: Task 4.

- [ ] **Step 1: Write the authored-slot happy-path tests**

Build small row and cache helpers in the test file. Each object entry should match the
real HPA-321 shape: `key`, `size`, `cache_status`, `sha256`, and `cache_path`.

```python
def test_l5_real_dtx_beats_l4_mas_dtx(tmp_path: Path) -> None:
    row, cache_dir = build_cached_row(
        tmp_path,
        simfile_id=42,
        files={
            "42/set.def": "#L4FILE: mas.dtx\n#L5FILE: real.dtx\n",
            "42/mas.dtx": "#TITLE: Master\n#DLEVEL: 80\n",
            "42/real.dtx": "#TITLE: Real\n#DLEVEL: 95\n",
        },
    )

    result = select_reference_chart(row, cache_dir, empty_overrides())

    assert result.status == "selected"
    assert result.method == "set_def"
    assert result.selected_chart_key == "42/real.dtx"
    assert result.selected_level_slot == 5
```

Add equivalent tests for L5 `full.dtx`, a custom L5 name, and missing L5 falling to a
valid L4 file with a warning.

- [ ] **Step 2: Run tests and verify failure**

```bash
uv run pytest tests/benchmark/test_reference_chart_selection.py -q
```

Expected: collection fails because the selector module does not exist.

- [ ] **Step 3: Define the selector domain records**

Use literals rather than a class hierarchy:

```python
SelectionStatus = Literal["selected", "quarantined"]
SelectionMethod = Literal[
    "override",
    "set_def",
    "fallback_single",
    "fallback_dlevel",
    "fallback_dlevel_filename_tiebreak",
]


@dataclass(frozen=True)
class SelectionOverride:
    chart_key: str
    reason: str


@dataclass(frozen=True)
class LoadedOverrides:
    values: Mapping[int, SelectionOverride]
    sha256: str


@dataclass(frozen=True)
class ChartSelection:
    status: SelectionStatus
    method: SelectionMethod | None
    reason_codes: tuple[str, ...]
    set_def_key: str | None
    set_def_content_hash: str | None
    selected_chart_key: str | None
    selected_chart_content_hash: str | None
    selected_level_slot: int | None
    selected_level_label: str | None
    dlevel_raw: str | None
    dlevel_normalized: int | None
    selected_chart_title: str | None
    selected_chart_artist: str | None
    override: SelectionOverride | None
    warnings: tuple[str, ...]
```

Add private `_CachedObject` and `_ChartCandidate` dataclasses to avoid passing raw dicts
through the selection algorithm.

- [ ] **Step 4: Implement and test cache-body verification**

The object loader must require `cache_status == "verified"`, lowercase SHA-256, and a
relative cache path. Resolve the path under `cache_dir`, require a regular file, compare
byte count, and stream SHA-256 before returning the body path.

Tests must cover missing files, mismatched size, mismatched digest, absolute paths, and
`..` traversal. These become row warnings or `cached_body_unavailable`, not command-wide
exceptions.

- [ ] **Step 5: Implement and test canonical `set.def` discovery**

Implement this exact policy:

```python
def _choose_set_def(objects: tuple[_CachedObject, ...], object_prefix: str) -> _CachedObject | None:
    candidates = tuple(obj for obj in objects if PurePosixPath(obj.key).name.casefold() == "set.def")
    root = tuple(obj for obj in candidates if len(_relative_parts(obj.key, object_prefix)) == 1)
    if len(root) == 1:
        return root[0]
    if len(root) > 1:
        exact = tuple(obj for obj in root if PurePosixPath(obj.key).name == "set.def")
        if len(exact) == 1:
            return exact[0]
        raise _Quarantine("ambiguous_set_def")
    if not candidates:
        return None
    minimum_depth = min(len(_relative_parts(obj.key, object_prefix)) for obj in candidates)
    shallowest = tuple(
        obj for obj in candidates if len(_relative_parts(obj.key, object_prefix)) == minimum_depth
    )
    if len(shallowest) != 1:
        raise _Quarantine("ambiguous_set_def")
    return shallowest[0]
```

Tests cover one root plus nested copies, one nested copy, exact lowercase preference, and
ambiguous same-depth copies.

- [ ] **Step 6: Implement and test referenced-key resolution**

Normalize `\` to `/`, resolve relative to the selected `set.def` directory, reject
absolute paths or traversal outside `object_prefix`, then match exact key before a
unique case-folded key.

Add tests for:

- nested `set.def` plus `../charts/real.dtx` that remains within the simfile prefix;
- backslash references;
- unique case-insensitive match with warning;
- duplicate case-insensitive matches producing `ambiguous_chart_key`;
- traversal above the simfile prefix producing quarantine.

- [ ] **Step 7: Implement authored L5-to-L1 selection**

For each non-empty slot file from 5 down to 1:

- missing reference: append a warning and continue;
- ambiguous reference: quarantine immediately;
- existing DTX: parse it and return `method="set_def"`;
- existing but invalid DTX: quarantine as `selected_chart_invalid`; do not continue to a lower slot.

Copy the slot label and parsed DLEVEL/title/artist into `ChartSelection`.

- [ ] **Step 8: Write override loader and override tests**

Create the committed empty file:

```json
{
  "schema_version": "crux.reference-chart-overrides/v1",
  "overrides": {}
}
```

`load_selection_overrides` must validate the exact schema, normalize decimal simfile
keys, reject aliases, require exact `chart_key` and non-empty `reason`, and compute the
SHA-256 of the exact file bytes. For no file, use the canonical empty-document bytes so
`sha256` remains deterministic.

Tests cover valid override, alias collision, wrong schema, missing reason, nonexistent
chart, non-DTX chart, and override parse failure. Invalid row-specific overrides produce
`override_invalid` without trying `set.def`.

- [ ] **Step 9: Implement fallback tests and code**

Add tests for:

- one valid DTX -> `fallback_single`;
- unique highest numeric DLEVEL -> `fallback_dlevel`;
- equal DLEVEL resolved by `real.dtx > full.dtx > mas.dtx` ->
  `fallback_dlevel_filename_tiebreak`;
- a lower recognized filename never beating a higher DLEVEL;
- multiple unranked ties -> `ambiguous_fallback`;
- no verified DTX -> `no_verified_dtx`.

Keep fallback private to this module. Never sort alphabetically to choose a winner.

- [ ] **Step 10: Run selector checks**

```bash
uv run pytest tests/benchmark/test_reference_chart_selection.py -q
uv run ruff check src/benchmark/reference_chart_selection.py tests/benchmark/test_reference_chart_selection.py
uv run black --check src/benchmark/reference_chart_selection.py tests/benchmark/test_reference_chart_selection.py
```

Expected: all pass.

- [ ] **Step 11: Commit Task 3**

```bash
git add src/benchmark/reference_chart_selection.py tests/benchmark/test_reference_chart_selection.py config/benchmark-reference-chart-overrides.json
git commit -m "feat: select authoritative cached charts"
```

---

### Task 4: Load and publish the enriched immutable manifest

**Files:**
- Create: `src/benchmark/reference_chart_manifest.py`
- Create: `tests/benchmark/test_reference_chart_manifest.py`

**Interfaces:**
- Consumes: `select_reference_chart` and the existing `canonical_json_line`, `render_manifest`, `publish_manifest`, and `publish_latest_manifest` helpers.
- Produces: `SelectionRequest(manifest_path, cache_dir, overrides_file, output_dir, default_overrides_missing_ok)`.
- Produces: `SelectionCounters(selected: int, quarantined: int)`.
- Produces: `SelectionOutcome(status, exit_code, manifest, counters)`.
- Produces: `select_reference_manifest(request: SelectionRequest) -> SelectionOutcome`.
- Consumed by: Task 5.

- [ ] **Step 1: Write failing input-validation tests**

Cover:

```python
def test_manifest_loader_rejects_mixed_source_corpus_versions(tmp_path: Path) -> None:
    manifest = write_jsonl(
        tmp_path / "input.jsonl",
        [base_row(1, corpus_version="sha256:a"), base_row(2, corpus_version="sha256:b")],
    )

    with pytest.raises(ValueError, match="one corpus_version"):
        select_reference_manifest(request_for(manifest, tmp_path))
```

Add tests for empty input, invalid JSON, unsupported schema, non-object rows, and duplicate
simfile IDs.

- [ ] **Step 2: Run tests and verify failure**

```bash
uv run pytest tests/benchmark/test_reference_chart_manifest.py -q
```

Expected: collection fails because the manifest module does not exist.

- [ ] **Step 3: Implement request, counters, and outcome dataclasses**

```python
REFERENCE_MANIFEST_SCHEMA = "crux.reference-chart-manifest/v1"
SELECTION_POLICY_VERSION = "crux.authoritative-chart-selection/v1"


@dataclass(frozen=True)
class SelectionRequest:
    manifest_path: Path
    cache_dir: Path
    overrides_file: Path | None
    output_dir: Path
    default_overrides_missing_ok: bool = False


@dataclass(frozen=True)
class SelectionCounters:
    selected: int = 0
    quarantined: int = 0


@dataclass(frozen=True)
class SelectionOutcome:
    status: Literal["complete", "partial"]
    exit_code: Literal[0, 1, 2]
    manifest: PublishedManifest | None
    counters: SelectionCounters
```

Keep fatal errors as `ValueError` or `ManifestPublicationError` until the CLI boundary;
row-level problems are already represented by `ChartSelection`.

- [ ] **Step 4: Implement exact JSONL loading and lineage**

Read the source bytes once, compute `source_manifest_sha256`, decode UTF-8, parse every
non-empty line, and validate the HPA-321 schema and unique IDs. Retain the common input
`corpus_version` as `source_corpus_version`.

Do not rewrite or normalize the input before hashing it.

- [ ] **Step 5: Implement row enrichment**

For each row, copy all fields except the old `corpus_version` and `schema_version`, then
add:

```python
{
    "schema_version": REFERENCE_MANIFEST_SCHEMA,
    "source_manifest_sha256": source_manifest_sha256,
    "source_corpus_version": source_corpus_version,
    "selection_policy_version": SELECTION_POLICY_VERSION,
    "override_file_sha256": overrides.sha256,
    "selection_status": selection.status,
    "selection_method": selection.method,
    "selection_reason_codes": list(selection.reason_codes),
    "set_def_key": selection.set_def_key,
    "set_def_content_hash": selection.set_def_content_hash,
    "selected_chart_key": selection.selected_chart_key,
    "selected_chart_content_hash": selection.selected_chart_content_hash,
    "selected_level_slot": selection.selected_level_slot,
    "selected_level_label": selection.selected_level_label,
    "dlevel_raw": selection.dlevel_raw,
    "dlevel_normalized": selection.dlevel_normalized,
    "selected_chart_title": selection.selected_chart_title,
    "selected_chart_artist": selection.selected_chart_artist,
    "selection_override": (
        None if selection.override is None else asdict(selection.override)
    ),
    "selection_warnings": list(selection.warnings),
}
```

Pass the normalized rows to existing `render_manifest`, `publish_manifest`, and
`publish_latest_manifest`.

- [ ] **Step 6: Test deterministic publication and partial results**

Assert that:

- repeated runs with identical inputs produce identical content and manifest SHA;
- changing override bytes changes `override_file_sha256` and derived identity;
- selected plus quarantined equals the number of input rows;
- any quarantine returns `status="partial"`, exit `1`, and still publishes a manifest;
- all selected returns `status="complete"`, exit `0`;
- publication failure does not return a fake manifest.

- [ ] **Step 7: Run focused checks**

```bash
uv run pytest tests/benchmark/test_reference_chart_manifest.py -q
uv run ruff check src/benchmark/reference_chart_manifest.py tests/benchmark/test_reference_chart_manifest.py
uv run black --check src/benchmark/reference_chart_manifest.py tests/benchmark/test_reference_chart_manifest.py
```

Expected: all pass.

- [ ] **Step 8: Commit Task 4**

```bash
git add src/benchmark/reference_chart_manifest.py tests/benchmark/test_reference_chart_manifest.py
git commit -m "feat: publish reference chart manifest"
```

---

### Task 5: Wire the CLI and offline acceptance fixture

**Files:**
- Modify: `src/cli/benchmark.py`
- Modify: `tests/test_cli_benchmark.py`
- Create: `tests/benchmark/test_reference_chart_acceptance.py`

**Interfaces:**
- Consumes: `SelectionRequest` and `select_reference_manifest` from Task 4.
- Produces: `crux benchmark select-reference-charts`.
- Produces: one JSON stdout summary and exit codes `0`, `1`, or `2`.

- [ ] **Step 1: Write failing CLI help and request-wiring tests**

```python
def test_select_reference_charts_help_lists_exact_options() -> None:
    result = runner.invoke(main, ["benchmark", "select-reference-charts", "--help"])

    assert result.exit_code == 0
    assert "--manifest FILE" in result.stdout
    assert "--cache-dir DIRECTORY" in result.stdout
    assert "--overrides-file FILE" in result.stdout
    assert "--output-dir DIRECTORY" in result.stdout


def test_select_reference_charts_wires_request(monkeypatch, tmp_path: Path) -> None:
    captured = []
    monkeypatch.setattr(
        "src.benchmark.reference_chart_manifest.select_reference_manifest",
        lambda request: captured.append(request) or complete_outcome(tmp_path),
    )

    result = runner.invoke(
        main,
        [
            "benchmark",
            "select-reference-charts",
            "--manifest",
            str(tmp_path / "manifests" / "input.jsonl"),
            "--cache-dir",
            str(tmp_path / "cache"),
            "--output-dir",
            str(tmp_path / "out"),
        ],
    )

    assert result.exit_code == 0
    assert captured[0].cache_dir == tmp_path / "cache"
```

- [ ] **Step 2: Run CLI tests and verify failure**

```bash
uv run pytest tests/test_cli_benchmark.py -k select_reference_charts -q
```

Expected: the command is unknown.

- [ ] **Step 3: Add the Click command**

Add lazy imports inside the command body, following existing benchmark commands. Use
`ctx.get_parameter_source("overrides_file")` so only an omitted default override may
be absent; an explicitly supplied missing path remains fatal:

```python
@benchmark.command("select-reference-charts")
@click.option("--manifest", type=click.Path(path_type=Path, dir_okay=False), required=True)
@click.option("--cache-dir", type=click.Path(path_type=Path, file_okay=False), default=None)
@click.option(
    "--overrides-file",
    type=click.Path(path_type=Path, dir_okay=False),
    default=Path("config/benchmark-reference-chart-overrides.json"),
    show_default=True,
)
@click.option(
    "--output-dir",
    type=click.Path(path_type=Path, file_okay=False),
    default=Path("artifacts/benchmark/reference-charts"),
    show_default=True,
)
@click.pass_context
def select_reference_charts_command(
    ctx: click.Context,
    manifest: Path,
    cache_dir: Path | None,
    overrides_file: Path,
    output_dir: Path,
) -> None:
    from click.core import ParameterSource

    from src.benchmark.corpus_manifest import ManifestPublicationError
    from src.benchmark.reference_chart_manifest import (
        SelectionRequest,
        select_reference_manifest,
    )

    resolved_cache = manifest.parent.parent / "cache" if cache_dir is None else cache_dir
    override_source = ctx.get_parameter_source("overrides_file")
    request = SelectionRequest(
        manifest_path=manifest,
        cache_dir=resolved_cache,
        overrides_file=overrides_file,
        output_dir=output_dir,
        default_overrides_missing_ok=override_source is ParameterSource.DEFAULT,
    )
    try:
        outcome = select_reference_manifest(request)
    except (OSError, ValueError, ManifestPublicationError) as exc:
        click.echo(f"reference_chart_selection_failed: {exc}", err=True)
        ctx.exit(2)
```

Emit a sorted JSON summary containing exactly `corpus_version`, `exit_code`,
`manifest_path`, `quarantined`, `selected`, and `status`. Do not include tracebacks or
file contents in normal errors.

- [ ] **Step 4: Test summaries and all exit codes**

Use monkeypatched outcomes to verify:

- complete -> JSON stdout and exit `0`;
- partial -> JSON stdout and exit `1`;
- fatal -> no JSON summary, sanitized stderr, exit `2`.

Also verify the default cache path derives from `manifest.parent.parent / "cache"`.

- [ ] **Step 5: Write the real offline acceptance test**

Create a fixture with at least four simfiles:

1. L5 `real.dtx` selected over L4 `mas.dtx`;
2. nested `set.def` resolving a custom relative L5 chart;
3. missing `set.def` using a unique highest DLEVEL fallback;
4. ambiguous fallback quarantined.

The test must build the real HPA-321 JSONL rows and content-addressed cache files, invoke
`main` through `CliRunner`, read the published manifest, and assert exact selected keys,
methods, source lineage, counts, and exit `1`.

Do not mock the parser, selector, manifest renderer, or filesystem.

- [ ] **Step 6: Run the feature test suite**

```bash
uv run pytest \
  tests/benchmark/test_dtx_text.py \
  tests/benchmark/test_dtx_parser.py \
  tests/benchmark/test_set_def_parser.py \
  tests/benchmark/test_reference_chart_selection.py \
  tests/benchmark/test_reference_chart_manifest.py \
  tests/benchmark/test_reference_chart_acceptance.py \
  tests/test_cli_benchmark.py -q
```

Expected: all pass.

- [ ] **Step 7: Run the repository validation stack**

```bash
uv run pytest
uv run ruff check src tests
uv run black --check src tests
uv run pylint src/app src/cli src/benchmark
```

Expected: all pass. If Pylint exposes unrelated pre-existing failures, record the exact
command and output in the implementation PR rather than weakening lint rules.

- [ ] **Step 8: Commit Task 5**

```bash
git add src/cli/benchmark.py tests/test_cli_benchmark.py tests/benchmark/test_reference_chart_acceptance.py
git commit -m "feat: expose reference chart selection"
```

---

## Final Verification

- [ ] Confirm every HPA-322 acceptance example has a named test.
- [ ] Confirm no code path imports `boto3`, accesses D1, or performs network I/O.
- [ ] Confirm HPA-321 manifests and cache files are never modified.
- [ ] Confirm no alphabetical-first chart selection exists.
- [ ] Confirm selected rows retain native object keys and SHA-256 hashes.
- [ ] Confirm quarantined rows publish actionable reason codes and do not abort other rows.
- [ ] Confirm the derived manifest is deterministic across repeated runs.
- [ ] Confirm the PR contains no HPA-323 timing, HPA-324 taxonomy, or HPA-326 inference work.
