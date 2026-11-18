"""Prepare and verify the fixed archived TF1 inputs for the TF2.15 experiment."""

from __future__ import annotations

import csv
import errno
import io
import stat
import tempfile
from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import replace
from decimal import Decimal, InvalidOperation
from pathlib import Path, PurePosixPath
from typing import Any, cast

from runtime.oaf_tf1.model import OafModelConfig, load_model_config
from src.benchmark.artifact_io import read_regular_file_no_follow
from src.benchmark.backend_identity import (
    BackendDescriptor,
    JsonValue,
    canonical_json_bytes,
    require_sha256,
    sha256_hex,
    strict_json_loads,
)
from src.benchmark.backends.base import CanonicalAudio
from src.benchmark.corpus_cache import CacheIndexStore, ResolvedSourceAudio, resolve_source_audio
from src.benchmark.input_view import load_materialized_audio, materialize_full_mix_audio
from src.benchmark.oaf_corpus_run import (
    OAF_FULL_MIX_INPUT_VIEW_ID,
    OAF_PREDICTION_MAP_ID,
    _expected_oaf_descriptor,
    _model_lock_path,
    build_inference_config,
    build_run_id,
    compute_model_lock_sha256,
    inference_config_sha256,
    parse_oaf_corpus_run,
)
from src.benchmark.prediction_artifact import (
    PredictionArtifact,
    prediction_artifact_matches_audio,
    prediction_artifact_matches_run_row,
    read_prediction_artifact,
)
from src.benchmark.reference_set_manifest import (
    LoadedReferenceSetManifest,
    LoadedReferenceSetRow,
    load_reference_set_manifest,
    preflight_reference_mappings,
)
from src.benchmark.reference_timing_manifest import (
    LoadedReferenceTimingManifest,
    load_reference_timing_manifest,
)
from src.benchmark.taxonomy import DTX_LANE_MAP_VERSION, TAXONOMY_VERSION

SELECTION_SCHEMA = "crux.oaf-tf215-selection/v1"
ARCHIVED_RUN_ID = "oaf-2d67684ad190163e"
REFERENCE_MANIFEST_SHA256 = "96c9971b2c9113e72cb293862badcc88d9146e689ea329f45576d7fb7349f041"
REFERENCE_TIMING_VERSION = "sha256:10868dfb1f87490692d28f6f115dfa164f20be69247115be52a76ef59c954fcf"
REPORT_SHA256 = {
    "summary.json": "919da729e0558eb6dfc919284a564d191a549f8830ae405848b61b967cfdf9f0",
    "items.csv": "206454a600ffb121a56cfe529a8c261599eb4f622757bf683b3693dd7da2bef6",
    "per_song.csv": "384ed0edce89284e50cfe125da8c26621e7442c93cd7a0f1909bf0de1b7cea82",
}
SELECTED_SIMFILE_IDS = (389, 391, 27, 241, 207)
ARCHIVED_SUCCESS_COUNT = 182
ARCHIVED_QUARANTINED_COUNT = 138
SELECTED_RANKS = tuple(q * (ARCHIVED_SUCCESS_COUNT - 1) // 4 for q in range(5))
_UNAVAILABLE = frozenset({errno.ENOENT, errno.EACCES, errno.EPERM})
_BASELINE_FIELDS = (
    "run_id",
    "reference_manifest_sha256",
    "reference_manifest_version",
    "reference_timing_manifest_sha256",
    "reference_timing_version",
    "backend_descriptor",
    "backend_descriptor_sha256",
    "model_id",
    "model_lock_sha256",
    "checkpoint_archive_sha256",
    "inference_config",
    "inference_config_sha256",
    "input_view_id",
    "adapter_revision",
    "canonicalization_revision",
)


class SelectionError(ValueError):
    """The evidence or selection has an invalid binding."""


class SelectionBlocked(SelectionError):
    """Required archived material is unavailable."""


def _io_error(path: Path, error: OSError) -> SelectionError:
    if error.errno in _UNAVAILABLE:
        return SelectionBlocked(f"required input unavailable: {path}")
    return SelectionError(f"could not validate {path}: {error}")


def _regular_file(path: Path) -> None:
    try:
        mode = path.lstat().st_mode
    except OSError as error:
        raise _io_error(path, error) from error
    if stat.S_ISLNK(mode):
        raise SelectionError(f"symlink is forbidden: {path}")
    if not stat.S_ISREG(mode):
        raise SelectionError(f"expected a regular file: {path}")


def _read_file(path: Path) -> bytes:
    _regular_file(path)
    try:
        return read_regular_file_no_follow(path)
    except OSError as error:
        raise _io_error(path, error) from error


def _relative_file(root: Path, value: object) -> Path:
    if (
        not isinstance(value, str)
        or not value
        or "\\" in value
        or "\x00" in value
        or any(part in {"", ".", ".."} for part in value.split("/"))
        or PurePosixPath(value).is_absolute()
    ):
        raise SelectionError(f"unsafe relative artifact path: {value!r}")
    path = root
    if path.is_symlink():
        raise SelectionError(f"symlink is forbidden: {path}")
    for part in value.split("/"):
        path /= part
        if path.is_symlink():
            raise SelectionError(f"symlink is forbidden: {path}")
    return path


def rank_scores(scores: Iterable[tuple[int, Decimal]]) -> list[tuple[int, Decimal]]:
    """Sort finite F1 values and resolve ties by the numerical simfile ID."""
    by_id: dict[int, Decimal] = {}
    for song, score in scores:
        if song in by_id:
            raise SelectionError(f"duplicate score for simfile {song}")
        if song <= 0 or not score.is_finite() or not 0 <= score <= 1:
            raise SelectionError(f"invalid score for simfile {song}")
        by_id[song] = score
    return sorted(by_id.items(), key=lambda row: (row[1], row[0]))


def _load_reports(
    directory: Path,
) -> tuple[dict[str, Any], dict[int, dict[str, str]], list[tuple[int, Decimal]]]:
    documents: dict[str, bytes] = {}
    for filename, expected in REPORT_SHA256.items():
        documents[filename] = _read_file(_relative_file(directory, filename))
        if sha256_hex(documents[filename]) != expected:
            raise SelectionError(f"archived report digest mismatch: {filename}")
    try:
        summary = cast(dict[str, Any], strict_json_loads(documents["summary.json"]))
        identity = summary["identity"]
        items: dict[int, dict[str, str]] = {}
        for item in csv.DictReader(io.StringIO(documents["items.csv"].decode("utf-8"))):
            simfile_id = int(item["simfile_id"])
            if simfile_id <= 0 or simfile_id in items or item["cohort_id"] != ARCHIVED_RUN_ID:
                raise SelectionError("invalid archived item identity")
            items[simfile_id] = item

        scores = []
        for row in csv.DictReader(io.StringIO(documents["per_song.csv"].decode("utf-8"))):
            if row["mode"] != "raw" or row["tolerance_ms"] != "50":
                continue
            if any(
                row[key] != identity[key]
                for key in (
                    "cohort_id",
                    "model_id",
                    "model_lock_sha256",
                    "prediction_map_version",
                    "input_view_id",
                    "scoring_version",
                )
            ):
                raise SelectionError("archived per-song report identity mismatch")
            scores.append((int(row["simfile_id"]), Decimal(row["f1"])))
    except (KeyError, TypeError, InvalidOperation, csv.Error) as error:
        raise SelectionError("malformed archived report") from error
    expected_population = {
        "success": ARCHIVED_SUCCESS_COUNT,
        "quarantined": ARCHIVED_QUARANTINED_COUNT,
    }
    if Counter(item["status"] for item in items.values()) != expected_population:
        raise SelectionError("archived item population mismatch")
    counts = {f"{status}_count": count for status, count in expected_population.items()}
    counts.update(total_count=len(items), failed_count=0, skipped_count=0)
    if any(summary["population"].get(key) != value for key, value in counts.items()):
        raise SelectionError("archived summary population mismatch")
    ordered = rank_scores(scores)
    if {song for song, _ in ordered} != {
        song for song, item in items.items() if item["status"] == "success"
    }:
        raise SelectionError("archived successful items and scores disagree")
    ranked = [ordered[index] for index in SELECTED_RANKS]
    if tuple(song for song, _ in ranked) != SELECTED_SIMFILE_IDS:
        raise SelectionError("archived selected membership mismatch")
    return identity, items, ranked


def _read_run(path: Path) -> tuple[dict[str, Any], str]:
    content = _read_file(path)
    try:
        run = parse_oaf_corpus_run(content, expected_run_id=ARCHIVED_RUN_ID)
    except ValueError as error:
        raise SelectionError(f"archived run is invalid: {error}") from error
    return run, sha256_hex(content)


def _run_identity(
    run: Mapping[str, Any], config: OafModelConfig, descriptor: BackendDescriptor
) -> dict[str, Any]:
    lock_digest = compute_model_lock_sha256(_model_lock_path())
    inference = build_inference_config(config, descriptor, lock_digest)
    inference_digest = inference_config_sha256(inference)
    expected = {
        "run_id": ARCHIVED_RUN_ID,
        "overall_status": "complete",
        "reference_manifest_sha256": REFERENCE_MANIFEST_SHA256,
        "reference_timing_version": REFERENCE_TIMING_VERSION,
        "backend_descriptor": dict(descriptor.payload),
        "backend_descriptor_sha256": descriptor.sha256,
        "model_id": config.model_id,
        "model_lock_sha256": lock_digest,
        "checkpoint_archive_sha256": config.checkpoint.archive_sha256,
        "input_view_id": OAF_FULL_MIX_INPUT_VIEW_ID,
        "inference_config": inference,
        "inference_config_sha256": inference_digest,
        "adapter_revision": inference["adapter_revision"],
        "canonicalization_revision": inference["canonicalization_revision"],
        "include_simfile_ids": [],
        "exclude_simfile_ids": [],
        "success_count": ARCHIVED_SUCCESS_COUNT,
        "quarantined_count": ARCHIVED_QUARANTINED_COUNT,
        "failed_count": 0,
        "skipped_count": 0,
    }
    for field, value in expected.items():
        if run.get(field) != value:
            raise SelectionError(f"archived run {field} mismatch")
    timing_digest = _required_digest(run, "reference_timing_manifest_sha256")
    reference_version = run.get("reference_manifest_version")
    if not isinstance(reference_version, str) or not reference_version.startswith("sha256:"):
        raise SelectionError("archived run reference manifest version is invalid")
    _required_digest(
        {"reference_manifest_version": reference_version[7:]}, "reference_manifest_version"
    )
    if (
        build_run_id(
            REFERENCE_MANIFEST_SHA256,
            timing_digest,
            descriptor.sha256,
            lock_digest,
            config.checkpoint.archive_sha256,
            inference_digest,
        )
        != ARCHIVED_RUN_ID
    ):
        raise SelectionError("archived run ID does not derive from its semantic identity")
    return {field: run[field] for field in _BASELINE_FIELDS}


def _required_digest(value: Mapping[str, Any], field: str) -> str:
    digest = value.get(field)
    if not isinstance(digest, str):
        raise SelectionError(f"invalid {field}")
    try:
        return require_sha256(digest, field)
    except ValueError as error:
        raise SelectionError(f"invalid {field}") from error


def _reference_rows(
    reference_path: Path, timing_path: Path, run: Mapping[str, Any]
) -> tuple[
    LoadedReferenceSetManifest, LoadedReferenceTimingManifest, tuple[LoadedReferenceSetRow, ...]
]:
    for path in (reference_path, timing_path):
        _regular_file(path)
    try:
        reference = load_reference_set_manifest(reference_path)
        timing = load_reference_timing_manifest(timing_path)
    except (FileNotFoundError, PermissionError) as error:
        raise SelectionBlocked(
            f"required reference or timing manifest unavailable: {error}"
        ) from error
    except (OSError, ValueError) as error:
        raise SelectionError(f"invalid reference or timing manifest: {error}") from error
    if (
        (
            reference.manifest_sha256,
            reference.corpus_version,
            timing.manifest_sha256,
            timing.corpus_version,
        )
        != (
            run["reference_manifest_sha256"],
            run["reference_manifest_version"],
            run["reference_timing_manifest_sha256"],
            run["reference_timing_version"],
        )
        or reference.source_reference_timing_manifest_sha256 != timing.manifest_sha256
        or reference.source_reference_timing_version != timing.corpus_version
    ):
        raise SelectionError("archived reference and timing manifest identities disagree")
    by_id = {row.view.simfile_id: row for row in reference.rows}
    try:
        selected = tuple(by_id[song] for song in SELECTED_SIMFILE_IDS)
    except KeyError as error:
        raise SelectionError(
            f"selected simfile {error} is absent from the reference manifest"
        ) from error
    if any(row.view.eligibility_status != "eligible" for row in selected):
        raise SelectionError("archived reference manifest marks a selected song ineligible")
    return reference, timing, selected


def _preflight_selected_references(
    reference: LoadedReferenceSetManifest,
    timing: LoadedReferenceTimingManifest,
    selected: tuple[LoadedReferenceSetRow, ...],
    timing_root: Path,
) -> None:
    timing_rows = {row.view.simfile_id: row for row in timing.rows}
    for row in selected:
        match = timing_rows.get(row.view.simfile_id)
        if match is None or match.view.timing_status != "ready":
            raise SelectionError(f"selected simfile {row.view.simfile_id} has no ready timing row")
        # Preflight folds missing and malformed native artifacts into one error.
        # Check availability first so the experiment can distinguish blockers.
        path = _relative_file(timing_root, match.view.reference_events_cache_path)
        _regular_file(path)
    try:
        preflight_reference_mappings(
            replace(reference, rows=selected), timing, timing_output_root=timing_root
        )
    except (OSError, ValueError) as error:
        raise SelectionError(f"archived reference preflight failed: {error}") from error


def _read_prediction(
    root: Path,
    row: Mapping[str, Any],
    descriptor: BackendDescriptor,
    expected_count: int,
    expected_classes: Mapping[str, int],
) -> PredictionArtifact:
    path = _relative_file(root, row.get("prediction_path"))
    content = _read_file(path)
    try:
        artifact = read_prediction_artifact(content)
    except ValueError as error:
        raise SelectionError(f"invalid archived prediction {path}: {error}") from error
    if artifact.prediction.descriptor != descriptor:
        raise SelectionError(f"archived prediction descriptor mismatch: {path}")
    if not prediction_artifact_matches_run_row(
        artifact, row, expected_input_view_id=OAF_FULL_MIX_INPUT_VIEW_ID
    ):
        raise SelectionError(f"archived prediction does not match its run row: {path}")
    events = artifact.prediction.events
    if any(event.prediction_map_version != OAF_PREDICTION_MAP_ID for event in events):
        raise SelectionError(f"archived prediction map mismatch: {path}")
    counts = Counter(event.native.native_class_id for event in events)
    if artifact.event_count != expected_count or counts != expected_classes:
        raise SelectionError(f"archived prediction native event or class counts mismatch: {path}")
    return artifact


def _archived_counts(item: Mapping[str, str]) -> tuple[int, dict[str, int]]:
    try:
        count = int(item["prediction_native_event_count"])
        classes: dict[str, int] = {}
        for entry in filter(None, item["prediction_native_class_counts"].split("|")):
            name, text = entry.split("=")
            if not name or name in classes or int(text) <= 0:
                raise ValueError("invalid class count")
            classes[name] = int(text)
    except (KeyError, ValueError) as error:
        raise SelectionError("archived report event/class counts are invalid") from error
    if count != sum(classes.values()):
        raise SelectionError("archived report event/class counts disagree")
    return count, classes


def _source_audio(
    loaded: LoadedReferenceSetRow,
    artifact: PredictionArtifact,
    cache_dir: Path,
    index: CacheIndexStore,
) -> ResolvedSourceAudio:
    audio = artifact.prediction.audio
    reference = loaded.source_row
    if (reference.get("source_audio_key"), reference.get("source_audio_content_hash")) != (
        audio.source_audio_id,
        audio.source_audio_sha256,
    ):
        raise SelectionError(
            f"archived prediction source does not match reference: {loaded.view.simfile_id}"
        )
    try:
        return resolve_source_audio(
            reference,
            cache_dir,
            index,
            source_audio_content_hash=audio.source_audio_sha256,
        )
    except OSError as error:
        raise _io_error(cache_dir, error) from error
    except ValueError as error:
        if str(error) in {"verified source audio unavailable", "verified cache body unavailable"}:
            digest = audio.source_audio_sha256
            path = _relative_file(cache_dir, f"sha256/{digest[:2]}/{digest}")
            # The cache resolver does not distinguish absent bytes from corrupt
            # bytes in its exceptions. Make that distinction on this failure path.
            if sha256_hex(_read_file(path)) != digest:
                raise SelectionError(
                    f"archived source audio failed its digest check: {path}"
                ) from error
            raise SelectionBlocked(
                f"verified source cache binding unavailable: {audio.source_audio_id}"
            ) from error
        raise SelectionError(
            f"invalid archived source audio {audio.source_audio_id}: {error}"
        ) from error


def _load_inputs(
    baseline: Path,
    run: Mapping[str, Any],
    descriptor: BackendDescriptor,
    report_items: dict[int, dict[str, str]],
    ranked: list[tuple[int, Decimal]],
) -> list[tuple[dict[str, Any], PredictionArtifact]]:
    by_id = {row["simfile_id"]: row for row in run["items"]}
    status_by_disposition = {
        "inferred": "success",
        "resumed": "success",
        "quarantined": "quarantined",
    }
    actual = {
        song: status_by_disposition.get(row.get("execution_disposition"))
        for song, row in by_id.items()
    }
    if actual != {song: row["status"] for song, row in report_items.items()}:
        raise SelectionError("archived run population does not match the item report")
    prepared = []
    for rank, (song, score) in zip(SELECTED_RANKS, ranked, strict=True):
        row = by_id[song]
        count, classes = _archived_counts(report_items[song])
        artifact = _read_prediction(baseline.parent, row, descriptor, count, classes)
        audio = artifact.prediction.audio
        prepared.append(
            (
                {
                    "rank": rank,
                    "simfile_id": song,
                    "archived_f1": str(score),
                    "prediction_native_event_count": count,
                    "prediction_native_class_counts": classes,
                    "prediction_path": row["prediction_path"],
                    "prediction_artifact_sha256": artifact.artifact_sha256,
                    "wav_path": f"{song}.wav",
                    "source_audio_id": audio.source_audio_id,
                    "source_audio_sha256": audio.source_audio_sha256,
                    "input_audio_sha256": audio.input_audio_sha256,
                    "audio_frame_count": audio.audio_frame_count,
                },
                artifact,
            )
        )
    return prepared


def prepare_selection(
    *,
    baseline_evidence_dir: Path,
    baseline_run: Path,
    reference_manifest: Path,
    timing_manifest: Path,
    cache_dir: Path,
    input_root: Path,
) -> dict[str, Any]:
    """Verify original evidence and materialize exactly the archived five WAVs."""
    if input_root.exists() or input_root.is_symlink():
        raise SelectionError(f"the experimental input directory must be new: {input_root}")
    identity, items, ranked = _load_reports(baseline_evidence_dir)
    config = load_model_config()
    descriptor = _expected_oaf_descriptor(config)
    run, run_digest = _read_run(baseline_run)
    baseline = _run_identity(run, config, descriptor)
    expected_identity = {
        **{
            key: baseline[key]
            for key in (
                "backend_descriptor_sha256",
                "model_id",
                "model_lock_sha256",
                "reference_manifest_sha256",
                "reference_timing_version",
                "input_view_id",
            )
        },
        "cohort_id": ARCHIVED_RUN_ID,
        "backend_id": config.backend_id,
        "prediction_map_version": OAF_PREDICTION_MAP_ID,
        "taxonomy_version": TAXONOMY_VERSION,
        "lane_map_version": DTX_LANE_MAP_VERSION,
    }
    if any(identity.get(key) != value for key, value in expected_identity.items()):
        raise SelectionError("archived run does not match the summary identity")
    reference, timing, selected = _reference_rows(reference_manifest, timing_manifest, run)
    _preflight_selected_references(reference, timing, selected, timing_manifest.parent.parent)
    prepared = _load_inputs(baseline_run, run, descriptor, items, ranked)
    try:
        index = CacheIndexStore.load(cache_dir)
    except OSError as error:
        raise _io_error(cache_dir, error) from error
    except ValueError as error:
        raise SelectionError(f"invalid archived source cache index: {error}") from error

    input_root.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".selected-wavs-", dir=input_root.parent) as staging:
        temporary = Path(staging)
        for loaded, (row, artifact) in zip(selected, prepared, strict=True):
            source = _source_audio(loaded, artifact, cache_dir, index)
            try:
                audio = materialize_full_mix_audio(
                    source,
                    temporary / row["wav_path"],
                    input_root=temporary,
                    input_view_id=OAF_FULL_MIX_INPUT_VIEW_ID,
                    max_input_audio_frames=config.max_input_audio_frames,
                )
            except (OSError, RuntimeError, ValueError) as error:
                raise SelectionError(
                    f"could not materialize simfile {row['simfile_id']}: {error}"
                ) from error
            _check_audio(audio, row, artifact, descriptor)
        # A fresh directory makes setup retryable without replacing any prior
        # selection. Validation above finishes before moving the five WAVs.
        input_root.mkdir()
        for row, _ in prepared:
            (temporary / row["wav_path"]).rename(input_root / row["wav_path"])

    return {
        "schema": SELECTION_SCHEMA,
        "report_sha256": dict(REPORT_SHA256),
        "baseline_run_sha256": run_digest,
        "baseline_identity": baseline,
        "prediction_map_version": OAF_PREDICTION_MAP_ID,
        "taxonomy_version": TAXONOMY_VERSION,
        "lane_map_version": DTX_LANE_MAP_VERSION,
        "items": [row for row, _ in prepared],
    }


def _check_audio(
    audio: CanonicalAudio,
    row: Mapping[str, Any],
    artifact: PredictionArtifact,
    descriptor: BackendDescriptor,
) -> None:
    if (
        audio.audio_frame_count != row["audio_frame_count"]
        or audio.audio_frame_count != artifact.prediction.audio.audio_frame_count
        or audio.input_audio_sha256 != row["input_audio_sha256"]
        or not prediction_artifact_matches_audio(
            artifact,
            source_audio_id=row["source_audio_id"],
            source_audio_sha256=row["source_audio_sha256"],
            audio=audio,
            descriptor=descriptor,
            prediction_map_version=OAF_PREDICTION_MAP_ID,
        )
    ):
        raise SelectionError(f"canonical WAV identity mismatch for simfile {row['simfile_id']}")


def selection_sha256(selection: Mapping[str, object]) -> str:
    """Hash the canonical selection object without a file newline."""
    return sha256_hex(canonical_json_bytes(cast(JsonValue, dict(selection))))


def verify_selection(
    *,
    selection_path: Path,
    expected_selection_sha256: str,
    baseline_run: Path,
    input_root: Path,
) -> tuple[dict[str, Any], tuple[tuple[CanonicalAudio, PredictionArtifact], ...]]:
    """Recheck the seal, archived predictions, and WAVs without source access."""
    content = _read_file(selection_path).removesuffix(b"\n")
    try:
        selection = strict_json_loads(content, require_canonical=True)
    except ValueError as error:
        raise SelectionError(f"invalid selection document: {error}") from error
    if not isinstance(selection, dict) or selection_sha256(selection) != expected_selection_sha256:
        raise SelectionError("selection digest does not match")
    expected = {
        "schema": SELECTION_SCHEMA,
        "report_sha256": REPORT_SHA256,
        "prediction_map_version": OAF_PREDICTION_MAP_ID,
        "taxonomy_version": TAXONOMY_VERSION,
        "lane_map_version": DTX_LANE_MAP_VERSION,
    }
    if any(selection.get(key) != value for key, value in expected.items()):
        raise SelectionError("selection does not match the pinned report or scoring identities")
    rows = selection.get("items")
    if (
        not isinstance(rows, list)
        or len(rows) != len(SELECTED_SIMFILE_IDS)
        or any(
            not isinstance(row, dict) or row.get("simfile_id") != song or row.get("rank") != rank
            for row, song, rank in zip(rows, SELECTED_SIMFILE_IDS, SELECTED_RANKS, strict=True)
        )
    ):
        raise SelectionError("selection does not match the pinned membership")
    run, digest = _read_run(baseline_run)
    if digest != selection.get("baseline_run_sha256"):
        raise SelectionError("archived run digest no longer matches the selection")
    config = load_model_config()
    descriptor = _expected_oaf_descriptor(config)
    if _run_identity(run, config, descriptor) != selection.get("baseline_identity"):
        raise SelectionError("archived run no longer matches the sealed lineage")

    run_rows = {row["simfile_id"]: row for row in run["items"]}
    inputs = []
    for value in rows:
        row = cast(dict[str, Any], value)
        archive = run_rows.get(row["simfile_id"])
        if archive is None or archive.get("execution_disposition") not in {"inferred", "resumed"}:
            raise SelectionError("sealed song is not successful in the archived run")
        for key in (
            "prediction_path",
            "prediction_artifact_sha256",
            "source_audio_id",
            "source_audio_sha256",
            "input_audio_sha256",
        ):
            if key not in row or row[key] != archive.get(key):
                raise SelectionError(f"sealed song {row['simfile_id']} has a different {key}")
        classes = row.get("prediction_native_class_counts")
        count = row.get("prediction_native_event_count")
        frames = row.get("audio_frame_count")
        if (
            type(count) is not int
            or count < 0
            or type(frames) is not int
            or frames <= 0
            or not isinstance(classes, dict)
            or any(type(n) is not int or n <= 0 for n in classes.values())
            or sum(classes.values()) != count
        ):
            raise SelectionError(f"invalid counts for sealed song {row['simfile_id']}")
        artifact = _read_prediction(baseline_run.parent, row, descriptor, count, classes)
        wav = _relative_file(input_root, row.get("wav_path"))
        _regular_file(wav)
        try:
            audio = load_materialized_audio(
                wav,
                source_audio_id=row["source_audio_id"],
                source_audio_sha256=row["source_audio_sha256"],
                input_view_id=OAF_FULL_MIX_INPUT_VIEW_ID,
                max_input_audio_frames=config.max_input_audio_frames,
            )
        except OSError as error:
            raise _io_error(wav, error) from error
        except ValueError as error:
            raise SelectionError(f"canonical WAV is invalid: {wav}: {error}") from error
        _check_audio(audio, row, artifact, descriptor)
        inputs.append((audio, artifact))
    return selection, tuple(inputs)
