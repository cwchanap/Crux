"""Small deterministic fixtures for the offline HPA-396 pilot tests."""

from __future__ import annotations

import hashlib
import io
import struct
import wave
from copy import deepcopy
from pathlib import Path

from src.benchmark.backends.base import CanonicalAudio, NativeEvent, NativePrediction
from src.benchmark.corpus_manifest import render_manifest
from src.benchmark.mapping import map_oaf_prediction
from src.benchmark.oaf_corpus_run import OAF_FULL_MIX_INPUT_VIEW_ID
from src.benchmark.prediction_artifact import render_prediction_artifact
from src.benchmark.reference_set import map_reference_events
from src.benchmark.reference_set_manifest import (
    LoadedReferenceSetManifest,
    LoadedReferenceSetRow,
    ReferenceSetRowView,
)
from src.benchmark.reference_timing import NativeReferenceEvent
from src.benchmark.reference_timing_manifest import LoadedReferenceTimingManifest
from src.benchmark.separation_handoff import LoadedSeparationPilotManifest
from src.benchmark.separation_pilot import SPLEETER_INPUT_VIEW_ID

SHA_A = "a" * 64
SHA_B = "b" * 64
SHA_C = "c" * 64
SHA_D = "d" * 64
CRUX_COMMIT = "1" * 40
IDM_MODEL_LOCK_PATH = Path(__file__).resolve().parents[2] / "runtime" / "idm" / "model.json"


def canonical_wav(*, frame_count: int = 32, sample: int = 0) -> bytes:
    """Return one tiny canonical 44.1 kHz mono PCM16 WAV."""
    frames = struct.pack("<" + "h" * frame_count, *(sample for _ in range(frame_count)))
    stream = io.BytesIO()
    output = wave.open(stream, "wb")
    output.setnchannels(1)
    output.setsampwidth(2)
    output.setframerate(44100)
    output.writeframes(frames)
    output.close()
    return stream.getvalue()


def sha256(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def oaf_descriptor() -> object:
    from src.benchmark.backend_identity import build_descriptor

    payload = {
        "architecture_id": "magenta-oaf-model-tpu-drums-v1",
        "backend_id": "magenta-egmd-tf1-94529798-8hit-v1",
        "descriptor_schema": "crux.transcription-backend-descriptor/v2",
        "model_id": "magenta-egmd-ckpt-569400-v1",
        "native_metadata_schema_id": "magenta-oaf-native-metadata-v1",
        "native_output_space_id": "magenta-oaf-midi88-a0-v1",
        "prediction_schema": "crux.drum-prediction-events/v2",
        "training_data_map_id": "magenta-egmd-data-8hit-94529798-v1",
        "upstream_source_commit": "94529798dfbbb14c27ddfd76f23027dc8e2ce185",
    }
    return build_descriptor(payload, frozenset(payload), payload["descriptor_schema"])


def oaf_artifact(audio: CanonicalAudio) -> bytes:
    descriptor = oaf_descriptor()
    native = NativePrediction(
        audio=audio,
        descriptor=descriptor,  # type: ignore[arg-type]
        events=(
            NativeEvent(
                time_sec=0.25,
                native_class_id="midi_36",
                model_output_bin=15,
                native_midi_note=36,
                native_metadata={"upstream_8hit_group_id": "kick"},
                confidence=0.9,
                velocity_midi=100,
            ),
        ),
    )
    mapped, _ = map_oaf_prediction(native)
    return render_prediction_artifact(mapped)


def reference_mapping(simfile_id: int):
    return map_reference_events(
        (
            NativeReferenceEvent(
                simfile_id=simfile_id,
                selected_chart_key=f"{simfile_id}/chart.dtx",
                selected_chart_content_hash=SHA_A,
                source_audio_key=f"{simfile_id}/audio.wav",
                source_audio_content_hash=SHA_B,
                source_order=0,
                measure=1,
                position=0.0,
                lane_id="13",
                note_id="kick-0",
                chart_time_sec=0.25,
                audio_time_sec=0.25,
            ),
        )
    )


def loaded_reference_manifests(
    *,
    ids: tuple[int, ...],
    reference_sha256: str,
    reference_version: str,
    timing_sha256: str,
    timing_version: str,
) -> tuple[LoadedReferenceSetManifest, LoadedReferenceTimingManifest, dict[int, object]]:
    rows = tuple(
        LoadedReferenceSetRow(
            source_row={
                "selected_chart_key": f"{simfile_id}/chart.dtx",
                "selected_chart_content_hash": SHA_A,
                "source_audio_key": f"{simfile_id}/audio.wav",
                "source_audio_content_hash": SHA_B,
                "source_endpoint_sha256": SHA_C,
                "source_bucket": "simfile-dtx",
            },
            view=ReferenceSetRowView(
                simfile_id=simfile_id,
                eligibility_status="eligible",
                eligibility_reason_codes=(),
                eligibility_warnings=(),
                mapped_event_count=1,
                common_scored_event_count=1,
                ignored_event_count=0,
                unmapped_event_count=0,
                duplicate_common_event_count=0,
            ),
        )
        for simfile_id in ids
    )
    reference = LoadedReferenceSetManifest(
        manifest_sha256=reference_sha256,
        corpus_version=reference_version,
        source_reference_timing_manifest_sha256=timing_sha256,
        source_reference_timing_version=timing_version,
        rows=rows,
    )
    timing = LoadedReferenceTimingManifest(
        manifest_sha256=timing_sha256,
        corpus_version=timing_version,
        rows=(),
    )
    return reference, timing, {simfile_id: reference_mapping(simfile_id) for simfile_id in ids}


def write_actual_handoff(path: Path, handoff: LoadedSeparationPilotManifest) -> bytes:
    """Render a synthetic handoff through the production immutable loader."""
    rows: list[dict[str, object]] = []
    comparison_names = (
        "summary.json",
        "summary.md",
        "spleeter/paired_per_song.csv",
        "spleeter/paired_per_class.csv",
        "htdemucs/paired_per_song.csv",
        "htdemucs/paired_per_class.csv",
    )
    for source in handoff.rows:
        row = deepcopy(dict(source))
        row.update(
            {
                "reviewed_subset_manifest_sha256": SHA_A,
                "reviewed_subset_manifest_version": "sha256:" + SHA_B,
                "parent_oaf_run_id": "oaf-parent-run",
                "full_mix": {
                    "status": "failed",
                    "failure_code": "source_audio_unavailable",
                    "separator_lock_sha256": None,
                    "input_view_id": OAF_FULL_MIX_INPUT_VIEW_ID,
                    "stem": None,
                    "input": None,
                    "prediction": None,
                },
                "spleeter": {
                    "status": "separation_failed",
                    "failure_code": "separation_failed",
                    "separator_lock_sha256": SHA_D,
                    "input_view_id": SPLEETER_INPUT_VIEW_ID,
                    "stem": None,
                    "input": None,
                    "prediction": None,
                },
                "comparison_artifacts": {
                    name: {"path": f"comparison/{name}", "sha256": SHA_A}
                    for name in comparison_names
                },
            }
        )
        rows.append(row)
    rendered = render_manifest(tuple(rows))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(rendered.content)
    return rendered.content
