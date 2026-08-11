from __future__ import annotations

import inspect
import io
import struct
from dataclasses import fields as dataclass_fields
from pathlib import Path

import pytest
from click.testing import CliRunner

from src.benchmark.backend_identity import (
    OAF_BACKEND_ID,
    OAF_DESCRIPTOR_SCHEMA,
    build_descriptor,
    sha256_hex,
)
from src.benchmark.backend_registry import BackendUnavailable, default_backend_registry
from src.benchmark.backends import CanonicalAudio, NativeEvent, NativePrediction
from src.benchmark.backends.oaf import OafBackend, OafBackendError, build_docker_command
from src.benchmark.mapping import map_oaf_prediction
from src.benchmark.prediction_artifact import (
    PREDICTION_SCHEMA,
    read_prediction_artifact,
    render_prediction_artifact,
)
from src.benchmark.scorer_input import read_scorer_events
from src.benchmark.taxonomy import OAF_PREDICTION_MAP_ID
from src.cli.main import main


def _descriptor() -> object:
    payload = {
        "architecture_id": "magenta-oaf-model-tpu-drums-v1",
        "backend_id": OAF_BACKEND_ID,
        "descriptor_schema": OAF_DESCRIPTOR_SCHEMA,
        "model_id": "magenta-egmd-ckpt-569400-v1",
        "native_metadata_schema_id": "magenta-oaf-native-metadata-v1",
        "native_output_space_id": "magenta-oaf-midi88-a0-v1",
        "prediction_schema": "crux.drum-prediction-events/v2",
        "training_data_map_id": "magenta-egmd-data-8hit-94529798-v1",
        "upstream_source_commit": "94529798dfbbb14c27ddfd76f23027dc8e2ce185",
    }
    return build_descriptor(payload, frozenset(payload), OAF_DESCRIPTOR_SCHEMA)


def _audio(path: Path) -> CanonicalAudio:
    content = (
        struct.pack("<4sI4s", b"RIFF", 40, b"WAVE")
        + struct.pack("<4sIHHIIHH", b"fmt ", 16, 1, 1, 44100, 88200, 2, 16)
        + struct.pack("<4sI", b"data", 4)
        + b"\x00\x00\x00\x00"
    )
    digest = sha256_hex(content)
    path.write_bytes(content)
    return CanonicalAudio(path, "song", digest, "view", digest, len(content), 44100, 1, 2, 2)


def _native(group: str | None, *, note: int = 38) -> NativeEvent:
    return NativeEvent(
        time_sec=0.5,
        native_class_id=f"midi_{note}",
        model_output_bin=note - 21,
        native_midi_note=note,
        native_metadata={"upstream_8hit_group_id": group},
        confidence=0.75,
        velocity_midi=96,
    )


def _prediction(tmp_path: Path, events: tuple[NativeEvent, ...]) -> NativePrediction:
    return NativePrediction(_audio(tmp_path / "song.wav"), _descriptor(), events)


def test_task_d_backend_contract_and_registry() -> None:
    field_names = {field.name for field in dataclass_fields(NativePrediction)}
    assert field_names == {"audio", "descriptor", "events"}
    assert not hasattr(NativePrediction, "verify")
    assert not hasattr(
        __import__("src.benchmark.backends", fromlist=["x"]), "Backend" + "Verification"
    )
    registry = default_backend_registry()
    assert registry.default_backend_id == "oaf"
    assert set(registry.factories) == {"oaf"}
    with pytest.raises(BackendUnavailable, match="unknown backend"):
        registry.create("legacy")


def test_task_d_oaf_mapping_and_scorer_bridge(tmp_path: Path) -> None:
    prediction = _prediction(tmp_path, (_native("hihat", note=46), _native("sticks", note=42)))
    mapped, diagnostics = map_oaf_prediction(prediction)
    assert mapped.events[0].common_class == "hihat"
    assert mapped.events[0].canonical_class is None
    assert mapped.events[1].mapping_status == "unmapped"
    assert diagnostics.unmapped == {"sticks": 1}
    artifact = read_prediction_artifact(render_prediction_artifact(mapped))
    events = read_scorer_events(artifact.content)
    assert len(events) == 1
    assert events[0].canonical_class == "hihat"
    assert events[0].metadata["detailed_canonical_class"] is None


def test_task_d_oaf_prediction_round_trip_preserves_common_mapping_seam(
    tmp_path: Path,
) -> None:
    native_events = (
        NativeEvent(
            time_sec=0.5,
            native_class_id="midi_46",
            model_output_bin=25,
            native_midi_note=46,
            native_metadata={"upstream_8hit_group_id": "hihat"},
            confidence=0.9,
            velocity_midi=100,
        ),
        NativeEvent(
            time_sec=1.0,
            native_class_id="midi_48",
            model_output_bin=27,
            native_midi_note=48,
            native_metadata={"upstream_8hit_group_id": "toms"},
            confidence=0.8,
            velocity_midi=101,
        ),
        NativeEvent(
            time_sec=1.5,
            native_class_id="midi_53",
            model_output_bin=32,
            native_midi_note=53,
            native_metadata={"upstream_8hit_group_id": "ride_bell"},
            confidence=0.7,
            velocity_midi=102,
        ),
        NativeEvent(
            time_sec=2.0,
            native_class_id="midi_75",
            model_output_bin=54,
            native_midi_note=75,
            native_metadata={"upstream_8hit_group_id": "sticks"},
            confidence=0.6,
            velocity_midi=103,
        ),
    )
    mapped, diagnostics = map_oaf_prediction(_prediction(tmp_path, native_events))

    assert diagnostics.unmapped == {"sticks": 1}
    assert [event.prediction_map_version for event in mapped.events] == [
        OAF_PREDICTION_MAP_ID
    ] * len(native_events)
    assert [
        (event.canonical_class, event.common_class, event.mapping_status) for event in mapped.events
    ] == [
        (None, "hihat", "mapped"),
        (None, "tom", "mapped"),
        ("ride", "ride", "mapped"),
        (None, None, "unmapped"),
    ]

    artifact = read_prediction_artifact(render_prediction_artifact(mapped))
    assert f'"schema":"{PREDICTION_SCHEMA}"'.encode() in artifact.content
    round_tripped = artifact.prediction.events
    assert [
        (
            event.native.native_class_id,
            event.native.model_output_bin,
            event.native.native_midi_note,
            event.native.native_metadata["upstream_8hit_group_id"],
            event.native.confidence,
            event.native.velocity_midi,
        )
        for event in round_tripped
    ] == [
        (
            event.native_class_id,
            event.model_output_bin,
            event.native_midi_note,
            event.native_metadata["upstream_8hit_group_id"],
            event.confidence,
            event.velocity_midi,
        )
        for event in native_events
    ]
    assert [
        (event.canonical_class, event.common_class, event.mapping_status) for event in round_tripped
    ] == [
        (None, "hihat", "mapped"),
        (None, "tom", "mapped"),
        ("ride", "ride", "mapped"),
        (None, None, "unmapped"),
    ]
    assert [event.prediction_map_version for event in round_tripped] == [
        OAF_PREDICTION_MAP_ID
    ] * len(native_events)

    scorer_events = read_scorer_events(artifact.content)
    assert [(event.canonical_class, event.time_sec) for event in scorer_events] == [
        ("hihat", 0.5),
        ("tom", 1.0),
        ("ride", 1.5),
    ]
    assert [event.metadata["detailed_canonical_class"] for event in scorer_events] == [
        None,
        None,
        "ride",
    ]


def test_task_d_mapping_rejects_descriptor_identity_mismatch(tmp_path: Path) -> None:
    prediction = _prediction(tmp_path, (_native("snare"),))
    bad = dict(prediction.descriptor.payload)
    bad["backend_id"] = "other"
    descriptor = build_descriptor(bad, frozenset(bad), OAF_DESCRIPTOR_SCHEMA)
    with pytest.raises(ValueError, match="backend_id"):
        map_oaf_prediction(NativePrediction(prediction.audio, descriptor, prediction.events))


class _FakeWorker:
    def __init__(self, ready: dict[str, object], response: dict[str, object]) -> None:
        self.ready = ready
        self.response = response
        self.requests: list[str] = []
        self.close_count = 0

    def request(self, path: str) -> dict[str, object]:
        self.requests.append(path)
        if "error" in self.response:
            raise RuntimeError(self.response["error"])
        return {"id": "one", "events": self.response["events"]}

    def close(self) -> None:
        self.close_count += 1


def test_task_d_oaf_adapter_translates_correlated_worker_error_and_reuses_process(
    tmp_path: Path,
) -> None:
    import sys

    input_root = tmp_path / "input"
    checkpoint = tmp_path / "checkpoint"
    input_root.mkdir()
    checkpoint.mkdir()
    audio = _audio(input_root / "song.wav")
    script = tmp_path / "worker.py"
    script.write_text(
        "import json, sys\n"
        "print(json.dumps({'type':'ready','backend_id':%r,'restored_tensor_count':78}), flush=True)\n"
        "first = True\n"
        "for line in sys.stdin:\n"
        "    request = json.loads(line)\n"
        "    if first:\n"
        "        first = False\n"
        "        print(json.dumps({'id':request['id'],'error':{'code':'audio_unavailable','message':'audio unavailable'}}), flush=True)\n"
        "    else:\n"
        "        print(json.dumps({'id':request['id'],'events':[]}), flush=True)\n"
        % OAF_BACKEND_ID,
        encoding="utf-8",
    )
    starts: list[list[str]] = []

    def process_factory(command: object, **kwargs: object) -> object:
        del command
        starts.append([sys.executable, str(script)])
        from src.benchmark.worker_process import WorkerProcess

        return WorkerProcess.start(
            starts[-1], timeout_seconds=float(kwargs.get("timeout_seconds", 1.0))
        )

    backend = OafBackend(checkpoint, input_root, process_factory=process_factory)
    try:
        with pytest.raises(OafBackendError) as raised:
            backend.transcribe(audio)
        assert raised.value.code == "audio_unavailable"
        assert backend.transcribe(audio).events == ()
        assert len(starts) == 1
    finally:
        backend.close()


def test_task_d_oaf_adapter_validates_ready_reuses_worker_and_closes(tmp_path: Path) -> None:
    input_root = tmp_path / "input"
    checkpoint = tmp_path / "checkpoint"
    input_root.mkdir()
    checkpoint.mkdir()
    audio = _audio(input_root / "song.wav")
    worker = _FakeWorker(
        {"type": "ready", "backend_id": OAF_BACKEND_ID, "restored_tensor_count": 78},
        {
            "events": [
                {
                    "time_sec": 0.5,
                    "native_class_id": "midi_38",
                    "model_output_bin": 17,
                    "native_midi_note": 38,
                    "upstream_8hit_group_id": "snare",
                    "confidence": 0.75,
                    "velocity_midi": 96,
                }
            ]
        },
    )
    backend = OafBackend(
        checkpoint,
        input_root,
        process_factory=lambda *_args, **_kwargs: worker,
    )
    assert backend.transcribe(audio).events[0].native_class_id == "midi_38"
    assert backend.transcribe(audio).events[0].native_metadata["upstream_8hit_group_id"] == "snare"
    assert worker.requests == ["song.wav", "song.wav"]
    backend.close()
    backend.close()
    assert worker.close_count == 1


def test_task_d_oaf_adapter_rejects_ready_override_keyword() -> None:
    assert "ready" not in inspect.signature(OafBackend).parameters


@pytest.mark.parametrize(
    ("backend_id", "restored_tensor_count", "error_code"),
    [
        ("wrong-backend", 78, "worker_identity_invalid"),
        (OAF_BACKEND_ID, 77, "worker_identity_invalid"),
    ],
)
def test_task_d_oaf_adapter_rejects_process_ready_identity(
    tmp_path: Path,
    backend_id: str,
    restored_tensor_count: int,
    error_code: str,
) -> None:
    input_root = tmp_path / "input"
    checkpoint = tmp_path / "checkpoint"
    input_root.mkdir()
    checkpoint.mkdir()
    audio = _audio(input_root / "song.wav")
    worker = _FakeWorker(
        {
            "type": "ready",
            "backend_id": backend_id,
            "restored_tensor_count": restored_tensor_count,
        },
        {"events": []},
    )
    backend = OafBackend(
        checkpoint,
        input_root,
        process_factory=lambda *_args, **_kwargs: worker,
    )
    with pytest.raises(OafBackendError) as raised:
        backend.transcribe(audio)
    assert raised.value.code == error_code
    assert worker.close_count == 1


def test_task_d_oaf_adapter_rejects_outside_input(tmp_path: Path) -> None:
    root = tmp_path / "input"
    checkpoint = tmp_path / "checkpoint"
    root.mkdir()
    checkpoint.mkdir()
    outside = _audio(tmp_path / "outside.wav")
    with pytest.raises(OafBackendError, match="input root"):
        OafBackend(checkpoint, root).transcribe(outside)


def test_task_d_oaf_launch_command_is_read_only_and_networkless(tmp_path: Path) -> None:
    command = build_docker_command(tmp_path / "checkpoint", tmp_path / "input")
    assert command[:6] == ["docker", "run", "--rm", "-i", "--network=none", "--read-only"]
    assert "--tmpfs=/tmp:rw" in command
    assert f"--mount=type=bind,src={tmp_path / 'checkpoint'},dst=/model,readonly" in command
    assert f"--mount=type=bind,src={tmp_path / 'input'},dst=/input,readonly" in command
    assert "--workdir=/input" in command
    assert command[-1] == "crux-oaf-tf1:local"


def test_task_d_docker_uses_non_root_defaults_without_workflow_build_args() -> None:
    repository = Path(__file__).parents[2]
    dockerfile = (repository / "runtime/oaf_tf1/Dockerfile").read_text(encoding="utf-8")
    workflow = (repository / ".github/workflows/oaf-smoke.yml").read_text(encoding="utf-8")

    assert "ARG RUNTIME_UID=65532" in dockerfile
    assert "ARG RUNTIME_GID=65532" in dockerfile
    assert "docker build -f runtime/oaf_tf1/Dockerfile -t crux-oaf-tf1:local ." in workflow
    assert "--build-arg RUNTIME_UID" not in workflow
    assert "--build-arg RUNTIME_GID" not in workflow


def test_task_d_docker_drops_retained_runner_manifest_consumer() -> None:
    repository = Path(__file__).parents[2]
    dockerfile = (repository / "runtime/oaf_tf1/Dockerfile").read_text(encoding="utf-8")
    assert "runner-source-manifest" not in dockerfile
    assert not (repository / "runtime/oaf_tf1/runner-source-manifest.json").exists()
    assert (
        "COPY --from=instrumented-source /opt/crux/vendor/ /opt/crux/runtime/vendor/" in dockerfile
    )


def test_task_d_docker_keeps_unmodified_upstream_parity_source() -> None:
    repository = Path(__file__).parents[2]
    dockerfile = (repository / "runtime/oaf_tf1/Dockerfile").read_text(encoding="utf-8")
    assert "COPY runtime/oaf_tf1/vendor/magenta/ /opt/crux/upstream/magenta/" in dockerfile
    assert "COPY --from=instrumented-source /opt/crux/upstream/ /opt/crux/upstream/" in dockerfile


def test_task_d_docker_copies_model_config_to_worker_runtime_path() -> None:
    repository = Path(__file__).parents[2]
    dockerfile = (repository / "runtime/oaf_tf1/Dockerfile").read_text(encoding="utf-8")
    assert "COPY runtime/oaf_tf1/model.json /opt/crux/runtime/model.json" in dockerfile


def test_task_d_worker_loads_model_config_from_runtime_sibling(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from runtime.oaf_tf1 import worker

    class Config:
        backend_id = OAF_BACKEND_ID

    class Model:
        restored_tensor_count = 78

    captured: list[Path] = []

    def load_config(path: Path) -> Config:
        captured.append(path)
        return Config()

    monkeypatch.setattr(worker, "load_model_config", load_config)
    output = io.StringIO()
    assert (
        worker.serve_requests(
            io.StringIO(), output, model_factory=lambda *_args, **_kwargs: Model()
        )
        == 0
    )
    assert captured == [Path(worker.__file__).with_name("model.json")]


def test_task_d_removes_legacy_runner_commands() -> None:
    runner = CliRunner()
    for command in ("score-midi", "export-reference-midi", "transcribe-and-score"):
        result = runner.invoke(main, ["benchmark", command, "--help"])
        assert result.exit_code != 0
        assert f"No such command '{command}'" in result.output
