from __future__ import annotations

import hashlib
import io
import json
import os
import struct
import subprocess
import sys
from decimal import Decimal
from pathlib import Path

import pytest

from runtime.oaf_tf1 import entrypoint as runtime_entrypoint
from runtime.oaf_tf1 import oaf_backend
from runtime.oaf_tf1.entrypoint import (
    EXPECTED_ENVIRONMENT,
    discard_interpreter_bootstrap_environment,
    validate_process_environment,
)
from runtime.oaf_tf1.protocol import (
    ProtocolFailure,
    canonical_json_line,
    load_authenticated_object,
    read_verified_canonical_wav,
    sanitize_diagnostic,
    serve_requests,
    validate_transcribe_request,
)


def _canonical_bytes(value: object) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True).encode("utf-8")
        + b"\n"
    )


def _canonical_wav(sample_frames: int = 4) -> bytes:
    samples = b"\x00\x00" * sample_frames
    return (
        b"RIFF"
        + struct.pack("<I", 36 + len(samples))
        + b"WAVEfmt "
        + struct.pack("<IHHIIHH", 16, 1, 1, 44100, 88200, 2, 16)
        + b"data"
        + struct.pack("<I", len(samples))
        + samples
    )


def _checkpoint_fixture(
    root: Path,
    *,
    data_payload: bytes = b"data-component",
) -> tuple[list[dict[str, object]], dict[str, bytes]]:
    payloads = {
        "model.ckpt-569400.data-00000-of-00001": data_payload,
        "model.ckpt-569400.index": b"index-component",
        "model.ckpt-569400.meta": b"meta-component",
    }
    root.mkdir()
    components = []
    for name, payload in payloads.items():
        (root / name).write_bytes(payload)
        components.append(
            {
                "name": name,
                "sha256": hashlib.sha256(payload).hexdigest(),
                "size": len(payload),
            }
        )
    return components, payloads


def test_entrypoint_rejects_python_hash_seed_before_runner_import(monkeypatch) -> None:
    monkeypatch.setenv("PYTHONHASHSEED", "unexpected")

    with pytest.raises(SystemExit) as error:
        validate_process_environment(EXPECTED_ENVIRONMENT)

    assert error.value.code == 2


@pytest.mark.parametrize(
    ("mutation", "expected_code"),
    [
        (lambda environment: environment.pop("OPENBLAS_NUM_THREADS"), 2),
        (lambda environment: environment.__setitem__("EXTRA", "1"), 2),
        (lambda environment: environment.__setitem__("OMP_NUM_THREADS", "2"), 2),
    ],
)
def test_entrypoint_rejects_any_process_environment_drift(
    monkeypatch, mutation, expected_code
) -> None:
    environment = dict(EXPECTED_ENVIRONMENT)
    mutation(environment)
    monkeypatch.setattr(os, "environ", environment)

    with pytest.raises(SystemExit) as error:
        validate_process_environment(EXPECTED_ENVIRONMENT)

    assert error.value.code == expected_code


def test_entrypoint_discards_only_the_required_python_locale_bootstrap(
    monkeypatch,
) -> None:
    environment = {
        **EXPECTED_ENVIRONMENT,
        "PYTHONCOERCECLOCALE": "0",
    }
    monkeypatch.setattr(os, "environ", environment)

    discard_interpreter_bootstrap_environment()

    assert os.environ == EXPECTED_ENVIRONMENT


def test_runtime_entrypoint_clears_image_environment_before_python() -> None:
    dockerfile = Path(__file__).resolve().parents[1] / "Dockerfile"
    if not dockerfile.is_file():
        pytest.skip("Dockerfile entrypoint shape is a host-side source check")
    entrypoint = next(
        json.loads(line[len("ENTRYPOINT ") :])
        for line in dockerfile.read_text(encoding="utf-8").splitlines()
        if line.startswith("ENTRYPOINT ")
    )

    assert entrypoint == [
        "/usr/bin/env",
        "-i",
        "CUDA_VISIBLE_DEVICES=-1",
        "MKL_NUM_THREADS=1",
        "OMP_NUM_THREADS=1",
        "OPENBLAS_NUM_THREADS=1",
        "PYTHONHASHSEED=0",
        "PYTHONCOERCECLOCALE=0",
        "TF_NUM_INTEROP_THREADS=1",
        "TF_NUM_INTRAOP_THREADS=1",
        "/opt/crux/venv/bin/python",
        "-s",
        "/opt/crux/runtime/entrypoint.py",
    ]


def test_entrypoint_dependency_import_failure_is_one_stable_stderr_line(
    tmp_path: Path,
) -> None:
    if not Path("/opt/crux/vendor").is_dir():
        pytest.skip("dependency import boundary runs in the runtime image")
    entrypoint_source = Path(runtime_entrypoint.__file__).resolve()
    isolated_entrypoint = tmp_path / "entrypoint.py"
    isolated_entrypoint.write_bytes(entrypoint_source.read_bytes())
    (tmp_path / "numpy.py").write_text(
        'raise RuntimeError("arbitrary dependency detail /private/secret")\n',
        encoding="utf-8",
    )

    completed = subprocess.run(
        [sys.executable, "-s", str(isolated_entrypoint)],
        check=False,
        capture_output=True,
        env={
            **EXPECTED_ENVIRONMENT,
            "PYTHONCOERCECLOCALE": "0",
        },
    )

    assert completed.returncode == 2
    assert completed.stdout == b""
    assert completed.stderr == b"code=runner_dependency_import_failed count=1\n"


def test_canonical_json_line_is_compact_sorted_utf8_and_bounded() -> None:
    assert canonical_json_line({"z": "é", "a": 1}, maximum_bytes=32) == (
        b'{"a":1,"z":"\xc3\xa9"}\n'
    )

    with pytest.raises(ProtocolFailure) as error:
        canonical_json_line({"value": "x" * 32}, maximum_bytes=16)

    assert error.value.code == "protocol_output_oversized"
    assert error.value.fatal is True


def test_smoke_match_uses_canonical_numbers_for_nonzero_frame_events() -> None:
    oracle_events = [
        {
            "confidence_raw": Decimal("0.625"),
            "frame_index": 1,
            "time_sec_raw": Decimal("0.011609977324263039"),
        }
    ]
    observed_events = [
        {
            "confidence_raw": 0.625,
            "frame_index": 1,
            "time_sec_raw": 0.011609977324263039,
        }
    ]

    assert oaf_backend.smoke_events_match(observed_events, oracle_events) is True
    oracle_events[0]["time_sec_raw"] = Decimal("0.011609977324263038")
    assert oaf_backend.smoke_events_match(observed_events, oracle_events) is False


def test_smoke_match_normalizes_exponent_numbers_without_weakening_equality() -> None:
    observed_events = [
        {
            "confidence_raw": 1e-7,
            "frame_index": 1,
            "time_sec_raw": -1e-7,
            "values": [1e20, -1e20],
        }
    ]
    oracle_events = [
        {
            "confidence_raw": Decimal("0.0000001"),
            "frame_index": 1,
            "time_sec_raw": Decimal("-1E-7"),
            "values": [Decimal("1E+20"), Decimal("-1E+20")],
        }
    ]

    assert oaf_backend.smoke_events_match(observed_events, oracle_events) is True
    oracle_events[0]["confidence_raw"] = Decimal("0.00000010000000000000001")
    assert oaf_backend.smoke_events_match(observed_events, oracle_events) is False


def test_checkpoint_consumers_receive_private_bytes_immune_to_later_mount_changes(
    tmp_path: Path,
) -> None:
    mounted_root = tmp_path / "mounted"
    components, payloads = _checkpoint_fixture(mounted_root)
    private_root = tmp_path / "private"

    checkpoint_prefix = Path(
        oaf_backend.materialize_authenticated_checkpoint(
            mounted_root,
            components,
            private_root=private_root,
        )
    )

    (mounted_root / components[0]["name"]).write_bytes(b"mutated-after-authentication")
    replacement = tmp_path / "replacement-index"
    replacement.write_bytes(b"swapped-after-authentication")
    os.replace(replacement, mounted_root / components[1]["name"])

    assert checkpoint_prefix == private_root / "model.ckpt-569400"
    for name, payload in payloads.items():
        assert (private_root / name).read_bytes() == payload


def test_checkpoint_copy_keeps_descriptor_identity_across_mid_read_path_swap(
    tmp_path: Path,
    monkeypatch,
) -> None:
    mounted_root = tmp_path / "mounted"
    original_data = b"a" * 70000
    components, _ = _checkpoint_fixture(mounted_root, data_payload=original_data)
    data_path = mounted_root / components[0]["name"]
    replacement = tmp_path / "replacement-data"
    replacement.write_bytes(b"b" * len(original_data))
    real_read = oaf_backend.os.read
    swapped = False

    def read_then_swap(descriptor: int, size: int) -> bytes:
        nonlocal swapped
        chunk = real_read(descriptor, size)
        if not swapped and chunk and os.fstat(descriptor).st_size == len(original_data):
            os.replace(replacement, data_path)
            swapped = True
        return chunk

    monkeypatch.setattr(oaf_backend.os, "read", read_then_swap)

    checkpoint_prefix = Path(
        oaf_backend.materialize_authenticated_checkpoint(
            mounted_root,
            components,
            private_root=tmp_path / "private",
        )
    )

    assert swapped is True
    assert data_path.read_bytes() == b"b" * len(original_data)
    private_data_path = Path(str(checkpoint_prefix) + ".data-00000-of-00001")
    assert private_data_path.read_bytes() == original_data


def test_mounted_object_rejects_duplicate_unknown_and_noncanonical_json(tmp_path: Path) -> None:
    target = tmp_path / "lock.json"
    cases = (
        b'{"schema":"fixture/v1","schema":"fixture/v1","value":1}\n',
        b'{"extra":1,"schema":"fixture/v1","value":1}\n',
        b'{ "schema":"fixture/v1","value":1}\n',
    )

    for content in cases:
        target.write_bytes(content)
        with pytest.raises(ProtocolFailure) as error:
            load_authenticated_object(
                target,
                label="fixture",
                exact_keys=frozenset({"schema", "value"}),
                expected_schema="fixture/v1",
            )
        assert error.value.code == "mounted_identity_invalid"


def test_mounted_object_returns_hash_of_exact_canonical_bytes(tmp_path: Path) -> None:
    target = tmp_path / "lock.json"
    payload = {"schema": "fixture/v1", "value": 1}
    content = _canonical_bytes(payload)
    target.write_bytes(content)

    authenticated = load_authenticated_object(
        target,
        label="fixture",
        exact_keys=frozenset(payload),
        expected_schema="fixture/v1",
    )

    assert authenticated.payload == payload
    assert authenticated.sha256 == hashlib.sha256(content).hexdigest()


def test_mounted_object_rejects_symlinks(tmp_path: Path) -> None:
    target = tmp_path / "lock.json"
    target.write_bytes(_canonical_bytes({"schema": "fixture/v1", "value": 1}))
    link = tmp_path / "link.json"
    link.symlink_to(target)

    with pytest.raises(ProtocolFailure) as error:
        load_authenticated_object(
            link,
            label="fixture",
            exact_keys=frozenset({"schema", "value"}),
            expected_schema="fixture/v1",
        )

    assert error.value.code == "mounted_identity_invalid"


@pytest.mark.parametrize(
    "payload",
    [
        {
            "audio_path": "audio.wav",
            "audio_sha256": "a" * 64,
            "backend_descriptor_sha256": "b" * 64,
            "request_id": "request-1",
        },
        {
            "type": "transcribe",
            "audio_path": "audio.wav",
            "audio_sha256": "a" * 64,
            "backend_descriptor_sha256": "b" * 64,
            "request_id": "request-1",
            "unknown": True,
        },
    ],
)
def test_transcribe_request_rejects_missing_or_unknown_keys(payload: dict[str, object]) -> None:
    with pytest.raises(ProtocolFailure) as error:
        validate_transcribe_request(payload, expected_descriptor_sha256="b" * 64)

    assert error.value.code == "request_invalid"
    assert error.value.fatal is False


@pytest.mark.parametrize(
    "path",
    [
        "",
        "/absolute.wav",
        "../escape.wav",
        "safe/../alias.wav",
        "./audio.wav",
        "safe//audio.wav",
        "safe\\audio.wav",
        "safe/audio.WAV",
        "safe/audio.wav\x00",
        "safe/e\u0301.wav",
    ],
)
def test_transcribe_request_rejects_noncanonical_runner_paths(path: str) -> None:
    payload = {
        "type": "transcribe",
        "audio_path": path,
        "audio_sha256": "a" * 64,
        "backend_descriptor_sha256": "b" * 64,
        "request_id": "request-1",
    }

    with pytest.raises(ProtocolFailure) as error:
        validate_transcribe_request(payload, expected_descriptor_sha256="b" * 64)

    assert error.value.code == "input_path_invalid"
    assert error.value.fatal is False


def test_transcribe_request_echoes_exact_valid_opaque_id() -> None:
    request = validate_transcribe_request(
        {
            "type": "transcribe",
            "audio_path": "safe/é.wav",
            "audio_sha256": "a" * 64,
            "backend_descriptor_sha256": "b" * 64,
            "request_id": "opaque._:-9",
        },
        expected_descriptor_sha256="b" * 64,
    )

    assert request.request_id == "opaque._:-9"
    assert request.audio_path == "safe/é.wav"


def test_transcribe_request_rejects_descriptor_mismatch() -> None:
    with pytest.raises(ProtocolFailure) as error:
        validate_transcribe_request(
            {
                "type": "transcribe",
                "audio_path": "audio.wav",
                "audio_sha256": "a" * 64,
                "backend_descriptor_sha256": "c" * 64,
                "request_id": "request-1",
            },
            expected_descriptor_sha256="b" * 64,
        )

    assert error.value.code == "request_descriptor_mismatch"
    assert error.value.fatal is False


def test_verified_wav_rejects_symlink_hash_mismatch_and_frame_overflow(tmp_path: Path) -> None:
    content = _canonical_wav(sample_frames=4)
    audio = tmp_path / "audio.wav"
    audio.write_bytes(content)
    link = tmp_path / "link.wav"
    link.symlink_to(audio)
    expected_hash = hashlib.sha256(content).hexdigest()

    for path, digest, maximum, code in (
        ("link.wav", expected_hash, 4, "input_path_invalid"),
        ("audio.wav", "0" * 64, 4, "input_hash_mismatch"),
        ("audio.wav", expected_hash, 3, "input_too_long"),
    ):
        with pytest.raises(ProtocolFailure) as error:
            read_verified_canonical_wav(tmp_path, path, digest, maximum)
        assert error.value.code == code
        assert error.value.fatal is False


def test_verified_wav_accepts_only_exact_canonical_pcm_contract(tmp_path: Path) -> None:
    content = _canonical_wav(sample_frames=4)
    audio = tmp_path / "audio.wav"
    audio.write_bytes(content)

    verified = read_verified_canonical_wav(
        tmp_path,
        "audio.wav",
        hashlib.sha256(content).hexdigest(),
        4,
    )

    assert verified.content == content
    assert verified.audio_frame_count == 4

    audio.write_bytes(content + b"JUNK")
    with pytest.raises(ProtocolFailure) as error:
        read_verified_canonical_wav(
            tmp_path,
            "audio.wav",
            hashlib.sha256(content + b"JUNK").hexdigest(),
            4,
        )
    assert error.value.code == "input_wav_invalid"


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (
            b"code=restore_ok tensor=onsets/count count=78 duration_ms=42",
            b"code=restore_ok tensor=onsets/count count=78 duration_ms=42",
        ),
        (b"Traceback (most recent call last):", b"[REDACTED]"),
        (b"path=/opt/crux/model/model.ckpt", b"[REDACTED]"),
        (b"https://user:secret@example.test/model?token=x", b"[REDACTED]"),
        (b"samples=[0.1,0.2]", b"[REDACTED]"),
        (b"arbitrary exception text", b"[REDACTED]"),
    ],
)
def test_stderr_diagnostics_are_stable_and_sanitized(raw: bytes, expected: bytes) -> None:
    assert sanitize_diagnostic(raw) == expected


def test_request_loop_emits_protocol_only_and_processes_one_request_at_a_time(
    tmp_path: Path,
) -> None:
    content = _canonical_wav()
    (tmp_path / "one.wav").write_bytes(content)
    (tmp_path / "two.wav").write_bytes(content)
    digest = hashlib.sha256(content).hexdigest()
    requests = b"".join(
        _canonical_bytes(
            {
                "type": "transcribe",
                "audio_path": path,
                "audio_sha256": digest,
                "backend_descriptor_sha256": "b" * 64,
                "request_id": request_id,
            }
        )
        for request_id, path in (("one", "one.wav"), ("two", "two.wav"))
    )
    stdin = io.BytesIO(requests)
    stdout = io.BytesIO()

    class Backend:
        active = False
        seen: list[str] = []

        def transcribe(self, verified):
            assert self.active is False
            self.active = True
            self.seen.append(verified.relative_path)
            self.active = False
            return [{"frame_index": len(self.seen)}]

    backend = Backend()
    serve_requests(
        stdin=stdin,
        stdout=stdout,
        backend=backend,
        input_root=tmp_path,
        descriptor_sha256="b" * 64,
        max_input_audio_frames=4,
        stdout_max_line_bytes=4096,
    )

    lines = stdout.getvalue().splitlines()
    assert len(lines) == 2
    responses = [json.loads(line) for line in lines]
    assert [response["request_id"] for response in responses] == ["one", "two"]
    assert all(set(response) == {"payload", "request_id", "type"} for response in responses)
    assert all(response["type"] == "response" for response in responses)
    assert all(
        set(response["payload"])
        == {
            "audio_sha256",
            "backend_descriptor_sha256",
            "native_events",
            "type",
        }
        for response in responses
    )
    assert all(response["payload"]["type"] == "transcription_result" for response in responses)
    assert all("request_id" not in response["payload"] for response in responses)
    assert backend.seen == ["one.wav", "two.wav"]


def test_request_loop_wraps_item_error_without_duplicate_request_id(tmp_path: Path) -> None:
    request = _canonical_bytes(
        {
            "type": "transcribe",
            "audio_path": "missing.wav",
            "audio_sha256": "a" * 64,
            "backend_descriptor_sha256": "b" * 64,
            "request_id": "request-1",
        }
    )
    stdout = io.BytesIO()

    serve_requests(
        stdin=io.BytesIO(request),
        stdout=stdout,
        backend=object(),
        input_root=tmp_path,
        descriptor_sha256="b" * 64,
        max_input_audio_frames=4,
        stdout_max_line_bytes=4096,
    )

    response = json.loads(stdout.getvalue())
    assert response == {
        "payload": {
            "code": "input_path_invalid",
            "message": "The canonical input path is unavailable or unsafe.",
            "type": "transcription_error",
        },
        "request_id": "request-1",
        "type": "response",
    }


def test_request_loop_rejects_noncanonical_json_before_item_validation(tmp_path: Path) -> None:
    with pytest.raises(ProtocolFailure) as error:
        serve_requests(
            stdin=io.BytesIO(b"{ }\n"),
            stdout=io.BytesIO(),
            backend=object(),
            input_root=tmp_path,
            descriptor_sha256="b" * 64,
            max_input_audio_frames=4,
            stdout_max_line_bytes=4096,
        )

    assert error.value.code == "protocol_input_invalid"
    assert error.value.fatal is True
