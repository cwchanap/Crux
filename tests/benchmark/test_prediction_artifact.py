from __future__ import annotations

from pathlib import Path

from src.benchmark.prediction_artifact import prediction_path

SHA_A = "a" * 64
SHA_B = "b" * 64
SHA_C = "c" * 64


def test_prediction_path_is_source_keyed_and_reference_independent(tmp_path: Path) -> None:
    path = prediction_path(
        tmp_path,
        simfile_id=10,
        source_audio_sha256=SHA_A,
        backend_descriptor_sha256=SHA_B,
        inference_config_sha256=SHA_C,
    )

    assert path == (tmp_path / "predictions" / "10" / SHA_A / SHA_B / f"{SHA_C}.jsonl")
    assert "input_audio_sha256" not in str(path)
    assert "reference_manifest" not in str(path)
