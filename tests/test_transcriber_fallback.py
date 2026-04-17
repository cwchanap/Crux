import numpy as np
import pytest
import requests
import soundfile as sf

from src.app.transcriber import DrumTranscriber


@pytest.mark.asyncio
async def test_transcribe_fallback_uses_onset_detection(monkeypatch, tmp_path):
    # Ensure model building is skipped and fallback path is used
    monkeypatch.setattr(DrumTranscriber, "_build_model", lambda self: None, raising=True)

    dt = DrumTranscriber(load_model=False, sample_rate=16000)

    # Create a tiny silent audio file that librosa can load
    sr = 16000
    audio = np.zeros(sr // 2, dtype=np.float32)  # 0.5s silence
    wav_path = tmp_path / "silence.wav"
    sf.write(wav_path, audio, sr)

    # Prepare job store
    job_id = "job-fallback"
    jobs_store = {
        job_id: {
            "status": "pending",
            "progress": 0,
            "created_at": "2024-01-01T00:00:00Z",
            "updated_at": "2024-01-01T00:00:00Z",
        }
    }

    midi_bytes = await dt.transcribe(str(wav_path), job_id, jobs_store)
    assert isinstance(midi_bytes, (bytes, bytearray))
    assert midi_bytes[:4] == b"MThd"
    # Progress should have advanced
    assert jobs_store[job_id]["progress"] >= 70


def test_init_falls_back_when_model_download_fails(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        DrumTranscriber,
        "_resolve_existing_path",
        classmethod(lambda cls, relative_path: None),
        raising=True,
    )

    class FailingResponse:
        def raise_for_status(self):
            raise requests.HTTPError("404 Client Error")

    def fake_get(*args, **kwargs):
        return FailingResponse()

    monkeypatch.setattr("src.app.transcriber.requests.get", fake_get)

    transcriber = DrumTranscriber(load_model=True)

    assert transcriber.model_path is None
    assert transcriber.model is None


def test_init_uses_shared_root_workspace_model_from_worktree(monkeypatch, tmp_path):
    repo_root = tmp_path / "Crux"
    worktree = repo_root / ".worktrees" / "feature"
    shared_model = repo_root / "models" / "e-gmd" / "tf2_model.weights.h5"
    worktree.mkdir(parents=True)
    shared_model.parent.mkdir(parents=True)
    shared_model.write_text("weights", encoding="utf-8")
    monkeypatch.chdir(worktree)

    def fail_download(self):
        raise AssertionError("should not download when shared root model exists")

    monkeypatch.setattr(DrumTranscriber, "_download_model", fail_download, raising=True)
    monkeypatch.setattr(DrumTranscriber, "_build_model", lambda self: None, raising=True)

    transcriber = DrumTranscriber(load_model=True)

    assert transcriber.model_path == str(shared_model)
    assert transcriber.model is None


def test_compute_spectrogram_shape():
    dt = DrumTranscriber(load_model=False, sample_rate=22050)
    # 0.1s of ones
    audio = np.ones(2205, dtype=np.float32)
    spec = dt._compute_spectrogram_for_model(audio, sr=22050)
    assert spec.ndim == 2
    # 229 mel bins expected per implementation
    assert spec.shape[1] == 229
    assert np.isfinite(spec).all()
