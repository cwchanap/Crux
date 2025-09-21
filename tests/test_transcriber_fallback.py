import numpy as np
import pytest
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


def test_compute_spectrogram_shape():
    dt = DrumTranscriber(load_model=False, sample_rate=22050)
    # 0.1s of ones
    audio = np.ones(2205, dtype=np.float32)
    spec = dt._compute_spectrogram_for_model(audio, sr=22050)
    assert spec.ndim == 2
    # 229 mel bins expected per implementation
    assert spec.shape[1] == 229
    assert np.isfinite(spec).all()
