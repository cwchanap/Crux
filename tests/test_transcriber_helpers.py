import numpy as np

from src.app.transcriber import DrumTranscriber


def test_find_onset_peaks_simple():
    dt = DrumTranscriber(load_model=False)
    signal = np.array([0.1, 0.2, 0.6, 0.4, 0.1, 0.7, 0.6, 0.5, 0.1])
    peaks = dt._find_onset_peaks(signal, threshold=0.3)
    assert isinstance(peaks, np.ndarray)
    assert peaks.tolist() == [2, 5]


def test_find_onset_peaks_respects_min_gap():
    dt = DrumTranscriber(load_model=False)
    signal = np.array([0.1, 0.82, 0.1, 0.8, 0.1, 0.91, 0.1], dtype=np.float32)
    peaks = dt._find_onset_peaks(signal, threshold=0.7, min_gap_frames=3)
    assert peaks.tolist() == [1, 5]


def test_find_onset_frames_simple():
    dt = DrumTranscriber(load_model=False)
    onset = np.array([0.1, 0.2, 0.7, 0.4, 0.1])
    frame = np.zeros_like(onset)
    frames = dt._find_onset_frames(onset, frame, threshold=0.5)
    assert frames == [2]


def test_create_midi_from_events():
    dt = DrumTranscriber(load_model=False)
    events = {
        36: [(0.0, 80), (0.5, 60)],  # Kick hits
        38: [(0.25, 70)],  # Snare hit
    }
    midi_bytes = dt._create_midi(events)
    assert isinstance(midi_bytes, (bytes, bytearray))
    assert midi_bytes[:4] == b"MThd"


def test_run_tf2_model_inference_uses_model_sample_rate(monkeypatch):
    dt = DrumTranscriber(load_model=False, sample_rate=44100)

    class FakeModel:
        def __call__(self, spec_input, training=False):  # noqa: ANN001, ARG002
            return {"fake": "outputs"}

    captured = {}

    def fake_compute(audio, sr):  # noqa: ANN001
        assert sr == 44100
        return np.zeros((10, 229), dtype=np.float32)

    def fake_process(outputs, sr):  # noqa: ANN001
        captured["sr"] = sr
        return {}

    dt.model = FakeModel()
    monkeypatch.setattr(dt, "_compute_spectrogram_for_model", fake_compute)
    monkeypatch.setattr(dt, "_process_tf2_model_outputs", fake_process)

    dt._run_tf2_model_inference(np.zeros(44100, dtype=np.float32), 44100)

    assert captured["sr"] == 16000
