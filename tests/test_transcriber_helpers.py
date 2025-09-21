import numpy as np

from src.app.transcriber import DrumTranscriber


def test_find_onset_peaks_simple():
    dt = DrumTranscriber(load_model=False)
    signal = np.array([0.1, 0.2, 0.6, 0.4, 0.1, 0.7, 0.6, 0.5, 0.1])
    peaks = dt._find_onset_peaks(signal, threshold=0.3)
    # Validate return type and that indices (if any) are non-negative
    assert isinstance(peaks, np.ndarray)
    assert (peaks >= 0).all()


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
