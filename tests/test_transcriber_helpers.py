import httpx
import numpy as np

from src.app.transcriber import DrumTranscriber


def test_onset_min_gap_allows_sixteenth_notes_at_120bpm():
    """MODEL_ONSET_MIN_GAP_FRAMES must be small enough to allow 16th notes at 120 BPM.

    At 120 BPM, 16th notes are 125 ms apart. With 16 kHz / 512-sample hop,
    each frame is ~32 ms so 2 frames = ~64 ms, well under 125 ms.
    """
    dt = DrumTranscriber(load_model=False)
    frame_duration_ms = dt.hop_length / dt.MODEL_SAMPLE_RATE * 1000
    gap_ms = dt.MODEL_ONSET_MIN_GAP_FRAMES * frame_duration_ms
    # Gap must be shorter than a 16th note at 120 BPM (125 ms)
    assert gap_ms < 125.0


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


def test_process_tf2_model_outputs_uses_calibrated_bin_mapping():
    dt = DrumTranscriber(load_model=False)
    onset_probs = np.zeros((1, 3, 88), dtype=np.float32)
    velocity_values = np.zeros((1, 3, 88), dtype=np.float32)

    calibrated_bins = {
        78: 36,  # kick
        63: 38,  # snare
        7: 42,  # closed hihat
        32: 46,  # open hihat
        35: 49,  # crash
        15: 51,  # ride
        72: 41,  # low tom
        57: 47,  # provisional mid tom
        66: 50,  # high tom
    }
    ignored_bin = 44  # was previously used by the overlapping mapping

    for model_bin in list(calibrated_bins) + [ignored_bin]:
        onset_probs[0, 1, model_bin] = 0.9
        velocity_values[0, 1, model_bin] = 0.5

    class FakeTensor:
        def __init__(self, value):
            self._value = value

        def numpy(self):
            return self._value

    outputs = {
        "onset_probs": FakeTensor(onset_probs),
        "velocity_values": FakeTensor(velocity_values),
    }

    drum_events = dt._process_tf2_model_outputs(outputs, sr=16000)

    for midi_note in dt.drum_mapping:
        expected_count = 1 if midi_note in calibrated_bins.values() else 0
        assert len(drum_events[midi_note]) == expected_count

    assert drum_events[36][0]["time"] == 512 / 16000
    assert drum_events[36][0]["velocity"] == 63
    assert len(drum_events[42]) == 1
    assert len(drum_events[46]) == 1


def test_download_model_uses_cached_tf1_checkpoint_with_full_filename(monkeypatch, tmp_path):
    checkpoint = tmp_path / "train" / "model.ckpt-10000"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.with_name(f"{checkpoint.name}.index").write_text("index", encoding="utf-8")

    monkeypatch.setattr(
        DrumTranscriber, "_shared_models_dir", classmethod(lambda cls: tmp_path), raising=True
    )
    monkeypatch.setattr(
        "src.app.transcriber.httpx.stream",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("unexpected download")),
    )

    transcriber = DrumTranscriber(load_model=False)

    assert transcriber._download_model() == str(checkpoint)


def test_find_checkpoint_discovers_extracted_ckpt569400(tmp_path):
    """_find_checkpoint_in_dir must find checkpoints regardless of step number
    or sub-directory layout."""
    # Simulate the actual e-gmd_checkpoint.zip layout: root-level ckpt
    index_file = tmp_path / "model.ckpt-569400.index"
    index_file.write_text("index", encoding="utf-8")

    result = DrumTranscriber._find_checkpoint_in_dir(tmp_path)
    assert result is not None
    assert result.name == "model.ckpt-569400"


def test_find_checkpoint_discovers_nested_checkpoint(tmp_path):
    """Some zip variants extract into a train/ subdirectory."""
    sub = tmp_path / "train"
    sub.mkdir()
    (sub / "model.ckpt-10000.index").write_text("index", encoding="utf-8")

    result = DrumTranscriber._find_checkpoint_in_dir(tmp_path)
    assert result is not None
    assert result.name == "model.ckpt-10000"


def test_find_checkpoint_returns_none_when_no_index(tmp_path):
    result = DrumTranscriber._find_checkpoint_in_dir(tmp_path)
    assert result is None


def test_download_model_tries_model_url_first(monkeypatch, tmp_path):
    """The known-good MODEL_URL must be the first URL attempted so that a clean
    checkout succeeds even when alternative mirrors are offline."""

    captured_urls: list[str] = []

    class FakeStreamContext:
        def __init__(self, url):
            captured_urls.append(url)

        def raise_for_status(self):
            raise httpx.HTTPStatusError(
                "404", request=httpx.Request("GET", "http://x"), response=httpx.Response(404)
            )

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def iter_bytes(self, **kwargs):
            return iter([])

    def fake_stream(method, url, **kwargs):
        return FakeStreamContext(url)

    model_dir = tmp_path / "models" / "e-gmd"
    monkeypatch.setattr(
        DrumTranscriber, "_shared_models_dir", classmethod(lambda cls: model_dir), raising=True
    )
    monkeypatch.setattr("src.app.transcriber.httpx.stream", fake_stream)

    transcriber = DrumTranscriber(load_model=False)
    transcriber._download_model()

    assert captured_urls[0] == DrumTranscriber.MODEL_URL
