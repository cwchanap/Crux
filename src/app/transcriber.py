"""
Drum transcription using trained Magenta E-GMD model
Directly loads the checkpoint without requiring full Magenta installation
"""

import logging
import os
import zipfile
from pathlib import Path
from typing import Any, Dict, Optional

import httpx
import librosa
import numpy as np
import pretty_midi

# Heavy dependencies (TensorFlow, TF2 model utilities) are intentionally NOT imported at
# module import time to keep tests and lightweight environments fast.

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DrumTranscriber:
    """
    Drum transcription using Magenta's approach
    Compatible with TensorFlow 2.x
    """

    # General MIDI drum mapping (drums on channel 9)
    DRUM_MAP = {
        36: "Kick",  # Bass Drum 1
        38: "Snare",  # Acoustic Snare
        42: "Hi-Hat Closed",  # Closed Hi-Hat
        46: "Hi-Hat Open",  # Open Hi-Hat
        49: "Crash",  # Crash Cymbal 1
        51: "Ride",  # Ride Cymbal 1
        45: "Tom Low",  # Low Tom
        47: "Tom Mid",  # Mid Tom
        50: "Tom High",  # High Tom
    }

    # Magenta E-GMD drum model checkpoint URL
    MODEL_URL = "https://storage.googleapis.com/magentadata/models/onsets_frames_transcription/e-gmd_checkpoint.zip"
    TF2_WEIGHTS_RELATIVE_PATH = Path("models/e-gmd/tf2_model.weights.h5")
    # The checkpoint filename inside the zip varies by source URL
    # (e.g. model.ckpt-569400, model.ckpt-10000).  _download_model()
    # discovers the actual name dynamically after extraction.
    MODEL_SAMPLE_RATE = 16000
    MODEL_ONSET_THRESHOLD = 0.7
    MODEL_ONSET_MIN_GAP_FRAMES = 2  # ~64 ms at 16 kHz / 512-sample hop
    # Empirically calibrated against the prepared benchmark corpus.
    # Mid tom remains provisional until we calibrate against charts that contain it.
    TF2_OUTPUT_BIN_TO_DRUM_MIDI = {
        7: 42,
        15: 51,
        32: 46,
        35: 49,
        57: 47,
        63: 38,
        66: 50,
        72: 41,
        78: 36,
    }

    def __init__(
        self,
        model_path: Optional[str] = None,
        sample_rate: int = 44100,
        load_model: bool = True,
    ):
        """
        Initialize drum transcriber

        Args:
            model_path: Path to pre-trained model checkpoint
            sample_rate: Sample rate for audio processing
        """
        self.sample_rate = sample_rate
        self.model_path = model_path
        self.model = None
        self.hop_length = 512  # Default hop length for spectrograms
        # Feature parameters (used across model building and feature extraction)
        self.n_mels = 229
        self.n_fft = 2048
        self.fmin = 30
        self.fmax = self.sample_rate // 2

        # Define drum mapping for E-GMD model (MIDI notes to drum names)
        self.drum_mapping = {
            36: "Kick",
            38: "Snare",
            42: "Hi-Hat Closed",
            46: "Hi-Hat Open",
            49: "Crash",
            51: "Ride",
            41: "Tom Low",
            47: "Tom Mid",
            50: "Tom High",
        }

        if load_model:
            if model_path is None:
                existing_weights_path = self._resolve_existing_path(self.TF2_WEIGHTS_RELATIVE_PATH)
                if existing_weights_path is not None:
                    self.model_path = str(existing_weights_path)
                    logger.info("Found converted TF2 weights at %s", existing_weights_path)
                else:
                    # Download E-GMD model if not provided
                    self.model_path = self._download_model()

            # Build model
            self.model = self._build_model()
        else:
            # Skip model initialization entirely (used for tests)
            self.model = None

    @classmethod
    def _resolve_existing_path(cls, relative_path: Path) -> Path | None:
        """Resolve project assets from the current repo or the shared workspace root."""
        repo_root = Path(__file__).resolve().parents[2]
        search_anchors = [Path.cwd().resolve(), repo_root]
        checked: set[Path] = set()

        for anchor in search_anchors:
            for candidate_root in [anchor, *anchor.parents]:
                if candidate_root in checked:
                    continue
                checked.add(candidate_root)

                candidate_path = candidate_root / relative_path
                if candidate_path.exists():
                    return candidate_path

                if candidate_root.name == ".worktrees":
                    shared_root_path = candidate_root.parent / relative_path
                    if shared_root_path.exists():
                        return shared_root_path

        return None

    @classmethod
    def _shared_models_dir(cls) -> Path:
        """Store models in the shared workspace root when running from a worktree."""
        repo_root = Path(__file__).resolve().parents[2]
        current_dir = Path.cwd().resolve()

        for anchor in (current_dir, repo_root):
            for candidate_root in [anchor, *anchor.parents]:
                if candidate_root.name == ".worktrees":
                    return candidate_root.parent / "models" / "e-gmd"

        return repo_root / "models" / "e-gmd"

    @staticmethod
    def _find_checkpoint_in_dir(model_dir: Path) -> Path | None:
        """Find the first TF1 checkpoint (.index file) under *model_dir*.

        The downloaded zip may extract ``model.ckpt-569400`` at the root or
        ``train/model.ckpt-10000`` in a sub-directory — the exact name depends
        on the source URL.  Scanning for ``.index`` avoids hard-coding a name
        that may not match the archive contents.
        """
        for index_file in sorted(model_dir.rglob("*.index")):
            name = index_file.name
            if name.startswith("model.ckpt") and name.endswith(".index"):
                # Strip the ".index" suffix to give the checkpoint base path.
                return index_file.with_suffix("")
        return None

    def _download_model(self) -> str | None:
        """Download the E-GMD model checkpoint if not already present"""
        model_dir = self._shared_models_dir()

        existing = self._find_checkpoint_in_dir(model_dir)
        if existing is not None:
            logger.info("E-GMD model already downloaded at %s", existing)
            return str(existing)

        # Try the known-good checkpoint URL first, then fallback alternatives.
        model_urls = [
            self.MODEL_URL,
            "https://storage.googleapis.com/magentadata/models/onsets_frames_transcription/e-gmd/model.ckpt-10000.zip",
            "https://storage.googleapis.com/magentadata/models/onsets_frames_transcription/e_gmd_checkpoint.zip",
        ]

        logger.info("Downloading E-GMD model checkpoint...")

        for model_url in model_urls:
            try:
                model_dir.mkdir(parents=True, exist_ok=True)
                zip_path = model_dir / "checkpoint.zip"

                with httpx.stream("GET", model_url, timeout=10, follow_redirects=True) as response:
                    response.raise_for_status()
                    with open(zip_path, "wb") as f:
                        for chunk in response.iter_bytes(chunk_size=8192):
                            f.write(chunk)

                with zipfile.ZipFile(zip_path, "r") as zip_ref:
                    zip_ref.extractall(model_dir)

                # Clean up zip file
                zip_path.unlink()

                discovered = self._find_checkpoint_in_dir(model_dir)
                if discovered is None:
                    logger.warning("Zip from %s did not contain a recognised checkpoint", model_url)
                    continue

                logger.info("E-GMD model downloaded successfully at %s", discovered)
                return str(discovered)

            except (httpx.HTTPError, zipfile.BadZipFile, OSError) as e:
                logger.warning("Failed with URL %s: %s", model_url, e)
                continue

        logger.error("Failed to download model from any source")
        return None

    def _build_model(self):
        """
        Load the trained Magenta drum transcription model directly
        """
        if not self.model_path:
            logger.warning("No model path available, using fallback method")
            return None

        try:
            from src.app.tf2_magenta_model import (
                create_drum_model,
                load_tf1_checkpoint_to_tf2,
            )

            # Create the TF2 model
            model = create_drum_model()

            # Load weights from the resolved model_path (set by __init__).
            # __init__ already handles the shared-weights search, so
            # _build_model must honour whatever path was chosen.
            model_available = self.model_path and (
                os.path.exists(self.model_path) or os.path.exists(self.model_path + ".index")
            )
            if model_available:
                if self.model_path.endswith(".weights.h5"):
                    # Load TF2 weights directly
                    model.load_weights(self.model_path)
                    logging.info("Loaded TF2 weights from %s", self.model_path)
                else:
                    # Try to load and convert TF1 checkpoint
                    try:
                        model = load_tf1_checkpoint_to_tf2(self.model_path, model)
                    except RuntimeError as e:
                        logging.error("Failed to convert TF1 checkpoint %s: %s", self.model_path, e)
                        return None
                    logging.info("Loaded and converted TF1 checkpoint from %s", self.model_path)
            else:
                logging.warning("No model path available, using fallback method")
                return None

            return model

        # ImportError/ModuleNotFoundError: TF not installed — fall back to onset detection.
        # OSError/ValueError: corrupt weights or bad checkpoint — fall back gracefully.
        # Unexpected TF runtime errors (RuntimeError etc.) propagate intentionally so that
        # broken model state is not silently hidden behind heuristic output.
        except (ImportError, ModuleNotFoundError, OSError, ValueError) as e:
            logging.error("Failed to build TF2 model: %s", e)
            return None

    def _load_model(self, checkpoint_path: str) -> bool:
        """Load the Magenta model from checkpoint"""
        try:
            # Create and load TF2-compatible model
            from src.app.tf2_magenta_model import create_drum_model  # Lazy import

            self.model = create_drum_model(checkpoint_path)
            logger.info("Successfully loaded TF2-compatible model")
            return True
        except Exception as e:  # pylint: disable=broad-except
            logger.error("Failed to load model: %s", e)
            logger.warning("Using fallback onset detection method")
            return False

    def _build_egmd_architecture(self):
        """
        Build the E-GMD model architecture
        Based on the Onsets and Frames architecture for drums
        """
        import tensorflow as tf  # Lazy import to avoid heavy dependency at module import time

        # Input: Mel spectrogram
        inputs = tf.keras.Input(shape=(None, self.n_mels), name="mel_input")

        # Onset stack - predicts when drum hits occur
        onset_x = tf.keras.layers.Conv1D(32, 3, padding="same", activation="relu")(inputs)
        onset_x = tf.keras.layers.BatchNormalization()(onset_x)
        onset_x = tf.keras.layers.Conv1D(32, 3, padding="same", activation="relu")(onset_x)
        onset_x = tf.keras.layers.BatchNormalization()(onset_x)
        onset_x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, return_sequences=True))(
            onset_x
        )
        onset_outputs = tf.keras.layers.Dense(9, activation="sigmoid", name="onset_probs")(onset_x)

        # Frame stack - predicts active drums at each frame
        frame_x = tf.keras.layers.Conv1D(32, 3, padding="same", activation="relu")(inputs)
        frame_x = tf.keras.layers.BatchNormalization()(frame_x)
        frame_x = tf.keras.layers.Concatenate()([frame_x, onset_outputs])
        frame_x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, return_sequences=True))(
            frame_x
        )
        frame_outputs = tf.keras.layers.Dense(9, activation="sigmoid", name="frame_probs")(frame_x)

        # Velocity stack - predicts hit strength
        velocity_x = tf.keras.layers.Concatenate()([frame_outputs, onset_outputs])
        velocity_outputs = tf.keras.layers.Dense(9, activation="linear", name="velocity")(
            velocity_x
        )

        model = tf.keras.Model(
            inputs=inputs,
            outputs={
                "onset_probs": onset_outputs,
                "frame_probs": frame_outputs,
                "velocity": velocity_outputs,
            },
        )

        return model

    def _predictions_to_events(self, predictions: Dict[str, np.ndarray]) -> Dict[int, list]:
        """
        Convert model predictions to drum events with timing and velocity
        """
        drum_events = {pitch: [] for pitch in self.DRUM_MAP.keys()}

        # E-GMD model outputs for 9 drum classes
        # Map to our DRUM_MAP pitches
        egmd_to_midi = {
            0: 36,  # Kick
            1: 38,  # Snare
            2: 42,  # Closed Hi-Hat
            3: 46,  # Open Hi-Hat
            4: 49,  # Crash
            5: 51,  # Ride
            6: 45,  # Low Tom
            7: 47,  # Mid Tom
            8: 50,  # High Tom
        }

        onset_probs = predictions.get("onset_probs")
        frame_probs = predictions.get("frame_probs")
        velocities = predictions.get("velocity")

        if onset_probs is None or frame_probs is None:
            return drum_events

        # Process each drum class
        for drum_idx in range(onset_probs.shape[1]):
            if drum_idx not in egmd_to_midi:
                continue

            midi_pitch = egmd_to_midi[drum_idx]
            if midi_pitch not in self.DRUM_MAP:
                continue

            # Detect onsets (peaks in onset probability)
            onset_threshold = 0.5
            onset_frames = self._find_onset_frames(
                onset_probs[:, drum_idx], frame_probs[:, drum_idx], onset_threshold
            )

            # Convert to time and add velocity
            for frame in onset_frames:
                time = frame * self.hop_length / self.sample_rate
                velocity = 64  # Default if velocity not available
                if velocities is not None:
                    velocity = int(np.clip(velocities[frame, drum_idx] * 127, 1, 127))

                drum_events[midi_pitch].append((time, velocity))

        return drum_events

    def _find_onset_frames(self, onset_probs, frame_probs, threshold):
        """Find onset frames from probability curves"""
        # Simple peak picking on onset probabilities
        onset_frames = []

        for i in range(1, len(onset_probs) - 1):
            if (
                onset_probs[i] > threshold
                and onset_probs[i] > onset_probs[i - 1]
                and onset_probs[i] > onset_probs[i + 1]
            ):
                onset_frames.append(i)

        return onset_frames

    def _detect_onsets_from_audio(self, audio: np.ndarray) -> Dict[int, list]:
        """
        Fallback: Detect drum onsets using signal processing when model unavailable
        """
        drum_onsets = {pitch: [] for pitch in self.DRUM_MAP.keys()}

        # Use onset detection for rhythm
        onset_envelope = librosa.onset.onset_strength(
            y=audio, sr=self.sample_rate, hop_length=self.hop_length
        )
        onsets = librosa.onset.onset_detect(
            onset_envelope=onset_envelope,
            sr=self.sample_rate,
            hop_length=self.hop_length,
            backtrack=True,
        )

        # Convert onset frames to time
        onset_times = librosa.frames_to_time(
            onsets, sr=self.sample_rate, hop_length=self.hop_length
        )

        # Analyze frequency content at each onset to guess drum type
        for onset_time, onset_frame in zip(onset_times, onsets):
            start_sample = int(onset_time * self.sample_rate)
            end_sample = min(start_sample + self.hop_length * 2, len(audio))
            onset_audio = audio[start_sample:end_sample]

            if len(onset_audio) == 0:
                continue

            # Get spectral centroid to estimate drum type
            spectral_centroid = librosa.feature.spectral_centroid(
                y=onset_audio, sr=self.sample_rate
            )[0].mean()

            # Enhanced frequency-based drum classification inspired by E-GMD
            # Also analyze zero crossing rate for better classification
            zcr = librosa.feature.zero_crossing_rate(onset_audio)[0].mean()

            # Improved drum classification based on spectral centroid and ZCR
            if spectral_centroid < 150:  # Very low frequency = kick
                drum_onsets[36].append((onset_time, 80))  # Higher velocity for kick
            elif spectral_centroid < 350 and zcr > 0.1:  # Mid-low with noise = snare
                drum_onsets[38].append((onset_time, 70))
            elif spectral_centroid > 3000 and zcr > 0.2:  # High freq with noise = hi-hat
                drum_onsets[42].append((onset_time, 60))
            elif spectral_centroid > 2000:  # High freq = cymbals
                drum_onsets[49].append((onset_time, 65))
            elif spectral_centroid < 1000:  # Mid frequencies = toms
                if spectral_centroid < 500:
                    drum_onsets[45].append((onset_time, 65))  # Low tom
                elif spectral_centroid < 750:
                    drum_onsets[47].append((onset_time, 65))  # Mid tom
                else:
                    drum_onsets[50].append((onset_time, 65))  # High tom

        return drum_onsets

    async def transcribe(self, audio_path: str, job_id: str, jobs_store: Dict[str, Any]) -> bytes:
        """
        Transcribe drums from audio file to MIDI
        """
        try:
            # Update progress
            jobs_store[job_id]["progress"] = 40

            # Load and preprocess audio
            logger.info("Loading audio file: %s", audio_path)
            audio, _ = librosa.load(audio_path, sr=self.sample_rate, mono=True)

            # Update progress
            jobs_store[job_id]["progress"] = 40

            # Initialize model if not already done
            if self.model is None:
                self.model = self._build_model()

            # Update progress
            jobs_store[job_id]["progress"] = 50

            if self.model is not None:
                # Use the trained E-GMD model
                logger.info("Using TF2 E-GMD model for transcription")

                # Update progress
                jobs_store[job_id]["progress"] = 60

                # Run inference with the TF2 model
                drum_events = self._run_tf2_model_inference(audio, self.sample_rate)

                # Update progress
                jobs_store[job_id]["progress"] = 70
            else:
                # Fallback to onset detection
                logger.info("Using onset detection for drum transcription")

                # Update progress
                jobs_store[job_id]["progress"] = 60

                # Detect onsets and classify drums
                drum_events = self._detect_onsets_from_audio(audio)

                # Update progress
                jobs_store[job_id]["progress"] = 70

            # Update progress
            jobs_store[job_id]["progress"] = 80

            # Convert to MIDI
            midi_data = self._create_midi(drum_events)

            # Update progress
            jobs_store[job_id]["progress"] = 90

            return midi_data

        except Exception as e:  # pylint: disable=broad-except
            logger.error("Transcription failed: %s", str(e))
            raise

    def _run_tf2_model_inference(self, audio: np.ndarray, sr: int) -> Dict:
        """
        Run TF2 model inference on audio
        """
        # Exceptions propagate intentionally — the caller skips the chart rather than silently
        # producing heuristic output disguised as ML transcription.
        spec = self._compute_spectrogram_for_model(audio, sr)
        spec_input = spec[np.newaxis, :, :, np.newaxis]
        outputs = self.model(spec_input, training=False)
        return self._process_tf2_model_outputs(outputs, self.MODEL_SAMPLE_RATE)

    def _extract_features(self, audio: np.ndarray) -> np.ndarray:
        """Extract mel-spectrogram features from audio"""
        # Compute mel-spectrogram
        mel_spec = librosa.feature.melspectrogram(
            y=audio,
            sr=self.sample_rate,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            n_mels=self.n_mels,
            fmin=self.fmin,
            fmax=self.fmax,
        )

        # Convert to log scale
        log_mel = librosa.power_to_db(mel_spec, ref=np.max)

        # Normalize
        std = log_mel.std()
        if std < 1e-8:
            raise ValueError(
                f"audio spectrogram has near-zero standard deviation ({std:.2e}); "
                "input may be silent or corrupt"
            )
        log_mel = (log_mel - log_mel.mean()) / std

        # Transpose for model input (time, features)
        return log_mel.T

    def _detect_drum_events(self, audio: np.ndarray, features: np.ndarray) -> Dict:
        """
        Detect drum events using onset detection and spectral analysis
        """
        drum_events = {pitch: [] for pitch in self.DRUM_MAP.keys()}

        # Onset detection
        onset_envelope = librosa.onset.onset_strength(
            y=audio, sr=self.sample_rate, hop_length=self.hop_length
        )

        # Find peaks (onsets)
        peaks = librosa.util.peak_pick(
            onset_envelope,
            pre_max=3,
            post_max=3,
            pre_avg=3,
            post_avg=5,
            delta=0.5,
            wait=10,
        )

        # Convert frame indices to time
        onset_times = librosa.frames_to_time(peaks, sr=self.sample_rate, hop_length=self.hop_length)

        # Classify each onset (simplified drum classification)
        for onset_time, peak_idx in zip(onset_times, peaks):
            # Extract spectral features around onset
            window_start = max(0, peak_idx - 2)
            window_end = min(len(features), peak_idx + 3)

            if window_end > window_start:
                spectral_window = features[window_start:window_end]

                # Simple frequency-based classification
                mean_spectrum = np.mean(spectral_window, axis=0)

                # Classify based on spectral centroid (simplified)
                if self._is_kick(mean_spectrum):
                    drum_events[36].append(
                        {
                            "time": onset_time,
                            "velocity": self._estimate_velocity(onset_envelope[peak_idx]),
                        }
                    )
                elif self._is_snare(mean_spectrum):
                    drum_events[38].append(
                        {
                            "time": onset_time,
                            "velocity": self._estimate_velocity(onset_envelope[peak_idx]),
                        }
                    )
                elif self._is_hihat(mean_spectrum):
                    drum_events[42].append(
                        {
                            "time": onset_time,
                            "velocity": self._estimate_velocity(onset_envelope[peak_idx]),
                        }
                    )
                else:
                    # Default to hi-hat for other percussion
                    drum_events[42].append(
                        {
                            "time": onset_time,
                            "velocity": self._estimate_velocity(onset_envelope[peak_idx]),
                        }
                    )

        return drum_events

    def _is_kick(self, spectrum: np.ndarray) -> bool:
        """Check if spectrum matches kick drum characteristics"""
        # Kick drums have strong low frequency content
        low_freq_energy = np.mean(spectrum[:20])
        high_freq_energy = np.mean(spectrum[20:])
        return low_freq_energy > high_freq_energy * 1.5

    def _is_snare(self, spectrum: np.ndarray) -> bool:
        """Check if spectrum matches snare drum characteristics"""
        # Snare drums have energy in mid frequencies
        mid_freq_energy = np.mean(spectrum[20:100])
        total_energy = np.mean(spectrum)
        return mid_freq_energy > total_energy * 0.6

    def _is_hihat(self, spectrum: np.ndarray) -> bool:
        """Check if spectrum matches hi-hat characteristics"""
        # Hi-hats have high frequency content
        high_freq_energy = np.mean(spectrum[100:])
        total_energy = np.mean(spectrum)
        return high_freq_energy > total_energy * 0.4

    def _estimate_velocity(self, strength: float) -> int:
        """Estimate MIDI velocity from onset strength"""
        # Normalize to MIDI velocity range (1-127)
        velocity = int(min(127, max(1, strength * 50)))
        return velocity

    def _create_midi(self, drum_events: Dict) -> bytes:
        """Create MIDI file from detected drum events"""
        midi = pretty_midi.PrettyMIDI()

        # Create drum track (channel 9 for drums in General MIDI)
        drum_track = pretty_midi.Instrument(program=0, is_drum=True)

        # Add notes for each drum type
        for drum_id, events in drum_events.items():
            for event in events:
                # Handle both tuple and dict formats
                if isinstance(event, tuple):
                    time, velocity = event
                else:
                    time = event["time"]
                    velocity = event["velocity"]

                note = pretty_midi.Note(
                    velocity=int(velocity),
                    pitch=drum_id,
                    start=time,
                    end=time + 0.1,  # Short duration for drums
                )
                drum_track.notes.append(note)

        # Add instrument to MIDI
        midi.instruments.append(drum_track)

        # Convert to bytes
        import io

        midi_io = io.BytesIO()
        midi.write(midi_io)
        midi_io.seek(0)

        return midi_io.read()

    def _compute_spectrogram_for_model(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """Compute mel spectrogram for TF2 model input"""
        # Resample to 16kHz if needed (standard for Magenta models)
        if sr != self.MODEL_SAMPLE_RATE:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=self.MODEL_SAMPLE_RATE)
            sr = self.MODEL_SAMPLE_RATE

        # Compute mel spectrogram
        mel_spec = librosa.feature.melspectrogram(
            y=audio,
            sr=sr,
            n_fft=2048,
            hop_length=512,
            n_mels=229,  # Magenta uses 229 mel bins
            fmin=30,
            fmax=sr / 2,
        )

        # Convert to log scale
        log_mel = librosa.power_to_db(mel_spec, ref=np.max)

        # Transpose to [time, freq] format
        log_mel = log_mel.T

        return log_mel

    def _process_tf2_model_outputs(self, outputs: Dict, sr: int) -> Dict:
        """Process TF2 model outputs into drum events"""
        drum_events = {drum_midi: [] for drum_midi in self.drum_mapping.keys()}

        # Get predictions from model outputs
        onset_probs = outputs["onset_probs"].numpy()[0]
        velocity_values = outputs["velocity_values"].numpy()[0]

        hop_length = 512
        for pitch, drum_midi in self.TF2_OUTPUT_BIN_TO_DRUM_MIDI.items():
            if drum_midi not in self.drum_mapping:
                continue
            if pitch >= onset_probs.shape[1]:
                continue

            pitch_onsets = onset_probs[:, pitch]
            pitch_velocities = velocity_values[:, pitch]

            onset_indices = self._find_onset_peaks(
                pitch_onsets,
                threshold=self.MODEL_ONSET_THRESHOLD,
                min_gap_frames=self.MODEL_ONSET_MIN_GAP_FRAMES,
            )

            for idx in onset_indices:
                onset_time = idx * hop_length / sr
                velocity = int(np.clip(pitch_velocities[idx] * 127, 1, 127))
                drum_events[drum_midi].append({"time": onset_time, "velocity": velocity})

        return drum_events

    def _find_onset_peaks(
        self,
        signal: np.ndarray,
        threshold: float = 0.3,
        min_gap_frames: int = 1,
    ) -> np.ndarray:
        """Find peaks in a signal above threshold"""
        peaks = []
        last_peak = -(10**9)
        for idx in range(1, len(signal) - 1):
            if signal[idx] < threshold:
                continue
            if signal[idx] <= signal[idx - 1]:
                continue
            if signal[idx] < signal[idx + 1]:
                continue
            if idx - last_peak < min_gap_frames:
                continue
            peaks.append(idx)
            last_peak = idx

        return np.array(peaks)
