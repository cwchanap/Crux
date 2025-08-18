"""
Drum transcription using trained Magenta E-GMD model
Directly loads the checkpoint without requiring full Magenta installation
"""

import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional

import librosa
import numpy as np
import pretty_midi
import requests
import tensorflow as tf

from .tf2_magenta_model import create_drum_model

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

    def __init__(self, model_path: Optional[str] = None, sample_rate: int = 44100):
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

        if model_path is None:
            # Check for converted TF2 weights first
            tf2_weights_path = "models/e-gmd/tf2_model.weights.h5"
            if os.path.exists(tf2_weights_path):
                self.model_path = tf2_weights_path
                logger.info(f"Found converted TF2 weights at {tf2_weights_path}")
            else:
                # Download E-GMD model if not provided
                self.model_path = self._download_model()

        # Build model
        try:
            self.model = self._build_model()
        except Exception as e:
            logger.warning(f"Could not load model: {e}. Using fallback method.")
            self.model = None

    def _download_model(self) -> str:
        """Download the E-GMD model checkpoint if not already present"""
        model_dir = Path("models/e-gmd")
        checkpoint_path = model_dir / "train/model.ckpt-10000"

        if checkpoint_path.with_suffix(".index").exists():
            logger.info("E-GMD model already downloaded")
            return str(checkpoint_path)

        # Try alternative download URLs
        model_urls = [
            "https://storage.googleapis.com/magentadata/models/onsets_frames_transcription/e-gmd/model.ckpt-10000.zip",
            "https://storage.googleapis.com/magentadata/models/onsets_frames_transcription/e_gmd_checkpoint.zip",
        ]

        logger.info("Downloading E-GMD model checkpoint...")

        for model_url in model_urls:
            try:
                response = requests.get(model_url, stream=True, timeout=10)
                response.raise_for_status()

                # Save the zip file
                model_dir.mkdir(parents=True, exist_ok=True)
                zip_path = model_dir / "checkpoint.zip"

                with open(zip_path, "wb") as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)

                # Extract the checkpoint
                import zipfile

                with zipfile.ZipFile(zip_path, "r") as zip_ref:
                    zip_ref.extractall(model_dir)

                # Clean up zip file
                zip_path.unlink()

                logger.info("E-GMD model downloaded successfully")
                return str(checkpoint_path)

            except Exception as e:
                logger.warning(f"Failed with URL {model_url}: {e}")
                continue

        logger.error("Failed to download model from any source")
        # For testing, create a dummy model
        logger.warning("Creating dummy model for testing purposes")
        return None

    def _build_model(self):
        """
        Load the trained Magenta drum transcription model directly
        """
        if not self.model_path:
            logger.warning("No model path available, using fallback method")
            return None

        try:
            from app.tf2_magenta_model import create_drum_model, load_tf1_checkpoint_to_tf2

            # Create the TF2 model
            model = create_drum_model()

            # Check for converted TF2 weights first
            tf2_weights_path = "models/e-gmd/tf2_model.weights.h5"
            if os.path.exists(tf2_weights_path):
                model.load_weights(tf2_weights_path)
                logging.info(f"Loaded converted TF2 weights from {tf2_weights_path}")
                return model

            # Try to load weights if checkpoint exists
            if self.model_path and os.path.exists(self.model_path):
                if self.model_path.endswith(".weights.h5"):
                    # Load TF2 weights directly
                    model.load_weights(self.model_path)
                    logging.info(f"Loaded TF2 weights from {self.model_path}")
                else:
                    # Try to load and convert TF1 checkpoint
                    model = load_tf1_checkpoint_to_tf2(self.model_path, model)
                    logging.info(f"Loaded and converted TF1 checkpoint from {self.model_path}")
            else:
                logging.warning("No model path available, using fallback method")
                return None

            return model

        except Exception as e:
            logging.error(f"Failed to build TF2 model: {e}")
            return None

    def _load_model(self, checkpoint_path: str) -> bool:
        """Load the Magenta model from checkpoint"""
        try:
            # Create and load TF2-compatible model
            self.model = create_drum_model(checkpoint_path)
            logger.info("Successfully loaded TF2-compatible model")
            return True
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            logger.warning("Using fallback onset detection method")
            return False

    def _build_egmd_architecture(self):
        """
        Build the E-GMD model architecture
        Based on the Onsets and Frames architecture for drums
        """
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
            logger.info(f"Loading audio file: {audio_path}")
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

        except Exception as e:
            logger.error(f"Transcription failed: {str(e)}")
            raise

    def _run_tf2_model_inference(self, audio: np.ndarray, sr: int) -> Dict:
        """
        Run TF2 model inference on audio
        """
        try:
            # Compute spectrogram for model input
            spec = self._compute_spectrogram_for_model(audio, sr)

            # Add batch and channel dimensions [batch, time, freq, channels]
            spec_input = spec[np.newaxis, :, :, np.newaxis]

            # Run inference
            outputs = self.model(spec_input, training=False)

            # Process outputs to drum events
            drum_events = self._process_tf2_model_outputs(outputs, sr)

            return drum_events
        except Exception as e:
            logger.error(f"TF2 model inference failed: {e}")
            logger.warning("Falling back to onset detection")
            return self._detect_onsets_from_audio(audio)

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
        log_mel = (log_mel - log_mel.mean()) / log_mel.std()

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
            onset_envelope, pre_max=3, post_max=3, pre_avg=3, post_avg=5, delta=0.5, wait=10
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
        if sr != 16000:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
            sr = 16000

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

        # Map piano keys to drum types (simplified mapping for E-GMD)
        drum_key_ranges = {
            36: range(35, 37),  # Kick drum
            38: range(37, 41),  # Snare
            42: range(41, 45),  # Hi-hat closed
            46: range(44, 46),  # Hi-hat open
            49: range(45, 52),  # Crash cymbal
            51: range(50, 53),  # Ride
            41: range(52, 55),  # Tom low
            47: range(55, 58),  # Tom mid
            50: range(58, 60),  # Tom high
        }

        # Process onsets for each drum type
        hop_length = 512
        for drum_midi, key_range in drum_key_ranges.items():
            if drum_midi not in self.drum_mapping:
                continue

            for pitch in key_range:
                if pitch >= onset_probs.shape[1]:
                    continue

                # Get onset probabilities for this pitch
                pitch_onsets = onset_probs[:, pitch]
                pitch_velocities = velocity_values[:, pitch]

                # Find onset peaks with threshold
                onset_indices = self._find_onset_peaks(pitch_onsets, threshold=0.3)

                # Get velocities for detected onsets
                if len(onset_indices) > 0:
                    for idx in onset_indices:
                        onset_time = idx * hop_length / sr

                        # Get velocity at onset frame
                        velocity = int(np.clip(pitch_velocities[idx] * 127, 1, 127))

                        drum_events[drum_midi].append({"time": onset_time, "velocity": velocity})

        return drum_events

    def _find_onset_peaks(self, signal: np.ndarray, threshold: float = 0.3) -> np.ndarray:
        """Find peaks in a signal above threshold"""
        above_threshold = signal > threshold

        # Find where signal crosses threshold
        diff = np.diff(np.concatenate(([False], above_threshold, [False])))
        starts = np.where(diff == 1)[0]
        ends = np.where(diff == -1)[0]

        # Find peaks within each above-threshold region
        peaks = []
        for start, end in zip(starts, ends):
            # Extend search window slightly
            start = max(0, start - 1)
            end = min(len(signal), end + 1)

            # Find local maxima
            while end < len(signal) and signal[end] > signal[end - 1]:
                end += 1

            # Only find peak if we have a valid range
            if end > start and end - start > 0:
                segment = signal[start:end]
                if len(segment) > 0:
                    peak_idx = start + np.argmax(segment)
                    peaks.append(peak_idx)

        return np.array(peaks)
