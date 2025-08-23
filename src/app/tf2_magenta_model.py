"""TensorFlow 2 compatible version of Magenta's onsets_frames model for drum transcription."""

import logging

import tensorflow as tf
from tensorflow.keras import layers

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ConvStack(tf.keras.Model):
    """Convolutional stack from Kelz 2016 architecture."""

    def __init__(self, hparams):
        super().__init__()
        self.conv_layers = []
        self.pool_layers = []
        self.dropout_layers = []
        self.batch_norms = []

        # Default hyperparameters based on Magenta's implementation
        temporal_sizes = getattr(hparams, "temporal_sizes", [3, 3, 3])
        freq_sizes = getattr(hparams, "freq_sizes", [3, 3, 3])
        num_filters = getattr(hparams, "num_filters", [48, 48, 96])
        pool_sizes = getattr(hparams, "pool_sizes", [1, 2, 2])
        dropout_keep_amts = getattr(hparams, "dropout_keep_amts", [1.0, 0.75, 0.75])

        for i, (
            conv_temporal_size,
            conv_freq_size,
            n_filters,
            freq_pool_size,
            dropout_amt,
        ) in enumerate(zip(temporal_sizes, freq_sizes, num_filters, pool_sizes, dropout_keep_amts)):
            # Convolutional layer
            self.conv_layers.append(
                layers.Conv2D(
                    n_filters,
                    (conv_temporal_size, conv_freq_size),
                    padding="same",
                    name=f"conv_{i}",
                )
            )

            # Batch normalization
            self.batch_norms.append(layers.BatchNormalization(name=f"batch_norm_{i}"))

            # Pooling layer (if needed)
            if freq_pool_size > 1:
                self.pool_layers.append(
                    layers.MaxPool2D(
                        (1, freq_pool_size), strides=(1, freq_pool_size), name=f"pool_{i}"
                    )
                )
            else:
                self.pool_layers.append(None)

            # Dropout layer (if needed)
            if dropout_amt < 1.0:
                self.dropout_layers.append(layers.Dropout(1.0 - dropout_amt, name=f"dropout_{i}"))
            else:
                self.dropout_layers.append(None)

        # Final fully connected layer
        fc_size = getattr(hparams, "fc_size", 768)
        fc_dropout = getattr(hparams, "fc_dropout_keep_amt", 0.5)

        self.flatten = layers.Reshape((-1, -1))  # Preserve time dimension
        self.fc = layers.Dense(fc_size, activation="relu", name="fc_end")
        self.fc_dropout = layers.Dropout(1.0 - fc_dropout, name="dropout_end")

    def call(self, inputs, training=False):
        net = inputs

        # Apply convolutional layers
        for i, (conv, bn, pool, dropout) in enumerate(
            zip(self.conv_layers, self.batch_norms, self.pool_layers, self.dropout_layers)
        ):
            net = conv(net)
            net = bn(net, training=training)
            net = tf.nn.relu(net)

            if pool is not None:
                net = pool(net)

            if dropout is not None:
                net = dropout(net, training=training)

        # Flatten spatial dimensions while preserving batch and time
        batch_size = tf.shape(net)[0]
        time_steps = tf.shape(net)[1]
        net = tf.reshape(net, (batch_size, time_steps, -1))

        # Final fully connected layer
        net = self.fc(net)
        net = self.fc_dropout(net, training=training)

        return net


class BiLSTMLayer(tf.keras.Model):
    """Bidirectional LSTM layer."""

    def __init__(self, num_units, stack_size=1):
        super().__init__()
        self.lstm_layers = []

        for i in range(stack_size):
            self.lstm_layers.append(
                layers.Bidirectional(
                    layers.LSTM(num_units, return_sequences=True), name=f"bilstm_{i}"
                )
            )

    def call(self, inputs, training=False):
        net = inputs
        for lstm in self.lstm_layers:
            net = lstm(net, training=training)
        return net


class AcousticModel(tf.keras.Model):
    """Acoustic model combining ConvNet and optional LSTM."""

    def __init__(self, hparams, lstm_units=0, lstm_stack_size=1):
        super().__init__()
        self.conv_stack = ConvStack(hparams)
        self.lstm_units = lstm_units

        if lstm_units > 0:
            self.lstm = BiLSTMLayer(lstm_units, lstm_stack_size)
        else:
            self.lstm = None

    def call(self, inputs, training=False):
        net = self.conv_stack(inputs, training=training)

        if self.lstm is not None:
            net = self.lstm(net, training=training)

        return net


class OnsetsFramesModel(tf.keras.Model):
    """TF2 implementation of the Onsets and Frames transcription model."""

    def __init__(self, hparams=None):
        super().__init__()

        # Default hyperparameters
        if hparams is None:
            hparams = type("Hparams", (), {})()
            hparams.onset_lstm_units = 256
            hparams.offset_lstm_units = 256
            hparams.velocity_lstm_units = 0
            hparams.frame_lstm_units = 0
            hparams.combined_lstm_units = 256
            hparams.acoustic_rnn_stack_size = 1
            hparams.combined_rnn_stack_size = 1

        self.hparams = hparams

        # Create sub-models for each output type
        self.onset_model = AcousticModel(
            hparams, hparams.onset_lstm_units, hparams.acoustic_rnn_stack_size
        )

        self.offset_model = AcousticModel(
            hparams, hparams.offset_lstm_units, hparams.acoustic_rnn_stack_size
        )

        self.velocity_model = AcousticModel(
            hparams, hparams.velocity_lstm_units, hparams.acoustic_rnn_stack_size
        )

        self.frame_model = AcousticModel(
            hparams, hparams.frame_lstm_units, hparams.acoustic_rnn_stack_size
        )

        # Output layers (88 piano keys or drum classes)
        self.onset_probs = layers.Dense(88, activation="sigmoid", name="onset_probs")
        self.offset_probs = layers.Dense(88, activation="sigmoid", name="offset_probs")
        self.velocity_values = layers.Dense(88, activation=None, name="velocity_values")
        self.activation_probs = layers.Dense(88, activation="sigmoid", name="activation_probs")

        # Combined model for frame predictions
        if hparams.combined_lstm_units > 0:
            self.combined_lstm = BiLSTMLayer(
                hparams.combined_lstm_units, hparams.combined_rnn_stack_size
            )
        else:
            self.combined_lstm = None

        self.frame_probs = layers.Dense(88, activation="sigmoid", name="frame_probs")

    def call(self, inputs, training=False):
        # Process through each sub-model
        onset_outputs = self.onset_model(inputs, training=training)
        onset_probs = self.onset_probs(onset_outputs)

        offset_outputs = self.offset_model(inputs, training=training)
        offset_probs = self.offset_probs(offset_outputs)

        velocity_outputs = self.velocity_model(inputs, training=training)
        velocity_values = self.velocity_values(velocity_outputs)

        frame_outputs = self.frame_model(inputs, training=training)
        activation_probs = self.activation_probs(frame_outputs)

        # Combine for frame predictions
        combined = tf.concat([onset_probs, activation_probs, offset_probs], axis=-1)

        if self.combined_lstm is not None:
            combined = self.combined_lstm(combined, training=training)

        frame_probs = self.frame_probs(combined)

        return {
            "onset_probs": onset_probs,
            "offset_probs": offset_probs,
            "velocity_values": velocity_values,
            "frame_probs": frame_probs,
        }


def load_tf1_checkpoint_to_tf2(model, checkpoint_path):
    """Load TF1 checkpoint weights into TF2 model"""
    # This is a simplified version - actual implementation would need
    # proper variable mapping from TF1 to TF2
    try:
        # Try to load as TF2 checkpoint first
        if checkpoint_path.endswith(".h5") or checkpoint_path.endswith(".weights.h5"):
            model.load_weights(checkpoint_path)
        else:
            # For TF1 checkpoints, we'd need manual conversion
            print("Note: TF1 checkpoint loading not fully implemented")
        return model
    except Exception as e:
        # If that fails, we'd need to manually map TF1 variables
        # This requires knowing the exact variable names in the checkpoint
        print(f"Note: Could not load checkpoint: {e}")
        return model


def create_drum_model(checkpoint_path=None):
    """Create a drum transcription model, optionally loading from checkpoint."""

    # Create model with drum-specific settings
    hparams = type("Hparams", (), {})()
    hparams.onset_lstm_units = 256
    hparams.offset_lstm_units = 256
    hparams.velocity_lstm_units = 0
    hparams.frame_lstm_units = 0
    hparams.combined_lstm_units = 256
    hparams.acoustic_rnn_stack_size = 1
    hparams.combined_rnn_stack_size = 1

    # Drum-specific parameters
    hparams.temporal_sizes = [3, 3, 3]
    hparams.freq_sizes = [3, 3, 3]
    hparams.num_filters = [48, 48, 96]
    hparams.pool_sizes = [1, 2, 2]
    hparams.dropout_keep_amts = [1.0, 0.75, 0.75]
    hparams.fc_size = 768
    hparams.fc_dropout_keep_amt = 0.5

    model = OnsetsFramesModel(hparams)

    # Build model with dummy input to initialize weights
    dummy_input = tf.zeros((1, 100, 229, 1))  # [batch, time, freq, channels]
    _ = model(dummy_input, training=False)
    # Load checkpoint if provided
    if checkpoint_path:
        model = load_tf1_checkpoint_to_tf2(model, checkpoint_path)

    return model


if __name__ == "__main__":
    # Test model creation
    model = create_drum_model()

    # Test inference
    test_input = tf.random.normal((1, 100, 229, 1))
    outputs = model(test_input, training=False)

    print("Model outputs:")
    for key, value in outputs.items():
        print(f"  {key}: {value.shape}")
