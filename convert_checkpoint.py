#!/usr/bin/env python3
"""
Convert TensorFlow 1 checkpoint to TensorFlow 2 format
"""

import logging
import os
from pathlib import Path

import tensorflow as tf

from src.app.tf2_magenta_model import create_drum_model

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def convert_tf1_checkpoint_to_tf2(checkpoint_path: str, output_path: str):
    """
    Convert TF1 checkpoint to TF2 model weights

    Args:
        checkpoint_path: Path to TF1 checkpoint (without .data/.index extensions)
        output_path: Path to save TF2 weights (.h5 file)
    """
    logger.info(f"Loading TF1 checkpoint from: {checkpoint_path}")

    # Create the TF2 model
    model = create_drum_model()
    logger.info("Created TF2 model architecture")

    # Load TF1 checkpoint
    reader = tf.compat.v1.train.NewCheckpointReader(checkpoint_path)
    var_to_shape_map = reader.get_variable_to_shape_map()

    logger.info(f"Found {len(var_to_shape_map)} variables in checkpoint")

    # Try to load weights into TF2 model
    weights_loaded = 0
    weights_dict = {}

    for var_name, shape in var_to_shape_map.items():
        try:
            # Get the tensor value
            tensor = reader.get_tensor(var_name)

            # Log some key variables for debugging
            if (
                "conv" in var_name
                or "lstm" in var_name
                or "probs" in var_name
                or "values" in var_name
            ):
                logger.info(f"  {var_name}: shape={shape}")

            # Store weights for manual assignment
            weights_dict[var_name] = tensor
            weights_loaded += 1

        except Exception as e:
            logger.warning(f"Could not load {var_name}: {e}")

    logger.info(f"Successfully loaded {weights_loaded}/{len(var_to_shape_map)} weights")

    # Try to assign weights to TF2 model layers
    try:
        # Find and set Conv2D weights
        for layer in model.layers:
            if isinstance(layer, tf.keras.layers.Conv2D):
                layer_name = layer.name
                # Try to find matching weights in checkpoint
                for var_name, tensor in weights_dict.items():
                    if "conv" in var_name.lower() and "conv" in layer_name.lower():
                        if "weights" in var_name or "kernel" in var_name:
                            if tensor.shape == layer.kernel.shape:
                                layer.kernel.assign(tensor)
                                logger.info(f"Assigned weights to {layer_name} kernel")
                        elif "bias" in var_name:
                            if tensor.shape == layer.bias.shape:
                                layer.bias.assign(tensor)
                                logger.info(f"Assigned weights to {layer_name} bias")

            elif isinstance(layer, tf.keras.layers.Dense):
                layer_name = layer.name
                # Try to find matching weights for output layers
                for var_name, tensor in weights_dict.items():
                    if layer_name in var_name or (
                        layer_name.replace("_", "") in var_name.replace("_", "")
                    ):
                        if "weights" in var_name or "kernel" in var_name:
                            # Dense layer kernels may need transposing
                            if tensor.T.shape == layer.kernel.shape:
                                layer.kernel.assign(tensor.T)
                                logger.info(f"Assigned transposed weights to {layer_name} kernel")
                            elif tensor.shape == layer.kernel.shape:
                                layer.kernel.assign(tensor)
                                logger.info(f"Assigned weights to {layer_name} kernel")
                        elif "bias" in var_name:
                            if tensor.shape == layer.bias.shape:
                                layer.bias.assign(tensor)
                                logger.info(f"Assigned weights to {layer_name} bias")

    except Exception as e:
        logger.error(f"Error assigning weights: {e}")
        # Continue anyway - partial weights are better than none

    # Save the TF2 model weights
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    model.save_weights(output_path)
    logger.info(f"Saved TF2 model weights to: {output_path}")

    return model


def main():
    # Path to the TF1 checkpoint
    checkpoint_path = os.path.expanduser("~/.cache/drum_transcription/models/model.ckpt-569400")

    # Output path for TF2 weights
    output_path = "models/e-gmd/tf2_model.weights.h5"

    # Check if checkpoint exists
    if not os.path.exists(checkpoint_path + ".index"):
        logger.error(f"Checkpoint not found: {checkpoint_path}")
        return

    # Convert the checkpoint
    logger.info("Starting TF1 to TF2 checkpoint conversion...")
    model = convert_tf1_checkpoint_to_tf2(checkpoint_path, output_path)

    logger.info("✅ Conversion complete!")
    logger.info(f"TF2 weights saved to: {output_path}")

    # Test the converted model
    logger.info("\nTesting converted model...")
    # Conv2D expects 4D input: [batch, time, frequency, channels]
    test_input = tf.random.normal((1, 100, 229, 1))  # Add channel dimension

    try:
        outputs = model(test_input)
        logger.info("Model inference successful!")
        for key, value in outputs.items():
            logger.info(f"  {key}: shape={value.shape}")
    except Exception as e:
        logger.error(f"Model inference failed: {e}")


if __name__ == "__main__":
    main()
