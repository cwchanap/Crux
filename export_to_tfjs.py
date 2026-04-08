#!/usr/bin/env python3
"""
Export the Drum Transcription TF2 Keras model to TensorFlow.js format.

This script:
- Builds the TF2 model architecture (same as `create_drum_model`)
- Loads TF2 weights (.h5) if provided
- Wraps the subclassed model in a functional Keras Model
- Exports to TFJS Layers format (model.json + weight shards)

Usage examples:
  uv run python export_to_tfjs.py \
    --weights models/e-gmd/tf2_model.weights.h5 \
    --output-dir web_model

  # If you prefer H5 first, then tfjs converter CLI:
  uv run python export_to_tfjs.py --weights models/e-gmd/tf2_model.weights.h5 --save-keras-h5 keras_model.h5
  tensorflowjs_converter --input_format=keras --output_format=tfjs_layers_model keras_model.h5 web_model

Requirements:
  pip install tensorflow tensorflowjs

Notes:
- Input shape is [batch, time, 229 mel bins, 1 channel]. Time is variable-length (None).
- Output is a dict with keys: onset_probs, offset_probs, velocity_values, frame_probs
  Those names are preserved in the exported TFJS layers model.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import tensorflow as tf

from src.app.tf2_magenta_model import create_drum_model

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("export_to_tfjs")


def build_functional_model(weights_path: str | None) -> tf.keras.Model:
    """Create a functional Keras model with named outputs and optional weights.

    If weights_path ends with .h5 or .weights.h5, it will be loaded via create_drum_model().
    """
    # Create the subclassed model and load weights if provided
    base = create_drum_model(checkpoint_path=weights_path)

    # Build functional wrapper with variable time dimension
    # Use batch_shape to ensure batch_input_shape is present in the config for TFJS importer
    inputs = tf.keras.Input(batch_shape=(None, None, 229, 1), name="inputs", dtype="float32")
    outputs = base(inputs, training=False)
    # Ensure list outputs with stable ordering and distinct output names
    # for TFJS Layers export.  A subclassed model called in a functional
    # context attributes every output tensor to the parent call layer, so
    # without explicit wrapping model.output_names collapses to duplicates
    # like ["onsets_frames_model", "onsets_frames_model_1", ...].
    if isinstance(outputs, dict):
        outputs = [
            tf.keras.layers.Activation("linear", name="onset_probs")(outputs["onset_probs"]),
            tf.keras.layers.Activation("linear", name="offset_probs")(outputs["offset_probs"]),
            tf.keras.layers.Activation("linear", name="velocity_values")(
                outputs["velocity_values"]
            ),
            tf.keras.layers.Activation("linear", name="frame_probs")(outputs["frame_probs"]),
        ]
    func_model = tf.keras.Model(inputs=inputs, outputs=outputs, name="drums_onsets_frames")

    # Show summary for verification
    try:
        func_model.summary(print_fn=lambda s: logger.info(s))
    except Exception:  # summary may fail in some terminals
        pass

    return func_model


def export_to_tfjs(
    weights_path: str | None,
    output_dir: Path,
    save_keras_h5: Path | None = None,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    model = build_functional_model(weights_path)

    if save_keras_h5 is None:
        # Try the Python API first
        try:
            import tensorflowjs as tfjs  # type: ignore

            logger.info("Exporting Keras model to TFJS (Layers format): %s", output_dir)
            tfjs.converters.save_keras_model(model, str(output_dir))
            logger.info("✅ Exported TFJS model to: %s", output_dir)
            return
        except Exception as e:
            logger.warning(
                "tensorflowjs Python package not available or failed (%s).\n"
                "Falling back to saving Keras H5 for CLI conversion.",
                e,
            )

    # Fallback: save to H5 and instruct user to run CLI
    h5_path = save_keras_h5 or output_dir / "keras_model.h5"
    h5_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Saving Keras H5 to: %s", h5_path)
    model.save(str(h5_path), include_optimizer=False)

    logger.info(
        "You can now run: \n  tensorflowjs_converter --input_format=keras --output_format=tfjs_layers_model %s %s",
        h5_path,
        output_dir,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export TF2 drum model to TFJS format")
    parser.add_argument(
        "--weights",
        type=str,
        default="models/e-gmd/tf2_model.weights.h5",
        help="Path to TF2 weights (.h5). If missing, the model exports with randomly initialized weights.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="web_model",
        help="Output directory for TFJS model (model.json + shards)",
    )
    parser.add_argument(
        "--save-keras-h5",
        action="store",
        dest="save_keras_h5",
        help="path to write Keras H5 model",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    weights_path = args.weights
    output_dir = Path(args.output_dir)
    save_keras_h5 = Path(args.save_keras_h5) if args.save_keras_h5 else None

    if weights_path and not Path(weights_path).exists():
        logger.warning("Weights not found at %s. Proceeding without loading weights.", weights_path)
        weights_path = None

    export_to_tfjs(weights_path, output_dir, save_keras_h5=save_keras_h5)


if __name__ == "__main__":
    main()
