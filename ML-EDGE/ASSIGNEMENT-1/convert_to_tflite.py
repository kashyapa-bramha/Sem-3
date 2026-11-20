"""Lab 1, Step 2: Convert the trained MNIST CNN to TensorFlow Lite."""

from pathlib import Path

import tensorflow as tf


def main():
    # Look beside this script for the trained model, regardless of the working directory.
    assignment_dir = Path(__file__).resolve().parent
    keras_path = assignment_dir / "mnist_model.keras"
    tflite_path = assignment_dir / "model.tflite"
    model = tf.keras.models.load_model(keras_path, compile=False)

    # Dynamic-range quantization compresses eligible weights without calibration images.
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_path.write_bytes(converter.convert())

    # Compare the actual files on disk; 1 KiB is 1,024 bytes.
    keras_bytes = keras_path.stat().st_size
    tflite_bytes = tflite_path.stat().st_size
    print(f"\nConversion complete: {tflite_path}")
    print(f"Original Keras model: {keras_bytes:,} bytes ({keras_bytes / 1024:.1f} KiB)")
    print(f"TFLite model: {tflite_bytes:,} bytes ({tflite_bytes / 1024:.1f} KiB)")
    print(f"Compression ratio (Keras / TFLite): {keras_bytes / tflite_bytes:.2f}x")


if __name__ == "__main__":
    main()