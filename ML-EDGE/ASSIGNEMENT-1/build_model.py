"""Lab 1, Step 1: Train a lightweight MNIST CNN for edge and cloud inference."""

from pathlib import Path

import numpy as np
import tensorflow as tf


def build_model():
    # Learn local stroke patterns, then combine them into ten digit probabilities.
    return tf.keras.Sequential(
        [
            tf.keras.Input(shape=(28, 28, 1)),
            tf.keras.layers.Conv2D(8, kernel_size=3, padding="same", activation="relu"),
            tf.keras.layers.MaxPooling2D(pool_size=2),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(32, activation="relu"),
            tf.keras.layers.Dense(10, activation="softmax"),
        ],
        name="mnist_cnn",
    )


def main():
    # Add one grayscale channel and scale pixel brightness from 0-255 to 0-1.
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train = x_train.reshape(-1, 28, 28, 1).astype("float32") / 255.0
    x_test = x_test.reshape(-1, 28, 28, 1).astype("float32") / 255.0
    print(f"Training samples: {len(x_train):,}")
    print(f"Test samples: {len(x_test):,}")

    # Sparse categorical cross-entropy uses the integer digit labels directly.
    model = build_model()
    model.summary()
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

    # Reserve 10% of the training set for validation; leave the test set untouched.
    model.fit(x_train, y_train, epochs=5, batch_size=64, validation_split=0.1, verbose=1)

    # Measure accuracy on digits the model did not use for training or validation.
    _, accuracy = model.evaluate(x_test, y_test, verbose=0)
    print(f"\nTest accuracy: {accuracy:.4f} ({accuracy:.2%})")

    # Keep the batch dimension on the first image so both benchmarks can use it directly.
    output_dir = Path(__file__).resolve().parent
    model.save(output_dir / "mnist_model.keras")
    np.save(output_dir / "test_sample.npy", x_test[:1])
    print(f"Model saved to {output_dir / 'mnist_model.keras'}")
    print(f"Test sample saved to {output_dir / 'test_sample.npy'} (true label: {y_test[0]})")


if __name__ == "__main__":
    main()