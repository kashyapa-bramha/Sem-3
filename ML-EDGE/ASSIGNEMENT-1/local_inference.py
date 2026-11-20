"""Lab 1, Step 3: Benchmark the MNIST TFLite model on the local CPU."""

from pathlib import Path
from time import perf_counter

import numpy as np
import tensorflow as tf


def main():
    # Use the same image for every benchmark, never a random replacement.
    assignment_dir = Path(__file__).resolve().parent
    sample_path = assignment_dir / "test_sample.npy"
    if not sample_path.is_file():
        raise SystemExit(f"Missing Step 1 sample: {sample_path}. Restore it before benchmarking.")
    sample = np.load(sample_path, allow_pickle=False)

    # Allocate working memory once, outside the timed loop.
    interpreter = tf.lite.Interpreter(model_path=str(assignment_dir / "model.tflite"))
    interpreter.allocate_tensors()
    input_tensor = interpreter.get_input_details()[0]
    output_tensor = interpreter.get_output_details()[0]
    input_index = input_tensor["index"]

    # The first invocation can include one-time setup; exclude it from the timings.
    interpreter.set_tensor(input_index, sample)
    interpreter.invoke()

    latencies_ms = []
    for _ in range(100):
        start = perf_counter()
        interpreter.set_tensor(input_index, sample)
        interpreter.invoke()
        latencies_ms.append((perf_counter() - start) * 1000)

    # Read the final softmax scores after timing so output handling is not measured.
    scores = interpreter.get_tensor(output_tensor["index"])[0]
    predicted_class = int(np.argmax(scores))
    mean_ms = np.mean(latencies_ms)

    print("\nEDGE INFERENCE RESULTS")
    print("=" * 48)
    print(f"Input shape     : {input_tensor['shape'].tolist()}")
    print(f"Output shape    : {output_tensor['shape'].tolist()}")
    print(f"Timed inferences: {len(latencies_ms)} (after 1 warm-up)")
    print(f"Predicted class : {predicted_class}")
    print(f"Confidence      : {scores[predicted_class]:.4f}")
    print(f"Mean latency    : {mean_ms:.4f} ms")
    print(f"Median latency  : {np.median(latencies_ms):.4f} ms")
    print(f"p95 latency     : {np.percentile(latencies_ms, 95):.4f} ms")
    print(f"Maximum latency : {max(latencies_ms):.4f} ms")
    print(f"Throughput      : {1000 / mean_ms:,.0f} inferences/sec")


if __name__ == "__main__":
    main()