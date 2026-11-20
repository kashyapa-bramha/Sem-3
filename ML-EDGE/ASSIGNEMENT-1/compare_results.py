"""Lab 1, Step 5: Compare local TFLite inference with cloud HTTP round trips."""

from pathlib import Path
from time import perf_counter

import numpy as np
import requests
import tensorflow as tf

from cloud_client import SERVER_URL


ITERATIONS = 100


def benchmark_edge(model_path, sample):
    interpreter = tf.lite.Interpreter(model_path=str(model_path))
    interpreter.allocate_tensors()
    input_index = interpreter.get_input_details()[0]["index"]
    output_index = interpreter.get_output_details()[0]["index"]

    interpreter.set_tensor(input_index, sample)
    interpreter.invoke()  # Untimed warm-up, as in Step 3.
    times_ms = []
    for _ in range(ITERATIONS):
        start = perf_counter()
        interpreter.set_tensor(input_index, sample)
        interpreter.invoke()
        times_ms.append((perf_counter() - start) * 1000)
    prediction = int(np.argmax(interpreter.get_tensor(output_index)[0]))
    return np.asarray(times_ms), prediction


def benchmark_cloud(sample):
    url = f"{SERVER_URL}/infer"
    payload = {"data": sample.flatten().tolist()}
    warmup = requests.post(url, json=payload, timeout=10)
    warmup.raise_for_status()
    payload_bytes = len(warmup.request.body)

    # Include JSON encoding and response transfer in RTT, but not response parsing.
    times_ms = []
    for _ in range(ITERATIONS):
        start = perf_counter()
        response = requests.post(url, json=payload, timeout=10)
        times_ms.append((perf_counter() - start) * 1000)
        response.raise_for_status()
    return np.asarray(times_ms), response.json()["prediction"], payload_bytes


def main():
    assignment_dir = Path(__file__).resolve().parent
    sample_path = assignment_dir / "test_sample.npy"
    if not sample_path.is_file():
        raise SystemExit(f"Missing Step 1 sample: {sample_path}. Restore it before benchmarking.")
    sample = np.load(sample_path, allow_pickle=False)

    try:
        health = requests.get(f"{SERVER_URL}/", timeout=5)
        health.raise_for_status()
        if health.json().get("status") != "healthy":
            raise SystemExit(f"Server at {SERVER_URL} did not report healthy.")
        edge, edge_prediction = benchmark_edge(assignment_dir / "model.tflite", sample)
        cloud, cloud_prediction, payload_bytes = benchmark_cloud(sample)
    except requests.RequestException as error:
        raise SystemExit(f"Comparison failed. Check the Step 4a server at {SERVER_URL}: {error}") from error

    edge_mean, cloud_mean = edge.mean(), cloud.mean()
    print(f"\n{ITERATIONS} timed inferences per path after 1 untimed warm-up each")
    print(f"{'METRIC':<28}{'EDGE':>16}{'CLOUD (RTT)':>16}")
    print("-" * 60)
    print(f"{'Mean latency (ms)':<28}{edge_mean:>16.4f}{cloud_mean:>16.4f}")
    print(f"{'Median latency (ms)':<28}{np.median(edge):>16.4f}{np.median(cloud):>16.4f}")
    print(f"{'p95 latency (ms)':<28}{np.percentile(edge, 95):>16.4f}{np.percentile(cloud, 95):>16.4f}")
    print(f"{'Throughput (inf/s)':<28}{1000 / edge_mean:>16,.1f}{1000 / cloud_mean:>16,.1f}")
    print(f"{'Upload per inference (B)':<28}{0:>16}{payload_bytes:>16,}")
    print(f"{'Network required':<28}{'No':>16}{'Yes':>16}")
    print(f"{'Prediction':<28}{edge_prediction:>16}{cloud_prediction:>16}")
    print(f"\nEdge speedup (mean RTT / mean edge): {cloud_mean / edge_mean:,.1f}x")
    print(f"Cloud upload at 30 FPS: {payload_bytes * 30 / 1024:.2f} KiB/s (request bodies only)")


if __name__ == "__main__":
    main()