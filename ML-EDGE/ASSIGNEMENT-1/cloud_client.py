"""Lab 1, Step 4b: Measure client round-trip time for local HTTP inference."""

from pathlib import Path
from time import perf_counter

import numpy as np
import requests


SERVER_URL = "http://127.0.0.1:8000"


def main():
    # Use the same normalized image as the edge benchmark, not a random substitute.
    sample_path = Path(__file__).with_name("test_sample.npy")
    if not sample_path.is_file():
        raise SystemExit(f"Missing Step 1 sample: {sample_path}. Restore it before benchmarking.")
    sample = np.load(sample_path, allow_pickle=False)
    payload = {"data": sample.flatten().tolist()}
    infer_url = f"{SERVER_URL}/infer"

    # Check readiness and warm up once; neither request counts toward the 100 timings.
    try:
        health = requests.get(f"{SERVER_URL}/", timeout=5)
        health.raise_for_status()
        if health.json().get("status") != "healthy":
            raise SystemExit(f"Server at {SERVER_URL} did not report healthy.")
        warmup = requests.post(infer_url, json=payload, timeout=10)
        warmup.raise_for_status()
        # Requests encodes json=payload as bytes. Count the body only, not HTTP headers.
        payload_bytes = len(warmup.request.body)

        rtt_times = []
        server_times = []
        for _ in range(100):
            start = perf_counter()
            response = requests.post(infer_url, json=payload, timeout=10)
            elapsed_ms = (perf_counter() - start) * 1000
            response.raise_for_status()
            result = response.json()
            rtt_times.append(elapsed_ms)
            server_times.append(result["server_latency_ms"])
    except requests.RequestException as error:
        raise SystemExit(f"Benchmark failed. Check the Step 4a server at {SERVER_URL}: {error}") from error

    # RTT minus server compute includes HTTP handling and serialization, not just transit.
    rtt_ms = np.asarray(rtt_times)
    server_ms = np.asarray(server_times)
    print("\nCLOUD INFERENCE RESULTS")
    print("=" * 48)
    print(f"Timed requests  : {len(rtt_ms)} (after 1 warm-up)")
    print(f"Prediction      : {result['prediction']}")
    print(f"Confidence      : {result['confidence']:.4f}")
    print(f"Server mean     : {server_ms.mean():.4f} ms")
    print(f"Server p95      : {np.percentile(server_ms, 95):.4f} ms")
    print(f"RTT mean        : {rtt_ms.mean():.4f} ms")
    print(f"RTT p95         : {np.percentile(rtt_ms, 95):.4f} ms")
    print(f"Mean overhead   : {(rtt_ms - server_ms).mean():.4f} ms (RTT minus server compute)")
    print(f"JSON payload    : {payload_bytes:,} bytes/request")
    print(f"Upload at 30 FPS: {payload_bytes * 30 / 1024:.2f} KiB/s (request bodies only)")


if __name__ == "__main__":
    main()