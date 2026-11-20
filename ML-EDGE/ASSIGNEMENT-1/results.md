# Lab Experiment 1: Experimental Results

## Model training and conversion

| Metric | Recorded result | Source |
|---|---:|---|
| Keras model test accuracy (MNIST test set) | 98.04% | Step 1 |
| Original Keras archive size | 637,645 bytes (622.7 KiB) | Step 2 |
| Dynamic-range quantized TFLite model size | 55,960 bytes (54.6 KiB) | Step 2 |
| Compression ratio (Keras archive / TFLite file) | 11.39x | Step 2 |
| TFLite model test-set accuracy | Not measured | Single-image inference only |

## Inference benchmarks

The standalone runs (Steps 3 and 4b) and the unified run (Step 5) used the same saved MNIST image (true label: 7). Each path used one untimed warm-up and 100 timed inferences. The cloud endpoint was on localhost, not a remote host.

| Metric | Standalone edge (Step 3) | Standalone cloud (Step 4b) | Unified edge (Step 5) | Unified cloud (Step 5) |
|---|---:|---:|---:|---:|
| Mean latency (ms) | 0.0082 | 17.0728 | 0.0079 | 18.1786 |
| Median latency (ms) | 0.0078 | Not measured | 0.0078 | 15.9766 |
| p95 latency (ms) | 0.0097 | 30.1196 | 0.0086 | 29.9176 |
| Maximum latency (ms) | 0.0105 | Not measured | Not measured | Not measured |
| Throughput (inferences/s) | 122,504 | Not measured | 126,454.2 | 55.0 |
| Predicted digit | 7 | 7 | 7 | 7 |

Edge latency includes copying the input tensor and executing the model. Cloud latency in this table is the full client round-trip time (RTT). "Not measured" indicates that the run did not report the statistic; no value has been inferred from a different run.

## Server compute, upload, and speedup

| Metric | Recorded result | Source |
|---|---:|---|
| Server inference latency, mean | 0.0701 ms | Step 4b |
| Server inference latency, p95 | 0.1052 ms | Step 4b |
| Mean request overhead (RTT minus server inference) | 17.0027 ms | Step 4b, paired requests |
| Network transit time alone | Not measured | Not isolated by the benchmark |
| Upload per inference (edge / cloud) | 0 / 5,687 bytes | Step 5; cloud value is the serialized JSON body |
| Cloud upload required at 30 FPS | 166.61 KiB/s | Steps 4b and 5; calculated from request-body bytes, not measured link traffic |
| Edge speedup (mean cloud RTT / mean edge latency) | 2,298.8x | Step 5, calculated before rounding the means |

The measured request overhead includes HTTP handling and serialization as well as local transport. It is not a measurement of network transit alone.