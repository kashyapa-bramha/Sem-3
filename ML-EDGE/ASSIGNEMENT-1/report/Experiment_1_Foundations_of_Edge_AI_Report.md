# 1. Title

## Build, Convert, Deploy, Compare: The Foundations of Edge AI

**Lab Experiment 1 (EC-1), Module 1: Foundations of Edge AI**
**Course:** AIML ZG535 / AML ZC421, Machine Learning on Edge
**Institution:** Birla Institute of Technology & Science, Pilani, Work Integrated Learning Programmes Division

### Abstract

This report compares edge and simulated-cloud inference for a compact convolutional neural network (CNN) trained on MNIST. The model has 50,618 parameters and reached 98.04% accuracy on the 10,000-image test set. It was converted to TensorFlow Lite with dynamic-range quantization, reducing the stored model from 637,645 bytes to 55,960 bytes (11.39 times smaller). The same saved test image was then classified 100 times locally with the TensorFlow Lite interpreter and 100 times through a FastAPI endpoint on localhost.

In the unified comparison run, mean local inference latency was 0.0094 ms and mean client round-trip time (RTT) was 16.4877 ms, an edge speedup of 1,748.4 times. In the separate cloud run, server-side inference accounted for 0.0711 ms of an 18.1398 ms mean RTT; the remaining 18.0688 ms was request overhead (serialization, HTTP handling, and local transport). Each cloud request carried 5,687 bytes of JSON, equivalent to 166.61 KiB/s at 30 requests per second, whereas local inference transmitted no image data. Both paths predicted digit 7 for the benchmark image. Because the server ran on the same laptop, the experiment demonstrates the cost of a request path but does not measure wide-area network (WAN) latency.

# 2. Objective

The objective was to execute the full Edge AI workflow on a single workstation: train a lightweight model, convert it for deployment, run it locally, serve the same model through a REST API, and compare the two deployments using measured evidence. The experiment measured:

1. Model accuracy and stored model size before and after conversion.
2. Local inference latency and throughput.
3. Server-side inference time and client round-trip time (RTT) for the cloud path.
4. The size of the serialized request body and the resulting upload rate at 30 requests per second.

The measurements support a discussion of three of the four drivers of Edge AI identified in the handout: latency, bandwidth, and privacy. Connectivity is addressed in the reflection analysis only.

# 3. Learning Outcomes

| Learning outcome (from the assignment)                       | Where it is demonstrated     | Evidence type                                         |
| ------------------------------------------------------------ | ---------------------------- | ----------------------------------------------------- |
| Build and train a lightweight CNN on MNIST                   | Sections 6 and 7             | Measured test accuracy                                |
| Convert a trained model to TensorFlow Lite with quantization | Section 8                    | Measured file sizes                                   |
| Run inference with the TensorFlow Lite interpreter           | Sections 9 and 12.2          | Measured latency statistics                           |
| Deploy the same model as a REST API with FastAPI             | Sections 10 and 12.3         | Measured server and RTT statistics                    |
| Measure and compare local latency with cloud RTT             | Sections 12.4 and 13         | Measured and derived values                           |
| Calculate bandwidth cost of sending image data               | Sections 14 and 15 (R3)      | Measured payload, derived rates, scenario calculation |
| Identify privacy risks of cloud-only inference               | Section 15 (R2)              | Reasoned analysis, not measured                       |
| Explain the Edge-Cloud continuum from measured evidence      | Sections 13, 15 (R5), and 16 | Interpretation                                        |

# 4. Introduction and Background

## 4.1 Motivation

The assignment opens with a factory camera that must classify one part every 100 ms. A cloud pipeline that serializes each image, transmits it, waits for inference, and receives the answer can miss this budget even when the model is fast, because every stage outside the model adds delay. The narrative figures in the handout (350 ms for a cloud cycle and 8 ms for an embedded device) are illustrative. They were not measured in this experiment and are not used as results.

## 4.2 The Four Drivers of Edge AI

Edge AI places inference close to the data source [1], [2]. The handout identifies four reasons that cloud-only inference is insufficient for some applications.

| Driver       | Relevance to this experiment                                                                              |
| ------------ | --------------------------------------------------------------------------------------------------------- |
| Latency      | Local inference time was compared with client RTT.                                                        |
| Bandwidth    | The bytes sent per cloud request were measured.                                                           |
| Privacy      | The pixel values that leave the client in the cloud design were identified.                               |
| Connectivity | Considered in the reflection analysis (R4). A paired server-stopped test was not recorded (Section 12.8). |

## 4.3 TensorFlow Lite and Quantization

TensorFlow Lite (TFLite) converts a trained Keras model into a compact FlatBuffer file that is executed by a lightweight interpreter without the full TensorFlow training framework [3]. Quantization stores numbers with fewer bits. In **dynamic-range quantization**, eligible weights are stored as 8-bit integers instead of 32-bit floating-point values, so the weights occupy roughly one quarter of their original space. No calibration dataset is needed, and the model's input and output remain floating-point. Quantization is an approximation and can change numerical results.

## 4.4 Simulating a Cloud API Locally

A cloud AI service exposes a model through an HTTP endpoint. In this experiment, a FastAPI [4] server on the same laptop simulates that architecture. The client must still encode the image as JSON, issue an HTTP POST, wait for the server, and parse the reply. The experiment therefore captures the cost of the request path, but it excludes physical distance, WAN routing, congestion, and remote-hosting delays. The measured cloud latency is accordingly a lower-bound illustration for a real remote deployment, not a measurement of one.

# 5. Experimental Environment

## 5.1 Hardware and Software

| Component                      | Value                                                                                |
| ------------------------------ | ------------------------------------------------------------------------------------ |
| Operating system               | Microsoft Windows 11 Enterprise, version 10.0.26200, 64-bit                          |
| Python                         | 3.12.10 (64-bit), in a dedicated virtual environment (`.venv`) for this assignment |
| TensorFlow                     | 2.21.0                                                                               |
| NumPy                          | 2.5.5                                                                                |
| FastAPI                        | 0.165.0                                                                              |
| Uvicorn                        | 0.54.0                                                                               |
| Requests                       | 2.34.2                                                                               |
| psutil                         | 7.2.2                                                                                |
| Execution device               | Laptop CPU, used for both edge and server-side inference                             |
| CPU model and installed memory | Not recorded                                                                         |

All versions were reported by an import check executed with the assignment's interpreter. The TensorFlow Lite runtime logged the creation of an XNNPACK CPU delegate. The laptop CPU stands in for a constrained edge device; no embedded hardware was used.

## 5.2 Implementation Files

| File                     | Step | Purpose                                                            |
| ------------------------ | ---- | ------------------------------------------------------------------ |
| `build_model.py`       | 1    | Train and evaluate the CNN; save the model and one benchmark image |
| `convert_to_tflite.py` | 2    | Convert the model with dynamic-range quantization                  |
| `local_inference.py`   | 3    | Benchmark local TFLite inference                                   |
| `cloud_server.py`      | 4a   | Serve the TFLite model through FastAPI on port 8000                |
| `cloud_client.py`      | 4b   | Benchmark the endpoint from the client                             |
| `compare_results.py`   | 5    | Run both benchmarks back to back and print a side-by-side table    |

# 6. CNN Model Architecture

The network follows the architecture specified in the assignment. Each convolutional filter learns a small 3 x 3 pattern and slides across the image, so the same pattern can be detected at different positions. Pooling retains the strongest local response while reducing computation. The dense layers combine the resulting features and output one probability for each digit.

| Layer                                           | Output shape per image |       Parameters | Function                              |
| ----------------------------------------------- | ---------------------: | ---------------: | ------------------------------------- |
| Input                                           |            28 x 28 x 1 |                0 | One grayscale channel                 |
| Conv2D: 8 filters, 3 x 3, ReLU,`same` padding |            28 x 28 x 8 |               80 | Learns local stroke and edge patterns |
| MaxPooling2D: 2 x 2                             |            14 x 14 x 8 |                0 | Downsamples feature maps              |
| Flatten                                         |                  1,568 |                0 | Converts feature maps to a vector     |
| Dense: 32 units, ReLU                           |                     32 |           50,208 | Combines features                     |
| Dense: 10 units, softmax                        |                     10 |              330 | Produces class probabilities          |
| **Total**                                 |                        | **50,618** |                                       |

Parameter counts follow from the layer definitions: convolution $3 \times 3 \times 1 \times 8 + 8 = 80$; first dense layer $1568 \times 32 + 32 = 50{,}208$; output layer $32 \times 10 + 10 = 330$. Keras reported the same total of 50,618 parameters for the implemented model when its construction was checked. This differs from the handout's prose figure of approximately 5,900 parameters. The report uses the verified value for the specified layers.

# 7. Model Training and Evaluation

## 7.1 Data and Preprocessing

MNIST contains 70,000 labeled 28 x 28 grayscale images of handwritten digits: 60,000 for training and 10,000 for testing. Pixel values are integers from 0 to 255. Each image was reshaped to 28 x 28 x 1, converted to `float32`, and divided by 255, so all pixels lie in [0, 1]. The same preprocessing was applied to the training and test sets [5].

## 7.2 Training Configuration

| Setting                             | Value                            |
| ----------------------------------- | -------------------------------- |
| Optimizer                           | Adam [6]                         |
| Loss function                       | Sparse categorical cross-entropy |
| Metric                              | Accuracy                         |
| Epochs                              | 5                                |
| Batch size                          | 64                               |
| Validation split                    | 10% of the training set          |
| Training / validation / test images | 54,000 / 6,000 / 10,000          |

With 54,000 training images and a batch size of 64, each epoch comprises $\lceil 54{,}000 / 64 \rceil = 844$ weight updates. The test set was not used in training or validation.

## 7.3 Evaluation and Saved Artifacts

The trained model was evaluated on the 10,000 test images. The first test image was saved to `test_sample.npy` with shape (1, 28, 28, 1), so every later benchmark used the same input. Its true label is 7.

```text
Test accuracy: 0.9804 (98.04%)
Model saved to ...\ASSIGNEMENT-1\mnist_model.keras
Test sample saved to ...\ASSIGNEMENT-1\test_sample.npy (true label: 7)
```

*Listing 1. Step 1 output (directory prefix abbreviated).*

The Keras model therefore classified 9,804 of the 10,000 test images correctly, corresponding to 98.04% accuracy. The handout's figure of approximately 98% is an illustrative reference; this value is the measured result. Per-epoch training and validation curves, test loss, and final validation accuracy were not part of the recorded outputs and are reported as not available.

# 8. TensorFlow Lite Conversion and Quantization

`convert_to_tflite.py` loads the saved Keras model, creates a `TFLiteConverter`, enables the default optimization (`tf.lite.Optimize.DEFAULT`), and writes the FlatBuffer to `model.tflite`. With no representative dataset, this applies dynamic-range quantization as specified in the assignment. The original model file is not modified.

```text
Conversion complete: ...\ASSIGNEMENT-1\model.tflite
Original Keras model: 637,645 bytes (622.7 KiB)
TFLite model: 55,960 bytes (54.6 KiB)
Compression ratio (Keras / TFLite): 11.39x
```

*Listing 2. Step 2 output (directory prefix abbreviated).*

| Measurement                       |   Bytes |    KiB | Decimal MB | Type             |
| --------------------------------- | ------: | -----: | ---------: | ---------------- |
| Original Keras model (`.keras`) | 637,645 |  622.7 |   0.637645 | Measured         |
| Quantized TFLite model            |  55,960 |   54.6 |   0.055960 | Measured         |
| Compression ratio                 |         |  11.39 |            | Derived, Eq. (1) |
| Size reduction                    |         | 91.22% |            | Derived, Eq. (2) |

$$
\text{Compression ratio} = \frac{\text{size of Keras file}}{\text{size of TFLite file}} = \frac{637{,}645}{55{,}960} = 11.39 \tag{1}
$$

$$
\text{Size reduction} = \left(1 - \frac{55{,}960}{637{,}645}\right) \times 100\% = 91.22\% \tag{2}
$$

The handout's checklist names a SavedModel directory, but the implemented workflow saved a single `.keras` file, and the report uses that file. The ratio compares complete files and should not be read as a pure measure of weight quantization. A Keras archive can include configuration and optimizer state in addition to the inference weights, and the experiment did not convert an unquantized TFLite model for comparison. As a rough check, storing the 50,618 parameters at one byte each requires about 50.6 kB, the same order of magnitude as the 55,960-byte TFLite file. This is an interpretation; the experiment did not inspect the converted tensors. The accuracy of the converted model on the full test set was not measured.

# 9. Edge Inference Implementation

`local_inference.py` represents the edge device. It uses the TFLite interpreter bundled with TensorFlow, as the handout's import fallback allows. The script:

1. Loads `test_sample.npy`, and stops with an explicit error if it is missing, so the benchmark never silently switches to random data.
2. Loads `model.tflite` and allocates the interpreter's tensors once, outside the timed loop.
3. Performs one untimed warm-up inference to exclude one-time runtime setup.
4. Runs 100 timed inferences. Each one starts a high-resolution timer (`time.perf_counter`), copies the image into the input tensor, invokes the interpreter, and stops the timer.
5. Reads the output scores after timing and reports the predicted class, confidence, and latency statistics.

No data leaves the process, and no network connection is required.

```text
EDGE INFERENCE RESULTS
================================================
Input shape     : [1, 28, 28, 1]
Output shape    : [1, 10]
Timed inferences: 100 (after 1 warm-up)
Predicted class : 7
Confidence      : 0.9999
Mean latency    : 0.0116 ms
Median latency  : 0.0103 ms
p95 latency     : 0.0163 ms
Maximum latency : 0.0970 ms
Throughput      : 86,007 inferences/sec
```

*Listing 3. Step 3 edge benchmark output.*

# 10. Cloud Inference Implementation

## 10.1 Server (`cloud_server.py`)

The server follows the API in the assignment. Its startup routine loads `model.tflite` and allocates the interpreter's tensors before the first request is accepted. It exposes `GET /` as a health check, returning `{"status": "healthy"}`, and `POST /infer`. The inference endpoint accepts JSON of the form `{"data": [...]}` containing the 784 flattened pixel values. The server converts the values to `float32`, reshapes them to (1, 28, 28, 1), and times only the tensor copy and interpreter invocation. It returns the prediction, rounded confidence, server latency, and `payload_bytes`.

The server adds two safeguards beyond the handout: it rejects a request that is not exactly 784 values in [0, 1], and it serializes access to the single shared interpreter with a lock. Inference errors return an HTTP 500 response without exposing internal details. The server listens on port 8000.

## 10.2 Client (`cloud_client.py`)

The client loads the same `test_sample.npy`, flattens it into a list, and builds the JSON payload. It checks the health endpoint, issues one untimed warm-up request, and then sends 100 sequential timed POST requests to `http://127.0.0.1:8000/infer`. The client timer covers the POST and receipt of the response, which includes JSON encoding by the Requests library. The JSON body size was taken from the request body that the library actually sent, excluding HTTP headers. Server latency is read from each response.

## 10.3 Two Payload Sizes

The server's `payload_bytes` field reports the decoded image: $784 \times 4 = 3{,}136$ bytes of `float32` values. The serialized JSON request body measured by the client was 5,687 bytes. The report uses the JSON size as the bandwidth per cloud request, because it is the quantity transmitted.

## 10.4 Cloud Simulation

The server and client ran on the same laptop. The measurements therefore include process-to-process HTTP communication, but not wide-area transit.

```text
CLOUD INFERENCE RESULTS
================================================
Timed requests  : 100 (after 1 warm-up)
Prediction      : 7
Confidence      : 0.9999
Server mean     : 0.0711 ms
Server p95      : 0.1125 ms
RTT mean        : 18.1398 ms
RTT p95         : 30.9058 ms
Mean overhead   : 18.0688 ms (RTT minus server compute)
JSON payload    : 5,687 bytes/request
Upload at 30 FPS: 166.61 KiB/s (request bodies only)
```

*Listing 4. Step 4b cloud client output.*

# 11. Experimental Methodology

## 11.1 Common Protocol

Every benchmark used the same saved image, the same `model.tflite`, one untimed warm-up, and 100 timed sequential inferences. The warm-up excludes one-time initialization. The run-to-run difference in the same quantity is itself informative, so the results are labeled by the step that produced them (Step 3, Step 4b, or Step 5).

## 11.2 Timed Intervals

| Quantity                 | Timer starts                                                   | Timer stops                    | Excludes                                                            |
| ------------------------ | -------------------------------------------------------------- | ------------------------------ | ------------------------------------------------------------------- |
| Edge latency             | Before copying the image into the input tensor                 | After the interpreter finishes | Model loading, tensor allocation, warm-up, output reading, printing |
| Server inference latency | Before copying the decoded image into the server's interpreter | After the interpreter finishes | JSON parsing, array reshaping, response creation                    |
| Client RTT               | Before the HTTP POST (including JSON encoding)                 | After the response is received | Health check, warm-up, parsing of the response JSON                 |

The server inference interval corresponds to the core operation timed locally. RTT additionally includes everything else in the request path. RTT minus server latency is therefore request overhead (serialization, HTTP handling, local transport, and server-side parsing), not pure network transit time.

## 11.3 Statistics and Derived Metrics

Let $t_1, \dots, t_{100}$ be the 100 timed latencies in milliseconds. The mean is

$$
\bar{t} = \frac{1}{100}\sum_{i=1}^{100} t_i \tag{3}
$$

The median is the middle value of the sorted measurements. The 95th percentile (p95) is the latency not exceeded by 95% of the measurements; NumPy's default linear interpolation was used. The maximum is the slowest observed run, not a guaranteed bound. Throughput for one sequential request stream is

$$
T = \frac{1000}{\bar{t}} \ \text{inferences per second} \tag{4}
$$

with $\bar{t}$ in milliseconds. It excludes camera capture and other pipeline stages. The comparison metrics are

$$
\text{Request overhead} = \overline{\text{RTT}} - \overline{t_{\text{server}}} \tag{5}
$$

$$
\text{Edge advantage} = \overline{\text{RTT}} - \bar{t}_{\text{edge}} \tag{6}
$$

$$
\text{Edge speedup} = \frac{\overline{\text{RTT}}}{\bar{t}_{\text{edge}}} \tag{7}
$$

$$
B_{30} = \frac{P \times 30}{1024} \ \text{KiB/s} \tag{8}
$$

where $P$ is the serialized request-body size in bytes and 30 is the number of requests per second assumed by the handout.

## 11.4 Classification of Results

Measured values are those printed by the scripts. **Derived** values are computed from measurements using the equations above. **Scenario calculations** use inputs specified in the assignment and are not measurements of this experiment. **Interpretation** explains what a result may imply. Where a printed statistic was not available, it is marked as "Not measured".

# 12. Results

## 12.1 Model Results

| Metric                               |                     Value | Type          |
| ------------------------------------ | ------------------------: | ------------- |
| Keras test accuracy (10,000 images)  |                    98.04% | Measured      |
| Original Keras file                  | 637,645 bytes (622.7 KiB) | Measured      |
| TFLite file                          |   55,960 bytes (54.6 KiB) | Measured      |
| Compression ratio                    |                     11.39 | Derived       |
| TFLite accuracy on the full test set |              Not measured | Not available |

## 12.2 Edge Inference (Step 3)

| Metric                         |               Value |
| ------------------------------ | ------------------: |
| Predicted digit (true label 7) |                   7 |
| Confidence                     |              0.9999 |
| Mean latency                   |          0.0116 ms |
| Median latency                 |           0.0103 ms |
| p95 latency                    |           0.0163 ms |
| Maximum latency                |           0.0970 ms |
| Throughput                     | 86,007 inferences/s |
| Data sent per inference        |             0 bytes |

## 12.3 Cloud Inference (Step 4b)

| Metric                                        |                   Value | Type                            |
| --------------------------------------------- | ----------------------: | ------------------------------- |
| Predicted digit                               |                       7 | Measured                        |
| Confidence                                    |                  0.9999 | Measured                        |
| Server inference latency, mean / p95          |   0.0711 ms / 0.1125 ms | Measured                        |
| Client RTT, mean / p95                        | 18.1398 ms / 30.9058 ms | Measured                        |
| Request overhead (RTT minus server inference) |              18.0688 ms | Measured by the script, Eq. (5) |
| Serialized JSON request body                  |             5,687 bytes | Measured                        |
| Upload at 30 requests/s                       |            166.61 KiB/s | Derived, Eq. (8)                |
| RTT median, RTT maximum, throughput           |            Not measured | Not reported by this run        |
| Network transit time alone                    |            Not measured | Not isolated by the benchmark   |

## 12.4 Unified Comparison (Step 5)

The unified script ran both benchmarks back to back, with one untimed warm-up and 100 timed inferences per path.

```text
METRIC                                  EDGE     CLOUD (RTT)
------------------------------------------------------------
Mean latency (ms)                     0.0094         16.4877
Median latency (ms)                   0.0091         15.7820
p95 latency (ms)                      0.0119         28.7203
Throughput (inf/s)                 106,044.2            60.7
Upload per inference (B)                   0           5,687
Network required                          No             Yes
Prediction                                 7               7

Edge speedup (mean RTT / mean edge): 1,748.4x
Cloud upload at 30 FPS: 166.61 KiB/s (request bodies only)
```

*Listing 5. Step 5 unified comparison output.*

## 12.5 Assignment Results Table

The assignment permits values from Step 5 or from Steps 3 and 4. The table uses the paired Step 5 run for edge and RTT, and Step 4b for server compute, which Step 5 does not print.

| Measurement scenario                   | Value (ms) | Network?   | Source                       |
| -------------------------------------- | ---------: | ---------- | ---------------------------- |
| Local TFLite inference (edge), mean    |     0.0094 | No         | Step 5                       |
| Cloud server compute only, mean        |     0.0711 | Server     | Step 4b                      |
| Cloud total round-trip (RTT), mean     |    16.4877 | Yes (full) | Step 5                       |
| Edge advantage (RTT minus local), mean |    16.4783 | n/a        | Derived from Step 5, Eq. (6) |
| Bandwidth per request (bytes)          |      5,687 | n/a        | Step 4b and Step 5           |

## 12.6 Quick-Reference Checklist

| Item                                             | Value                     |
| ------------------------------------------------ | ------------------------- |
| TFLite model file size                           | 54.6 KiB (55,960 bytes)   |
| Keras model file size (the saved original model) | 622.7 KiB (637,645 bytes) |
| Compression ratio (Keras file / TFLite)          | 11.39                     |
| Model test-set accuracy                          | 98.04%                    |
| Edge speedup factor (cloud RTT mean / edge mean) | 1,748.4 (Step 5)          |
| Bandwidth per request                            | 5,687 bytes               |
| Equivalent bandwidth at 30 FPS                   | 166.61 KiB/s              |
| Edge prediction matches cloud prediction         | Yes (digit 7 in all runs) |

## 12.7 Consistency Checks

- All four inference reports used the same image and produced class 7, matching its true label. Edge and cloud confidence were both 0.9999 in the standalone runs.
- The serialized payload was 5,687 bytes in both the Step 4b and Step 5 runs, and the 30 FPS rate of 166.61 KiB/s agrees with $5687 \times 30 / 1024$.
- The mean edge latency was 0.0116 ms in Step 3 and 0.0094 ms in Step 5, a difference of 0.0022 ms or 19% of the Step 3 value. Mean cloud RTT was 18.1398 ms in Step 4b and 16.4877 ms in Step 5, a difference of 1.6521 ms or 9.1% of the Step 4b value. Latency varies between runs, so the report does not mix statistics from different runs inside one derived calculation.
- Printed throughput can differ slightly from $1000/\bar{t}$ calculated with rounded means. The scripts compute from unrounded values: for example, $1000/0.0094 = 106{,}383$, compared with the printed 106,044.2.

## 12.8 Measurements Not Available

| Item                                                                                                 | Status                                                  |
| ---------------------------------------------------------------------------------------------------- | ------------------------------------------------------- |
| Per-epoch training and validation loss and accuracy; final validation accuracy; test loss            | Not available in the recorded outputs                   |
| Accuracy of the quantized TFLite model on the test set                                               | Not measured                                            |
| Unquantized (FP32) TFLite size, latency, and accuracy                                                | Not measured                                            |
| Cloud RTT median and maximum latency (Step 4b); maximum latency in Step 5                            | Not measured                                            |
| Cloud throughput in Step 4b                                                                          | Not measured (a derived estimate appears in Section 13) |
| Pure network transit time; WAN latency                                                               | Not measured                                            |
| CPU model and installed memory                                                                       | Not recorded                                            |
| Cloud client output with the server stopped, and local inference during the same period              | Not recorded                                            |
| Concurrent-request behavior, full-integer quantization, and automatic fallback (optional extensions) | Not performed                                           |

# 13. Edge vs Cloud Comparison

## 13.1 Same-Run Comparison (Step 5)

| Metric                    |      Edge | Cloud (client RTT) | Difference or ratio | Type                        |
| ------------------------- | --------: | -----------------: | ------------------: | --------------------------- |
| Mean latency (ms)         |    0.0094 |            16.4877 |          16.4783 ms | Derived, Eq. (6)            |
| Median latency (ms)       |    0.0091 |            15.7820 |          15.7729 ms | Derived                     |
| p95 latency (ms)          |    0.0119 |            28.7203 |          28.7084 ms | Derived                     |
| Throughput (inferences/s) | 106,044.2 |               60.7 |                     | Measured                    |
| Data sent per inference   |   0 bytes |        5,687 bytes |         5,687 bytes | Measured                    |
| Network required          |        No |                Yes |                     | Observed from design        |
| Prediction                |         7 |                  7 |               Match | Measured                    |
| Edge speedup              |           |                    |       1,748.4 times | Measured by script, Eq. (7) |

The printed speedup of 1,748.4 uses the unrounded means. Dividing the rounded means gives 16.4877/0.0094 = 1,754.0. The difference arises because the edge mean is reported to only two significant figures.

## 13.2 Cloud Request Breakdown (Step 4b)

| Component                                     |                 Mean | Share of mean RTT |
| --------------------------------------------- | -------------------: | ----------------: |
| Server inference (timed on the server)        |            0.0711 ms |             0.39% |
| Request overhead (RTT minus server inference) |           18.0688 ms |            99.61% |
| **Client RTT**                          | **18.1398 ms** |              100% |

Shares are derived from Eq. (5). The script computes the overhead from unrounded values; the difference of the rounded means is 18.0687 ms, which differs only by rounding. Server inference was thus a very small part of the measured cloud latency, and the dominant cost was the request path. For the same run, the edge advantage is $18.1398 - 0.0116 = 18.1282$ ms. The derived speedup is $18.1398/0.0116 = 1{,}564$; because the edge mean has only three significant figures, this ratio is uncertain by roughly 0.4% from rounding alone. The derived cloud throughput is $1000/18.1398 = 55.1$ inferences per second, compared with 60.7 printed in the separate Step 5 run.

## 13.3 Observations

1. **Compute latency differs from end-to-end latency.** The server's own timer reported a mean of 0.0711 ms, while the client waited 18.1398 ms. Using server latency alone would understate the cloud path by more than two orders of magnitude.
2. **The localhost cloud path was slower than local inference by a large factor.** Localhost removes physical distance, but not JSON encoding, the HTTP stack, process boundaries, or response handling. The model computation is only microseconds, so these fixed costs dominate.
3. **Server-side inference was not faster than local inference.** The server-side mean (0.0711 ms) exceeded the Step 3 edge mean (0.0116 ms). The server's timed region covers the same operations, but it runs in a separate server process under different conditions. The experiment did not isolate the cause.
4. **Latency variability was larger in the cloud path.** The cloud RTT p95 was 30.9058 ms in Step 4b and 28.7203 ms in Step 5, and the Step 4b value was approximately 1.7 times its mean. By contrast, edge p95 values (0.0163 ms and 0.0119 ms) stayed within about 1.4 times the edge means, although the Step 3 maximum of 0.0970 ms was more than 8 times its mean.
5. **Edge inference transmitted no image data.** Its zero-byte transmission follows from the design; no network traffic was measured.

# 14. Bandwidth Calculations

## 14.1 Measured Request Size

The measured JSON request body was $P = 5{,}687$ bytes, or 5.55 KiB. It is 1.81 times the 3,136-byte decoded image, because JSON represents each pixel as text and adds punctuation and the `data` field. The size excludes HTTP headers and the response.

## 14.2 Upload Rate at 30 Requests per Second

$$
B_{30} = \frac{5687 \times 30}{1024} = \frac{170{,}610\ \text{B/s}}{1024} = 166.61\ \text{KiB/s} \tag{9}
$$

Equivalently, 170,610 B/s is 0.163 MiB/s or about 1.36 Mb/s (decimal). This is derived from the measured request size and counts request bodies only. It is not the measured traffic on a network link.

## 14.3 Edge Comparison

Local inference sent 0 bytes of image data per inference, so its upload requirement is 0 KiB/s. The saving is 166.61 KiB/s per image stream at 30 inferences per second, relative to this MNIST cloud request. The CCTV scenario in the assignment is addressed in R3, because it uses scenario inputs rather than these measurements.

# 15. Reflection Questions R1-R5

## R1. Latency and Real-Time Constraints

**Requirements (theoretical).** The factory runs at 600 parts per minute, which gives 10 parts per second and a decision budget of 100 ms per part. The automotive pedestrian-detection requirement is under 10 ms.

**Measurements (Step 5, mean unless stated).** Local inference took 0.0094 ms. Cloud RTT was 16.4877 ms, with a median of 15.7820 ms and a p95 of 28.7203 ms.

| Requirement                       | Edge (mean 0.0094 ms)      | Cloud RTT (mean 16.4877 ms)     |
| --------------------------------- | -------------------------- | ------------------------------- |
| Factory: 100 ms per part          | Uses 0.0094% of the budget | Uses 16.49% of the budget       |
| Pedestrian detection: under 10 ms | Uses 0.094% of the budget  | Exceeds the budget by 6.4877 ms |

**Factory line.** Both systems met the 100 ms budget on average. Under sequential requests, the measured cloud throughput of 60.7 inferences per second also exceeds the 10 parts per second required. The cloud p95 of 28.7203 ms is below 100 ms, so about 95% of the measured requests completed within 29 ms. The maximum cloud RTT was not measured, so the experiment does not establish that every request met the deadline. The 350 ms cloud cycle in the handout's story was not reproduced: the localhost server removes WAN delay. The edge mean applies to the MNIST model on a laptop CPU, not to a camera processor.

**Automotive.** The cloud path exceeded 10 ms on average, and its median (15.7820 ms) was also above 10 ms. This was a localhost measurement, so a real remote service would add WAN latency. The experiment therefore implies that a cloud round trip is a poor fit for a hard sub-10 ms safety function, and that such a function should run on the vehicle. It does not prove that any automotive detector meets the deadline, since the tested model classifies digits and the experiment excludes camera capture, the actual detector, vehicle hardware, and operating conditions. A safety claim would require end-to-end testing.

## R2. Privacy and Data Sovereignty

In the cloud design, raw pixel values leave the client, whereas the edge design sent no image data. Because the server ran on the same laptop, no data left the machine in this experiment. The table identifies applications where uploading raw data would create privacy risk in a real deployment. Regulatory references are summarized at a high level and do not constitute legal advice.

| Application and regulation                                                                                                                                                                               | Sensitive data and risk of cloud processing                                                                                                                                                                         | How on-device inference reduces exposure                                                                                                                                                                             |
| -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Hospital bedside fall detection (United States).** HIPAA Privacy and Security Rules apply to identifiable patient health information held by covered entities and their business associates [8]. | Room video can identify a patient and reveal condition, treatment, or intimate care. Continuous upload exposes more information than is needed to raise an alert.                                                   | Detect falls on a bedside device, discard frames after processing, and send only an alert and location. The cloud need not receive or retain the video. The alert itself remains protected information.              |
| **Always-on smart speaker in an EU household.** GDPR Article 25 requires data protection by design and by default [7].                                                                             | Audio captured before activation can contain private conversations and bystanders who never addressed the device.                                                                                                   | Detect the wake word locally using a short buffer that is continuously overwritten, so pre-activation audio need not leave the device. Any later cloud processing of the command is a narrower, separate disclosure. |
| **Facial access control at a Singapore office.** The Personal Data Protection Act 2012 (PDPA) governs the handling of personal data, including identifiable facial and access records [9].         | Face images and linked entry logs identify employees and reveal attendance patterns. A central or overseas store increases the consequence of unauthorized access and may engage cross-border transfer obligations. | Match the face at the door against a protected on-device template, discard the captured image, and send only an allow/deny signal to the lock. Retained templates and identity-linked logs still require safeguards. |

On-device inference removes the need to transfer raw inputs for prediction. It does not eliminate regulatory obligations: diagnostic uploads, backups, or stored templates would reintroduce exposure, so the data flow of the complete system determines compliance.

## R3. Bandwidth Economics

**Scenario calculation (not measured).** The inputs are 500 cameras, 25 frames per second, and 50 KB per frame, assuming every transmitted frame is 50 KB. The handout calls this a keyframe size, so the calculation is an approximation. Decimal units are used ($1\ \text{KB} = 1000$ bytes, 1 byte = 8 bits), and protocol overhead is excluded.

**One camera, all frames.**

$$
50{,}000\ \tfrac{\text{B}}{\text{frame}} \times 25\ \tfrac{\text{frames}}{\text{s}} = 1{,}250{,}000\ \tfrac{\text{B}}{\text{s}} = 1.25\ \tfrac{\text{MB}}{\text{s}} = 10\ \tfrac{\text{Mb}}{\text{s}} \tag{10}
$$

**500 cameras, all frames.**

$$
500 \times 10\ \text{Mb/s} = 5000\ \text{Mb/s} = 5\ \text{Gb/s} \quad (625\ \text{MB/s}) \tag{11}
$$

**One camera, one-byte labels.**

$$
1\ \tfrac{\text{B}}{\text{frame}} \times 25\ \tfrac{\text{frames}}{\text{s}} = 25\ \tfrac{\text{B}}{\text{s}} = 200\ \tfrac{\text{bit}}{\text{s}} \tag{12}
$$

**500 cameras, labels only.**

$$
500 \times 200\ \text{bit/s} = 100{,}000\ \text{bit/s} = 100\ \text{kb/s} \quad (12.5\ \text{kB/s}) \tag{13}
$$

**Saving and reduction.**

$$
\text{Saving} = 5\ \text{Gb/s} - 0.0001\ \text{Gb/s} = 4.9999\ \text{Gb/s} \tag{14}
$$

$$
\text{Reduction} = \frac{50{,}000 - 1}{50{,}000} \times 100\% = 99.998\% \tag{15}
$$

| Quantity                             | All frames | Labels only |
| ------------------------------------ | ---------: | ----------: |
| One camera                           |    10 Mb/s |   200 bit/s |
| 500 cameras                          |     5 Gb/s |    100 kb/s |
| Volume per day at the sustained rate |      54 TB |     1.08 GB |

**Implications.** Cloud-only inference requires the network and central service to sustain a 5 Gb/s video payload, with extra capacity for protocol overhead and peaks. About 54 TB per day would also be ingested if every frame is transmitted continuously. Classifying frames at the camera reduces uplink capacity, central inference load, and any central storage of raw footage. The costs move to the cameras: processing hardware, power, and model maintenance. A label cannot reconstruct footage, so where evidence is required, a practical design sends labels routinely and uploads selected event clips. The 100 kb/s figure is a payload rate, not a provisioned link rate, since per-message headers can exceed a one-byte label.

## R4. Connectivity and Resilience

**Evidence.** The cloud server was stopped at the end of the Step 4a check, and the server log shows a clean shutdown. The client output while the server was stopped, and a local run during that period, were not recorded. The behavior below is therefore expected from the implementation, not a measured result of this experiment.

| Path            | Expected behavior with the server stopped                                                                                                         | Dependency                                                              |
| --------------- | ------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------- |
| Cloud inference | The client cannot complete its health check, because no server is listening on port 8000. It stops before benchmarking and returns no prediction. | Requires a reachable endpoint and a running inference service           |
| Edge inference  | The local script loads the saved image and model and produces a prediction without contacting the server.                                         | Requires a functioning device and local model, but no network or server |

Stopping the local server simulates a service outage, not a physical network failure. The distinction matters: a healthy network cannot compensate for a stopped service, and a healthy remote server cannot serve a disconnected client. Edge inference removes both dependencies from the immediate decision, which improves resilience to external failure. The device itself can still fail, and the lab does not implement automatic failover between the two paths.

**Environments where offline operation is essential.**

1. **Underground mine.** A worker-proximity detector must issue a local warning as machinery moves between tunnels with little or no radio coverage.
2. **Aircraft in flight.** Onboard condition monitoring must keep evaluating sensor data throughout the flight and cannot depend on continuous contact with a ground service.
3. **Air-gapped industrial facility.** An inspection or anomaly-detection system must operate where external connections are deliberately prohibited, even when equipment and the internal network are healthy.

## R5. Edge-Cloud Continuum

The handout refers to a five-question decision framework from the module but does not reproduce the questions. The recommendations below apply the four constraints named in R5 (latency, bandwidth, privacy, connectivity). They are reasoned judgments, not experimental results.

| Scenario                                                        | Tier                                       | Primary driver | Rationale                                                                                                                                                                                                                           | Trade-offs of the alternative tiers                                                                                                                                                                                                  |
| --------------------------------------------------------------- | ------------------------------------------ | -------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| (a) Always-on keyword spotting on a smart speaker               | Edge                                       | Privacy        | A small model detects the wake word on the device, so audio captured before activation need not leave it, and detection works offline.                                                                                              | Fog keeps audio in the home but still moves it to another device. Cloud detection would transmit audio before the user has invoked the assistant.                                                                                    |
| (b) Monthly batch analysis of patient outcome records           | Cloud, in an approved research environment | Privacy        | Privacy determines whether and where records may be pooled. If pooling is authorized, a governed cloud environment can analyze a large cohort with access controls and audit logs, and a monthly schedule tolerates transfer delay. | Edge keeps records at their source but cannot directly produce a pooled result. Fog (a hospital-controlled server) is preferable when identifiable records must remain on site, but cross-hospital analysis then needs coordination. |
| (c) Industrial robot arm collision avoidance                    | Edge                                       | Latency        | The decision belongs in the local control loop. Sensing and issuing a stop command must not wait for a network round trip, even if the network is congested or down.                                                                | Fog helps coordinate several robots but adds a network dependency. Cloud suits incident analysis and model training afterward, but its variable round-trip time is unsuitable for the immediate stop decision.                       |
| (d) Real-time credit card fraud detection at a payment terminal | Fog (regional payment-processing node)     | Latency        | A nearby processor can score a transaction within the authorization deadline while using recent activity from many terminals.                                                                                                       | Edge is fastest but cannot see suspicious use of a card across merchants. Cloud offers broader history but adds delay and dependence on the wider network.                                                                           |

Low latency does not always imply edge placement. A robot can act on its own sensor data, whereas fraud scoring benefits from shared transaction history, which makes a nearby fog node a better compromise, provided it has fresh data and meets the authorization deadline. For (b), privacy is a placement constraint and not an inherent advantage of the cloud: if policy prohibits the proposed transfer, processing should stay under the hospital's control.

# 16. Discussion

## 16.1 Why Local Inference Was Faster

The local path copies a 3,136-byte tensor and invokes the interpreter in the same process. The cloud path additionally encodes 784 values as a 5,687-byte JSON body, passes it through the HTTP stack and operating system to another process, decodes and reshapes it, builds a JSON response, and parses it on return. The model's compute time is a negligible part of this chain: server inference accounted for 0.39% of the Step 4b mean RTT. The measured penalty is thus a property of the request path, not evidence that a remote model computes more slowly. A real remote deployment adds distance, routing, and queuing, so this experiment is best read as an illustrative lower bound on cloud-path cost for this small model.

## 16.2 Compute Latency Versus End-to-End Latency

The comparison is meaningful only when the interval is stated. Server inference latency measures the model call inside the server, RTT measures what the client experiences, and the difference (request overhead) is the cost of communication and serialization. Reporting only server latency would hide 99.61% of the observed cloud latency. Likewise, request overhead is not equal to network transit time, which was not isolated.

## 16.3 Bandwidth and Privacy Implications

A 28 x 28 grayscale image required 5,687 bytes of JSON per request, and 166.61 KiB/s at 30 requests per second. This is already larger than the decoded image because of text encoding. Larger images or streams scale this cost, as the R3 scenario illustrates, which motivates transmitting predictions instead of raw frames. The same transmitted pixels are also the privacy exposure: the cloud request carries the complete image, while the edge design requires no raw upload.

## 16.4 Where Cloud AI Remains Beneficial

The experiment does not show that the cloud is generally inferior. Cloud deployment remains appropriate for tasks that are not latency-critical, such as batch analytics (R5b), for models too large for device memory, for training and retraining, for aggregating data across many devices, and for centralized updates and monitoring. The tested model occupies only 54.6 KiB and fits easily on a device; a much larger model may not. This is reasoning, not a measured result.

## 16.5 Limitations

- **Localhost simulation.** The server ran on the same machine, so WAN latency, congestion, and remote-hosting effects were not measured.
- **Single input, single machine.** All benchmarks repeated one image on one laptop CPU. Latency was not measured across different images or on embedded hardware, and the laptop is not a constrained edge device.
- **Sequential requests.** No concurrent requests were tested, so scalability and queuing behavior are unknown.
- **Run-to-run variation.** Means differed between runs (edge 0.0116 and 0.0094 ms; cloud RTT 18.1398 and 16.4877 ms), and no repeated-run confidence intervals were computed. At microsecond scale, scheduler and call overhead are significant relative to model compute.
- **Unattributed size reduction.** The size ratio compares a Keras archive with a quantized TFLite file, and no FP32 TFLite model was produced, so the share due to quantization alone is unknown. Quantized accuracy on the full test set was not measured, so the experiment does not show that conversion preserved accuracy beyond the one benchmark image.
- **Bandwidth estimate.** The 30 FPS figure counts request bodies only and excludes HTTP headers, responses, and retransmissions.
- **Runtime deprecation.** TensorFlow 2.21.0 emits a deprecation notice for its bundled `tf.lite.Interpreter` and recommends migration to LiteRT. The experiment used the bundled interpreter, as in the handout's fallback.
- **Reflection analysis.** R2, R3, and R5 are based on reasoning and scenario inputs; R4 describes expected behavior, as noted.

# 17. Conclusion

The MNIST CNN reached 98.04% test accuracy, and dynamic-range quantization with TensorFlow Lite conversion reduced its stored size from 637,645 to 55,960 bytes (11.39 times smaller). On the same input, the edge and cloud paths both predicted digit 7. In the unified benchmark, the mean local inference latency was 0.0094 ms against a mean client RTT of 16.4877 ms through a localhost FastAPI endpoint, an edge speedup of 1,748.4 times. The separate cloud run showed that server inference accounted for only 0.39% of the RTT, and that 99.61% was request overhead. Each cloud request sent 5,687 bytes of JSON (166.61 KiB/s at 30 requests per second), whereas local inference sent no image data.

These results support the lab's central point for this model and setup: the cost of the cloud request path, not model computation, dominated cloud latency, and local inference avoided both that cost and raw-image transmission. The conclusions are limited to one image, one laptop, and a localhost server. They do not establish WAN latency, behavior under concurrency, the accuracy of the quantized model across the test set, or the suitability of any specific safety-critical system.

# 18. References

[1] Birla Institute of Technology & Science, Pilani, WILP Division, "Lab Experiment 1: Build, Convert, Deploy, Compare: The Foundations of Edge AI," AIML ZG535 / AML ZC421, Machine Learning on Edge (assignment specification).

[2] W. Shi, J. Cao, Q. Zhang, Y. Li, and L. Xu, "Edge Computing: Vision and Challenges," *IEEE Internet of Things Journal*, vol. 3, no. 5, pp. 637-646, 2016, doi: 10.1109/JIOT.2016.2579198.

[3] Google, "TensorFlow Lite Guide." https://www.tensorflow.org/lite/guide

[4] FastAPI Documentation. https://fastapi.tiangolo.com/

[5] Y. LeCun, L. Bottou, Y. Bengio, and P. Haffner, "Gradient-based learning applied to document recognition," *Proceedings of the IEEE*, vol. 86, no. 11, pp. 2278-2324, 1998.

[6] D. P. Kingma and J. Ba, "Adam: A Method for Stochastic Optimization," arXiv:1412.6980, 2014 (published at ICLR 2015).

[7] Regulation (EU) 2016/679 (General Data Protection Regulation), Article 25: Data protection by design and by default.

[8] U.S. Department of Health and Human Services, HIPAA Privacy Rule and Security Rule, 45 CFR Parts 160 and 164.

[9] Personal Data Protection Act 2012 (Singapore).

*Experimental data: terminal outputs from the author's executions of `build_model.py`, `convert_to_tflite.py`, `local_inference.py`, `cloud_client.py`, and `compare_results.py`, as reproduced in Sections 7 to 12.*
