# Lab Experiment 1: Foundations of Edge AI

Build, Convert, Deploy, Compare: the same MNIST CNN is run locally with TensorFlow Lite (edge) and behind a FastAPI endpoint on localhost (simulated cloud), and the two are compared on latency, bandwidth, and prediction.

```text
Build  ->  Convert  ->  Deploy  ->  Compare
Step 1     Step 2       Step 3 + 4   Step 5
```

## Before You Start

You need:

- Windows with PowerShell (the commands below are for PowerShell)
- Python 3.12, 64-bit. Check with `py -3.12 --version`; it prints `Python 3.12.x`.
- An internet connection. Setup downloads packages from public PyPI, and Step 1 downloads the MNIST dataset the first time it runs.

Follow the steps in order and do not skip any. Each step creates files that the next step needs.

**Run every command from the assignment folder.** Open a terminal in VS Code, then run this from the repository root (`Sem-3`):

```powershell
cd .\ML-EDGE\ASSIGNEMENT-1
```

The folder name is spelled `ASSIGNEMENT-1`. Every command below uses the virtual environment's Python (`.\.venv\Scripts\python.exe`) directly, so you never need to activate the environment.

**Messages you can ignore.** TensorFlow prints informational lines such as `oneDNN custom operations are on`, `XNNPACK delegate`, and a notice that `tf.lite.Interpreter` is deprecated. These are normal and do not affect the results.

## Setup (once)

Create the virtual environment:

```powershell
py -3.12 -m venv .venv
```

Install the dependencies:

```powershell
.\.venv\Scripts\python.exe -m pip install -r .\requirements\requirements.txt
```

Confirm the installation:

```powershell
.\.venv\Scripts\python.exe -c "import tensorflow, numpy, fastapi, uvicorn, requests, psutil; print('All dependencies installed successfully')"
```

The last line of the output must be `All dependencies installed successfully`. Continue only after you see it.

## Run the Experiment

### Step 1. Build: train the CNN

```powershell
.\.venv\Scripts\python.exe build_model.py
```

Trains for 5 epochs (batch size 64, 10% validation split) and evaluates on the test set.

**You should see:** `Test accuracy:` of about 98%, then lines saying the model and `test_sample.npy` were saved, with the sample's true label (7).
**Creates:** `mnist_model.keras` and `test_sample.npy`, the single image reused by every later benchmark.

### Step 2. Convert: Keras to TensorFlow Lite

```powershell
.\.venv\Scripts\python.exe convert_to_tflite.py
```

Applies dynamic-range quantization.

**You should see:** `Conversion complete:` followed by the original size, the TFLite size, and a `Compression ratio` of roughly 11x.
**Creates:** `model.tflite`.

### Step 3. Deploy (edge): local inference

```powershell
.\.venv\Scripts\python.exe local_inference.py
```

Runs one warm-up and 100 timed inferences on the local TFLite interpreter.

**You should see:** an `EDGE INFERENCE RESULTS` report with `Predicted class : 7` and the mean, median, p95, and maximum latency in milliseconds, plus throughput. Timings differ slightly from run to run.

### Step 4. Deploy (cloud): FastAPI server and client

This step uses two terminals. Open the second one from the VS Code **Terminal** menu, **New Terminal**. In each terminal, run `cd .\ML-EDGE\ASSIGNEMENT-1` from the repository root.

**Terminal 1: check that port 8000 is free, then start the server**

```powershell
Get-NetTCPConnection -LocalPort 8000 -State Listen -ErrorAction SilentlyContinue
```

This must print nothing. If it prints a connection, an earlier server is still running: press `Ctrl+C` in the window that runs it, then repeat the check.

```powershell
.\.venv\Scripts\python.exe cloud_server.py
```

Wait until the terminal shows `Application startup complete`. Leave this terminal running and do not type in it.

**Terminal 2: run the cloud benchmark**

```powershell
.\.venv\Scripts\python.exe cloud_client.py
```

Sends one warm-up request and 100 timed requests.

**You should see:** a `CLOUD INFERENCE RESULTS` report with `Prediction : 7`, server latency, round-trip time (RTT), request overhead, `JSON payload : 5,687 bytes/request`, and the upload rate at 30 FPS.

### Step 5. Compare: edge versus cloud

The server in Terminal 1 must still be running. In Terminal 2, run:

```powershell
.\.venv\Scripts\python.exe compare_results.py
```

Benchmarks both paths back to back.

**You should see:** a table with edge and cloud columns for mean, median, and p95 latency and throughput, `Network required` of `No` and `Yes`, a prediction of `7` in both columns, then the edge speedup and the 30 FPS upload requirement.

### Finish

Stop the server by pressing `Ctrl+C` in Terminal 1. The experiment is complete.

## Files

| File                              | Step   | Purpose                                                    |
| --------------------------------- | ------ | ---------------------------------------------------------- |
| `build_model.py`                | 1      | Train and evaluate the CNN; save the model and test image  |
| `convert_to_tflite.py`          | 2      | Convert to TensorFlow Lite with dynamic-range quantization |
| `local_inference.py`            | 3      | Edge inference benchmark                                   |
| `cloud_server.py`               | 4      | FastAPI inference server on port 8000                      |
| `cloud_client.py`               | 4      | Cloud benchmark client                                     |
| `compare_results.py`            | 5      | Side-by-side edge versus cloud comparison                  |
| `requirements/requirements.txt` | Setup  | Python dependencies                                        |
| `report/`                       | Report | Lab report                                                 |
