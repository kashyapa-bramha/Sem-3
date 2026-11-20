**BIRLA INSTITUTE OF TECHNOLOGY & SCIENCE, PILANI**

Work Integrated Learning Programmes Division

AIML ZG535 / AML ZC421 — Machine Learning on Edge

**LAB EXPERIMENT 1**

**Build → Convert → Deploy → Compare: The Foundations of Edge AI**

|                 |                                                                |
|-----------------|----------------------------------------------------------------|
| **Field**       | **Details**                                                    |
| Module          | Module 1 — Foundations of Edge AI                              |
| Lab Number      | Experiment 1 (EC-1)                                            |
| Duration        | 90 – 120 minutes                                               |
| Submission      | eLearn portal within 2 WEEKS of the session                    |
| Tools           | Python 3.9+, TensorFlow 2.x, FastAPI, uvicorn, requests, NumPy |
| Lab Type        | Build • Convert • Deploy • Measure • Compare                   |
| Primary Concept | Edge AI vs Cloud AI — latency, bandwidth, privacy              |

# 1. The Story: Why Edge AI Exists

> Imagine you are an engineer at a robotics company. Your team has
> deployed a smart camera on a factory assembly line. The camera
> captures images of every component that passes beneath it and uses a
> machine learning model to detect defects — scratches, cracks, or
> missing parts — in real time.
>
> On Day 1, the system routes every captured image to a cloud server for
> inference. The image is serialised into JSON, sent over the network,
> processed by a GPU cluster 800 km away, and the prediction is
> returned. The cycle time: 350 ms per image. The production line runs
> at 600 parts per minute — one part every 100 ms. The cloud system
> cannot keep up. Parts flash past the camera without a verdict.
> Defective components slip through undetected.
>
> On Day 2, the engineering team converts the same model to TensorFlow
> Lite and deploys it directly inside the camera's embedded processor.
> Inference now takes 8 ms. No network. No serialisation. No latency
> spikes. The line runs at full speed.
>
> This lab is that story, compressed into a single session. You will
> build the model, convert it to TFLite, run it on a simulated edge
> device, deploy the same model behind a cloud API, measure the latency
> and bandwidth difference, and reason carefully about privacy. By the
> end, the motivations for Edge AI will not be abstract slides — they
> will be numbers you measured yourself.
>
> **Learning Outcomes — after completing this lab you will be able to:**
>
> ✔ Build and train a lightweight CNN using TensorFlow / Keras on MNIST
>
> ✔ Convert a trained model to TensorFlow Lite (.tflite) with
> quantization
>
> ✔ Run inference using the TFLite interpreter (simulated edge device)
>
> ✔ Deploy the same model as a REST API using FastAPI (cloud simulation)
>
> ✔ Measure and compare local latency vs cloud round-trip time (RTT)
>
> ✔ Calculate the bandwidth cost of sending image data to the cloud
>
> ✔ Identify privacy risks introduced by cloud-only inference
>
> ✔ Explain the Edge–Cloud continuum using your own measured evidence

# 2. Conceptual Background

## 2.1 The Four Drivers of Edge AI

> Module 1 established four fundamental reasons why cloud-only AI is
> insufficient for modern real-time applications. This lab gives you
> quantitative, hands-on evidence for three of them.

|              |                                                                |
|--------------|----------------------------------------------------------------|
| **Driver**   | **What You Will Measure in This Lab**                          |
| Latency      | Local TFLite mean latency ms vs cloud round-trip time ms       |
| Bandwidth    | Bytes transmitted per inference request to the cloud API       |
| Privacy      | What raw data leaves the device in the cloud scenario          |
| Connectivity | Failure behaviour when the server is unreachable (Extension D) |

## 2.2 TensorFlow Lite and the Conversion Pipeline

> TensorFlow Lite (TFLite) is Google's inference runtime for edge
> devices. It accepts a trained Keras or SavedModel and compiles it into
> a flat, memory-mapped FlatBuffer (.tflite). The result has no
> TensorFlow dependencies, runs on CPUs, MCUs, Coral Edge TPUs, and
> Android/iOS Neural Processing Units (NPUs), and is typically 4–15×
> smaller than the original SavedModel.
>
> **Conversion Pipeline**
>
> Keras Model → TFLiteConverter → .tflite FlatBuffer → TFLite
> Interpreter → Inference
>
> Key operations applied by the converter:
>
> • Operator fusion: Conv + BatchNorm + ReLU merged into a single kernel
>
> • Constant folding: pre-compute fixed subgraphs at compile time
>
> • Quantization: compress weights from FP32 (4 bytes) to INT8 (1 byte)

## 2.3 Simulating a Cloud API Locally

> A cloud AI service exposes its model through an HTTP REST endpoint. In
> production this is hosted on AWS, Azure, or GCP, introducing 50–500 ms
> of WAN propagation plus queuing. In this lab you simulate it by
> running FastAPI on localhost — the same machine as the edge client.
> This deliberately underestimates the real cloud latency penalty. The
> edge advantage you measure is therefore a lower bound: in a production
> setting it would be significantly larger.

# 3. Lab Setup

## 3.1 Environment

> All steps work on a laptop (Windows, macOS, or Linux) or Google Colab.
> No physical edge device is needed. The TFLite interpreter running on
> your laptop CPU simulates the constrained edge compute environment.
>
> **Install Dependencies**
>
> \# Create and activate a virtual environment
>
> python -m venv edgeai_lab1
>
> source edgeai_lab1/bin/activate \# macOS / Linux
>
> \# edgeai_lab1\Scripts\activate \# Windows
>
> \# Install all required packages
>
> pip install --upgrade pip
>
> pip install tensorflow\>=2.13.0
>
> pip install fastapi uvicorn requests numpy psutil
>
> \# Verify
>
> python -c "import tensorflow as tf; print('TF:', tf.\_\_version\_\_)"

## 3.2 Workspace Structure

> module1-lab/
>
> ├── build_model.py \# Step 1: train the MNIST CNN
>
> ├── convert_to_tflite.py \# Step 2: convert to TFLite
>
> ├── local_inference.py \# Step 3: edge inference benchmark
>
> ├── cloud_server.py \# Step 4a: FastAPI inference server
>
> ├── cloud_client.py \# Step 4b: cloud benchmark client
>
> ├── compare_results.py \# Step 5: unified side-by-side comparison
>
> ├── mnist_model/ \# ← generated by Step 1
>
> └── model.tflite \# ← generated by Step 2

# 4. Step-by-Step Instructions

> **STEP 1 — Build and Train a Lightweight MNIST CNN** ⏱ ~15 min
>
> **Concept**
>
> Before deploying AI anywhere we must first train a model. We use the
> MNIST dataset: 70,000 grayscale handwritten digit images (0–9), each
> 28×28 pixels. We build a minimal Convolutional Neural Network (CNN) —
> one convolutional layer with 8 filters followed by a 32-unit dense
> layer and a 10-class softmax output. Total parameters: approximately
> 5,900. This trains in under 60 seconds on a laptop CPU and reaches
> ~98% accuracy on the test set.
>
> **Why CNN?**
>
> Convolutional layers learn spatial features (edges, curves, stroke
> endings) that are
>
> invariant to small positional shifts — making them far more accurate
> than a dense
>
> network on image data, even at very small sizes.
>
> **File: build_model.py**
>
> """
>
> build_model.py
>
> Lab 1 — Step 1: Train a compact MNIST CNN.
>
> Represents the cloud training phase of the Edge AI lifecycle.
>
> """
>
> import tensorflow as tf
>
> import numpy as np
>
> \# ── 1. Load and preprocess MNIST ──────────────────────────
>
> (x_train, y_train), (x_test, y_test) =
> tf.keras.datasets.mnist.load_data()
>
> x_train = x_train.reshape(-1, 28, 28, 1).astype('float32') / 255.0
>
> x_test = x_test.reshape(-1, 28, 28, 1).astype('float32') / 255.0
>
> print(f'Training samples : {len(x_train):,}')
>
> print(f'Test samples : {len(x_test):,}')
>
> \# ── 2. Define model architecture ──────────────────────────
>
> model = tf.keras.Sequential(\[
>
> tf.keras.layers.Conv2D(8, kernel_size=3, activation='relu',
>
> padding='same', input_shape=(28, 28, 1)),
>
> tf.keras.layers.MaxPooling2D(pool_size=2),
>
> tf.keras.layers.Flatten(),
>
> tf.keras.layers.Dense(32, activation='relu'),
>
> tf.keras.layers.Dense(10, activation='softmax'),
>
> \], name='mnist_cnn')
>
> model.summary() \# Print parameter count
>
> \# ── 3. Compile ─────────────────────────────────────────────
>
> model.compile(
>
> optimizer='adam',
>
> loss='sparse_categorical_crossentropy',
>
> metrics=\['accuracy'\]
>
> )
>
> \# ── 4. Train ───────────────────────────────────────────────
>
> model.fit(x_train, y_train, epochs=5, batch_size=64,
>
> validation_split=0.1, verbose=1)
>
> \# ── 5. Evaluate ────────────────────────────────────────────
>
> loss, acc = model.evaluate(x_test, y_test, verbose=0)
>
> print(f'\nTest accuracy : {acc:.4f} ({acc\*100:.2f}%)')
>
> \# ── 6. Save ────────────────────────────────────────────────
>
> model.save('mnist_model.keras')
>
> print('\n✔ Model saved to mnist_model/')
>
> \# Save one test sample for consistent benchmarking later
>
> np.save('test_sample.npy', x_test\[:1\])
>
> print(f'✔ Test sample saved (true label: {y_test\[0\]})')
>
> **Run**
>
> python build_model.py
>
> **Expected Output**
>
> Test accuracy : 0.9820 (98.20%)
>
> ✔ Model saved to mnist_model/
>
> ✔ Test sample saved (true label: \<digit\>)
>
> **STEP 2 — Convert the Keras Model to TensorFlow Lite** ⏱ ~5 min
>
> **Concept**
>
> The TF SavedModel format requires the full TensorFlow library and
> keeps weights in 32-bit floats. Edge devices cannot load this
> efficiently. TFLite conversion produces a single .tflite FlatBuffer.
> With dynamic-range quantization (Optimize.DEFAULT), weight tensors are
> compressed to INT8, reducing model size by roughly 4× and enabling
> faster compute on hardware with INT8 SIMD.
>
> **File: convert_to_tflite.py**
>
> """
>
> convert_to_tflite.py
>
> Lab 1 — Step 2: Convert trained model to TFLite with quantization.
>
> """
>
> import tensorflow as tf
>
> import os
>
> \# Load the saved Keras model
>
> model = tf.keras.models.load_model('mnist_model.keras')
>
> print('Loaded model from mnist_model/')
>
> \# Create converter
>
> converter = tf.lite.TFLiteConverter.from_keras_model(model)
>
> \# Enable dynamic-range quantization:
>
> \# - Weights: FP32 → INT8 (~4× size reduction)
>
> \# - Activations: remain FP32 (no representative dataset needed)
>
> converter.optimizations = \[tf.lite.Optimize.DEFAULT\]
>
> \# Run conversion
>
> tflite_model = converter.convert()
>
> \# Save the FlatBuffer
>
> with open('model.tflite', 'wb') as f:
>
> f.write(tflite_model)
>
> \# Report compression
>
> sm_size = os.path.getsize('mnist_model.keras') / 1024
>
> tfl_size = os.path.getsize('model.tflite') / 1024
>
> print(f'\n✔ Conversion complete!')
>
> print(f' SavedModel size : {sm_size:,.1f} KB')
>
> print(f' TFLite size : {tfl_size:,.1f} KB')
>
> print(f' Compression : {sm_size/tfl_size:.1f}×')
>
> **Run**
>
> python convert_to_tflite.py
>
> **STEP 3 — Edge Inference — Run the TFLite Model Locally** ⏱ ~10 min
>
> **Concept**
>
> Your laptop CPU now acts as the edge compute unit. The TFLite
> interpreter loads the FlatBuffer, allocates tensors in fixed memory,
> and executes inference completely offline — no network, no data
> transmission. We run 100 inferences and report mean, median, and p95
> (95th-percentile) latency. In production, p95 determines whether your
> system meets its real-time SLA.
>
> **File: local_inference.py**
>
> """
>
> local_inference.py
>
> Lab 1 — Step 3: Edge inference benchmark using TFLite.
>
> """
>
> import numpy as np
>
> import time, os
>
> try:
>
> from tflite_runtime.interpreter import Interpreter
>
> except ImportError:
>
> from tensorflow.lite.python.interpreter import Interpreter
>
> \# ── 1. Load model ──────────────────────────────────────────
>
> interpreter = Interpreter(model_path='model.tflite')
>
> interpreter.allocate_tensors()
>
> inp = interpreter.get_input_details()\[0\]
>
> out = interpreter.get_output_details()\[0\]
>
> print(f'Input shape : {inp\["shape"\]}')
>
> print(f'Output shape : {out\["shape"\]}')
>
> \# ── 2. Load test sample ────────────────────────────────────
>
> x = np.load('test_sample.npy') if os.path.exists('test_sample.npy') \\
>
> else np.random.rand(1,28,28,1).astype(np.float32)
>
> \# ── 3. Warm-up (JIT compilation, not measured) ─────────────
>
> interpreter.set_tensor(inp\['index'\], x)
>
> interpreter.invoke()
>
> \# ── 4. Benchmark: 100 inferences ───────────────────────────
>
> N = 100
>
> times = \[\]
>
> for \_ in range(N):
>
> t0 = time.perf_counter()
>
> interpreter.set_tensor(inp\['index'\], x)
>
> interpreter.invoke()
>
> times.append((time.perf_counter()-t0)\*1000)
>
> times = np.array(times)
>
> pred = int(np.argmax(interpreter.get_tensor(out\['index'\])))
>
> conf = float(np.max(interpreter.get_tensor(out\['index'\])))
>
> \# ── 5. Report ──────────────────────────────────────────────
>
> print('\n' + '='\*50)
>
> print(' EDGE INFERENCE RESULTS')
>
> print('='\*50)
>
> print(f' Predicted class : {pred}')
>
> print(f' Confidence : {conf:.4f}')
>
> print(f' Mean latency : {times.mean():.3f} ms')
>
> print(f' Median latency : {np.median(times):.3f} ms')
>
> print(f' p95 latency : {np.percentile(times,95):.3f} ms')
>
> print(f' Max latency : {times.max():.3f} ms')
>
> print(f' Throughput : {1000/times.mean():.0f} inferences/sec')
>
> print('='\*50)
>
> print(' RECORD THESE VALUES IN YOUR RESULTS TABLE')
>
> **Run**
>
> python local_inference.py
>
> **STEP 4 — Cloud Deployment — FastAPI Server and Client** ⏱ ~20 min
>
> **Concept**
>
> In cloud AI, the model lives on a remote server. The client must
> serialise raw data into JSON, transmit it over the network, wait for
> inference, and receive a response. Every step adds latency and exposes
> data. In this step you build both sides: the server that hosts the
> TFLite model, and the client that measures total round-trip time.
>
> **What Leaves the Device?**
>
> Edge scenario : NOTHING. Inference is fully local.
>
> Cloud scenario : Raw image pixels — 784 float values — serialised as
> JSON.
>
> • A 28×28 greyscale image ≈ 6 KB of JSON per request
>
> • A 1080p colour frame ≈ 50 MB of JSON per request (impractical)
>
> This is why bandwidth is a primary driver of Edge AI adoption.
>
> **File: cloud_server.py**
>
> """
>
> cloud_server.py
>
> Lab 1 — Step 4a: FastAPI server simulating a cloud inference API.
>
> Run this FIRST in Terminal 1, then run cloud_client.py in Terminal 2.
>
> """
>
> from fastapi import FastAPI, HTTPException
>
> from pydantic import BaseModel
>
> import numpy as np, time
>
> try:
>
> from tflite_runtime.interpreter import Interpreter
>
> except ImportError:
>
> from tensorflow.lite.python.interpreter import Interpreter
>
> app = FastAPI(title='Lab1 Cloud Inference')
>
> \# Load model once at server startup
>
> interpreter = Interpreter(model_path='model.tflite')
>
> interpreter.allocate_tensors()
>
> inp = interpreter.get_input_details()\[0\]
>
> out = interpreter.get_output_details()\[0\]
>
> print('✔ Model ready on port 8000')
>
> class Request(BaseModel):
>
> data: list \# 784 float values (1×28×28×1 flattened)
>
> @app.get('/')
>
> def health(): return {'status': 'healthy'}
>
> @app.post('/infer')
>
> def infer(req: Request):
>
> try:
>
> x = np.array(req.data, dtype=np.float32).reshape(1,28,28,1)
>
> t0 = time.perf_counter()
>
> interpreter.set_tensor(inp\['index'\], x)
>
> interpreter.invoke()
>
> server_ms = (time.perf_counter()-t0)\*1000
>
> pred = int(np.argmax(interpreter.get_tensor(out\['index'\])))
>
> conf = float(np.max(interpreter.get_tensor(out\['index'\])))
>
> return {'prediction': pred, 'confidence': round(conf,4),
>
> 'server_latency_ms': round(server_ms,3),
>
> 'payload_bytes': len(req.data)\*4}
>
> except Exception as e:
>
> raise HTTPException(status_code=500, detail=str(e))
>
> if \_\_name\_\_ == '\_\_main\_\_':
>
> import uvicorn
>
> uvicorn.run(app, host='0.0.0.0', port=8000, log_level='warning')
>
> **File: cloud_client.py**
>
> """
>
> cloud_client.py
>
> Lab 1 — Step 4b: Benchmarks the cloud REST API from the client side.
>
> Measures total round-trip time including serialisation and network.
>
> """
>
> import requests, numpy as np, time, os, json
>
> URL = 'http://localhost:8000/infer'
>
> N = 100
>
> x = np.load('test_sample.npy') if os.path.exists('test_sample.npy') \\
>
> else np.random.rand(1,28,28,1).astype(np.float32)
>
> payload = {'data': x.flatten().tolist()}
>
> payload_bytes = len(json.dumps(payload).encode('utf-8'))
>
> print(f'Payload size : {payload_bytes:,} bytes
> ({payload_bytes/1024:.1f} KB)')
>
> \# Check server
>
> try:
>
> requests.get('http://localhost:8000/', timeout=5)
>
> print('Server reachable ✔')
>
> except:
>
> print('ERROR: start cloud_server.py first'); exit(1)
>
> \# Warm-up
>
> requests.post(URL, json=payload, timeout=10)
>
> \# Benchmark
>
> rtt_times, server_times = \[\], \[\]
>
> for \_ in range(N):
>
> t0 = time.perf_counter()
>
> res = requests.post(URL, json=payload, timeout=10)
>
> rtt_times.append((time.perf_counter()-t0)\*1000)
>
> server_times.append(res.json()\['server_latency_ms'\])
>
> rtt, srv = np.array(rtt_times), np.array(server_times)
>
> net = rtt - srv
>
> r = res.json()
>
> print('\n' + '='\*50)
>
> print(' CLOUD INFERENCE RESULTS')
>
> print('='\*50)
>
> print(f' Prediction : {r\["prediction"\]}')
>
> print(f' Server mean ms : {srv.mean():.3f}')
>
> print(f' Server p95 ms : {np.percentile(srv,95):.3f}')
>
> print(f' RTT mean ms : {rtt.mean():.3f}')
>
> print(f' RTT p95 ms : {np.percentile(rtt,95):.3f}')
>
> print(f' Net overhead ms : {net.mean():.3f} (serialise+transfer)')
>
> print(f' Payload size : {payload_bytes:,} bytes')
>
> print(f' At 30 FPS needs : {payload_bytes\*30/1024:.0f} KB/s')
>
> print('='\*50)
>
> print(' RECORD THESE VALUES IN YOUR RESULTS TABLE')
>
> **Run (Two Terminals Required)**
>
> \# Terminal 1 — start the server
>
> python cloud_server.py
>
> \# Terminal 2 — run the benchmark
>
> python cloud_client.py
>
> **STEP 5 — Unified Comparison** ⏱ ~10 min
>
> **File: compare_results.py**
>
> This script runs both benchmarks back-to-back and prints a
> side-by-side table. Ensure the cloud server is still running before
> executing this script.
>
> """
>
> compare_results.py
>
> Lab 1 — Step 5: Side-by-side edge vs cloud comparison.
>
> The cloud server must be running (python cloud_server.py).
>
> """
>
> import numpy as np, time, os, requests, json
>
> try:
>
> from tflite_runtime.interpreter import Interpreter
>
> except ImportError:
>
> from tensorflow.lite.python.interpreter import Interpreter
>
> x = np.load('test_sample.npy') if os.path.exists('test_sample.npy') \\
>
> else np.random.rand(1,28,28,1).astype(np.float32)
>
> N = 100
>
> \# ── Edge benchmark ─────────────────────────────────────────
>
> interp = Interpreter('model.tflite')
>
> interp.allocate_tensors()
>
> inp, out = interp.get_input_details()\[0\],
> interp.get_output_details()\[0\]
>
> interp.set_tensor(inp\['index'\], x); interp.invoke() \# warm-up
>
> edge_t = \[\]
>
> for \_ in range(N):
>
> t0=time.perf_counter(); interp.set_tensor(inp\['index'\],x)
>
> interp.invoke(); edge_t.append((time.perf_counter()-t0)\*1000)
>
> e_pred = int(np.argmax(interp.get_tensor(out\['index'\])))
>
> \# ── Cloud benchmark ─────────────────────────────────────────
>
> URL = 'http://localhost:8000/infer'
>
> payload = {'data': x.flatten().tolist()}
>
> payload_bytes = len(json.dumps(payload).encode())
>
> requests.post(URL, json=payload, timeout=10) \# warm-up
>
> cloud_t = \[\]
>
> for \_ in range(N):
>
> t0=time.perf_counter()
>
> res=requests.post(URL,json=payload,timeout=10)
>
> cloud_t.append((time.perf_counter()-t0)\*1000)
>
> c_pred = res.json()\['prediction'\]
>
> e, c = np.array(edge_t), np.array(cloud_t)
>
> print('\n' + '='\*65)
>
> print(f' {"METRIC":\<28} {"EDGE":\>16} {"CLOUD RTT":\>16}')
>
> print('='\*65)
>
> print(f' {"Mean latency (ms)":\<28} {e.mean():\>16.3f}
> {c.mean():\>16.3f}')
>
> print(f' {"Median (ms)":\<28} {np.median(e):\>16.3f}
> {np.median(c):\>16.3f}')
>
> print(f' {"p95 latency (ms)":\<28} {np.percentile(e,95):\>16.3f}
> {np.percentile(c,95):\>16.3f}')
>
> print(f' {"Throughput (inf/s)":\<28} {1000/e.mean():\>16.0f}
> {1000/c.mean():\>16.0f}')
>
> print(f' {"Data sent per call":\<28} {"0 bytes":\>16}
> {payload_bytes:\>13,} B')
>
> print(f' {"Network required":\<28} {"No":\>16} {"Yes":\>16}')
>
> print(f' {"Prediction":\<28} {e_pred:\>16} {c_pred:\>16}')
>
> print('='\*65)
>
> print(f'\n Edge speedup : {c.mean()/e.mean():.1f}x faster than cloud
> RTT')
>
> print(f' Bandwidth per inference : {payload_bytes:,} bytes
> ({payload_bytes/1024:.1f} KB)')
>
> print(f' At 30 FPS bandwidth need : {payload_bytes\*30/1024:.0f} KB/s
> ({payload_bytes\*30/1024/1024:.2f} MB/s)')
>
> print('='\*65)
>
> **Run**
>
> python compare_results.py

# 5. Record Your Results

> **Fill in the table below using your measured values from Step 5 (or
> from Steps 3 and 4 individually).**

|                               |                        |              |                             |
|-------------------------------|------------------------|--------------|-----------------------------|
| **Measurement Scenario**      | **Value (ms)**         | **Network?** | **Notes**                   |
| Local TFLite inference (edge) | \_\_\_\_\_\_\_\_\_\_\_ | No           | Pure on-device compute      |
| Cloud server compute only     | \_\_\_\_\_\_\_\_\_\_\_ | Server       | Returned in JSON body       |
| Cloud total round-trip (RTT)  | \_\_\_\_\_\_\_\_\_\_\_ | Yes (full)   | Upload+compute+download     |
| Edge advantage (RTT − local)  | \_\_\_\_\_\_\_\_\_\_\_ | —            | Network overhead quantified |
| Bandwidth per request (bytes) | \_\_\_\_\_\_\_\_\_\_\_ | —            | Size of JSON payload        |

> **Quick-Reference Checklist**

|                                                  |                |
|--------------------------------------------------|----------------|
| **Item to Record**                               | **Your Value** |
| TFLite model file size (KB)                      |                |
| SavedModel directory size (KB)                   |                |
| Compression ratio (SavedModel / TFLite)          |                |
| Model test set accuracy (%)                      |                |
| Edge speedup factor (cloud RTT mean / edge mean) |                |
| Bandwidth per request (bytes)                    |                |
| Equivalent bandwidth at 30 FPS (KB/s)            |                |
| Edge prediction matches cloud prediction?        | Yes / No       |

# 6. Reflection Questions

> Answer these questions in your submitted lab report. Ground each
> answer in the measurements you recorded above.
>
> **R1 — Latency and Real-Time Constraints**
>
> The factory assembly line in the opening story runs at 600 parts per
> minute (one part every 100 ms). Using your measured edge mean latency
> and cloud RTT, which system meets this requirement? Now consider an
> autonomous vehicle that requires pedestrian detection in under 10 ms.
> What does your experiment imply for cloud AI in automotive safety
> applications?
>
> **R2 — Privacy and Data Sovereignty**
>
> In the cloud scenario, raw image pixel values are serialised and
> transmitted to the server. Identify three real-world applications —
> beyond factory inspection — where this would create unacceptable
> privacy risks. For each, identify the applicable regulation (GDPR
> Article 25, HIPAA, or PDPA) and explain how on-device inference
> architecturally eliminates the compliance exposure.
>
> **R3 — Bandwidth Economics**
>
> Your 28×28 MNIST image requires approximately 6 KB of JSON payload. A
> smart city deploys 500 CCTV cameras capturing 1080p video at 25 FPS.
> Assume H.264 compression produces 50 KB per keyframe. Calculate the
> total uplink bandwidth required if all inference is done in the cloud.
> Now calculate how much bandwidth edge AI would save if each camera
> transmitted only a 1-byte class label. What are the infrastructure and
> economic implications of this reduction?
>
> **R4 — Connectivity and Resilience**
>
> Stop your cloud server (Ctrl+C). Attempt to run cloud_client.py and
> observe the error. Now run local_inference.py and observe it succeeds.
> What does this demonstrate about the availability requirements of edge
> vs cloud AI? Give three operational environments — beyond a network
> outage — where offline operation is non-negotiable.
>
> **R5 — Edge–Cloud Continuum**
>
> Module 1 introduced a 5-question decision framework for choosing the
> appropriate AI tier. Apply it to each of the following scenarios.
> State which tier (Edge / Fog / Cloud) you recommend and which of the
> four constraints (latency, bandwidth, privacy, connectivity) is the
> primary decision driver: (a) always-on keyword spotting on a smart
> speaker; (b) monthly batch analysis of patient outcome records; (c)
> industrial robot arm collision avoidance; (d) real-time credit card
> fraud detection at a payment terminal.

# 7. Extensions for Advanced Students

> **Extension A — Full Integer Quantization (INT8)**
>
> Modify convert_to_tflite.py to use full integer quantization where
> both weights AND activations are converted to INT8. You need to supply
> a representative dataset generator for calibration. Compare the
> resulting model's size and latency against the dynamic-range quantized
> version, and measure any accuracy difference across 1,000 test images.
>
> \# Extension A snippet — full INT8 quantization
>
> import tensorflow as tf, numpy as np
>
> model = tf.keras.models.load_model('mnist_model')
>
> (x_train,\_),\_ = tf.keras.datasets.mnist.load_data()
>
> x_train = x_train.reshape(-1,28,28,1).astype('float32')/255.0
>
> def representative_dataset():
>
> for img in x_train\[:200\]:
>
> yield \[img\[np.newaxis\]\]
>
> converter = tf.lite.TFLiteConverter.from_keras_model(model)
>
> converter.optimizations = \[tf.lite.Optimize.DEFAULT\]
>
> converter.representative_dataset = representative_dataset
>
> converter.target_spec.supported_ops =
> \[tf.lite.OpsSet.TFLITE_BUILTINS_INT8\]
>
> converter.inference_input_type = tf.uint8
>
> converter.inference_output_type = tf.uint8
>
> tflite_int8 = converter.convert()
>
> open('model_int8.tflite','wb').write(tflite_int8)
>
> print('INT8 model size:', len(tflite_int8)//1024, 'KB')
>
> **Extension B — Concurrent Cloud Requests**
>
> Modify cloud_client.py to use Python's asyncio and httpx to simulate 4
> simultaneous inference requests (representing 4 cameras feeding a
> single cloud endpoint). Compare the per-request latency and total
> throughput under concurrency against your single-request baseline.
> Relate your findings to the scalability limitations of cloud AI.
>
> **Extension C — Model Comparison: FP32 vs INT8**
>
> Convert the model twice: (1) without any optimizations (FP32
> baseline), and (2) with dynamic-range quantization (INT8). Evaluate
> both on the full 10,000-image MNIST test set and record top-1
> accuracy, mean latency, and model file size for each. Build a bar
> chart comparing the three metrics. At what accuracy threshold would
> INT8 quantization become unacceptable for a given application?
>
> **Extension D — Connectivity Failure Simulation**
>
> Implement a simple retry-with-local-fallback pattern: when
> cloud_client.py cannot reach the server within a configurable timeout,
> it falls back to local TFLite inference and logs the decision. This
> models a production edge AI system with graceful degradation. Test
> both the failure path (server down) and the success path (server up).

# 8. References and Further Reading

> **Textbooks**
>
> 1\. Situnayake, D., & Plunkett, J. (2023). AI at the Edge. O'Reilly
> Media. Chapters 1 (Foundations) and 3 (Model Optimization). ISBN:
> 978-1098120207.
>
> 2\. Rao, M., et al. (2022). Edge AI: Convergence of Edge Computing and
> Artificial Intelligence. Springer. Chapters 1 and 9. ISBN:
> 978-3030967079.
>
> **Academic Papers**
>
> 3\. Shi, W., Cao, J., Zhang, Q., Li, Y., & Xu, L. (2016). Edge
> Computing: Vision and Challenges. IEEE Internet of Things Journal,
> 3(5), 637–646. https://doi.org/10.1109/JIOT.2016.2579198
>
> 4\. Howard, A., et al. (2017). MobileNets: Efficient Convolutional
> Neural Networks for Mobile Vision Applications. arXiv:1704.04861.
> \[Background on lightweight architectures underlying TFLite
> optimization.\]
>
> **Documentation and Tools**
>
> 5\. Google. TensorFlow Lite Guide.
> https://www.tensorflow.org/lite/guide
>
> 6\. FastAPI Documentation. https://fastapi.tiangolo.com/
>
> 7\. MLPerf Inference Benchmark (Edge division).
> https://mlcommons.org/en/inference-edge-21/
>
> **Industry Reports**
>
> 8\. IDC. (2023). The Digitization of the World — From Edge to Core.
> \[Source for the 175 ZB global data projection.\]
>
> 9\. Gartner. (2021). Gartner Identifies Top Technology Trends Shaping
> Infrastructure. \[Source for the 75% enterprise data outside cloud
> projection by 2025.\]

*End of Lab Experiment 1 \| AIML ZG535 / AML ZC421 \| BITS Pilani WILPD*
