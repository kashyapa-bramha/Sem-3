# Lab Experiment 1: Academic Report Outline

Status: outline for review, not the final report. Use the assignment, the six implemented programs, the student's recorded runs, and the agreed R1-R5 answers. Do not substitute validation-run timings or the handout's illustrative values.

| Section | Planned content and evidence |
|---|---|
| 1. Title | **Build, Convert, Deploy, Compare: The Foundations of Edge AI**. Identify BITS Pilani WILP, Machine Learning on Edge, and Experiment 1 (EC-1). Reserve fields for student details and applicable course code. |
| 2. Objective | Compare the same quantized MNIST CNN locally and through a localhost HTTP service. State what is measured: file size, inference latency, client RTT, and request-body bytes. |
| 3. Learning Outcomes | Map training, conversion, local inference, API deployment, measurement, and tier selection to the artifacts or results that demonstrate each outcome. |
| 4. Introduction and Background | Introduce the factory's 100 ms decision budget; explain latency, bandwidth, privacy, and connectivity. Identify the factory story as motivation and localhost as a cloud simulation. |
| 5. Experimental Environment | Record Windows 11 Enterprise build 26200, Python 3.12.10 (64-bit), VS Code, assignment virtual environment, and verified library versions. Identify the laptop CPU as the execution device for both paths. **Evidence A:** interpreter and package-version output only. |
| 6. CNN Model Architecture | Table of layer, output shape, and parameter count: convolution with 8 filters, max pooling, flatten, dense 32, softmax 10. Use the verified total of 50,618 parameters, not the handout's approximate count. **Evidence B:** model summary. |
| 7. Model Training and Evaluation | Describe `build_model.py`: normalized float32 images with one grayscale channel; 54,000 training, 6,000 validation, 10,000 test images; Adam, sparse categorical cross-entropy, five epochs, batch size 64. Report 98.04% Keras test accuracy and the saved benchmark sample's true label, 7. **Evidence C:** evaluation and save messages. |
| 8. TensorFlow Lite Conversion and Quantization | Describe `convert_to_tflite.py`, dynamic-range optimization of eligible weights, and the saved FlatBuffer. Define compression as original archive bytes divided by TFLite bytes: 637,645 / 55,960 = 11.39 (rounded). **Evidence D:** conversion summary with both file sizes. |
| 9. Edge Inference Implementation | Trace `local_inference.py`: allocate tensors once, load the saved image, invoke locally, and read the output scores. Explain the missing-sample error and zero network upload. **Evidence E:** Step 3 report with prediction, confidence, and latency statistics. |
| 10. Cloud Inference Implementation | Trace `cloud_client.py` and `cloud_server.py`: health check, flattened JSON pixels, POST `/infer`, float32 reconstruction, shared interpreter, JSON response. Distinguish the server's 3,136 decoded pixel bytes from 5,687 serialized request-body bytes. **Evidence F:** healthy response and Step 4b client report. |
| 11. Experimental Methodology | Diagram the two timing intervals: edge input copy through inference completion; client POST through receipt of response. Show the separate server compute timer. Explain the common sample, one excluded warm-up and 100 timed calls per path, `perf_counter`, mean, median, p95, and sequential throughput. |
| 12. Results | Reproduce the assignment's measurement table and quick-reference checklist from `results.md`. Use Step 5 for paired edge/RTT results and identify Step 4b as the source for server compute. Preserve standalone statistics separately, including Step 3 maximum latency. Mark unavailable measurements explicitly. |
| 13. Edge vs Cloud Comparison | Use `compare_results.py` output: Step 5 mean edge 0.0079 ms versus RTT 18.1786 ms; edge advantage 18.1707 ms from displayed means; speedup 2,298.8x from unrounded means. Include median, p95, throughput, network requirement, and matching predictions. **Evidence G:** full Step 5 terminal table. |
| 14. Bandwidth Calculations | Separate measured payload from derived rate: 5,687 bytes/request multiplied by 30 requests/s and divided by 1,024 gives 166.61 KiB/s. Contrast with zero local upload; exclude headers and responses. Keep the hypothetical CCTV calculation under R3. |
| 15. Reflection Questions R1-R5 | Use the reflection plan below; label measured, calculated, and expected results distinctly. |
| 16. Discussion | Explain what the placement comparison isolates and what it does not: HTTP overhead is not pure transit, the size ratio compares whole files, and no FP32-versus-quantized timing or accuracy comparison was performed. Address one repeated image, one machine, and no WAN or embedded hardware test. |
| 17. Conclusion | Summarize the observed agreement on one digit, measured storage reduction, local latency advantage, and avoided image upload. Do not generalize to all models, devices, or safety deadlines. |
| 18. References | Cite the assignment as the primary source and identify the implementation and terminal results as original experimental evidence. Include only supplied references actually used; do not imply that every work in the handout was consulted. |

## Reflection plan

| Question | Planned argument |
|---|---|
| R1: Latency | Use the student's Step 3 mean of 0.0082 ms and Step 4b RTT of 17.0728 ms. Compare with the 100 ms factory and under-10 ms automotive budgets. Separate average performance from a deadline guarantee. |
| R2: Privacy | Use the agreed examples: hospital bedside monitoring (HIPAA), smart-speaker audio (GDPR Article 25), and Singapore workplace access control (PDPA). Identify sensitive data, exposure avoided, and residual obligations for each. |
| R3: Bandwidth economics | Show decimal-unit calculations for 500 cameras at 25 FPS, assuming 50 KB per transmitted frame. Compare 5 Gb/s with 100 kb/s for one-byte labels; derive 4.9999 Gb/s saved and 99.998% reduction. Discuss link capacity, storage, edge costs, and loss of footage. Label these as scenario calculations. |
| R4: Resilience | Distinguish a stopped service from a network outage. Contrast cloud dependency with local availability; use the mine, aircraft, and air-gapped facility examples. **Evidence H:** failed cloud health check and successful local inference while the server is stopped. |
| R5: Placement | Give tier, primary driver, and alternatives for all four cases: keyword spotting (edge/privacy), monthly outcomes analysis (approved cloud/privacy), collision avoidance (edge/latency), and payment fraud detection (regional fog/latency). State the conditions behind the cloud and fog recommendations. |

## Before finalization

- **Direct R4 evidence:** Shutdown was observed; the paired post-shutdown cloud failure and local success were not supplied. Provide those terminal outputs or keep the behavior explicitly labeled as expected, not observed. Do not replace the original benchmark timings with this demonstration.
- **Course details:** Confirm the applicable course code (the handout lists AIML ZG535 / AML ZC421). Supply student name, student ID, session/submission dates, and any cover-page requirements, or leave placeholders for private completion.
- **R5 framework:** The handout refers to the module's five decision questions but does not reproduce them. If exact mapping to their wording is required, provide the relevant slide or excerpt; otherwise use the supplied constraints and agreed answers.
- **Evidence format:** Original terminal excerpts can support the results. Supply screenshot assets only if required or preferred; the A-H markers indicate placement, not a claim that screenshots exist. Redact personal paths and never include credential-bearing package-index logs.
- **Optional reproducibility details:** CPU model and installed RAM have not been established here. Add only if verified. Quantized test-set accuracy, WAN latency, pure network transit time, and end-to-end camera latency were not measured and are not prerequisites for this report.