# Starter issues

## 1. Download and explore the dataset

**Description:** Download the Google Isolated Sign Language Recognition
competition data from Kaggle and optionally compare it with a WLASL subset.

**Acceptance criteria:** Dataset license/rules are checked; a small, documented
subset is downloaded through DVC; labels, signers, class balance, and sample
counts are summarized without committing raw data.

**Suggested assignee:** A

## 2. Landmark extraction and normalization pipeline

**Description:** Extract MediaPipe landmarks and normalize around the shoulders.

**Acceptance criteria:** Reproducible script processes a DVC input, handles
missing hands, resamples to the configured sequence length, and writes a
versionable output with a short quality report.

**Suggested assignee:** A

## 3. Baseline GRU model with MLflow logging

**Description:** Train a lightweight landmark-based GRU baseline.

**Acceptance criteria:** Signer-aware train/validation/test split, seed control,
MLflow parameters and metrics, checkpoint artifact, and baseline report.

**Suggested assignee:** A

## 4. Model comparison

**Description:** Compare GRU, light Transformer, CNN+LSTM, and a pretrained video
model reference.

**Acceptance criteria:** Same signer-aware evaluation protocol, comparison table
covering top-1/top-5 accuracy, macro-F1, FPS, latency, and model size.

**Suggested assignee:** A

## 5. Live webcam pipeline with temporal smoothing

**Description:** Connect webcam landmarks to inference, confidence thresholds,
and a running transcript.

**Acceptance criteria:** Live skeleton overlay, buffered predictions, smoothing,
low-confidence warning, graceful camera failure, and a recorded fallback demo.

**Suggested assignee:** B

## 6. DVC pipeline, FastAPI skeleton, and Dockerfile

**Description:** Add `dvc.yaml`, a minimal FastAPI endpoint/WebSocket skeleton,
and a reproducible Docker development image.

**Acceptance criteria:** `dvc repro` describes the pipeline; API health check
works; Docker image builds; no credentials are stored in source control.

**Suggested assignee:** B
