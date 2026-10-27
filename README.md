# SignBridge

SignBridge is a free, open-source university project for real-time recognition of
isolated sign-language words from webcam or uploaded video. It extracts hand,
body, and face landmarks with MediaPipe, compares temporal models for a
50--100-sign vocabulary, smooths predictions, and displays a running transcript.
Sentence generation and speech are optional extensions.

## Team

| Student | Responsibilities |
| --- | --- |
|Ikram ferchichi | Dataset preparation, landmark extraction, normalization, model training, comparison, and MLflow |
| Malak ghannouchi | Live pipeline, temporal smoothing, DVC pipeline, API, frontend, Docker, and drift monitoring |
| Both | Evaluation, custom signer clips, report, and presentation |

## Setup

PowerShell:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
pytest
```

The pinned MediaPipe version is 0.10.21 and is verified by the setup test
through the classic `mp.solutions.holistic` API.

Initialize DVC and use the local placeholder remote:

```powershell
dvc init
dvc remote add -d local ..\signbridge-dvc-storage
```

The directories `data/raw`, `data/landmarks`, `data/processed`, and `models`
are intended for DVC-tracked artifacts. Their empty `.gitkeep` files are
tracked now; datasets and trained models are deliberately not downloaded or
committed in Step 1. DVC metadata files remain trackable.

To switch to a future DagsHub remote, provide the repository URL and credentials
your team receives, then use:

```powershell
dvc remote add -d dagshub https://dagshub.com/<owner>/<repo>.dvc
dvc remote modify dagshub auth basic
$env:DAGSHUB_USERNAME = "<your-username>"
$env:DAGSHUB_TOKEN = "<your-token>"
```

For Google Drive, provide a shared folder ID and configure OAuth on the machine:

```powershell
dvc remote add -d gdrive gdrive://<folder-id>
dvc remote modify gdrive gdrive_use_service_account true
```

Do not commit tokens. The exact DagsHub owner/repository, token, or Google
service-account/OAuth details must be supplied by the team.

## Verification and MLflow

## Dataset

SignBridge uses a subset of **WLASL** (Word-Level American Sign Language),
distributed under the C-UDA licence for computational use only.

| Item | Value |
| --- | --- |
| Signs | 100 |
| Videos | 1,102 (7-19 per sign) |
| Signers | 74 (one signer provides about 15% of the videos) |
| Sources | YouTube and ASL dictionary sites, so resolution, lighting, and signer vary |

Notes:

- The videos are **not** in Git. Only the small `data/raw/wlasl.dvc` pointer is
  committed. Do not publish or redistribute the videos (C-UDA).
- Only about 44% of the missing original links were still available, so the
  subset is smaller than the 1,860 videos listed in WLASL for these 100 signs.
- The original WLASL `split` column is **not** used: the same signer appears in
  train and validation. Split with `StratifiedGroupKFold` (group = `signer_id`,
  stratified by sign, 5 folds) and report the mean and standard deviation.
- Cite WLASL (Li et al., "Word-level Deep Sign Language Recognition from Video:
  A New Large-scale Dataset and Methods Comparison") in reports.

## Data pipeline

Run these in PowerShell from the project root with the virtual environment active.

1. **Download the videos** (only the glosses already in `data/raw/wlasl/metadata.csv`;
   resumable; needs `yt-dlp` inside the venv):

   ```powershell
   python -m pip install yt-dlp
   python src\download_wlasl_subset.py --wlasl-json "<path>\WLASL_v0.3.json"
   ```

   `WLASL_v0.3.json` comes from the official WLASL repository (`start_kit`).
   Failures are logged in `reports/wlasl_download_log.csv`. Then version the data:

   ```powershell
   dvc add data/raw/wlasl
   git add data/raw/wlasl.dvc
   ```

2. **Extract landmarks** (MediaPipe Holistic; 25 pose + 21 + 21 hand + 60 face points;
   normalized by shoulder midpoint and shoulder width; resampled to 32 frames):

   ```powershell
   python src\extract_landmarks.py --limit 10 --workers 2   # quick test
   python src\extract_landmarks.py --workers 4              # full, resumable
   dvc add data/landmarks/wlasl
   ```

   Output in `data/landmarks/wlasl/`: one `.npz` per video (`x` = `[32, 381]`
   features, `mask` = detected pose/left hand/right hand/face per frame),
   `dataset_T32.npz` (all sequences plus labels and signer ids),
   `classes.json`, and `feature_spec.json`. Quality statistics (share of frames
   with detected hands) are in `reports/landmark_extraction_log.csv`.

3. **Benchmark parallel extraction** (Big Data component):

   ```powershell
   python src\extract_landmarks.py --benchmark 1 2 4 8 --limit 40
   ```

   Results go to `reports/extraction_benchmark.csv` (time, speedup, efficiency).

## Project status

| Step | Status |
| --- | --- |
| Environment, DVC, MLflow setup | Done |
| WLASL subset downloaded and DVC-tracked | Done (100 signs, 1,102 videos, 74 signers) |
| Landmark extraction and normalization | Script ready; full run and quality check pending |
| Signer-independent splits | Planned |
| GRU baseline, Transformer, CNN+LSTM, pretrained reference | Planned |
| Live pipeline, smoothing, API, frontend, Docker, drift monitoring | Planned |
Run the headless check with a video:

```powershell
python -m src.test_setup --video path\to\video.mp4 --no-display
```

For a webcam:

```powershell
python -m src.test_setup
```

Press `q` to stop the webcam window. The script prints package versions,
creates a dummy MLflow run, and reports detected hands/pose and FPS. To view
the local tracking UI (do not commit or leave it running in automation):

```powershell
mlflow ui --backend-store-uri .\mlruns
```

Open the displayed local URL in a browser.

## Folder structure

```text
data/{raw,landmarks,processed}/  DVC-managed data
src/                             Python source and setup checks
notebooks/                       Exploration notebooks
configs/                         YAML experiment configuration
app/                             API/frontend code
models/                          DVC-managed model artifacts
reports/                         Evaluation outputs
tests/                           Automated tests
docs/                            Project documentation and issues
```

## Requirements mapping

- GitHub: branches, issues, pull requests, and visible student contributions.
- DVC: reproducible versioning of raw videos, landmarks, processed data, and models.
- MLflow: experiment parameters, metrics, artifacts, and model comparison.
- Multiple models: landmark GRU, light Transformer, CNN+LSTM, and a pretrained
  video model as an accuracy reference.
- Free tools: PyTorch, MediaPipe, OpenCV, FastAPI, Streamlit/Gradio, Docker,
  Evidently, and optional Airflow.

## Planned workflow

1. Download and version a manageable isolated-sign subset; split by signer.
2. Extract and normalize landmarks, resample sequences to 32--64 frames, and
   add carefully selected augmentations.
3. Train the baseline and alternatives with every run logged to MLflow.
4. Compare accuracy, macro-F1, latency, FPS, model size, and signer generalization.
5. Connect the best lightweight model to the webcam buffer and temporal smoothing.
6. Evaluate on held-out signers and custom clips, monitor drift, and package the demo.

Continuous translation is out of scope for this phase. The system is an assistant,
not a replacement for qualified interpreters.
