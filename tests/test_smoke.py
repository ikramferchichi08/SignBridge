from pathlib import Path

import mlflow
import yaml

from src.utils.mlflow_utils import start_configured_run


def test_imports_and_params() -> None:
    import cv2
    import mediapipe
    import torch

    assert cv2.__version__
    assert mediapipe.__version__
    assert torch.__version__
    params = yaml.safe_load(Path("configs/params.yaml").read_text(encoding="utf-8"))
    assert params["experiment_name"] == "signbridge"
    assert params["sequence_length"] == 32


def test_mlflow_helper_creates_run(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"file:{tmp_path / 'mlruns'}")
    params = {"experiment_name": "smoke", "seed": 42}
    with start_configured_run(params) as run:
        mlflow.log_metric("smoke_metric", 1.0)
    assert run.info.run_id
