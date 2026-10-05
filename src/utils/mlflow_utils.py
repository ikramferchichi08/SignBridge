"""Small MLflow helpers shared by experiments and setup checks."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import mlflow
import yaml


def load_params(path: str | Path = "configs/params.yaml") -> dict[str, Any]:
    """Load experiment parameters from YAML."""
    with Path(path).open(encoding="utf-8") as file:
        return yaml.safe_load(file)


def configure_mlflow(
    params: dict[str, Any] | None = None,
    params_path: str | Path = "configs/params.yaml",
) -> str:
    """Configure a local or environment-provided MLflow tracking URI."""
    loaded = params if params is not None else load_params(params_path)
    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "file:./mlruns")
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(loaded.get("experiment_name", "signbridge"))
    return tracking_uri


def start_configured_run(
    params: dict[str, Any] | None = None,
    params_path: str | Path = "configs/params.yaml",
):
    """Start a run and log the scalar and nested configuration values."""
    loaded = params if params is not None else load_params(params_path)
    configure_mlflow(loaded)
    run = mlflow.start_run()
    flat_params = {
        key: value
        for key, value in loaded.items()
        if isinstance(value, (str, int, float, bool)) and value is not None
    }
    mlflow.log_params(flat_params)
    return run
