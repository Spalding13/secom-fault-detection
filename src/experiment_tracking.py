"""Small MLflow helpers for SECOM experiment tracking."""

from __future__ import annotations

import os
from pathlib import Path

import mlflow
import mlflow.sklearn


EXPERIMENT_NAME = "secom-fault-detection"


def configure_mlflow(project_root: Path, experiment_name: str = EXPERIMENT_NAME):
    """Configure local file-based MLflow tracking for this project."""
    tracking_dir = project_root / "mlruns"
    os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
    mlflow.set_tracking_uri(tracking_dir.as_uri())
    mlflow.set_experiment(experiment_name)
    return tracking_dir


def log_model_run(
    run_name,
    params,
    metrics,
    artifact_paths=None,
    sklearn_model=None,
    model_name="model",
):
    """Log one manually controlled MLflow run."""
    artifact_paths = artifact_paths or []

    with mlflow.start_run(run_name=run_name):
        mlflow.log_params(_stringify_params(params))
        mlflow.log_metrics(_clean_metrics(metrics))

        for artifact_path in artifact_paths:
            artifact_path = Path(artifact_path)
            if artifact_path.exists():
                mlflow.log_artifact(str(artifact_path))

        if sklearn_model is not None:
            mlflow.sklearn.log_model(
                sk_model=sklearn_model,
                name=model_name,
                serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_PICKLE,
            )


def _stringify_params(params):
    """Convert complex or missing parameter values into MLflow-safe strings."""
    return {
        key: "None" if value is None else str(value)
        for key, value in params.items()
    }


def _clean_metrics(metrics):
    """Drop missing metrics before MLflow logging."""
    return {
        key: float(value)
        for key, value in metrics.items()
        if value is not None
    }
