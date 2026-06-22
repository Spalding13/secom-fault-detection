"""Utilities for loading the SECOM dataset."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


EXPECTED_FEATURE_SHAPE = (1567, 590)
EXPECTED_LABELS = {-1, 1}
DOWNLOAD_HINT = "python scripts/download_data.py"


def load_secom_data(data_dir: Path) -> tuple[pd.DataFrame, pd.Series]:
    """Load SECOM feature and target files from a raw data directory.

    Parameters
    ----------
    data_dir:
        Directory containing ``secom.data`` and ``secom_labels.data``.

    Returns
    -------
    X, y:
        Feature matrix and target labels using the original SECOM convention:
        ``1 = Fail`` and ``-1 = Pass``.
    """
    data_dir = Path(data_dir)
    features_path = data_dir / "secom.data"
    labels_path = data_dir / "secom_labels.data"

    missing = [path.name for path in (features_path, labels_path) if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing SECOM raw data file(s): "
            + ", ".join(missing)
            + f". Run `{DOWNLOAD_HINT}` from the repository root."
        )

    X = pd.read_csv(features_path, sep=r"\s+", header=None, na_values="NaN")
    labels = pd.read_csv(
        labels_path,
        sep=r"\s+",
        header=None,
        names=["target", "timestamp"],
    )
    y = labels["target"]

    if X.shape[0] != len(y):
        raise ValueError(
            f"Feature rows ({X.shape[0]}) do not match labels ({len(y)})."
        )

    if X.shape != EXPECTED_FEATURE_SHAPE:
        raise ValueError(
            f"Unexpected SECOM feature shape {X.shape}; "
            f"expected {EXPECTED_FEATURE_SHAPE}."
        )

    observed_labels = set(y.dropna().astype(int).unique())
    if observed_labels != EXPECTED_LABELS:
        raise ValueError(
            f"Unexpected target labels {sorted(observed_labels)}; "
            f"expected {sorted(EXPECTED_LABELS)}."
        )

    return X, y
