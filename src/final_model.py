"""Shared final-model configuration and persistence helpers."""

from __future__ import annotations

import pickle
from pathlib import Path

from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, VarianceThreshold, f_classif
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from src.preprocessing import (
    DEFAULT_MAX_MISSING_FRACTION,
    MissingValueColumnFilter,
)


FINAL_MODEL_NAME = "RF + SelectKBest tuned"
FINAL_FAIL_THRESHOLD = 0.35
FINAL_THRESHOLD_SELECTION_METRIC = "Fail F1"
FINAL_THRESHOLD_SELECTION_METHOD = (
    "maximize Fail F1 on training-set out-of-fold predictions"
)
FINAL_THRESHOLD_CV_FOLDS = 3
FINAL_RANDOM_STATE = 42
FINAL_POSITIVE_LABEL = 1
FINAL_NEGATIVE_LABEL = -1


FINAL_MODEL_PARAMS = {
    "n_estimators": 100,
    "class_weight": "balanced",
    "max_depth": None,
    "min_samples_leaf": 5,
    "random_state": FINAL_RANDOM_STATE,
    "n_jobs": -1,
}


FINAL_FEATURE_SELECTION = {
    "method": "SelectKBest",
    "score_func": "f_classif",
    "k": 200,
}


def build_final_pipeline():
    """Build the selected final SECOM preprocessing-and-model pipeline."""
    return Pipeline(steps=[
        (
            "missingness_filter",
            MissingValueColumnFilter(
                max_missing_fraction=DEFAULT_MAX_MISSING_FRACTION
            ),
        ),
        ("imputer", SimpleImputer(strategy="median")),
        ("variance_filter", VarianceThreshold(threshold=0.0)),
        (
            "feature_selector",
            SelectKBest(score_func=f_classif, k=FINAL_FEATURE_SELECTION["k"]),
        ),
        ("classifier", RandomForestClassifier(**FINAL_MODEL_PARAMS)),
    ])


def build_final_model_artifact(fitted_pipeline):
    """Create the persisted artifact dictionary for final inference."""
    return {
        "pipeline": fitted_pipeline,
        "fail_threshold": FINAL_FAIL_THRESHOLD,
        "positive_label": FINAL_POSITIVE_LABEL,
        "negative_label": FINAL_NEGATIVE_LABEL,
        "model_name": FINAL_MODEL_NAME,
        "selection_metric": FINAL_THRESHOLD_SELECTION_METRIC,
        "threshold_selection_method": FINAL_THRESHOLD_SELECTION_METHOD,
        "threshold_cv_folds": FINAL_THRESHOLD_CV_FOLDS,
        "random_state": FINAL_RANDOM_STATE,
        "model_params": FINAL_MODEL_PARAMS.copy(),
        "feature_selection": FINAL_FEATURE_SELECTION.copy(),
        "max_missing_fraction": DEFAULT_MAX_MISSING_FRACTION,
    }


def save_final_model_artifact(artifact, path):
    """Save a final model artifact dictionary with the highest pickle protocol."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as file:
        pickle.dump(artifact, file, protocol=pickle.HIGHEST_PROTOCOL)


def load_final_model_artifact(path):
    """Load a trusted final model artifact dictionary."""
    with Path(path).open("rb") as file:
        return pickle.load(file)
