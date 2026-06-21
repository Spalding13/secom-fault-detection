"""Evaluation helpers for SECOM classification experiments."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def get_fail_probabilities(fitted_model, X, positive_label: int = 1):
    """Return predicted probabilities for the Fail class."""
    fail_class_index = list(fitted_model.classes_).index(positive_label)
    return fitted_model.predict_proba(X)[:, fail_class_index]


def summarize_model(model_name, y_true, y_pred, y_fail_proba):
    """Return the main metrics for model comparison."""
    return {
        "model": model_name,
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "fail_precision": precision_score(
            y_true,
            y_pred,
            pos_label=1,
            zero_division=0,
        ),
        "fail_recall": recall_score(
            y_true,
            y_pred,
            pos_label=1,
            zero_division=0,
        ),
        "fail_f1": f1_score(
            y_true,
            y_pred,
            pos_label=1,
            zero_division=0,
        ),
        "roc_auc": roc_auc_score(y_true == 1, y_fail_proba),
        "pr_auc": average_precision_score(y_true == 1, y_fail_proba),
    }


def evaluate_model_cv(model_name, estimator, X, y, cv):
    """Evaluate one estimator with stratified cross-validation."""
    fold_results = []

    for fold, (train_idx, valid_idx) in enumerate(cv.split(X, y), start=1):
        X_fold_train = X.iloc[train_idx]
        X_fold_valid = X.iloc[valid_idx]
        y_fold_train = y.iloc[train_idx]
        y_fold_valid = y.iloc[valid_idx]

        fold_model = clone(estimator)
        fold_model.fit(X_fold_train, y_fold_train)

        y_fold_pred = fold_model.predict(X_fold_valid)
        y_fold_fail_proba = get_fail_probabilities(fold_model, X_fold_valid)

        row = summarize_model(
            model_name,
            y_fold_valid,
            y_fold_pred,
            y_fold_fail_proba,
        )
        row["fold"] = fold
        fold_results.append(row)

    return pd.DataFrame(fold_results)


def get_oof_fail_probabilities(estimator, X, y, cv):
    """Generate out-of-fold Fail probabilities using only training data."""
    oof_fail_proba = np.zeros(len(y))

    for train_idx, valid_idx in cv.split(X, y):
        X_fold_train = X.iloc[train_idx]
        X_fold_valid = X.iloc[valid_idx]
        y_fold_train = y.iloc[train_idx]

        fold_model = clone(estimator)
        fold_model.fit(X_fold_train, y_fold_train)

        oof_fail_proba[valid_idx] = get_fail_probabilities(
            fold_model,
            X_fold_valid,
        )

    return oof_fail_proba
