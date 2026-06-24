"""Reusable preprocessing transformers for the SECOM project."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted, validate_data


DEFAULT_MAX_MISSING_FRACTION = 0.50


class MissingValueColumnFilter(TransformerMixin, BaseEstimator):
    """Remove columns whose training-set missing fraction exceeds a limit."""

    def __init__(self, max_missing_fraction=DEFAULT_MAX_MISSING_FRACTION):
        self.max_missing_fraction = max_missing_fraction

    def fit(self, X, y=None):
        """Learn which columns have acceptable coverage in the training data."""
        if not 0 <= self.max_missing_fraction <= 1:
            raise ValueError("max_missing_fraction must be between 0 and 1.")

        values = validate_data(
            self,
            X,
            reset=True,
            dtype="numeric",
            ensure_all_finite="allow-nan",
        )
        self.missing_fraction_ = pd.isna(values).mean(axis=0)
        self.support_ = self.missing_fraction_ <= self.max_missing_fraction
        self.n_features_retained_ = int(self.support_.sum())
        self.n_features_removed_ = int((~self.support_).sum())

        if self.n_features_retained_ == 0:
            raise ValueError(
                "The missingness threshold removed every input column."
            )

        if hasattr(X, "columns"):
            self.input_columns_ = np.asarray(X.columns, dtype=object)
        elif hasattr(self, "input_columns_"):
            del self.input_columns_

        return self

    def transform(self, X):
        """Return only the columns retained during fit."""
        check_is_fitted(self, attributes=["support_", "n_features_in_"])
        values = validate_data(
            self,
            X,
            reset=False,
            dtype="numeric",
            ensure_all_finite="allow-nan",
        )

        if hasattr(X, "columns") and hasattr(self, "input_columns_"):
            current_columns = np.asarray(X.columns, dtype=object)
            if not np.array_equal(current_columns, self.input_columns_):
                raise ValueError(
                    "Input columns must match the columns and order seen during fit."
                )
            return X.loc[:, self.input_columns_[self.support_]]

        return values[:, self.support_]

    def get_support(self, indices=False):
        """Return the retained-column mask or retained-column indices."""
        check_is_fitted(self, attributes=["support_"])
        if indices:
            return np.flatnonzero(self.support_)
        return self.support_.copy()

    def get_feature_names_out(self, input_features=None):
        """Return feature names for the retained columns."""
        check_is_fitted(self, attributes=["support_", "n_features_in_"])

        if input_features is None:
            if hasattr(self, "input_columns_"):
                input_features = self.input_columns_
            else:
                input_features = np.asarray(
                    [f"x{i}" for i in range(self.n_features_in_)],
                    dtype=object,
                )
        else:
            input_features = np.asarray(input_features, dtype=object)

        if len(input_features) != self.n_features_in_:
            raise ValueError(
                "input_features must contain one name for each fitted column."
            )

        return input_features[self.support_]

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.input_tags.allow_nan = True
        return tags
