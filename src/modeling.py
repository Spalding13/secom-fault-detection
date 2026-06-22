"""Model definitions and training helpers."""

from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.kernel_ridge import KernelRidge


class KernelRidgeClassifier(ClassifierMixin, BaseEstimator):
    """Small classifier wrapper around sklearn's KernelRidge regressor.

    The SECOM UCI baseline summary refers to a kernel ridge classifier. Since
    sklearn exposes KernelRidge as a regression estimator, this wrapper fits
    the regressor on labels -1 and 1 and thresholds the continuous score at 0.
    """

    def __init__(
        self,
        alpha=1.0,
        kernel="rbf",
        gamma=None,
        degree=3,
        coef0=1,
    ):
        self.alpha = alpha
        self.kernel = kernel
        self.gamma = gamma
        self.degree = degree
        self.coef0 = coef0

    def fit(self, X, y):
        """Fit KernelRidge using Pass=-1 and Fail=1 labels."""
        self.classes_ = np.array([-1, 1])
        y_encoded = np.asarray(y)

        self.model_ = KernelRidge(
            alpha=self.alpha,
            kernel=self.kernel,
            gamma=self.gamma,
            degree=self.degree,
            coef0=self.coef0,
        )
        self.model_.fit(X, y_encoded)
        return self

    def decision_function(self, X):
        """Return the raw continuous KernelRidge prediction score."""
        return self.model_.predict(X)

    def predict(self, X):
        """Predict Fail for scores >= 0 and Pass otherwise."""
        scores = self.decision_function(X)
        return np.where(scores >= 0, 1, -1)
