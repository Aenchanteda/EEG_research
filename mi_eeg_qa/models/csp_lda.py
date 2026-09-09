"""CSP-LDA baseline decoder."""

from __future__ import annotations

import numpy as np
from scipy.linalg import eigh
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.validation import check_is_fitted


class CSPTransformer(BaseEstimator, TransformerMixin):
    """Binary common spatial patterns transformer with log-variance features."""

    def __init__(self, n_components: int = 6, reg: float = 1e-6):
        self.n_components = n_components
        self.reg = reg

    def fit(self, X: np.ndarray, y: np.ndarray):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        classes = np.unique(y)
        if classes.size != 2:
            raise ValueError("CSPTransformer currently supports binary classification")
        covs = []
        for cls in classes:
            cls_cov = np.mean([self._cov(epoch) for epoch in X[y == cls]], axis=0)
            covs.append(cls_cov)
        composite = covs[0] + covs[1] + self.reg * np.eye(X.shape[1])
        eigvals, eigvecs = eigh(covs[0], composite)
        order = np.argsort(eigvals)
        half = max(1, self.n_components // 2)
        pick = np.r_[order[:half], order[-half:]]
        self.filters_ = eigvecs[:, pick].T
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        check_is_fitted(self, "filters_")
        projected = np.einsum("fc,nct->nft", self.filters_, np.asarray(X, dtype=float))
        var = np.var(projected, axis=-1)
        var /= var.sum(axis=1, keepdims=True) + 1e-12
        return np.log(var + 1e-12)

    def _cov(self, epoch: np.ndarray) -> np.ndarray:
        cov = epoch @ epoch.T
        return cov / (np.trace(cov) + 1e-12)


class CSPLDAClassifier(BaseEstimator, ClassifierMixin):
    """CSP feature extraction followed by Linear Discriminant Analysis."""

    def __init__(self, n_components: int = 6, reg: float = 1e-6):
        self.n_components = n_components
        self.reg = reg

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.label_encoder_ = LabelEncoder().fit(y)
        encoded = self.label_encoder_.transform(y)
        self.pipeline_ = Pipeline(
            [
                ("csp", CSPTransformer(n_components=self.n_components, reg=self.reg)),
                ("lda", LinearDiscriminantAnalysis()),
            ]
        )
        self.pipeline_.fit(X, encoded)
        self.classes_ = self.label_encoder_.classes_
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        check_is_fitted(self, "pipeline_")
        return self.pipeline_.predict_proba(X)

    def predict(self, X: np.ndarray) -> np.ndarray:
        encoded = np.argmax(self.predict_proba(X), axis=1)
        return self.label_encoder_.inverse_transform(encoded)
