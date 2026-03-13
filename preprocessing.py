# preprocessing.py
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

class PositiveClipper(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = np.where(np.isfinite(X), X, 0)
        return np.clip(X, a_min=0, a_max=None)