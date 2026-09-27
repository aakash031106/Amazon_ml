"""
src/model.py
Member 3: Entity Resolution Classifier Module

Trains machine learning classifiers (LightGBM, Logistic Regression) on pairwise
features and predicts match probabilities.
"""

from typing import List, Dict, Any, Tuple
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression


FEATURE_COLUMNS = [
    "name_exact",
    "name_jaccard",
    "name_recall",
    "name_char_cosine",
    "name_pfx_match",
    "name_len_diff",
    "name_token_diff",
    "addr_exact",
    "addr_jaccard",
    "addr_char_cosine",
    "addr_num_match",
    "is_addr_empty",
    "country_match",
    "combined_jaccard",
]


class ERMatchClassifier:
    """
    Binary classifier that predicts whether a candidate pair (S1, S2/S3) is a true match.
    """

    def __init__(self, model_type: str = "lightgbm", random_state: int = 42):
        self.model_type = model_type
        self.feature_names = FEATURE_COLUMNS

        if model_type == "lightgbm":
            self.model = LGBMClassifier(
                n_estimators=100,
                learning_rate=0.08,
                num_leaves=31,
                random_state=random_state,
                n_jobs=-1,
                verbose=-1,
            )
        elif model_type == "logistic_regression":
            self.model = LogisticRegression(
                max_iter=1000,
                random_state=random_state,
            )
        else:
            raise ValueError(f"Unknown model_type: {model_type}")

    def fit(self, X: np.ndarray, y: np.ndarray):
        """Train the classifier on feature matrix X and binary labels y."""
        self.model.fit(X, y)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Return the predicted probability of being a true match (class 1).
        Array of floats between 0.0 and 1.0.
        """
        if len(X) == 0:
            return np.array([])
        return self.model.predict_proba(X)[:, 1]

    def get_feature_importances(self) -> Dict[str, float]:
        """Return relative importance of each feature."""
        if hasattr(self.model, "feature_importances_"):
            importances = self.model.feature_importances_
            total = float(sum(importances)) or 1.0
            return {
                name: round(float(imp) / total, 4)
                for name, imp in zip(self.feature_names, importances)
            }
        elif hasattr(self.model, "coef_"):
            coefs = np.abs(self.model.coef_[0])
            total = float(sum(coefs)) or 1.0
            return {
                name: round(float(c) / total, 4)
                for name, c in zip(self.feature_names, coefs)
            }
        return {}
