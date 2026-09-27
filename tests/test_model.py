"""
tests/test_model.py
Unit tests and training validation for Member 3's Features and Model modules.
"""

import sys
import os
import numpy as np

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.normalization import normalize_record
from src.features import compute_pairwise_features
from src.model import ERMatchClassifier, FEATURE_COLUMNS


def test_features_computation():
    print("--- 1. Testing Feature Computation ---")
    s1 = normalize_record({
        "entity_id": "S1-100",
        "business_name": "Payne Enterprises Inc",
        "business_address": "3315 Fremont St, Peoria, IL",
        "country": "US"
    })

    # True Match candidate (with minor typo & abbreviation)
    target_match = normalize_record({
        "entity_id": "S2-200",
        "business_name": "Payne Enterpires",
        "business_address": "3315 Fremont Street, Peoria, IL",
        "country": "US"
    })

    # False Match candidate (different business)
    target_nonmatch = normalize_record({
        "entity_id": "S2-300",
        "business_name": "Payne Medical Clinic",
        "business_address": "900 Main St, Chicago, IL",
        "country": "US"
    })

    feat_match = compute_pairwise_features(s1, target_match)
    feat_nonmatch = compute_pairwise_features(s1, target_nonmatch)

    print("True Match Features:")
    for k, v in feat_match.items():
        print(f"  {k:20s}: {v:.4f}")

    print("\nNon-Match Features:")
    for k, v in feat_nonmatch.items():
        print(f"  {k:20s}: {v:.4f}")

    assert feat_match["name_char_cosine"] > 0.60, "Char cosine should be solid for typo match!"
    assert feat_match["addr_num_match"] == 1.0, "Street number 3315 should match!"
    assert feat_match["combined_jaccard"] > feat_nonmatch["combined_jaccard"], "Match score should be higher!"
    print("\nFeature computation tests PASSED!\n")


def test_model_training():
    print("--- 2. Testing Classifier Training & Probability Output ---")
    
    # Generate synthetic training batch
    # Positive pairs: high similarity
    # Negative pairs: low similarity
    X_pos = np.random.uniform(0.7, 1.0, size=(100, len(FEATURE_COLUMNS)))
    y_pos = np.ones(100)

    X_neg = np.random.uniform(0.0, 0.4, size=(200, len(FEATURE_COLUMNS)))
    y_neg = np.zeros(200)

    X = np.vstack([X_pos, X_neg])
    y = np.concatenate([y_pos, y_neg])

    for model_type in ["logistic_regression", "lightgbm"]:
        clf = ERMatchClassifier(model_type=model_type)
        clf.fit(X, y)
        probs = clf.predict_proba(X[:5])
        importances = clf.get_feature_importances()

        print(f"Model: {model_type}")
        print(f"  Sample probabilities: {[round(p, 4) for p in probs]}")
        print(f"  Top 3 Important Features: {sorted(importances.items(), key=lambda x: x[1], reverse=True)[:3]}")
        assert all(0.0 <= p <= 1.0 for p in probs), "Probabilities must be in [0, 1]!"

    print("\nALL MODEL AND FEATURE TESTS PASSED SUCCESSFULLY!")

if __name__ == "__main__":
    test_features_computation()
    test_model_training()
