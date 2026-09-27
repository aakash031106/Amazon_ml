"""
src/pipeline.py
Leader / Member 4: End-to-End Entity Resolution Pipeline

Integrates Normalization -> Blocking -> Features -> Classification -> Output Formatting.
"""

import os
import time
from typing import List, Dict, Any, Tuple
import numpy as np
import pandas as pd

from src.normalization import normalize_record
from src.blocking import MultiPassBlocker
from src.features import compute_pairwise_features
from src.model import ERMatchClassifier, FEATURE_COLUMNS


class ERPipeline:
    """
    End-to-End Entity Resolution Pipeline for Amazon ML Challenge 2026.
    """

    def __init__(self, model_type: str = "lightgbm", threshold: float = 0.70, max_candidates: int = 30):
        self.model_type = model_type
        self.threshold = threshold
        self.max_candidates = max_candidates
        self.blocker = MultiPassBlocker(max_candidates_per_s1=max_candidates)
        self.classifier = ERMatchClassifier(model_type=model_type)
        self.is_trained = False

    def train(
        self,
        s1_train_records: List[Dict[str, Any]],
        target_train_records: List[Dict[str, Any]],
        ground_truth_map: Dict[str, List[str]],
    ):
        """
        Train the pipeline on training data:
        1. Indexes target records in blocker
        2. Generates candidate pairs for training S1 records
        3. Extracts features for candidate pairs
        4. Labels pairs (1 if in ground truth, else 0)
        5. Fits the classifier
        """
        print("\n--- [Pipeline] Step 1: Indexing Targets for Training ---")
        self.blocker.index_targets(target_train_records)

        print("\n--- [Pipeline] Step 2: Generating Training Feature Matrix ---")
        t0 = time.time()
        X_list = []
        y_list = []
        target_dict = {r["entity_id"]: r for r in target_train_records}

        pos_count = 0
        neg_count = 0

        for s1 in s1_train_records:
            s1_id = s1["entity_id"]
            true_targets = set(ground_truth_map.get(s1_id, []))
            candidates = self.blocker.find_candidates(s1)

            for tid in candidates:
                if tid not in target_dict:
                    continue
                target = target_dict[tid]
                feat_dict = compute_pairwise_features(s1, target)
                feat_vec = [feat_dict[col] for col in FEATURE_COLUMNS]

                is_match = 1.0 if tid in true_targets else 0.0
                X_list.append(feat_vec)
                y_list.append(is_match)

                if is_match == 1.0:
                    pos_count += 1
                else:
                    neg_count += 1

        dt = time.time() - t0
        print(f"Generated {len(X_list):,} training pairs in {dt:.2f}s (Positives: {pos_count:,}, Negatives: {neg_count:,})")

        X = np.array(X_list)
        y = np.array(y_list)

        print("\n--- [Pipeline] Step 3: Fitting Match Classifier ---")
        self.classifier.fit(X, y)
        self.is_trained = True
        importances = self.classifier.get_feature_importances()
        print(f"Classifier trained successfully. Top features: {sorted(importances.items(), key=lambda x: x[1], reverse=True)[:4]}")

    def predict(
        self,
        s1_records: List[Dict[str, Any]],
        target_records: List[Dict[str, Any]],
        threshold: float = None,
    ) -> Tuple[Dict[str, List[str]], Dict[str, List[str]]]:
        """
        Run inference on test / evaluation records:
        Returns:
          - candidate_map: {s1_id: [candidate_ids]}
          - matching_map: {s1_id: [matched_ids]}
        """
        if not self.is_trained:
            raise RuntimeError("Pipeline must be trained before predicting.")

        th = threshold if threshold is not None else self.threshold

        # 1. Index test targets
        test_blocker = MultiPassBlocker(max_candidates_per_s1=self.max_candidates)
        test_blocker.index_targets(target_records)
        target_dict = {r["entity_id"]: r for r in target_records}

        candidate_map = {}
        matching_map = {}

        print(f"\n--- [Pipeline] Predicting matches for {len(s1_records):,} S1 entities (Threshold={th}) ---")
        t0 = time.time()

        for s1 in s1_records:
            s1_id = s1["entity_id"]
            candidates = test_blocker.find_candidates(s1)
            candidate_map[s1_id] = candidates

            if not candidates:
                matching_map[s1_id] = []
                continue

            # Batch compute features for all candidates of this S1
            pair_feats = []
            valid_cands = []
            for tid in candidates:
                if tid in target_dict:
                    f = compute_pairwise_features(s1, target_dict[tid])
                    pair_feats.append([f[col] for col in FEATURE_COLUMNS])
                    valid_cands.append(tid)

            if not pair_feats:
                matching_map[s1_id] = []
                continue

            probs = self.classifier.predict_proba(np.array(pair_feats))
            
            # Keep all candidates with probability >= threshold
            matched = [valid_cands[i] for i, p in enumerate(probs) if p >= th]
            matching_map[s1_id] = matched

        dt = time.time() - t0
        print(f"Prediction complete in {dt:.2f}s.")
        return candidate_map, matching_map


def save_output_tsvs(
    s1_ids: List[str],
    candidate_map: Dict[str, List[str]],
    matching_map: Dict[str, List[str]],
    output_dir: str = "outputs"
):
    """
    Save the two required competition TSVs in exact required format:
    1. candidate_pairs.tsv (source1_entity_id \t candidate_entity_ids)
    2. matching_results.tsv (source1_entity_id \t matched_entity_ids)
    """
    os.makedirs(output_dir, exist_ok=True)
    cand_path = os.path.join(output_dir, "candidate_pairs.tsv")
    match_path = os.path.join(output_dir, "matching_results.tsv")

    print(f"\nWriting output TSVs to {output_dir}...")
    
    # Write candidate_pairs.tsv
    with open(cand_path, "w", encoding="utf-8") as f:
        f.write("source1_entity_id\tcandidate_entity_ids\n")
        for sid in s1_ids:
            cands = ",".join(candidate_map.get(sid, []))
            f.write(f"{sid}\t{cands}\n")

    # Write matching_results.tsv
    with open(match_path, "w", encoding="utf-8") as f:
        f.write("source1_entity_id\tmatched_entity_ids\n")
        for sid in s1_ids:
            matches = ",".join(matching_map.get(sid, []))
            f.write(f"{sid}\t{matches}\n")

    print(f"Saved: {cand_path}")
    print(f"Saved: {match_path}")
