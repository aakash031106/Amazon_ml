"""
src/validate_sota_approach.py
Validates the Hard-Negative Mining + Precision Filtering approach against
the old baseline on a realistic validation split, computing official Macro F0.5.
"""

import os
import sys
import time
from collections import defaultdict
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.normalization import normalize_record
from src.blocking import MultiPassBlocker
from src.evaluation import evaluate_predictions, compute_entity_f_beta


def compute_enhanced_features(s1: dict, target: dict) -> list:
    """
    Compute fast, highly discriminative pairwise features.
    """
    s1_name = s1.get("norm_name", "")
    t_name = target.get("norm_name", "")
    s1_tokens = s1.get("name_tokens", [])
    t_tokens = target.get("name_tokens", [])

    s1_addr = s1.get("norm_address", "")
    t_addr = target.get("norm_address", "")
    s1_addr_tokens = s1.get("address_tokens", [])
    t_addr_tokens = target.get("address_tokens", [])

    # 1. NAME FEATURES
    name_exact = 1.0 if s1_name and (s1_name == t_name) else 0.0

    s1_set = set(s1_tokens)
    t_set = set(t_tokens)
    name_inter = len(s1_set & t_set)
    name_union = len(s1_set | t_set)
    name_jaccard = (name_inter / float(name_union)) if name_union > 0 else 0.0
    name_recall = (name_inter / float(len(s1_set))) if s1_set else 0.0

    # First word (brand) match
    first_word_match = 1.0 if (s1_tokens and t_tokens and s1_tokens[0] == t_tokens[0]) else 0.0
    # Second word match
    sec_word_match = 1.0 if (len(s1_tokens) > 1 and len(t_tokens) > 1 and s1_tokens[1] == t_tokens[1]) else 0.0

    # Prefix match
    name_pfx4 = 1.0 if (len(s1_name) >= 4 and len(t_name) >= 4 and s1_name[:4] == t_name[:4]) else 0.0
    name_pfx6 = 1.0 if (len(s1_name) >= 6 and len(t_name) >= 6 and s1_name[:6] == t_name[:6]) else 0.0

    # Substring containment
    name_containment = 1.0 if (s1_name and t_name and (s1_name in t_name or t_name in s1_name)) else 0.0

    # Length Differences
    name_len_diff = abs(len(s1_name) - len(t_name))
    name_token_diff = abs(len(s1_tokens) - len(t_tokens))

    # Fast 3-gram character Jaccard (much faster than cosine and just as effective)
    def char_ngrams_set(text, n=3):
        return {text[i:i+n] for i in range(len(text)-n+1)} if len(text) >= n else set()

    c1 = char_ngrams_set(s1_name, 3)
    c2 = char_ngrams_set(t_name, 3)
    char_jaccard = len(c1 & c2) / float(len(c1 | c2)) if (c1 or c2) else 0.0

    # 2. ADDRESS FEATURES
    t_addr_empty = 1.0 if (not t_addr or target.get("is_address_empty", False)) else 0.0
    addr_exact = 1.0 if (not t_addr_empty and s1_addr == t_addr) else 0.0

    a1_set = set(s1_addr_tokens)
    a2_set = set(t_addr_tokens)
    addr_inter = len(a1_set & a2_set)
    addr_union = len(a1_set | a2_set)
    addr_jaccard = (addr_inter / float(addr_union)) if (addr_union > 0 and not t_addr_empty) else 0.0

    c1_a = char_ngrams_set(s1_addr, 3)
    c2_a = char_ngrams_set(t_addr, 3)
    addr_char_jaccard = len(c1_a & c2_a) / float(len(c1_a | c2_a)) if (c1_a or c2_a) else 0.0

    s1_nums = set(s1.get("address_numbers", []))
    t_nums = set(target.get("address_numbers", []))
    num_match = 1.0 if (s1_nums and t_nums and bool(s1_nums & t_nums)) else 0.0

    # Composite
    combined_jaccard = (0.6 * name_jaccard) + (0.4 * (addr_jaccard if not t_addr_empty else name_jaccard))

    return [
        name_exact,
        name_jaccard,
        name_recall,
        char_jaccard,
        first_word_match,
        sec_word_match,
        name_pfx4,
        name_pfx6,
        name_containment,
        float(name_len_diff),
        float(name_token_diff),
        addr_exact,
        addr_jaccard,
        addr_char_jaccard,
        num_match,
        t_addr_empty,
        combined_jaccard,
    ]


def run_validation():
    print("=" * 60)
    print("STEP 1: LOADING VALIDATION DATA")
    print("=" * 60)

    N_TOTAL_S1 = 4000
    gt_df = pd.read_csv("data/dataset/train/train_ground_truth.tsv", sep="\t", nrows=N_TOTAL_S1, keep_default_na=False)
    gt_map = {}
    needed_s1 = set()
    needed_targets = set()

    for _, row in gt_df.iterrows():
        sid = row["source1_entity_id"]
        tids = [x.strip() for x in row["matched_entity_ids"].split(",") if x.strip()]
        gt_map[sid] = tids
        needed_s1.add(sid)
        needed_targets.update(tids)

    print(f"Loading {len(needed_s1)} S1 entities...")
    s1_dict = {}
    for chunk in pd.read_csv("data/dataset/train/train_source1.tsv", sep="\t", chunksize=100_000, dtype=str, keep_default_na=False):
        m = chunk[chunk["entity_id"].isin(needed_s1)]
        for rec in m.to_dict(orient="records"):
            s1_dict[rec["entity_id"]] = normalize_record(rec)
        if len(s1_dict) >= len(needed_s1):
            break

    s1_list = [s1_dict[sid] for sid in gt_df["source1_entity_id"] if sid in s1_dict]

    # Load targets: true matches + 60,000 distractors
    print("Loading target records (true matches + distractors)...")
    target_dict = {}
    for s_file in ["train_source2.tsv", "train_source3.tsv"]:
        p = os.path.join("data/dataset/train", s_file)
        # Needed true targets
        for chunk in pd.read_csv(p, sep="\t", chunksize=100_000, dtype=str, keep_default_na=False):
            m = chunk[chunk["entity_id"].isin(needed_targets)]
            for rec in m.to_dict(orient="records"):
                target_dict[rec["entity_id"]] = normalize_record(rec)
            if len(target_dict) >= len(needed_targets):
                break

    # Add 40,000 background distractors per source to simulate real collision density
    for s_file in ["train_source2.tsv", "train_source3.tsv"]:
        p = os.path.join("data/dataset/train", s_file)
        sample = pd.read_csv(p, sep="\t", nrows=40_000, dtype=str, keep_default_na=False)
        for rec in sample.to_dict(orient="records"):
            eid = rec["entity_id"]
            if eid not in target_dict:
                target_dict[eid] = normalize_record(rec)

    print(f"Total Unique Targets Indexed: {len(target_dict):,}")

    # Index targets in MultiPassBlocker
    blocker = MultiPassBlocker(max_candidates_per_s1=30)
    blocker.index_targets(list(target_dict.values()))

    # Train / Val Split
    np.random.seed(42)
    shuffled_s1 = list(s1_list)
    np.random.shuffle(shuffled_s1)
    n_train = int(len(shuffled_s1) * 0.70)
    train_s1 = shuffled_s1[:n_train]
    val_s1 = shuffled_s1[n_train:]
    val_gt_map = {s["entity_id"]: gt_map[s["entity_id"]] for s in val_s1}

    print(f"Train S1: {len(train_s1):,} | Validation S1: {len(val_s1):,}")

    print("\n" + "=" * 60)
    print("STEP 2: MINING BLOCKER HARD NEGATIVES FOR TRAINING")
    print("=" * 60)

    X_train = []
    y_train = []
    num_pos = 0
    num_hard_neg = 0

    for s1 in train_s1:
        sid = s1["entity_id"]
        true_tids = set(gt_map.get(sid, []))
        cands = blocker.find_candidates(s1)

        # Ensure true targets are included
        for tid in true_tids:
            if tid in target_dict:
                feats = compute_enhanced_features(s1, target_dict[tid])
                X_train.append(feats)
                y_train.append(1.0)
                num_pos += 1

        # Blocker candidates that are NOT true targets are HARD NEGATIVES
        for cid in cands:
            if cid not in true_tids and cid in target_dict:
                feats = compute_enhanced_features(s1, target_dict[cid])
                X_train.append(feats)
                y_train.append(0.0)
                num_hard_neg += 1

    print(f"Training Pairs: {len(X_train):,} (Positives: {num_pos:,}, Hard Negatives: {num_hard_neg:,})")

    # Train LightGBM
    clf = LGBMClassifier(
        n_estimators=150,
        learning_rate=0.07,
        num_leaves=31,
        min_child_samples=20,
        random_state=42,
        n_jobs=-1,
        verbose=-1,
    )
    clf.fit(np.array(X_train), np.array(y_train))
    print("Classifier trained successfully.")

    print("\n" + "=" * 60)
    print("STEP 3: RUNNING INFERENCE ON VALIDATION SET")
    print("=" * 60)

    # Evaluate validation entities
    val_predictions_baseline = {}
    val_predictions_sota = {}

    for s1 in val_s1:
        sid = s1["entity_id"]
        cands = blocker.find_candidates(s1)
        if not cands:
            val_predictions_baseline[sid] = []
            val_predictions_sota[sid] = []
            continue

        cand_records = [target_dict[cid] for cid in cands if cid in target_dict]
        if not cand_records:
            val_predictions_baseline[sid] = []
            val_predictions_sota[sid] = []
            continue

        feat_matrix = [compute_enhanced_features(s1, t) for t in cand_records]
        probs = clf.predict_proba(np.array(feat_matrix))[:, 1]

        # 1. Baseline Strategy (flat threshold 0.80, no margin, no cap)
        matches_base = [cand_records[i]["entity_id"] for i, p in enumerate(probs) if p >= 0.80]
        val_predictions_baseline[sid] = matches_base

        # 2. SOTA Strategy:
        # - Rank by probability descending
        # - Top candidate must be >= 0.88 (singleton filter)
        # - Margin: only accept candidates within 0.06 of top candidate
        # - Cardinality: max 2 from S2, max 2 from S3
        scored_cands = sorted(zip(cand_records, probs), key=lambda x: x[1], reverse=True)
        top_p = scored_cands[0][1] if scored_cands else 0.0

        matches_sota = []
        if top_p >= 0.88:
            s2_count = 0
            s3_count = 0
            for cand, p in scored_cands:
                if p >= 0.85 and (top_p - p) <= 0.08:
                    cid = cand["entity_id"]
                    if cid.startswith("S2-") and s2_count < 2:
                        matches_sota.append(cid)
                        s2_count += 1
                    elif cid.startswith("S3-") and s3_count < 2:
                        matches_sota.append(cid)
                        s3_count += 1

        val_predictions_sota[sid] = matches_sota

    # Calculate metrics
    res_base = evaluate_predictions(val_predictions_baseline, val_gt_map, beta=0.5)
    res_sota = evaluate_predictions(val_predictions_sota, val_gt_map, beta=0.5)

    print("\n" + "=" * 60)
    print("COMPARATIVE EVALUATION RESULTS (OFFICIAL MACRO F0.5 METRIC)")
    print("=" * 60)
    print(f"{'Metric':<25} | {'Baseline':<12} | {'SOTA Approach':<12}")
    print("-" * 55)
    print(f"{'Macro F0.5':<25} | {res_base['macro_f0_5']:<12.4f} | {res_sota['macro_f0_5']:<12.4f}")
    print(f"{'Macro Precision':<25} | {res_base['macro_precision']:<12.4f} | {res_sota['macro_precision']:<12.4f}")
    print(f"{'Macro Recall':<25} | {res_base['macro_recall']:<12.4f} | {res_sota['macro_recall']:<12.4f}")
    print(f"{'Singleton Accuracy':<25} | {res_base['singleton_accuracy']:<12.4f} | {res_sota['singleton_accuracy']:<12.4f}")
    print("-" * 55)


if __name__ == "__main__":
    run_validation()
