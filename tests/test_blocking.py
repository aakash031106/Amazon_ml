"""
tests/test_blocking.py
Unit tests and real-data candidate recall validation for Member 2's Blocker.
"""

import sys
import os
import pandas as pd

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.normalization import normalize_record
from src.blocking import MultiPassBlocker, evaluate_blocking_recall

def test_blocking_on_synthetic():
    print("--- 1. Testing Blocker Logic on Synthetic Pairs ---")
    blocker = MultiPassBlocker(max_candidates_per_s1=10)

    targets = [
        normalize_record({
            "entity_id": "S2-001",
            "business_name": "Payne Enterprises Inc",
            "business_address": "3315 Fremont St, Peoria, IL",
            "country": "US"
        }),
        normalize_record({
            "entity_id": "S2-002",
            "business_name": "Random Bakery",
            "business_address": "123 Main St, Chicago, IL",
            "country": "US"
        }),
        normalize_record({
            "entity_id": "S3-003",
            "business_name": "Payne Enterpires", # typo
            "business_address": "Peoria, IL, 3315 Fremont St",
            "country": "US"
        }),
    ]

    blocker.index_targets(targets)

    # Query with S1
    s1 = normalize_record({
        "entity_id": "S1-999",
        "business_name": "Payne Énterprises",
        "business_address": "3315 Fremont Street, Peoria, IL",
        "country": "US"
    })

    candidates = blocker.find_candidates(s1)
    print(f"Candidates found for {s1['business_name']}: {candidates}")

    assert "S2-001" in candidates, "Failed: S2-001 should be a candidate!"
    assert "S3-003" in candidates, "Failed: S3-003 (typo) should be caught by 4-char prefix or number!"
    assert "S2-002" not in candidates, "S2-002 should not be a candidate!"
    print("Synthetic blocking test PASSED!\n")


def test_blocking_on_real_training_data():
    print("--- 2. Testing Blocker on Real Training Ground Truth ---")
    gt_path = "data/dataset/train/train_ground_truth.tsv"
    
    # Load 1,000 ground truth rows
    gt_df = pd.read_csv(gt_path, sep="\t", nrows=1000, keep_default_na=False)
    
    # Filter those with matches
    matched_gt = gt_df[gt_df["matched_entity_ids"].str.strip() != ""].head(200)
    
    gt_map = {}
    needed_s1 = set()
    needed_targets = set()
    
    for _, row in matched_gt.iterrows():
        s1_id = row["source1_entity_id"]
        tids = [x.strip() for x in row["matched_entity_ids"].split(",") if x.strip()]
        gt_map[s1_id] = tids
        needed_s1.add(s1_id)
        needed_targets.update(tids)

    print(f"Testing on {len(needed_s1)} real S1 entities with {len(needed_targets)} true matches...")

    # Load matching S1 records
    s1_records = []
    for chunk in pd.read_csv("data/dataset/train/train_source1.tsv", sep="\t", chunksize=50_000, dtype=str, keep_default_na=False):
        m = chunk[chunk["entity_id"].isin(needed_s1)]
        for rec in m.to_dict(orient="records"):
            s1_records.append(normalize_record(rec))
        if len(s1_records) >= len(needed_s1):
            break

    # Load matching targets + extra background targets from S2 and S3 to test precision/filtering
    target_records = []
    # 1. Load the exact true targets first so we test true recall
    for s_file in ["train_source2.tsv", "train_source3.tsv"]:
        p = os.path.join("data/dataset/train", s_file)
        for chunk in pd.read_csv(p, sep="\t", chunksize=100_000, dtype=str, keep_default_na=False):
            m = chunk[chunk["entity_id"].isin(needed_targets)]
            for rec in m.to_dict(orient="records"):
                target_records.append(normalize_record(rec))
            if len(target_records) >= len(needed_targets):
                break

    # 2. Add 20,000 background distractors
    for s_file in ["train_source2.tsv", "train_source3.tsv"]:
        p = os.path.join("data/dataset/train", s_file)
        sample = pd.read_csv(p, sep="\t", nrows=10_000, dtype=str, keep_default_na=False)
        for rec in sample.to_dict(orient="records"):
            target_records.append(normalize_record(rec))

    print(f"Loaded {len(s1_records)} S1 entities and indexed {len(target_records):,} target records.")

    blocker = MultiPassBlocker(max_candidates_per_s1=30)
    blocker.index_targets(target_records)

    # Generate candidate map
    candidate_map = {}
    for s1 in s1_records:
        candidate_map[s1["entity_id"]] = blocker.find_candidates(s1)

    # Filter GT map to only targets present in our target sample
    indexed_target_ids = set(blocker.targets.keys())
    eval_gt_map = {}
    for s1_id, tids in gt_map.items():
        present_tids = [t for t in tids if t in indexed_target_ids]
        if present_tids:
            eval_gt_map[s1_id] = present_tids

    results = evaluate_blocking_recall(candidate_map, eval_gt_map)
    print("\n--- BLOCKING EVALUATION RESULTS ---")
    for k, v in results.items():
        print(f"  {k}: {v}")

    assert results["candidate_recall"] > 0.85, f"Candidate recall too low: {results['candidate_recall']}"
    print("\nREAL DATA BLOCKING VALIDATION PASSED!")

if __name__ == "__main__":
    test_blocking_on_synthetic()
    test_blocking_on_real_training_data()
