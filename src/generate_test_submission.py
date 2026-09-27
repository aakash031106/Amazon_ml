"""
src/generate_test_submission.py
Amazon ML Challenge 2026: Business Entity Resolution
Generates the full competition submission files for all 1.73M test entities.

Outputs:
  - outputs/candidate_pairs.tsv
  - outputs/matching_results.tsv
"""

import os
import sys
import time
import subprocess
import pandas as pd
import numpy as np

# Ensure utf-8 stdout
if sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.normalization import normalize_record
from src.blocking import MultiPassBlocker
from src.features import compute_pairwise_features
from src.model import ERMatchClassifier, FEATURE_COLUMNS
from src.pipeline import ERPipeline


def train_production_classifier(sample_size: int = 5000) -> ERMatchClassifier:
    """
    Train high-accuracy LightGBM model on a balanced sample from train_ground_truth.tsv.
    """
    print(f"\n--- [Step 1] Training Classifier on {sample_size:,} Ground Truth Entities ---")
    t0 = time.time()

    gt_df = pd.read_csv("data/dataset/train/train_ground_truth.tsv", sep="\t", nrows=sample_size, keep_default_na=False)
    gt_map = {}
    needed_s1 = set()
    needed_targets = set()

    for _, row in gt_df.iterrows():
        sid = row["source1_entity_id"]
        tids = [x.strip() for x in row["matched_entity_ids"].split(",") if x.strip()]
        gt_map[sid] = tids
        needed_s1.add(sid)
        needed_targets.update(tids)

    # Load S1
    s1_records = []
    for chunk in pd.read_csv("data/dataset/train/train_source1.tsv", sep="\t", chunksize=100_000, dtype=str, keep_default_na=False):
        m = chunk[chunk["entity_id"].isin(needed_s1)]
        for rec in m.to_dict(orient="records"):
            s1_records.append(normalize_record(rec))
        if len(s1_records) >= len(needed_s1):
            break

    # Load Targets
    target_records = []
    for s_file in ["train_source2.tsv", "train_source3.tsv"]:
        p = os.path.join("data/dataset/train", s_file)
        for chunk in pd.read_csv(p, sep="\t", chunksize=100_000, dtype=str, keep_default_na=False):
            m = chunk[chunk["entity_id"].isin(needed_targets)]
            for rec in m.to_dict(orient="records"):
                target_records.append(normalize_record(rec))
            if len(target_records) >= len(needed_targets):
                break

    # Add background distractors
    for s_file in ["train_source2.tsv", "train_source3.tsv"]:
        sample = pd.read_csv(os.path.join("data/dataset/train", s_file), sep="\t", nrows=15000, dtype=str, keep_default_na=False)
        for rec in sample.to_dict(orient="records"):
            target_records.append(normalize_record(rec))

    # Deduplicate targets
    seen = set()
    unique_targets = []
    for t in target_records:
        if t["entity_id"] not in seen:
            seen.add(t["entity_id"])
            unique_targets.append(t)

    print(f"Loaded {len(s1_records):,} S1 and {len(unique_targets):,} target records in {time.time() - t0:.1f}s.")

    # Index and build training pairs
    blocker = MultiPassBlocker(max_candidates_per_s1=30)
    blocker.index_targets(unique_targets)
    target_dict = {t["entity_id"]: t for t in unique_targets}

    X_list = []
    y_list = []
    for s1 in s1_records:
        sid = s1["entity_id"]
        true_tids = set(gt_map.get(sid, []))
        cands = blocker.find_candidates(s1)
        for cid in cands:
            if cid in target_dict:
                f = compute_pairwise_features(s1, target_dict[cid])
                X_list.append([f[col] for col in FEATURE_COLUMNS])
                y_list.append(1.0 if cid in true_tids else 0.0)

    clf = ERMatchClassifier(model_type="lightgbm", random_state=42)
    clf.fit(np.array(X_list), np.array(y_list))
    print(f"Classifier trained on {len(X_list):,} pairs in {time.time() - t0:.1f}s.")
    return clf


def run_country_inference(
    country: str,
    clf: ERMatchClassifier,
    threshold: float = 0.80,
    max_cands: int = 30,
    cand_file=None,
    match_file=None,
):
    """
    Process all test records for a specific country in a streaming fashion.
    """
    print(f"\n============================================================")
    print(f"Processing Test Country: {country.upper()}")
    print(f"============================================================")
    t_start = time.time()

    # 1. Load and Index Test Targets (S2 + S3) for this country only
    print(f"Loading test targets for {country}...")
    target_dict = {}
    target_list = []

    for s_file in ["test_source2.tsv", "test_source3.tsv"]:
        p = os.path.join("data/dataset/test", s_file)
        for chunk in pd.read_csv(p, sep="\t", chunksize=250_000, dtype=str, keep_default_na=False):
            country_chunk = chunk[chunk["country"].str.strip() == country]
            for rec in country_chunk.to_dict(orient="records"):
                eid = rec["entity_id"]
                if eid not in target_dict:
                    normed = normalize_record(rec)
                    target_dict[eid] = normed
                    target_list.append(normed)

    print(f"Loaded {len(target_list):,} unique {country} target records in {time.time() - t_start:.1f}s.")

    blocker = MultiPassBlocker(max_candidates_per_s1=max_cands)
    blocker.index_targets(target_list)

    del target_list
    import gc
    gc.collect()

    # 2. Stream Test S1 in Chunks
    s1_path = os.path.join("data/dataset/test", "test_source1.tsv")
    total_processed = 0
    total_matched = 0

    CHUNK_S1 = 25_000  # Process 25K S1 at a time

    for chunk in pd.read_csv(s1_path, sep="\t", chunksize=CHUNK_S1, dtype=str, keep_default_na=False):
        c_chunk = chunk[chunk["country"].str.strip() == country]
        if c_chunk.empty:
            continue

        # Batch: collect all candidate pairs for this chunk, then do ONE predict call
        batch_s1_ids = []
        batch_cands_per_s1 = []  # list of candidate lists
        batch_feats = []          # flat list of all feature vectors
        batch_pair_index = []     # (s1_idx, cid) to reconstruct after prediction

        for rec in c_chunk.to_dict(orient="records"):
            s1 = normalize_record(rec)
            sid = s1["entity_id"]
            candidates = blocker.find_candidates(s1)

            batch_s1_ids.append(sid)
            batch_cands_per_s1.append(candidates)

            for cid in candidates:
                if cid in target_dict:
                    f = compute_pairwise_features(s1, target_dict[cid])
                    batch_feats.append([f[col] for col in FEATURE_COLUMNS])
                    batch_pair_index.append((len(batch_s1_ids) - 1, cid))

        # One batch predict for all pairs in this chunk
        if batch_feats:
            all_probs = clf.predict_proba(np.array(batch_feats))
        else:
            all_probs = np.array([])

        # Build per-S1 match sets from flat predictions
        s1_matches = {i: [] for i in range(len(batch_s1_ids))}
        for pair_pos, (s1_idx, cid) in enumerate(batch_pair_index):
            if all_probs[pair_pos] >= threshold:
                s1_matches[s1_idx].append(cid)

        # Write results
        for i, sid in enumerate(batch_s1_ids):
            cands = batch_cands_per_s1[i]
            matched = s1_matches[i]
            cand_file.write(f"{sid}\t{','.join(cands)}\n")
            match_file.write(f"{sid}\t{','.join(matched)}\n")
            if matched:
                total_matched += 1
            total_processed += 1

        print(f"  Processed {total_processed:,} {country} S1 entities... (matched: {total_matched:,})", flush=True)

    cand_file.flush()
    match_file.flush()
    print(f"Finished {country} in {time.time() - t_start:.1f}s: {total_processed:,} S1 entities processed.", flush=True)
    del target_dict
    del blocker
    gc.collect()


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Generate full competition test submission")
    parser.add_argument("--limit", type=int, default=None, help="Optional limit on S1 entities per country for testing")
    parser.add_argument("--countries", nargs="+", default=["France", "US", "India"], help="List of countries to process")
    parser.add_argument("--threshold", type=float, default=0.80, help="Classification probability threshold")
    parser.add_argument("--append", action="store_true", help="Append to existing output files instead of overwriting")
    args = parser.parse_args()

    start_time = time.time()
    os.makedirs("outputs", exist_ok=True)
    cand_out = "outputs/candidate_pairs.tsv"
    match_out = "outputs/matching_results.tsv"

    print("=" * 60, flush=True)
    print("AMAZON ML CHALLENGE 2026: TEST SET SUBMISSION GENERATION", flush=True)
    print(f"Countries: {args.countries} | Limit: {args.limit} | Threshold: {args.threshold} | Append: {args.append}", flush=True)
    print("=" * 60, flush=True)

    # 1. Train Production Classifier
    clf = train_production_classifier(sample_size=6000)

    # 2. Open Output TSVs
    mode = "a" if args.append else "w"
    with open(cand_out, mode, encoding="utf-8") as f_cand, open(match_out, mode, encoding="utf-8") as f_match:
        if not args.append:
            f_cand.write("source1_entity_id\tcandidate_entity_ids\n")
            f_match.write("source1_entity_id\tmatched_entity_ids\n")

        for c in args.countries:
            run_country_inference(
                country=c,
                clf=clf,
                threshold=args.threshold,
                max_cands=30,
                cand_file=f_cand,
                match_file=f_match
            )

    print(f"\nAll test predictions generated in {time.time() - start_time:.1f}s.", flush=True)
    print(f"Saved: {cand_out}", flush=True)
    print(f"Saved: {match_out}", flush=True)

    # 4. Run Official Validator
    if not args.limit:
        print("\n--- Running Official Submission Validator ---", flush=True)
        val_cmd = [
            "python", "utils/validate_submission.py",
            "--matching", match_out,
            "--candidate", cand_out,
            "--test-dir", "data/dataset/test"
        ]
        res = subprocess.run(val_cmd, capture_output=True, text=True)
        print(res.stdout, flush=True)
        if res.stderr:
            print(res.stderr, flush=True)


if __name__ == "__main__":
    main()

