"""
src/generate_test_submission_sota.py
Amazon ML Challenge 2026: Business Entity Resolution
State-of-the-Art (SOTA) Submission Pipeline:
1. Hard Negative Mining on Ground Truth (18,000 S1 + 100,000 targets)
2. 17-Dimensional High-Precision Feature Engineering
3. LightGBM Gradient Boosted Decision Trees
4. Precision-Dominant Post-Processing:
   - Probability ranking
   - Singleton threshold guard
   - Dynamic relative confidence margin (top_p - p <= 0.15)
   - Source-specific cardinality constraints (max 3 S2, max 3 S3)
5. Streaming Test Inference by Country
6. Official Validation & Submission Packaging
"""

import os
import sys
import time
import zipfile
import subprocess
from collections import defaultdict
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier

if sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.normalization import normalize_record
from src.blocking import MultiPassBlocker


def char_ngrams_set(text: str, n: int = 3) -> set:
    if not text or len(text) < n:
        return set()
    return {text[i:i+n] for i in range(len(text) - n + 1)}


def compute_sota_features(s1: dict, target: dict) -> list:
    """
    Computes 17 high-precision numerical similarity features.
    Optimized for execution speed during million-scale batch inference.
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

    # Prefix match (length 4 and 6)
    name_pfx4 = 1.0 if (len(s1_name) >= 4 and len(t_name) >= 4 and s1_name[:4] == t_name[:4]) else 0.0
    name_pfx6 = 1.0 if (len(s1_name) >= 6 and len(t_name) >= 6 and s1_name[:6] == t_name[:6]) else 0.0

    # Name containment (one is exact substring of another)
    name_containment = 1.0 if (s1_name and t_name and (s1_name in t_name or t_name in s1_name)) else 0.0

    # Length Differences
    name_len_diff = float(abs(len(s1_name) - len(t_name)))
    name_token_diff = float(abs(len(s1_tokens) - len(t_tokens)))

    # Fast 3-gram character Jaccard
    c1 = char_ngrams_set(s1_name, 3)
    c2 = char_ngrams_set(t_name, 3)
    char_jaccard = (len(c1 & c2) / float(len(c1 | c2))) if (c1 or c2) else 0.0

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
    addr_char_jaccard = (len(c1_a & c2_a) / float(len(c1_a | c2_a))) if (c1_a or c2_a and not t_addr_empty) else 0.0

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
        name_len_diff,
        name_token_diff,
        addr_exact,
        addr_jaccard,
        addr_char_jaccard,
        num_match,
        t_addr_empty,
        combined_jaccard,
    ]


def train_sota_classifier(n_s1_samples: int = 18_000) -> LGBMClassifier:
    """
    Train high-precision LightGBM model on real blocker hard negatives.
    """
    print(f"\n============================================================", flush=True)
    print(f"STEP 1: Training SOTA Classifier on {n_s1_samples:,} Ground Truth Entities", flush=True)
    print(f"============================================================", flush=True)
    t0 = time.time()

    gt_df = pd.read_csv("data/dataset/train/train_ground_truth.tsv", sep="\t", nrows=n_s1_samples, keep_default_na=False)
    gt_map = {}
    needed_s1 = set()
    needed_targets = set()

    for _, row in gt_df.iterrows():
        sid = row["source1_entity_id"]
        tids = [x.strip() for x in row["matched_entity_ids"].split(",") if x.strip()]
        gt_map[sid] = tids
        needed_s1.add(sid)
        needed_targets.update(tids)

    print(f"Loading {len(needed_s1):,} S1 entities from train_source1.tsv...", flush=True)
    s1_dict = {}
    for chunk in pd.read_csv("data/dataset/train/train_source1.tsv", sep="\t", chunksize=100_000, dtype=str, keep_default_na=False):
        m = chunk[chunk["entity_id"].isin(needed_s1)]
        for rec in m.to_dict(orient="records"):
            s1_dict[rec["entity_id"]] = normalize_record(rec)
        if len(s1_dict) >= len(needed_s1):
            break

    s1_list = [s1_dict[sid] for sid in gt_df["source1_entity_id"] if sid in s1_dict]
    print(f"Loaded {len(s1_list):,} normalized S1 records.", flush=True)

    # Load targets: true matches + 60,000 distractors per source
    print("Loading target records (true matches + realistic distractors)...", flush=True)
    target_dict = {}
    for s_file in ["train_source2.tsv", "train_source3.tsv"]:
        p = os.path.join("data/dataset/train", s_file)
        for chunk in pd.read_csv(p, sep="\t", chunksize=100_000, dtype=str, keep_default_na=False):
            m = chunk[chunk["entity_id"].isin(needed_targets)]
            for rec in m.to_dict(orient="records"):
                target_dict[rec["entity_id"]] = normalize_record(rec)
            if len(target_dict) >= len(needed_targets):
                break

    # Add 50,000 background distractors per source
    for s_file in ["train_source2.tsv", "train_source3.tsv"]:
        p = os.path.join("data/dataset/train", s_file)
        sample = pd.read_csv(p, sep="\t", nrows=50_000, dtype=str, keep_default_na=False)
        for rec in sample.to_dict(orient="records"):
            eid = rec["entity_id"]
            if eid not in target_dict:
                target_dict[eid] = normalize_record(rec)

    print(f"Total Unique Targets Indexed: {len(target_dict):,} in {time.time() - t0:.1f}s.", flush=True)

    # Index targets in MultiPassBlocker
    blocker = MultiPassBlocker(max_candidates_per_s1=30)
    blocker.index_targets(list(target_dict.values()))

    print("Mining Blocker Hard Negatives...", flush=True)
    t_mine = time.time()
    X_train = []
    y_train = []
    num_pos = 0
    num_hard_neg = 0

    for s1 in s1_list:
        sid = s1["entity_id"]
        true_tids = set(gt_map.get(sid, []))
        cands = blocker.find_candidates(s1)

        # True targets (Positive pairs)
        for tid in true_tids:
            if tid in target_dict:
                X_train.append(compute_sota_features(s1, target_dict[tid]))
                y_train.append(1.0)
                num_pos += 1

        # Blocker candidate that is NOT a true target (Hard Negative pair)
        for cid in cands:
            if cid not in true_tids and cid in target_dict:
                X_train.append(compute_sota_features(s1, target_dict[cid]))
                y_train.append(0.0)
                num_hard_neg += 1

    print(f"Mined {len(X_train):,} training pairs ({num_pos:,} Positives, {num_hard_neg:,} Hard Negatives) in {time.time() - t_mine:.1f}s.", flush=True)

    # Train LightGBM
    print("Fitting LightGBM Classifier...", flush=True)
    clf = LGBMClassifier(
        n_estimators=180,
        learning_rate=0.07,
        num_leaves=31,
        min_child_samples=25,
        subsample=0.85,
        colsample_bytree=0.85,
        random_state=42,
        n_jobs=-1,
        verbose=-1,
    )
    clf.fit(np.array(X_train), np.array(y_train))
    print(f"LightGBM Classifier trained successfully in {time.time() - t0:.1f}s total.", flush=True)

    del X_train, y_train, target_dict, blocker, s1_dict, s1_list
    import gc
    gc.collect()

    return clf


def run_country_inference_sota(
    country: str,
    clf: LGBMClassifier,
    cand_file,
    match_file,
    min_prob: float = 0.50,
    prob_margin: float = 0.15,
    max_per_source: int = 3,
):
    """
    Process all test records for a specific country with SOTA precision filtering.
    """
    print(f"\n============================================================", flush=True)
    print(f"STARTING TEST INFERENCE: {country.upper()}", flush=True)
    print(f"Params: min_prob={min_prob}, margin={prob_margin}, max_per_source={max_per_source}", flush=True)
    print(f"============================================================", flush=True)
    t_start = time.time()

    # 1. Load and Index Targets (S2 + S3) for this country
    print(f"Loading {country} test target pool (Source 2 + Source 3)...", flush=True)
    target_dict = {}
    target_list = []

    for s_file in ["test_source2.tsv", "test_source3.tsv"]:
        p = os.path.join("data/dataset/test", s_file)
        for chunk in pd.read_csv(p, sep="\t", chunksize=250_000, dtype=str, keep_default_na=False):
            c_chunk = chunk[chunk["country"].str.strip() == country]
            for rec in c_chunk.to_dict(orient="records"):
                eid = rec["entity_id"]
                if eid not in target_dict:
                    normed = normalize_record(rec)
                    target_dict[eid] = normed
                    target_list.append(normed)

    print(f"Loaded {len(target_list):,} unique {country} targets in {time.time() - t_start:.1f}s.", flush=True)

    # Build blocker index
    blocker = MultiPassBlocker(max_candidates_per_s1=30)
    blocker.index_targets(target_list)

    del target_list
    import gc
    gc.collect()

    # 2. Stream Test S1 in Batches
    s1_path = os.path.join("data/dataset/test", "test_source1.tsv")
    total_processed = 0
    total_matched_entities = 0
    total_predicted_matches = 0

    CHUNK_S1 = 25_000

    for chunk in pd.read_csv(s1_path, sep="\t", chunksize=CHUNK_S1, dtype=str, keep_default_na=False):
        c_chunk = chunk[chunk["country"].str.strip() == country]
        if c_chunk.empty:
            continue

        batch_s1_ids = []
        batch_cands_per_s1 = []
        batch_feats = []
        batch_pair_index = []  # (s1_idx, cid)

        for rec in c_chunk.to_dict(orient="records"):
            s1 = normalize_record(rec)
            sid = s1["entity_id"]
            candidates = blocker.find_candidates(s1)

            batch_s1_ids.append(sid)
            batch_cands_per_s1.append(candidates)

            s1_idx = len(batch_s1_ids) - 1
            for cid in candidates:
                if cid in target_dict:
                    f = compute_sota_features(s1, target_dict[cid])
                    batch_feats.append(f)
                    batch_pair_index.append((s1_idx, cid))

        # Batch predict all candidate pairs
        if batch_feats:
            all_probs = clf.predict_proba(np.array(batch_feats))[:, 1]
        else:
            all_probs = np.array([])

        # Group scored candidates per S1
        s1_candidate_scores = defaultdict(list)
        for pair_pos, (s1_idx, cid) in enumerate(batch_pair_index):
            s1_candidate_scores[s1_idx].append((cid, all_probs[pair_pos]))

        # High-Precision Calibrated Post-Processing per S1 entity
        for i, sid in enumerate(batch_s1_ids):
            cands = batch_cands_per_s1[i]
            scored_pairs = s1_candidate_scores[i]

            matched = []
            if scored_pairs:
                scored_pairs.sort(key=lambda x: x[1], reverse=True)
                top_p = scored_pairs[0][1]

                # Singleton guard: if top candidate is not high confidence, predict empty
                if top_p >= min_prob:
                    s2_count = 0
                    s3_count = 0
                    for cid, p in scored_pairs:
                        if p >= min_prob and (top_p - p) <= prob_margin:
                            if cid.startswith("S2-") and s2_count < max_per_source:
                                matched.append(cid)
                                s2_count += 1
                            elif cid.startswith("S3-") and s3_count < max_per_source:
                                matched.append(cid)
                                s3_count += 1

            cand_file.write(f"{sid}\t{','.join(cands)}\n")
            match_file.write(f"{sid}\t{','.join(matched)}\n")

            if matched:
                total_matched_entities += 1
                total_predicted_matches += len(matched)
            total_processed += 1

        print(f"  [{country}] Processed {total_processed:,} S1 entities... (matched: {total_matched_entities:,}, total matches: {total_predicted_matches:,})", flush=True)

    cand_file.flush()
    match_file.flush()

    avg_matches = (total_predicted_matches / total_matched_entities) if total_matched_entities > 0 else 0.0
    print(f"Completed {country} in {time.time() - t_start:.1f}s! Total S1: {total_processed:,} | Entities with matches: {total_matched_entities:,} | Avg matches per matched S1: {avg_matches:.2f}", flush=True)

    del target_dict, blocker
    gc.collect()


def main():
    start_time = time.time()
    os.makedirs("outputs", exist_ok=True)
    cand_out = "outputs/candidate_pairs.tsv"
    match_out = "outputs/matching_results.tsv"

    print("=" * 65, flush=True)
    print("AMAZON ML CHALLENGE 2026: SOTA SUBMISSION PIPELINE EXECUTION", flush=True)
    print("Target Metric: Macro F0.5 (Optimized for High Precision & Cardinality)", flush=True)
    print("=" * 65, flush=True)

    # 1. Train Production Classifier with Hard Negative Mining
    clf = train_sota_classifier(n_s1_samples=18_000)

    # 2. Run Inference for all 3 Countries
    countries = ["France", "India", "US"]

    with open(cand_out, "w", encoding="utf-8") as f_cand, open(match_out, "w", encoding="utf-8") as f_match:
        f_cand.write("source1_entity_id\tcandidate_entity_ids\n")
        f_match.write("source1_entity_id\tmatched_entity_ids\n")

        for country in countries:
            run_country_inference_sota(
                country=country,
                clf=clf,
                cand_file=f_cand,
                match_file=f_match,
                min_prob=0.50,
                prob_margin=0.15,
                max_per_source=3,
            )

    print(f"\nAll 1.73M test predictions generated in {time.time() - start_time:.1f}s!", flush=True)
    print(f"Saved: {cand_out}", flush=True)
    print(f"Saved: {match_out}", flush=True)

    # 3. Validate Submission against Official Rules
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

    # 4. Package Submission Zip
    zip_path = "outputs/submission.zip"
    print(f"\nCompressing outputs to {zip_path}...", flush=True)
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.write(cand_out, arcname="candidate_pairs.tsv")
        zf.write(match_out, arcname="matching_results.tsv")

    size_mb = os.path.getsize(zip_path) / (1024 * 1024)
    print(f"Submission archive ready: {zip_path} ({size_mb:.2f} MB)", flush=True)
    print(f"TOTAL PIPELINE COMPLETED IN {time.time() - start_time:.1f}s!", flush=True)


if __name__ == "__main__":
    main()
