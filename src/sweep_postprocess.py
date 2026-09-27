"""
Sweep post-processing parameters on validation set to find the optimal combination.
"""
import os, sys, time
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.normalization import normalize_record
from src.blocking import MultiPassBlocker
from src.evaluation import evaluate_predictions, compute_entity_f_beta
from src.validate_sota_approach import compute_enhanced_features

def sweep():
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

    s1_dict = {}
    for chunk in pd.read_csv("data/dataset/train/train_source1.tsv", sep="\t", chunksize=100_000, dtype=str, keep_default_na=False):
        m = chunk[chunk["entity_id"].isin(needed_s1)]
        for rec in m.to_dict(orient="records"):
            s1_dict[rec["entity_id"]] = normalize_record(rec)
        if len(s1_dict) >= len(needed_s1):
            break

    s1_list = [s1_dict[sid] for sid in gt_df["source1_entity_id"] if sid in s1_dict]

    target_dict = {}
    for s_file in ["train_source2.tsv", "train_source3.tsv"]:
        p = os.path.join("data/dataset/train", s_file)
        for chunk in pd.read_csv(p, sep="\t", chunksize=100_000, dtype=str, keep_default_na=False):
            m = chunk[chunk["entity_id"].isin(needed_targets)]
            for rec in m.to_dict(orient="records"):
                target_dict[rec["entity_id"]] = normalize_record(rec)
            if len(target_dict) >= len(needed_targets):
                break

    for s_file in ["train_source2.tsv", "train_source3.tsv"]:
        p = os.path.join("data/dataset/train", s_file)
        sample = pd.read_csv(p, sep="\t", nrows=40_000, dtype=str, keep_default_na=False)
        for rec in sample.to_dict(orient="records"):
            eid = rec["entity_id"]
            if eid not in target_dict:
                target_dict[eid] = normalize_record(rec)

    blocker = MultiPassBlocker(max_candidates_per_s1=30)
    blocker.index_targets(list(target_dict.values()))

    np.random.seed(42)
    shuffled_s1 = list(s1_list)
    np.random.shuffle(shuffled_s1)
    n_train = int(len(shuffled_s1) * 0.70)
    train_s1 = shuffled_s1[:n_train]
    val_s1 = shuffled_s1[n_train:]
    val_gt_map = {s["entity_id"]: gt_map[s["entity_id"]] for s in val_s1}

    X_train = []
    y_train = []
    for s1 in train_s1:
        sid = s1["entity_id"]
        true_tids = set(gt_map.get(sid, []))
        cands = blocker.find_candidates(s1)
        for tid in true_tids:
            if tid in target_dict:
                X_train.append(compute_enhanced_features(s1, target_dict[tid]))
                y_train.append(1.0)
        for cid in cands:
            if cid not in true_tids and cid in target_dict:
                X_train.append(compute_enhanced_features(s1, target_dict[cid]))
                y_train.append(0.0)

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

    # Precompute validation candidates and probabilities
    val_scored = {}
    for s1 in val_s1:
        sid = s1["entity_id"]
        cands = blocker.find_candidates(s1)
        cand_records = [target_dict[cid] for cid in cands if cid in target_dict]
        if not cand_records:
            val_scored[sid] = []
            continue
        feat_matrix = [compute_enhanced_features(s1, t) for t in cand_records]
        probs = clf.predict_proba(np.array(feat_matrix))[:, 1]
        pairs = sorted(zip([c["entity_id"] for c in cand_records], probs), key=lambda x: x[1], reverse=True)
        val_scored[sid] = pairs

    print("\n--- Sweeping Post-Processing Parameters ---", flush=True)
    best_f = 0.0
    best_cfg = None

    for min_th in [0.50, 0.60, 0.65, 0.70, 0.75, 0.80]:
        for margin in [0.10, 0.15, 0.20, 0.25, 0.35, 1.0]:
            for max_per_source in [2, 3, 4, 10]:
                preds = {}
                for sid, pairs in val_scored.items():
                    if not pairs:
                        preds[sid] = []
                        continue
                    top_p = pairs[0][1]
                    matches = []
                    s2_c = 0
                    s3_c = 0
                    for cid, p in pairs:
                        if p >= min_th and (top_p - p) <= margin:
                            if cid.startswith("S2-") and s2_c < max_per_source:
                                matches.append(cid)
                                s2_c += 1
                            elif cid.startswith("S3-") and s3_c < max_per_source:
                                matches.append(cid)
                                s3_c += 1
                    preds[sid] = matches

                res = evaluate_predictions(preds, val_gt_map, beta=0.5)
                f05 = res["macro_f0_5"]
                if f05 > best_f:
                    best_f = f05
                    best_cfg = (min_th, margin, max_per_source, res)
                    print(f"NEW BEST -> Macro F0.5: {f05:.4f} | Prec: {res['macro_precision']:.4f} | Rec: {res['macro_recall']:.4f} | Params: min_th={min_th}, margin={margin}, max_per_src={max_per_source}", flush=True)

    print("\n" + "=" * 60, flush=True)
    print(f"Optimal Configuration: min_th={best_cfg[0]}, margin={best_cfg[1]}, max_per_source={best_cfg[2]}", flush=True)
    print(f"Optimal Macro F0.5: {best_f:.4f}", flush=True)
    print("=" * 60, flush=True)

if __name__ == "__main__":
    sweep()
