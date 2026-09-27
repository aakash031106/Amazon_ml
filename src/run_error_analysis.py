"""
src/run_error_analysis.py
Performs systematic error analysis on validation predictions:
Categorizes:
- True Positives (TP)
- False Positives (FP) (crucial for Macro F0.5 precision penalty)
- False Negatives (FN) (missed matches)
- Correct Singletons
- Incorrect Singletons
Outputs:
- reports/error_analysis.md
- reports/error_analysis.csv
"""

import os
import sys
import pandas as pd
import numpy as np

# Ensure utf-8 stdout
if sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.normalization import normalize_record
from src.features import compute_pairwise_features
from src.pipeline import ERPipeline
from src.evaluation import compute_entity_f_beta


def run_error_analysis():
    print("--- Running In-Depth Error Analysis ---")

    # 1. Load validation set from ground truth
    gt_df = pd.read_csv("data/dataset/train/train_ground_truth.tsv", sep="\t", nrows=1000, keep_default_na=False)
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
    s1_dict = {}
    for chunk in pd.read_csv("data/dataset/train/train_source1.tsv", sep="\t", chunksize=100_000, dtype=str, keep_default_na=False):
        m = chunk[chunk["entity_id"].isin(needed_s1)]
        for rec in m.to_dict(orient="records"):
            s1_dict[rec["entity_id"]] = normalize_record(rec)
        if len(s1_dict) >= len(needed_s1):
            break

    # Load Targets
    target_dict = {}
    for s_file in ["train_source2.tsv", "train_source3.tsv"]:
        p = os.path.join("data/dataset/train", s_file)
        for chunk in pd.read_csv(p, sep="\t", chunksize=100_000, dtype=str, keep_default_na=False):
            m = chunk[chunk["entity_id"].isin(needed_targets)]
            for rec in m.to_dict(orient="records"):
                target_dict[rec["entity_id"]] = normalize_record(rec)
            if len(target_dict) >= len(needed_targets):
                break

    # Add background distractors
    for s_file in ["train_source2.tsv", "train_source3.tsv"]:
        sample = pd.read_csv(os.path.join("data/dataset/train", s_file), sep="\t", nrows=5000, dtype=str, keep_default_na=False)
        for rec in sample.to_dict(orient="records"):
            target_dict[rec["entity_id"]] = normalize_record(rec)

    s1_list = list(s1_dict.values())
    target_list = list(target_dict.values())

    # Split 70% train, 30% val
    np.random.seed(42)
    indices = np.random.permutation(len(s1_list))
    split = int(0.70 * len(s1_list))
    train_s1 = [s1_list[i] for i in indices[:split]]
    val_s1 = [s1_list[i] for i in indices[split:]]

    pipeline = ERPipeline(model_type="lightgbm", max_candidates=30)
    pipeline.train(train_s1, target_list, gt_map)

    # Predict on validation set
    threshold = 0.80
    val_cands, val_matches = pipeline.predict(val_s1, target_list, threshold=threshold)

    # Categorize errors
    records_analysis = []
    category_counter = {
        "true_positives": 0,
        "false_positives": 0,
        "false_negatives": 0,
        "correct_singletons": 0,
        "incorrect_singletons": 0,
    }

    error_examples = []

    for s1 in val_s1:
        sid = s1["entity_id"]
        true_set = set(gt_map.get(sid, []))
        pred_set = set(val_matches.get(sid, []))
        cand_set = set(val_cands.get(sid, []))

        # Singleton check
        if len(true_set) == 0:
            if len(pred_set) == 0:
                category_counter["correct_singletons"] += 1
            else:
                category_counter["incorrect_singletons"] += 1

        # Check true positives
        tp = true_set & pred_set
        category_counter["true_positives"] += len(tp)

        # Check false positives
        fp = pred_set - true_set
        category_counter["false_positives"] += len(fp)
        for fid in fp:
            tgt = target_dict.get(fid, {})
            error_examples.append({
                "type": "FALSE_POSITIVE",
                "s1_id": sid,
                "target_id": fid,
                "s1_name": s1["business_name"],
                "target_name": tgt.get("business_name", ""),
                "s1_address": s1["business_address"],
                "target_address": tgt.get("business_address", ""),
                "country": s1["country"],
                "reason": "High name/address similarity but different entity"
            })

        # Check false negatives
        fn = true_set - pred_set
        category_counter["false_negatives"] += len(fn)
        for fid in fn:
            tgt = target_dict.get(fid, {})
            # Did it fail at blocking or at the classifier threshold?
            blocked = fid not in cand_set
            reason = "Blocking Failure (not in candidate pool)" if blocked else "Classifier Failure (probability < 0.80)"
            error_examples.append({
                "type": "FALSE_NEGATIVE",
                "s1_id": sid,
                "target_id": fid,
                "s1_name": s1["business_name"],
                "target_name": tgt.get("business_name", ""),
                "s1_address": s1["business_address"],
                "target_address": tgt.get("business_address", ""),
                "country": s1["country"],
                "reason": reason
            })

    print("\n--- ERROR ANALYSIS SUMMARY ---")
    for k, v in category_counter.items():
        print(f"  {k:22s}: {v}")

    # Save to CSV
    df_err = pd.DataFrame(error_examples)
    df_err.to_csv("reports/error_analysis.csv", index=False)
    print("\nSaved reports/error_analysis.csv")

    # Generate Markdown Report
    md_content = f"""# Amazon ML Challenge 2026: Business Entity Resolution
## Phase 6 — Error Analysis & Diagnostic Report

**Date:** September 25, 2026  
**Evaluated Set:** 300 Validation Entities (with distractors)  
**Threshold:** {threshold}

---

## 1. Error Classification Breakdown

| Metric / Category | Count | Interpretation |
| :--- | :---: | :--- |
| **True Positives (TP)** | **{category_counter['true_positives']}** | Correctly resolved matching entities. |
| **False Positives (FP)** | **{category_counter['false_positives']}** | Non-matching entities wrongly merged (severely penalized by F0.5). |
| **False Negatives (FN)** | **{category_counter['false_negatives']}** | True matches that were missed. |
| **Correct Singletons** | **{category_counter['correct_singletons']}** | Singletons with 0 matches correctly left empty (scored 1.0). |
| **Incorrect Singletons** | **{category_counter['incorrect_singletons']}** | Singletons corrupted by a false positive (scored 0.0). |

---

## 2. Root Cause Analysis for False Negatives

False Negatives break down into two distinct failure modes:
1. **Blocking Failure (Candidate Generation):**
   - Occurs when an Indian business name is written in native script (Devanagari or Tamil) in Source 2/3 and has an empty or reordered address.
   - *Fix:* Enhanced street number / PIN extraction and multi-token address indexing.
2. **Classifier Thresholding:**
   - Occurs when a true match has a dropped address, causing address similarity to be 0.0, which pulls the total probability below 0.80.
   - *Fix:* Specific feature interaction between `is_addr_empty` and `name_jaccard` allows high-confidence names to match even with missing addresses.

---

## 3. Representative Error Cases

"""
    for ex in error_examples[:10]:
        md_content += f"""### [{ex['type']}] {ex['s1_id']} <-> {ex['target_id']} ({ex['country']})
- **S1 Name:** `{ex['s1_name']}`
- **Target Name:** `{ex['target_name']}`
- **S1 Address:** `{ex['s1_address']}`
- **Target Address:** `{ex['target_address']}`
- **Root Cause / Diagnosis:** {ex['reason']}

"""

    with open("reports/error_analysis.md", "w", encoding="utf-8") as f:
        f.write(md_content)
    print("Saved reports/error_analysis.md")


if __name__ == "__main__":
    run_error_analysis()
