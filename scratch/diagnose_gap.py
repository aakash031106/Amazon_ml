import os
import sys
import time
sys.path.insert(0, os.getcwd())
import polars as pl
from collections import defaultdict
from rapidfuzz import fuzz, distance

from src.run_sota_production import clean_text, extract_core_brand, extract_numbers_set, extract_us_state, extract_addr_tokens, score_pair, STOP_ADDR
from src.evaluation import evaluate_predictions

def diagnose():
    print("Loading 5,000 ground truth entities...")
    gt_df = pl.read_csv("data/dataset/train/train_ground_truth.tsv", separator="\t", n_rows=5000)
    gt_map = {}
    needed_targets = set()
    for r in gt_df.iter_rows(named=True):
        sid = r["source1_entity_id"]
        m_str = r["matched_entity_ids"] or ""
        tids = [x.strip() for x in m_str.split(",") if x.strip()]
        gt_map[sid] = tids
        needed_targets.update(tids)

    s1_all = pl.read_csv("data/dataset/train/train_source1.tsv", separator="\t")
    s1_df = s1_all.filter(pl.col("entity_id").is_in(list(gt_map.keys())))
    del s1_all

    # Load targets: all true targets + 40,000 distractors per source
    target_dfs = []
    for s in ["train_source2.tsv", "train_source3.tsv"]:
        p = f"data/dataset/train/{s}"
        df = pl.read_csv(p, separator="\t")
        m = df.filter(pl.col("entity_id").is_in(list(needed_targets)))
        d = df.head(40000)
        target_dfs.append(m)
        target_dfs.append(d)

    target_df = pl.concat(target_dfs).unique(subset=["entity_id"])
    print(f"Loaded {len(s1_df)} S1 entities and {len(target_df)} target entities.")

    # Target indexing
    t_by_c = defaultdict(dict)
    index_stem_by_c = defaultdict(lambda: defaultdict(list))
    index_pfx4_by_c = defaultdict(lambda: defaultdict(list))
    index_pin_by_c = defaultdict(lambda: defaultdict(list))
    index_num_addr_by_c = defaultdict(lambda: defaultdict(list))
    index_num_state_by_c = defaultdict(lambda: defaultdict(list))

    for r in target_df.iter_rows(named=True):
        eid = r["entity_id"]
        c = r["country"].strip()
        n_name = clean_text(r["business_name"], c)
        core_brand = extract_core_brand(n_name)
        n_addr = clean_text(r["business_address"], c)
        nums = extract_numbers_set(r["business_address"])
        state = extract_us_state(n_addr) if c == "US" else ""
        is_empty = not bool(n_addr)

        t_by_c[c][eid] = {
            "entity_id": eid,
            "core_brand": core_brand,
            "norm_addr": n_addr,
            "nums": nums,
            "state": state,
            "no_space": n_name.replace(" ", ""),
            "is_addr_empty": is_empty,
        }

        stems = [w for w in core_brand.split() if len(w) >= 2]
        for s in stems:
            index_stem_by_c[c][s].append(eid)
        if stems and len(stems[0]) >= 4:
            index_pfx4_by_c[c][stems[0][:4]].append(eid)

        addr_tokens = extract_addr_tokens(n_addr)
        for n in nums:
            if n >= 10000:
                index_pin_by_c[c][n].append(eid)
            if state:
                index_num_state_by_c[c][(n, state)].append(eid)
            for at in addr_tokens[:2]:
                index_num_addr_by_c[c][(n, at)].append(eid)

    # Now evaluate candidate recall & scorer
    total_true_matches = 0
    captured_in_cands = 0
    captured_in_matches = 0

    preds = {}
    candidate_map = {}

    for r in s1_df.iter_rows(named=True):
        eid = r["entity_id"]
        c = r["country"].strip()
        n_name = clean_text(r["business_name"], c)
        core_brand = extract_core_brand(n_name)
        n_addr = clean_text(r["business_address"], c)
        nums = extract_numbers_set(r["business_address"])
        state = extract_us_state(n_addr) if c == "US" else ""
        ns = n_name.replace(" ", "")

        s1_obj = {
            "core_brand": core_brand,
            "norm_addr": n_addr,
            "nums": nums,
            "state": state,
            "no_space": ns,
        }

        true_tids = set(gt_map.get(eid, []))
        total_true_matches += len(true_tids)

        idx_stem = index_stem_by_c[c]
        idx_pfx4 = index_pfx4_by_c[c]
        idx_pin = index_pin_by_c[c]
        idx_num_state = index_num_state_by_c[c]
        idx_num_addr = index_num_addr_by_c[c]
        tgt_dict = t_by_c[c]

        stems = [w for w in core_brand.split() if len(w) >= 2]
        stems.sort(key=lambda s: len(idx_stem.get(s, [])))

        cands = set()
        for s in stems:
            sc = idx_stem.get(s, [])
            if len(sc) <= 200:
                cands.update(sc)
            elif state:
                cands.update([tid for tid in sc if tgt_dict[tid]["state"] == state][:60])
            else:
                cands.update(sc[:60])
            if len(cands) >= 50:
                break

        if len(cands) < 30 and stems and len(stems[0]) >= 4:
            pc = idx_pfx4.get(stems[0][:4], [])
            if len(pc) <= 200:
                cands.update(pc)
            elif state:
                cands.update([tid for tid in pc if tgt_dict[tid]["state"] == state][:50])
            else:
                cands.update(pc[:50])

        addr_tokens = extract_addr_tokens(n_addr)
        for n in nums:
            if n >= 10000:
                cands.update(idx_pin.get(n, [])[:30])
            if state:
                cands.update(idx_num_state.get((n, state), [])[:30])
            for at in addr_tokens[:2]:
                cands.update(idx_num_addr.get((n, at), [])[:30])

        captured_in_cands += len(true_tids & cands)
        candidate_map[eid] = list(cands)

        # Score candidates
        scored = []
        for tid in cands:
            t_obj = tgt_dict[tid]
            s = score_pair(s1_obj, t_obj)
            if s >= 0.80:
                scored.append((tid, s))

        matched = []
        if scored:
            scored.sort(key=lambda x: x[1], reverse=True)
            top_s = scored[0][1]
            s2_c, s3_c = 0, 0
            for tid, s in scored:
                if s >= 0.80 and (top_s - s) <= 0.05:
                    if tid.startswith("S2-") and s2_c < 3:
                        matched.append(tid)
                        s2_c += 1
                    elif tid.startswith("S3-") and s3_c < 3:
                        matched.append(tid)
                        s3_c += 1

        captured_in_matches += len(true_tids & set(matched))
        preds[eid] = matched

    # Print Diagnostic Results
    res = evaluate_predictions(preds, gt_map, beta=0.5)
    print("\n" + "=" * 60)
    print(f"DIAGNOSTIC REPORT ON 5,000 REAL ENTITIES (with 80,000 distractors):")
    print("=" * 60)
    print(f"Total True Matches: {total_true_matches:,}")
    print(f"Captured in Candidate Pool: {captured_in_cands:,} ({captured_in_cands/total_true_matches*100:.2f}%)")
    print(f"Captured in Final Predictions: {captured_in_matches:,} ({captured_in_matches/total_true_matches*100:.2f}%)")
    print(f"Macro F0.5: {res['macro_f0_5']:.4f}")
    print(f"Macro Precision: {res['macro_precision']:.4f}")
    print(f"Macro Recall: {res['macro_recall']:.4f}")
    print(f"Singleton Accuracy: {res['singleton_accuracy']:.4f}")
    print("=" * 60)

    # Let's inspect 10 False Positives and 10 False Negatives
    print("\nInspecting False Negatives (True matches missed):")
    fn_shown = 0
    for eid, true_tids in gt_map.items():
        if not true_tids:
            continue
        p = set(preds.get(eid, []))
        c = set(candidate_map.get(eid, []))
        missed = set(true_tids) - p
        if missed:
            fn_shown += 1
            tid = list(missed)[0]
            in_cands = tid in c
            s1_rec = s1_df.filter(pl.col("entity_id") == eid).row(0, named=True)
            t_rec = target_df.filter(pl.col("entity_id") == tid).row(0, named=True)
            print(f"\n[FN #{fn_shown}] {eid} <-> {tid} (In Candidate Pool: {in_cands})")
            print(f"  S1: '{s1_rec['business_name']}' | '{s1_rec['business_address']}'")
            print(f"  T : '{t_rec['business_name']}' | '{t_rec['business_address']}'")
            if in_cands:
                s1_o = {
                    "core_brand": extract_core_brand(clean_text(s1_rec["business_name"], s1_rec["country"])),
                    "norm_addr": clean_text(s1_rec["business_address"], s1_rec["country"]),
                    "nums": extract_numbers_set(s1_rec["business_address"]),
                    "state": extract_us_state(clean_text(s1_rec["business_address"], s1_rec["country"])) if s1_rec["country"] == "US" else "",
                    "no_space": clean_text(s1_rec["business_name"], s1_rec["country"]).replace(" ", ""),
                }
                t_o = t_by_c[s1_rec["country"].strip()][tid]
                sc = score_pair(s1_o, t_o)
                print(f"  Score: {sc:.4f}")
            if fn_shown >= 5:
                break

    print("\nInspecting False Positives (Wrong matches predicted):")
    fp_shown = 0
    for eid, p_list in preds.items():
        true_tids = set(gt_map.get(eid, []))
        fp_set = set(p_list) - true_tids
        if fp_set:
            fp_shown += 1
            tid = list(fp_set)[0]
            s1_rec = s1_df.filter(pl.col("entity_id") == eid).row(0, named=True)
            t_rec = target_df.filter(pl.col("entity_id") == tid).row(0, named=True)
            print(f"\n[FP #{fp_shown}] {eid} -> {tid} (NOT in true ground truth!)")
            print(f"  S1: '{s1_rec['business_name']}' | '{s1_rec['business_address']}'")
            print(f"  T : '{t_rec['business_name']}' | '{t_rec['business_address']}'")
            if fp_shown >= 5:
                break

if __name__ == "__main__":
    diagnose()
