import os, sys, time
sys.path.insert(0, os.getcwd())
import polars as pl
from collections import defaultdict
import unicodedata, re
from rapidfuzz import fuzz, distance
from src.evaluation import evaluate_predictions

def run_val_benchmark():
    t0 = time.time()
    print("Running Baseline SOTA v5 on 10,000 Validation Entities...")
    
    # 1. Load Validation Data
    val_s1 = pl.read_csv("data/val_split/val_s1.tsv", separator="\t")
    val_targets = pl.read_csv("data/val_split/val_targets.tsv", separator="\t")
    val_gt = pl.read_csv("data/val_split/val_gt.tsv", separator="\t")
    
    gt_map = {}
    total_true = 0
    for r in val_gt.iter_rows(named=True):
        sid = r["source1_entity_id"]
        m = r["matched_entity_ids"] or ""
        tids = [x.strip() for x in m.split(",") if x.strip()]
        gt_map[sid] = tids
        total_true += len(tids)

    # Import clean & score functions from SOTA v5
    from src.run_sota_v5_optimal import (
        clean_text, extract_core_brand, extract_numbers_set, 
        extract_geo_code, extract_addr_tokens, score_pair_optimal, NULL_ADDRS
    )

    # 2. Index Targets by Country
    t_by_c = defaultdict(dict)
    idx_stem_by_c = defaultdict(lambda: defaultdict(list))
    idx_stem_geo_by_c = defaultdict(lambda: defaultdict(list))
    idx_pfx4_by_c = defaultdict(lambda: defaultdict(list))
    idx_pfx4_geo_by_c = defaultdict(lambda: defaultdict(list))
    idx_num_tok_by_c = defaultdict(lambda: defaultdict(list))
    idx_num_geo_by_c = defaultdict(lambda: defaultdict(list))
    idx_pin_by_c = defaultdict(lambda: defaultdict(list))

    for r in val_targets.iter_rows(named=True):
        eid = r["entity_id"]
        c = r["country"].strip()
        raw_n = r["business_name"] or ""
        raw_a = r["business_address"] or ""
        n_name = clean_text(raw_n, c)
        core_brand = extract_core_brand(n_name)
        n_addr = clean_text(raw_a, c)
        nums = extract_numbers_set(raw_a)
        geo = extract_geo_code(n_addr, raw_a, c)
        is_empty = (not n_addr) or (n_addr in NULL_ADDRS)
        is_indic = any(ord(ch) > 127 for ch in raw_n)

        t_by_c[c][eid] = {
            "entity_id": eid,
            "core_brand": core_brand,
            "norm_addr": n_addr,
            "nums": nums,
            "geo": geo,
            "is_addr_empty": is_empty,
            "is_indic": is_indic,
            "no_space": core_brand.replace(" ", ""),
        }

        stems = [w for w in core_brand.split() if len(w) >= 2]
        for s in stems:
            idx_stem_by_c[c][s].append(eid)
            if geo:
                idx_stem_geo_by_c[c][(s, geo)].append(eid)
        if stems and len(stems[0]) >= 4:
            pfx = stems[0][:4]
            idx_pfx4_by_c[c][pfx].append(eid)
            if geo:
                idx_pfx4_geo_by_c[c][(pfx, geo)].append(eid)
        ns = core_brand.replace(" ", "")
        if len(ns) >= 5:
            idx_stem_by_c[c][ns].append(eid)
            if geo:
                idx_stem_geo_by_c[c][(ns, geo)].append(eid)

        atok = extract_addr_tokens(n_addr)
        for num in nums:
            if len(num) >= 5:
                idx_pin_by_c[c][num].append(eid)
            if geo:
                idx_num_geo_by_c[c][(num, geo)].append(eid)
            for tok in atok[:2]:
                idx_num_tok_by_c[c][(num, tok)].append(eid)

    # 3. Query Matching
    preds = {}
    captured_cands = 0
    captured_preds = 0

    for r in val_s1.iter_rows(named=True):
        sid = r["entity_id"]
        c = r["country"].strip()
        raw_n = r["business_name"] or ""
        raw_a = r["business_address"] or ""
        n_name = clean_text(raw_n, c)
        core_brand = extract_core_brand(n_name)
        n_addr = clean_text(raw_a, c)
        nums = extract_numbers_set(raw_a)
        geo = extract_geo_code(n_addr, raw_a, c)
        is_indic = any(ord(ch) > 127 for ch in raw_n)

        s1_obj = {
            "core_brand": core_brand,
            "norm_addr": n_addr,
            "nums": nums,
            "geo": geo,
            "is_indic": is_indic,
            "no_space": core_brand.replace(" ", ""),
        }

        true_tids = set(gt_map.get(sid, []))
        idx_stem = idx_stem_by_c[c]
        idx_stem_geo = idx_stem_geo_by_c[c]
        idx_pfx4 = idx_pfx4_by_c[c]
        idx_pfx4_geo = idx_pfx4_geo_by_c[c]
        idx_num_tok = idx_num_tok_by_c[c]
        idx_num_geo = idx_num_geo_by_c[c]
        idx_pin = idx_pin_by_c[c]
        tgt_dict = t_by_c[c]

        # Candidate retrieval
        cands = set()
        stems = [w for w in core_brand.split() if len(w) >= 2]
        stems.sort(key=lambda x: len(idx_stem.get(x, [])))

        for s in stems:
            sc = idx_stem.get(s, [])
            if len(sc) <= 200:
                cands.update(sc)
            elif geo:
                cands.update(idx_stem_geo.get((s, geo), [])[:50])
            else:
                cands.update(sc[:50])
            if len(cands) >= 50:
                break

        if len(cands) < 30 and stems and len(stems[0]) >= 4:
            pfx = stems[0][:4]
            pc = idx_pfx4.get(pfx, [])
            if len(pc) <= 200:
                cands.update(pc)
            elif geo:
                cands.update(idx_pfx4_geo.get((pfx, geo), [])[:40])
            else:
                cands.update(pc[:40])

        atok = extract_addr_tokens(n_addr)
        for num in nums:
            if len(num) >= 5:
                cands.update(idx_pin.get(num, [])[:30])
            if geo:
                cands.update(idx_num_geo.get((num, geo), [])[:30])
            for tok in atok[:2]:
                cands.update(idx_num_tok.get((num, tok), [])[:30])

        captured_cands += len(true_tids & cands)

        # Scorer
        scored = []
        for tid in cands:
            t_obj = tgt_dict[tid]
            score = score_pair_optimal(s1_obj, t_obj)
            if score >= 0.80:
                scored.append((tid, score))

        matched = []
        if scored:
            scored.sort(key=lambda x: x[1], reverse=True)
            top_s = scored[0][1]
            s2_c, s3_c = 0, 0
            for tid, score in scored:
                if score >= 0.80 and (top_s - score) <= 0.04:
                    if tid.startswith("S2-") and s2_c < 3:
                        matched.append(tid)
                        s2_c += 1
                    elif tid.startswith("S3-") and s3_c < 3:
                        matched.append(tid)
                        s3_c += 1

        captured_preds += len(true_tids & set(matched))
        preds[sid] = matched

    # 4. Evaluation
    res = evaluate_predictions(preds, gt_map, beta=0.5)
    print("\n" + "=" * 65)
    print("BENCHMARK VALIDATION RESULTS (10,000 S1, 233,851 Targets):")
    print("=" * 65)
    print(f"Total True Matches: {total_true:,}")
    print(f"Candidate Recall:   {captured_cands:,} / {total_true:,} ({captured_cands/total_true*100:.2f}%)")
    print(f"Captured Matches:   {captured_preds:,} / {total_true:,} ({captured_preds/total_true*100:.2f}%)")
    print(f"Macro F0.5:         {res['macro_f0_5']:.4f}")
    print(f"Macro Precision:    {res['macro_precision']:.4f}")
    print(f"Macro Recall:       {res['macro_recall']:.4f}")
    print(f"Singleton Accuracy: {res['singleton_accuracy']:.4f}")
    print(f"Total Time:         {time.time()-t0:.1f}s")
    print("=" * 65)

if __name__ == "__main__":
    run_val_benchmark()
