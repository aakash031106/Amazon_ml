import os
import sys
import time
sys.path.insert(0, os.getcwd())
import polars as pl
from collections import defaultdict
from rapidfuzz import fuzz, distance

from src.run_sota_production import clean_text, extract_core_brand, extract_numbers_set, extract_us_state, extract_addr_tokens, STOP_ADDR
from src.evaluation import evaluate_predictions

def test_fixed_scorer():
    print("Testing Fixed Scorer on 5,000 real ground truth entities...")
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

    t_by_c = defaultdict(dict)
    index_stem_by_c = defaultdict(lambda: defaultdict(list))
    index_pfx4_by_c = defaultdict(lambda: defaultdict(list))
    index_pin_by_c = defaultdict(lambda: defaultdict(list))
    index_num_addr_by_c = defaultdict(lambda: defaultdict(list))
    index_num_state_by_c = defaultdict(lambda: defaultdict(list))

    NULL_ADDRS = {"", "none", "null", "nan", "<null>"}

    for r in target_df.iter_rows(named=True):
        eid = r["entity_id"]
        c = r["country"].strip()
        n_name = clean_text(r["business_name"], c)
        core_brand = extract_core_brand(n_name)
        n_addr = clean_text(r["business_address"], c)
        nums = extract_numbers_set(r["business_address"])
        state = extract_us_state(n_addr) if c == "US" else ""
        is_empty = (not n_addr) or (n_addr in NULL_ADDRS)

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

    def score_pair_fixed(s1: dict, t: dict) -> float:
        if s1["state"] and t["state"] and s1["state"] != t["state"]:
            return 0.0

        b1, b2 = s1["core_brand"], t["core_brand"]
        a1, a2 = s1["norm_addr"], t["norm_addr"]

        if b1 == b2:
            b_sort = 100.0
        else:
            b_sort = fuzz.token_sort_ratio(b1, b2)

        ns1, ns2 = s1["no_space"], t["no_space"]
        if ns1 and ns2 and (ns1 in ns2 or ns2 in ns1):
            b_sort = max(b_sort, 92.0)

        # 1. Missing target address
        if t["is_addr_empty"]:
            if b_sort >= 85 and distance.JaroWinkler.similarity(b1, b2) >= 0.85:
                return 0.92 + (b_sort / 1000.0)
            return 0.0

        has_nums = bool(s1["nums"] and t["nums"])
        nums_overlap = bool(s1["nums"] & t["nums"])

        # Strict street number conflict: different building numbers = NEVER match unless brand is 98%
        if has_nums and not nums_overlap and b_sort < 98:
            return 0.0

        # Pruning
        if b_sort < 40 and not (has_nums and nums_overlap):
            return 0.0

        if a1 == a2:
            a_sort = 100.0
        else:
            a_sort = fuzz.token_sort_ratio(a1, a2)

        # 2. Strong brand + reasonable address
        if b_sort >= 70 and a_sort >= 50:
            return 0.90 + (0.5 * b_sort + 0.5 * a_sort) / 1000.0

        # 3. Moderate brand + strong address
        if b_sort >= 55 and a_sort >= 68:
            return 0.88 + (0.4 * b_sort + 0.6 * a_sort) / 1000.0

        # 4. Near-identical address + street number overlap (transliterations in native script)
        # CRITICAL: Require brand overlap >= 30 OR near-perfect address >= 88 to avoid merging different shops!
        if has_nums and nums_overlap:
            if a_sort >= 88 and b_sort >= 30:
                return 0.88 + (a_sort / 1000.0)
            elif a_sort >= 92:
                return 0.88 + (a_sort / 1000.0)

        # 5. Exact brand match
        if b_sort >= 90:
            return 0.85 + (b_sort / 1000.0)

        return 0.0

    for min_th, margin, max_s in [
        (0.80, 0.05, 3),
        (0.80, 0.06, 4),
        (0.80, 0.08, 4),
    ]:
        preds = {}
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

            scored = []
            for tid in cands:
                t_obj = tgt_dict[tid]
                s = score_pair_fixed(s1_obj, t_obj)
                if s >= min_th:
                    scored.append((tid, s))

            matched = []
            if scored:
                scored.sort(key=lambda x: x[1], reverse=True)
                top_s = scored[0][1]
                s2_c, s3_c = 0, 0
                for tid, s in scored:
                    if s >= min_th and (top_s - s) <= margin:
                        if tid.startswith("S2-") and s2_c < max_s:
                            matched.append(tid)
                            s2_c += 1
                        elif tid.startswith("S3-") and s3_c < max_s:
                            matched.append(tid)
                            s3_c += 1

            preds[eid] = matched

        res = evaluate_predictions(preds, gt_map, beta=0.5)
        print(f"min_th={min_th}, margin={margin}, max_s={max_s} -> Macro F0.5: {res['macro_f0_5']:.4f} | Prec: {res['macro_precision']:.4f} | Rec: {res['macro_recall']:.4f} | SingAcc: {res['singleton_accuracy']:.4f}")

if __name__ == "__main__":
    test_fixed_scorer()
