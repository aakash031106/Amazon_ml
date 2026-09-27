import os
import sys
import time
sys.path.insert(0, os.getcwd())
import polars as pl
from collections import defaultdict
from rapidfuzz import fuzz, distance

from src.run_sota_v4 import clean_base, extract_core_brand, extract_numbers, extract_state_code, extract_distinctive_addr_tokens, STOP_ADDR
from src.evaluation import evaluate_predictions

def test_per_country():
    print("Testing 5,000 entities with Country-level breakdown...")
    t0 = time.time()

    # Load 5,000 GT
    gt_df = pl.read_csv("data/dataset/train/train_ground_truth.tsv", separator="\t", n_rows=5000)
    gt_map = {}
    needed_s1 = set()
    needed_targets = set()
    for row in gt_df.iter_rows(named=True):
        sid = row["source1_entity_id"]
        m_str = row["matched_entity_ids"] or ""
        tids = [x.strip() for x in m_str.split(",") if x.strip()]
        gt_map[sid] = tids
        needed_s1.add(sid)
        needed_targets.update(tids)

    # Load S1
    s1_all = pl.read_csv("data/dataset/train/train_source1.tsv", separator="\t")
    s1_df = s1_all.filter(pl.col("entity_id").is_in(list(needed_s1)))
    del s1_all

    # Load Targets
    target_dfs = []
    for s_file in ["train_source2.tsv", "train_source3.tsv"]:
        p = os.path.join("data/dataset/train", s_file)
        df = pl.read_csv(p, separator="\t")
        true_m = df.filter(pl.col("entity_id").is_in(list(needed_targets)))
        distractors = df.head(25000)
        target_dfs.append(true_m)
        target_dfs.append(distractors)
        del df

    target_df = pl.concat(target_dfs).unique(subset=["entity_id"])
    print(f"Loaded {len(s1_df)} S1 entities and {len(target_df)} target entities in {time.time() - t0:.1f}s.")

    # Index targets by country
    target_by_country = defaultdict(dict)
    index_stem_by_c = defaultdict(lambda: defaultdict(list))
    index_pfx4_by_c = defaultdict(lambda: defaultdict(list))
    index_pin_by_c = defaultdict(lambda: defaultdict(list))
    index_num_state_by_c = defaultdict(lambda: defaultdict(list))
    index_num_addr_by_c = defaultdict(lambda: defaultdict(list))

    for r in target_df.iter_rows(named=True):
        eid = r["entity_id"]
        c = r["country"].strip()
        raw_n = r["business_name"]
        raw_a = r["business_address"]
        n_name = clean_base(raw_n, c)
        core_brand = extract_core_brand(n_name)
        n_addr = clean_base(raw_a, c)
        nums = extract_numbers(raw_a)
        state = extract_state_code(n_addr) if c == "US" else ""
        stems = [t for t in core_brand.split() if len(t) >= 2]
        addr_tokens = extract_distinctive_addr_tokens(n_addr)

        target_by_country[c][eid] = {
            "entity_id": eid,
            "country": c,
            "norm_name": n_name,
            "core_brand": core_brand,
            "no_space_name": n_name.replace(" ", ""),
            "norm_addr": n_addr,
            "stems": stems,
            "nums": set(nums),
            "state": state,
            "is_addr_empty": not bool(n_addr),
        }

        for s in stems:
            index_stem_by_c[c][s].append(eid)
        if stems and len(stems[0]) >= 4:
            index_pfx4_by_c[c][stems[0][:4]].append(eid)
        for n in nums:
            if len(n) >= 5:
                index_pin_by_c[c][n].append(eid)
            if state:
                index_num_state_by_c[c][(n, state)].append(eid)
            for at in addr_tokens[:2]:
                index_num_addr_by_c[c][(n, at)].append(eid)

    del target_df

    for min_thresh, margin, max_per_src in [
        (0.80, 0.05, 3),
        (0.80, 0.06, 3),
        (0.80, 0.07, 3),
        (0.80, 0.08, 4),
    ]:
        preds = {}
        preds_by_c = defaultdict(dict)
        gt_by_c = defaultdict(dict)

        for r in s1_df.iter_rows(named=True):
            eid = r["entity_id"]
            c = r["country"].strip()
            raw_n = r["business_name"]
            raw_a = r["business_address"]
            n_name = clean_base(raw_n, c)
            core_brand = extract_core_brand(n_name)
            n_addr = clean_base(raw_a, c)
            nums = extract_numbers(raw_a)
            state = extract_state_code(n_addr) if c == "US" else ""
            stems = [t for t in core_brand.split() if len(t) >= 2]
            addr_tokens = extract_distinctive_addr_tokens(n_addr)

            s1 = {
                "entity_id": eid,
                "country": c,
                "norm_name": n_name,
                "core_brand": core_brand,
                "no_space_name": n_name.replace(" ", ""),
                "norm_addr": n_addr,
                "stems": stems,
                "nums": set(nums),
                "state": state,
            }

            idx_stem = index_stem_by_c[c]
            idx_pfx4 = index_pfx4_by_c[c]
            idx_pin = index_pin_by_c[c]
            idx_num_state = index_num_state_by_c[c]
            idx_num_addr = index_num_addr_by_c[c]
            tgt_dict = target_by_country[c]

            cands = set()
            for s in stems:
                sc = idx_stem.get(s, [])
                if len(sc) <= 200:
                    cands.update(sc)
                else:
                    cands.update(sc[:50])
                if len(cands) >= 50:
                    break

            if len(cands) < 30 and stems and len(stems[0]) >= 4:
                pc = idx_pfx4.get(stems[0][:4], [])
                if len(pc) <= 200:
                    cands.update(pc)
                else:
                    cands.update(pc[:40])

            for n in nums:
                if len(n) >= 5:
                    cands.update(idx_pin.get(n, [])[:30])
                if state:
                    cands.update(idx_num_state.get((n, state), [])[:30])
                for at in addr_tokens[:2]:
                    cands.update(idx_num_addr.get((n, at), [])[:30])

            scored = []
            for tid in cands:
                t = tgt_dict[tid]
                if s1["state"] and t["state"] and s1["state"] != t["state"]:
                    continue

                b1, b2 = s1["core_brand"], t["core_brand"]
                a1, a2 = s1["norm_addr"], t["norm_addr"]

                b_sort = fuzz.token_sort_ratio(b1, b2)
                if s1["no_space_name"] and t["no_space_name"]:
                    if s1["no_space_name"] in t["no_space_name"] or t["no_space_name"] in s1["no_space_name"]:
                        b_sort = max(b_sort, 92.0)

                if t["is_addr_empty"]:
                    if b_sort >= 85 and distance.JaroWinkler.similarity(b1, b2) >= 0.85:
                        scored.append((tid, 0.92 + b_sort / 1000.0))
                    continue

                has_nums = bool(s1["nums"] and t["nums"])
                nums_overlap = bool(s1["nums"] & t["nums"])
                if has_nums and not nums_overlap and b_sort < 95:
                    continue

                if b_sort < 50 and not (has_nums and nums_overlap):
                    continue

                a_sort = fuzz.token_sort_ratio(a1, a2)

                if b_sort >= 70 and a_sort >= 50:
                    scored.append((tid, 0.90 + (0.5 * b_sort + 0.5 * a_sort) / 1000.0))
                elif b_sort >= 55 and a_sort >= 68:
                    scored.append((tid, 0.88 + (0.4 * b_sort + 0.6 * a_sort) / 1000.0))
                elif has_nums and nums_overlap and a_sort >= 80:
                    scored.append((tid, 0.88 + a_sort / 1000.0))
                elif b_sort >= 88:
                    scored.append((tid, 0.85 + b_sort / 1000.0))

            matched = []
            if scored:
                scored.sort(key=lambda x: x[1], reverse=True)
                top_s = scored[0][1]
                s2_c, s3_c = 0, 0
                for tid, s in scored:
                    if s >= min_thresh and (top_s - s) <= margin:
                        if tid.startswith("S2-") and s2_c < max_per_src:
                            matched.append(tid)
                            s2_c += 1
                        elif tid.startswith("S3-") and s3_c < max_per_src:
                            matched.append(tid)
                            s3_c += 1

            preds[eid] = matched
            preds_by_c[c][eid] = matched
            gt_by_c[c][eid] = gt_map[eid]

        res = evaluate_predictions(preds, gt_map, beta=0.5)
        print(f"\nOVERALL: min_th={min_thresh}, margin={margin}, max_per_src={max_per_src}")
        print(f"  Macro F0.5: {res['macro_f0_5']:.4f} | Prec: {res['macro_precision']:.4f} | Rec: {res['macro_recall']:.4f} | SingAcc: {res['singleton_accuracy']:.4f}")
        for c in ["France", "India", "US"]:
            r_c = evaluate_predictions(preds_by_c[c], gt_by_c[c], beta=0.5)
            print(f"  [{c:6}] F0.5: {r_c['macro_f0_5']:.4f} | Prec: {r_c['macro_precision']:.4f} | Rec: {r_c['macro_recall']:.4f} (N={len(gt_by_c[c])})")

if __name__ == "__main__":
    test_per_country()
