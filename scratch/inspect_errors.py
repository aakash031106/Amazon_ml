import sys, os
sys.path.insert(0, os.getcwd())
import polars as pl
from collections import defaultdict
from src.evaluation import evaluate_predictions
from src.run_sota_v5_optimal import (
    clean_text, extract_core_brand, extract_numbers_set, 
    extract_geo_code, extract_addr_tokens, score_pair_optimal, NULL_ADDRS
)

def inspect_errors():
    val_s1 = pl.read_csv('data/val_split/val_s1.tsv', separator='\t')
    val_targets = pl.read_csv('data/val_split/val_targets.tsv', separator='\t')
    val_gt = pl.read_csv('data/val_split/val_gt.tsv', separator='\t')

    s1_dict = {r['entity_id']: r for r in val_s1.iter_rows(named=True)}
    t_dict = {r['entity_id']: r for r in val_targets.iter_rows(named=True)}
    gt_map = {r['source1_entity_id']: [x.strip() for x in (r['matched_entity_ids'] or '').split(',') if x.strip()] for r in val_gt.iter_rows(named=True)}

    # Target Indexing
    t_by_c = defaultdict(dict)
    idx_stem_by_c = defaultdict(lambda: defaultdict(list))
    idx_stem_geo_by_c = defaultdict(lambda: defaultdict(list))
    idx_pfx4_by_c = defaultdict(lambda: defaultdict(list))
    idx_pfx4_geo_by_c = defaultdict(lambda: defaultdict(list))
    idx_num_tok_by_c = defaultdict(lambda: defaultdict(list))
    idx_num_geo_by_c = defaultdict(lambda: defaultdict(list))
    idx_pin_by_c = defaultdict(lambda: defaultdict(list))

    for r in val_targets.iter_rows(named=True):
        eid = r['entity_id']
        c = r['country'].strip()
        raw_n = r['business_name'] or ''
        raw_a = r['business_address'] or ''
        n_name = clean_text(raw_n, c)
        core_brand = extract_core_brand(n_name)
        n_addr = clean_text(raw_a, c)
        nums = extract_numbers_set(raw_a)
        geo = extract_geo_code(n_addr, raw_a, c)
        is_empty = (not n_addr) or (n_addr in NULL_ADDRS)
        is_indic = any(ord(ch) > 127 for ch in raw_n)

        t_by_c[c][eid] = {
            'entity_id': eid,
            'core_brand': core_brand,
            'norm_addr': n_addr,
            'nums': nums,
            'geo': geo,
            'is_addr_empty': is_empty,
            'is_indic': is_indic,
            'no_space': core_brand.replace(' ', ''),
        }

        stems = [w for w in core_brand.split() if len(w) >= 2]
        for s in stems:
            idx_stem_by_c[c][s].append(eid)
            if geo: idx_stem_geo_by_c[c][(s, geo)].append(eid)
        if stems and len(stems[0]) >= 4:
            pfx = stems[0][:4]
            idx_pfx4_by_c[c][pfx].append(eid)
            if geo: idx_pfx4_geo_by_c[c][(pfx, geo)].append(eid)
        ns = core_brand.replace(' ', '')
        if len(ns) >= 5:
            idx_stem_by_c[c][ns].append(eid)
            if geo: idx_stem_geo_by_c[c][(ns, geo)].append(eid)

        atok = extract_addr_tokens(n_addr)
        for num in nums:
            if len(num) >= 5: idx_pin_by_c[c][num].append(eid)
            if geo: idx_num_geo_by_c[c][(num, geo)].append(eid)
            for tok in atok[:2]: idx_num_tok_by_c[c][(num, tok)].append(eid)

    all_candidates_scored = []
    cands_by_s1 = defaultdict(set)
    for r in val_s1.iter_rows(named=True):
        sid = r['entity_id']
        c = r['country'].strip()
        raw_n = r['business_name'] or ''
        raw_a = r['business_address'] or ''
        n_name = clean_text(raw_n, c)
        core_brand = extract_core_brand(n_name)
        n_addr = clean_text(raw_a, c)
        nums = extract_numbers_set(raw_a)
        geo = extract_geo_code(n_addr, raw_a, c)
        is_indic = any(ord(ch) > 127 for ch in raw_n)

        s1_obj = {
            'core_brand': core_brand,
            'norm_addr': n_addr,
            'nums': nums,
            'geo': geo,
            'is_indic': is_indic,
            'no_space': core_brand.replace(' ', ''),
        }

        idx_stem = idx_stem_by_c[c]
        idx_stem_geo = idx_stem_geo_by_c[c]
        idx_pfx4 = idx_pfx4_by_c[c]
        idx_pfx4_geo = idx_pfx4_geo_by_c[c]
        idx_num_tok = idx_num_tok_by_c[c]
        idx_num_geo = idx_num_geo_by_c[c]
        idx_pin = idx_pin_by_c[c]
        tgt_dict = t_by_c[c]

        cands = set()
        stems = [w for w in core_brand.split() if len(w) >= 2]
        stems.sort(key=lambda x: len(idx_stem.get(x, [])))

        for s in stems:
            sc = idx_stem.get(s, [])
            if len(sc) <= 200: cands.update(sc)
            elif geo: cands.update(idx_stem_geo.get((s, geo), [])[:50])
            else: cands.update(sc[:50])
            if len(cands) >= 50: break

        if len(cands) < 30 and stems and len(stems[0]) >= 4:
            pfx = stems[0][:4]
            pc = idx_pfx4.get(pfx, [])
            if len(pc) <= 200: cands.update(pc)
            elif geo: cands.update(idx_pfx4_geo.get((pfx, geo), [])[:40])
            else: cands.update(pc[:40])

        atok = extract_addr_tokens(n_addr)
        for num in nums:
            if len(num) >= 5: cands.update(idx_pin.get(num, [])[:30])
            if geo: cands.update(idx_num_geo.get((num, geo), [])[:30])
            for tok in atok[:2]: cands.update(idx_num_tok.get((num, tok), [])[:30])

        cands_by_s1[sid] = cands

        for tid in cands:
            s = score_pair_optimal(s1_obj, tgt_dict[tid])
            if s >= 0.80:
                all_candidates_scored.append((sid, tid, s))

    # Global 1-to-1 Target Unique Matching
    preds_unique = defaultdict(list)
    assigned_targets = set()
    all_candidates_scored.sort(key=lambda x: x[2], reverse=True)
    best_by_s1 = {}
    for sid, tid, s in all_candidates_scored:
        if tid in assigned_targets:
            continue
        if sid not in best_by_s1:
            best_by_s1[sid] = s
        if (best_by_s1[sid] - s) <= 0.04:
            s2_c = sum(1 for x in preds_unique[sid] if x.startswith('S2-'))
            s3_c = sum(1 for x in preds_unique[sid] if x.startswith('S3-'))
            if tid.startswith('S2-') and s2_c < 3:
                preds_unique[sid].append(tid)
                assigned_targets.add(tid)
            elif tid.startswith('S3-') and s3_c < 3:
                preds_unique[sid].append(tid)
                assigned_targets.add(tid)

    # Inspect False Positives
    fps = []
    fns = []
    for sid, true_tids in gt_map.items():
        predicted = set(preds_unique.get(sid, []))
        true_set = set(true_tids)
        for p in predicted - true_set:
            fps.append((sid, p))
        for m in true_set - predicted:
            in_cands = m in cands_by_s1[sid]
            fns.append((sid, m, in_cands))

    print(f"\nTOTAL FALSE POSITIVES IN VAL: {len(fps)}")
    print(f"TOTAL FALSE NEGATIVES IN VAL: {len(fns)}")
    blocking_fns = sum(1 for _, _, in_c in fns if not in_c)
    scorer_fns = sum(1 for _, _, in_c in fns if in_c)
    print(f"  - Missed by Blocker (not in candidates): {blocking_fns} ({blocking_fns/len(fns)*100:.1f}%)")
    print(f"  - Missed by Scorer (in candidates, score < 0.80): {scorer_fns} ({scorer_fns/len(fns)*100:.1f}%)")

    print("\n" + "="*70)
    print("SAMPLE 5 FALSE POSITIVES (Why did the model wrongly merge?):")
    print("="*70)
    for sid, tid in fps[:5]:
        s = s1_dict[sid]
        t = t_dict[tid]
        print(f"\n[FP] {sid} ({s['country']}) -> {tid}")
        print(f"  S1: '{s['business_name']}' | '{s['business_address']}'")
        print(f"  T : '{t['business_name']}' | '{t['business_address']}'")

    print("\n" + "="*70)
    print("SAMPLE 5 FALSE NEGATIVES MISSED BY BLOCKER (Why not in candidates?):")
    print("="*70)
    shown = 0
    for sid, tid, in_c in fns:
        if not in_c:
            s = s1_dict[sid]
            t = t_dict[tid]
            print(f"\n[FN - Blocking] {sid} ({s['country']}) <-> {tid}")
            print(f"  S1: '{s['business_name']}' | '{s['business_address']}'")
            print(f"  T : '{t['business_name']}' | '{t['business_address']}'")
            shown += 1
            if shown >= 5: break

    print("\n" + "="*70)
    print("SAMPLE 5 FALSE NEGATIVES MISSED BY SCORER (Why rejected by threshold?):")
    print("="*70)
    shown = 0
    for sid, tid, in_c in fns:
        if in_c:
            s = s1_dict[sid]
            t = t_dict[tid]
            print(f"\n[FN - Scorer] {sid} ({s['country']}) <-> {tid}")
            print(f"  S1: '{s['business_name']}' | '{s['business_address']}'")
            print(f"  T : '{t['business_name']}' | '{t['business_address']}'")
            shown += 1
            if shown >= 5: break

if __name__ == "__main__":
    inspect_errors()
