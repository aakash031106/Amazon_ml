import sys, os, time
sys.path.insert(0, os.getcwd())
import polars as pl
from collections import defaultdict
import unicodedata, re
from rapidfuzz import fuzz, distance

from src.evaluation import evaluate_predictions

def test():
    t0 = time.time()
    print("Loading 5,000 Ground Truth Entities & 117,000 Targets...")
    gt_df = pl.read_csv("data/dataset/train/train_ground_truth.tsv", separator="\t", n_rows=5000)
    gt_map = {}
    needed_t = set()
    for r in gt_df.iter_rows(named=True):
        sid = r["source1_entity_id"]
        m = r["matched_entity_ids"] or ""
        tids = [x.strip() for x in m.split(",") if x.strip()]
        gt_map[sid] = tids
        needed_t.update(tids)

    s1_df = pl.read_csv("data/dataset/train/train_source1.tsv", separator="\t").filter(pl.col("entity_id").is_in(list(gt_map.keys())))

    target_dfs = []
    for s in ["train_source2.tsv", "train_source3.tsv"]:
        df = pl.read_csv(f"data/dataset/train/{s}", separator="\t")
        m = df.filter(pl.col("entity_id").is_in(list(needed_t)))
        d = df.head(50000)
        target_dfs.append(m)
        target_dfs.append(d)
    targets = pl.concat(target_dfs).unique(subset=["entity_id"])
    print(f"Loaded in {time.time()-t0:.1f}s. Targets: {len(targets)}")

    NUM_REGEX = re.compile(r"\d+")
    CLEAN_REGEX = re.compile(r"[^\w\s]")
    SPACE_REGEX = re.compile(r"\s+")
    GENERIC_REGEX = re.compile(r"\b(pvt|ltd|limited|private|llc|inc|corp|corporation|co|company|llp|gmbh|sa|sas|sarl|eurl|sci|snc)\b", re.I)
    URL_CLEAN = re.compile(r"\.(com|org|net|fr|in|co|io)\b", re.I)

    US_STATES = {
        "al", "ak", "az", "ar", "ca", "co", "ct", "de", "fl", "ga", "hi", "id", "il", "in", "ia",
        "ks", "ky", "la", "me", "md", "ma", "mi", "mn", "ms", "mo", "mt", "ne", "nv", "nh", "nj",
        "nm", "ny", "nc", "nd", "oh", "ok", "or", "pa", "ri", "sc", "sd", "tn", "tx", "ut", "vt",
        "va", "wa", "wv", "wi", "wy"
    }
    NULL_ADDRS = {"", "none", "null", "nan", "<null>"}
    STOP = {"street", "st", "road", "rd", "avenue", "ave", "drive", "dr", "lane", "ln", "near", "nr", "opp", "floor", "fl", "bldg", "block", "no", "flat", "null", "none"}

    def clean(t):
        if not t: return ""
        s = unicodedata.normalize("NFKD", str(t))
        s = "".join(c for c in s if not unicodedata.combining(c)).lower()
        s = URL_CLEAN.sub("", s)
        s = CLEAN_REGEX.sub(" ", s)
        return SPACE_REGEX.sub(" ", s).strip()

    def brand(n):
        return SPACE_REGEX.sub(" ", GENERIC_REGEX.sub(" ", n)).strip()

    def extract_nums(a):
        if not a: return set()
        found = NUM_REGEX.findall(str(a))
        res = set()
        for f in found:
            val = f.lstrip("0")
            if val and len(val) <= 8:
                res.add(val)
        return res

    def extract_state(a, country):
        if country == "US":
            for t in a.split():
                if t in US_STATES:
                    return t
        return ""

    def extract_addr_tokens(a):
        if not a: return []
        tokens = [w for w in clean(a).split() if len(w) >= 4 and not w.isdigit() and w not in STOP]
        return tokens

    # Target index structures
    t_by_c = defaultdict(dict)
    idx_stem = defaultdict(lambda: defaultdict(list))
    idx_pfx4 = defaultdict(lambda: defaultdict(list))
    idx_num_tok = defaultdict(lambda: defaultdict(list))
    idx_num_state = defaultdict(lambda: defaultdict(list))
    idx_pin = defaultdict(lambda: defaultdict(list))

    print("Indexing target entities...")
    t1 = time.time()
    for r in targets.iter_rows(named=True):
        tid = r["entity_id"]
        c = r["country"].strip()
        raw_n = r["business_name"] or ""
        raw_a = r["business_address"] or ""
        n = clean(raw_n)
        b = brand(n)
        a = clean(raw_a)
        nm = extract_nums(raw_a)
        st = extract_state(a, c)
        is_empty = (not a) or (a in NULL_ADDRS)
        is_indic = any(ord(ch) > 127 for ch in raw_n)

        t_by_c[c][tid] = {
            "entity_id": tid,
            "core_brand": b,
            "norm_addr": a,
            "nums": nm,
            "state": st,
            "is_empty": is_empty,
            "is_indic": is_indic,
            "no_space": b.replace(" ", "")
        }

        stems = [w for w in b.split() if len(w) >= 2]
        for s in stems:
            idx_stem[c][s].append(tid)
        if stems and len(stems[0]) >= 4:
            idx_pfx4[c][stems[0][:4]].append(tid)
        ns = b.replace(" ", "")
        if len(ns) >= 5:
            idx_stem[c][ns].append(tid)

        atok = extract_addr_tokens(raw_a)
        for num in nm:
            if len(num) >= 5: # PIN code
                idx_pin[c][num].append(tid)
            if st:
                idx_num_state[c][(num, st)].append(tid)
            for tok in atok[:2]:
                idx_num_tok[c][(num, tok)].append(tid)

    print(f"Indexed in {time.time()-t1:.1f}s.")

    def score_pair_optimal(s1, t):
        # 1. State conflict check (Zero-tolerance)
        if s1["state"] and t["state"] and s1["state"] != t["state"]:
            return 0.0

        b1, b2 = s1["core_brand"], t["core_brand"]
        a1, a2 = s1["norm_addr"], t["norm_addr"]

        # Exact brand match fast path
        if b1 == b2:
            b_sort = 100.0
        else:
            b_sort = fuzz.token_sort_ratio(b1, b2)

        ns1, ns2 = s1["no_space"], t["no_space"]
        if ns1 and ns2 and (ns1 in ns2 or ns2 in ns1):
            b_sort = max(b_sort, 92.0)

        # 2. Case: Target Address is Missing
        if t["is_empty"]:
            # Only match if brand similarity is extremely high
            if b_sort >= 85 and distance.JaroWinkler.similarity(b1, b2) >= 0.85:
                return 0.92 + (b_sort / 1000.0)
            return 0.0

        # Number overlap analysis
        has_nums = bool(s1["nums"] and t["nums"])
        nums_overlap = bool(s1["nums"] & t["nums"])

        # Strict street number conflict: different building numbers = NEVER match unless brand is near-identical
        if has_nums and not nums_overlap and b_sort < 95:
            return 0.0

        if a1 == a2:
            a_sort = 100.0
        else:
            a_sort = fuzz.token_sort_ratio(a1, a2)

        # 3. Case: Target Name is Transliterated (Indic Script / Non-ASCII)
        if t["is_indic"] and not s1["is_indic"]:
            # Name cannot match in ASCII, must rely on address & number grounding
            if nums_overlap and a_sort >= 78:
                return 0.91 + (a_sort / 1000.0)
            if a_sort >= 88:
                return 0.90 + (a_sort / 1000.0)
            return 0.0

        # 4. Standard Case: Both Names & Addresses Present
        # Tier 1: Strong brand + good address
        if b_sort >= 70 and a_sort >= 50:
            return 0.90 + (0.5 * b_sort + 0.5 * a_sort) / 1000.0

        # Tier 2: Moderate brand + strong address
        if b_sort >= 52 and a_sort >= 68:
            return 0.88 + (0.4 * b_sort + 0.6 * a_sort) / 1000.0

        # Tier 3: Same building/street number + strong address + non-zero brand
        if has_nums and nums_overlap and a_sort >= 82 and b_sort >= 30:
            return 0.88 + (a_sort / 1000.0)

        # Tier 4: Exact brand + consistent address (prevent cross-city false matches!)
        if b_sort >= 90 and a_sort >= 40:
            return 0.86 + (b_sort / 1000.0)

        return 0.0

    print("Evaluating predictions...")
    preds = {}
    total_true = sum(len(v) for v in gt_map.values())
    captured_cands = 0
    captured_preds = 0

    for r in s1_df.iter_rows(named=True):
        sid = r["entity_id"]
        c = r["country"].strip()
        raw_n = r["business_name"] or ""
        raw_a = r["business_address"] or ""
        n = clean(raw_n)
        b = brand(n)
        a = clean(raw_a)
        nm = extract_nums(raw_a)
        st = extract_state(a, c)
        is_indic = any(ord(ch) > 127 for ch in raw_n)

        s1_obj = {
            "core_brand": b,
            "norm_addr": a,
            "nums": nm,
            "state": st,
            "is_indic": is_indic,
            "no_space": b.replace(" ", "")
        }

        true_tids = set(gt_map.get(sid, []))
        idx_s = idx_stem[c]
        idx_p = idx_pfx4[c]
        idx_nt = idx_num_tok[c]
        idx_ns = idx_num_state[c]
        idx_pn = idx_pin[c]
        tgt_dict = t_by_c[c]

        # Candidate generation
        cands = set()
        stems = [w for w in b.split() if len(w) >= 2]
        stems.sort(key=lambda x: len(idx_s.get(x, [])))

        for s in stems:
            sc = idx_s.get(s, [])
            if len(sc) <= 200:
                cands.update(sc)
            elif st:
                cands.update([tid for tid in sc if tgt_dict[tid]["state"] == st][:50])
            else:
                cands.update(sc[:50])
            if len(cands) >= 50:
                break

        if len(cands) < 30 and stems and len(stems[0]) >= 4:
            pc = idx_p.get(stems[0][:4], [])
            if len(pc) <= 200:
                cands.update(pc)
            else:
                cands.update(pc[:40])

        # Address & Number candidate retrieval
        atok = extract_addr_tokens(raw_a)
        for num in nm:
            if len(num) >= 5:
                cands.update(idx_pn.get(num, [])[:30])
            if st:
                cands.update(idx_ns.get((num, st), [])[:30])
            for tok in atok[:2]:
                cands.update(idx_nt.get((num, tok), [])[:30])

        captured_cands += len(true_tids & cands)

        # Score candidates
        scored = []
        for tid in cands:
            t_obj = tgt_dict[tid]
            score = score_pair_optimal(s1_obj, t_obj)
            if score >= 0.80:
                scored.append((tid, score))

        matched = []
        if scored:
            scored.sort(key=lambda x: x[1], reverse=True)
            top_score = scored[0][1]
            s2_c, s3_c = 0, 0
            for tid, score in scored:
                if score >= 0.80 and (top_score - score) <= 0.05:
                    if tid.startswith("S2-") and s2_c < 3:
                        matched.append(tid)
                        s2_c += 1
                    elif tid.startswith("S3-") and s3_c < 3:
                        matched.append(tid)
                        s3_c += 1

        captured_preds += len(true_tids & set(matched))
        preds[sid] = matched

    res = evaluate_predictions(preds, gt_map, beta=0.5)
    print("\n" + "="*60)
    print("RESULTS ON 5,000 REAL ENTITIES WITH 117,000 TARGETS:")
    print("="*60)
    print(f"Total True Matches: {total_true:,}")
    print(f"Candidate Recall: {captured_cands:,} / {total_true:,} ({captured_cands/total_true*100:.2f}%)")
    print(f"Captured Predictions: {captured_preds:,} / {total_true:,} ({captured_preds/total_true*100:.2f}%)")
    print(f"Macro F0.5: {res['macro_f0_5']:.4f}")
    print(f"Macro Precision: {res['macro_precision']:.4f}")
    print(f"Macro Recall: {res['macro_recall']:.4f}")
    print(f"Singleton Accuracy: {res['singleton_accuracy']:.4f}")
    print("="*60)

if __name__ == "__main__":
    test()
