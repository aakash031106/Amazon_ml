import sys, os, time
sys.path.insert(0, os.getcwd())
import polars as pl
from collections import defaultdict
import unicodedata, re
import numpy as np
from rapidfuzz import fuzz, distance
import lightgbm as lgb
from src.evaluation import evaluate_predictions

def train_and_eval_lgbm():
    t0 = time.time()
    print("=" * 70)
    print("TRAINING LIGHTGBM PAIRWISE MATCH CLASSIFIER WITH HARD NEGATIVES")
    print("=" * 70)

    # 1. Load Training Data (First 15,000 S1 for Training, Next 10,000 for Validation)
    print("Loading Ground Truth and Source Entities...")
    gt_df = pl.read_csv("data/dataset/train/train_ground_truth.tsv", separator="\t", n_rows=25000)
    
    train_gt_map = {}
    val_gt_map = {}
    needed_train_t = set()
    needed_val_t = set()

    for i, r in enumerate(gt_df.iter_rows(named=True)):
        sid = r["source1_entity_id"]
        m_str = r["matched_entity_ids"] or ""
        tids = [x.strip() for x in m_str.split(",") if x.strip()]
        if i < 15000:
            train_gt_map[sid] = tids
            needed_train_t.update(tids)
        else:
            val_gt_map[sid] = tids
            needed_val_t.update(tids)

    print(f"Train S1: {len(train_gt_map):,} | Val S1: {len(val_gt_map):,}")

    # Load S1
    all_s1_ids = list(train_gt_map.keys()) + list(val_gt_map.keys())
    s1_df = pl.read_csv("data/dataset/train/train_source1.tsv", separator="\t").filter(pl.col("entity_id").is_in(all_s1_ids))
    train_s1_df = s1_df.filter(pl.col("entity_id").is_in(list(train_gt_map.keys())))
    val_s1_df = s1_df.filter(pl.col("entity_id").is_in(list(val_gt_map.keys())))
    del s1_df

    # Load Targets (all true targets + 60,000 distractors)
    all_needed_t = needed_train_t | needed_val_t
    target_dfs = []
    for s_file in ["train_source2.tsv", "train_source3.tsv"]:
        p = os.path.join("data/dataset/train", s_file)
        df = pl.read_csv(p, separator="\t")
        true_m = df.filter(pl.col("entity_id").is_in(list(all_needed_t)))
        distractors = df.slice(50000, 60000)
        target_dfs.append(true_m)
        target_dfs.append(distractors)
        del df
        
    targets_df = pl.concat(target_dfs).unique(subset=["entity_id"])
    print(f"Target Pool: {len(targets_df):,} records loaded in {time.time()-t0:.1f}s.")

    # 2. Text Normalization and Indexing
    GENERIC_WORDS = {
        'hotel', 'restaurant', 'cafe', 'hospital', 'clinic', 'medical', 'pharmacy', 'chemist',
        'store', 'mart', 'supermarket', 'textiles', 'hardware', 'electronics', 'telecom', 'motors',
        'school', 'academy', 'college', 'enterprises', 'enterprise', 'solutions', 'services', 'group',
        'industries', 'industry', 'builders', 'developers', 'trading', 'logistics', 'finance', 'investments'
    }
    GENERIC_REGEX = re.compile(
        r"\b(pvt|ltd|limited|private|llc|inc|corp|corporation|co|company|llp|gmbh|sa|sas|sarl|eurl|sci|snc|"
        r"enterprises|enterprise|solutions|services|group|holdings|associates|consulting|industries)\b", re.I
    )
    CLEAN_REGEX = re.compile(r"[^\w\s]")
    SPACE_REGEX = re.compile(r"\s+")
    URL_CLEAN = re.compile(r"\.(com|org|net|fr|in|co|io)\b", re.I)
    SECTOR_STRIP = re.compile(r"\b(sector|sec|phase|ph|plot|plt|nh|ward)\s*[-#]?\s*\d+\b", re.I)
    NUM_REGEX = re.compile(r"\d+")

    US_STATES = {
        "al", "ak", "az", "ar", "ca", "co", "ct", "de", "fl", "ga", "hi", "id", "il", "in", "ia",
        "ks", "ky", "la", "me", "md", "ma", "mi", "mn", "ms", "mo", "mt", "ne", "nv", "nh", "nj",
        "nm", "ny", "nc", "nd", "oh", "ok", "or", "pa", "ri", "sc", "sd", "tn", "tx", "ut", "vt",
        "va", "wa", "wv", "wi", "wy"
    }
    INDIAN_STATES = {
        "maharashtra": "MH", "mh": "MH", "tamil nadu": "TN", "tn": "TN",
        "uttar pradesh": "UP", "up": "UP", "delhi": "DL", "dl": "DL",
        "karnataka": "KA", "ka": "KA", "west bengal": "WB", "wb": "WB",
        "gujarat": "GJ", "gj": "GJ", "rajasthan": "RJ", "rj": "RJ",
        "madhya pradesh": "MP", "mp": "MP", "haryana": "HR", "hr": "HR",
        "punjab": "PB", "pb": "PB", "andhra pradesh": "AP", "ap": "AP",
        "telangana": "TS", "ts": "TS", "tg": "TS", "kerala": "KL", "kl": "KL",
        "odisha": "OD", "od": "OD", "orissa": "OD", "bihar": "BR", "br": "BR",
        "jharkhand": "JH", "jh": "JH", "assam": "AS", "as": "AS",
        "chhattisgarh": "CG", "cg": "CG", "uttarakhand": "UK", "uk": "UK",
        "goa": "GA", "ga": "GA", "himachal pradesh": "HP", "hp": "HP",
        "jammu and kashmir": "JK", "jk": "JK", "chandigarh": "CH", "ch": "CH",
        "puducherry": "PY", "py": "PY", "pondicherry": "PY"
    }
    STOP_ADDR = {"street", "st", "road", "rd", "avenue", "ave", "drive", "dr", "lane", "ln", "near", "nr", "opp", "floor", "fl", "bldg", "block", "no", "flat", "none", "nan", "null"}
    NULL_ADDRS = {"", "none", "null", "nan", "<null>"}

    def clean(t):
        if not t: return ""
        s = unicodedata.normalize("NFKD", str(t))
        s = "".join(c for c in s if not unicodedata.combining(c)).lower()
        s = URL_CLEAN.sub("", s)
        s = CLEAN_REGEX.sub(" ", s)
        return SPACE_REGEX.sub(" ", s).strip()

    def brand(n):
        return SPACE_REGEX.sub(" ", GENERIC_REGEX.sub(" ", n)).strip()

    def extract_nums(raw_a):
        if not raw_a: return set()
        cleaned_a = SECTOR_STRIP.sub(" ", str(raw_a))
        found = NUM_REGEX.findall(cleaned_a)
        res = set()
        for f in found:
            v = f.lstrip("0")
            if v and len(v) <= 6:
                res.add(v)
        return res

    def extract_geo(norm_a, raw_a, country):
        if not norm_a: return ""
        if country == "US":
            for t in reversed(norm_a.split()):
                if t in US_STATES: return t.upper()
        elif country == "India":
            low = norm_a.lower()
            for s_name, code in INDIAN_STATES.items():
                if re.search(r"\b" + re.escape(s_name) + r"\b", low): return code
        return ""

    def extract_addr_tokens(norm_a):
        return [w for w in norm_a.split() if len(w) >= 4 and not w.isdigit() and w not in STOP_ADDR]

    # Index targets
    t_by_c = defaultdict(dict)
    idx_stem = defaultdict(lambda: defaultdict(list))
    idx_stem_geo = defaultdict(lambda: defaultdict(list))
    idx_pfx4 = defaultdict(lambda: defaultdict(list))
    idx_pfx4_geo = defaultdict(lambda: defaultdict(list))
    idx_num_tok = defaultdict(lambda: defaultdict(list))
    idx_num_geo = defaultdict(lambda: defaultdict(list))
    idx_pin = defaultdict(lambda: defaultdict(list))

    print("Building multi-pass candidate indices...")
    for r in targets_df.iter_rows(named=True):
        eid = r['entity_id']
        c = r['country'].strip()
        raw_n = r['business_name'] or ''
        raw_a = r['business_address'] or ''
        n_name = clean(raw_n)
        core_brand = brand(n_name)
        n_addr = clean(raw_a)
        nums = extract_nums(raw_a)
        raw_nums = set(NUM_REGEX.findall(str(raw_a)))
        geo = extract_geo(n_addr, raw_a, c)
        is_empty = (not n_addr) or (n_addr in NULL_ADDRS)
        is_indic = any(ord(ch) > 127 for ch in raw_n)
        is_generic = core_brand in GENERIC_WORDS or len(core_brand) <= 2

        t_by_c[c][eid] = {
            'entity_id': eid,
            'core_brand': core_brand,
            'norm_name': n_name,
            'norm_addr': n_addr,
            'nums': nums,
            'geo': geo,
            'is_addr_empty': is_empty,
            'is_indic': is_indic,
            'is_generic': is_generic,
            'no_space': core_brand.replace(' ', ''),
        }

        stems = [w for w in core_brand.split() if len(w) >= 2]
        for s in stems:
            idx_stem[c][s].append(eid)
            if geo: idx_stem_geo[c][(s, geo)].append(eid)
        if stems and len(stems[0]) >= 4:
            pfx = stems[0][:4]
            idx_pfx4[c][pfx].append(eid)
            if geo: idx_pfx4_geo[c][(pfx, geo)].append(eid)
        ns = core_brand.replace(' ', '')
        if len(ns) >= 5:
            idx_stem[c][ns].append(eid)
            if geo: idx_stem_geo[c][(ns, geo)].append(eid)

        atok = extract_addr_tokens(n_addr)
        for num in raw_nums:
            if len(num) >= 5: idx_pin[c][num].append(eid)
            if geo: idx_num_geo[c][(num, geo)].append(eid)
            for tok in atok[:2]: idx_num_tok[c][(num, tok)].append(eid)

    def compute_features(s1, t):
        b1, b2 = s1['core_brand'], t['core_brand']
        a1, a2 = s1['norm_addr'], t['norm_addr']

        # Brand similarities
        b_sort = fuzz.token_sort_ratio(b1, b2)
        b_set = fuzz.token_set_ratio(b1, b2)
        b_jaro = distance.JaroWinkler.similarity(b1, b2)
        b_lev = fuzz.ratio(b1, b2)

        # Address similarities
        is_empty = 1.0 if t['is_addr_empty'] else 0.0
        if is_empty:
            a_sort = 0.0
            a_set = 0.0
            a_jaro = 0.0
        else:
            a_sort = fuzz.token_sort_ratio(a1, a2)
            a_set = fuzz.token_set_ratio(a1, a2)
            a_jaro = distance.JaroWinkler.similarity(a1, a2)

        # Number overlap
        has_nums = 1.0 if (s1['nums'] and t['nums']) else 0.0
        num_inter = len(s1['nums'] & t['nums'])
        nums_overlap = 1.0 if num_inter > 0 else 0.0
        nums_conflict = 1.0 if (has_nums and not nums_overlap) else 0.0

        # Geo match
        geo_match = 1.0 if (s1['geo'] and t['geo'] and s1['geo'] == t['geo']) else 0.0
        geo_conflict = 1.0 if (s1['geo'] and t['geo'] and s1['geo'] != t['geo']) else 0.0

        # Indic / Script flag
        is_indic = 1.0 if (t['is_indic'] and not s1['is_indic']) else 0.0

        # Brand token prefix match
        w1 = b1.split()[:2]
        w2 = b2.split()[:2]
        w1_match = 1.0 if (w1 and w2 and w1[0] == w2[0]) else 0.0
        w2_match = 1.0 if (len(w1) >= 2 and len(w2) >= 2 and w1 == w2) else 0.0

        # No space containment
        ns_contain = 1.0 if (s1['no_space'] and t['no_space'] and (s1['no_space'] in t['no_space'] or t['no_space'] in s1['no_space'])) else 0.0

        # Source type (S2=1, S3=0)
        is_s2 = 1.0 if t['entity_id'].startswith("S2-") else 0.0

        return [
            b_sort, b_set, b_jaro, b_lev,
            a_sort, a_set, a_jaro, is_empty,
            has_nums, nums_overlap, nums_conflict, num_inter,
            geo_match, geo_conflict, is_indic,
            w1_match, w2_match, ns_contain, is_s2
        ]

    FEATURE_NAMES = [
        "b_sort", "b_set", "b_jaro", "b_lev",
        "a_sort", "a_set", "a_jaro", "is_empty",
        "has_nums", "nums_overlap", "nums_conflict", "num_inter",
        "geo_match", "geo_conflict", "is_indic",
        "w1_match", "w2_match", "ns_contain", "is_s2"
    ]

    # 3. Extract Training Pairs (Positive + Hard Negatives)
    print("\nGenerating Training Dataset (Positive + Hard Negatives from 15,000 S1)...")
    X_train = []
    y_train = []

    for r in train_s1_df.iter_rows(named=True):
        sid = r['entity_id']
        c = r['country'].strip()
        raw_n = r['business_name'] or ''
        raw_a = r['business_address'] or ''
        n_name = clean(raw_n)
        core_brand = brand(n_name)
        n_addr = clean(raw_a)
        nums = extract_nums(raw_a)
        raw_nums = set(NUM_REGEX.findall(str(raw_a)))
        geo = extract_geo(n_addr, raw_a, c)
        is_indic = any(ord(ch) > 127 for ch in raw_n)
        is_generic = core_brand in GENERIC_WORDS or len(core_brand) <= 2

        s1_obj = {
            'core_brand': core_brand,
            'norm_name': n_name,
            'norm_addr': n_addr,
            'nums': nums,
            'geo': geo,
            'is_indic': is_indic,
            'is_generic': is_generic,
            'no_space': core_brand.replace(' ', ''),
        }

        true_tids = set(train_gt_map.get(sid, []))
        idx_s = idx_stem[c]
        idx_sg = idx_stem_geo[c]
        idx_p = idx_pfx4[c]
        idx_nt = idx_num_tok[c]
        idx_ng = idx_num_geo[c]
        idx_pn = idx_pin[c]
        tgt_dict = t_by_c[c]

        cands = set()
        stems = [w for w in core_brand.split() if len(w) >= 2]
        stems.sort(key=lambda x: len(idx_s.get(x, [])))

        for s in stems:
            sc = idx_s.get(s, [])
            if len(sc) <= 200: cands.update(sc)
            elif geo: cands.update(idx_sg.get((s, geo), [])[:40])
            else: cands.update(sc[:40])
            if len(cands) >= 40: break

        atok = extract_addr_tokens(n_addr)
        for num in raw_nums:
            if len(num) >= 5: cands.update(idx_pn.get(num, [])[:20])
            if geo: cands.update(idx_ng.get((num, geo), [])[:20])
            for tok in atok[:2]: cands.update(idx_nt.get((num, tok), [])[:20])

        # Add all true matches to ensure complete positive coverage
        for tid in true_tids:
            if tid in tgt_dict:
                cands.add(tid)

        # Create pairs: positive (y=1) and hard negatives (y=0)
        neg_count = 0
        for tid in cands:
            t_obj = tgt_dict[tid]
            is_match = 1 if tid in true_tids else 0
            if is_match == 0:
                if neg_count >= 8: continue # Cap negatives per entity
                neg_count += 1
            feat = compute_features(s1_obj, t_obj)
            X_train.append(feat)
            y_train.append(is_match)

    X_train = np.array(X_train, dtype=np.float32)
    y_train = np.array(y_train, dtype=np.int32)
    pos_count = int(np.sum(y_train))
    neg_count = len(y_train) - pos_count
    print(f"Training Matrix: {len(X_train):,} pairs ({pos_count:,} positive, {neg_count:,} hard negatives).")

    # 4. Train LightGBM Matcher
    print("\nTraining LightGBM Classifier...")
    t_train = time.time()
    lgb_train = lgb.Dataset(X_train, label=y_train, feature_name=FEATURE_NAMES)
    
    params = {
        'objective': 'binary',
        'metric': 'binary_logloss',
        'boosting_type': 'gbdt',
        'learning_rate': 0.08,
        'num_leaves': 31,
        'max_depth': 6,
        'min_data_in_leaf': 20,
        'feature_fraction': 0.85,
        'bagging_fraction': 0.85,
        'bagging_freq': 1,
        'verbose': -1,
        'random_state': 42
    }
    
    bst = lgb.train(params, lgb_train, num_boost_round=120)
    print(f"Model trained in {time.time()-t_train:.1f}s!")
    
    # Feature importances
    imp = bst.feature_importance(importance_type='gain')
    imp_dict = sorted(zip(FEATURE_NAMES, imp), key=lambda x: x[1], reverse=True)
    print("\nTop 8 Most Discriminative Features:")
    for fn, val in imp_dict[:8]:
        print(f"  - {fn:<15}: {val:,.1f}")

    # 5. Evaluate on Validation Set (10,000 S1 Entities)
    print("\nEvaluating on Validation Set (10,000 S1 Entities)...")
    t_val = time.time()
    
    val_pairs = []
    val_pair_meta = [] # (sid, tid)

    for r in val_s1_df.iter_rows(named=True):
        sid = r['entity_id']
        c = r['country'].strip()
        raw_n = r['business_name'] or ''
        raw_a = r['business_address'] or ''
        n_name = clean(raw_n)
        core_brand = brand(n_name)
        n_addr = clean(raw_a)
        nums = extract_nums(raw_a)
        raw_nums = set(NUM_REGEX.findall(str(raw_a)))
        geo = extract_geo(n_addr, raw_a, c)
        is_indic = any(ord(ch) > 127 for ch in raw_n)
        is_generic = core_brand in GENERIC_WORDS or len(core_brand) <= 2

        s1_obj = {
            'core_brand': core_brand,
            'norm_name': n_name,
            'norm_addr': n_addr,
            'nums': nums,
            'geo': geo,
            'is_indic': is_indic,
            'is_generic': is_generic,
            'no_space': core_brand.replace(' ', ''),
        }

        idx_s = idx_stem[c]
        idx_sg = idx_stem_geo[c]
        idx_p = idx_pfx4[c]
        idx_nt = idx_num_tok[c]
        idx_ng = idx_num_geo[c]
        idx_pn = idx_pin[c]
        tgt_dict = t_by_c[c]

        cands = set()
        stems = [w for w in core_brand.split() if len(w) >= 2]
        stems.sort(key=lambda x: len(idx_s.get(x, [])))

        for s in stems:
            sc = idx_s.get(s, [])
            if len(sc) <= 200: cands.update(sc)
            elif geo: cands.update(idx_sg.get((s, geo), [])[:40])
            else: cands.update(sc[:40])
            if len(cands) >= 40: break

        atok = extract_addr_tokens(n_addr)
        for num in raw_nums:
            if len(num) >= 5: cands.update(idx_pn.get(num, [])[:20])
            if geo: cands.update(idx_ng.get((num, geo), [])[:20])
            for tok in atok[:2]: cands.update(idx_nt.get((num, tok), [])[:20])

        for tid in cands:
            t_obj = tgt_dict[tid]
            feat = compute_features(s1_obj, t_obj)
            val_pairs.append(feat)
            val_pair_meta.append((sid, tid))

    X_val = np.array(val_pairs, dtype=np.float32)
    print(f"Scoring {len(X_val):,} validation candidate pairs with LightGBM...")
    probs = bst.predict(X_val)

    # 6. Threshold Sweep for Macro F0.5
    print("\nSweeping Decision Thresholds for Optimal Macro F0.5...")
    best_f05 = 0.0
    best_th = 0.5
    best_preds = None

    for th in [0.60, 0.70, 0.75, 0.80, 0.85, 0.88, 0.90]:
        scored_pairs = []
        for i in range(len(probs)):
            if probs[i] >= th:
                sid, tid = val_pair_meta[i]
                scored_pairs.append((sid, tid, float(probs[i])))

        # Global Unique Matching
        scored_pairs.sort(key=lambda x: x[2], reverse=True)
        preds_th = defaultdict(list)
        assigned = set()
        best_by_s = {}

        for sid, tid, p in scored_pairs:
            if tid in assigned: continue
            if sid not in best_by_s: best_by_s[sid] = p
            if (best_by_s[sid] - p) <= 0.08:
                s2_c = sum(1 for x in preds_th[sid] if x.startswith("S2-"))
                s3_c = sum(1 for x in preds_th[sid] if x.startswith("S3-"))
                if tid.startswith("S2-") and s2_c < 3:
                    preds_th[sid].append(tid)
                    assigned.add(tid)
                elif tid.startswith("S3-") and s3_c < 3:
                    preds_th[sid].append(tid)
                    assigned.add(tid)

        res = evaluate_predictions(preds_th, val_gt_map, beta=0.5)
        print(f"Threshold = {th:.2f} -> Macro F0.5: {res['macro_f0_5']:.4f} | Prec: {res['macro_precision']:.4f} | Rec: {res['macro_recall']:.4f} | SingAcc: {res['singleton_accuracy']:.4f}")
        if res['macro_f0_5'] > best_f05:
            best_f05 = res['macro_f0_5']
            best_th = th
            best_preds = res

    print("\n" + "=" * 70)
    print(f"BEST LIGHTGBM RESULT: Threshold = {best_th:.2f} | Macro F0.5 = {best_f05:.4f}")
    print(f"Precision: {best_preds['macro_precision']:.4f} | Recall: {best_preds['macro_recall']:.4f}")
    print(f"Total Workflow Time: {time.time()-t0:.1f}s")
    print("=" * 70)

if __name__ == "__main__":
    train_and_eval_lgbm()
