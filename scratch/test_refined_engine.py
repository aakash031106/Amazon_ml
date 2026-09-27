import sys, os, time
sys.path.insert(0, os.getcwd())
import polars as pl
from collections import defaultdict
import unicodedata, re
from rapidfuzz import fuzz, distance
from src.evaluation import evaluate_predictions

def test_refined():
    t0 = time.time()
    val_s1 = pl.read_csv('data/val_split/val_s1.tsv', separator='\t')
    val_targets = pl.read_csv('data/val_split/val_targets.tsv', separator='\t')
    val_gt = pl.read_csv('data/val_split/val_gt.tsv', separator='\t')

    gt_map = {r['source1_entity_id']: [x.strip() for x in (r['matched_entity_ids'] or '').split(',') if x.strip()] for r in val_gt.iter_rows(named=True)}
    total_true = sum(len(v) for v in gt_map.values())

    # Expanded Generic Industry Words that should NEVER be matched alone without strong distinctive brand
    GENERIC_WORDS = {
        'hotel', 'hotels', 'restaurant', 'restro', 'cafe', 'dhaba', 'caterers', 'bakery', 'bakers',
        'hospital', 'clinic', 'nursing', 'dental', 'eye', 'care', 'medical', 'pharmacy', 'chemist', 'druggist',
        'store', 'mart', 'bazar', 'supermarket', 'hypermarket', 'grocery', 'kirana', 'general',
        'textiles', 'cloth', 'saree', 'garments', 'creations', 'boutique', 'collection', 'emporium',
        'hardware', 'electricals', 'electronics', 'telecom', 'mobile', 'motors', 'automobiles',
        'school', 'academy', 'classes', 'college', 'institute', 'education', 'tutorials',
        'enterprises', 'enterprise', 'solutions', 'services', 'group', 'holdings', 'associates', 'consulting',
        'industries', 'industry', 'engineering', 'construction', 'builders', 'developers', 'properties',
        'realty', 'real estate', 'trading', 'traders', 'agency', 'logistics', 'transport', 'carriers',
        'finance', 'investments', 'holdings', 'securities', 'capital', 'advisors'
    }

    GENERIC_REGEX = re.compile(
        r"\b(pvt|ltd|limited|private|llc|inc|corp|corporation|co|company|llp|gmbh|sa|sas|sarl|eurl|sci|snc|"
        r"enterprises|enterprise|solutions|services|group|holdings|associates|consulting|industries)\b",
        re.I
    )

    CLEAN_REGEX = re.compile(r"[^\w\s]")
    SPACE_REGEX = re.compile(r"\s+")
    URL_CLEAN = re.compile(r"\.(com|org|net|fr|in|co|io)\b", re.I)

    # Patterns to strip from address before number extraction to prevent false sector/phase overlaps
    SECTOR_STRIP = re.compile(r"\b(sector|sec|phase|ph|plot|plt|nh|gali|road|rd|floor|fl|ward)\s*[-#]?\s*\d+\b", re.I)
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

    STOP_ADDR = {
        "street", "saint", "st", "road", "rd", "avenue", "ave", "drive", "dr",
        "lane", "ln", "boulevard", "bd", "rue", "near", "nr", "opp", "null",
        "floor", "fl", "bldg", "block", "no", "flat", "none", "nan", "shop"
    }
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

    def extract_building_nums(raw_a):
        if not raw_a: return set()
        # First remove Sector 9, Phase 2, etc. so their numbers don't pollute street numbers
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
        elif country == "France":
            matches = re.findall(r"\b(\d{5})\b", raw_a or "")
            if matches: return matches[0][:2]
        return ""

    def extract_addr_tokens(norm_a):
        return [w for w in norm_a.split() if len(w) >= 4 and not w.isdigit() and w not in STOP_ADDR]

    # Index targets
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
        n_name = clean(raw_n)
        core_brand = brand(n_name)
        n_addr = clean(raw_a)
        nums = extract_building_nums(raw_a)
        raw_nums = set(NUM_REGEX.findall(str(raw_a)))
        geo = extract_geo(n_addr, raw_a, c)
        is_empty = (not n_addr) or (n_addr in NULL_ADDRS)
        is_indic = any(ord(ch) > 127 for ch in raw_n)
        is_generic_brand = core_brand in GENERIC_WORDS or len(core_brand) <= 2

        t_by_c[c][eid] = {
            'entity_id': eid,
            'core_brand': core_brand,
            'norm_name': n_name,
            'norm_addr': n_addr,
            'nums': nums,
            'geo': geo,
            'is_addr_empty': is_empty,
            'is_indic': is_indic,
            'is_generic_brand': is_generic_brand,
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
        for num in raw_nums:
            if len(num) >= 5: idx_pin_by_c[c][num].append(eid)
            if geo: idx_num_geo_by_c[c][(num, geo)].append(eid)
            for tok in atok[:4]: idx_num_tok_by_c[c][(num, tok)].append(eid)

    def score_pair_refined(s1, t):
        # 1. Hard Geographic Conflict Check
        if s1['geo'] and t['geo'] and s1['geo'] != t['geo']:
            return 0.0

        b1, b2 = s1['core_brand'], t['core_brand']
        a1, a2 = s1['norm_addr'], t['norm_addr']

        # Prevent generic word false merges (e.g. "Hotel" matching "Hotel" across town)
        if (s1['is_generic_brand'] or t['is_generic_brand']) and (b1 != b2 or not s1['nums'] or not (s1['nums'] & t['nums'])):
            return 0.0

        # Fast Brand Similarities
        if b1 == b2:
            b_sort = 100.0
            b_set = 100.0
        else:
            b_sort = fuzz.token_sort_ratio(b1, b2)
            b_set = fuzz.token_set_ratio(b1, b2)

        ns1, ns2 = s1['no_space'], t['no_space']
        if ns1 and ns2 and (ns1 in ns2 or ns2 in ns1):
            b_sort = max(b_sort, 92.0)
            b_set = max(b_set, 92.0)

        # First 2 words match fast path (e.g. "Maure Williams Colombier" <-> "Maure Williams Inc Center")
        w1 = b1.split()[:2]
        w2 = b2.split()[:2]
        if len(w1) >= 2 and len(w2) >= 2 and w1 == w2:
            b_sort = max(b_sort, 88.0)
            b_set = max(b_set, 90.0)

        # 2. Case: Target Address is Missing / Empty
        if t['is_addr_empty']:
            if b_sort >= 85 and distance.JaroWinkler.similarity(b1, b2) >= 0.85:
                return 0.92 + (b_sort / 1000.0)
            return 0.0

        # Building Number Overlap Check
        has_nums = bool(s1['nums'] and t['nums'])
        nums_overlap = bool(s1['nums'] & t['nums'])

        # Strict street number conflict: different building numbers = NEVER match unless brand is 98%+
        if has_nums and not nums_overlap and b_sort < 95:
            return 0.0

        if a1 == a2:
            a_sort = 100.0
            a_set = 100.0
        else:
            a_sort = fuzz.token_sort_ratio(a1, a2)
            a_set = fuzz.token_set_ratio(a1, a2)

        a_eff = max(a_sort, a_set)

        # 3. Case: Target Name is Transliterated (Indic Script / Non-ASCII)
        if t['is_indic'] and not s1['is_indic']:
            # Must ground on physical address
            if nums_overlap and a_eff >= 75:
                return 0.91 + (a_eff / 1000.0)
            if a_eff >= 88:
                return 0.90 + (a_eff / 1000.0)
            return 0.0

        # 4. Standard Case: Both Names & Addresses Present
        # Tier 1: Strong brand + good address
        if b_sort >= 68 and a_eff >= 50:
            return 0.90 + (0.5 * b_sort + 0.5 * a_eff) / 1000.0

        # Tier 2: Moderate brand + strong address
        if b_sort >= 50 and a_eff >= 68:
            return 0.88 + (0.4 * b_sort + 0.6 * a_eff) / 1000.0

        # Tier 3: Same building number + strong address containment (e.g. DBA or trade names)
        if has_nums and nums_overlap and a_eff >= 85:
            return 0.88 + (a_eff / 1000.0)

        # Tier 4: Exact brand + consistent address
        if b_sort >= 90 and a_eff >= 40:
            return 0.86 + (b_sort / 1000.0)

        return 0.0

    print("Running candidate retrieval and refined scoring...")
    all_scored = []
    cands_by_s1 = defaultdict(set)
    captured_cands = 0

    for r in val_s1.iter_rows(named=True):
        sid = r['entity_id']
        c = r['country'].strip()
        raw_n = r['business_name'] or ''
        raw_a = r['business_address'] or ''
        n_name = clean(raw_n)
        core_brand = brand(n_name)
        n_addr = clean(raw_a)
        nums = extract_building_nums(raw_a)
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
            'is_generic_brand': is_generic,
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

        # Also search no_space for URLs / compound names
        ns = s1_obj['no_space']
        if len(ns) >= 5:
            sc = idx_stem.get(ns, [])
            if len(sc) <= 200: cands.update(sc)
            elif geo: cands.update(idx_stem_geo.get((ns, geo), [])[:40])

        if len(cands) < 30 and stems and len(stems[0]) >= 4:
            pfx = stems[0][:4]
            pc = idx_pfx4.get(pfx, [])
            if len(pc) <= 200: cands.update(pc)
            elif geo: cands.update(idx_pfx4_geo.get((pfx, geo), [])[:40])
            else: cands.update(pc[:40])

        atok = extract_addr_tokens(n_addr)
        for num in raw_nums:
            if len(num) >= 5: cands.update(idx_pin.get(num, [])[:30])
            if geo: cands.update(idx_num_geo.get((num, geo), [])[:30])
            for tok in atok[:4]: cands.update(idx_num_tok.get((num, tok), [])[:30])

        cands_by_s1[sid] = cands
        true_tids = set(gt_map.get(sid, []))
        captured_cands += len(true_tids & cands)

        for tid in cands:
            s = score_pair_refined(s1_obj, tgt_dict[tid])
            if s >= 0.80:
                all_scored.append((sid, tid, s))

    # Global 1-to-1 Target Unique Matching
    preds_unique = defaultdict(list)
    assigned_targets = set()
    all_scored.sort(key=lambda x: x[2], reverse=True)
    best_by_s1 = {}
    for sid, tid, s in all_scored:
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

    res = evaluate_predictions(preds_unique, gt_map, beta=0.5)
    captured_preds = sum(len(set(preds_unique.get(sid, [])) & set(gt_map[sid])) for sid in gt_map)

    print("\n" + "=" * 65)
    print("REFINED SOTA RESULTS ON 10,000 S1 VALIDATION SET:")
    print("=" * 65)
    print(f"Total True Matches: {total_true:,}")
    print(f"Candidate Recall:   {captured_cands:,} / {total_true:,} ({captured_cands/total_true*100:.2f}%)")
    print(f"Captured Matches:   {captured_preds:,} / {total_true:,} ({captured_preds/total_true*100:.2f}%)")
    print(f"Macro F0.5:         {res['macro_f0_5']:.4f}")
    print(f"Macro Precision:    {res['macro_precision']:.4f}")
    print(f"Macro Recall:       {res['macro_recall']:.4f}")
    print(f"Singleton Accuracy: {res['singleton_accuracy']:.4f}")
    print(f"Total Runtime:      {time.time()-t0:.1f}s")
    print("=" * 65)

if __name__ == "__main__":
    test_refined()
