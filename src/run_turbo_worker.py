#!/usr/bin/env python3
"""
Turbo Worker Engine for Amazon ML Challenge 2026
Executes high-precision matching for a specific country partition.
High-throughput C++ RapidFuzz + Polars Vectorized Normalization.
"""

import argparse
import os
import sys
import time
from collections import defaultdict
import polars as pl
from rapidfuzz import fuzz, distance

FRENCH_ABBR_PAIRS = [
    (r"\bbd\b", "boulevard"), (r"\bav\b", "avenue"), (r"\br\b", "rue"),
    (r"\bpl\b", "place"), (r"\ball\b", "allee"), (r"\brte\b", "route"),
    (r"\bn°\b", " "), (r"\bbis\b", "b"), (r"\bter\b", "t")
]

US_INDIA_ABBR_PAIRS = [
    (r"\bst\b", "street"), (r"\brd\b", "road"), (r"\bave\b", "avenue"),
    (r"\bdr\b", "drive"), (r"\bln\b", "lane"), (r"\bhwy\b", "highway"),
    (r"\bste\b", "suite"), (r"\bapt\b", "apartment"), (r"\bopp\b", "opposite"),
    (r"\bnr\b", "near"),
]

GENERIC_REGEX = (
    r"\b(pvt|ltd|limited|private|llc|inc|corp|corporation|co|company|llp|gmbh|sa|sas|sarl|eurl|"
    r"france|india|usa|us|enterprises|enterprise|solutions|services|group|holdings|associates|"
    r"consulting|industries|pharmacy|chemist|medical|druggist|store|mart|bazar|supermarket|"
    r"hypermarket|grocery|restaurant|restro|cafe|hotel|dhaba|bhojanalaya|caterers|bakers|bakery|"
    r"sweets|jewellers|jeweller|textiles|cloth|saree|creations|boutique|collection|emporium|"
    r"stationery|hardware|electricals|electronics|mobile|telecom|motors|automobiles|hospital|"
    r"nursing|clinic|pathology|diagnostic|dental|eye|care|school|academy|classes|college|"
    r"institute|education|tutorials)\b"
)

STOP_ADDR = {
    "street", "saint", "st", "road", "rd", "avenue", "ave", "drive", "dr",
    "lane", "ln", "boulevard", "bd", "rue", "near", "nr", "opp", "null",
    "floor", "fl", "bldg", "block", "no", "flat"
}

US_STATES = {
    "al", "ak", "az", "ar", "ca", "co", "ct", "de", "fl", "ga",
    "hi", "id", "il", "in", "ia", "ks", "ky", "la", "me", "md",
    "ma", "mi", "mn", "ms", "mo", "mt", "ne", "nv", "nh", "nj",
    "nm", "ny", "nc", "nd", "oh", "ok", "or", "pa", "ri", "sc",
    "sd", "tn", "tx", "ut", "vt", "va", "wa", "wv", "wi", "wy"
}


def build_vectorized_df(df: pl.DataFrame, country: str) -> pl.DataFrame:
    abbr_pairs = FRENCH_ABBR_PAIRS if country == "France" else US_INDIA_ABBR_PAIRS

    name_expr = (
        pl.col("business_name").fill_null("")
        .str.to_lowercase()
        .str.replace_all("&", " and ")
        .str.replace_all(r"[^\w\s]", " ")
        .str.replace_all(r"\s+", " ")
        .str.strip_chars()
    )

    addr_expr = (
        pl.col("business_address").fill_null("")
        .str.to_lowercase()
        .str.replace_all("&", " and ")
        .str.replace_all(r"[^\w\s]", " ")
        .str.replace_all(r"\s+", " ")
        .str.strip_chars()
    )

    for pat, rep in abbr_pairs:
        addr_expr = addr_expr.str.replace_all(pat, rep)

    df = df.with_columns([
        name_expr.alias("norm_name"),
        addr_expr.alias("norm_addr"),
    ]).with_columns([
        pl.col("norm_name").str.replace_all(GENERIC_REGEX, " ").str.replace_all(r"\s+", " ").str.strip_chars().alias("core_brand_raw"),
        pl.col("norm_addr").str.extract_all(r"\b\d+\b").alias("nums_raw")
    ]).with_columns([
        pl.when(pl.col("core_brand_raw") == "")
        .then(pl.col("norm_name"))
        .otherwise(pl.col("core_brand_raw"))
        .alias("core_brand")
    ])

    return df


def extract_state(norm_addr: str) -> str:
    for t in norm_addr.split():
        if t in US_STATES:
            return t
    return ""


def run_worker(country: str, part: int, num_parts: int, output_dir: str):
    t_start = time.time()
    tag = f"[{country} Part {part+1}/{num_parts}]"
    print(f"\n{'='*70}", flush=True)
    print(f"{tag} Starting Turbo Worker...", flush=True)
    print(f"{'='*70}", flush=True)

    os.makedirs(output_dir, exist_ok=True)
    cand_out_path = os.path.join(output_dir, f"candidate_{country}_p{part}.tsv")
    match_out_path = os.path.join(output_dir, f"matching_{country}_p{part}.tsv")

    # 1. Load Country Targets
    print(f"{tag} Loading target pool from test_source2 and test_source3...", flush=True)
    t0 = time.time()
    s2_df = pl.read_csv("data/dataset/test/test_source2.tsv", separator="\t", columns=["entity_id", "business_name", "business_address", "country"])
    s3_df = pl.read_csv("data/dataset/test/test_source3.tsv", separator="\t", columns=["entity_id", "business_name", "business_address", "country"])
    s2_c = s2_df.filter(pl.col("country").str.strip_chars() == country)
    s3_c = s3_df.filter(pl.col("country").str.strip_chars() == country)
    target_df = pl.concat([s2_c, s3_c]).unique(subset=["entity_id"])
    del s2_df, s3_df, s2_c, s3_c
    import gc; gc.collect()

    print(f"{tag} Normalizing {len(target_df):,} targets with Polars vectorized Rust...", flush=True)
    target_df = build_vectorized_df(target_df, country)

    t_eids = target_df["entity_id"].to_list()
    t_brands = target_df["core_brand"].to_list()
    t_names = target_df["norm_name"].to_list()
    t_addrs = target_df["norm_addr"].to_list()
    t_nums_raw = target_df["nums_raw"].to_list()
    t_nums = [frozenset(str(int(n)) for n in (nl or []) if n.isdigit()) for nl in t_nums_raw]
    t_nospace = [n.replace(" ", "") for n in t_names]
    t_empty = [not bool(a) for a in t_addrs]
    t_states = [extract_state(a) if country == "US" else "" for a in t_addrs]
    del target_df
    gc.collect()

    # 2. Build Inverted Indexes
    print(f"{tag} Building inverted indexes for {len(t_eids):,} targets...", flush=True)
    t_idx = time.time()
    index_stem = defaultdict(list)
    index_pfx4 = defaultdict(list)
    index_pin = defaultdict(list)
    index_num_addr = defaultdict(list)
    index_num_state = defaultdict(list)

    for i in range(len(t_eids)):
        brand = t_brands[i]
        addr = t_addrs[i]
        nums = t_nums[i]
        state = t_states[i]
        stems = [w for w in brand.split() if len(w) >= 2]
        for s in stems:
            index_stem[s].append(i)
        if stems and len(stems[0]) >= 4:
            index_pfx4[stems[0][:4]].append(i)
        addr_tokens = [t for t in addr.split() if len(t) >= 3 and not t.isdigit() and t not in STOP_ADDR]
        for n in nums:
            if len(n) >= 5:
                index_pin[n].append(i)
            if state:
                index_num_state[(n, state)].append(i)
            for at in addr_tokens[:2]:
                index_num_addr[(n, at)].append(i)

    print(f"{tag} Indexed in {time.time()-t_idx:.1f}s.", flush=True)

    # 3. Load Partitioned S1 Entities
    print(f"{tag} Loading test_source1 for {country}...", flush=True)
    s1_all = pl.read_csv("data/dataset/test/test_source1.tsv", separator="\t")
    s1_country = s1_all.filter(pl.col("country").str.strip_chars() == country)
    total_country_entities = len(s1_country)

    # Slice partition
    chunk_size = (total_country_entities + num_parts - 1) // num_parts
    start_idx = part * chunk_size
    end_idx = min(start_idx + chunk_size, total_country_entities)
    s1_partition = s1_country.slice(start_idx, end_idx - start_idx)
    del s1_all, s1_country
    gc.collect()

    print(f"{tag} Normalizing partition: {len(s1_partition):,} entities (rows {start_idx:,} to {end_idx:,})...", flush=True)
    s1_partition = build_vectorized_df(s1_partition, country)

    s1_eids = s1_partition["entity_id"].to_list()
    s1_brands = s1_partition["core_brand"].to_list()
    s1_names = s1_partition["norm_name"].to_list()
    s1_addrs = s1_partition["norm_addr"].to_list()
    s1_nums_raw = s1_partition["nums_raw"].to_list()
    s1_nums = [frozenset(str(int(n)) for n in (nl or []) if n.isdigit()) for nl in s1_nums_raw]
    s1_nospace = [n.replace(" ", "") for n in s1_names]
    s1_states = [extract_state(a) if country == "US" else "" for a in s1_addrs]
    s1_stems = [[w for w in b.split() if len(w) >= 2] for b in s1_brands]
    s1_addr_tokens = [[t for t in a.split() if len(t) >= 3 and not t.isdigit() and t not in STOP_ADDR] for a in s1_addrs]
    del s1_partition
    gc.collect()

    # Sort stems by rarity
    for stems in s1_stems:
        stems.sort(key=lambda s: len(index_stem.get(s, [])))

    # 4. Inference Execution
    print(f"{tag} Starting high-speed inference for {len(s1_eids):,} entities...", flush=True)
    t_inf = time.time()
    matched_count = 0
    total_matches = 0
    min_thresh = 0.80
    margin = 0.05
    max_per_source = 3
    BATCH_PRINT = 10000

    f_cand = open(cand_out_path, "w", encoding="utf-8")
    f_match = open(match_out_path, "w", encoding="utf-8")

    for i in range(len(s1_eids)):
        eid = s1_eids[i]
        b1 = s1_brands[i]
        ns1 = s1_nospace[i]
        a1 = s1_addrs[i]
        num1 = s1_nums[i]
        stems = s1_stems[i]
        state1 = s1_states[i]
        addr_toks = s1_addr_tokens[i]

        cands = set()
        for s in stems:
            sc = index_stem.get(s, [])
            if len(sc) <= 200:
                cands.update(sc)
            elif state1:
                cands.update([tid for tid in sc if t_states[tid] == state1][:100])
            else:
                cands.update(sc[:100])
            if len(cands) >= 50:
                break

        if len(cands) < 30 and stems and len(stems[0]) >= 4:
            pc = index_pfx4.get(stems[0][:4], [])
            if len(pc) <= 200:
                cands.update(pc)
            elif state1:
                cands.update([tid for tid in pc if t_states[tid] == state1][:80])
            else:
                cands.update(pc[:80])

        for n in num1:
            if len(n) >= 5:
                cands.update(index_pin.get(n, []))
            if state1:
                cands.update(index_num_state.get((n, state1), []))
            for at in addr_toks[:2]:
                cands.update(index_num_addr.get((n, at), []))

        scored = []
        for idx in cands:
            # US state consistency check
            if state1 and t_states[idx] and state1 != t_states[idx]:
                continue

            b2 = t_brands[idx]
            ns2 = t_nospace[idx]
            num2 = t_nums[idx]

            # Fast brand calculation with equality shortcut
            if b1 == b2:
                brand_sort = 100.0
            else:
                brand_sort = fuzz.token_sort_ratio(b1, b2)

            if ns1 and ns2 and (ns1 in ns2 or ns2 in ns1):
                brand_sort = max(brand_sort, 92.0)

            # Target address is empty
            if t_empty[idx]:
                if brand_sort >= 85 and distance.JaroWinkler.similarity(b1, b2) >= 0.85:
                    scored.append((idx, 0.92 + brand_sort / 1000.0))
                continue

            # Strict street number conflict check (CRITICAL FOR MACRO F0.5)
            has_nums = bool(num1 and num2)
            nums_overlap = bool(num1 & num2)
            if has_nums and not nums_overlap and brand_sort < 95:
                continue

            # Early exit: skip address comparison if brand is completely dissimilar
            if brand_sort < 55 and not (has_nums and nums_overlap):
                continue

            a2 = t_addrs[idx]
            if a1 == a2:
                a_sort = 100.0
            else:
                a_sort = fuzz.token_sort_ratio(a1, a2)

            # Match evaluation
            if brand_sort >= 70 and a_sort >= 50:
                scored.append((idx, 0.90 + (0.5 * brand_sort + 0.5 * a_sort) / 1000.0))
            elif brand_sort >= 55 and a_sort >= 68:
                scored.append((idx, 0.88 + (0.4 * brand_sort + 0.6 * a_sort) / 1000.0))
            elif has_nums and nums_overlap and a_sort >= 80:
                scored.append((idx, 0.88 + a_sort / 1000.0))
            elif brand_sort >= 88:
                scored.append((idx, 0.85 + brand_sort / 1000.0))

        matched = []
        if scored:
            scored.sort(key=lambda x: x[1], reverse=True)
            top_s = scored[0][1]
            s2_c, s3_c = 0, 0
            for idx, s in scored:
                if s >= min_thresh and (top_s - s) <= margin:
                    eid_m = t_eids[idx]
                    if eid_m.startswith("S2-") and s2_c < max_per_source:
                        matched.append(eid_m)
                        s2_c += 1
                    elif eid_m.startswith("S3-") and s3_c < max_per_source:
                        matched.append(eid_m)
                        s3_c += 1

        cand_str = ",".join(t_eids[idx] for idx in cands)
        match_str = ",".join(matched)
        f_cand.write(f"{eid}\t{cand_str}\n")
        f_match.write(f"{eid}\t{match_str}\n")

        if matched:
            matched_count += 1
            total_matches += len(matched)

        if (i + 1) % BATCH_PRINT == 0 or (i + 1) == len(s1_eids):
            avg_m = (total_matches / matched_count) if matched_count > 0 else 0.0
            rate = (i + 1) / (time.time() - t_inf)
            print(f"{tag} {i+1:,}/{len(s1_eids):,} done | Speed: {rate:.1f} ent/s | Matched: {matched_count:,} ({matched_count/(i+1)*100:.1f}%) | Avg matches: {avg_m:.2f}", flush=True)

    f_cand.close()
    f_match.close()
    print(f"\n{tag} COMPLETED in {time.time() - t_start:.1f}s! Total: {len(s1_eids):,}, Matched: {matched_count:,}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--country", required=True, choices=["France", "India", "US"])
    parser.add_argument("--part", type=int, default=0)
    parser.add_argument("--num_parts", type=int, default=1)
    parser.add_argument("--output_dir", default="outputs/parts")
    args = parser.parse_args()

    run_worker(args.country, args.part, args.num_parts, args.output_dir)
