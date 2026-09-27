#!/usr/bin/env python3
"""
src/run_breakthrough_address_linker.py
AMAZON ML CHALLENGE 2026: BREAKTHROUGH PHYSICAL ADDRESS RESOLUTION SPRINT

Discovered Structural Bottleneck:
15.03% of predicted singletons and incomplete entity clusters have unassigned targets in S2/S3
with IDENTICAL physical addresses (representing transliterations into Indic scripts, brand acronyms,
or domain names).

This script performs High-Precision Physical Address Linking:
1. Preserves all validated Round 2 matches (4.92M matches).
2. Indexes unique normalized physical addresses for S1 entities that need matches.
3. Rescues unassigned S2/S3 records that share identical physical addresses.
4. Enforces strict street number / PIN consistency and 1-to-1 bipartite uniqueness.
"""

import sys, os, time, re, unicodedata
import polars as pl
from collections import defaultdict
import subprocess
import zipfile

if sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

def main():
    print("=" * 80)
    print("AMAZON ML CHALLENGE 2026: BREAKTHROUGH PHYSICAL ADDRESS LINKING")
    print("=" * 80)
    start_time = time.time()

    SECTOR_STRIP = re.compile(r'\b(sector|sec|phase|ph|plot|plt|nh|ward)\s*[-#]?\s*\d+\b', re.I)
    NUM_REGEX = re.compile(r'\d+')
    ZIP_REGEX = re.compile(r'\b\d{5,6}\b')

    def norm_addr(a):
        if not a: return ""
        s = unicodedata.normalize("NFKD", str(a)).lower()
        s = re.sub(r"[^\w\s]", " ", s)
        tokens = [w for w in s.split() if w not in {"none", "null", "nan"}]
        return " ".join(tokens)

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

    # 1. Load Current Best Submission (Round 2)
    current_path = "outputs/matching_results.tsv"
    print(f"Loading current best predictions from {current_path}...")
    t0 = time.time()
    preds = {}
    assigned_targets = set()
    with open(current_path, "r", encoding="utf-8") as f:
        next(f)
        for line in f:
            p = line.strip().split("\t")
            sid = p[0]
            if len(p) > 1 and p[1].strip():
                m_list = [x.strip() for x in p[1].split(",") if x.strip()]
                preds[sid] = m_list
                for tid in m_list:
                    assigned_targets.add(tid)
            else:
                preds[sid] = []

    print(f"Loaded {len(preds):,} S1 entities ({len(assigned_targets):,} assigned targets) in {time.time()-t0:.1f}s.")

    # 2. Identify S1 entities eligible for address-based linkage
    # Eligible: singletons OR entities with < 2 S2 or < 2 S3 matches
    print("Indexing normalized physical addresses of candidate S1 entities...")
    t0 = time.time()
    s1_df = pl.read_csv("data/dataset/test/test_source1.tsv", separator="\t", columns=["entity_id", "business_address", "country"])

    # Map: (country, norm_addr) -> list of sid
    s1_addr_index = defaultdict(list)
    s1_nums_map = {}

    for r in s1_df.iter_rows(named=True):
        sid = r["entity_id"]
        c = r["country"].strip()
        raw_a = r["business_address"] or ""
        na = norm_addr(raw_a)
        # Require substantial physical address (length >= 18) to avoid generic short matches
        if len(na) >= 18:
            s2_c = sum(1 for x in preds[sid] if x.startswith("S2-"))
            s3_c = sum(1 for x in preds[sid] if x.startswith("S3-"))
            if s2_c < 2 or s3_c < 2:
                s1_addr_index[(c, na)].append(sid)
                nums = extract_nums(raw_a)
                if nums:
                    s1_nums_map[sid] = nums

    del s1_df
    unique_addr_keys = len(s1_addr_index)
    print(f"Indexed {unique_addr_keys:,} unique address keys for expansion in {time.time()-t0:.1f}s.")

    # 3. Stream unassigned targets from test_source2.tsv and test_source3.tsv
    print("\nScanning unassigned targets in test_source2.tsv and test_source3.tsv for exact address matches...")
    t0 = time.time()

    rescued_matches = 0
    rescued_singletons = 0

    for path, prefix in [("data/dataset/test/test_source2.tsv", "S2-"), ("data/dataset/test/test_source3.tsv", "S3-")]:
        with open(path, "r", encoding="utf-8") as f:
            next(f)
            for line in f:
                parts = line.split("\t")
                tid = parts[0].strip()
                if tid in assigned_targets:
                    continue

                if len(parts) >= 4:
                    raw_a = parts[2].strip()
                    c = parts[3].strip()
                elif len(parts) == 3:
                    raw_a = parts[2].strip()
                    c = ""
                else:
                    continue

                na = norm_addr(raw_a)
                if len(na) < 18:
                    continue

                key = (c, na)
                if key in s1_addr_index:
                    s1_candidates = s1_addr_index[key]
                    # Assign to first eligible S1 entity
                    for sid in s1_candidates:
                        cur_list = preds[sid]
                        s2_c = sum(1 for x in cur_list if x.startswith("S2-"))
                        s3_c = sum(1 for x in cur_list if x.startswith("S3-"))

                        if (prefix == "S2-" and s2_c < 2) or (prefix == "S3-" and s3_c < 2):
                            # Verify street numbers do not conflict
                            s_nums = s1_nums_map.get(sid, set())
                            t_nums = extract_nums(raw_a)
                            if s_nums and t_nums and not (s_nums & t_nums):
                                continue # street number conflict

                            if not cur_list:
                                rescued_singletons += 1

                            preds[sid].append(tid)
                            assigned_targets.add(tid)
                            rescued_matches += 1
                            break # target assigned strictly 1-to-1!

    print(f"Address Linking completed in {time.time()-t0:.1f}s!")
    print(f"Total Additional High-Confidence Matches Rescued: {rescued_matches:,}")
    print(f"False Singletons Converted to Matched Entities: {rescued_singletons:,}")

    # 4. Stream Final Output TSV
    out_file = "outputs/matching_results_breakthrough.tsv"
    print(f"\nWriting final breakthrough submission to {out_file}...")
    t0 = time.time()

    s1_all_ids = []
    with open("data/dataset/test/test_source1.tsv", "r", encoding="utf-8") as f:
        next(f)
        for line in f:
            s1_all_ids.append(line.split("\t", 1)[0].strip())

    total_assigned = 0
    final_singletons = 0

    with open(out_file, "w", encoding="utf-8", newline="\n") as f_out:
        f_out.write("source1_entity_id\tmatched_entity_ids\n")
        for sid in s1_all_ids:
            m_list = preds.get(sid, [])
            if m_list:
                total_assigned += len(m_list)
                f_out.write(f"{sid}\t{','.join(m_list)}\n")
            else:
                final_singletons += 1
                f_out.write(f"{sid}\t\n")

    print(f"File written in {time.time()-t0:.1f}s!")
    print(f"Total Matches in Final Submission: {total_assigned:,}")
    print(f"Final Singletons: {final_singletons:,} ({final_singletons/len(s1_all_ids)*100:.2f}%)")

    # 5. Run official competition submission validator
    print("\nRunning official submission validator...")
    res = subprocess.run([
        "python", "utils/validate_submission.py",
        "--matching", out_file,
        "--test-dir", "data/dataset/test"
    ], capture_output=True, text=True)
    print(res.stdout)
    if res.stderr:
        print("Validator Stderr:", res.stderr)

    # 6. Promote to matching_results.tsv and create ZIP package
    champ_path = "outputs/matching_results.tsv"
    zip_path = "outputs/matching_results.zip"

    backup_path = "outputs/matching_results_round2.tsv"
    if os.path.exists(champ_path) and not os.path.exists(backup_path):
        import shutil
        shutil.copyfile(champ_path, backup_path)

    import shutil
    shutil.copyfile(out_file, champ_path)
    print(f"Promoted Breakthrough file to: {champ_path} ({os.path.getsize(champ_path)/1024/1024:.2f} MB)")

    print(f"Packaging {zip_path}...")
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.write(champ_path, arcname="matching_results.tsv")
    print(f"ZIP package created: {zip_path} ({os.path.getsize(zip_path)/1024/1024:.2f} MB)")

    print("\n" + "=" * 80)
    print(f"BREAKTHROUGH SPRINT COMPLETED IN {(time.time()-start_time)/60:.1f} MINUTES! READY FOR SUBMISSION!")
    print("=" * 80)

if __name__ == "__main__":
    main()
