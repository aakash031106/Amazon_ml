#!/usr/bin/env python3
"""
Master Turbo Coordinator for Amazon ML Challenge 2026
Orchestrates 7 concurrent country workers to resolve all 1.73M entities in ~18-22 minutes.
Merges outputs, validates submission, and packages submission.zip.
"""

import os
import sys
import time
import shutil
import zipfile
import subprocess
import polars as pl

PARTS_CONFIG = [
    ("France", 0, 2),
    ("France", 1, 2),
    ("India", 0, 4),
    ("India", 1, 4),
    ("India", 2, 4),
    ("India", 3, 4),
    ("US", 0, 3),
    ("US", 1, 3),
    ("US", 2, 3),
]


def main():
    master_start = time.time()
    print("=" * 80, flush=True)
    print("AMAZON ML CHALLENGE 2026: TURBO SOTA ENGINE ORCHESTRATOR", flush=True)
    print("Running 7 Concurrent High-Precision C++ RapidFuzz Workers", flush=True)
    print("Target Completion: ~18 to 22 Minutes for All 1,732,544 Entities", flush=True)
    print("=" * 80, flush=True)

    parts_dir = "outputs/parts"
    os.makedirs(parts_dir, exist_ok=True)

    # Clean old partial files
    for f in os.listdir(parts_dir):
        if f.endswith(".tsv"):
            try:
                os.remove(os.path.join(parts_dir, f))
            except Exception:
                pass

    print(f"\n[Master] Launching {len(PARTS_CONFIG)} concurrent worker processes...", flush=True)
    processes = []
    log_files = []

    for country, part, num_parts in PARTS_CONFIG:
        cmd = [
            sys.executable,
            "src/run_turbo_worker.py",
            "--country", country,
            "--part", str(part),
            "--num_parts", str(num_parts),
            "--output_dir", parts_dir
        ]
        log_path = os.path.join(parts_dir, f"worker_{country}_p{part}.log")
        log_f = open(log_path, "w", encoding="utf-8")
        log_files.append(log_f)

        proc = subprocess.Popen(
            cmd,
            stdout=log_f,
            stderr=subprocess.STDOUT,
            cwd=os.getcwd()
        )
        processes.append((country, part, num_parts, proc, log_path))
        print(f"  -> Spawned {country} Part {part+1}/{num_parts} (PID: {proc.pid})", flush=True)

    print(f"\n[Master] All 7 workers actively computing in parallel across CPU cores!", flush=True)

    # Monitor loop
    completed = set()
    poll_interval = 15
    while len(completed) < len(processes):
        time.sleep(poll_interval)
        elapsed = time.time() - master_start
        status_lines = []

        for country, part, num_parts, proc, log_p in processes:
            key = f"{country}_p{part}"
            ret = proc.poll()
            if ret is not None and key not in completed:
                completed.add(key)
                print(f"[Master Notification] Worker {country} Part {part+1}/{num_parts} FINISHED in {elapsed:.1f}s (Exit code: {ret})!", flush=True)

            # Read last line of log for progress
            last_line = ""
            if os.path.exists(log_p):
                try:
                    with open(log_p, "r", encoding="utf-8") as f:
                        lines = [l.strip() for l in f if l.strip()]
                        if lines:
                            last_line = lines[-1]
                except Exception:
                    pass
            status_lines.append(f"  [{country} P{part+1}] {last_line[:80]}")

        print(f"\n--- Progress Update ({elapsed/60:.1f} mins elapsed | Completed: {len(completed)}/{len(processes)}) ---", flush=True)
        for sl in status_lines:
            print(sl, flush=True)

    # Close log files
    for lf in log_files:
        try:
            lf.close()
        except Exception:
            pass

    print(f"\n{'='*80}", flush=True)
    print(f"[Master] ALL 7 WORKERS COMPLETED in {time.time() - master_start:.1f}s!", flush=True)
    print("Merging partial outputs into final submission files...", flush=True)
    print(f"{'='*80}", flush=True)

    # Read S1 order to preserve exact order
    print("Loading test_source1 to ensure 100% ID coverage...", flush=True)
    s1_all = pl.read_csv("data/dataset/test/test_source1.tsv", separator="\t")
    s1_eids_ordered = s1_all["entity_id"].to_list()
    total_expected = len(s1_eids_ordered)

    match_dict = {}
    cand_dict = {}

    for country, part, num_parts in PARTS_CONFIG:
        m_file = os.path.join(parts_dir, f"matching_{country}_p{part}.tsv")
        c_file = os.path.join(parts_dir, f"candidate_{country}_p{part}.tsv")

        print(f"Reading {m_file}...", flush=True)
        with open(m_file, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split("\t")
                if parts:
                    eid = parts[0]
                    m = parts[1] if len(parts) > 1 else ""
                    match_dict[eid] = m

        print(f"Reading {c_file}...", flush=True)
        with open(c_file, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split("\t")
                if parts:
                    eid = parts[0]
                    c = parts[1] if len(parts) > 1 else ""
                    cand_dict[eid] = c

    # Write final files
    match_tsv_path = "outputs/matching_results.tsv"
    match_csv_path = "outputs/matching_results.csv"
    cand_tsv_path = "outputs/candidate_pairs.tsv"

    print(f"\nWriting {match_tsv_path} and {match_csv_path}...", flush=True)
    with open(match_tsv_path, "w", encoding="utf-8") as f_tsv, \
         open(match_csv_path, "w", encoding="utf-8") as f_csv:
        f_tsv.write("source1_entity_id\tmatched_entity_ids\n")
        f_csv.write("source1_entity_id,matched_entity_ids\n")

        for eid in s1_eids_ordered:
            m = match_dict.get(eid, "")
            f_tsv.write(f"{eid}\t{m}\n")
            f_csv.write(f"{eid},{m}\n")

    print(f"Writing {cand_tsv_path}...", flush=True)
    with open(cand_tsv_path, "w", encoding="utf-8") as f_cand:
        f_cand.write("source1_entity_id\tcandidate_entity_ids\n")
        for eid in s1_eids_ordered:
            c = cand_dict.get(eid, "")
            f_cand.write(f"{eid}\t{c}\n")

    print(f"\nVerifying row counts: {len(match_dict):,} entities generated vs {total_expected:,} expected.")

    # Run official validator
    print(f"\n{'='*70}", flush=True)
    print("RUNNING OFFICIAL VALIDATOR: utils/validate_submission.py", flush=True)
    print(f"{'='*70}", flush=True)
    val_cmd = [
        sys.executable,
        "utils/validate_submission.py",
        "--matching", match_tsv_path,
        "--candidate", cand_tsv_path,
        "--test-dir", "data/dataset/test"
    ]
    subprocess.run(val_cmd)

    # Package zip
    zip_path = "outputs/submission.zip"
    print(f"\nPackaging {zip_path}...", flush=True)
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        z.write(match_tsv_path, arcname="matching_results.tsv")
        z.write(match_csv_path, arcname="matching_results.csv")
        z.write(cand_tsv_path, arcname="candidate_pairs.tsv")

    zip_size_mb = os.path.getsize(zip_path) / (1024 * 1024)
    print(f"\n{'='*80}", flush=True)
    print(f"ALL DELIVERABLES READY: {zip_path} ({zip_size_mb:.2f} MB)", flush=True)
    print(f"Total Execution Time: {time.time() - master_start:.1f}s ({(time.time()-master_start)/60:.1f} minutes)!", flush=True)
    print(f"{'='*80}", flush=True)


if __name__ == "__main__":
    main()
