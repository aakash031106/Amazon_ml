"""
src/run_data_audit.py
Comprehensive Data Audit Script for Amazon ML Challenge 2026
Business Entity Resolution
"""

import os
import sys
import glob
import time
import json
import re
from collections import Counter, defaultdict
import pandas as pd
import numpy as np

# Ensure utf-8 stdout
if sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

DATA_DIR = "data/dataset"
CHUNK_SIZE = 250_000

def audit_sources():
    print("--- Auditing Source Tables (Train & Test) ---")
    files = {
        'train_source1': os.path.join(DATA_DIR, 'train', 'train_source1.tsv'),
        'train_source2': os.path.join(DATA_DIR, 'train', 'train_source2.tsv'),
        'train_source3': os.path.join(DATA_DIR, 'train', 'train_source3.tsv'),
        'test_source1': os.path.join(DATA_DIR, 'test', 'test_source1.tsv'),
        'test_source2': os.path.join(DATA_DIR, 'test', 'test_source2.tsv'),
        'test_source3': os.path.join(DATA_DIR, 'test', 'test_source3.tsv'),
    }

    profiles = []
    country_counts = {}
    name_stats = {}
    address_stats = {}

    for name, path in files.items():
        print(f"\nProcessing {name} ({path})...")
        t0 = time.time()
        file_size_mb = os.path.getsize(path) / (1024 * 1024)

        total_rows = 0
        null_counts = defaultdict(int)
        empty_counts = defaultdict(int)
        seen_ids = set()
        duplicate_id_count = 0
        expected_prefix = 'S1-' if 'source1' in name else ('S2-' if 'source2' in name else 'S3-')
        bad_prefix_count = 0

        c_counter = Counter()
        name_len_samples = []
        name_words_samples = []
        addr_len_samples = []
        addr_words_samples = []

        # Common legal suffix counters
        legal_suffix_counter = Counter()
        pin_counter = 0

        # Sample for name / addr inspection
        sample_records = []

        chunk_idx = 0
        for chunk in pd.read_csv(path, sep='\t', chunksize=CHUNK_SIZE, dtype=str, keep_default_na=False):
            total_rows += len(chunk)
            chunk_idx += 1

            # Check nulls and empties
            for col in chunk.columns:
                # empty strings
                empties = (chunk[col].str.strip() == '').sum()
                empty_counts[col] += empties

            # ID checks
            ids = chunk['entity_id'].values
            for eid in ids:
                if eid in seen_ids:
                    duplicate_id_count += 1
                else:
                    seen_ids.add(eid)
                if not eid.startswith(expected_prefix):
                    bad_prefix_count += 1

            # Country counts
            c_counter.update(chunk['country'].value_counts().to_dict())

            # Sample lengths from first chunk
            if chunk_idx == 1:
                names = chunk['business_name'].fillna('')
                addrs = chunk['business_address'].fillna('')
                
                n_lens = names.str.len().tolist()
                n_words = names.str.split().str.len().tolist()
                name_len_samples.extend(n_lens)
                name_words_samples.extend(n_words)

                a_lens = addrs.str.len().tolist()
                a_words = addrs.str.split().str.len().tolist()
                addr_len_samples.extend(a_lens)
                addr_words_samples.extend(a_words)

            if len(sample_records) < 10:
                sample_records.extend(chunk.head(2).to_dict(orient='records'))

        dt = time.time() - t0
        print(f"Done {name} in {dt:.1f}s: {total_rows:,} rows, {duplicate_id_count} duplicate IDs, Countries: {dict(c_counter)}")

        country_counts[name] = dict(c_counter)

        profiles.append({
            'file_name': name + '.tsv',
            'file_path': path,
            'size_mb': round(file_size_mb, 2),
            'total_rows': total_rows,
            'columns': list(chunk.columns),
            'num_columns': len(chunk.columns),
            'duplicate_ids': duplicate_id_count,
            'bad_prefix_count': bad_prefix_count,
            'empty_name_count': empty_counts.get('business_name', 0),
            'empty_address_count': empty_counts.get('business_address', 0),
            'empty_country_count': empty_counts.get('country', 0),
            'unique_countries': list(c_counter.keys()),
        })

        name_stats[name] = {
            'mean_length': float(np.mean(name_len_samples)) if name_len_samples else 0.0,
            'median_length': float(np.median(name_len_samples)) if name_len_samples else 0.0,
            'min_length': int(np.min(name_len_samples)) if name_len_samples else 0,
            'max_length': int(np.max(name_len_samples)) if name_len_samples else 0,
            'mean_words': float(np.mean(name_words_samples)) if name_words_samples else 0.0,
            'median_words': float(np.median(name_words_samples)) if name_words_samples else 0.0,
        }
        address_stats[name] = {
            'mean_length': float(np.mean(addr_len_samples)) if addr_len_samples else 0.0,
            'median_length': float(np.median(addr_len_samples)) if addr_len_samples else 0.0,
            'min_length': int(np.min(addr_len_samples)) if addr_len_samples else 0,
            'max_length': int(np.max(addr_len_samples)) if addr_len_samples else 0,
            'mean_words': float(np.mean(addr_words_samples)) if addr_words_samples else 0.0,
            'median_words': float(np.median(addr_words_samples)) if addr_words_samples else 0.0,
        }

    return profiles, country_counts, name_stats, address_stats


def audit_ground_truth():
    print("\n--- Auditing Ground Truth ---")
    gt_path = os.path.join(DATA_DIR, 'train', 'train_ground_truth.tsv')
    file_size_mb = os.path.getsize(gt_path) / (1024 * 1024)

    t0 = time.time()
    total_rows = 0
    zero_matches = 0
    single_matches = 0
    multi_matches = 0
    match_distribution = Counter()
    source_breakdown = Counter() # e.g. S2 only, S3 only, Both
    total_matched_entities = 0
    all_matched_ids = set()
    duplicate_s1_in_gt = 0
    seen_s1 = set()

    sample_zero = []
    sample_single = []
    sample_multi = []

    for chunk in pd.read_csv(gt_path, sep='\t', chunksize=CHUNK_SIZE, dtype=str, keep_default_na=False):
        total_rows += len(chunk)
        for _, row in chunk.iterrows():
            s1_id = row['source1_entity_id'].strip()
            if s1_id in seen_s1:
                duplicate_s1_in_gt += 1
            seen_s1.add(s1_id)

            m_str = row['matched_entity_ids'].strip()
            if not m_str:
                zero_matches += 1
                match_distribution[0] += 1
                if len(sample_zero) < 5:
                    sample_zero.append(s1_id)
            else:
                m_list = [x.strip() for x in m_str.split(',') if x.strip()]
                num_m = len(m_list)
                match_distribution[num_m] += 1
                total_matched_entities += num_m
                all_matched_ids.update(m_list)

                has_s2 = any(x.startswith('S2-') for x in m_list)
                has_s3 = any(x.startswith('S3-') for x in m_list)
                if has_s2 and has_s3:
                    source_breakdown['both_s2_and_s3'] += 1
                elif has_s2:
                    source_breakdown['s2_only'] += 1
                elif has_s3:
                    source_breakdown['s3_only'] += 1

                if num_m == 1:
                    single_matches += 1
                    if len(sample_single) < 5:
                        sample_single.append((s1_id, m_list))
                else:
                    multi_matches += 1
                    if len(sample_multi) < 5:
                        sample_multi.append((s1_id, m_list))

    dt = time.time() - t0
    print(f"Ground truth parsed in {dt:.1f}s.")
    print(f"Total S1 entities in GT: {total_rows:,}")
    print(f"Zero matches: {zero_matches:,} ({zero_matches/total_rows*100:.2f}%)")
    print(f"Single matches: {single_matches:,} ({single_matches/total_rows*100:.2f}%)")
    print(f"Multi matches: {multi_matches:,} ({multi_matches/total_rows*100:.2f}%)")
    print(f"Total matched target records: {total_matched_entities:,} (unique target IDs: {len(all_matched_ids):,})")
    print(f"Max matches for a single S1: {max(match_distribution.keys())}")
    print(f"Source breakdown: {dict(source_breakdown)}")
    print(f"Match distribution top keys: {sorted(match_distribution.items())[:10]}")

    gt_summary = {
        'file_name': 'train_ground_truth.tsv',
        'file_path': gt_path,
        'size_mb': round(file_size_mb, 2),
        'total_s1_rows': total_rows,
        'duplicate_s1_rows': duplicate_s1_in_gt,
        'zero_match_count': zero_matches,
        'zero_match_pct': round(zero_matches / total_rows * 100, 2),
        'single_match_count': single_matches,
        'single_match_pct': round(single_matches / total_rows * 100, 2),
        'multi_match_count': multi_matches,
        'multi_match_pct': round(multi_matches / total_rows * 100, 2),
        'total_target_matches': total_matched_entities,
        'unique_target_matches': len(all_matched_ids),
        'max_matches_single_s1': max(match_distribution.keys()),
        'match_distribution': {int(k): int(v) for k, v in sorted(match_distribution.items())},
        'source_breakdown': dict(source_breakdown),
        'sample_zero': sample_zero,
        'sample_single': sample_single,
        'sample_multi': sample_multi,
    }
    return gt_summary


def audit_match_cross_country_and_noise():
    print("\n--- Auditing Cross-Country and Match Noise Patterns ---")
    # Load 50k S1 with ground truth matches and check whether their matched S2/S3 have same country, same name, etc.
    gt_df = pd.read_csv(os.path.join(DATA_DIR, 'train', 'train_ground_truth.tsv'), sep='\t', nrows=50_000, dtype=str, keep_default_na=False)
    # Filter only those with matches
    matched_gt = gt_df[gt_df['matched_entity_ids'].str.strip() != ''].head(5000)
    
    # Collect all needed S1 IDs and target IDs
    s1_ids = set(matched_gt['source1_entity_id'].values)
    target_ids = set()
    s1_to_targets = {}
    for _, row in matched_gt.iterrows():
        tids = [x.strip() for x in row['matched_entity_ids'].split(',') if x.strip()]
        s1_to_targets[row['source1_entity_id']] = tids
        target_ids.update(tids)

    print(f"Sampled {len(matched_gt)} S1 records with matches, needing {len(target_ids)} target records.")

    # Find S1 records
    s1_records = {}
    for chunk in pd.read_csv(os.path.join(DATA_DIR, 'train', 'train_source1.tsv'), sep='\t', chunksize=100_000, dtype=str, keep_default_na=False):
        matches = chunk[chunk['entity_id'].isin(s1_ids)]
        for _, r in matches.iterrows():
            s1_records[r['entity_id']] = r.to_dict()
        if len(s1_records) >= len(s1_ids):
            break

    # Find target records across S2 and S3
    target_records = {}
    for s_name in ['train_source2.tsv', 'train_source3.tsv']:
        p = os.path.join(DATA_DIR, 'train', s_name)
        for chunk in pd.read_csv(p, sep='\t', chunksize=100_000, dtype=str, keep_default_na=False):
            matches = chunk[chunk['entity_id'].isin(target_ids)]
            for _, r in matches.iterrows():
                target_records[r['entity_id']] = r.to_dict()
            if len(target_records) >= len(target_ids):
                break

    print(f"Loaded {len(s1_records)} S1 records and {len(target_records)} target records.")

    # Compare pairs
    cross_country_count = 0
    total_pairs_checked = 0
    exact_name_count = 0
    exact_addr_count = 0
    case_insensitive_name_match = 0

    pair_examples = []

    for s1_id, tids in s1_to_targets.items():
        if s1_id not in s1_records:
            continue
        s1 = s1_records[s1_id]
        for tid in tids:
            if tid not in target_records:
                continue
            t = target_records[tid]
            total_pairs_checked += 1

            if s1['country'] != t['country']:
                cross_country_count += 1

            s1_n = s1['business_name'].strip()
            t_n = t['business_name'].strip()
            s1_a = s1['business_address'].strip()
            t_a = t['business_address'].strip()

            if s1_n == t_n:
                exact_name_count += 1
            if s1_n.lower() == t_n.lower():
                case_insensitive_name_match += 1
            if s1_a.lower() == t_a.lower():
                exact_addr_count += 1

            pair_examples.append({
                's1_id': s1_id,
                'target_id': tid,
                's1_name': s1_n,
                'target_name': t_n,
                's1_addr': s1_a,
                'target_addr': t_a,
                's1_country': s1['country'],
                'target_country': t['country'],
            })

    cross_country_analysis = {
        'total_pairs_checked': total_pairs_checked,
        'cross_country_count': cross_country_count,
        'cross_country_pct': round(cross_country_count / max(total_pairs_checked, 1) * 100, 4),
        'exact_name_count': exact_name_count,
        'exact_name_pct': round(exact_name_count / max(total_pairs_checked, 1) * 100, 2),
        'case_insensitive_name_count': case_insensitive_name_match,
        'case_insensitive_name_pct': round(case_insensitive_name_match / max(total_pairs_checked, 1) * 100, 2),
        'exact_address_pct': round(exact_addr_count / max(total_pairs_checked, 1) * 100, 2),
        'examples': pair_examples[:25]
    }
    print(f"Checked {total_pairs_checked} true pairs:")
    print(f"  Cross-country pairs: {cross_country_count} ({cross_country_analysis['cross_country_pct']}%)")
    print(f"  Exact name matches: {exact_name_count} ({cross_country_analysis['exact_name_pct']}%)")
    print(f"  Case-insensitive name matches: {case_insensitive_name_match} ({cross_country_analysis['case_insensitive_name_pct']}%)")
    print(f"  Exact address matches: {exact_addr_count} ({cross_country_analysis['exact_address_pct']}%)")

    return cross_country_analysis


def main():
    start_all = time.time()
    profiles, country_counts, name_stats, address_stats = audit_sources()
    gt_summary = audit_ground_truth()
    cross_country_analysis = audit_match_cross_country_and_noise()

    # Save data profile csv
    df_profile = pd.DataFrame(profiles)
    df_profile.to_csv('reports/data_profile.csv', index=False)
    print("\nSaved reports/data_profile.csv")

    # Save full audit json
    audit_results = {
        'profiles': profiles,
        'country_counts': country_counts,
        'name_stats': name_stats,
        'address_stats': address_stats,
        'ground_truth': gt_summary,
        'cross_country_and_noise': cross_country_analysis,
    }
    with open('reports/audit_data.json', 'w', encoding='utf-8') as f:
        json.dump(audit_results, f, indent=2, ensure_ascii=False)
    print("Saved reports/audit_data.json")

    print(f"All auditing completed in {time.time() - start_all:.1f}s.")

if __name__ == '__main__':
    main()
