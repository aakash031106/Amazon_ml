"""
src/extract_noise_examples.py
Extract representative match pairs from train data to analyze noise patterns in detail:
- Easy matches
- Hard matches (name variations, typos, abbreviation, word order)
- Missing addresses in matched targets
- Multi-matches
- Singletons
"""
import os
import sys
import json
import pandas as pd

if sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

DATA_DIR = "data/dataset"

def get_examples():
    gt_df = pd.read_csv(os.path.join(DATA_DIR, 'train', 'train_ground_truth.tsv'), sep='\t', nrows=20_000, dtype=str, keep_default_na=False)
    
    # 1. Singletons
    singletons = gt_df[gt_df['matched_entity_ids'].str.strip() == ''].head(10)['source1_entity_id'].tolist()
    
    # 2. Multi-match
    multis = gt_df[gt_df['matched_entity_ids'].str.contains(',')].head(10)
    
    # Collect sample of matches
    matched = gt_df[gt_df['matched_entity_ids'].str.strip() != ''].head(100)
    s1_ids = set(matched['source1_entity_id'].values) | set(singletons) | set(multis['source1_entity_id'].values)
    
    target_ids = set()
    s1_to_targets = {}
    for _, r in matched.iterrows():
        tids = [x.strip() for x in r['matched_entity_ids'].split(',') if x.strip()]
        s1_to_targets[r['source1_entity_id']] = tids
        target_ids.update(tids)
    for _, r in multis.iterrows():
        tids = [x.strip() for x in r['matched_entity_ids'].split(',') if x.strip()]
        s1_to_targets[r['source1_entity_id']] = tids
        target_ids.update(tids)

    # Load S1
    s1_dict = {}
    for chunk in pd.read_csv(os.path.join(DATA_DIR, 'train', 'train_source1.tsv'), sep='\t', chunksize=100_000, dtype=str, keep_default_na=False):
        m = chunk[chunk['entity_id'].isin(s1_ids)]
        for _, r in m.iterrows():
            s1_dict[r['entity_id']] = r.to_dict()
        if len(s1_dict) >= len(s1_ids):
            break

    # Load S2 / S3
    targets_dict = {}
    for s_file in ['train_source2.tsv', 'train_source3.tsv']:
        for chunk in pd.read_csv(os.path.join(DATA_DIR, 'train', s_file), sep='\t', chunksize=100_000, dtype=str, keep_default_na=False):
            m = chunk[chunk['entity_id'].isin(target_ids)]
            for _, r in m.iterrows():
                targets_dict[r['entity_id']] = r.to_dict()
            if len(targets_dict) >= len(target_ids):
                break

    print(f"Loaded {len(s1_dict)} S1 and {len(targets_dict)} targets.")
    
    # Classify pairs
    samples = []
    for s1_id, tids in s1_to_targets.items():
        if s1_id not in s1_dict:
            continue
        s1 = s1_dict[s1_id]
        for tid in tids:
            if tid not in targets_dict:
                continue
            t = targets_dict[tid]
            
            s1_n, tn = s1['business_name'], t['business_name']
            s1_a, ta = s1['business_address'], t['business_address']
            
            # Simple category detection
            has_empty_addr = (ta.strip() == '')
            name_exact = (s1_n.strip().lower() == tn.strip().lower())
            addr_exact = (s1_a.strip().lower() == ta.strip().lower())
            
            samples.append({
                's1_id': s1_id,
                'target_id': tid,
                'country': s1['country'],
                's1_name': s1_n,
                'target_name': tn,
                's1_address': s1_a,
                'target_address': ta,
                'name_exact_case_insensitive': name_exact,
                'addr_exact_case_insensitive': addr_exact,
                'target_address_empty': has_empty_addr
            })

    # Save to json
    with open('reports/sample_pairs.json', 'w', encoding='utf-8') as f:
        json.dump({
            'singletons': [{ 'id': sid, 'details': s1_dict.get(sid, {}) } for sid in singletons[:5]],
            'pairs': samples[:50]
        }, f, indent=2, ensure_ascii=False)
    print("Saved reports/sample_pairs.json")

if __name__ == '__main__':
    get_examples()
