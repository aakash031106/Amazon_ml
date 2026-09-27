import os, sys, time
sys.path.insert(0, os.getcwd())
import polars as pl
from collections import defaultdict

def create_val_set():
    t0 = time.time()
    print("Building Gold-Standard Local Validation Split (10,000 S1 Entities + Distractors)...")
    
    # 1. Load Ground Truth for first 10,000 S1 entities
    gt_df = pl.read_csv("data/dataset/train/train_ground_truth.tsv", separator="\t", n_rows=10000)
    
    gt_map = {}
    needed_targets = set()
    for r in gt_df.iter_rows(named=True):
        sid = r["source1_entity_id"]
        m_str = r["matched_entity_ids"] or ""
        tids = [x.strip() for x in m_str.split(",") if x.strip()]
        gt_map[sid] = tids
        needed_targets.update(tids)
        
    print(f"Loaded 10,000 S1 entities with {len(needed_targets):,} true target matches.")
    
    # 2. Load matching S1 records
    s1_all = pl.read_csv("data/dataset/train/train_source1.tsv", separator="\t")
    val_s1 = s1_all.filter(pl.col("entity_id").is_in(list(gt_map.keys())))
    del s1_all
    
    # 3. Load Targets: All true targets + 100,000 distractors from each of S2 and S3
    target_dfs = []
    for s_file in ["train_source2.tsv", "train_source3.tsv"]:
        p = os.path.join("data/dataset/train", s_file)
        df = pl.read_csv(p, separator="\t")
        true_m = df.filter(pl.col("entity_id").is_in(list(needed_targets)))
        # Distractors: skip first 100k, take 100k
        distractors = df.slice(100000, 100000)
        target_dfs.append(true_m)
        target_dfs.append(distractors)
        del df
        
    val_targets = pl.concat(target_dfs).unique(subset=["entity_id"])
    print(f"Validation Target Pool: {len(val_targets):,} entities (including {len(needed_targets):,} true targets and {len(val_targets)-len(needed_targets):,} distractors).")
    
    os.makedirs("data/val_split", exist_ok=True)
    val_s1.write_csv("data/val_split/val_s1.tsv", separator="\t")
    val_targets.write_csv("data/val_split/val_targets.tsv", separator="\t")
    gt_df.write_csv("data/val_split/val_gt.tsv", separator="\t")
    print(f"Saved validation set in {time.time()-t0:.1f}s to data/val_split/")

if __name__ == "__main__":
    create_val_set()
