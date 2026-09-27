"""
src/blocking.py
Member 2: Blocking & Candidate Generation Module

Generates candidate pairs (S1 <-> S2/S3) using multi-pass union blocking
partitioned strictly by country.
"""

from collections import defaultdict
from typing import List, Dict, Set, Tuple, Any
import time


def get_blocking_keys(record: Dict[str, Any]) -> List[str]:
    """
    Generate multiple blocking keys for a single record.
    If two records share at least one blocking key, they become candidates!
    
    Keys generated:
    1. Country + First word of name (e.g., 'US:payne')
    2. Country + 4-char prefix of name (e.g., 'US:pfx:payn')
    3. Country + Second word (if name has >= 2 words) (e.g., 'US:w2:enterprises')
    4. Country + Street number / PIN code (e.g., 'US:num:3315')
    """
    country = record.get("country", "")
    norm_name = record.get("norm_name", "")
    tokens = record.get("name_tokens", [])
    numbers = record.get("address_numbers", [])

    keys = []
    
    # 1. First word (min length 3 to avoid noisy 1-2 letter noise)
    if tokens:
        w1 = tokens[0]
        if len(w1) >= 3:
            keys.append(f"{country}:w1:{w1}")
            # 3-character prefix (handles aggressive stem variations)
            keys.append(f"{country}:pfx3:{w1[:3]}")
        
        # 2. 4-character prefix of first word (catches typos)
        if len(w1) >= 4:
            keys.append(f"{country}:pfx:{w1[:4]}")

        # 3. Second word (catches flipped word orders)
        if len(tokens) >= 2:
            w2 = tokens[1]
            if len(w2) >= 3:
                keys.append(f"{country}:w2:{w2}")

    # 4. First extracted numeric token from address (e.g. house number or postal code)
    if numbers:
        keys.append(f"{country}:num:{numbers[0]}")

    # 5. First significant address token (e.g. city or street name)
    addr_tokens = record.get("address_tokens", [])
    if addr_tokens:
        for at in addr_tokens:
            if len(at) >= 4 and not at.isdigit():
                keys.append(f"{country}:addr:{at}")
                break

    return keys


def fast_jaccard_overlap(tokens1: List[str], tokens2: List[str]) -> float:
    """Fast Jaccard token similarity for candidate ranking."""
    if not tokens1 or not tokens2:
        return 0.0
    s1, s2 = set(tokens1), set(tokens2)
    inter = len(s1 & s2)
    if not inter:
        return 0.0
    return inter / float(len(s1 | s2))


class MultiPassBlocker:
    """
    In-memory multi-pass blocker designed to index target records (S2 & S3)
    and rapidly query candidates for S1 entities.
    """

    def __init__(self, max_candidates_per_s1: int = 30):
        self.max_candidates_per_s1 = max_candidates_per_s1
        # Inverted index: blocking_key -> list of target entity IDs
        self.index = defaultdict(list)
        # Fast target lookup: entity_id -> target record summary
        self.targets = {}

    def index_targets(self, target_records: List[Dict[str, Any]]):
        """Build the inverted index from target records (S2 and S3)."""
        print(f"Indexing {len(target_records):,} target records...")
        t0 = time.time()
        for rec in target_records:
            tid = rec["entity_id"]
            self.targets[tid] = {
                "norm_name": rec.get("norm_name", ""),
                "name_tokens": rec.get("name_tokens", []),
                "address_numbers": rec.get("address_numbers", []),
            }
            for key in get_blocking_keys(rec):
                self.index[key].append(tid)

        dt = time.time() - t0
        print(f"Index built in {dt:.2f}s with {len(self.index):,} unique blocking keys.")

    def find_candidates(self, s1_record: Dict[str, Any]) -> List[str]:
        """
        Find candidate target IDs for a single S1 record across all blocking keys.
        Ranks candidates by token similarity and caps at max_candidates_per_s1.
        """
        keys = get_blocking_keys(s1_record)
        candidate_ids = set()

        for key in keys:
            matched_ids = self.index.get(key, [])
            # Skip massive runaway keys (e.g. if > 2,000 entities share the same common word)
            if len(matched_ids) <= 2000:
                candidate_ids.update(matched_ids)

        if not candidate_ids:
            return []

        # If candidates are within budget, return them directly
        if len(candidate_ids) <= self.max_candidates_per_s1:
            return list(candidate_ids)

        # If more than max_candidates_per_s1, rank them by token overlap
        s1_tokens = s1_record.get("name_tokens", [])
        scored = []
        for tid in candidate_ids:
            tgt = self.targets.get(tid, {})
            score = fast_jaccard_overlap(s1_tokens, tgt.get("name_tokens", []))
            scored.append((score, tid))

        # Sort descending by score and keep top-K
        scored.sort(key=lambda x: x[0], reverse=True)
        return [tid for _, tid in scored[: self.max_candidates_per_s1]]


def evaluate_blocking_recall(
    candidate_map: Dict[str, List[str]],
    ground_truth_map: Dict[str, List[str]]
) -> Dict[str, Any]:
    """
    Evaluate candidate recall on ground truth:
    Recall = (True matches found in candidate pool) / (Total true matches)
    """
    total_true_matches = 0
    captured_true_matches = 0
    candidate_counts = []

    for s1_id, true_targets in ground_truth_map.items():
        if not true_targets:
            continue  # singletons have no targets to capture
        
        candidates = set(candidate_map.get(s1_id, []))
        candidate_counts.append(len(candidates))

        for tid in true_targets:
            total_true_matches += 1
            if tid in candidates:
                captured_true_matches += 1

    recall = (captured_true_matches / total_true_matches) if total_true_matches > 0 else 1.0
    avg_candidates = sum(candidate_counts) / len(candidate_counts) if candidate_counts else 0.0
    max_candidates = max(candidate_counts) if candidate_counts else 0

    return {
        "total_true_matches": total_true_matches,
        "captured_true_matches": captured_true_matches,
        "missed_true_matches": total_true_matches - captured_true_matches,
        "candidate_recall": round(recall, 4),
        "recall_percentage": f"{recall * 100:.2f}%",
        "avg_candidates_per_s1": round(avg_candidates, 2),
        "max_candidates": max_candidates,
    }
