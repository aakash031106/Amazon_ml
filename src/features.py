"""
src/features.py
Member 3: Pairwise Feature Engineering Module

Computes numerical similarity features for any candidate pair (S1 <-> S2/S3).
"""

from typing import Dict, Any, List
import math
from collections import Counter


def get_char_ngrams(text: str, n: int = 3) -> Counter:
    """Generate character n-grams and return their frequency counts."""
    if not text:
        return Counter()
    cleaned = f" {text} "
    return Counter(cleaned[i : i + n] for i in range(len(cleaned) - n + 1))


def cosine_sim_ngrams(c1: Counter, c2: Counter) -> float:
    """Compute cosine similarity between two character n-gram frequency counters."""
    if not c1 or not c2:
        return 0.0
    common = set(c1.keys()) & set(c2.keys())
    if not common:
        return 0.0
    dot = sum(c1[k] * c2[k] for k in common)
    norm1 = math.sqrt(sum(v * v for v in c1.values()))
    norm2 = math.sqrt(sum(v * v for v in c2.values()))
    if norm1 == 0.0 or norm2 == 0.0:
        return 0.0
    return dot / (norm1 * norm2)


def compute_pairwise_features(s1: Dict[str, Any], target: Dict[str, Any]) -> Dict[str, float]:
    """
    Compute a dictionary of similarity features between an S1 entity and a candidate S2/S3 entity.
    All features are numerical floats.
    """
    s1_name = s1.get("norm_name", "")
    t_name = target.get("norm_name", "")
    s1_tokens = s1.get("name_tokens", [])
    t_tokens = target.get("name_tokens", [])

    s1_addr = s1.get("norm_address", "")
    t_addr = target.get("norm_address", "")
    s1_addr_tokens = s1.get("address_tokens", [])
    t_addr_tokens = target.get("address_tokens", [])

    # 1. NAME FEATURES
    # Exact Match
    name_exact = 1.0 if s1_name and (s1_name == t_name) else 0.0

    # Token Overlap & Jaccard
    s1_set = set(s1_tokens)
    t_set = set(t_tokens)
    name_inter = len(s1_set & t_set)
    name_union = len(s1_set | t_set)
    name_jaccard = (name_inter / float(name_union)) if name_union > 0 else 0.0
    name_recall = (name_inter / float(len(s1_set))) if s1_set else 0.0

    # Character 3-gram Cosine Similarity
    c1_name = get_char_ngrams(s1_name, n=3)
    c2_name = get_char_ngrams(t_name, n=3)
    name_char_cosine = cosine_sim_ngrams(c1_name, c2_name)

    # Prefix match
    name_pfx_match = 1.0 if (s1_name[:4] == t_name[:4] and len(s1_name) >= 4) else 0.0

    # Length Differences
    name_len_diff = abs(len(s1_name) - len(t_name))
    name_token_diff = abs(len(s1_tokens) - len(t_tokens))

    # 2. ADDRESS FEATURES
    t_addr_empty = 1.0 if (not t_addr or target.get("is_address_empty", False)) else 0.0
    addr_exact = 1.0 if (not t_addr_empty and s1_addr == t_addr) else 0.0

    a1_set = set(s1_addr_tokens)
    a2_set = set(t_addr_tokens)
    addr_inter = len(a1_set & a2_set)
    addr_union = len(a1_set | a2_set)
    addr_jaccard = (addr_inter / float(addr_union)) if (addr_union > 0 and not t_addr_empty) else 0.0

    c1_addr = get_char_ngrams(s1_addr, n=3)
    c2_addr = get_char_ngrams(t_addr, n=3)
    addr_char_cosine = cosine_sim_ngrams(c1_addr, c2_addr) if not t_addr_empty else 0.0

    # Number / PIN match
    s1_nums = set(s1.get("address_numbers", []))
    t_nums = set(target.get("address_numbers", []))
    num_match = 1.0 if (s1_nums and t_nums and bool(s1_nums & t_nums)) else 0.0

    # 3. METADATA & COMPOSITE FEATURES
    country_match = 1.0 if s1.get("country") == target.get("country") else 0.0
    combined_jaccard = (0.6 * name_jaccard) + (0.4 * (addr_jaccard if not t_addr_empty else name_jaccard))

    return {
        "name_exact": name_exact,
        "name_jaccard": name_jaccard,
        "name_recall": name_recall,
        "name_char_cosine": name_char_cosine,
        "name_pfx_match": name_pfx_match,
        "name_len_diff": float(name_len_diff),
        "name_token_diff": float(name_token_diff),
        "addr_exact": addr_exact,
        "addr_jaccard": addr_jaccard,
        "addr_char_cosine": addr_char_cosine,
        "addr_num_match": num_match,
        "is_addr_empty": t_addr_empty,
        "country_match": country_match,
        "combined_jaccard": combined_jaccard,
    }
