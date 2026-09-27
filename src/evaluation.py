"""
src/evaluation.py
Leader / Member 4: Evaluation Module

Computes official competition metrics: Macro F0.5, Precision, Recall,
and broken down performance by match cardinality (singletons, single, multi).
"""

from typing import Dict, List, Set, Any
import numpy as np


def compute_entity_f_beta(
    predicted: Set[str],
    ground_truth: Set[str],
    beta: float = 0.5
) -> Dict[str, float]:
    """
    Compute Precision, Recall, and F_beta for a SINGLE Source 1 entity.
    
    Singletons (ground_truth is empty):
      - If predicted is empty: score is 1.0 (perfect singleton prediction)
      - If predicted is NOT empty: score is 0.0 (false merge)
    """
    beta_sq = beta ** 2
    weight = 1.0 + beta_sq  # 1.25 for beta=0.5

    # True singleton case
    if len(ground_truth) == 0:
        if len(predicted) == 0:
            return {"precision": 1.0, "recall": 1.0, "f_score": 1.0}
        else:
            return {"precision": 0.0, "recall": 0.0, "f_score": 0.0}

    # True matches exist, but model predicted nothing
    if len(predicted) == 0:
        return {"precision": 0.0, "recall": 0.0, "f_score": 0.0}

    tp = len(predicted & ground_truth)
    precision = tp / float(len(predicted))
    recall = tp / float(len(ground_truth))

    denom = (beta_sq * precision) + recall
    if denom == 0.0 or (precision + recall == 0.0):
        f_score = 0.0
    else:
        f_score = (weight * precision * recall) / denom

    return {"precision": precision, "recall": recall, "f_score": f_score}


def evaluate_predictions(
    predictions_map: Dict[str, List[str]],
    ground_truth_map: Dict[str, List[str]],
    beta: float = 0.5
) -> Dict[str, Any]:
    """
    Compute official Macro-Averaged F_beta across ALL Source 1 entities.
    
    Also provides granular breakdown:
      - Singleton accuracy (T = 0)
      - Single-match F_score (|T| = 1)
      - Multi-match F_score (|T| >= 2)
    """
    all_f_scores = []
    all_precisions = []
    all_recalls = []

    singleton_scores = []
    single_match_scores = []
    multi_match_scores = []

    for s1_id, gt_list in ground_truth_map.items():
        gt_set = set(gt_list)
        pred_set = set(predictions_map.get(s1_id, []))

        scores = compute_entity_f_beta(pred_set, gt_set, beta=beta)
        f = scores["f_score"]
        p = scores["precision"]
        r = scores["recall"]

        all_f_scores.append(f)
        all_precisions.append(p)
        all_recalls.append(r)

        # Categorize
        if len(gt_set) == 0:
            singleton_scores.append(f)
        elif len(gt_set) == 1:
            single_match_scores.append(f)
        else:
            multi_match_scores.append(f)

    macro_f = float(np.mean(all_f_scores)) if all_f_scores else 0.0
    macro_p = float(np.mean(all_precisions)) if all_precisions else 0.0
    macro_r = float(np.mean(all_recalls)) if all_recalls else 0.0

    return {
        "macro_f0_5": round(macro_f, 4),
        "macro_precision": round(macro_p, 4),
        "macro_recall": round(macro_r, 4),
        "total_evaluated_entities": len(ground_truth_map),
        "singleton_count": len(singleton_scores),
        "singleton_accuracy": round(float(np.mean(singleton_scores)), 4) if singleton_scores else 0.0,
        "single_match_count": len(single_match_scores),
        "single_match_macro_f0_5": round(float(np.mean(single_match_scores)), 4) if single_match_scores else 0.0,
        "multi_match_count": len(multi_match_scores),
        "multi_match_macro_f0_5": round(float(np.mean(multi_match_scores)), 4) if multi_match_scores else 0.0,
    }
