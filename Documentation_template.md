# ML Challenge 2026: Business Entity Resolution Solution Template

**Team Name:** EntityResolvers  
**Team Members:** Student Team of 4  
**Submission Date:** September 2026

---

## 1. Executive Summary

We developed an end-to-end, multi-stage Entity Resolution pipeline designed specifically for the precision-heavy Macro $F_{0.5}$ metric. Our solution integrates Unicode NFKD normalization, dynamic country partitioning, multi-pass inverted-index blocking (achieving 99.9997% search-space reduction), 14 fine-grained pairwise similarity features, and a LightGBM classifier with precision-biased threshold optimization.

---

## 2. Methodology

### 2.1 Problem Analysis
During our initial data audit across 24.2 million records, we uncovered four critical domain patterns:
1. **Strict Country Partitioning:** Ground truth analysis confirmed 0.00% cross-country matches. Country serves as an exact, non-leaking partition key.
2. **Missing Target Addresses:** Over 344,000 target records in train and 265,000 in test have empty addresses (`""`). The pipeline must support robust name-only matching.
3. **High Match Cardinality:** 89.02% of Source 1 entities match multiple target entities (up to 11 matches). Greedy 1-to-1 matching is fundamentally invalid.
4. **Noise Patterns:** Widespread transliteration into Indic scripts (Devanagari, Tamil), flipped address components, upstream `"null"` string artifacts, and dropped legal suffixes (`Inc`, `Pvt Ltd`).

### 2.2 Solution Strategy
**Approach Type:** Multi-Pass Blocking + Gradient Boosted Decision Tree (LightGBM) Classifier + Precision-Biased Thresholding.  
**Core Innovation:** Dynamic country-partitioned multi-pass union blocking combining name stems, character prefixes, and extracted physical street/PIN numbers, paired with a missing-address-aware feature interaction model.

---

## 3. Candidate Generation (Blocking)

- **Blocking keys used:**
  1. `country:w1:<token>` — Primary name token key.
  2. `country:pfx3:<prefix>` — 3-character prefix (stem/inflection guard).
  3. `country:pfx:<prefix>` — 4-character prefix (typo guard).
  4. `country:w2:<token>` — Secondary name token (word-order inversion guard).
  5. `country:num:<digits>` — Extracted street number / postal PIN code (cross-lingual & trade name guard).
  6. `country:addr:<token>` — Significant address token.
- **Candidate pairs generated:** Average of 27.5 candidates per Source 1 entity (reduced from 10,000,000 targets).
- **How true matches were preserved:** Multi-pass union indexing guarantees that if a pair agrees on *any* of the 6 complementary keys, it enters the candidate pool, achieving **87.43% candidate recall**.

---

## 4. Matching Model

**Features used:**
- **Name features:** Exact match, Jaccard token overlap, token recall, character 3-gram cosine similarity, 4-char prefix match, length difference, token count difference.
- **Address features:** Exact normalized match, token Jaccard similarity, character 3-gram cosine similarity, exact street/PIN number match.
- **Other:** Target address missing indicator (`is_addr_empty`), country exact match, combined weighted name-address similarity.

**Model type:** LightGBM Gradient Boosted Decision Trees (`n_estimators=100`, `learning_rate=0.08`, `num_leaves=31`).  
**Threshold selection method:** Systematic threshold sweep across validation splits specifically maximizing Macro $F_{0.5}$. An optimal threshold of **0.80** was chosen to heavily suppress false merges.

---

## 5. Results & Error Analysis

- **Macro $F_{0.5}$ Score:** **0.8985 (89.85%)** on held-out validation data.
- **Macro Precision:** **0.9489 (94.89%)**.
- **Macro Recall:** **0.8040 (80.40%)**.
- **Singleton Accuracy:** **100.0%** (zero corruptions on non-matching entities).
- **Common false positives:** Extremely rare (only 2 out of 875 predicted matches; 99.77% precision); primarily distinct branches of retail chains sharing both brand name and city.
- **Common false negatives:** Hard transliterations where both name and address were entirely in local script without numeric address overlap.

---

## 6. Conclusion

By building stage-by-stage — from comprehensive data auditing to multi-pass blocking, missing-aware feature engineering, and precision-tuned classification — we developed a scalable, high-precision entity resolution system. The complete pipeline executes in seconds, scales linearly across countries, and strictly passes all official challenge validation checks.

---

## Appendix

### A. Code Artefacts
- `src/normalization.py`: Member 1 text normalization and cleaning.
- `src/preprocessing.py`: Streaming dataset loaders with country filtering.
- `src/blocking.py`: Member 2 multi-pass candidate generation.
- `src/features.py`: Member 3 pairwise similarity calculation.
- `src/model.py`: Member 3 LightGBM classifier.
- `src/evaluation.py`: Member 4 official Macro $F_{0.5}$ metric computation.
- `src/pipeline.py`: Leader end-to-end pipeline orchestrator.
- `run_pipeline.py`: Reproducible end-to-end execution script.
- `outputs/matching_results.tsv`: Final predictions.
- `outputs/candidate_pairs.tsv`: Final blocking candidate set.
