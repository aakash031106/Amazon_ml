# Amazon ML Challenge 2026: Business Entity Resolution
## Phase 3 — Blocking & Candidate Generation Report

**Date:** September 25, 2026  
**Author:** Member 2 & Engineering Team  
**Scope:** Multi-Pass Union Blocking Strategy, Candidate Recall, Efficiency, and Scalability

---

## 1. Executive Summary & Objective

In Entity Resolution, comparing all $1.73 \times 10^6$ Source 1 entities against all $9.97 \times 10^6$ Source 2 & Source 3 target entities requires $\approx 17.27 \text{ Trillion}$ pairwise comparisons.

**Member 2's Goal:**
Design a high-recall, low-latency **Multi-Pass Union Blocker** that reduces the search space from $10,000,000$ records to a narrow pool of $\le 30$ candidates per entity, while preserving over $87\%$ of all true matches.

---

## 2. Multi-Pass Union Blocking Strategy

All blocking is strictly partitioned within country (`US`, `India`, `France`). Within each country partition, we generate 5 complementary blocking keys:

1. **Pass 1 — First Word Key (`country:w1:<word>`):**
   - Captures records sharing the primary entity name token (e.g. `US:w1:payne`).
2. **Pass 2 — 3-Character Prefix (`country:pfx3:<prefix>`):**
   - Catches inflectional variants and aggressive typos (e.g. `US:pfx3:pay`).
3. **Pass 3 — 4-Character Prefix (`country:pfx:<prefix>`):**
   - Catches common spelling variations and suffix truncations (e.g. `US:pfx:payn`).
4. **Pass 4 — Second Word Key (`country:w2:<word>`):**
   - Guards against word-order inversions (e.g., `International Automation` vs `Automation International`).
5. **Pass 5 — Address Street Number / PIN Code (`country:num:<digits>`):**
   - Catches entities whose names are transliterated into Indian scripts or written under trade names (DBAs), but share the same physical address number.
6. **Pass 6 — Primary Address Token (`country:addr:<token>`):**
   - Catches matching locations and municipalities.

---

## 3. Experimental Validation on Real Ground Truth

Tested on real training data with 716 true ground truth matches:

| Metric | Measured Value | Target Goal | Status |
| :--- | :---: | :---: | :---: |
| **Total True Matches Evaluated** | 716 | - | - |
| **Captured True Matches** | **626** | - | - |
| **Candidate Recall** | **87.43%** | $> 85.0\%$ | **EXCEEDED** |
| **Average Candidates per $S_1$** | **27.50** | $\le 30.0$ | **OPTIMAL** |
| **Maximum Candidates per $S_1$** | **30** | $\le 50.0$ | **BOUNDED** |
| **Indexing Time (20,716 records)**| **0.06 seconds** | $< 2.0\text{ s}$ | **BLAZING FAST** |
| **Reduction Ratio** | **$99.9997\%$** | $> 99.9\%$ | **OPTIMAL** |

---

## 4. Key Takeaways & Hand-off to Member 3

1. **Massive Search Space Reduction:** The comparison space is reduced by a factor of over **300,000×**!
2. **High Recall Ceiling:** With **87.43%** candidate recall, Member 3's ML model has a strong candidate pool from which to achieve a competitive $F_{0.5}$ score.
3. **Candidate Pairs Output:** The output candidate set adheres strictly to the required TSV schema:
   `source1_entity_id \t candidate_entity_ids` (comma-separated).
