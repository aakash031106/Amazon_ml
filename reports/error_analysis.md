# Amazon ML Challenge 2026: Business Entity Resolution
## Phase 6 — Error Analysis & Diagnostic Report

**Date:** September 25, 2026  
**Evaluated Set:** 300 Validation Entities (with distractors)  
**Threshold:** 0.8

---

## 1. Error Classification Breakdown

| Metric / Category | Count | Interpretation |
| :--- | :---: | :--- |
| **True Positives (TP)** | **873** | Correctly resolved matching entities. |
| **False Positives (FP)** | **2** | Non-matching entities wrongly merged (severely penalized by F0.5). |
| **False Negatives (FN)** | **134** | True matches that were missed. |
| **Correct Singletons** | **18** | Singletons with 0 matches correctly left empty (scored 1.0). |
| **Incorrect Singletons** | **0** | Singletons corrupted by a false positive (scored 0.0). |

---

## 2. Root Cause Analysis for False Negatives

False Negatives break down into two distinct failure modes:
1. **Blocking Failure (Candidate Generation):**
   - Occurs when an Indian business name is written in native script (Devanagari or Tamil) in Source 2/3 and has an empty or reordered address.
   - *Fix:* Enhanced street number / PIN extraction and multi-token address indexing.
2. **Classifier Thresholding:**
   - Occurs when a true match has a dropped address, causing address similarity to be 0.0, which pulls the total probability below 0.80.
   - *Fix:* Specific feature interaction between `is_addr_empty` and `name_jaccard` allows high-confidence names to match even with missing addresses.

---

## 3. Representative Error Cases

### [FALSE_NEGATIVE] S1-385608858 <-> S3-393804899 (India)
- **S1 Name:** `Janashakti (India) Square Private Limited`
- **Target Name:** `Mr janashaktiindiasquare.com`
- **S1 Address:** `105, Mangalmurti Sapphire, Nr Omkar School, 90 Ft Rd, Kalyan, Thane, Maharashtra`
- **Target Address:** `Mangalmurti Sapphire, Nr Omkar School, 90 Ft Rd, 105, Kalyan, MH, Thane`
- **Root Cause / Diagnosis:** Blocking Failure (not in candidate pool)

### [FALSE_NEGATIVE] S1-385608858 <-> S2-635808285 (India)
- **S1 Name:** `Janashakti (India) Square Private Limited`
- **Target Name:** `Shri Janashakti Square Private Limited Services #68886`
- **S1 Address:** `105, Mangalmurti Sapphire, Nr Omkar School, 90 Ft Rd, Kalyan, Thane, Maharashtra`
- **Target Address:** `105-, MANGALMURTI SAPPHIRE, NR OMKAR SCHOOL, 90 FT RD, KALYAN, महाराष्ट्र`
- **Root Cause / Diagnosis:** Blocking Failure (not in candidate pool)

### [FALSE_NEGATIVE] S1-231519650 <-> S2-504051668 (India)
- **S1 Name:** `Sai Care Private Limited`
- **Target Name:** `साई केयर प्राइवेट लिमिटेड`
- **S1 Address:** `Flat Number 2B, 1St Floor, Shrinivas Apartment, Survey Number 1202/29, Plot Number 567/2B/3, Apte Road, Near Santosh Bakery, Haveli, Pune, Maharashtra`
- **Target Address:** `FLAT NUMBER B3/2B, 1ST FLOOR, SHRINIVAS APARTMENT, SURVEY NUMBER 1202/29, PLOT NUMBER 567/2B/3, APTE ROAD, NEAR SANTOSH BAKERY, PUNE REGION, HAVELI, Maharashtra`
- **Root Cause / Diagnosis:** Blocking Failure (not in candidate pool)

### [FALSE_NEGATIVE] S1-231519650 <-> S3-286238001 (India)
- **S1 Name:** `Sai Care Private Limited`
- **Target Name:** `Sai`
- **S1 Address:** `Flat Number 2B, 1St Floor, Shrinivas Apartment, Survey Number 1202/29, Plot Number 567/2B/3, Apte Road, Near Santosh Bakery, Haveli, Pune, Maharashtra`
- **Target Address:** `Flat Number 2B, 1St Floor, Shrinivas Apartment, Survey Number 1202/29, Plot Number 567/2B/3, Apte Road, Near Santosh Bakery, Haveli, Pune, Maharashtra`
- **Root Cause / Diagnosis:** Blocking Failure (not in candidate pool)

### [FALSE_NEGATIVE] S1-686913759 <-> S2-584238851 (US)
- **S1 Name:** `Behavioral Health Clinic`
- **Target Name:** `BEHAVIORALHEALTHCLINIC.COM`
- **S1 Address:** `2493 Pennwood Court, Round Lake Beach, IL`
- **Target Address:** `2493 PENNWOOD CT, ROUND LAKE BEACH, IL`
- **Root Cause / Diagnosis:** Blocking Failure (not in candidate pool)

### [FALSE_NEGATIVE] S1-849274176 <-> S2-34421487 (India)
- **S1 Name:** `Aarvita Biosciences Private Limited`
- **Target Name:** `Jaxrizagild`
- **S1 Address:** `No.41, Ponnambalam Salai, K.K.Nagar, Chennai, Tamil Nadu`
- **Target Address:** `41, CHENNAI CITY REGION, தமிழ்நாடு`
- **Root Cause / Diagnosis:** Classifier Failure (probability < 0.80)

### [FALSE_NEGATIVE] S1-986414977 <-> S2-510349301 (India)
- **S1 Name:** `Jai Finance Private Limited`
- **Target Name:** `जय फाइनेंस प्राइवेट लिमिटेड`
- **S1 Address:** `Prop No. 169 & 170, Upper Ground Floor Pocket-20, Sec-24, Rohini, Delhi, North West, Delhi`
- **Target Address:** `DELHI, UPPER GROUND FLOOR POCKET-20, SEC-24, ROHINI, <NULL>, PROP NO. 169 & 170, Delhi`
- **Root Cause / Diagnosis:** Blocking Failure (not in candidate pool)

### [FALSE_NEGATIVE] S1-520706871 <-> S2-413724070 (India)
- **S1 Name:** `Shyam Agro Pvt Ltd`
- **Target Name:** `श्याम एग्रो प्रा. लि.`
- **S1 Address:** `Sco No-19 Sec-56, Gurugram, Gurgaon, Haryana`
- **Target Address:** `PLOT 555 SCO NO-19 SEC-56, GURGAON, Haryana`
- **Root Cause / Diagnosis:** Blocking Failure (not in candidate pool)

### [FALSE_NEGATIVE] S1-520706871 <-> S2-905169888 (India)
- **S1 Name:** `Shyam Agro Pvt Ltd`
- **Target Name:** `श्याम एग्रो प्रा. लि.`
- **S1 Address:** `Sco No-19 Sec-56, Gurugram, Gurgaon, Haryana`
- **Target Address:** `GURGAON, Haryana, PLOT 555 SCO NO-19 SEC-56, GURUGRAM, GURGAON`
- **Root Cause / Diagnosis:** Blocking Failure (not in candidate pool)

### [FALSE_NEGATIVE] S1-327804795 <-> S3-836465899 (India)
- **S1 Name:** `YML Five Pvt Ltd`
- **Target Name:** `ymlfive.com`
- **S1 Address:** `D.No. 23/1490, Flat No.8, Deekshita Towers, Behind Pavani Prestige Apartment, Magunt, A Layout, Nellore, Andhra Pradesh`
- **Target Address:** `G-23/1490, Flat No.8, Deekshita Towers, Behind Pavani Prestige Apartment, Magunt, A Layout, Nellore, ఆంధ్రప్రదేశ్`
- **Root Cause / Diagnosis:** Blocking Failure (not in candidate pool)

