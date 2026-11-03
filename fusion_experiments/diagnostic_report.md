# Forensic Procurement Cartel Detection: Diagnostic and Experimental Evaluation Report

**Date:** October 2026  
**Scope:** Chronological Evaluation (Train: 2004–2015, Test: 2016–2022)  
**Baseline Evaluated:** Rank Reliability-Aware Evidence Fusion  
**Workspace:** `Model Training/fusion_experiments/`

---

## Executive Summary

This study conducted a rigorous forensic audit and empirical investigation into the low precision and recall observed in the **Rank Reliability-Aware Fusion** model.

### Verified Findings:
1. **Reconciliation of Discrepant Test Counts:**
   - The test parquet (`labelled_fusion_scores.parquet`) contains **6,208 records** (226 cartel-labelled, 5,982 non-cartel-labelled).
   - In the baseline evaluation, exactly **5 records** (1 cartel, 4 non-cartel) are excluded from scoring because they have **NaN across all four evidence modules ($M_1, M_2, M_3, M_4$)**.
   - This leaves **6,203 scored records** (225 cartels, 5,978 non-cartels). The 225 vs 226 and 6,203 vs 6,208 counts are completely resolved: the difference is 100% accounted for by tenders with zero observed module evidence.

2. **Root Cause of Low Precision and Screening Recall:**
   - **Prevalence Collapse:** Test positive prevalence dropped from **62.06%** in training (2004–2015) to **3.63%** in testing (2016–2022). Furthermore, ground-truth cartel convictions end in 2017: years 2018–2022 contain 2,714 records with zero positive labels, generating structural false positives.
   - **Severe Module Dilution:** The baseline weighted mean formula $\frac{\sum \text{rank}_i \cdot \text{rel}_i}{\sum \text{rel}_i}$ severely penalizes cartels that exhibit evidence in only one or two modules. When $M_2$ rank $\ge 0.85$, **69.5%** of tenders are dragged below 0.70 by neutral/absent modules.
   - **Module Imbalance ($M_4$ Dominance):** $M_4$ is available in 99.9% of test tenders and accounts for **46.53% of all fusion weight**, while $M_1$ has only 2.64% test coverage (accounting for 0.30% of total weight). Standalone $M_4$ actually achieved AP of 0.1558; linearly fusing $M_1, M_2, M_3$ with $M_4$ diluted AP down to 0.0937.

3. **Key Experimental Breakthrough:**
   - **Soft-OR Blend Fusion ($\alpha = 0.5$):** Tuned strictly within the training period (2004–2012 train, 2013–2015 val) without touching test data, this method blends the maximum single-module reliability-weighted signal with the weighted mean.
   - On the identical 6,203 test records:
     - **Average Precision more than doubled** from **0.0937** to **0.2062**.
     - **Top 1% Precision increased from 3.17% to 47.62%** (catching 30 cartels vs 2 in baseline, lift jumping from 0.88× to 13.13×).
     - **Top 5% Recall increased from 18.67% to 33.33%** (catching 75 cartels vs 42 in baseline, precision 24.12% vs 13.50%).

---

## 1. Audit of Current Evaluation & Discrepancy Resolution

### 1.1 Dataset Accounting
| Dataset Level | Total Records | Cartel (1) | Non-Cartel (0) | Cartel Prevalence | Exclusion Reason |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **All Test Records (>= 2016)** | 6,208 | 226 | 5,982 | 3.6405% | Full raw test split |
| **Zero-Evidence Test Records** | 5 | 1 | 4 | 20.0000% | All 4 modules NaN (`row_ids`: 466809, 3090101, 3005451, 3153078, 3153079) |
| **Scored Test Records (Baseline)** | **6,203** | **225** | **5,978** | **3.6273%** | Evaluated in `final_ranking_metrics.csv` & `final_top_k_results.csv` |

### 1.2 Module Coverage Denominators
Module coverage figures must be stated relative to the correct denominator:
- **$M_1$ (Price Variance / Clustering):** 164 non-null evidence rows = **2.64% of all test rows** (2.64% of scored rows). Mean test reliability: 0.1420.
- **$M_2$ (Single Bidding / Low Competition):** 4,920 non-null evidence rows = **79.25% of all test rows** (79.32% of scored rows). Mean test reliability: 0.5034.
- **$M_3$ (Bidder Rotation / Market Sharing):** 3,776 non-null evidence rows = **60.82% of all test rows** (60.87% of scored rows). Mean test reliability: 0.4232.
- **$M_4$ (Price / Estimate Discrepancy):** 6,202 non-null evidence rows = **99.90% of all test rows** (99.98% of scored rows). Mean test reliability: 0.5751.

### 1.3 Fusion Scoring Mechanics
For each row, score $F$ is calculated as:
$$F = \frac{\sum_{i \in \{1,2,3,4\}} \text{rank}_i \cdot \text{reliability}_i \cdot \mathbb{I}(\text{rank}_i \text{ valid} \land \text{reliability}_i > 0)}{\sum_{i \in \{1,2,3,4\}} \text{reliability}_i \cdot \mathbb{I}(\text{rank}_i \text{ valid} \land \text{reliability}_i > 0)}$$
- Missing evidence scores remain NaN and do not contribute to numerator or denominator.
- If denominator is 0, score is NaN.
- Reliability acts as a linear weighting factor across active modules.

---

## 2. Diagnostic Investigation of the 5 Hypotheses

### Hypothesis 1: Temporal Prevalence Shift vs. Concept Drift
- **Verified:** Training period cartel prevalence is **62.06%** (7,819 cartels / 12,599 records). Test period cartel prevalence is **3.63%** (225 / 6,203).
- **Mechanism of Threshold Breakdown:** In training, predicting positive on 99.8% of records gave $F_1 = 0.765$ at $\text{threshold} = 0.01$. Applying this threshold to test produced **5,971 false positives** (99.9% false positive rate), collapsing test precision to **3.63%**.
- **Label Truncation Effect:** Test set cartels appear only in 2016 ($N=184$, prevalence 12.04%) and 2017 ($N=42$, prevalence 2.14%). From 2018 to 2022 ($N=2,714$), positive labels are completely absent.

### Hypothesis 2: Module Dilution
- **Verified & Quantified:** When a tender triggers a high module rank ($\ge 0.85$):
  - For $M_2$: 69.5% (301 of 433) are dragged below 0.70.
  - For $M_3$: 82.2% (438 of 533) are dragged below 0.70.
  - For $M_4$: 43.5% (215 of 494) are dragged below 0.70.
- When only 1 module is available ($N=789$, 24 cartels), ranking performance is essentially random ($\text{AUC} = 0.4524$, $\text{AP} = 0.0283$).
- When 3 modules are available ($N=3,321$, 111 cartels), multi-signal corroboration yields $\text{AUC} = 0.8868$ and $\text{AP} = 0.2901$.

### Hypothesis 3: Module Dominance and Reliability
- **Verified:** $M_4$ contributes **46.53%** of all denominator weight in the test set. $M_2$ contributes **32.31%**, $M_3$ contributes **20.85%**, and $M_1$ contributes only **0.30%**.
- $M_4$'s dominance stems from both ubiquitous availability (99.9%) and relatively high average reliability (0.575).
- Standalone $M_4$ achieves $\text{AP} = 0.1558$, while the baseline fused model achieves only $\text{AP} = 0.0937$. Fusing uninformative low-reliability signals from other modules dragged down $M_4$'s precision.

### Hypothesis 4: Ranking Quality & Subgroup False Positives
- At Top 5% screening (score $\ge 0.7936$, $N=311$ selected):
  - 40 of 42 True Positives are in 2016, 2 in 2017, and 0 in 2018–2022.
  - False Positives are heavily concentrated in Spain (87 FPs), Portugal (71 FPs), Latvia (47 FPs), and France (41 FPs). Latvia and France had 0 true cartels in the top 5% selected tenders.
  - CPV market group `market_23` accounts for both the highest TPs (29) and highest FPs (58).

### Hypothesis 5: Rank Normalization & Leakage Audit
- **Verified Clean:** The empirical percentile reference distributions for $M_1, M_2, M_3, M_4$ were fit strictly on training records ($N=511, 7552, 6915, 12544$ respectively).
- Test records were transformed using `np.searchsorted` against these frozen training references.
- No test labels or test distributions were leaked into reference construction.

---

## 3. Controlled Experimental Results

All experiments were executed using identical chronological train (2004–2015) and test (2016–2022) splits, with hyperparameters tuned exclusively on an internal chronological validation split (train: 2004–2012, val: 2013–2015).

### 3.1 Primary Comparison: Available Scored Records
| Model / Experiment | Scored Records | Cartel Cases | ROC-AUC | Average Precision | Top 1% Precision | Top 1% Recall | Top 5% Precision | Top 5% Recall | Top 5% Lift |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **A: Baseline Rank Reliability** | 6,203 | 225 | 0.7559 | 0.0937 | 3.17% | 0.89% | 13.50% | 18.67% | 3.72× |
| **A: Baseline Rank Equal-Weight** | 6,203 | 225 | 0.7388 | 0.0824 | 3.17% | 0.89% | 11.90% | 16.44% | 3.28× |
| **B: Supervised Nonlinear Meta-Model** | 6,203 | 225 | 0.5565 | 0.1062 | 26.98% | 7.56% | 11.25% | 15.56% | 3.10× |
| **C: Soft-OR Blend Fusion ($\alpha=0.5$)** | **6,203** | **225** | **0.7580** | **0.2062** | **47.62%** | **13.33%** | **24.12%** | **33.33%** | **6.65×** |
| **D: Reliability Gating ($\tau=0.2$)** | 6,190 | 225 | 0.7558 | 0.0963 | 4.84% | 1.33% | 14.52% | 20.00% | 3.99× |
| **D: Reliability Gating ($\tau=0.4$)** | 5,780 | 211 | 0.7753 | 0.1309 | 27.59% | 7.58% | 16.26% | 22.27% | 4.45× |
| **E: Non-Cartel Reference Fusion** | 6,203 | 225 | 0.7449 | 0.0853 | 3.17% | 0.89% | 12.22% | 16.89% | 3.37× |

### 3.2 Common Set Comparison (Identical 5,780 Records Scored Across All Models)
*Restricted to records meeting the strict reliability threshold $\tau=0.4$ (Total: 5,780 records, 211 cartels, baseline prevalence = 3.65%):*

| Model / Experiment | Scored Records | Cartels | ROC-AUC | Average Precision | Top 5% Precision | Top 5% Recall | Top 5% Lift |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **A: Baseline Rank Reliability** | 5,780 | 211 | 0.8207 | 0.1343 | 15.92% | 21.80% | 4.36× |
| **B: Supervised Nonlinear Meta-Model** | 5,780 | 211 | 0.5694 | 0.1115 | 11.42% | 15.64% | 3.13× |
| **C: Soft-OR Blend Fusion ($\alpha=0.5$)** | **5,780** | **211** | **0.8211** | **0.2216** | **24.57%** | **33.65%** | **6.73×** |
| **D: Reliability Gating ($\tau=0.4$)** | 5,780 | 211 | 0.7753 | 0.1309 | 16.26% | 22.27% | 4.45× |
| **E: Non-Cartel Reference Fusion** | 5,780 | 211 | 0.7937 | 0.1202 | 14.19% | 19.43% | 3.89× |

---

## 4. Key Takeaways & Recommendations

1. **Defensible Improvement: Soft-OR Blend Fusion ($\alpha=0.5$)**
   - Combining the strongest single-module evidence signal with the overall reliability-weighted mean resolves the dilution problem without discarding unobserved modules.
   - At an audit budget of Top 1% (63 tenders), precision increases from **3.17% to 47.62%** (catching 30 cartels instead of 2).
   - At Top 5% (311 tenders), precision increases from **13.50% to 24.12%** and recall jumps from **18.67% to 33.33%** (75 cartels detected vs 42).

2. **Investigation Capacity Framing:**
   - In anti-cartel enforcement, fixed classification thresholds (like 0.5 or training-tuned 0.01) are misleading due to prior probability collapse.
   - Screening should strictly be operated as a **Top-$K$ ranked investigation queue** based on auditing capacity.

3. **Status of Existing Baseline:**
   - As requested, no baseline scripts, dashboards, or presentations were modified.
   - All experimental data, validation logs, predictions, and comparative outputs are preserved in `fusion_experiments/`.
