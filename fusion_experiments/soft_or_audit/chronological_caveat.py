# fusion_experiments/soft_or_audit/chronological_caveat.py
"""
Audit Step 6: Chronological Evaluation Caveat (2016-2017 vs 2018-2022)
Evaluates:
1. Performance on 2016-2017 (period with active cartel ground truth).
2. Analysis of 2018-2022 (period with zero labelled positives).
3. Quantifies how 2018-2022 introduces structural false positives due to label truncation.
"""

from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score

BASE_DIR = Path(__file__).resolve().parent.parent.parent
OUTPUT_DIR = BASE_DIR / "fusion_experiments" / "soft_or_audit"
AUDIT_PARQUET = OUTPUT_DIR / "audit_scored_test_records.parquet"

def evaluate_subperiod_top_k(df, score_col, model_name, k):
    work = df[["row_id", "is_cartel", score_col]].dropna(subset=["is_cartel", score_col]).copy()
    work = work.sort_values(by=score_col, ascending=False, kind="mergesort").reset_index(drop=True)
    n_select = max(1, int(np.ceil(len(work) * k / 100.0)))
    selected = work.iloc[:n_select]
    tp = int(selected["is_cartel"].sum())
    total_c = int(work["is_cartel"].sum())
    p = tp / len(selected) if len(selected) > 0 else 0.0
    r = tp / total_c if total_c > 0 else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    prev = total_c / len(work) if len(work) > 0 else 0.0
    lift = p / prev if prev > 0 else np.nan
    return {
        "model": model_name,
        "top_k_percent": k,
        "total_available_rows": len(work),
        "rows_selected": len(selected),
        "total_cartel": total_c,
        "cartel_selected": tp,
        "precision": p,
        "recall": r,
        "f1": f1,
        "lift": lift,
    }

def main():
    df = pd.read_parquet(AUDIT_PARQUET)
    
    p1 = df[df["tender_year"].isin([2016, 2017])].copy()
    p2 = df[df["tender_year"] >= 2018].copy()

    print("=" * 70)
    print("CHRONOLOGICAL SPLIT DIAGNOSTICS: 2016-2017 vs 2018-2022")
    print("=" * 70)
    print(f"Period 1 (2016-2017): Total={len(p1):,}, Cartels={p1['is_cartel'].sum()} ({p1['is_cartel'].mean():.4%})")
    print(f"Period 2 (2018-2022): Total={len(p2):,}, Cartels={p2['is_cartel'].sum()} ({p2['is_cartel'].mean():.4%})")

    # Metrics on Period 1 (2016-2017)
    b_auc_p1 = roc_auc_score(p1["is_cartel"], p1["baseline_score"])
    b_ap_p1 = average_precision_score(p1["is_cartel"], p1["baseline_score"])
    s_auc_p1 = roc_auc_score(p1["is_cartel"], p1["soft_or_score"])
    s_ap_p1 = average_precision_score(p1["is_cartel"], p1["soft_or_score"])

    print("\n--- Period 1 (2016-2017) Standalone Ranking Performance ---")
    print(f"Baseline Rank Reliability: ROC-AUC = {b_auc_p1:.4f}, Average Precision = {b_ap_p1:.4f}")
    print(f"Soft-OR Blend (alpha=0.5): ROC-AUC = {s_auc_p1:.4f}, Average Precision = {s_ap_p1:.4f}")

    p1_top_k = []
    for m_name, col in [("Baseline", "baseline_score"), ("Soft-OR", "soft_or_score")]:
        for k in [1, 2, 5, 10, 20]:
            p1_top_k.append(evaluate_subperiod_top_k(p1, col, m_name, k))
    df_p1_top_k = pd.DataFrame(p1_top_k)
    df_p1_top_k.to_csv(OUTPUT_DIR / "audit_period_1_2016_2017_metrics.csv", index=False)
    print("\nPeriod 1 (2016-2017) Top-K Metrics:")
    print(df_p1_top_k[["model", "top_k_percent", "rows_selected", "cartel_selected", "precision", "recall", "lift"]])

    # Period 2 (2018-2022) Analysis
    print("\n--- Period 2 (2018-2022) Label Truncation Analysis ---")
    print(f"Total tenders in 2018-2022: {len(p2):,}")
    print(f"Confirmed cartels in 2018-2022: {p2['is_cartel'].sum()}")
    print("Note: Because positive count is exactly zero, Recall is mathematically undefined (0/0),")
    print("and Precision is mathematically 0.00% regardless of the ranking model.")

    # How many 2018-2022 records were flagged in the overall test Top 5%?
    b_flagged_p2 = (df["in_top5_baseline"] & (df["tender_year"] >= 2018)).sum()
    s_flagged_p2 = (df["in_top5_soft_or"] & (df["tender_year"] >= 2018)).sum()
    print(f"\nOverall Top 5% Selected Tenders from 2018-2022 (All Automatically Counted as False Positives):")
    print(f"Baseline: {b_flagged_p2} / 311 ({b_flagged_p2/311:.1%})")
    print(f"Soft-OR : {s_flagged_p2} / 311 ({s_flagged_p2/311:.1%})")

    # Save summary table
    summary = {
        "period": ["2016-2017 (Active Labels)", "2018-2022 (Truncated Labels)", "Combined (2016-2022)"],
        "total_records": [len(p1), len(p2), len(df)],
        "cartels": [int(p1["is_cartel"].sum()), int(p2["is_cartel"].sum()), int(df["is_cartel"].sum())],
        "cartel_prevalence": [p1["is_cartel"].mean(), p2["is_cartel"].mean(), df["is_cartel"].mean()],
        "baseline_auc": [b_auc_p1, np.nan, roc_auc_score(df["is_cartel"], df["baseline_score"])],
        "baseline_ap": [b_ap_p1, np.nan, average_precision_score(df["is_cartel"], df["baseline_score"])],
        "soft_or_auc": [s_auc_p1, np.nan, roc_auc_score(df["is_cartel"], df["soft_or_score"])],
        "soft_or_ap": [s_ap_p1, np.nan, average_precision_score(df["is_cartel"], df["soft_or_score"])],
    }
    df_summary = pd.DataFrame(summary)
    df_summary.to_csv(OUTPUT_DIR / "audit_chronological_split_summary.csv", index=False)
    print("\nSummary Comparison across Time Windows:")
    print(df_summary)

if __name__ == "__main__":
    main()
