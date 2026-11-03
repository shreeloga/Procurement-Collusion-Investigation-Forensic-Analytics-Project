# fusion_experiments/aggregate_comparisons.py
"""
Aggregates all experiment results (A, B, C, D, E) and generates:
1. master_comparison_available.csv (performance on each method's available scored records)
2. master_comparison_common_set.csv (performance on the strictly identical common subset of records scored by all methods)
3. master_top_k_comparison.csv (top 1%, 2%, 5%, 10%, 20% screening tables)
"""

from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

BASE_DIR = Path(__file__).resolve().parent.parent
EXP_DIR = BASE_DIR / "fusion_experiments"

# Load parquet score files
df_base = pd.read_parquet(EXP_DIR / "results_a_baseline" / "test_scores_baseline.parquet")
df_b = pd.read_parquet(EXP_DIR / "results_b_nonlinear" / "test_scores_nonlinear.parquet")
df_c = pd.read_parquet(EXP_DIR / "results_c_soft_or" / "test_scores_soft_or.parquet")
df_d = pd.read_parquet(EXP_DIR / "results_d_reliability_gating" / "test_scores_gating.parquet")
df_e = pd.read_parquet(EXP_DIR / "results_e_non_cartel_ref" / "test_scores_non_cartel_ref.parquet")

# Merge on row_id
merged = df_base[["row_id", "tender_year", "is_cartel", "rank_reliability_fusion", "rank_equal_weight_fusion"]].copy()
merged = merged.merge(df_b[["row_id", "xgb_nonlinear_fusion"]], on="row_id", how="left")
merged = merged.merge(df_c[["row_id", "score_p_mean", "score_soft_or"]], on="row_id", how="left")
merged = merged.merge(df_d[["row_id", "score_tau_20", "score_tau_40"]], on="row_id", how="left")
merged = merged.merge(df_e[["row_id", "rank_clean_reliability_fusion"]], on="row_id", how="left")

# Models to compare
MODELS = [
    ("A: Baseline Rank Reliability-Aware", "rank_reliability_fusion"),
    ("A: Baseline Rank Equal-Weight", "rank_equal_weight_fusion"),
    ("B: Supervised Nonlinear Meta-Fusion", "xgb_nonlinear_fusion"),
    ("C: Soft-OR Blend Fusion (alpha=0.5)", "score_soft_or"),
    ("C: Generalized Mean Fusion (p=1.0)", "score_p_mean"),
    ("D: Reliability Gating (tau=0.2)", "score_tau_20"),
    ("D: Reliability Gating (tau=0.4)", "score_tau_40"),
    ("E: Non-Cartel Reference Fusion", "rank_clean_reliability_fusion"),
]

TOP_K_VALUES = [1, 2, 5, 10, 20]

def compute_ranking_row(df, score_col, model_name):
    valid = df[score_col].notna() & df["is_cartel"].notna()
    sub = df[valid]
    y = sub["is_cartel"].astype(int).to_numpy()
    s = sub[score_col].astype(float).to_numpy()
    auc = roc_auc_score(y, s)
    ap = average_precision_score(y, s)
    pred_05 = (s >= 0.5).astype(int)
    p_05 = precision_score(y, pred_05, zero_division=0)
    r_05 = recall_score(y, pred_05, zero_division=0)
    f1_05 = f1_score(y, pred_05, zero_division=0)
    tn, fp, fn, tp = confusion_matrix(y, pred_05, labels=[0, 1]).ravel()
    spec_05 = tn / (tn + fp) if (tn + fp) > 0 else np.nan
    return {
        "model": model_name,
        "scored_rows": len(y),
        "total_test_rows": len(df),
        "coverage_pct": len(y) / len(df) * 100,
        "cartel_cases": int(y.sum()),
        "non_cartel_cases": int((y == 0).sum()),
        "cartel_prevalence": y.sum() / len(y),
        "roc_auc": auc,
        "average_precision": ap,
        "precision_at_0.5": p_05,
        "recall_at_0.5": r_05,
        "specificity_at_0.5": spec_05,
        "f1_at_0.5": f1_05,
        "tn_at_0.5": int(tn),
        "fp_at_0.5": int(fp),
        "fn_at_0.5": int(fn),
        "tp_at_0.5": int(tp),
    }

def compute_top_k_row(df, score_col, model_name, k):
    sub = df.dropna(subset=[score_col, "is_cartel"]).copy()
    y = sub["is_cartel"].astype(int)
    s = sub[score_col].astype(float)
    sub = sub.sort_values(by=score_col, ascending=False, kind="mergesort").reset_index(drop=True)
    n_select = max(1, int(np.ceil(len(sub) * k / 100.0)))
    selected = sub.iloc[:n_select]
    tp = int(selected["is_cartel"].sum())
    total_c = int(y.sum())
    p = tp / len(selected) if len(selected) > 0 else 0.0
    r = tp / total_c if total_c > 0 else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    prev = total_c / len(sub)
    lift = p / prev if prev > 0 else np.nan
    return {
        "model": model_name,
        "top_k_percent": k,
        "total_available_rows": len(sub),
        "rows_selected": len(selected),
        "total_cartel": total_c,
        "cartel_selected": tp,
        "precision": p,
        "recall": r,
        "f1": f1,
        "lift": lift,
    }

# 1. Available records evaluation
avail_metrics = []
top_k_avail = []
for m_name, s_col in MODELS:
    avail_metrics.append(compute_ranking_row(merged, s_col, m_name))
    for k in TOP_K_VALUES:
        top_k_avail.append(compute_top_k_row(merged, s_col, m_name, k))

df_avail_metrics = pd.DataFrame(avail_metrics)
df_avail_metrics.to_csv(EXP_DIR / "master_comparison_available.csv", index=False)

df_top_k_avail = pd.DataFrame(top_k_avail)
df_top_k_avail.to_csv(EXP_DIR / "master_top_k_comparison.csv", index=False)

# 2. Common set evaluation (records scored by all models)
common_mask = pd.Series(True, index=merged.index)
for _, s_col in MODELS:
    common_mask &= merged[s_col].notna()

common_subset = merged[common_mask].copy()
print(f"Common set of records scored by all 8 models: {len(common_subset):,} / {len(merged):,}")
print(f"Common set cartels: {common_subset['is_cartel'].sum()} ({common_subset['is_cartel'].mean():.4%})")

common_metrics = []
top_k_common = []
for m_name, s_col in MODELS:
    common_metrics.append(compute_ranking_row(common_subset, s_col, f"{m_name} [Common Set]"))
    for k in TOP_K_VALUES:
        top_k_common.append(compute_top_k_row(common_subset, s_col, f"{m_name} [Common Set]", k))

df_common_metrics = pd.DataFrame(common_metrics)
df_common_metrics.to_csv(EXP_DIR / "master_comparison_common_set.csv", index=False)

df_top_k_common = pd.DataFrame(top_k_common)
df_top_k_common.to_csv(EXP_DIR / "master_top_k_common_set.csv", index=False)

print("\n--- MASTER COMPARISON: AVAILABLE RECORDS ---")
print(df_avail_metrics[["model", "scored_rows", "cartel_cases", "roc_auc", "average_precision"]])

print("\n--- MASTER COMPARISON: COMMON SET RECORDS (N=5,780) ---")
print(df_common_metrics[["model", "scored_rows", "cartel_cases", "roc_auc", "average_precision"]])

print("\n--- TOP-5% COMPARISON (AVAILABLE) ---")
top5 = df_top_k_avail[df_top_k_avail["top_k_percent"] == 5]
print(top5[["model", "rows_selected", "cartel_selected", "precision", "recall", "lift"]])
