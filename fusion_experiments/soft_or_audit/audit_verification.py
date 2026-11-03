# fusion_experiments/soft_or_audit/audit_verification.py
"""
Audit Step 1 & 2 & 3:
1. Verify exact formula and implementation in exp_c_soft_or.py.
2. Reproduce Baseline and Soft-OR results on identical 6,203 test records from raw inputs.
3. Check Top-K selection rules, rounding, and lift calculations.
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

BASE_DIR = Path(__file__).resolve().parent.parent.parent
INPUT_FILE = BASE_DIR / "labelled_fusion_scores.parquet"
OUTPUT_DIR = BASE_DIR / "fusion_experiments" / "soft_or_audit"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

EVIDENCE_COLUMNS = [
    "m1_evidence_score_v3",
    "m2_evidence_score_v4",
    "m3_evidence_score_v3",
    "m4_evidence_score_v3",
]

RELIABILITY_COLUMNS = [
    "m1_reliability_v2",
    "m2_reliability_v2",
    "m3_reliability_v2",
    "m4_reliability_v2",
]

MODULE_NAMES = ["M1", "M2", "M3", "M4"]
TOP_K_VALUES = [1, 2, 5, 10, 20]

def fit_rank_reference(train_series):
    values = pd.to_numeric(train_series, errors="coerce")
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return np.array([], dtype=float)
    return np.sort(values.to_numpy(dtype=float))

def apply_rank_reference(series, reference):
    values = pd.to_numeric(series, errors="coerce")
    result = pd.Series(np.nan, index=series.index, dtype=float)
    if len(reference) == 0:
        return result
    valid = values.notna()
    if not valid.any():
        return result
    x = values.loc[valid].to_numpy(dtype=float)
    positions = np.searchsorted(reference, x, side="right")
    result.loc[valid] = positions / len(reference)
    return result

def weighted_rank_fusion(dataframe, rank_cols, rel_cols):
    numerator = pd.Series(0.0, index=dataframe.index)
    denominator = pd.Series(0.0, index=dataframe.index)
    for r_col, rel_col in zip(rank_cols, rel_cols):
        r = pd.to_numeric(dataframe[r_col], errors="coerce")
        w = pd.to_numeric(dataframe[rel_col], errors="coerce")
        valid = r.notna() & w.notna() & np.isfinite(r) & np.isfinite(w) & (w > 0)
        numerator.loc[valid] += r.loc[valid] * w.loc[valid]
        denominator.loc[valid] += w.loc[valid]
    score = numerator / denominator
    score.loc[denominator == 0] = np.nan
    return score

def soft_or_blend_fusion(dataframe, rank_cols, rel_cols, alpha=0.5):
    """
    Exact implementation from exp_c_soft_or.py:
    F = alpha * max(rank_i * rel_i) + (1 - alpha) * weighted_mean
    """
    num = pd.Series(0.0, index=dataframe.index)
    den = pd.Series(0.0, index=dataframe.index)
    max_val = pd.Series(np.nan, index=dataframe.index)
    
    for r_col, rel_col in zip(rank_cols, rel_cols):
        r = pd.to_numeric(dataframe[r_col], errors="coerce")
        w = pd.to_numeric(dataframe[rel_col], errors="coerce")
        valid = r.notna() & w.notna() & np.isfinite(r) & np.isfinite(w) & (w > 0)
        contrib = r.loc[valid] * w.loc[valid]
        num.loc[valid] += contrib
        den.loc[valid] += w.loc[valid]
        
        curr_max = max_val.loc[valid]
        new_max = np.where(curr_max.isna(), contrib, np.maximum(curr_max, contrib))
        max_val.loc[valid] = new_max
        
    score = pd.Series(np.nan, index=dataframe.index)
    valid_den = den > 0
    w_mean = num.loc[valid_den] / den.loc[valid_den]
    score.loc[valid_den] = alpha * max_val.loc[valid_den] + (1.0 - alpha) * w_mean
    return score, max_val, w_mean

def evaluate_metrics(y_true, y_score, model_name):
    valid = pd.Series(y_true).notna() & pd.Series(y_score).notna()
    y = pd.Series(y_true).loc[valid].astype(int).to_numpy()
    s = pd.Series(y_score).loc[valid].astype(float).to_numpy()
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
        "rows": len(y),
        "cartel": int(y.sum()),
        "non_cartel": int((y == 0).sum()),
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

def evaluate_top_k(df, score_col, model_name, k):
    work = df[["row_id", "is_cartel", score_col]].dropna(subset=["is_cartel", score_col]).copy()
    work["is_cartel"] = work["is_cartel"].astype(int)
    # Mergesort for stable tie-breaking
    work = work.sort_values(by=score_col, ascending=False, kind="mergesort").reset_index(drop=True)
    n_select = max(1, int(np.ceil(len(work) * k / 100.0)))
    selected = work.iloc[:n_select]
    tp = int(selected["is_cartel"].sum())
    total_c = int(work["is_cartel"].sum())
    p = tp / len(selected) if len(selected) > 0 else 0.0
    r = tp / total_c if total_c > 0 else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    prevalence = total_c / len(work)
    lift = p / prevalence if prevalence > 0 else np.nan
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
    print("=" * 70)
    print("AUDIT: INDEPENDENT RE-RUN OF BASELINE AND SOFT-OR FUSION")
    print("=" * 70)

    df = pd.read_parquet(INPUT_FILE)
    df = df[df["is_cartel"].isin([0, 1])].copy()
    df["is_cartel"] = df["is_cartel"].astype(int)

    train = df[df["tender_year"] < 2016].copy()
    test = df[df["tender_year"] >= 2016].copy()

    rank_cols = []
    for m, col in zip(MODULE_NAMES, EVIDENCE_COLUMNS):
        ref = fit_rank_reference(train[col])
        train[f"{m.lower()}_rank"] = apply_rank_reference(train[col], ref)
        test[f"{m.lower()}_rank"] = apply_rank_reference(test[col], ref)
        rank_cols.append(f"{m.lower()}_rank")

    test["baseline_rank_reliability"] = weighted_rank_fusion(test, rank_cols, RELIABILITY_COLUMNS)
    test["soft_or_alpha_05"], test["max_contrib"], test["w_mean"] = soft_or_blend_fusion(
        test, rank_cols, RELIABILITY_COLUMNS, alpha=0.5
    )

    # Scale check between max_contrib and w_mean
    valid_both = test["max_contrib"].notna() & test["w_mean"].notna()
    print("\nScale Comparison on Test Set (N=6,203):")
    print(f"max(rank_i * rel_i): mean={test.loc[valid_both, 'max_contrib'].mean():.4f}, median={test.loc[valid_both, 'max_contrib'].median():.4f}, min={test.loc[valid_both, 'max_contrib'].min():.4f}, max={test.loc[valid_both, 'max_contrib'].max():.4f}")
    print(f"weighted_mean      : mean={test.loc[valid_both, 'w_mean'].mean():.4f}, median={test.loc[valid_both, 'w_mean'].median():.4f}, min={test.loc[valid_both, 'w_mean'].min():.4f}, max={test.loc[valid_both, 'w_mean'].max():.4f}")

    # Ranking metrics
    metrics = [
        evaluate_metrics(test["is_cartel"], test["baseline_rank_reliability"], "Baseline Rank Reliability-Aware"),
        evaluate_metrics(test["is_cartel"], test["soft_or_alpha_05"], "Soft-OR Blend (alpha=0.5)"),
    ]
    df_metrics = pd.DataFrame(metrics)
    df_metrics.to_csv(OUTPUT_DIR / "audit_reproduced_ranking_metrics.csv", index=False)

    top_k = []
    for m_name, s_col in [
        ("Baseline Rank Reliability-Aware", "baseline_rank_reliability"),
        ("Soft-OR Blend (alpha=0.5)", "soft_or_alpha_05"),
    ]:
        for k in TOP_K_VALUES:
            top_k.append(evaluate_top_k(test, s_col, m_name, k))
    df_top_k = pd.DataFrame(top_k)
    df_top_k.to_csv(OUTPUT_DIR / "audit_reproduced_top_k.csv", index=False)

    print("\nReproduced Ranking Metrics:")
    for _, r in df_metrics.iterrows():
        print(f"[{r['model']}] Scored Rows: {r['rows']}, Cartels: {r['cartel']}, AUC: {r['roc_auc']:.6f}, AP: {r['average_precision']:.6f}")

    print("\nReproduced Top-K Screening Metrics:")
    for _, r in df_top_k.iterrows():
        print(f"[{r['model']}] Top {r['top_k_percent']:>2}%: Selected={r['rows_selected']}, Cartels={r['cartel_selected']}, Precision={r['precision']:.4%}, Recall={r['recall']:.4%}, Lift={r['lift']:.2f}x")

    # Verify against expected targets
    print("\nVerification Checklist:")
    b_auc = df_metrics.loc[df_metrics["model"] == "Baseline Rank Reliability-Aware", "roc_auc"].iloc[0]
    b_ap = df_metrics.loc[df_metrics["model"] == "Baseline Rank Reliability-Aware", "average_precision"].iloc[0]
    b_top5_p = df_top_k.loc[(df_top_k["model"] == "Baseline Rank Reliability-Aware") & (df_top_k["top_k_percent"] == 5), "precision"].iloc[0]
    b_top5_c = df_top_k.loc[(df_top_k["model"] == "Baseline Rank Reliability-Aware") & (df_top_k["top_k_percent"] == 5), "cartel_selected"].iloc[0]

    s_auc = df_metrics.loc[df_metrics["model"] == "Soft-OR Blend (alpha=0.5)", "roc_auc"].iloc[0]
    s_ap = df_metrics.loc[df_metrics["model"] == "Soft-OR Blend (alpha=0.5)", "average_precision"].iloc[0]
    s_top5_p = df_top_k.loc[(df_top_k["model"] == "Soft-OR Blend (alpha=0.5)") & (df_top_k["top_k_percent"] == 5), "precision"].iloc[0]
    s_top5_c = df_top_k.loc[(df_top_k["model"] == "Soft-OR Blend (alpha=0.5)") & (df_top_k["top_k_percent"] == 5), "cartel_selected"].iloc[0]

    print(f"Baseline AUC target: 0.7559 | Actual: {b_auc:.4f} -> {'MATCH' if np.isclose(b_auc, 0.7559, atol=1e-4) else 'MISMATCH'}")
    print(f"Baseline AP target: 0.0937  | Actual: {b_ap:.4f}  -> {'MATCH' if np.isclose(b_ap, 0.0937, atol=1e-4) else 'MISMATCH'}")
    print(f"Baseline Top-5 P: 13.50%    | Actual: {b_top5_p:.2%} -> {'MATCH' if np.isclose(b_top5_p, 0.1350, atol=1e-4) else 'MISMATCH'}")
    print(f"Baseline Top-5 C: 42        | Actual: {b_top5_c}     -> {'MATCH' if b_top5_c == 42 else 'MISMATCH'}")

    print(f"Soft-OR AUC target: 0.7580  | Actual: {s_auc:.4f} -> {'MATCH' if np.isclose(s_auc, 0.7580, atol=1e-4) else 'MISMATCH'}")
    print(f"Soft-OR AP target: 0.2062   | Actual: {s_ap:.4f}  -> {'MATCH' if np.isclose(s_ap, 0.2062, atol=1e-4) else 'MISMATCH'}")
    print(f"Soft-OR Top-5 P: 24.12%     | Actual: {s_top5_p:.2%} -> {'MATCH' if np.isclose(s_top5_p, 0.2412, atol=1e-4) else 'MISMATCH'}")
    print(f"Soft-OR Top-5 C: 75         | Actual: {s_top5_c}     -> {'MATCH' if s_top5_c == 75 else 'MISMATCH'}")

if __name__ == "__main__":
    main()
