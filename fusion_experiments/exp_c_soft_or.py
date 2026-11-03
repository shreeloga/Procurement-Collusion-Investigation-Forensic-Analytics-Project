# fusion_experiments/exp_c_soft_or.py
"""
Experiment C: Soft-OR and Generalized Mean Aggregation
Allows strong signal from a single module to elevate the risk score without being heavily diluted.
Tuning performed strictly on chronological validation within the training set:
- Train: 2004-2012
- Val: 2013-2015
Best aggregation rule selected by validation Average Precision.
Evaluated once on test period (2016-2022).
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
INPUT_FILE = BASE_DIR / "labelled_fusion_scores.parquet"
OUTPUT_DIR = BASE_DIR / "fusion_experiments" / "results_c_soft_or"
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

def generalized_p_mean_fusion(dataframe, rank_cols, rel_cols, p=2.0):
    """
    Generalized p-mean reliability-weighted fusion:
    F = [ sum(w_i * rank_i^p) / sum(w_i) ]^(1/p)
    When p=1, equals standard weighted mean.
    When p>1, emphasizes highest module ranks (soft maximum).
    """
    num = pd.Series(0.0, index=dataframe.index)
    den = pd.Series(0.0, index=dataframe.index)
    for r_col, rel_col in zip(rank_cols, rel_cols):
        r = pd.to_numeric(dataframe[r_col], errors="coerce")
        w = pd.to_numeric(dataframe[rel_col], errors="coerce")
        valid = r.notna() & w.notna() & np.isfinite(r) & np.isfinite(w) & (w > 0)
        num.loc[valid] += w.loc[valid] * (r.loc[valid] ** p)
        den.loc[valid] += w.loc[valid]
    score = pd.Series(np.nan, index=dataframe.index)
    valid_den = den > 0
    score.loc[valid_den] = (num.loc[valid_den] / den.loc[valid_den]) ** (1.0 / p)
    return score

def soft_or_blend_fusion(dataframe, rank_cols, rel_cols, alpha=0.5):
    """
    Blends the maximum single-module reliability-weighted score with the weighted mean:
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
        
        # Track element-wise max
        curr_max = max_val.loc[valid]
        new_max = np.where(curr_max.isna(), contrib, np.maximum(curr_max, contrib))
        max_val.loc[valid] = new_max
        
    score = pd.Series(np.nan, index=dataframe.index)
    valid_den = den > 0
    w_mean = num.loc[valid_den] / den.loc[valid_den]
    score.loc[valid_den] = alpha * max_val.loc[valid_den] + (1.0 - alpha) * w_mean
    return score

def compute_metrics(y_true, y_score, model_name):
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

def compute_top_k(df, score_col, model_name, k):
    work = df[["row_id", "is_cartel", score_col]].dropna(subset=["is_cartel", score_col]).copy()
    work["is_cartel"] = work["is_cartel"].astype(int)
    work = work.sort_values(by=score_col, ascending=False, kind="mergesort").reset_index(drop=True)
    n_select = max(1, int(np.ceil(len(work) * k / 100.0)))
    selected = work.iloc[:n_select]
    tp = int(selected["is_cartel"].sum())
    total_c = int(work["is_cartel"].sum())
    p = tp / len(selected) if len(selected) > 0 else 0.0
    r = tp / total_c if total_c > 0 else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    prev = total_c / len(work)
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

def run_experiment_c():
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

    # Chronological validation within train period: 2004-2012 train, 2013-2015 val
    val = train[train["tender_year"] >= 2013].copy()
    print(f"Validation set size: {len(val):,}, Cartels: {val['is_cartel'].sum()}")

    # Search space for p in Generalized P-Mean
    p_grid = [1.0, 1.5, 2.0, 3.0, 4.0]
    p_results = []
    for p in p_grid:
        val_score = generalized_p_mean_fusion(val, rank_cols, RELIABILITY_COLUMNS, p=p)
        valid = val_score.notna()
        auc = roc_auc_score(val.loc[valid, "is_cartel"], val_score.loc[valid])
        ap = average_precision_score(val.loc[valid, "is_cartel"], val_score.loc[valid])
        p_results.append({"method": "p_mean", "param": p, "val_auc": auc, "val_ap": ap})
        print(f"p-mean p={p}: Val AUC={auc:.4f}, Val AP={ap:.4f}")

    # Search space for alpha in Soft-OR blend
    alpha_grid = [0.1, 0.25, 0.5, 0.75, 0.9]
    for alpha in alpha_grid:
        val_score = soft_or_blend_fusion(val, rank_cols, RELIABILITY_COLUMNS, alpha=alpha)
        valid = val_score.notna()
        auc = roc_auc_score(val.loc[valid, "is_cartel"], val_score.loc[valid])
        ap = average_precision_score(val.loc[valid, "is_cartel"], val_score.loc[valid])
        p_results.append({"method": "soft_or_blend", "param": alpha, "val_auc": auc, "val_ap": ap})
        print(f"soft-OR alpha={alpha}: Val AUC={auc:.4f}, Val AP={ap:.4f}")

    tuning_df = pd.DataFrame(p_results)
    tuning_df.to_csv(OUTPUT_DIR / "tuning_results_soft_or.csv", index=False)

    best_p_row = tuning_df[tuning_df["method"] == "p_mean"].sort_values("val_ap", ascending=False).iloc[0]
    best_p = best_p_row["param"]
    best_blend_row = tuning_df[tuning_df["method"] == "soft_or_blend"].sort_values("val_ap", ascending=False).iloc[0]
    best_alpha = best_blend_row["param"]

    print(f"\nBest Generalized p-Mean: p={best_p} (Val AP={best_p_row['val_ap']:.4f})")
    print(f"Best Soft-OR Blend: alpha={best_alpha} (Val AP={best_blend_row['val_ap']:.4f})")

    # Evaluate both best rules on the test set
    test["score_p_mean"] = generalized_p_mean_fusion(test, rank_cols, RELIABILITY_COLUMNS, p=best_p)
    test["score_soft_or"] = soft_or_blend_fusion(test, rank_cols, RELIABILITY_COLUMNS, alpha=best_alpha)

    # Save test scores
    test[["row_id", "tender_year", "is_cartel", "score_p_mean", "score_soft_or"]].to_parquet(
        OUTPUT_DIR / "test_scores_soft_or.parquet", index=False
    )

    ranking_metrics = [
        compute_metrics(test["is_cartel"], test["score_p_mean"], f"Generalized Mean Fusion (p={best_p})"),
        compute_metrics(test["is_cartel"], test["score_soft_or"], f"Soft-OR Blend Fusion (alpha={best_alpha})"),
    ]
    ranking_df = pd.DataFrame(ranking_metrics)
    ranking_df.to_csv(OUTPUT_DIR / "ranking_metrics_soft_or.csv", index=False)

    top_k_metrics = []
    for m_name, s_col in [
        (f"Generalized Mean Fusion (p={best_p})", "score_p_mean"),
        (f"Soft-OR Blend Fusion (alpha={best_alpha})", "score_soft_or"),
    ]:
        for k in TOP_K_VALUES:
            top_k_metrics.append(compute_top_k(test, s_col, m_name, k))
    top_k_df = pd.DataFrame(top_k_metrics)
    top_k_df.to_csv(OUTPUT_DIR / "top_k_soft_or.csv", index=False)

    print("\nExperiment C Ranking Metrics on Test Set:")
    print(ranking_df[["model", "rows", "cartel", "roc_auc", "average_precision", "precision_at_0.5", "recall_at_0.5"]])
    print("\nExperiment C Top-K Metrics on Test Set:")
    print(top_k_df[["model", "top_k_percent", "rows_selected", "cartel_selected", "precision", "recall", "lift"]])

if __name__ == "__main__":
    run_experiment_c()
