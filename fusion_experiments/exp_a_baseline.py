# fusion_experiments/exp_a_baseline.py
"""
Experiment A: Reproduce Current Final Baseline
Verifies exact match with final_fusion_results:
- Rank Reliability-aware Fusion
- Rank Equal-weight Fusion
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
OUTPUT_DIR = BASE_DIR / "fusion_experiments" / "results_a_baseline"
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

def equal_rank_fusion(dataframe, rank_cols):
    values = dataframe[rank_cols].copy().apply(pd.to_numeric, errors="coerce")
    return values.mean(axis=1, skipna=True)

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

def run_experiment_a():
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

    test["rank_reliability_fusion"] = weighted_rank_fusion(test, rank_cols, RELIABILITY_COLUMNS)
    test["rank_equal_weight_fusion"] = equal_rank_fusion(test, rank_cols)

    # Save test scores
    test[["row_id", "tender_year", "is_cartel", "rank_reliability_fusion", "rank_equal_weight_fusion"]].to_parquet(
        OUTPUT_DIR / "test_scores_baseline.parquet", index=False
    )

    ranking_metrics = [
        compute_metrics(test["is_cartel"], test["rank_reliability_fusion"], "Rank Reliability-aware Fusion"),
        compute_metrics(test["is_cartel"], test["rank_equal_weight_fusion"], "Rank Equal-weight Fusion"),
    ]
    ranking_df = pd.DataFrame(ranking_metrics)
    ranking_df.to_csv(OUTPUT_DIR / "ranking_metrics_baseline.csv", index=False)

    top_k_metrics = []
    for m_name, s_col in [
        ("Rank Reliability-aware Fusion", "rank_reliability_fusion"),
        ("Rank Equal-weight Fusion", "rank_equal_weight_fusion"),
    ]:
        for k in TOP_K_VALUES:
            top_k_metrics.append(compute_top_k(test, s_col, m_name, k))
    top_k_df = pd.DataFrame(top_k_metrics)
    top_k_df.to_csv(OUTPUT_DIR / "top_k_baseline.csv", index=False)

    print("Experiment A (Baseline) completed successfully.")
    print("\nRanking Metrics:")
    print(ranking_df[["model", "rows", "cartel", "roc_auc", "average_precision", "precision_at_0.5", "recall_at_0.5"]])
    print("\nTop-K Metrics (Rank Reliability-aware):")
    print(top_k_df[top_k_df["model"] == "Rank Reliability-aware Fusion"][["top_k_percent", "rows_selected", "cartel_selected", "precision", "recall", "lift"]])

if __name__ == "__main__":
    run_experiment_a()
