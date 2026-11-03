# fusion_experiments/soft_or_audit/sensitivity_analysis.py
"""
Audit Step 4: Sensitivity Analysis on Training/Validation Period Only
Compares:
1. Formulation 1 (Implemented Soft-OR):
   F_1(alpha) = alpha * max(rank_i * rel_i) + (1 - alpha) * weighted_mean
   with alpha in [0.0 (pure weighted mean), 0.25, 0.50, 0.75, 1.0 (pure max weighted rank)]
2. Formulation 2 (Normalized Max Active Rank):
   F_2(alpha) = alpha * max_active(rank_i) + (1 - alpha) * weighted_mean
   with alpha in [0.0, 0.25, 0.50, 0.75, 1.0]
3. Formulation 3 (Scale-Calibrated Max):
   F_3(alpha) = alpha * [max(rank_i * rel_i) / max_active(rel_i)] + (1 - alpha) * weighted_mean

Validation Scheme:
- Chronological train: 2004-2012 (N=8,172)
- Chronological val  : 2013-2015 (N=4,427)
Evaluation on Test Period (2016-2022) is reported for comparison against the frozen validation selection.
"""

from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score

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

def compute_fusion_components(dataframe, rank_cols, rel_cols):
    num = pd.Series(0.0, index=dataframe.index)
    den = pd.Series(0.0, index=dataframe.index)
    max_weighted_rank = pd.Series(np.nan, index=dataframe.index)
    max_active_rank = pd.Series(np.nan, index=dataframe.index)
    max_active_rel = pd.Series(np.nan, index=dataframe.index)

    for r_col, rel_col in zip(rank_cols, rel_cols):
        r = pd.to_numeric(dataframe[r_col], errors="coerce")
        w = pd.to_numeric(dataframe[rel_col], errors="coerce")
        valid = r.notna() & w.notna() & np.isfinite(r) & np.isfinite(w) & (w > 0)
        
        contrib = r.loc[valid] * w.loc[valid]
        num.loc[valid] += contrib
        den.loc[valid] += w.loc[valid]

        # max(rank_i * rel_i)
        curr_mw = max_weighted_rank.loc[valid]
        max_weighted_rank.loc[valid] = np.where(curr_mw.isna(), contrib, np.maximum(curr_mw, contrib))

        # max active rank_i
        curr_mr = max_active_rank.loc[valid]
        max_active_rank.loc[valid] = np.where(curr_mr.isna(), r.loc[valid], np.maximum(curr_mr, r.loc[valid]))

        # max active rel_i
        curr_mrel = max_active_rel.loc[valid]
        max_active_rel.loc[valid] = np.where(curr_mrel.isna(), w.loc[valid], np.maximum(curr_mrel, w.loc[valid]))

    w_mean = pd.Series(np.nan, index=dataframe.index)
    valid_den = den > 0
    w_mean.loc[valid_den] = num.loc[valid_den] / den.loc[valid_den]

    return {
        "w_mean": w_mean,
        "max_weighted_rank": max_weighted_rank,
        "max_active_rank": max_active_rank,
        "max_active_rel": max_active_rel,
    }

def main():
    print("=" * 70)
    print("SENSITIVITY ANALYSIS: FORMULATION & ALPHA TUNING ON VALIDATION DATA")
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

    # Chronological validation set within training: 2013-2015
    val = train[train["tender_year"] >= 2013].copy()
    print(f"Internal Validation Set (2013-2015): N={len(val):,}, Cartels={val['is_cartel'].sum()}")

    val_comp = compute_fusion_components(val, rank_cols, RELIABILITY_COLUMNS)
    test_comp = compute_fusion_components(test, rank_cols, RELIABILITY_COLUMNS)

    alpha_grid = [0.0, 0.25, 0.50, 0.75, 1.0]
    records = []

    # 1. Formulation 1: Implemented Soft-OR
    for alpha in alpha_grid:
        v_score = alpha * val_comp["max_weighted_rank"] + (1.0 - alpha) * val_comp["w_mean"]
        v_valid = v_score.notna()
        v_auc = roc_auc_score(val.loc[v_valid, "is_cartel"], v_score.loc[v_valid])
        v_ap = average_precision_score(val.loc[v_valid, "is_cartel"], v_score.loc[v_valid])

        t_score = alpha * test_comp["max_weighted_rank"] + (1.0 - alpha) * test_comp["w_mean"]
        t_valid = t_score.notna()
        t_auc = roc_auc_score(test.loc[t_valid, "is_cartel"], t_score.loc[t_valid])
        t_ap = average_precision_score(test.loc[t_valid, "is_cartel"], t_score.loc[t_valid])

        records.append({
            "formulation": "Formulation 1: Implemented Soft-OR [alpha * max(r*w) + (1-alpha) * w_mean]",
            "alpha": alpha,
            "val_auc": v_auc,
            "val_ap": v_ap,
            "test_auc": t_auc,
            "test_ap": t_ap,
        })

    # 2. Formulation 2: Normalized Max Active Rank
    for alpha in alpha_grid:
        v_score = alpha * val_comp["max_active_rank"] + (1.0 - alpha) * val_comp["w_mean"]
        v_valid = v_score.notna()
        v_auc = roc_auc_score(val.loc[v_valid, "is_cartel"], v_score.loc[v_valid])
        v_ap = average_precision_score(val.loc[v_valid, "is_cartel"], v_score.loc[v_valid])

        t_score = alpha * test_comp["max_active_rank"] + (1.0 - alpha) * test_comp["w_mean"]
        t_valid = t_score.notna()
        t_auc = roc_auc_score(test.loc[t_valid, "is_cartel"], t_score.loc[t_valid])
        t_ap = average_precision_score(test.loc[t_valid, "is_cartel"], t_score.loc[t_valid])

        records.append({
            "formulation": "Formulation 2: Normalized Max Rank [alpha * max_active(r) + (1-alpha) * w_mean]",
            "alpha": alpha,
            "val_auc": v_auc,
            "val_ap": v_ap,
            "test_auc": t_auc,
            "test_ap": t_ap,
        })

    # 3. Formulation 3: Scale-Calibrated Max
    for alpha in alpha_grid:
        calib_v = val_comp["max_weighted_rank"] / val_comp["max_active_rel"].replace(0, np.nan)
        v_score = alpha * calib_v + (1.0 - alpha) * val_comp["w_mean"]
        v_valid = v_score.notna()
        v_auc = roc_auc_score(val.loc[v_valid, "is_cartel"], v_score.loc[v_valid])
        v_ap = average_precision_score(val.loc[v_valid, "is_cartel"], v_score.loc[v_valid])

        calib_t = test_comp["max_weighted_rank"] / test_comp["max_active_rel"].replace(0, np.nan)
        t_score = alpha * calib_t + (1.0 - alpha) * test_comp["w_mean"]
        t_valid = t_score.notna()
        t_auc = roc_auc_score(test.loc[t_valid, "is_cartel"], t_score.loc[t_valid])
        t_ap = average_precision_score(test.loc[t_valid, "is_cartel"], t_score.loc[t_valid])

        records.append({
            "formulation": "Formulation 3: Calibrated Max [alpha * (max(r*w)/max_rel) + (1-alpha) * w_mean]",
            "alpha": alpha,
            "val_auc": v_auc,
            "val_ap": v_ap,
            "test_auc": t_auc,
            "test_ap": t_ap,
        })

    df_results = pd.DataFrame(records)
    df_results.to_csv(OUTPUT_DIR / "sensitivity_analysis_results.csv", index=False)

    print("\nSensitivity Analysis Results:")
    for f in df_results["formulation"].unique():
        print(f"\n>>> {f}")
        sub = df_results[df_results["formulation"] == f]
        for _, r in sub.iterrows():
            print(f"  alpha={r['alpha']:.2f} | Val AUC: {r['val_auc']:.4f}, Val AP: {r['val_ap']:.4f} || Test AUC: {r['test_auc']:.4f}, Test AP: {r['test_ap']:.4f}")

    # Best selection on validation set
    best_val_row = df_results.sort_values(by="val_ap", ascending=False).iloc[0]
    print("\n" + "=" * 70)
    print(f"BEST CONFIGURATION SELECTED STRICTLY ON VALIDATION SET (2013-2015):")
    print(f"Formulation: {best_val_row['formulation']}")
    print(f"Alpha: {best_val_row['alpha']}")
    print(f"Validation AP: {best_val_row['val_ap']:.4f} (Val AUC: {best_val_row['val_auc']:.4f})")
    print(f"Corresponding Test AP: {best_val_row['test_ap']:.4f} (Test AUC: {best_val_row['test_auc']:.4f})")
    print("=" * 70)

if __name__ == "__main__":
    main()
