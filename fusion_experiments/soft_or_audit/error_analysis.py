# fusion_experiments/soft_or_audit/error_analysis.py
"""
Audit Step 5: Error and Subgroup Analysis
Compares the top-ranked test records between Baseline and Soft-OR:
1. Which modules provide the strongest evidence (dominant module in max_contrib).
2. Module availability and reliability profiles for TPs and FPs.
3. Country, Year, and CPV distribution.
4. Deep dive into the 33 newly discovered cartels found by Soft-OR in Top 5% that Baseline missed.
"""

from pathlib import Path
import numpy as np
import pandas as pd

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

def main():
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

    # Baseline calculation
    num = pd.Series(0.0, index=test.index)
    den = pd.Series(0.0, index=test.index)
    max_contrib = pd.Series(np.nan, index=test.index)
    dominant_mod = pd.Series(None, index=test.index, dtype=object)

    for r_col, rel_col, m_name in zip(rank_cols, RELIABILITY_COLUMNS, MODULE_NAMES):
        r = test[r_col]
        w = test[rel_col]
        valid = r.notna() & w.notna() & (w > 0)
        contrib = r * w
        num.loc[valid] += contrib.loc[valid]
        den.loc[valid] += w.loc[valid]

        # Element-wise dominant module tracker
        for idx in test[valid].index:
            c = contrib.loc[idx]
            if pd.isna(max_contrib.loc[idx]) or c > max_contrib.loc[idx]:
                max_contrib.loc[idx] = c
                dominant_mod.loc[idx] = m_name

    test["baseline_score"] = num / den
    test["soft_or_score"] = 0.5 * max_contrib + 0.5 * test["baseline_score"]
    test["dominant_module"] = dominant_mod
    test["max_contrib"] = max_contrib

    # Filter to scored records (6,203)
    scored = test[test["baseline_score"].notna()].copy()

    # Sort and rank for Top 5% screening
    top5_count = int(np.ceil(len(scored) * 0.05)) # 311

    scored = scored.sort_values(by="baseline_score", ascending=False, kind="mergesort").reset_index(drop=True)
    scored["rank_baseline"] = np.arange(1, len(scored) + 1)
    scored["in_top5_baseline"] = scored["rank_baseline"] <= top5_count

    scored = scored.sort_values(by="soft_or_score", ascending=False, kind="mergesort").reset_index(drop=True)
    scored["rank_soft_or"] = np.arange(1, len(scored) + 1)
    scored["in_top5_soft_or"] = scored["rank_soft_or"] <= top5_count

    # Save detailed audit records
    export_cols = [
        "row_id", "country", "tender_year", "main_cpv_2", "is_cartel",
        "baseline_score", "rank_baseline", "in_top5_baseline",
        "soft_or_score", "rank_soft_or", "in_top5_soft_or",
        "dominant_module", "max_contrib",
        "m1_rank", "m2_rank", "m3_rank", "m4_rank",
        "m1_reliability_v2", "m2_reliability_v2", "m3_reliability_v2", "m4_reliability_v2"
    ]
    scored[export_cols].to_parquet(OUTPUT_DIR / "audit_scored_test_records.parquet", index=False)

    # Error analysis in Top 5%
    top5_b = scored[scored["in_top5_baseline"]]
    top5_s = scored[scored["in_top5_soft_or"]]

    print("=" * 70)
    print("ERROR & SUBGROUP ANALYSIS: TOP 5% SCREENING (N=311)")
    print("=" * 70)
    print(f"Baseline Top 5%: TPs={top5_b['is_cartel'].sum()} ({top5_b['is_cartel'].mean():.2%}), FPs={(top5_b['is_cartel']==0).sum()}")
    print(f"Soft-OR  Top 5%: TPs={top5_s['is_cartel'].sum()} ({top5_s['is_cartel'].mean():.2%}), FPs={(top5_s['is_cartel']==0).sum()}")

    # Dominant module in Top 5% selections
    print("\nDominant Module for TPs in Soft-OR Top 5%:")
    print(top5_s[top5_s["is_cartel"] == 1]["dominant_module"].value_counts())

    print("\nDominant Module for FPs in Soft-OR Top 5%:")
    print(top5_s[top5_s["is_cartel"] == 0]["dominant_module"].value_counts())

    # Newly identified cartels by Soft-OR that Baseline missed
    newly_found = scored[scored["in_top5_soft_or"] & (~scored["in_top5_baseline"]) & (scored["is_cartel"] == 1)]
    print(f"\nNewly Caught Cartels in Top 5% by Soft-OR (N={len(newly_found)}):")
    print(f"Dominant Module distribution for newly caught cartels:")
    print(newly_found["dominant_module"].value_counts())
    print(f"Country distribution for newly caught cartels:")
    print(newly_found["country"].value_counts())
    print(f"Tender year distribution for newly caught cartels:")
    print(newly_found["tender_year"].value_counts())
    print(f"Mean baseline rank of newly caught cartels: {newly_found['rank_baseline'].mean():.1f} (median: {newly_found['rank_baseline'].median():.1f})")

    # Lost cartels (in baseline Top 5% but not in Soft-OR Top 5%)
    lost_cartels = scored[(~scored["in_top5_soft_or"]) & scored["in_top5_baseline"] & (scored["is_cartel"] == 1)]
    print(f"\nCartels Lost from Top 5% (in Baseline but not Soft-OR) (N={len(lost_cartels)}):")
    if len(lost_cartels) > 0:
        print(lost_cartels[["row_id", "country", "tender_year", "dominant_module", "baseline_score", "soft_or_score", "rank_baseline", "rank_soft_or"]])
    else:
        print("None! Exactly zero cartels were lost.")

    # Country breakdown
    print("\nCountry Distribution in Soft-OR Top 5%:")
    ctry_tp = top5_s[top5_s["is_cartel"] == 1]["country"].value_counts()
    ctry_fp = top5_s[top5_s["is_cartel"] == 0]["country"].value_counts()
    ctry_df = pd.DataFrame({"TP": ctry_tp, "FP": ctry_fp}).fillna(0).astype(int)
    ctry_df["Precision"] = ctry_df["TP"] / (ctry_df["TP"] + ctry_df["FP"])
    ctry_df.to_csv(OUTPUT_DIR / "audit_top5_country_breakdown.csv")
    print(ctry_df)

    # CPV breakdown
    print("\nTop 5 CPV Groups in Soft-OR Top 5%:")
    cpv_tp = top5_s[top5_s["is_cartel"] == 1]["main_cpv_2"].value_counts()
    cpv_fp = top5_s[top5_s["is_cartel"] == 0]["main_cpv_2"].value_counts()
    cpv_df = pd.DataFrame({"TP": cpv_tp, "FP": cpv_fp}).fillna(0).astype(int)
    cpv_df["Precision"] = cpv_df["TP"] / (cpv_df["TP"] + cpv_df["FP"])
    cpv_df = cpv_df.sort_values(by=["TP", "FP"], ascending=[False, True])
    cpv_df.to_csv(OUTPUT_DIR / "audit_top5_cpv_breakdown.csv")
    print(cpv_df.head(5))

if __name__ == "__main__":
    main()
