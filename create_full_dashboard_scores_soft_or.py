# create_full_dashboard_scores_soft_or.py
"""
Generates full-dataset procurement cartel risk scores using the audited Soft-OR Fusion method.
Formula:
    F_SoftOR(alpha=0.5) = 0.5 * max(r_i * q_i) + 0.5 * [ sum(r_i * q_i) / sum(q_i) ]
where r_i is module percentile rank and q_i is module reliability (q_i > 0).

Output:
    dashboard_data/full_tender_risk_scores_soft_or.parquet
    (The baseline file dashboard_data/full_tender_risk_scores.parquet is strictly preserved)
"""

import json
from pathlib import Path
import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent

print("=" * 70)
print("CREATING FULL-DATASET DASHBOARD SCORES (SOFT-OR FUSION, ALPHA=0.5)")
print("=" * 70)

# Check input sources
DASHBOARD_DATA_DIR = BASE_DIR / "dashboard_data"
DASHBOARD_DATA_DIR.mkdir(parents=True, exist_ok=True)

BASELINE_FILE = DASHBOARD_DATA_DIR / "full_tender_risk_scores.parquet"
OUTPUT_FILE = DASHBOARD_DATA_DIR / "full_tender_risk_scores_soft_or.parquet"

# Modules configuration
MODULE_DEFS = {
    "m1": {"evidence": "m1_evidence_score_v3", "reliability": "m1_reliability_v2"},
    "m2": {"evidence": "m2_evidence_score_v4", "reliability": "m2_reliability_v2"},
    "m3": {"evidence_score": "m3_evidence_score_v3", "evidence": "m3_evidence_score_v3", "reliability": "m3_reliability_v2"},
    "m4": {"evidence_score": "m4_evidence_score_v3", "evidence": "m4_evidence_score_v3", "reliability": "m4_reliability_v2"},
}

def percentile_rank_full(series):
    values = pd.to_numeric(series, errors="coerce")
    valid = values.notna()
    result = pd.Series(np.nan, index=series.index, dtype="float64")
    if valid.sum() == 0:
        return result
    valid_values = values.loc[valid].to_numpy()
    sorted_values = np.sort(valid_values)
    ranks = (
        np.searchsorted(sorted_values, valid_values, side="right")
        / len(sorted_values)
    )
    result.loc[valid] = ranks
    return result

def main():
    if BASELINE_FILE.exists():
        print(f"\nLoading base dataset from:\n{BASELINE_FILE}")
        df = pd.read_parquet(BASELINE_FILE)
    else:
        # Search for raw evidence/reliability file
        matches = list(BASE_DIR.parent.rglob("evidence_scores_reliability_v2.parquet"))
        if not matches:
            raise FileNotFoundError("Could not find evidence_scores_reliability_v2.parquet or full_tender_risk_scores.parquet")
        input_path = matches[0]
        print(f"\nLoading raw evidence dataset from:\n{input_path}")
        df = pd.read_parquet(input_path)

    print(f"Loaded dataset: {len(df):,} records, {len(df.columns)} columns")

    # Ensure module percentile ranks exist
    for mod in ["m1", "m2", "m3", "m4"]:
        r_col = f"{mod}_rank"
        ev_col = MODULE_DEFS[mod]["evidence"]
        if r_col not in df.columns:
            print(f"Calculating full-dataset rank for {mod.upper()} ({ev_col})...")
            df[r_col] = percentile_rank_full(df[ev_col])

    # Calculate Soft-OR Fusion (alpha=0.5)
    print("\nCalculating Soft-OR Fusion (alpha=0.5)...")
    r_cols = ["m1_rank", "m2_rank", "m3_rank", "m4_rank"]
    rel_cols = ["m1_reliability_v2", "m2_reliability_v2", "m3_reliability_v2", "m4_reliability_v2"]

    num = pd.Series(0.0, index=df.index)
    den = pd.Series(0.0, index=df.index)
    max_contrib = pd.Series(np.nan, index=df.index)

    for r_col, rel_col in zip(r_cols, rel_cols):
        r = pd.to_numeric(df[r_col], errors="coerce")
        w = pd.to_numeric(df[rel_col], errors="coerce")
        valid = r.notna() & w.notna() & np.isfinite(r) & np.isfinite(w) & (w > 0)

        contrib = r.loc[valid] * w.loc[valid]
        num.loc[valid] += contrib
        den.loc[valid] += w.loc[valid]

        curr_max = max_contrib.loc[valid]
        new_max = np.where(curr_max.isna(), contrib, np.maximum(curr_max, contrib))
        max_contrib.loc[valid] = new_max

    valid_den = den > 0
    w_mean = pd.Series(np.nan, index=df.index)
    w_mean.loc[valid_den] = num.loc[valid_den] / den.loc[valid_den]

    soft_or_score = pd.Series(np.nan, index=df.index)
    soft_or_score.loc[valid_den] = 0.5 * max_contrib.loc[valid_den] + 0.5 * w_mean.loc[valid_den]

    # Preserve baseline score as separate column for audit/comparison
    if "final_risk_score" in df.columns:
        df["baseline_risk_score"] = df["final_risk_score"]
    else:
        df["baseline_risk_score"] = w_mean

    # Set new candidate final score
    df["soft_or_risk_score"] = soft_or_score
    df["final_risk_score"] = soft_or_score
    df["scoring_method"] = "Soft-OR Fusion (alpha=0.5)"

    # Investigation rank and percentile
    print("Computing investigation ranks and percentile queues...")
    df["investigation_rank"] = df["final_risk_score"].rank(ascending=False, method="first")
    df["risk_percentile"] = df["final_risk_score"].rank(pct=True, method="average")

    # Risk level categorization
    valid_scores = df["final_risk_score"].dropna()
    q70 = float(valid_scores.quantile(0.70))
    q90 = float(valid_scores.quantile(0.90))

    def classify_risk(val):
        if pd.isna(val):
            return "Unavailable"
        if val >= q90:
            return "High"
        if val >= q70:
            return "Medium"
        return "Low"

    df["risk_level"] = df["final_risk_score"].apply(classify_risk)
    df["available_module_count"] = df[r_cols].notna().sum(axis=1)

    # Save versioned Soft-OR parquet
    print(f"\nSaving versioned Soft-OR full-dataset scores to:\n{OUTPUT_FILE}")
    df.to_parquet(OUTPUT_FILE, index=False)

    metadata = {
        "scoring_method": "Soft-OR Evidence Fusion",
        "alpha": 0.5,
        "formula": "0.5 * max(rank_i * rel_i) + 0.5 * sum(rank_i * rel_i) / sum(rel_i)",
        "total_records": len(df),
        "usable_scores": int(df["final_risk_score"].notna().sum()),
        "unavailable_scores": int(df["final_risk_score"].isna().sum()),
        "threshold_q70": q70,
        "threshold_q90": q90,
        "risk_level_counts": df["risk_level"].value_counts().to_dict(),
        "baseline_preserved_file": str(BASELINE_FILE),
        "soft_or_file": str(OUTPUT_FILE),
    }

    metadata_path = DASHBOARD_DATA_DIR / "soft_or_scoring_metadata.json"
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print("\n" + "=" * 70)
    print("SUCCESS: Soft-OR Full Dataset Scoring Complete")
    print(f"Total Records: {len(df):,}")
    print(f"Usable Soft-OR Scores: {metadata['usable_scores']:,}")
    print(f"Unavailable Scores: {metadata['unavailable_scores']:,}")
    print(f"Risk Breakdown: {metadata['risk_level_counts']}")
    print(f"Metadata saved to: {metadata_path}")
    print("=" * 70)

if __name__ == "__main__":
    main()
