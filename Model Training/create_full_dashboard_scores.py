import pandas as pd
import numpy as np
from pathlib import Path


# ============================================================
# PROJECT PATH
# ============================================================

BASE_DIR = Path(__file__).resolve().parent


# ============================================================
# FIND RELIABILITY FILE AUTOMATICALLY
# ============================================================

print("=" * 70)
print("CREATING FULL-DATASET DASHBOARD SCORES")
print("=" * 70)

print("\nSearching for:")
print("evidence_scores_reliability_v2.parquet")


matches = list(
    BASE_DIR.parent.rglob(
        "evidence_scores_reliability_v2.parquet"
    )
)


if not matches:

    raise FileNotFoundError(
        "\nCould not find evidence_scores_reliability_v2.parquet "
        "inside the Project folder.\n\n"
        "Please check that the reliability pipeline was run successfully."
    )


print("\nFound file:")

for i, path in enumerate(matches, 1):
    print(f"{i}. {path}")


# If multiple copies exist, prefer the one inside
# the current Model Training project if available.

preferred = [
    p for p in matches
    if "Model Training" in str(p)
]


if preferred:
    INPUT_FILE = preferred[0]
else:
    INPUT_FILE = matches[0]


print("\nUsing:")
print(INPUT_FILE)


# ============================================================
# OUTPUT
# ============================================================

OUTPUT_DIR = BASE_DIR / "dashboard_data"

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

OUTPUT_FILE = (
    OUTPUT_DIR
    / "full_tender_risk_scores.parquet"
)


# ============================================================
# LOAD DATA
# ============================================================

print("\nLoading full evidence/reliability dataset...")

df = pd.read_parquet(INPUT_FILE)

print(
    f"Dataset shape: {df.shape}"
)


# ============================================================
# REQUIRED COLUMNS
# ============================================================

required_columns = [

    "row_id",

    # M1-M4 evidence
    "m1_evidence_score_v3",
    "m2_evidence_score_v4",
    "m3_evidence_score_v3",
    "m4_evidence_score_v3",

    # M1-M4 reliability
    "m1_reliability_v2",
    "m2_reliability_v2",
    "m3_reliability_v2",
    "m4_reliability_v2",
]


missing = [
    c for c in required_columns
    if c not in df.columns
]


if missing:

    raise ValueError(
        "\nMissing required columns:\n"
        + "\n".join(missing)
    )


# ============================================================
# MODULE DEFINITIONS
# ============================================================

modules = {

    "m1": {
        "evidence": "m1_evidence_score_v3",
        "reliability": "m1_reliability_v2"
    },

    "m2": {
        "evidence": "m2_evidence_score_v4",
        "reliability": "m2_reliability_v2"
    },

    "m3": {
        "evidence": "m3_evidence_score_v3",
        "reliability": "m3_reliability_v2"
    },

    "m4": {
        "evidence": "m4_evidence_score_v3",
        "reliability": "m4_reliability_v2"
    }
}


# ============================================================
# FULL-DATASET PERCENTILE RANK
# ============================================================

def percentile_rank_full(series):

    values = pd.to_numeric(
        series,
        errors="coerce"
    )

    valid = values.notna()

    result = pd.Series(
        np.nan,
        index=series.index,
        dtype="float64"
    )

    if valid.sum() == 0:
        return result

    valid_values = (
        values.loc[valid]
        .to_numpy()
    )

    sorted_values = np.sort(
        valid_values
    )

    ranks = (
        np.searchsorted(
            sorted_values,
            valid_values,
            side="right"
        )
        / len(sorted_values)
    )

    result.loc[valid] = ranks

    return result


# ============================================================
# MODULE RANKS
# ============================================================

print("\nCalculating module ranks...")

for module, cols in modules.items():

    evidence_col = cols["evidence"]

    rank_col = f"{module}_rank"

    df[rank_col] = percentile_rank_full(
        df[evidence_col]
    )

    available = int(
        df[rank_col].notna().sum()
    )

    print(
        f"{module.upper()} available: "
        f"{available:,} / {len(df):,}"
    )


# ============================================================
# RELIABILITY-AWARE RANK FUSION
# ============================================================

print("\nCalculating reliability-aware rank fusion...")


weighted_sum = pd.Series(
    0.0,
    index=df.index
)

weight_sum = pd.Series(
    0.0,
    index=df.index
)


for module, cols in modules.items():

    rank = pd.to_numeric(
        df[f"{module}_rank"],
        errors="coerce"
    )

    reliability = pd.to_numeric(
        df[cols["reliability"]],
        errors="coerce"
    )

    valid = (
        rank.notna()
        & reliability.notna()
        & (reliability > 0)
    )

    weighted_sum.loc[valid] += (
        rank.loc[valid]
        * reliability.loc[valid]
    )

    weight_sum.loc[valid] += (
        reliability.loc[valid]
    )


df["final_risk_score"] = (
    weighted_sum
    / weight_sum.replace(0, np.nan)
)


# ============================================================
# EQUAL RANK FUSION
# ============================================================

rank_columns = [
    "m1_rank",
    "m2_rank",
    "m3_rank",
    "m4_rank"
]


df["rank_equal_fusion"] = (
    df[rank_columns]
    .mean(
        axis=1,
        skipna=True
    )
)


# ============================================================
# INVESTIGATION RANK
# ============================================================

df["investigation_rank"] = (
    df["final_risk_score"]
    .rank(
        ascending=False,
        method="first"
    )
)


# ============================================================
# RISK PERCENTILE
# ============================================================

df["risk_percentile"] = (
    df["final_risk_score"]
    .rank(
        pct=True,
        method="average"
    )
)


# ============================================================
# RISK LEVEL
# ============================================================

score = df["final_risk_score"]

q70 = score.quantile(0.70)
q90 = score.quantile(0.90)


def classify_risk(value):

    if pd.isna(value):
        return "Unavailable"

    if value >= q90:
        return "High"

    if value >= q70:
        return "Medium"

    return "Low"


df["risk_level"] = (
    df["final_risk_score"]
    .apply(classify_risk)
)


# ============================================================
# MODULE AVAILABILITY COUNT
# ============================================================

df["available_module_count"] = (
    df[rank_columns]
    .notna()
    .sum(axis=1)
)


# ============================================================
# DASHBOARD COLUMNS
# ============================================================

dashboard_columns = [

    # Identification
    "row_id",

    # Procurement information
    "country",
    "tender_year",
    "buyer_id",
    "bidder_id",

    # CPV
    "main_cpv_2",
    "main_cpv_3",
    "main_cpv_3x",
    "main_cpv_4",

    # Competition
    "lot_bidscount",
    "singleb",
    "bid_iswinning",
    "bid_isconsortium",
    "bid_issubcontracted",

    # Prices
    "bid_price",
    "lot_estimatedprice",
    "tender_estimatedprice",
    "tender_finalprice",

    # Dates
    "tender_publications_firstcallfortenderdate",
    "tender_publications_firstdcontractawarddate",

    # Evidence
    "m1_evidence_score_v3",
    "m2_evidence_score_v4",
    "m3_evidence_score_v3",
    "m4_evidence_score_v3",

    # Reliability
    "m1_reliability_v2",
    "m2_reliability_v2",
    "m3_reliability_v2",
    "m4_reliability_v2",

    # Rank
    "m1_rank",
    "m2_rank",
    "m3_rank",
    "m4_rank",

    # Final
    "rank_equal_fusion",
    "final_risk_score",
    "investigation_rank",
    "risk_percentile",
    "risk_level",
    "available_module_count",

    # Internal evaluation only
    "is_cartel",
    "cartel_id"
]


dashboard_columns = [
    c
    for c in dashboard_columns
    if c in df.columns
]


dashboard_df = df[
    dashboard_columns
].copy()


# ============================================================
# SAVE
# ============================================================

print("\nSaving dashboard data...")

dashboard_df.to_parquet(
    OUTPUT_FILE,
    index=False
)


# ============================================================
# SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("SUCCESS")
print("=" * 70)

print(
    f"\nOutput file:\n{OUTPUT_FILE}"
)

print(
    f"\nTotal records: "
    f"{len(dashboard_df):,}"
)

print(
    f"Usable final risk scores: "
    f"{dashboard_df['final_risk_score'].notna().sum():,}"
)

print(
    f"Unavailable scores: "
    f"{dashboard_df['final_risk_score'].isna().sum():,}"
)

print("\nRisk distribution:")

print(
    dashboard_df["risk_level"]
    .value_counts(
        dropna=False
    )
)

print("\nModule availability:")

for module in ["m1", "m2", "m3", "m4"]:

    count = dashboard_df[
        f"{module}_rank"
    ].notna().sum()

    print(
        f"{module.upper()}: "
        f"{count:,} / {len(dashboard_df):,}"
    )

print("\nDashboard data creation complete.")