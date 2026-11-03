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


# ============================================================
# 1. PATHS
# ============================================================

SCRIPT_DIR = Path(__file__).resolve().parent

INPUT_FILE = SCRIPT_DIR / "labelled_fusion_scores.parquet"

OUTPUT_DIR = SCRIPT_DIR / "rank_based_fusion_results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 2. LOAD DATA
# ============================================================

print("=" * 70)
print("RANK-BASED EVIDENCE FUSION")
print("=" * 70)

print("\nLoading:")
print(INPUT_FILE)

df = pd.read_parquet(INPUT_FILE)

print(f"Dataset shape: {df.shape}")


# ============================================================
# 3. REQUIRED COLUMNS
# ============================================================

required_columns = [
    "row_id",
    "tender_year",
    "is_cartel",

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

    # Existing fusion methods
    "equal_weight_fusion",
    "reliability_aware_fusion",
]

missing = [
    c for c in required_columns
    if c not in df.columns
]

if missing:
    raise ValueError(
        "Missing required columns:\n"
        + "\n".join(missing)
    )

print("\nAll required columns are present.")


# ============================================================
# 4. LABELLED DATA ONLY
# ============================================================

df = df[
    df["is_cartel"].notna()
].copy()

df["is_cartel"] = df["is_cartel"].astype(int)

print(f"\nLabelled rows: {len(df):,}")


# ============================================================
# 5. MODULE DEFINITIONS
# ============================================================

evidence_columns = {
    "m1": "m1_evidence_score_v3",
    "m2": "m2_evidence_score_v4",
    "m3": "m3_evidence_score_v3",
    "m4": "m4_evidence_score_v3",
}

reliability_columns = {
    "m1": "m1_reliability_v2",
    "m2": "m2_reliability_v2",
    "m3": "m3_reliability_v2",
    "m4": "m4_reliability_v2",
}


# ============================================================
# 6. CHRONOLOGICAL SPLIT
# ============================================================

TRAIN_END_YEAR = 2016

train = df[
    df["tender_year"] < TRAIN_END_YEAR
].copy()

test = df[
    df["tender_year"] >= TRAIN_END_YEAR
].copy()

print("\nChronological split:")
print(f"Training rows: {len(train):,}")
print(f"Testing rows : {len(test):,}")

print(
    f"Training years: "
    f"{train['tender_year'].min()} - "
    f"{train['tender_year'].max()}"
)

print(
    f"Testing years : "
    f"{test['tender_year'].min()} - "
    f"{test['tender_year'].max()}"
)

print("\nTraining class distribution:")
print(
    train["is_cartel"]
    .value_counts()
    .sort_index()
)

print("\nTesting class distribution:")
print(
    test["is_cartel"]
    .value_counts()
    .sort_index()
)


# ============================================================
# 7. RANK / PERCENTILE TRANSFORMATION
# ============================================================

def percentile_rank(series):
    """
    Convert evidence scores to percentile ranks.

    Lowest score  -> approximately 0
    Highest score -> 1

    Missing values remain missing.

    Ranking is performed separately for each module.
    """

    result = pd.Series(
        np.nan,
        index=series.index,
        dtype=float
    )

    valid = series.notna()

    if valid.sum() == 0:
        return result

    values = series.loc[valid]

    ranks = values.rank(
        method="average",
        pct=True
    )

    result.loc[valid] = ranks

    return result


# ============================================================
# 8. CREATE RANK SCORES
# ============================================================

print("\n" + "=" * 70)
print("CREATING MODULE PERCENTILE RANKS")
print("=" * 70)

for module, column in evidence_columns.items():

    rank_column = f"{module}_rank"

    # IMPORTANT:
    # Rank is calculated separately inside each chronological
    # period. This avoids using future test observations to
    # define the training score scale.

    train[rank_column] = percentile_rank(
        pd.to_numeric(
            train[column],
            errors="coerce"
        )
    )

    test[rank_column] = percentile_rank(
        pd.to_numeric(
            test[column],
            errors="coerce"
        )
    )

    train_available = train[rank_column].notna().sum()
    test_available = test[rank_column].notna().sum()

    print(
        f"{module.upper()}: "
        f"train available = {train_available:,}, "
        f"test available = {test_available:,}"
    )


# ============================================================
# 9. RANK-BASED FUSION
# ============================================================

def rank_equal_fusion(data):
    """
    Equal-weight percentile-rank fusion.

    Missing modules are ignored.

        F =
        sum(rank_i)
        -------------
        number available
    """

    rank_columns = [
        "m1_rank",
        "m2_rank",
        "m3_rank",
        "m4_rank",
    ]

    rank_matrix = data[
        rank_columns
    ]

    score = rank_matrix.mean(
        axis=1,
        skipna=True
    )

    return score


def rank_reliability_fusion(data):
    """
    Reliability-weighted rank fusion.

        F =
        sum(rank_i * reliability_i)
        ----------------------------
        sum(reliability_i)

    Missing evidence contributes zero.
    """

    numerator = np.zeros(
        len(data),
        dtype=float
    )

    denominator = np.zeros(
        len(data),
        dtype=float
    )

    for module in ["m1", "m2", "m3", "m4"]:

        rank = pd.to_numeric(
            data[f"{module}_rank"],
            errors="coerce"
        )

        reliability = pd.to_numeric(
            data[
                reliability_columns[module]
            ],
            errors="coerce"
        )

        # Evidence is available only when the rank exists.
        available = rank.notna()

        # Missing reliability is treated as zero.
        reliability = reliability.fillna(0.0)

        rank = rank.fillna(0.0)

        weight = (
            reliability
            .to_numpy()
        )

        available_array = (
            available
            .to_numpy()
        )

        numerator += (
            rank.to_numpy()
            * weight
            * available_array
        )

        denominator += (
            weight
            * available_array
        )

    score = np.full(
        len(data),
        np.nan,
        dtype=float
    )

    valid = denominator > 0

    score[valid] = (
        numerator[valid]
        / denominator[valid]
    )

    return pd.Series(
        score,
        index=data.index
    )


# ============================================================
# 10. CALCULATE RANK FUSION
# ============================================================

train["rank_equal_fusion"] = (
    rank_equal_fusion(train)
)

test["rank_equal_fusion"] = (
    rank_equal_fusion(test)
)

train["rank_reliability_fusion"] = (
    rank_reliability_fusion(train)
)

test["rank_reliability_fusion"] = (
    rank_reliability_fusion(test)
)


# ============================================================
# 11. EVALUATION FUNCTION
# ============================================================

def evaluate_model(
    data,
    score_column,
    model_name
):

    subset = data[
        data[score_column].notna()
    ].copy()

    y_true = subset[
        "is_cartel"
    ].astype(int)

    y_score = subset[
        score_column
    ].astype(float)

    auc = roc_auc_score(
        y_true,
        y_score
    )

    ap = average_precision_score(
        y_true,
        y_score
    )

    # Use 0.5 because percentile-rank fusion
    # is naturally on a 0-1 scale.
    y_pred = (
        y_score >= 0.5
    ).astype(int)

    precision = precision_score(
        y_true,
        y_pred,
        zero_division=0
    )

    recall = recall_score(
        y_true,
        y_pred,
        zero_division=0
    )

    f1 = f1_score(
        y_true,
        y_pred,
        zero_division=0
    )

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1]
    ).ravel()

    specificity = (
        tn / (tn + fp)
        if (tn + fp) > 0
        else 0.0
    )

    return {
        "model": model_name,
        "rows": len(subset),
        "cartel_rows": int(y_true.sum()),
        "non_cartel_rows": int(
            (y_true == 0).sum()
        ),
        "roc_auc": auc,
        "average_precision": ap,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "f1": f1,
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


# ============================================================
# 12. EVALUATE TEST SET
# ============================================================

print("\n" + "=" * 70)
print("TEST RESULTS")
print("=" * 70)

results = []

# New rank-based methods
results.append(
    evaluate_model(
        test,
        "rank_equal_fusion",
        "Rank Equal-weight Fusion"
    )
)

results.append(
    evaluate_model(
        test,
        "rank_reliability_fusion",
        "Rank Reliability-aware Fusion"
    )
)

# Existing methods
results.append(
    evaluate_model(
        test,
        "equal_weight_fusion",
        "Raw Equal-weight Fusion"
    )
)

results.append(
    evaluate_model(
        test,
        "reliability_aware_fusion",
        "Raw Reliability-aware Fusion"
    )
)

results_df = pd.DataFrame(results)

display_columns = [
    "model",
    "rows",
    "roc_auc",
    "average_precision",
    "precision",
    "recall",
    "specificity",
    "f1",
]

print(
    results_df[
        display_columns
    ].to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)


# ============================================================
# 13. DELTA COMPARISONS
# ============================================================

rank_equal = results_df[
    results_df["model"]
    == "Rank Equal-weight Fusion"
].iloc[0]

rank_reliability = results_df[
    results_df["model"]
    == "Rank Reliability-aware Fusion"
].iloc[0]

raw_equal = results_df[
    results_df["model"]
    == "Raw Equal-weight Fusion"
].iloc[0]

raw_reliability = results_df[
    results_df["model"]
    == "Raw Reliability-aware Fusion"
].iloc[0]


delta_rows = [

    {
        "comparison":
            "Rank Equal vs Raw Equal",

        "delta_auc":
            rank_equal["roc_auc"]
            - raw_equal["roc_auc"],

        "delta_ap":
            rank_equal["average_precision"]
            - raw_equal["average_precision"],
    },

    {
        "comparison":
            "Rank Reliability vs Raw Reliability",

        "delta_auc":
            rank_reliability["roc_auc"]
            - raw_reliability["roc_auc"],

        "delta_ap":
            rank_reliability["average_precision"]
            - raw_reliability["average_precision"],
    },

    {
        "comparison":
            "Rank Reliability vs Rank Equal",

        "delta_auc":
            rank_reliability["roc_auc"]
            - rank_equal["roc_auc"],

        "delta_ap":
            rank_reliability["average_precision"]
            - rank_equal["average_precision"],
    },
]

deltas_df = pd.DataFrame(
    delta_rows
)


print("\n" + "=" * 70)
print("PERFORMANCE DELTAS")
print("=" * 70)

print(
    deltas_df.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)


# ============================================================
# 14. MODULE RANK STATISTICS
# ============================================================

rank_statistics = []

for module in ["m1", "m2", "m3", "m4"]:

    column = f"{module}_rank"

    rank_statistics.append({
        "module": module.upper(),

        "train_available":
            int(train[column].notna().sum()),

        "train_mean_rank":
            train[column].mean(),

        "test_available":
            int(test[column].notna().sum()),

        "test_mean_rank":
            test[column].mean(),

        "test_median_rank":
            test[column].median(),
    })

rank_statistics_df = pd.DataFrame(
    rank_statistics
)


# ============================================================
# 15. SAVE RESULTS
# ============================================================

results_file = (
    OUTPUT_DIR /
    "rank_based_fusion_metrics.csv"
)

deltas_file = (
    OUTPUT_DIR /
    "rank_based_fusion_deltas.csv"
)

rank_stats_file = (
    OUTPUT_DIR /
    "rank_statistics.csv"
)

test_scores_file = (
    OUTPUT_DIR /
    "rank_based_test_scores.parquet"
)

results_df.to_csv(
    results_file,
    index=False
)

deltas_df.to_csv(
    deltas_file,
    index=False
)

rank_statistics_df.to_csv(
    rank_stats_file,
    index=False
)


test_output_columns = [
    "row_id",
    "tender_year",
    "is_cartel",

    "m1_evidence_score_v3",
    "m2_evidence_score_v4",
    "m3_evidence_score_v3",
    "m4_evidence_score_v3",

    "m1_rank",
    "m2_rank",
    "m3_rank",
    "m4_rank",

    "rank_equal_fusion",
    "rank_reliability_fusion",

    "equal_weight_fusion",
    "reliability_aware_fusion",
]

test[
    test_output_columns
].to_parquet(
    test_scores_file,
    index=False
)


# ============================================================
# 16. FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("FILES SAVED")
print("=" * 70)

print(results_file)
print(deltas_file)
print(rank_stats_file)
print(test_scores_file)


print("\n" + "=" * 70)
print("FINAL COMPARISON")
print("=" * 70)

print(
    f"\nRank Equal-weight AUC: "
    f"{rank_equal['roc_auc']:.4f}"
)

print(
    f"Rank Reliability-aware AUC: "
    f"{rank_reliability['roc_auc']:.4f}"
)

print(
    f"Raw Equal-weight AUC: "
    f"{raw_equal['roc_auc']:.4f}"
)

print(
    f"Raw Reliability-aware AUC: "
    f"{raw_reliability['roc_auc']:.4f}"
)

print(
    f"\nRank Equal AP: "
    f"{rank_equal['average_precision']:.4f}"
)

print(
    f"Rank Reliability-aware AP: "
    f"{rank_reliability['average_precision']:.4f}"
)

print(
    f"Raw Equal AP: "
    f"{raw_equal['average_precision']:.4f}"
)

print(
    f"Raw Reliability-aware AP: "
    f"{raw_reliability['average_precision']:.4f}"
)

print("\nDone.")