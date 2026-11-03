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


# ================================================================
# 1. PATHS
# ================================================================

SCRIPT_DIR = Path(__file__).resolve().parent

INPUT_FILE = SCRIPT_DIR / "labelled_fusion_scores.parquet"

OUTPUT_DIR = SCRIPT_DIR / "final_fusion_results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ================================================================
# 2. CONFIGURATION
# ================================================================

TEST_START_YEAR = 2016

TOP_K_VALUES = [1, 2, 5, 10, 20]

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


# ================================================================
# 3. HELPER FUNCTIONS
# ================================================================

def safe_numeric(series):
    """
    Convert a series to numeric values.
    Invalid values become NaN.
    """
    return pd.to_numeric(series, errors="coerce")


def fit_rank_reference(train_series):
    """
    Fit an empirical percentile-rank reference using TRAINING data only.

    The returned sorted array is later used to transform future/test
    observations without looking at the test distribution.

    NaN values are ignored.
    """
    values = pd.to_numeric(train_series, errors="coerce")
    values = values[np.isfinite(values)]

    if len(values) == 0:
        return np.array([], dtype=float)

    return np.sort(values.to_numpy(dtype=float))


def apply_rank_reference(series, reference):
    """
    Transform a series into percentile ranks using a reference
    distribution fitted ONLY on the training data.

    For a test value x:

        rank = proportion of training values <= x

    Therefore, the test distribution itself is never used to
    determine the transformation.

    Missing values remain NaN.
    """
    values = pd.to_numeric(series, errors="coerce")

    result = pd.Series(np.nan, index=series.index, dtype=float)

    if len(reference) == 0:
        return result

    valid = values.notna()

    if not valid.any():
        return result

    x = values.loc[valid].to_numpy(dtype=float)

    # Number of training observations <= each x
    positions = np.searchsorted(reference, x, side="right")

    # Convert to [0,1]
    ranks = positions / len(reference)

    result.loc[valid] = ranks

    return result


def weighted_rank_fusion(
    dataframe,
    rank_columns,
    reliability_columns,
):
    """
    Reliability-aware fusion of rank-normalized module evidence.

    For each row:

        F = sum(rank_i * reliability_i) /
            sum(reliability_i)

    Only modules with both a valid rank and valid reliability
    contribute.

    This means missing module evidence does not automatically
    become zero evidence.
    """

    numerator = pd.Series(0.0, index=dataframe.index)
    denominator = pd.Series(0.0, index=dataframe.index)

    for rank_col, rel_col in zip(rank_columns, reliability_columns):

        rank = pd.to_numeric(dataframe[rank_col], errors="coerce")
        reliability = pd.to_numeric(
            dataframe[rel_col],
            errors="coerce"
        )

        valid = (
            rank.notna()
            & reliability.notna()
            & np.isfinite(rank)
            & np.isfinite(reliability)
            & (reliability > 0)
        )

        numerator.loc[valid] += (
            rank.loc[valid] * reliability.loc[valid]
        )

        denominator.loc[valid] += reliability.loc[valid]

    fusion = numerator / denominator

    fusion.loc[denominator == 0] = np.nan

    return fusion


def equal_rank_fusion(dataframe, rank_columns):
    """
    Equal-weight rank fusion.

    Each available module receives equal weight.
    """

    values = dataframe[rank_columns].copy()

    values = values.apply(pd.to_numeric, errors="coerce")

    return values.mean(axis=1, skipna=True)


def calculate_metrics(y_true, scores, model_name):
    """
    Calculate ranking and threshold-free classification metrics.

    A fixed threshold is intentionally NOT used for the final
    screening system. Precision/recall/F1 here use 0.5 only as
    descriptive diagnostics and are not the operational decision rule.
    """

    valid = (
        pd.Series(y_true).notna()
        & pd.Series(scores).notna()
    )

    y = pd.Series(y_true).loc[valid].astype(int).to_numpy()
    s = pd.Series(scores).loc[valid].astype(float).to_numpy()

    if len(np.unique(y)) < 2:
        return {
            "model": model_name,
            "rows": len(y),
            "cartel": int(y.sum()),
            "non_cartel": int((y == 0).sum()),
            "roc_auc": np.nan,
            "average_precision": np.nan,
        }

    auc = roc_auc_score(y, s)
    ap = average_precision_score(y, s)

    # Descriptive threshold diagnostic only.
    pred = (s >= 0.5).astype(int)

    precision = precision_score(
        y,
        pred,
        zero_division=0
    )

    recall = recall_score(
        y,
        pred,
        zero_division=0
    )

    f1 = f1_score(
        y,
        pred,
        zero_division=0
    )

    tn, fp, fn, tp = confusion_matrix(
        y,
        pred,
        labels=[0, 1]
    ).ravel()

    specificity = (
        tn / (tn + fp)
        if (tn + fp) > 0
        else np.nan
    )

    return {
        "model": model_name,
        "rows": len(y),
        "cartel": int(y.sum()),
        "non_cartel": int((y == 0).sum()),
        "roc_auc": auc,
        "average_precision": ap,
        "precision_at_0.5": precision,
        "recall_at_0.5": recall,
        "specificity_at_0.5": specificity,
        "f1_at_0.5": f1,
        "tn_at_0.5": int(tn),
        "fp_at_0.5": int(fp),
        "fn_at_0.5": int(fn),
        "tp_at_0.5": int(tp),
    }


def calculate_top_k(
    dataframe,
    score_column,
    model_name,
    k_percent,
):
    """
    Evaluate forensic screening by investigating the top K% of
    available scored tenders.

    This is the primary operational evaluation because the project
    is intended to prioritize tenders for investigation rather than
    produce a universal probability threshold.
    """

    work = dataframe[
        ["row_id", "is_cartel", score_column]
    ].copy()

    work[score_column] = pd.to_numeric(
        work[score_column],
        errors="coerce"
    )

    work["is_cartel"] = pd.to_numeric(
        work["is_cartel"],
        errors="coerce"
    )

    work = work.dropna(
        subset=[score_column, "is_cartel"]
    ).copy()

    work["is_cartel"] = work["is_cartel"].astype(int)

    total_rows = len(work)

    if total_rows == 0:
        return None

    total_cartel = int(work["is_cartel"].sum())

    # Number of rows to investigate.
    selected_count = max(
        1,
        int(np.ceil(total_rows * k_percent / 100.0))
    )

    # Sort descending by risk score.
    work = work.sort_values(
        by=score_column,
        ascending=False,
        kind="mergesort"
    ).reset_index(drop=True)

    work["selected"] = False
    work.loc[
        :selected_count - 1,
        "selected"
    ] = True

    selected = work[work["selected"]]

    cartel_selected = int(
        selected["is_cartel"].sum()
    )

    non_cartel_selected = int(
        len(selected) - cartel_selected
    )

    cartel_not_selected = (
        total_cartel - cartel_selected
    )

    non_cartel_total = total_rows - total_cartel

    non_cartel_not_selected = (
        non_cartel_total - non_cartel_selected
    )

    precision = (
        cartel_selected / len(selected)
        if len(selected) > 0
        else 0.0
    )

    recall = (
        cartel_selected / total_cartel
        if total_cartel > 0
        else 0.0
    )

    specificity = (
        non_cartel_not_selected / non_cartel_total
        if non_cartel_total > 0
        else 0.0
    )

    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    random_recall = k_percent / 100.0

    lift = (
        recall / random_recall
        if random_recall > 0
        else np.nan
    )

    return {
        "model": model_name,
        "top_k_percent": k_percent,
        "total_available_rows": total_rows,
        "rows_selected": len(selected),
        "total_cartel": total_cartel,
        "cartel_selected": cartel_selected,
        "cartel_not_selected": cartel_not_selected,
        "non_cartel_selected": non_cartel_selected,
        "non_cartel_not_selected": non_cartel_not_selected,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "f1": f1,
        "random_recall": random_recall,
        "lift_over_random": lift,
        "recall_gain_percentage_points": (
            (recall - random_recall) * 100
        ),
    }


# ================================================================
# 4. LOAD DATA
# ================================================================

print("=" * 70)
print("FINAL LEAKAGE-SAFE FUSION EVALUATION")
print("=" * 70)

print()
print("Loading:")
print(INPUT_FILE)

if not INPUT_FILE.exists():
    raise FileNotFoundError(
        f"Input file not found:\n{INPUT_FILE}"
    )

df = pd.read_parquet(INPUT_FILE)

print(f"Dataset shape: {df.shape}")


# ================================================================
# 5. CHECK REQUIRED COLUMNS
# ================================================================

required_columns = [
    "row_id",
    "tender_year",
    "is_cartel",
] + EVIDENCE_COLUMNS + RELIABILITY_COLUMNS

missing_columns = [
    col
    for col in required_columns
    if col not in df.columns
]

if missing_columns:
    raise ValueError(
        "Missing required columns:\n"
        + "\n".join(missing_columns)
    )

print()
print("All required columns are present.")


# ================================================================
# 6. KEEP LABELLED DATA ONLY
# ================================================================

df["tender_year"] = safe_numeric(
    df["tender_year"]
)

df["is_cartel"] = safe_numeric(
    df["is_cartel"]
)

labelled = df[
    df["is_cartel"].isin([0, 1])
].copy()

labelled["is_cartel"] = labelled[
    "is_cartel"
].astype(int)

print()
print("=" * 70)
print("LABELLED DATA")
print("=" * 70)

print(f"Labelled rows: {len(labelled):,}")
print(
    f"Cartel rows: "
    f"{int(labelled['is_cartel'].sum()):,}"
)
print(
    f"Non-cartel rows: "
    f"{int((labelled['is_cartel'] == 0).sum()):,}"
)


# ================================================================
# 7. CHRONOLOGICAL TRAIN / TEST SPLIT
# ================================================================

train = labelled[
    labelled["tender_year"] < TEST_START_YEAR
].copy()

test = labelled[
    labelled["tender_year"] >= TEST_START_YEAR
].copy()

if len(train) == 0 or len(test) == 0:
    raise ValueError(
        "Chronological split produced an empty train or test set."
    )

if train["is_cartel"].nunique() < 2:
    raise ValueError(
        "Training set does not contain both classes."
    )

if test["is_cartel"].nunique() < 2:
    raise ValueError(
        "Test set does not contain both classes."
    )

print()
print("=" * 70)
print("CHRONOLOGICAL SPLIT")
print("=" * 70)

print(
    f"Training years: "
    f"{train['tender_year'].min()} - "
    f"{train['tender_year'].max()}"
)

print(
    f"Testing years:  "
    f"{test['tender_year'].min()} - "
    f"{test['tender_year'].max()}"
)

print(f"Training rows: {len(train):,}")
print(f"Testing rows:  {len(test):,}")

print()
print("Training class distribution:")
print(
    train["is_cartel"].value_counts()
    .sort_index()
    .rename(
        index={
            0: "Non-cartel",
            1: "Cartel"
        }
    )
)

print()
print("Testing class distribution:")
print(
    test["is_cartel"].value_counts()
    .sort_index()
    .rename(
        index={
            0: "Non-cartel",
            1: "Cartel"
        }
    )
)


# ================================================================
# 8. TRAIN-ONLY RANK TRANSFORMATION
# ================================================================

print()
print("=" * 70)
print("FITTING TRAIN-ONLY RANK REFERENCES")
print("=" * 70)

rank_columns = []

rank_reference_statistics = []

for module_name, evidence_column in zip(
    MODULE_NAMES,
    EVIDENCE_COLUMNS
):

    reference = fit_rank_reference(
        train[evidence_column]
    )

    if len(reference) == 0:
        raise ValueError(
            f"No valid training values for {evidence_column}"
        )

    train_rank_column = (
        f"{module_name.lower()}_rank"
    )

    test_rank_column = (
        f"{module_name.lower()}_rank"
    )

    train[train_rank_column] = apply_rank_reference(
        train[evidence_column],
        reference
    )

    test[test_rank_column] = apply_rank_reference(
        test[evidence_column],
        reference
    )

    rank_columns.append(
        train_rank_column
    )

    rank_reference_statistics.append({
        "module": module_name,
        "evidence_column": evidence_column,
        "training_reference_size": len(reference),
        "training_min": float(reference.min()),
        "training_max": float(reference.max()),
        "training_median": float(
            np.median(reference)
        ),
    })

    print(
        f"{module_name}: "
        f"{evidence_column} -> "
        f"{train_rank_column}"
    )

rank_reference_df = pd.DataFrame(
    rank_reference_statistics
)

rank_reference_df.to_csv(
    OUTPUT_DIR / "rank_reference_statistics.csv",
    index=False
)


# ================================================================
# 9. RELIABILITY-AWARE RANK FUSION
# ================================================================

print()
print("=" * 70)
print("CALCULATING FINAL FUSION SCORES")
print("=" * 70)

train["rank_reliability_fusion"] = (
    weighted_rank_fusion(
        train,
        rank_columns,
        RELIABILITY_COLUMNS
    )
)

test["rank_reliability_fusion"] = (
    weighted_rank_fusion(
        test,
        rank_columns,
        RELIABILITY_COLUMNS
    )
)


# Also calculate rank equal-weight fusion for comparison.

train["rank_equal_weight_fusion"] = (
    equal_rank_fusion(
        train,
        rank_columns
    )
)

test["rank_equal_weight_fusion"] = (
    equal_rank_fusion(
        test,
        rank_columns
    )
)


# Existing raw fusion scores are retained if available.

raw_equal_available = (
    "equal_weight_fusion" in labelled.columns
)

raw_reliability_available = (
    "reliability_aware_fusion" in labelled.columns
)

if raw_equal_available:
    test["raw_equal_weight_fusion"] = pd.to_numeric(
        test["equal_weight_fusion"],
        errors="coerce"
    )

if raw_reliability_available:
    test["raw_reliability_aware_fusion"] = pd.to_numeric(
        test["reliability_aware_fusion"],
        errors="coerce"
    )


# ================================================================
# 10. FUSION AVAILABILITY
# ================================================================

print()
print("Fusion availability:")

for name in [
    "rank_reliability_fusion",
    "rank_equal_weight_fusion",
]:

    available = test[name].notna().sum()

    print(
        f"{name}: "
        f"{available:,} / {len(test):,}"
    )

if raw_equal_available:
    print(
        "raw_equal_weight_fusion: "
        f"{test['raw_equal_weight_fusion'].notna().sum():,} / "
        f"{len(test):,}"
    )

if raw_reliability_available:
    print(
        "raw_reliability_aware_fusion: "
        f"{test['raw_reliability_aware_fusion'].notna().sum():,} / "
        f"{len(test):,}"
    )


# ================================================================
# 11. SAVE TEST SCORES
# ================================================================

test_output_columns = [
    "row_id",
    "tender_year",
    "is_cartel",
] + EVIDENCE_COLUMNS + RELIABILITY_COLUMNS + rank_columns + [
    "rank_equal_weight_fusion",
    "rank_reliability_fusion",
]

if raw_equal_available:
    test_output_columns.append(
        "raw_equal_weight_fusion"
    )

if raw_reliability_available:
    test_output_columns.append(
        "raw_reliability_aware_fusion"
    )

test_scores = test[
    test_output_columns
].copy()

test_scores.to_parquet(
    OUTPUT_DIR / "final_test_scores.parquet",
    index=False
)


# ================================================================
# 12. RANKING METRICS
# ================================================================

print()
print("=" * 70)
print("RANKING PERFORMANCE")
print("=" * 70)

evaluation_scores = [
    (
        "Rank Reliability-aware Fusion",
        "rank_reliability_fusion"
    ),
    (
        "Rank Equal-weight Fusion",
        "rank_equal_weight_fusion"
    ),
]

if raw_equal_available:
    evaluation_scores.append(
        (
            "Raw Equal-weight Fusion",
            "raw_equal_weight_fusion"
        )
    )

if raw_reliability_available:
    evaluation_scores.append(
        (
            "Raw Reliability-aware Fusion",
            "raw_reliability_aware_fusion"
        )
    )

ranking_results = []

for model_name, score_column in evaluation_scores:

    metrics = calculate_metrics(
        test["is_cartel"],
        test[score_column],
        model_name
    )

    ranking_results.append(metrics)

    print()
    print(model_name)
    print(
        f"  Rows: {metrics['rows']:,}"
    )
    print(
        f"  ROC-AUC: {metrics['roc_auc']:.4f}"
    )
    print(
        f"  Average Precision: "
        f"{metrics['average_precision']:.4f}"
    )

ranking_df = pd.DataFrame(
    ranking_results
)

ranking_df.to_csv(
    OUTPUT_DIR / "final_ranking_metrics.csv",
    index=False
)


# ================================================================
# 13. TOP-K FORENSIC SCREENING
# ================================================================

print()
print("=" * 70)
print("TOP-K FORENSIC SCREENING")
print("=" * 70)

top_k_results = []

for model_name, score_column in evaluation_scores:

    print()
    print(model_name)

    for k in TOP_K_VALUES:

        result = calculate_top_k(
            test,
            score_column,
            model_name,
            k
        )

        if result is None:
            continue

        top_k_results.append(result)

        print()
        print(
            f"Top {k}%:"
        )
        print(
            f"  Tenders investigated: "
            f"{result['rows_selected']:,}"
        )
        print(
            f"  Cartel tenders found: "
            f"{result['cartel_selected']:,}"
        )
        print(
            f"  Precision: "
            f"{result['precision']:.4f}"
        )
        print(
            f"  Recall: "
            f"{result['recall']:.4f}"
        )
        print(
            f"  Specificity: "
            f"{result['specificity']:.4f}"
        )
        print(
            f"  F1: "
            f"{result['f1']:.4f}"
        )
        print(
            f"  Lift over random: "
            f"{result['lift_over_random']:.2f}x"
        )

top_k_df = pd.DataFrame(
    top_k_results
)

top_k_df.to_csv(
    OUTPUT_DIR / "final_top_k_results.csv",
    index=False
)


# ================================================================
# 14. MAIN FINAL RESULT
# ================================================================

print()
print("=" * 70)
print("FINAL RECOMMENDED FUSION RESULT")
print("=" * 70)

final_model = ranking_df[
    ranking_df["model"]
    == "Rank Reliability-aware Fusion"
].iloc[0]

print(
    f"ROC-AUC: "
    f"{final_model['roc_auc']:.4f}"
)

print(
    f"Average Precision: "
    f"{final_model['average_precision']:.4f}"
)

final_top5 = top_k_df[
    (top_k_df["model"]
     == "Rank Reliability-aware Fusion")
    &
    (top_k_df["top_k_percent"] == 5)
]

if len(final_top5) > 0:

    final_top5 = final_top5.iloc[0]

    print()
    print("Top 5% screening:")
    print(
        f"  Available test tenders: "
        f"{final_top5['total_available_rows']:,}"
    )
    print(
        f"  Tenders investigated: "
        f"{final_top5['rows_selected']:,}"
    )
    print(
        f"  Cartel tenders found: "
        f"{final_top5['cartel_selected']:,}"
    )
    print(
        f"  Precision: "
        f"{final_top5['precision']:.4f}"
    )
    print(
        f"  Recall: "
        f"{final_top5['recall']:.4f}"
    )
    print(
        f"  Lift: "
        f"{final_top5['lift_over_random']:.2f}x"
    )


# ================================================================
# 15. SAVE FINAL SUMMARY
# ================================================================

summary = {
    "test_start_year": TEST_START_YEAR,
    "training_start_year": train["tender_year"].min(),
    "training_end_year": train["tender_year"].max(),
    "test_end_year": test["tender_year"].max(),
    "training_rows": len(train),
    "test_rows": len(test),
    "test_cartel_rows": int(
        test["is_cartel"].sum()
    ),
    "test_non_cartel_rows": int(
        (test["is_cartel"] == 0).sum()
    ),
    "final_model": (
        "Rank Reliability-aware Fusion"
    ),
    "final_auc": float(
        final_model["roc_auc"]
    ),
    "final_average_precision": float(
        final_model["average_precision"]
    ),
}

if len(final_top5) > 0:

    summary.update({
        "top_5_percent_rows_selected": int(
            final_top5["rows_selected"]
        ),
        "top_5_percent_cartel_selected": int(
            final_top5["cartel_selected"]
        ),
        "top_5_percent_precision": float(
            final_top5["precision"]
        ),
        "top_5_percent_recall": float(
            final_top5["recall"]
        ),
        "top_5_percent_lift": float(
            final_top5["lift_over_random"]
        ),
    })

summary_df = pd.DataFrame(
    [summary]
)

summary_df.to_csv(
    OUTPUT_DIR / "final_result_summary.csv",
    index=False
)


# ================================================================
# 16. FINISHED
# ================================================================

print()
print("=" * 70)
print("FILES SAVED")
print("=" * 70)

print(
    OUTPUT_DIR / "rank_reference_statistics.csv"
)

print(
    OUTPUT_DIR / "final_test_scores.parquet"
)

print(
    OUTPUT_DIR / "final_ranking_metrics.csv"
)

print(
    OUTPUT_DIR / "final_top_k_results.csv"
)

print(
    OUTPUT_DIR / "final_result_summary.csv"
)

print()
print("=" * 70)
print("FINAL EVALUATION COMPLETE")
print("=" * 70)