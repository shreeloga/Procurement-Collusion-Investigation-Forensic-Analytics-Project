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

INPUT_FILE = (
    SCRIPT_DIR
    / "rank_based_fusion_results"
    / "rank_based_test_scores.parquet"
)

ORIGINAL_FILE = (
    SCRIPT_DIR
    / "labelled_fusion_scores.parquet"
)

OUTPUT_DIR = (
    SCRIPT_DIR
    / "rank_threshold_results"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# 2. LOAD DATA
# ============================================================

print("=" * 70)
print("THRESHOLD OPTIMIZATION")
print("RANK RELIABILITY-AWARE FUSION")
print("=" * 70)

print("\nLoading rank-based test scores:")
print(INPUT_FILE)

rank_test = pd.read_parquet(
    INPUT_FILE
)

print(
    f"Rank test file shape: "
    f"{rank_test.shape}"
)


# We also need the training-period rank scores.
# The previous rank-based script saved only the test scores,
# so reconstruct the ranks from the original labelled file.

print("\nLoading original labelled fusion file:")
print(ORIGINAL_FILE)

df = pd.read_parquet(
    ORIGINAL_FILE
)

df = df[
    df["is_cartel"].notna()
].copy()

df["is_cartel"] = (
    df["is_cartel"]
    .astype(int)
)

print(
    f"Labelled rows: "
    f"{len(df):,}"
)


# ============================================================
# 3. CHRONOLOGICAL SPLIT
# ============================================================

TRAIN_END_YEAR = 2016

train = df[
    df["tender_year"] < TRAIN_END_YEAR
].copy()

test = df[
    df["tender_year"] >= TRAIN_END_YEAR
].copy()

print("\nChronological split:")
print(
    f"Training rows: "
    f"{len(train):,}"
)

print(
    f"Testing rows : "
    f"{len(test):,}"
)

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


# ============================================================
# 4. MODULE DEFINITIONS
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
# 5. PERCENTILE RANK FUNCTION
# ============================================================

def percentile_rank(series):

    result = pd.Series(
        np.nan,
        index=series.index,
        dtype=float
    )

    valid = series.notna()

    if valid.sum() == 0:
        return result

    values = series.loc[valid]

    result.loc[valid] = (
        values.rank(
            method="average",
            pct=True
        )
    )

    return result


# ============================================================
# 6. RECREATE MODULE RANKS
# ============================================================

print("\n" + "=" * 70)
print("RECREATING TRAINING RANKS")
print("=" * 70)

for module, column in evidence_columns.items():

    rank_column = (
        f"{module}_rank"
    )

    train[rank_column] = (
        percentile_rank(
            pd.to_numeric(
                train[column],
                errors="coerce"
            )
        )
    )

    print(
        f"{module.upper()}: "
        f"{train[rank_column].notna().sum():,} "
        f"available training rows"
    )


# ============================================================
# 7. RANK RELIABILITY-AWARE FUSION
# ============================================================

def rank_reliability_fusion(data):

    numerator = np.zeros(
        len(data),
        dtype=float
    )

    denominator = np.zeros(
        len(data),
        dtype=float
    )

    for module in [
        "m1",
        "m2",
        "m3",
        "m4"
    ]:

        rank = pd.to_numeric(
            data[
                f"{module}_rank"
            ],
            errors="coerce"
        )

        reliability = pd.to_numeric(
            data[
                reliability_columns[module]
            ],
            errors="coerce"
        )

        available = rank.notna()

        rank = rank.fillna(0.0)

        reliability = (
            reliability
            .fillna(0.0)
        )

        numerator += (
            rank.to_numpy()
            * reliability.to_numpy()
            * available.to_numpy()
        )

        denominator += (
            reliability.to_numpy()
            * available.to_numpy()
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


train[
    "rank_reliability_fusion"
] = rank_reliability_fusion(
    train
)


# ============================================================
# 8. ALIGN TEST SCORE FROM PREVIOUS EXPERIMENT
# ============================================================

# The previous script already calculated the test rank
# reliability-aware score.
#
# Align by row_id so that we use exactly the same test scores.

test_score_lookup = (
    rank_test[
        [
            "row_id",
            "is_cartel",
            "rank_reliability_fusion",
        ]
    ]
    .drop_duplicates(
        subset=["row_id"]
    )
)

test = test.merge(
    test_score_lookup,
    on="row_id",
    how="left",
    suffixes=(
        "",
        "_from_rank_file"
    )
)


# ============================================================
# 9. VERIFY LABEL CONSISTENCY
# ============================================================

if "is_cartel_from_rank_file" in test.columns:

    mismatch = (
        test["is_cartel"]
        != test[
            "is_cartel_from_rank_file"
        ]
    ).sum()

    print(
        f"\nLabel mismatches: "
        f"{mismatch}"
    )

    if mismatch != 0:
        raise ValueError(
            "Label mismatch detected."
        )


# ============================================================
# 10. THRESHOLD METRIC FUNCTION
# ============================================================

def calculate_threshold_metrics(
    y_true,
    y_score,
    threshold
):

    y_pred = (
        y_score >= threshold
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

    tn, fp, fn, tp = (
        confusion_matrix(
            y_true,
            y_pred,
            labels=[0, 1]
        ).ravel()
    )

    specificity = (
        tn / (tn + fp)
        if (tn + fp) > 0
        else 0.0
    )

    return {
        "threshold": threshold,
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
# 11. PREPARE TRAINING DATA
# ============================================================

train_valid = train[
    train[
        "rank_reliability_fusion"
    ].notna()
].copy()

y_train = (
    train_valid[
        "is_cartel"
    ].astype(int)
)

score_train = (
    train_valid[
        "rank_reliability_fusion"
    ].astype(float)
)

print("\nTraining fusion-score rows:")
print(
    f"{len(train_valid):,}"
)


# ============================================================
# 12. THRESHOLD GRID
# ============================================================

thresholds = np.arange(
    0.01,
    1.001,
    0.01
)

training_threshold_results = []

for threshold in thresholds:

    metrics = (
        calculate_threshold_metrics(
            y_train,
            score_train,
            threshold
        )
    )

    training_threshold_results.append(
        metrics
    )


training_threshold_df = pd.DataFrame(
    training_threshold_results
)


# ============================================================
# 13. SELECT THRESHOLD USING TRAINING DATA
# ============================================================

# Primary criterion:
# maximize F1 on the TRAINING period only.
#
# Tie-break:
# choose the threshold with the highest precision.

training_threshold_df = (
    training_threshold_df
    .sort_values(
        by=[
            "f1",
            "precision"
        ],
        ascending=[
            False,
            False
        ]
    )
    .reset_index(drop=True)
)

selected_threshold = float(
    training_threshold_df
    .iloc[0]["threshold"]
)

selected_train_metrics = (
    training_threshold_df
    .iloc[0]
)


print("\n" + "=" * 70)
print("SELECTED THRESHOLD")
print("=" * 70)

print(
    f"Selected using TRAINING data only: "
    f"{selected_threshold:.2f}"
)

print(
    f"Training precision: "
    f"{selected_train_metrics['precision']:.4f}"
)

print(
    f"Training recall: "
    f"{selected_train_metrics['recall']:.4f}"
)

print(
    f"Training specificity: "
    f"{selected_train_metrics['specificity']:.4f}"
)

print(
    f"Training F1: "
    f"{selected_train_metrics['f1']:.4f}"
)


# ============================================================
# 14. APPLY SELECTED THRESHOLD TO TEST
# ============================================================

test_valid = test[
    test[
        "rank_reliability_fusion"
    ].notna()
].copy()

y_test = (
    test_valid[
        "is_cartel"
    ].astype(int)
)

score_test = (
    test_valid[
        "rank_reliability_fusion"
    ].astype(float)
)

test_selected_metrics = (
    calculate_threshold_metrics(
        y_test,
        score_test,
        selected_threshold
    )
)


# ============================================================
# 15. TEST AUC / AP
# ============================================================

test_auc = roc_auc_score(
    y_test,
    score_test
)

test_ap = average_precision_score(
    y_test,
    score_test
)


print("\n" + "=" * 70)
print("TEST PERFORMANCE USING TRAINING-SELECTED THRESHOLD")
print("=" * 70)

print(
    f"Test rows: "
    f"{len(test_valid):,}"
)

print(
    f"ROC-AUC: "
    f"{test_auc:.4f}"
)

print(
    f"Average Precision: "
    f"{test_ap:.4f}"
)

print(
    f"Threshold: "
    f"{selected_threshold:.2f}"
)

print(
    f"Precision: "
    f"{test_selected_metrics['precision']:.4f}"
)

print(
    f"Recall: "
    f"{test_selected_metrics['recall']:.4f}"
)

print(
    f"Specificity: "
    f"{test_selected_metrics['specificity']:.4f}"
)

print(
    f"F1: "
    f"{test_selected_metrics['f1']:.4f}"
)

print(
    f"TN: "
    f"{test_selected_metrics['tn']}"
)

print(
    f"FP: "
    f"{test_selected_metrics['fp']}"
)

print(
    f"FN: "
    f"{test_selected_metrics['fn']}"
)

print(
    f"TP: "
    f"{test_selected_metrics['tp']}"
)


# ============================================================
# 16. TEST-ONLY BEST THRESHOLD
# ============================================================

# This is ONLY a diagnostic.
#
# We do NOT use this threshold as the final threshold because
# that would use test labels to optimize the decision rule.

test_threshold_results = []

for threshold in thresholds:

    metrics = (
        calculate_threshold_metrics(
            y_test,
            score_test,
            threshold
        )
    )

    test_threshold_results.append(
        metrics
    )

test_threshold_df = pd.DataFrame(
    test_threshold_results
)

test_threshold_sorted = (
    test_threshold_df
    .sort_values(
        by=[
            "f1",
            "precision"
        ],
        ascending=[
            False,
            False
        ]
    )
    .reset_index(drop=True)
)

best_test_threshold = float(
    test_threshold_sorted
    .iloc[0]["threshold"]
)

best_test_metrics = (
    test_threshold_sorted
    .iloc[0]
)


print("\n" + "=" * 70)
print("TEST-ONLY DIAGNOSTIC")
print("=" * 70)

print(
    "IMPORTANT: This threshold is NOT used as the final "
    "model threshold."
)

print(
    f"Best test-only threshold: "
    f"{best_test_threshold:.2f}"
)

print(
    f"Test-only precision: "
    f"{best_test_metrics['precision']:.4f}"
)

print(
    f"Test-only recall: "
    f"{best_test_metrics['recall']:.4f}"
)

print(
    f"Test-only specificity: "
    f"{best_test_metrics['specificity']:.4f}"
)

print(
    f"Test-only F1: "
    f"{best_test_metrics['f1']:.4f}"
)


# ============================================================
# 17. SAVE TRAINING THRESHOLD SEARCH
# ============================================================

training_threshold_file = (
    OUTPUT_DIR
    / "training_threshold_search.csv"
)

training_threshold_df.to_csv(
    training_threshold_file,
    index=False
)


# ============================================================
# 18. SAVE TEST THRESHOLD SEARCH
# ============================================================

test_threshold_file = (
    OUTPUT_DIR
    / "test_threshold_diagnostic.csv"
)

test_threshold_df.to_csv(
    test_threshold_file,
    index=False
)


# ============================================================
# 19. SAVE FINAL METRICS
# ============================================================

final_metrics = pd.DataFrame([
    {
        "model":
            "Rank Reliability-aware Fusion",

        "threshold_selection":
            "Training 2004-2015",

        "threshold":
            selected_threshold,

        "test_rows":
            len(test_valid),

        "test_cartel_rows":
            int(y_test.sum()),

        "test_non_cartel_rows":
            int((y_test == 0).sum()),

        "roc_auc":
            test_auc,

        "average_precision":
            test_ap,

        "precision":
            test_selected_metrics[
                "precision"
            ],

        "recall":
            test_selected_metrics[
                "recall"
            ],

        "specificity":
            test_selected_metrics[
                "specificity"
            ],

        "f1":
            test_selected_metrics[
                "f1"
            ],

        "tn":
            test_selected_metrics[
                "tn"
            ],

        "fp":
            test_selected_metrics[
                "fp"
            ],

        "fn":
            test_selected_metrics[
                "fn"
            ],

        "tp":
            test_selected_metrics[
                "tp"
            ],
    }
])


final_metrics_file = (
    OUTPUT_DIR
    / "rank_reliability_threshold_metrics.csv"
)

final_metrics.to_csv(
    final_metrics_file,
    index=False
)


# ============================================================
# 20. SAVE TEST PREDICTIONS
# ============================================================

test_valid[
    "rank_reliability_prediction"
] = (
    score_test
    >= selected_threshold
).astype(int)

test_valid[
    "rank_reliability_threshold"
] = (
    selected_threshold
)

prediction_file = (
    OUTPUT_DIR
    / "rank_reliability_threshold_predictions.parquet"
)

prediction_columns = [
    "row_id",
    "tender_year",
    "is_cartel",
    "rank_reliability_fusion",
    "rank_reliability_threshold",
    "rank_reliability_prediction",
]

test_valid[
    prediction_columns
].to_parquet(
    prediction_file,
    index=False
)


# ============================================================
# 21. FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("FILES SAVED")
print("=" * 70)

print(training_threshold_file)
print(test_threshold_file)
print(final_metrics_file)
print(prediction_file)

print("\n" + "=" * 70)
print("FINAL RESULT")
print("=" * 70)

print(
    f"\nTraining-selected threshold: "
    f"{selected_threshold:.2f}"
)

print(
    f"Test ROC-AUC: "
    f"{test_auc:.4f}"
)

print(
    f"Test AP: "
    f"{test_ap:.4f}"
)

print(
    f"Test precision: "
    f"{test_selected_metrics['precision']:.4f}"
)

print(
    f"Test recall: "
    f"{test_selected_metrics['recall']:.4f}"
)

print(
    f"Test specificity: "
    f"{test_selected_metrics['specificity']:.4f}"
)

print(
    f"Test F1: "
    f"{test_selected_metrics['f1']:.4f}"
)

print("\nDone.")