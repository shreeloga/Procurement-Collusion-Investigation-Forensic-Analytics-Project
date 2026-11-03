# ============================================================
# LEARNED NON-NEGATIVE MODULE-WEIGHT FUSION
# ============================================================
#
# Goal:
# Learn the optimal contribution of M1, M2, M3 and M4
# using ONLY the chronological training period.
#
# Constraint:
#     w1 >= 0
#     w2 >= 0
#     w3 >= 0
#     w4 >= 0
#     w1 + w2 + w3 + w4 = 1
#
# Validation:
#     Training: 2004-2015
#     Testing : 2016-2022
#
# Comparisons:
#     1. Equal-weight fusion
#     2. Existing reliability-aware fusion
#     3. Learned-weight fusion
#
# ============================================================

from pathlib import Path
import numpy as np
import pandas as pd

from scipy.optimize import minimize

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)


# ============================================================
# 1. PATHS
# ============================================================

SCRIPT_DIR = Path(__file__).resolve().parent

INPUT_FILE = (
    SCRIPT_DIR /
    "labelled_fusion_scores.parquet"
)

OUTPUT_DIR = (
    SCRIPT_DIR /
    "learned_weight_fusion_results"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# 2. CONFIGURATION
# ============================================================

RANDOM_STATE = 42

TEST_START_YEAR = 2016

TARGET_COL = "is_cartel"
YEAR_COL = "tender_year"

EVIDENCE_COLS = [
    "m1_evidence_score_v3",
    "m2_evidence_score_v4",
    "m3_evidence_score_v3",
    "m4_evidence_score_v3"
]

FUSION_COLS = [
    "equal_weight_fusion",
    "reliability_aware_fusion"
]


# ============================================================
# 3. LOAD DATA
# ============================================================

print("=" * 70)
print("LEARNED NON-NEGATIVE MODULE-WEIGHT FUSION")
print("=" * 70)

print("\nLoading:")
print(INPUT_FILE)

df = pd.read_parquet(INPUT_FILE)

print("\nDataset shape:", df.shape)


# ============================================================
# 4. CHECK COLUMNS
# ============================================================

required_columns = (
    [TARGET_COL, YEAR_COL]
    + EVIDENCE_COLS
    + FUSION_COLS
)

missing_columns = [
    col
    for col in required_columns
    if col not in df.columns
]

if missing_columns:
    raise ValueError(
        "\nMissing required columns:\n"
        + "\n".join(missing_columns)
    )

print("\nAll required columns are present.")


# ============================================================
# 5. KEEP LABELLED DATA
# ============================================================

df = df[
    df[TARGET_COL].notna()
].copy()

df[TARGET_COL] = (
    df[TARGET_COL]
    .astype(int)
)

df[YEAR_COL] = pd.to_numeric(
    df[YEAR_COL],
    errors="coerce"
)

df = df[
    df[YEAR_COL].notna()
].copy()

print("\nLabelled rows:", len(df))

print("\nLabel distribution:")
print(
    df[TARGET_COL]
    .value_counts()
    .sort_index()
)


# ============================================================
# 6. CONVERT EVIDENCE SCORES TO NUMERIC
# ============================================================

for col in EVIDENCE_COLS:
    df[col] = pd.to_numeric(
        df[col],
        errors="coerce"
    )


# ============================================================
# 7. CHRONOLOGICAL SPLIT
# ============================================================

train_mask = (
    df[YEAR_COL] < TEST_START_YEAR
)

test_mask = (
    df[YEAR_COL] >= TEST_START_YEAR
)

train_df = df.loc[
    train_mask
].copy()

test_df = df.loc[
    test_mask
].copy()


print("\n" + "=" * 70)
print("CHRONOLOGICAL SPLIT")
print("=" * 70)

print(
    f"\nTraining years: "
    f"{train_df[YEAR_COL].min():.0f}"
    f" - "
    f"{train_df[YEAR_COL].max():.0f}"
)

print(
    f"Testing years: "
    f"{test_df[YEAR_COL].min():.0f}"
    f" - "
    f"{test_df[YEAR_COL].max():.0f}"
)

print("\nTraining rows:", len(train_df))
print("Testing rows :", len(test_df))

print("\nTraining class distribution:")
print(
    train_df[TARGET_COL]
    .value_counts()
    .sort_index()
)

print("\nTesting class distribution:")
print(
    test_df[TARGET_COL]
    .value_counts()
    .sort_index()
)


# ============================================================
# 8. PREPARE EVIDENCE MATRICES
# ============================================================

X_train = train_df[
    EVIDENCE_COLS
].copy()

X_test = test_df[
    EVIDENCE_COLS
].copy()

y_train = train_df[
    TARGET_COL
].values

y_test = test_df[
    TARGET_COL
].values


# ============================================================
# 9. IMPORTANT:
#    MISSING EVIDENCE IS NOT THE SAME AS ZERO EVIDENCE
# ============================================================
#
# For learning the module weights, we calculate each module's
# contribution only where that module actually has evidence.
#
# The final learned fusion is:
#
#   F = sum(w_i * E_i) / sum(w_i for available modules)
#
# This prevents missing M1 from automatically behaving like
# zero-risk evidence.
#
# ============================================================


def weighted_fusion(
    evidence_df,
    weights
):

    values = evidence_df.values.astype(float)

    available = ~np.isnan(values)

    weighted_values = (
        np.nan_to_num(values, nan=0.0)
        * weights
    )

    numerator = weighted_values.sum(
        axis=1
    )

    denominator = (
        available * weights
    ).sum(axis=1)

    scores = np.divide(
        numerator,
        denominator,
        out=np.zeros_like(numerator),
        where=denominator > 0
    )

    return scores


# ============================================================
# 10. OPTIMIZATION OBJECTIVE
# ============================================================
#
# We optimize ROC-AUC on TRAINING data.
#
# IMPORTANT:
# The test period is never used while learning weights.
#
# ============================================================


def objective(weights):

    scores = weighted_fusion(
        X_train,
        weights
    )

    valid = np.isfinite(scores)

    if valid.sum() < 2:
        return 1.0

    try:

        auc = roc_auc_score(
            y_train[valid],
            scores[valid]
        )

    except ValueError:

        return 1.0

    # scipy minimizes, so minimize negative AUC.
    return -auc


# ============================================================
# 11. INITIAL WEIGHTS
# ============================================================

initial_weights = np.array([
    0.25,
    0.25,
    0.25,
    0.25
])


# ============================================================
# 12. CONSTRAINTS
# ============================================================

constraints = [
    {
        "type": "eq",
        "fun": lambda w: np.sum(w) - 1.0
    }
]

bounds = [
    (0.0, 1.0),
    (0.0, 1.0),
    (0.0, 1.0),
    (0.0, 1.0)
]


# ============================================================
# 13. LEARN WEIGHTS
# ============================================================

print("\n" + "=" * 70)
print("LEARNING MODULE WEIGHTS")
print("=" * 70)

print("\nInitial weights:")

for module, weight in zip(
    ["M1", "M2", "M3", "M4"],
    initial_weights
):
    print(
        f"  {module}: {weight:.4f}"
    )


result = minimize(
    objective,
    initial_weights,
    method="SLSQP",
    bounds=bounds,
    constraints=constraints,
    options={
        "maxiter": 1000,
        "ftol": 1e-10,
        "disp": True
    }
)


# ============================================================
# 14. CHECK OPTIMIZATION
# ============================================================

if not result.success:

    print(
        "\nWARNING: Weight optimization did not "
        "fully converge."
    )

    print(
        "Optimizer message:",
        result.message
    )

learned_weights = result.x

# Numerical cleanup.
learned_weights[
    learned_weights < 0
] = 0

learned_weights = (
    learned_weights /
    learned_weights.sum()
)


print("\nLearned weights:")

for module, weight in zip(
    ["M1", "M2", "M3", "M4"],
    learned_weights
):

    print(
        f"  {module}: {weight:.6f}"
    )

print(
    "\nWeight sum:",
    learned_weights.sum()
)


# ============================================================
# 15. TRAINING PERFORMANCE
# ============================================================

train_learned_scores = weighted_fusion(
    X_train,
    learned_weights
)

train_valid = np.isfinite(
    train_learned_scores
)

train_auc = roc_auc_score(
    y_train[train_valid],
    train_learned_scores[train_valid]
)

train_ap = average_precision_score(
    y_train[train_valid],
    train_learned_scores[train_valid]
)

print("\nTraining performance:")
print(
    f"  ROC-AUC: {train_auc:.4f}"
)

print(
    f"  AP     : {train_ap:.4f}"
)


# ============================================================
# 16. TEST PERFORMANCE
# ============================================================

test_learned_scores = weighted_fusion(
    X_test,
    learned_weights
)

test_valid = np.isfinite(
    test_learned_scores
)

test_y_valid = y_test[
    test_valid
]

test_scores_valid = test_learned_scores[
    test_valid
]

test_auc = roc_auc_score(
    test_y_valid,
    test_scores_valid
)

test_ap = average_precision_score(
    test_y_valid,
    test_scores_valid
)


# ============================================================
# 17. THRESHOLD METRICS
# ============================================================

# Use 0.5 only for a simple classification summary.
# AUC and AP are the main ranking metrics.

threshold = 0.5

test_predictions = (
    test_scores_valid >= threshold
).astype(int)

precision = precision_score(
    test_y_valid,
    test_predictions,
    zero_division=0
)

recall = recall_score(
    test_y_valid,
    test_predictions,
    zero_division=0
)

f1 = f1_score(
    test_y_valid,
    test_predictions,
    zero_division=0
)

tn, fp, fn, tp = confusion_matrix(
    test_y_valid,
    test_predictions,
    labels=[0, 1]
).ravel()

specificity = (
    tn / (tn + fp)
    if (tn + fp) > 0
    else 0
)


print("\n" + "=" * 70)
print("LEARNED-WEIGHT TEST PERFORMANCE")
print("=" * 70)

print(
    f"\nValid test rows: {test_valid.sum()}"
)

print(
    f"ROC-AUC      : {test_auc:.4f}"
)

print(
    f"Average Prec.: {test_ap:.4f}"
)

print(
    f"Precision    : {precision:.4f}"
)

print(
    f"Recall       : {recall:.4f}"
)

print(
    f"Specificity  : {specificity:.4f}"
)

print(
    f"F1           : {f1:.4f}"
)

print("\nConfusion matrix:")
print(
    np.array([
        [tn, fp],
        [fn, tp]
    ])
)


# ============================================================
# 18. COMPARE WITH EXISTING FUSIONS
# ============================================================

print("\n" + "=" * 70)
print("COMPARISON WITH EXISTING FUSIONS")
print("=" * 70)


def evaluate_existing_fusion(
    name,
    score_column
):

    scores = pd.to_numeric(
        test_df[score_column],
        errors="coerce"
    ).values

    valid = np.isfinite(scores)

    valid_y = y_test[
        valid
    ]

    valid_scores = scores[
        valid
    ]

    auc = roc_auc_score(
        valid_y,
        valid_scores
    )

    ap = average_precision_score(
        valid_y,
        valid_scores
    )

    print(
        f"\n{name}"
    )

    print(
        f"  Rows: {valid.sum()}"
    )

    print(
        f"  ROC-AUC: {auc:.4f}"
    )

    print(
        f"  AP     : {ap:.4f}"
    )

    return {
        "model": name,
        "rows": int(valid.sum()),
        "roc_auc": auc,
        "average_precision": ap
    }


equal_results = evaluate_existing_fusion(
    "Equal-Weight Fusion",
    "equal_weight_fusion"
)

ra_results = evaluate_existing_fusion(
    "Reliability-Aware Fusion",
    "reliability_aware_fusion"
)

learned_results = {
    "model": "Learned-Weight Fusion",
    "rows": int(test_valid.sum()),
    "roc_auc": test_auc,
    "average_precision": test_ap,
    "precision": precision,
    "recall": recall,
    "specificity": specificity,
    "f1": f1,
    "tn": tn,
    "fp": fp,
    "fn": fn,
    "tp": tp
}


# ============================================================
# 19. PERFORMANCE DELTAS
# ============================================================

print("\n" + "=" * 70)
print("PERFORMANCE DELTAS")
print("=" * 70)

print(
    "\nLearned vs Equal-weight:"
)

print(
    f"  ΔAUC: "
    f"{test_auc - equal_results['roc_auc']:+.4f}"
)

print(
    f"  ΔAP : "
    f"{test_ap - equal_results['average_precision']:+.4f}"
)

print(
    "\nLearned vs Reliability-aware:"
)

print(
    f"  ΔAUC: "
    f"{test_auc - ra_results['roc_auc']:+.4f}"
)

print(
    f"  ΔAP : "
    f"{test_ap - ra_results['average_precision']:+.4f}"
)


# ============================================================
# 20. SAVE LEARNED WEIGHTS
# ============================================================

weights_df = pd.DataFrame({
    "module": [
        "M1",
        "M2",
        "M3",
        "M4"
    ],
    "learned_weight": learned_weights
})

weights_df.to_csv(
    OUTPUT_DIR /
    "learned_module_weights.csv",
    index=False
)


# ============================================================
# 21. SAVE METRICS
# ============================================================

metrics_df = pd.DataFrame([
    equal_results,
    ra_results,
    learned_results
])

metrics_df.to_csv(
    OUTPUT_DIR /
    "learned_weight_fusion_metrics.csv",
    index=False
)


# ============================================================
# 22. SAVE TEST PREDICTIONS
# ============================================================

prediction_df = pd.DataFrame({
    "row_id": (
        test_df["row_id"].values
        if "row_id" in test_df.columns
        else np.arange(len(test_df))
    ),

    "tender_year": (
        test_df[YEAR_COL].values
    ),

    "is_cartel": y_test,

    "equal_weight_fusion": (
        test_df[
            "equal_weight_fusion"
        ].values
    ),

    "reliability_aware_fusion": (
        test_df[
            "reliability_aware_fusion"
        ].values
    ),

    "learned_weight_fusion": (
        test_learned_scores
    )
})

prediction_df.to_csv(
    OUTPUT_DIR /
    "learned_weight_fusion_test_predictions.csv",
    index=False
)


# ============================================================
# 23. SAVE TRAINING/TEST SUMMARY
# ============================================================

summary_df = pd.DataFrame([
    {
        "dataset": "training",
        "rows": len(train_df),
        "min_year": train_df[YEAR_COL].min(),
        "max_year": train_df[YEAR_COL].max(),
        "cartel": int(y_train.sum()),
        "non_cartel": int(
            (y_train == 0).sum()
        )
    },
    {
        "dataset": "testing",
        "rows": len(test_df),
        "min_year": test_df[YEAR_COL].min(),
        "max_year": test_df[YEAR_COL].max(),
        "cartel": int(y_test.sum()),
        "non_cartel": int(
            (y_test == 0).sum()
        )
    }
])

summary_df.to_csv(
    OUTPUT_DIR /
    "learned_weight_split_summary.csv",
    index=False
)


# ============================================================
# 24. FINAL OUTPUT
# ============================================================

print("\n" + "=" * 70)
print("FINAL SUMMARY")
print("=" * 70)

print(
    "\nLearned module weights:"
)

for module, weight in zip(
    ["M1", "M2", "M3", "M4"],
    learned_weights
):

    print(
        f"  {module}: {weight:.4f}"
    )

print("\nPerformance:")

print(
    metrics_df[
        [
            "model",
            "rows",
            "roc_auc",
            "average_precision"
        ]
    ].to_string(
        index=False
    )
)

print("\nOutput directory:")
print(OUTPUT_DIR)

print("\nFiles created:")

for file in sorted(
    OUTPUT_DIR.iterdir()
):
    print(
        "  ",
        file.name
    )

print("\nDone.")