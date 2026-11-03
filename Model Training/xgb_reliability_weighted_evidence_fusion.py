from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

from xgboost import XGBClassifier


# ============================================================
# 1. PATHS
# ============================================================

SCRIPT_DIR = Path(__file__).resolve().parent

INPUT_PATH = SCRIPT_DIR / "labelled_fusion_scores.parquet"

OUTPUT_DIR = (
    SCRIPT_DIR /
    "strict_xgb_reliability_weighted_evidence_results"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


print("=" * 70)
print("XGBOOST RELIABILITY-WEIGHTED EVIDENCE FUSION")
print("=" * 70)

print(f"Input : {INPUT_PATH}")
print(f"Output: {OUTPUT_DIR}")


# ============================================================
# 2. LOAD DATA
# ============================================================

df = pd.read_parquet(INPUT_PATH)

print("\nDataset shape:", df.shape)


# ============================================================
# 3. REQUIRED COLUMNS
# ============================================================

required_columns = [
    "row_id",
    "tender_year",
    "is_cartel",

    "m1_evidence_score_v3",
    "m1_reliability_v2",

    "m2_evidence_score_v4",
    "m2_reliability_v2",

    "m3_evidence_score_v3",
    "m3_reliability_v2",

    "m4_evidence_score_v3",
    "m4_reliability_v2",
]

missing_columns = [
    col for col in required_columns
    if col not in df.columns
]

if missing_columns:
    raise ValueError(
        "Missing required columns:\n"
        + "\n".join(missing_columns)
    )

print("All required columns are present.")


# ============================================================
# 4. KEEP LABELLED ROWS ONLY
# ============================================================

df = df[df["is_cartel"].notna()].copy()

df["is_cartel"] = df["is_cartel"].astype(int)

print("\nLabelled rows:", len(df))

print("\nClass distribution:")
print(
    df["is_cartel"]
    .value_counts()
    .sort_index()
)

print("\nCartel prevalence:")
print(
    f"{df['is_cartel'].mean():.6f}"
)


# ============================================================
# 5. SAME 2016 CHRONOLOGICAL SPLIT
# ============================================================

TEST_START_YEAR = 2016

train = df[
    df["tender_year"] < TEST_START_YEAR
].copy()

test = df[
    df["tender_year"] >= TEST_START_YEAR
].copy()


print("\n" + "=" * 70)
print("CHRONOLOGICAL SPLIT")
print("=" * 70)

print(
    f"Test start year: {TEST_START_YEAR}"
)

print(
    f"\nTraining rows: {len(train):,}"
)

print(
    f"Testing rows : {len(test):,}"
)

print(
    f"\nTraining years: "
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
# 6. CREATE RELIABILITY-WEIGHTED EVIDENCE FEATURES
#
# These are exactly:
#
#     Evidence × Reliability
#
# We deliberately DO NOT provide the raw reliability values
# to XGBoost.
# ============================================================

train["m1_reliability_weighted_evidence"] = (
    pd.to_numeric(
        train["m1_evidence_score_v3"],
        errors="coerce"
    )
    *
    pd.to_numeric(
        train["m1_reliability_v2"],
        errors="coerce"
    )
)

train["m2_reliability_weighted_evidence"] = (
    pd.to_numeric(
        train["m2_evidence_score_v4"],
        errors="coerce"
    )
    *
    pd.to_numeric(
        train["m2_reliability_v2"],
        errors="coerce"
    )
)

train["m3_reliability_weighted_evidence"] = (
    pd.to_numeric(
        train["m3_evidence_score_v3"],
        errors="coerce"
    )
    *
    pd.to_numeric(
        train["m3_reliability_v2"],
        errors="coerce"
    )
)

train["m4_reliability_weighted_evidence"] = (
    pd.to_numeric(
        train["m4_evidence_score_v3"],
        errors="coerce"
    )
    *
    pd.to_numeric(
        train["m4_reliability_v2"],
        errors="coerce"
    )
)


test["m1_reliability_weighted_evidence"] = (
    pd.to_numeric(
        test["m1_evidence_score_v3"],
        errors="coerce"
    )
    *
    pd.to_numeric(
        test["m1_reliability_v2"],
        errors="coerce"
    )
)

test["m2_reliability_weighted_evidence"] = (
    pd.to_numeric(
        test["m2_evidence_score_v4"],
        errors="coerce"
    )
    *
    pd.to_numeric(
        test["m2_reliability_v2"],
        errors="coerce"
    )
)

test["m3_reliability_weighted_evidence"] = (
    pd.to_numeric(
        test["m3_evidence_score_v3"],
        errors="coerce"
    )
    *
    pd.to_numeric(
        test["m3_reliability_v2"],
        errors="coerce"
    )
)

test["m4_reliability_weighted_evidence"] = (
    pd.to_numeric(
        test["m4_evidence_score_v3"],
        errors="coerce"
    )
    *
    pd.to_numeric(
        test["m4_reliability_v2"],
        errors="coerce"
    )
)


# ============================================================
# 7. DEFINE THE FOUR FEATURES
# ============================================================

feature_columns = [
    "m1_reliability_weighted_evidence",
    "m2_reliability_weighted_evidence",
    "m3_reliability_weighted_evidence",
    "m4_reliability_weighted_evidence",
]


print("\n" + "=" * 70)
print("FEATURES USED BY XGBOOST")
print("=" * 70)

for col in feature_columns:
    print("  ", col)


# ============================================================
# 8. PREPARE X AND y
# ============================================================

X_train = train[feature_columns].copy()

X_test = test[feature_columns].copy()

y_train = train["is_cartel"].astype(int)

y_test = test["is_cartel"].astype(int)


# ============================================================
# 9. FORCE NUMERIC
# ============================================================

for col in feature_columns:

    X_train[col] = pd.to_numeric(
        X_train[col],
        errors="coerce"
    )

    X_test[col] = pd.to_numeric(
        X_test[col],
        errors="coerce"
    )


# ============================================================
# 10. MISSING VALUE CHECK
# ============================================================

print("\n" + "=" * 70)
print("FEATURE INFORMATION")
print("=" * 70)

print(
    "Number of features:",
    len(feature_columns)
)

print(
    "Missing training values:",
    int(X_train.isna().sum().sum())
)

print(
    "Missing testing values:",
    int(X_test.isna().sum().sum())
)


# ============================================================
# 11. TRAINING CLASS WEIGHT
# ============================================================

negative_count = int(
    (y_train == 0).sum()
)

positive_count = int(
    (y_train == 1).sum()
)

scale_pos_weight = (
    negative_count /
    positive_count
)

print(
    "\nTraining non-cartel:",
    negative_count
)

print(
    "Training cartel    :",
    positive_count
)

print(
    "scale_pos_weight   :",
    scale_pos_weight
)


# ============================================================
# 12. XGBOOST MODEL
#
# Only FOUR features.
#
# The model learns nonlinear interactions between:
#
#   M1 weighted evidence
#   M2 weighted evidence
#   M3 weighted evidence
#   M4 weighted evidence
# ============================================================

model = XGBClassifier(
    n_estimators=300,
    max_depth=3,
    learning_rate=0.05,

    subsample=0.8,
    colsample_bytree=1.0,

    objective="binary:logistic",
    eval_metric="logloss",

    scale_pos_weight=scale_pos_weight,

    random_state=42,
    n_jobs=-1,
    tree_method="hist"
)


print("\n" + "=" * 70)
print("TRAINING XGBOOST")
print("=" * 70)

model.fit(
    X_train,
    y_train
)

print("Training completed.")


# ============================================================
# 13. TEST PREDICTIONS
# ============================================================

test_probability = (
    model.predict_proba(X_test)[:, 1]
)

test_prediction = (
    test_probability >= 0.5
).astype(int)


# ============================================================
# 14. METRICS
# ============================================================

roc_auc = roc_auc_score(
    y_test,
    test_probability
)

average_precision = average_precision_score(
    y_test,
    test_probability
)

precision = precision_score(
    y_test,
    test_prediction,
    zero_division=0
)

recall = recall_score(
    y_test,
    test_prediction,
    zero_division=0
)

f1 = f1_score(
    y_test,
    test_prediction,
    zero_division=0
)


# ============================================================
# 15. CONFUSION MATRIX
# ============================================================

tn, fp, fn, tp = confusion_matrix(
    y_test,
    test_prediction,
    labels=[0, 1]
).ravel()


specificity = (
    tn / (tn + fp)
    if (tn + fp) > 0
    else 0.0
)


# ============================================================
# 16. PRINT RESULTS
# ============================================================

print("\n" + "=" * 70)
print("RESULTS")
print("=" * 70)

print(
    f"Test rows        : {len(y_test):,}"
)

print(
    f"Cartel rows      : {int(y_test.sum()):,}"
)

print(
    f"Non-cartel rows  : {int((y_test == 0).sum()):,}"
)

print(
    f"\nROC-AUC          : {roc_auc:.4f}"
)

print(
    f"Average Precision: {average_precision:.4f}"
)

print(
    f"Precision        : {precision:.4f}"
)

print(
    f"Recall           : {recall:.4f}"
)

print(
    f"Specificity      : {specificity:.4f}"
)

print(
    f"F1               : {f1:.4f}"
)

print("\nConfusion Matrix:")

print(
    pd.DataFrame(
        [[tn, fp], [fn, tp]],
        index=[
            "Actual Non-cartel",
            "Actual Cartel"
        ],
        columns=[
            "Predicted Non-cartel",
            "Predicted Cartel"
        ]
    )
)


# ============================================================
# 17. SAVE METRICS
# ============================================================

metrics = pd.DataFrame([{

    "model":
        "XGBoost_Reliability_Weighted_Evidence",

    "test_start_year":
        TEST_START_YEAR,

    "train_rows":
        len(train),

    "test_rows":
        len(test),

    "train_min_year":
        train["tender_year"].min(),

    "train_max_year":
        train["tender_year"].max(),

    "test_min_year":
        test["tender_year"].min(),

    "test_max_year":
        test["tender_year"].max(),

    "train_cartel":
        int(y_train.sum()),

    "train_non_cartel":
        int((y_train == 0).sum()),

    "test_cartel":
        int(y_test.sum()),

    "test_non_cartel":
        int((y_test == 0).sum()),

    "n_features":
        len(feature_columns),

    "roc_auc":
        roc_auc,

    "average_precision":
        average_precision,

    "precision":
        precision,

    "recall":
        recall,

    "specificity":
        specificity,

    "f1":
        f1,

    "tn":
        tn,

    "fp":
        fp,

    "fn":
        fn,

    "tp":
        tp,

    "scale_pos_weight":
        scale_pos_weight

}])


metrics.to_csv(
    OUTPUT_DIR /
    "xgb_reliability_weighted_evidence_metrics.csv",
    index=False
)


# ============================================================
# 18. SAVE CONFUSION MATRIX
# ============================================================

confusion_df = pd.DataFrame(
    [[tn, fp], [fn, tp]],
    index=[
        "Actual Non-cartel",
        "Actual Cartel"
    ],
    columns=[
        "Predicted Non-cartel",
        "Predicted Cartel"
    ]
)

confusion_df.to_csv(
    OUTPUT_DIR /
    "xgb_reliability_weighted_evidence_confusion_matrix.csv"
)


# ============================================================
# 19. SAVE TEST PREDICTIONS
# ============================================================

predictions = test[
    [
        "row_id",
        "tender_year",
        "is_cartel"
    ]
].copy()

predictions[
    "xgb_reliability_weighted_evidence_score"
] = test_probability

predictions[
    "xgb_reliability_weighted_evidence_prediction"
] = test_prediction


# Save the four inputs used by XGBoost

for col in feature_columns:
    predictions[col] = test[col].values


predictions.to_csv(
    OUTPUT_DIR /
    "xgb_reliability_weighted_evidence_test_predictions.csv",
    index=False
)


# ============================================================
# 20. FEATURE IMPORTANCE
# ============================================================

importance = pd.DataFrame({

    "feature":
        feature_columns,

    "importance":
        model.feature_importances_

})


importance = importance.sort_values(
    "importance",
    ascending=False
).reset_index(drop=True)


importance.to_csv(
    OUTPUT_DIR /
    "xgb_reliability_weighted_evidence_feature_importance.csv",
    index=False
)


# ============================================================
# 21. PRINT FEATURE IMPORTANCE
# ============================================================

print("\n" + "=" * 70)
print("FEATURE IMPORTANCE")
print("=" * 70)

print(
    importance.to_string(index=False)
)


# ============================================================
# 22. SAVE MODEL
# ============================================================

model.save_model(
    OUTPUT_DIR /
    "xgb_reliability_weighted_evidence_model.json"
)


# ============================================================
# 23. FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("COMPLETED")
print("=" * 70)

print(
    f"ROC-AUC           : {roc_auc:.4f}"
)

print(
    f"Average Precision : {average_precision:.4f}"
)

print(
    f"Precision         : {precision:.4f}"
)

print(
    f"Recall            : {recall:.4f}"
)

print(
    f"Specificity       : {specificity:.4f}"
)

print(
    f"F1                : {f1:.4f}"
)

print("\nResults saved to:")
print(OUTPUT_DIR)
