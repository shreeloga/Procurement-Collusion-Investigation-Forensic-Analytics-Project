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

OUTPUT_DIR = SCRIPT_DIR / "strict_xgb_evidence_fusion_results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 70)
print("XGBOOST RELIABILITY-AWARE EVIDENCE FUSION")
print("=" * 70)

print(f"Input : {INPUT_PATH}")
print(f"Output: {OUTPUT_DIR}")


# ============================================================
# 2. LOAD DATA
# ============================================================

df = pd.read_parquet(INPUT_PATH)

print("\nDataset shape:", df.shape)

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

missing = [c for c in required_columns if c not in df.columns]

if missing:
    raise ValueError(
        "The following required columns are missing:\n"
        + "\n".join(missing)
    )

print("All required columns are present.")


# ============================================================
# 3. KEEP LABELLED DATA ONLY
# ============================================================

df = df[df["is_cartel"].notna()].copy()

df["is_cartel"] = df["is_cartel"].astype(int)

print("\nLabelled rows:", len(df))

print("\nClass distribution:")
print(df["is_cartel"].value_counts().sort_index())

print("\nCartel prevalence:")
print(df["is_cartel"].mean())


# ============================================================
# 4. SAME CHRONOLOGICAL SPLIT
#
# Training: years before 2016
# Testing : 2016 onwards
# ============================================================

TEST_START_YEAR = 2016

train = df[df["tender_year"] < TEST_START_YEAR].copy()
test = df[df["tender_year"] >= TEST_START_YEAR].copy()

print("\n" + "=" * 70)
print("CHRONOLOGICAL SPLIT")
print("=" * 70)

print(f"Test start year: {TEST_START_YEAR}")

print(f"\nTraining rows: {len(train):,}")
print(f"Testing rows : {len(test):,}")

print(
    f"\nTraining years: "
    f"{train['tender_year'].min()} - {train['tender_year'].max()}"
)

print(
    f"Testing years : "
    f"{test['tender_year'].min()} - {test['tender_year'].max()}"
)

print("\nTraining class distribution:")
print(train["is_cartel"].value_counts().sort_index())

print("\nTesting class distribution:")
print(test["is_cartel"].value_counts().sort_index())


# ============================================================
# 5. DEFINE EVIDENCE + RELIABILITY FEATURES
# ============================================================

evidence_columns = [
    "m1_evidence_score_v3",
    "m2_evidence_score_v4",
    "m3_evidence_score_v3",
    "m4_evidence_score_v3",
]

reliability_columns = [
    "m1_reliability_v2",
    "m2_reliability_v2",
    "m3_reliability_v2",
    "m4_reliability_v2",
]


# ============================================================
# 6. CREATE EVIDENCE × RELIABILITY FEATURES
#
# This explicitly gives XGBoost the reliability-adjusted
# evidence signal.
# ============================================================

for i in range(4):

    evidence_col = evidence_columns[i]
    reliability_col = reliability_columns[i]

    weighted_col = f"m{i + 1}_evidence_x_reliability"

    df[weighted_col] = (
        pd.to_numeric(df[evidence_col], errors="coerce")
        * pd.to_numeric(df[reliability_col], errors="coerce")
    )

weighted_columns = [
    "m1_evidence_x_reliability",
    "m2_evidence_x_reliability",
    "m3_evidence_x_reliability",
    "m4_evidence_x_reliability",
]

feature_columns = (
    evidence_columns
    + reliability_columns
    + weighted_columns
)

print("\nFeatures used by XGBoost:")
for col in feature_columns:
    print("  ", col)


# ============================================================
# IMPORTANT:
# The weighted columns were created on df after train/test
# were created above, so update train/test from the modified df.
# ============================================================

train = df[df["tender_year"] < TEST_START_YEAR].copy()
test = df[df["tender_year"] >= TEST_START_YEAR].copy()


# ============================================================
# 7. PREPARE X / y
# ============================================================

X_train = train[feature_columns].copy()
X_test = test[feature_columns].copy()

y_train = train["is_cartel"].astype(int)
y_test = test["is_cartel"].astype(int)


# Convert everything explicitly to numeric

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
# 8. CHECK MISSING VALUES
# ============================================================

print("\n" + "=" * 70)
print("FEATURE INFORMATION")
print("=" * 70)

print("Number of features:", len(feature_columns))

print(
    "Missing training values:",
    int(X_train.isna().sum().sum())
)

print(
    "Missing testing values:",
    int(X_test.isna().sum().sum())
)


# ============================================================
# 9. CLASS WEIGHT
# ============================================================

negative_count = int((y_train == 0).sum())
positive_count = int((y_train == 1).sum())

scale_pos_weight = negative_count / positive_count

print("\nTraining non-cartel:", negative_count)
print("Training cartel    :", positive_count)
print("scale_pos_weight   :", scale_pos_weight)


# ============================================================
# 10. TRAIN XGBOOST
#
# This is intentionally a relatively conservative model.
# The objective is to learn how M1-M4 evidence and reliability
# should be combined, not to use hundreds of raw procurement
# variables.
# ============================================================

model = XGBClassifier(
    n_estimators=300,
    max_depth=3,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.9,

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
# 11. GENERATE TEST PROBABILITIES
# ============================================================

test_probability = model.predict_proba(X_test)[:, 1]

# Use 0.5 as the classification threshold.
test_prediction = (test_probability >= 0.5).astype(int)


# ============================================================
# 12. EVALUATION
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


# Confusion matrix

tn, fp, fn, tp = confusion_matrix(
    y_test,
    test_prediction,
    labels=[0, 1]
).ravel()

specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0


# ============================================================
# 13. PRINT RESULTS
# ============================================================

print("\n" + "=" * 70)
print("RESULTS")
print("=" * 70)

print(f"Test rows        : {len(y_test):,}")
print(f"Cartel rows      : {int(y_test.sum()):,}")
print(f"Non-cartel rows  : {int((y_test == 0).sum()):,}")

print(f"\nROC-AUC          : {roc_auc:.4f}")
print(f"Average Precision: {average_precision:.4f}")
print(f"Precision        : {precision:.4f}")
print(f"Recall           : {recall:.4f}")
print(f"Specificity      : {specificity:.4f}")
print(f"F1               : {f1:.4f}")

print("\nConfusion Matrix:")
print(
    pd.DataFrame(
        [[tn, fp], [fn, tp]],
        index=["Actual Non-cartel", "Actual Cartel"],
        columns=["Predicted Non-cartel", "Predicted Cartel"]
    )
)


# ============================================================
# 14. SAVE METRICS
# ============================================================

metrics = pd.DataFrame([{
    "model": "XGBoost_Reliability_Aware_Evidence_Fusion",

    "test_start_year": TEST_START_YEAR,

    "train_rows": len(train),
    "test_rows": len(test),

    "train_min_year": train["tender_year"].min(),
    "train_max_year": train["tender_year"].max(),

    "test_min_year": test["tender_year"].min(),
    "test_max_year": test["tender_year"].max(),

    "train_cartel": int(y_train.sum()),
    "train_non_cartel": int((y_train == 0).sum()),

    "test_cartel": int(y_test.sum()),
    "test_non_cartel": int((y_test == 0).sum()),

    "n_features": len(feature_columns),

    "roc_auc": roc_auc,
    "average_precision": average_precision,

    "precision": precision,
    "recall": recall,
    "specificity": specificity,
    "f1": f1,

    "tn": tn,
    "fp": fp,
    "fn": fn,
    "tp": tp,

    "scale_pos_weight": scale_pos_weight
}])

metrics.to_csv(
    OUTPUT_DIR / "xgb_reliability_evidence_metrics.csv",
    index=False
)


# ============================================================
# 15. SAVE CONFUSION MATRIX
# ============================================================

confusion_df = pd.DataFrame(
    [[tn, fp], [fn, tp]],
    index=["Actual Non-cartel", "Actual Cartel"],
    columns=["Predicted Non-cartel", "Predicted Cartel"]
)

confusion_df.to_csv(
    OUTPUT_DIR / "xgb_reliability_evidence_confusion_matrix.csv"
)


# ============================================================
# 16. SAVE TEST PREDICTIONS
# ============================================================

predictions = test[
    [
        "row_id",
        "tender_year",
        "is_cartel"
    ]
].copy()

predictions["xgb_reliability_evidence_score"] = test_probability
predictions["xgb_reliability_evidence_prediction"] = test_prediction

# Also save the original evidence/reliability inputs for auditing.

for col in feature_columns:
    predictions[col] = test[col].values

predictions.to_csv(
    OUTPUT_DIR / "xgb_reliability_evidence_test_predictions.csv",
    index=False
)


# ============================================================
# 17. FEATURE IMPORTANCE
# ============================================================

importance = pd.DataFrame({
    "feature": feature_columns,
    "importance": model.feature_importances_
})

importance = importance.sort_values(
    "importance",
    ascending=False
).reset_index(drop=True)

importance.to_csv(
    OUTPUT_DIR / "xgb_reliability_evidence_feature_importance.csv",
    index=False
)


# ============================================================
# 18. PRINT TOP FEATURES
# ============================================================

print("\n" + "=" * 70)
print("TOP FEATURES")
print("=" * 70)

print(
    importance.head(20).to_string(index=False)
)


# ============================================================
# 19. SAVE MODEL
# ============================================================

model.save_model(
    OUTPUT_DIR / "xgb_reliability_evidence_model.json"
)


# ============================================================
# 20. FINAL SUMMARY
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

