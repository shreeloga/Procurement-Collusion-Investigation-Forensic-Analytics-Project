from pathlib import Path
import warnings

import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from xgboost import XGBClassifier


warnings.filterwarnings("ignore")


# ============================================================
# 1. PATHS
# ============================================================

SCRIPT_DIR = Path(__file__).resolve().parent

INPUT_FILE = SCRIPT_DIR / "labelled_fusion_scores.parquet"

OUTPUT_DIR = SCRIPT_DIR / "hybrid_meta_learning_results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 2. CONFIGURATION
# ============================================================

RANDOM_STATE = 42

TARGET_COL = "is_cartel"

DATE_COL = "tender_year"

# Same chronological split used in your strict XGBoost experiment.
TEST_START_YEAR = 2016


# ============================================================
# 3. REQUIRED COLUMNS
# ============================================================

EVIDENCE_COLS = [
    "m1_evidence_score_v3",
    "m2_evidence_score_v4",
    "m3_evidence_score_v3",
    "m4_evidence_score_v3",
]

RELIABILITY_COLS = [
    "m1_reliability_v2",
    "m2_reliability_v2",
    "m3_reliability_v2",
    "m4_reliability_v2",
]

# These are NOT required to already exist.
# We will create them below.
WEIGHTED_EVIDENCE_COLS = [
    "m1_evidence_x_reliability",
    "m2_evidence_x_reliability",
    "m3_evidence_x_reliability",
    "m4_evidence_x_reliability",
]

FUSION_COLS = [
    "equal_weight_fusion",
    "reliability_aware_fusion",
]

# ============================================================
# 4. LOAD DATA
# ============================================================

print("=" * 70)
print("HYBRID META-LEARNING")
print("=" * 70)

print("\nLoading dataset:")
print(INPUT_FILE)

df = pd.read_parquet(INPUT_FILE)

print("\nDataset shape:", df.shape)


# ============================================================
# 5. CHECK REQUIRED COLUMNS
# ============================================================

required_columns = (
    [TARGET_COL, DATE_COL]
    + EVIDENCE_COLS
    + RELIABILITY_COLS
    + FUSION_COLS
)

missing_columns = [
    col for col in required_columns
    if col not in df.columns
]

if missing_columns:
    raise ValueError(
        "\nMissing required columns:\n"
        + "\n".join(missing_columns)
    )

print("\nAll required columns are present.")


# ============================================================
# 6. KEEP LABELLED DATA
# ============================================================

# Only labelled observations can be used for supervised learning.
df = df[df[TARGET_COL].notna()].copy()

# Convert label to integer.
df[TARGET_COL] = df[TARGET_COL].astype(int)

print("\nLabelled rows:", len(df))

print("\nLabel distribution:")
print(df[TARGET_COL].value_counts().sort_index())

print("\nCartel prevalence:")
print(df[TARGET_COL].mean())


# ============================================================
# 7. CONVERT TENDER YEAR
# ============================================================

df[DATE_COL] = pd.to_numeric(
    df[DATE_COL],
    errors="coerce"
)

df = df[df[DATE_COL].notna()].copy()

df[DATE_COL] = df[DATE_COL].astype(float)


# ============================================================
# 8. CREATE AVAILABILITY FEATURES
# ============================================================

print("\nCreating evidence availability features...")

availability_cols = []

for evidence_col in EVIDENCE_COLS:

    module = evidence_col.split("_")[0]

    availability_col = f"{module}_available"

    df[availability_col] = (
        df[evidence_col].notna().astype(int)
    )

    availability_cols.append(availability_col)


# ============================================================
# 9. CREATE EXPLICIT EVIDENCE-RELIABILITY INTERACTIONS
# ============================================================

print("Creating evidence × reliability features...")

# These already exist in the dataset, but we verify them.
for evidence_col, reliability_col, weighted_col in zip(
    EVIDENCE_COLS,
    RELIABILITY_COLS,
    WEIGHTED_EVIDENCE_COLS
):

    # Do not overwrite existing columns.
    # If a weighted column is missing, create it.
    if weighted_col not in df.columns:
        df[weighted_col] = (
            pd.to_numeric(
                df[evidence_col],
                errors="coerce"
            )
            *
            pd.to_numeric(
                df[reliability_col],
                errors="coerce"
            )
        )


# ============================================================
# 10. META-MODEL FEATURE SET
# ============================================================

META_FEATURES = (
    EVIDENCE_COLS
    + RELIABILITY_COLS
    + WEIGHTED_EVIDENCE_COLS
    + availability_cols
    + FUSION_COLS
)

print("\nMeta-model features:")
for feature in META_FEATURES:
    print("  ", feature)

print("\nNumber of meta-model features:", len(META_FEATURES))


# ============================================================
# 11. CREATE MODEL DATAFRAME
# ============================================================

X = df[META_FEATURES].copy()
y = df[TARGET_COL].copy()


# ============================================================
# 12. NUMERIC CONVERSION
# ============================================================

for col in META_FEATURES:
    X[col] = pd.to_numeric(
        X[col],
        errors="coerce"
    )


# ============================================================
# 13. CHRONOLOGICAL SPLIT
# ============================================================

train_mask = df[DATE_COL] < TEST_START_YEAR
test_mask = df[DATE_COL] >= TEST_START_YEAR

X_train = X.loc[train_mask].copy()
X_test = X.loc[test_mask].copy()

y_train = y.loc[train_mask].copy()
y_test = y.loc[test_mask].copy()

years_train = df.loc[train_mask, DATE_COL]
years_test = df.loc[test_mask, DATE_COL]


print("\n" + "=" * 70)
print("CHRONOLOGICAL SPLIT")
print("=" * 70)

print(
    f"\nTraining years: "
    f"{years_train.min():.0f} - {years_train.max():.0f}"
)

print(
    f"Testing years: "
    f"{years_test.min():.0f} - {years_test.max():.0f}"
)

print("\nTraining rows:", len(X_train))
print("Testing rows :", len(X_test))

print("\nTraining class distribution:")
print(y_train.value_counts().sort_index())

print("\nTesting class distribution:")
print(y_test.value_counts().sort_index())

print(
    "\nTest cartel prevalence:",
    round(y_test.mean(), 6)
)


# ============================================================
# 14. HANDLE MISSING VALUES
# ============================================================

# For meta-learning, missing evidence means the module did not
# provide evidence for that procurement record.
#
# We use:
#   - 0 for missing evidence/reliability/interactions
#   - explicit availability flags to tell the model whether
#     the module was actually available.
#
# This is important because:
#
# evidence = 0
#
# and
#
# evidence unavailable
#
# are NOT necessarily the same thing.

X_train = X_train.fillna(0.0)
X_test = X_test.fillna(0.0)


print("\nMissing values after filling:")
print(
    "Training:",
    int(X_train.isna().sum().sum())
)

print(
    "Testing :",
    int(X_test.isna().sum().sum())
)


# ============================================================
# 15. SAVE SPLIT INFORMATION
# ============================================================

split_summary = pd.DataFrame({
    "dataset": [
        "training",
        "testing"
    ],
    "rows": [
        len(X_train),
        len(X_test)
    ],
    "min_year": [
        years_train.min(),
        years_test.min()
    ],
    "max_year": [
        years_train.max(),
        years_test.max()
    ],
    "cartel": [
        int(y_train.sum()),
        int(y_test.sum())
    ],
    "non_cartel": [
        int((y_train == 0).sum()),
        int((y_test == 0).sum())
    ]
})

split_summary.to_csv(
    OUTPUT_DIR / "hybrid_split_summary.csv",
    index=False
)


# ============================================================
# 16. EVALUATION FUNCTION
# ============================================================

def evaluate_model(
    model_name,
    y_true,
    probabilities,
    threshold=0.5
):

    predictions = (
        probabilities >= threshold
    ).astype(int)

    auc = roc_auc_score(
        y_true,
        probabilities
    )

    ap = average_precision_score(
        y_true,
        probabilities
    )

    precision = precision_score(
        y_true,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        y_true,
        predictions,
        zero_division=0
    )

    f1 = f1_score(
        y_true,
        predictions,
        zero_division=0
    )

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        predictions,
        labels=[0, 1]
    ).ravel()

    specificity = (
        tn / (tn + fp)
        if (tn + fp) > 0
        else 0
    )

    print("\n" + "-" * 70)
    print(model_name)
    print("-" * 70)

    print(f"ROC-AUC      : {auc:.4f}")
    print(f"Average Prec.: {ap:.4f}")
    print(f"Precision    : {precision:.4f}")
    print(f"Recall       : {recall:.4f}")
    print(f"Specificity  : {specificity:.4f}")
    print(f"F1           : {f1:.4f}")

    print("\nConfusion matrix:")
    print(
        np.array([
            [tn, fp],
            [fn, tp]
        ])
    )

    return {
        "model": model_name,
        "rows": len(y_true),
        "cartel": int(y_true.sum()),
        "non_cartel": int((y_true == 0).sum()),
        "roc_auc": auc,
        "average_precision": ap,
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
# 17. MODEL 1 — LOGISTIC REGRESSION META-FUSION
# ============================================================

print("\n" + "=" * 70)
print("MODEL 1: LOGISTIC REGRESSION META-FUSION")
print("=" * 70)

# Standardization is appropriate for Logistic Regression.
#
# class_weight='balanced' is intentionally NOT used here.
#
# Your strict XGBoost experiment used scale_pos_weight based
# on the training distribution. To keep the meta-learning
# comparison consistent, we calculate the same ratio.

negative_train = int((y_train == 0).sum())
positive_train = int((y_train == 1).sum())

scale_pos_weight = (
    negative_train / positive_train
)

print(
    "\nTraining scale_pos_weight:",
    round(scale_pos_weight, 4)
)

logistic_model = Pipeline([
    (
        "scaler",
        StandardScaler()
    ),
    (
        "logistic",
        LogisticRegression(
            max_iter=2000,
            C=1.0,
            class_weight=None,
            random_state=RANDOM_STATE
        )
    )
])

logistic_model.fit(
    X_train,
    y_train
)

logistic_probabilities = (
    logistic_model.predict_proba(X_test)[:, 1]
)

logistic_results = evaluate_model(
    "Logistic Regression Meta-Fusion",
    y_test,
    logistic_probabilities
)


# ============================================================
# 18. SAVE LOGISTIC PREDICTIONS
# ============================================================

logistic_predictions = pd.DataFrame({
    "row_id": df.loc[test_mask, "row_id"].values
    if "row_id" in df.columns
    else np.arange(len(X_test)),
    "tender_year": years_test.values,
    "is_cartel": y_test.values,
    "logistic_meta_probability": logistic_probabilities,
    "logistic_meta_prediction": (
        logistic_probabilities >= 0.5
    ).astype(int)
})

logistic_predictions.to_csv(
    OUTPUT_DIR /
    "logistic_meta_test_predictions.csv",
    index=False
)


# ============================================================
# 19. LOGISTIC REGRESSION COEFFICIENTS
# ============================================================

logistic_estimator = (
    logistic_model.named_steps["logistic"]
)

coefficients = logistic_estimator.coef_[0]

logistic_coefficients = pd.DataFrame({
    "feature": META_FEATURES,
    "coefficient": coefficients,
    "absolute_coefficient": np.abs(coefficients)
}).sort_values(
    "absolute_coefficient",
    ascending=False
)

logistic_coefficients.to_csv(
    OUTPUT_DIR /
    "logistic_meta_coefficients.csv",
    index=False
)


# ============================================================
# 20. MODEL 2 — XGBOOST META-FUSION
# ============================================================

print("\n" + "=" * 70)
print("MODEL 2: XGBOOST META-FUSION")
print("=" * 70)

xgb_meta_model = XGBClassifier(
    n_estimators=300,
    max_depth=3,
    learning_rate=0.03,
    subsample=0.8,
    colsample_bytree=0.8,

    objective="binary:logistic",
    eval_metric="logloss",

    scale_pos_weight=scale_pos_weight,

    random_state=RANDOM_STATE,
    n_jobs=-1,

    tree_method="hist"
)

xgb_meta_model.fit(
    X_train,
    y_train
)

xgb_meta_probabilities = (
    xgb_meta_model.predict_proba(X_test)[:, 1]
)

xgb_results = evaluate_model(
    "XGBoost Meta-Fusion",
    y_test,
    xgb_meta_probabilities
)


# ============================================================
# 21. SAVE XGBOOST PREDICTIONS
# ============================================================

xgb_predictions = pd.DataFrame({
    "row_id": df.loc[test_mask, "row_id"].values
    if "row_id" in df.columns
    else np.arange(len(X_test)),
    "tender_year": years_test.values,
    "is_cartel": y_test.values,
    "xgb_meta_probability": xgb_meta_probabilities,
    "xgb_meta_prediction": (
        xgb_meta_probabilities >= 0.5
    ).astype(int)
})

xgb_predictions.to_csv(
    OUTPUT_DIR /
    "xgb_meta_test_predictions.csv",
    index=False
)


# ============================================================
# 22. XGBOOST FEATURE IMPORTANCE
# ============================================================

xgb_importance = pd.DataFrame({
    "feature": META_FEATURES,
    "importance": xgb_meta_model.feature_importances_
}).sort_values(
    "importance",
    ascending=False
)

xgb_importance.to_csv(
    OUTPUT_DIR /
    "xgb_meta_feature_importance.csv",
    index=False
)


# ============================================================
# 23. BASELINE COMPARISON ON SAME TEST SET
# ============================================================

print("\n" + "=" * 70)
print("BASELINE FUSION COMPARISON")
print("=" * 70)

baseline_results = []


# ------------------------------------------------------------
# Equal-weight fusion
# ------------------------------------------------------------

equal_score = pd.to_numeric(
    df.loc[test_mask, "equal_weight_fusion"],
    errors="coerce"
)

valid_equal = equal_score.notna()

if valid_equal.sum() > 0:

    equal_prob = equal_score.loc[
        valid_equal
    ].values

    equal_y = y_test.loc[
        valid_equal
    ].values

    baseline_results.append(
        evaluate_model(
            "Equal-Weight Fusion",
            equal_y,
            equal_prob
        )
    )


# ------------------------------------------------------------
# Reliability-aware fusion
# ------------------------------------------------------------

ra_score = pd.to_numeric(
    df.loc[test_mask, "reliability_aware_fusion"],
    errors="coerce"
)

valid_ra = ra_score.notna()

if valid_ra.sum() > 0:

    ra_prob = ra_score.loc[
        valid_ra
    ].values

    ra_y = y_test.loc[
        valid_ra
    ].values

    baseline_results.append(
        evaluate_model(
            "Reliability-Aware Fusion",
            ra_y,
            ra_prob
        )
    )


# ============================================================
# 24. COMBINE ALL RESULTS
# ============================================================

all_results = pd.DataFrame(
    baseline_results
    + [
        logistic_results,
        xgb_results
    ]
)

all_results.to_csv(
    OUTPUT_DIR /
    "hybrid_meta_learning_metrics.csv",
    index=False
)


# ============================================================
# 25. PERFORMANCE DELTAS
# ============================================================

print("\n" + "=" * 70)
print("PERFORMANCE COMPARISON")
print("=" * 70)

results_lookup = all_results.set_index("model")


if (
    "Reliability-Aware Fusion" in results_lookup.index
    and "Equal-Weight Fusion" in results_lookup.index
):

    ra_auc = results_lookup.loc[
        "Reliability-Aware Fusion",
        "roc_auc"
    ]

    equal_auc = results_lookup.loc[
        "Equal-Weight Fusion",
        "roc_auc"
    ]

    ra_ap = results_lookup.loc[
        "Reliability-Aware Fusion",
        "average_precision"
    ]

    equal_ap = results_lookup.loc[
        "Equal-Weight Fusion",
        "average_precision"
    ]

    print(
        "\nReliability-aware vs Equal-weight:"
    )

    print(
        f"ΔAUC: {ra_auc - equal_auc:+.4f}"
    )

    print(
        f"ΔAP : {ra_ap - equal_ap:+.4f}"
    )


if (
    "XGBoost Meta-Fusion" in results_lookup.index
    and "Reliability-Aware Fusion" in results_lookup.index
):

    meta_auc = results_lookup.loc[
        "XGBoost Meta-Fusion",
        "roc_auc"
    ]

    ra_auc = results_lookup.loc[
        "Reliability-Aware Fusion",
        "roc_auc"
    ]

    meta_ap = results_lookup.loc[
        "XGBoost Meta-Fusion",
        "average_precision"
    ]

    ra_ap = results_lookup.loc[
        "Reliability-Aware Fusion",
        "average_precision"
    ]

    print(
        "\nXGBoost Meta-Fusion vs Reliability-aware:"
    )

    print(
        f"ΔAUC: {meta_auc - ra_auc:+.4f}"
    )

    print(
        f"ΔAP : {meta_ap - ra_ap:+.4f}"
    )


# ============================================================
# 26. SAVE ALL TEST PREDICTIONS TOGETHER
# ============================================================

combined_predictions = pd.DataFrame({
    "row_id": (
        df.loc[test_mask, "row_id"].values
        if "row_id" in df.columns
        else np.arange(len(X_test))
    ),

    "tender_year": years_test.values,

    "is_cartel": y_test.values,

    "equal_weight_fusion": (
        df.loc[
            test_mask,
            "equal_weight_fusion"
        ].values
    ),

    "reliability_aware_fusion": (
        df.loc[
            test_mask,
            "reliability_aware_fusion"
        ].values
    ),

    "logistic_meta_probability":
        logistic_probabilities,

    "xgb_meta_probability":
        xgb_meta_probabilities
})

combined_predictions.to_csv(
    OUTPUT_DIR /
    "hybrid_meta_test_predictions.csv",
    index=False
)


# ============================================================
# 27. FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("FINAL SUMMARY")
print("=" * 70)

print("\nResults:")
print(
    all_results[
        [
            "model",
            "rows",
            "roc_auc",
            "average_precision",
            "precision",
            "recall",
            "specificity",
            "f1"
        ]
    ].to_string(index=False)
)

print("\nOutput directory:")
print(OUTPUT_DIR)

print("\nFiles created:")

for file in sorted(OUTPUT_DIR.iterdir()):
    print("  ", file.name)

print("\nDone.")