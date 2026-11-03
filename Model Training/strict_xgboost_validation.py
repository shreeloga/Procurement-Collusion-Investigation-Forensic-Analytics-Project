import re
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


# ================================================================
# 1. PATHS
# ================================================================

SCRIPT_DIR = Path(__file__).resolve().parent

# Dataset is in the same folder as this script
DATA_FILE = SCRIPT_DIR / "df_full.parquet"

# Output folder
OUTPUT_DIR = SCRIPT_DIR / "strict_xgboost_results"
OUTPUT_DIR.mkdir(exist_ok=True)


# ================================================================
# 2. FEATURE CONFIGURATION
# ================================================================

# Features intentionally used for the strict model
FEATURES = [
    "country",
    "tender_year",
    "main_cpv_2",
    "main_cpv_3",
    "main_cpv_3x",
    "main_cpv_4",
    "lot_bidscount",
    "singleb",
    "bid_isconsortium",
    "bid_issubcontracted",
    "bid_price",
    "lot_estimatedprice",
    "tender_estimatedprice"
]

# Explicitly excluded features
EXCLUDED_FEATURES = [
    "tender_finalprice",
    "bid_iswinning",
    "cartel_id",
    "buyer_id",
    "bidder_id",
    "row_id",
    "is_cartel"
]


# ================================================================
# 3. FEATURE NAME SANITIZATION
# ================================================================

def sanitize_feature_names(columns):
    """
    XGBoost can reject feature names containing characters such as
    [, ], <, >, etc.

    Replace unsafe characters with underscores.
    """

    sanitized = []

    for col in columns:
        col = str(col)

        # Replace characters that may cause XGBoost issues
        col = re.sub(r"[\[\]<>]", "_", col)

        # Replace other unusual characters
        col = re.sub(r"[^0-9a-zA-Z_]+", "_", col)

        # Avoid empty feature names
        if col == "":
            col = "feature"

        sanitized.append(col)

    # Make names unique
    seen = {}
    unique_names = []

    for name in sanitized:

        if name not in seen:
            seen[name] = 0
            unique_names.append(name)

        else:
            seen[name] += 1
            unique_names.append(
                f"{name}_{seen[name]}"
            )

    return unique_names


# ================================================================
# 4. LOAD DATA
# ================================================================

print("=" * 70)
print("STRICT XGBOOST VALIDATION")
print("=" * 70)

print()
print("Loading original dataset...")
print(DATA_FILE)

if not DATA_FILE.exists():
    raise FileNotFoundError(
        f"\nDataset not found:\n{DATA_FILE}\n"
    )

df = pd.read_parquet(DATA_FILE)

print()
print(f"Rows:    {len(df):,}")
print(f"Columns: {len(df.columns)}")


# ================================================================
# 5. CHECK REQUIRED COLUMNS
# ================================================================

required_columns = set(FEATURES + ["is_cartel"])

missing_columns = [
    col for col in required_columns
    if col not in df.columns
]

if missing_columns:
    raise ValueError(
        "Missing required columns:\n"
        + "\n".join(missing_columns)
    )


# ================================================================
# 6. KEEP ONLY LABELLED OBSERVATIONS
# ================================================================
#
# is_cartel can contain:
#   True
#   False
#   None
#
# None means unlabelled and must NOT be treated as non-cartel.
# ================================================================

labelled = df[
    df["is_cartel"].isin([True, False, 0, 1])
].copy()

# Convert label explicitly to integer
labelled["is_cartel"] = (
    labelled["is_cartel"]
    .astype(int)
)

print()
print(f"Labelled observations: {len(labelled):,}")

print()
print("Class distribution:")
print(
    labelled["is_cartel"]
    .value_counts()
    .sort_index()
)


# ================================================================
# 7. PREPARE FEATURES
# ================================================================

print()
print("Features used:")

for feature in FEATURES:
    print(f"  {feature}")

print()
print("Explicitly excluded:")

print("  tender_finalprice  -> possible post-outcome information")
print("  bid_iswinning       -> outcome-related information")
print("  cartel_id           -> label information")
print("  buyer_id            -> identifier")
print("  bidder_id           -> identifier")


# ================================================================
# 8. REMOVE ROWS WITHOUT A VALID YEAR
# ================================================================

labelled = labelled[
    labelled["tender_year"].notna()
].copy()

# Convert year to numeric
labelled["tender_year"] = pd.to_numeric(
    labelled["tender_year"],
    errors="coerce"
)

labelled = labelled[
    labelled["tender_year"].notna()
].copy()

print()
print("Year range:")
print(
    f"Minimum year: {labelled['tender_year'].min()}"
)
print(
    f"Maximum year: {labelled['tender_year'].max()}"
)


# ================================================================
# 9. SORT CHRONOLOGICALLY
# ================================================================

labelled = labelled.sort_values(
    ["tender_year", "row_id"]
).reset_index(drop=True)


# ================================================================
# 10. CHOOSE A STRICT TIME-BASED SPLIT
# ================================================================
#
# We do NOT split randomly.
#
# We search for a year cutoff such that:
#
#   training years < test years
#
# and the test set contains enough cartel examples to calculate
# meaningful evaluation metrics.
#
# Target:
#   approximately 20% of labelled observations in testing
#   and at least 100 cartel examples.
#
# The exact split is printed below.
# ================================================================

print()
print("=" * 70)
print("TIME-BASED SPLIT")
print("=" * 70)

year_counts = (
    labelled.groupby("tender_year")
    .agg(
        rows=("is_cartel", "size"),
        cartel=("is_cartel", "sum")
    )
    .reset_index()
)

total_rows = len(labelled)
total_cartel = labelled["is_cartel"].sum()

target_test_rows = total_rows * 0.20

candidate_splits = []

for test_start_year in sorted(
    year_counts["tender_year"].unique()
):

    train_part = labelled[
        labelled["tender_year"] < test_start_year
    ]

    test_part = labelled[
        labelled["tender_year"] >= test_start_year
    ]

    if len(train_part) == 0 or len(test_part) == 0:
        continue

    train_cartel = train_part["is_cartel"].sum()
    test_cartel = test_part["is_cartel"].sum()

    # Both classes must exist in both sets
    if train_part["is_cartel"].nunique() < 2:
        continue

    if test_part["is_cartel"].nunique() < 2:
        continue

    # Prefer a test size close to 20%
    distance = abs(
        len(test_part) - target_test_rows
    )

    candidate_splits.append(
        (
            distance,
            test_start_year,
            len(train_part),
            len(test_part),
            train_cartel,
            test_cartel
        )
    )


# Prefer splits with at least 100 cartel examples.
good_splits = [
    item
    for item in candidate_splits
    if item[5] >= 100
]

if good_splits:
    candidate_splits = good_splits

if not candidate_splits:
    raise ValueError(
        "Could not find a valid chronological split "
        "containing both cartel and non-cartel examples "
        "in both training and testing."
    )

# Select split closest to 20% test size
candidate_splits.sort(
    key=lambda x: x[0]
)

(
    _distance,
    TEST_START_YEAR,
    _train_rows,
    _test_rows,
    _train_cartel,
    _test_cartel
) = candidate_splits[0]


# Actual split
train_df = labelled[
    labelled["tender_year"] < TEST_START_YEAR
].copy()

test_df = labelled[
    labelled["tender_year"] >= TEST_START_YEAR
].copy()


print()
print(f"Selected test start year: {TEST_START_YEAR}")

print()
print(f"Training rows: {len(train_df):,}")
print(f"Testing rows:  {len(test_df):,}")

print()
print(
    "Training years: "
    f"{train_df['tender_year'].min()} - "
    f"{train_df['tender_year'].max()}"
)

print(
    "Testing years:  "
    f"{test_df['tender_year'].min()} - "
    f"{test_df['tender_year'].max()}"
)

print()
print("Training class distribution:")
print(
    train_df["is_cartel"]
    .value_counts()
    .sort_index()
)

print()
print("Testing class distribution:")
print(
    test_df["is_cartel"]
    .value_counts()
    .sort_index()
)


# ================================================================
# 11. CHECK CLASS DISTRIBUTION
# ================================================================

if test_df["is_cartel"].nunique() < 2:

    raise ValueError(
        "\nThe selected test period contains only one class. "
        "AUC/AP cannot be meaningfully calculated."
    )

if train_df["is_cartel"].nunique() < 2:

    raise ValueError(
        "\nThe selected training period contains only one class."
    )


# ================================================================
# 12. CREATE X AND y
# ================================================================

X_train = train_df[FEATURES].copy()
y_train = train_df["is_cartel"].copy()

X_test = test_df[FEATURES].copy()
y_test = test_df["is_cartel"].copy()


# ================================================================
# 13. ENCODE CATEGORICAL FEATURES
# ================================================================

print()
print("Encoding categorical features...")

# Identify object/category columns
categorical_cols = X_train.select_dtypes(
    include=["object", "category"]
).columns.tolist()

print()
print("Categorical columns detected:")

for col in categorical_cols:
    print(f"  {col}")


# Convert categorical columns to strings
#
# This is important for:
#   bid_isconsortium
#   bid_issubcontracted
#
# because they are stored as object dtype in the parquet file.

for col in categorical_cols:

    X_train[col] = X_train[col].astype(str)
    X_test[col] = X_test[col].astype(str)


# One-hot encode training data
X_train = pd.get_dummies(
    X_train,
    columns=categorical_cols,
    dummy_na=True
)


# One-hot encode test data
X_test = pd.get_dummies(
    X_test,
    columns=categorical_cols,
    dummy_na=True
)


# ================================================================
# 14. ALIGN TRAINING AND TEST FEATURES
# ================================================================
#
# A category may exist in training but not testing, or vice versa.
#
# Reindexing guarantees that both datasets have exactly the
# same feature columns.
# ================================================================

X_test = X_test.reindex(
    columns=X_train.columns,
    fill_value=0
)


# ================================================================
# 15. FORCE NUMERIC DATA TYPES
# ================================================================

X_train = X_train.apply(
    pd.to_numeric,
    errors="coerce"
)

X_test = X_test.apply(
    pd.to_numeric,
    errors="coerce"
)


# Replace infinite values with NaN
X_train = X_train.replace(
    [np.inf, -np.inf],
    np.nan
)

X_test = X_test.replace(
    [np.inf, -np.inf],
    np.nan
)


print()
print(
    f"Feature count before sanitization: "
    f"{X_train.shape[1]}"
)


# ================================================================
# 16. SANITIZE FEATURE NAMES
# ================================================================

sanitized_columns = sanitize_feature_names(
    X_train.columns
)

X_train.columns = sanitized_columns
X_test.columns = sanitized_columns

print(
    f"Final feature count: "
    f"{X_train.shape[1]}"
)

print(
    "Feature names sanitized for XGBoost."
)


# ================================================================
# 17. VERIFY NO OBJECT COLUMNS REMAIN
# ================================================================

object_columns_train = X_train.select_dtypes(
    include=["object", "category"]
).columns.tolist()

object_columns_test = X_test.select_dtypes(
    include=["object", "category"]
).columns.tolist()

if object_columns_train:
    raise ValueError(
        "Object columns remain in X_train:\n"
        + "\n".join(object_columns_train)
    )

if object_columns_test:
    raise ValueError(
        "Object columns remain in X_test:\n"
        + "\n".join(object_columns_test)
    )


# ================================================================
# 18. MISSING VALUES
# ================================================================

train_missing = int(
    X_train.isna().sum().sum()
)

test_missing = int(
    X_test.isna().sum().sum()
)

print()
print("Missing values:")
print(f"Training: {train_missing:,}")
print(f"Testing:  {test_missing:,}")


# ================================================================
# 19. CLASS WEIGHT
# ================================================================

negative_count = int(
    (y_train == 0).sum()
)

positive_count = int(
    (y_train == 1).sum()
)

scale_pos_weight = (
    negative_count / positive_count
)

print()
print(
    f"Training non-cartel: {negative_count:,}"
)

print(
    f"Training cartel:     {positive_count:,}"
)

print(
    f"scale_pos_weight: {scale_pos_weight:.4f}"
)


# ================================================================
# 20. TRAIN XGBOOST
# ================================================================

print()
print("Training strict-validation XGBoost model...")

model = XGBClassifier(

    n_estimators=500,

    max_depth=5,

    learning_rate=0.05,

    subsample=0.8,

    colsample_bytree=0.8,

    objective="binary:logistic",

    eval_metric="logloss",

    scale_pos_weight=scale_pos_weight,

    random_state=42,

    n_jobs=-1,

    tree_method="hist"
)


model.fit(
    X_train,
    y_train
)


# ================================================================
# 21. PREDICTIONS
# ================================================================

print()
print("Generating test predictions...")

test_probability = model.predict_proba(
    X_test
)[:, 1]

test_prediction = (
    test_probability >= 0.5
).astype(int)


# ================================================================
# 22. EVALUATION METRICS
# ================================================================

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


# ================================================================
# 23. CONFUSION MATRIX
# ================================================================

cm = confusion_matrix(
    y_test,
    test_prediction,
    labels=[0, 1]
)

tn, fp, fn, tp = cm.ravel()

specificity = (
    tn / (tn + fp)
    if (tn + fp) > 0
    else 0
)


# ================================================================
# 24. PRINT RESULTS
# ================================================================

print()
print("=" * 70)
print("STRICT XGBOOST RESULTS")
print("=" * 70)

print()
print(f"Test rows:              {len(y_test):,}")

print(
    f"Test cartel:            "
    f"{int(y_test.sum()):,}"
)

print(
    f"Test non-cartel:        "
    f"{int((y_test == 0).sum()):,}"
)

print()
print(
    f"ROC-AUC:                {roc_auc:.4f}"
)

print(
    f"Average Precision:      {average_precision:.4f}"
)

print(
    f"Precision:              {precision:.4f}"
)

print(
    f"Recall:                 {recall:.4f}"
)

print(
    f"Specificity:            {specificity:.4f}"
)

print(
    f"F1 Score:              {f1:.4f}"
)

print()
print("Confusion Matrix:")
print(cm)

print()
print("TN:", tn)
print("FP:", fp)
print("FN:", fn)
print("TP:", tp)


# ================================================================
# 25. SAVE METRICS
# ================================================================

metrics_df = pd.DataFrame({

    "metric": [
        "test_rows",
        "test_cartel",
        "test_non_cartel",
        "roc_auc",
        "average_precision",
        "precision",
        "recall",
        "specificity",
        "f1",
        "training_start_year",
        "training_end_year",
        "testing_start_year",
        "testing_end_year",
        "training_rows",
        "testing_rows",
        "training_cartel",
        "training_non_cartel"
    ],

    "value": [
        len(y_test),
        int(y_test.sum()),
        int((y_test == 0).sum()),
        roc_auc,
        average_precision,
        precision,
        recall,
        specificity,
        f1,
        train_df["tender_year"].min(),
        train_df["tender_year"].max(),
        test_df["tender_year"].min(),
        test_df["tender_year"].max(),
        len(y_train),
        len(y_test),
        positive_count,
        negative_count
    ]
})

metrics_path = (
    OUTPUT_DIR /
    "strict_xgboost_metrics.csv"
)

metrics_df.to_csv(
    metrics_path,
    index=False
)


# ================================================================
# 26. SAVE CONFUSION MATRIX
# ================================================================

confusion_df = pd.DataFrame({

    "actual": [
        0,
        0,
        1,
        1
    ],

    "predicted": [
        0,
        1,
        0,
        1
    ],

    "count": [
        tn,
        fp,
        fn,
        tp
    ]
})

confusion_path = (
    OUTPUT_DIR /
    "strict_xgboost_confusion_matrix.csv"
)

confusion_df.to_csv(
    confusion_path,
    index=False
)


# ================================================================
# 27. SAVE TEST PREDICTIONS
# ================================================================

predictions_df = pd.DataFrame({

    "row_id": test_df["row_id"].values,

    "tender_year": test_df["tender_year"].values,

    "is_cartel": y_test.values,

    "xgboost_probability": test_probability,

    "xgboost_prediction": test_prediction
})

predictions_path = (
    OUTPUT_DIR /
    "strict_xgboost_test_predictions.csv"
)

predictions_df.to_csv(
    predictions_path,
    index=False
)


# ================================================================
# 28. FEATURE IMPORTANCE
# ================================================================

importance_df = pd.DataFrame({

    "feature": X_train.columns,

    "importance": model.feature_importances_

})

importance_df = importance_df.sort_values(
    "importance",
    ascending=False
).reset_index(drop=True)


importance_path = (
    OUTPUT_DIR /
    "strict_xgboost_feature_importance.csv"
)

importance_df.to_csv(
    importance_path,
    index=False
)


# ================================================================
# 29. PRINT TOP FEATURES
# ================================================================

print()
print("=" * 70)
print("TOP 20 FEATURE IMPORTANCES")
print("=" * 70)

print(
    importance_df.head(20).to_string(
        index=False
    )
)


# ================================================================
# 30. OUTPUT LOCATIONS
# ================================================================

print()
print("=" * 70)
print("OUTPUT FILES")
print("=" * 70)

print()
print(metrics_path)

print(confusion_path)

print(predictions_path)

print(importance_path)

print()
print("=" * 70)
print("STRICT VALIDATION COMPLETE")
print("=" * 70)