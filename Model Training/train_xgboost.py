import os
import re
import warnings

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

from xgboost import XGBClassifier


warnings.filterwarnings("ignore")


# ================================================================
# CONFIGURATION
# ================================================================

INPUT_FILE = "df_full.parquet"

FUSION_FILE = (
    "converted_data/fusion_results/"
    "labelled_fusion_scores.parquet"
)

OUTPUT_DIR = (
    "converted_data/fusion_results/"
    "xgboost_results"
)

RANDOM_STATE = 42
TEST_SIZE = 0.20


# ================================================================
# START
# ================================================================

print("=" * 80)
print("XGBOOST CARTEL DETECTION - ORIGINAL DATASET")
print("=" * 80)


# ================================================================
# LOAD ORIGINAL DATASET
# ================================================================

print("\nLoading dataset:")
print(INPUT_FILE)

df = pd.read_parquet(INPUT_FILE)

print("\nDataset information:")
print(f"Rows:    {len(df):,}")
print(f"Columns: {len(df.columns)}")


print("\nColumns:")

for col in df.columns:
    print("  ", col)


# ================================================================
# CHECK TARGET
# ================================================================

if "is_cartel" not in df.columns:

    raise ValueError(
        "Column 'is_cartel' was not found in df_full.parquet."
    )


# ================================================================
# KEEP ONLY LABELLED ROWS
# ================================================================

print("\n" + "=" * 80)
print("LABEL FILTERING")
print("=" * 80)

print("\nOriginal is_cartel values:")

print(
    df["is_cartel"]
    .value_counts(dropna=False)
)


# Keep ONLY known labels 0 and 1.
labelled = df[
    df["is_cartel"].isin([0, 1])
].copy()


print("\nLabelled observations:")
print(f"Rows: {len(labelled):,}")


print("\nLabel distribution:")

print(
    labelled["is_cartel"]
    .value_counts()
    .sort_index()
)


# ================================================================
# TARGET
# ================================================================

y = labelled["is_cartel"].astype(int)


# ================================================================
# FEATURE SELECTION
# ================================================================

print("\n" + "=" * 80)
print("FEATURE SELECTION")
print("=" * 80)


# Direct identifiers / target leakage / outcome information.
EXCLUDED_COLUMNS = [
    "row_id",
    "cartel_id",
    "buyer_id",
    "bidder_id",

    # Winner information is excluded from the first experiment.
    "bid_iswinning",

    # Date fields are excluded in this first clean experiment.
    "tender_publications_firstcallfortenderdate",
    "tender_publications_firstdcontractawarddate",

    # Target.
    "is_cartel",
]


# Features from the original dataset.
FEATURE_COLUMNS = [
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
    "tender_estimatedprice",
    "tender_finalprice",
]


# Check that requested columns actually exist.
missing_features = [
    col
    for col in FEATURE_COLUMNS
    if col not in labelled.columns
]


if missing_features:

    raise ValueError(
        "The following expected feature columns are missing:\n"
        + "\n".join(missing_features)
    )


X = labelled[
    FEATURE_COLUMNS
].copy()


print("\nFeatures used:")

for col in FEATURE_COLUMNS:
    print("  ", col)


# ================================================================
# DATA CLEANING
# ================================================================

print("\n" + "=" * 80)
print("DATA CLEANING")
print("=" * 80)


# ------------------------------------------------
# Boolean columns
# ------------------------------------------------

BOOLEAN_COLUMNS = [
    "singleb",
    "bid_isconsortium",
    "bid_issubcontracted",
]


for col in BOOLEAN_COLUMNS:

    if col in X.columns:

        X[col] = X[col].astype("float")


# ------------------------------------------------
# Numeric columns
# ------------------------------------------------

NUMERIC_COLUMNS = [
    "tender_year",
    "lot_bidscount",
    "bid_price",
    "lot_estimatedprice",
    "tender_estimatedprice",
    "tender_finalprice",
]


for col in NUMERIC_COLUMNS:

    if col in X.columns:

        X[col] = pd.to_numeric(
            X[col],
            errors="coerce"
        )


# ------------------------------------------------
# Categorical columns
# ------------------------------------------------

CATEGORICAL_COLUMNS = [
    "country",
    "main_cpv_2",
    "main_cpv_3",
    "main_cpv_3x",
    "main_cpv_4",
]


for col in CATEGORICAL_COLUMNS:

    if col in X.columns:

        X[col] = X[col].astype("string")


# ================================================================
# TRAIN / TEST SPLIT
# ================================================================

print("\n" + "=" * 80)
print("TRAIN / TEST SPLIT")
print("=" * 80)


X_train, X_test, y_train, y_test = train_test_split(

    X,
    y,

    test_size=TEST_SIZE,

    random_state=RANDOM_STATE,

    stratify=y,
)


print(
    f"\nTraining rows: {len(X_train):,}"
)

print(
    f"Testing rows:  {len(X_test):,}"
)


print("\nTraining label distribution:")

print(
    y_train.value_counts()
)


print("\nTesting label distribution:")

print(
    y_test.value_counts()
)


# ================================================================
# CATEGORICAL ENCODING
# ================================================================

print("\n" + "=" * 80)
print("CATEGORICAL ENCODING")
print("=" * 80)


# One-hot encode categorical columns.

X_train = pd.get_dummies(

    X_train,

    columns=CATEGORICAL_COLUMNS,

    dummy_na=True

)


X_test = pd.get_dummies(

    X_test,

    columns=CATEGORICAL_COLUMNS,

    dummy_na=True

)


# Make train and test have exactly the same columns.

X_train, X_test = X_train.align(

    X_test,

    join="left",

    axis=1,

    fill_value=0

)


# ================================================================
# BOOLEAN TO INTEGER
# ================================================================

for col in X_train.columns:

    if X_train[col].dtype == bool:

        X_train[col] = X_train[col].astype(int)

    if X_test[col].dtype == bool:

        X_test[col] = X_test[col].astype(int)


# ================================================================
# NUMERIC CONVERSION
# ================================================================

X_train = X_train.apply(
    pd.to_numeric,
    errors="coerce"
)

X_test = X_test.apply(
    pd.to_numeric,
    errors="coerce"
)


# ================================================================
# REPLACE INFINITE VALUES
# ================================================================

X_train = X_train.replace(
    [np.inf, -np.inf],
    np.nan
)

X_test = X_test.replace(
    [np.inf, -np.inf],
    np.nan
)


# ================================================================
# SANITIZE FEATURE NAMES
# ================================================================

print("\n" + "=" * 80)
print("SANITIZING FEATURE NAMES")
print("=" * 80)


def clean_feature_name(name):

    name = str(name)

    # XGBoost does not allow these characters.
    name = name.replace("[", "_")
    name = name.replace("]", "_")
    name = name.replace("<", "_")

    # Replace any other unusual characters.
    name = re.sub(
        r"[^A-Za-z0-9_]+",
        "_",
        name
    )

    # Remove repeated underscores.
    name = re.sub(
        r"_+",
        "_",
        name
    )

    # Remove leading/trailing underscores.
    name = name.strip("_")

    # Avoid empty names.
    if name == "":
        name = "feature"

    return name


# Clean training feature names.

cleaned_columns = [
    clean_feature_name(col)
    for col in X_train.columns
]


X_train.columns = cleaned_columns


# X_test must use exactly the same names.

X_test.columns = cleaned_columns


# ================================================================
# CHECK FOR DUPLICATE FEATURE NAMES
# ================================================================

if X_train.columns.duplicated().any():

    print(
        "\nDuplicate feature names detected."
    )

    new_columns = []

    counts = {}

    for col in X_train.columns:

        if col not in counts:

            counts[col] = 0

            new_columns.append(col)

        else:

            counts[col] += 1

            new_columns.append(
                f"{col}_{counts[col]}"
            )


    X_train.columns = new_columns

    X_test.columns = new_columns


print(
    f"\nFinal feature count: "
    f"{X_train.shape[1]:,}"
)

print(
    "Feature names sanitized for XGBoost."
)


# ================================================================
# MISSING VALUES
# ================================================================

print("\nMissing values:")

print(
    f"Training: "
    f"{int(X_train.isna().sum().sum()):,}"
)

print(
    f"Testing:  "
    f"{int(X_test.isna().sum().sum()):,}"
)


# XGBoost can handle NaN values directly.


# ================================================================
# CLASS BALANCE
# ================================================================

negative = int(
    (y_train == 0).sum()
)

positive = int(
    (y_train == 1).sum()
)


scale_pos_weight = (
    negative / positive
)


print("\n" + "=" * 80)
print("CLASS BALANCE")
print("=" * 80)


print(
    f"Non-cartel training rows: "
    f"{negative:,}"
)

print(
    f"Cartel training rows:     "
    f"{positive:,}"
)

print(
    f"scale_pos_weight:         "
    f"{scale_pos_weight:.4f}"
)


# ================================================================
# XGBOOST MODEL
# ================================================================

print("\n" + "=" * 80)
print("TRAINING XGBOOST")
print("=" * 80)


model = XGBClassifier(

    n_estimators=500,

    max_depth=5,

    learning_rate=0.05,

    subsample=0.8,

    colsample_bytree=0.8,

    objective="binary:logistic",

    eval_metric="logloss",

    scale_pos_weight=scale_pos_weight,

    random_state=RANDOM_STATE,

    n_jobs=-1,

    tree_method="hist",
)


model.fit(

    X_train,

    y_train,

    eval_set=[
        (X_test, y_test)
    ],

    verbose=False,
)


print(
    "\nXGBoost training completed."
)


# ================================================================
# PREDICTIONS
# ================================================================

print("\n" + "=" * 80)
print("GENERATING PREDICTIONS")
print("=" * 80)


# Probability of cartel class.

y_probability = model.predict_proba(
    X_test
)[:, 1]


# Default classification threshold.

THRESHOLD = 0.50


y_prediction = (
    y_probability >= THRESHOLD
).astype(int)


# ================================================================
# PERFORMANCE METRICS
# ================================================================

roc_auc = roc_auc_score(

    y_test,

    y_probability
)


average_precision = average_precision_score(

    y_test,

    y_probability
)


precision = precision_score(

    y_test,

    y_prediction,

    zero_division=0
)


recall = recall_score(

    y_test,

    y_prediction,

    zero_division=0
)


f1 = f1_score(

    y_test,

    y_prediction,

    zero_division=0
)


# ================================================================
# CONFUSION MATRIX
# ================================================================

tn, fp, fn, tp = confusion_matrix(

    y_test,

    y_prediction

).ravel()


specificity = (

    tn / (tn + fp)

    if (tn + fp) > 0

    else 0
)


# ================================================================
# DISPLAY RESULTS
# ================================================================

print("\n" + "=" * 80)
print("XGBOOST PERFORMANCE")
print("=" * 80)


print(
    f"\nROC-AUC:           "
    f"{roc_auc:.4f}"
)


print(
    f"Average Precision: "
    f"{average_precision:.4f}"
)


print(
    f"Precision:         "
    f"{precision:.4f}"
)


print(
    f"Recall:            "
    f"{recall:.4f}"
)


print(
    f"Specificity:       "
    f"{specificity:.4f}"
)


print(
    f"F1 Score:          "
    f"{f1:.4f}"
)


print("\nConfusion Matrix:")


confusion_df = pd.DataFrame(

    [
        [tn, fp],
        [fn, tp]
    ],

    index=[
        "Actual Non-Cartel",
        "Actual Cartel"
    ],

    columns=[
        "Predicted Non-Cartel",
        "Predicted Cartel"
    ]

)


print(
    confusion_df
)


# ================================================================
# FEATURE IMPORTANCE
# ================================================================

print("\n" + "=" * 80)
print("TOP XGBOOST FEATURES")
print("=" * 80)


feature_importance = pd.DataFrame({

    "feature":
        X_train.columns,

    "importance":
        model.feature_importances_

})


feature_importance = (
    feature_importance
    .sort_values(
        "importance",
        ascending=False
    )
)


print(
    feature_importance
    .head(30)
    .to_string(index=False)
)


# ================================================================
# CREATE OUTPUT DIRECTORY
# ================================================================

os.makedirs(

    OUTPUT_DIR,

    exist_ok=True
)


# ================================================================
# SAVE XGBOOST METRICS
# ================================================================

metrics = pd.DataFrame([{

    "model":
        "XGBoost",

    "dataset":
        "df_full.parquet",

    "training_rows":
        len(X_train),

    "test_rows":
        len(X_test),

    "positive_test_rows":
        int((y_test == 1).sum()),

    "negative_test_rows":
        int((y_test == 0).sum()),

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

    "threshold":
        THRESHOLD,

    "n_features":
        X_train.shape[1],

}])


metrics.to_csv(

    os.path.join(
        OUTPUT_DIR,
        "xgboost_metrics.csv"
    ),

    index=False
)


# ================================================================
# SAVE FEATURE IMPORTANCE
# ================================================================

feature_importance.to_csv(

    os.path.join(
        OUTPUT_DIR,
        "xgboost_feature_importance.csv"
    ),

    index=False
)


# ================================================================
# SAVE TEST PREDICTIONS
# ================================================================

test_predictions = labelled.loc[

    X_test.index,

    [
        "row_id",
        "is_cartel"
    ]

].copy()


test_predictions[
    "xgboost_probability"
] = y_probability


test_predictions[
    "xgboost_prediction"
] = y_prediction


test_predictions.to_csv(

    os.path.join(
        OUTPUT_DIR,
        "xgboost_test_predictions.csv"
    ),

    index=False
)


# ================================================================
# COMPARE WITH EXISTING FUSION
# ================================================================

print("\n" + "=" * 80)
print("COMPARING WITH EXISTING FUSION")
print("=" * 80)


if os.path.exists(FUSION_FILE):

    print(
        "\nLoading fusion results:"
    )

    print(
        FUSION_FILE
    )


    fusion_df = pd.read_parquet(
        FUSION_FILE
    )


    # Get the row IDs belonging to the
    # XGBoost test set.

    test_row_ids = set(
        test_predictions["row_id"]
    )


    fusion_test = fusion_df[
        fusion_df["row_id"].isin(
            test_row_ids
        )
    ].copy()


    comparison = []


    # ------------------------------------------------------------
    # XGBoost
    # ------------------------------------------------------------

    comparison.append({

        "method":
            "XGBoost",

        "type":
            "Supervised ML",

        "rows":
            len(y_test),

        "coverage_percent":
            100.0,

        "roc_auc":
            roc_auc,

        "average_precision":
            average_precision,

    })


    # ------------------------------------------------------------
    # Equal-weight Fusion
    # ------------------------------------------------------------

    if (
        "equal_weight_fusion"
        in fusion_test.columns
    ):

        valid = fusion_test[
            fusion_test[
                "equal_weight_fusion"
            ].notna()
        ].copy()


        if len(valid) > 0:

            equal_auc = roc_auc_score(

                valid["is_cartel"],

                valid[
                    "equal_weight_fusion"
                ]

            )


            equal_ap = (
                average_precision_score(

                    valid["is_cartel"],

                    valid[
                        "equal_weight_fusion"
                    ]

                )
            )


            comparison.append({

                "method":
                    "Equal-weight Fusion",

                "type":
                    "Evidence Fusion",

                "rows":
                    len(valid),

                "coverage_percent":
                    (
                        len(valid)
                        / len(fusion_test)
                        * 100
                    ),

                "roc_auc":
                    equal_auc,

                "average_precision":
                    equal_ap,

            })


    # ------------------------------------------------------------
    # Reliability-aware Fusion
    # ------------------------------------------------------------

    if (
        "reliability_aware_fusion"
        in fusion_test.columns
    ):

        valid = fusion_test[
            fusion_test[
                "reliability_aware_fusion"
            ].notna()
        ].copy()


        if len(valid) > 0:

            ra_auc = roc_auc_score(

                valid["is_cartel"],

                valid[
                    "reliability_aware_fusion"
                ]

            )


            ra_ap = (
                average_precision_score(

                    valid["is_cartel"],

                    valid[
                        "reliability_aware_fusion"
                    ]

                )
            )


            comparison.append({

                "method":
                    "Reliability-Aware Fusion",

                "type":
                    "Evidence Fusion",

                "rows":
                    len(valid),

                "coverage_percent":
                    (
                        len(valid)
                        / len(fusion_test)
                        * 100
                    ),

                "roc_auc":
                    ra_auc,

                "average_precision":
                    ra_ap,

            })


    # ------------------------------------------------------------
    # DISPLAY COMPARISON
    # ------------------------------------------------------------

    comparison_df = pd.DataFrame(
        comparison
    )


    print("\nPerformance comparison:")

    print(
        comparison_df.to_string(
            index=False
        )
    )


    # Save comparison.

    comparison_df.to_csv(

        os.path.join(
            OUTPUT_DIR,
            "xgboost_vs_fusion.csv"
        ),

        index=False
    )


else:

    print(
        "\nFusion file was not found:"
    )

    print(
        FUSION_FILE
    )

    print(
        "\nXGBoost results will still be saved."
    )


# ================================================================
# SAVE CONFUSION MATRIX
# ================================================================

confusion_df.to_csv(

    os.path.join(
        OUTPUT_DIR,
        "xgboost_confusion_matrix.csv"
    )
)


# ================================================================
# FINAL
# ================================================================

print("\n" + "=" * 80)
print("XGBOOST EXPERIMENT COMPLETE")
print("=" * 80)


print("\nOutput directory:")

print(
    OUTPUT_DIR
)


print("\nFiles created:")

print(
    "  xgboost_metrics.csv"
)

print(
    "  xgboost_feature_importance.csv"
)

print(
    "  xgboost_test_predictions.csv"
)

print(
    "  xgboost_confusion_matrix.csv"
)


if os.path.exists(FUSION_FILE):

    print(
        "  xgboost_vs_fusion.csv"
    )


print("\nDone.")