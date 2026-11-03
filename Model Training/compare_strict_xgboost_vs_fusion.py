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


# ================================================================
# 1. PATHS
# ================================================================

SCRIPT_DIR = Path(__file__).resolve().parent

XGB_FILE = (
    SCRIPT_DIR
    / "strict_xgboost_results"
    / "strict_xgboost_test_predictions.csv"
)

FUSION_FILE = (
    SCRIPT_DIR
    / "labelled_fusion_scores.parquet"
)

OUTPUT_DIR = (
    SCRIPT_DIR
    / "strict_xgboost_vs_fusion_results"
)

OUTPUT_DIR.mkdir(
    exist_ok=True
)


# ================================================================
# 2. CHECK FILES
# ================================================================

print("=" * 70)
print("STRICT XGBOOST vs FUSION COMPARISON")
print("=" * 70)

print()
print("XGBoost predictions:")
print(XGB_FILE)

print()
print("Fusion scores:")
print(FUSION_FILE)

if not XGB_FILE.exists():
    raise FileNotFoundError(
        f"\nXGBoost prediction file not found:\n{XGB_FILE}"
    )

if not FUSION_FILE.exists():
    raise FileNotFoundError(
        f"\nFusion file not found:\n{FUSION_FILE}"
    )


# ================================================================
# 3. LOAD DATA
# ================================================================

print()
print("Loading XGBoost predictions...")

xgb = pd.read_csv(
    XGB_FILE
)

print(
    f"XGBoost rows: {len(xgb):,}"
)

print()
print("Loading fusion scores...")

fusion = pd.read_parquet(
    FUSION_FILE
)

print(
    f"Fusion rows: {len(fusion):,}"
)


# ================================================================
# 4. CHECK REQUIRED COLUMNS
# ================================================================

xgb_required = [
    "row_id",
    "is_cartel",
    "xgboost_probability",
    "xgboost_prediction"
]

fusion_required = [
    "row_id",
    "is_cartel",
    "equal_weight_fusion",
    "reliability_aware_fusion"
]

missing_xgb = [
    col
    for col in xgb_required
    if col not in xgb.columns
]

missing_fusion = [
    col
    for col in fusion_required
    if col not in fusion.columns
]

if missing_xgb:
    raise ValueError(
        "Missing XGBoost columns:\n"
        + "\n".join(missing_xgb)
    )

if missing_fusion:
    raise ValueError(
        "Missing fusion columns:\n"
        + "\n".join(missing_fusion)
    )


# ================================================================
# 5. KEEP ONLY REQUIRED COLUMNS
# ================================================================

xgb = xgb[
    xgb_required
].copy()

fusion = fusion[
    fusion_required
].copy()


# ================================================================
# 6. REMOVE DUPLICATE ROW IDs
# ================================================================

xgb = xgb.drop_duplicates(
    subset=["row_id"]
)

fusion = fusion.drop_duplicates(
    subset=["row_id"]
)


# ================================================================
# 7. MERGE ON EXACT SAME ROWS
# ================================================================

print()
print("Matching XGBoost test rows with fusion rows...")

merged = pd.merge(
    xgb,
    fusion,
    on="row_id",
    how="inner",
    suffixes=("_xgb", "_fusion")
)

print(
    f"Matched rows: {len(merged):,}"
)


# ================================================================
# 8. VERIFY LABEL CONSISTENCY
# ================================================================

label_mismatch = (
    merged["is_cartel_xgb"]
    != merged["is_cartel_fusion"]
)

mismatch_count = int(
    label_mismatch.sum()
)

print()
print(
    f"Label mismatches: {mismatch_count:,}"
)

if mismatch_count > 0:

    raise ValueError(
        "The same row_id has different is_cartel labels "
        "between XGBoost and fusion data."
    )


# Use the verified label
merged["is_cartel"] = (
    merged["is_cartel_xgb"]
    .astype(int)
)


# ================================================================
# 9. CHECK MISSING SCORES
# ================================================================

score_columns = [
    "xgboost_probability",
    "equal_weight_fusion",
    "reliability_aware_fusion"
]

print()
print("Missing scores:")

for col in score_columns:

    missing = int(
        merged[col].isna().sum()
    )

    print(
        f"  {col}: {missing:,}"
    )


# ================================================================
# 10. COMMON-ROW DATASETS
# ================================================================
#
# XGBoost itself should have a score for every strict test row.
#
# Fusion may have a small number of unavailable scores.
#
# Therefore, each comparison uses rows where BOTH models being
# compared have valid scores.
# ================================================================

xgb_fusion = merged.dropna(
    subset=[
        "xgboost_probability",
        "equal_weight_fusion"
    ]
).copy()

xgb_ra = merged.dropna(
    subset=[
        "xgboost_probability",
        "reliability_aware_fusion"
    ]
).copy()

fusion_common = merged.dropna(
    subset=[
        "equal_weight_fusion",
        "reliability_aware_fusion"
    ]
).copy()


# ================================================================
# 11. METRIC FUNCTION
# ================================================================

def calculate_metrics(
    df,
    score_column,
    prediction_column=None
):

    y_true = df["is_cartel"].astype(int)

    scores = pd.to_numeric(
        df[score_column],
        errors="coerce"
    )

    valid = (
        y_true.notna()
        & scores.notna()
    )

    y_true = y_true[valid]
    scores = scores[valid]

    if y_true.nunique() < 2:
        raise ValueError(
            f"Cannot calculate ROC-AUC for {score_column}: "
            "only one class is present."
        )

    auc = roc_auc_score(
        y_true,
        scores
    )

    ap = average_precision_score(
        y_true,
        scores
    )

    # Use supplied prediction if available.
    if prediction_column is not None:

        predictions = (
            df.loc[valid, prediction_column]
            .astype(int)
        )

    else:

        # 0.5 threshold for fusion scores and XGBoost
        predictions = (
            scores >= 0.5
        ).astype(int)

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

    cm = confusion_matrix(
        y_true,
        predictions,
        labels=[0, 1]
    )

    tn, fp, fn, tp = cm.ravel()

    specificity = (
        tn / (tn + fp)
        if (tn + fp) > 0
        else 0
    )

    return {
        "rows": len(y_true),
        "cartel": int(y_true.sum()),
        "non_cartel": int((y_true == 0).sum()),
        "roc_auc": auc,
        "average_precision": ap,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "f1": f1,
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp)
    }


# ================================================================
# 12. CALCULATE XGBOOST METRICS
# ================================================================

print()
print("=" * 70)
print("XGBOOST vs EQUAL-WEIGHT FUSION")
print("=" * 70)

xgb_equal_metrics = calculate_metrics(
    xgb_fusion,
    "xgboost_probability",
    "xgboost_prediction"
)

equal_metrics = calculate_metrics(
    xgb_fusion,
    "equal_weight_fusion"
)


# ================================================================
# 13. CALCULATE XGBOOST vs RA FUSION
# ================================================================

print()
print("=" * 70)
print("XGBOOST vs RELIABILITY-AWARE FUSION")
print("=" * 70)

xgb_ra_metrics = calculate_metrics(
    xgb_ra,
    "xgboost_probability",
    "xgboost_prediction"
)

ra_metrics = calculate_metrics(
    xgb_ra,
    "reliability_aware_fusion"
)


# ================================================================
# 14. PRINT RESULTS
# ================================================================

def print_metrics(
    name,
    metrics
):

    print()
    print(name)

    print(
        f"  Rows:              {metrics['rows']:,}"
    )

    print(
        f"  Cartel:            {metrics['cartel']:,}"
    )

    print(
        f"  Non-cartel:        {metrics['non_cartel']:,}"
    )

    print(
        f"  ROC-AUC:           {metrics['roc_auc']:.4f}"
    )

    print(
        f"  Average Precision: {metrics['average_precision']:.4f}"
    )

    print(
        f"  Precision:         {metrics['precision']:.4f}"
    )

    print(
        f"  Recall:            {metrics['recall']:.4f}"
    )

    print(
        f"  Specificity:       {metrics['specificity']:.4f}"
    )

    print(
        f"  F1:                {metrics['f1']:.4f}"
    )

    print()
    print("  Confusion Matrix:")

    print(
        f"    TN = {metrics['tn']:,}"
    )

    print(
        f"    FP = {metrics['fp']:,}"
    )

    print(
        f"    FN = {metrics['fn']:,}"
    )

    print(
        f"    TP = {metrics['tp']:,}"
    )


print_metrics(
    "XGBoost - same rows as Equal Fusion",
    xgb_equal_metrics
)

print_metrics(
    "Equal-weight Fusion",
    equal_metrics
)

print_metrics(
    "XGBoost - same rows as RA Fusion",
    xgb_ra_metrics
)

print_metrics(
    "Reliability-aware Fusion",
    ra_metrics
)


# ================================================================
# 15. DIFFERENCES
# ================================================================

delta_equal_auc = (
    xgb_equal_metrics["roc_auc"]
    - equal_metrics["roc_auc"]
)

delta_equal_ap = (
    xgb_equal_metrics["average_precision"]
    - equal_metrics["average_precision"]
)

delta_ra_auc = (
    xgb_ra_metrics["roc_auc"]
    - ra_metrics["roc_auc"]
)

delta_ra_ap = (
    xgb_ra_metrics["average_precision"]
    - ra_metrics["average_precision"]
)


print()
print("=" * 70)
print("XGBOOST vs FUSION DIFFERENCES")
print("=" * 70)

print()
print(
    f"XGBoost - Equal Fusion ΔAUC: "
    f"{delta_equal_auc:+.4f}"
)

print(
    f"XGBoost - Equal Fusion ΔAP:  "
    f"{delta_equal_ap:+.4f}"
)

print()
print(
    f"XGBoost - RA Fusion ΔAUC: "
    f"{delta_ra_auc:+.4f}"
)

print(
    f"XGBoost - RA Fusion ΔAP:  "
    f"{delta_ra_ap:+.4f}"
)


# ================================================================
# 16. SAVE METRICS TABLE
# ================================================================

results = [

    {
        "model": "Strict XGBoost",
        "comparison_group": "XGBoost vs Equal Fusion",
        **xgb_equal_metrics
    },

    {
        "model": "Equal-weight Fusion",
        "comparison_group": "XGBoost vs Equal Fusion",
        **equal_metrics
    },

    {
        "model": "Strict XGBoost",
        "comparison_group": "XGBoost vs RA Fusion",
        **xgb_ra_metrics
    },

    {
        "model": "Reliability-aware Fusion",
        "comparison_group": "XGBoost vs RA Fusion",
        **ra_metrics
    }
]

results_df = pd.DataFrame(
    results
)

metrics_path = (
    OUTPUT_DIR
    / "strict_xgboost_vs_fusion_metrics.csv"
)

results_df.to_csv(
    metrics_path,
    index=False
)


# ================================================================
# 17. SAVE DELTA TABLE
# ================================================================

delta_df = pd.DataFrame({

    "comparison": [
        "XGBoost vs Equal-weight Fusion",
        "XGBoost vs Reliability-aware Fusion"
    ],

    "xgboost_auc": [
        xgb_equal_metrics["roc_auc"],
        xgb_ra_metrics["roc_auc"]
    ],

    "fusion_auc": [
        equal_metrics["roc_auc"],
        ra_metrics["roc_auc"]
    ],

    "delta_auc": [
        delta_equal_auc,
        delta_ra_auc
    ],

    "xgboost_average_precision": [
        xgb_equal_metrics["average_precision"],
        xgb_ra_metrics["average_precision"]
    ],

    "fusion_average_precision": [
        equal_metrics["average_precision"],
        ra_metrics["average_precision"]
    ],

    "delta_average_precision": [
        delta_equal_ap,
        delta_ra_ap
    ]
})

delta_path = (
    OUTPUT_DIR
    / "strict_xgboost_vs_fusion_deltas.csv"
)

delta_df.to_csv(
    delta_path,
    index=False
)


# ================================================================
# 18. SAVE MATCHED ROWS
# ================================================================

matched_columns = [
    "row_id",
    "is_cartel",
    "xgboost_probability",
    "xgboost_prediction",
    "equal_weight_fusion",
    "reliability_aware_fusion"
]

matched_path = (
    OUTPUT_DIR
    / "strict_xgboost_fusion_matched_rows.csv"
)

merged[matched_columns].to_csv(
    matched_path,
    index=False
)


# ================================================================
# 19. FINAL SUMMARY
# ================================================================

print()
print("=" * 70)
print("FINAL COMPARISON SUMMARY")
print("=" * 70)

print()
print(
    "Strict XGBoost ROC-AUC: "
    f"{xgb_equal_metrics['roc_auc']:.4f}"
)

print(
    "Equal-weight Fusion ROC-AUC: "
    f"{equal_metrics['roc_auc']:.4f}"
)

print(
    "Reliability-aware Fusion ROC-AUC: "
    f"{ra_metrics['roc_auc']:.4f}"
)

print()

print(
    "Strict XGBoost Average Precision: "
    f"{xgb_equal_metrics['average_precision']:.4f}"
)

print(
    "Equal-weight Fusion Average Precision: "
    f"{equal_metrics['average_precision']:.4f}"
)

print(
    "Reliability-aware Fusion Average Precision: "
    f"{ra_metrics['average_precision']:.4f}"
)


# ================================================================
# 20. OUTPUT FILES
# ================================================================

print()
print("=" * 70)
print("OUTPUT FILES")
print("=" * 70)

print()
print(metrics_path)

print(delta_path)

print(matched_path)

print()
print("=" * 70)
print("COMPARISON COMPLETE")
print("=" * 70)