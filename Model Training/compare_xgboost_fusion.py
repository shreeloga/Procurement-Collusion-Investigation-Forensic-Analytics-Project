import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import roc_auc_score, average_precision_score


# ============================================================
# PATHS
# ============================================================

SCRIPT_DIR = Path(__file__).resolve().parent

XGB_PREDICTIONS = (
    SCRIPT_DIR
    / "converted_data"
    / "fusion_results"
    / "xgboost_results"
    / "xgboost_test_predictions.csv"
)

FUSION_FILE = SCRIPT_DIR / "labelled_fusion_scores.parquet"

OUTPUT_DIR = SCRIPT_DIR / "comparison_results"
OUTPUT_DIR.mkdir(exist_ok=True)

# ============================================================
# CHECK FILES
# ============================================================

print("=" * 70)
print("XGBOOST vs FUSION COMPARISON")
print("=" * 70)

print("\nChecking files...")

if not XGB_PREDICTIONS.exists():
    raise FileNotFoundError(
        f"\nXGBoost prediction file not found:\n{XGB_PREDICTIONS}"
    )

if not FUSION_FILE.exists():
    raise FileNotFoundError(
        f"\nFusion file not found:\n{FUSION_FILE}"
    )

print(f"XGBoost predictions: {XGB_PREDICTIONS}")
print(f"Fusion scores:       {FUSION_FILE}")


# ============================================================
# LOAD FILES
# ============================================================

print("\nLoading files...")

xgb = pd.read_csv(XGB_PREDICTIONS)
fusion = pd.read_parquet(FUSION_FILE)

print(f"XGBoost rows: {len(xgb):,}")
print(f"Fusion rows:  {len(fusion):,}")


# ============================================================
# DISPLAY COLUMNS
# ============================================================

print("\nXGBoost columns:")
print(xgb.columns.tolist())

print("\nFusion columns:")
print(fusion.columns.tolist())


# ============================================================
# FIND REQUIRED COLUMNS
# ============================================================

# XGBoost prediction column
possible_xgb_pred = [
    "xgboost_probability",
    "xgb_probability",
    "predicted_probability",
    "prediction_probability",
    "probability",
    "y_pred_proba",
]

xgb_pred_col = None

for col in possible_xgb_pred:
    if col in xgb.columns:
        xgb_pred_col = col
        break

if xgb_pred_col is None:
    raise ValueError(
        "\nCould not find XGBoost probability column.\n"
        "Please check the printed XGBoost columns above."
    )


# Row identifier
if "row_id" not in xgb.columns:
    raise ValueError(
        "\n'xgb_test_predictions.csv' does not contain 'row_id'.\n"
        "The comparison requires row_id to safely match the same observations."
    )

if "row_id" not in fusion.columns:
    raise ValueError(
        "\n'labelled_fusion_scores.parquet' does not contain 'row_id'.\n"
        "The comparison requires row_id to safely match the same observations."
    )


# Label column
possible_label_cols = [
    "is_cartel",
    "label",
    "target",
    "y"
]

label_col = None

for col in possible_label_cols:
    if col in xgb.columns:
        label_col = col
        break

if label_col is None:
    if "is_cartel" in fusion.columns:
        label_col = "is_cartel"
    else:
        raise ValueError(
            "\nCould not find the cartel label column."
        )


# ============================================================
# IDENTIFY FUSION COLUMNS
# ============================================================

equal_col = "equal_weight_fusion"
ra_col = "reliability_aware_fusion"

if equal_col not in fusion.columns:
    raise ValueError(
        f"\nFusion column '{equal_col}' was not found."
    )

if ra_col not in fusion.columns:
    raise ValueError(
        f"\nFusion column '{ra_col}' was not found."
    )


print("\nUsing:")
print(f"XGBoost probability:       {xgb_pred_col}")
print(f"Label:                     {label_col}")
print(f"Equal-weight fusion:       {equal_col}")
print(f"Reliability-aware fusion:  {ra_col}")


# ============================================================
# PREPARE XGBOOST DATA
# ============================================================

xgb_compare = xgb[
    ["row_id", xgb_pred_col, label_col]
].copy()

xgb_compare = xgb_compare.rename(
    columns={
        xgb_pred_col: "xgboost_score",
        label_col: "is_cartel"
    }
)


# ============================================================
# PREPARE FUSION DATA
# ============================================================

fusion_compare = fusion[
    ["row_id", equal_col, ra_col]
].copy()

fusion_compare = fusion_compare.rename(
    columns={
        equal_col: "equal_weight_fusion",
        ra_col: "reliability_aware_fusion"
    }
)


# ============================================================
# MERGE USING row_id
# ============================================================

print("\nMatching XGBoost test rows with fusion scores...")

comparison = pd.merge(
    xgb_compare,
    fusion_compare,
    on="row_id",
    how="inner"
)

print(f"Matched rows: {len(comparison):,}")


if len(comparison) == 0:
    raise ValueError(
        "\nNo matching row_id values were found between "
        "XGBoost predictions and fusion scores."
    )


# ============================================================
# CLEAN LABEL
# ============================================================

comparison["is_cartel"] = comparison["is_cartel"].astype(int)


# ============================================================
# CALCULATE PERFORMANCE
# ============================================================

y = comparison["is_cartel"]

results = []


def evaluate_model(model_name, score_column):

    data = comparison[
        ["is_cartel", score_column]
    ].dropna()

    y_true = data["is_cartel"]
    y_score = data[score_column]

    auc = roc_auc_score(y_true, y_score)
    ap = average_precision_score(y_true, y_score)

    results.append({
        "Model": model_name,
        "Rows": len(data),
        "ROC-AUC": auc,
        "Average Precision": ap
    })


evaluate_model(
    "XGBoost",
    "xgboost_score"
)

evaluate_model(
    "Equal-weight Fusion",
    "equal_weight_fusion"
)

evaluate_model(
    "Reliability-aware Fusion",
    "reliability_aware_fusion"
)


# ============================================================
# RESULTS TABLE
# ============================================================

results_df = pd.DataFrame(results)

print("\n")
print("=" * 70)
print("PERFORMANCE COMPARISON")
print("=" * 70)

print(
    results_df.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)


# ============================================================
# DELTAS
# ============================================================

xgb_auc = results_df.loc[
    results_df["Model"] == "XGBoost",
    "ROC-AUC"
].iloc[0]

equal_auc = results_df.loc[
    results_df["Model"] == "Equal-weight Fusion",
    "ROC-AUC"
].iloc[0]

ra_auc = results_df.loc[
    results_df["Model"] == "Reliability-aware Fusion",
    "ROC-AUC"
].iloc[0]

xgb_ap = results_df.loc[
    results_df["Model"] == "XGBoost",
    "Average Precision"
].iloc[0]

equal_ap = results_df.loc[
    results_df["Model"] == "Equal-weight Fusion",
    "Average Precision"
].iloc[0]

ra_ap = results_df.loc[
    results_df["Model"] == "Reliability-aware Fusion",
    "Average Precision"
].iloc[0]


print("\n")
print("=" * 70)
print("COMPARISON DELTAS")
print("=" * 70)

print(f"\nXGBoost vs Equal-weight Fusion:")
print(f"  Δ ROC-AUC: {xgb_auc - equal_auc:+.4f}")
print(f"  Δ AP:      {xgb_ap - equal_ap:+.4f}")

print(f"\nXGBoost vs Reliability-aware Fusion:")
print(f"  Δ ROC-AUC: {xgb_auc - ra_auc:+.4f}")
print(f"  Δ AP:      {xgb_ap - ra_ap:+.4f}")


# ============================================================
# SAVE MATCHED DATA
# ============================================================

comparison_output = OUTPUT_DIR / "xgboost_fusion_matched_rows.csv"

comparison.to_csv(
    comparison_output,
    index=False
)


# ============================================================
# SAVE RESULTS
# ============================================================

results_output = OUTPUT_DIR / "xgboost_vs_fusion_metrics.csv"

results_df.to_csv(
    results_output,
    index=False
)


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n")
print("=" * 70)
print("FILES SAVED")
print("=" * 70)

print(f"\nMatched predictions:")
print(comparison_output)

print(f"\nPerformance comparison:")
print(results_output)

print("\nComparison completed successfully.")
print("=" * 70)