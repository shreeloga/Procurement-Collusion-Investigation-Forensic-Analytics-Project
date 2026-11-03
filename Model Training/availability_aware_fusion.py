# availability_aware_fusion.py

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

INPUT_FILE = SCRIPT_DIR / "labelled_fusion_scores.parquet"

OUTPUT_DIR = SCRIPT_DIR / "availability_aware_fusion_results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 2. LOAD DATA
# ============================================================

print("=" * 70)
print("AVAILABILITY-AWARE EVIDENCE FUSION")
print("=" * 70)

print("\nLoading:")
print(INPUT_FILE)

df = pd.read_parquet(INPUT_FILE)

print(f"Dataset shape: {df.shape}")


# ============================================================
# 3. REQUIRED COLUMNS
# ============================================================

required_columns = [
    "row_id",
    "tender_year",
    "is_cartel",

    # Evidence scores
    "m1_evidence_score_v3",
    "m2_evidence_score_v4",
    "m3_evidence_score_v3",
    "m4_evidence_score_v3",

    # Reliability
    "m1_reliability_v2",
    "m2_reliability_v2",
    "m3_reliability_v2",
    "m4_reliability_v2",

    # Existing fusion scores
    "equal_weight_fusion",
    "reliability_aware_fusion",
]


missing = [c for c in required_columns if c not in df.columns]

if missing:
    raise ValueError(
        "Missing required columns:\n"
        + "\n".join(missing)
    )

print("\nAll required columns are present.")


# ============================================================
# 4. KEEP LABELLED ROWS ONLY
# ============================================================

df = df[df["is_cartel"].notna()].copy()

print(f"\nLabelled rows: {len(df):,}")


# Convert label to integer
df["is_cartel"] = df["is_cartel"].astype(int)


# ============================================================
# 5. DEFINE MODULE AVAILABILITY
# ============================================================

# Evidence is considered available when the evidence score exists.
#
# We explicitly create availability flags rather than relying only
# on reliability. This lets us test whether explicit availability
# handling improves the existing reliability-aware fusion.

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


for module, col in evidence_columns.items():

    df[f"{module}_available_explicit"] = (
        df[col].notna().astype(float)
    )


# ============================================================
# 6. CHRONOLOGICAL SPLIT
# ============================================================

# Same split used in the strict XGBoost and previous fusion
# experiments:
#
# Training: 2004-2015
# Testing : 2016-2022

TRAIN_END_YEAR = 2016

train = df[df["tender_year"] < TRAIN_END_YEAR].copy()
test = df[df["tender_year"] >= TRAIN_END_YEAR].copy()

print("\nChronological split:")
print(f"Training rows: {len(train):,}")
print(f"Testing rows : {len(test):,}")

print(
    f"Training years: "
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
# 7. DEFINE AVAILABILITY-AWARE FUSION
# ============================================================

def availability_aware_fusion(
    data,
    weights
):
    """
    Availability-aware fusion:

        F =
        sum(w_i * E_i * A_i)
        --------------------
        sum(w_i * A_i)

    where:

        E_i = module evidence score
        A_i = module availability
        w_i = module weight

    Equal module weights are used here because the purpose of
    this experiment is to test explicit availability handling.

    Missing evidence does not contribute to the numerator or
    denominator.
    """

    numerator = np.zeros(len(data), dtype=float)
    denominator = np.zeros(len(data), dtype=float)

    for module in ["m1", "m2", "m3", "m4"]:

        evidence = pd.to_numeric(
            data[evidence_columns[module]],
            errors="coerce"
        ).fillna(0.0).to_numpy()

        available = data[
            f"{module}_available_explicit"
        ].to_numpy()

        weight = weights[module]

        numerator += weight * evidence * available
        denominator += weight * available

    score = np.full(len(data), np.nan, dtype=float)

    valid = denominator > 0

    score[valid] = (
        numerator[valid] /
        denominator[valid]
    )

    return pd.Series(
        score,
        index=data.index,
        name="availability_aware_fusion"
    )


# ============================================================
# 8. EQUAL MODULE WEIGHTS
# ============================================================

weights = {
    "m1": 0.25,
    "m2": 0.25,
    "m3": 0.25,
    "m4": 0.25,
}

print("\nAvailability-aware weights:")
for module, weight in weights.items():
    print(f"{module.upper()}: {weight:.4f}")


# ============================================================
# 9. CALCULATE AVAILABILITY-AWARE SCORE
# ============================================================

train["availability_aware_fusion"] = (
    availability_aware_fusion(train, weights)
)

test["availability_aware_fusion"] = (
    availability_aware_fusion(test, weights)
)


# ============================================================
# 10. EVALUATION FUNCTION
# ============================================================

def evaluate_model(
    data,
    score_column,
    model_name
):

    subset = data[
        data[score_column].notna()
    ].copy()

    y_true = subset["is_cartel"].astype(int)
    y_score = subset[score_column].astype(float)

    # ROC-AUC and Average Precision
    auc = roc_auc_score(
        y_true,
        y_score
    )

    ap = average_precision_score(
        y_true,
        y_score
    )

    # Threshold = 0.5
    y_pred = (
        y_score >= 0.5
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

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1]
    ).ravel()

    specificity = (
        tn / (tn + fp)
        if (tn + fp) > 0
        else 0.0
    )

    return {
        "model": model_name,
        "rows": len(subset),
        "cartel_rows": int(y_true.sum()),
        "non_cartel_rows": int((y_true == 0).sum()),
        "roc_auc": auc,
        "average_precision": ap,
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
# 11. TEST AVAILABILITY
# ============================================================

print("\n" + "=" * 70)
print("MODULE AVAILABILITY")
print("=" * 70)

availability_rows = []

for module in ["m1", "m2", "m3", "m4"]:

    col = f"{module}_available_explicit"

    total_available = int(
        df[col].sum()
    )

    train_available = int(
        train[col].sum()
    )

    test_available = int(
        test[col].sum()
    )

    availability_rows.append({
        "module": module.upper(),
        "full_available": total_available,
        "full_coverage_percent":
            100 * total_available / len(df),

        "train_available": train_available,
        "train_coverage_percent":
            100 * train_available / len(train),

        "test_available": test_available,
        "test_coverage_percent":
            100 * test_available / len(test),
    })

availability_df = pd.DataFrame(
    availability_rows
)

print(
    availability_df.to_string(
        index=False
    )
)


# ============================================================
# 12. EVALUATE MODELS ON TEST SET
# ============================================================

results = []

# Availability-aware fusion
results.append(
    evaluate_model(
        test,
        "availability_aware_fusion",
        "Availability-aware Fusion"
    )
)


# Existing equal-weight fusion
results.append(
    evaluate_model(
        test,
        "equal_weight_fusion",
        "Equal-weight Fusion"
    )
)


# Existing reliability-aware fusion
results.append(
    evaluate_model(
        test,
        "reliability_aware_fusion",
        "Reliability-aware Fusion"
    )
)


results_df = pd.DataFrame(results)


# ============================================================
# 13. PRINT RESULTS
# ============================================================

print("\n" + "=" * 70)
print("TEST RESULTS")
print("=" * 70)

display_columns = [
    "model",
    "rows",
    "roc_auc",
    "average_precision",
    "precision",
    "recall",
    "specificity",
    "f1",
]

print(
    results_df[
        display_columns
    ].to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)


# ============================================================
# 14. CALCULATE DELTAS
# ============================================================

availability_result = results_df[
    results_df["model"] ==
    "Availability-aware Fusion"
].iloc[0]

equal_result = results_df[
    results_df["model"] ==
    "Equal-weight Fusion"
].iloc[0]

reliability_result = results_df[
    results_df["model"] ==
    "Reliability-aware Fusion"
].iloc[0]


delta_rows = [

    {
        "comparison":
            "Availability-aware vs Equal-weight",

        "delta_auc":
            availability_result["roc_auc"]
            - equal_result["roc_auc"],

        "delta_ap":
            availability_result["average_precision"]
            - equal_result["average_precision"],
    },

    {
        "comparison":
            "Availability-aware vs Reliability-aware",

        "delta_auc":
            availability_result["roc_auc"]
            - reliability_result["roc_auc"],

        "delta_ap":
            availability_result["average_precision"]
            - reliability_result["average_precision"],
    },

    {
        "comparison":
            "Reliability-aware vs Equal-weight",

        "delta_auc":
            reliability_result["roc_auc"]
            - equal_result["roc_auc"],

        "delta_ap":
            reliability_result["average_precision"]
            - equal_result["average_precision"],
    },
]


deltas_df = pd.DataFrame(delta_rows)


print("\n" + "=" * 70)
print("PERFORMANCE DELTAS")
print("=" * 70)

print(
    deltas_df.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)


# ============================================================
# 15. SAVE RESULTS
# ============================================================

results_file = (
    OUTPUT_DIR /
    "availability_aware_fusion_metrics.csv"
)

availability_file = (
    OUTPUT_DIR /
    "availability_statistics.csv"
)

deltas_file = (
    OUTPUT_DIR /
    "availability_aware_fusion_deltas.csv"
)

test_scores_file = (
    OUTPUT_DIR /
    "availability_aware_test_scores.parquet"
)


results_df.to_csv(
    results_file,
    index=False
)

availability_df.to_csv(
    availability_file,
    index=False
)

deltas_df.to_csv(
    deltas_file,
    index=False
)


# Save only useful test columns
test_output_columns = [
    "row_id",
    "tender_year",
    "is_cartel",

    "m1_evidence_score_v3",
    "m2_evidence_score_v4",
    "m3_evidence_score_v3",
    "m4_evidence_score_v3",

    "m1_available_explicit",
    "m2_available_explicit",
    "m3_available_explicit",
    "m4_available_explicit",

    "equal_weight_fusion",
    "reliability_aware_fusion",
    "availability_aware_fusion",
]

test[
    test_output_columns
].to_parquet(
    test_scores_file,
    index=False
)


# ============================================================
# 16. FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("FILES SAVED")
print("=" * 70)

print(results_file)
print(availability_file)
print(deltas_file)
print(test_scores_file)


print("\n" + "=" * 70)
print("FINAL INTERPRETATION")
print("=" * 70)

print(
    f"\nAvailability-aware Fusion AUC: "
    f"{availability_result['roc_auc']:.4f}"
)

print(
    f"Equal-weight Fusion AUC: "
    f"{equal_result['roc_auc']:.4f}"
)

print(
    f"Reliability-aware Fusion AUC: "
    f"{reliability_result['roc_auc']:.4f}"
)

print(
    f"\nAvailability-aware vs Equal AUC delta: "
    f"{availability_result['roc_auc'] - equal_result['roc_auc']:+.4f}"
)

print(
    f"Availability-aware vs Reliability-aware AUC delta: "
    f"{availability_result['roc_auc'] - reliability_result['roc_auc']:+.4f}"
)

print(
    f"\nAvailability-aware AP: "
    f"{availability_result['average_precision']:.4f}"
)

print(
    f"Equal-weight AP: "
    f"{equal_result['average_precision']:.4f}"
)

print(
    f"Reliability-aware AP: "
    f"{reliability_result['average_precision']:.4f}"
)

print("\nDone.")