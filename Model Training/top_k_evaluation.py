from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# 1. PATHS
# ============================================================

SCRIPT_DIR = Path(__file__).resolve().parent

INPUT_FILE = (
    SCRIPT_DIR
    / "rank_based_fusion_results"
    / "rank_based_test_scores.parquet"
)

OUTPUT_DIR = (
    SCRIPT_DIR
    / "top_k_results"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# 2. LOAD DATA
# ============================================================

print("=" * 70)
print("TOP-K FORENSIC SCREENING EVALUATION")
print("=" * 70)

print("\nLoading:")
print(INPUT_FILE)

df = pd.read_parquet(
    INPUT_FILE
)

print(
    f"Dataset shape: "
    f"{df.shape}"
)


# ============================================================
# 3. REQUIRED COLUMNS
# ============================================================

required_columns = [
    "row_id",
    "tender_year",
    "is_cartel",
    "rank_reliability_fusion",
    "equal_weight_fusion",
    "reliability_aware_fusion",
]

missing = [
    c for c in required_columns
    if c not in df.columns
]

if missing:
    raise ValueError(
        "Missing required columns:\n"
        + "\n".join(missing)
    )

print("\nAll required columns are present.")


# ============================================================
# 4. KEEP TEST PERIOD
# ============================================================

TEST_START_YEAR = 2016

test = df[
    df["tender_year"] >= TEST_START_YEAR
].copy()

print("\nTest period:")
print(
    f"Years: "
    f"{test['tender_year'].min()} - "
    f"{test['tender_year'].max()}"
)

print(
    f"Rows: "
    f"{len(test):,}"
)


# ============================================================
# 5. REMOVE MISSING SCORES
# ============================================================

score_columns = [
    "rank_reliability_fusion",
    "equal_weight_fusion",
    "reliability_aware_fusion",
]

for column in score_columns:

    available = (
        test[column]
        .notna()
        .sum()
    )

    print(
        f"{column}: "
        f"{available:,} available"
    )


# ============================================================
# 6. TOP-K FUNCTION
# ============================================================

def evaluate_top_k(
    data,
    score_column,
    model_name,
    k_percent
):
    """
    Select the top K% of tenders according to the
    fusion score.

    Example:
        k_percent = 5

    means the highest-scoring 5% of test tenders
    are selected for investigation.
    """

    subset = data[
        data[score_column].notna()
    ].copy()

    # Sort highest risk first
    subset = subset.sort_values(
        by=score_column,
        ascending=False
    ).reset_index(drop=True)

    total_rows = len(subset)

    total_cartel = int(
        subset["is_cartel"].sum()
    )

    total_non_cartel = (
        total_rows
        - total_cartel
    )

    # Number to investigate
    k_rows = int(
        np.ceil(
            total_rows
            * k_percent
            / 100.0
        )
    )

    # Make sure at least one row is selected
    k_rows = max(
        1,
        k_rows
    )

    selected = subset.iloc[
        :k_rows
    ].copy()

    selected_cartel = int(
        selected["is_cartel"].sum()
    )

    selected_non_cartel = (
        len(selected)
        - selected_cartel
    )

    # --------------------------------------------------------
    # Precision
    # --------------------------------------------------------

    precision = (
        selected_cartel
        / len(selected)
        if len(selected) > 0
        else 0.0
    )

    # --------------------------------------------------------
    # Recall
    # --------------------------------------------------------

    recall = (
        selected_cartel
        / total_cartel
        if total_cartel > 0
        else 0.0
    )

    # --------------------------------------------------------
    # Specificity
    #
    # TN = non-cartel rows NOT selected
    # FP = non-cartel rows selected
    # --------------------------------------------------------

    fp = selected_non_cartel

    tn = (
        total_non_cartel
        - fp
    )

    specificity = (
        tn
        / (tn + fp)
        if (tn + fp) > 0
        else 0.0
    )

    # --------------------------------------------------------
    # F1
    # --------------------------------------------------------

    if precision + recall > 0:

        f1 = (
            2
            * precision
            * recall
            / (precision + recall)
        )

    else:

        f1 = 0.0

    # --------------------------------------------------------
    # Random baseline
    #
    # If we randomly selected the same percentage of rows,
    # expected cartel capture = k_percent.
    # --------------------------------------------------------

    random_recall = (
        k_percent
        / 100.0
    )

    # Lift:
    #
    # actual recall / random recall
    #
    # Example:
    # Top 5% captures 20% of cartel cases
    # lift = 20 / 5 = 4x
    # --------------------------------------------------------

    lift = (
        recall
        / random_recall
        if random_recall > 0
        else np.nan
    )

    # --------------------------------------------------------
    # Cartel capture improvement over random
    # --------------------------------------------------------

    recall_gain_percentage_points = (
        recall * 100
        - k_percent
    )

    return {
        "model": model_name,
        "top_k_percent": k_percent,

        "total_rows": total_rows,

        "rows_selected": len(selected),

        "selection_percent":
            100
            * len(selected)
            / total_rows,

        "total_cartel":
            total_cartel,

        "cartel_selected":
            selected_cartel,

        "cartel_not_selected":
            total_cartel
            - selected_cartel,

        "non_cartel_selected":
            selected_non_cartel,

        "non_cartel_not_selected":
            tn,

        "precision":
            precision,

        "recall":
            recall,

        "specificity":
            specificity,

        "f1":
            f1,

        "random_recall":
            random_recall,

        "lift_over_random":
            lift,

        "recall_gain_percentage_points":
            recall_gain_percentage_points,
    }


# ============================================================
# 7. TOP-K LEVELS
# ============================================================

top_k_levels = [
    1,
    2,
    5,
    10,
    20,
]


# ============================================================
# 8. RUN EVALUATION
# ============================================================

results = []

models = [
    (
        "rank_reliability_fusion",
        "Rank Reliability-aware Fusion"
    ),

    (
        "reliability_aware_fusion",
        "Raw Reliability-aware Fusion"
    ),

    (
        "equal_weight_fusion",
        "Raw Equal-weight Fusion"
    ),
]


for score_column, model_name in models:

    print("\n" + "=" * 70)
    print(model_name)
    print("=" * 70)

    for k in top_k_levels:

        result = evaluate_top_k(
            test,
            score_column,
            model_name,
            k
        )

        results.append(
            result
        )

        print(
            f"\nTop {k}%:"
        )

        print(
            f"  Selected rows : "
            f"{result['rows_selected']:,}"
        )

        print(
            f"  Cartel found  : "
            f"{result['cartel_selected']:,}"
        )

        print(
            f"  Precision     : "
            f"{result['precision']:.4f}"
        )

        print(
            f"  Recall        : "
            f"{result['recall']:.4f}"
        )

        print(
            f"  Specificity   : "
            f"{result['specificity']:.4f}"
        )

        print(
            f"  F1            : "
            f"{result['f1']:.4f}"
        )

        print(
            f"  Lift          : "
            f"{result['lift_over_random']:.2f}x"
        )


results_df = pd.DataFrame(
    results
)


# ============================================================
# 9. SAVE COMPLETE RESULTS
# ============================================================

results_file = (
    OUTPUT_DIR
    / "top_k_screening_results.csv"
)

results_df.to_csv(
    results_file,
    index=False
)


# ============================================================
# 10. CREATE COMPARISON TABLE
# ============================================================

comparison_df = results_df[
    [
        "model",
        "top_k_percent",
        "rows_selected",
        "cartel_selected",
        "precision",
        "recall",
        "specificity",
        "f1",
        "lift_over_random",
    ]
].copy()


comparison_file = (
    OUTPUT_DIR
    / "top_k_comparison.csv"
)

comparison_df.to_csv(
    comparison_file,
    index=False
)


# ============================================================
# 11. RANK RELIABILITY-ONLY SUMMARY
# ============================================================

rank_summary = results_df[
    results_df["model"]
    == "Rank Reliability-aware Fusion"
].copy()

rank_summary_file = (
    OUTPUT_DIR
    / "rank_reliability_top_k_summary.csv"
)

rank_summary.to_csv(
    rank_summary_file,
    index=False
)


# ============================================================
# 12. SELECTED TOP-K ROWS
# ============================================================

# Save the actual highest-risk tenders for each Top-K level.
#
# This is useful later if you want to inspect which tenders
# the forensic system is prioritizing.

selected_rows = []

rank_data = test[
    test[
        "rank_reliability_fusion"
    ].notna()
].copy()

rank_data = rank_data.sort_values(
    by="rank_reliability_fusion",
    ascending=False
).reset_index(drop=True)


for k in top_k_levels:

    n_rows = int(
        np.ceil(
            len(rank_data)
            * k
            / 100.0
        )
    )

    n_rows = max(
        1,
        n_rows
    )

    selected = rank_data.iloc[
        :n_rows
    ].copy()

    selected[
        "top_k_percent"
    ] = k

    selected[
        "risk_rank"
    ] = np.arange(
        1,
        len(selected) + 1
    )

    selected_rows.append(
        selected
    )


selected_df = pd.concat(
    selected_rows,
    ignore_index=True
)


selected_file = (
    OUTPUT_DIR
    / "rank_reliability_selected_tenders.parquet"
)

selected_df.to_parquet(
    selected_file,
    index=False
)


# ============================================================
# 13. PRINT FINAL COMPARISON
# ============================================================

print("\n" + "=" * 70)
print("FINAL TOP-K COMPARISON")
print("=" * 70)

print(
    comparison_df.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)


# ============================================================
# 14. BEST SCREENING LEVEL
# ============================================================

print("\n" + "=" * 70)
print("RANK RELIABILITY-AWARE SUMMARY")
print("=" * 70)

for _, row in rank_summary.iterrows():

    print(
        f"\nTop {int(row['top_k_percent'])}%"
    )

    print(
        f"  Tenders investigated: "
        f"{int(row['rows_selected']):,}"
    )

    print(
        f"  Cartel tenders found: "
        f"{int(row['cartel_selected']):,}"
    )

    print(
        f"  Precision: "
        f"{row['precision']:.4f}"
    )

    print(
        f"  Recall: "
        f"{row['recall']:.4f}"
    )

    print(
        f"  Specificity: "
        f"{row['specificity']:.4f}"
    )

    print(
        f"  F1: "
        f"{row['f1']:.4f}"
    )

    print(
        f"  Lift over random: "
        f"{row['lift_over_random']:.2f}x"
    )


# ============================================================
# 15. FILES
# ============================================================

print("\n" + "=" * 70)
print("FILES SAVED")
print("=" * 70)

print(results_file)
print(comparison_file)
print(rank_summary_file)
print(selected_file)

print("\nDone.")