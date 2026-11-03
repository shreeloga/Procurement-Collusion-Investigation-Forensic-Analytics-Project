# test_soft_or_integration.py
"""
Automated Integration and Smoke Test Suite for Soft-OR Candidate Final Integration.
Verifies:
1. Full-dataset Soft-OR score file exists, has correct schema, row count, and missingness.
2. Baseline full-dataset score file is preserved and intact.
3. Dashboard (app.py) dependencies, data loading, and column privacy (hidden cartel labels).
4. Exact numerical reproduction of chronological evaluation metrics:
   - Soft-OR: AUC ~ 0.7580, AP ~ 0.2062, Top 5% precision ~ 24.12% (75 cartels).
   - Baseline: AUC ~ 0.7559, AP ~ 0.0937, Top 5% precision ~ 13.50% (42 cartels).
"""

from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score

BASE_DIR = Path(__file__).resolve().parent
DASHBOARD_DATA_DIR = BASE_DIR / "dashboard_data"
SOFT_OR_FILE = DASHBOARD_DATA_DIR / "full_tender_risk_scores_soft_or.parquet"
BASELINE_FILE = DASHBOARD_DATA_DIR / "full_tender_risk_scores.parquet"
AUDIT_SCORES = BASE_DIR / "fusion_experiments" / "soft_or_audit" / "audit_scored_test_records.parquet"

def test_1_baseline_file_preservation():
    print("Test 1: Checking baseline file preservation...")
    assert BASELINE_FILE.exists(), f"Baseline file {BASELINE_FILE} missing!"
    df_base = pd.read_parquet(BASELINE_FILE)
    assert len(df_base) == 3861477, f"Expected 3,861,477 rows, got {len(df_base):,}"
    assert "final_risk_score" in df_base.columns, "final_risk_score missing from baseline!"
    print(f"  PASS: Baseline file intact ({len(df_base):,} rows).")

def test_2_soft_or_dataset_integrity():
    print("Test 2: Checking Soft-OR full dataset integrity...")
    assert SOFT_OR_FILE.exists(), f"Soft-OR file {SOFT_OR_FILE} missing!"
    df_soft = pd.read_parquet(SOFT_OR_FILE)
    assert len(df_soft) == 3861477, f"Expected 3,861,477 rows, got {len(df_soft):,}"
    
    # Check new versioned columns
    required_cols = [
        "final_risk_score", "soft_or_risk_score", "baseline_risk_score",
        "investigation_rank", "risk_percentile", "risk_level",
        "m1_rank", "m2_rank", "m3_rank", "m4_rank",
        "m1_reliability_v2", "m2_reliability_v2", "m3_reliability_v2", "m4_reliability_v2"
    ]
    for col in required_cols:
        assert col in df_soft.columns, f"Required column {col} missing from Soft-OR parquet!"

    valid_count = df_soft["final_risk_score"].notna().sum()
    na_count = df_soft["final_risk_score"].isna().sum()
    assert valid_count == 3854645, f"Expected 3,854,645 valid scores, got {valid_count:,}"
    assert na_count == 6832, f"Expected 6,832 NaN scores, got {na_count:,}"
    print(f"  PASS: Soft-OR dataset valid ({valid_count:,} usable scores, {na_count:,} missing).")

def test_3_privacy_and_dashboard_display_columns():
    print("Test 3: Checking dashboard privacy & column separation...")
    import ast
    with open(BASE_DIR / "app.py", "r", encoding="utf-8") as f:
        tree = ast.parse(f.read())
    
    # Check display_columns definitions in app.py
    display_cols = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "display_columns":
                    if isinstance(node.value, ast.List):
                        for elt in node.value.elts:
                            if isinstance(elt, ast.Constant):
                                display_cols.append(elt.value)
    
    assert "is_cartel" not in display_cols, "Privacy violation: is_cartel found in display_columns!"
    assert "cartel_id" not in display_cols, "Privacy violation: cartel_id found in display_columns!"
    print("  PASS: Ground-truth cartel labels are strictly hidden from investigation ranking table.")

def test_4_chronological_evaluation_reproduction():
    print("Test 4: Verifying exact chronological evaluation metrics...")
    assert AUDIT_SCORES.exists(), f"Audit scores {AUDIT_SCORES} missing!"
    df = pd.read_parquet(AUDIT_SCORES)
    assert len(df) == 6203, f"Expected 6,203 scored test records, got {len(df)}"
    assert df["is_cartel"].sum() == 225, f"Expected 225 cartels, got {df['is_cartel'].sum()}"

    # Soft-OR metrics
    s_auc = roc_auc_score(df["is_cartel"], df["soft_or_score"])
    s_ap = average_precision_score(df["is_cartel"], df["soft_or_score"])
    
    # Baseline metrics
    b_auc = roc_auc_score(df["is_cartel"], df["baseline_score"])
    b_ap = average_precision_score(df["is_cartel"], df["baseline_score"])

    # Top 5% screening
    top5_count = int(np.ceil(len(df) * 0.05)) # 311
    s_top5 = df.sort_values(by="soft_or_score", ascending=False, kind="mergesort").iloc[:top5_count]
    b_top5 = df.sort_values(by="baseline_score", ascending=False, kind="mergesort").iloc[:top5_count]

    s_tp = int(s_top5["is_cartel"].sum())
    b_tp = int(b_top5["is_cartel"].sum())
    s_p = s_tp / top5_count
    b_p = b_tp / top5_count

    print(f"  Baseline: AUC={b_auc:.6f}, AP={b_ap:.6f}, Top-5% P={b_p:.4%} ({b_tp} cartels)")
    print(f"  Soft-OR : AUC={s_auc:.6f}, AP={s_ap:.6f}, Top-5% P={s_p:.4%} ({s_tp} cartels)")

    assert np.isclose(b_auc, 0.7559, atol=1e-4), f"Baseline AUC mismatch: {b_auc}"
    assert np.isclose(b_ap, 0.0937, atol=1e-4), f"Baseline AP mismatch: {b_ap}"
    assert b_tp == 42, f"Baseline Top-5 positives mismatch: {b_tp} != 42"

    assert np.isclose(s_auc, 0.7580, atol=1e-4), f"Soft-OR AUC mismatch: {s_auc}"
    assert np.isclose(s_ap, 0.2062, atol=1e-4), f"Soft-OR AP mismatch: {s_ap}"
    assert s_tp == 75, f"Soft-OR Top-5 positives mismatch: {s_tp} != 75"
    assert np.isclose(s_p, 0.241158, atol=1e-4), f"Soft-OR Top-5 Precision mismatch: {s_p}"

    print("  PASS: Exact reproduction verified.")

def main():
    print("=" * 70)
    print("RUNNING AUTOMATED INTEGRATION TESTS FOR SOFT-OR CANDIDATE FINAL METHOD")
    print("=" * 70)
    test_1_baseline_file_preservation()
    test_2_soft_or_dataset_integrity()
    test_3_privacy_and_dashboard_display_columns()
    test_4_chronological_evaluation_reproduction()
    print("=" * 70)
    print("ALL 4 INTEGRATION TESTS PASSED SUCCESSFULLY!")
    print("=" * 70)

if __name__ == "__main__":
    main()
