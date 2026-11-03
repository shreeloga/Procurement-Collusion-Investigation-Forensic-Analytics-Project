# fusion_experiments/run_diagnostics.py
import os
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score

BASE_DIR = Path(__file__).resolve().parent.parent
INPUT_FILE = BASE_DIR / "labelled_fusion_scores.parquet"
EXP_DIR = BASE_DIR / "fusion_experiments"
EXP_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_parquet(INPUT_FILE)
df = df[df["is_cartel"].isin([0, 1])].copy()
df["is_cartel"] = df["is_cartel"].astype(int)

train = df[df["tender_year"] < 2016].copy()
test = df[df["tender_year"] >= 2016].copy()

print("=" * 60)
print("DIAGNOSTIC RUN")
print(f"Train rows: {len(train):,}, Cartel: {train['is_cartel'].sum()} ({train['is_cartel'].mean():.4%})")
print(f"Test rows: {len(test):,}, Cartel: {test['is_cartel'].sum()} ({test['is_cartel'].mean():.4%})")
print("=" * 60)

# Check module availability
for m, col, rel in [
    ("M1", "m1_evidence_score_v3", "m1_reliability_v2"),
    ("M2", "m2_evidence_score_v4", "m2_reliability_v2"),
    ("M3", "m3_evidence_score_v3", "m3_reliability_v2"),
    ("M4", "m4_evidence_score_v3", "m4_reliability_v2"),
]:
    t_avail = train[col].notna().sum()
    te_avail = test[col].notna().sum()
    te_rel_mean = test.loc[test[col].notna(), rel].mean()
    te_rel_med = test.loc[test[col].notna(), rel].median()
    print(f"{m} ({col}): Train avail={t_avail} ({t_avail/len(train):.2%}), Test avail={te_avail} ({te_avail/len(test):.2%}), Test Rel mean={te_rel_mean:.4f}, med={te_rel_med:.4f}")

# Compute train references
rank_cols = []
for m, col in [
    ("m1", "m1_evidence_score_v3"),
    ("m2", "m2_evidence_score_v4"),
    ("m3", "m3_evidence_score_v3"),
    ("m4", "m4_evidence_score_v3"),
]:
    ref = np.sort(train[col].dropna().to_numpy(dtype=float))
    train[f"{m}_rank"] = np.nan
    test[f"{m}_rank"] = np.nan
    
    t_valid = train[col].notna()
    train.loc[t_valid, f"{m}_rank"] = np.searchsorted(ref, train.loc[t_valid, col].to_numpy(dtype=float), side="right") / len(ref)
    
    te_valid = test[col].notna()
    test.loc[te_valid, f"{m}_rank"] = np.searchsorted(ref, test.loc[te_valid, col].to_numpy(dtype=float), side="right") / len(ref)
    rank_cols.append(f"{m}_rank")

# Weighted fusion
num = pd.Series(0.0, index=test.index)
den = pd.Series(0.0, index=test.index)
m_contr = {}
for m, r_col, rel_col in [
    ("m1", "m1_rank", "m1_reliability_v2"),
    ("m2", "m2_rank", "m2_reliability_v2"),
    ("m3", "m3_rank", "m3_reliability_v2"),
    ("m4", "m4_rank", "m4_reliability_v2"),
]:
    valid = test[r_col].notna() & test[rel_col].notna() & (test[rel_col] > 0)
    w = test.loc[valid, rel_col]
    r = test.loc[valid, r_col]
    num.loc[valid] += r * w
    den.loc[valid] += w
    m_contr[m] = {
        "weight_sum": w.sum(),
        "valid_count": valid.sum(),
        "mean_rank": r.mean(),
        "auc": roc_auc_score(test.loc[valid, "is_cartel"], r) if test.loc[valid, "is_cartel"].nunique() > 1 else np.nan,
        "ap": average_precision_score(test.loc[valid, "is_cartel"], r) if test.loc[valid, "is_cartel"].nunique() > 1 else np.nan,
    }

test["rank_reliability_fusion"] = num / den
test.loc[den == 0, "rank_reliability_fusion"] = np.nan

print("\n--- Module Contributions & Standalone Test Performance ---")
total_den_sum = den.sum()
for m, d in m_contr.items():
    pct_weight = d["weight_sum"] / total_den_sum * 100
    print(f"{m.upper()}: count={d['valid_count']}, share of total weight={pct_weight:.2f}%, mean rank={d['mean_rank']:.4f}, AUC={d['auc']:.4f}, AP={d['ap']:.4f}")

# Number of available modules per record
test["n_modules"] = (
    test["m1_rank"].notna().astype(int) +
    test["m2_rank"].notna().astype(int) +
    test["m3_rank"].notna().astype(int) +
    test["m4_rank"].notna().astype(int)
)

print("\n--- Performance by Number of Available Modules ---")
for n_mod in sorted(test["n_modules"].unique()):
    sub = test[test["n_modules"] == n_mod]
    valid_sub = sub[sub["rank_reliability_fusion"].notna()]
    c_count = valid_sub["is_cartel"].sum()
    if len(valid_sub) > 0 and valid_sub["is_cartel"].nunique() > 1:
        auc = roc_auc_score(valid_sub["is_cartel"], valid_sub["rank_reliability_fusion"])
        ap = average_precision_score(valid_sub["is_cartel"], valid_sub["rank_reliability_fusion"])
    else:
        auc = np.nan
        ap = np.nan
    rate_str = f"{c_count/len(valid_sub):.2%}" if len(valid_sub) > 0 else "N/A"
    print(f"{n_mod} modules available: {len(sub):,} total ({len(valid_sub):,} scored), {c_count} cartels ({rate_str}), AUC={auc:.4f}, AP={ap:.4f}")

# Module Dilution Analysis: Tenders where one module is very high (>0.85) but final score is dragged down
print("\n--- Dilution Analysis ---")
for m in ["m1", "m2", "m3", "m4"]:
    high_m = test[(test[f"{m}_rank"] >= 0.85) & test["rank_reliability_fusion"].notna()]
    mean_final = high_m["rank_reliability_fusion"].mean()
    dragged = (high_m["rank_reliability_fusion"] < 0.70).sum()
    print(f"When {m.upper()} rank >= 0.85 (N={len(high_m)}, Cartels={high_m['is_cartel'].sum()}): mean final score = {mean_final:.4f}, dragged below 0.70 = {dragged}/{len(high_m)} ({dragged/len(high_m):.1%})")

# Subgroup distribution of False Positives at Top 5% screening
top5_cutoff = test["rank_reliability_fusion"].quantile(0.95)
top5_selected = test[test["rank_reliability_fusion"] >= top5_cutoff]
fps = top5_selected[top5_selected["is_cartel"] == 0]
tps = top5_selected[top5_selected["is_cartel"] == 1]
print(f"\n--- Top 5% Screening Breakdown (Score >= {top5_cutoff:.4f}) ---")
print(f"Total selected: {len(top5_selected)}, TPs: {len(tps)}, FPs: {len(fps)}")

print("\nTop FP Countries:")
print(fps["country"].value_counts().head(5))
print("\nTop TP Countries:")
print(tps["country"].value_counts().head(5))

print("\nTop FP Tender Years:")
print(fps["tender_year"].value_counts().sort_index())
print("\nTop TP Tender Years:")
print(tps["tender_year"].value_counts().sort_index())

print("\nTop FP CPV-2:")
print(fps["main_cpv_2"].value_counts().head(5))
print("\nTop TP CPV-2:")
print(tps["main_cpv_2"].value_counts().head(5))
