# analyze_high_phq_subgroup.py
# -*- coding: utf-8 -*-
"""
PHQ high-heavy subgroup 분석 (cohort-level threshold)

목적:
- PHQ_n_high_pre가 높은 사람들(high-heavy)을 cohort 기준으로 정의
- 그 중 improved vs non-improved(same+worse) 비교
- check values / services 차이를 분석

핵심 원칙:
1) threshold는 train/val 나누기 전 "전체 cohort" 기준
2) batch에 실제 들어간 환자만 사용
3) PHQ-path feature는 NO-LAST 규칙 유지
"""

import numpy as np
import pandas as pd
from scipy.stats import ttest_ind

# ===============================
# utils
# ===============================

def to_numpy(x):
    if hasattr(x, "detach"):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def cliff_delta(x, y):
    """Cliff's delta (효과크기, 비모수)"""
    x = np.asarray(x)
    y = np.asarray(y)
    nx, ny = len(x), len(y)
    gt = sum(xx > yy for xx in x for yy in y)
    lt = sum(xx < yy for xx in x for yy in y)
    return (gt - lt) / (nx * ny)


# ===============================
# PHQ high-heavy subgroup 정의
# ===============================

def build_high_phq_subgroup(
    batch,
    raw2_path,
    phq_srvy_name="PHQ-9",
    high_cut=2,
    quantile=0.8,
):
    """
    cohort-level 기준으로 PHQ high-heavy subgroup 생성

    Returns
    -------
    dict with:
      - high_heavy_ids
      - improved_ids
      - non_improved_ids
      - summary_df
    """

    # ---- batch에 실제 들어간 환자 ----
    pids = list(batch.stats["patients"])
    pids_set = set(pids)

    # ---- dY (0=improved, 1=same, 2=worse) ----
    dY = to_numpy(batch.dY_phq)
    pid2dy = dict(zip(pids, dY))

    # ---- raw2 로드 ----
    raw2 = pd.read_csv(raw2_path)
    raw2["srvy_result"] = pd.to_numeric(raw2["srvy_result"], errors="coerce")

    phq = raw2[
        (raw2["menti_seq"].isin(pids_set)) &
        (raw2["srvy_name"] == phq_srvy_name)
    ].copy()

    # ---- high 판정 ----
    phq["is_high"] = phq["srvy_result"] >= high_cut

    # ---- 사람별 high count ----
    high_cnt = phq.groupby("menti_seq")["is_high"].sum()

    # ---- cohort 기준 threshold ----
    thr = high_cnt.quantile(quantile)

    high_heavy_ids = high_cnt[high_cnt >= thr].index.tolist()

    improved_ids = [pid for pid in high_heavy_ids if pid2dy.get(pid) == 0]
    non_improved_ids = [pid for pid in high_heavy_ids if pid2dy.get(pid) in (1, 2)]

    summary = pd.DataFrame({
        "metric": [
            "total_patients",
            "high_heavy_threshold",
            "n_high_heavy",
            "n_improved",
            "n_non_improved",
        ],
        "value": [
            len(pids),
            float(thr),
            len(high_heavy_ids),
            len(improved_ids),
            len(non_improved_ids),
        ]
    })

    return {
        "high_heavy_ids": high_heavy_ids,
        "improved_ids": improved_ids,
        "non_improved_ids": non_improved_ids,
        "summary_df": summary,
    }


# ===============================
# check / service 비교
# ===============================

def compare_features_between_groups(
    raw1_df,
    improved_ids,
    non_improved_ids,
    feature_cols,
):
    """
    improved vs non-improved 비교
    """
    rows = []

    for col in feature_cols:
        x = raw1_df.loc[raw1_df["menti_seq"].isin(improved_ids), col].dropna()
        y = raw1_df.loc[raw1_df["menti_seq"].isin(non_improved_ids), col].dropna()

        if len(x) < 5 or len(y) < 5:
            continue

        t_stat, p_val = ttest_ind(x, y, equal_var=False)
        cd = cliff_delta(x, y)

        rows.append({
            "feature": col,
            "mean_improved": x.mean(),
            "mean_non_improved": y.mean(),
            "mean_diff": x.mean() - y.mean(),
            "p_value": p_val,
            "cliffs_delta": cd,
            "n_improved": len(x),
            "n_non_improved": len(y),
        })

    return (
        pd.DataFrame(rows)
        .sort_values("cliffs_delta", key=lambda s: np.abs(s), ascending=False)
        .reset_index(drop=True)
    )


# ===============================
# main runner
# ===============================

def run_high_phq_subgroup_analysis(
    batch,
    raw1_path,
    raw2_path,
    phq_srvy_name="PHQ-9",
    high_cut=2,
    quantile=0.8,
    check_cols=None,
    service_cols=None,
):
    """
    전체 파이프라인 실행
    """

    # ---- subgroup ----
    sub = build_high_phq_subgroup(
        batch=batch,
        raw2_path=raw2_path,
        phq_srvy_name=phq_srvy_name,
        high_cut=high_cut,
        quantile=quantile,
    )

    print("\n[SUBGROUP SUMMARY]")
    print(sub["summary_df"].to_string(index=False))

    # ---- raw1 로드 ----
    raw1 = pd.read_csv(raw1_path)
    raw1 = raw1[raw1["menti_seq"].isin(batch.stats["patients"])]

    results = {}

    if check_cols:
        print("\n[CHECK VALUE COMPARISON]")
        df_check = compare_features_between_groups(
            raw1,
            sub["improved_ids"],
            sub["non_improved_ids"],
            check_cols,
        )
        results["checks"] = df_check
        print(df_check.head(10).to_string(index=False))

    if service_cols:
        print("\n[SERVICE COMPARISON]")
        df_service = compare_features_between_groups(
            raw1,
            sub["improved_ids"],
            sub["non_improved_ids"],
            service_cols,
        )
        results["services"] = df_service
        print(df_service.head(10).to_string(index=False))

    return {
        "subgroup": sub,
        "results": results,
    }


# ===============================
# example usage (not auto-run)
# ===============================
if __name__ == "__main__":
    print("Import and run run_high_phq_subgroup_analysis() from notebook.")
