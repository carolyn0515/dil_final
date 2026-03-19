# logit_phqpath_experiments.py
# -*- coding: utf-8 -*-
"""
PHQ-path (raw_data2.csv) 기반 로지스틱 실험 (누출(leakage) 방지 버전)

- batch(너의 build_dataset 결과)와 raw_data2를 매칭해서 feature를 만들고,
- multinomial logistic regression(3-class: improved/same/worse) 실험을 돌린다.

✅ 핵심: PHQ-path feature 만들 때 "마지막 PHQ 값"을 절대 쓰지 않는다.
- vals_pre = vals[:-1] 로 마지막 관측치를 제외하고 요약 feature 구성
- 따라서 last/delta/last_is_high/range/max 등 last를 통해 Δ를 암시할 수 있는 것 제거

실험 세트:
  (1) y0 only
  (2) checks only
  (3) y0 + checks
  (4) phq-path only (NO LAST)
  (5) phq-path + checks (NO LAST)
  (6) y0 + phq-path + checks (NO LAST)

추가(중요 피처 분석):
- 저장된 metrics_{tag}.json + feature_columns.json 기반으로 PHQ-path 피처 중요도(coef) 분석
- (옵션) X_val의 표준편차로 coef 스케일링한 "scaled_coef" 제공
- (옵션) PHQ-path 8개 피처 ablation(하나씩 제거)으로 f1 변화 확인

주의:
- batch 구조는 네 프로젝트 기준으로 추정(fallback 포함)
- raw2.csv는 최소 컬럼: menti_seq, srvy_name, srvy_result, reg_date 필요
"""

import os
import json
import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    confusion_matrix, ConfusionMatrixDisplay
)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ---------------- utils ----------------

def ensure_dir(d: str) -> str:
    os.makedirs(d, exist_ok=True)
    return d


def _to_list(x):
    if x is None:
        return []
    if isinstance(x, (list, tuple, set, np.ndarray, pd.Index)):
        return list(x)
    return [x]  # str이면 단일 원소 리스트로


def _as_numpy_1d(x):
    if hasattr(x, "detach"):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _safe_float(x):
    try:
        return float(x)
    except Exception:
        return np.nan


def get_batch_ids_in_order(batch):
    """
    (중요) raw_data2의 menti_seq와 매칭할 "환자 id 리스트"를 batch 순서(B dimension)대로 가져오기.
    - 가장 우선: batch.stats['patients']
    - 그 외: batch.ids / batch.pids / batch.patient_ids 등 fallback
    """
    if hasattr(batch, "stats") and isinstance(batch.stats, dict):
        if "patients" in batch.stats:
            return list(batch.stats["patients"])

    # optional: batch.ids 같은 필드가 있다면
    for k in ["ids", "pids", "pid", "patient_ids"]:
        if hasattr(batch, k):
            v = getattr(batch, k)
            if hasattr(v, "detach"):
                v = v.detach().cpu().numpy().tolist()
            return list(v)

    if hasattr(batch, "stats") and isinstance(batch.stats, dict):
        for k in ["id_list", "ids", "pid_list", "patient_ids"]:
            if k in batch.stats:
                return list(batch.stats[k])

    raise AttributeError(
        "batch에서 환자 id 리스트를 못 찾았어. "
        "batch.stats['patients'] 또는 batch.ids/pids/patient_ids 형태로 넣어줘."
    )


def _read_raw2(raw2_path: str) -> pd.DataFrame:
    df = pd.read_csv(raw2_path)
    # 기대 컬럼: menti_seq, srvy_name, srvy_result, reg_date
    if "reg_date" in df.columns:
        df["reg_date"] = pd.to_datetime(df["reg_date"], errors="coerce")
    else:
        df["reg_date"] = pd.NaT

    if "srvy_result" in df.columns:
        df["srvy_result"] = pd.to_numeric(df["srvy_result"], errors="coerce")
    else:
        df["srvy_result"] = np.nan

    return df


def _impute_nan_with_col_mean(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    col_mean = np.nanmean(X, axis=0)
    inds = np.where(np.isnan(X))
    if len(inds[0]) > 0:
        X[inds] = np.take(col_mean, inds[1])
    return X


def _safe_std(X: np.ndarray) -> np.ndarray:
    s = np.nanstd(X, axis=0)
    s[s == 0] = 1.0
    return s


def _load_json(p: str):
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def _save_json(p: str, obj):
    with open(p, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


# ---------------- feature builders ----------------

def build_features_y0_from_batch(batch):
    """y0_PHQ만 feature로 (단일 컬럼)."""
    y0 = _as_numpy_1d(batch.y0_phq).astype(float)
    X = y0.reshape(-1, 1)
    cols = ["y0_phq"]
    return X, cols


def build_features_checks_from_batch(batch, use_z_checks: bool = True):
    """
    weekly checks sequence를 환자 단위로 요약(평균/표준편차/최대/최소).
    - use_z_checks=True: batch.C (z-scored)
    - False: batch.Craw (raw)
    """
    C = batch.C if use_z_checks else batch.Craw
    C_np = _as_numpy_1d(C)  # (B,T,6) 가정

    mean = np.nanmean(C_np, axis=1)
    std  = np.nanstd(C_np, axis=1)
    mx   = np.nanmax(C_np, axis=1)
    mn   = np.nanmin(C_np, axis=1)

    X = np.concatenate([mean, std, mx, mn], axis=1)  # (B, 6*4)

    base = None
    if hasattr(batch, "stats") and isinstance(batch.stats, dict):
        base = batch.stats.get("check_basenames", None) or batch.stats.get("check_cols", None)

    if base is None:
        base = [f"check{i+1}" for i in range(C_np.shape[-1])]
    base = list(base)

    cols = []
    for stat, _arr in [("mean", mean), ("std", std), ("max", mx), ("min", mn)]:
        for name in base:
            cols.append(f"{name}_{stat}")

    return X, cols


def build_features_phq_path_from_raw2_no_last(
    batch,
    raw2_path: str,
    phq_srvy_names,
    high_thr=2.0,
    min_points_pre: int = 2,
):
    """
    누출 방지 PHQ-path feature:
    raw_data2에서 PHQ 설문 "경로"를 만들되, 마지막 관측치(vals[-1])는 절대 사용하지 않는다.

    phq_srvy_names: ['PHQ-9'] 같은 list-like 권장 (str도 가능; 내부에서 list로 변환)
    feature (환자당, last 제외 vals_pre = vals[:-1]):
      - phq_n_obs_pre
      - phq_first_pre
      - phq_mean_pre
      - phq_std_pre
      - phq_min_pre
      - phq_slope_pre
      - phq_n_high_pre
      - phq_frac_high_pre

    조건:
      - 원본 vals가 0/1개면: 전부 NaN
      - vals_pre가 min_points_pre 미만이면: 전부 NaN (기본 2)
    """
    pids = get_batch_ids_in_order(batch)
    names = _to_list(phq_srvy_names)

    df = _read_raw2(raw2_path)

    required = ["menti_seq", "srvy_name", "srvy_result"]
    for c in required:
        if c not in df.columns:
            raise ValueError(f"raw2에 '{c}' 컬럼이 없어. 현재 컬럼: {list(df.columns)}")

    df = df.dropna(subset=["srvy_name"]).copy()
    d = df[df["srvy_name"].isin(names)].copy()

    feats = []
    for pid in pids:
        g = d[d["menti_seq"] == pid].sort_values("reg_date")
        vals = g["srvy_result"].dropna().to_numpy(dtype=float)

        # 마지막 제외를 하려면 최소 2개는 있어야 함
        if len(vals) < 2:
            feats.append([np.nan] * 8)
            continue

        vals_pre = vals[:-1]  # last 제거
        if len(vals_pre) < min_points_pre:
            feats.append([np.nan] * 8)
            continue

        n = len(vals_pre)
        first = float(vals_pre[0])
        mean  = float(np.mean(vals_pre))
        std   = float(np.std(vals_pre)) if n > 1 else 0.0
        vmin  = float(np.min(vals_pre))
        slope = float((vals_pre[-1] - vals_pre[0]) / (n - 1)) if n > 1 else 0.0
        n_high = int((vals_pre >= high_thr).sum())
        frac_high = float(n_high / n)

        feats.append([
            float(n),
            _safe_float(first),
            _safe_float(mean),
            _safe_float(std),
            _safe_float(vmin),
            _safe_float(slope),
            float(n_high),
            _safe_float(frac_high),
        ])

    X = np.asarray(feats, dtype=float)
    cols = [
        "phq_n_obs_pre",
        "phq_first_pre",
        "phq_mean_pre",
        "phq_std_pre",
        "phq_min_pre",
        "phq_slope_pre",
        "phq_n_high_pre",
        "phq_frac_high_pre",
    ]
    return X, cols


# ---------------- modeling / eval ----------------

def fit_eval_logit_multiclass(X, y, idx_tr, idx_val, outdir: str, tag: str, C: float = 1.0):
    """
    multinomial logistic regression 학습/평가 + confusion matrix 저장
    """
    OUT = ensure_dir(outdir)

    X = _impute_nan_with_col_mean(X)
    y = np.asarray(y, dtype=int)

    idx_tr = np.asarray(idx_tr, dtype=int)
    idx_val = np.asarray(idx_val, dtype=int)

    X_tr, X_va = X[idx_tr], X[idx_val]
    y_tr, y_va = y[idx_tr], y[idx_val]

    clf = LogisticRegression(
        multi_class="multinomial",
        solver="lbfgs",
        C=float(C),
        max_iter=2000,
        n_jobs=None,
    )
    clf.fit(X_tr, y_tr)
    pred = clf.predict(X_va)

    acc  = float(accuracy_score(y_va, pred))
    prec = float(precision_score(y_va, pred, average="macro", zero_division=0))
    rec  = float(recall_score(y_va, pred, average="macro", zero_division=0))
    f1   = float(f1_score(y_va, pred, average="macro", zero_division=0))

    prec_each = precision_score(y_va, pred, average=None, labels=[0, 1, 2], zero_division=0).tolist()
    rec_each  = recall_score(y_va, pred, average=None, labels=[0, 1, 2], zero_division=0).tolist()

    cm = confusion_matrix(y_va, pred, labels=[0, 1, 2])

    disp = ConfusionMatrixDisplay(cm, display_labels=["improved", "same", "worse"])
    disp.plot(values_format="d", cmap="Blues")
    plt.title(f"{tag} (val)")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, f"cm_{tag}.png"), dpi=150)
    plt.close()

    res = dict(
        tag=tag,
        acc=acc,
        precision=prec,
        recall=rec,
        f1=f1,
        precision_each=prec_each,
        recall_each=rec_each,
        n_val=int(len(y_va)),
        cm=cm.tolist(),
        coef_=clf.coef_.tolist(),
        intercept_=clf.intercept_.tolist(),
    )

    with open(os.path.join(OUT, f"metrics_{tag}.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2, ensure_ascii=False)

    print(f"[{tag}] acc={acc:.4f} prec={prec:.6f} rec={rec:.6f} f1={f1:.6f} (n={len(y_va)})")
    return res


def summarize_to_dataframe(summary: dict) -> pd.DataFrame:
    """
    summary(dict of dict)를 보기 좋게 DataFrame으로 반환.
    """
    rows = []
    for k, v in summary.items():
        rows.append(dict(
            exp=k,
            tag=v.get("tag"),
            acc=v.get("acc"),
            prec=v.get("precision"),
            rec=v.get("recall"),
            f1=v.get("f1"),
            n=v.get("n_val"),
        ))
    df = pd.DataFrame(rows)
    # f1 기준 내림차순
    df = df.sort_values(["f1", "acc"], ascending=[False, False]).reset_index(drop=True)
    return df


# ---------------- 중요 피처 분석 (PHQ-path) ----------------

def analyze_phq_feature_importance(
    outdir: str,
    exp_tag: str,
    X_val_phq: np.ndarray = None,   # (n_val, n_phq_features) 제공 시 scaled_coef 계산
    topk: int = 10,
    class_names=("improved", "same", "worse"),
):
    """
    outdir/feature_columns.json + outdir/metrics_{exp_tag}.json 를 사용해
    PHQ-path 피처 중요도(계수 기반)를 테이블로 만든다.

    - phq_path_plus_checks: concat([X_p, X_c]) => PHQ 피처는 앞쪽 8개
    - full_y0_path_checks : concat([X_y0, X_p, X_c]) => PHQ 피처는 [1:9]
    - phq_path_only       : PHQ 피처만 => 전체가 8개
    """
    feat_cols = _load_json(os.path.join(outdir, "feature_columns.json"))
    phq_cols = feat_cols["phq_path_no_last"]
    n_phq = len(phq_cols)

    metrics = _load_json(os.path.join(outdir, f"metrics_{exp_tag}.json"))
    coef = np.asarray(metrics["coef_"], dtype=float)  # (3, n_total)

    # PHQ slice 결정(네 기존 concat 순서를 "건들지 말라" 조건 그대로 가정)
    if "full_y0_path_checks" in exp_tag:
        sl = slice(1, 1 + n_phq)
    else:
        sl = slice(0, n_phq)

    coef_phq = coef[:, sl]  # (3, n_phq)

    # scaled coef (optional)
    if X_val_phq is not None:
        X_val_phq = _impute_nan_with_col_mean(X_val_phq)
        std = _safe_std(X_val_phq)
        scaled = coef_phq * std[None, :]
    else:
        scaled = None

    # raw rows
    rows = []
    for ci, cname in enumerate(class_names):
        for j, fn in enumerate(phq_cols):
            rows.append({
                "class": cname,
                "feature": fn,
                "coef": float(coef_phq[ci, j]),
                "abs_coef": float(abs(coef_phq[ci, j])),
                "scaled_coef": float(scaled[ci, j]) if scaled is not None else np.nan,
                "abs_scaled_coef": float(abs(scaled[ci, j])) if scaled is not None else np.nan,
            })
    df_raw = pd.DataFrame(rows)

    # overall importance: mean(|coef|) across classes
    key = "abs_scaled_coef" if scaled is not None else "abs_coef"
    df_overall = (
        df_raw.groupby("feature")[[key]]
              .mean()
              .rename(columns={key: "importance"})
              .sort_values("importance", ascending=False)
              .reset_index()
    )

    # per-class topk
    df_per_class_top = (
        df_raw.sort_values(key, ascending=False)
              .groupby("class")
              .head(topk)
              .reset_index(drop=True)
    )

    # contrast: worse - improved (방향성)
    df_contrast = pd.DataFrame({
        "feature": phq_cols,
        "coef_worse_minus_improved": (coef_phq[2] - coef_phq[0]).astype(float),
    })
    df_contrast["abs"] = df_contrast["coef_worse_minus_improved"].abs()
    df_contrast = df_contrast.sort_values("abs", ascending=False).reset_index(drop=True)

    return {
        "overall": df_overall,
        "per_class_top": df_per_class_top,
        "contrast_worse_vs_improved": df_contrast,
        "raw": df_raw,
        "meta": {"exp_tag": exp_tag, "used_scaled": (scaled is not None), "n_phq": n_phq}
    }


def save_phq_importance_reports(outdir: str, exp_tag: str, X_val_phq: np.ndarray = None, topk: int = 10):
    """
    중요도 분석 결과를 csv/json로 저장.
    """
    rep = analyze_phq_feature_importance(
        outdir=outdir,
        exp_tag=exp_tag,
        X_val_phq=X_val_phq,
        topk=topk,
    )

    imp_dir = ensure_dir(os.path.join(outdir, "phq_importance"))
    rep["overall"].to_csv(os.path.join(imp_dir, f"overall_{exp_tag}.csv"), index=False, encoding="utf-8")
    rep["per_class_top"].to_csv(os.path.join(imp_dir, f"per_class_top_{exp_tag}.csv"), index=False, encoding="utf-8")
    rep["contrast_worse_vs_improved"].to_csv(os.path.join(imp_dir, f"contrast_{exp_tag}.csv"), index=False, encoding="utf-8")
    rep["raw"].to_csv(os.path.join(imp_dir, f"raw_{exp_tag}.csv"), index=False, encoding="utf-8")

    # meta + top few rows만 json로
    small = {
        "meta": rep["meta"],
        "overall_top20": rep["overall"].head(20).to_dict(orient="records"),
        "contrast_top20": rep["contrast_worse_vs_improved"].head(20).to_dict(orient="records"),
    }
    _save_json(os.path.join(imp_dir, f"summary_{exp_tag}.json"), small)

    print(f"[PHQ-IMPORTANCE] saved to: {imp_dir}")
    print("\n[Top overall]")
    print(rep["overall"].head(12).to_string(index=False))
    print("\n[Top contrast (worse - improved)]")
    print(rep["contrast_worse_vs_improved"].head(12).to_string(index=False))

    return rep


def ablation_drop_one_feature_phq_only(
    X_p: np.ndarray,
    y: np.ndarray,
    idx_tr,
    idx_val,
    outdir: str,
    C_logit: float = 1.0,
    feature_names=None,
    tag_prefix: str = "ablate_phq_path_only",
):
    """
    PHQ-path feature들(보통 8개)을 하나씩 제거하면서 성능 변화(f1_delta)를 확인.
    - "정말 중요한 피처"를 성능 변화로 검증하는 방식
    """
    ensure_dir(outdir)

    X_p = _impute_nan_with_col_mean(X_p)
    y = np.asarray(y, dtype=int)
    idx_tr = np.asarray(idx_tr, dtype=int)
    idx_val = np.asarray(idx_val, dtype=int)

    if feature_names is None:
        feature_names = [f"f{i}" for i in range(X_p.shape[1])]

    base = fit_eval_logit_multiclass(X_p, y, idx_tr, idx_val, outdir, f"{tag_prefix}_BASE", C=C_logit)
    base_f1 = float(base["f1"])

    rows = []
    for j, fn in enumerate(feature_names):
        keep = [k for k in range(X_p.shape[1]) if k != j]
        X_drop = X_p[:, keep]
        r = fit_eval_logit_multiclass(X_drop, y, idx_tr, idx_val, outdir, f"{tag_prefix}_DROP_{fn}", C=C_logit)
        rows.append({
            "dropped_feature": fn,
            "f1_after_drop": float(r["f1"]),
            "f1_delta_vs_base": float(r["f1"]) - base_f1,
            "acc_after_drop": float(r["acc"]),
        })

    df = pd.DataFrame(rows).sort_values("f1_delta_vs_base").reset_index(drop=True)
    df.to_csv(os.path.join(outdir, f"{tag_prefix}_drop1_summary.csv"), index=False, encoding="utf-8")

    print("\n[ABLATION] drop-one feature summary (more negative = more important)")
    print(df.head(12).to_string(index=False))
    return df


# ---------------- main runner ----------------

def run_phq_logit_experiments_with_raw2_phqpath(
    batch,
    raw2_path: str,
    outdir: str,
    phq_srvy_names=("PHQ-9",),
    use_z_checks: bool = True,
    C_logit: float = 1.0,
    high_thr: float = 10.0,
    min_points_pre: int = 2,
    # ---- 추가 옵션(중요도/ablation) ----
    run_importance: bool = True,
    importance_topk: int = 10,
    run_ablation_phq_only: bool = False,
):
    """
    실험 세트:
      (1) y0 only
      (2) checks only
      (3) y0 + checks
      (4) phq-path only (NO LAST)
      (5) phq-path + checks (NO LAST)
      (6) y0 + phq-path + checks (NO LAST)
    """
    OUT = ensure_dir(outdir)

    # labels / split
    y = _as_numpy_1d(batch.dY_phq).astype(int)
    idx_tr = np.asarray(batch.idx_tr)
    idx_val = np.asarray(batch.idx_val)

    # features
    X_y0, cols_y0 = build_features_y0_from_batch(batch)
    X_c, cols_c = build_features_checks_from_batch(batch, use_z_checks=use_z_checks)
    X_p, cols_p = build_features_phq_path_from_raw2_no_last(
        batch,
        raw2_path,
        phq_srvy_names=phq_srvy_names,
        high_thr=high_thr,
        min_points_pre=min_points_pre,
    )

    # run
    summary = {}
    summary["y0_only"] = fit_eval_logit_multiclass(X_y0, y, idx_tr, idx_val, OUT, "logit_y0_only", C=C_logit)
    summary["checks_only"] = fit_eval_logit_multiclass(X_c, y, idx_tr, idx_val, OUT, "logit_checks_only", C=C_logit)
    summary["y0_plus_checks"] = fit_eval_logit_multiclass(
        np.concatenate([X_y0, X_c], axis=1), y, idx_tr, idx_val, OUT, "logit_y0_plus_checks", C=C_logit
    )
    summary["phq_path_only"] = fit_eval_logit_multiclass(
        X_p, y, idx_tr, idx_val, OUT, "logit_phq_path_only_NO_LAST", C=C_logit
    )
    summary["phq_path_plus_checks"] = fit_eval_logit_multiclass(
        np.concatenate([X_p, X_c], axis=1), y, idx_tr, idx_val, OUT, "logit_phq_path_plus_checks_NO_LAST", C=C_logit
    )
    summary["full_y0_path_checks"] = fit_eval_logit_multiclass(
        np.concatenate([X_y0, X_p, X_c], axis=1), y, idx_tr, idx_val, OUT, "logit_full_y0_path_checks_NO_LAST", C=C_logit
    )

    # feature columns 저장(나중에 해석용)
    with open(os.path.join(OUT, "feature_columns.json"), "w", encoding="utf-8") as f:
        json.dump(
            dict(
                y0=cols_y0,
                checks=cols_c,
                phq_path_no_last=cols_p,
            ),
            f,
            indent=2,
            ensure_ascii=False,
        )

    # 요약 테이블 저장
    df_sum = summarize_to_dataframe(summary)
    df_sum.to_csv(os.path.join(OUT, "summary_table.csv"), index=False, encoding="utf-8")
    print("\n=== Summary (sorted by f1) ===")
    print(df_sum.to_string(index=False))

    # ---------------- 중요도 분석 ----------------
    if run_importance:
        # X_val_phq(=PHQ-path의 val 부분) 만들어서 scaled_coef까지 계산
        idx_val_np = np.asarray(idx_val, dtype=int)
        X_val_phq = X_p[idx_val_np]  # (n_val, n_phq_features)

        # phq_path_only / phq_path_plus_checks / full_y0_path_checks 3개 보고서 저장
        for tag in [
            "logit_phq_path_only_NO_LAST",
            "logit_phq_path_plus_checks_NO_LAST",
            "logit_full_y0_path_checks_NO_LAST",
        ]:
            save_phq_importance_reports(
                outdir=OUT,
                exp_tag=tag,
                X_val_phq=X_val_phq,
                topk=importance_topk,
            )

    # ---------------- Ablation(옵션) ----------------
    if run_ablation_phq_only:
        ab_dir = ensure_dir(os.path.join(OUT, "phq_ablation"))
        ablation_drop_one_feature_phq_only(
            X_p=X_p,
            y=y,
            idx_tr=idx_tr,
            idx_val=idx_val,
            outdir=ab_dir,
            C_logit=C_logit,
            feature_names=cols_p,
            tag_prefix="ablate_phq_path_only_NO_LAST",
        )

    return summary


if __name__ == "__main__":
    print("Run this module from your notebook by importing run_phq_logit_experiments_with_raw2_phqpath.")
