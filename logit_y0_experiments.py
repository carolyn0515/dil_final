# -*- coding: utf-8 -*-
"""
Logistic regression experiments using the SAME Batch built by build_dataset():
- Task: predict ΔPHQ (0=improved, 1=same, 2=worse)
- Experiments:
  A) y0(PHQ) only
  B) checks summary only
  C) y0(PHQ) + checks summary

Outputs (under outdir):
- logit_metrics.json
- logit_coef_y0_phq_only.csv
- logit_coef_checks_only.csv
- logit_coef_y0_phq_plus_checks.csv
- cm_logit_y0_phq_only.png
- cm_logit_checks_only.png
- cm_logit_y0_phq_plus_checks.png
"""

import os
import json
import numpy as np
import pandas as pd

from typing import Dict, Any, List, Tuple

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ---------------- utils ----------------

def ensure_dir(d: str) -> str:
    os.makedirs(d, exist_ok=True)
    return d

def _eval_multiclass(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
    labels = [0, 1, 2]
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average="macro", zero_division=0)
    rec = recall_score(y_true, y_pred, average="macro", zero_division=0)
    f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    prec_each = precision_score(y_true, y_pred, average=None, labels=labels, zero_division=0).tolist()
    rec_each  = recall_score(y_true, y_pred, average=None, labels=labels, zero_division=0).tolist()
    return dict(
        acc=float(acc),
        precision=float(prec),
        recall=float(rec),
        f1=float(f1),
        precision_each=prec_each,
        recall_each=rec_each,
        n=int(len(y_true)),
        cm=cm.tolist(),
    )

def _save_cm(cm: np.ndarray, outpath: str, title: str):
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["improved","same","worse"])
    disp.plot(values_format="d", cmap="Blues")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(outpath, dpi=160)
    plt.close()


# ---------------- feature builders ----------------

def build_features_y0_phq_only(batch) -> Tuple[np.ndarray, List[str]]:
    """
    y0 features (PHQ only).
    Keep it minimal + interpretable.

    Features:
      - y0_phq
      - y0_phq^2  (optional nonlinearity, still interpretable)
    """
    y0_phq = batch.y0_phq.detach().cpu().numpy().astype(float)

    X = np.stack([
        y0_phq,
        y0_phq ** 2,
    ], axis=1)

    cols = ["y0_phq", "y0_phq_sq"]
    return X, cols


def build_features_checks_summary(batch, use_z: bool = True) -> Tuple[np.ndarray, List[str]]:
    """
    weekly checks summary features from sequence C (z-scored) or Craw (raw):
    For each of 6 channels:
      - mean, std, min, max
      - slope over time (linear trend)
      - change_rate: mean(|Δ|) over time
    Total: 6 channels * 6 stats = 36 features
    """
    if use_z:
        C = batch.C.detach().cpu().numpy()     # (B,T,6) z-scored
        prefix = "Cz"
        check_names = batch.stats.get("check_basenames", None) or batch.stats.get("check_cols", None)
    else:
        C = batch.Craw.detach().cpu().numpy()  # (B,T,6) raw
        prefix = "Craw"
        check_names = batch.stats.get("check_basenames", None) or batch.stats.get("check_cols", None)

    B, T, d = C.shape
    t = np.arange(T, dtype=float)
    t_center = t - t.mean()
    denom = (t_center**2).sum() + 1e-12

    feats = []
    cols = []

    for j in range(d):
        x = C[:, :, j]  # (B,T)

        mean = x.mean(axis=1)
        std  = x.std(axis=1)
        mn   = x.min(axis=1)
        mx   = x.max(axis=1)

        slope = (x * t_center[None, :]).sum(axis=1) / denom

        dx = np.diff(x, axis=1)
        chg = np.abs(dx).mean(axis=1)

        feats.extend([mean, std, mn, mx, slope, chg])

        cname = check_names[j] if (check_names is not None and j < len(check_names)) else f"check{j+1}"
        cols.extend([
            f"{prefix}_{cname}_mean",
            f"{prefix}_{cname}_std",
            f"{prefix}_{cname}_min",
            f"{prefix}_{cname}_max",
            f"{prefix}_{cname}_slope",
            f"{prefix}_{cname}_chg",
        ])

    X = np.stack(feats, axis=1)
    return X, cols


def build_features_y0_phq_plus_checks(batch, use_z: bool = True) -> Tuple[np.ndarray, List[str]]:
    Xy0, cy0 = build_features_y0_phq_only(batch)
    Xc,  cc  = build_features_checks_summary(batch, use_z=use_z)
    X = np.concatenate([Xy0, Xc], axis=1)
    cols = cy0 + cc
    return X, cols


# ---------------- training / evaluation ----------------

def fit_eval_logit_multiclass(
    X: np.ndarray,
    y: np.ndarray,
    idx_tr: np.ndarray,
    idx_val: np.ndarray,
    outdir: str,
    tag: str,
    class_weight: str = "balanced",
    C: float = 1.0,
    max_iter: int = 2000,
) -> Dict[str, Any]:
    """
    Multinomial logistic regression for 3-class ΔPHQ.
    """
    OUT = ensure_dir(outdir)

    Xtr, Xva = X[idx_tr], X[idx_val]
    ytr, yva = y[idx_tr], y[idx_val]

    clf = LogisticRegression(
        multi_class="multinomial",
        solver="lbfgs",
        class_weight=class_weight,
        C=C,
        max_iter=max_iter,
        n_jobs=None,
        random_state=42,
    )
    clf.fit(Xtr, ytr)

    yhat_tr = clf.predict(Xtr)
    yhat_va = clf.predict(Xva)

    met_tr = _eval_multiclass(ytr, yhat_tr)
    met_va = _eval_multiclass(yva, yhat_va)

    cm_va = np.array(met_va["cm"])
    _save_cm(cm_va, os.path.join(OUT, f"cm_{tag}.png"), title=f"ΔPHQ Logistic ({tag})")

    return dict(
        tag=tag,
        train=met_tr,
        val=met_va,
        coef=clf.coef_.tolist(),   # (3, n_features)
        intercept=clf.intercept_.tolist(),
    )


def coef_table_multiclass(coef: np.ndarray, cols: List[str]) -> pd.DataFrame:
    """
    coef: (3, n_features) for classes [0,1,2]
    return df with columns: feature, beta_improved, beta_same, beta_worse
    """
    df = pd.DataFrame({
        "feature": cols,
        "beta_improved": coef[0],
        "beta_same": coef[1],
        "beta_worse": coef[2],
    })
    df["max_abs_beta"] = np.max(np.abs(coef), axis=0)
    df = df.sort_values("max_abs_beta", ascending=False).drop(columns=["max_abs_beta"])
    return df


def run_phq_logit_experiments_3way(
    batch,
    outdir: str,
    use_z_checks: bool = True,
    C: float = 1.0,
) -> Dict[str, Any]:
    """
    Runs 3 baselines for ΔPHQ prediction:
      A) y0(PHQ) only
      B) checks only
      C) y0(PHQ) + checks
    """
    OUT = ensure_dir(outdir)

    # target
    y = batch.dY_phq.detach().cpu().numpy().astype(int)
    valid = (y >= 0)

    # same split indices
    idx_tr = np.array([i for i in batch.idx_tr if valid[i]], dtype=int)
    idx_val = np.array([i for i in batch.idx_val if valid[i]], dtype=int)

    # A) y0(PHQ) only
    X_a, cols_a = build_features_y0_phq_only(batch)
    res_a = fit_eval_logit_multiclass(
        X_a, y, idx_tr, idx_val, outdir=OUT,
        tag="logit_y0_phq_only",
        class_weight="balanced",
        C=C,
    )
    coef_table_multiclass(np.array(res_a["coef"]), cols_a).to_csv(
        os.path.join(OUT, "logit_coef_y0_phq_only.csv"), index=False
    )

    # B) checks only
    X_b, cols_b = build_features_checks_summary(batch, use_z=use_z_checks)
    res_b = fit_eval_logit_multiclass(
        X_b, y, idx_tr, idx_val, outdir=OUT,
        tag="logit_checks_only",
        class_weight="balanced",
        C=C,
    )
    coef_table_multiclass(np.array(res_b["coef"]), cols_b).to_csv(
        os.path.join(OUT, "logit_coef_checks_only.csv"), index=False
    )

    # C) y0(PHQ) + checks
    X_c, cols_c = build_features_y0_phq_plus_checks(batch, use_z=use_z_checks)
    res_c = fit_eval_logit_multiclass(
        X_c, y, idx_tr, idx_val, outdir=OUT,
        tag="logit_y0_phq_plus_checks",
        class_weight="balanced",
        C=C,
    )
    coef_table_multiclass(np.array(res_c["coef"]), cols_c).to_csv(
        os.path.join(OUT, "logit_coef_y0_phq_plus_checks.csv"), index=False
    )

    summary = {
        "A_y0_phq_only": res_a,
        "B_checks_only": res_b,
        "C_y0_phq_plus_checks": res_c,
        "notes": {
            "target": "ΔPHQ (0=improved,1=same,2=worse)",
            "y0": "PHQ only (y0_phq, y0_phq^2)",
            "checks_input": "batch.C (z-scored)" if use_z_checks else "batch.Craw (raw)",
            "logit": "multinomial, class_weight=balanced",
        },
    }

    with open(os.path.join(OUT, "logit_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("\n[LOGIT-3WAY] Done. Saved under:", OUT)
    print("  - logit_metrics.json")
    print("  - logit_coef_y0_phq_only.csv")
    print("  - logit_coef_checks_only.csv")
    print("  - logit_coef_y0_phq_plus_checks.csv")
    print("  - cm_logit_y0_phq_only.png")
    print("  - cm_logit_checks_only.png")
    print("  - cm_logit_y0_phq_plus_checks.png")
    return summary


# ---------------- main (example) ----------------
if __name__ == "__main__":
    raise SystemExit(
        "Import and run from your pipeline:\n"
        "  from logit_y0_experiments import run_phq_logit_experiments_3way\n"
        "  summary = run_phq_logit_experiments_3way(batch, outdir='figs_phq/logit_3way')\n"
    )
