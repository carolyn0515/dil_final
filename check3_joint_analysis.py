# -*- coding: utf-8 -*-
"""
check3_joint_analysis_v2.py

Goal
-----
"service 연결" 직전 단계까지: time attention(a_t) × check3 channel attention(b_t) -> joint j_t를 만들고,
PHQ improved vs else 기준으로
(1) 언제 check3가 중요했는지 (mean curves, peak week),
(2) 집중도(entropy),
(3) check3 dynamics(level/change/volatility) 정렬(alignment)
+ (4) 수치적 검증: improved vs else 차이가 "얼마나" 나는지 + 통계적으로 유의한지
    - Mann–Whitney U test (two-sided, normal approx + tie correction)  [no scipy]
    - Cliff's delta (effect size)
    - Bootstrap 95% CI for median/mean difference

Inputs
------
- raw weekly CSV (raw_data1.csv)  : check*_value + service* + reg_date + menti_seq
- raw survey CSV (raw_data2.csv) : PHQ-9/P4/Loneliness longitudinal
- trained checkpoint (phq_multi_seq.ckpt) produced by approach3.py

Outputs (outdir)
----------------
- per_patient_summary_<split>.csv
- peak_window_stats_<split>.csv
- global_summary_check3_joint_<split>.txt
- stats_improved_vs_else_<split>.csv
- stats_improved_vs_else_<split>.txt
- Figures (*.png):
    * check3_a_mean_by_group_<split>.png
    * check3_b_mean_by_group_<split>.png
    * check3_j_mean_by_group_<split>.png
    * check3_peak_t_hist_by_group_<split>.png
    * check3_entropy_by_group_<split>.png
    * check3_dynamics_weighted_by_group_<split>.png
    * check3_global_mean_j_<split>.png

Usage
-----
python check3_joint_analysis_v2.py \
  --raw1 data/raw_data1.csv --raw2 data/raw_data2.csv \
  --ckpt_dir figs_phq_multi_attn --ckpt_name phq_multi_seq.ckpt \
  --outdir figs_check3_joint --split val --T 40 --check_name check3

Note
----
This script imports functions/classes from approach3.py, so keep it in the same folder or in PYTHONPATH.
"""

from __future__ import annotations

import argparse
import os
import math
from typing import Dict, Any, Tuple

import numpy as np
import pandas as pd
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---- import your existing pipeline ----
import approach3  # requires approach3.py on PYTHONPATH / same dir


# ----------------------------
# helpers
# ----------------------------

def _find_check_index(batch: "approach3.Batch", check_name: str) -> int:
    """
    Find channel index for check_name (e.g., "check3") from batch.stats["check_basenames"].
    Falls back to substring match.
    """
    names = batch.stats.get("check_basenames", batch.stats.get("check_cols", []))
    if not names:
        raise ValueError("Cannot find check names in batch.stats.")
    # exact match first
    for i, n in enumerate(names):
        if n == check_name:
            return i
    # try startswith / contains
    low = check_name.lower()
    for i, n in enumerate(names):
        if str(n).lower().startswith(low):
            return i
    for i, n in enumerate(names):
        if low in str(n).lower():
            return i
    raise ValueError(f"check_name='{check_name}' not found in check_basenames={names}")


def _safe_normalize_rows(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Row-wise normalize to sum=1. If row sum ~0, make uniform."""
    s = x.sum(axis=1, keepdims=True)
    out = np.empty_like(x, dtype=np.float64)
    bad = (s[:, 0] < eps)
    out[~bad] = x[~bad] / s[~bad]
    if bad.any():
        out[bad] = 1.0 / x.shape[1]
    return out


def _row_entropy(p: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Entropy per row: -sum p log p. p is assumed non-negative."""
    p = np.clip(p, eps, 1.0)
    return -(p * np.log(p)).sum(axis=1)


def _rolling_std(x: np.ndarray, window: int = 3) -> np.ndarray:
    """
    Simple trailing rolling std:
      vol[t] = std(x[max(0,t-window+1):t+1])
    """
    T = x.shape[0]
    out = np.zeros(T, dtype=np.float64)
    for t in range(T):
        s = max(0, t - window + 1)
        out[t] = float(np.std(x[s:t+1], ddof=0))
    return out


def _boxplot_by_group(ax, values0, values1, labels=("improved", "else"), ylabel=""):
    ax.boxplot([values0, values1], labels=list(labels), showfliers=False)
    ax.set_ylabel(ylabel)


# ----------------------------
# stats (no scipy dependency)
# ----------------------------

def _norm_cdf(z: float) -> float:
    """Standard normal CDF."""
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def _mann_whitney_u(x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    """
    Mann–Whitney U test (two-sided) using normal approximation with tie correction.
    Returns (U, p_two_sided).

    Notes:
    - Independent samples.
    - Average ranks for ties.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    x = x[np.isfinite(x)]
    y = y[np.isfinite(y)]
    n1, n2 = len(x), len(y)
    if n1 == 0 or n2 == 0:
        return float("nan"), float("nan")

    allv = np.concatenate([x, y])
    order = np.argsort(allv, kind="mergesort")
    ranks = np.empty_like(allv, dtype=np.float64)
    ranks[order] = np.arange(1, len(allv) + 1, dtype=np.float64)

    # average ranks for ties
    sorted_vals = allv[order]
    i = 0
    while i < len(sorted_vals):
        j = i + 1
        while j < len(sorted_vals) and sorted_vals[j] == sorted_vals[i]:
            j += 1
        if j - i > 1:
            avg = ranks[order[i:j]].mean()
            ranks[order[i:j]] = avg
        i = j

    r1 = ranks[:n1].sum()
    U1 = r1 - n1 * (n1 + 1) / 2.0
    U2 = n1 * n2 - U1
    U = min(U1, U2)

    mu_U = n1 * n2 / 2.0

    # tie correction variance
    _, counts = np.unique(allv, return_counts=True)
    tie_sum = np.sum(counts**3 - counts)
    N = n1 + n2
    sigma2 = (n1 * n2 / 12.0) * ((N + 1) - tie_sum / (N * (N - 1)))
    if sigma2 <= 0:
        return float(U), float("nan")
    sigma = math.sqrt(sigma2)

    # continuity correction
    z = (U - mu_U + 0.5) / sigma
    p = 2.0 * (1.0 - _norm_cdf(abs(z)))
    return float(U), float(p)


def _cliffs_delta(x: np.ndarray, y: np.ndarray) -> float:
    """Cliff's delta in [-1, 1]."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    x = x[np.isfinite(x)]
    y = y[np.isfinite(y)]
    if len(x) == 0 or len(y) == 0:
        return float("nan")
    X = x.reshape(-1, 1)
    Y = y.reshape(1, -1)
    return float(((X > Y).sum() - (X < Y).sum()) / (X.size * Y.size))


def _bootstrap_ci_diff(x: np.ndarray,
                       y: np.ndarray,
                       stat: str = "median",
                       n_boot: int = 5000,
                       seed: int = 42,
                       alpha: float = 0.05) -> Tuple[float, float, float]:
    """
    Bootstrap 95% CI for (stat(x) - stat(y)).
    Returns: (diff_hat, ci_lo, ci_hi)
    """
    rng = np.random.default_rng(seed)
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    x = x[np.isfinite(x)]
    y = y[np.isfinite(y)]
    if len(x) == 0 or len(y) == 0:
        return float("nan"), float("nan"), float("nan")

    if stat == "median":
        f = np.median
    elif stat == "mean":
        f = np.mean
    else:
        raise ValueError("stat must be 'median' or 'mean'")

    diff_hat = float(f(x) - f(y))
    boots = np.empty(n_boot, dtype=np.float64)
    for i in range(n_boot):
        xb = rng.choice(x, size=len(x), replace=True)
        yb = rng.choice(y, size=len(y), replace=True)
        boots[i] = f(xb) - f(yb)
    lo = float(np.quantile(boots, alpha / 2.0))
    hi = float(np.quantile(boots, 1.0 - alpha / 2.0))
    return diff_hat, lo, hi


def _iqr(a: np.ndarray) -> Tuple[float, float, float]:
    a = np.asarray(a, dtype=np.float64)
    a = a[np.isfinite(a)]
    if len(a) == 0:
        return float("nan"), float("nan"), float("nan")
    q1 = float(np.quantile(a, 0.25))
    q2 = float(np.quantile(a, 0.50))
    q3 = float(np.quantile(a, 0.75))
    return q1, q2, q3


def compute_group_stats(per_patient: pd.DataFrame,
                        outdir: str,
                        split: str,
                        seed: int = 42,
                        n_boot: int = 5000) -> Tuple[str, str]:
    """
    Compute numeric summaries + significance tests for improved vs else.
    Saves:
      - stats_improved_vs_else_<split>.csv
      - stats_improved_vs_else_<split>.txt
    Returns (csv_path, txt_path).
    """
    OUT = approach3.ensure_dir(outdir)
    g = per_patient["group_improved_else"].to_numpy()
    m0 = (g == 0)  # improved
    m1 = (g == 1)  # else

    metrics = [
        ("mu_lvl", "Attention-weighted level (P(check3=1) during critical windows)"),
        ("mu_change", "Attention-weighted change (0↔1 transitions)"),
        ("mu_vol", "Attention-weighted volatility (rolling std, window=vol_window)"),
        ("entropy", "Entropy of normalized joint attention"),
        ("peak_t", "Peak week t* of joint attention"),
    ]

    rows = []
    for col, desc in metrics:
        x0 = per_patient.loc[m0, col].to_numpy(dtype=float)
        x1 = per_patient.loc[m1, col].to_numpy(dtype=float)

        q10, med0, q30 = _iqr(x0)
        q11, med1, q31 = _iqr(x1)

        mean0 = float(np.nanmean(x0)) if len(x0) else float("nan")
        mean1 = float(np.nanmean(x1)) if len(x1) else float("nan")
        std0 = float(np.nanstd(x0)) if len(x0) else float("nan")
        std1 = float(np.nanstd(x1)) if len(x1) else float("nan")

        U, p = _mann_whitney_u(x0, x1)
        delta = _cliffs_delta(x0, x1)

        diff_med, lo_med, hi_med = _bootstrap_ci_diff(x0, x1, stat="median", n_boot=n_boot, seed=seed)
        diff_mean, lo_mean, hi_mean = _bootstrap_ci_diff(x0, x1, stat="mean", n_boot=n_boot, seed=seed)

        rows.append({
            "metric": col,
            "description": desc,
            "n_improved": int(np.isfinite(x0).sum()),
            "n_else": int(np.isfinite(x1).sum()),
            "median_improved": med0,
            "iqr_improved_q1": q10,
            "iqr_improved_q3": q30,
            "median_else": med1,
            "iqr_else_q1": q11,
            "iqr_else_q3": q31,
            "mean_improved": mean0,
            "std_improved": std0,
            "mean_else": mean1,
            "std_else": std1,
            "mw_u": U,
            "mw_p_two_sided": p,
            "cliffs_delta": delta,
            "diff_median_improved_minus_else": diff_med,
            "diff_median_ci95_lo": lo_med,
            "diff_median_ci95_hi": hi_med,
            "diff_mean_improved_minus_else": diff_mean,
            "diff_mean_ci95_lo": lo_mean,
            "diff_mean_ci95_hi": hi_mean,
        })

    df_stats = pd.DataFrame(rows)
    csv_path = os.path.join(OUT, f"stats_improved_vs_else_{split}.csv")
    df_stats.to_csv(csv_path, index=False)

    txt_path = os.path.join(OUT, f"stats_improved_vs_else_{split}.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("Improved vs Else (Mann–Whitney U, Cliff's delta, bootstrap 95% CI)\n")
        for r in rows:
            f.write(f"\n[{r['metric']}] {r['description']}\n")
            f.write(f"  n: improved={r['n_improved']} else={r['n_else']}\n")
            f.write(
                f"  median (IQR): improved={r['median_improved']:.6g} "
                f"[{r['iqr_improved_q1']:.6g},{r['iqr_improved_q3']:.6g}]"
                f" | else={r['median_else']:.6g} "
                f"[{r['iqr_else_q1']:.6g},{r['iqr_else_q3']:.6g}]\n"
            )
            f.write(f"  MWU p={r['mw_p_two_sided']:.3g}, Cliff's delta={r['cliffs_delta']:.3g}\n")
            f.write(
                f"  median diff (imp-else)={r['diff_median_improved_minus_else']:.6g}"
                f" (95% CI {r['diff_median_ci95_lo']:.6g}..{r['diff_median_ci95_hi']:.6g})\n"
            )
            f.write(
                f"  mean diff (imp-else)={r['diff_mean_improved_minus_else']:.6g}"
                f" (95% CI {r['diff_mean_ci95_lo']:.6g}..{r['diff_mean_ci95_hi']:.6g})\n"
            )

    return csv_path, txt_path


# ----------------------------
# core computation
# ----------------------------

@torch.no_grad()
def compute_attn_arrays(batch: "approach3.Batch",
                        model: torch.nn.Module,
                        split: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns:
      pid_list: array of patient ids aligned to split indices
      dY: (B,) ΔPHQ labels (0/1/2)
      a:  (B,T) time attention
      ch: (B,T,d_c) channel attention
      xraw: (B,T,6) raw checks
    """
    if split == "val":
        idx = batch.idx_val
    elif split == "train":
        idx = batch.idx_tr
    else:
        raise ValueError("split must be 'train' or 'val'")

    S = batch.S[idx]
    C = batch.C[idx]
    y0p = batch.y0_phq[idx]
    y0p4 = batch.y0_p4[idx]
    y0lon = batch.y0_lon[idx]
    dY = batch.dY_phq[idx]
    Craw = batch.Craw[idx]

    model.eval()
    _, _, _, attn, ch_alpha = model.forward_multi(S, C, y0p, y0p4, y0lon)
    a = attn.detach().cpu().numpy().astype(np.float64)          # (B,T)
    ch = ch_alpha.detach().cpu().numpy().astype(np.float64)     # (B,T,d_c)
    dY_np = dY.detach().cpu().numpy().astype(np.int64)          # (B,)
    xraw = Craw.detach().cpu().numpy().astype(np.float64)       # (B,T,6)

    # patient ids
    all_pids = batch.stats.get("patients", None)
    if all_pids is None:
        pid_list = [f"idx{int(i)}" for i in idx]
    else:
        pid_list = [all_pids[int(i)] for i in idx]

    return np.array(pid_list, dtype=object), dY_np, a, ch, xraw


def build_check3_joint_tables(pid: np.ndarray,
                              dY: np.ndarray,
                              a: np.ndarray,
                              ch: np.ndarray,
                              xraw: np.ndarray,
                              check_idx: int,
                              vol_window: int = 3,
                              peak_window: int = 1) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """
    Creates:
      - per_patient_summary df
      - peak_window_stats df
      - global_summary dict

    Notes:
      - For categorical check3 (0/1), level is simply x_t.
      - change uses |x_t - x_{t-1}|, which becomes 1 only when a 0↔1 transition occurs.
    """
    B, T = a.shape
    b = ch[:, :, check_idx]           # (B,T)
    j = a * b                         # (B,T)
    jn = _safe_normalize_rows(j)      # (B,T)

    peak_t = np.argmax(j, axis=1).astype(np.int64)
    ent = _row_entropy(jn)

    x = xraw[:, :, check_idx]         # (B,T) raw check3 (categorical 0/1)

    # dynamics
    lvl = x
    chg = np.zeros_like(lvl)
    chg[:, 1:] = np.abs(lvl[:, 1:] - lvl[:, :-1])
    vol = np.zeros_like(lvl)
    for i in range(B):
        vol[i] = _rolling_std(lvl[i], window=vol_window)

    mu_lvl = (jn * lvl).sum(axis=1)
    mu_chg = (jn * chg).sum(axis=1)
    mu_vol = (jn * vol).sum(axis=1)

    # peak-window stats
    pw_level = np.zeros(B, dtype=np.float64)
    pw_chg = np.zeros(B, dtype=np.float64)
    pw_vol = np.zeros(B, dtype=np.float64)
    for i in range(B):
        t0 = int(peak_t[i])
        s = max(0, t0 - peak_window)
        e = min(T - 1, t0 + peak_window)
        W = slice(s, e + 1)
        pw_level[i] = float(np.max(lvl[i, W]))
        pw_chg[i] = float(np.max(chg[i, W]))
        pw_vol[i] = float(np.max(vol[i, W]))

    # improved vs else (else := same or worse)
    improved = (dY == 0)
    dY_bin = np.where(improved, 0, 1)

    per_patient = pd.DataFrame({
        "pid": pid,
        "dY_phq": dY,
        "group_improved_else": dY_bin,
        "peak_t": peak_t,
        "entropy": ent,
        "mu_lvl": mu_lvl,
        "mu_change": mu_chg,
        "mu_vol": mu_vol,
    })

    peak_stats = pd.DataFrame({
        "pid": pid,
        "dY_phq": dY,
        "group_improved_else": dY_bin,
        "peak_t": peak_t,
        "peak_level": pw_level,
        "peak_change": pw_chg,
        "peak_vol": pw_vol,
    })

    global_mean_j = j.mean(axis=0)
    global_peak_t = int(np.argmax(global_mean_j))

    global_summary = {
        "B": int(B),
        "T": int(T),
        "check_idx": int(check_idx),
        "global_peak_t": global_peak_t,
        "top3_weeks_by_mean_j": [int(x) for x in np.argsort(-global_mean_j)[:3]],
        "mean_entropy": float(ent.mean()),
        "mean_mu_change": float(mu_chg.mean()),
        "mean_mu_vol": float(mu_vol.mean()),
    }

    # Also return arrays for plotting via global_summary
    global_summary["_arrays"] = {
        "a": a, "b": b, "j": j, "jn": jn,
        "lvl": lvl, "chg": chg, "vol": vol,
        "global_mean_j": global_mean_j,
    }
    return per_patient, peak_stats, global_summary


def save_figures(outdir: str,
                 per_patient: pd.DataFrame,
                 global_summary: Dict[str, Any],
                 split: str):
    OUT = approach3.ensure_dir(outdir)
    arr = global_summary["_arrays"]
    a = arr["a"]; b = arr["b"]; j = arr["j"]
    mean_j = arr["global_mean_j"]
    T = a.shape[1]
    weeks = np.arange(T)

    g = per_patient["group_improved_else"].to_numpy()
    m0 = (g == 0)  # improved
    m1 = (g == 1)  # else

    # ---- Figure 1: mean a by group ----
    fig, ax = plt.subplots(1, 1, figsize=(7, 3))
    ax.plot(weeks, a[m0].mean(axis=0), marker="o", linewidth=1.5, label="a_t (improved)")
    ax.plot(weeks, a[m1].mean(axis=0), marker="o", linewidth=1.5, label="a_t (else)")
    ax.set_xlabel("week (t)"); ax.set_ylabel("mean a_t"); ax.set_title(f"Time attention a_t ({split.upper()})")
    ax.legend(); fig.tight_layout()
    fig.savefig(os.path.join(OUT, f"check3_a_mean_by_group_{split}.png"), dpi=150); plt.close(fig)

    # ---- Figure 1b: mean b by group ----
    fig, ax = plt.subplots(1, 1, figsize=(7, 3))
    ax.plot(weeks, b[m0].mean(axis=0), marker="o", linewidth=1.5, label="b_t (improved)")
    ax.plot(weeks, b[m1].mean(axis=0), marker="o", linewidth=1.5, label="b_t (else)")
    ax.set_xlabel("week (t)"); ax.set_ylabel("mean b_t"); ax.set_title(f"Check3 channel attention b_t ({split.upper()})")
    ax.legend(); fig.tight_layout()
    fig.savefig(os.path.join(OUT, f"check3_b_mean_by_group_{split}.png"), dpi=150); plt.close(fig)

    # ---- Figure 1c: mean j by group ----
    fig, ax = plt.subplots(1, 1, figsize=(7, 3))
    ax.plot(weeks, j[m0].mean(axis=0), marker="o", linewidth=1.5, label="j_t=a*b (improved)")
    ax.plot(weeks, j[m1].mean(axis=0), marker="o", linewidth=1.5, label="j_t=a*b (else)")
    ax.set_xlabel("week (t)"); ax.set_ylabel("mean j_t"); ax.set_title(f"Joint attention j_t ({split.upper()})")
    ax.legend(); fig.tight_layout()
    fig.savefig(os.path.join(OUT, f"check3_j_mean_by_group_{split}.png"), dpi=150); plt.close(fig)

    # ---- Figure 2: peak_t distribution (bar) ----
    peak_t = per_patient["peak_t"].to_numpy()
    c0 = np.bincount(peak_t[m0], minlength=T)
    c1 = np.bincount(peak_t[m1], minlength=T)
    fig, ax = plt.subplots(1, 1, figsize=(8, 3))
    ax.bar(weeks - 0.15, c0, width=0.3, label="improved")
    ax.bar(weeks + 0.15, c1, width=0.3, label="else")
    ax.set_xlabel("peak week t*"); ax.set_ylabel("count"); ax.set_title(f"Peak week of j_t (t*) by group ({split.upper()})")
    ax.legend(); fig.tight_layout()
    fig.savefig(os.path.join(OUT, f"check3_peak_t_hist_by_group_{split}.png"), dpi=150); plt.close(fig)

    # ---- Figure 3: entropy boxplot ----
    ent0 = per_patient.loc[m0, "entropy"].to_numpy()
    ent1 = per_patient.loc[m1, "entropy"].to_numpy()
    fig, ax = plt.subplots(1, 1, figsize=(5, 3))
    _boxplot_by_group(ax, ent0, ent1, ylabel="entropy of normalized j_t")
    ax.set_title(f"Entropy of joint attention ({split.upper()})")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, f"check3_entropy_by_group_{split}.png"), dpi=150); plt.close(fig)

    # ---- Figure 4: attention-weighted dynamics (mu_*) ----
    fig, axes = plt.subplots(1, 3, figsize=(10, 3))
    _boxplot_by_group(axes[0],
                      per_patient.loc[m0, "mu_lvl"].to_numpy(),
                      per_patient.loc[m1, "mu_lvl"].to_numpy(),
                      ylabel="mu_lvl")
    _boxplot_by_group(axes[1],
                      per_patient.loc[m0, "mu_change"].to_numpy(),
                      per_patient.loc[m1, "mu_change"].to_numpy(),
                      ylabel="mu_change")
    _boxplot_by_group(axes[2],
                      per_patient.loc[m0, "mu_vol"].to_numpy(),
                      per_patient.loc[m1, "mu_vol"].to_numpy(),
                      ylabel="mu_vol")
    for ax in axes:
        ax.grid(True, axis="y", alpha=0.3)
    fig.suptitle(f"Attention-weighted check3 dynamics ({split.upper()})", y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, f"check3_dynamics_weighted_by_group_{split}.png"), dpi=150); plt.close(fig)

    # ---- Figure 5: global mean j ----
    fig, ax = plt.subplots(1, 1, figsize=(7, 3))
    ax.plot(weeks, mean_j, marker="o", linewidth=1.5)
    ax.set_xlabel("week (t)"); ax.set_ylabel("mean j_t")
    ax.set_title(f"Global mean joint attention (all patients, {split.upper()})")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, f"check3_global_mean_j_{split}.png"), dpi=150); plt.close(fig)


def write_global_txt(outdir: str, global_summary: Dict[str, Any], split: str, check_name: str):
    OUT = approach3.ensure_dir(outdir)
    path = os.path.join(OUT, f"global_summary_check3_joint_{split}.txt")
    gs = {k: v for k, v in global_summary.items() if k != "_arrays"}
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"check_name: {check_name}\n")
        for k, v in gs.items():
            f.write(f"{k}: {v}\n")
    print("[WRITE]", path)


# ----------------------------
# Notebook-friendly runner
# ----------------------------

def run_pipeline(raw1: str,
                 raw2: str,
                 ckpt_dir: str,
                 ckpt_name: str,
                 outdir: str,
                 split: str = "val",
                 T: int = 40,
                 seed: int = 42,
                 check_name: str = "check3",
                 vol_window: int = 3,
                 peak_window: int = 1) -> Dict[str, str]:
    """
    Run the full pipeline programmatically (useful inside a Jupyter notebook).
    Returns a dict of important output paths.
    """
    OUT = approach3.ensure_dir(outdir)

    batch = approach3.build_dataset(raw1, raw2, T=T, outdir=ckpt_dir, seed=seed)
    device = batch.C.device
    model = approach3.load_model(ckpt_dir, device=device, filename=ckpt_name)

    pid, dY, a, ch, xraw = compute_attn_arrays(batch, model, split=split)
    check_idx = _find_check_index(batch, check_name)

    per_patient, peak_stats, global_summary = build_check3_joint_tables(
        pid, dY, a, ch, xraw,
        check_idx=check_idx,
        vol_window=vol_window,
        peak_window=peak_window,
    )

    per_path = os.path.join(OUT, f"per_patient_summary_{split}.csv")
    peak_path = os.path.join(OUT, f"peak_window_stats_{split}.csv")
    per_patient.to_csv(per_path, index=False)
    peak_stats.to_csv(peak_path, index=False)

    save_figures(OUT, per_patient, global_summary, split=split)
    write_global_txt(OUT, global_summary, split=split, check_name=check_name)

    stats_csv, stats_txt = compute_group_stats(per_patient, OUT, split=split, seed=seed, n_boot=5000)

    outputs = {
        "outdir": OUT,
        "per_patient_csv": per_path,
        "peak_window_csv": peak_path,
        "global_summary_txt": os.path.join(OUT, f"global_summary_check3_joint_{split}.txt"),
        "stats_csv": stats_csv,
        "stats_txt": stats_txt,
        "fig_a_mean_a": os.path.join(OUT, f"check3_a_mean_by_group_{split}.png"),
        "fig_b_mean_b": os.path.join(OUT, f"check3_b_mean_by_group_{split}.png"),
        "fig_j_mean_j": os.path.join(OUT, f"check3_j_mean_by_group_{split}.png"),
        "fig_peak_hist": os.path.join(OUT, f"check3_peak_t_hist_by_group_{split}.png"),
        "fig_entropy": os.path.join(OUT, f"check3_entropy_by_group_{split}.png"),
        "fig_mu_box": os.path.join(OUT, f"check3_dynamics_weighted_by_group_{split}.png"),
        "fig_global_j": os.path.join(OUT, f"check3_global_mean_j_{split}.png"),
    }
    return outputs


def verify_outputs(outputs: Dict[str, str], n_head: int = 5) -> None:
    """
    Notebook helper: quickly check that files exist and preview CSV heads.
    """
    print("=== Output files ===")
    for k, p in outputs.items():
        if k == "outdir":
            continue
        ok = os.path.exists(p)
        print(f"[{'OK' if ok else 'MISSING'}] {k}: {p}")

    # Preview CSVs
    for k in ["per_patient_csv", "peak_window_csv", "stats_csv"]:
        p = outputs.get(k, "")
        if p and os.path.exists(p):
            df = pd.read_csv(p)
            print(f"\n=== {k} head({n_head}) ===")
            try:
                display(df.head(n_head))  # noqa: F821  (Jupyter)
            except Exception:
                print(df.head(n_head).to_string(index=False))
        else:
            print(f"\n[SKIP] cannot preview {k}: file missing")

    # Show images (if running in notebook)
    try:
        from PIL import Image
        from IPython.display import display as ipy_display
        for k in ["fig_j_mean_j", "fig_peak_hist", "fig_entropy", "fig_mu_box", "fig_global_j"]:
            p = outputs.get(k, "")
            if p and os.path.exists(p):
                print(f"\n=== {k} ===")
                ipy_display(Image.open(p))
    except Exception as e:
        print("\n[NOTE] Image preview skipped (PIL/IPython not available):", repr(e))


def _fmt_med_iqr(x: np.ndarray) -> str:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return "nan"
    q1, med, q3 = _iqr(x)
    return f"{med:.4g} [{q1:.4g},{q3:.4g}]"

def _fmt_mean_sd(x: np.ndarray) -> str:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return "nan"
    return f"{np.mean(x):.4g} ± {np.std(x):.4g}"

def print_key_results(per_patient: pd.DataFrame,
                      global_summary: Dict[str, Any],
                      split: str = "val",
                      topk: int = 3) -> None:
    """
    Notebook/console용: 파일 저장 말고도 핵심 수치들을 즉시 출력.
    """
    g = per_patient["group_improved_else"].to_numpy()
    m0 = (g == 0)
    m1 = (g == 1)

    n0, n1 = int(m0.sum()), int(m1.sum())
    print(f"\n=== [KEY RESULTS] split={split} ===")
    print(f"n(improved)={n0}, n(else)={n1}")

    # Global peak info
    gp = global_summary.get("global_peak_t", None)
    top3 = global_summary.get("top3_weeks_by_mean_j", None)
    print(f"global_peak_t: {gp}")
    print(f"top3_weeks_by_mean_j: {top3}")

    # Peak concentration
    peak = per_patient["peak_t"].to_numpy()
    def topk_share(mask):
        if mask.sum() == 0:
            return []
        vc = np.bincount(peak[mask])
        top = np.argsort(-vc)[:topk]
        shares = [(int(t), int(vc[t]), float(vc[t]/mask.sum())) for t in top]
        return shares

    print("\n-- Peak week concentration (top-k) --")
    print("improved:", topk_share(m0))
    print("else    :", topk_share(m1))

    # Core metrics numeric summary
    cols = ["mu_lvl", "mu_change", "mu_vol", "entropy", "peak_t"]
    print("\n-- Numeric summary (median[IQR] | mean±sd) --")
    for c in cols:
        x0 = per_patient.loc[m0, c].to_numpy()
        x1 = per_patient.loc[m1, c].to_numpy()
        print(f"* {c}")
        print(f"  improved: {_fmt_med_iqr(x0)} | {_fmt_mean_sd(x0)}")
        print(f"  else    : {_fmt_med_iqr(x1)} | {_fmt_mean_sd(x1)}")

    # Significance + effect size (same as stats file but print now)
    print("\n-- Significance (MWU p) + Effect size (Cliff's delta) + Bootstrap CI --")
    for c in cols:
        x0 = per_patient.loc[m0, c].to_numpy(dtype=float)
        x1 = per_patient.loc[m1, c].to_numpy(dtype=float)
        U, p = _mann_whitney_u(x0, x1)
        d = _cliffs_delta(x0, x1)
        diff_med, lo_med, hi_med = _bootstrap_ci_diff(x0, x1, stat="median", n_boot=5000, seed=42)
        print(f"* {c}: p={p:.3g}, cliffs_delta={d:.3g}, median_diff(imp-else)={diff_med:.4g} [{lo_med:.4g},{hi_med:.4g}]")

def load_and_report(outdir: str, split: str = "val") -> None:
    """
    outdir에 저장된 csv/txt를 다시 로드해서 핵심 수치를 한 번에 출력.
    (모델 다시 안 돌리고 결과만 보고 싶을 때)
    """
    per_path = os.path.join(outdir, f"per_patient_summary_{split}.csv")
    glob_path = os.path.join(outdir, f"global_summary_check3_joint_{split}.txt")
    if not os.path.exists(per_path):
        raise FileNotFoundError(per_path)

    per = pd.read_csv(per_path)

    # global_summary txt 파싱 (간단)
    gs = {}
    if os.path.exists(glob_path):
        with open(glob_path, "r", encoding="utf-8") as f:
            for line in f:
                if ":" not in line:
                    continue
                k, v = line.split(":", 1)
                gs[k.strip()] = v.strip()
    print_key_results(per, gs, split=split)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw1", type=str, default="data/raw_data1.csv")
    ap.add_argument("--raw2", type=str, default="data/raw_data2.csv")
    ap.add_argument("--ckpt_dir", type=str, default="figs_phq_multi_attn")
    ap.add_argument("--ckpt_name", type=str, default="phq_multi_seq.ckpt")
    ap.add_argument("--outdir", type=str, default="figs_check3_joint")
    ap.add_argument("--split", type=str, choices=["train", "val"], default="val")
    ap.add_argument("--T", type=int, default=40)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--check_name", type=str, default="check3")
    ap.add_argument("--vol_window", type=int, default=3)
    ap.add_argument("--peak_window", type=int, default=1)
    args = ap.parse_args()

    OUT = approach3.ensure_dir(args.outdir)

    batch = approach3.build_dataset(args.raw1, args.raw2, T=args.T, outdir=args.ckpt_dir, seed=args.seed)
    device = batch.C.device
    model = approach3.load_model(args.ckpt_dir, device=device, filename=args.ckpt_name)

    pid, dY, a, ch, xraw = compute_attn_arrays(batch, model, split=args.split)
    check_idx = _find_check_index(batch, args.check_name)

    per_patient, peak_stats, global_summary = build_check3_joint_tables(
        pid, dY, a, ch, xraw,
        check_idx=check_idx,
        vol_window=args.vol_window,
        peak_window=args.peak_window,
    )

    per_path = os.path.join(OUT, f"per_patient_summary_{args.split}.csv")
    peak_path = os.path.join(OUT, f"peak_window_stats_{args.split}.csv")
    per_patient.to_csv(per_path, index=False)
    peak_stats.to_csv(peak_path, index=False)
    print("[SAVE]", per_path)
    print("[SAVE]", peak_path)

    save_figures(OUT, per_patient, global_summary, split=args.split)
    write_global_txt(OUT, global_summary, split=args.split, check_name=args.check_name)

    stats_csv, stats_txt = compute_group_stats(per_patient, OUT, split=args.split, seed=args.seed, n_boot=5000)
    print("[SAVE]", stats_csv)
    print("[SAVE]", stats_txt)

    print("[DONE] outputs under:", OUT)


if __name__ == "__main__":
    main()
