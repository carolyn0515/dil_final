# -*- coding: utf-8 -*-
"""
check3_improved_feature_discovery.py

Goal
----
Discover "check3 characteristic features" that separate improved vs else:
1) Extract attention tensors (attn: time, ch_alpha: time×channel) from trained model
2) Build per-patient derived features focused on check3
3) Validate with:
   - Group tests: permutation p, bootstrap CI, effect sizes, BH-FDR
   - V1: univariate AUC (bootstrap CI) on improved-vs-else
   - V4: window faithfulness using IMPROVED-LOGIT drop (targeted vs random)
   - V3: model-level ablation on INPUT check3

Assumptions
-----------
- model.forward_multi(S, C, y0_phq, y0_p4, y0_lon) returns:
    (logit_phq, logit_p4, logit_lon, attn, ch_alpha)
  where:
    attn: (B,T) time attention
    ch_alpha: (B,T,C) channel attention (prob-like weights across channels per time)

- batch has:
    batch.S, batch.C, batch.y0_phq, batch.y0_p4, batch.y0_lon, batch.dY_phq
    batch.idx_val / batch.idx_tr
    batch.stats["patients"]
    batch.stats["check_basenames"] or batch.stats["check_cols"]

- "improved" label is cfg.improved_label (default 0)
"""

import os
from dataclasses import dataclass
from typing import Dict, Any, Optional, List, Tuple

import numpy as np
import pandas as pd
import torch

try:
    from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
except Exception:
    roc_auc_score = None
    accuracy_score = None
    f1_score = None


# -----------------------------
# utils
# -----------------------------
def ensure_dir(d: str) -> str:
    os.makedirs(d, exist_ok=True)
    return d

def _smooth_1d(x: np.ndarray, k: int) -> np.ndarray:
    if k <= 1:
        return x
    if k % 2 == 0:
        k += 1
    pad = k // 2
    xp = np.pad(x, (pad, pad), mode="edge")
    w = np.ones(k, dtype=np.float32) / float(k)
    return np.convolve(xp, w, mode="valid")

def find_channel_index(check_names: List[str], target: str = "check3") -> int:
    target_l = target.strip().lower()
    for i, n in enumerate(check_names):
        if str(n).strip().lower() == target_l:
            return i
    for i, n in enumerate(check_names):
        if target_l in str(n).strip().lower():
            return i
    raise ValueError(f"Cannot find channel '{target}' from check_names={check_names}")

def _argmax_safe(x: np.ndarray) -> int:
    if len(x) == 0:
        return -1
    return int(np.argmax(x))

def _entropy(p: np.ndarray, eps: float = 1e-12) -> float:
    p = np.clip(p, eps, 1.0)
    p = p / p.sum()
    return float(-(p * np.log(p)).sum())


# -----------------------------
# stats helpers
# -----------------------------
def _cliffs_delta(x: np.ndarray, y: np.ndarray) -> float:
    x = x[np.isfinite(x)]
    y = y[np.isfinite(y)]
    if len(x) == 0 or len(y) == 0:
        return np.nan
    gt = 0
    lt = 0
    for xi in x:
        gt += np.sum(xi > y)
        lt += np.sum(xi < y)
    return float((gt - lt) / (len(x) * len(y)))

def _cohens_d(x: np.ndarray, y: np.ndarray) -> float:
    x = x[np.isfinite(x)]
    y = y[np.isfinite(y)]
    if len(x) < 2 or len(y) < 2:
        return np.nan
    mx, my = x.mean(), y.mean()
    sx, sy = x.std(ddof=1), y.std(ddof=1)
    sp = np.sqrt(((len(x)-1)*sx*sx + (len(y)-1)*sy*sy) / (len(x)+len(y)-2))
    if sp == 0:
        return np.nan
    return float((mx - my) / sp)

def _bootstrap_ci_diff(x: np.ndarray, y: np.ndarray, n_boot=2000, seed=0) -> Tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    x = x[np.isfinite(x)]
    y = y[np.isfinite(y)]
    if len(x) == 0 or len(y) == 0:
        return np.nan, np.nan, np.nan
    diffs = []
    for _ in range(n_boot):
        xb = rng.choice(x, size=len(x), replace=True)
        yb = rng.choice(y, size=len(y), replace=True)
        diffs.append(xb.mean() - yb.mean())
    diffs = np.array(diffs)
    return float(np.mean(diffs)), float(np.quantile(diffs, 0.025)), float(np.quantile(diffs, 0.975))

def _perm_test_diff(x: np.ndarray, y: np.ndarray, n_perm=5000, seed=0) -> float:
    rng = np.random.default_rng(seed)
    x = x[np.isfinite(x)]
    y = y[np.isfinite(y)]
    if len(x) == 0 or len(y) == 0:
        return np.nan
    obs = x.mean() - y.mean()
    z = np.concatenate([x, y])
    n1 = len(x)
    cnt = 0
    for _ in range(n_perm):
        rng.shuffle(z)
        d = z[:n1].mean() - z[n1:].mean()
        if abs(d) >= abs(obs):
            cnt += 1
    return float((cnt + 1) / (n_perm + 1))

def _bh_fdr(pvals: np.ndarray) -> np.ndarray:
    pvals = np.asarray(pvals, dtype=float)
    m = len(pvals)
    order = np.argsort(pvals)
    q = np.empty_like(pvals, dtype=float)
    prev = 1.0
    for rank, _ in enumerate(order[::-1], start=1):
        i = order[-rank]
        val = min(prev, pvals[i] * m / (m - rank + 1))
        q[i] = val
        prev = val
    return q

def _bootstrap_auc(y_true: np.ndarray, y_score: np.ndarray, n_boot=3000, seed=0) -> Dict[str, float]:
    if roc_auc_score is None:
        raise ImportError("sklearn is required for AUC. Please install scikit-learn.")
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score).astype(float)
    n = len(y_true)
    aucs = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        yt = y_true[idx]
        ys = y_score[idx]
        if len(np.unique(yt)) < 2:
            continue
        aucs.append(roc_auc_score(yt, ys))
    aucs = np.array(aucs, dtype=float)
    return {
        "auc_mean": float(np.mean(aucs)) if len(aucs) else np.nan,
        "auc_ci95_lo": float(np.quantile(aucs, 0.025)) if len(aucs) else np.nan,
        "auc_ci95_hi": float(np.quantile(aucs, 0.975)) if len(aucs) else np.nan,
        "n_boot_used": int(len(aucs)),
    }


# -----------------------------
# model extraction
# -----------------------------
@torch.no_grad()
def extract_tensors(batch, model, split: str = "val") -> Dict[str, Any]:
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
    dY = batch.dY_phq[idx]  # (B,) 0/1/2

    model.eval()
    logit_phq, logit_p4, logit_lon, attn, ch_alpha = model.forward_multi(S, C, y0p, y0p4, y0lon)

    return {
        "S": S,
        "C": C,
        "y0p": y0p,
        "y0p4": y0p4,
        "y0lon": y0lon,
        "dPHQ": dY.detach().cpu().numpy().astype(int),
        "patients": np.array(batch.stats["patients"])[idx],
        "check_names": list(batch.stats.get("check_basenames", batch.stats.get("check_cols"))),
        "logit_phq": logit_phq,
        "attn": attn.detach().cpu().numpy(),        # (B,T)
        "ch_alpha": ch_alpha.detach().cpu().numpy() # (B,T,C)
    }


# -----------------------------
# feature engineering (check3-focused)
# -----------------------------
@dataclass
class FeatureConfig:
    target_channel: str = "check3"
    smooth: int = 3
    eps: float = 1e-9

def compute_check3_features_for_patient(
    attn_t: np.ndarray,           # (T,)
    ch_alpha_t: np.ndarray,       # (T,C)
    c_raw_t: np.ndarray,          # (T,) raw check3 input values from C[:, :, ch3]
    ch3_idx: int,
    cfg: FeatureConfig
) -> Dict[str, Any]:
    """
    Creates *check3 characterization* features:
      A) channel-attention dynamics for check3 (from ch_alpha)
      B) coupling between time-attention and check3 input (attn × check3_input)
    """
    eps = cfg.eps
    T, C = ch_alpha_t.shape

    attn = _smooth_1d(attn_t, cfg.smooth)
    ch3a = _smooth_1d(ch_alpha_t[:, ch3_idx], cfg.smooth)   # channel-attn for check3
    c3   = _smooth_1d(c_raw_t, cfg.smooth)                  # raw check3 input

    # other channels summary
    others = np.delete(ch_alpha_t, ch3_idx, axis=1)
    other_mean = others.mean(axis=1)
    other_max  = others.max(axis=1)

    # derivatives
    d_ch3a = np.diff(ch3a, prepend=ch3a[0])
    d_c3   = np.diff(c3,   prepend=c3[0])

    # attention-weighted check3 "exposure"
    # (time-attn tells "important time", raw check3 tells "what happened")
    aw_c3 = attn * c3
    aw_ch3a = attn * ch3a

    # entropy/top1
    ent = []
    top1 = []
    ranks = np.argsort(-ch_alpha_t, axis=1)
    for t in range(T):
        p = np.clip(ch_alpha_t[t], eps, 1.0)
        p = p / p.sum()
        ent.append(_entropy(p, eps=eps))
        top1.append(float(np.max(p)))
    ent = np.asarray(ent)
    top1 = np.asarray(top1)

    feat: Dict[str, Any] = {}
    feat["T"] = int(T)

    # --- A) channel-attn based (your original family + 조금 보강) ---
    feat["ch3a_mean"]   = float(ch3a.mean())
    feat["ch3a_max"]    = float(ch3a.max())
    feat["ch3a_peak_t"] = _argmax_safe(ch3a)

    feat["ch3a_vol"]        = float(np.std(d_ch3a))
    feat["ch3a_dpeak_t"]    = _argmax_safe(np.abs(d_ch3a))
    feat["ch3a_dpeak_val"]  = float(np.max(np.abs(d_ch3a)))

    feat["dom_ratio_mean"]  = float(np.mean(ch3a / (other_mean + eps)))
    feat["dom_ratio_max"]   = float(np.max(ch3a / (other_mean + eps)))
    feat["margin_mean"]     = float(np.mean(ch3a - other_max))
    feat["margin_min"]      = float(np.min(ch3a - other_max))

    feat["ch_ent_mean"]     = float(ent.mean())
    feat["ch_top1_mean"]    = float(top1.mean())
    feat["ch3a_top1_rate"]  = float(np.mean(ranks[:, 0] == ch3_idx))
    feat["ch3a_top2_rate"]  = float(np.mean(np.any(ranks[:, :2] == ch3_idx, axis=1)))

    # --- B) raw check3 input “shape” ---
    feat["c3_mean"]      = float(np.mean(c3))
    feat["c3_max"]       = float(np.max(c3))
    feat["c3_vol"]       = float(np.std(d_c3))
    feat["c3_dpeak_val"] = float(np.max(np.abs(d_c3)))
    feat["c3_peak_t"]    = _argmax_safe(c3)

    # --- C) coupling time-attn × check3 ---
    feat["attn_peak_t"]   = _argmax_safe(attn)
    feat["aw_c3_sum"]      = float(np.sum(aw_c3))
    feat["aw_c3_max"]      = float(np.max(aw_c3))
    feat["aw_c3_peak_t"]   = _argmax_safe(aw_c3)

    feat["aw_ch3a_sum"]    = float(np.sum(aw_ch3a))
    feat["aw_ch3a_max"]    = float(np.max(aw_ch3a))
    feat["aw_ch3a_peak_t"] = _argmax_safe(aw_ch3a)

    # alignment / correlation (robust하게: 분산 0이면 0 처리)
    def _corr(a, b):
        a = np.asarray(a, float); b = np.asarray(b, float)
        if np.std(a) < 1e-12 or np.std(b) < 1e-12:
            return 0.0
        return float(np.corrcoef(a, b)[0, 1])

    feat["corr_attn_c3"]    = _corr(attn, c3)
    feat["corr_attn_ch3a"]  = _corr(attn, ch3a)
    feat["corr_c3_ch3a"]    = _corr(c3, ch3a)

    return feat


# -----------------------------
# group tests: improved vs else
# -----------------------------
@dataclass
class GroupConfig:
    split: str = "val"
    target_channel: str = "check3"
    improved_label: int = 0

    smooth: int = 3
    n_boot: int = 2000
    n_perm: int = 5000
    seed: int = 0

    # candidate criteria
    fdr_q_max: float = 0.10
    abs_cohens_d_min: float = 0.20

def build_feature_table(batch, model, outdir: str, cfg: GroupConfig) -> pd.DataFrame:
    OUT = ensure_dir(outdir)
    data = extract_tensors(batch, model, split=cfg.split)

    attn = data["attn"]          # (B,T)
    ch_alpha = data["ch_alpha"]  # (B,T,C)
    C_in = data["C"]             # torch (B,T,C)
    dPHQ = data["dPHQ"]          # (B,)
    pids = data["patients"]
    check_names = data["check_names"]

    ch3_idx = find_channel_index(check_names, cfg.target_channel)
    fcfg = FeatureConfig(target_channel=cfg.target_channel, smooth=cfg.smooth)

    C_np = C_in.detach().cpu().numpy()  # (B,T,C)
    rows = []
    for i in range(attn.shape[0]):
        c3_raw = C_np[i, :, ch3_idx]  # raw check3 input along time
        feats = compute_check3_features_for_patient(attn[i], ch_alpha[i], c3_raw, ch3_idx, fcfg)

        rows.append({
            "pid": pids[i],
            "split": cfg.split,
            "dPHQ": int(dPHQ[i]),              # 0/1/2
            "is_improved": int(dPHQ[i] == cfg.improved_label),
            "check3_idx": int(ch3_idx),
            **feats
        })

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, f"per_patient_features_{cfg.split}.csv"), index=False)
    return df

def run_group_tests(df: pd.DataFrame, outdir: str, cfg: GroupConfig) -> Dict[str, pd.DataFrame]:
    OUT = ensure_dir(outdir)

    g1 = df["is_improved"].values.astype(int) == 1
    g2 = ~g1

    exclude = {"pid", "split", "dPHQ", "is_improved", "check3_idx"}
    feat_cols = [c for c in df.columns if c not in exclude and np.issubdtype(df[c].dtype, np.number)]

    rows = []
    for col in feat_cols:
        x = df.loc[g1, col].values.astype(float)
        y = df.loc[g2, col].values.astype(float)

        obs = float(np.nanmean(x) - np.nanmean(y))
        d = _cohens_d(x, y)
        cliffs = _cliffs_delta(x, y)
        boot_mean, ci_lo, ci_hi = _bootstrap_ci_diff(x, y, n_boot=cfg.n_boot, seed=cfg.seed)
        p = _perm_test_diff(x, y, n_perm=cfg.n_perm, seed=cfg.seed)

        rows.append({
            "feature": col,
            "diff_mean(improved-else)": obs,
            "cohens_d": d,
            "cliffs_delta": cliffs,
            "boot_diff_mean": boot_mean,
            "ci95_lo": ci_lo,
            "ci95_hi": ci_hi,
            "perm_p": p,
            "n_improved": int(np.sum(g1)),
            "n_else": int(np.sum(g2)),
        })

    res = pd.DataFrame(rows)
    res["abs_cohens_d"] = np.abs(res["cohens_d"].values.astype(float))
    res = res.sort_values(["perm_p", "abs_cohens_d"], ascending=[True, False])
    res["fdr_q"] = _bh_fdr(res["perm_p"].values)

    res_path = os.path.join(OUT, f"stats_tests_{cfg.split}.csv")
    res.to_csv(res_path, index=False)

    cand = res[(res["fdr_q"] <= cfg.fdr_q_max) & (res["abs_cohens_d"] >= cfg.abs_cohens_d_min)].copy()
    cand = cand.sort_values(["fdr_q", "perm_p", "abs_cohens_d"], ascending=[True, True, False])
    cand_path = os.path.join(OUT, f"candidate_findings_{cfg.split}.csv")
    cand.to_csv(cand_path, index=False)

    return {"tests": res, "candidates": cand}


# -----------------------------
# validation suite (improved-centered)
# -----------------------------
@dataclass
class ValConfig:
    split: str = "val"
    target_channel: str = "check3"
    improved_label: int = 0

    smooth: int = 3

    # V1 AUC
    n_boot_auc: int = 3000

    # V4 window faithfulness
    half_width: int = 1
    n_random: int = 50
    n_perm_window: int = 8000
    window_mode: str = "set_patient_mean"  # "set_patient_mean" | "permute_within_window"

    # V3 ablation
    ablation_modes: Optional[List[str]] = None

    seed: int = 0

@torch.no_grad()
def forward_phq_and_attn(model, S, C, y0p, y0p4, y0lon):
    model.eval()
    logit_phq, _, _, attn, ch_alpha = model.forward_multi(S, C, y0p, y0p4, y0lon)
    return logit_phq, attn, ch_alpha

def compute_tstar_from_ch3a_dpeak(ch_alpha: np.ndarray, ch3_idx: int, smooth: int = 3) -> Tuple[np.ndarray, np.ndarray]:
    """
    ch_alpha: (B,T,C)
    Returns:
      marker: max |Δ ch3a|
      tstar:  argmax |Δ ch3a|
    """
    B, T, C = ch_alpha.shape
    marker = np.zeros(B, dtype=float)
    tstar = np.zeros(B, dtype=int)
    for i in range(B):
        ch3a = _smooth_1d(ch_alpha[i, :, ch3_idx], smooth)
        d = np.diff(ch3a, prepend=ch3a[0])
        a = np.abs(d)
        t = int(np.argmax(a))
        marker[i] = float(a[t])
        tstar[i] = t
    return marker, tstar

def ablate_check3_window(C: torch.Tensor, ch3_idx: int, centers: np.ndarray, half_width: int = 1,
                         mode: str = "set_patient_mean", seed: int = 0) -> torch.Tensor:
    rng = np.random.default_rng(seed)
    C2 = C.clone()
    B, T, _ = C2.shape

    if mode == "set_patient_mean":
        for i in range(B):
            c = int(centers[i])
            s = max(0, c - half_width)
            e = min(T - 1, c + half_width)
            mean_i = float(C2[i, :, ch3_idx].mean().item())
            C2[i, s:e+1, ch3_idx] = mean_i

    elif mode == "permute_within_window":
        for i in range(B):
            c = int(centers[i])
            s = max(0, c - half_width)
            e = min(T - 1, c + half_width)
            win = C2[i, s:e+1, ch3_idx].detach().cpu().numpy()
            perm = rng.permutation(len(win))
            C2[i, s:e+1, ch3_idx] = torch.tensor(win[perm], device=C2.device, dtype=C2.dtype)

    else:
        raise ValueError(f"Unknown window mode={mode}")

    return C2

def ablate_check3_input(C: torch.Tensor, ch3_idx: int, mode: str = "permute_time", seed: int = 0) -> torch.Tensor:
    rng = np.random.default_rng(seed)
    C2 = C.clone()
    B, T, _ = C2.shape

    if mode == "permute_time":
        for i in range(B):
            perm = rng.permutation(T)
            C2[i, :, ch3_idx] = C2[i, perm, ch3_idx]

    elif mode == "set_patient_mean":
        m = C2[:, :, ch3_idx].mean(dim=1, keepdim=True)
        C2[:, :, ch3_idx] = m.repeat(1, T)

    elif mode == "set_global_mode":
        v = C2[:, :, ch3_idx].detach().cpu().numpy().ravel()
        v_ = np.round(v).astype(int)
        vals, cnts = np.unique(v_, return_counts=True)
        mode_val = int(vals[np.argmax(cnts)])
        C2[:, :, ch3_idx] = float(mode_val)

    else:
        raise ValueError(f"Unknown ablation mode={mode}")

    return C2

def _perm_test_signflip(delta: np.ndarray, n_perm=8000, seed=0) -> float:
    rng = np.random.default_rng(seed)
    delta = np.asarray(delta, dtype=float)
    delta = delta[np.isfinite(delta)]
    if len(delta) == 0:
        return np.nan
    obs = float(np.mean(delta))
    cnt = 0
    for _ in range(n_perm):
        signs = rng.choice([-1.0, 1.0], size=len(delta))
        d = float(np.mean(delta * signs))
        if abs(d) >= abs(obs):
            cnt += 1
    return float((cnt + 1) / (n_perm + 1))

@torch.no_grad()
def eval_model_phq_macro(model, S, C, y0p, y0p4, y0lon, dPHQ_true_np) -> Dict[str, float]:
    if accuracy_score is None:
        raise ImportError("sklearn is required for acc/F1. Please install scikit-learn.")
    logit, _, _ = forward_phq_and_attn(model, S, C, y0p, y0p4, y0lon)
    pred = torch.argmax(logit, dim=1).detach().cpu().numpy().astype(int)
    y = np.asarray(dPHQ_true_np).astype(int)
    return {
        "acc": float(accuracy_score(y, pred)),
        "macro_f1": float(f1_score(y, pred, average="macro")),
    }

def run_validation_from_marker(
    batch, model, outdir: str, marker_feature: str, df_features: pd.DataFrame, cfg: ValConfig
) -> Dict[str, Any]:
    """
    marker_feature: selected feature column in df_features used for V1 AUC.
    V4 uses improved-logit drop around tstar from ch3a_dpeak (attention-based).
    """
    OUT = ensure_dir(outdir)
    data = extract_tensors(batch, model, split=cfg.split)

    S = data["S"]
    C = data["C"]
    y0p = data["y0p"]
    y0p4 = data["y0p4"]
    y0lon = data["y0lon"]
    dPHQ = data["dPHQ"]  # 0/1/2
    check_names = data["check_names"]
    ch3_idx = find_channel_index(check_names, cfg.target_channel)

    # binary label
    y_bin = (dPHQ == cfg.improved_label).astype(int)

    # ---------- V1: univariate AUC on chosen marker_feature ----------
    marker = df_features[marker_feature].values.astype(float)
    v1_auc = {
        "feature": marker_feature,
        "auc": float(roc_auc_score(y_bin, marker)) if roc_auc_score is not None and len(np.unique(y_bin)) == 2 else np.nan,
        **(_bootstrap_auc(y_bin, marker, n_boot=cfg.n_boot_auc, seed=cfg.seed) if roc_auc_score is not None else {})
    }
    pd.DataFrame([v1_auc]).to_csv(os.path.join(OUT, f"v1_auc_{cfg.split}.csv"), index=False)

    # ---------- V4: window faithfulness (IMPROVED-LOGIT 기준) ----------
    logit0, attn0, ch0 = forward_phq_and_attn(model, S, C, y0p, y0p4, y0lon)
    ch0_np = ch0.detach().cpu().numpy()
    _, tstar = compute_tstar_from_ch3a_dpeak(ch0_np, ch3_idx, smooth=cfg.smooth)

    # improved logit (class=0) ONLY
    base_impr_logit = logit0[:, cfg.improved_label].detach().cpu().numpy().astype(float)

    # targeted window
    C_targ = ablate_check3_window(C, ch3_idx, tstar, half_width=cfg.half_width,
                                  mode=cfg.window_mode, seed=cfg.seed)
    logit_t, _, _ = forward_phq_and_attn(model, S, C_targ, y0p, y0p4, y0lon)
    targ_impr_logit = logit_t[:, cfg.improved_label].detach().cpu().numpy().astype(float)
    delta_targ = targ_impr_logit - base_impr_logit

    # random windows
    rng = np.random.default_rng(cfg.seed)
    B, T, _ = C.shape
    deltas_rand = []
    for r in range(cfg.n_random):
        centers = rng.integers(0, T, size=B)
        C_r = ablate_check3_window(C, ch3_idx, centers, half_width=cfg.half_width,
                                   mode=cfg.window_mode, seed=cfg.seed + 13 + r)
        logit_r, _, _ = forward_phq_and_attn(model, S, C_r, y0p, y0p4, y0lon)
        rand_impr_logit = logit_r[:, cfg.improved_label].detach().cpu().numpy().astype(float)
        deltas_rand.append(rand_impr_logit - base_impr_logit)
    delta_rand_mean = np.mean(np.stack(deltas_rand, axis=0), axis=0)

    diff = delta_targ - delta_rand_mean
    p_win = _perm_test_signflip(diff, n_perm=cfg.n_perm_window, seed=cfg.seed)

    v4 = {
        "metric": "improved_logit_drop",
        "mean_delta_targeted": float(np.mean(delta_targ)),
        "mean_delta_random": float(np.mean(delta_rand_mean)),
        "mean_diff(targeted-random)": float(np.mean(diff)),
        "perm_p": float(p_win),
        "half_width": int(cfg.half_width),
        "n_random": int(cfg.n_random),
        "window_mode": cfg.window_mode,
        "note": "More negative targeted delta than random => targeted window more influential for improved-logit."
    }
    pd.DataFrame([v4]).to_csv(os.path.join(OUT, f"v4_window_faithfulness_{cfg.split}.csv"), index=False)

    # ---------- V3: model-level ablation on INPUT check3 ----------
    modes = cfg.ablation_modes or ["permute_time", "set_patient_mean", "set_global_mode"]
    base = eval_model_phq_macro(model, S, C, y0p, y0p4, y0lon, dPHQ)
    rows = [{"mode": "baseline", **base, "delta_acc": 0.0, "delta_macro_f1": 0.0}]
    for m in modes:
        C_abl = ablate_check3_input(C, ch3_idx, mode=m, seed=cfg.seed)
        met = eval_model_phq_macro(model, S, C_abl, y0p, y0p4, y0lon, dPHQ)
        rows.append({
            "mode": m,
            **met,
            "delta_acc": float(met["acc"] - base["acc"]),
            "delta_macro_f1": float(met["macro_f1"] - base["macro_f1"]),
        })
    v3_df = pd.DataFrame(rows)
    v3_df.to_csv(os.path.join(OUT, f"v3_model_ablation_{cfg.split}.csv"), index=False)

    return {"v1_auc": v1_auc, "v4": v4, "v3": v3_df}


# -----------------------------
# Orchestrator: discover top check3 features
# -----------------------------
@dataclass
class DiscoverConfig:
    outdir: str = "./out_check3_discovery"
    group: GroupConfig = GroupConfig()
    val: ValConfig = ValConfig()

    # how to choose the "marker feature" for V1/V4/V3 reporting
    # option: pick best by fdr_q then |d|, or explicitly set
    marker_feature: Optional[str] = None

def discover_check3_characteristics(batch, model, cfg: Optional[DiscoverConfig] = None) -> Dict[str, Any]:
    cfg = cfg or DiscoverConfig()
    OUT = ensure_dir(cfg.outdir)

    # 1) build features
    df_feat = build_feature_table(batch, model, outdir=OUT, cfg=cfg.group)

    # 2) group tests improved vs else
    tests = run_group_tests(df_feat, outdir=OUT, cfg=cfg.group)
    df_tests = tests["tests"]
    df_cand = tests["candidates"]

    # 3) pick marker feature
    if cfg.marker_feature is not None:
        marker = cfg.marker_feature
    else:
        # default: best candidate; if none, best overall by fdr_q then abs_d
        if len(df_cand) > 0:
            marker = str(df_cand.iloc[0]["feature"])
        else:
            marker = str(df_tests.sort_values(["fdr_q", "abs_cohens_d"], ascending=[True, False]).iloc[0]["feature"])

    # 4) validation suite for chosen marker (AUC + faithfulness + ablation)
    val_out = run_validation_from_marker(batch, model, outdir=OUT, marker_feature=marker, df_features=df_feat, cfg=cfg.val)

    # 5) Top findings summary (human-friendly)
    topk = df_tests.sort_values(["fdr_q", "abs_cohens_d"], ascending=[True, False]).head(20).copy()
    topk.insert(0, "rank", np.arange(1, len(topk) + 1))
    topk_path = os.path.join(OUT, f"top_findings_{cfg.group.split}.csv")
    topk.to_csv(topk_path, index=False)

    # 6) Save a compact "final answer" row
    final = pd.DataFrame([{
        "chosen_marker_feature": marker,
        "marker_auc": val_out["v1_auc"].get("auc", np.nan),
        "marker_auc_ci95_lo": val_out["v1_auc"].get("auc_ci95_lo", np.nan),
        "marker_auc_ci95_hi": val_out["v1_auc"].get("auc_ci95_hi", np.nan),
        "v4_perm_p": val_out["v4"]["perm_p"],
        "v4_mean_diff(targeted-random)": val_out["v4"]["mean_diff(targeted-random)"],
        "baseline_macro_f1": float(val_out["v3"].query("mode=='baseline'")["macro_f1"].iloc[0]),
    }])
    final_path = os.path.join(OUT, f"final_summary_{cfg.group.split}.csv")
    final.to_csv(final_path, index=False)

    return {
        "paths": {
            "features_csv": os.path.join(OUT, f"per_patient_features_{cfg.group.split}.csv"),
            "tests_csv": os.path.join(OUT, f"stats_tests_{cfg.group.split}.csv"),
            "candidates_csv": os.path.join(OUT, f"candidate_findings_{cfg.group.split}.csv"),
            "top_findings_csv": topk_path,
            "final_summary_csv": final_path,
            "v1_auc_csv": os.path.join(OUT, f"v1_auc_{cfg.group.split}.csv"),
            "v4_window_csv": os.path.join(OUT, f"v4_window_faithfulness_{cfg.group.split}.csv"),
            "v3_ablation_csv": os.path.join(OUT, f"v3_model_ablation_{cfg.group.split}.csv"),
        },
        "marker_feature": marker,
        "df_features": df_feat,
        "df_tests": df_tests,
        "df_candidates": df_cand,
        "validation": val_out,
    }


# -----------------------------
# Example usage (you will adapt to your training script)
# -----------------------------
if __name__ == "__main__":
    """
    You should call discover_check3_characteristics(batch, model, cfg)
    from your training/analysis environment where `batch` and `model` exist.
    """
    print("Run this from an environment where `batch` and `model` are available.")
