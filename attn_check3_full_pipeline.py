# -*- coding: utf-8 -*-
"""
attn_check3_full_pipeline.py

ONE file that:
1) Builds per-patient attention-derived features (including ch3_dpeak_val)
2) Runs group stats (improved vs else): bootstrap CI, permutation p, effect sizes, FDR
3) Runs validation suite:
   - V1 Univariate AUC (bootstrap CI)
   - V4 Window faithfulness (targeted vs random occlusion, perm p)
   - V3 Model-level ablation on INPUT check3 (several modes)

Outputs (saved to outdir)
-------------------------
- per_patient_features_{split}.csv
- stats_tests_{split}.csv
- candidate_findings_{split}.csv
- marker_ch3_dpeak_{split}.csv
- model_ablation_{split}.csv
- window_faithfulness_{split}.csv

Assumptions
-----------
- approach3 model exposes:
    model.forward_multi(S, C, y0_phq, y0_p4, y0_lon)
    -> (logit_phq, logit_p4, logit_lon, attn, ch_alpha)
  where:
    attn: (B,T)
    ch_alpha: (B,T,C)

- batch has:
    batch.S, batch.C, batch.y0_phq, batch.y0_p4, batch.y0_lon, batch.dY_phq
    batch.idx_val / batch.idx_tr
    batch.stats["patients"]
    batch.stats["check_basenames"] or batch.stats["check_cols"]

Notes
-----
- We treat "improved" as label cfg.improved_label (default 0),
  and "else" as the remaining labels.
"""

import os
from dataclasses import dataclass
from typing import Dict, Any, Optional, List, Tuple

import numpy as np
import pandas as pd
import torch

# optional sklearn (for AUC / basic metrics)
try:
    from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
except Exception as e:
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

def nanmean(x):
    x = np.asarray(x, dtype=float)
    return float(np.nanmean(x)) if np.isfinite(x).any() else np.nan

def nanstd(x):
    x = np.asarray(x, dtype=float)
    return float(np.nanstd(x)) if np.isfinite(x).any() else np.nan


# -----------------------------
# statistics helpers
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
    """
    Benjamini–Hochberg FDR q-values
    """
    pvals = np.asarray(pvals, dtype=float)
    m = len(pvals)
    order = np.argsort(pvals)
    q = np.empty_like(pvals, dtype=float)
    prev = 1.0
    for rank, _idx in enumerate(order[::-1], start=1):
        i = order[-rank]
        val = min(prev, pvals[i] * m / (m - rank + 1))
        q[i] = val
        prev = val
    return q


# -----------------------------
# model extraction
# -----------------------------

@torch.no_grad()
def extract_attn_tensors(batch, model, split: str = "val") -> Dict[str, Any]:
    """
    Expects model.forward_multi to return:
      (logit_phq, logit_p4, logit_lon, attn, ch_alpha)
    attn: (B,T)
    ch_alpha: (B,T,C)
    """
    if split == "val":
        idx = batch.idx_val
    elif split == "train":
        idx = batch.idx_tr
    else:
        raise ValueError("split must be 'train' or 'val'")

    S   = batch.S[idx]
    C   = batch.C[idx]
    y0p = batch.y0_phq[idx]
    y0p4 = batch.y0_p4[idx]
    y0lon = batch.y0_lon[idx]

    dY   = batch.dY_phq[idx]  # (B,)

    model.eval()
    logit_phq, logit_p4, logit_lon, attn, ch_alpha = model.forward_multi(S, C, y0p, y0p4, y0lon)

    out = {
        "S": S,
        "C": C,
        "y0p": y0p,
        "y0p4": y0p4,
        "y0lon": y0lon,
        "dPHQ": dY.detach().cpu().numpy().astype(int),    # (B,)
        "y0PHQ": y0p.detach().cpu().numpy().astype(int),
        "patients": np.array(batch.stats["patients"])[idx],
        "check_names": list(batch.stats.get("check_basenames", batch.stats.get("check_cols"))),

        "logit_phq": logit_phq,
        "attn": attn.detach().cpu().numpy(),              # (B,T)
        "ch_alpha": ch_alpha.detach().cpu().numpy(),      # (B,T,C)
    }
    return out


# -----------------------------
# feature engineering
# -----------------------------

@dataclass
class FeatureConfig:
    target_channel: str = "check3"
    smooth: int = 1
    eps: float = 1e-9

def compute_features_for_patient(attn: np.ndarray,
                                 ch_alpha: np.ndarray,
                                 ch3_idx: int,
                                 cfg: FeatureConfig) -> Dict[str, Any]:
    """
    attn: (T,)
    ch_alpha: (T,C)
    """
    T, C = ch_alpha.shape
    eps = cfg.eps

    attn_s = _smooth_1d(attn, cfg.smooth)
    ch3 = _smooth_1d(ch_alpha[:, ch3_idx], cfg.smooth)

    others = np.delete(ch_alpha, ch3_idx, axis=1)
    other_mean = others.mean(axis=1)
    other_max = others.max(axis=1)

    # dynamics on channel attention itself
    d_ch3 = np.diff(ch3, prepend=ch3[0])

    feat: Dict[str, Any] = {}
    feat["T"] = int(T)

    # main marker + some companions
    feat["ch3_mean"] = float(ch3.mean())
    feat["ch3_max"]  = float(ch3.max())
    feat["ch3_peak_t"]  = _argmax_safe(ch3)
    feat["attn_peak_t"] = _argmax_safe(attn_s)

    feat["ch3_vol"] = float(np.std(d_ch3))
    feat["ch3_dpeak_t"] = _argmax_safe(np.abs(d_ch3))
    feat["ch3_dpeak_val"] = float(np.max(np.abs(d_ch3)))

    # relativity / dominance
    feat["dom_ratio_mean"] = float(np.mean(ch3 / (other_mean + eps)))
    feat["dom_ratio_max"]  = float(np.max(ch3 / (other_mean + eps)))
    feat["margin_mean"] = float(np.mean(ch3 - other_max))
    feat["margin_min"]  = float(np.min(ch3 - other_max))

    # channel entropy and top1
    ent = []
    top1 = []
    ranks = np.argsort(-ch_alpha, axis=1)  # (T,C)
    for t in range(T):
        p = np.clip(ch_alpha[t], eps, 1.0)
        p = p / p.sum()
        ent.append(_entropy(p, eps=eps))
        top1.append(float(np.max(p)))
    ent = np.array(ent)
    top1 = np.array(top1)
    feat["ch_ent_mean"] = float(ent.mean())
    feat["ch_top1_mean"] = float(top1.mean())
    feat["ch3_top1_rate"] = float(np.mean(ranks[:, 0] == ch3_idx))
    feat["ch3_top2_rate"] = float(np.mean(np.any(ranks[:, :2] == ch3_idx, axis=1)))

    return feat


# -----------------------------
# group tests (improved vs else)
# -----------------------------

@dataclass
class GroupTestConfig:
    split: str = "val"
    target_channel: str = "check3"
    smooth: int = 1

    n_boot: int = 2000
    n_perm: int = 5000
    seed: int = 0

    improved_label: int = 0

def build_feature_table(batch, model, outdir: str, cfg: GroupTestConfig) -> pd.DataFrame:
    OUT = ensure_dir(outdir)
    data = extract_attn_tensors(batch, model, split=cfg.split)

    attn = data["attn"]
    ch_alpha = data["ch_alpha"]
    dPHQ = data["dPHQ"]
    y0PHQ = data["y0PHQ"]
    pids = data["patients"]
    check_names = data["check_names"]

    ch3_idx = find_channel_index(check_names, cfg.target_channel)
    fcfg = FeatureConfig(target_channel=cfg.target_channel, smooth=cfg.smooth)

    rows = []
    for i in range(attn.shape[0]):
        feats = compute_features_for_patient(attn[i], ch_alpha[i], ch3_idx, fcfg)
        rows.append({
            "pid": pids[i],
            "split": cfg.split,
            "dPHQ": int(dPHQ[i]),
            "y0PHQ": int(y0PHQ[i]),
            "check3_idx": int(ch3_idx),
            **feats
        })

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, f"per_patient_features_{cfg.split}.csv"), index=False)
    return df

def run_group_tests(df: pd.DataFrame, outdir: str, cfg: GroupTestConfig) -> pd.DataFrame:
    OUT = ensure_dir(outdir)

    g1 = df["dPHQ"].values == cfg.improved_label
    g2 = ~g1

    exclude = {"pid", "split", "dPHQ", "y0PHQ", "check3_idx"}
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

    res.to_csv(os.path.join(OUT, f"stats_tests_{cfg.split}.csv"), index=False)

    # candidates
    cand = res[(res["fdr_q"] <= 0.10) & (np.abs(res["cohens_d"]) >= 0.20)].copy()
    cand["abs_cohens_d"] = np.abs(cand["cohens_d"].values.astype(float))
    cand = cand.sort_values(["fdr_q", "perm_p", "abs_cohens_d"], ascending=[True, True, False])
    cand.to_csv(os.path.join(OUT, f"candidate_findings_{cfg.split}.csv"), index=False)

    return res


# -----------------------------
# validation suite
# -----------------------------

def compute_ch3_dpeak_val_and_tstar(ch_alpha: np.ndarray, ch3_idx: int, smooth: int = 1) -> Tuple[np.ndarray, np.ndarray]:
    """
    ch_alpha: (B,T,C)
    Returns:
      marker: (B,) = max |Δ ch3_attn|
      tstar:  (B,) = argmax |Δ ch3_attn|
    """
    B, T, C = ch_alpha.shape
    marker = np.zeros(B, dtype=float)
    tstar = np.zeros(B, dtype=int)
    for i in range(B):
        ch3 = _smooth_1d(ch_alpha[i, :, ch3_idx], smooth)
        d = np.diff(ch3, prepend=ch3[0])
        a = np.abs(d)
        t = int(np.argmax(a))
        marker[i] = float(a[t])
        tstar[i] = t
    return marker, tstar

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
        "auc_mean": float(np.mean(aucs)),
        "auc_ci95_lo": float(np.quantile(aucs, 0.025)),
        "auc_ci95_hi": float(np.quantile(aucs, 0.975)),
        "n_boot_used": int(len(aucs)),
    }

@torch.no_grad()
def eval_model_phq(model, S, C, y0p, y0p4, y0lon, dPHQ_true) -> Dict[str, float]:
    if accuracy_score is None:
        raise ImportError("sklearn is required for acc/F1. Please install scikit-learn.")
    logit_phq, _, _, _, _ = model.forward_multi(S, C, y0p, y0p4, y0lon)
    pred = torch.argmax(logit_phq, dim=1).detach().cpu().numpy().astype(int)
    y = dPHQ_true.detach().cpu().numpy().astype(int)
    return {
        "acc": float(accuracy_score(y, pred)),
        "macro_f1": float(f1_score(y, pred, average="macro")),
    }

def ablate_check3_input(C: torch.Tensor, ch3_idx: int, mode: str = "permute_time", seed: int = 0) -> torch.Tensor:
    rng = np.random.default_rng(seed)
    C2 = C.clone()
    B, T, Cc = C2.shape

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

@torch.no_grad()
def forward_phq_and_attn(model, S, C, y0p, y0p4, y0lon):
    model.eval()
    logit_phq, logit_p4, logit_lon, attn, ch_alpha = model.forward_multi(S, C, y0p, y0p4, y0lon)
    return logit_phq, attn, ch_alpha

@torch.no_grad()
def _true_class_logit(logit_phq: torch.Tensor, y_true: torch.Tensor) -> np.ndarray:
    B = logit_phq.size(0)
    idx = torch.arange(B, device=logit_phq.device)
    v = logit_phq[idx, y_true.long()].detach().cpu().numpy().astype(float)
    return v

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


@dataclass
class ValidationConfig:
    split: str = "val"
    target_channel: str = "check3"
    improved_label: int = 0

    smooth_attn: int = 3

    # AUC
    n_boot_auc: int = 3000

    # window faithfulness
    half_width: int = 1
    n_random: int = 50
    n_perm_window: int = 8000
    window_mode: str = "set_patient_mean"

    # model ablation
    ablation_modes: Optional[List[str]] = None

    seed: int = 0


def run_validation(batch, model, outdir: str, cfg: ValidationConfig) -> Dict[str, Any]:
    OUT = ensure_dir(outdir)
    data = extract_attn_tensors(batch, model, split=cfg.split)

    S = data["S"]
    C = data["C"]
    y0p = data["y0p"]
    y0p4 = data["y0p4"]
    y0lon = data["y0lon"]
    dPHQ_np = data["dPHQ"]
    y0PHQ_np = data["y0PHQ"]
    pids = data["patients"]
    check_names = data["check_names"]

    ch3_idx = find_channel_index(check_names, cfg.target_channel)

    # forward once (get ch_alpha for marker)
    logit0, attn0, ch0 = forward_phq_and_attn(model, S, C, y0p, y0p4, y0lon)
    marker, tstar = compute_ch3_dpeak_val_and_tstar(ch0.detach().cpu().numpy(), ch3_idx, smooth=cfg.smooth_attn)

    df_marker = pd.DataFrame({
        "pid": pids,
        "dPHQ": dPHQ_np.astype(int),
        "y0PHQ": y0PHQ_np.astype(int),
        "ch3_dpeak_val": marker.astype(float),
        "t_star": tstar.astype(int),
    })
    marker_path = os.path.join(OUT, f"marker_ch3_dpeak_{cfg.split}.csv")
    df_marker.to_csv(marker_path, index=False)

    # V1: AUC
    y_bin = (dPHQ_np == cfg.improved_label).astype(int)
    auc = float(roc_auc_score(y_bin, marker)) if roc_auc_score is not None and len(np.unique(y_bin)) == 2 else np.nan
    boot = _bootstrap_auc(y_bin, marker, n_boot=cfg.n_boot_auc, seed=cfg.seed) if roc_auc_score is not None else {}
    v1 = {**boot, "auc": auc}

    # V4: window faithfulness (targeted vs random)
    y_true_t = torch.tensor(dPHQ_np, device=logit0.device)
    base_true_logit = _true_class_logit(logit0, y_true_t)

    # targeted window around tstar
    C_targ = ablate_check3_window(C, ch3_idx, tstar, half_width=cfg.half_width,
                                  mode=cfg.window_mode, seed=cfg.seed)
    logit_t, _, _ = forward_phq_and_attn(model, S, C_targ, y0p, y0p4, y0lon)
    targ_true_logit = _true_class_logit(logit_t, y_true_t)
    delta_targ = targ_true_logit - base_true_logit

    # random windows
    rng = np.random.default_rng(cfg.seed)
    B, T, _ = C.shape
    deltas_rand = []
    for r in range(cfg.n_random):
        centers = rng.integers(0, T, size=B)
        C_r = ablate_check3_window(C, ch3_idx, centers, half_width=cfg.half_width,
                                   mode=cfg.window_mode, seed=cfg.seed + 13 + r)
        logit_r, _, _ = forward_phq_and_attn(model, S, C_r, y0p, y0p4, y0lon)
        rand_true_logit = _true_class_logit(logit_r, y_true_t)
        deltas_rand.append(rand_true_logit - base_true_logit)
    delta_rand_mean = np.mean(np.stack(deltas_rand, axis=0), axis=0)

    diff = delta_targ - delta_rand_mean
    p_win = _perm_test_signflip(diff, n_perm=cfg.n_perm_window, seed=cfg.seed)

    v4 = {
        "mean_delta_targeted": float(np.mean(delta_targ)),
        "mean_delta_random": float(np.mean(delta_rand_mean)),
        "mean_diff(targeted-random)": float(np.mean(diff)),
        "perm_p": float(p_win),
        "half_width": int(cfg.half_width),
        "n_random": int(cfg.n_random),
        "window_mode": cfg.window_mode,
        "note": "Negative delta means true-class logit dropped. If targeted causes larger drop than random, evidence of faithfulness.",
    }
    win_path = os.path.join(OUT, f"window_faithfulness_{cfg.split}.csv")
    pd.DataFrame([v4]).to_csv(win_path, index=False)

    # V3: model-level ablation on INPUT check3
    modes = cfg.ablation_modes or ["permute_time", "set_patient_mean", "set_global_mode"]
    base = eval_model_phq(model, S, C, y0p, y0p4, y0lon, torch.tensor(dPHQ_np, device=S.device))
    rows = [{"mode": "baseline", **base, "delta_acc": 0.0, "delta_macro_f1": 0.0}]

    for m in modes:
        C_abl = ablate_check3_input(C, ch3_idx, mode=m, seed=cfg.seed)
        met = eval_model_phq(model, S, C_abl, y0p, y0p4, y0lon, torch.tensor(dPHQ_np, device=S.device))
        rows.append({
            "mode": m,
            **met,
            "delta_acc": float(met["acc"] - base["acc"]),
            "delta_macro_f1": float(met["macro_f1"] - base["macro_f1"]),
        })

    v3_df = pd.DataFrame(rows)
    ab_path = os.path.join(OUT, f"model_ablation_{cfg.split}.csv")
    v3_df.to_csv(ab_path, index=False)

    return {
        "marker_csv": marker_path,
        "v1_auc": v1,
        "v4_window_faithfulness": v4,
        "v3_model_ablation": v3_df,
        "paths": {
            "marker_csv": marker_path,
            "window_csv": win_path,
            "ablation_csv": ab_path,
        }
    }


# -----------------------------
# One-shot runner: group-tests + validation
# -----------------------------

@dataclass
class FullPipelineConfig:
    group: GroupTestConfig = GroupTestConfig()
    val: ValidationConfig = ValidationConfig()

def run_all(batch, model, outdir: str, cfg: Optional[FullPipelineConfig] = None) -> Dict[str, Any]:
    cfg = cfg or FullPipelineConfig()
    OUT = ensure_dir(outdir)

    # 1) feature table + group tests
    df_feat = build_feature_table(batch, model, outdir=OUT, cfg=cfg.group)
    df_tests = run_group_tests(df_feat, outdir=OUT, cfg=cfg.group)

    # 2) validation suite
    val_out = run_validation(batch, model, outdir=OUT, cfg=cfg.val)

    return {
        "df_features": df_feat,
        "df_tests": df_tests,
        "validation": val_out,
        "paths": {
            "features_csv": os.path.join(OUT, f"per_patient_features_{cfg.group.split}.csv"),
            "tests_csv": os.path.join(OUT, f"stats_tests_{cfg.group.split}.csv"),
            "candidates_csv": os.path.join(OUT, f"candidate_findings_{cfg.group.split}.csv"),
            **val_out["paths"],
        }
    }
