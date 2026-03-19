# -*- coding: utf-8 -*-
"""
attn_check3_timestep_analysis.py

Goal
----
Compare improved vs else groups at the TIMESTEP level for check3 channel attention:
1) timestep slope: Δα(t) = α(t+1)-α(t)
2) find t_max where |mean_slope_imp(t) - mean_slope_else(t)| is maximized
3) visualize:
   - slope difference curve + t_max marker
   - mean attention trajectory for each group + t_max marker
4) export CSVs:
   - timestep_summary_{split}.csv
   - per_patient_series_{split}.npz (optional)

Assumptions
-----------
- You already have the same batch/model interface used in your pipeline:
  extract_attn_tensors(batch, model, split) returns dict with:
    "dPHQ": (B,) int labels 0/1/2
    "patients": (B,) patient ids
    "check_names": list of check channel names
    "ch_alpha": (B,T,C) channel attention
- improved label = 0 by default
"""

import os
from dataclasses import dataclass
from typing import Dict, Any, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# -----------------------------
# small utils
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

def find_channel_index(check_names, target: str = "check3") -> int:
    target_l = target.strip().lower()
    for i, n in enumerate(check_names):
        if str(n).strip().lower() == target_l:
            return i
    for i, n in enumerate(check_names):
        if target_l in str(n).strip().lower():
            return i
    raise ValueError(f"Cannot find channel '{target}' from check_names={check_names}")


# -----------------------------
# core timestep analysis
# -----------------------------
@dataclass
class TimeStepCfg:
    split: str = "val"
    target_channel: str = "check3"
    improved_label: int = 0
    smooth: int = 1          # smoothing over time for attention series
    save_npz: bool = False   # optionally save per-patient series to npz


def analyze_check3_timestep(batch, model, outdir: str,
                            extract_attn_tensors_fn,
                            cfg: Optional[TimeStepCfg] = None) -> Dict[str, Any]:
    """
    extract_attn_tensors_fn must be a callable:
      data = extract_attn_tensors_fn(batch, model, split=cfg.split)
      returns dict containing at least: dPHQ, patients, check_names, ch_alpha

    Returns a dict with:
      t_max, tables, saved paths
    """
    cfg = cfg or TimeStepCfg()
    OUT = ensure_dir(outdir)

    data = extract_attn_tensors_fn(batch, model, split=cfg.split)

    dPHQ = np.asarray(data["dPHQ"]).astype(int)                 # (B,)
    pids = np.asarray(data["patients"])
    check_names = list(data["check_names"])
    ch_alpha = np.asarray(data["ch_alpha"]).astype(float)       # (B,T,C)

    ch3_idx = find_channel_index(check_names, cfg.target_channel)
    ch3_attn = ch_alpha[:, :, ch3_idx]                          # (B,T)

    # optional smoothing over time
    if cfg.smooth > 1:
        ch3_attn_sm = np.stack([_smooth_1d(ch3_attn[i], cfg.smooth) for i in range(ch3_attn.shape[0])], axis=0)
    else:
        ch3_attn_sm = ch3_attn

    # slope Δα(t) = α(t+1)-α(t)
    slope = ch3_attn_sm[:, 1:] - ch3_attn_sm[:, :-1]            # (B,T-1)

    mask_imp = (dPHQ == cfg.improved_label)
    mask_else = ~mask_imp

    if mask_imp.sum() == 0 or mask_else.sum() == 0:
        raise ValueError(f"Empty group: n_improved={mask_imp.sum()}, n_else={mask_else.sum()}")

    mean_attn_imp = ch3_attn_sm[mask_imp].mean(axis=0)          # (T,)
    mean_attn_else = ch3_attn_sm[mask_else].mean(axis=0)

    mean_slope_imp = slope[mask_imp].mean(axis=0)               # (T-1,)
    mean_slope_else = slope[mask_else].mean(axis=0)

    diff_slope = mean_slope_imp - mean_slope_else               # (T-1,)
    t_max = int(np.argmax(np.abs(diff_slope)))                  # index in slope domain (0..T-2)

    # make summary table
    T = ch3_attn_sm.shape[1]
    ts_attn = np.arange(T)
    ts_slope = np.arange(T - 1)

    df_attn = pd.DataFrame({
        "t": ts_attn,
        "mean_attn_improved": mean_attn_imp,
        "mean_attn_else": mean_attn_else,
        "diff_attn(improved-else)": mean_attn_imp - mean_attn_else,
        "n_improved": int(mask_imp.sum()),
        "n_else": int(mask_else.sum()),
    })

    df_slope = pd.DataFrame({
        "t": ts_slope,
        "mean_slope_improved": mean_slope_imp,
        "mean_slope_else": mean_slope_else,
        "diff_slope(improved-else)": diff_slope,
        "abs_diff_slope": np.abs(diff_slope),
        "is_t_max": (ts_slope == t_max).astype(int),
        "n_improved": int(mask_imp.sum()),
        "n_else": int(mask_else.sum()),
    })

    # save CSV
    attn_csv = os.path.join(OUT, f"timestep_attn_{cfg.split}.csv")
    slope_csv = os.path.join(OUT, f"timestep_slope_{cfg.split}.csv")
    df_attn.to_csv(attn_csv, index=False)
    df_slope.to_csv(slope_csv, index=False)

    # plots
    fig1 = os.path.join(OUT, f"plot_slope_diff_{cfg.split}.png")
    plt.figure(figsize=(9, 4))
    plt.plot(ts_slope, diff_slope, label="diff_slope (improved-else)")
    plt.axhline(0, linewidth=1)
    plt.axvline(t_max, linestyle="--", label=f"t_max={t_max}")
    plt.xlabel("t (slope index, corresponds to t -> t+1)")
    plt.ylabel("mean Δattention difference")
    plt.title(f"Check3 attention slope difference (split={cfg.split})")
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig1, dpi=200)
    plt.close()

    fig2 = os.path.join(OUT, f"plot_attn_mean_{cfg.split}.png")
    plt.figure(figsize=(9, 4))
    plt.plot(ts_attn, mean_attn_imp, label="mean attention (improved)", linewidth=2)
    plt.plot(ts_attn, mean_attn_else, label="mean attention (else)", linewidth=2)
    plt.axvline(t_max, linestyle="--", label=f"t_max={t_max}")
    plt.xlabel("timestep")
    plt.ylabel("mean check3 attention")
    plt.title(f"Check3 mean attention trajectory (split={cfg.split})")
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig2, dpi=200)
    plt.close()

    # optional: save per-patient arrays for later deep dives
    npz_path = None
    if cfg.save_npz:
        npz_path = os.path.join(OUT, f"per_patient_check3_series_{cfg.split}.npz")
        np.savez_compressed(
            npz_path,
            pid=pids,
            dPHQ=dPHQ,
            ch3_attn=ch3_attn_sm,
            ch3_slope=slope,
            ch3_idx=np.array([ch3_idx], dtype=int),
            check_names=np.array(check_names, dtype=object),
        )

    return {
        "t_max": t_max,
        "ch3_idx": ch3_idx,
        "paths": {
            "attn_csv": attn_csv,
            "slope_csv": slope_csv,
            "plot_slope": fig1,
            "plot_attn": fig2,
            "npz": npz_path,
        },
        "tables": {
            "attn": df_attn,
            "slope": df_slope,
        }
    }
