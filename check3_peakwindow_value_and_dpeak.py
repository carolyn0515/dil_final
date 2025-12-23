# -*- coding: utf-8 -*-
"""
check3_peakwindow_value_and_dpeak.py

Goal
-----
(1) 핵심 time-attention 구간들(예: t=34, 38)에서 check3 원값(0/1) 분포를 improved/else로 비교
(2) 같은 구간에서 check3 상태(dpeak: x - peak)를 요약해 improved/else 비교
(3) 결과 csv/fig 저장

Assumptions
-----------
- approach3.py provides:
  - build_dataset(raw1_csv, raw2_csv, T, outdir, seed) -> Batch
  - load_model(outdir, device, filename) -> model
  - model.forward_multi(S, C, y0_phq, y0_p4, y0_lon) -> (..., attn, ch_alpha)
    (approach3 eval path shows this exact call pattern)  :contentReference[oaicite:2]{index=2}
- check3 is categorical binary in batch.Craw (0/1)

Outputs
-------
- per_patient_windows_<split>.csv
- group_summary_<split>.csv
- figs:
  - window_check3_raw_rate_<split>.png
  - window_check3_dpeak_rate_<split>.png
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from typing import Dict, Any, List, Tuple

import numpy as np
import pandas as pd
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import approach3


# -----------------------------
# helpers
# -----------------------------
def ensure_dir(d: str) -> str:
    os.makedirs(d, exist_ok=True)
    return d


def find_check_index(batch: "approach3.Batch", check_name: str = "check3") -> int:
    # same logic as in your check3_joint_analysis helper. :contentReference[oaicite:3]{index=3}
    names = batch.stats.get("check_basenames", batch.stats.get("check_cols", []))
    if not names:
        raise ValueError("Cannot find check names in batch.stats.")
    check_name = check_name.strip().lower()
    for i, n in enumerate(names):
        if str(n).strip().lower() == check_name:
            return i
    for i, n in enumerate(names):
        if str(n).strip().lower().startswith(check_name):
            return i
    for i, n in enumerate(names):
        if check_name in str(n).strip().lower():
            return i
    raise ValueError(f"check_name='{check_name}' not found in check_basenames={names}")


def get_split_indices(batch: "approach3.Batch", split: str) -> np.ndarray:
    if split == "val":
        return np.asarray(batch.idx_val, dtype=int)
    if split == "train":
        return np.asarray(batch.idx_tr, dtype=int)
    raise ValueError("split must be 'val' or 'train'")


@torch.no_grad()
def extract_attn_and_raw(batch: "approach3.Batch", model: torch.nn.Module, split: str) -> Dict[str, Any]:
    idx = get_split_indices(batch, split)
    S = batch.S[idx]
    C = batch.C[idx]
    y0p = batch.y0_phq[idx]
    y0p4 = batch.y0_p4[idx]
    y0lon = batch.y0_lon[idx]
    dY = batch.dY_phq[idx].detach().cpu().numpy().astype(int)

    # patient ids (same idea as your joint analysis script) :contentReference[oaicite:4]{index=4}
    all_pids = batch.stats.get("patients", None)
    if all_pids is None:
        pids = np.array([f"idx{int(i)}" for i in idx], dtype=object)
    else:
        pids = np.array([all_pids[int(i)] for i in idx], dtype=object)

    model.eval()
    # forward_multi returns attn, ch_alpha as last two tensors in approach3 usage :contentReference[oaicite:5]{index=5}
    _, _, _, attn, ch_alpha = model.forward_multi(S, C, y0p, y0p4, y0lon)
    attn = attn.detach().cpu().numpy()          # (B,T)
    ch_alpha = ch_alpha.detach().cpu().numpy()  # (B,T,C)

    # raw check values for dynamics/stat (Craw exists in Batch definition) :contentReference[oaicite:6]{index=6}
    Craw = batch.Craw[idx].detach().cpu().numpy()  # (B,T,C)

    return dict(pids=pids, dY=dY, attn=attn, ch_alpha=ch_alpha, Craw=Craw)


def compute_joint(attn: np.ndarray, ch_alpha: np.ndarray, check_idx: int) -> np.ndarray:
    """
    j_t = a_t * b_t (per patient)
    - Same definition as your joint analysis: j = a * b :contentReference[oaicite:7]{index=7}
    """
    a = attn
    b = ch_alpha[:, :, check_idx]
    return a * b  # (B,T)


def window_slice(T: int, center: int, half_width: int) -> slice:
    s = max(0, int(center) - int(half_width))
    e = min(T - 1, int(center) + int(half_width))
    return slice(s, e + 1)


def compute_dpeak_from_raw(x: np.ndarray) -> np.ndarray:
    """
    x: (B,T) raw check3 in {0,1}
    dpeak = x - peak_i where peak_i = max_t x_it
    - For binary, dpeak ∈ {0, -1} for peak=1 cohort, or {0} if peak=0.
    """
    peak = np.max(x, axis=1, keepdims=True)  # (B,1) in {0,1}
    return x - peak


def bootstrap_diff_mean(x0: np.ndarray, x1: np.ndarray, n_boot: int = 2000, seed: int = 0) -> Dict[str, float]:
    rng = np.random.default_rng(seed)
    x0 = np.asarray(x0, dtype=float)
    x1 = np.asarray(x1, dtype=float)
    n0, n1 = len(x0), len(x1)
    diffs = []
    for _ in range(n_boot):
        s0 = x0[rng.integers(0, n0, size=n0)]
        s1 = x1[rng.integers(0, n1, size=n1)]
        diffs.append(np.mean(s0) - np.mean(s1))
    diffs = np.asarray(diffs, dtype=float)
    return dict(
        diff_mean=float(np.mean(x0) - np.mean(x1)),
        ci95_lo=float(np.quantile(diffs, 0.025)),
        ci95_hi=float(np.quantile(diffs, 0.975)),
    )


def perm_test_diff_mean(x0: np.ndarray, x1: np.ndarray, n_perm: int = 5000, seed: int = 0) -> float:
    rng = np.random.default_rng(seed)
    x0 = np.asarray(x0, dtype=float)
    x1 = np.asarray(x1, dtype=float)
    obs = float(np.mean(x0) - np.mean(x1))
    z = np.concatenate([x0, x1], axis=0)
    n0 = len(x0)
    cnt = 0
    for _ in range(n_perm):
        rng.shuffle(z)
        d = float(np.mean(z[:n0]) - np.mean(z[n0:]))
        if abs(d) >= abs(obs):
            cnt += 1
    return float((cnt + 1) / (n_perm + 1))


@dataclass
class Config:
    raw1: str
    raw2: str
    ckpt_dir: str
    ckpt_name: str
    outdir: str = "figs_check3_peakwindows"
    split: str = "val"
    T: int = 40
    seed: int = 42

    check_name: str = "check3"
    windows: List[int] = None
    half_width: int = 1

    n_boot: int = 2000
    n_perm: int = 5000


def run(cfg: Config) -> Dict[str, Any]:
    OUT = ensure_dir(cfg.outdir)

    # build dataset (same pipeline entrypoint)
    batch = approach3.build_dataset(cfg.raw1, cfg.raw2, T=cfg.T, outdir=cfg.ckpt_dir, seed=cfg.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = approach3.load_model(cfg.ckpt_dir, device=device, filename=cfg.ckpt_name)

    data = extract_attn_and_raw(batch, model, split=cfg.split)
    pids = data["pids"]
    dY = data["dY"]  # 0/1/2 where 0=improved (approach3 delta3 definition) :contentReference[oaicite:8]{index=8}
    improved = (dY == 0)
    else_mask = ~improved

    check_idx = find_check_index(batch, cfg.check_name)

    Craw = data["Craw"][:, :, check_idx]  # (B,T) raw check3
    dpeak = compute_dpeak_from_raw(Craw)  # (B,T)

    # joint attention for possible future use (not mandatory for this step)
    j = compute_joint(data["attn"], data["ch_alpha"], check_idx)  # (B,T)
    T = j.shape[1]

    if cfg.windows is None or len(cfg.windows) == 0:
        cfg.windows = [34, 38]

    rows = []
    for t0 in cfg.windows:
        W = window_slice(T, t0, cfg.half_width)

        # raw check3 summary in window
        raw_mean = np.mean(Craw[:, W], axis=1)          # (B,)
        raw_any1 = (np.max(Craw[:, W], axis=1) >= 1).astype(float)

        # dpeak summary in window (binary: 0 or -1)
        dpeak_min = np.min(dpeak[:, W], axis=1)         # (B,)
        post_event = (dpeak_min < 0).astype(float)      # 1 if -1 observed in window

        # store per-patient
        for i in range(len(pids)):
            rows.append(dict(
                pid=str(pids[i]),
                split=cfg.split,
                dY_phq=int(dY[i]),
                group_improved=int(improved[i]),
                window_center=int(t0),
                half_width=int(cfg.half_width),
                raw_mean=float(raw_mean[i]),
                raw_any1=float(raw_any1[i]),
                dpeak_min=float(dpeak_min[i]),
                post_event=float(post_event[i]),
            ))

    df = pd.DataFrame(rows)
    per_path = os.path.join(OUT, f"per_patient_windows_{cfg.split}.csv")
    df.to_csv(per_path, index=False)

    # group summaries per window
    summ_rows = []
    for t0 in cfg.windows:
        sub = df[df["window_center"] == int(t0)]
        for col in ["raw_mean", "raw_any1", "dpeak_min", "post_event"]:
            x0 = sub.loc[sub["group_improved"] == 1, col].values.astype(float)
            x1 = sub.loc[sub["group_improved"] == 0, col].values.astype(float)

            boot = bootstrap_diff_mean(x0, x1, n_boot=cfg.n_boot, seed=cfg.seed + 11 + int(t0))
            p = perm_test_diff_mean(x0, x1, n_perm=cfg.n_perm, seed=cfg.seed + 23 + int(t0))

            summ_rows.append(dict(
                split=cfg.split,
                window_center=int(t0),
                metric=col,
                n_improved=int(len(x0)),
                n_else=int(len(x1)),
                mean_improved=float(np.mean(x0)),
                mean_else=float(np.mean(x1)),
                diff_mean_improved_minus_else=float(boot["diff_mean"]),
                ci95_lo=float(boot["ci95_lo"]),
                ci95_hi=float(boot["ci95_hi"]),
                perm_p=float(p),
            ))

    df_s = pd.DataFrame(summ_rows)
    summ_path = os.path.join(OUT, f"group_summary_{cfg.split}.csv")
    df_s.to_csv(summ_path, index=False)

    # figures: per window raw_any1 rate and post_event rate
    def plot_rate(metric: str, title: str, fname: str):
        xs = []
        ys0 = []
        ys1 = []
        for t0 in cfg.windows:
            sub = df[df["window_center"] == int(t0)]
            m0 = sub["group_improved"] == 1
            m1 = sub["group_improved"] == 0
            xs.append(int(t0))
            ys0.append(float(np.mean(sub.loc[m0, metric].values.astype(float))))
            ys1.append(float(np.mean(sub.loc[m1, metric].values.astype(float))))

        x = np.arange(len(xs))
        fig, ax = plt.subplots(1, 1, figsize=(7, 3))
        ax.bar(x - 0.15, ys0, width=0.3, label="improved")
        ax.bar(x + 0.15, ys1, width=0.3, label="else")
        ax.set_xticks(x)
        ax.set_xticklabels([f"t={v}" for v in xs])
        ax.set_ylim(0.0, 1.0)
        ax.set_ylabel("rate")
        ax.set_title(title)
        ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(OUT, fname), dpi=150)
        plt.close(fig)

    plot_rate(
        metric="raw_any1",
        title=f"check3=1 rate in attention windows ({cfg.split.upper()})",
        fname=f"window_check3_raw_rate_{cfg.split}.png"
    )
    plot_rate(
        metric="post_event",
        title=f"post-event rate (dpeak<0) in attention windows ({cfg.split.upper()})",
        fname=f"window_check3_dpeak_rate_{cfg.split}.png"
    )

    return dict(
        per_patient_csv=per_path,
        summary_csv=summ_path,
        outdir=OUT,
        windows=cfg.windows,
    )


def parse_args() -> Config:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw1", required=True, help="raw weekly CSV (raw_data1.csv)")
    ap.add_argument("--raw2", required=True, help="raw survey CSV (raw_data2.csv)")
    ap.add_argument("--ckpt_dir", required=True, help="directory containing checkpoint (same as approach3 outdir)")
    ap.add_argument("--ckpt_name", default="phq_multi_seq.ckpt")
    ap.add_argument("--outdir", default="figs_check3_peakwindows")
    ap.add_argument("--split", default="val", choices=["val", "train"])
    ap.add_argument("--T", type=int, default=40)
    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument("--check_name", default="check3")
    ap.add_argument("--windows", nargs="*", type=int, default=[34, 38], help="window centers, e.g. 34 38")
    ap.add_argument("--half_width", type=int, default=1)

    ap.add_argument("--n_boot", type=int, default=2000)
    ap.add_argument("--n_perm", type=int, default=5000)
    args = ap.parse_args()

    return Config(
        raw1=args.raw1,
        raw2=args.raw2,
        ckpt_dir=args.ckpt_dir,
        ckpt_name=args.ckpt_name,
        outdir=args.outdir,
        split=args.split,
        T=args.T,
        seed=args.seed,
        check_name=args.check_name,
        windows=list(args.windows),
        half_width=args.half_width,
        n_boot=args.n_boot,
        n_perm=args.n_perm,
    )


if __name__ == "__main__":
    cfg = parse_args()
    out = run(cfg)
    print("[DONE]")
    print(out)
