# -*- coding: utf-8 -*-
"""
Channel-attention analysis for your PHQ multi-task seq model.

What it does:
- Extracts channel attention ch_alpha from model.forward_multi -> (B,T,6)
- Builds A_ch as either (N,C) mean over time or keeps (N,T,C)
- Compares attention importance between:
    * improved (Δ=0) vs else (Δ=1/2)
    * and also per-class summaries (Δ=0/1/2)
- Reports:
    * per-class mean/std
    * improved-vs-else: mean diff, Cliff's delta, permutation p-value
    * channel rank by overall mean importance
- Saves CSV/JSON into outdir

Usage:
- import and call run_check3_attn_analysis(...)
- or run as script (see __main__ at bottom)

NOTE:
- This script assumes your training code exists somewhere, providing:
    build_dataset, train_model, load_model
    and a model object with forward_multi(...) returning ch_alpha.
"""

import os, json
import numpy as np
import pandas as pd


# ----------------------------
# Utils
# ----------------------------
def ensure_dir(d: str) -> str:
    os.makedirs(d, exist_ok=True)
    return d

def _safe_mean(x, axis=None):
    x = np.asarray(x)
    return float(np.nanmean(x, axis=axis))

def _safe_std(x, axis=None):
    x = np.asarray(x)
    return float(np.nanstd(x, axis=axis))

def cliffs_delta(x, y):
    """
    Cliff's delta effect size for two distributions.
    delta in [-1,1]. Negative => x tends to be smaller than y.
    """
    x = np.asarray(x)
    y = np.asarray(y)
    x = x[np.isfinite(x)]
    y = y[np.isfinite(y)]
    nx = len(x); ny = len(y)
    if nx == 0 or ny == 0:
        return np.nan
    # O(nx*ny) is fine here because C=6; but x,y lengths may be large.
    # Use rank-based approximation to be safe.
    # Exact Cliff's delta can be computed from Mann–Whitney U:
    # delta = (2U)/(nx*ny) - 1
    import scipy.stats as st
    U, _ = st.mannwhitneyu(x, y, alternative="two-sided")
    delta = (2.0 * U) / (nx * ny) - 1.0
    return float(delta)

def permutation_pvalue(x, y, n_perm=2000, seed=42):
    """
    Two-sided permutation test for difference in means.
    Returns p-value.
    """
    rng = np.random.default_rng(seed)
    x = np.asarray(x); y = np.asarray(y)
    x = x[np.isfinite(x)]; y = y[np.isfinite(y)]
    if len(x) == 0 or len(y) == 0:
        return np.nan

    obs = abs(np.mean(x) - np.mean(y))
    pooled = np.concatenate([x, y], axis=0)
    n_x = len(x)
    count = 0

    for _ in range(int(n_perm)):
        rng.shuffle(pooled)
        x_p = pooled[:n_x]
        y_p = pooled[n_x:]
        stat = abs(np.mean(x_p) - np.mean(y_p))
        if stat >= obs:
            count += 1
    # add-one smoothing
    return float((count + 1) / (n_perm + 1))


# ----------------------------
# Core analysis
# ----------------------------
def analyze_channel_attention(
    A_ch,
    y,
    channel_names,
    check3_idx=2,
    task_name="PHQΔ",
    reduce_time="mean",  # "mean" or "sum" or None (keep T)
    n_perm=2000,
    seed=42
):
    """
    Parameters
    ----------
    A_ch : np.ndarray
        channel attention array (N,C) or (N,T,C)
    y : np.ndarray
        labels (N,), values in {0,1,2}. -1 allowed (ignored).
    channel_names : list[str]
        length C
    check3_idx : int
        index for check3
    reduce_time : str or None
        if A_ch is (N,T,C):
          - "mean": average over T -> (N,C)
          - "sum":  sum over T -> (N,C)
          - None : keep (N,T,C) but summary will still reduce internally
    Returns
    -------
    dict with:
      - summary_df : per-class mean/std per channel
      - improved_vs_else : dict per channel (mean_improved, mean_else, diff, p_value, cliffs_delta)
      - channel_rank_df : overall mean rank
    """
    A_ch = np.asarray(A_ch)
    y = np.asarray(y)
    assert A_ch.ndim in (2, 3), f"A_ch must be (N,C) or (N,T,C), got {A_ch.shape}"
    N = A_ch.shape[0]
    assert y.shape[0] == N, f"y length {y.shape[0]} != N {N}"

    C = A_ch.shape[-1]
    assert len(channel_names) == C, f"channel_names length {len(channel_names)} != C {C}"

    # filter labeled
    m = (y >= 0)
    A = A_ch[m]
    yy = y[m]

    # reduce time to (N,C) for stats
    if A.ndim == 3:
        if reduce_time == "mean":
            A2 = A.mean(axis=1)
        elif reduce_time == "sum":
            A2 = A.sum(axis=1)
        elif reduce_time is None:
            # keep, but for stats we still need a (N,C)
            A2 = A.mean(axis=1)
        else:
            raise ValueError("reduce_time must be one of ['mean','sum',None]")
    else:
        A2 = A  # (N,C)

    # 1) per-class summary
    rows = []
    for cls in [0, 1, 2]:
        mm = (yy == cls)
        if not mm.any():
            continue
        for j, ch in enumerate(channel_names):
            vals = A2[mm, j]
            rows.append({
                "task": task_name,
                "class": int(cls),
                "class_name": {0:"improved",1:"same",2:"worse"}.get(int(cls), str(cls)),
                "channel": ch,
                "mean": _safe_mean(vals),
                "std": _safe_std(vals),
                "n": int(mm.sum())
            })
    summary_df = pd.DataFrame(rows)

    # 2) improved vs else
    imp = (yy == 0)
    els = (yy != 0)
    improved_vs_else = {}

    for j, ch in enumerate(channel_names):
        x = A2[imp, j]
        z = A2[els, j]

        mean_imp = float(np.mean(x)) if len(x) else np.nan
        mean_else = float(np.mean(z)) if len(z) else np.nan
        diff = mean_imp - mean_else

        p = permutation_pvalue(x, z, n_perm=n_perm, seed=seed) if (len(x) and len(z)) else np.nan
        try:
            cd = cliffs_delta(x, z)
        except Exception:
            cd = np.nan

        improved_vs_else[ch] = {
            "mean_improved": mean_imp,
            "mean_else": mean_else,
            "mean_diff": float(diff),
            "p_value_perm": p,
            "cliffs_delta": cd,
            "n_improved": int(len(x)),
            "n_else": int(len(z)),
        }

    # 3) rank by overall mean
    overall_mean = A2.mean(axis=0)  # (C,)
    rank_df = pd.DataFrame({
        "channel": channel_names,
        "overall_mean": overall_mean
    }).sort_values("overall_mean", ascending=False).reset_index(drop=True)
    rank_df["rank"] = np.arange(1, len(rank_df) + 1)

    # handy check3 summary
    check3_name = channel_names[check3_idx] if 0 <= check3_idx < len(channel_names) else None

    out = {
        "summary_df": summary_df,
        "improved_vs_else": improved_vs_else,
        "channel_rank_df": rank_df,
        "check3_name": check3_name,
    }
    return out


# ----------------------------
# Glue: extract A_ch from your batch+model (torch)
# ----------------------------
def extract_channel_attention_from_model(batch, model, split="val"):
    """
    Returns:
      A_ch: (B,T,C) channel attention
      y   : (B,) ΔPHQ labels
      channel_names: list[str]
    """
    import torch

    if split == "val":
        idx = batch.idx_val
    elif split == "train":
        idx = batch.idx_tr
    else:
        raise ValueError("split must be 'val' or 'train'")

    S   = batch.S[idx]
    C   = batch.C[idx]
    y0p = batch.y0_phq[idx]
    y0p4 = batch.y0_p4[idx]
    y0lon = batch.y0_lon[idx]
    dY  = batch.dY_phq[idx]

    model.eval()
    with torch.no_grad():
        _, _, _, _, ch_alpha = model.forward_multi(S, C, y0p, y0p4, y0lon)  # (B,T,C)

    A_ch = ch_alpha.detach().cpu().numpy()
    y = dY.detach().cpu().numpy()

    channel_names = batch.stats.get("check_basenames", None)
    if channel_names is None:
        check_cols = batch.stats.get("check_cols", [f"check{i+1}" for i in range(A_ch.shape[-1])])
        channel_names = [str(c).replace("_value", "") for c in check_cols]

    return A_ch, y, channel_names


def run_check3_attn_analysis(batch, model, outdir, split="val", n_perm=2000, seed=42):
    OUT = ensure_dir(outdir)
    A_ch, y, channel_names = extract_channel_attention_from_model(batch, model, split=split)

    check3_idx = channel_names.index("check3") if "check3" in channel_names else 2

    out = analyze_channel_attention(
        A_ch=A_ch,
        y=y,
        channel_names=channel_names,
        check3_idx=check3_idx,
        task_name=f"PHQΔ ({split})",
        reduce_time="mean",
        n_perm=n_perm,
        seed=seed
    )

    # save
    out["summary_df"].to_csv(os.path.join(OUT, f"channel_attn_summary_{split}.csv"), index=False)
    out["channel_rank_df"].to_csv(os.path.join(OUT, f"channel_attn_rank_{split}.csv"), index=False)
    with open(os.path.join(OUT, f"channel_attn_improved_vs_else_{split}.json"), "w", encoding="utf-8") as f:
        json.dump(out["improved_vs_else"], f, indent=2, ensure_ascii=False)

    # also: check3-only quick csv
    if out["check3_name"] is not None:
        ch = out["check3_name"]
        r = out["improved_vs_else"].get(ch, {})
        pd.DataFrame([{"channel": ch, **r}]).to_csv(
            os.path.join(OUT, f"check3_improved_vs_else_{split}.csv"), index=False
        )

    return out


# ----------------------------
# __main__ example
# ----------------------------
if __name__ == "__main__":
    """
    Example CLI run:
      python attn_channel_analysis.py

    You MUST provide your own environment that defines:
      build_dataset, train_model, load_model
    For script mode, simplest is:
      - put this file next to your big model script
      - edit the import below to match your module name
    """
    # ---- EDIT THIS IMPORT to your actual filename/module ----
    # from phq_multi_seq_model import build_dataset, train_model, load_model
    raise SystemExit(
        "This script is meant to be imported in your notebook, or edit __main__ imports for standalone runs."
    )
