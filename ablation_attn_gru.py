# -*- coding: utf-8 -*-
"""
ablation_attn_gru.py

Purpose
-------
Prove (with experiments) why:
  (1) attention is necessary (performance + interpretability)
  (2) GRU is a reasonable backbone (stability vs alternatives)

Assumptions
-----------
- You already have approach3.py (your main file) that contains:
  - Batch dataclass
  - build_dataset(...)
  - _eval_task(...) or equivalent metrics helper
  - SeqModelPHQ (your attention+GRU multi-task model)

This file imports build_dataset + Batch and provides controlled ablations.

What it produces
----------------
- CSV summary table: ablation_summary.csv
- JSON per-run logs: ablation_runs/*.json
- Optional plots: learning curves etc. (kept minimal)
"""

import os, json, copy, random
from dataclasses import dataclass, asdict
from typing import Dict, Any, Optional, List, Tuple

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score

# ---- import from your main model file ----
# If your file is not named approach3.py, rename below accordingly.
import approach3 as base


# -------------------------
# Repro / utils
# -------------------------
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def ensure_dir(d: str) -> str:
    os.makedirs(d, exist_ok=True)
    return d


def metrics_only(dY: torch.Tensor, logits: torch.Tensor) -> Dict[str, float]:
    mask = (dY >= 0)
    if not mask.any():
        return {"acc": np.nan, "f1": np.nan, "n": 0}
    y_true = dY[mask].detach().cpu().numpy()
    y_pred = logits[mask].argmax(-1).detach().cpu().numpy()
    return {
        "acc": float(accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred, average="macro")),
        "n": int(len(y_true)),
    }


# ============================================================
# 1) Model variants: remove temporal attn / channel attn / both
# ============================================================

class SeqModel_AblateAttention(base.SeqModelPHQ):
    """
    Inherits your main SeqModelPHQ, but allows turning off:
      - temporal attention (use mean pooling or last hidden)
      - channel attention (use uniform channel weights)
    """

    def __init__(
        self,
        *args,
        use_time_attn: bool = True,
        time_pool: str = "attn",   # "attn" | "last" | "mean"
        use_channel_attn: bool = True,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.use_time_attn = use_time_attn
        self.time_pool = time_pool
        self.use_channel_attn = use_channel_attn

    def _forward_all(self, C_in, y0_phq, y0_p4, y0_lon):
        """
        Same outputs as your original:
          logit_phq, logit_p4, logit_lon, attn, ch_alpha
        but attention parts can be ablated.
        """
        B, T, d_c = C_in.size()
        assert d_c == self.d_c

        y0e = self._y0_embed_all(y0_phq, y0_p4, y0_lon)
        h_t = self.y0_to_h0(y0e)
        t_idx = torch.arange(T, device=C_in.device)

        h_list = []
        alpha_list = []

        # ---------- recurrent roll-out ----------
        for t in range(T):
            c_t = C_in[:, t, :]
            pos_t = self.pos(t_idx[t]).unsqueeze(0).expand(B, -1)

            if self.use_channel_attn:
                z_t = torch.cat([c_t, y0e, h_t], dim=-1)
                ch_logits_t = self.ch_attn(z_t)
                ch_alpha_t = F.softmax(ch_logits_t, dim=-1)
            else:
                # uniform channel weights
                ch_alpha_t = torch.full_like(c_t, 1.0 / d_c)

            c_t_weighted = c_t * ch_alpha_t

            x_parts = [c_t, c_t_weighted, pos_t]
            if self.use_y0_in_step:
                x_parts.append(y0e)
            x_t = torch.cat(x_parts, dim=-1)

            h_t = self.gru_cell(x_t, h_t)
            h_list.append(h_t.unsqueeze(1))
            alpha_list.append(ch_alpha_t.unsqueeze(1))

        h = torch.cat(h_list, dim=1)       # (B,T,hs)
        ch_alpha = torch.cat(alpha_list, dim=1)  # (B,T,d_c)
        h = self.drop(h)

        # ---------- temporal aggregation ----------
        if (not self.use_time_attn) or (self.time_pool in ["last", "mean"]):
            if self.time_pool == "last":
                context = h[:, -1, :]  # (B,hs)
                attn = torch.full((B, T), 1.0 / T, device=h.device)
            else:
                # mean pooling
                context = h.mean(dim=1)
                attn = torch.full((B, T), 1.0 / T, device=h.device)
        else:
            # original temporal attention
            y0_expand = y0e.unsqueeze(1).expand(B, T, y0e.size(-1))
            attn_logits = self.attn_v(torch.tanh(self.attn_h(h) + self.attn_y(y0_expand))).squeeze(-1)
            attn = F.softmax(attn_logits, dim=1)
            context = torch.sum(attn.unsqueeze(-1) * h, dim=1)

        enc_in = torch.cat([context, y0e], dim=-1)
        h_enc = self.enc(enc_in)

        logit_phq = self.head_phq(h_enc)
        logit_p4  = self.head_p4(h_enc)
        logit_lon = self.head_lon(h_enc)

        return logit_phq, logit_p4, logit_lon, attn, ch_alpha


# ============================================================
# 2) GRU alternatives: LSTM and TransformerEncoder
# ============================================================

class SeqModel_LSTM_Attn(nn.Module):
    """
    LSTM backbone + (optional) temporal attention.
    Keeps the same y0 embeddings + y0 masking logic from base.SeqModelPHQ.
    For fairness, we reuse base.SeqModelPHQ masking / loss methods by composition.
    """
    def __init__(
        self,
        d_c: int,
        d_pos: int = 16,
        hs: int = 128,
        d_y0_each: int = 16,
        use_time_attn: bool = True,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.use_time_attn = use_time_attn
        self.hs = hs
        self.d_c = d_c

        self.pos = nn.Embedding(128, d_pos)

        # y0 embeddings (same shapes)
        self.y0_emb_phq = nn.Embedding(4 + 1, d_y0_each)
        self.y0_emb_p4  = nn.Embedding(3 + 1, d_y0_each)
        self.y0_emb_lon = nn.Embedding(2 + 1, d_y0_each)
        self.y0_dim = 3 * d_y0_each

        self.y0_to_h0 = nn.Sequential(nn.Linear(self.y0_dim, hs), nn.Tanh())
        self.y0_to_c0 = nn.Sequential(nn.Linear(self.y0_dim, hs), nn.Tanh())

        # LSTMCell input: [C_t, pos_t, y0e]
        self.cell = nn.LSTMCell(d_c + d_pos + self.y0_dim, hs)
        self.drop = nn.Dropout(dropout)

        # temporal attention
        attn_dim = hs
        self.attn_h = nn.Linear(hs, attn_dim, bias=False)
        self.attn_y = nn.Linear(self.y0_dim, attn_dim, bias=False)
        self.attn_v = nn.Linear(attn_dim, 1, bias=False)

        self.enc = nn.Sequential(nn.Linear(hs + self.y0_dim, 256), nn.ReLU(), nn.Dropout(0.2))
        self.head_phq = nn.Linear(256, 3)
        self.head_p4  = nn.Linear(256, 3)
        self.head_lon = nn.Linear(256, 3)

        # reuse masking + loss from base
        self._masker = base.SeqModelPHQ(d_c=d_c, d_y0_each=d_y0_each)

    def _y0_embed_all(self, y0_phq, y0_p4, y0_lon):
        phq = torch.where(y0_phq < 0, torch.full_like(y0_phq, 4), y0_phq).clamp(0, 4)
        p4  = torch.where(y0_p4  < 0, torch.full_like(y0_p4,  3), y0_p4 ).clamp(0, 3)
        lon = torch.where(y0_lon < 0, torch.full_like(y0_lon, 2), y0_lon).clamp(0, 2)
        e = torch.cat([self.y0_emb_phq(phq), self.y0_emb_p4(p4), self.y0_emb_lon(lon)], dim=-1)
        return e

    def forward_multi(self, S, C_in, y0_phq, y0_p4, y0_lon):
        B, T, d_c = C_in.size()
        y0e = self._y0_embed_all(y0_phq, y0_p4, y0_lon)

        h_t = self.y0_to_h0(y0e)
        c_t = self.y0_to_c0(y0e)

        t_idx = torch.arange(T, device=C_in.device)
        h_list = []

        for t in range(T):
            pos_t = self.pos(t_idx[t]).unsqueeze(0).expand(B, -1)
            x_t = torch.cat([C_in[:, t, :], pos_t, y0e], dim=-1)
            h_t, c_t = self.cell(x_t, (h_t, c_t))
            h_list.append(h_t.unsqueeze(1))

        h = self.drop(torch.cat(h_list, dim=1))  # (B,T,hs)

        if self.use_time_attn:
            y0_expand = y0e.unsqueeze(1).expand(B, T, y0e.size(-1))
            attn_logits = self.attn_v(torch.tanh(self.attn_h(h) + self.attn_y(y0_expand))).squeeze(-1)
            attn = F.softmax(attn_logits, dim=1)
            context = torch.sum(attn.unsqueeze(-1) * h, dim=1)
        else:
            attn = torch.full((B, T), 1.0 / T, device=h.device)
            context = h.mean(dim=1)

        h_enc = self.enc(torch.cat([context, y0e], dim=-1))
        return self.head_phq(h_enc), self.head_p4(h_enc), self.head_lon(h_enc), attn, None

    def loss(self, *args, **kwargs):
        # delegate to base implementation (same signature)
        return self._masker.loss(*args, **kwargs)

    @staticmethod
    def _delta_mask_from_y0(y0, y_max):
        return base.SeqModelPHQ._delta_mask_from_y0(y0, y_max)


class SeqModel_Transformer(nn.Module):
    """
    TransformerEncoder backbone (small) + pooling.
    Kept intentionally tiny to avoid overfitting.
    """
    def __init__(self, d_c: int, hs: int = 128, nhead: int = 4, nlayers: int = 2, d_y0_each: int = 16):
        super().__init__()
        self.d_c = d_c
        self.hs = hs

        self.y0_emb_phq = nn.Embedding(5, d_y0_each)
        self.y0_emb_p4  = nn.Embedding(4, d_y0_each)
        self.y0_emb_lon = nn.Embedding(3, d_y0_each)
        self.y0_dim = 3 * d_y0_each

        self.in_proj = nn.Linear(d_c + self.y0_dim, hs)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=hs, nhead=nhead, dim_feedforward=hs * 4, dropout=0.1, batch_first=True
        )
        self.enc = nn.TransformerEncoder(enc_layer, num_layers=nlayers)

        self.out = nn.Sequential(nn.Linear(hs + self.y0_dim, 256), nn.ReLU(), nn.Dropout(0.2))
        self.head_phq = nn.Linear(256, 3)
        self.head_p4  = nn.Linear(256, 3)
        self.head_lon = nn.Linear(256, 3)

        self._masker = base.SeqModelPHQ(d_c=d_c, d_y0_each=d_y0_each)

    def _y0_embed_all(self, y0_phq, y0_p4, y0_lon):
        phq = torch.where(y0_phq < 0, torch.full_like(y0_phq, 4), y0_phq).clamp(0, 4)
        p4  = torch.where(y0_p4  < 0, torch.full_like(y0_p4,  3), y0_p4 ).clamp(0, 3)
        lon = torch.where(y0_lon < 0, torch.full_like(y0_lon, 2), y0_lon).clamp(0, 2)
        return torch.cat([self.y0_emb_phq(phq), self.y0_emb_p4(p4), self.y0_emb_lon(lon)], dim=-1)

    def forward_multi(self, S, C_in, y0_phq, y0_p4, y0_lon):
        B, T, _ = C_in.size()
        y0e = self._y0_embed_all(y0_phq, y0_p4, y0_lon)          # (B,y0dim)
        y0_seq = y0e.unsqueeze(1).expand(B, T, y0e.size(-1))     # (B,T,y0dim)
        x = torch.cat([C_in, y0_seq], dim=-1)
        x = self.in_proj(x)                                     # (B,T,hs)
        h = self.enc(x)                                         # (B,T,hs)

        # simple mean pool (keep it stable)
        context = h.mean(dim=1)
        attn = torch.full((B, T), 1.0 / T, device=h.device)

        z = self.out(torch.cat([context, y0e], dim=-1))
        return self.head_phq(z), self.head_p4(z), self.head_lon(z), attn, None

    def loss(self, *args, **kwargs):
        return self._masker.loss(*args, **kwargs)

    @staticmethod
    def _delta_mask_from_y0(y0, y_max):
        return base.SeqModelPHQ._delta_mask_from_y0(y0, y_max)


# ============================================================
# 3) Training / evaluation wrapper (seed-repeated)
# ============================================================

@dataclass
class AblationConfig:
    outdir: str = "figs_ablation_attn_gru"
    epochs: int = 50
    lr: float = 1e-3
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
    es_patience: int = 8
    es_min_delta: float = 1e-4
    seeds: Tuple[int, ...] = (41, 42, 43)
    # fairness: same task weights as your main training
    w_phq: float = 3.0
    w_p4: float = 1.0
    w_lon: float = 1.0
    use_y0_mask: bool = True
    use_class_weight_phq: bool = True


def train_one_model(model: nn.Module, batch: base.Batch, cfg: AblationConfig, seed: int) -> Dict[str, Any]:
    set_seed(seed)
    OUT = ensure_dir(cfg.outdir)

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    best_val = float("inf")
    best_state = None
    patience = cfg.es_patience

    def run_split(idx, train: bool):
        S = batch.S[idx]
        C = batch.C[idx]
        y0_phq = batch.y0_phq[idx]
        y0_p4  = batch.y0_p4[idx]
        y0_lon = batch.y0_lon[idx]
        dY_phq = batch.dY_phq[idx]
        dY_p4  = batch.dY_p4[idx]
        dY_lon = batch.dY_lon[idx]

        if train:
            model.train()
            opt.zero_grad()
            logit_phq, logit_p4, logit_lon, attn, ch_alpha = model.forward_multi(S, C, y0_phq, y0_p4, y0_lon)
            loss, comp = model.loss(
                logit_phq, logit_p4, logit_lon,
                dY_phq, dY_p4, dY_lon,
                y0_phq, y0_p4, y0_lon,
                use_y0_mask=cfg.use_y0_mask,
                w_phq=cfg.w_phq, w_p4=cfg.w_p4, w_lon=cfg.w_lon,
                use_class_weight_phq=cfg.use_class_weight_phq,
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            opt.step()
            return float(loss.item()), comp

        else:
            model.eval()
            with torch.no_grad():
                logit_phq, logit_p4, logit_lon, attn, ch_alpha = model.forward_multi(S, C, y0_phq, y0_p4, y0_lon)

                # apply y0 feasibility masks for eval consistency
                m_phq = model._delta_mask_from_y0(y0_phq, y_max=3)
                m_p4  = model._delta_mask_from_y0(y0_p4,  y_max=2)
                m_lon = model._delta_mask_from_y0(y0_lon, y_max=1)

                logit_phq = logit_phq.masked_fill(~m_phq, -1e9)
                logit_p4  = logit_p4.masked_fill(~m_p4,  -1e9)
                logit_lon = logit_lon.masked_fill(~m_lon, -1e9)

                loss, comp = model.loss(
                    logit_phq, logit_p4, logit_lon,
                    dY_phq, dY_p4, dY_lon,
                    y0_phq, y0_p4, y0_lon,
                    use_y0_mask=cfg.use_y0_mask,
                    w_phq=cfg.w_phq, w_p4=cfg.w_p4, w_lon=cfg.w_lon,
                    use_class_weight_phq=cfg.use_class_weight_phq,
                )

                met_phq = metrics_only(dY_phq, logit_phq)
                met_p4  = metrics_only(dY_p4,  logit_p4)
                met_lon = metrics_only(dY_lon, logit_lon)

            return float(loss.item()), comp, met_phq, met_p4, met_lon

    hist = []
    for ep in range(1, cfg.epochs + 1):
        tr_loss, _ = run_split(batch.idx_tr, train=True)
        va_loss, _, m1, m2, m3 = run_split(batch.idx_val, train=False)
        hist.append({"ep": ep, "train_loss": tr_loss, "val_loss": va_loss})

        if va_loss < best_val - cfg.es_min_delta:
            best_val = va_loss
            best_state = copy.deepcopy(model.state_dict())
            patience = cfg.es_patience
        else:
            patience -= 1
            if patience <= 0:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    # final eval
    _, _, met_phq, met_p4, met_lon = run_split(batch.idx_val, train=False)

    return {
        "seed": seed,
        "best_val_loss": best_val,
        "metrics_phq": met_phq,
        "metrics_p4": met_p4,
        "metrics_lon": met_lon,
        "hist": hist,
    }


def run_ablation_suite(batch: base.Batch, cfg: AblationConfig) -> pd.DataFrame:
    OUT = ensure_dir(cfg.outdir)
    ensure_dir(os.path.join(OUT, "ablation_runs"))

    device = batch.C.device
    levels = batch.levels.tolist()

    # ---- Define experiment variants ----
    variants = []

    # Baseline: GRU + time attn + channel attn (your current design)
    variants.append(("GRU+TimeAttn+ChAttn", lambda: SeqModel_AblateAttention(
        d_s=batch.S.size(-1), d_c=batch.C.size(-1), d_pos=16, hs=128,
        lambda_c=0.0, d_y0_each=16, use_y0_in_step=True, levels=levels, dropout=0.1,
        use_time_attn=True, time_pool="attn", use_channel_attn=True
    ).to(device)))

    # Attention ablations
    variants.append(("GRU+NoTimeAttn(Last)", lambda: SeqModel_AblateAttention(
        d_s=batch.S.size(-1), d_c=batch.C.size(-1), d_pos=16, hs=128,
        lambda_c=0.0, d_y0_each=16, use_y0_in_step=True, levels=levels, dropout=0.1,
        use_time_attn=False, time_pool="last", use_channel_attn=True
    ).to(device)))

    variants.append(("GRU+NoTimeAttn(Mean)", lambda: SeqModel_AblateAttention(
        d_s=batch.S.size(-1), d_c=batch.C.size(-1), d_pos=16, hs=128,
        lambda_c=0.0, d_y0_each=16, use_y0_in_step=True, levels=levels, dropout=0.1,
        use_time_attn=False, time_pool="mean", use_channel_attn=True
    ).to(device)))

    variants.append(("GRU+NoChAttn", lambda: SeqModel_AblateAttention(
        d_s=batch.S.size(-1), d_c=batch.C.size(-1), d_pos=16, hs=128,
        lambda_c=0.0, d_y0_each=16, use_y0_in_step=True, levels=levels, dropout=0.1,
        use_time_attn=True, time_pool="attn", use_channel_attn=False
    ).to(device)))

    variants.append(("GRU+NoAttn(Both)", lambda: SeqModel_AblateAttention(
        d_s=batch.S.size(-1), d_c=batch.C.size(-1), d_pos=16, hs=128,
        lambda_c=0.0, d_y0_each=16, use_y0_in_step=True, levels=levels, dropout=0.1,
        use_time_attn=False, time_pool="mean", use_channel_attn=False
    ).to(device)))

    # GRU choice comparisons
    variants.append(("LSTM+TimeAttn", lambda: SeqModel_LSTM_Attn(
        d_c=batch.C.size(-1), d_pos=16, hs=128, d_y0_each=16, use_time_attn=True
    ).to(device)))

    variants.append(("Transformer(meanpool)", lambda: SeqModel_Transformer(
        d_c=batch.C.size(-1), hs=128, nhead=4, nlayers=2, d_y0_each=16
    ).to(device)))

    all_rows = []
    for name, ctor in variants:
        for seed in cfg.seeds:
            model = ctor()
            res = train_one_model(model, batch, cfg, seed=seed)

            row = {
                "variant": name,
                "seed": seed,
                "best_val_loss": res["best_val_loss"],

                "phq_acc": res["metrics_phq"]["acc"],
                "phq_f1":  res["metrics_phq"]["f1"],
                "p4_acc":  res["metrics_p4"]["acc"],
                "p4_f1":   res["metrics_p4"]["f1"],
                "lon_acc": res["metrics_lon"]["acc"],
                "lon_f1":  res["metrics_lon"]["f1"],
            }
            all_rows.append(row)

            # save per-run json
            with open(os.path.join(OUT, "ablation_runs", f"{name}__seed{seed}.json"), "w") as f:
                json.dump(res, f, indent=2)

    df = pd.DataFrame(all_rows)

    # aggregate over seeds
    agg = df.groupby("variant").agg({
        "phq_acc": ["mean", "std"],
        "phq_f1":  ["mean", "std"],
        "p4_acc":  ["mean", "std"],
        "p4_f1":   ["mean", "std"],
        "lon_acc": ["mean", "std"],
        "lon_f1":  ["mean", "std"],
        "best_val_loss": ["mean", "std"],
    })
    agg.columns = ["_".join(c).strip() for c in agg.columns.values]
    agg = agg.reset_index()

    # compute drops vs baseline
    base_row = agg[agg["variant"] == "GRU+TimeAttn+ChAttn"].iloc[0]
    for metric in ["phq_acc_mean", "phq_f1_mean", "p4_acc_mean", "p4_f1_mean", "lon_acc_mean", "lon_f1_mean"]:
        agg[f"drop_{metric}"] = base_row[metric] - agg[metric]

    out_csv = os.path.join(OUT, "ablation_summary.csv")
    agg.to_csv(out_csv, index=False)
    print("[ABLATION] summary saved ->", out_csv)
    return agg


if __name__ == "__main__":
    # Example main (optional).
    RAW1 = "data/raw_data1.csv"
    RAW2 = "data/raw_data2.csv"
    OUT  = ensure_dir("figs_ablation_attn_gru")

    batch = base.build_dataset(RAW1, RAW2, T=40, outdir=OUT, seed=42)

    cfg = AblationConfig(
        outdir=OUT,
        epochs=80,
        lr=1e-3,
        seeds=(41, 42, 43),
    )

    df_summary = run_ablation_suite(batch, cfg)
    print(df_summary)
