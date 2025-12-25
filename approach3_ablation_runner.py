# -*- coding: utf-8 -*-
"""
approach3_ablation_runner.py  (CHECK3 DROP 버전)

- approach3.py는 수정하지 않음
- approach3의 build_dataset / Batch / eval 유틸을 그대로 사용
- "y0 input / check3 drop / time attention / channel attention" ablation을 위한
  별도 모델(Variant) + 러너 제공

사용 (ipynb):
    from approach3_ablation_runner import run_suite
    results = run_suite(RAW1, RAW2, OUT, T=40, epochs=120, lr=1e-3, seed=42)

주의:
- check3_drop은 "값을 0으로" 만드는 게 아니라
  입력 차원에서 check3 채널 자체를 제거(d_c 감소)하는 진짜 feature drop임.
"""

import os, json
from dataclasses import dataclass
from typing import Optional, List, Dict, Any, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# approach3.py에서 재사용
import approach3
from approach3 import ensure_dir, set_seed, Batch, build_dataset


# -------------------------
# Ablation config
# -------------------------
@dataclass
class AblationConfig:
    name: str

    # (A) y0 input ablation
    use_y0: bool = True               # y0 자체를 쓰는지(encoder/attn/step 모두)

    # (B) channel attention ablation
    use_channel_attn: bool = True     # False면 채널 가중치 uniform으로 고정

    # (C) time attention ablation
    time_pool: str = "attn"           # "attn"(base), "mean", "last", "uniform"

    # (D) check3 ablation (input-level) : TRUE DROP
    check3_mode: Optional[str] = None   # None | "drop"
    check3_name: str = "check3"         # stats["check_basenames"]에서 찾을 이름


# -------------------------
# Variant model
# -------------------------
class SeqModelPHQ_Variant(nn.Module):
    """
    approach3.SeqModelPHQ를 그대로 복붙하지 않고,
    핵심 구조는 동일하게 유지하되 "사용/비사용"만 깔끔하게 제어하는 variant.

    - use_y0=False:
        * y0 embedding = 0
        * y0_to_h0: learnable h0 파라미터로 대체
        * step 입력에서 y0 제외
        * time attention에서 y0 term 제외
        * encoder 입력에서 y0 제외

    - use_channel_attn=False:
        * channel weights = uniform (1/d_c)

    - time_pool:
        * "attn": additive attention
        * "mean": mean pooling
        * "last": last hidden
        * "uniform": time-attn을 uniform으로 강제(=mean과 유사하지만 attn을 반환하기 위해 분리)
    """
    def __init__(
        self,
        d_s: int = 16,
        d_c: int = 6,
        d_pos: int = 16,
        hs: int = 128,
        d_y0_each: int = 16,
        use_y0: bool = True,
        use_channel_attn: bool = True,
        time_pool: str = "attn",
        dropout: float = 0.1,
        levels: Optional[List[int]] = None,
    ):
        super().__init__()
        self.d_c = d_c
        self.hs = hs
        self.use_y0 = bool(use_y0)
        self.use_channel_attn = bool(use_channel_attn)
        self.time_pool = str(time_pool)

        # time embedding
        self.pos = nn.Embedding(128, d_pos)

        # --- y0 embeddings (base와 동일 스펙) ---
        # (use_y0=False면 forward에서 0으로 대체)
        self.y0_emb_phq = nn.Embedding(4 + 1, d_y0_each)  # 0..3, 4=unknown
        self.y0_emb_p4  = nn.Embedding(3 + 1, d_y0_each)  # 0..2, 3=unknown
        self.y0_emb_lon = nn.Embedding(2 + 1, d_y0_each)  # 0..1, 2=unknown
        self.y0_dim = 3 * d_y0_each

        # base: y0 -> h0, ablated: learnable h0
        self.y0_to_h0 = nn.Sequential(nn.Linear(self.y0_dim, hs), nn.Tanh())
        self.h0_param = nn.Parameter(torch.zeros(hs))

        # channel attention (base와 동일 구조)
        self.ch_attn = nn.Sequential(
            nn.Linear(d_c + self.y0_dim + hs, 64),
            nn.Tanh(),
            nn.Linear(64, d_c),
        )

        # GRUCell 입력 차원
        # base: [c_t, c_t_weighted, pos_t, y0e]
        # ablate y0: [c_t, c_t_weighted, pos_t]
        cin_dim = 2 * d_c + d_pos + (self.y0_dim if self.use_y0 else 0)
        self.gru_cell = nn.GRUCell(cin_dim, hs)
        self.drop = nn.Dropout(dropout)

        # additive time attention (base와 동일 구조)
        attn_dim = hs
        self.attn_h = nn.Linear(hs, attn_dim, bias=False)
        self.attn_y = nn.Linear(self.y0_dim, attn_dim, bias=False)
        self.attn_v = nn.Linear(attn_dim, 1, bias=False)

        # encoder + heads
        # base: [context, y0e], ablate y0: [context]
        enc_in = hs + (self.y0_dim if self.use_y0 else 0)
        self.enc = nn.Sequential(
            nn.Linear(enc_in, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
        )
        self.head_phq = nn.Linear(256, 3)
        self.head_p4  = nn.Linear(256, 3)
        self.head_lon = nn.Linear(256, 3)

        # 기록용(config)
        self._config = dict(
            d_s=d_s, d_c=d_c, d_pos=d_pos, hs=hs,
            d_y0_each=d_y0_each,
            use_y0=self.use_y0,
            use_channel_attn=self.use_channel_attn,
            time_pool=self.time_pool,
            dropout=dropout,
            levels=levels or [2]*d_c,
        )

    # -------- y0 embed (approach3와 동일 semantics) --------
    def _y0_embed_all(self, y0_phq, y0_p4, y0_lon):
        phq = y0_phq.clone()
        phq = torch.where(phq < 0, torch.full_like(phq, 4), phq)
        phq = torch.clamp(phq, 0, 4)

        p4 = y0_p4.clone()
        p4 = torch.where(p4 < 0, torch.full_like(p4, 3), p4)
        p4 = torch.clamp(p4, 0, 3)

        lon = y0_lon.clone()
        lon = torch.where(lon < 0, torch.full_like(lon, 2), lon)
        lon = torch.clamp(lon, 0, 2)

        e_phq = self.y0_emb_phq(phq.long())
        e_p4  = self.y0_emb_p4(p4.long())
        e_lon = self.y0_emb_lon(lon.long())
        return torch.cat([e_phq, e_p4, e_lon], dim=-1)  # (B, y0_dim)

    # y0 기반 Δ 마스크/CE는 approach3 구현을 그대로 사용(재사용)
    _delta_mask_from_y0 = staticmethod(approach3.SeqModelPHQ._delta_mask_from_y0)
    _masked_ce          = staticmethod(approach3.SeqModelPHQ._masked_ce)

    def loss(
        self,
        logit_phq, logit_p4, logit_lon,
        dY_phq, dY_p4, dY_lon,
        y0_phq, y0_p4, y0_lon,
        use_y0_mask: bool = True,
        w_phq: float = 2.0, w_p4: float = 1.0, w_lon: float = 1.0,
        use_class_weight_phq: bool = True
    ):
        # approach3.SeqModelPHQ.loss 로직과 동일 계열
        class_weight_phq = None
        if use_class_weight_phq:
            with torch.no_grad():
                mask_valid = (dY_phq >= 0)
                cnt = torch.bincount(dY_phq[mask_valid].long(), minlength=3).float()
                cnt = torch.clamp(cnt, min=1.0)
                alpha = 0.5
                inv = (cnt.sum() / cnt) ** alpha
                class_weight_phq = inv / inv.mean()

        if use_y0_mask:
            m_phq = self._delta_mask_from_y0(y0_phq, y_max=3)
            m_p4  = self._delta_mask_from_y0(y0_p4,  y_max=2)
            m_lon = self._delta_mask_from_y0(y0_lon, y_max=1)

            ce_phq = self._masked_ce(logit_phq, dY_phq, m_phq, weight=class_weight_phq)
            ce_p4  = self._masked_ce(logit_p4,  dY_p4,  m_p4)
            ce_lon = self._masked_ce(logit_lon, dY_lon, m_lon)
        else:
            ce_phq = F.cross_entropy(logit_phq, dY_phq.long())
            ce_p4  = F.cross_entropy(logit_p4,  dY_p4.long())
            ce_lon = F.cross_entropy(logit_lon, dY_lon.long())

        num = w_phq * ce_phq + w_p4 * ce_p4 + w_lon * ce_lon
        den = w_phq + w_p4 + w_lon
        ce_mean = num / den

        comp = dict(
            ce_mean=float(ce_mean.item()),
            ce_phq=float(ce_phq.item()),
            ce_p4=float(ce_p4.item()),
            ce_lon=float(ce_lon.item()),
        )
        return ce_mean, comp

    def forward_multi(self, S, C_in, y0_phq, y0_p4, y0_lon):
        B, T, d_c = C_in.size()
        assert d_c == self.d_c, f"input d_c={d_c} != model d_c={self.d_c}"

        # y0 embedding
        if self.use_y0:
            y0e = self._y0_embed_all(y0_phq, y0_p4, y0_lon)     # (B, y0_dim)
            h_t = self.y0_to_h0(y0e)                             # (B, hs)
        else:
            y0e = torch.zeros(B, self.y0_dim, device=C_in.device, dtype=C_in.dtype)
            h_t = self.h0_param.unsqueeze(0).expand(B, -1)       # (B, hs)

        t_idx = torch.arange(T, device=C_in.device)

        h_list = []
        alpha_list = []

        for t in range(T):
            c_t = C_in[:, t, :]  # (B, d_c)
            pos_t = self.pos(t_idx[t]).unsqueeze(0).expand(B, -1)

            # channel attention
            if self.use_channel_attn:
                z_t = torch.cat([c_t, y0e, h_t], dim=-1)
                ch_logits_t = self.ch_attn(z_t)
                ch_alpha_t = F.softmax(ch_logits_t, dim=-1)
            else:
                ch_alpha_t = torch.full_like(c_t, 1.0 / d_c)  # uniform

            c_t_weighted = c_t * ch_alpha_t

            x_parts = [c_t, c_t_weighted, pos_t]
            if self.use_y0:
                x_parts.append(y0e)
            x_t = torch.cat(x_parts, dim=-1)

            h_t = self.gru_cell(x_t, h_t)
            h_list.append(h_t.unsqueeze(1))
            alpha_list.append(ch_alpha_t.unsqueeze(1))

        h = torch.cat(h_list, dim=1)            # (B,T,hs)
        ch_alpha = torch.cat(alpha_list, dim=1) # (B,T,d_c)
        h = self.drop(h)

        # time pooling
        if self.time_pool == "attn":
            if self.use_y0:
                y0_expand = y0e.unsqueeze(1).expand(B, T, y0e.size(-1))
                attn_logits = self.attn_v(
                    torch.tanh(self.attn_h(h) + self.attn_y(y0_expand))
                ).squeeze(-1)
            else:
                attn_logits = self.attn_v(torch.tanh(self.attn_h(h))).squeeze(-1)

            attn = F.softmax(attn_logits, dim=1)            # (B,T)
            context = torch.sum(attn.unsqueeze(-1) * h, dim=1)
        elif self.time_pool == "mean":
            attn = torch.full((B, T), 1.0 / T, device=h.device, dtype=h.dtype)
            context = h.mean(dim=1)
        elif self.time_pool == "uniform":
            attn = torch.full((B, T), 1.0 / T, device=h.device, dtype=h.dtype)
            context = torch.sum(attn.unsqueeze(-1) * h, dim=1)
        elif self.time_pool == "last":
            attn = torch.zeros((B, T), device=h.device, dtype=h.dtype)
            attn[:, -1] = 1.0
            context = h[:, -1, :]
        else:
            raise ValueError("time_pool must be one of: ['attn','mean','uniform','last']")

        # encoder input
        if self.use_y0:
            enc_in = torch.cat([context, y0e], dim=-1)
        else:
            enc_in = context

        h_enc = self.enc(enc_in)
        logit_phq = self.head_phq(h_enc)
        logit_p4  = self.head_p4(h_enc)
        logit_lon = self.head_lon(h_enc)

        return logit_phq, logit_p4, logit_lon, attn, ch_alpha


# -------------------------
# check3 DROP helper
# -------------------------
def _get_check_names(batch: Batch) -> List[str]:
    return batch.stats.get("check_basenames", batch.stats.get("check_cols", []))


def apply_check3_drop(batch: Batch, C: torch.Tensor, channel_name: str) -> torch.Tensor:
    """
    TRUE DROP: check3 채널 자체를 제거.
    C: (B,T,d_c) -> (B,T,d_c-1)
    """
    check_names = _get_check_names(batch)
    if channel_name not in check_names:
        raise ValueError(f"channel '{channel_name}' not found in check names: {check_names}")
    ch_idx = check_names.index(channel_name)
    return torch.cat([C[:, :, :ch_idx], C[:, :, ch_idx+1:]], dim=-1)


def drop_level(levels: List[int], batch: Batch, channel_name: str) -> List[int]:
    """
    levels 리스트에서도 check3에 해당하는 원소를 제거.
    """
    check_names = _get_check_names(batch)
    if channel_name not in check_names:
        raise ValueError(f"channel '{channel_name}' not found in check names: {check_names}")
    ch_idx = check_names.index(channel_name)
    return levels[:ch_idx] + levels[ch_idx+1:]


# -------------------------
# train / eval
# -------------------------
def train_variant(
    batch: Batch,
    cfg: AblationConfig,
    outdir: str,
    epochs: int = 300,
    lr: float = 1e-3,
    es_patience: int = 8,
    es_min_delta: float = 1e-4,
    seed: int = 42,
) -> Tuple[nn.Module, Dict[str, Any]]:
    set_seed(seed)
    OUT = ensure_dir(outdir)

    # ---- d_c / levels 결정 (check3 drop이면 차원 줄임) ----
    base_levels = batch.levels.tolist() if hasattr(batch, "levels") else [2] * batch.C.size(-1)

    if cfg.check3_mode == "drop":
        d_c = batch.C.size(-1) - 1
        levels = drop_level(base_levels, batch, cfg.check3_name)
    else:
        d_c = batch.C.size(-1)
        levels = base_levels

    model = SeqModelPHQ_Variant(
        d_s=batch.S.size(-1),
        d_c=d_c,
        d_pos=16,
        hs=128,
        d_y0_each=16,
        use_y0=cfg.use_y0,
        use_channel_attn=cfg.use_channel_attn,
        time_pool=cfg.time_pool,
        dropout=0.1,
        levels=levels,
    ).to(batch.C.device)

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    def run_split(idx, train: bool):
        S = batch.S[idx]
        C_in = batch.C[idx]
        y0_phq = batch.y0_phq[idx]
        y0_p4  = batch.y0_p4[idx]
        y0_lon = batch.y0_lon[idx]
        dY_phq = batch.dY_phq[idx]
        dY_p4  = batch.dY_p4[idx]
        dY_lon = batch.dY_lon[idx]

        # y0 input ablation: y0를 unknown(-1)로 만들어 동일 인터페이스 유지
        if not cfg.use_y0:
            y0_phq = torch.full_like(y0_phq, -1)
            y0_p4  = torch.full_like(y0_p4,  -1)
            y0_lon = torch.full_like(y0_lon, -1)

        # check3 TRUE DROP: 입력 C를 차원에서 제거
        if cfg.check3_mode == "drop":
            C_in = apply_check3_drop(batch, C_in, cfg.check3_name)

        if train:
            model.train()
            opt.zero_grad()
            logit_phq, logit_p4, logit_lon, attn, ch_alpha = model.forward_multi(S, C_in, y0_phq, y0_p4, y0_lon)
            loss, comp = model.loss(
                logit_phq, logit_p4, logit_lon,
                dY_phq, dY_p4, dY_lon,
                y0_phq, y0_p4, y0_lon,
                use_y0_mask=True,   # 논문에서 동일하게 유지
                w_phq=3,
                use_class_weight_phq=True,
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            return float(loss.item()), comp
        else:
            model.eval()
            with torch.no_grad():
                logit_phq, logit_p4, logit_lon, attn, ch_alpha = model.forward_multi(S, C_in, y0_phq, y0_p4, y0_lon)
                loss, comp = model.loss(
                    logit_phq, logit_p4, logit_lon,
                    dY_phq, dY_p4, dY_lon,
                    y0_phq, y0_p4, y0_lon,
                    use_y0_mask=True,
                    w_phq=3,
                    use_class_weight_phq=True,
                )
            return float(loss.item()), comp

    hist = {"ep": [], "train": [], "val": [], "ce_mean": []}
    best_val = float("inf")
    best_ep = -1
    patience_left = es_patience
    best_path = os.path.join(OUT, f"best_{cfg.name}.pt")

    for ep in range(1, epochs + 1):
        tr_loss, comp_tr = run_split(batch.idx_tr, train=True)
        va_loss, comp_va = run_split(batch.idx_val, train=False)

        hist["ep"].append(ep)
        hist["train"].append(tr_loss)
        hist["val"].append(va_loss)
        hist["ce_mean"].append(comp_va["ce_mean"])

        print(f"[{cfg.name}][EP{ep:03d}] train={tr_loss:.4f} val={va_loss:.4f} (CEmean={comp_va['ce_mean']:.4f})")

        if va_loss < best_val - es_min_delta:
            best_val = va_loss
            best_ep = ep
            patience_left = es_patience
            torch.save(model.state_dict(), best_path)
        else:
            patience_left -= 1
            if patience_left <= 0:
                print(f"[{cfg.name}][EarlyStop] ep={ep}, best_ep={best_ep}, best_val={best_val:.6f}")
                break

    # load best
    if os.path.exists(best_path):
        model.load_state_dict(torch.load(best_path, map_location=batch.C.device))
    model.eval()

    info = dict(history=hist, best_ep=best_ep, best_val=best_val, best_path=best_path)
    return model, info


@torch.no_grad()
def eval_variant_on_split(batch: Batch, model: nn.Module, cfg: AblationConfig, split: str, outdir: str):
    """
    approach3.eval_multi_on_split과 동일한 출력 형식(metrics dict)로 만들기.
    단, cfg에 따라 y0/check3 input을 동일하게 ablate한 상태로 평가.
    """
    OUT = ensure_dir(outdir)
    batch.stats["_outdir_for_eval"] = OUT

    if split == "val":
        idx = batch.idx_val
        prefix = "val"
    elif split == "train":
        idx = batch.idx_tr
        prefix = "train"
    else:
        raise ValueError("split must be 'train' or 'val'")

    S = batch.S[idx]
    C = batch.C[idx]
    y0p = batch.y0_phq[idx]
    y0p4 = batch.y0_p4[idx]
    y0lon = batch.y0_lon[idx]
    dYp = batch.dY_phq[idx]
    dYp4 = batch.dY_p4[idx]
    dYlon = batch.dY_lon[idx]

    if not cfg.use_y0:
        y0p = torch.full_like(y0p, -1)
        y0p4 = torch.full_like(y0p4, -1)
        y0lon = torch.full_like(y0lon, -1)

    if cfg.check3_mode == "drop":
        C = apply_check3_drop(batch, C, cfg.check3_name)

    logit_phq, logit_p4, logit_lon, attn, ch_alpha = model.forward_multi(S, C, y0p, y0p4, y0lon)

    # y0 제약은 동일하게 적용(논문에서 일관성 유지)
    m_phq = approach3.SeqModelPHQ._delta_mask_from_y0(y0p, y_max=3)
    m_p4  = approach3.SeqModelPHQ._delta_mask_from_y0(y0p4, y_max=2)
    m_lon = approach3.SeqModelPHQ._delta_mask_from_y0(y0lon, y_max=1)

    logit_phq = logit_phq.masked_fill(~m_phq, -1e9)
    logit_p4  = logit_p4.masked_fill(~m_p4,  -1e9)
    logit_lon = logit_lon.masked_fill(~m_lon, -1e9)

    metrics_phq = approach3._eval_task("PHQ-9", dYp, logit_phq, OUT, f"PHQ9_{prefix}_{cfg.name}")
    metrics_p4  = approach3._eval_task("P4", dYp4, logit_p4, OUT, f"P4_{prefix}_{cfg.name}")
    metrics_lon = approach3._eval_task("Loneliness", dYlon, logit_lon, OUT, f"Lon_{prefix}_{cfg.name}")
    return dict(metrics_phq=metrics_phq, metrics_p4=metrics_p4, metrics_lon=metrics_lon)


def summarize_metrics(m: Dict[str, Any]) -> Dict[str, float]:
    """
    metrics dict -> flat numbers for table
    """
    out = {}
    for task_key in ["metrics_phq", "metrics_p4", "metrics_lon"]:
        mm = m.get(task_key)
        if mm is None:
            out[f"{task_key}_acc"] = np.nan
            out[f"{task_key}_f1"] = np.nan
            continue
        out[f"{task_key}_acc"] = float(mm["acc"])
        out[f"{task_key}_f1"]  = float(mm["f1"])
    return out


def run_suite(
    RAW1: str,
    RAW2: str,
    OUT: str,
    T: int = 40,
    epochs: int = 300,
    lr: float = 1e-3,
    seed: int = 42,
):
    """
    base vs 핵심 ablation을 한번에 수행하고
    OUT/ablation_suite_results.json + OUT/ablation_suite_table.csv 저장
    """
    OUT = ensure_dir(OUT)
    set_seed(seed)

    # dataset
    batch = build_dataset(RAW1, RAW2, T=T, outdir=OUT, seed=seed)

    # 실험 설정들
    cfgs = [
        AblationConfig(
            name="BASE",
            use_y0=True,
            use_channel_attn=True,
            time_pool="attn",
            check3_mode=None,
        ),

        # 1) y0 input ablation
        AblationConfig(
            name="ABL_noY0",
            use_y0=False,
            use_channel_attn=True,
            time_pool="attn",
            check3_mode=None,
        ),

        # 2) check3 TRUE DROP (feature 제거)
        AblationConfig(
            name="ABL_check3_drop",
            use_y0=True,
            use_channel_attn=True,
            time_pool="attn",
            check3_mode="drop",
            check3_name="check3",
        ),

        # 3) time attention ablation (mean pooling)
        AblationConfig(
            name="ABL_time_mean",
            use_y0=True,
            use_channel_attn=True,
            time_pool="mean",
            check3_mode=None,
        ),

        # 4) channel attention ablation (uniform channels)
        AblationConfig(
            name="ABL_ch_uniform",
            use_y0=True,
            use_channel_attn=False,
            time_pool="attn",
            check3_mode=None,
        ),
    ]

    results: Dict[str, Any] = {}
    table_rows: List[Dict[str, Any]] = []

    for cfg in cfgs:
        exp_out = ensure_dir(os.path.join(OUT, f"ablation_{cfg.name}"))
        print("\n==============================")
        print("Running:", cfg)
        print("==============================\n")

        model, info = train_variant(
            batch, cfg, outdir=exp_out,
            epochs=epochs, lr=lr, seed=seed
        )

        metrics_val = eval_variant_on_split(batch, model, cfg, split="val", outdir=exp_out)

        results[cfg.name] = dict(
            cfg=cfg.__dict__,
            train_info=info,
            metrics_val=metrics_val,
        )

        flat = summarize_metrics(metrics_val)
        row = dict(name=cfg.name, **flat)
        table_rows.append(row)

        # per-exp json
        with open(os.path.join(exp_out, "result.json"), "w", encoding="utf-8") as f:
            json.dump(results[cfg.name], f, indent=2, ensure_ascii=False)

    # base 대비 drop 계산
    base = results["BASE"]["metrics_val"]
    base_flat = summarize_metrics(base)

    for row in table_rows:
        if row["name"] == "BASE":
            row["drop_PHQ_acc"] = 0.0
            row["drop_PHQ_f1"] = 0.0
            continue
        row["drop_PHQ_acc"] = float(base_flat["metrics_phq_acc"] - row["metrics_phq_acc"])
        row["drop_PHQ_f1"]  = float(base_flat["metrics_phq_f1"]  - row["metrics_phq_f1"])

    # save suite json/csv
    suite_path = os.path.join(OUT, "ablation_suite_results.json")
    with open(suite_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    # csv
    import pandas as pd
    df = pd.DataFrame(table_rows)
    csv_path = os.path.join(OUT, "ablation_suite_table.csv")
    df.to_csv(csv_path, index=False)

    print("\n[SAVED]")
    print(" -", suite_path)
    print(" -", csv_path)
    print(df.sort_values("drop_PHQ_f1", ascending=True))

    return results
