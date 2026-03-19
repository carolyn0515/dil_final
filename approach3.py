# -*- coding: utf-8 -*-
"""
멀티 설문 Seq 모델 + Attention 기반 상호작용 분석 (+ y0 기반 Δ 제약)

    y0(PHQ-9 / P4 / Lon) -> C(weekly checks seq) -> Δy_PHQ9 / Δy_P4 / Δy_Loneliness
- 결측 없는 환자만 채택(체크 + 세 설문 모두 기준)
- 서비스(s1..service16)는 모델/분석에서 사용하지 않음 (S는 Batch 안에만 정보용으로 존재)
- 체크는 approach_sc.py와 동일하게 전 채널 ordinal feature로 사용
- 끝단 Δ(0=improved, 1=same, 2=worse) 3-클래스 CE
- 중간 C 재구성(aux loss)는 제거
- GRU + additive attention 으로 어떤 시점의 C에 집중하는지 해석 가능하도록 설계
- 추가: y0에 따라 불가능한 Δ(improved/worse)를 막는 hard constraint 적용
"""

import os, json, random, re
from dataclasses import dataclass
from typing import List, Optional, Dict, Any

import numpy as np
import pandas as pd
from itertools import combinations
from sklearn.linear_model import LogisticRegression

# (옵션) 분석 유틸 있으면 사용
try:
    from analysis_util import *
    _HAS_AUTIL = True
except Exception:
    _HAS_AUTIL = False

try:
    import statsmodels.api as sm
    _HAS_SM = True
except Exception:
    _HAS_SM = False


import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ---------------- I/O & utils ----------------

def ensure_dir(d: str) -> str:
    os.makedirs(d, exist_ok=True)
    return d

def _to_int(x):
    try:
        return int(str(x).strip())
    except Exception:
        return np.nan

SURVEY_PHQ9       = "PHQ-9"
SURVEY_P4         = "P4"
SURVEY_LONELINESS = "Loneliness"

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def _extract_int(s: str, default: int = 10**9) -> int:
    m = re.search(r"(\d+)", s)
    return int(m.group(1)) if m else default

# 채널별 레벨 수 정의 (통계용 / 설명용)
LEVELS_PER_CHECK = {
    "check1": 3,  # {0,1,2}
    "check2": 2,  # 이진
    "check3": 2,  # 이진
    "check4": 2,  # 이진
    "check5": 3,  # {1,2,3} → 라벨 {0,1,2}로 해석
    "check6": 4,  # {0,1,2,3}
}

def load_weekly(raw1_csv: str, id_col: str = "menti_seq", time_col: str = "reg_date"):
    df = pd.read_csv(raw1_csv)

    # 1) check*_value 우선
    check_val_cols = [c for c in df.columns if c.startswith("check") and c.endswith("_value")]
    check_val_cols = sorted(check_val_cols, key=_extract_int)

    # 2) 없으면 *_type 제외하고 check* 6개 선택
    if len(check_val_cols) == 0:
        cand = [c for c in df.columns if c.startswith("check")]
        cand = [c for c in cand if not c.endswith("_type")]
        cand = sorted(cand, key=_extract_int)[:6]
        check_val_cols = cand

    if len(check_val_cols) < 6:
        raise AssertionError(f"expect 6 check value columns, got {len(check_val_cols)}: {check_val_cols}")
    check_val_cols = check_val_cols[:6]

    # service 16개 정렬 (모델에는 사용하지 않지만 정보용으로만 유지)
    service_cols = [c for c in df.columns if c.startswith("service")]
    service_cols = sorted(service_cols, key=_extract_int)[:16]
    if len(service_cols) < 16:
        raise AssertionError(f"expect 16 service columns, got {len(service_cols)}: {service_cols}")

    # 시간 정렬
    df[time_col] = pd.to_datetime(df[time_col], errors="coerce")
    df = df.sort_values([id_col, time_col]).reset_index(drop=True)
    return df, check_val_cols, service_cols, id_col, time_col


def load_surveys(raw2_csv: str, id_col: str = "menti_seq") -> pd.DataFrame:
    df = pd.read_csv(raw2_csv)
    df["srvy_result"] = df["srvy_result"].apply(_to_int)
    df["reg_date"] = pd.to_datetime(df["reg_date"], errors="coerce")
    return df[[id_col, "srvy_name", "srvy_result", "reg_date"]]\
        .dropna(subset=["srvy_name"]).reset_index(drop=True)


def first_last_survey(sdf: pd.DataFrame, name: str):
    """해당 환자 설문에서 특정 이름(name)의 최초/최종 값 반환 (둘 다 있어야 Δ 계산)."""
    g = sdf[sdf["srvy_name"] == name].sort_values("reg_date")
    if len(g) < 2:
        return np.nan, np.nan
    y0 = g["srvy_result"].iloc[0]
    yT = g["srvy_result"].iloc[-1]
    return y0, yT


def delta3(y0, yT) -> int:
    """
    y0, yT(raw)로부터 Δ 3-class 반환:
      - 결측: -1 (loss / metric 계산에서 제외)
      - dy < 0: 0 (improved)
      - dy = 0: 1 (same)
      - dy > 0: 2 (worse)
    """
    if np.isnan(y0) or np.isnan(yT):
        return -1
    dy = int(np.sign(yT - y0))  # -1,0,+1
    return dy + 1               # 0,1,2


# ---------------- Dataset build ----------------

@dataclass
class Batch:
    # weekly
    S: torch.Tensor        # (B,T,16) services (정보용)
    C: torch.Tensor        # (B,T,6)  checks (z-scored, 모델 입력)
    Craw: torch.Tensor     # (B,T,6)  checks (원시값, 통계용)
    levels: torch.Tensor   # (6,)     각 채널 level 수 (stat용)
    is_bin: torch.Tensor   # (6,)     이진 여부 (levels==2)

    # labels (초기 상태)
    y0_phq: torch.Tensor   # (B,) in {0,1,2,3}
    y0_p4:  torch.Tensor   # (B,) in {0,1,2}
    y0_lon: torch.Tensor   # (B,) in {0,1}

    # Δ labels
    dY_phq: torch.Tensor   # (B,) in {0,1,2}
    dY_p4:  torch.Tensor   # (B,) in {0,1,2}
    dY_lon: torch.Tensor   # (B,) in {0,1,2}

    idx_tr: np.ndarray
    idx_val: np.ndarray
    stats: Dict[str, Any]


def build_dataset(raw1_csv: str,
                  raw2_csv: str,
                  T: int = 40,
                  outdir: str = "figs_phq_multi",
                  seed: int = 42) -> Batch:
    set_seed(seed)
    r1, check_cols, service_cols, id_col, time_col = load_weekly(raw1_csv)
    r2 = load_surveys(raw2_csv, id_col=id_col)
    OUT = ensure_dir(outdir)

    S_list, C_list = [], []
    y0_phq_list, y0_p4_list, y0_lon_list = [], [], []
    dY_phq_list, dY_p4_list, dY_lon_list = [], [], []
    keep_ids = []

    for pid, g in r1.groupby(id_col):
        g = g.sort_values(time_col)

        # 정확히 T주 확보
        if len(g) < T:
            continue
        g = g.iloc[:T]

        # 체크에 결측 있으면 제외
        if g[check_cols].isna().any().any():
            continue

        # 설문 정보
        sdf = r2[r2[id_col] == pid]

        # --- PHQ-9 (필수) ---
        y0_phq, yT_phq = first_last_survey(sdf, SURVEY_PHQ9)
        if np.isnan(y0_phq) or np.isnan(yT_phq):
            continue
        y0_phq = int(y0_phq)
        dY_phq = delta3(y0_phq, yT_phq)

        # --- P4 (필수: 세 설문 모두 있는 cohort만 사용) ---
        y0_p4, yT_p4 = first_last_survey(sdf, SURVEY_P4)
        if np.isnan(y0_p4) or np.isnan(yT_p4):
            continue
        y0_p4 = int(y0_p4)
        dY_p4 = delta3(y0_p4, yT_p4)

        # --- Loneliness (필수) ---
        y0_lon, yT_lon = first_last_survey(sdf, SURVEY_LONELINESS)
        if np.isnan(y0_lon) or np.isnan(yT_lon):
            continue
        y0_lon = int(y0_lon)
        dY_lon = delta3(y0_lon, yT_lon)

        # weekly C / S
        C = g[check_cols].to_numpy(float)                # (T,6)
        S = g[[c for c in g.columns if c in service_cols]].to_numpy(float)  # (T,16)

        S_list.append(S)
        C_list.append(C)
        y0_phq_list.append(y0_phq)
        y0_p4_list.append(y0_p4)
        y0_lon_list.append(y0_lon)

        dY_phq_list.append(dY_phq)
        dY_p4_list.append(dY_p4)
        dY_lon_list.append(dY_lon)
        keep_ids.append(pid)

    if not S_list:
        raise RuntimeError("No complete sequences found under the filtering.")

    S    = np.stack(S_list, axis=0)              # (B,T,16)
    Craw = np.stack(C_list, axis=0)              # (B,T,6)
    dY_phq = np.array(dY_phq_list, dtype=np.int64)
    dY_p4  = np.array(dY_p4_list,  dtype=np.int64)
    dY_lon = np.array(dY_lon_list, dtype=np.int64)

    # 체크 전역 z-score (모델 입력용)
    flat = Craw.reshape(-1, Craw.shape[-1])
    mu = flat.mean(axis=0)
    sd = flat.std(axis=0)
    sd = np.where(sd < 1e-6, 1.0, sd)
    Cz = (Craw - mu) / sd                          # (B,T,6)

    # 레벨 수(이진/순서형) 확정 (stat 용)
    base_names = [re.sub(r"_value$", "", c) for c in check_cols]
    levels = np.array([LEVELS_PER_CHECK.get(b, 2) for b in base_names], dtype=np.int64)
    is_bin = (levels == 2)

    # split (PHQ-9 Δy 기반 stratify)
    idx = np.arange(Cz.shape[0])
    idx_tr, idx_val = train_test_split(
        idx, test_size=0.2, random_state=seed, stratify=dY_phq
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    y0_phq = np.array(y0_phq_list, dtype=np.int64)
    y0_p4  = np.array(y0_p4_list,  dtype=np.int64)
    y0_lon = np.array(y0_lon_list, dtype=np.int64)

    batch = Batch(
        S=torch.tensor(S,      dtype=torch.float32, device=device),
        C=torch.tensor(Cz,     dtype=torch.float32, device=device),
        Craw=torch.tensor(Craw, dtype=torch.float32, device=device),
        levels=torch.tensor(levels, dtype=torch.long, device=device),
        is_bin=torch.tensor(is_bin, dtype=torch.bool, device=device),
        y0_phq=torch.tensor(y0_phq, dtype=torch.long, device=device),
        y0_p4=torch.tensor(y0_p4,   dtype=torch.long, device=device),
        y0_lon=torch.tensor(y0_lon, dtype=torch.long, device=device),
        dY_phq=torch.tensor(dY_phq, dtype=torch.long, device=device),
        dY_p4=torch.tensor(dY_p4,   dtype=torch.long, device=device),
        dY_lon=torch.tensor(dY_lon, dtype=torch.long, device=device),
        idx_tr=idx_tr,
        idx_val=idx_val,
        stats={
            "check_mu": mu.tolist(),
            "check_sd": sd.tolist(),
            "service_cols": service_cols,
            "check_cols": check_cols,
            "levels_per_check": levels.tolist(),
            "check_basenames": base_names,
            "patients": keep_ids,
            "T": T,
            "id_col": id_col,
        }
    )

    # analysis_util 호환용: PHQ-9 Δ를 dY 이름으로도 제공
    batch.dY = batch.dY_phq  # ΔPHQ
    batch.y0 = batch.y0_phq  # 초기 PHQ

    with open(os.path.join(OUT, "build_stats_multi.json"), "w") as f:
        json.dump(batch.stats, f, indent=2, ensure_ascii=False)
    return batch


# ---------------- Model (멀티 설문 + Attention + y0-제약) ----------------

class SeqModelPHQ(nn.Module):
    def __init__(
        self,
        d_s: int = 16,
        d_c: int = 6,
        d_pos: int = 16,
        hs: int = 128,
        lambda_c: float = 0.0,
        d_y0_each: int = 8,
        use_y0_in_step: bool = True,
        levels: Optional[List[int]] = None,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.use_y0_in_step = use_y0_in_step
        self.levels = list(int(k) for k in (levels or [2] * d_c))
        self.d_c = d_c
        self.hs = hs

        # time embedding
        self.pos = nn.Embedding(128, d_pos)

        # --- y0 embeddings ---
        self.y0_emb_phq = nn.Embedding(4 + 1, d_y0_each)  # 0..3, 4=unknown
        self.y0_emb_p4  = nn.Embedding(3 + 1, d_y0_each)  # 0..2, 3=unknown
        self.y0_emb_lon = nn.Embedding(2 + 1, d_y0_each)  # 0..1, 2=unknown

        y0_dim = 3 * d_y0_each

        # y0_all -> 초기 hidden
        self.y0_to_h0 = nn.Sequential(
            nn.Linear(y0_dim, hs),
            nn.Tanh(),
        )

        # ---------- NEW: 동적 채널 attention ----------
        # 입력: [C_t (d_c), y0_embed (y0_dim), h_{t-1} (hs)]
        self.ch_attn = nn.Sequential(
            nn.Linear(d_c + y0_dim + hs, 64),
            nn.Tanh(),
            nn.Linear(64, d_c),
        )
        # ------------------------------------------------

        # C-branch GRUCell: (check + pos + opt y0) → h_t
        # C-branch GRUCell: (원본 C + weighted C + pos + opt y0) → h_t
        cin_dim = 2 * d_c + d_pos + (y0_dim if use_y0_in_step else 0)
        self.gru_cell = nn.GRUCell(cin_dim, hs)
        self.drop  = nn.Dropout(dropout)

        # --- additive time attention ---
        attn_dim = hs
        self.attn_h = nn.Linear(hs, attn_dim, bias=False)
        self.attn_y = nn.Linear(y0_dim, attn_dim, bias=False)
        self.attn_v = nn.Linear(attn_dim, 1, bias=False)

        # endpoint encoder + heads
        enc_in = hs + y0_dim
        self.enc = nn.Sequential(
            nn.Linear(enc_in, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
        )
        self.head_phq = nn.Linear(256, 3)
        self.head_p4  = nn.Linear(256, 3)
        self.head_lon = nn.Linear(256, 3)

    # ------------ helper: y0에 따른 Δ 가능/불가능 마스크 ------------

    @staticmethod
    def _delta_mask_from_y0(y0: torch.Tensor, y_max: int) -> torch.Tensor:
        """
        y0: (B,) 0..y_max, or -1(unknown)
        y_max: 최대 레벨 (PHQ=3, P4=2, Lon=1)
        return: (B,3) bool mask, True=가능한 클래스
          Δ index: 0=improved, 1=same, 2=worse
        """
        device = y0.device
        B = y0.size(0)
        mask = torch.ones(B, 3, dtype=torch.bool, device=device)  # 기본: 다 가능

        valid = (y0 >= 0)  # -1(unknown)은 건들지 않음

        # y0 == 0  → improved(0) 불가
        mask[valid & (y0 == 0), 0] = False
        # y0 == y_max → worse(2) 불가
        mask[valid & (y0 == y_max), 2] = False

        return mask

    @staticmethod
    def _masked_ce(logits: torch.Tensor,
                   targets: torch.Tensor,
                   mask: torch.Tensor,
                   weight: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        logits: (B,3)
        targets: (B,)
        mask: (B,3) bool, False=불가능한 클래스
        weight: (3,) or None  # 클래스별 가중치 (예: [w_impr, w_same, w_worse])
        """
        logits_masked = logits.masked_fill(~mask, -1e9)
        log_probs = F.log_softmax(logits_masked, dim=-1)   # (B,3)

        if weight is not None:
            # weight: (3,) -> (1,3)
            w = weight.to(logits.device).view(1, -1)       # (1,3)
            n_cls = log_probs.size(1)
            # one-hot target
            one_hot = F.one_hot(
                targets.long().clamp(0, n_cls - 1),
                num_classes=n_cls
            ).float()                                      # (B,3)
            loss_per_sample = -(one_hot * w * log_probs).sum(dim=1)  # (B,)
            loss = loss_per_sample.mean()
        else:
            loss = F.nll_loss(log_probs, targets.long(), reduction="mean")

        return loss


    # -------- y0 임베딩: 세 개 설문 동시에 --------
    def _y0_embed_all(
        self,
        y0_phq: torch.Tensor,  # (B,) in {0,1,2,3} or -1
        y0_p4:  torch.Tensor,  # (B,) in {0,1,2} or -1
        y0_lon: torch.Tensor,  # (B,) in {0,1}   or -1
    ) -> torch.Tensor:
        # PHQ: valid 0..3, unknown=4
        phq = y0_phq.clone()
        phq = torch.where(phq < 0, torch.full_like(phq, 4), phq)
        phq = torch.clamp(phq, 0, 4)
        # P4: valid 0..2, unknown=3
        p4  = y0_p4.clone()
        p4  = torch.where(p4 < 0, torch.full_like(p4, 3), p4)
        p4  = torch.clamp(p4, 0, 3)
        # Loneliness: valid 0..1, unknown=2
        lon = y0_lon.clone()
        lon = torch.where(lon < 0, torch.full_like(lon, 2), lon)
        lon = torch.clamp(lon, 0, 2)

        e_phq = self.y0_emb_phq(phq.long())
        e_p4  = self.y0_emb_p4(p4.long())
        e_lon = self.y0_emb_lon(lon.long())
        return torch.cat([e_phq, e_p4, e_lon], dim=-1)  # (B, y0_dim)

    # -------- 공통 forward (time / channel attention까지) --------
    def _forward_all(
        self,
        C_in: torch.Tensor,      # (B,T,6)
        y0_phq: torch.Tensor,    # (B,)
        y0_p4: torch.Tensor,     # (B,)
        y0_lon: torch.Tensor,    # (B,)
    ):
        """
        반환:
          logit_phq: (B,3)
          logit_p4 : (B,3)
          logit_lon: (B,3)
          attn     : (B,T)         - time attention
          ch_alpha : (B,T,d_c)     - channel attention per step
        """
        B, T, d_c = C_in.size()
        assert d_c == self.d_c

        y0e = self._y0_embed_all(y0_phq, y0_p4, y0_lon)  # (B, y0_dim)

        # 초기 hidden h_0
        h_t = self.y0_to_h0(y0e)                         # (B, hs)

        # 시간 인덱스
        t_idx = torch.arange(T, device=C_in.device)

        h_list = []
        alpha_list = []

        for t in range(T):
            c_t = C_in[:, t, :]                          # (B, d_c)
            pos_t = self.pos(t_idx[t]).unsqueeze(0).expand(B, -1)  # (B, d_pos)

            # ----- 동적 채널 attention: [C_t, y0e, h_{t-1}] -----
            z_t = torch.cat([c_t, y0e, h_t], dim=-1)     # (B, d_c + y0_dim + hs)
            ch_logits_t = self.ch_attn(z_t)              # (B, d_c)
            ch_alpha_t  = F.softmax(ch_logits_t, dim=-1) # (B, d_c)

            # (선택) 너무 한 채널로 몰리는 것 방지하고 싶으면:
            # eps = 0.05
            # ch_alpha_t = (1 - eps) * ch_alpha_t + eps * (1.0 / d_c)

            c_t_weighted = c_t * ch_alpha_t              # (B, d_c)
            # ---------------------------------------------------

            # GRUCell 입력
            x_parts = [c_t, c_t_weighted, pos_t]
            if self.use_y0_in_step:
                x_parts.append(y0e)
            x_t = torch.cat(x_parts, dim=-1)             # (B, cin_dim)

            # 한 스텝 업데이트
            h_t = self.gru_cell(x_t, h_t)                # (B, hs)

            h_list.append(h_t.unsqueeze(1))              # (B,1,hs)
            alpha_list.append(ch_alpha_t.unsqueeze(1))   # (B,1,d_c)

        h = torch.cat(h_list, dim=1)                     # (B,T,hs)
        ch_alpha = torch.cat(alpha_list, dim=1)          # (B,T,d_c)
        h = self.drop(h)

        # --------- 시간 attention ---------
        y0_expand = y0e.unsqueeze(1).expand(B, T, y0e.size(-1))
        attn_logits = self.attn_v(
            torch.tanh(self.attn_h(h) + self.attn_y(y0_expand))
        ).squeeze(-1)                                    # (B,T)
        attn = F.softmax(attn_logits, dim=1)             # (B,T)

        context = torch.sum(attn.unsqueeze(-1) * h, dim=1)  # (B,hs)

        enc_in = torch.cat([context, y0e], dim=-1)       # (B,hs + y0_dim)
        h_enc = self.enc(enc_in)                         # (B,256)

        logit_phq = self.head_phq(h_enc)
        logit_p4  = self.head_p4(h_enc)
        logit_lon = self.head_lon(h_enc)

        return logit_phq, logit_p4, logit_lon, attn, ch_alpha

    # 기존 analysis_util 호환용 forward (PHQ만)
    def forward(self, S, C_in, y0_phq):
        """
        옛날 유틸들이 (S, C, y0_phq) 만 주는 경우를 위해
        P4/Lon은 unknown(-1)으로 채워서 사용.
        이때도 y0 제약을 적용해서 불가능한 Δ는 로짓에서 제거한다.
        """
        B = y0_phq.size(0)
        device = y0_phq.device
        y0_p4_dummy  = torch.full_like(y0_phq, -1, device=device)
        y0_lon_dummy = torch.full_like(y0_phq, -1, device=device)

        logit_phq, _, _, attn, _ = self._forward_all(C_in, y0_phq, y0_p4_dummy, y0_lon_dummy)

        m_phq = self._delta_mask_from_y0(y0_phq, y_max=3)
        logit_phq = logit_phq.masked_fill(~m_phq, -1e9)

        return logit_phq, attn

    # 새 멀티태스크 forward
    def forward_multi(self,
                      S: torch.Tensor,
                      C_in: torch.Tensor,
                      y0_phq: torch.Tensor,
                      y0_p4: torch.Tensor,
                      y0_lon: torch.Tensor):
        return self._forward_all(C_in, y0_phq, y0_p4, y0_lon)

    # -------- Loss: ΔPHQ / ΔP4 / ΔLon CE (+ y0제약) --------
    def loss(self,
             logit_phq: torch.Tensor,
             logit_p4: torch.Tensor,
             logit_lon: torch.Tensor,
             dY_phq: torch.Tensor,
             dY_p4: torch.Tensor,
             dY_lon: torch.Tensor,
             y0_phq: torch.Tensor,
             y0_p4: torch.Tensor,
             y0_lon: torch.Tensor,
             use_y0_mask: bool = True,
             # (1) 태스크별 가중치
             w_phq: float = 2.0,
             w_p4: float = 1.0,
             w_lon: float = 1.0,
             # (2) PHQ 클래스 가중치 사용할지 여부
             use_class_weight_phq: bool = True):

        # ----- (2) PHQ 클래스 가중치 계산 (inverse frequency) -----
        class_weight_phq = None
        if use_class_weight_phq:
            with torch.no_grad():
                mask_valid = (dY_phq >= 0)
                cnt = torch.bincount(
                    dY_phq[mask_valid].long(),
                    minlength=3
                ).float()  # (3,)
                cnt = torch.clamp(cnt, min=1.0)

                alpha = 0.5  # 0.5 ~ 0.7 정도 추천
                inv = (cnt.sum() / cnt) ** alpha  # inverse freq 에 지수 적용
                class_weight_phq = inv / inv.mean()  # 평균 1로 정규화

        # ----- CE 계산 -----
        if use_y0_mask:
            m_phq = self._delta_mask_from_y0(y0_phq, y_max=3)
            m_p4 = self._delta_mask_from_y0(y0_p4, y_max=2)
            m_lon = self._delta_mask_from_y0(y0_lon, y_max=1)

            ce_phq = self._masked_ce(
                logit_phq, dY_phq, m_phq,
                weight=class_weight_phq  # 여기만 weight 적용
            )
            ce_p4 = self._masked_ce(logit_p4, dY_p4, m_p4)
            ce_lon = self._masked_ce(logit_lon, dY_lon, m_lon)
        else:
            # y0 마스크 안 쓰는 경우 (현재는 안 씀)
            if class_weight_phq is not None:
                log_probs = F.log_softmax(logit_phq, dim=-1)
                w = class_weight_phq.to(logit_phq.device).view(1, -1)
                one_hot = F.one_hot(dY_phq.long(), num_classes=3).float()
                ce_phq = -(one_hot * w * log_probs).sum(dim=1).mean()
            else:
                ce_phq = F.cross_entropy(logit_phq, dY_phq.long())

            ce_p4 = F.cross_entropy(logit_p4, dY_p4.long())
            ce_lon = F.cross_entropy(logit_lon, dY_lon.long())

        # ----- (1) 태스크별 가중 평균 -----
        num = w_phq * ce_phq + w_p4 * ce_p4 + w_lon * ce_lon
        den = w_phq + w_p4 + w_lon
        ce_mean = num / den

        comp = {
            "ce_mean": float(ce_mean.item()),
            "ce_phq": float(ce_phq.item()),
            "ce_p4": float(ce_p4.item()),
            "ce_lon": float(ce_lon.item()),
        }
        return ce_mean, comp


# ---------------- Train / Eval (단일 태스크 평가 유틸) ----------------

def _eval_task(name: str,
               dY: torch.Tensor,
               logits: torch.Tensor,
               outdir: str,
               prefix: str):
    """
    단일 태스크(PHQ-9, P4, Loneliness)에 대해
      - confusion matrix
      - acc, macro f1
    를 계산하고 그림 저장.
    """
    mask = (dY >= 0)
    if not mask.any():
        print(f"[VAL] {name}: no labeled samples, skip.")
        return None

    y_true = dY[mask].cpu().numpy()
    y_pred = logits[mask].argmax(-1).cpu().numpy()

    labels_order = [0, 1, 2]
    names_order  = ["improved", "same", "worse"]

    cm = confusion_matrix(y_true, y_pred, labels=labels_order)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=names_order)
    disp.plot(values_format="d", cmap="Blues")
    plt.title(f"{name} Δ")
    plt.tight_layout()
    fname = os.path.join(outdir, f"cm_{prefix}.png")
    plt.savefig(fname, dpi=150)
    plt.close()

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average="macro", zero_division=0)
    rec  = recall_score(y_true, y_pred, average="macro", zero_division=0)
    f1   = f1_score(y_true, y_pred, average="macro")
    prec_each = precision_score(y_true, y_pred, average=None, labels=[0,1,2], zero_division=0).tolist()
    rec_each  = recall_score(y_true, y_pred, average=None, labels=[0,1,2], zero_division=0).tolist()

    print(
        f"[VAL] {name} Δ  "
        f"acc={acc:.3f}  prec={prec:.3f}  rec={rec:.3f}  f1={f1:.3f}  "
        f"(n={len(y_true)})"
    )
    print("precision_each:", prec_each)
    print("recall_each:", rec_each)
    
    return dict(
    acc=acc,
    precision=prec,
    recall=rec,
    f1=f1,
    precision_each=prec_each,   # [improved, same, worse]
    recall_each=rec_each,       # [improved, same, worse]
    n=len(y_true),
    cm=cm.tolist(),
)


def train_model(
    batch: Batch,
    outdir: str = "figs_phq_multi",
    epochs: int = 30,
    lr: float = 1e-3,
    lambda_c: float = 0.0,   # 더이상 사용하지 않음 (인터페이스용)
    es_patience: int = 8,
    es_min_delta: float = 1e-4,
    seed: int = 42,
):
    set_seed(seed)
    OUT = ensure_dir(outdir)

    levels = batch.levels.tolist()
    model = SeqModelPHQ(
        d_s=batch.S.size(-1),
        d_c=batch.C.size(-1),
        d_pos=16,
        hs=128,
        lambda_c=lambda_c,
        d_y0_each=16,
        use_y0_in_step=True,
        levels=levels,
        dropout=0.1,
    ).to(batch.C.device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    def run_split(idx, train: bool = False):
        S = batch.S[idx]
        C_in = batch.C[idx]
        y0_phq = batch.y0_phq[idx]
        y0_p4  = batch.y0_p4[idx]
        y0_lon = batch.y0_lon[idx]
        dY_phq = batch.dY_phq[idx]
        dY_p4  = batch.dY_p4[idx]
        dY_lon = batch.dY_lon[idx]

        if train:
            model.train()
            opt.zero_grad()
            logit_phq, logit_p4, logit_lon, attn, ch_alpha = model.forward_multi(S, C_in, y0_phq, y0_p4, y0_lon)
            loss, comp = model.loss(
                logit_phq, logit_p4, logit_lon,
                dY_phq, dY_p4, dY_lon,
                y0_phq, y0_p4, y0_lon,
                use_y0_mask=True,
                w_phq=3,  # 더 세게
                use_class_weight_phq=True,
            )

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            return loss.item(), comp
        else:
            model.eval()
            with torch.no_grad():
                logit_phq, logit_p4, logit_lon, attn, ch_alpha = model.forward_multi(
                    S, C_in, y0_phq, y0_p4, y0_lon
                )
                loss, comp = model.loss(
                    logit_phq, logit_p4, logit_lon,
                    dY_phq, dY_p4, dY_lon,
                    y0_phq, y0_p4, y0_lon,
                    use_y0_mask=True,
                )

                m_phq = model._delta_mask_from_y0(y0_phq, y_max=3)
                logit_phq_eval = logit_phq.masked_fill(~m_phq, -1e9)

                mask_phq = (dY_phq >= 0)
                preds_phq = logit_phq_eval.argmax(-1)[mask_phq].cpu().numpy()
                trues_phq = dY_phq[mask_phq].cpu().numpy()
            return loss.item(), comp, preds_phq, trues_phq

    hist = {"ep": [], "train": [], "val": [], "ce_mean": []}

    best_val = float("inf")
    best_ep  = -1
    patience_left = es_patience

    for ep in range(1, epochs + 1):
        tr_loss, comp_tr = run_split(batch.idx_tr, train=True)
        va_loss, comp_va, preds, trues = run_split(batch.idx_val, train=False)

        hist["ep"].append(ep)
        hist["train"].append(tr_loss)
        hist["val"].append(va_loss)
        hist["ce_mean"].append(comp_va["ce_mean"])

        print(
            f"[EP{ep:02d}] "
            f"train={tr_loss:.4f}  val={va_loss:.4f}  "
            f"(CEmean={comp_va['ce_mean']:.4f})"
        )

        # --- Early Stopping ---
        if va_loss < best_val - es_min_delta:
            best_val = va_loss
            best_ep = ep
            patience_left = es_patience
            torch.save(model.state_dict(), os.path.join(OUT, "best.pt"))
        else:
            patience_left -= 1
            if patience_left <= 0:
                print(f"[EarlyStop] Stopping at ep {ep}. Best ep={best_ep} val={best_val:.6f}")
                break

    # 학습 곡선
    plt.figure(figsize=(7, 3))
    plt.plot(hist["ep"], hist["train"], label="train")
    plt.plot(hist["ep"], hist["val"], label="val")
    plt.title("Multi Δ loss (mean CE)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, "learning_multi.png"), dpi=150)
    plt.close()

    # 최적 가중치 로드 후 평가 (세 설문 모두)
    if os.path.exists(os.path.join(OUT, "best.pt")):
        model.load_state_dict(torch.load(os.path.join(OUT, "best.pt"), map_location=batch.C.device))
    else:
        print("[WARN] best.pt가 없어 현재 가중치로 평가합니다.")

    model.eval()
    with torch.no_grad():
        S_val     = batch.S[batch.idx_val]
        C_val     = batch.C[batch.idx_val]
        y0_phq_v  = batch.y0_phq[batch.idx_val]
        y0_p4_v   = batch.y0_p4[batch.idx_val]
        y0_lon_v  = batch.y0_lon[batch.idx_val]
        dY_phq_v  = batch.dY_phq[batch.idx_val]
        dY_p4_v   = batch.dY_p4[batch.idx_val]
        dY_lon_v  = batch.dY_lon[batch.idx_val]

        logit_phq_v, logit_p4_v, logit_lon_v, attn_val, ch_alpha_val = model.forward_multi(
            S_val, C_val, y0_phq_v, y0_p4_v, y0_lon_v
        )

        # 평가 시에도 y0 제약 적용
        m_phq_v = model._delta_mask_from_y0(y0_phq_v, y_max=3)
        m_p4_v  = model._delta_mask_from_y0(y0_p4_v,  y_max=2)
        m_lon_v = model._delta_mask_from_y0(y0_lon_v, y_max=1)

        logit_phq_eval = logit_phq_v.masked_fill(~m_phq_v, -1e9)
        logit_p4_eval  = logit_p4_v.masked_fill(~m_p4_v,   -1e9)
        logit_lon_eval = logit_lon_v.masked_fill(~m_lon_v, -1e9)

    print("\n========== FINAL EVAL (VAL SPLIT) ==========")
    metrics_phq = _eval_task("PHQ-9",     dY_phq_v, logit_phq_eval, OUT, "PHQ9_multi")
    metrics_p4  = _eval_task("P4",        dY_p4_v,  logit_p4_eval,  OUT, "P4_multi")
    metrics_lon = _eval_task("Loneliness", dY_lon_v, logit_lon_eval, OUT, "Loneliness_multi")
    print("===========================================\n")

    # 간단한 attention 시각화: 평균 attention over val
    attn_mean = attn_val.mean(dim=0).cpu().numpy()  # (T,)
    plt.figure(figsize=(6,3))
    plt.plot(np.arange(attn_mean.shape[0]), attn_mean, marker="o")
    plt.xlabel("week (t)")
    plt.ylabel("mean attention weight")
    plt.title("Mean temporal attention (VAL)")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, "attn_mean_val.png"), dpi=150)
    plt.close()

    # 저장용 config 부착
    model._config = dict(
        d_s=batch.S.size(-1),
        d_c=batch.C.size(-1),
        d_pos=16,
        hs=128,
        lambda_c=lambda_c,
        d_y0_each=16,
        use_y0_in_step=True,
        levels=levels,
    )

    info = {
        "history": hist,
        "best_ep": best_ep,
        "best_val": best_val,
        "metrics_phq": metrics_phq,
        "metrics_p4": metrics_p4,
        "metrics_lon": metrics_lon,
    }
    return model, info


# ====== Save / Load utilities ======

def _default_model_config():
    """SeqModelPHQ 생성에 필요한 하이퍼파라미터를 기록/복원용으로 유지."""
    return dict(
        d_s=16,
        d_c=6,
        d_pos=16,
        hs=128,
        lambda_c=0.0,
        d_y0_each=8,
        use_y0_in_step=True,
        levels=[2] * 6,
    )

@torch.no_grad()
def plot_ch_attn_overall(batch: Batch, model: nn.Module, outdir: str, split: str = "val"):
    """
    split (train/val) cohort 전체에서
    (시간 포함) 채널 attention 평균을 barplot으로 그림.
    """
    OUT = ensure_dir(outdir)
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

    model.eval()
    _, _, _, _, ch_alpha = model.forward_multi(S, C, y0p, y0p4, y0lon)  # (B,T,d_c)
    ch_alpha = ch_alpha.cpu().numpy()
    # B, T 두 축 모두 평균
    mean_alpha = ch_alpha.mean(axis=(0, 1))        # (d_c,)

    check_names = batch.stats.get("check_basenames", batch.stats.get("check_cols"))
    d_c = mean_alpha.shape[0]
    x = np.arange(d_c)

    plt.figure(figsize=(6, 3))
    plt.bar(x, mean_alpha)
    plt.xticks(x, check_names, rotation=45, ha="right")
    plt.ylabel("mean channel attention")
    plt.title(f"Mean channel attention ({split.upper()})")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, f"ch_attn_overall_{split}.png"), dpi=150)
    plt.close()

@torch.no_grad()
def plot_ch_attn_by_y0_phq(batch: Batch, model: nn.Module, outdir: str, split: str = "val"):
    """
    초기 PHQ 그룹(y0_PHQ)에 따라 채널 attention 평균을 비교.
    시간축(T)도 평균해서 채널별 한 점으로 만든다.
    """
    OUT = ensure_dir(outdir)
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

    model.eval()
    _, _, _, _, ch_alpha = model.forward_multi(S, C, y0p, y0p4, y0lon)  # (B,T,d_c)
    ch_alpha = ch_alpha.cpu().numpy()
    y0_np = y0p.cpu().numpy()

    check_names = batch.stats.get("check_basenames", batch.stats.get("check_cols"))
    d_c = ch_alpha.shape[2]
    x = np.arange(d_c)

    plt.figure(figsize=(7, 3))
    for v in sorted(np.unique(y0_np)):
        mask = (y0_np == v)
        if not mask.any():
            continue
        # 해당 그룹의 B/T 평균
        mean_alpha = ch_alpha[mask].mean(axis=(0, 1))   # (d_c,)
        plt.plot(x, mean_alpha, marker="o", linewidth=1.5, label=f"y0_PHQ={int(v)}")

    plt.xticks(x, check_names, rotation=45, ha="right")
    plt.ylabel("mean channel attention")
    plt.title(f"Channel attention by baseline PHQ ({split.upper()})")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, f"ch_attn_by_y0PHQ_{split}.png"), dpi=150)
    plt.close()

@torch.no_grad()
def plot_ch_attn_by_delta_phq(batch: Batch, model: nn.Module, outdir: str, split: str = "val"):
    """
    ΔPHQ(improved/same/worse) 그룹별로 채널 attention 평균 비교.
    시간축(T)도 평균해서 채널별 한 점으로 만든다.
    """
    OUT = ensure_dir(outdir)
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
    dY   = batch.dY_phq[idx]   # 0,1,2

    model.eval()
    _, _, _, _, ch_alpha = model.forward_multi(S, C, y0p, y0p4, y0lon)  # (B,T,d_c)
    ch_alpha = ch_alpha.cpu().numpy()
    dY_np = dY.cpu().numpy()

    check_names = batch.stats.get("check_basenames", batch.stats.get("check_cols"))
    d_c = ch_alpha.shape[2]
    x = np.arange(d_c)

    labels = {0: "improved", 1: "same", 2: "worse"}

    plt.figure(figsize=(7, 3))
    for cls in [0, 1, 2]:
        mask = (dY_np == cls)
        if not mask.any():
            continue
        # 해당 Δ 그룹의 B/T 평균
        mean_alpha = ch_alpha[mask].mean(axis=(0, 1))   # (d_c,)
        plt.plot(x, mean_alpha, marker="o", linewidth=1.5, label=f"ΔPHQ={labels[cls]}")

    plt.xticks(x, check_names, rotation=45, ha="right")
    plt.ylabel("mean channel attention")
    plt.title(f"Channel attention by ΔPHQ ({split.upper()})")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, f"ch_attn_by_dPHQ_{split}.png"), dpi=150)
    plt.close()


def save_model(outdir: str,
               model: nn.Module,
               extra: Optional[dict] = None,
               filename: str = "phq_multi_seq.ckpt"):
    """모델 가중치와 설정을 한 파일에 저장."""
    os.makedirs(outdir, exist_ok=True)
    payload = {
        "state_dict": model.state_dict(),
        "config": getattr(model, "_config", _default_model_config()),
        "extra": extra or {},
        "format": "phq-multi-seq-v2-attn-y0mask"
    }
    torch.save(payload, os.path.join(outdir, filename))
    print(f"[SAVE] model -> {os.path.join(outdir, filename)}")


def load_model(outdir: str,
               device: torch.device,
               filename: str = "phq_multi_seq.ckpt") -> nn.Module:
    """저장된 체크포인트에서 모델을 복원."""
    ckpt_path = os.path.join(outdir, filename)
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"checkpoint not found: {ckpt_path}")

    payload = torch.load(ckpt_path, map_location=device)
    config = payload.get("config", _default_model_config())

    model = SeqModelPHQ(**config).to(device)
    model.load_state_dict(payload["state_dict"], strict=True)
    model._config = config
    print(f"[LOAD] model <- {ckpt_path}")
    return model


# ---------------- Interaction Analysis ----------------

def analyze_baseline_comorbidity(raw2_csv: str,
                                 batch: Batch,
                                 outdir: str,
                                 cut_phq: int = 2,
                                 cut_p4: float | None = None,
                                 cut_lon: float | None = None):
    """
    초기 설문 점수 기반 베이스라인 공병 분석:
      - PHQ-9 high vs P4 high
      - PHQ-9 high vs Loneliness high

    여기서는 PHQ-9 / P4 / Lon 값이 이미
      PHQ: 0~3, P4: 0~2, Lon: 0~1
    스케일이라고 가정.
    HIGH/LOW 기준:
      - PHQ-9: cut_phq (기본 2 → 2 이상이면 high)
      - P4/Loneliness: 주어지지 않으면 이 코호트의 median 사용
    """
    id_col = batch.stats.get("id_col", "menti_seq")
    patients = batch.stats["patients"]

    r2 = load_surveys(raw2_csv, id_col=id_col)

    rows = []
    for pid in patients:
        sdf = r2[r2[id_col] == pid]
        y0_phq, _ = first_last_survey(sdf, SURVEY_PHQ9)
        y0_p4,  _ = first_last_survey(sdf, SURVEY_P4)
        y0_lon, _ = first_last_survey(sdf, SURVEY_LONELINESS)
        if np.isnan(y0_phq) or np.isnan(y0_p4) or np.isnan(y0_lon):
            continue
        rows.append(dict(
            pid=pid,
            phq0=float(y0_phq),
            p40=float(y0_p4),
            lon0=float(y0_lon),
        ))

    if not rows:
        print("[COMORB] no patients with all three baseline scores; skip.")
        return None

    df = pd.DataFrame(rows)

    # 컷오프 설정
    if cut_p4 is None:
        cut_p4 = float(np.nanmedian(df["p40"].values))   # 기록용(이제 high/low에는 안 씀)
    if cut_lon is None:
        cut_lon = float(np.nanmedian(df["lon0"].values))

    df["phq_high"] = df["phq0"] >= cut_phq
    # df["p4_high"] = df["p40"] >= cut_p4   # <- P4는 이제 0/1/2 원값 그대로 사용
    df["lon_high"] = df["lon0"] >= cut_lon

    OUT = ensure_dir(outdir)

    # --- P4는 원값(0/1/2) 그대로 cross-tab ---
    ct_phq_p4 = pd.crosstab(df["phq_high"], df["p40"], normalize="index")

    # 열 순서 고정: 0,1,2가 전부 보이게 (없는 값은 0으로)
    ct_phq_p4 = ct_phq_p4.reindex(columns=[0, 1, 2], fill_value=0.0)

    # 보기 좋은 라벨로 변경
    ct_phq_p4.index = ct_phq_p4.index.map({False: "PHQ_low", True: "PHQ_high"})
    ct_phq_p4.columns = [f"P4={c}" for c in ct_phq_p4.columns]

    # --- Loneliness는 기존대로 high/low ---
    ct_phq_lon = pd.crosstab(df["phq_high"], df["lon_high"], normalize="index")
    ct_phq_lon.index = ct_phq_lon.index.map({False: "PHQ_low", True: "PHQ_high"})
    ct_phq_lon.columns = ct_phq_lon.columns.map({False: "Lon_low", True: "Lon_high"})

    # 저장
    ct_phq_p4.to_csv(os.path.join(OUT, "xtab_baseline_PHQ_P4.csv"))
    ct_phq_lon.to_csv(os.path.join(OUT, "xtab_baseline_PHQ_Loneliness.csv"))

    print("\n[COMORB] Baseline PHQ-high vs P4 (0/1/2) (row-normalized):")
    print(ct_phq_p4)
    print("\n[COMORB] Baseline PHQ-high vs Loneliness-high (row-normalized):")
    print(ct_phq_lon)

    return {
        "cutoffs": dict(cut_phq=cut_phq, cut_p4=cut_p4, cut_lon=cut_lon),
        "xtab_phq_p4": ct_phq_p4.to_dict(),
        "xtab_phq_lon": ct_phq_lon.to_dict(),
    }


def analyze_delta_interactions(batch: Batch,
                               model: nn.Module,
                               outdir: str):
    """
    Δ 수준에서 상호작용 분석:
      - 실제 ΔPHQ vs ΔP4 / ΔLoneliness cross-tab
      - 예측 ΔPHQ vs ΔP4 / ΔLoneliness cross-tab
      - 각 조합은 row-normalized cross-tab CSV로 저장
      - 클래스 "worse"/"improved" 확률 간 상관계수 계산 후 JSON 저장
    """
    OUT = ensure_dir(outdir)

    with torch.no_grad():
        S_val    = batch.S[batch.idx_val]
        C_val    = batch.C[batch.idx_val]
        y0_phq_v = batch.y0_phq[batch.idx_val]
        y0_p4_v  = batch.y0_p4[batch.idx_val]
        y0_lon_v = batch.y0_lon[batch.idx_val]
        dY_phq_v = batch.dY_phq[batch.idx_val]
        dY_p4_v  = batch.dY_p4[batch.idx_val]
        dY_lon_v = batch.dY_lon[batch.idx_val]

        logit_phq_v, logit_p4_v, logit_lon_v, attn_val, ch_alpha_val = model.forward_multi(
            S_val, C_val, y0_phq_v, y0_p4_v, y0_lon_v
        )

        # y0 제약 적용 후 확률/예측 사용
        m_phq_v = model._delta_mask_from_y0(y0_phq_v, y_max=3)
        m_p4_v  = model._delta_mask_from_y0(y0_p4_v,  y_max=2)
        m_lon_v = model._delta_mask_from_y0(y0_lon_v, y_max=1)

        logit_phq_eval = logit_phq_v.masked_fill(~m_phq_v, -1e9)
        logit_p4_eval  = logit_p4_v.masked_fill(~m_p4_v,   -1e9)
        logit_lon_eval = logit_lon_v.masked_fill(~m_lon_v, -1e9)

    patients = np.array(batch.stats["patients"])
    pid_val = patients[batch.idx_val]

    prob_phq = F.softmax(logit_phq_eval, dim=-1).cpu().numpy()
    prob_p4  = F.softmax(logit_p4_eval,  dim=-1).cpu().numpy()
    prob_lon = F.softmax(logit_lon_eval, dim=-1).cpu().numpy()

    df = pd.DataFrame(dict(
        pid=pid_val,
        dPHQ_true=dY_phq_v.cpu().numpy(),
        dP4_true=dY_p4_v.cpu().numpy(),
        dLon_true=dY_lon_v.cpu().numpy(),
        dPHQ_pred=logit_phq_eval.argmax(-1).cpu().numpy(),
        dP4_pred=logit_p4_eval.argmax(-1).cpu().numpy(),
        dLon_pred=logit_lon_eval.argmax(-1).cpu().numpy(),
        pPHQ_imp=prob_phq[:, 0],
        pPHQ_same=prob_phq[:, 1],
        pPHQ_worse=prob_phq[:, 2],
        pP4_imp=prob_p4[:, 0],
        pP4_same=prob_p4[:, 1],
        pP4_worse=prob_p4[:, 2],
        pLon_imp=prob_lon[:, 0],
        pLon_same=prob_lon[:, 1],
        pLon_worse=prob_lon[:, 2],
    ))

    # ---- TRUE Δ cross-tabs ----
    def _xtab_true(col_y, col_x, fname):
        m = (df[col_y] >= 0) & (df[col_x] >= 0)
        if not m.any():
            print(f"[Δ-TRUE] no data for {col_y} vs {col_x}")
            return None
        ct = pd.crosstab(df.loc[m, col_y], df.loc[m, col_x], normalize="index")
        ct.to_csv(os.path.join(OUT, fname))
        print(f"[Δ-TRUE] {col_y} vs {col_x} (row-normalized):")
        print(ct)
        return ct.to_dict()

    xtab_true_phq_p4  = _xtab_true("dPHQ_true", "dP4_true", "xtab_true_PHQ_P4.csv")
    xtab_true_phq_lon = _xtab_true("dPHQ_true", "dLon_true", "xtab_true_PHQ_Loneliness.csv")

    # ---- PRED Δ cross-tabs ----
    def _xtab_pred(col_y, col_x, fname):
        m = (df[col_y] >= 0) & (df[col_x] >= 0)
        if not m.any():
            print(f"[Δ-PRED] no data for {col_y} vs {col_x}")
            return None
        ct = pd.crosstab(df.loc[m, col_y], df.loc[m, col_x], normalize="index")
        ct.to_csv(os.path.join(OUT, fname))
        print(f"[Δ-PRED] {col_y} vs {col_x} (row-normalized):")
        print(ct)
        return ct.to_dict()

    xtab_pred_phq_p4  = _xtab_pred("dPHQ_pred", "dP4_pred", "xtab_pred_PHQ_P4.csv")
    xtab_pred_phq_lon = _xtab_pred("dPHQ_pred", "dLon_pred", "xtab_pred_PHQ_Loneliness.csv")

    # ---- 조건부 분포 (예: PHQ worse일 때 P4/Lon 분포) ----
    cond: Dict[str, Any] = {}
    for name_y, col_y, col_x in [
        ("P4_given_PHQ_true", "dPHQ_true", "dP4_true"),
        ("Lon_given_PHQ_true", "dPHQ_true", "dLon_true"),
        ("P4_given_PHQ_pred", "dPHQ_pred", "dP4_pred"),
        ("Lon_given_PHQ_pred", "dPHQ_pred", "dLon_pred"),
    ]:
        m = (df[col_y] >= 0) & (df[col_x] >= 0)
        if not m.any():
            continue
        sub = df.loc[m, [col_y, col_x]]
        out = {}
        for k in [0, 1, 2]:  # PHQ improved/same/worse
            subk = sub[sub[col_y] == k]
            if len(subk) == 0:
                continue
            vc = subk[col_x].value_counts(normalize=True).sort_index()
            out[int(k)] = vc.to_dict()
        cond[name_y] = out

    # ---- probability correlations ----
    corr: Dict[str, float] = {}
    def _corr(a, b):
        return float(np.corrcoef(a, b)[0, 1])

    corr["PHQw_P4w"]  = _corr(df["pPHQ_worse"], df["pP4_worse"])
    corr["PHQw_Lonw"] = _corr(df["pPHQ_worse"], df["pLon_worse"])
    corr["PHQi_P4i"]  = _corr(df["pPHQ_imp"],   df["pP4_imp"])
    corr["PHQi_Loni"] = _corr(df["pPHQ_imp"],   df["pLon_imp"])

    with open(os.path.join(OUT, "interactions_delta.json"), "w") as f:
        json.dump(
            dict(
                xtab_true_phq_p4=xtab_true_phq_p4,
                xtab_true_phq_lon=xtab_true_phq_lon,
                xtab_pred_phq_p4=xtab_pred_phq_p4,
                xtab_pred_phq_lon=xtab_pred_phq_lon,
                conditional=cond,
                prob_corr=corr,
            ),
            f,
            indent=2,
            ensure_ascii=False,
        )

    print("\n[Δ-PROB] correlation of probs:")
    print("  corr(PHQ worse, P4 worse) =", corr["PHQw_P4w"])
    print("  corr(PHQ worse, Lon worse)=", corr["PHQw_Lonw"])
    print("  corr(PHQ impr,  P4 impr)  =", corr["PHQi_P4i"])
    print("  corr(PHQ impr,  Lon impr) =", corr["PHQi_Loni"])
    print()


def run_full_interaction_analysis(raw2_csv: str,
                                  batch: Batch,
                                  model: nn.Module,
                                  outdir: str):
    """
    상호작용 분석 풀 패키지:
      1) baseline comorbidity (PHQ-9 high vs P4/Lon high)
      2) Δ 수준 true/pred cross-tabs, 조건부 분포, 확률 상관
    """
    OUT = ensure_dir(outdir)
    print("\n===== [1] Baseline comorbidity =====")
    baseline_info = analyze_baseline_comorbidity(raw2_csv, batch, OUT)
    print("Baseline info:", baseline_info)

    print("\n===== [2] Δ-level interactions & prob correlations =====")
    analyze_delta_interactions(batch, model, OUT)
    print("Interaction analysis outputs saved under:", OUT)


@torch.no_grad()
def plot_attn_by_delta_phq(batch: Batch, model: nn.Module, outdir: str, split: str = "val"):
    """
    ΔPHQ 그룹별(mean over patients) 시간축 attention을 그림.
      - split: "val" 또는 "train"
      - 결과: attn_by_dPHQ.png 로 저장
    """
    OUT = ensure_dir(outdir)

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
    dY   = batch.dY_phq[idx]        # 0=improved,1=same,2=worse

    model.eval()
    logit_phq, logit_p4, logit_lon, attn, ch_alpha = model.forward_multi(S, C, y0p, y0p4, y0lon)
    attn = attn.cpu().numpy()
    dY_np = dY.cpu().numpy()
    T = attn.shape[1]
    weeks = np.arange(T)

    plt.figure(figsize=(7, 3))
    labels = {0: "improved", 1: "same", 2: "worse"}

    for cls in [0, 1, 2]:
        mask = (dY_np == cls)
        if not mask.any():
            continue
        mean_attn = attn[mask].mean(axis=0)
        plt.plot(weeks, mean_attn, marker="o", label=f"ΔPHQ={labels[cls]}", linewidth=1.5)

    plt.xlabel("week (t)")
    plt.ylabel("mean attention weight")
    plt.title(f"Temporal attention by ΔPHQ ({split.upper()})")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, f"attn_by_dPHQ_{split}.png"), dpi=150)
    plt.close()


@torch.no_grad()
def plot_attn_by_y0_phq(batch: Batch, model: nn.Module, outdir: str, split: str = "val"):
    """
    초기 PHQ(y0_PHQ) 그룹별(mean over patients) 시간축 attention을 그림.
      - y0_phq는 {0,1,2,3} 카테고리라고 가정.
      - 결과: attn_by_y0PHQ.png 로 저장
    """
    OUT = ensure_dir(outdir)

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

    model.eval()
    logit_phq, logit_p4, logit_lon, attn, ch_alpha = model.forward_multi(S, C, y0p, y0p4, y0lon)
    attn = attn.cpu().numpy()
    y0_np = y0p.cpu().numpy()
    T = attn.shape[1]
    weeks = np.arange(T)

    plt.figure(figsize=(7, 3))
    uniq_y0 = sorted(np.unique(y0_np))
    for v in uniq_y0:
        mask = (y0_np == v)
        if not mask.any():
            continue
        mean_attn = attn[mask].mean(axis=0)
        plt.plot(weeks, mean_attn, marker="o", label=f"y0_PHQ={int(v)}", linewidth=1.5)

    plt.xlabel("week (t)")
    plt.ylabel("mean attention weight")
    plt.title(f"Temporal attention by baseline PHQ ({split.upper()})")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, f"attn_by_y0PHQ_{split}.png"), dpi=150)
    plt.close()

@torch.no_grad()
def analyze_services_by_attention(batch: Batch,
                                  model: nn.Module,
                                  outdir: str,
                                  split: str = "val",
                                  topk: int = 5):
    """
    temporal attention + service를 엮어서,
    - 각 환자별로 'attention이 높은 주(t)'에서의 서비스 사용 평균 벡터를 만들고
    - ΔPHQ 그룹별/개선 vs 비개선 그룹별로 서비스 패턴을 비교하는 분석.

    결과:
      - service_top{topk}_per_patient_{split}.csv
      - service_top{topk}_by_dPHQ_{split}.csv
      - service_top{topk}_by_improved_vs_non_{split}.csv
      - service_diff_improved_vs_non_top{topk}_{split}.png
    """
    OUT = ensure_dir(outdir)

    # --- split 선택 ---
    if split == "val":
        idx = batch.idx_val
    elif split == "train":
        idx = batch.idx_tr
    else:
        raise ValueError("split must be 'train' or 'val'")

    # subset
    S   = batch.S[idx]        # (B', T, d_s)  - 서비스
    C   = batch.C[idx]        # (B', T, d_c)  - 체크 (모델 입력)
    y0p = batch.y0_phq[idx]
    y0p4 = batch.y0_p4[idx]
    y0lon = batch.y0_lon[idx]
    dY   = batch.dY_phq[idx]  # (B',)  0/1/2

    model.eval()
    # attn: (B', T), ch_alpha: (B', T, d_c)
    logit_phq, logit_p4, logit_lon, attn, ch_alpha = model.forward_multi(
        S, C, y0p, y0p4, y0lon
    )

    attn_np = attn.cpu().numpy()      # (B', T)
    S_np    = S.cpu().numpy()         # (B', T, d_s)
    dY_np   = dY.cpu().numpy()        # (B',)
    patients = np.array(batch.stats["patients"])[idx]  # (B',)

    Bp, T, d_s = S_np.shape
    topk = min(topk, T)

    # 각 환자별로 "attention 상위 topk 주차"에서의 서비스 평균 벡터
    S_top_mean = np.zeros((Bp, d_s), dtype=np.float32)

    for i in range(Bp):
        alpha_i = attn_np[i]                   # (T,)
        # 상위 topk 주 인덱스
        top_idx = np.argsort(alpha_i)[-topk:]  # 오름차순 -> 뒤에서 topk
        # 보기 좋게 주차 순으로 정렬 (선택 사항)
        top_idx = np.sort(top_idx)
        # 해당 주차들의 서비스 벡터 평균
        S_top_mean[i] = S_np[i, top_idx].mean(axis=0)

    service_cols = batch.stats["service_cols"]
    assert len(service_cols) == d_s, "service_cols length mismatch"

    # 1) 개별 환자 레벨 CSV
    rows = []
    for i in range(Bp):
        row = {
            "pid": patients[i],
            "dPHQ_true": int(dY_np[i]),  # 0(improved)/1(same)/2(worse)
        }
        for j, sname in enumerate(service_cols):
            row[sname] = float(S_top_mean[i, j])
        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(
        os.path.join(OUT, f"service_top{topk}_per_patient_{split}.csv"),
        index=False,
    )

    # 2) ΔPHQ(0/1/2) 그룹별 평균 서비스 사용 (attention topk 구간 기준)
    df_by_delta = df.groupby("dPHQ_true").mean(numeric_only=True)
    df_by_delta.to_csv(
        os.path.join(OUT, f"service_top{topk}_by_dPHQ_{split}.csv")
    )

    # 3) 개선(improved=0) vs 비개선(1/2) 이진 그룹
    df["delta_bin"] = np.where(df["dPHQ_true"] == 0, "improved", "non_improved")
    df_bin = df.groupby("delta_bin").mean(numeric_only=True)[service_cols]
    df_bin.to_csv(
        os.path.join(OUT, f"service_top{topk}_by_improved_vs_non_{split}.csv")
    )

    # 개선 vs 비개선 간 서비스 사용 차이(bar plot)
    if "improved" in df_bin.index and "non_improved" in df_bin.index:
        diff = (df_bin.loc["improved"] - df_bin.loc["non_improved"]).values
        x = np.arange(d_s)

        plt.figure(figsize=(8, 3))
        plt.bar(x, diff)
        plt.xticks(x, service_cols, rotation=45, ha="right")
        plt.ylabel("mean(S | improved) - mean(S | non-improved)")
        plt.title(f"Service usage at high-attention weeks (top{topk}, {split})")
        plt.tight_layout()
        plt.savefig(
            os.path.join(OUT, f"service_diff_improved_vs_non_top{topk}_{split}.png"),
            dpi=150,
        )
        plt.close()
    else:
        print("[SERV-ATTN] either 'improved' or 'non_improved' group missing; skip diff plot.")

@torch.no_grad()
def explain_patient_with_services(batch: Batch,
                                  model: nn.Module,
                                  pid,
                                  outdir: str,
                                  split: str = "val",
                                  topk: int = 5):
    """
    특정 환자(pid)에 대해:
      - temporal attention 상위 topk 주차 t들을 찾고
      - 그 주차별 (week index, attention, 서비스, 체크값)을 테이블로 저장.
    """
    OUT = ensure_dir(outdir)

    # 어떤 split에서 찾을지
    if split == "val":
        idx_all = batch.idx_val
    elif split == "train":
        idx_all = batch.idx_tr
    else:
        raise ValueError("split must be 'train' or 'val'")

    patients = np.array(batch.stats["patients"])
    mask = (patients[idx_all] == pid)
    if not mask.any():
        print(f"[EXPLAIN] pid={pid} not found in {split} split.")
        return

    # 해당 pid의 인덱스 (split 내부 index -> batch 전체 index)
    pos = np.where(mask)[0][0]
    bi = idx_all[pos]

    # 1 x T
    S = batch.S[bi:bi+1]    # (1,T,d_s)
    C = batch.C[bi:bi+1]    # (1,T,d_c)
    y0p = batch.y0_phq[bi:bi+1]
    y0p4 = batch.y0_p4[bi:bi+1]
    y0lon = batch.y0_lon[bi:bi+1]
    dY = batch.dY_phq[bi].item()

    model.eval()
    logit_phq, logit_p4, logit_lon, attn, ch_alpha = model.forward_multi(
        S, C, y0p, y0p4, y0lon
    )

    attn_np = attn.cpu().numpy()[0]  # (T,)
    S_np    = S.cpu().numpy()[0]     # (T,d_s)
    C_np    = C.cpu().numpy()[0]     # (T,d_c)

    T = attn_np.shape[0]
    topk = min(topk, T)
    top_idx = np.argsort(attn_np)[-topk:]
    top_idx = np.sort(top_idx)

    service_cols = batch.stats["service_cols"]
    check_cols   = batch.stats["check_cols"]

    rows = []
    for t in top_idx:
        row = {
            "pid": pid,
            "week_index": int(t),
            "attn": float(attn_np[t]),
            "dPHQ_true": int(dY),
        }
        for j, sname in enumerate(service_cols):
            row[f"S::{sname}"] = float(S_np[t, j])
        for k, cname in enumerate(check_cols):
            row[f"C::{cname}"] = float(C_np[t, k])
        rows.append(row)

    df = pd.DataFrame(rows)
    fname = os.path.join(OUT, f"explain_pid_{pid}_top{topk}_{split}.csv")
    df.to_csv(fname, index=False)
    print(f"[EXPLAIN] saved explanation for pid={pid} -> {fname}")

@torch.no_grad()
def build_service_features_topk(batch: Batch,
                                model: nn.Module,
                                outdir: str,
                                split: str = "val",
                                topk: int = 5) -> pd.DataFrame:
    """
    - split ("train" / "val") 코호트에서
    - 각 환자별로 time-attention 상위 topk 주를 뽑고
    - 그 주의 service 사용량을 평균내어 feature로 만든다.
    반환: patient-level DataFrame
      [pid, improved, dPHQ, y0_PHQ, service_total, service1..16]
    """
    OUT = ensure_dir(outdir)

    if split == "val":
        idx = batch.idx_val
    elif split == "train":
        idx = batch.idx_tr
    else:
        raise ValueError("split must be 'train' or 'val'")

    device = batch.C.device

    S   = batch.S[idx]          # (B,T,16)
    C   = batch.C[idx]          # (B,T,6)
    y0p = batch.y0_phq[idx]     # (B,)
    y0p4 = batch.y0_p4[idx]
    y0lon = batch.y0_lon[idx]
    dY   = batch.dY_phq[idx]    # (B,)

    patients = np.array(batch.stats["patients"])[idx]

    model.eval()
    logit_phq, logit_p4, logit_lon, attn, ch_alpha = model.forward_multi(
        S, C, y0p, y0p4, y0lon
    )  # attn: (B,T)

    attn_np = attn.cpu().numpy()
    S_np    = S.cpu().numpy()
    dY_np   = dY.cpu().numpy()
    y0_np   = y0p.cpu().numpy()

    service_cols = batch.stats["service_cols"]  # 길이 16, 순서 = S 채널 순서

    rows = []
    B, T, d_s = S_np.shape
    for i in range(B):
        a = attn_np[i]  # (T,)
        # top-k attention 주 index (가장 큰 값 k개)
        top_idx = np.argsort(a)[-topk:]
        top_idx = np.sort(top_idx)  # 시간순 정렬 (옵션)

        S_top = S_np[i, top_idx, :]          # (k, 16)
        S_mean = S_top.mean(axis=0)          # (16,)
        S_total = S_np[i].sum()              # 전체 40주 서비스 사용량 합

        improved = 1 if dY_np[i] == 0 else 0

        row = {
            "pid": patients[i],
            "improved": improved,
            "dPHQ": int(dY_np[i]),
            "y0_PHQ": int(y0_np[i]),
            "service_total": float(S_total),
        }
        for j, sc in enumerate(service_cols):
            row[sc] = float(S_mean[j])
        rows.append(row)

    df = pd.DataFrame(rows)
    csv_path = os.path.join(OUT, f"service_top{topk}_features_{split}.csv")
    df.to_csv(csv_path, index=False)
    print(f"[SERVICE] saved top-{topk} service features -> {csv_path}")
    return df

def run_service_logistic_regression(df: pd.DataFrame,
                                    outdir: str,
                                    prefix: str = "topk_val"):
    """
    입력 df: build_service_features_topk 에서 만든 DataFrame
    개선(improved=1) vs 비개선(0)을 종속변수로,
    service1..16 + y0_PHQ + service_total 을 공변량으로 넣어
    로지스틱 회귀 수행. (statsmodels 사용)
    """
    if not _HAS_SM:
        print("[WARN] statsmodels 가 없어 OR 분석을 건너뜁니다. "
              "pip install statsmodels 후 다시 실행해주세요.")
        return None

    OUT = ensure_dir(outdir)

    service_cols = [c for c in df.columns if c.startswith("service")]
    covar_cols = service_cols + ["y0_PHQ", "service_total"]

    y = df["improved"].astype(int).values
    X = df[covar_cols].values

    # 스케일 차이를 줄이기 위해 간단히 표준화
    X_mean = X.mean(axis=0, keepdims=True)
    X_std  = X.std(axis=0, keepdims=True) + 1e-6
    Xz = (X - X_mean) / X_std

    X_sm = sm.add_constant(Xz)   # intercept
    logit = sm.Logit(y, X_sm)
    res = logit.fit(disp=False)

    params = res.params
    bse    = res.bse
    pvals  = res.pvalues

    OR      = np.exp(params)
    OR_low  = np.exp(params - 1.96 * bse)
    OR_high = np.exp(params + 1.96 * bse)

    idx = ["intercept"] + covar_cols
    or_df = pd.DataFrame({
        "OR": OR,
        "CI_low": OR_low,
        "CI_high": OR_high,
        "p_value": pvals,
    }, index=idx)

    csv_path = os.path.join(OUT, f"service_logit_OR_{prefix}.csv")
    or_df.to_csv(csv_path)
    print(f"[SERVICE] logistic OR table saved -> {csv_path}")

    # 콘솔에 서비스 관련 항목만 간단히 출력
    print("\n[SERVICE] Adjusted odds ratios (services only):")
    print(or_df.loc[service_cols].sort_values("OR", ascending=False))

    return or_df

def build_patient_service_explanations(df: pd.DataFrame,
                                       outdir: str,
                                       prefix: str = "topk_val",
                                       top_m: int = 3):
    """
    df: build_service_features_topk 가 만든 patient-level DF
    - service feature들을 전체 평균/표준편차로 z-score 변환
    - 각 환자별로 z-score가 가장 큰 service top_m 개를 뽑아
      설명용 테이블을 만든다.
    """
    OUT = ensure_dir(outdir)

    service_cols = [c for c in df.columns if c.startswith("service")]

    S = df[service_cols].values
    mu = S.mean(axis=0, keepdims=True)
    sd = S.std(axis=0, keepdims=True) + 1e-6
    Z = (S - mu) / sd    # z-score, (N, 16)

    rows = []
    for i in range(df.shape[0]):
        z_i = Z[i]
        # 가장 큰 z-score 순으로 top_m 서비스 선택
        top_idx = np.argsort(z_i)[-top_m:][::-1]

        row = {
            "pid": df.loc[i, "pid"],
            "improved": int(df.loc[i, "improved"]),
            "dPHQ": int(df.loc[i, "dPHQ"]),
            "y0_PHQ": int(df.loc[i, "y0_PHQ"]),
        }
        for k, j in enumerate(top_idx):
            svc_name = service_cols[j]
            row[f"top{k+1}_service"] = svc_name
            row[f"top{k+1}_z"] = float(z_i[j])
        rows.append(row)

    expl_df = pd.DataFrame(rows)
    csv_path = os.path.join(OUT, f"service_patient_explanations_{prefix}.csv")
    expl_df.to_csv(csv_path, index=False)
    print(f"[SERVICE] patient-level explanations saved -> {csv_path}")
    return expl_df

def run_service_explanation_pipeline(batch: Batch,
                                     model: nn.Module,
                                     outdir: str,
                                     split: str = "val",
                                     topk: int = 5):
    """
    1) attention top-k 기반 service feature 추출
    2) baseline(y0_PHQ) & total usage 통제 로지스틱 회귀 (OR)
    3) 환자별 service 설명 테이블 생성
    """
    df_feat = build_service_features_topk(batch, model, outdir, split=split, topk=topk)
    or_df   = run_service_logistic_regression(df_feat, outdir, prefix=f"top{topk}_{split}")
    expl_df = build_patient_service_explanations(df_feat, outdir, prefix=f"top{topk}_{split}", top_m=3)
    return dict(features=df_feat, or_table=or_df, explanations=expl_df)

@torch.no_grad()
def analyze_service_seq_and_pairs(
    batch: Batch,
    model: nn.Module,
    outdir: str,
    split: str = "val",
    topk_attn: int = 5,
    window: int = 1,
    min_support_pair: int = 5,
):
    """
    high-attention week '근처' 서비스 사용 + 서비스 조합까지 보는 확장 분석.

    1) 각 환자별로 attn이 높은 주(topk_attn)를 고르고,
       그 주변 window(예: +-1주)까지 합쳐서 "관심 구간"으로 정의.
    2) 관심 구간에서 서비스별 사용률(>0 비율)과
       서비스 쌍별 동시 사용률(>0 & >0 비율)을 환자 단위 feature로 만듦.
    3) 개선 여부(ΔPHQ=0 vs 나머지)를 타겟으로
       각 feature별 단변량 로지스틱 회귀 → OR(odds ratio) 계산.
    4) 결과를 csv + 간단 barplot 으로 저장.

    출력:
      - service_seq_features_<split>.csv : 환자 x (서비스 + 서비스쌍) feature
      - service_OR_seq_<split>.csv      : feature별 OR 정렬
      - service_OR_seq_top20_<split>.png: OR 상위 20개 시각화
    """
    OUT = ensure_dir(outdir)

    if split == "val":
        idx = batch.idx_val
    elif split == "train":
        idx = batch.idx_tr
    else:
        raise ValueError("split must be 'train' or 'val'")

    # 기본 텐서
    S   = batch.S[idx]          # (B,T,16)
    C   = batch.C[idx]
    y0p = batch.y0_phq[idx]
    y0p4 = batch.y0_p4[idx]
    y0lon = batch.y0_lon[idx]
    dY   = batch.dY_phq[idx]    # (B,)

    # attention 뽑기
    model.eval()
    _, _, _, attn, _ = model.forward_multi(S, C, y0p, y0p4, y0lon)  # attn:(B,T)

    S_np    = S.cpu().numpy()
    attn_np = attn.cpu().numpy()
    dY_np   = dY.cpu().numpy()

    patients = np.array(batch.stats["patients"])[idx]
    T = S_np.shape[1]
    n_svc = S_np.shape[2]

    service_cols = batch.stats.get(
        "service_cols",
        [f"service{i+1}" for i in range(n_svc)]
    )

    pair_indices = list(combinations(range(n_svc), 2))
    pair_names = [
        f"{service_cols[i]}+{service_cols[j]}"
        for (i, j) in pair_indices
    ]

    rows = []

    for b in range(S_np.shape[0]):
        if dY_np[b] < 0:
            # Δ 라벨 없는 환자면 스킵
            continue

        label = int(dY_np[b])  # 0=improved,1=same,2=worse
        improved = int(label == 0)

        att_b = attn_np[b]     # (T,)
        # topk attention 주 인덱스
        top_idx = np.argsort(att_b)[-topk_attn:]

        # top 주 주변 window 포함하는 boolean mask
        mask_week = np.zeros(T, dtype=bool)
        for t in top_idx:
            lo = max(0, t - window)
            hi = min(T - 1, t + window)
            mask_week[lo:hi + 1] = True

        if not mask_week.any():
            continue

        # 관심 구간의 서비스 사용
        S_win = S_np[b, mask_week, :]          # (W, n_svc)
        used  = (S_win > 0).astype(float)      # 사용 여부로 단순화

        # 서비스별 사용률 (W 중 몇 주에서 썼는지 비율)
        svc_feat = used.mean(axis=0)           # (n_svc,)

        # 서비스쌍 동시 사용률
        pair_feat = []
        for (i, j) in pair_indices:
            joint = (used[:, i] * used[:, j]).mean()
            pair_feat.append(joint)
        pair_feat = np.asarray(pair_feat)

        row = {
            "pid": patients[b],
            "dPHQ": label,
            "improved": improved,
            "n_weeks_window": int(mask_week.sum()),
        }

        for name, val in zip(service_cols, svc_feat):
            row[name] = float(val)

        for name, val in zip(pair_names, pair_feat):
            row[name] = float(val)

        rows.append(row)

    if not rows:
        print("[SERVICE-SEQ] no rows created; check data / masks.")
        return

    df = pd.DataFrame(rows)
    feat_path = os.path.join(OUT, f"service_seq_features_{split}.csv")
    df.to_csv(feat_path, index=False)
    print(f"[SERVICE-SEQ] per-patient features -> {feat_path}")

    # ---------- 단변량 logit으로 OR 계산 ----------
    # ---------- 단변량 logit으로 OR + 사용량 통계 계산 ----------
    y = df["improved"].values.astype(int)
    feat_cols = [c for c in df.columns if c.startswith("service")]
    N = len(df)

    rows_or = []
    for col in feat_cols:
        X = df[[col]].values  # (N, 1)

        # 변동성이 거의 없으면 스킵 (모두 0 또는 모두 1 등)
        if X.std() < 1e-6:
            continue

        # 해당 feature를 1 이상 가진 환자 수
        n_with = int((X[:, 0] > 0).sum())
        n_total = int(N)
        n_without = n_total - n_with
        support = n_with / n_total if n_total > 0 else 0.0

        # co-use feature인 경우 최소 support 체크
        if "+" in col and n_with < min_support_pair:
            continue

        try:
            lr = LogisticRegression(
                fit_intercept=True,
                C=1e6,  # 거의 unregularized
                solver="lbfgs",
                max_iter=1000,
            )
            lr.fit(X, y)
            beta = float(lr.coef_[0, 0])
            OR = float(np.exp(beta))

            rows_or.append(dict(
                feature=col,
                beta=beta,
                OR=OR,
                n_with=n_with,
                n_without=n_without,
                n_total=n_total,
                support=support,
            ))
        except Exception as e:
            print(f"[SERVICE-SEQ] logistic failed for {col}: {e}")
            continue

    if not rows_or:
        print("[SERVICE-SEQ] no logistic results.")
        return

    res_df = pd.DataFrame(rows_or)

    # log_OR 및 composite_score 추가
    res_df["log_OR"] = np.log(res_df["OR"])
    # 많이 쓰이면서(=support 높고) 효과도 큰(=log_OR 큰) 피처를 위로 올리기 위한 스코어
    res_df["composite_score"] = res_df["log_OR"] * res_df["support"]

    # composite_score 기준으로 정렬 (필요하면 OR로 바꿔도 됨)
    res_df = res_df.sort_values("composite_score", ascending=False)

    or_path = os.path.join(OUT, f"service_OR_seq_{split}.csv")
    res_df.to_csv(or_path, index=False)
    print(f"[SERVICE-SEQ] feature-wise OR + usage stats -> {or_path}")

    # ---------- OR 상위 몇 개 barplot ----------
    topN = min(20, len(res_df))
    top = res_df.head(topN)

    x = np.arange(topN)
    plt.figure(figsize=(12, 4))
    plt.bar(x, top["OR"] - 1.0)  # 1보다 얼마나 큰지/작은지 보기 쉽게
    plt.xticks(x, top["feature"], rotation=60, ha="right")
    plt.ylabel("OR - 1 (improved vs non)")
    plt.title(f"Top{topN} service / service-pair OR around high-attn weeks ({split})")
    plt.tight_layout()
    fig_path = os.path.join(OUT, f"service_OR_seq_top{topN}_{split}.png")
    plt.savefig(fig_path, dpi=150)
    plt.close()
    print(f"[SERVICE-SEQ] top OR plot -> {fig_path}")

import pandas as pd
import numpy as np
import os

@torch.no_grad()
def service_basic_stats(batch: Batch, outdir: str, split: str = "val"):
    """
    서비스별 사용량 기본 통계.
      - split: "train" 또는 "val"
      - 가정: S > 0 이면 그 주에 해당 서비스 사용
    결과:
      - service_usage_overall_{split}.csv 저장
    """
    OUT = ensure_dir(outdir)
    if split == "val":
        idx = batch.idx_val
    elif split == "train":
        idx = batch.idx_tr
    else:
        raise ValueError("split must be 'train' or 'val'")

    S = batch.S[idx].detach().cpu().numpy()      # (B,T,16)
    B, T, d_s = S.shape

    # 서비스 이름
    service_cols = batch.stats.get("service_cols", [f"service{i+1}" for i in range(d_s)])

    used = (S > 0).astype(float)                # (B,T,16), 0/1 사용 여부

    # --- 주 단위 요약 (전체 cohort 기준) ---
    # 한 주에 해당 서비스가 사용된 비율 (환자×주 전체 기준)
    prop_used_week = used.mean(axis=(0, 1))     # (16,)

    # --- 환자 단위 요약 ---
    weeks_used_per_patient = used.sum(axis=1)   # (B,16) : 각 환자가 몇 주 사용했는지
    mean_weeks_per_patient = weeks_used_per_patient.mean(axis=0)      # (16,)
    std_weeks_per_patient  = weeks_used_per_patient.std(axis=0)       # (16,)
    # 최소 한 번이라도 사용한 환자 비율
    prop_patients_any_use  = (weeks_used_per_patient > 0).mean(axis=0)  # (16,)

    df = pd.DataFrame({
        "service": service_cols,
        "prop_used_week": prop_used_week,
        "mean_weeks_per_patient": mean_weeks_per_patient,
        "std_weeks_per_patient": std_weeks_per_patient,
        "prop_patients_any_use": prop_patients_any_use,
    })

    csv_path = os.path.join(OUT, f"service_usage_overall_{split}.csv")
    df.to_csv(csv_path, index=False)
    print(f"[SERVICE] overall usage stats saved -> {csv_path}")
    print(df.head())

@torch.no_grad()
def service_stats_by_y0_and_delta_phq(batch: Batch, outdir: str, split: str = "val"):
    """
    PHQ 기준:
      - y0_PHQ 그룹별
      - ΔPHQ(improved/same/worse) 그룹별
    환자 단위 서비스 사용량 통계.
      - 각 환자에 대해: 서비스별 사용 주 수 (#weeks with S>0)
      - 그룹별로 그 평균을 계산해서 CSV로 저장.
    """
    OUT = ensure_dir(outdir)
    if split == "val":
        idx = batch.idx_val
    elif split == "train":
        idx = batch.idx_tr
    else:
        raise ValueError("split must be 'train' or 'val'")

    S   = batch.S[idx].detach().cpu().numpy()       # (B,T,16)
    y0p = batch.y0_phq[idx].detach().cpu().numpy()  # (B,)
    dYp = batch.dY_phq[idx].detach().cpu().numpy()  # (B,)

    B, T, d_s = S.shape
    service_cols = batch.stats.get("service_cols", [f"service{i+1}" for i in range(d_s)])

    used = (S > 0).astype(float)
    weeks_used_per_patient = used.sum(axis=1)       # (B,16)

    # --------- 1) y0_PHQ 그룹별 ---------
    rows_y0 = []
    for y0_val in sorted(np.unique(y0p)):
        mask = (y0p == y0_val)
        if not mask.any():
            continue
        grp = weeks_used_per_patient[mask]          # (n_grp,16)
        mean_weeks = grp.mean(axis=0)
        std_weeks  = grp.std(axis=0)
        n_grp      = grp.shape[0]

        for s_idx, s_name in enumerate(service_cols):
            rows_y0.append(dict(
                y0_phq=int(y0_val),
                service=s_name,
                mean_weeks=mean_weeks[s_idx],
                std_weeks=std_weeks[s_idx],
                n_patients=int(n_grp),
            ))

    df_y0 = pd.DataFrame(rows_y0)
    path_y0 = os.path.join(OUT, f"service_usage_by_y0PHQ_{split}.csv")
    df_y0.to_csv(path_y0, index=False)
    print(f"[SERVICE] by y0_PHQ stats saved -> {path_y0}")

    # --------- 2) ΔPHQ 그룹별 (0=improved,1=same,2=worse) ---------
    rows_d = []
    for d_val in sorted(np.unique(dYp)):
        if d_val < 0:
            continue    # -1 은 label 없음
        mask = (dYp == d_val)
        if not mask.any():
            continue
        grp = weeks_used_per_patient[mask]
        mean_weeks = grp.mean(axis=0)
        std_weeks  = grp.std(axis=0)
        n_grp      = grp.shape[0]

        for s_idx, s_name in enumerate(service_cols):
            rows_d.append(dict(
                dPHQ=int(d_val),   # 0=improved,1=same,2=worse
                service=s_name,
                mean_weeks=mean_weeks[s_idx],
                std_weeks=std_weeks[s_idx],
                n_patients=int(n_grp),
            ))

    df_d = pd.DataFrame(rows_d)
    path_d = os.path.join(OUT, f"service_usage_by_dPHQ_{split}.csv")
    df_d.to_csv(path_d, index=False)
    print(f"[SERVICE] by ΔPHQ stats saved -> {path_d}")

import os
import numpy as np
import pandas as pd

def _parse_services_from_feature(feature: str):
    """
    예시 feature 문자열:
      - 'service9_highAttn'
      - 'service1+service9_highAttn'
      - 'service2+service8+service9_highAttn' (3-way도 대응 가능)

    반환:
      - ['service9']
      - ['service1', 'service9']
      - ['service2', 'service8', 'service9']
    """
    # 뒤에 붙은 '_highAttn' 같은 suffix 제거
    base = feature
    base = base.replace("_highAttn", "")
    base = base.replace("_highAttn0", "")
    base = base.replace("_highAttn1", "")

    # '+' 기준으로 split
    parts = base.split("+")
    # 'service1', 'service9', ... 만 남도록 필터링 (혹시 다른 토큰 섞여 있을 경우 대비)
    services = [p for p in parts if p.startswith("service")]
    return services


def build_pair_seq_or_table(
    or_seq_csv: str = "figs_phq_multi_attn/service_OR_seq_val.csv",
    out_csv: str = "figs_phq_multi_attn/service_pair_OR_summary_val.csv",
):
    """
    service_OR_seq_val.csv → 시퀀스/조합 OR 테이블 요약.

    기대하는 입력 컬럼(유연하게 처리):
      - feature : 문자열 (예: 'service1+service9_highAttn')
      - OR      : odds ratio
      - (optional) n_with, n_total 또는 count_with, count_without
        -> support = n_with / n_total (해당 feature가 관찰된 샘플 비율)

    출력 CSV 컬럼:
      - feature            : 원본 feature 이름
      - kind               : 'single' / 'pair' / 'combo>=3'
      - n_services         : feature 안에 들어있는 서비스 개수
      - services           : 'service1+service9' 같은 형태로 정규화된 서비스 조합
      - OR                 : odds ratio
      - log_OR             : np.log(OR)
      - n_with             : feature가 나타난 샘플 수 (있는 경우)
      - n_total            : OR 계산에 쓰인 전체 샘플 수 (있는 경우)
      - support            : n_with / n_total (있는 경우)
      - composite_score    : support * log_OR  (support 있으면)
                             혹은 log_OR (support 없으면)
    """
    if not os.path.exists(or_seq_csv):
        raise FileNotFoundError(f"OR seq csv not found: {or_seq_csv}")

    df = pd.read_csv(or_seq_csv)

    if "feature" not in df.columns or "OR" not in df.columns:
        raise ValueError("CSV must contain at least 'feature' and 'OR' columns.")

    # ---- support 계산용 컬럼 후보 찾기 ----
    n_with_col = None
    n_total_col = None

    # 1) n_with / n_total 직접 있는 경우
    if "n_with" in df.columns and "n_total" in df.columns:
        n_with_col = "n_with"
        n_total_col = "n_total"
    # 2) count_with / count_without 으로 있는 경우
    elif {"count_with", "count_without"}.issubset(df.columns):
        df["n_with"] = df["count_with"]
        df["n_total"] = df["count_with"] + df["count_without"]
        n_with_col = "n_with"
        n_total_col = "n_total"

    # ---- feature 파싱: 서비스 목록, 개수, kind ----
    services_list = []
    n_services_list = []
    kind_list = []

    for feat in df["feature"].astype(str).tolist():
        svcs = _parse_services_from_feature(feat)
        services_list.append("+".join(svcs) if svcs else "")
        k = len(svcs)
        n_services_list.append(k)
        if k <= 1:
            kind_list.append("single")
        elif k == 2:
            kind_list.append("pair")
        else:
            kind_list.append("combo>=3")

    df["services"] = services_list
    df["n_services"] = n_services_list
    df["kind"] = kind_list

    # ---- log_OR & support & composite_score ----
    # OR <= 0 인 값은 log가 정의 안되므로 NaN 처리
    df["log_OR"] = np.where(df["OR"] > 0, np.log(df["OR"]), np.nan)

    if n_with_col is not None and n_total_col is not None:
        df["support"] = df[n_with_col] / df[n_total_col].replace(0, np.nan)
        # support가 존재하면 composite = support * log_OR
        df["composite_score"] = df["support"] * df["log_OR"]
    else:
        # support 계산 불가 → composite를 log_OR으로만 두고, 나중에 사람이 해석
        df["support"] = np.nan
        df["composite_score"] = df["log_OR"]

    # ---- 정렬: composite_score 큰 순서 ----
    df_sorted = df.sort_values("composite_score", ascending=False)

    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    df_sorted.to_csv(out_csv, index=False)
    print(f"[SAVE] pair/sequence OR summary -> {out_csv}")

    return df_sorted

@torch.no_grad()
def analyze_service_seq_with_high_attention(
    batch: Batch,
    model: nn.Module,
    outdir: str,
    split: str = "val",
    attn_quantile: float = 0.8,
    min_support_pair: int = 5,
):
    """
    서비스/서비스 조합이
      - 전체 시퀀스(anytime)에서 쓰였는지
      - high-attention week 근처에서 쓰였는지
    를 모두 feature로 만들어서,
    개선(improved vs non) odds에 대한 단변량 OR를 계산.

    출력:
      - service_seq_features_highattn_{split}.csv : per-patient feature 테이블
      - service_OR_seq_highattn_{split}.csv      : feature별 any/high OR 및 support, composite score
    """
    from sklearn.linear_model import LogisticRegression  # 위에서 이미 import 했다면 생략 가능

    OUT = ensure_dir(outdir)

    # ----- split 선택 -----
    if split == "val":
        idx = batch.idx_val
    elif split == "train":
        idx = batch.idx_tr
    else:
        raise ValueError("split must be 'train' or 'val'")

    # 데이터 슬라이스
    S   = batch.S[idx]            # (B,T,16)
    C   = batch.C[idx]
    y0p = batch.y0_phq[idx]
    y0p4 = batch.y0_p4[idx]
    y0lon = batch.y0_lon[idx]
    dY   = batch.dY_phq[idx]      # 0=improved,1=same,2=worse

    # patient id 매핑
    all_pids = np.array(batch.stats["patients"])
    pid_sel = all_pids[idx]

    # ----- attention 구하기 -----
    model.eval()
    _, _, _, attn, _ = model.forward_multi(S, C, y0p, y0p4, y0lon)  # attn: (B,T)
    attn = attn.cpu().numpy()

    S_np = S.cpu().numpy()        # (B,T,16)
    B, T, d_s = S_np.shape
    service_cols = batch.stats["service_cols"]  # 예: ["service1",..., "service16"]
    assert len(service_cols) == d_s

    rows = []
    for b in range(B):
        pid = pid_sel[b]
        dY_b = int(dY[b].item())
        improved = int(dY_b == 0)  # ΔPHQ=0 → improved

        s_bt = S_np[b]             # (T,16)
        attn_b = attn[b]           # (T,)

        # high-attention window (예: 상위 20% 주)
        thr = np.quantile(attn_b, attn_quantile)
        high_mask = (attn_b >= thr)      # (T,)

        # 서비스 사용 여부 (주 단위)
        used = (s_bt > 0)                # (T,16) bool
        used_high = used & high_mask[:, None]  # (T,16)

        row = {"pid": pid, "improved": improved}

        # ----- 단일 서비스: any / high -----
        for k, svc in enumerate(service_cols):
            any_flag = used[:, k].any()
            high_flag = used_high[:, k].any()
            row[f"{svc}_any"] = float(any_flag)
            row[f"{svc}_high"] = float(high_flag)

        # ----- 서비스 페어: any / high -----
        for i in range(d_s):
            for j in range(i + 1, d_s):
                pair_name = f"{service_cols[i]}+{service_cols[j]}"
                co_any = used[:, i] & used[:, j]           # 어떤 주든 둘 다 사용
                co_high = co_any & high_mask               # high-attn 주에서 둘 다 사용

                row[f"{pair_name}_any"] = float(co_any.any())
                row[f"{pair_name}_high"] = float(co_high.any())

        rows.append(row)

    df = pd.DataFrame(rows)
    feat_path = os.path.join(OUT, f"service_seq_features_highattn_{split}.csv")
    df.to_csv(feat_path, index=False)
    print(f"[SERVICE-SEQ-ATTN] per-patient features -> {feat_path}")

    # ---------- 단변량 logit으로 OR 계산 (any / high 둘 다) ----------
    y = df["improved"].values.astype(int)
    N = len(df)

    # base feature 이름: "_any" 붙은 컬럼에서 suffix 제거해서 수집
    feature_bases = sorted({
        col.rsplit("_", 1)[0]
        for col in df.columns
        if col.startswith("service") and col.endswith("_any")
    })

    results = []
    for base in feature_bases:
        for mode in ["any", "high"]:
            col = f"{base}_{mode}"
            if col not in df.columns:
                continue

            X = df[[col]].values  # (N,1)

            # 변동성 없으면 스킵
            if X.std() < 1e-6:
                continue

            # support (해당 feature가 1인 환자 비율/수)
            mask_on = (X[:, 0] > 0)
            n_with = int(mask_on.sum())
            support = n_with / float(N)

            # 페어(feature 이름에 "+" 포함)는 최소 support 체크
            if "+" in base and n_with < min_support_pair:
                continue

            try:
                lr = LogisticRegression(
                    fit_intercept=True,
                    C=1e6,
                    solver="lbfgs",
                    max_iter=1000,
                )
                lr.fit(X, y)
                beta = float(lr.coef_[0, 0])
                OR = float(np.exp(beta))
            except Exception as e:
                print(f"[SERVICE-SEQ-ATTN] logistic failed for {col}: {e}")
                continue

            results.append(dict(
                feature=base,
                mode=mode,
                beta=beta,
                OR=OR,
                n_with=n_with,
                n_total=N,
                support=support,
            ))

    if not results:
        print("[SERVICE-SEQ-ATTN] no logistic results.")
        return

    res_df = pd.DataFrame(results)

    # any / high를 한 줄로 합치기
    df_any = res_df[res_df["mode"] == "any"].drop(columns=["mode"]).rename(columns={
        "beta": "beta_any",
        "OR": "OR_any",
        "support": "support_any",
        "n_with": "n_with_any",
    })
    df_high = res_df[res_df["mode"] == "high"].drop(columns=["mode"]).rename(columns={
        "beta": "beta_high",
        "OR": "OR_high",
        "support": "support_high",
        "n_with": "n_with_high",
    })

    merged = pd.merge(
        df_any,
        df_high,
        on=["feature", "n_total"],
        how="outer",
        suffixes=("", "_dup"),
    )

    # log_OR 및 composite_score 계산
    # log_OR 및 composite_score 계산
    for pref in ["any", "high"]:
        OR_col = f"OR_{pref}"
        log_col = f"log_OR_{pref}"
        comp_col = f"composite_{pref}"

        if OR_col in merged.columns:
            # 숫자로 캐스팅 + NaN 방어
            vals = pd.to_numeric(merged[OR_col], errors="coerce").fillna(1.0)
            # pandas는 min 대신 lower/upper 사용
            vals = vals.clip(lower=1e-8)
            merged[log_col] = np.log(vals)

            supp = merged.get(f"support_{pref}")
            if supp is not None:
                merged[comp_col] = merged[log_col] * supp.fillna(0.0)
            else:
                merged[comp_col] = np.nan
        else:
            merged[log_col] = np.nan
            merged[comp_col] = np.nan

    # high-attn composite 기준으로 정렬
    if "composite_high" in merged.columns:
        merged = merged.sort_values("composite_high", ascending=False)

    out_path = os.path.join(OUT, f"service_OR_seq_highattn_{split}.csv")
    merged.to_csv(out_path, index=False)
    print(f"[SERVICE-SEQ-ATTN] feature-wise OR (any vs high-attn) -> {out_path}")

@torch.no_grad()
def eval_multi_on_split(batch: Batch, model: nn.Module, split: str = "val"):
    """
    현재 batch/model로 split에서 3개 태스크(PHQ/P4/Lon) 성능 평가.
    return: dict(metrics_phq, metrics_p4, metrics_lon)
    """
    if split == "val":
        idx = batch.idx_val
        prefix = "val"
    elif split == "train":
        idx = batch.idx_tr
        prefix = "train"
    else:
        raise ValueError("split must be 'train' or 'val'")

    OUT = batch.stats.get("_outdir_for_eval", "figs_phq_multi_attn")  # 편의용
    OUT = ensure_dir(OUT)

    S   = batch.S[idx]
    C   = batch.C[idx]
    y0p = batch.y0_phq[idx]
    y0p4 = batch.y0_p4[idx]
    y0lon = batch.y0_lon[idx]
    dYp = batch.dY_phq[idx]
    dYp4 = batch.dY_p4[idx]
    dYlon = batch.dY_lon[idx]

    model.eval()
    logit_phq, logit_p4, logit_lon, attn, ch_alpha = model.forward_multi(S, C, y0p, y0p4, y0lon)

    # y0 제약 적용(평가 일관성)
    m_phq = model._delta_mask_from_y0(y0p, y_max=3)
    m_p4  = model._delta_mask_from_y0(y0p4, y_max=2)
    m_lon = model._delta_mask_from_y0(y0lon, y_max=1)

    logit_phq = logit_phq.masked_fill(~m_phq, -1e9)
    logit_p4  = logit_p4.masked_fill(~m_p4,  -1e9)
    logit_lon = logit_lon.masked_fill(~m_lon, -1e9)

    metrics_phq = _eval_task("PHQ-9", dYp,   logit_phq, OUT, f"PHQ9_{prefix}_base")
    metrics_p4  = _eval_task("P4",    dYp4,  logit_p4,  OUT, f"P4_{prefix}_base")
    metrics_lon = _eval_task("Loneliness", dYlon, logit_lon, OUT, f"Lon_{prefix}_base")

    return dict(metrics_phq=metrics_phq, metrics_p4=metrics_p4, metrics_lon=metrics_lon)


@torch.no_grad()
def ablation_channel_eval(batch: Batch,
                          model: nn.Module,
                          outdir: str,
                          split: str = "val",
                          channel: int | str = "check3",
                          mode: str = "zero",
                          save_cm: bool = False):
    """
    단일 채널 ablation:
      - mode="zero": 해당 채널을 0으로(=z-score 평균값으로 고정)
      - mode="shuffle_time": 환자별로 시간축(T)에서 그 채널 값을 섞음(정보 파괴)
      - mode="shuffle_patient": 환자 축(B)에서 섞음(집단 분포 유지, 개인 매칭 파괴)

    return: dict(base=..., ablated=..., delta=...)
    """
    OUT = ensure_dir(outdir)
    batch.stats["_outdir_for_eval"] = OUT  # eval 함수가 저장할 경로

    # split 선택
    if split == "val":
        idx = batch.idx_val
        prefix = "val"
    elif split == "train":
        idx = batch.idx_tr
        prefix = "train"
    else:
        raise ValueError("split must be 'train' or 'val'")

    # 채널 index 결정
    check_names = batch.stats.get("check_basenames", batch.stats.get("check_cols"))
    if isinstance(channel, str):
        if check_names is None:
            raise ValueError("check_names not found in batch.stats")
        if channel not in check_names:
            raise ValueError(f"channel '{channel}' not in check names: {check_names}")
        ch_idx = check_names.index(channel)
        ch_name = channel
    else:
        ch_idx = int(channel)
        ch_name = check_names[ch_idx] if check_names is not None else f"ch{ch_idx}"

    # ---------- base 평가 ----------
    model.eval()
    base = eval_multi_on_split(batch, model, split=split)

    # ---------- ablated 입력 만들기 ----------
    S   = batch.S[idx]
    C   = batch.C[idx].clone()  # (B',T,6)
    y0p = batch.y0_phq[idx]
    y0p4 = batch.y0_p4[idx]
    y0lon = batch.y0_lon[idx]
    dYp = batch.dY_phq[idx]
    dYp4 = batch.dY_p4[idx]
    dYlon = batch.dY_lon[idx]

    if mode == "zero":
        C[:, :, ch_idx] = 0.0
    elif mode == "shuffle_time":
        # 환자별로 T 축 shuffle
        Bp, T, _ = C.shape
        for i in range(Bp):
            perm = torch.randperm(T, device=C.device)
            C[i, :, ch_idx] = C[i, perm, ch_idx]
    elif mode == "shuffle_patient":
        # B 축 shuffle
        Bp = C.shape[0]
        perm = torch.randperm(Bp, device=C.device)
        C[:, :, ch_idx] = C[perm, :, ch_idx]
    else:
        raise ValueError("mode must be one of: ['zero','shuffle_time','shuffle_patient']")

    # ---------- ablated forward ----------
    logit_phq, logit_p4, logit_lon, attn, ch_alpha = model.forward_multi(S, C, y0p, y0p4, y0lon)

    m_phq = model._delta_mask_from_y0(y0p, y_max=3)
    m_p4  = model._delta_mask_from_y0(y0p4, y_max=2)
    m_lon = model._delta_mask_from_y0(y0lon, y_max=1)

    logit_phq = logit_phq.masked_fill(~m_phq, -1e9)
    logit_p4  = logit_p4.masked_fill(~m_p4,  -1e9)
    logit_lon = logit_lon.masked_fill(~m_lon, -1e9)

    # CM 저장 여부에 따라 prefix 분기
    if save_cm:
        metrics_phq = _eval_task("PHQ-9", dYp,   logit_phq, OUT, f"PHQ9_{prefix}_abl_{ch_name}_{mode}")
        metrics_p4  = _eval_task("P4",    dYp4,  logit_p4,  OUT, f"P4_{prefix}_abl_{ch_name}_{mode}")
        metrics_lon = _eval_task("Loneliness", dYlon, logit_lon, OUT, f"Lon_{prefix}_abl_{ch_name}_{mode}")
    else:
        # _eval_task는 항상 CM 저장하니까, 저장을 안 하고 싶으면 여기서 수치만 계산
        def _metrics_only(dY, logits):
            mask = (dY >= 0)
            if not mask.any():
                return None
            y_true = dY[mask].cpu().numpy()
            y_pred = logits[mask].argmax(-1).cpu().numpy()
            return dict(
                acc=float(accuracy_score(y_true, y_pred)),
                f1=float(f1_score(y_true, y_pred, average="macro")),
                n=int(len(y_true)),
            )

        metrics_phq = _metrics_only(dYp, logit_phq)
        metrics_p4  = _metrics_only(dYp4, logit_p4)
        metrics_lon = _metrics_only(dYlon, logit_lon)

    ablated = dict(metrics_phq=metrics_phq, metrics_p4=metrics_p4, metrics_lon=metrics_lon)

    # ---------- delta(하락폭) ----------
    def _delta(base_m, abl_m):
        if (base_m is None) or (abl_m is None):
            return None
        return dict(
            acc_drop=float(base_m["acc"] - abl_m["acc"]),
            f1_drop=float(base_m["f1"] - abl_m["f1"]),
        )

    delta = dict(
        phq=_delta(base["metrics_phq"], ablated["metrics_phq"]),
        p4=_delta(base["metrics_p4"], ablated["metrics_p4"]),
        lon=_delta(base["metrics_lon"], ablated["metrics_lon"]),
    )

    # 저장
    payload = dict(
        channel=ch_name,
        ch_idx=int(ch_idx),
        mode=mode,
        split=split,
        base=base,
        ablated=ablated,
        delta=delta,
    )
    with open(os.path.join(OUT, f"ablation_{ch_name}_{mode}_{split}.json"), "w") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    print(f"[ABLATION] saved -> ablation_{ch_name}_{mode}_{split}.json")
    print("[ABLATION] delta drop:", delta)
    return payload


# ---------------- Main ----------------

if __name__ == "__main__":
    RAW1 = "data/raw_data1.csv"
    RAW2 = "data/raw_data2.csv"
    OUT  = ensure_dir("figs_phq_multi_attn")
    TRAIN = True
    T = 40

    batch = build_dataset(RAW1, RAW2, T=T, outdir=OUT, seed=42)

    # ---- 서비스 사용량 기본 통계 & y0/ΔPHQ별 통계 ----
    service_basic_stats(batch, outdir=OUT, split="train")
    service_stats_by_y0_and_delta_phq(batch, outdir=OUT, split="train")

    if not TRAIN:
        device = batch.C.device
        model = load_model(OUT, device, filename="phq_multi_seq.ckpt")
    else:
        model, info = train_model(batch, outdir=OUT, epochs=300, lr=1e-3, lambda_c=0.0, seed=42)
        save_model(
            OUT,
            model,
            extra={
                "best_ep": info["best_ep"],
                "best_val": info["best_val"],
                "metrics_phq": info["metrics_phq"],
                "metrics_p4": info["metrics_p4"],   # 오타 수정
                "metrics_lon": info["metrics_lon"],
            },
            filename="phq_multi_seq.ckpt"
        )

    # ---- (NEW) ΔPHQ / y0별 attention 패턴 ----
    plot_attn_by_delta_phq(batch, model, outdir=OUT, split="val")
    plot_attn_by_y0_phq(batch, model, outdir=OUT, split="val")

    plot_ch_attn_overall(batch, model, outdir=OUT, split="val")
    plot_ch_attn_by_y0_phq(batch, model, outdir=OUT, split="val")
    plot_ch_attn_by_delta_phq(batch, model, outdir=OUT, split="val")

    # ---- (NEW) 서비스 패턴 분석 (attention + service) ----
    analyze_services_by_attention(batch, model, outdir=OUT, split="val", topk=5)
    run_service_explanation_pipeline(batch, model, OUT, split="val", topk=5)
    analyze_service_seq_and_pairs(batch, model, outdir=OUT, split="val",
                                  topk_attn=5, window=1, min_support_pair=5)
    # ---- (NEW) 서비스 seq + high-attention 분석 ----
    analyze_service_seq_with_high_attention(batch, model, OUT, split="val", attn_quantile=0.8, min_support_pair=5)



    # ---- (NEW) 서비스 페어/시퀀스 OR 요약 ----
    try:
        from pathlib import Path
        or_seq_csv = str(Path(OUT) / "service_OR_seq_val.csv")
        out_pair_csv = str(Path(OUT) / "service_pair_OR_summary_val.csv")
        df_pairs = build_pair_seq_or_table(or_seq_csv=or_seq_csv,
                                           out_csv=out_pair_csv)
        print(df_pairs.head(20))
    except Exception as e:
        print("[WARN] build_pair_seq_or_table failed:", e)



    # (옵션) 기존 분석 유틸 사용 (PHQ-9 Δ 기준)
    if _HAS_AUTIL:
        try:
            basic_report(model, batch, outdir=OUT)
        except Exception as e:
            print("[WARN] basic_report failed:", e)
        try:
            reliability_diagram(model, batch, outdir=OUT, n_bins=15)
        except Exception as e:
            print("[WARN] reliability_diagram failed:", e)
        try:
            subgroup_by_y0(model, batch, outdir=OUT)
        except Exception as e:
            print("[WARN] subgroup_by_y0 failed:", e)

    # ---- (NEW) 세 설문 사이 상호작용 분석 ----
    run_full_interaction_analysis(RAW2, batch, model, OUT)

