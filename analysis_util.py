# =========================
# PHQ-9: 후속 분석 유틸
# 붙여넣기만 하면 돌아가도록 구성
# =========================

import os, json, math, itertools
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (
    confusion_matrix, ConfusionMatrixDisplay,
    accuracy_score, f1_score, roc_auc_score
)


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
        cut_p4 = float(np.nanmedian(df["p40"].values))
    if cut_lon is None:
        cut_lon = float(np.nanmedian(df["lon0"].values))

    df["phq_high"] = df["phq0"] >= cut_phq
    df["p4_high"]  = df["p40"] >= cut_p4
    df["lon_high"] = df["lon0"] >= cut_lon

    OUT = ensure_dir(outdir)

    ct_phq_p4 = pd.crosstab(df["phq_high"], df["p4_high"], normalize="index")
    ct_phq_lon = pd.crosstab(df["phq_high"], df["lon_high"], normalize="index")

    ct_phq_p4.to_csv(os.path.join(OUT, "xtab_baseline_PHQ_P4.csv"))
    ct_phq_lon.to_csv(os.path.join(OUT, "xtab_baseline_PHQ_Loneliness.csv"))

    print("\n[COMORB] Baseline PHQ-high vs P4-high (row-normalized):")
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



# -----------------------------
# 0) 헬퍼: 예측/확률/정답 뽑기
# -----------------------------
@torch.no_grad()
def phq_predict(model, batch, idx=None):
    model.eval()
    if idx is None:
        idx = batch.idx_val
    S = batch.S[idx]; C = batch.C[idx]; y0 = batch.y0[idx]; dY = batch.dY[idx]
    logit, _ = model(S, C, y0)
    prob = F.softmax(logit, dim=-1)             # (N, 3)
    pred = prob.argmax(dim=-1)                  # (N,)
    return pred.cpu().numpy(), prob.cpu().numpy(), dY.cpu().numpy()

# -----------------------------------------
# 1) 클래스 분포/기본지표/오분류 샘플 리포트
# -----------------------------------------
def basic_report(model, batch, outdir="figs_phq"):
    os.makedirs(outdir, exist_ok=True)
    yhat, prob, y = phq_predict(model, batch)

    # 분포
    labels = ["improved(0)","same(1)","worse(2)"]
    uniq, cnt = np.unique(y, return_counts=True)
    print("[Dist] y val:", dict(zip(uniq.tolist(), cnt.tolist())))

    # 지표
    acc = accuracy_score(y, yhat)
    f1m = f1_score(y, yhat, average="macro")
    print(f"[VAL] acc={acc:.4f}  f1_macro={f1m:.4f}")

    # CM
    cm = confusion_matrix(y, yhat, labels=[0,1,2])
    disp = ConfusionMatrixDisplay(cm, display_labels=["improved","same","worse"])
    disp.plot(values_format="d", cmap="Blues")
    plt.title("PHQ-9 Δ (val)")
    plt.tight_layout(); plt.savefig(os.path.join(outdir,"cm_val.png"), dpi=150); plt.close()

    # 오분류 Top-N (최대 20개)
    wrong = np.where(y != yhat)[0]
    conf = prob[np.arange(len(prob)), yhat]  # 예측 확신도
    top = wrong[np.argsort(conf[wrong])[::-1]][:20]  # 확신 높은 오분류
    with open(os.path.join(outdir, "misclassified.json"), "w") as f:
        json.dump({
            "count_val": len(y),
            "acc": acc, "f1_macro": f1m,
            "wrong_top20_indices": top.tolist(),
            "wrong_top20_pred": yhat[top].tolist(),
            "wrong_top20_true": y[top].tolist(),
            "wrong_top20_conf": conf[top].round(4).tolist()
        }, f, indent=2)

# --------------------------------
# 2) 멀티클래스 캘리브레이션(ECE)
# --------------------------------
def expected_calibration_error(probs, y_true, n_bins=15):
    """
    Multiclass ECE (top-label 기준)
    """
    conf = probs.max(axis=1)
    pred = probs.argmax(axis=1)
    correct = (pred == y_true).astype(float)

    bins = np.linspace(0.0, 1.0, n_bins+1)
    ece = 0.0
    for b0, b1 in zip(bins[:-1], bins[1:]):
        mask = (conf >= b0) & (conf < b1)
        if not np.any(mask):
            continue
        acc_bin = correct[mask].mean()
        conf_bin = conf[mask].mean()
        ece += (mask.mean()) * abs(acc_bin - conf_bin)
    return ece

def reliability_diagram(model, batch, outdir="figs_phq", n_bins=15):
    os.makedirs(outdir, exist_ok=True)
    _, prob, y = phq_predict(model, batch)
    conf = prob.max(axis=1)
    pred = prob.argmax(axis=1)
    correct = (pred == y).astype(float)

    bins = np.linspace(0.0, 1.0, n_bins+1)
    xs, accs, confs, sizes = [], [], [], []
    for b0, b1 in zip(bins[:-1], bins[1:]):
        mask = (conf >= b0) & (conf < b1)
        if not np.any(mask):
            continue
        xs.append((b0+b1)/2)
        accs.append(correct[mask].mean())
        confs.append(conf[mask].mean())
        sizes.append(mask.sum())

    ece = expected_calibration_error(prob, y, n_bins=n_bins)
    plt.figure(figsize=(5,5))
    plt.plot([0,1],[0,1], "--", label="perfect")
    plt.plot(xs, accs, marker="o", label="accuracy per bin")
    plt.bar(xs, [s/len(y) for s in sizes], width=1.0/n_bins, alpha=0.2, label="bin freq")
    plt.xlabel("confidence"); plt.ylabel("accuracy / freq")
    plt.title(f"Reliability (ECE={ece:.3f})")
    plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(outdir,"reliability.png"), dpi=150); plt.close()

# -------------------------------------
# 3) Subgroup 성능: y0 구간/초기상태별
# -------------------------------------
def subgroup_by_y0(model, batch, outdir="figs_phq"):
    os.makedirs(outdir, exist_ok=True)
    _, prob, y = phq_predict(model, batch)
    y0 = batch.y0[batch.idx_val].cpu().numpy()

    # 예: 4구간(0~3)로 clip된 인덱스 사용 (모델에서와 같은 binning)
    bins = [0,1,2,3]
    rows = []
    for b in bins:
        m = (np.clip(y0,0,3) == b)
        if m.sum() < 5:
            continue
        yhat = prob[m].argmax(axis=1)
        acc = accuracy_score(y[m], yhat)
        f1m = f1_score(y[m], yhat, average="macro")
        rows.append((int(b), int(m.sum()), float(acc), float(f1m)))
        print(f"[y0_bin={b}] n={m.sum()} acc={acc:.3f} f1={f1m:.3f}")

    with open(os.path.join(outdir,"subgroup_y0.json"), "w") as f:
        json.dump({"rows":[{"y0_bin":a,"n":b,"acc":c,"f1":d} for a,b,c,d in rows]}, f, indent=2)

# ---------------------------------------------------
# 4) 시계열 Saliency(Gradient)로 서비스/시간 중요도
# ---------------------------------------------------

# analysis_util.py

import os
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

def temporal_service_saliency(
    model,
    batch,
    outdir="figs_phq",
    use_val_only=True,
    mode="coupled",          # "teacher" or "coupled"
    mix_alpha=0.0,           # 0.0이면 C_in = z(chat), 0.5면 C_in = 0.5*z(chat)+0.5*C_in
    target="pred",           # "pred" | "improved" | "same" | "worse"
    take_mean=True,          # 여러 샘플 평균 saliency 저장
    max_samples=None,        # 너무 많으면 잘라서 분석
):
    """
    서비스 saliency를 계산.
    - teacher: 기존 forward(teacher forcing). S가 endpoint에 영향 없어서 grad가 0/None일 수 있음.
    - coupled: 분석 시점에 C_in을 chat에서 유도한 z(chat) (또는 z(chat)와 기존 C_in을 혼합)으로 대체하여
               S → chat → z(chat) → C_in → h_c → logit 경로를 강제로 연결.

    target:
      - "pred": 각 샘플의 예측 클래스 확률 평균을 스코어로 사용
      - "improved"/"same"/"worse": 해당 클래스 확률 평균을 스코어로 사용 (0/1/2)
    """
    os.makedirs(outdir, exist_ok=True)
    model.eval()

    # --- index 선정 ---
    if use_val_only:
        idx = batch.idx_val
    else:
        idx = np.arange(batch.S.size(0))
    if max_samples is not None:
        idx = idx[:max_samples]

    device = batch.S.device
    S_base  = batch.S[idx].detach()       # (N,T,K)
    C_in0   = batch.C[idx].detach()       # (N,T,6)  z-scored
    Craw    = batch.Craw[idx].detach()    # (N,T,6)  raw
    y0      = batch.y0[idx].detach()      # (N,)
    dY      = batch.dY[idx].detach()      # (N,)

    # z-score를 위해 통계치를 텐서로
    mu = torch.tensor(batch.stats["check_mu"], device=device, dtype=Craw.dtype)    # (6,)
    sd = torch.tensor(batch.stats["check_sd"], device=device, dtype=Craw.dtype)    # (6,)
    sd = torch.where(sd < 1e-6, torch.ones_like(sd), sd)

    # --- requires_grad on Services ---
    S = S_base.clone().requires_grad_(True)

    # --- 1st pass: S → chat (teacher forcing와 동일) ---
    # 여기서는 C_in0으로 h_c를 만들지 않고, chat만 뽑는다.
    logit_t, chat = model(S, C_in0, y0)  # chat: (N,T,6)

    if mode == "teacher":
        # teacher forcing 그대로 logit_t 사용 (S가 endpoint에 영향 없을 수 있어 grad가 0/None)
        logit = logit_t
    elif mode == "coupled":
        # chat을 z-score로 변환해 C_in으로 사용 (또는 혼합)
        # chat은 raw scale로 예측되므로, z(chat) = (chat - mu) / sd
        zchat = (chat - mu.view(1, 1, -1)) / sd.view(1, 1, -1)

        # 혼합 옵션: mix_alpha in [0,1], 0이면 zchat만, 1이면 기존 C_in만
        C_in_coupled = (1.0 - mix_alpha) * zchat + mix_alpha * C_in0

        # --- 2nd pass: S, C_in_coupled → logit (S의 영향이 endpoint까지 연결됨) ---
        logit, _ = model(S, C_in_coupled, y0)
    else:
        raise ValueError("mode must be 'teacher' or 'coupled'")

    # --- target 선택 ---
    prob = F.softmax(logit, dim=-1)  # (N,3)
    if target == "pred":
        cls = prob.argmax(dim=1)                 # 각 샘플의 예측 클래스
        score = prob.gather(1, cls.unsqueeze(1)).squeeze(1)  # (N,)
    else:
        name2id = {"improved": 0, "same": 1, "worse": 2}
        tid = name2id[target]
        score = prob[:, tid]  # (N,)

    # 여러 샘플을 하나의 스칼라로 만들어 역전파 (평균)
    score_mean = score.mean()

    # --- backprop to get d(score)/dS ---
    model.zero_grad(set_to_none=True)
    if S.grad is not None:
        S.grad.zero_()
    score_mean.backward()

    if S.grad is None:
        raise RuntimeError(
            "S.grad is None. If you used mode='teacher', try mode='coupled' "
            "so that services influence the endpoint via C_in := z(chat)."
        )

    # --- saliency: |∂score/∂S| ---
    sal = S.grad.detach().abs().cpu().numpy()   # (N,T,K)

    # --- 시각화 & 요약 ---
    if take_mean:
        sal_mean = sal.mean(axis=0)  # (T,K)
        plt.figure(figsize=(10, 4))
        plt.imshow(sal_mean.T, aspect="auto", origin="lower")
        plt.yticks(range(batch.S.shape[-1]), batch.stats["service_cols"])
        plt.xlabel("time (t)")
        plt.ylabel("service")
        plt.title(f"Temporal service saliency (mode={mode}, alpha={mix_alpha}, target={target})")
        plt.colorbar()
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, f"saliency_S_{mode}_a{mix_alpha}_{target}.png"), dpi=150)
        plt.close()

        # 시간/서비스별 합계도 저장
        np.save(os.path.join(outdir, f"saliency_S_{mode}_a{mix_alpha}_{target}.npy"), sal_mean)
    else:
        # 개별 샘플 저장 (필요 시)
        np.save(os.path.join(outdir, f"saliency_S_{mode}_a{mix_alpha}_{target}_per_sample.npy"), sal)

    return sal


# --------------------------------------------------------
# 5) Counterfactual: 특정 서비스 on/off 간 확률 변화
# --------------------------------------------------------
@torch.no_grad()
def counterfactual_toggle(model, batch, k, v0=0.0, v1=1.0, outdir="figs_phq"):
    """
    서비스 k 채널 전체를 v0 -> v1 로 바꿨을 때, 각 클래스 확률 평균 변화
    """
    os.makedirs(outdir, exist_ok=True)
    model.eval()
    idx = batch.idx_val
    S0 = batch.S[idx].clone()
    C  = batch.C[idx]; y0 = batch.y0[idx]

    S_off = S0.clone(); S_off[:,:,k] = v0
    S_on  = S0.clone(); S_on [:,:,k] = v1

    p_off = F.softmax(model(S_off, C, y0)[0], dim=-1).cpu().numpy()
    p_on  = F.softmax(model(S_on , C, y0)[0], dim=-1).cpu().numpy()

    delta = (p_on - p_off).mean(axis=0)  # (3,)
    cls_names = ["improved","same","worse"]
    print("[Counterfactual Δ prob]", {n: float(d) for n,d in zip(cls_names, delta)})

    plt.figure(figsize=(4.6,3))
    plt.bar(range(3), delta)
    plt.xticks(range(3), cls_names, rotation=0)
    plt.ylabel("Δ probability (on - off)")
    plt.title(f"Service[{k}] toggle: {batch.stats['service_cols'][k]}")
    plt.tight_layout(); plt.savefig(os.path.join(outdir, f"cf_toggle_{k}.png"), dpi=150); plt.close()

    return delta

# ----------------------------------------------------------------
# 6) 시간 창(윈도우)별 Permutation: 어느 구간이 더 중요한가?
# ----------------------------------------------------------------
@torch.no_grad()
def window_permutation_importance(model, batch, win=4, outdir="figs_phq"):
    """
    길이 win 의 시간 창 단위로 서비스 전체를 섞어 ΔLoss 측정
    """
    os.makedirs(outdir, exist_ok=True)
    model.eval()
    # 기준 손실
    def _loss(S_over=None):
        idx = batch.idx_val
        S = batch.S if S_over is None else S_over
        logit, chat = model(S[idx], batch.C[idx], batch.y0[idx])
        loss, _ = model.loss(logit, chat, batch.dY[idx], batch.Craw[idx], batch.is_bin)
        return float(loss.item())

    base = _loss()
    N, T, K = batch.S.shape
    deltas = []
    for start in range(0, T, win):
        end = min(T, start+win)
        S_ = batch.S.clone()
        perm = torch.randperm(N, device=S_.device)
        S_[:, start:end, :] = S_[perm, start:end, :]
        val = _loss(S_over=S_)
        deltas.append(val - base)

    plt.figure(figsize=(max(6, 0.35*math.ceil(T/win)+2), 2.8))
    xs = [f"{s}-{min(T,s+win)-1}" for s in range(0,T,win)]
    plt.bar(range(len(deltas)), deltas)
    plt.xticks(range(len(deltas)), xs, rotation=45, ha="right")
    plt.ylabel("ΔLoss"); plt.title(f"Window Permutation (win={win})")
    plt.tight_layout(); plt.savefig(os.path.join(outdir, f"win_perm_w{win}.png"), dpi=150); plt.close()
    return np.array(deltas)

# -----------------------------------------------------------------
# 7) ICE(Individual) 곡선: 특정 개인에 대한 서비스 반응 곡선
# -----------------------------------------------------------------
@torch.no_grad()
def ice_curve_one(model, batch, k, idx_sample=None, steps=11, outdir="figs_phq"):
    """
    한 명(idx_sample)의 service k 값을 0..1 그리드로 바꾸며 PHQ-9 Δ 확률 곡선
    """
    os.makedirs(outdir, exist_ok=True)
    if idx_sample is None:
        idx_sample = batch.idx_val[0]
    model.eval()

    S0 = batch.S[idx_sample:idx_sample+1].clone()
    C0 = batch.C[idx_sample:idx_sample+1]
    y0 = batch.y0[idx_sample:idx_sample+1]

    grid = torch.linspace(0,1,steps=steps, device=S0.device)
    probs = []
    for g in grid:
        Sg = S0.clone(); Sg[:,:,k] = g
        p = F.softmax(model(Sg, C0, y0)[0], dim=-1).squeeze(0).cpu().numpy()
        probs.append(p)
    probs = np.stack(probs, 0)

    plt.figure(figsize=(6.2,3.6))
    for c, name in enumerate(["improved","same","worse"]):
        plt.plot(np.linspace(0,1,steps), probs[:,c], label=name)
    plt.xlabel(f"{batch.stats['service_cols'][k]}"); plt.ylabel("prob.")
    plt.title(f"ICE (idx={int(idx_sample)})")
    plt.legend(); plt.tight_layout(); plt.savefig(os.path.join(outdir, f"ice_k{k}_idx{int(idx_sample)}.png"), dpi=150); plt.close()
    return probs

# -------------------------------------------------------
# 8) 체크 예측 보조손실 분해: BCE vs Huber 채널별
# -------------------------------------------------------
@torch.no_grad()
def decompose_aux_loss(model, batch, outdir="figs_phq"):
    os.makedirs(outdir, exist_ok=True)
    model.eval()
    idx = batch.idx_val
    S = batch.S[idx]; C = batch.C[idx]; y0 = batch.y0[idx]
    _, chat = model(S, C, y0)
    Craw = batch.Craw[idx]
    is_bin = batch.is_bin

    bce_vals, huber_vals, names = [], [], batch.stats["check_cols"]
    for d, isb in enumerate(is_bin):
        if bool(isb.item()):
            bce = F.binary_cross_entropy_with_logits(chat[:, :, d], Craw[:, :, d]).item()
            bce_vals.append(bce); huber_vals.append(0.0)
        else:
            hub = F.smooth_l1_loss(chat[:, :, d], Craw[:, :, d], beta=1.0).item()
            bce_vals.append(0.0); huber_vals.append(hub)

    x = np.arange(len(names))
    plt.figure(figsize=(7,3.2))
    plt.bar(x-0.18, bce_vals, width=0.36, label="BCE(bin)")
    plt.bar(x+0.18, huber_vals, width=0.36, label="Huber(ord)")
    plt.xticks(x, names, rotation=45, ha="right")
    plt.title("Aux loss per check channel"); plt.legend()
    plt.tight_layout(); plt.savefig(os.path.join(outdir,"aux_per_check.png"), dpi=150); plt.close()

# -------------------------------------------------------
# 9) 학습량-성능 곡선 (learning curve)
# -------------------------------------------------------
def learning_curve(model_ctor, batch, fracs=(0.1, 0.25, 0.5, 0.75, 1.0), epochs=80, outdir="figs_phq"):
    """
    model_ctor: lambda -> 새 모델 생성, 예: lambda: SeqModelPHQ(lambda_c=0.2).to(device)
    """
    os.makedirs(outdir, exist_ok=True)
    results = []
    # 고정된 train 집합에서 일부 비율만 사용
    full_tr = batch.idx_tr.copy()
    for fr in fracs:
        n = max(8, int(len(full_tr)*fr))
        sub = np.random.RandomState(0).choice(full_tr, size=n, replace=False)

        model = model_ctor()
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

        for ep in range(1, epochs+1):
            # 한 에폭 미니멀 트레이닝
            S = batch.S[sub]; C = batch.C[sub]; y0 = batch.y0[sub]; dY = batch.dY[sub]
            logit, chat = model(S, C, y0); loss, _ = model.loss(logit, chat, dY, batch.Craw[sub], batch.is_bin)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

        # 검증 성능
        yhat, _, y = phq_predict(model, batch)
        acc = accuracy_score(y, yhat); f1m = f1_score(y, yhat, average="macro")
        results.append((float(fr), float(acc), float(f1m)))
        print(f"[LC] frac={fr:.2f}  acc={acc:.3f}  f1={f1m:.3f}")

    # Plot
    xs, accs, f1s = zip(*results)
    plt.figure(figsize=(6,3))
    plt.plot(xs, accs, marker="o", label="acc")
    plt.plot(xs, f1s, marker="o", label="f1_macro")
    plt.xlabel("train fraction"); plt.title("Learning Curve")
    plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(outdir,"learning_curve.png"), dpi=150); plt.close()
    return results

# -------------------------------------------------------
# 10) One-vs-Rest AUROC (참고용)
# -------------------------------------------------------
def multiclass_ovr_auc(model, batch):
    yhat, prob, y = phq_predict(model, batch)
    y_bin = np.eye(3, dtype=int)[y]          # (N,3)
    aucs = []
    for c in range(3):
        try:
            auc = roc_auc_score(y_bin[:,c], prob[:,c])
        except Exception:
            auc = float("nan")
        aucs.append(float(auc))
    print("[AUROC ovr]", {"improved":aucs[0], "same":aucs[1], "worse":aucs[2]})
    return aucs

