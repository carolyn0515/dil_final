# y0_ablation_runner.py
# -*- coding: utf-8 -*-
"""
y0 ablation runner (새 파일)
- 기존 approach3.py는 "수정하지 않고"
- y0를 (1) 모델 입력으로 쓸지, (2) hard mask로 쓸지 조합(E1~E4)만 바꿔서 실험

전제 (approach3.py 안에 아래 심볼이 있어야 함):
- build_dataset, SeqModelPHQ, _eval_task, ensure_dir, set_seed

노트북 사용:
    import importlib
    import y0_ablation_runner as abr
    importlib.reload(abr)

    batch = abr.build_batch(
        raw1="data/raw_data1.csv",
        raw2="data/raw_data2.csv",
        T=40,
        outdir="figs_phq_multi_attn",
        seed=42
    )

    results = abr.run_all_y0_ablations(
        batch=batch,
        base_outdir="figs_y0_ablations",
        epochs=300,
        lr=1e-3,
        seed=42
    )
"""

import os
import json
import numpy as np
import torch
import torch.nn as nn

# =========================
# 0) 기존 코드 모듈 import (✅ approach3.py)
# =========================
from approach3 import build_dataset, SeqModelPHQ, _eval_task, ensure_dir, set_seed


# =========================
# 1) 배치 로더 래퍼
# =========================
def build_batch(raw1: str, raw2: str, T: int = 40, outdir: str = "figs_phq_multi_attn", seed: int = 42):
    return build_dataset(raw1, raw2, T=T, outdir=outdir, seed=seed)


# =========================
# 2) y0 ablation 정의
# =========================
ABLATIONS = {
    "E1_full_inputON_maskON": dict(use_y0_input=True,  use_y0_mask=True),
    "E2_inputOFF_maskON":     dict(use_y0_input=False, use_y0_mask=True),
    "E3_inputON_maskOFF":     dict(use_y0_input=True,  use_y0_mask=False),
    "E4_inputOFF_maskOFF":    dict(use_y0_input=False, use_y0_mask=False),
}


def _make_y0_dummy_like(y0: torch.Tensor) -> torch.Tensor:
    """y0 입력을 끊기 위한 dummy(-1)."""
    return torch.full_like(y0, -1)


@torch.no_grad()
def _apply_eval_mask_if_needed(model: nn.Module,
                               logits: torch.Tensor,
                               y0_true: torch.Tensor,
                               y_max: int,
                               use_y0_mask: bool) -> torch.Tensor:
    """평가 시 hard constraint 적용 여부."""
    if not use_y0_mask:
        return logits
    m = model._delta_mask_from_y0(y0_true, y_max=y_max)
    return logits.masked_fill(~m, -1e9)


# =========================
# 3) 학습/평가 루프 (토글 가능 버전)
# =========================
def train_eval_one_setting(
    batch,
    setting_name: str,
    use_y0_input: bool,
    use_y0_mask: bool,
    outdir: str,
    epochs: int = 300,
    lr: float = 1e-3,
    es_patience: int = 8,
    es_min_delta: float = 1e-4,
    seed: int = 42,
    w_phq: float = 3.0,
    w_p4: float = 1.0,
    w_lon: float = 1.0,
    use_class_weight_phq: bool = True,
):
    set_seed(seed)
    OUT = ensure_dir(outdir)

    # 모델 생성 (원본과 동일 하이퍼파라미터)
    levels = batch.levels.tolist()
    model = SeqModelPHQ(
        d_s=batch.S.size(-1),
        d_c=batch.C.size(-1),
        d_pos=16,
        hs=128,
        lambda_c=0.0,
        d_y0_each=16,
        use_y0_in_step=True,   # ✅ 모델 구조 고정
        levels=levels,
        dropout=0.1,
    ).to(batch.C.device)

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    def _forward_with_choice(S, C, y0_phq_true, y0_p4_true, y0_lon_true):
        if use_y0_input:
            y0_in_phq, y0_in_p4, y0_in_lon = y0_phq_true, y0_p4_true, y0_lon_true
        else:
            y0_in_phq = _make_y0_dummy_like(y0_phq_true)
            y0_in_p4  = _make_y0_dummy_like(y0_p4_true)
            y0_in_lon = _make_y0_dummy_like(y0_lon_true)

        return model.forward_multi(S, C, y0_in_phq, y0_in_p4, y0_in_lon)

    def run_split(idx, train: bool):
        S = batch.S[idx]
        C = batch.C[idx]

        y0_phq_true = batch.y0_phq[idx]
        y0_p4_true  = batch.y0_p4[idx]
        y0_lon_true = batch.y0_lon[idx]

        dY_phq = batch.dY_phq[idx]
        dY_p4  = batch.dY_p4[idx]
        dY_lon = batch.dY_lon[idx]

        if train:
            model.train()
            opt.zero_grad()

            logit_phq, logit_p4, logit_lon, attn, ch_alpha = _forward_with_choice(
                S, C, y0_phq_true, y0_p4_true, y0_lon_true
            )

            loss, comp = model.loss(
                logit_phq, logit_p4, logit_lon,
                dY_phq, dY_p4, dY_lon,
                y0_phq_true, y0_p4_true, y0_lon_true,
                use_y0_mask=use_y0_mask,
                w_phq=w_phq, w_p4=w_p4, w_lon=w_lon,
                use_class_weight_phq=use_class_weight_phq,
            )

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            return float(loss.item()), comp

        else:
            model.eval()
            with torch.no_grad():
                logit_phq, logit_p4, logit_lon, attn, ch_alpha = _forward_with_choice(
                    S, C, y0_phq_true, y0_p4_true, y0_lon_true
                )

                loss, comp = model.loss(
                    logit_phq, logit_p4, logit_lon,
                    dY_phq, dY_p4, dY_lon,
                    y0_phq_true, y0_p4_true, y0_lon_true,
                    use_y0_mask=use_y0_mask,
                    w_phq=w_phq, w_p4=w_p4, w_lon=w_lon,
                    use_class_weight_phq=use_class_weight_phq,
                )

                logit_phq_eval = _apply_eval_mask_if_needed(model, logit_phq, y0_phq_true, y_max=3, use_y0_mask=use_y0_mask)
                logit_p4_eval  = _apply_eval_mask_if_needed(model, logit_p4,  y0_p4_true,  y_max=2, use_y0_mask=use_y0_mask)
                logit_lon_eval = _apply_eval_mask_if_needed(model, logit_lon, y0_lon_true, y_max=1, use_y0_mask=use_y0_mask)

            return float(loss.item()), comp, (logit_phq_eval, logit_p4_eval, logit_lon_eval)

    # ----------------
    # train + early stop
    # ----------------
    hist = {"ep": [], "train_loss": [], "val_loss": [], "val_ce_mean": []}

    best_val = float("inf")
    best_ep = -1
    patience_left = es_patience
    best_path = os.path.join(OUT, "best.pt")

    for ep in range(1, epochs + 1):
        tr_loss, _ = run_split(batch.idx_tr, train=True)
        va_loss, comp_va, _ = run_split(batch.idx_val, train=False)

        hist["ep"].append(ep)
        hist["train_loss"].append(tr_loss)
        hist["val_loss"].append(va_loss)
        hist["val_ce_mean"].append(float(comp_va.get("ce_mean", np.nan)))

        print(f"[{setting_name}][EP{ep:03d}] train={tr_loss:.4f} val={va_loss:.4f} (CEmean={hist['val_ce_mean'][-1]:.4f})")

        if va_loss < best_val - es_min_delta:
            best_val = va_loss
            best_ep = ep
            patience_left = es_patience
            torch.save(model.state_dict(), best_path)
        else:
            patience_left -= 1
            if patience_left <= 0:
                print(f"[{setting_name}][EarlyStop] ep={ep} best_ep={best_ep} best_val={best_val:.6f}")
                break

    if os.path.exists(best_path):
        model.load_state_dict(torch.load(best_path, map_location=batch.C.device))

    # ----------------
    # final eval (val)
    # ----------------
    model.eval()
    with torch.no_grad():
        idx = batch.idx_val
        S = batch.S[idx]
        C = batch.C[idx]
        y0_phq_true = batch.y0_phq[idx]
        y0_p4_true  = batch.y0_p4[idx]
        y0_lon_true = batch.y0_lon[idx]
        dY_phq = batch.dY_phq[idx]
        dY_p4  = batch.dY_p4[idx]
        dY_lon = batch.dY_lon[idx]

        logit_phq, logit_p4, logit_lon, attn, ch_alpha = _forward_with_choice(
            S, C, y0_phq_true, y0_p4_true, y0_lon_true
        )

        logit_phq_eval = _apply_eval_mask_if_needed(model, logit_phq, y0_phq_true, y_max=3, use_y0_mask=use_y0_mask)
        logit_p4_eval  = _apply_eval_mask_if_needed(model, logit_p4,  y0_p4_true,  y_max=2, use_y0_mask=use_y0_mask)
        logit_lon_eval = _apply_eval_mask_if_needed(model, logit_lon, y0_lon_true, y_max=1, use_y0_mask=use_y0_mask)

    print(f"\n========== [{setting_name}] FINAL EVAL (VAL) ==========")
    metrics_phq = _eval_task("PHQ-9", dY_phq, logit_phq_eval, OUT, f"{setting_name}_PHQ9")
    metrics_p4  = _eval_task("P4", dY_p4,  logit_p4_eval,  OUT, f"{setting_name}_P4")
    metrics_lon = _eval_task("Loneliness", dY_lon, logit_lon_eval, OUT, f"{setting_name}_LON")
    print("======================================================\n")

    summary = {
        "setting": setting_name,
        "use_y0_input": bool(use_y0_input),
        "use_y0_mask": bool(use_y0_mask),
        "best_ep": int(best_ep),
        "best_val": float(best_val),
        "metrics_phq": metrics_phq,
        "metrics_p4": metrics_p4,
        "metrics_lon": metrics_lon,
        "history": hist,
    }
    with open(os.path.join(OUT, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    return model, summary


# =========================
# 4) 4개 세팅 일괄 실행
# =========================
def run_all_y0_ablations(
    batch,
    base_outdir: str = "figs_y0_ablations",
    epochs: int = 300,
    lr: float = 1e-3,
    seed: int = 42,
):
    base_outdir = ensure_dir(base_outdir)
    all_summaries = {}

    for name, cfg in ABLATIONS.items():
        outdir = os.path.join(base_outdir, name)
        _, summary = train_eval_one_setting(
            batch=batch,
            setting_name=name,
            use_y0_input=cfg["use_y0_input"],
            use_y0_mask=cfg["use_y0_mask"],
            outdir=outdir,
            epochs=epochs,
            lr=lr,
            seed=seed,
        )
        all_summaries[name] = summary

    rows = []
    for name, s in all_summaries.items():
        m = s.get("metrics_phq") or {}
        rows.append({
            "setting": name,
            "use_y0_input": s["use_y0_input"],
            "use_y0_mask": s["use_y0_mask"],
            "best_ep": s["best_ep"],
            "best_val": s["best_val"],
            "PHQ_f1": m.get("f1"),
            "PHQ_acc": m.get("acc"),
            "PHQ_rec": m.get("recall"),
            "PHQ_prec": m.get("precision"),
            "PHQ_n": m.get("n"),
        })

    try:
        import pandas as pd
        df = pd.DataFrame(rows)
        df.to_csv(os.path.join(base_outdir, "ablation_compare_PHQ.csv"), index=False)
        print("[SAVE] compare table ->", os.path.join(base_outdir, "ablation_compare_PHQ.csv"))
        print(df.sort_values("PHQ_f1", ascending=False))
    except Exception as e:
        print("[WARN] failed to save compare csv:", e)

    with open(os.path.join(base_outdir, "all_summaries.json"), "w", encoding="utf-8") as f:
        json.dump(all_summaries, f, indent=2, ensure_ascii=False)

    return all_summaries


# =========================
# 5) CLI 실행도 가능
# =========================
if __name__ == "__main__":
    RAW1 = "data/raw_data1.csv"
    RAW2 = "data/raw_data2.csv"
    T = 40
    SEED = 42

    # ✅ outdir는 폴더명이어야 함
    batch = build_batch(RAW1, RAW2, T=T, outdir="figs_phq_multi_attn", seed=SEED)

    run_all_y0_ablations(
        batch=batch,
        base_outdir="figs_y0_ablations",
        epochs=300,
        lr=1e-3,
        seed=SEED,
    )
