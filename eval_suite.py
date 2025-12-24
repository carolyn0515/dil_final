# -*- coding: utf-8 -*-
"""
eval_suite.py

- class distribution table (val)
- richer metrics: balanced acc, per-class F1, macro/weighted F1, Cohen's kappa
- baselines: majority, stratified, logistic regression on aggregated features
- ablations (eval-only):
    (a) check3 zero / shuffle_time / shuffle_patient  (reuses your ablation_channel_eval)
    (b) y0-mask OFF at evaluation (inflate risk, but shows mask effect)

Usage:
  python eval_suite.py \
    --raw1 data/raw_data1.csv --raw2 data/raw_data2.csv \
    --out figs_eval_suite --T 40 \
    --ckpt figs_phq_multi_attn/phq_multi_seq.ckpt
"""

import os, json, argparse
import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score, f1_score, balanced_accuracy_score, cohen_kappa_score,
    confusion_matrix
)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

import torch
import torch.nn.functional as F

# 네가 올린 파일(approach3.py 또는 그 이름)에서 import
# 파일명이 다르면 아래 import만 맞춰줘
from approach3 import (
    build_dataset, load_model, ensure_dir,
    SURVEY_PHQ9, SURVEY_P4, SURVEY_LONELINESS,
    SeqModelPHQ, Batch,
    ablation_channel_eval,  # 이미 구현된 check3 ablation
)

LABELS = ["improved", "same", "worse"]  # 0,1,2

def class_dist(y: np.ndarray):
    # y in {0,1,2}
    vc = pd.Series(y).value_counts().reindex([0,1,2], fill_value=0)
    dist = (vc / vc.sum()).values
    return vc.values, dist

def metrics_all(y_true: np.ndarray, y_pred: np.ndarray):
    out = {}
    out["acc"] = float(accuracy_score(y_true, y_pred))
    out["bal_acc"] = float(balanced_accuracy_score(y_true, y_pred))
    out["macro_f1"] = float(f1_score(y_true, y_pred, average="macro"))
    out["weighted_f1"] = float(f1_score(y_true, y_pred, average="weighted"))
    out["per_class_f1"] = (
        f1_score(y_true, y_pred, average=None, labels=[0,1,2]).astype(float).tolist()
    )
    out["kappa"] = float(cohen_kappa_score(y_true, y_pred, labels=[0,1,2]))
    out["cm"] = confusion_matrix(y_true, y_pred, labels=[0,1,2]).tolist()
    return out

@torch.no_grad()
def predict_multi(batch: Batch, model: SeqModelPHQ, split="val", y0_mask=True):
    if split == "val":
        idx = batch.idx_val
    else:
        idx = batch.idx_tr

    S = batch.S[idx]
    C = batch.C[idx]
    y0_phq = batch.y0_phq[idx]
    y0_p4  = batch.y0_p4[idx]
    y0_lon = batch.y0_lon[idx]

    dPHQ = batch.dY_phq[idx].detach().cpu().numpy()
    dP4  = batch.dY_p4[idx].detach().cpu().numpy()
    dLon = batch.dY_lon[idx].detach().cpu().numpy()

    model.eval()
    logit_phq, logit_p4, logit_lon, attn, ch_alpha = model.forward_multi(
        S, C, y0_phq, y0_p4, y0_lon
    )

    if y0_mask:
        m_phq = model._delta_mask_from_y0(y0_phq, y_max=3)
        m_p4  = model._delta_mask_from_y0(y0_p4,  y_max=2)
        m_lon = model._delta_mask_from_y0(y0_lon, y_max=1)
        logit_phq = logit_phq.masked_fill(~m_phq, -1e9)
        logit_p4  = logit_p4.masked_fill(~m_p4,  -1e9)
        logit_lon = logit_lon.masked_fill(~m_lon, -1e9)

    pred_phq = logit_phq.argmax(-1).detach().cpu().numpy()
    pred_p4  = logit_p4.argmax(-1).detach().cpu().numpy()
    pred_lon = logit_lon.argmax(-1).detach().cpu().numpy()

    return (dPHQ, pred_phq), (dP4, pred_p4), (dLon, pred_lon)

def save_class_dist_table(outdir, split_name, dPHQ, dP4, dLon):
    rows = []
    for task, y in [("PHQ-9 Δ", dPHQ), ("P4 Δ", dP4), ("Loneliness Δ", dLon)]:
        counts, dist = class_dist(y)
        rows.append({
            "task": task,
            "n": int(np.sum(counts)),
            "improved_n": int(counts[0]),
            "same_n": int(counts[1]),
            "worse_n": int(counts[2]),
            "improved_%": float(dist[0]),
            "same_%": float(dist[1]),
            "worse_%": float(dist[2]),
        })
    df = pd.DataFrame(rows)
    path = os.path.join(outdir, f"class_dist_{split_name}.csv")
    df.to_csv(path, index=False)
    print("[SAVE]", path)
    print(df)

def baseline_majority(y_tr: np.ndarray, y_te: np.ndarray):
    maj = pd.Series(y_tr).value_counts().idxmax()
    pred = np.full_like(y_te, maj)
    return pred, {"majority_class": int(maj)}

def baseline_stratified(y_tr: np.ndarray, y_te: np.ndarray, seed=42):
    rng = np.random.default_rng(seed)
    p = pd.Series(y_tr).value_counts(normalize=True).reindex([0,1,2], fill_value=0).values
    pred = rng.choice([0,1,2], size=len(y_te), p=p)
    return pred, {"train_dist": p.tolist()}

def baseline_logreg_agg(C_tr: np.ndarray, C_te: np.ndarray, y_tr: np.ndarray, seed=42):
    """
    Aggregated features from checks:
      - mean over time per channel (6)
      - std over time per channel (6)
      - last value per channel (6)
    total 18 dims.
    """
    def featurize(C):
        mu = C.mean(axis=1)
        sd = C.std(axis=1)
        last = C[:, -1, :]
        return np.concatenate([mu, sd, last], axis=1)

    X_tr = featurize(C_tr)
    X_te = featurize(C_te)

    clf = LogisticRegression(
        max_iter=2000, solver="lbfgs", multi_class="auto",
        class_weight="balanced", random_state=seed
    )
    clf.fit(X_tr, y_tr)
    pred = clf.predict(X_te)
    return pred, {"coef_shape": list(clf.coef_.shape)}

def run_baselines(outdir, batch: Batch):
    out = {}
    # train/val split indices
    idx_tr, idx_val = batch.idx_tr, batch.idx_val

    C = batch.C.detach().cpu().numpy()
    dPHQ = batch.dY_phq.detach().cpu().numpy()
    dP4  = batch.dY_p4.detach().cpu().numpy()
    dLon = batch.dY_lon.detach().cpu().numpy()

    for name, y in [("phq", dPHQ), ("p4", dP4), ("lon", dLon)]:
        y_tr, y_te = y[idx_tr], y[idx_val]
        C_tr, C_te = C[idx_tr], C[idx_val]

        pred_maj, info_maj = baseline_majority(y_tr, y_te)
        pred_str, info_str = baseline_stratified(y_tr, y_te, seed=42)
        pred_lr,  info_lr  = baseline_logreg_agg(C_tr, C_te, y_tr, seed=42)

        out[name] = {
            "majority": {**metrics_all(y_te, pred_maj), **info_maj},
            "stratified": {**metrics_all(y_te, pred_str), **info_str},
            "logreg_agg": {**metrics_all(y_te, pred_lr), **info_lr},
        }

    path = os.path.join(outdir, "baselines.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print("[SAVE]", path)
    return out

def run_model_metrics(outdir, batch: Batch, model: SeqModelPHQ):
    out = {}

    # (A) 정상 평가 (y0-mask ON)
    (t_phq, p_phq), (t_p4, p_p4), (t_lon, p_lon) = predict_multi(batch, model, split="val", y0_mask=True)
    out["model_y0mask_on"] = {
        "phq": metrics_all(t_phq, p_phq),
        "p4":  metrics_all(t_p4,  p_p4),
        "lon": metrics_all(t_lon, p_lon),
    }

    # (B) y0-mask OFF (마스크 효과 보여주기)
    (t_phq2, p_phq2), (t_p42, p_p42), (t_lon2, p_lon2) = predict_multi(batch, model, split="val", y0_mask=False)
    out["model_y0mask_off"] = {
        "phq": metrics_all(t_phq2, p_phq2),
        "p4":  metrics_all(t_p42,  p_p42),
        "lon": metrics_all(t_lon2, p_lon2),
    }

    path = os.path.join(outdir, "model_metrics.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print("[SAVE]", path)
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw1", type=str, required=True)
    ap.add_argument("--raw2", type=str, required=True)
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--out", type=str, default="figs_eval_suite")
    ap.add_argument("--T", type=int, default=40)
    args = ap.parse_args()

    OUT = ensure_dir(args.out)

    # 1) dataset build (너 코드 그대로)
    batch = build_dataset(args.raw1, args.raw2, T=args.T, outdir=OUT, seed=42)

    # 2) model load
    device = batch.C.device
    # ckpt path를 outdir/filename 형태로 쓰는 load_model이라면,
    # 여기선 outdir, filename을 나눠서 맞춰줘야 함.
    ckpt_dir = os.path.dirname(args.ckpt)
    ckpt_file = os.path.basename(args.ckpt)
    model = load_model(ckpt_dir, device, filename=ckpt_file)

    # 3) class distribution (val)
    idx_val = batch.idx_val
    dPHQ = batch.dY_phq[idx_val].detach().cpu().numpy()
    dP4  = batch.dY_p4[idx_val].detach().cpu().numpy()
    dLon = batch.dY_lon[idx_val].detach().cpu().numpy()
    save_class_dist_table(OUT, "val", dPHQ, dP4, dLon)

    # 4) model metrics (expanded)
    run_model_metrics(OUT, batch, model)

    # 5) baselines (3종)
    run_baselines(OUT, batch)

    # 6) eval-only ablations (check3 중심)
    #    -> 네가 이미 만든 ablation_channel_eval 재사용
    for mode in ["zero", "shuffle_time", "shuffle_patient"]:
        ablation_channel_eval(batch, model, OUT, split="val", channel="check3", mode=mode, save_cm=True)

    print("\n[DONE] eval suite outputs saved in:", OUT)

if __name__ == "__main__":
    main()
