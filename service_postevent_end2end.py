# -*- coding: utf-8 -*-
"""
service_postevent_end2end.py

✅ CSV 생성 → 분석까지 "한 번에" 도는 올인원 스크립트.

지원 모드
---------
(A) from_batch 모드 (처음부터):
    1) batch(.pt/.pkl로 저장된 객체) 로드
    2) batch.S 기반으로 service features 생성 (binary/count/recent)
    3) post-event 라벨 테이블(df_postevent)과 pid로 merge
    4) screening + multivariate logit 수행
    5) merged CSV + 결과 CSV 모두 저장

(B) from_merged 모드 (이미 merged CSV 3개 있는 경우):
    - 이전에 만든 merged CSV 3개를 넣어서 바로 분석

요구사항(중요)
-------------
- df_postevent CSV에는 최소한 다음 컬럼이 있어야 함:
    - pid
    - post_event  (0/1)
  (그 외 컬럼은 있어도 됨)

- batch 객체는 torch.load로 로드 가능해야 함.
  (예: torch.save(batch, "batch.pt") 로 저장해둔 것)

사용 예시
---------
# 1) 처음부터 (batch -> merged -> 분석)
python service_postevent_end2end.py from_batch \
  --batch_path /home/nakyung/projects/DILFINAL/batch.pt \
  --postevent_csv figs_check3_peakwindows/per_patient_windows_val.csv \
  --outdir figs_check3_peakwindows/service_postevent_end2end \
  --split val \
  --t_cut 34 \
  --recent_window 3 \
  --p_screen 0.10 \
  --count_transform log1p

# 2) 이미 merged 3개가 있으면 (분석만)
python service_postevent_end2end.py from_merged \
  --binary_csv figs_check3_peakwindows/merged_postevent_services_binary_val.csv \
  --count_csv  figs_check3_peakwindows/merged_postevent_services_count_val.csv \
  --recent_csv figs_check3_peakwindows/merged_postevent_services_recent_val.csv \
  --outdir figs_check3_peakwindows/service_postevent_compare \
  --p_screen 0.10 \
  --count_transform log1p
"""

from __future__ import annotations

import argparse
import os
from typing import List, Dict, Tuple, Optional

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import fisher_exact

# torch는 batch 로드할 때만 필요
try:
    import torch
except Exception:
    torch = None


# -------------------------
# utils
# -------------------------
def ensure_dir(d: str) -> str:
    os.makedirs(d, exist_ok=True)
    return d


def parse_service_cols(arg: str) -> List[str]:
    cols = [c.strip() for c in arg.split(",") if c.strip()]
    if not cols:
        raise ValueError("service_cols is empty")
    return cols


def _safe_read_csv(path: str, required: Optional[List[str]] = None) -> pd.DataFrame:
    df = pd.read_csv(path)
    if required:
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"Missing columns in {path}: {missing}")
    return df


def _check_binary(y: pd.Series, name: str = "post_event") -> None:
    uniq = set(pd.Series(y).dropna().unique().tolist())
    if not uniq.issubset({0, 1}):
        raise ValueError(f"'{name}' must be binary 0/1. got unique={sorted(list(uniq))}")


# -------------------------
# feature builder (from batch)
# -------------------------
def build_service_features_from_batch(
    batch,
    split: str = "val",
    t_cut: int = 34,
    mode: str = "binary",
    recent_window: int = 3,
) -> pd.DataFrame:
    """
    batch.S 기반 서비스 feature 생성

    Assumptions:
      - batch.S shape: (B, T, d_s)
      - batch.idx_val / batch.idx_tr 존재
      - batch.stats["patients"] : list-like of pid aligned with batch dimension
      - batch.stats["service_cols"] : list of service col names length == d_s

    Output:
      DataFrame with columns: pid, <service_cols...>
    """
    if split == "val":
        idx = batch.idx_val
    elif split == "train":
        idx = batch.idx_tr
    else:
        raise ValueError("split must be 'val' or 'train'")

    S = batch.S[idx].detach().cpu().numpy()
    pids = np.array(batch.stats["patients"])[idx]
    service_cols = list(batch.stats["service_cols"])

    B, T, d_s = S.shape
    if len(service_cols) != d_s:
        raise ValueError(f"len(service_cols)={len(service_cols)} != d_s={d_s}")

    rows = []
    for i in range(B):
        row = {"pid": pids[i]}
        for j, sname in enumerate(service_cols):
            if mode == "binary":
                row[sname] = int(S[i, :t_cut, j].max() > 0)
            elif mode == "count":
                row[sname] = int(S[i, :t_cut, j].sum())
            elif mode == "recent":
                lo = max(0, t_cut - recent_window)
                row[sname] = int(S[i, lo:t_cut, j].max() > 0)
            else:
                raise ValueError("unknown mode")
        rows.append(row)

    return pd.DataFrame(rows)


def make_merged_tables_from_batch(
    batch,
    postevent_csv: str,
    outdir: str,
    split: str,
    t_cut: int,
    recent_window: int,
    post_event_col: str = "post_event",
) -> Dict[str, str]:
    """
    Create merged CSVs:
      - merged_postevent_services_binary_<split>.csv
      - merged_postevent_services_count_<split>.csv
      - merged_postevent_services_recent_<split>.csv

    postevent_csv must contain: pid, post_event (0/1).
    """
    OUT = ensure_dir(outdir)
    df_post = _safe_read_csv(postevent_csv, required=["pid", post_event_col]).copy()
    _check_binary(df_post[post_event_col], name=post_event_col)

    paths = {}
    for mode in ["binary", "count", "recent"]:
        Xs = build_service_features_from_batch(
            batch=batch,
            split=split,
            t_cut=t_cut,
            mode=mode,
            recent_window=recent_window,
        )
        dfm = df_post.merge(Xs, on="pid", how="inner")

        out_path = os.path.join(OUT, f"merged_postevent_services_{mode}_{split}.csv")
        dfm.to_csv(out_path, index=False)
        paths[mode] = out_path

    return paths


# -------------------------
# screening
# -------------------------
def screening_binary_like(df: pd.DataFrame, service_cols: List[str], y_col: str) -> pd.DataFrame:
    rows = []
    for s in service_cols:
        tab = pd.crosstab(df[s], df[y_col])
        if tab.shape != (2, 2):
            continue
        OR, p = fisher_exact(tab)
        rows.append({"service": s, "OR": float(OR), "p": float(p)})
    out = pd.DataFrame(rows)
    if len(out) == 0:
        return pd.DataFrame(columns=["service", "OR", "p"])
    return out.sort_values("p", ascending=True).reset_index(drop=True)


def _univ_logit(y: np.ndarray, x: np.ndarray) -> Tuple[Optional[float], Optional[float]]:
    try:
        X = sm.add_constant(x.astype(float))
        m = sm.Logit(y.astype(float), X).fit(disp=False)
        beta = float(m.params[1])
        p = float(m.pvalues[1])
        OR = float(np.exp(beta))
        return OR, p
    except Exception:
        return None, None


def screening_count(df: pd.DataFrame, service_cols: List[str], y_col: str, transform: str) -> pd.DataFrame:
    y = df[y_col].values.astype(float)
    rows = []
    for s in service_cols:
        x = df[s].values.astype(float)
        if transform == "log1p":
            x = np.log1p(x)
        elif transform == "none":
            pass
        else:
            raise ValueError("transform must be 'log1p' or 'none'")
        OR, p = _univ_logit(y, x)
        if OR is None:
            continue
        rows.append({"service": s, "OR": float(OR), "p": float(p)})
    out = pd.DataFrame(rows)
    if len(out) == 0:
        return pd.DataFrame(columns=["service", "OR", "p"])
    return out.sort_values("p", ascending=True).reset_index(drop=True)


# -------------------------
# multivariate logistic
# -------------------------
def multivar_logit(
    df: pd.DataFrame,
    y_col: str,
    x_cols: List[str],
    mode: str,
    count_transform: str = "log1p",
) -> pd.DataFrame:
    if len(x_cols) == 0:
        return pd.DataFrame(columns=["var", "Coef.", "Std.Err.", "z", "P>|z|", "OR", "CI_lo", "CI_hi"])

    X = df[x_cols].copy()

    if mode == "count":
        if count_transform == "log1p":
            for c in x_cols:
                X[c] = np.log1p(X[c].astype(float))
        elif count_transform == "none":
            X = X.astype(float)
        else:
            raise ValueError("count_transform must be 'log1p' or 'none'")

    X = sm.add_constant(X)
    y = df[y_col].astype(float)

    m = sm.Logit(y, X).fit(disp=False)
    tab = m.summary2().tables[1].copy()

    tab["OR"] = np.exp(tab["Coef."])
    tab["CI_lo"] = np.exp(tab["Coef."] - 1.96 * tab["Std.Err."])
    tab["CI_hi"] = np.exp(tab["Coef."] + 1.96 * tab["Std.Err."])

    return tab.reset_index().rename(columns={"index": "var"})


# -------------------------
# analysis runner
# -------------------------
def run_one_mode(
    csv_path: str,
    outdir: str,
    mode: str,
    service_cols: List[str],
    y_col: str,
    p_screen: float,
    count_transform: str,
) -> Dict[str, str]:
    df = _safe_read_csv(csv_path)
    for c in ["pid", y_col] + service_cols:
        if c not in df.columns:
            raise ValueError(f"Missing column '{c}' in {csv_path}")

    _check_binary(df[y_col], name=y_col)

    if mode in ("binary", "recent"):
        screen = screening_binary_like(df, service_cols, y_col=y_col)
    elif mode == "count":
        screen = screening_count(df, service_cols, y_col=y_col, transform=count_transform)
    else:
        raise ValueError("mode must be binary/count/recent")

    screen_path = os.path.join(outdir, f"screening_{mode}.csv")
    screen.to_csv(screen_path, index=False)

    cand = screen.loc[screen["p"] < p_screen, "service"].tolist() if len(screen) else []

    logit = multivar_logit(
        df=df,
        y_col=y_col,
        x_cols=cand,
        mode=mode,
        count_transform=count_transform,
    )
    logit_path = os.path.join(outdir, f"logit_{mode}.csv")
    logit.to_csv(logit_path, index=False)

    return {
        "mode": mode,
        "csv": csv_path,
        "screening_csv": screen_path,
        "logit_csv": logit_path,
        "n_rows": str(len(df)),
        "n_candidates": str(len(cand)),
    }


def run_analysis_from_merged(
    binary_csv: str,
    count_csv: str,
    recent_csv: str,
    outdir: str,
    service_cols: List[str],
    y_col: str,
    p_screen: float,
    count_transform: str,
) -> str:
    OUT = ensure_dir(outdir)
    outs = []
    outs.append(run_one_mode(binary_csv, OUT, "binary", service_cols, y_col, p_screen, count_transform))
    outs.append(run_one_mode(count_csv,  OUT, "count",  service_cols, y_col, p_screen, count_transform))
    outs.append(run_one_mode(recent_csv, OUT, "recent", service_cols, y_col, p_screen, count_transform))

    summary = pd.DataFrame(outs)
    summary_path = os.path.join(OUT, "run_summary.csv")
    summary.to_csv(summary_path, index=False)

    print("[DONE] saved:", summary_path)
    print(summary.to_string(index=False))
    return summary_path


# -------------------------
# CLI
# -------------------------
def cli_from_batch(args) -> None:
    if torch is None:
        raise RuntimeError("PyTorch is required for from_batch mode, but torch import failed.")

    OUT = ensure_dir(args.outdir)

    # load batch
    batch = torch.load(args.batch_path, map_location="cpu")

    # create merged csvs (binary/count/recent)
    merged_paths = make_merged_tables_from_batch(
        batch=batch,
        postevent_csv=args.postevent_csv,
        outdir=OUT,
        split=args.split,
        t_cut=args.t_cut,
        recent_window=args.recent_window,
        post_event_col=args.y_col,
    )

    # service columns: batch.stats["service_cols"] 기준(권장)
    service_cols = list(batch.stats["service_cols"])

    # run analysis
    summary_path = run_analysis_from_merged(
        binary_csv=merged_paths["binary"],
        count_csv=merged_paths["count"],
        recent_csv=merged_paths["recent"],
        outdir=OUT,
        service_cols=service_cols,
        y_col=args.y_col,
        p_screen=args.p_screen,
        count_transform=args.count_transform,
    )
    print("[END2END DONE]", summary_path)


def cli_from_merged(args) -> None:
    OUT = ensure_dir(args.outdir)
    service_cols = parse_service_cols(args.service_cols)

    run_analysis_from_merged(
        binary_csv=args.binary_csv,
        count_csv=args.count_csv,
        recent_csv=args.recent_csv,
        outdir=OUT,
        service_cols=service_cols,
        y_col=args.y_col,
        p_screen=args.p_screen,
        count_transform=args.count_transform,
    )


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    # from_batch
    ap_b = sub.add_parser("from_batch", help="batch -> merged CSVs -> analysis end2end")
    ap_b.add_argument("--batch_path", required=True, help="torch.load 가능한 batch 파일(.pt/.pkl)")
    ap_b.add_argument("--postevent_csv", required=True, help="pid + post_event 컬럼 있는 CSV")
    ap_b.add_argument("--outdir", default="service_postevent_end2end_outputs")

    ap_b.add_argument("--split", default="val", choices=["val", "train"])
    ap_b.add_argument("--t_cut", type=int, default=34)
    ap_b.add_argument("--recent_window", type=int, default=3)

    ap_b.add_argument("--y_col", default="post_event", help="outcome column name in postevent_csv")
    ap_b.add_argument("--p_screen", type=float, default=0.10)
    ap_b.add_argument("--count_transform", default="log1p", choices=["log1p", "none"])

    # from_merged
    ap_m = sub.add_parser("from_merged", help="already-merged CSVs -> analysis")
    ap_m.add_argument("--binary_csv", required=True)
    ap_m.add_argument("--count_csv", required=True)
    ap_m.add_argument("--recent_csv", required=True)
    ap_m.add_argument("--outdir", default="service_postevent_compare")
    ap_m.add_argument("--service_cols", default=",".join([f"service{i}" for i in range(1, 17)]))
    ap_m.add_argument("--y_col", default="post_event")
    ap_m.add_argument("--p_screen", type=float, default=0.10)
    ap_m.add_argument("--count_transform", default="log1p", choices=["log1p", "none"])

    args = ap.parse_args()

    if args.cmd == "from_batch":
        cli_from_batch(args)
    elif args.cmd == "from_merged":
        cli_from_merged(args)
    else:
        raise ValueError("unknown cmd")


if __name__ == "__main__":
    main()
