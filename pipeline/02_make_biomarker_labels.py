#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

FEATURES: List[str] = [
    "rate_proxy",
    "pause_ratio",
    "pitch_var",
    "jitter_local",
    "shimmer_local",
    "hnr",
    "vsa_proxy",
]

PREFIX = {
    "rate_proxy": "RATE",
    "pause_ratio": "PAUSE",
    "pitch_var": "PITCH",
    "jitter_local": "JITTER",
    "shimmer_local": "SHIMMER",
    "hnr": "HNR",
    "vsa_proxy": "VSA",
}


def quantize_level(feature: str, z: float, low_thr: float, high_thr: float) -> str:
    if pd.isna(z):
        if feature == "pitch_var":
            return "VAR_MED"
        if feature == "vsa_proxy":
            return "MED"
        return "MED"

    if feature == "pitch_var":
        if z <= low_thr:
            return "STABLE"
        if z >= high_thr:
            return "VAR_HIGH"
        return "VAR_MED"

    if feature == "vsa_proxy":
        if z <= low_thr:
            return "SMALL"
        if z >= high_thr:
            return "LARGE"
        return "MED"

    if z <= low_thr:
        return "LOW"
    if z >= high_thr:
        return "HIGH"
    return "MED"


def make_token(feature: str, level: str) -> str:
    pref = PREFIX[feature]
    return f"[{pref}_{level}]"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_csv", type=Path, default=Path("pipeline/artifacts/biomarkers_continuous.csv"))
    ap.add_argument("--splits", type=Path, default=Path("pipeline/artifacts/splits.json"))
    ap.add_argument("--out_csv", type=Path, default=Path("pipeline/artifacts/biomarkers_labeled.csv"))
    ap.add_argument("--stats_out", type=Path, default=Path("pipeline/artifacts/normalization_stats.json"))
    ap.add_argument("--thr_out", type=Path, default=Path("pipeline/artifacts/token_thresholds.json"))
    ap.add_argument("--low_thr", type=float, default=-0.5)
    ap.add_argument("--high_thr", type=float, default=0.5)
    args = ap.parse_args()

    df = pd.read_csv(args.in_csv).copy()
    if not {"subject_id", "meta_cohort", "audio_path"}.issubset(df.columns):
        raise ValueError("Input CSV must contain subject_id, meta_cohort, and audio_path columns")

    splits = json.loads(Path(args.splits).read_text())
    train_ids = set(splits.get("train", []))

    controls_train = df[(df["subject_id"].isin(train_ids)) & (df["meta_cohort"] == "control")].copy()
    if len(controls_train) == 0:
        raise ValueError("No TRAIN control rows found; cannot z-normalize against train controls.")

    mu_sigma: Dict[str, Dict[str, float]] = {}
    for feat in FEATURES:
        mu = float(controls_train[feat].mean())
        sigma = float(controls_train[feat].std(ddof=0))
        if not np.isfinite(sigma) or sigma == 0.0:
            sigma = 1.0
        mu_sigma[feat] = {"mean": mu, "std": float(sigma)}

    out = df[["subject_id", "meta_cohort", "audio_path"]].copy()

    level_cols = []
    token_cols = []
    z_cols = []

    for feat in FEATURES:
        mu = mu_sigma[feat]["mean"]
        sigma = mu_sigma[feat]["std"]
        z = (df[feat] - mu) / (sigma if sigma != 0 else 1.0)
        level = z.apply(lambda v: quantize_level(feat, v, args.low_thr, args.high_thr))
        token = level.apply(lambda L: make_token(feat, L))

        z_col = f"{feat}_z"
        level_col = f"{feat}_level"
        token_col = f"{feat}_tok"

        out[z_col] = z
        out[level_col] = level
        out[token_col] = token

        z_cols.append(z_col)
        level_cols.append(level_col)
        token_cols.append(token_col)

    out["prosody_label"] = (
        out["rate_proxy_level"].astype(str)
        + "__"
        + out["pause_ratio_level"].astype(str)
        + "__"
        + out["pitch_var_level"].astype(str)
    )
    out["phonation_label"] = (
        out["jitter_local_level"].astype(str)
        + "__"
        + out["shimmer_local_level"].astype(str)
        + "__"
        + out["hnr_level"].astype(str)
    )
    out["articulation_label"] = out["vsa_proxy_level"].astype(str)

    out["prosody_text"] = (
        "rate_" + out["rate_proxy_level"].astype(str).str.lower()
        + " pause_" + out["pause_ratio_level"].astype(str).str.lower()
        + " pitch_" + out["pitch_var_level"].astype(str).str.lower()
    )
    out["phonation_text"] = (
        "jitter_" + out["jitter_local_level"].astype(str).str.lower()
        + " shimmer_" + out["shimmer_local_level"].astype(str).str.lower()
        + " hnr_" + out["hnr_level"].astype(str).str.lower()
    )
    out["articulation_text"] = "vsa_" + out["vsa_proxy_level"].astype(str).str.lower()

    out = out.sort_values(["subject_id", "audio_path"]).reset_index(drop=True)

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out_csv, index=False)

    stats_obj = {
        "computed_from": "train_controls",
        "n_controls_train": int(len(controls_train)),
        "ddof": 0,
        "features": mu_sigma,
    }
    Path(args.stats_out).write_text(json.dumps(stats_obj, indent=2))

    thr_obj: Dict[str, Dict[str, object]] = {}
    for feat in FEATURES:
        pref = PREFIX[feat]
        if feat == "pitch_var":
            tokens = [f"{pref}_STABLE", f"{pref}_VAR_MED", f"{pref}_VAR_HIGH"]
        elif feat == "vsa_proxy":
            tokens = [f"{pref}_SMALL", f"{pref}_MED", f"{pref}_LARGE"]
        else:
            tokens = [f"{pref}_LOW", f"{pref}_MED", f"{pref}_HIGH"]
        thr_obj[feat] = {
            "z_low_thr": args.low_thr,
            "z_high_thr": args.high_thr,
            "tokens": tokens,
            "source_column": feat,
        }
    Path(args.thr_out).write_text(json.dumps(thr_obj, indent=2))

    print(f"Wrote labels: {args.out_csv}")
    print(f"Wrote stats:  {args.stats_out}")
    print(f"Wrote thr:    {args.thr_out}")


if __name__ == "__main__":
    main()
    