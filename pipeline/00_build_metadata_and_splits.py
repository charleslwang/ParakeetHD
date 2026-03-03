#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def clean_str(x) -> str:
    if x is None:
        return ""
    return str(x).replace("\ufeff", "").strip()


def normalize_cohort(x: str) -> str:
    s = clean_str(x).lower()
    s = " ".join(s.split())
    if s in {"control", "healthy", "hc"}:
        return "control"
    if s in {"manifest", "manifest hd", "hd", "symptomatic"}:
        return "manifest"
    if s in {"prodromal", "prodromal hd"}:
        return "prodromal"
    if s in {"prehd", "pre-hd", "pre hd", "premanifest", "pre-manifest"}:
        return "prehd"
    return s


def stratified_split(subjects, cohorts, seed=13, train=0.7, dev=0.1):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({"subject_id": subjects, "cohort": cohorts}).drop_duplicates()

    train_ids, dev_ids, test_ids = [], [], []

    for coh, g in df.groupby("cohort"):
        ids = g["subject_id"].tolist()
        rng.shuffle(ids)
        n = len(ids)

        n_train = int(round(train * n))
        n_dev = int(round(dev * n))

        if n >= 3:
            n_train = max(1, min(n_train, n - 2))
            n_dev = max(1, min(n_dev, n - n_train - 1))
        elif n == 2:
            n_train = 1
            n_dev = 0
        else:
            n_train = 1
            n_dev = 0

        train_ids += ids[:n_train]
        dev_ids += ids[n_train:n_train + n_dev]
        test_ids += ids[n_train + n_dev:]

    return sorted(train_ids), sorted(dev_ids), sorted(test_ids)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--meta", type=Path, default=Path("data/metadata.csv"))
    ap.add_argument("--out_dir", type=Path, default=Path("pipeline/artifacts"))
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--min_text_len", type=int, default=10)
    ap.add_argument(
        "--audio_root",
        type=Path,
        default=Path("data"),
        help="Root directory that metadata file_name paths are relative to.",
    )
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    meta = pd.read_csv(args.meta).fillna("")
    required = ["subject_id", "file_name", "text", "class_classification"]
    for c in required:
        if c not in meta.columns:
            raise ValueError(f"metadata.csv missing required column: {c}")

    meta["subject_id"] = meta["subject_id"].apply(clean_str)
    meta["file_name"] = meta["file_name"].apply(clean_str)
    meta["text"] = meta["text"].apply(clean_str)
    meta["class_classification"] = meta["class_classification"].apply(clean_str)

    meta = meta[(meta["subject_id"] != "") & (meta["file_name"] != "")]
    meta = meta[meta["text"].astype(str).str.len() >= args.min_text_len].copy()

    meta["audio_path"] = meta["file_name"].apply(
        lambda p: str((args.audio_root / p).resolve()).replace("\\", "/")
    )
    exists_all = meta["audio_path"].apply(lambda p: Path(p).exists())
    missing_all = int((~exists_all).sum())
    if missing_all > 0:
        print(f"WARNING: dropping {missing_all} rows with non-existent audio_path after rooting.")

    meta_all = meta[exists_all].copy()
    meta_all["meta_cohort"] = meta_all["class_classification"].apply(normalize_cohort)
    meta_all = meta_all[meta_all["meta_cohort"].astype(str).str.strip() != ""].copy()

    out_meta_all = args.out_dir / "metadata_all.csv"
    meta_all.to_csv(out_meta_all, index=False)

    m = meta_all.sort_values(["subject_id"]).drop_duplicates(subset=["subject_id"], keep="first").copy()
    m["meta_cohort"] = m["class_classification"].apply(normalize_cohort)
    m = m[m["meta_cohort"].astype(str).str.strip() != ""].copy()

    out_meta_subject = args.out_dir / "metadata_subjects.csv"
    m.to_csv(out_meta_subject, index=False)

    train_ids, dev_ids, test_ids = stratified_split(
        m["subject_id"].tolist(),
        m["meta_cohort"].astype(str).tolist(),
        seed=args.seed,
    )
    splits = {"train": train_ids, "dev": dev_ids, "test": test_ids}
    out_splits = args.out_dir / "splits.json"
    out_splits.write_text(json.dumps(splits, indent=2))

    print(f"Wrote: {out_meta_all} rows={len(meta_all)} (all utterances)")
    print(f"Wrote: {out_meta_subject} rows={len(m)} unique_speakers={m['subject_id'].nunique()}")
    print(f"Wrote: {out_splits} train={len(train_ids)} dev={len(dev_ids)} test={len(test_ids)}")
    print("Cohort counts:")
    print(m["meta_cohort"].value_counts())


if __name__ == "__main__":
    main()
    