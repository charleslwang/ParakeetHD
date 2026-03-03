#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import pandas as pd


def norm_text(s: str) -> str:
    s = (s or "").replace("\n", " ").strip()
    return " ".join(s.split())


def write_jsonl(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for obj in rows:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--meta", type=Path, default=Path("pipeline/artifacts/metadata_all.csv"))
    ap.add_argument("--labels", type=Path, default=Path("pipeline/artifacts/biomarkers_labeled.csv"))
    ap.add_argument("--splits", type=Path, default=Path("pipeline/artifacts/splits.json"))
    ap.add_argument("--out_dir", type=Path, default=Path("pipeline/manifests/plain"))
    args = ap.parse_args()

    meta = pd.read_csv(args.meta).fillna("")
    labels = pd.read_csv(args.labels).fillna("")
    splits = json.loads(args.splits.read_text())

    if "audio_path" not in meta.columns:
        raise ValueError("metadata_all.csv must contain audio_path")
    if "audio_path" not in labels.columns:
        raise ValueError("biomarkers_labeled.csv must contain audio_path")

    df = meta.merge(labels, on=["subject_id", "meta_cohort", "audio_path"], how="inner", suffixes=("", "_bio"))

    for split_name, ids in splits.items():
        split_df = df[df["subject_id"].isin(ids)].copy()
        rows = []
        for _, r in split_df.iterrows():
            audio_abs = str(r["audio_path"]).strip().replace("\\", "/")
            if not audio_abs:
                continue

            rows.append(
                {
                    "audio": audio_abs,
                    "text": norm_text(str(r.get("text", ""))),
                    "speaker": str(r.get("subject_id", "")),
                    "cohort": str(r.get("meta_cohort", "")),
                    "prosody_label": str(r.get("prosody_label", "")),
                    "phonation_label": str(r.get("phonation_label", "")),
                    "articulation_label": str(r.get("articulation_label", "")),
                }
            )

        out_path = args.out_dir / f"{split_name}.jsonl"
        write_jsonl(out_path, rows)
        print(f"Wrote {out_path} lines={len(rows)}")


if __name__ == "__main__":
    main()
    