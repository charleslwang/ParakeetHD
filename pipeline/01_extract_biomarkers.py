#!/usr/bin/env python3
import argparse
from pathlib import Path

import librosa
import numpy as np
import opensmile
import pandas as pd
import parselmouth
import soundfile as sf
from tqdm import tqdm

SR = 16000


def load_audio(path: Path, target_sr: int = SR):
    y, sr = sf.read(str(path), always_2d=False)
    y = np.asarray(y, dtype=np.float32)
    if y.ndim == 2:
        y = y.mean(axis=1)
    if sr != target_sr:
        y = librosa.resample(y, orig_sr=sr, target_sr=target_sr, res_type="kaiser_best")
        sr = target_sr
    return y, sr


def vad_pause_ratio(y: np.ndarray, sr: int = SR):
    frame_len = int(0.025 * sr)
    hop = int(0.010 * sr)
    rms = librosa.feature.rms(y=y, frame_length=frame_len, hop_length=hop, center=True)[0]
    if rms.size == 0:
        return float("nan")
    rms_db = librosa.amplitude_to_db(rms, ref=np.max)
    thr = max(np.percentile(rms_db, 40.0), -60.0)
    speech = rms_db > thr
    return 1.0 - float(speech.mean())


def speech_rate_proxy(y: np.ndarray, sr: int = SR):
    frame_len = int(0.025 * sr)
    hop = int(0.010 * sr)
    rms = librosa.feature.rms(y=y, frame_length=frame_len, hop_length=hop, center=True)[0]
    if rms.size == 0:
        return float("nan")
    rms_db = librosa.amplitude_to_db(rms, ref=np.max)
    thr = max(np.percentile(rms_db, 40.0), -60.0)
    speech = rms_db > thr
    fps = sr / hop
    dur = (len(y) / sr) + 1e-9
    return float(speech.sum() / fps) / dur


def opensmile_pitch_var(wav_path: Path):
    sm = opensmile.Smile(
        feature_set=opensmile.FeatureSet.eGeMAPSv02,
        feature_level=opensmile.FeatureLevel.Functionals,
    )
    df = sm.process_file(str(wav_path))
    cols = list(df.columns)
    preferred = [c for c in cols if "F0" in c and ("stddev" in c.lower())]
    if not preferred:
        preferred = [c for c in cols if "F0" in c and ("iqr" in c.lower() or "range" in c.lower())]
    return float(df[preferred[0]].iloc[0]) if preferred else float("nan")


def praat_phonation_and_vsa_proxy(wav_path: Path):
    snd = parselmouth.Sound(str(wav_path))

    pitch_floor = 75
    pitch_ceiling = 500
    point = parselmouth.praat.call(snd, "To PointProcess (periodic, cc)", pitch_floor, pitch_ceiling)

    jitter_local = float(
        parselmouth.praat.call(point, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3)
    )
    shimmer_local = float(
        parselmouth.praat.call([snd, point], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
    )

    harm = parselmouth.praat.call(snd, "To Harmonicity (cc)", 0.01, pitch_floor, 0.1, 1.0)
    hnr = float(parselmouth.praat.call(harm, "Get mean", 0, 0))

    form = snd.to_formant_burg(
        time_step=0.01,
        max_number_of_formants=5,
        maximum_formant=5500,
        window_length=0.025,
        pre_emphasis_from=50,
    )

    duration = snd.get_total_duration()
    ts = np.arange(0, duration, 0.01)
    f1, f2 = [], []
    for t in ts:
        v1 = parselmouth.praat.call(form, "Get value at time", 1, float(t), "Hertz", "Linear")
        v2 = parselmouth.praat.call(form, "Get value at time", 2, float(t), "Hertz", "Linear")
        if v1 and v2 and 150 < v1 < 1200 and 500 < v2 < 4000:
            f1.append(float(v1))
            f2.append(float(v2))

    if len(f1) < 50:
        vsa_proxy = float("nan")
    else:
        f1 = np.array(f1)
        f2 = np.array(f2)
        vsa_proxy = float(np.sqrt(np.var(f1) + np.var(f2)))

    return jitter_local, shimmer_local, hnr, vsa_proxy


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--meta", type=Path, default=Path("pipeline/artifacts/metadata_all.csv"))
    ap.add_argument("--out", type=Path, default=Path("pipeline/artifacts/biomarkers_continuous.csv"))
    args = ap.parse_args()

    m = pd.read_csv(args.meta).fillna("")
    rows = []

    for _, r in tqdm(m.iterrows(), total=len(m)):
        sid = str(r["subject_id"]).strip()
        cohort = str(r["meta_cohort"]).strip()
        wav_path = Path(str(r["audio_path"]).strip())

        if not wav_path.exists():
            rows.append(
                {"subject_id": sid, "meta_cohort": cohort, "audio_path": str(wav_path), "missing": True}
            )
            continue

        y, sr = load_audio(wav_path, target_sr=SR)
        pause_ratio = vad_pause_ratio(y, sr)
        rate_proxy = speech_rate_proxy(y, sr)
        pitch_var = opensmile_pitch_var(wav_path)
        jitter, shimmer, hnr, vsa = praat_phonation_and_vsa_proxy(wav_path)

        rows.append(
            {
                "subject_id": sid,
                "meta_cohort": cohort,
                "audio_path": str(wav_path).replace("\\", "/"),
                "rate_proxy": rate_proxy,
                "pause_ratio": pause_ratio,
                "pitch_var": pitch_var,
                "jitter_local": jitter,
                "shimmer_local": shimmer,
                "hnr": hnr,
                "vsa_proxy": vsa,
                "missing": False,
            }
        )

    out = pd.DataFrame(rows)
    out.to_csv(args.out, index=False)
    missing_n = int((out["missing"] == True).sum()) if "missing" in out.columns else 0
    print(f"Wrote: {args.out} rows={len(out)} missing={missing_n}")


if __name__ == "__main__":
    main()
    