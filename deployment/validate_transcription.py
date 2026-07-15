#!/usr/bin/env python3
"""Experimental end-to-end ParakeetHD Core ML transcription parity check.

This complements validate_coreml.py. The tensor validator proves each exported
subnetwork matches PyTorch on saved fixtures; this script exercises real WAV
files and compares an experimental Python Core ML greedy decode with NeMo.
The Swift app runtime should use the same decoding semantics and must be
validated separately on device.
"""

from __future__ import annotations

import argparse
import json
import platform
import re
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

try:
    from .model_loader import load_parakeet_hd
except ImportError:
    from model_loader import load_parakeet_hd


def normalize_text(value: str) -> str:
    value = value.lower()
    value = re.sub(r"[^\w\s']", " ", value)
    return re.sub(r"\s+", " ", value).strip()


def load_wav_mono(path: Path, sample_rate: int) -> np.ndarray:
    audio, sr = sf.read(str(path), always_2d=False)
    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    if sr != sample_rate:
        import librosa

        audio = librosa.resample(audio, orig_sr=sr, target_sr=sample_rate, res_type="kaiser_best")
    return audio.astype(np.float32, copy=False)


def read_manifest(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def audio_path_from_row(row: dict[str, Any]) -> Path:
    for key in ("audio", "audio_filepath", "wav", "path"):
        if key in row and str(row[key]).strip():
            return Path(str(row[key]))
    raise KeyError("Manifest row does not contain audio or audio_filepath")


def text_from_row(row: dict[str, Any]) -> str:
    for key in ("text", "transcript", "reference"):
        if key in row:
            return str(row[key])
    return ""


def semantic_name(features: list[dict[str, Any]], semantic: str) -> str:
    for feature in features:
        if feature.get("semantic") == semantic:
            return str(feature["name"])
    raise KeyError(f"Bundle does not define a feature with semantic={semantic!r}")


def load_initial_decoder_inputs(bundle_dir: Path) -> dict[str, np.ndarray]:
    manifest = json.loads((bundle_dir / "validation" / "manifest.json").read_text(encoding="utf-8"))
    decoder_inputs = bundle_dir / "validation" / manifest["decoder_joint"]["inputs"]
    with np.load(decoder_inputs, allow_pickle=False) as stored:
        return {name: stored[name] for name in stored.files}


def detokenize(token_ids: list[int], tokens: list[str]) -> str:
    pieces = [tokens[token_id] for token_id in token_ids if 0 <= token_id < len(tokens)]
    text = "".join(pieces).replace("▁", " ")
    return re.sub(r"\s+", " ", text).strip()


class CoreMLTDTGreedyDecoder:
    def __init__(self, bundle_path: Path):
        import coremltools as ct

        self.bundle_path = bundle_path.expanduser().resolve()
        self.bundle_dir = self.bundle_path.parent
        self.bundle = json.loads(self.bundle_path.read_text(encoding="utf-8"))
        self.sample_rate = int(self.bundle["audio"]["sample_rate"])
        self.tokens = (self.bundle_dir / self.bundle["tokenizer"]["tokens_path"]).read_text(
            encoding="utf-8"
        ).splitlines()
        self.decoding = self.bundle["decoding"]
        self.blank_id = int(self.decoding["blank_id"])
        self.durations = [int(x) for x in self.decoding.get("durations", [1])]

        encoder_cfg = self.bundle["coreml"]["encoder"]
        decoder_cfg = self.bundle["coreml"]["decoder_joint"]
        if self.bundle["audio"]["encoder_input"] != "waveform":
            raise ValueError(
                "validate_transcription.py currently requires a waveform-input bundle. "
                "Feature-input bundles need a preprocessor parity implementation."
            )

        self.encoder = ct.models.MLModel(str(self.bundle_dir / encoder_cfg["path"]), compute_units=ct.ComputeUnit.ALL)
        self.decoder = ct.models.MLModel(str(self.bundle_dir / decoder_cfg["path"]), compute_units=ct.ComputeUnit.ALL)
        self.encoder_inputs = encoder_cfg["inputs"]
        self.encoder_outputs = encoder_cfg["outputs"]
        self.decoder_inputs = decoder_cfg["inputs"]
        self.decoder_outputs = decoder_cfg["outputs"]
        self.initial_decoder_inputs = load_initial_decoder_inputs(self.bundle_dir)

    def _split_decoder_outputs(
        self, prediction: dict[str, np.ndarray]
    ) -> tuple[np.ndarray, np.ndarray | None, np.ndarray, np.ndarray]:
        output_by_semantic = {
            feature["semantic"]: prediction[feature["name"]]
            for feature in self.decoder_outputs
            if feature["name"] in prediction
        }
        states_1 = output_by_semantic["output_states_1"]
        states_2 = output_by_semantic["output_states_2"]
        if "token_logits" in output_by_semantic and "duration_logits" in output_by_semantic:
            return output_by_semantic["token_logits"], output_by_semantic["duration_logits"], states_1, states_2

        logits = output_by_semantic["joint_logits"]
        layout = self.decoding.get("joint_output_layout")
        token_count = int(self.decoding["token_class_count"])
        duration_count = int(self.decoding["duration_class_count"])
        if layout == "token_scores_then_duration_scores":
            return logits[..., :token_count], logits[..., token_count : token_count + duration_count], states_1, states_2
        return logits, None, states_1, states_2

    def transcribe(self, wav_path: Path, max_symbols_per_frame: int) -> str:
        audio = load_wav_mono(wav_path, self.sample_rate)
        encoder_input = {
            semantic_name(self.encoder_inputs, "audio_signal"): audio.reshape(1, -1),
            semantic_name(self.encoder_inputs, "length"): np.array([audio.shape[0]], dtype=np.int32),
        }
        encoded = self.encoder.predict(encoder_input)
        encoder_outputs = encoded[semantic_name(self.encoder_outputs, "encoder_outputs")]
        encoded_length = int(np.asarray(encoded[semantic_name(self.encoder_outputs, "encoded_length")]).reshape(-1)[0])

        target_name = semantic_name(self.decoder_inputs, "targets")
        target_length_name = semantic_name(self.decoder_inputs, "target_length")
        input_state_1_name = semantic_name(self.decoder_inputs, "input_states_1")
        input_state_2_name = semantic_name(self.decoder_inputs, "input_states_2")
        encoder_frame_name = semantic_name(self.decoder_inputs, "encoder_outputs")
        states_1 = self.initial_decoder_inputs[input_state_1_name]
        states_2 = self.initial_decoder_inputs[input_state_2_name]
        target = np.array([[self.blank_id]], dtype=np.int32)
        target_length = np.array([1], dtype=np.int32)

        token_ids: list[int] = []
        frame = 0
        while frame < encoded_length:
            symbols = 0
            need_loop = True
            last_skip = 1
            while need_loop and symbols < max_symbols_per_frame:
                frame_slice = encoder_outputs[:, :, frame : frame + 1]
                prediction = self.decoder.predict(
                    {
                        encoder_frame_name: frame_slice,
                        target_name: target,
                        target_length_name: target_length,
                        input_state_1_name: states_1,
                        input_state_2_name: states_2,
                    }
                )
                token_logits, duration_logits, next_states_1, next_states_2 = self._split_decoder_outputs(prediction)
                token_id = int(np.argmax(np.asarray(token_logits).reshape(-1)))
                duration_index = int(np.argmax(np.asarray(duration_logits).reshape(-1))) if duration_logits is not None else 0
                skip = self.durations[min(duration_index, len(self.durations) - 1)]
                last_skip = skip

                if token_id != self.blank_id:
                    token_ids.append(token_id)
                    target = np.array([[token_id]], dtype=np.int32)
                    states_1, states_2 = next_states_1, next_states_2
                symbols += 1

                frame += skip
                need_loop = skip == 0

            if last_skip == 0 or symbols >= max_symbols_per_frame:
                frame += 1

        return detokenize(token_ids, self.tokens)


def nemo_transcribe(model, wav_path: Path) -> str:
    result = model.transcribe([str(wav_path)], batch_size=1)
    first = result[0] if isinstance(result, (list, tuple)) else result
    if hasattr(first, "text"):
        return str(first.text)
    if isinstance(first, dict):
        return str(first.get("text", first))
    return str(first)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--base-model", default="nvidia/parakeet-tdt-0.6b-v2")
    parser.add_argument("--adapter-source", default="charleslwang/parakeet-tdt-0.6b-HD")
    parser.add_argument("--adapter-filename", default="best_adapter.pt")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--max-symbols-per-frame", type=int, default=10)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()

    decoder = CoreMLTDTGreedyDecoder(args.bundle)
    nemo_model, _ = load_parakeet_hd(
        base_model=args.base_model,
        adapter_source=args.adapter_source,
        adapter_filename=args.adapter_filename,
        device=args.device,
    )

    rows = read_manifest(args.manifest)
    if args.limit is not None:
        rows = rows[: args.limit]

    results = []
    for row in rows:
        wav_path = audio_path_from_row(row)
        reference = text_from_row(row)
        nemo_text = nemo_transcribe(nemo_model, wav_path)
        coreml_text = decoder.transcribe(wav_path, args.max_symbols_per_frame)
        results.append(
            {
                "audio": str(wav_path),
                "reference": reference,
                "nemo": nemo_text,
                "coreml": coreml_text,
                "nemo_matches_coreml_normalized": normalize_text(nemo_text) == normalize_text(coreml_text),
            }
        )

    report = {
        "host": platform.platform(),
        "bundle": str(args.bundle),
        "manifest": str(args.manifest),
        "count": len(results),
        "all_nemo_coreml_matches_normalized": all(r["nemo_matches_coreml_normalized"] for r in results),
        "results": results,
    }
    text = json.dumps(report, indent=2) + "\n"
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(text, encoding="utf-8")
    print(text, end="")
    return 0 if report["all_nemo_coreml_matches_normalized"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
