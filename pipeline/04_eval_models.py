#!/usr/bin/env python3
import argparse
import json
import os
import re
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from typing import Any as TypingAny
from unittest.mock import patch

import numpy as np
import soundfile as sf
import torch
from huggingface_hub import hf_hub_download
from jiwer import process_words
from transformers import WhisperForConditionalGeneration, WhisperProcessor
from omegaconf import DictConfig, ListConfig
from omegaconf.base import ContainerMetadata


def normalize_text(s: str) -> str:
    s = s.lower()
    s = re.sub(r"[^\w\s]", "", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def safe_name(x: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", x)


def load_manifest(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def load_wav_mono(path: str) -> Tuple[np.ndarray, int]:
    y, sr = sf.read(path, always_2d=False)
    y = np.asarray(y, dtype=np.float32)
    if y.ndim == 2:
        y = y.mean(axis=1)
    return y, sr


def resample_if_needed(y: np.ndarray, sr: int, target_sr: int = 16000) -> Tuple[np.ndarray, int]:
    if sr == target_sr:
        return y, sr
    import librosa
    y = librosa.resample(y, orig_sr=sr, target_sr=target_sr, res_type="kaiser_best")
    return y, target_sr


def normalize_transcribe_output(x: Any) -> str:
    if isinstance(x, str):
        return x
    if hasattr(x, "text"):
        return str(x.text)
    if isinstance(x, dict):
        for k in ["text", "pred_text", "transcript"]:
            if k in x:
                return str(x[k])
    return str(x)


def aggregate_rows(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    S = sum(r["substitutions"] for r in rows)
    D = sum(r["deletions"] for r in rows)
    I = sum(r["insertions"] for r in rows)
    H = sum(r["hits"] for r in rows)
    N = sum(r["reference_words"] for r in rows)
    wer = float((S + D + I) / N) if N > 0 else None
    return {
        "substitutions": S,
        "deletions": D,
        "insertions": I,
        "hits": H,
        "reference_words": N,
        "wer": wer,
    }


def enable_nemo_safe_unpickling():
    """
    PyTorch 2.6+ defaults to safer unpickling behavior. NeMo adapter bundles
    saved via model.save_adapters(...) can contain OmegaConf objects and typing metadata.
    Since these are our own trusted checkpoints, allowlist the needed classes.
    """
    safe = [DictConfig, ListConfig, ContainerMetadata, TypingAny]

    try:
        from omegaconf.nodes import (
            AnyNode,
            BooleanNode,
            BytesNode,
            EnumNode,
            FloatNode,
            IntegerNode,
            PathNode,
            StringNode,
        )
        safe.extend([
            AnyNode,
            BooleanNode,
            BytesNode,
            EnumNode,
            FloatNode,
            IntegerNode,
            PathNode,
            StringNode,
        ])
    except Exception:
        pass

    try:
        torch.serialization.add_safe_globals(safe)
        print("[INFO] Added OmegaConf/typing classes to torch safe globals.")
    except Exception as e:
        print(f"[WARN] Could not add safe globals: {e}")


class WhisperRunner:
    def __init__(self, model_id: str, device: str):
        self.processor = WhisperProcessor.from_pretrained(model_id)
        self.model = WhisperForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=torch.float16 if device == "cuda" else None,
        )
        self.model.generation_config.language = "en"
        self.model.generation_config.task = "transcribe"
        self.model.generation_config.forced_decoder_ids = None
        self.model.generation_config.suppress_tokens = []
        self.model.eval()
        self.model.to(device)
        self.device = device

        try:
            self.model_dtype = next(self.model.parameters()).dtype
        except StopIteration:
            self.model_dtype = torch.float16 if device == "cuda" else torch.float32

        self.target_sr = 16000
        self.chunk_length_s = 25.0
        self.chunk_overlap_s = 5.0

    @staticmethod
    def _normalize_ws(s: str) -> str:
        return re.sub(r"\s+", " ", s).strip()

    @staticmethod
    def _longest_suffix_prefix_word_overlap(left: str, right: str, max_words: int = 30) -> int:
        left_words = WhisperRunner._normalize_ws(left).split()
        right_words = WhisperRunner._normalize_ws(right).split()
        max_k = min(len(left_words), len(right_words), max_words)
        for k in range(max_k, 0, -1):
            if left_words[-k:] == right_words[:k]:
                return k
        return 0

    def _stitch_texts(self, texts: List[str]) -> str:
        merged = ""
        for t in texts:
            t = self._normalize_ws(t)
            if not t:
                continue
            if not merged:
                merged = t
                continue

            k = self._longest_suffix_prefix_word_overlap(merged, t)
            if k > 0:
                t_words = t.split()
                merged = merged + " " + " ".join(t_words[k:])
            else:
                merged = merged + " " + t

            merged = self._normalize_ws(merged)
        return merged

    def _transcribe_chunk(self, y_chunk: np.ndarray, sr: int, max_length: int, num_beams: int, temperature: float) -> str:
        feats = self.processor.feature_extractor(
            y_chunk,
            sampling_rate=sr,
            return_tensors="pt",
        )
        input_features = feats.input_features.to(device=self.device, dtype=self.model_dtype)

        kwargs = dict(
            input_features=input_features,
            max_length=max_length,
            num_beams=num_beams,
            do_sample=False,
            temperature=temperature,
        )

        if hasattr(feats, "attention_mask") and feats.attention_mask is not None:
            kwargs["attention_mask"] = feats.attention_mask.to(device=self.device, dtype=torch.long)

        with torch.no_grad():
            pred_ids = self.model.generate(**kwargs)

        return self.processor.tokenizer.decode(pred_ids[0], skip_special_tokens=True)

    def transcribe(self, wav_path: str, max_length: int, num_beams: int, temperature: float) -> str:
        y, sr = load_wav_mono(wav_path)
        y, sr = resample_if_needed(y, sr, self.target_sr)

        duration_s = len(y) / float(sr) if sr else 0.0
        effective_max_length = max(max_length, 448)

        if duration_s <= 28.0:
            return self._transcribe_chunk(y, sr, effective_max_length, num_beams, temperature)

        chunk_size = int(self.chunk_length_s * sr)
        overlap = int(self.chunk_overlap_s * sr)
        step = chunk_size - overlap
        if step <= 0:
            raise ValueError("chunk_length_s must be greater than chunk_overlap_s")

        chunk_texts = []
        start = 0
        n = len(y)
        while start < n:
            end = min(start + chunk_size, n)
            y_chunk = y[start:end]
            if len(y_chunk) < int(0.5 * sr):
                break

            txt = self._transcribe_chunk(y_chunk, sr, effective_max_length, num_beams, temperature)
            chunk_texts.append(txt)

            if end >= n:
                break
            start += step

        return self._stitch_texts(chunk_texts)


class ParakeetRunner:
    def __init__(self, spec: Dict[str, Any], device: str):
        self.spec = spec
        self.device = device
        self.mode = spec.get("mode", "baseline")
        self.adapter_loaded = False

        # Build the model in adapter-compatible form only for PEFT runs.
        self.model = self._load_model(
            model_ref=spec["model_id"],
            enable_adapter_support=(self.mode == "peft"),
        )

        # Match the decoding patch you used elsewhere.
        self._patch_rnnt_decoding()

        # IMPORTANT:
        # Load adapters BEFORE moving the model to GPU.
        # This mirrors the load path that worked in 07_train_parakeet_aux.py.
        if self.mode == "peft":
            adapter_repo_id = spec["adapter_repo_id"]
            adapter_filename = spec.get("adapter_filename", "best_adapter.pt")
            self._load_adapter_from_hf(adapter_repo_id, adapter_filename)

        # Optional eval-time chunking switch if you want parity with training.
        if spec.get("enable_subsampling_chunking", False):
            try:
                self.model.change_subsampling_conv_chunking_factor(1)
                print("[INFO] Enabled Parakeet subsampling conv auto-chunking for eval.")
            except Exception as e:
                print(f"[WARN] Could not enable eval subsampling conv chunking: {e}")

        if hasattr(self.model, "eval"):
            self.model.eval()
        if hasattr(self.model, "to"):
            self.model.to(device)

    def _load_model(self, model_ref: str, enable_adapter_support: bool = False):
        import nemo.collections.asr as nemo_asr

        model_path = Path(model_ref)

        if enable_adapter_support:
            from nemo.core import adapter_mixins
            from omegaconf import open_dict

            if model_path.exists():
                cfg = nemo_asr.models.ASRModel.restore_from(
                    str(model_path),
                    return_config=True,
                )
            else:
                cfg = nemo_asr.models.ASRModel.from_pretrained(
                    model_name=model_ref,
                    return_config=True,
                )

            with open_dict(cfg):
                adapter_metadata = adapter_mixins.get_registered_adapter(cfg.encoder._target_)
                if adapter_metadata is None:
                    raise RuntimeError(
                        f"No registered adapter-compatible encoder found for {cfg.encoder._target_}"
                    )
                cfg.encoder._target_ = adapter_metadata.adapter_class_path

            print(f"[INFO] Updated encoder target for adapter support: {cfg.encoder._target_}")

            if model_path.exists():
                print(f"[INFO] Loading local NeMo checkpoint with adapter-compatible encoder: {model_path}")
                return nemo_asr.models.ASRModel.restore_from(
                    str(model_path),
                    override_config_path=cfg,
                )
            else:
                print(f"[INFO] Loading pretrained NeMo model with adapter-compatible encoder: {model_ref}")
                return nemo_asr.models.ASRModel.from_pretrained(
                    model_name=model_ref,
                    override_config_path=cfg,
                )

        if model_path.exists():
            print(f"[INFO] Loading local NeMo checkpoint: {model_path}")
            return nemo_asr.models.ASRModel.restore_from(str(model_path))

        print(f"[INFO] Loading pretrained NeMo model: {model_ref}")
        return nemo_asr.models.ASRModel.from_pretrained(model_name=model_ref)

    def _patch_rnnt_decoding(self):
        try:
            from omegaconf import open_dict

            dec_cfg = self.model.cfg.get("rnnt_decoding", None)
            if dec_cfg is not None:
                with open_dict(dec_cfg):
                    dec_cfg.strategy = "greedy_batch"

                    if hasattr(dec_cfg, "greedy") and dec_cfg.greedy is not None:
                        if hasattr(dec_cfg.greedy, "use_cuda_graph_decoder"):
                            dec_cfg.greedy.use_cuda_graph_decoder = False
                        if hasattr(dec_cfg.greedy, "allow_cuda_graphs"):
                            dec_cfg.greedy.allow_cuda_graphs = False
                        if hasattr(dec_cfg.greedy, "cuda_graphs"):
                            dec_cfg.greedy.cuda_graphs = False

                    if hasattr(dec_cfg, "greedy_batch") and dec_cfg.greedy_batch is not None:
                        if hasattr(dec_cfg.greedy_batch, "use_cuda_graph_decoder"):
                            dec_cfg.greedy_batch.use_cuda_graph_decoder = False
                        if hasattr(dec_cfg.greedy_batch, "allow_cuda_graphs"):
                            dec_cfg.greedy_batch.allow_cuda_graphs = False
                        if hasattr(dec_cfg.greedy_batch, "cuda_graphs"):
                            dec_cfg.greedy_batch.cuda_graphs = False

                self.model.change_decoding_strategy(dec_cfg)
                print("[INFO] Updated rnnt_decoding config to disable CUDA graphs.")

            if hasattr(self.model, "decoding") and self.model.decoding is not None:
                decoding_obj = self.model.decoding.decoding

                if hasattr(decoding_obj, "use_cuda_graph_decoder"):
                    decoding_obj.use_cuda_graph_decoder = False

                if hasattr(decoding_obj, "disable_cuda_graphs"):
                    decoding_obj.disable_cuda_graphs()
                    print("[INFO] Called disable_cuda_graphs() on decoding object.")

                if hasattr(decoding_obj, "decoding_computer"):
                    computer = decoding_obj.decoding_computer
                    if hasattr(computer, "use_cuda_graphs"):
                        computer.use_cuda_graphs = False
                    if hasattr(computer, "_use_cuda_graphs"):
                        computer._use_cuda_graphs = False
                    if hasattr(computer, "_graphs"):
                        computer._graphs = None
                    if hasattr(computer, "allow_cuda_graphs"):
                        computer.allow_cuda_graphs = False
                    print("[INFO] Disabled CUDA graphs on underlying decoding_computer.")

        except Exception as e:
            print(f"[WARN] Failed to completely patch RNNT decoding config: {e}")

    def _download_hf_file_if_exists(self, repo_id: str, filename: str) -> Optional[str]:
        try:
            return hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                repo_type="model",
            )
        except Exception:
            return None

    def _load_peft_metadata(self, repo_id: str) -> Dict[str, Any]:
        path = self._download_hf_file_if_exists(repo_id, "peft_metadata.json")
        if path is None:
            return {}
        try:
            return json.loads(Path(path).read_text())
        except Exception:
            return {}

    def _load_adapter_from_hf(self, repo_id: str, filename: str):
        enable_nemo_safe_unpickling()

        adapter_path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            repo_type="model",
        )
        print(f"[INFO] Downloaded adapter from HF: {repo_id}/{filename} -> {adapter_path}")

        meta = self._load_peft_metadata(repo_id)
        if meta:
            print(f"[INFO] Loaded PEFT metadata: {meta}")

        if not hasattr(self.model, "load_adapters"):
            raise RuntimeError(
                "This NeMo model does not expose load_adapters(); cannot load saved adapter bundle."
            )

        orig_torch_load = torch.load

        def patched_torch_load(*args, **kwargs):
            # Since this is your own trusted adapter checkpoint, force legacy behavior.
            kwargs.setdefault("weights_only", False)
            return orig_torch_load(*args, **kwargs)

        try:
            with patch("torch.load", patched_torch_load):
                self.model.load_adapters(adapter_path)

            self.adapter_loaded = True
            print(f"[INFO] Loaded adapter bundle via model.load_adapters(): {adapter_path}")

        except Exception as e:
            raise RuntimeError(
                f"Official NeMo adapter load failed for {repo_id}/{filename}: "
                f"{type(e).__name__}: {e}"
            )

        try:
            if hasattr(self.model, "get_enabled_adapters"):
                print(f"[INFO] Enabled adapters after load: {self.model.get_enabled_adapters()}")
        except Exception:
            pass

    def transcribe(self, wav_path: str, max_length: int, num_beams: int, temperature: float) -> str:
        # Keep baseline behavior the same: use NeMo transcribe directly.
        # We try a couple call signatures for robustness across NeMo versions.
        try:
            out = self.model.transcribe([wav_path], batch_size=1)
        except TypeError:
            try:
                out = self.model.transcribe([wav_path])
            except TypeError:
                out = self.model.transcribe(audio=[wav_path], batch_size=1)

        if isinstance(out, list) and len(out) > 0:
            return normalize_transcribe_output(out[0])
        return normalize_transcribe_output(out)

class OmniRunner:
    def __init__(self, model_card: str, lang: str):
        from omnilingual_asr.models.inference.pipeline import ASRInferencePipeline

        self.pipeline = ASRInferencePipeline(model_card=model_card)
        self.lang = lang
        self.target_sr = 16000
        self.chunk_length_s = 35.0
        self.chunk_overlap_s = 5.0

    @staticmethod
    def _normalize_ws(s: str) -> str:
        return re.sub(r"\s+", " ", s).strip()

    @staticmethod
    def _longest_suffix_prefix_word_overlap(left: str, right: str, max_words: int = 30) -> int:
        left_words = OmniRunner._normalize_ws(left).split()
        right_words = OmniRunner._normalize_ws(right).split()
        max_k = min(len(left_words), len(right_words), max_words)
        for k in range(max_k, 0, -1):
            if left_words[-k:] == right_words[:k]:
                return k
        return 0

    def _stitch_texts(self, texts: List[str]) -> str:
        merged = ""
        for t in texts:
            t = self._normalize_ws(normalize_transcribe_output(t))
            if not t:
                continue
            if not merged:
                merged = t
                continue
            k = self._longest_suffix_prefix_word_overlap(merged, t)
            if k > 0:
                t_words = t.split()
                merged = merged + " " + " ".join(t_words[k:])
            else:
                merged = merged + " " + t
            merged = self._normalize_ws(merged)
        return merged

    def _transcribe_chunk(self, wav_path: str) -> str:
        out = self.pipeline.transcribe([wav_path], lang=[self.lang], batch_size=1)
        if isinstance(out, list) and len(out) > 0:
            return normalize_transcribe_output(out[0])
        return normalize_transcribe_output(out)

    def transcribe(self, wav_path: str, max_length: int, num_beams: int, temperature: float) -> str:
        y, sr = load_wav_mono(wav_path)
        y, sr = resample_if_needed(y, sr, self.target_sr)

        duration_s = len(y) / float(sr) if sr else 0.0
        if duration_s <= self.chunk_length_s:
            return self._transcribe_chunk(wav_path)

        chunk_size = int(self.chunk_length_s * sr)
        overlap = int(self.chunk_overlap_s * sr)
        step = chunk_size - overlap
        if step <= 0:
            raise ValueError("chunk_length_s must be greater than chunk_overlap_s")

        chunk_texts = []
        start = 0
        n = len(y)

        while start < n:
            end = min(start + chunk_size, n)
            y_chunk = y[start:end]
            if len(y_chunk) < int(0.5 * sr):
                break

            tf = tempfile.NamedTemporaryFile(prefix="omni_chunk_", suffix=".wav", delete=False)
            tf.close()
            sf.write(tf.name, y_chunk, sr)

            try:
                txt = self._transcribe_chunk(tf.name)
                chunk_texts.append(txt)
            finally:
                try:
                    os.remove(tf.name)
                except Exception:
                    pass

            if end >= n:
                break
            start += step

        return self._stitch_texts(chunk_texts)


def build_runner(spec: Dict[str, Any], device: str):
    family = spec["family"]

    if family == "whisper":
        return WhisperRunner(model_id=spec["model_id"], device=device)

    if family == "parakeet":
        return ParakeetRunner(spec=spec, device=device)

    if family == "omnilingual":
        return OmniRunner(
            model_card=spec["model_card"],
            lang=spec.get("lang", "eng_Latn"),
        )

    raise ValueError(f"Unsupported family: {family}")


def run_one(
    spec: Dict[str, Any],
    manifest_path: Path,
    out_dir: Path,
    max_length: int,
    num_beams: int,
    temperature: float,
    device: str,
):
    rows = load_manifest(manifest_path)
    runner = build_runner(spec, device=device)

    model_label = spec["label"]
    output_rows = []
    missing_audio = 0

    for ex in rows:
        wav_path = ex["audio"]
        if not Path(wav_path).exists():
            missing_audio += 1
            continue

        ref_text = normalize_text(ex["text"])
        pred_text_raw = runner.transcribe(
            wav_path=wav_path,
            max_length=max_length,
            num_beams=num_beams,
            temperature=temperature,
        )
        pred_text = normalize_text(pred_text_raw)

        measures = process_words(ref_text, pred_text)
        N = int(measures.hits + measures.substitutions + measures.deletions)

        output_rows.append(
            {
                "model_label": model_label,
                "family": spec["family"],
                "split": manifest_path.stem,
                "speaker": ex.get("speaker", ""),
                "cohort": ex.get("cohort", "unknown"),
                "audio_path": wav_path,
                "reference_text": ref_text,
                "prediction_text": pred_text,
                "substitutions": int(measures.substitutions),
                "deletions": int(measures.deletions),
                "insertions": int(measures.insertions),
                "hits": int(measures.hits),
                "reference_words": N,
                "utterance_wer": float(measures.wer),
            }
        )

    overall = aggregate_rows(output_rows)
    by_cohort = {}
    for cohort in sorted(set(r["cohort"] for r in output_rows)):
        cohort_rows = [r for r in output_rows if r["cohort"] == cohort]
        by_cohort[cohort] = aggregate_rows(cohort_rows)

    stem = f"{safe_name(model_label)}__{manifest_path.stem}"
    pred_path = out_dir / f"predictions__{stem}.jsonl"
    summary_path = out_dir / f"summary__{stem}.json"

    with pred_path.open("w", encoding="utf-8") as f:
        for r in output_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    summary = {
        "model_label": model_label,
        "family": spec["family"],
        "split": manifest_path.stem,
        "n_scored": len(output_rows),
        "n_missing_audio": missing_audio,
        "overall": overall,
        "by_cohort": by_cohort,
    }
    summary_path.write_text(json.dumps(summary, indent=2))

    print(f"Wrote {pred_path}")
    print(f"Wrote {summary_path}")
    print(json.dumps(summary, indent=2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--specs", type=Path, required=True, help="JSON file with model specs")
    ap.add_argument("--manifest", type=Path, default=Path("pipeline/manifests/plain/test.jsonl"))
    ap.add_argument("--out_dir", type=Path, default=Path("pipeline/eval_outputs"))
    ap.add_argument("--max_length", type=int, default=256)
    ap.add_argument("--num_beams", type=int, default=1)
    ap.add_argument("--temperature", type=float, default=0.0)
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    specs = json.loads(args.specs.read_text())
    device = "cuda" if torch.cuda.is_available() else "cpu"

    for spec in specs:
        run_one(
            spec=spec,
            manifest_path=args.manifest,
            out_dir=args.out_dir,
            max_length=args.max_length,
            num_beams=args.num_beams,
            temperature=args.temperature,
            device=device,
        )


if __name__ == "__main__":
    main()
    