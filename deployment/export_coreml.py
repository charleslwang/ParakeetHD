#!/usr/bin/env python3
"""Export ParakeetHD encoder and decoder/joint subnetworks to Core ML.

This script reconstructs the full model from the NVIDIA base checkpoint and
the released NeMo encoder adapter. It then traces deployment-only wrappers,
converts them to ML Program packages, and writes the tokenizer and decoding
metadata needed by a native client.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import shutil
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import torch
from omegaconf import OmegaConf

try:
    from .model_loader import load_parakeet_hd
except ImportError:  # Allows `python deployment/export_coreml.py`.
    from model_loader import load_parakeet_hd


class AudioEncoderForCoreML(torch.nn.Module):
    """Combine NeMo waveform preprocessing and the acoustic encoder."""

    def __init__(self, preprocessor: torch.nn.Module, encoder: torch.nn.Module):
        super().__init__()
        self.preprocessor = preprocessor
        self.encoder = encoder

    def forward(self, audio_signal: torch.Tensor, length: torch.Tensor):
        features, feature_length = self.preprocessor(input_signal=audio_signal, length=length)
        encoded = self.encoder(audio_signal=features, length=feature_length)
        if isinstance(encoded, (tuple, list)):
            return encoded[0], encoded[1]
        return encoded, feature_length


class AcousticEncoderForCoreML(torch.nn.Module):
    """Export only the acoustic encoder when preprocessing lives in Swift."""

    def __init__(self, encoder: torch.nn.Module):
        super().__init__()
        self.encoder = encoder

    def forward(self, audio_features: torch.Tensor, feature_length: torch.Tensor):
        encoded = self.encoder(audio_signal=audio_features, length=feature_length)
        if isinstance(encoded, (tuple, list)):
            return encoded[0], encoded[1]
        return encoded, feature_length


class DecoderJointForCoreML(torch.nn.Module):
    """Expose one prediction-network + joint-network decoding step."""

    def __init__(self, decoder: torch.nn.Module, joint: torch.nn.Module):
        super().__init__()
        self.decoder = decoder
        self.joint = joint

    def forward(
        self,
        encoder_outputs: torch.Tensor,
        targets: torch.Tensor,
        target_length: torch.Tensor,
        input_states_1: torch.Tensor,
        input_states_2: torch.Tensor,
    ):
        decoder_outputs = self.decoder(
            targets=targets,
            target_length=target_length,
            states=(input_states_1, input_states_2),
        )
        decoder_output = decoder_outputs[0]
        decoder_length = decoder_outputs[1]
        output_state_1, output_state_2 = decoder_outputs[2]
        joint_output = self.joint(
            encoder_outputs=encoder_outputs,
            decoder_outputs=decoder_output,
        )
        if isinstance(joint_output, (tuple, list)):
            return (*joint_output, decoder_length, output_state_1, output_state_2)
        return joint_output, decoder_length, output_state_1, output_state_2


def _flatten_tensors(value: Any) -> list[torch.Tensor]:
    if isinstance(value, torch.Tensor):
        return [value]
    if isinstance(value, (tuple, list)):
        result: list[torch.Tensor] = []
        for item in value:
            result.extend(_flatten_tensors(item))
        return result
    raise TypeError(f"Export wrapper returned unsupported output type: {type(value).__name__}")


def _as_numpy(tensor: torch.Tensor) -> np.ndarray:
    return tensor.detach().cpu().numpy()


def _shape(tensor: torch.Tensor) -> list[int]:
    return [int(value) for value in tensor.shape]


def _integer_list(value: Any) -> list[int]:
    if value is None:
        return []
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().flatten().tolist()
    if isinstance(value, np.ndarray):
        value = value.flatten().tolist()
    return [int(item) for item in value]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _public_source(value: str) -> str:
    return "local" if Path(value).expanduser().exists() else value


def _prepare_modules_for_export(model) -> None:
    preprocessor = getattr(model, "preprocessor", None)
    featurizer = getattr(preprocessor, "featurizer", None)
    if featurizer is not None:
        if hasattr(featurizer, "dither"):
            featurizer.dither = 0.0
        if hasattr(featurizer, "pad_to"):
            featurizer.pad_to = 0

    for name in ["encoder", "decoder", "joint"]:
        module = getattr(model, name, None)
        prepare = getattr(module, "_prepare_for_export", None)
        if prepare is None:
            continue
        try:
            prepare()
        except Exception as exc:
            print(f"[WARN] {name}._prepare_for_export() skipped: {type(exc).__name__}: {exc}")


def _preprocessor_metadata(model) -> dict[str, Any]:
    metadata = _config_value(model, ["preprocessor"], default={})
    if not isinstance(metadata, dict):
        metadata = {}
    metadata = dict(metadata)
    featurizer = getattr(getattr(model, "preprocessor", None), "featurizer", None)
    for key in ["dither", "pad_to"]:
        if hasattr(featurizer, key):
            metadata[key] = getattr(featurizer, key)
    return metadata


def _config_value(model, paths: Iterable[str], default: Any = None) -> Any:
    for path in paths:
        value = OmegaConf.select(model.cfg, path, default=None)
        if value is not None:
            return OmegaConf.to_container(value, resolve=True) if OmegaConf.is_config(value) else value
    return default


def _runtime_value(model, names: Iterable[str], default: Any = None) -> Any:
    owners = [model, getattr(model, "decoding", None)]
    decoding = getattr(model, "decoding", None)
    owners.append(getattr(decoding, "decoding", None))
    owners.extend([getattr(model, "decoder", None), getattr(model, "joint", None)])
    for owner in owners:
        if owner is None:
            continue
        for name in names:
            value = getattr(owner, name, None)
            if value is not None:
                return value
    return default


def _extract_tokens(model) -> list[str]:
    tokenizer = getattr(model, "tokenizer", None)
    if tokenizer is None:
        vocabulary = getattr(getattr(model, "decoder", None), "vocabulary", None)
        if vocabulary:
            return [str(token) for token in vocabulary]
        raise RuntimeError("Could not locate the NeMo tokenizer or decoder vocabulary.")

    vocab_size = getattr(tokenizer, "vocab_size", None)
    if callable(vocab_size):
        vocab_size = vocab_size()
    if vocab_size is None:
        vocab_size = _runtime_value(model, ["vocab_size"])
    if vocab_size is None:
        raise RuntimeError("Could not determine tokenizer vocabulary size.")

    token_ids = list(range(int(vocab_size)))
    if hasattr(tokenizer, "ids_to_tokens"):
        tokens = tokenizer.ids_to_tokens(token_ids)
    elif hasattr(tokenizer, "tokenizer") and hasattr(tokenizer.tokenizer, "id_to_piece"):
        tokens = [tokenizer.tokenizer.id_to_piece(token_id) for token_id in token_ids]
    else:
        raise RuntimeError("Tokenizer does not expose ids_to_tokens() or id_to_piece().")

    result = [str(token) for token in tokens]
    if any("\n" in token or "\r" in token for token in result):
        raise RuntimeError("Tokenizer contains a line break and cannot be represented by tokens.txt.")
    return result


def _extract_decoding_metadata(model, tokens: Sequence[str]) -> dict[str, Any]:
    blank_id = _runtime_value(model, ["blank_id", "blank_idx"])
    if isinstance(blank_id, torch.Tensor):
        blank_id = blank_id.item()
    if blank_id is None:
        blank_id = len(tokens)

    durations = _runtime_value(model, ["durations", "tdt_durations"])
    if durations is None:
        durations = _config_value(
            model,
            [
                "model_defaults.tdt_durations",
                "tdt_durations",
                "rnnt_decoding.tdt_durations",
                "decoding.tdt_durations",
            ],
            default=[],
        )
    big_blank_durations = _runtime_value(model, ["big_blank_durations"])
    if big_blank_durations is None:
        big_blank_durations = _config_value(
            model,
            ["model_defaults.big_blank_durations", "rnnt_decoding.big_blank_durations"],
            default=[],
        )

    duration_values = _integer_list(durations)
    if not duration_values:
        raise RuntimeError("Could not locate TDT duration metadata in the NeMo model.")
    if int(blank_id) < 0:
        raise RuntimeError(f"Invalid blank token ID: {blank_id}")

    return {
        "type": "tdt",
        "blank_id": int(blank_id),
        "vocab_size": len(tokens),
        "token_class_count": len(tokens) + 1,
        "durations": duration_values,
        "duration_class_count": len(duration_values),
        "big_blank_durations": _integer_list(big_blank_durations),
    }


def _record_joint_output_layout(decoding: dict[str, Any], outputs: Sequence[torch.Tensor]) -> None:
    joint_size = int(outputs[0].shape[-1])
    token_count = int(decoding["token_class_count"])
    duration_count = int(decoding["duration_class_count"])
    decoding["joint_output_size"] = joint_size
    if len(outputs) == 5:
        duration_size = int(outputs[1].shape[-1])
        decoding["duration_output_size"] = duration_size
        if joint_size == token_count and duration_size == duration_count:
            decoding["joint_output_layout"] = "separate_token_and_duration_tensors"
        else:
            decoding["joint_output_layout"] = "unknown"
    elif joint_size == token_count + duration_count:
        decoding["joint_output_layout"] = "token_scores_then_duration_scores"
    elif joint_size == token_count:
        decoding["joint_output_layout"] = "token_scores_only"
    else:
        decoding["joint_output_layout"] = "unknown"


def _trace_module(
    module: torch.nn.Module,
    inputs: tuple[torch.Tensor, ...],
    output_path: Path,
) -> tuple[torch.jit.ScriptModule, list[torch.Tensor]]:
    module.eval()
    with torch.inference_mode():
        eager_outputs = _flatten_tensors(module(*inputs))
        traced = torch.jit.trace(module, inputs, strict=False, check_trace=False)
    traced.save(str(output_path))
    return traced, eager_outputs


def _coreml_feature_names(ml_model, kind: str) -> tuple[list[str], list[str]]:
    spec = ml_model.get_spec()
    inputs = [feature.name for feature in spec.description.input]
    outputs = [feature.name for feature in spec.description.output]
    if not inputs or not outputs:
        raise RuntimeError(f"Converted {kind} model has an empty Core ML signature.")
    return inputs, outputs


def _semantic_decoder_outputs(count: int) -> list[str]:
    if count == 4:
        return ["joint_logits", "prednet_length", "output_states_1", "output_states_2"]
    if count == 5:
        return [
            "token_logits",
            "duration_logits",
            "prednet_length",
            "output_states_1",
            "output_states_2",
        ]
    return [f"decoder_output_{index}" for index in range(count)]


def _write_fixture(
    fixture_dir: Path,
    kind: str,
    input_names: Sequence[str],
    input_tensors: Sequence[torch.Tensor],
    output_names: Sequence[str],
    output_tensors: Sequence[torch.Tensor],
) -> dict[str, Any]:
    fixture_dir.mkdir(parents=True, exist_ok=True)
    input_path = fixture_dir / f"{kind}_inputs.npz"
    output_path = fixture_dir / f"{kind}_expected_outputs.npz"
    np.savez(input_path, **{name: _as_numpy(value) for name, value in zip(input_names, input_tensors)})
    np.savez(
        output_path,
        **{f"output_{index}": _as_numpy(value) for index, value in enumerate(output_tensors)},
    )
    return {
        "inputs": str(input_path.name),
        "expected_outputs": str(output_path.name),
        "coreml_output_names": list(output_names),
    }


def _convert_encoder(ct, traced, example_inputs, args):
    if args.encoder_input == "waveform":
        signal_name = "audio_signal"
        signal_shape = (
            1,
            ct.RangeDim(
                lower_bound=args.min_audio_samples,
                upper_bound=args.max_audio_samples,
                default=args.example_audio_samples,
            ),
        )
        length_name = "length"
    else:
        signal_name = "audio_features"
        signal_shape = (
            int(example_inputs[0].shape[0]),
            int(example_inputs[0].shape[1]),
            ct.RangeDim(
                lower_bound=args.min_feature_frames,
                upper_bound=args.max_feature_frames,
                default=int(example_inputs[0].shape[2]),
            ),
        )
        length_name = "feature_length"
    return ct.convert(
        traced,
        source="pytorch",
        convert_to="mlprogram",
        minimum_deployment_target=getattr(ct.target, f"iOS{args.ios_target}"),
        compute_precision=ct.precision.FLOAT16 if args.precision == "float16" else ct.precision.FLOAT32,
        inputs=[
            ct.TensorType(name=signal_name, shape=signal_shape, dtype=np.float32),
            ct.TensorType(name=length_name, shape=_shape(example_inputs[1]), dtype=np.int32),
        ],
    )


def _convert_decoder(ct, traced, example_inputs, args):
    names = ["encoder_outputs", "targets", "target_length", "input_states_1", "input_states_2"]
    dtypes = [np.float32, np.int32, np.int32, np.float32, np.float32]
    inputs = [
        ct.TensorType(name=name, shape=_shape(tensor), dtype=dtype)
        for name, tensor, dtype in zip(names, example_inputs, dtypes)
    ]
    return ct.convert(
        traced,
        source="pytorch",
        convert_to="mlprogram",
        minimum_deployment_target=getattr(ct.target, f"iOS{args.ios_target}"),
        compute_precision=ct.precision.FLOAT16 if args.precision == "float16" else ct.precision.FLOAT32,
        inputs=inputs,
    )


def _describe_features(names: Sequence[str], semantics: Sequence[str], tensors: Sequence[torch.Tensor]):
    return [
        {"name": name, "semantic": semantic, "example_shape": _shape(tensor)}
        for name, semantic, tensor in zip(names, semantics, tensors)
    ]


def _set_model_metadata(ml_model, name: str, base_model: str, adapter_source: str) -> None:
    ml_model.author = "ParakeetHD contributors"
    ml_model.license = "Apache-2.0 adapter; base model subject to its own license"
    ml_model.short_description = f"{name} subnetwork for ParakeetHD"
    ml_model.user_defined_metadata["parakeethd.base_model"] = base_model
    ml_model.user_defined_metadata["parakeethd.adapter_source"] = adapter_source


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model", default="nvidia/parakeet-tdt-0.6b-v2")
    parser.add_argument("--adapter-source", default="charleslwang/parakeet-tdt-0.6b-HD")
    parser.add_argument("--adapter-filename", default="best_adapter.pt")
    parser.add_argument("--output-dir", type=Path, default=Path("coreml-export"))
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--ios-target", type=int, choices=[16, 17, 18], default=17)
    parser.add_argument("--precision", choices=["float16", "float32"], default="float32")
    parser.add_argument(
        "--encoder-input",
        choices=["waveform", "features"],
        default="waveform",
        help="Use features if Core ML cannot convert NeMo's waveform/STFT preprocessor.",
    )
    parser.add_argument("--example-audio-seconds", type=float, default=1.0)
    parser.add_argument("--min-audio-seconds", type=float, default=0.25)
    parser.add_argument("--max-audio-seconds", type=float, default=30.0)
    parser.add_argument("--keep-intermediates", action="store_true")
    parser.add_argument("--clean", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("--device cuda was requested, but CUDA is not available.")

    try:
        import coremltools as ct
    except ImportError as exc:
        raise SystemExit("coremltools is required; install requirements-coreml.txt") from exc

    if not 0 < args.min_audio_seconds <= args.example_audio_seconds <= args.max_audio_seconds:
        raise SystemExit("Audio durations must satisfy 0 < min <= example <= max.")

    output_dir = args.output_dir.expanduser().resolve()
    if args.clean and output_dir.exists():
        shutil.rmtree(output_dir)
    models_dir = output_dir / "models"
    intermediates_dir = output_dir / "intermediates"
    fixtures_dir = output_dir / "validation"
    models_dir.mkdir(parents=True, exist_ok=True)
    intermediates_dir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Loading {args.base_model} with adapter {args.adapter_source}/{args.adapter_filename}")
    model, adapter_path = load_parakeet_hd(
        base_model=args.base_model,
        adapter_source=args.adapter_source,
        adapter_filename=args.adapter_filename,
        device=args.device,
    )
    _prepare_modules_for_export(model)
    sample_rate = int(_config_value(model, ["preprocessor.sample_rate", "sample_rate"], 16000))
    preprocessor_metadata = _preprocessor_metadata(model)
    args.example_audio_samples = int(round(args.example_audio_seconds * sample_rate))
    args.min_audio_samples = int(round(args.min_audio_seconds * sample_rate))
    args.max_audio_samples = int(round(args.max_audio_seconds * sample_rate))
    tokens = _extract_tokens(model)
    decoding = _extract_decoding_metadata(model, tokens)

    device = torch.device(args.device)
    torch.manual_seed(0)
    waveform = torch.randn(1, args.example_audio_samples, dtype=torch.float32, device=device) * 0.01
    waveform_length = torch.tensor([args.example_audio_samples], dtype=torch.int32, device=device)
    if args.encoder_input == "waveform":
        encoder_inputs = (waveform, waveform_length)
        encoder_input_semantics = ["audio_signal", "length"]
        encoder_wrapper = AudioEncoderForCoreML(model.preprocessor, model.encoder).eval()
    else:
        with torch.inference_mode():
            audio_features, feature_length = model.preprocessor(
                input_signal=waveform,
                length=waveform_length,
            )
            min_features, _ = model.preprocessor(
                input_signal=torch.zeros(1, args.min_audio_samples, dtype=torch.float32, device=device),
                length=torch.tensor([args.min_audio_samples], dtype=torch.int32, device=device),
            )
            max_features, _ = model.preprocessor(
                input_signal=torch.zeros(1, args.max_audio_samples, dtype=torch.float32, device=device),
                length=torch.tensor([args.max_audio_samples], dtype=torch.int32, device=device),
            )
        args.min_feature_frames = int(min_features.shape[2])
        args.max_feature_frames = int(max_features.shape[2])
        encoder_inputs = (audio_features, feature_length.to(dtype=torch.int32))
        encoder_input_semantics = ["audio_features", "feature_length"]
        encoder_wrapper = AcousticEncoderForCoreML(model.encoder).eval()
    encoder_trace_path = intermediates_dir / "ParakeetEncoder.pt"
    if args.encoder_input == "waveform":
        print("[INFO] Tracing waveform preprocessor + encoder")
    else:
        print("[INFO] Tracing acoustic encoder with precomputed features")
    encoder_trace, encoder_outputs = _trace_module(encoder_wrapper, encoder_inputs, encoder_trace_path)

    blank_id = int(decoding["blank_id"])
    target = torch.tensor([[blank_id]], dtype=torch.int32, device=device)
    target_length = torch.ones(1, dtype=torch.int32, device=device)
    with torch.inference_mode():
        decoder_example = model.decoder.input_example(max_batch=1, max_dim=1)
        initial_states = decoder_example[-1]
        initial_states = tuple(state.to(device=device, dtype=torch.float32) for state in initial_states)
    if any(not torch.allclose(state, torch.zeros_like(state)) for state in initial_states):
        raise RuntimeError(
            "Decoder input_example() returned non-zero initial states. "
            "Update the bundle format and Swift runtime to load explicit initial states."
        )
    encoder_channels = int(encoder_outputs[0].shape[1])
    decoder_inputs = (
        torch.randn(1, encoder_channels, 1, dtype=torch.float32, device=device) * 0.01,
        target,
        target_length,
        initial_states[0],
        initial_states[1],
    )
    decoder_wrapper = DecoderJointForCoreML(model.decoder, model.joint).eval()
    decoder_trace_path = intermediates_dir / "ParakeetDecoderJoint.pt"
    print("[INFO] Tracing decoder + joint network")
    decoder_trace, decoder_outputs = _trace_module(decoder_wrapper, decoder_inputs, decoder_trace_path)
    _record_joint_output_layout(decoding, decoder_outputs)

    # Core ML conversion is CPU-based. Reloading maps traced constants off CUDA.
    encoder_trace = torch.jit.load(str(encoder_trace_path), map_location="cpu").eval()
    decoder_trace = torch.jit.load(str(decoder_trace_path), map_location="cpu").eval()
    encoder_inputs_cpu = tuple(tensor.cpu() for tensor in encoder_inputs)
    decoder_inputs_cpu = tuple(tensor.cpu() for tensor in decoder_inputs)

    print("[INFO] Converting encoder to Core ML")
    try:
        encoder_ml = _convert_encoder(ct, encoder_trace, encoder_inputs_cpu, args)
    except Exception as exc:
        if args.encoder_input == "waveform":
            message = (
                "Core ML conversion of the waveform preprocessor + encoder failed. "
                "The likely cause is an unsupported audio/STFT operation. Re-run with "
                "--encoder-input features and implement the recorded preprocessor configuration in Swift. "
                "The failed run's TorchScript intermediates were preserved."
            )
        else:
            message = "Core ML conversion of the acoustic encoder failed; TorchScript intermediates were preserved."
        raise RuntimeError(message) from exc
    public_base_model = _public_source(args.base_model)
    public_adapter_source = _public_source(args.adapter_source)
    _set_model_metadata(encoder_ml, "Encoder", public_base_model, public_adapter_source)
    encoder_path = models_dir / "ParakeetEncoder.mlpackage"
    encoder_ml.save(str(encoder_path))

    print("[INFO] Converting decoder/joint to Core ML")
    decoder_ml = _convert_decoder(ct, decoder_trace, decoder_inputs_cpu, args)
    _set_model_metadata(decoder_ml, "DecoderJoint", public_base_model, public_adapter_source)
    decoder_path = models_dir / "ParakeetDecoderJoint.mlpackage"
    decoder_ml.save(str(decoder_path))

    encoder_input_names, encoder_output_names = _coreml_feature_names(encoder_ml, "encoder")
    decoder_input_names, decoder_output_names = _coreml_feature_names(decoder_ml, "decoder")
    if len(encoder_output_names) != len(encoder_outputs):
        raise RuntimeError("Core ML encoder output count differs from the traced model.")
    if len(decoder_output_names) != len(decoder_outputs):
        raise RuntimeError("Core ML decoder output count differs from the traced model.")

    tokens_path = output_dir / "tokens.txt"
    tokens_path.write_text("\n".join(tokens) + "\n", encoding="utf-8")

    encoder_fixture = _write_fixture(
        fixtures_dir,
        "encoder",
        encoder_input_names,
        encoder_inputs_cpu,
        encoder_output_names,
        encoder_outputs,
    )
    decoder_fixture = _write_fixture(
        fixtures_dir,
        "decoder_joint",
        decoder_input_names,
        decoder_inputs_cpu,
        decoder_output_names,
        decoder_outputs,
    )
    validation_manifest = {"encoder": encoder_fixture, "decoder_joint": decoder_fixture}
    (fixtures_dir / "manifest.json").write_text(
        json.dumps(validation_manifest, indent=2) + "\n",
        encoding="utf-8",
    )

    bundle = {
        "format": "parakeethd-coreml-bundle",
        "format_version": 1,
        "base_model": public_base_model,
        "adapter": {"source": public_adapter_source, "filename": args.adapter_filename},
        "audio": {
            "sample_rate": sample_rate,
            "channels": 1,
            "dtype": "float32",
            "encoder_input": args.encoder_input,
            "preprocessor": preprocessor_metadata,
        },
        "tokenizer": {"type": "bpe", "tokens_path": "tokens.txt", "tokens_count": len(tokens)},
        "decoding": decoding,
        "decoder_initial_state": {"type": "zeros"},
        "coreml": {
            "minimum_ios": args.ios_target,
            "compute_precision": args.precision,
            "encoder": {
                "path": "models/ParakeetEncoder.mlpackage",
                "inputs": _describe_features(
                    encoder_input_names,
                    encoder_input_semantics,
                    encoder_inputs_cpu,
                ),
                "outputs": _describe_features(
                    encoder_output_names,
                    ["encoder_outputs", "encoded_length"],
                    encoder_outputs,
                ),
            },
            "decoder_joint": {
                "path": "models/ParakeetDecoderJoint.mlpackage",
                "inputs": _describe_features(
                    decoder_input_names,
                    ["encoder_outputs", "targets", "target_length", "input_states_1", "input_states_2"],
                    decoder_inputs_cpu,
                ),
                "outputs": _describe_features(
                    decoder_output_names,
                    _semantic_decoder_outputs(len(decoder_outputs)),
                    decoder_outputs,
                ),
            },
        },
        "export": {
            "adapter_sha256": _sha256(adapter_path),
            "torch_version": torch.__version__,
            "coremltools_version": getattr(ct, "__version__", "unknown"),
            "host": platform.platform(),
        },
    }
    (output_dir / "bundle.json").write_text(json.dumps(bundle, indent=2) + "\n", encoding="utf-8")

    if not args.keep_intermediates:
        shutil.rmtree(intermediates_dir)

    print(f"[OK] Core ML bundle written to {output_dir}")
    print(f"[NEXT] Validate on macOS: python deployment/validate_coreml.py --bundle {output_dir / 'bundle.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
