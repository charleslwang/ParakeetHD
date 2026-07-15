<div align="center">

# 🦜 ParakeetHD: Huntington Disease ASR Model Suite

[![arXiv](https://img.shields.io/badge/arXiv-2603.11168-b31b1b.svg)](https://arxiv.org/abs/2603.11168)
[![Hugging Face Collection](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Models-FFD21E)](https://huggingface.co/collections/charleslwang/parakeethd)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

*Official repository for the paper **"Huntington Disease Automatic Speech Recognition with Biomarker Supervision."***

</div>

---

## 📌 Quick Links
- [Summary](#-summary)
- [Models](#-models)
- [Building applications](#-building-applications)
- [Citation](#-citation)

---

## 📖 Summary

**ParakeetHD** is a model suite designed for automatic speech recognition (ASR) on speech affected by Huntington disease (HD). In this work, we:

*   🔍 **Compare** multiple ASR model families on HD speech under a unified evaluation pipeline.
*   🛠️ **Adapt** **Parakeet-TDT** to HD speech using parameter-efficient encoder-side adapters.
*   📊 **Evaluate** performance with **WER** and detailed **substitution / deletion / insertion** analysis.
*   🩺 **Study** whether **prosodic, phonatory, and articulatory biomarkers** can be used as auxiliary supervision during adaptation.

> **Key Finding:** Our results show that **HD-specific adaptation** gives the strongest overall performance, while biomarker-aware supervision helps reveal clinically meaningful changes in error behavior.

---

## 🤖 Models

All models in the suite are hosted in our [Hugging Face Collection](https://huggingface.co/collections/charleslwang/parakeethd). 

| Model | Focus / Supervision | Link |
| :--- | :--- | :--- |
| **Parakeet-HD** | Baseline HD Adaptation | [View Model](https://huggingface.co/collections/charleslwang/parakeethd) |
| **Parakeet-HD-Prosody** | Prosodic Biomarkers | [View Model](https://huggingface.co/charleslwang/parakeet-tdt-0.6b-HD-prosody) |
| **Parakeet-HD-Phonation** | Phonatory Biomarkers | [View Model](https://huggingface.co/charleslwang/parakeet-tdt-0.6b-HD-phonation) |
| **Parakeet-HD-Articulation**| Articulatory Biomarkers | [View Model](https://huggingface.co/charleslwang/parakeet-tdt-0.6b-HD-articulation) |

---

## 🧩 Building applications

This repository is structured so developers can build their own applications
with ParakeetHD without starting from model internals. The intended contract is:

1. Export or download a Core ML bundle.
2. Treat `bundle.json` as the model/runtime entry point.
3. Embed the Swift Package in `ios/` or port its runtime logic to another
   native stack.
4. Build your own app UI, audio workflow, deployment policy, and validation.

See [APPLICATION_DEVELOPERS.md](APPLICATION_DEVELOPERS.md) for the integration
guide.

---

## 🍎 Core ML export

The Hugging Face ParakeetHD repositories contain NeMo encoder adapters rather
than complete standalone checkpoints. The deployment scripts reconstruct the
model from the NVIDIA base checkpoint and adapter before producing two Core ML
ML Program packages:

```text
coreml-export/
  bundle.json
  tokens.txt
  models/
    ParakeetEncoder.mlpackage/
    ParakeetDecoderJoint.mlpackage/
  validation/
    manifest.json
    *.npz
```

You can either build these Core ML artifacts locally with the scripts below or
download the validated prebuilt bundle from Hugging Face:

```text
https://huggingface.co/charleslwang/parakeet-tdt-0.6b-HD/tree/main/coreml
```

Native clients should treat `coreml/bundle.json` as the entry point. It points
to the two `.mlpackage` models and records the tokenizer, audio shape, sample
rate, blank token, and TDT duration metadata needed by the app runtime.

> [!IMPORTANT]
> The public repository provides export, tensor validation, experimental
> transcription parity checking, benchmark utilities, and a Swift Package
> runtime for waveform-input Core ML bundles. A native app must still add
> its own UI/audio-session integration and on-device validation. Feature-input bundles
> still require exact Swift log-mel preprocessing parity.
> The default waveform-input export accepts 0.25–30 seconds of audio; apps
> that need longer recordings should add chunking or streaming.

Create a dedicated environment on a Mac and install the deployment
dependencies:

```bash
python3.12 -m venv .venv-coreml
source .venv-coreml/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-coreml.txt
```

Export the baseline Parakeet-HD adapter:

```bash
python deployment/export_coreml.py --clean
```

The exporter defaults to:

- base model: `nvidia/parakeet-tdt-0.6b-v2`
- adapter: `charleslwang/parakeet-tdt-0.6b-HD/best_adapter.pt`
- target: iOS 17 ML Program with float32 compute precision
- waveform input: mono float32 audio at the model's configured sample rate
- flexible chunk duration: 0.25–30 seconds

Float16 export is available for smaller/faster experimental bundles, but it
should be revalidated carefully:

```bash
python deployment/export_coreml.py --precision float16 --clean
```

Local adapter files and alternate ParakeetHD variants are also supported:

```bash
python deployment/export_coreml.py \
  --adapter-source ./checkpoints/best_adapter.pt \
  --output-dir ./coreml-export \
  --clean
```

If Core ML reports an unsupported waveform preprocessing or STFT operation,
export the acoustic encoder alone:

```bash
python deployment/export_coreml.py \
  --encoder-input features \
  --output-dir ./coreml-export \
  --clean
```

In feature-input mode, `bundle.json` includes the NeMo preprocessor
configuration that must be reproduced by the Swift audio pipeline.

Validate both Core ML subnetworks against the PyTorch outputs captured during
export. Core ML prediction validation must run on macOS:

```bash
python deployment/validate_coreml.py \
  --bundle ./coreml-export/bundle.json
```

Run an experimental end-to-end transcription parity check on real WAVs. The
manifest can use either `audio` or `audio_filepath` plus `text` fields:

```bash
python deployment/validate_transcription.py \
  --bundle ./coreml-export/bundle.json \
  --manifest ./pipeline/manifests/plain/test.jsonl \
  --limit 8 \
  --report ./coreml-export/validation/e2e_report.json
```

This script is intentionally stricter than the tensor validator: it returns a
non-zero exit code if normalized NeMo and Core ML transcripts differ. It
currently supports waveform-input bundles. Feature-input bundles require an
independent preprocessing parity implementation before full transcription
validation is meaningful.

Measure local Core ML package size and prediction latency:

```bash
python deployment/benchmark_coreml.py \
  --bundle ./coreml-export/bundle.json \
  --report ./coreml-export/validation/benchmark_report.json
```

Build and try the Swift runtime CLI:

```bash
cd ios
swift build
swift run parakeet-transcribe \
  --bundle ../coreml-export/bundle.json \
  --audio /path/to/audio.wav
```

Publishing the hosted Core ML bundle is a maintainer-only step and is not part
of the public repository. Do not commit Hugging Face tokens, generated
`coreml-export/` folders, or `.mlpackage` artifacts to git.

The uploaded `bundle.json` is the native client's contract. It records the
actual Core ML feature names, example tensor shapes, tokenizer location, sample
rate, blank token, and TDT duration metadata. A Swift app does not use NeMo or
`model_config.yaml` after these artifacts have been exported.

> [!NOTE]
> Core ML conversion support depends on the exact PyTorch, NeMo, and
> `coremltools` versions. If conversion of the waveform preprocessor fails on
> an unsupported audio/STFT operation, the exporter preserves its TorchScript
> intermediates; retry with `--encoder-input features` and move preprocessing
> into Swift. Never publish a bundle that has not
> passed `validate_coreml.py`, end-to-end transcription testing, and practical
> device benchmarks. The default float32 export is primarily a parity target;
> float16 and additional compression should be evaluated before treating a
> bundle as suitable for iPhone deployment.

---

## 📝 Citation

If you use this repository or the released models, please cite our paper:

```bibtex
@article{wang2026huntington,
  title={Huntington Disease Automatic Speech Recognition with Biomarker Supervision},
  author={Wang, Charles L. and Chen, Cady and Gong, Ziwei and Hirschberg, Julia},
  journal={arXiv preprint arXiv:2603.11168},
  year={2026},
  url={https://arxiv.org/abs/2603.11168}
}
