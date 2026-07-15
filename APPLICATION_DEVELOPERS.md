# Building Applications With ParakeetHD

This repository is intended to provide the model export pipeline and native
runtime components that application developers need to build their own
products. It is not intended to prescribe a particular app UI.

## What This Repo Provides

- Scripts to reconstruct ParakeetHD from the open Hugging Face model/adapters.
- Core ML export for the encoder and decoder/joint subnetworks.
- Tensor-level Core ML validation.
- End-to-end transcription parity checks on real WAV files.
- Bundle benchmarking utilities.
- A Swift Package runtime that app developers can embed in iOS or macOS apps.

## What Application Developers Provide

- Their own app shell, UI, and product workflow.
- Audio recording, file import, privacy prompts, and app lifecycle behavior.
- On-device validation on their target devices.
- Distribution-specific model hosting or bundling decisions.

## Build A Core ML Bundle

From the repository root:

```bash
python3.12 -m venv .venv-coreml
source .venv-coreml/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-coreml.txt
python deployment/export_coreml.py --clean
python deployment/validate_coreml.py --bundle ./coreml-export/bundle.json
```

The exported bundle contains:

```text
coreml-export/
  bundle.json
  tokens.txt
  models/
    ParakeetEncoder.mlpackage/
    ParakeetDecoderJoint.mlpackage/
  validation/
```

Apps should treat `bundle.json` as the stable entry point. Do not hard-code
Core ML feature names; they are recorded in the bundle.

## Use The Swift Runtime In An App

Add `ios/` as a local Swift Package dependency, or publish it as a package from
your fork. Then import the runtime:

```swift
import ParakeetRuntime

let runtime = try ParakeetRuntime(
    bundleJSON: Bundle.main.url(forResource: "bundle", withExtension: "json")!
)

let transcript = try runtime.transcribe(audioURL: audioFileURL)
```

For apps that download model artifacts after install, point `bundleJSON` at the
downloaded bundle directory instead of `Bundle.main`.

## Validate Before Shipping

Run the Python parity checker against real audio:

```bash
python deployment/validate_transcription.py \
  --bundle ./coreml-export/bundle.json \
  --manifest ./pipeline/manifests/plain/test.jsonl \
  --report ./coreml-export/validation/e2e_report.json
```

Run the local benchmark:

```bash
python deployment/benchmark_coreml.py \
  --bundle ./coreml-export/bundle.json \
  --report ./coreml-export/validation/benchmark_report.json
```

Then repeat application-level validation on the oldest and slowest devices you
intend to support. Measure package size, peak memory, latency, real-time
factor, thermal behavior, and transcript quality.

## Current Runtime Scope

The Swift Package currently supports waveform-input bundles, where NeMo audio
preprocessing is included in the exported Core ML encoder. If you export with
`--encoder-input features`, your app must provide exact log-mel preprocessing
parity before calling the encoder.

The runtime is deliberately UI-free. It is suitable for apps with very
different surfaces: clinical tools, research data collection, accessibility
workflows, batch transcription utilities, or internal demos.

## Audio Length

The default waveform-input export accepts mono audio from 0.25 to 30 seconds,
as recorded in `bundle.json`. Applications that need longer recordings,
continuous microphone transcription, or clinical interviews should segment
audio into supported chunks or build a streaming layer around the runtime.
Do not pass recordings longer than the exported maximum directly to the
encoder.
