# ParakeetHD Swift Runtime SDK

This package is the native runtime layer for exported ParakeetHD Core ML
bundles. It loads `bundle.json`, `tokens.txt`, `ParakeetEncoder.mlpackage`,
and `ParakeetDecoderJoint.mlpackage`, then runs waveform-input transcription
with a greedy TDT decoding loop.

The package is intentionally UI-free. Application developers should embed this
runtime in their own iOS or macOS apps and provide their own recording,
permissions, file import, progress UI, storage, networking, and product logic.

## Build

```bash
cd ios
swift build
```

## Run The CLI

Export and validate a waveform-input bundle first:

```bash
python deployment/export_coreml.py --clean
python deployment/validate_coreml.py --bundle ./coreml-export/bundle.json
```

Then run:

```bash
cd ios
swift run parakeet-transcribe \
  --bundle ../coreml-export/bundle.json \
  --audio /path/to/audio.wav
```

The CLI uses `AVAudioFile`/`AVAudioConverter` to load and resample local audio
to the sample rate recorded in `bundle.json`.

Waveform-input bundles are exported with bounded flexible input shapes. The
default exporter supports 0.25 to 30 seconds of mono audio. Apps that need
longer files or continuous recording should chunk audio before calling the
runtime.

## Library Use

```swift
import ParakeetRuntime

let runtime = try ParakeetRuntime(
    bundleJSON: URL(fileURLWithPath: "/path/to/coreml-export/bundle.json")
)
let transcript = try runtime.transcribe(audioURL: URL(fileURLWithPath: "/path/to/audio.wav"))
```

For bundled app resources, copy `bundle.json`, `tokens.txt`, and the `models/`
directory into your app target and pass the URL for `bundle.json`. For
downloaded models, pass a URL in your app's documents or cache directory.

## Current Scope

Supported:

- Waveform-input Core ML bundles.
- Bundle-driven Core ML feature names.
- Greedy TDT decoding with recurrent decoder states.
- Blank handling, duration-based frame advancement, and max symbols per frame.
- SentencePiece-style `tokens.txt` detokenization using the `▁` word marker.

Application-level responsibilities:

- On-device validation on target iPhones.
- Transcript parity testing against `deployment/validate_transcription.py`.
- UI/audio session integration for recording or file import.
- Chunking or streaming for audio longer than the bundle's exported maximum.
- User consent, local storage, privacy, and product-specific workflows.
- Feature-input support with exact Swift log-mel preprocessing parity.
- Tokenizer parity tests if a future tokenizer requires more than `tokens.txt`.
