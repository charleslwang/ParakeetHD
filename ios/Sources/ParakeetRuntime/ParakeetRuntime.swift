import CoreML
import Foundation

public final class ParakeetRuntime {
    public struct Options {
        public var computeUnits: MLComputeUnits
        public var maxSymbolsPerFrame: Int

        public init(computeUnits: MLComputeUnits = .all, maxSymbolsPerFrame: Int = 10) {
            self.computeUnits = computeUnits
            self.maxSymbolsPerFrame = maxSymbolsPerFrame
        }
    }

    private let bundleDirectory: URL
    public let config: ParakeetBundleConfig
    private let tokenizer: ParakeetTokenizer
    private let encoder: ParakeetCoreMLModel
    private let decoderJoint: ParakeetCoreMLModel
    private let options: Options

    public init(bundleJSON: URL, options: Options = Options()) throws {
        let config = try ParakeetBundleConfig.load(from: bundleJSON)
        guard config.format == "parakeethd-coreml-bundle" else {
            throw ParakeetError.invalidBundle("Unexpected format '\(config.format)'")
        }
        guard config.decoding.type == "tdt" else {
            throw ParakeetError.unsupportedBundle("Only TDT decoding is supported")
        }
        guard config.audio.encoderInput == "waveform" else {
            throw ParakeetError.unsupportedBundle(
                "Feature-input bundles require AudioPreprocessor parity before Swift runtime use"
            )
        }

        let bundleDirectory = bundleJSON.deletingLastPathComponent()
        self.bundleDirectory = bundleDirectory
        self.config = config
        self.tokenizer = try ParakeetTokenizer(tokensURL: bundleDirectory.appendingPathComponent(config.tokenizer.tokensPath))
        self.encoder = try ParakeetCoreMLModel(
            url: bundleDirectory.appendingPathComponent(config.coreml.encoder.path),
            computeUnits: options.computeUnits
        )
        self.decoderJoint = try ParakeetCoreMLModel(
            url: bundleDirectory.appendingPathComponent(config.coreml.decoderJoint.path),
            computeUnits: options.computeUnits
        )
        self.options = options
    }

    public func transcribe(audioURL: URL) throws -> String {
        let audio = try ParakeetAudioLoader.loadMonoFloat32(
            url: audioURL,
            targetSampleRate: Double(config.audio.sampleRate)
        )
        if ProcessInfo.processInfo.environment["PARAKEET_DEBUG"] == "1" {
            let minValue = audio.min() ?? 0
            let maxValue = audio.max() ?? 0
            fputs("[ParakeetRuntime] audio count=\(audio.count) min=\(minValue) max=\(maxValue)\n", stderr)
        }
        return try transcribe(samples: audio)
    }

    public func transcribe(samples: [Float]) throws -> String {
        guard !samples.isEmpty else {
            return ""
        }

        let encoderInputs = config.coreml.encoder.inputs
        let encoderOutputsConfig = config.coreml.encoder.outputs
        let audioName = try encoderInputs.name(forSemantic: "audio_signal")
        let lengthName = try encoderInputs.name(forSemantic: "length")
        let encodedName = try encoderOutputsConfig.name(forSemantic: "encoder_outputs")
        let encodedLengthName = try encoderOutputsConfig.name(forSemantic: "encoded_length")

        let encoderPrediction = try encoder.predict([
            audioName: try MLMultiArray.parakeetFloat32(shape: [1, NSNumber(value: samples.count)], values: samples),
            lengthName: try MLMultiArray.parakeetInt32(shape: [1], values: [Int32(samples.count)]),
        ])
        guard let encoded = encoderPrediction[encodedName],
              let encodedLengthArray = encoderPrediction[encodedLengthName] else {
            throw ParakeetError.missingFeature("Encoder outputs did not include expected semantic outputs")
        }
        let encodedLength = try encodedLengthArray.intValue()
        if ProcessInfo.processInfo.environment["PARAKEET_DEBUG"] == "1" {
            var minValue = Float.greatestFiniteMagnitude
            var maxValue = -Float.greatestFiniteMagnitude
            for index in 0..<encoded.count {
                let value = encoded.floatValue(atFlatIndex: index)
                minValue = min(minValue, value)
                maxValue = max(maxValue, value)
            }
            debugLog(
                "encoded shape=\(encoded.shape) strides=\(encoded.strides) dtype=\(encoded.dataType.rawValue) length=\(encodedLength) min=\(minValue) max=\(maxValue)"
            )
        }
        let tokenIDs = try decode(encoded: encoded, encodedLength: encodedLength)
        return tokenizer.decode(tokenIDs: tokenIDs)
    }

    private func decode(encoded: MLMultiArray, encodedLength: Int) throws -> [Int] {
        let decoderInputs = config.coreml.decoderJoint.inputs
        let decoderOutputs = config.coreml.decoderJoint.outputs
        let encoderFrameName = try decoderInputs.name(forSemantic: "encoder_outputs")
        let targetName = try decoderInputs.name(forSemantic: "targets")
        let targetLengthName = try decoderInputs.name(forSemantic: "target_length")
        let inputState1Name = try decoderInputs.name(forSemantic: "input_states_1")
        let inputState2Name = try decoderInputs.name(forSemantic: "input_states_2")
        let outputState1Name = try decoderOutputs.name(forSemantic: "output_states_1")
        let outputState2Name = try decoderOutputs.name(forSemantic: "output_states_2")

        guard encoded.shape.count == 3 else {
            throw ParakeetError.invalidTensor("Expected encoder output rank 3, got shape \(encoded.shape)")
        }
        let channels = encoded.shape[1].intValue
        var states = try makeInitialStates()
        var targetID = config.decoding.blankID
        var emitted: [Int] = []
        var frame = 0

        while frame < encodedLength {
            var symbols = 0
            var needLoop = true
            var lastSkip = 1
            while needLoop && symbols < options.maxSymbolsPerFrame {
                let frameArray = try encoderFrame(encoded, frame: frame, channels: channels)
                let prediction = try decoderJoint.predict([
                    encoderFrameName: frameArray,
                    targetName: try MLMultiArray.parakeetInt32(shape: [1, 1], values: [Int32(targetID)]),
                    targetLengthName: try MLMultiArray.parakeetInt32(shape: [1], values: [1]),
                    inputState1Name: states.0,
                    inputState2Name: states.1,
                ])

                let scores = try splitScores(prediction: prediction)
                guard let nextState1 = prediction[outputState1Name],
                      let nextState2 = prediction[outputState2Name] else {
                    throw ParakeetError.missingFeature("Decoder prediction is missing recurrent output states")
                }

                let tokenID = try scores.tokenScores.argmaxLastDimension(prefixCount: config.decoding.tokenClassCount)
                let durationIndex: Int
                if let durationScores = scores.durationScores {
                    durationIndex = try durationScores.argmaxLastDimension(prefixCount: config.decoding.durationClassCount)
                } else if config.decoding.jointOutputLayout == "token_scores_then_duration_scores" {
                    durationIndex = try scores.tokenScores.argmaxLastDimension(
                        prefixCount: config.decoding.durationClassCount,
                        offset: config.decoding.tokenClassCount
                    )
                } else {
                    durationIndex = 0
                }
                let skip = config.decoding.durations[min(durationIndex, config.decoding.durations.count - 1)]
                lastSkip = skip
                debugLog("frame=\(frame) token=\(tokenID) durationIndex=\(durationIndex) skip=\(skip)")

                if tokenID != config.decoding.blankID {
                    emitted.append(tokenID)
                    targetID = tokenID
                    states = (nextState1, nextState2)
                }

                symbols += 1
                frame += skip
                needLoop = skip == 0
            }
            if lastSkip == 0 {
                frame += 1
            }
        }
        return emitted
    }

    private func splitScores(prediction: [String: MLMultiArray]) throws -> (tokenScores: MLMultiArray, durationScores: MLMultiArray?) {
        let outputs = config.coreml.decoderJoint.outputs
        if let tokenName = try? outputs.name(forSemantic: "token_logits"),
           let durationName = try? outputs.name(forSemantic: "duration_logits"),
           let tokenScores = prediction[tokenName],
           let durationScores = prediction[durationName] {
            return (tokenScores, durationScores)
        }
        let jointName = try outputs.name(forSemantic: "joint_logits")
        guard let jointScores = prediction[jointName] else {
            throw ParakeetError.missingFeature("Decoder prediction is missing joint logits")
        }
        return (jointScores, nil)
    }

    private func makeInitialStates() throws -> (MLMultiArray, MLMultiArray) {
        let inputs = config.coreml.decoderJoint.inputs
        let state1 = try zeroArray(shape: inputs.first(where: { $0.semantic == "input_states_1" })?.exampleShape)
        let state2 = try zeroArray(shape: inputs.first(where: { $0.semantic == "input_states_2" })?.exampleShape)
        return (state1, state2)
    }

    private func zeroArray(shape: [Int]?) throws -> MLMultiArray {
        guard let shape, !shape.isEmpty else {
            throw ParakeetError.invalidBundle("Missing recurrent state example shape")
        }
        return try MLMultiArray.parakeetFloat32(
            shape: shape.map { NSNumber(value: $0) },
            values: Array(repeating: 0, count: shape.reduce(1, *))
        )
    }

    private func encoderFrame(_ encoded: MLMultiArray, frame: Int, channels: Int) throws -> MLMultiArray {
        let strides = encoded.strides.map(\.intValue)
        let values = (0..<channels).map { channel in
            encoded.floatValue(atFlatIndex: channel * strides[1] + frame * strides[2])
        }
        return try MLMultiArray.parakeetFloat32(
            shape: [1, NSNumber(value: channels), 1],
            values: values
        )
    }

    private func debugLog(_ message: String) {
        if ProcessInfo.processInfo.environment["PARAKEET_DEBUG"] == "1" {
            fputs("[ParakeetRuntime] \(message)\n", stderr)
        }
    }
}
