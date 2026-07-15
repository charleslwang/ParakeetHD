import Foundation

public struct ParakeetBundleConfig: Decodable {
    public let format: String
    public let formatVersion: Int
    public let audio: AudioConfig
    public let tokenizer: TokenizerConfig
    public let decoding: DecodingConfig
    public let coreml: CoreMLConfig

    enum CodingKeys: String, CodingKey {
        case format
        case formatVersion = "format_version"
        case audio
        case tokenizer
        case decoding
        case coreml
    }

    public static func load(from bundleJSON: URL) throws -> ParakeetBundleConfig {
        let data = try Data(contentsOf: bundleJSON)
        let decoder = JSONDecoder()
        return try decoder.decode(ParakeetBundleConfig.self, from: data)
    }
}

public struct AudioConfig: Decodable {
    public let sampleRate: Int
    public let channels: Int
    public let dtype: String
    public let encoderInput: String

    enum CodingKeys: String, CodingKey {
        case sampleRate = "sample_rate"
        case channels
        case dtype
        case encoderInput = "encoder_input"
    }
}

public struct TokenizerConfig: Decodable {
    public let type: String
    public let tokensPath: String
    public let tokensCount: Int

    enum CodingKeys: String, CodingKey {
        case type
        case tokensPath = "tokens_path"
        case tokensCount = "tokens_count"
    }
}

public struct DecodingConfig: Decodable {
    public let type: String
    public let blankID: Int
    public let vocabSize: Int
    public let tokenClassCount: Int
    public let durations: [Int]
    public let durationClassCount: Int
    public let jointOutputSize: Int?
    public let durationOutputSize: Int?
    public let jointOutputLayout: String?

    enum CodingKeys: String, CodingKey {
        case type
        case blankID = "blank_id"
        case vocabSize = "vocab_size"
        case tokenClassCount = "token_class_count"
        case durations
        case durationClassCount = "duration_class_count"
        case jointOutputSize = "joint_output_size"
        case durationOutputSize = "duration_output_size"
        case jointOutputLayout = "joint_output_layout"
    }
}

public struct CoreMLConfig: Decodable {
    public let minimumIOS: Int
    public let computePrecision: String
    public let encoder: ModelConfig
    public let decoderJoint: ModelConfig

    enum CodingKeys: String, CodingKey {
        case minimumIOS = "minimum_ios"
        case computePrecision = "compute_precision"
        case encoder
        case decoderJoint = "decoder_joint"
    }
}

public struct ModelConfig: Decodable {
    public let path: String
    public let inputs: [FeatureConfig]
    public let outputs: [FeatureConfig]
}

public struct FeatureConfig: Decodable {
    public let name: String
    public let semantic: String
    public let exampleShape: [Int]

    enum CodingKeys: String, CodingKey {
        case name
        case semantic
        case exampleShape = "example_shape"
    }
}

extension Array where Element == FeatureConfig {
    func name(forSemantic semantic: String) throws -> String {
        guard let feature = first(where: { $0.semantic == semantic }) else {
            throw ParakeetError.missingFeature("No feature declared for semantic '\(semantic)'")
        }
        return feature.name
    }
}
