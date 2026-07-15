import Foundation

public enum ParakeetError: Error, LocalizedError {
    case invalidBundle(String)
    case unsupportedBundle(String)
    case missingFeature(String)
    case invalidTensor(String)
    case modelPredictionFailed(String)

    public var errorDescription: String? {
        switch self {
        case .invalidBundle(let message):
            return "Invalid Parakeet bundle: \(message)"
        case .unsupportedBundle(let message):
            return "Unsupported Parakeet bundle: \(message)"
        case .missingFeature(let message):
            return "Missing Core ML feature: \(message)"
        case .invalidTensor(let message):
            return "Invalid tensor: \(message)"
        case .modelPredictionFailed(let message):
            return "Core ML prediction failed: \(message)"
        }
    }
}
