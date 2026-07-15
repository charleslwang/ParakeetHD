import CoreML
import Foundation

final class DictionaryFeatureProvider: MLFeatureProvider {
    private let values: [String: MLFeatureValue]

    var featureNames: Set<String> {
        Set(values.keys)
    }

    init(_ values: [String: MLFeatureValue]) {
        self.values = values
    }

    func featureValue(for featureName: String) -> MLFeatureValue? {
        values[featureName]
    }
}

final class ParakeetCoreMLModel {
    private let model: MLModel

    init(url: URL, computeUnits: MLComputeUnits = .all) throws {
        let configuration = MLModelConfiguration()
        configuration.computeUnits = computeUnits
        let loadURL: URL
        if url.pathExtension == "mlmodelc" {
            loadURL = url
        } else {
            loadURL = try MLModel.compileModel(at: url)
        }
        self.model = try MLModel(contentsOf: loadURL, configuration: configuration)
    }

    func predict(_ inputs: [String: MLMultiArray]) throws -> [String: MLMultiArray] {
        let provider = DictionaryFeatureProvider(inputs.mapValues { MLFeatureValue(multiArray: $0) })
        let output: MLFeatureProvider
        do {
            output = try model.prediction(from: provider)
        } catch {
            throw ParakeetError.modelPredictionFailed(error.localizedDescription)
        }

        var arrays: [String: MLMultiArray] = [:]
        for name in output.featureNames {
            if let array = output.featureValue(for: name)?.multiArrayValue {
                arrays[name] = array
            }
        }
        return arrays
    }
}
