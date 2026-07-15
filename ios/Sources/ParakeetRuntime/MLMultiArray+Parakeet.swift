import CoreML
import Foundation

extension MLMultiArray {
    static func parakeetFloat32(shape: [NSNumber], values: [Float]) throws -> MLMultiArray {
        let array = try MLMultiArray(shape: shape, dataType: .float32)
        guard array.count == values.count else {
            throw ParakeetError.invalidTensor("Expected \(array.count) values for shape \(shape), got \(values.count)")
        }
        let pointer = array.dataPointer.bindMemory(to: Float.self, capacity: array.count)
        for index in 0..<values.count {
            pointer[index] = values[index]
        }
        return array
    }

    static func parakeetInt32(shape: [NSNumber], values: [Int32]) throws -> MLMultiArray {
        let array = try MLMultiArray(shape: shape, dataType: .int32)
        guard array.count == values.count else {
            throw ParakeetError.invalidTensor("Expected \(array.count) values for shape \(shape), got \(values.count)")
        }
        let pointer = array.dataPointer.bindMemory(to: Int32.self, capacity: array.count)
        for index in 0..<values.count {
            pointer[index] = values[index]
        }
        return array
    }

    func intValue(at index: Int = 0) throws -> Int {
        guard index >= 0 && index < count else {
            throw ParakeetError.invalidTensor("Index \(index) is out of bounds for count \(count)")
        }
        switch dataType {
        case .int32:
            return Int(dataPointer.bindMemory(to: Int32.self, capacity: count)[index])
        case .float32:
            return Int(dataPointer.bindMemory(to: Float.self, capacity: count)[index])
        case .double:
            return Int(dataPointer.bindMemory(to: Double.self, capacity: count)[index])
        default:
            return self[index].intValue
        }
    }

    func argmaxLastDimension(prefixCount: Int? = nil, offset: Int = 0) throws -> Int {
        let lastDimension = shape.last?.intValue ?? count
        let width = prefixCount ?? lastDimension
        guard width > 0, offset >= 0, offset + width <= lastDimension else {
            throw ParakeetError.invalidTensor("Invalid argmax range offset=\(offset) width=\(width) lastDimension=\(lastDimension)")
        }
        let base = strides.dropLast().enumerated().reduce(0) { partial, item in
            let (axis, stride) = item
            return partial + (shape[axis].intValue - 1) * stride.intValue
        }
        let lastStride = strides.last?.intValue ?? 1
        var bestIndex = 0
        var bestValue = -Float.greatestFiniteMagnitude
        for index in 0..<width {
            let value = floatValue(atFlatIndex: base + (offset + index) * lastStride)
            if value > bestValue {
                bestValue = value
                bestIndex = index
            }
        }
        return bestIndex
    }

    func floatValue(atFlatIndex index: Int) -> Float {
        switch dataType {
        case .float32:
            return dataPointer.bindMemory(to: Float.self, capacity: count)[index]
        case .double:
            return Float(dataPointer.bindMemory(to: Double.self, capacity: count)[index])
        case .float16:
            return self[index].floatValue
        default:
            return self[index].floatValue
        }
    }
}
