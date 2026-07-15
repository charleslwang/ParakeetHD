import AVFoundation
import Foundation

public enum ParakeetAudioLoader {
    public static func loadMonoFloat32(url: URL, targetSampleRate: Double) throws -> [Float] {
        let file = try AVAudioFile(forReading: url)
        let sourceFormat = file.processingFormat
        let frameCount = AVAudioFrameCount(file.length)
        guard let sourceBuffer = AVAudioPCMBuffer(pcmFormat: sourceFormat, frameCapacity: frameCount) else {
            throw ParakeetError.invalidTensor("Could not allocate source audio buffer")
        }
        try file.read(into: sourceBuffer)

        let monoFormat = AVAudioFormat(
            commonFormat: .pcmFormatFloat32,
            sampleRate: targetSampleRate,
            channels: 1,
            interleaved: false
        )
        guard let monoFormat else {
            throw ParakeetError.invalidTensor("Could not create mono audio format")
        }

        if abs(sourceFormat.sampleRate - targetSampleRate) < 0.001 && sourceFormat.channelCount == 1,
           let channel = sourceBuffer.floatChannelData?[0] {
            return Array(UnsafeBufferPointer(start: channel, count: Int(sourceBuffer.frameLength)))
        }

        guard let converter = AVAudioConverter(from: sourceFormat, to: monoFormat) else {
            throw ParakeetError.invalidTensor("Could not create audio converter")
        }

        let ratio = targetSampleRate / sourceFormat.sampleRate
        let outputCapacity = AVAudioFrameCount(Double(sourceBuffer.frameLength) * ratio + 1024)
        guard let outputBuffer = AVAudioPCMBuffer(pcmFormat: monoFormat, frameCapacity: outputCapacity) else {
            throw ParakeetError.invalidTensor("Could not allocate converted audio buffer")
        }

        var didProvideInput = false
        var conversionError: NSError?
        converter.convert(to: outputBuffer, error: &conversionError) { _, status in
            if didProvideInput {
                status.pointee = .noDataNow
                return nil
            }
            didProvideInput = true
            status.pointee = .haveData
            return sourceBuffer
        }
        if let conversionError {
            throw conversionError
        }
        guard let channel = outputBuffer.floatChannelData?[0] else {
            throw ParakeetError.invalidTensor("Converted audio is missing channel data")
        }
        return Array(UnsafeBufferPointer(start: channel, count: Int(outputBuffer.frameLength)))
    }
}
