import Foundation
import ParakeetRuntime

func usage() -> Never {
    fputs("Usage: parakeet-transcribe [--cpu-only|--cpu-gpu] --bundle /path/to/bundle.json --audio /path/to/audio.wav\n", stderr)
    exit(2)
}

var bundlePath: String?
var audioPath: String?
var computeUnits: ParakeetRuntime.Options = .init()
var args = Array(CommandLine.arguments.dropFirst())
while !args.isEmpty {
    let arg = args.removeFirst()
    switch arg {
    case "--bundle":
        guard !args.isEmpty else { usage() }
        bundlePath = args.removeFirst()
    case "--audio":
        guard !args.isEmpty else { usage() }
        audioPath = args.removeFirst()
    case "--cpu-only":
        computeUnits = .init(computeUnits: .cpuOnly)
    case "--cpu-gpu":
        computeUnits = .init(computeUnits: .cpuAndGPU)
    default:
        usage()
    }
}

guard let bundlePath, let audioPath else {
    usage()
}

do {
    let runtime = try ParakeetRuntime(bundleJSON: URL(fileURLWithPath: bundlePath), options: computeUnits)
    let text = try runtime.transcribe(audioURL: URL(fileURLWithPath: audioPath))
    print(text)
} catch {
    fputs("\(error.localizedDescription)\n", stderr)
    exit(1)
}
