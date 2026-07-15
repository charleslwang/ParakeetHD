// swift-tools-version: 5.10

import PackageDescription

let package = Package(
    name: "ParakeetHD",
    platforms: [
        .iOS(.v17),
        .macOS(.v14),
    ],
    products: [
        .library(name: "ParakeetRuntime", targets: ["ParakeetRuntime"]),
        .executable(name: "parakeet-transcribe", targets: ["ParakeetTranscribe"]),
    ],
    targets: [
        .target(name: "ParakeetRuntime"),
        .executableTarget(
            name: "ParakeetTranscribe",
            dependencies: ["ParakeetRuntime"]
        ),
    ]
)
