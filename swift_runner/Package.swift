// swift-tools-version: 6.0
import PackageDescription

let package = Package(
    name: "swift_runner",
    platforms: [
        .macOS(.v13),
    ],
    products: [
        .executable(name: "swift_runner", targets: ["swift_runner"]),
        .executable(name: "swift_cached_runner", targets: ["swift_cached_runner"]),
        .executable(name: "swift_fullseq_runner", targets: ["swift_fullseq_runner"]),
        .executable(name: "pure_coreml_asr_cli", targets: ["pure_coreml_asr_cli"]),
        .library(name: "ChunkMergeCore", targets: ["ChunkMergeCore"]),
    ],
    dependencies: [],
    targets: [
        .target(
            name: "ChunkMergeCore",
            dependencies: []
        ),
        .executableTarget(
            name: "swift_runner",
            dependencies: []
        ),
        .executableTarget(
            name: "swift_cached_runner",
            dependencies: []
        ),
        .executableTarget(
            name: "swift_fullseq_runner",
            dependencies: []
        ),
        .executableTarget(
            name: "pure_coreml_asr_cli",
            dependencies: ["ChunkMergeCore"]
        ),
        .testTarget(
            name: "ChunkMergeCoreTests",
            dependencies: ["ChunkMergeCore"]
        ),
    ]
)
