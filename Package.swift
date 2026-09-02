// swift-tools-version: 6.2
import PackageDescription

let package = Package(
    name: "TUFF",
    platforms: [
        .macOS(.v15),
        .iOS(.v26),
    ],
    products: [
        .library(name: "TUFFEngine", targets: ["TUFFEngine"]),
        .executable(name: "TUFFRepack", targets: ["TUFFRepack"]),
        .executable(name: "TUFFCLI", targets: ["TUFFCLI"]),
        .executable(name: "TUFFCommand", targets: ["TUFFCommand"]),
        .executable(name: "TUFF", targets: ["TUFFMac"]),
        .executable(name: "TUFFDecodeService", targets: ["TUFFDecodeService"]),
        .executable(name: "TUFFServer", targets: ["TUFFServer"]),
    ],
    dependencies: [
        .package(url: "https://github.com/huggingface/swift-transformers", from: "1.3.0"),
        .package(url: "https://github.com/apple/swift-nio.git", exact: "2.101.3"),
        .package(url: "https://github.com/sparkle-project/Sparkle", exact: "2.9.2"),
        .package(url: "https://github.com/mgriebling/SwiftMath.git", exact: "1.7.3"),
    ],
    targets: [
        .target(
            name: "TUFFModelCatalog",
            path: "Sources/TUFFModelCatalog"
        ),
        .target(
            name: "TUFFFormat",
            path: "Sources/TUFFFormat"
        ),
        .target(
            name: "TUFFEngine",
            dependencies: [
                "TUFFFormat",
                .product(name: "Tokenizers", package: "swift-transformers"),
                .product(name: "Hub", package: "swift-transformers"),
            ],
            path: "Sources/TUFFEngine",
            resources: [
                .copy("Metal"),
            ]
        ),
        .target(
            name: "TUFFRepackCore",
            dependencies: ["TUFFModelCatalog", "TUFFFormat"],
            path: "Sources/TUFFRepack/Core"
        ),
        .executableTarget(
            name: "TUFFRepack",
            dependencies: ["TUFFRepackCore"],
            path: "Sources/TUFFRepack/Command"
        ),
        .target(
            name: "TUFFCLICore",
            dependencies: ["TUFFEngine"],
            path: "Sources/TUFFCLI",
            exclude: ["Command"]
        ),
        .executableTarget(
            name: "TUFFCLI",
            dependencies: ["TUFFCLICore"],
            path: "Sources/TUFFCLI/Command"
        ),
        .target(
            name: "TUFFCommandCore",
            dependencies: ["TUFFModelCatalog"],
            path: "Sources/TUFFCommand/Core"
        ),
        .executableTarget(
            name: "TUFFCommand",
            dependencies: ["TUFFCommandCore"],
            path: "Sources/TUFFCommand/Command"
        ),
        .target(
            name: "TUFFAppCore",
            dependencies: ["TUFFModelCatalog", "TUFFEngine", "TUFFRepackCore", "TUFFDecodeProtocol"],
            path: "Sources/TUFFApp/Core",
            resources: [
                .copy("Resources/app-prompts.json"),
            ]
        ),
        .target(
            name: "TUFFMacPresentation",
            dependencies: [
                "TUFFAppCore",
                .product(name: "SwiftMath", package: "SwiftMath"),
            ],
            path: "Sources/TUFFApp/MacPresentation"
        ),
        .target(
            name: "TUFFDecodeProtocol",
            dependencies: ["TUFFEngine"],
            path: "Sources/TUFFDecodeProtocol"
        ),
        .executableTarget(
            name: "TUFFDecodeService",
            dependencies: ["TUFFAppCore", "TUFFDecodeProtocol"],
            path: "Sources/TUFFDecodeService"
        ),
        .target(
            name: "TUFFServerCore",
            dependencies: [
                "TUFFModelCatalog",
                "TUFFEngine",
                .product(name: "NIOCore", package: "swift-nio"),
                .product(name: "NIOPosix", package: "swift-nio"),
                .product(name: "NIOHTTP1", package: "swift-nio"),
            ],
            path: "Sources/TUFFServer/Core"
        ),
        .target(
            name: "TUFFAppServer",
            dependencies: [
                "TUFFAppCore",
                "TUFFServerCore",
                .product(name: "NIOCore", package: "swift-nio"),
            ],
            path: "Sources/TUFFApp/Server"
        ),
        .target(
            name: "TUFFAppUpdater",
            dependencies: [
                .product(name: "Sparkle", package: "Sparkle"),
            ],
            path: "Sources/TUFFApp/Updater"
        ),
        .executableTarget(
            name: "TUFFServer",
            dependencies: ["TUFFServerCore"],
            path: "Sources/TUFFServer/Command"
        ),
        .executableTarget(
            name: "TUFFMac",
            dependencies: [
                "TUFFAppCore",
                "TUFFAppServer",
                "TUFFAppUpdater",
                "TUFFMacPresentation",
            ],
            path: "Sources/TUFFApp/Mac",
            resources: [
                .copy("Resources/tuff-app-icon.png"),
            ]
        ),
        .target(
            name: "TUFFValidationSupport",
            dependencies: ["TUFFEngine"],
            path: "Sources/TUFFValidation/Support"
        ),
        .testTarget(
            name: "TUFFModelCatalogTests",
            dependencies: ["TUFFModelCatalog"],
            path: "Tests/TUFFModelCatalog"
        ),
        .testTarget(
            name: "TUFFCommandTests",
            dependencies: ["TUFFCommandCore", "TUFFModelCatalog"],
            path: "Tests/TUFFCommand"
        ),
        .testTarget(
            name: "TUFFFormatTests",
            dependencies: ["TUFFFormat"],
            path: "Tests/TUFFFormat"
        ),
        .testTarget(
            name: "TUFFFormatCompatibilityTests",
            dependencies: ["TUFFFormat", "TUFFEngine", "TUFFRepackCore"],
            path: "Tests/TUFFFormatCompatibility",
            resources: [.copy("Fixtures")]
        ),
        .testTarget(
            name: "TUFFTestsCore",
            dependencies: [
                "TUFFEngine",
                "TUFFValidationSupport",
                "TUFFRepackCore",
                "TUFFCLICore",
                .product(name: "Hub", package: "swift-transformers"),
            ],
            path: "Tests/TUFFEngine/Core",
            resources: [
                .copy("Tokenization/Fixtures"),
                .copy("Runtime/Vision/Fixtures/images"),
            ]
        ),
        .testTarget(
            name: "TUFFRepackTests",
            dependencies: ["TUFFFormat", "TUFFRepackCore"],
            path: "Tests/TUFFRepack/Core"
        ),
        .testTarget(
            name: "TUFFAppCoreTests",
            dependencies: [
                "TUFFModelCatalog",
                "TUFFAppCore",
                "TUFFEngine",
                "TUFFRepackCore",
                "TUFFDecodeProtocol",
            ],
            path: "Tests/TUFFApp/Core"
        ),
        .testTarget(
            name: "TUFFDecodeServiceTests",
            dependencies: ["TUFFDecodeService", "TUFFAppCore", "TUFFDecodeProtocol"],
            path: "Tests/TUFFDecodeService"
        ),
        .testTarget(
            name: "TUFFMacPresentationTests",
            dependencies: [
                "TUFFModelCatalog",
                "TUFFAppCore",
                "TUFFAppServer",
                "TUFFAppUpdater",
                "TUFFMac",
                "TUFFMacPresentation",
            ],
            path: "Tests/TUFFApp/MacPresentation"
        ),
        .testTarget(
            name: "TUFFServerTests",
            dependencies: [
                "TUFFServerCore",
                .product(name: "NIOEmbedded", package: "swift-nio"),
            ],
            path: "Tests/TUFFServer",
            resources: [.copy("Fixtures")]
        ),
        .testTarget(
            name: "TUFFAppServerTests",
            dependencies: [
                "TUFFAppCore",
                "TUFFAppServer",
                "TUFFServerCore",
            ],
            path: "Tests/TUFFAppServer"
        ),
        .testTarget(
            name: "TUFFAppUpdaterTests",
            dependencies: ["TUFFAppUpdater"],
            path: "Tests/TUFFAppUpdater"
        ),
    ]
)
