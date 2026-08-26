// swift-tools-version: 6.2
import PackageDescription

let package = Package(
    name: "TurboFieldfare",
    platforms: [
        .macOS(.v15),
        .iOS(.v26),
    ],
    products: [
        .library(name: "TurboFieldfare", targets: ["TurboFieldfare"]),
        .executable(name: "TUFFRepack", targets: ["TurboFieldfareRepack"]),
        .executable(name: "TUFFCLI", targets: ["TurboFieldfareCLI"]),
        .executable(name: "TUFF", targets: ["TurboFieldfareMac"]),
        .executable(name: "TUFFDecodeService", targets: ["TurboFieldfareDecodeService"]),
        .executable(name: "TUFFServer", targets: ["TurboFieldfareServer"]),
    ],
    dependencies: [
        .package(url: "https://github.com/huggingface/swift-transformers", from: "1.3.0"),
        .package(url: "https://github.com/apple/swift-nio.git", exact: "2.101.3"),
        .package(url: "https://github.com/sparkle-project/Sparkle", exact: "2.9.2"),
    ],
    targets: [
        .target(
            name: "TUFFModelCatalog",
            path: "Sources/TUFFModelCatalog"
        ),
        .target(
            name: "TurboFieldfareFormat",
            path: "Sources/TurboFieldfareFormat"
        ),
        .target(
            name: "TurboFieldfare",
            dependencies: [
                "TurboFieldfareFormat",
                .product(name: "Tokenizers", package: "swift-transformers"),
                .product(name: "Hub", package: "swift-transformers"),
            ],
            path: "Sources/TurboFieldfare",
            resources: [
                .copy("Metal"),
            ]
        ),
        .target(
            name: "TurboFieldfareRepackCore",
            dependencies: ["TUFFModelCatalog", "TurboFieldfareFormat"],
            path: "Sources/TurboFieldfareRepack/Core"
        ),
        .executableTarget(
            name: "TurboFieldfareRepack",
            dependencies: ["TurboFieldfareRepackCore"],
            path: "Sources/TurboFieldfareRepack/Command"
        ),
        .target(
            name: "TurboFieldfareCLICore",
            dependencies: ["TurboFieldfare"],
            path: "Sources/TurboFieldfareCLI",
            exclude: ["Command"]
        ),
        .executableTarget(
            name: "TurboFieldfareCLI",
            dependencies: ["TurboFieldfareCLICore"],
            path: "Sources/TurboFieldfareCLI/Command"
        ),
        .target(
            name: "TurboFieldfareAppCore",
            dependencies: ["TUFFModelCatalog", "TurboFieldfare", "TurboFieldfareRepackCore", "TurboFieldfareDecodeProtocol"],
            path: "Sources/TurboFieldfareApp/Core",
            resources: [
                .copy("Resources/app-prompts.json"),
            ]
        ),
        .target(
            name: "TurboFieldfareMacPresentation",
            dependencies: ["TurboFieldfareAppCore"],
            path: "Sources/TurboFieldfareApp/MacPresentation"
        ),
        .target(
            name: "TurboFieldfareDecodeProtocol",
            dependencies: ["TurboFieldfare"],
            path: "Sources/TurboFieldfareDecodeProtocol"
        ),
        .executableTarget(
            name: "TurboFieldfareDecodeService",
            dependencies: ["TurboFieldfareAppCore", "TurboFieldfareDecodeProtocol"],
            path: "Sources/TurboFieldfareDecodeService"
        ),
        .target(
            name: "TurboFieldfareServerCore",
            dependencies: [
                "TUFFModelCatalog",
                "TurboFieldfare",
                .product(name: "NIOCore", package: "swift-nio"),
                .product(name: "NIOPosix", package: "swift-nio"),
                .product(name: "NIOHTTP1", package: "swift-nio"),
            ],
            path: "Sources/TurboFieldfareServer/Core"
        ),
        .target(
            name: "TurboFieldfareAppServer",
            dependencies: [
                "TurboFieldfareAppCore",
                "TurboFieldfareServerCore",
                .product(name: "NIOCore", package: "swift-nio"),
            ],
            path: "Sources/TurboFieldfareApp/Server"
        ),
        .target(
            name: "TurboFieldfareAppUpdater",
            dependencies: [
                .product(name: "Sparkle", package: "Sparkle"),
            ],
            path: "Sources/TurboFieldfareApp/Updater"
        ),
        .executableTarget(
            name: "TurboFieldfareServer",
            dependencies: ["TurboFieldfareServerCore"],
            path: "Sources/TurboFieldfareServer/Command"
        ),
        .executableTarget(
            name: "TurboFieldfareMac",
            dependencies: [
                "TurboFieldfareAppCore",
                "TurboFieldfareAppServer",
                "TurboFieldfareAppUpdater",
                "TurboFieldfareMacPresentation",
            ],
            path: "Sources/TurboFieldfareApp/Mac",
            resources: [
                .copy("Resources/tuff-app-icon.png"),
            ]
        ),
        .target(
            name: "TurboFieldfareValidationSupport",
            dependencies: ["TurboFieldfare"],
            path: "Sources/TurboFieldfareValidation/Support"
        ),
        .testTarget(
            name: "TUFFModelCatalogTests",
            dependencies: ["TUFFModelCatalog"],
            path: "Tests/TUFFModelCatalog"
        ),
        .testTarget(
            name: "TurboFieldfareFormatTests",
            dependencies: ["TurboFieldfareFormat"],
            path: "Tests/TurboFieldfareFormat"
        ),
        .testTarget(
            name: "TurboFieldfareFormatCompatibilityTests",
            dependencies: ["TurboFieldfareFormat", "TurboFieldfare", "TurboFieldfareRepackCore"],
            path: "Tests/TurboFieldfareFormatCompatibility",
            resources: [.copy("Fixtures")]
        ),
        .testTarget(
            name: "TurboFieldfareTestsCore",
            dependencies: [
                "TurboFieldfare",
                "TurboFieldfareValidationSupport",
                "TurboFieldfareRepackCore",
                "TurboFieldfareCLICore",
                .product(name: "Hub", package: "swift-transformers"),
            ],
            path: "Tests/TurboFieldfare/Core",
            resources: [
                .copy("Tokenization/Fixtures"),
                .copy("Runtime/Vision/Fixtures/images"),
            ]
        ),
        .testTarget(
            name: "TurboFieldfareRepackTests",
            dependencies: ["TurboFieldfareFormat", "TurboFieldfareRepackCore"],
            path: "Tests/TurboFieldfareRepack/Core"
        ),
        .testTarget(
            name: "TurboFieldfareAppCoreTests",
            dependencies: [
                "TUFFModelCatalog",
                "TurboFieldfareAppCore",
                "TurboFieldfare",
                "TurboFieldfareRepackCore",
                "TurboFieldfareDecodeProtocol",
            ],
            path: "Tests/TurboFieldfareApp/Core"
        ),
        .testTarget(
            name: "TurboFieldfareDecodeServiceTests",
            dependencies: ["TurboFieldfareDecodeService", "TurboFieldfareAppCore", "TurboFieldfareDecodeProtocol"],
            path: "Tests/TurboFieldfareDecodeService"
        ),
        .testTarget(
            name: "TurboFieldfareMacPresentationTests",
            dependencies: ["TurboFieldfareAppCore", "TurboFieldfareMacPresentation"],
            path: "Tests/TurboFieldfareApp/MacPresentation"
        ),
        .testTarget(
            name: "TurboFieldfareServerTests",
            dependencies: [
                "TurboFieldfareServerCore",
                .product(name: "NIOEmbedded", package: "swift-nio"),
            ],
            path: "Tests/TurboFieldfareServer",
            resources: [.copy("Fixtures")]
        ),
        .testTarget(
            name: "TurboFieldfareAppServerTests",
            dependencies: [
                "TurboFieldfareAppCore",
                "TurboFieldfareAppServer",
                "TurboFieldfareServerCore",
            ],
            path: "Tests/TurboFieldfareAppServer"
        ),
        .testTarget(
            name: "TurboFieldfareAppUpdaterTests",
            dependencies: ["TurboFieldfareAppUpdater"],
            path: "Tests/TurboFieldfareAppUpdater"
        ),
    ]
)
