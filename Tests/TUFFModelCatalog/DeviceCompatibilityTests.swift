import Testing
@testable import TUFFModelCatalog

@Suite struct DeviceCompatibilityTests {
    private func device(memoryGiB: UInt64,
                        macOS: Int = 26,
                        generation: Int = 2) -> TUFFDeviceCapabilities {
        TUFFDeviceCapabilities(
            unifiedMemoryBytes: memoryGiB * TUFFModelCatalog.oneGiB,
            macOSMajorVersion: macOS,
            appleSiliconGeneration: generation)
    }

    @Test func qualifiedModelsAcceptExistingMemoryTiers() {
        for memoryGiB: UInt64 in [8, 16, 24, 64] {
            for model in [TUFFModelCatalog.gemma4_E4B,
                          TUFFModelCatalog.gemma4_26B_A4B,
                          TUFFModelCatalog.qwen36_35B_A3B] {
                #expect(model.compatibility(with: device(memoryGiB: memoryGiB))
                    .isCompatible)
            }
        }
    }

    @Test func qualifiedGPTOSS20BUsesMeasuredSixteenGBGate() {
        let model = TUFFModelCatalog.gptOss_20B
        #expect(!model.compatibility(with: device(memoryGiB: 8)).isCompatible)
        #expect(model.compatibility(with: device(memoryGiB: 16)).isCompatible)
        #expect(model.compatibility(
            with: device(memoryGiB: 16), contextTokens: 4_096).isCompatible)
        #expect(model.memory.qualifiedDefaultWorkingSetBytes == 5_487_695_296)
    }

    @Test func qualifiedGPTOSS120BUsesMeasuredSixteenGBGate() {
        let model = TUFFModelCatalog.gptOss_120B
        #expect(model.qualification == .qualified)
        #expect(!model.compatibility(with: device(memoryGiB: 8)).isCompatible)
        #expect(model.compatibility(with: device(memoryGiB: 16)).isCompatible)
        #expect(model.compatibility(
            with: device(memoryGiB: 16), contextTokens: 4_096).isCompatible)
        #expect(model.memory.qualifiedDefaultWorkingSetBytes == 7_990_582_952)
        #expect(model.source.approximateDownloadBytes >= 65_248_815_744)
    }

    @Test func unqualifiedCheckpointFailsClosedAtEveryMemoryTier() {
        let base = TUFFModelCatalog.gptOss_120B
        let model = TUFFModelDescriptor(
            id: base.id,
            selector: base.selector,
            aliases: base.aliases,
            apiModelID: base.apiModelID,
            displayName: base.displayName,
            shortName: base.shortName,
            summary: base.summary,
            family: base.family,
            architecture: base.architecture,
            installDirectoryName: base.installDirectoryName,
            source: base.source,
            hardware: base.hardware,
            memory: base.memory,
            qualification: .requiresRealModelValidation,
            runtimeDefaults: base.runtimeDefaults,
            capabilities: base.capabilities,
            reasoningControl: base.reasoningControl,
            addons: base.addons)

        for memoryGiB: UInt64 in [16, 64, 192] {
            let compatibility = model.compatibility(
                with: device(memoryGiB: memoryGiB))
            #expect(!compatibility.isCompatible)
            #expect(compatibility.issues.contains(.requiresRealModelValidation))
        }
    }

    @Test func insufficientMemoryAndPlatformReturnConcreteIssues() {
        let result = TUFFModelCatalog.default.compatibility(
            with: device(memoryGiB: 4, macOS: 14, generation: 0))
        #expect(result.issues.contains(.insufficientUnifiedMemory(
            requiredBytes: 8 * TUFFModelCatalog.oneGiB,
            actualBytes: 4 * TUFFModelCatalog.oneGiB)))
        #expect(result.issues.contains(.unsupportedMacOS(
            requiredMajorVersion: 15, actualMajorVersion: 14)))
        #expect(result.issues.contains(.unsupportedAppleSilicon(
            requiredGeneration: 1, actualGeneration: 0)))
    }

    @Test func contextEstimatesAreModelSpecificAndMonotonic() {
        let gemma = TUFFModelCatalog.gemma4_26B_A4B.memory
        let qwen = TUFFModelCatalog.qwen36_35B_A3B.memory
        #expect(gemma.kvCache.estimatedBytes(contextTokens: 8_192)
                > qwen.kvCache.estimatedBytes(contextTokens: 8_192))
        #expect(gemma.estimatedWorkingSetBytes(contextTokens: 64_000,
                                               expertCacheSlots: 16)
                > gemma.estimatedWorkingSetBytes(contextTokens: 8_000,
                                                  expertCacheSlots: 16))
        #expect(qwen.estimatedWorkingSetBytes(contextTokens: 8_000,
                                              expertCacheSlots: 24)
                > qwen.estimatedWorkingSetBytes(contextTokens: 8_000,
                                                 expertCacheSlots: 8))
    }

    @Test func e4bUsesTheMeasuredEightKPeak() {
        let model = TUFFModelCatalog.gemma4_E4B
        #expect(model.memory.qualifiedDefaultWorkingSetBytes == 1_833_438_160)
        #expect(model.runtimeDefaults.contextTokens == 8_192)
        #expect(model.compatibility(
            with: device(memoryGiB: 8), contextTokens: 8_192).isCompatible)
    }

    @Test func excessiveContextIsRejectedWithoutRejectingTheModel() {
        let model = TUFFModelCatalog.gemma4_26B_A4B
        let host = device(memoryGiB: 8)
        #expect(model.compatibility(with: host).isCompatible)
        let context = model.compatibility(
            with: host, contextTokens: 262_144, expertCacheSlots: 32)
        #expect(!context.isCompatible)
        #expect(context.issues.contains { issue in
            if case .contextExceedsSafeMemory = issue { return true }
            return false
        })
    }

    @Test(arguments: [
        ("Apple M1", 1),
        ("Apple M2 Pro", 2),
        ("Apple M5 Max", 5),
    ])
    func parsesAppleSiliconGeneration(brand: String, expected: Int) {
        #expect(TUFFDeviceCapabilities.appleSiliconGeneration(brandString: brand)
                == expected)
    }

    @Test func rejectsUnknownProcessorBrand() {
        #expect(TUFFDeviceCapabilities.appleSiliconGeneration(
            brandString: "Intel(R) Core(TM)") == nil)
    }
}
