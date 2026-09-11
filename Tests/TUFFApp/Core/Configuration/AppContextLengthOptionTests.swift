import Testing
@testable import TUFFAppCore

@Suite struct AppContextLengthOptionTests {
    @Test func optionsUseSupportedContextLengthsInAscendingOrder() {
        #expect(AppContextLengthOption.allCases.map(\.tokens)
            == [2_048, 4_096, 8_192, 16_384, 32_768, 65_536, 131_072, 196_608, 204_800, 262_144])
    }

    @Test func menusStopAtEachCheckpointsNativeLimit() {
        #expect(AppContextLengthOption.options(for: .qwen38FlashNext).last?.tokens == 262_144)
        #expect(AppContextLengthOption.options(for: .gemma4E4B).last?.tokens == 131_072)
        #expect(AppContextLengthOption.options(for: .minimaxM27).last?.tokens == 204_800)
        #expect(AppModelSettingsProfile.defaults(for: AppModelInstallDescriptor.qwen38FlashNext.settingsProfileKey).isValid())
    }

    @Test func optionsReportProductionFP16KVAllocation() {
        let originalOptions: [AppContextLengthOption] = [.fourK, .eightK, .sixteenK, .thirtyTwoK, .sixtyFourK]
        let mebibytes = originalOptions.map {
            $0.fp16KVBytes / 1_048_576
        }
        // 29 MiB above the older figures at every size, because the sliding
        // ring is sized for the widest prefill chunk the runtime may see - the
        // pooled image-token count - and the estimate had used the smaller text
        // chunk. The menu deltas are unchanged: every option grew equally.
        #expect(mebibytes == [334, 414, 574, 894, 1_534])
        // Deltas are relative to the default, which is 8K as of 2026-08-17 so
        // that an image and its prompt fit without the user changing anything.
        // 334 - 414 = -80 MiB, 574 - 414 = +160 MiB, and so on.
        #expect(originalOptions.map(\.menuLabel) == [
            "4K, -85 MB",
            "8K, Default",
            "16K, +170 MB",
            "32K, +500 MB",
            "64K, +1.17 GB",
        ])
    }
}
