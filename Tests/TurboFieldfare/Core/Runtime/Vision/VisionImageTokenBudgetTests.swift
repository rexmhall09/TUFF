import Testing
@testable import TurboFieldfare

@Suite struct VisionImageTokenBudgetTests {
    @Test func textOnlyFamilyHasNoImageBudget() {
        #expect(VisionImageTokenBudget.maximumTokensPerImage(family: .gptOss) == 0)
        #expect(VisionImageTokenBudget.capacity(
            maxContext: 131_072,
            reservedTextTokens: 0,
            family: .gptOss) == 0)
    }

    @Test func visionFamiliesKeepPositiveBudgets() {
        for family in [ModelFamily.gemma4, .qwen36] {
            #expect(VisionImageTokenBudget.maximumTokensPerImage(family: family) > 0)
            #expect(VisionImageTokenBudget.capacity(
                maxContext: 131_072,
                reservedTextTokens: 1_024,
                family: family) > 0)
        }
    }
}
