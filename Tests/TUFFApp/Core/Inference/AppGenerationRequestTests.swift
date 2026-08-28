import Foundation
import Testing
import TUFFEngine
@testable import TUFFAppCore

@Suite struct AppGenerationRequestTests {
    private let existingDirectory = FileManager.default.temporaryDirectory

    @Test func defaultRequestUsesDocumentedSamplingPolicy() {
        let request = AppGenerationRequest(modelDirectory: existingDirectory, prompt: "hello")
        #expect(request.maxNewTokens == 4_096)
        #expect(request.temperature == 0.2)
        #expect(request.topK == 64)
        #expect(request.topP == 0.95)
        #expect(request.repetitionPenalty == 1)
        #expect(!request.isPureGreedy)
    }

    @Test func temperatureZeroRemainsPureGreedyWithTruncationDefaults() {
        let request = AppGenerationRequest(modelDirectory: existingDirectory,
                                           prompt: "hello",
                                           temperature: 0)
        #expect(request.topK == 64)
        #expect(request.topP == 0.95)
        #expect(request.isPureGreedy)
    }

    @Test func emptyPromptRejected() {
        let request = AppGenerationRequest(modelDirectory: existingDirectory, prompt: "   ")
        #expect(throws: AppInferenceError.self) {
            try request.validate()
        }
    }

    @Test func invalidMaxTokensRejected() {
        let request = AppGenerationRequest(modelDirectory: existingDirectory,
                                           prompt: "hello", maxNewTokens: 0)
        #expect(throws: AppInferenceError.self) {
            try request.validate()
        }
    }

    @Test func invalidSlotCountRejected() {
        var options = AppRuntimeOptions()
        options.expertCacheSlots = 7
        let request = AppGenerationRequest(modelDirectory: existingDirectory,
                                           prompt: "hello", runtimeOptions: options)
        #expect(throws: AppInferenceError.self) {
            try request.validate()
        }
    }

    @Test func repetitionPenaltyBelowOneRejected() {
        let request = AppGenerationRequest(modelDirectory: existingDirectory,
                                           prompt: "hello", repetitionPenalty: 0.9)
        #expect(throws: AppInferenceError.self) {
            try request.validate()
        }
    }

    @Test func invalidTopKRejected() {
        for topK in [0, 257] {
            let request = AppGenerationRequest(modelDirectory: existingDirectory,
                                               prompt: "hello", topK: topK)
            #expect(throws: AppInferenceError.self) {
                try request.validate()
            }
        }
    }

    @Test func invalidTopPRejected() {
        let request = AppGenerationRequest(modelDirectory: existingDirectory,
                                           prompt: "hello", topP: 1.1)
        #expect(throws: AppInferenceError.self) {
            try request.validate()
        }
    }

    @Test func stochasticTopPRequiresTopK() {
        let request = AppGenerationRequest(modelDirectory: existingDirectory,
                                           prompt: "hello", topK: nil, topP: 0.95)
        #expect(throws: AppInferenceError.self) {
            try request.validate()
        }
    }

    @Test func missingModelDirectoryRejected() {
        let request = AppGenerationRequest(
            modelDirectory: URL(fileURLWithPath: "/nonexistent/model.gturbo"),
            prompt: "hello")
        #expect(throws: AppInferenceError.self) {
            try request.validate()
        }
    }
    @Test func duplicateAndMalformedImageDescriptorsAreRejected() {
        let id = UUID()
        let attachment = AppImageAttachment(
            id: id,
            fileURL: URL(fileURLWithPath: "/tmp/image.png"),
            displayName: "image.png",
            encodedBytes: 4,
            sha256: String(repeating: "a", count: 64))
        #expect(throws: AppInferenceError.self) {
            try AppGenerationRequest(
                modelDirectory: existingDirectory,
                prompt: "",
                imageAttachments: [attachment, attachment]).validate()
        }
        let malformed = AppImageAttachment(
            fileURL: URL(fileURLWithPath: "/tmp/image.png"),
            displayName: "image.png",
            encodedBytes: 4,
            sha256: "not-a-digest")
        #expect(throws: AppInferenceError.self) {
            try AppGenerationRequest(
                modelDirectory: existingDirectory,
                prompt: "",
                imageAttachments: [malformed]).validate()
        }
    }

    @Test func imageOnlyRequestIsValid() throws {
        let attachment = AppImageAttachment(
            fileURL: URL(fileURLWithPath: "/tmp/image.png"),
            displayName: "image.png",
            encodedBytes: 4,
            sha256: String(repeating: "a", count: 64))
        let request = AppGenerationRequest(
            modelDirectory: existingDirectory,
            prompt: "",
            imageAttachments: [attachment])
        try request.validate()
    }

    @Test func reasoningControlsMatchTheModelFamily() throws {
        try AppGenerationRequest.validateReasoning(
            family: .gemma4,
            reasoning: .on,
            reasoningEffort: nil)
        try AppGenerationRequest.validateReasoning(
            family: .gptOss,
            reasoning: .off,
            reasoningEffort: .high)

        #expect(throws: AppInferenceError.self) {
            try AppGenerationRequest.validateReasoning(
                family: .gptOss,
                reasoning: .on,
                reasoningEffort: .medium)
        }
        #expect(throws: AppInferenceError.self) {
            try AppGenerationRequest.validateReasoning(
                family: .qwen36,
                reasoning: .off,
                reasoningEffort: .low)
        }
    }

    @Test func gptOssTextSkipsVisionBudgetAndImagesFailClosed() throws {
        try AppGenerationRequest.validateImageCapacity(
            imageCount: 0,
            maxContextTokens: 4_096,
            family: .gptOss)

        #expect(throws: AppInferenceError.invalidRequest(
            "GPT-OSS does not support image input.")) {
            try AppGenerationRequest.validateImageCapacity(
                imageCount: 1,
                maxContextTokens: 4_096,
                family: .gptOss)
        }
    }

    @Test func structuredMessagesCanReplaceTheSimplePrompt() throws {
        let request = AppGenerationRequest(
            modelDirectory: existingDirectory,
            prompt: "",
            structuredMessages: [
                GFTokenizer.Message(role: .user, content: "hello"),
            ],
            repetitionPenalty: 0.8,
            seed: 17,
            stopStrings: ["done"])

        try request.validate()
        #expect(request.seed == 17)
        #expect(request.stopStrings == ["done"])
    }

    @Test func structuredImagesRequireExactMultimodalReferences() throws {
        let attachment = AppImageAttachment(
            fileURL: URL(fileURLWithPath: "/tmp/image.png"),
            displayName: "image.png",
            encodedBytes: 4,
            sha256: String(repeating: "a", count: 64))
        let messages = [GFTokenizer.Message(role: .user, content: "describe")]

        #expect(throws: AppInferenceError.self) {
            try AppGenerationRequest(
                modelDirectory: existingDirectory,
                prompt: "",
                structuredMessages: messages,
                imageAttachments: [attachment]).validate()
        }
        #expect(throws: AppInferenceError.self) {
            try AppGenerationRequest(
                modelDirectory: existingDirectory,
                prompt: "",
                structuredMessages: messages,
                multimodalMessages: [MultimodalMessage(
                    role: .user,
                    content: [.image(id: UUID())])],
                imageAttachments: [attachment]).validate()
        }

        let valid = AppGenerationRequest(
            modelDirectory: existingDirectory,
            prompt: "",
            structuredMessages: messages,
            multimodalMessages: [MultimodalMessage(
                role: .user,
                content: [.image(id: attachment.id), .text("describe")])],
            imageAttachments: [attachment])
        try valid.validate()
    }

    @Test func structuredStopStringsCannotBeEmpty() {
        let request = AppGenerationRequest(
            modelDirectory: existingDirectory,
            prompt: "",
            structuredMessages: [
                GFTokenizer.Message(role: .user, content: "hello"),
            ],
            stopStrings: [""])

        #expect(throws: AppInferenceError.self) {
            try request.validate()
        }
    }

}
