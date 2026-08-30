import Testing
@testable import TUFFRepackCore

/// Which dense Gemma a checkpoint is, from its own shape.
///
/// Every dense Gemma was labelled E4B, which held while E4B was the only one.
/// E2B repacked correctly — every shape in its manifest was its own — and then
/// carried E4B's variant name, so the installed model was validated against
/// E4B's architecture and a complete, correct download was rejected as
/// "completed install did not pass metadata validation".
@Suite struct DenseGemmaVariantTests {
    @Test func e2bShapesAreNotLabelledE4B() throws {
        #expect(try denseGemmaVariant(hiddenSize: 1_536, numLayers: 35) == .gemma4_E2B)
    }

    @Test func e4bKeepsItsName() throws {
        #expect(try denseGemmaVariant(hiddenSize: 2_560, numLayers: 42) == .gemma4_E4B)
    }

    @Test func qat12BHasAnExplicitIdentity() throws {
        #expect(try denseGemmaVariant(hiddenSize: 3_840, numLayers: 48)
                == .gemma4_12B_QAT)
    }

    /// Shape typos and future variants fail closed instead of silently
    /// inheriting another dense architecture's runtime contract.
    @Test func partialOrUnknownShapeMatchesAreRejected() {
        #expect(throws: RepackError.self) {
            try denseGemmaVariant(hiddenSize: 1_536, numLayers: 42)
        }
        #expect(throws: RepackError.self) {
            try denseGemmaVariant(hiddenSize: 2_560, numLayers: 35)
        }
    }
}
