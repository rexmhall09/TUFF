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
    @Test func e2bShapesAreNotLabelledE4B() {
        #expect(denseGemmaVariant(hiddenSize: 1_536, numLayers: 35) == .gemma4_E2B)
    }

    @Test func e4bKeepsItsName() {
        #expect(denseGemmaVariant(hiddenSize: 2_560, numLayers: 42) == .gemma4_E4B)
    }

    /// Both dimensions have to agree before a checkpoint is called E2B, so a
    /// future dense Gemma sharing one of them cannot inherit the name.
    @Test func aPartialShapeMatchDoesNotClaimE2B() {
        #expect(denseGemmaVariant(hiddenSize: 1_536, numLayers: 42) == .gemma4_E4B)
        #expect(denseGemmaVariant(hiddenSize: 2_560, numLayers: 35) == .gemma4_E4B)
    }
}
