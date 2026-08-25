import Foundation
import Testing
@testable import TurboFieldfare
import TurboFieldfareValidationSupport

@Suite struct GPTOSSMoEPrimitiveTests {
    @Test func cappedSwiGLUMatchesReferenceAcrossClampEdges() throws {
        let gate: [Float] = [-20, -7, -1, 0, 1, 6.996, 7, 8, 20]
        let linear: [Float] = [20, -20, -7, -1, 0, 1, 7, 8, -8]
        let context = try MetalContext()
        let primitive = try GPTOSSMoEPrimitives(context: context)
        let gateHalf = gate.map(Float16.init)
        let linearHalf = linear.map(Float16.init)
        let gateBuffer = try #require(Fp16Buffer.make(context.device, halves: gateHalf))
        let linearBuffer = try #require(Fp16Buffer.make(context.device, halves: linearHalf))
        let output = try #require(Fp16Buffer.make(context.device, count: gate.count))
        let commandBuffer = try #require(context.queue.makeCommandBuffer())

        primitive.encodeCappedSwiGLU(
            commandBuffer: commandBuffer,
            gate: gateBuffer,
            linear: linearBuffer,
            output: output,
            count: UInt32(gate.count))
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        try checkCommandBufferError(commandBuffer.error)

        let actual = Fp16Buffer.read(output, count: gate.count)
        let reference = GPTOSSMoERef.cappedSwiGLU(
            gate: gateHalf.map(Float.init),
            linear: linearHalf.map(Float.init))
        let error = RelError.compute(actual: actual, reference: reference)
        #expect(error < Tolerance.fp16Reduction, "rel=\(error)")
    }

    @Test("top-4 router matches stable CPU selection", arguments: [
        (Int(32), UInt64(0x2001)),
        (Int(128), UInt64(0x2002)),
    ])
    func top4RouterMatchesReference(numExperts: Int, seed: UInt64) throws {
        var random = SeedTree(seed).key("gpt-oss-router")
        var logits = (0..<numExperts).map { _ in random.uniform(-4, 4) }
        // An exact boundary tie verifies stable lower-index selection.
        logits[3] = 6
        logits[17] = 6
        let expected = GPTOSSMoERef.routerTop4(logits: logits)

        let context = try MetalContext()
        let primitive = try GPTOSSMoEPrimitives(context: context)
        let logitBuffer = try #require(context.device.makeBuffer(
            bytes: logits,
            length: logits.count * MemoryLayout<Float>.stride,
            options: .storageModeShared))
        let indices = try #require(context.device.makeBuffer(
            length: 4 * MemoryLayout<UInt32>.stride,
            options: .storageModeShared))
        let weights = try #require(Fp16Buffer.make(context.device, count: 4))
        let commandBuffer = try #require(context.queue.makeCommandBuffer())
        primitive.encodeRouterTop4(
            commandBuffer: commandBuffer,
            logits: logitBuffer,
            outputIndices: indices,
            outputWeights: weights,
            numExperts: UInt32(numExperts))
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        try checkCommandBufferError(commandBuffer.error)

        let pointer = indices.contents().bindMemory(to: UInt32.self, capacity: 4)
        let actualIndices = (0..<4).map { pointer[$0] }
        let actualWeights = Fp16Buffer.read(weights, count: 4)
        #expect(actualIndices == expected.indices)
        let error = RelError.compute(actual: actualWeights, reference: expected.weights)
        #expect(error < Tolerance.fp16Reduction, "rel=\(error)")
    }
}
