import Foundation
import Metal
import Testing
@testable import TurboFieldfare
import TurboFieldfareValidationSupport

private enum GPTOSSToyCPUReference {
    static func logits(model: Model, token: Int32) async throws -> [Float] {
        let config = model.config
        var hidden = half(readBF16(model.embedding,
                                   count: config.vocabSize * config.hiddenSize)[
            (Int(token) * config.hiddenSize)..<((Int(token) + 1) * config.hiddenSize)
        ])
        for layer in 0..<config.numLayers {
            let inputNorm = readBF16(try model.inputNorm(layer: layer),
                                     count: config.hiddenSize)
            let normed = half(RmsNormRef.apply(
                x: hidden, weight: inputNorm, eps: 1e-5))
            var query = half(project(
                matrix: readBF16(try model.qProj(layer: layer),
                                 count: config.numHeads * config.headDim
                                    * config.hiddenSize),
                input: normed,
                bias: readBF16(try model.qProjBias(layer: layer),
                               count: config.numHeads * config.headDim),
                rows: config.numHeads * config.headDim,
                columns: config.hiddenSize))
            var key = half(project(
                matrix: readBF16(try model.kProj(layer: layer),
                                 count: config.numKVHeads * config.headDim
                                    * config.hiddenSize),
                input: normed,
                bias: readBF16(try model.kProjBias(layer: layer),
                               count: config.numKVHeads * config.headDim),
                rows: config.numKVHeads * config.headDim,
                columns: config.hiddenSize))
            let value = half(project(
                matrix: readBF16(try model.vProj(layer: layer),
                                 count: config.numKVHeads * config.headDim
                                    * config.hiddenSize),
                input: normed,
                bias: readBF16(try model.vProjBias(layer: layer),
                               count: config.numKVHeads * config.headDim),
                rows: config.numKVHeads * config.headDim,
                columns: config.hiddenSize))
            let yarn = config.yarnRope!
            query = half(RopeRef.applyYaRNNeox(
                input: query,
                numTokens: 1,
                numHeads: config.numHeads,
                headDim: config.headDim,
                position: 0,
                theta: Float(config.ropeTheta),
                originalContextLength: yarn.originalContextLength,
                scalingFactor: Float(yarn.scalingFactor),
                betaFast: Float(yarn.betaFast),
                betaSlow: Float(yarn.betaSlow)))
            key = half(RopeRef.applyYaRNNeox(
                input: key,
                numTokens: 1,
                numHeads: config.numKVHeads,
                headDim: config.headDim,
                position: 0,
                theta: Float(config.ropeTheta),
                originalContextLength: yarn.originalContextLength,
                scalingFactor: Float(yarn.scalingFactor),
                betaFast: Float(yarn.betaFast),
                betaSlow: Float(yarn.betaSlow)))
            let sinks = readBF16(try model.attentionSinks(layer: layer),
                                 count: config.numHeads)
            let attended = half(AttentionRef.apply(
                q: query,
                k: key,
                v: value,
                headDim: config.headDim,
                numQHeads: config.numHeads,
                numKVHeads: config.numKVHeads,
                seqLen: 1,
                window: config.layerIsFull(layer) ? nil : config.slidingWindow,
                scale: Float(config.attentionScale),
                sinks: sinks))
            let projected = half(project(
                matrix: readBF16(try model.oProj(layer: layer),
                                 count: config.hiddenSize * config.numHeads
                                    * config.headDim),
                input: attended,
                bias: readBF16(try model.oProjBias(layer: layer),
                               count: config.hiddenSize),
                rows: config.hiddenSize,
                columns: config.numHeads * config.headDim))
            hidden = zip(hidden, projected).map { Float(Float16($0 + $1)) }

            let postNorm = readBF16(try model.postAttnNorm(layer: layer),
                                    count: config.hiddenSize)
            let expertInput = half(RmsNormRef.apply(
                x: hidden, weight: postNorm, eps: 1e-5))
            let routerLogits = project(
                matrix: readBF16(try model.router(layer: layer),
                                 count: config.numExperts * config.hiddenSize),
                input: expertInput,
                bias: readBF16(try model.routerBias(layer: layer),
                               count: config.numExperts),
                rows: config.numExperts,
                columns: config.hiddenSize)
            let route = GPTOSSMoERef.routerTop4(logits: routerLogits)
            let routeWeights = route.weights.map { Float(Float16($0)) }
            let expertIDs = route.indices.map(Int.init)
            let blobs = try await model.fetchRoutedExperts(
                layer: layer, experts: expertIDs)
            let offsets = try model.gptOssRoutedExpertOffsets(layer: layer)
            var partials: [[Float]] = []
            for blob in blobs {
                partials.append(expert(
                    blob: blob,
                    offsets: offsets,
                    input: expertInput,
                    config: config))
            }
            hidden = (0..<config.hiddenSize).map { column in
                var result = hidden[column]
                for routeIndex in 0..<4 {
                    result += routeWeights[routeIndex] * partials[routeIndex][column]
                }
                return Float(Float16(result))
            }
        }
        let final = half(RmsNormRef.apply(
            x: hidden,
            weight: readBF16(model.finalNorm, count: config.hiddenSize),
            eps: 1e-5))
        return half(project(
            matrix: readBF16(model.lmHead,
                             count: config.vocabSize * config.hiddenSize),
            input: final,
            bias: [Float](repeating: 0, count: config.vocabSize),
            rows: config.vocabSize,
            columns: config.hiddenSize))
    }

    private static func expert(blob: TensorView,
                               offsets: GPTOSSExpertOffsets,
                               input: [Float],
                               config: ArchConfig) -> [Float] {
        let hidden = config.hiddenSize
        let intermediate = config.moeIntermediateSize
        let firstPacked = readBytes(
            blob, offset: offsets.mlp1Weights,
            count: 2 * intermediate * hidden / 2)
        let firstScales = readBytes(
            blob, offset: offsets.mlp1Scales,
            count: 2 * intermediate * hidden / 32)
        let firstWeights = Quantization.dequantizeMXFP4(
            packed: firstPacked,
            scales: firstScales,
            rows: 2 * intermediate,
            columns: hidden)
        let firstBias = readBF16(
            blob, relativeOffset: offsets.mlp1Bias,
            count: 2 * intermediate)
        let first = half(project(
            matrix: firstWeights,
            input: input,
            bias: firstBias,
            rows: 2 * intermediate,
            columns: hidden))
        let gate = stride(from: 0, to: first.count, by: 2).map { first[$0] }
        let linear = stride(from: 1, to: first.count, by: 2).map { first[$0] }
        let activated = half(GPTOSSMoERef.cappedSwiGLU(
            gate: gate, linear: linear, limit: Float(config.swigluLimit)))

        let secondPacked = readBytes(
            blob, offset: offsets.mlp2Weights,
            count: hidden * intermediate / 2)
        let secondScales = readBytes(
            blob, offset: offsets.mlp2Scales,
            count: hidden * intermediate / 32)
        let secondWeights = Quantization.dequantizeMXFP4(
            packed: secondPacked,
            scales: secondScales,
            rows: hidden,
            columns: intermediate)
        let secondBias = readBF16(
            blob, relativeOffset: offsets.mlp2Bias,
            count: hidden)
        return half(project(
            matrix: secondWeights,
            input: activated,
            bias: secondBias,
            rows: hidden,
            columns: intermediate))
    }

    private static func project(matrix: [Float], input: [Float],
                                bias: [Float], rows: Int,
                                columns: Int) -> [Float] {
        (0..<rows).map { row in
            var result = bias[row]
            for column in 0..<columns {
                result += matrix[row * columns + column] * input[column]
            }
            return result
        }
    }

    private static func half<S: Sequence>(_ values: S) -> [Float]
        where S.Element == Float {
        values.map { Float(Float16($0)) }
    }

    private static func readBF16(_ view: TensorView, count: Int) -> [Float] {
        readBF16(view, relativeOffset: 0, count: count)
    }

    private static func readBF16(_ view: TensorView,
                                 relativeOffset: Int,
                                 count: Int) -> [Float] {
        let pointer = view.buffer.contents()
            .advanced(by: Int(view.offset) + relativeOffset)
            .assumingMemoryBound(to: UInt16.self)
        return (0..<count).map { Quantization.bf16ToFloat(pointer[$0]) }
    }

    private static func readBytes(_ view: TensorView,
                                  offset: Int,
                                  count: Int) -> [UInt8] {
        let pointer = view.buffer.contents()
            .advanced(by: Int(view.offset) + offset)
            .assumingMemoryBound(to: UInt8.self)
        return Array(UnsafeBufferPointer(start: pointer, count: count))
    }
}

@Suite(.serialized) struct GPTOSSRunnerTests {
    private func makeRunner() throws
        -> (URL, MetalContext, Model, ModelForwardRunner) {
        let directory = try GPTOSSToySynthetic.write()
        let context = try MetalContext()
        let model = try Model.load(
            directoryURL: directory,
            device: context.device,
            expecting: .gptOssToy(),
            streamingMode: .pread(slotCount: 8))
        let runtime = RuntimeConfiguration(
            expertCacheSlots: 8,
            prefillEnabled: true,
            prefillChunkTokens: 32,
            forceLogitsHead: true)
        let runner = try ModelForwardRunner(
            model: model,
            context: context,
            maxContext: 64,
            runtimeConfiguration: runtime)
        return (directory, context, model, runner)
    }

    private func logits(_ context: MetalContext) throws -> MTLBuffer {
        try #require(context.device.makeBuffer(
            length: ArchConfig.gptOssToy().vocabSize
                * MemoryLayout<Float16>.stride,
            options: .storageModeShared))
    }

    private func values(_ logits: MTLBuffer) -> [Float16] {
        let pointer = logits.contents().assumingMemoryBound(to: Float16.self)
        return (0..<ArchConfig.gptOssToy().vocabSize).map { pointer[$0] }
    }

    @Test func toyInstallSelectsGPTOSSBackendAndStreamsEveryLayer() async throws {
        let (directory, context, model, runner) = try makeRunner()
        defer { try? FileManager.default.removeItem(at: directory) }
        #expect(model.config.family == .gptOss)
        #expect(!runner.usesFusedGreedyHead)
        #expect(model.openLayerFileCount() == 0)

        let output = try logits(context)
        try await runner.produce(token: 7, position: 0, into: output)
        #expect(runner.continuationPosition == 1)
        #expect(model.openLayerFileCount() == 2)
        #expect(values(output).allSatisfy { Float($0).isFinite })
        #expect(runner.totalCb1Nanos > 0)
        #expect(runner.totalIoNanos > 0)
        #expect(runner.totalCb2Nanos > 0)
        #expect(runner.totalHeadNanos > 0)
    }

    @Test func scalarPrefillMatchesDecodeAndContinuation() async throws {
        let (directory, context, _, runner) = try makeRunner()
        defer { try? FileManager.default.removeItem(at: directory) }
        let output = try logits(context)

        try await runner.produce(token: 3, position: 0, into: output)
        try await runner.produce(token: 11, position: 1, into: output)
        let decodeReference = values(output)

        runner.reset()
        var progress: [Int] = []
        let prompt: [Int32] = [3, 11]
        let result = try await runner.prefillChunked(
            tokens: prompt[...],
            startPosition: 0,
            outputMode: .greedyIfAvailable,
            config: .production(chunkTokens: 32),
            into: output,
            onProgress: { progress.append($0) })
        #expect(result == PrefillResult(newPosition: 2, seed: .logitsWritten))
        #expect(progress == [1, 2])
        #expect(values(output) == decodeReference)

        try runner.prepareForContinuation(expectedPosition: 2)
        try await runner.produce(token: 19, position: 2, into: output)
        #expect(runner.continuationPosition == 3)
    }

    @Test func resetRestoresDeterministicKVState() async throws {
        let (directory, context, _, runner) = try makeRunner()
        defer { try? FileManager.default.removeItem(at: directory) }
        let output = try logits(context)

        try await runner.produce(token: 29, position: 0, into: output)
        let first = values(output)
        runner.reset()
        #expect(runner.continuationPosition == 0)
        try await runner.produce(token: 29, position: 0, into: output)
        #expect(values(output) == first)
    }

    @Test func toyLogitsMatchIndependentOpenAIEquations() async throws {
        let (directory, context, model, runner) = try makeRunner()
        defer { try? FileManager.default.removeItem(at: directory) }
        let output = try logits(context)
        let reference = try await GPTOSSToyCPUReference.logits(
            model: model, token: 7)
        try await runner.produce(token: 7, position: 0, into: output)
        let actual = values(output).map(Float.init)
        let relative = RelError.compute(actual: actual, reference: reference)
        #expect(relative < 0.01, "toy GPT-OSS logits rel=\(relative)")
        let actualToken = actual.indices.max { actual[$0] < actual[$1] }
        let referenceToken = reference.indices.max {
            reference[$0] < reference[$1]
        }
        #expect(actualToken == referenceToken)
    }
}
