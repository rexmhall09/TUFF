import Metal

struct GPTOSSExpertOffsets: Sendable, Equatable {
    let mlp1Weights: Int
    let mlp1Scales: Int
    let mlp1Bias: Int
    let mlp2Weights: Int
    let mlp2Scales: Int
    let mlp2Bias: Int

    func validate(blobBytes: Int, hiddenSize: Int, intermediateSize: Int) throws {
        let fields: [(String, Int, Int)] = [
            ("mlp1", mlp1Weights, 2 * intermediateSize * hiddenSize / 2),
            ("mlp1_scales", mlp1Scales,
             2 * intermediateSize * hiddenSize / Quantization.mxfp4GroupSize),
            ("mlp1_bias", mlp1Bias,
             2 * intermediateSize * MemoryLayout<UInt16>.stride),
            ("mlp2", mlp2Weights, hiddenSize * intermediateSize / 2),
            ("mlp2_scales", mlp2Scales,
             hiddenSize * intermediateSize / Quantization.mxfp4GroupSize),
            ("mlp2_bias", mlp2Bias, hiddenSize * MemoryLayout<UInt16>.stride),
        ]
        for (name, offset, size) in fields {
            guard offset >= 0, size > 0, offset <= blobBytes,
                  size <= blobBytes - offset else {
                throw GPTOSSExpertRuntimeError.invalidBlobRange(name)
            }
        }
    }
}

enum GPTOSSExpertRuntimeError: Error, Equatable {
    case invalidBlobRange(String)
    case invalidScratchLayout
}

/// Persistent scratch for both one-token decode and bounded chunked prefill.
/// Projection intermediates are reused route-by-route; only the final partial
/// vectors scale with `queryCapacity * topK`.
struct GPTOSSExpertScratchLayout: Sendable, Equatable {
    static let maximumPrefillQueries = 256

    let hiddenSize: Int
    let intermediateSize: Int
    let topK: Int
    let queryCapacity: Int

    init(hiddenSize: Int, intermediateSize: Int,
         topK: Int = 4, queryCapacity: Int) {
        precondition(hiddenSize > 0 && hiddenSize.isMultiple(of: 32))
        precondition(intermediateSize > 0 && intermediateSize.isMultiple(of: 32))
        precondition(topK > 0)
        self.hiddenSize = hiddenSize
        self.intermediateSize = intermediateSize
        self.topK = topK
        self.queryCapacity = max(1, min(queryCapacity, Self.maximumPrefillQueries))
    }

    var routeCapacity: Int { queryCapacity * topK }
    var mlp1Elements: Int { 2 * intermediateSize }
    var activationElements: Int { intermediateSize }
    var routePartialElements: Int { routeCapacity * hiddenSize }
    var privateBytes: Int {
        (mlp1Elements + activationElements + routePartialElements)
            * MemoryLayout<Float16>.stride
    }
    var sharedBytes: Int {
        routeCapacity * MemoryLayout<Float16>.stride
    }
    var totalBytes: Int { privateBytes + sharedBytes }
}

struct GPTOSSExpertScratchBuffers {
    let layout: GPTOSSExpertScratchLayout
    let mlp1: MTLBuffer
    let activation: MTLBuffer
    let routePartials: MTLBuffer
    let routeWeights: MTLBuffer

    static func allocate(device: MTLDevice,
                         layout: GPTOSSExpertScratchLayout) throws
        -> GPTOSSExpertScratchBuffers {
        func buffer(elements: Int, mode: MTLResourceOptions,
                    label: String) throws -> MTLBuffer {
            guard let result = device.makeBuffer(
                length: max(1, elements) * MemoryLayout<Float16>.stride,
                options: mode) else {
                throw ModelError.residentBufferWrapFailed
            }
            result.label = label
            return result
        }
        return GPTOSSExpertScratchBuffers(
            layout: layout,
            mlp1: try buffer(elements: layout.mlp1Elements,
                             mode: .storageModePrivate,
                             label: "gptoss.expert.mlp1"),
            activation: try buffer(elements: layout.activationElements,
                                   mode: .storageModePrivate,
                                   label: "gptoss.expert.activation"),
            routePartials: try buffer(elements: layout.routePartialElements,
                                      mode: .storageModePrivate,
                                      label: "gptoss.expert.routePartials"),
            routeWeights: try buffer(elements: layout.routeCapacity,
                                     mode: .storageModeShared,
                                     label: "gptoss.expert.routeWeights"))
    }
}

/// Executes one already-streamed GPT-OSS expert at a time. The caller owns
/// cache planning and keeps each `blob` alive until the command buffer ends.
final class GPTOSSExpertRuntime {
    private let mxfp4: MXFP4GEMV
    private let primitives: GPTOSSMoEPrimitives

    init(context: MetalContext) throws {
        mxfp4 = try MXFP4GEMV(context: context)
        primitives = try GPTOSSMoEPrimitives(context: context)
    }

    func encodeExpert(commandBuffer: MTLCommandBuffer,
                      blob: TensorView,
                      offsets: GPTOSSExpertOffsets,
                      input: MTLBuffer,
                      queryIndex: Int,
                      routeSlot: Int,
                      scratch: GPTOSSExpertScratchBuffers,
                      swigluLimit: Float = 7) throws {
        let layout = scratch.layout
        guard queryIndex >= 0, queryIndex < layout.queryCapacity,
              routeSlot >= 0, routeSlot < layout.topK else {
            throw GPTOSSExpertRuntimeError.invalidScratchLayout
        }
        try offsets.validate(blobBytes: Int(blob.length),
                             hiddenSize: layout.hiddenSize,
                             intermediateSize: layout.intermediateSize)
        guard let blobBase = Int(exactly: blob.offset) else {
            throw GPTOSSExpertRuntimeError.invalidBlobRange("blob")
        }
        let halfBytes = MemoryLayout<Float16>.stride
        let inputOffset = queryIndex * layout.hiddenSize * halfBytes
        mxfp4.encode(
            commandBuffer: commandBuffer,
            weights: blob.buffer, weightsOffset: blobBase + offsets.mlp1Weights,
            scales: blob.buffer, scalesOffset: blobBase + offsets.mlp1Scales,
            input: input, inputOffset: inputOffset,
            output: scratch.mlp1,
            bias: blob.buffer, biasOffset: blobBase + offsets.mlp1Bias,
            rows: UInt32(2 * layout.intermediateSize),
            columns: UInt32(layout.hiddenSize))
        primitives.encodeCappedSwiGLUInterleaved(
            commandBuffer: commandBuffer,
            input: scratch.mlp1,
            output: scratch.activation,
            count: UInt32(layout.intermediateSize),
            limit: swigluLimit)
        let routeIndex = queryIndex * layout.topK + routeSlot
        mxfp4.encode(
            commandBuffer: commandBuffer,
            weights: blob.buffer, weightsOffset: blobBase + offsets.mlp2Weights,
            scales: blob.buffer, scalesOffset: blobBase + offsets.mlp2Scales,
            input: scratch.activation,
            output: scratch.routePartials,
            outputOffset: routeIndex * layout.hiddenSize * halfBytes,
            bias: blob.buffer, biasOffset: blobBase + offsets.mlp2Bias,
            rows: UInt32(layout.hiddenSize),
            columns: UInt32(layout.intermediateSize))
    }

    func encodeReduce(commandBuffer: MTLCommandBuffer,
                      scratch: GPTOSSExpertScratchBuffers,
                      residual: MTLBuffer,
                      output: MTLBuffer,
                      queryStart: Int = 0,
                      queryCount: Int) throws {
        let layout = scratch.layout
        guard queryStart >= 0, queryCount > 0,
              queryStart + queryCount <= layout.queryCapacity else {
            throw GPTOSSExpertRuntimeError.invalidScratchLayout
        }
        let halfBytes = MemoryLayout<Float16>.stride
        primitives.encodeRouteReduce(
            commandBuffer: commandBuffer,
            routePartials: scratch.routePartials,
            routePartialsOffset: queryStart * layout.topK
                * layout.hiddenSize * halfBytes,
            routeWeights: scratch.routeWeights,
            routeWeightsOffset: queryStart * layout.topK * halfBytes,
            residual: residual,
            residualOffset: queryStart * layout.hiddenSize * halfBytes,
            output: output,
            outputOffset: queryStart * layout.hiddenSize * halfBytes,
            queryCount: UInt32(queryCount),
            hiddenSize: UInt32(layout.hiddenSize),
            topK: UInt32(layout.topK))
    }

    func encodeFloatResidualReduce(commandBuffer: MTLCommandBuffer,
                                   scratch: GPTOSSExpertScratchBuffers,
                                   residual: MTLBuffer,
                                   output: MTLBuffer,
                                   queryStart: Int = 0,
                                   queryCount: Int) throws {
        let layout = scratch.layout
        guard queryStart >= 0, queryCount > 0,
              queryStart + queryCount <= layout.queryCapacity else {
            throw GPTOSSExpertRuntimeError.invalidScratchLayout
        }
        let halfBytes = MemoryLayout<Float16>.stride
        let floatBytes = MemoryLayout<Float>.stride
        primitives.encodeFloatResidualRouteReduce(
            commandBuffer: commandBuffer,
            routePartials: scratch.routePartials,
            routePartialsOffset: queryStart * layout.topK
                * layout.hiddenSize * halfBytes,
            routeWeights: scratch.routeWeights,
            routeWeightsOffset: queryStart * layout.topK * halfBytes,
            residual: residual,
            residualOffset: queryStart * layout.hiddenSize * floatBytes,
            output: output,
            outputOffset: queryStart * layout.hiddenSize * floatBytes,
            queryCount: UInt32(queryCount),
            hiddenSize: UInt32(layout.hiddenSize),
            topK: UInt32(layout.topK))
    }
}
