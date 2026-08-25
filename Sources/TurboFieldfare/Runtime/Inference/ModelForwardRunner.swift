import Metal

/// Family-neutral production runner used by the CLI, app, and server. The
/// existing Gemma/Qwen implementation remains untouched; GPT-OSS selects its
/// BF16/MXFP4 layer graph after the model has passed normal load validation.
public final class ModelForwardRunner: ChunkedPrefillRunner,
    MultimodalPrefillRunner, ContextWindowReporting,
    ContinuableLogitProducer, GreedyHeadReporting, @unchecked Sendable {
    private enum Backend {
        case affine(RealForwardRunner)
        case gptOss(GPTOSSForwardRunner)
    }

    private let backend: Backend

    public init(model: Model, context: MetalContext, maxContext: Int,
                runtimeConfiguration: RuntimeConfiguration = .production) throws {
        if model.config.family == .gptOss {
            backend = .gptOss(try GPTOSSForwardRunner(
                model: model,
                context: context,
                maxContext: maxContext,
                runtimeConfiguration: runtimeConfiguration))
        } else {
            backend = .affine(try RealForwardRunner(
                model: model,
                context: context,
                maxContext: maxContext,
                runtimeConfiguration: runtimeConfiguration))
        }
    }

    public func reset() {
        switch backend {
        case .affine(let runner): runner.reset()
        case .gptOss(let runner): runner.reset()
        }
    }

    public func produce(token: Int32, position: Int,
                        into logits: MTLBuffer) async throws {
        switch backend {
        case .affine(let runner):
            try await runner.produce(token: token, position: position, into: logits)
        case .gptOss(let runner):
            try await runner.produce(token: token, position: position, into: logits)
        }
    }

    public func prepareForContinuation(expectedPosition: Int) throws {
        switch backend {
        case .affine(let runner):
            try runner.prepareForContinuation(expectedPosition: expectedPosition)
        case .gptOss(let runner):
            try runner.prepareForContinuation(expectedPosition: expectedPosition)
        }
    }

    public var continuationPosition: Int {
        switch backend {
        case .affine(let runner): return runner.continuationPosition
        case .gptOss(let runner): return runner.continuationPosition
        }
    }

    public var maxContext: Int {
        switch backend {
        case .affine(let runner): return runner.maxContext
        case .gptOss(let runner): return runner.maxContext
        }
    }

    func prefillChunked(tokens: ArraySlice<Int32>,
                        startPosition: Int,
                        outputMode: PrefillOutputMode,
                        config: PrefillRuntimeConfig,
                        into logits: MTLBuffer,
                        onProgress: (Int) -> Void) async throws -> PrefillResult {
        switch backend {
        case .affine(let runner):
            return try await runner.prefillChunked(
                tokens: tokens,
                startPosition: startPosition,
                outputMode: outputMode,
                config: config,
                into: logits,
                onProgress: onProgress)
        case .gptOss(let runner):
            return try await runner.prefillChunked(
                tokens: tokens,
                startPosition: startPosition,
                outputMode: outputMode,
                config: config,
                into: logits,
                onProgress: onProgress)
        }
    }

    func prefillMultimodal(input: MultimodalPrefillInput,
                           startPosition: Int,
                           outputMode: PrefillOutputMode,
                           config: PrefillRuntimeConfig,
                           into logits: MTLBuffer,
                           onProgress: (Int) -> Void) async throws -> PrefillResult {
        switch backend {
        case .affine(let runner):
            return try await runner.prefillMultimodal(
                input: input,
                startPosition: startPosition,
                outputMode: outputMode,
                config: config,
                into: logits,
                onProgress: onProgress)
        case .gptOss:
            throw PrefillError.chunkedUnsupported(
                "GPT-OSS v2 does not support image input")
        }
    }

    var usesFusedGreedyHead: Bool {
        switch backend {
        case .affine(let runner): return runner.usesFusedGreedyHead
        case .gptOss: return false
        }
    }

    var lastGreedyToken: UInt32 {
        switch backend {
        case .affine(let runner): return runner.lastGreedyToken
        case .gptOss: return 0
        }
    }

    public var totalIoNanos: UInt64 {
        switch backend {
        case .affine(let runner): return runner.totalIoNanos
        case .gptOss(let runner): return runner.totalIoNanos
        }
    }
    public var totalCb1Nanos: UInt64 {
        switch backend {
        case .affine(let runner): return runner.totalCb1Nanos
        case .gptOss(let runner): return runner.totalCb1Nanos
        }
    }
    public var totalCb2Nanos: UInt64 {
        switch backend {
        case .affine(let runner): return runner.totalCb2Nanos
        case .gptOss(let runner): return runner.totalCb2Nanos
        }
    }
    public var totalHeadNanos: UInt64 {
        switch backend {
        case .affine(let runner): return runner.totalHeadNanos
        case .gptOss(let runner): return runner.totalHeadNanos
        }
    }
    public var totalHeadFusedNanos: UInt64 {
        switch backend {
        case .affine(let runner): return runner.totalHeadFusedNanos
        case .gptOss: return 0
        }
    }
    public var totalRDAdviseNanos: UInt64 {
        switch backend {
        case .affine(let runner): return runner.totalRDAdviseNanos
        case .gptOss: return 0
        }
    }
    public var totalRDAdviseCalls: UInt64 {
        switch backend {
        case .affine(let runner): return runner.totalRDAdviseCalls
        case .gptOss: return 0
        }
    }
    public var totalRDAdviseBytes: UInt64 {
        switch backend {
        case .affine(let runner): return runner.totalRDAdviseBytes
        case .gptOss: return 0
        }
    }
    public var totalRDAdviseFailures: UInt64 {
        switch backend {
        case .affine(let runner): return runner.totalRDAdviseFailures
        case .gptOss: return 0
        }
    }
    public var totalRDAdviseSkipped: UInt64 {
        switch backend {
        case .affine(let runner): return runner.totalRDAdviseSkipped
        case .gptOss: return 0
        }
    }
}
