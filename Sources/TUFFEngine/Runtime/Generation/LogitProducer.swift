import Metal

/// Produces next-token logits for the `Generator`. The production
/// implementation is `RealForwardRunner`; tests use scripted logits so decode
/// behavior stays independent of the kernel stack.
public protocol LogitProducer: AnyObject, Sendable {
    /// Clear any per-generation state, such as KV cache.
    func reset()
    /// Run one token at `position`, leaving FP16 logits in `logits`.
    func produce(token: Int32, position: Int, into logits: MTLBuffer) async throws
}

/// A producer that can carry extra GPU work on the command buffer which
/// finishes a token. Sampling reads only the logits that buffer produces, so
/// appending it there saves a second submit-and-wait per token.
public protocol EpilogueFusingLogitProducer: LogitProducer {
    /// Runs one token and, when the runtime's layer graph allows it, encodes
    /// `epilogue` onto the same command buffer before committing. Returns
    /// `true` when `epilogue` ran; `false` when it was not encoded and the
    /// caller must submit that work itself.
    func produce(token: Int32, position: Int, into logits: MTLBuffer,
                 appendingToFinalCommandBuffer epilogue: (MTLCommandBuffer) -> Void)
        async throws -> Bool
}

public protocol ContinuableLogitProducer: LogitProducer {
    var continuationPosition: Int { get }
    func prepareForContinuation(expectedPosition: Int) throws
}

protocol ContextWindowReporting: Sendable {
    var maxContext: Int { get }
}

public enum PrefillOutputMode: Sendable, Equatable {
    case logits
    case greedyIfAvailable
}

public enum PrefillSeed: Sendable, Equatable {
    case logitsWritten
    case greedyToken(UInt32)
}

public struct PrefillResult: Sendable, Equatable {
    public let newPosition: Int
    public let seed: PrefillSeed

    public init(newPosition: Int, seed: PrefillSeed) {
        self.newPosition = newPosition
        self.seed = seed
    }
}

protocol ChunkedPrefillRunner: LogitProducer {
    /// Prefill a prompt slice using the chunked production runtime.
    func prefillChunked(tokens: ArraySlice<Int32>,
                        startPosition: Int,
                        outputMode: PrefillOutputMode,
                        config: PrefillRuntimeConfig,
                        into logits: MTLBuffer,
                        onProgress: (Int) -> Void) async throws -> PrefillResult
}

protocol MultimodalPrefillRunner: LogitProducer {
    func prefillMultimodal(input: MultimodalPrefillInput,
                           startPosition: Int,
                           outputMode: PrefillOutputMode,
                           config: PrefillRuntimeConfig,
                           into logits: MTLBuffer,
                           onProgress: (Int) -> Void) async throws -> PrefillResult
}
