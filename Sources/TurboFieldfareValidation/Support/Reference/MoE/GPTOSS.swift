import Foundation

public enum GPTOSSMoERef {
    public static func cappedSwiGLU(gate: [Float],
                                    linear: [Float],
                                    limit: Float = 7) -> [Float] {
        precondition(gate.count == linear.count)
        return zip(gate, linear).map { gateValue, linearValue in
            let cappedGate = min(gateValue, limit)
            let cappedLinear = max(-limit, min(limit, linearValue))
            let silu = cappedGate / (1 + Foundation.exp(-1.702 * cappedGate))
            return silu * (cappedLinear + 1)
        }
    }

    public static func routerTop4(logits: [Float])
        -> (indices: [UInt32], weights: [Float]) {
        precondition((4...128).contains(logits.count))
        var ranked: [(index: UInt32, score: Float)] = []
        ranked.reserveCapacity(logits.count)
        for (index, score) in logits.enumerated() {
            ranked.append((UInt32(index), score))
        }
        ranked.sort { lhs, rhs in
            lhs.score == rhs.score ? lhs.index < rhs.index : lhs.score > rhs.score
        }
        let selected = Array(ranked.prefix(4))
        let maximum = selected.first?.score ?? 0
        let exponentials = selected.map { Foundation.exp($0.score - maximum) }
        let denominator = exponentials.reduce(0, +)
        return (selected.map { $0.index }, exponentials.map { $0 / denominator })
    }
}
