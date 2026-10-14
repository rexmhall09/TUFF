import Foundation

/// Timings of one `tuff prompt` run, written by the engine for the host. No
/// prompt or output text, only counts and seconds.
public struct CLIRunReport: Codable, Equatable, Sendable {
    public var prefillTokens: Int
    public var newTokens: Int
    public var prefillSeconds: Double
    public var decodeSeconds: Double

    public init(prefillTokens: Int, newTokens: Int, prefillSeconds: Double, decodeSeconds: Double) {
        self.prefillTokens = prefillTokens
        self.newTokens = newTokens
        self.prefillSeconds = prefillSeconds
        self.decodeSeconds = decodeSeconds
    }
}
