import Foundation

public protocol AppModelInstallerClient: Sendable {
    var descriptor: AppModelInstallDescriptor { get }
    func checkInstallRequirement(outputDirectory: URL) throws -> AppModelInstallRequirement
    func installDefaultModel(outputDirectory: URL) -> AsyncThrowingStream<AppModelInstallEvent, Error>
    func discardPartialInstall(outputDirectory: URL) async throws
    /// Paths that make the destination unusable, such as a symlink left where the
    /// model directory belongs. Empty when the destination is fine.
    func blockedInstallEntries(outputDirectory: URL) -> [String]
    /// Remove exactly those entries so a download can start again.
    func clearBlockedInstallPath(outputDirectory: URL) async throws
    func cancel()
}
