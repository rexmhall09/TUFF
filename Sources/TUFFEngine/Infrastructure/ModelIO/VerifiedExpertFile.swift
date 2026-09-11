import Darwin
import Foundation

/// Owns the exact descriptor being checked, through its eventual streamer
/// handoff. A pending check retains itself until completion, including when
/// a generation is cancelled and its model is released.
final class VerifiedExpertFile: @unchecked Sendable {
    let descriptor: Int32
    private let completion = DispatchGroup()
    private var failure: Error?

    init(descriptor: Int32, name: String, expectedSHA256: String?) {
        self.descriptor = descriptor
        guard let expectedSHA256 else { return }
        completion.enter()
        DispatchQueue.global(qos: .userInitiated).async { [self] in
            do {
                try Sha256Verifier.verifyFile(fileDescriptor: descriptor,
                                               named: name, expectedHex: expectedSHA256)
            } catch { failure = error }
            completion.leave()
        }
    }

    func waitUntilVerified() throws {
        // The group establishes the happens-before edge for failure. There
        // is one writer, and no reader accesses it before this wait.
        completion.wait()
        if let failure { throw failure }
    }

    deinit { close(descriptor) }
}
