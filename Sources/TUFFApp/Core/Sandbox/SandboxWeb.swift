#if os(macOS)
import Foundation

/// The one way web code reaches the running sandbox. It takes a lease, waits
/// a while if the sandbox is still starting, and gives the lease back. It
/// never offers another path to the web: if the sandbox is not ready, the
/// request fails with a message. The one thing it may start is a sandbox
/// that was stopped for being idle, and only while the person still has the
/// setting on (`mayRestart`).
struct SandboxWebAccess: Sendable {
    let controller: SandboxController
    /// How long a request waits for a sandbox that is still starting.
    var startupWait: Duration = .seconds(60)
    var pollInterval: Duration = .milliseconds(250)
    var sleep: @Sendable (Duration) async throws -> Void = { try await Task.sleep(for: $0) }
    var makeClient: @Sendable (SandboxLease) -> SandboxClient = { SandboxClient(lease: $0) }
    /// Whether a sandbox that is off (stopped when idle) may be started again
    /// for this request: true only while the setting is on.
    var mayRestart: @Sendable () async -> Bool = { false }

    /// Runs `body` with a client for the running sandbox. With
    /// `confirmFirst` the sandbox is asked, with a signed request, whether it
    /// still answers before `body` runs: use it for requests that carry a
    /// secret, so a key never goes to a port that something else may hold.
    /// Without it the check still runs when the sandbox has not given a signed
    /// answer for a few seconds (`SandboxDependencies.proofValidity`), so the
    /// first request after a quiet time, or after the sandbox died, is never
    /// sent to a port that something else may have taken.
    /// If a request fails because the sandbox did not answer or its answer was
    /// not signed, the sandbox is checked again and, if it fails that, stopped
    /// and marked failed, so the next message starts a fresh one.
    func withClient<T: Sendable>(
        confirmFirst: Bool, _ body: @Sendable (SandboxClient) async throws -> T
    ) async throws -> T {
        let lease = try await acquireWaitingForStart()
        let client = makeClient(lease)
        defer { client.invalidate() }
        do {
            let stale = await controller.needsProof(lease)
            if confirmFirst || stale {
                try await controller.confirmRunning(lease)
                // The sandbox can be stopped between the check and the
                // request; nothing is sent to a port that is no longer ours.
                guard await controller.isCurrent(lease) else { throw SandboxError.notReady }
            }
            let result = try await body(client)
            await controller.noteAnswered(lease)
            await controller.release(lease)
            return result
        } catch {
            if !(error is CancellationError), !Task.isCancelled, Self.suggestsDeadSandbox(error) {
                try? await controller.confirmRunning(lease)
            }
            await controller.release(lease)
            throw error
        }
    }

    private static func suggestsDeadSandbox(_ error: Error) -> Bool {
        switch error as? SandboxClientError {
        case .unreachable, .notSigned, .tooLarge, .malformed: true
        // A signed refusal because the clocks differ (the VM's clock can fall
        // behind while the Mac sleeps): this sandbox will refuse everything,
        // so it is stopped and the next message starts a fresh one.
        case .remote(_, "stale_request", _): true
        default: false
        }
    }

    private func acquireWaitingForStart() async throws -> SandboxLease {
        var waited = Duration.zero
        var restartedAt: Duration?
        while true {
            do {
                return try await controller.acquire()
            } catch SandboxError.stillStarting {
                guard waited < startupWait else { throw SandboxError.stillStarting }
                try await sleep(pollInterval)
                waited += pollInterval
            } catch SandboxError.notReady {
                let state = await controller.state
                // A restart was just asked for and has not begun yet.
                let startPending = restartedAt.map { waited < $0 + .seconds(2) } ?? false
                if (state == .stopping || (state == .off && startPending)), waited < startupWait {
                    // An idle stop is finishing: wait for it, then look again.
                    try await sleep(pollInterval)
                    waited += pollInterval
                    continue
                }
                // Off after an idle stop: start again once if the person
                // still wants the sandbox. A failed start stays failed.
                guard restartedAt == nil, state == .off, await mayRestart() else {
                    throw SandboxError.notReady
                }
                restartedAt = waited
                // The start survives this request being stopped; the loop
                // then waits for it like for any start under way.
                let controller = controller
                _ = Task { await controller.start() }
                try await sleep(pollInterval)
                waited += pollInterval
            }
        }
    }
}

/// How sandbox failures read to the rest of the app: as the errors the web
/// code already shows, with the sandbox's own sentence when there is no
/// closer match. Nothing here falls back to a direct request.
enum SandboxWebFailure {
    static func http(_ error: Error, host: String, byteLimit: Int) -> Error {
        if error is CancellationError { return error }
        if let error = error as? AppHTTPError { return error }
        if let error = error as? SandboxError { return AppHTTPError.sandbox(sentence(error.description)) }
        guard let error = error as? SandboxClientError else {
            return AppHTTPError.sandbox(sentence(SandboxText.oneLine(error.localizedDescription)))
        }
        switch error {
        case .remote(_, let code, let message):
            switch code {
            case "private_address" where message.hasPrefix(host):
                return AppHTTPError.privateAddress(host)
            case "timed_out": return AppHTTPError.timedOut
            case "oversized": return AppHTTPError.oversized(limit: byteLimit)
            case "fetch_failed":
                return AppHTTPError.sandbox("The sandbox could not get an answer from the site.")
            case "dns_error": return AppHTTPError.network("The host could not be resolved.")
            case "insecure_redirect" where message.contains("https to http"):
                return AppHTTPError.insecureRedirect
            case "too_many_redirects": return AppHTTPError.tooManyRedirects
            default: return AppHTTPError.sandbox(sentence("The sandbox refused the request: \(message)"))
            }
        case .notSigned, .tooLarge, .unreachable, .malformed:
            return AppHTTPError.sandbox(sentence(error.description))
        }
    }

    private static func sentence(_ text: String) -> String {
        guard let last = text.last, !".!?".contains(last) else { return text }
        return text + "."
    }
}

/// `AppHTTPTransport` for the search providers: each request goes to the
/// sandbox's `/v1/http`, which reaches only the three search sites.
struct SandboxHTTPTransport: AppHTTPTransport {
    static let maximumFinalURLCharacters = 2_048
    let access: SandboxWebAccess

    private struct Payload: Encodable {
        let url: String
        let method: String
        let headers: [String: String]?
        let bodyBase64: String?
        let maxBytes: Int
        let timeoutSeconds: Double

        enum CodingKeys: String, CodingKey {
            case url, method, headers
            case bodyBase64 = "body_base64"
            case maxBytes = "max_bytes"
            case timeoutSeconds = "timeout_seconds"
        }
    }

    private struct Answer: Decodable {
        let finalURL: String
        let status: Int
        let contentType: String
        let bodyBase64: String

        enum CodingKeys: String, CodingKey {
            case finalURL = "final_url"
            case status
            case contentType = "content_type"
            case bodyBase64 = "body_base64"
        }
    }

    func perform(_ request: AppHTTPRequest) async throws -> AppHTTPResponse {
        guard let host = request.url.host, request.url.scheme?.lowercased() == "https" else {
            throw AppHTTPError.unsupportedScheme
        }
        let timeout = min(max(request.timeoutSeconds, 1), 60)
        let payload = Payload(
            url: request.url.absoluteString, method: request.method.uppercased(),
            headers: request.headers.isEmpty ? nil : request.headers,
            bodyBase64: request.body.map { $0.base64EncodedString() },
            maxBytes: request.maximumBytes, timeoutSeconds: timeout)
        // A key travels only in these two headers.
        let carriesSecret = request.headers.keys.contains {
            ["authorization", "x-subscription-token"].contains($0.lowercased())
        }
        let limit = (request.maximumBytes + 2) / 3 * 4 + 64 * 1_024
        let data: Data
        do {
            data = try await access.withClient(confirmFirst: carriesSecret) { client in
                try await client.post("/v1/http", body: payload, answerLimit: limit,
                                      timeout: timeout + 10)
            }
        } catch {
            throw SandboxWebFailure.http(error, host: host, byteLimit: request.maximumBytes)
        }
        let answer: Answer
        do {
            answer = try JSONDecoder().decode(Answer.self, from: data)
        } catch {
            throw AppHTTPError.sandbox("The sandbox's answer was not what was expected.")
        }
        guard answer.finalURL.count <= Self.maximumFinalURLCharacters,
              let finalURL = URL(string: answer.finalURL), finalURL.scheme?.lowercased() == "https",
              finalURL.host != nil, (100...599).contains(answer.status),
              let body = Data(base64Encoded: answer.bodyBase64) else {
            throw AppHTTPError.sandbox("The sandbox's answer was not what was expected.")
        }
        guard body.count <= request.maximumBytes else {
            throw AppHTTPError.oversized(limit: request.maximumBytes)
        }
        return AppHTTPResponse(
            finalURL: finalURL, statusCode: answer.status,
            contentType: answer.contentType.isEmpty ? nil : answer.contentType, body: body)
    }
}

/// Reads one page through the sandbox's `/v1/page`: the sandbox fetches it
/// and turns the HTML or PDF into text, and the app gets only that text.
struct SandboxPageReader: Sendable {
    static let maximumCharacters = 200_000
    static let maximumTitleCharacters = 300
    static let maximumFinalURLCharacters = 2_048

    let access: SandboxWebAccess
    let timeoutSeconds: Double

    private struct Payload: Encodable {
        let url: String
        let maxBytes: Int
        let timeoutSeconds: Double
        let maxCharacters: Int

        enum CodingKeys: String, CodingKey {
            case url
            case maxBytes = "max_bytes"
            case timeoutSeconds = "timeout_seconds"
            case maxCharacters = "max_characters"
        }
    }

    private struct Answer: Decodable {
        let finalURL: String
        let status: Int
        let title: String
        let text: String

        enum CodingKeys: String, CodingKey {
            case finalURL = "final_url"
            case status, title, text
        }
    }

    func read(_ url: URL) async throws -> AppWebPage {
        guard let host = url.host, let scheme = url.scheme?.lowercased(),
              scheme == "http" || scheme == "https" else {
            throw AppWebPageError.transport(.unsupportedScheme)
        }
        let timeout = min(max(timeoutSeconds, 1), 60)
        let byteLimit = AppWebPageReader.maximumBytes
        let payload = Payload(url: url.absoluteString, maxBytes: byteLimit, timeoutSeconds: timeout,
                              maxCharacters: Self.maximumCharacters)
        // The text is at most maximumCharacters; JSON may spell each in up to
        // twelve bytes, and the rest of the answer is small.
        let limit = Self.maximumCharacters * 12 + 64 * 1_024
        let data: Data
        do {
            // The sandbox reads for up to the timeout plus its processing time.
            data = try await access.withClient(confirmFirst: false) { client in
                try await client.post("/v1/page", body: payload, answerLimit: limit,
                                      timeout: timeout + 40)
            }
        } catch SandboxClientError.remote(_, "unsupported_content", let message)
                    where !message.hasPrefix("unsupported content encoding")
                        && !message.hasPrefix("response body could not be decompressed")
                        && !message.hasPrefix("this sandbox cannot read PDFs") {
            throw AppWebPageError.unsupportedContent("another kind of file")
        } catch {
            let mapped = SandboxWebFailure.http(error, host: host, byteLimit: byteLimit)
            if mapped is CancellationError { throw mapped }
            throw AppWebPageError.transport(mapped as? AppHTTPError ?? .sandbox("\(mapped)"))
        }
        let answer: Answer
        do {
            answer = try JSONDecoder().decode(Answer.self, from: data)
        } catch {
            throw AppWebPageError.transport(.sandbox("The sandbox's answer was not what was expected."))
        }
        guard answer.finalURL.count <= Self.maximumFinalURLCharacters,
              let finalURL = URL(string: answer.finalURL), finalURL.host != nil,
              let finalScheme = finalURL.scheme?.lowercased(),
              finalScheme == "http" || finalScheme == "https", (100...599).contains(answer.status) else {
            throw AppWebPageError.transport(.sandbox("The sandbox's answer was not what was expected."))
        }
        guard (200..<300).contains(answer.status) else { throw AppWebPageError.http(answer.status) }
        let text = SandboxWebText.clean(answer.text, limit: Self.maximumCharacters)
            .trimmingCharacters(in: .whitespacesAndNewlines)
        guard !text.isEmpty else { throw AppWebPageError.empty }
        let title = SandboxWebText.clean(answer.title, limit: Self.maximumTitleCharacters)
            .replacingOccurrences(of: "\n", with: " ")
            .trimmingCharacters(in: .whitespacesAndNewlines)
        return AppWebPage(finalURL: finalURL, title: title.isEmpty ? nil : title, text: text)
    }
}

/// Text from the sandbox is data, not trusted: the sandbox already strips
/// control and invisible characters, and the app does it again before the
/// text can reach the model or the screen.
enum SandboxWebText {
    static func clean(_ text: String, limit: Int) -> String {
        var cleaned = String.UnicodeScalarView()
        var count = 0
        for scalar in text.unicodeScalars {
            if count >= limit { break }
            switch scalar.properties.generalCategory {
            case .control:
                guard scalar == "\n" || scalar == "\t" else { continue }
                cleaned.append(scalar)
            case .lineSeparator, .paragraphSeparator:
                cleaned.append("\n")
            case .format, .surrogate, .privateUse:
                continue
            default:
                cleaned.append(scalar)
            }
            count += 1
        }
        return String(cleaned)
    }
}
#endif
