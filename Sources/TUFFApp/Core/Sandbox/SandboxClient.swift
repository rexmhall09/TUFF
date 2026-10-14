#if os(macOS)
import Foundation

/// What a sandbox request can go wrong with. Every case but `remote` means
/// the answer cannot be trusted or was not received; `remote` is the
/// sandbox's own signed refusal or failure.
enum SandboxClientError: Error, Equatable, Sendable, CustomStringConvertible {
    /// No signature, or one made with another secret or for another request.
    case notSigned
    /// More bytes than the caller allowed, so reading stopped.
    case tooLarge(limit: Int)
    /// Nothing answered, or the connection failed.
    case unreachable(String)
    /// The sandbox answered, signed, with an error: its code (such as
    /// `private_address` or `timed_out`), message and HTTP status.
    case remote(status: Int, code: String, message: String)
    /// Signed, but not the answer this request should get.
    case malformed(String)

    var description: String {
        switch self {
        case .notSigned:
            "The sandbox's answer was not signed with this start's secret, so it was not used."
        case .tooLarge(let limit):
            "The sandbox's answer was larger than \(limit / 1_024) KB, so reading stopped."
        case .unreachable(let detail):
            "The sandbox did not answer: \(detail)"
        case .remote(_, _, let message):
            message
        case .malformed(let detail):
            "The sandbox's answer was not what was expected (\(detail))."
        }
    }
}

/// The sandbox never redirects. If something on its port does, the redirect
/// is not followed: the app does not connect anywhere else.
private final class RefuseRedirects: NSObject, URLSessionTaskDelegate, @unchecked Sendable {
    func urlSession(_ session: URLSession, task: URLSessionTask,
                    willPerformHTTPRedirection response: HTTPURLResponse,
                    newRequest request: URLRequest) async -> URLRequest? { nil }
}

/// One signed request to the running sandbox and its checked answer. The
/// signature of the answer is verified before anything else about it is read
/// or believed, including its status.
struct SandboxClient: Sendable {
    let port: Int
    let key: Data
    private let session: URLSession

    /// `protocolClasses` lets tests answer without a network.
    init(port: Int, key: Data, protocolClasses: [AnyClass]? = nil) {
        self.port = port
        self.key = key
        let configuration = URLSessionConfiguration.ephemeral
        if let protocolClasses { configuration.protocolClasses = protocolClasses }
        // Never through a proxy configured on the Mac, and no cookies,
        // credentials or cache for a local, signed conversation.
        configuration.connectionProxyDictionary = [:]
        configuration.httpCookieStorage = nil
        configuration.httpShouldSetCookies = false
        configuration.urlCache = nil
        configuration.urlCredentialStorage = nil
        configuration.requestCachePolicy = .reloadIgnoringLocalAndRemoteCacheData
        configuration.httpMaximumConnectionsPerHost = 4
        session = URLSession(configuration: configuration)
    }

    init(lease: SandboxLease, protocolClasses: [AnyClass]? = nil) {
        self.init(port: lease.port, key: lease.key, protocolClasses: protocolClasses)
    }

    /// Ends this client's session once its running requests are done. Call
    /// it when the client is no longer needed.
    func invalidate() { session.finishTasksAndInvalidate() }

    /// POSTs `body` as JSON to `path` and returns the signed 200 answer's
    /// body. `answerLimit` bounds the bytes read; `timeout` the whole call.
    func post<Body: Encodable & Sendable>(_ path: String, body: Body, answerLimit: Int,
                               timeout: TimeInterval) async throws -> Data {
        try await withThrowingTaskGroup(of: Data?.self) { group in
            group.addTask { try await self.exchange(path, body: body, answerLimit: answerLimit, timeout: timeout) }
            group.addTask {
                try await Task.sleep(for: .seconds(timeout))
                return nil
            }
            defer { group.cancelAll() }
            guard let first = try await group.next(), let data = first else {
                throw SandboxClientError.unreachable("no answer within \(Int(timeout)) seconds")
            }
            return data
        }
    }

    private func exchange<Body: Encodable & Sendable>(_ path: String, body: Body, answerLimit: Int,
                                           timeout: TimeInterval) async throws -> Data {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys, .withoutEscapingSlashes]
        let payload = try encoder.encode(body)
        guard let url = URL(string: "http://127.0.0.1:\(port)\(path)") else {
            throw SandboxClientError.malformed("bad address")
        }
        let nonce = try SandboxSigning.newNonce()
        let time = String(Int(Date().timeIntervalSince1970))
        var request = URLRequest(url: url, cachePolicy: .reloadIgnoringLocalAndRemoteCacheData,
                                 timeoutInterval: timeout)
        request.httpMethod = "POST"
        request.httpBody = payload
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.setValue(nonce, forHTTPHeaderField: "X-Sandbox-Nonce")
        request.setValue(time, forHTTPHeaderField: "X-Sandbox-Time")
        request.setValue(
            SandboxSigning.requestAuth(key: key, method: "POST", path: path, nonce: nonce,
                                       time: time, body: payload),
            forHTTPHeaderField: "X-Sandbox-Auth")

        let data: Data
        let response: URLResponse
        do {
            let (bytes, received) = try await session.bytes(for: request, delegate: RefuseRedirects())
            if let http = received as? HTTPURLResponse,
               let declared = http.value(forHTTPHeaderField: "Content-Length").flatMap(Int.init),
               declared > answerLimit {
                throw SandboxClientError.tooLarge(limit: answerLimit)
            }
            var collected = [UInt8]()
            collected.reserveCapacity(min(answerLimit, 64 * 1_024))
            for try await byte in bytes {
                collected.append(byte)
                if collected.count > answerLimit { throw SandboxClientError.tooLarge(limit: answerLimit) }
            }
            data = Data(collected)
            response = received
        } catch let error as SandboxClientError {
            throw error
        } catch is CancellationError {
            throw CancellationError()
        } catch let error as URLError where error.code == .cancelled {
            throw CancellationError()
        } catch {
            throw SandboxClientError.unreachable(SandboxText.oneLine(error.localizedDescription))
        }
        guard let http = response as? HTTPURLResponse else {
            throw SandboxClientError.malformed("not an HTTP answer")
        }
        // The signature first: nothing in an unsigned answer is read.
        guard let signature = http.value(forHTTPHeaderField: "X-Sandbox-Auth"),
              SandboxSigning.isValidResponse(signature, key: key, method: "POST", path: path,
                                             nonce: nonce, status: http.statusCode, body: data) else {
            throw SandboxClientError.notSigned
        }
        guard http.statusCode == 200 else {
            throw Self.remoteError(status: http.statusCode, body: data)
        }
        return data
    }

    /// The sandbox's signed error answer: `{"error": {"message", "code"}}`.
    static func remoteError(status: Int, body: Data) -> SandboxClientError {
        struct Failure: Decodable {
            struct Detail: Decodable { let message: String?; let code: String? }
            let error: Detail
        }
        let detail = (try? JSONDecoder().decode(Failure.self, from: body))?.error
        let code = SandboxText.oneLine(detail?.code ?? "error", limit: 40)
        let message = SandboxText.oneLine(detail?.message ?? "status \(status)", limit: 300)
        return .remote(status: status, code: code, message: message)
    }
}
#endif
