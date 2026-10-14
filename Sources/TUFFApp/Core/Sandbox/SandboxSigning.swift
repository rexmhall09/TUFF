#if os(macOS)
import CryptoKit
import Foundation
import Security

/// The signatures of the sandbox API, as written in the docstring of
/// Sandbox/server.py. Every request carries one and every answer must carry
/// one, so a program that merely listens on the sandbox's port cannot pass
/// for the sandbox. `Sandbox/test-vectors/signing.json` holds fixed examples
/// that this code and the server both reproduce.
enum SandboxSigning {
    static let requestLabel = "tuff-sandbox-req-v1"
    static let responseLabel = "tuff-sandbox-resp-v1"

    /// `X-Sandbox-Auth` of a request: lowercase hex of the HMAC-SHA256.
    static func requestAuth(key: Data, method: String, path: String, nonce: String,
                            time: String, body: Data) -> String {
        hex(mac(key: key, message: [requestLabel, method, path, nonce, time, sha256Hex(body)]))
    }

    /// `X-Sandbox-Auth` an answer to that request must carry.
    static func responseAuth(key: Data, method: String, path: String, nonce: String,
                             status: Int, body: Data) -> String {
        hex(mac(key: key, message: [responseLabel, method, path, nonce, String(status), sha256Hex(body)]))
    }

    /// Whether `given` is the signature of this answer. The comparison does
    /// not stop at the first different byte.
    static func isValidResponse(_ given: String, key: Data, method: String, path: String,
                                nonce: String, status: Int, body: Data) -> Bool {
        guard let decoded = bytes(fromHex: given), decoded.count == SHA256.byteCount else { return false }
        let message = [responseLabel, method, path, nonce, String(status), sha256Hex(body)]
            .joined(separator: "\n")
        return HMAC<SHA256>.isValidAuthenticationCode(
            decoded, authenticating: Data(message.utf8), using: SymmetricKey(data: key))
    }

    /// 64 random hexadecimal characters, new for every request.
    static func newNonce() throws -> String {
        var bytes = [UInt8](repeating: 0, count: 32)
        guard SecRandomCopyBytes(kSecRandomDefault, bytes.count, &bytes) == errSecSuccess else {
            throw SandboxError.secretUnavailable("no random numbers")
        }
        return hex(bytes)
    }

    static func sha256Hex(_ data: Data) -> String {
        hex(Array(SHA256.hash(data: data)))
    }

    private static func mac(key: Data, message: [String]) -> [UInt8] {
        let code = HMAC<SHA256>.authenticationCode(
            for: Data(message.joined(separator: "\n").utf8), using: SymmetricKey(data: key))
        return Array(code)
    }

    private static func hex(_ bytes: [UInt8]) -> String {
        bytes.map { String(format: "%02x", $0) }.joined()
    }

    private static func bytes(fromHex text: String) -> [UInt8]? {
        let characters = Array(text.utf8)
        guard characters.count % 2 == 0 else { return nil }
        var result: [UInt8] = []
        result.reserveCapacity(characters.count / 2)
        var index = 0
        while index < characters.count {
            guard let high = nibble(characters[index]), let low = nibble(characters[index + 1]) else {
                return nil
            }
            result.append(high << 4 | low)
            index += 2
        }
        return result
    }

    private static func nibble(_ character: UInt8) -> UInt8? {
        if character >= UInt8(ascii: "0"), character <= UInt8(ascii: "9") {
            return character - UInt8(ascii: "0")
        }
        if character >= UInt8(ascii: "a"), character <= UInt8(ascii: "f") {
            return character - UInt8(ascii: "a") + 10
        }
        if character >= UInt8(ascii: "A"), character <= UInt8(ascii: "F") {
            return character - UInt8(ascii: "A") + 10
        }
        return nil
    }
}

/// What `GET /health` reports.
struct SandboxHealth: Equatable, Sendable {
    let profile: String
    let api: Int
    /// How many Mac addresses the server was told to refuse.
    let blockedAddresses: Int
}

/// Asks a running sandbox whether it answers, with a signed request. Tests
/// replace it with a fake.
protocol SandboxHealthProbing: Sendable {
    /// Throws if nothing answers, if the answer is not signed with `key`, or
    /// if it is not what a web sandbox of this API version says.
    func health(port: Int, key: Data) async throws -> SandboxHealth
}

struct SandboxHealthProbe: SandboxHealthProbing {
    /// Larger than any answer of `/health`. A server that sends more is not
    /// the sandbox.
    static let maximumBytes = 64 * 1_024

    init() {}

    func health(port: Int, key: Data) async throws -> SandboxHealth {
        guard let url = URL(string: "http://127.0.0.1:\(port)/health") else {
            throw SandboxError.notHealthy("bad address")
        }
        let nonce = try SandboxSigning.newNonce()
        let time = String(Int(Date().timeIntervalSince1970))
        var request = URLRequest(url: url, cachePolicy: .reloadIgnoringLocalAndRemoteCacheData,
                                 timeoutInterval: 3)
        request.httpMethod = "GET"
        request.setValue(nonce, forHTTPHeaderField: "X-Sandbox-Nonce")
        request.setValue(time, forHTTPHeaderField: "X-Sandbox-Time")
        request.setValue(
            SandboxSigning.requestAuth(key: key, method: "GET", path: "/health", nonce: nonce,
                                       time: time, body: Data()),
            forHTTPHeaderField: "X-Sandbox-Auth")
        let configuration = URLSessionConfiguration.ephemeral
        configuration.connectionProxyDictionary = [:]
        configuration.timeoutIntervalForRequest = 3
        configuration.timeoutIntervalForResource = 5
        configuration.httpCookieStorage = nil
        configuration.urlCache = nil
        let session = URLSession(configuration: configuration)
        defer { session.finishTasksAndInvalidate() }
        let data: Data
        let response: URLResponse
        do {
            let (bytes, received) = try await session.bytes(for: request)
            var collected = Data()
            for try await byte in bytes {
                collected.append(byte)
                if collected.count > Self.maximumBytes {
                    throw SandboxError.notHealthy("the answer is far too long")
                }
            }
            data = collected
            response = received
        } catch let error as SandboxError {
            throw error
        } catch {
            throw SandboxError.notHealthy(SandboxText.oneLine(error.localizedDescription))
        }
        guard let http = response as? HTTPURLResponse else {
            throw SandboxError.notHealthy("not an HTTP answer")
        }
        return try Self.check(status: http.statusCode,
                              signature: http.value(forHTTPHeaderField: "X-Sandbox-Auth"),
                              body: data, nonce: nonce, key: key)
    }

    /// The signature first, then the content: nothing in an unsigned answer
    /// is read.
    static func check(status: Int, signature: String?, body: Data, nonce: String,
                      key: Data) throws -> SandboxHealth {
        guard let signature, SandboxSigning.isValidResponse(
            signature, key: key, method: "GET", path: "/health", nonce: nonce,
            status: status, body: body) else {
            throw SandboxError.notHealthy("the answer is not signed with this start's secret")
        }
        guard status == 200 else {
            throw SandboxError.notHealthy("status \(status)")
        }
        struct Answer: Decodable {
            let status: String
            let profile: String
            let api: Int
            let blocked_addresses: Int
        }
        guard let answer = try? JSONDecoder().decode(Answer.self, from: body),
              answer.status == "ok" else {
            throw SandboxError.notHealthy("unexpected content")
        }
        guard answer.profile == "web", answer.api == 1 else {
            throw SandboxError.notHealthy(
                "a different kind of sandbox (\(SandboxText.oneLine(answer.profile, limit: 20)), API \(answer.api))")
        }
        return SandboxHealth(profile: answer.profile, api: answer.api,
                             blockedAddresses: answer.blocked_addresses)
    }
}
#endif
