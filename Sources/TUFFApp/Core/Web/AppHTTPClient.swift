import Darwin
import Foundation

public struct AppHTTPRequest: Equatable, Sendable {
    public var url: URL
    public var method: String
    public var headers: [String: String]
    public var body: Data?
    public var maximumBytes: Int
    public var timeoutSeconds: Double

    public init(url: URL, method: String = "GET", headers: [String: String] = [:],
                body: Data? = nil, maximumBytes: Int, timeoutSeconds: Double) {
        self.url = url
        self.method = method
        self.headers = headers
        self.body = body
        self.maximumBytes = maximumBytes
        self.timeoutSeconds = timeoutSeconds
    }
}

public struct AppHTTPResponse: Equatable, Sendable {
    public var finalURL: URL
    public var statusCode: Int
    public var contentType: String?
    public var body: Data

    public init(finalURL: URL, statusCode: Int, contentType: String? = nil, body: Data) {
        self.finalURL = finalURL
        self.statusCode = statusCode
        self.contentType = contentType
        self.body = body
    }

    /// The charset named in Content-Type, lowercased.
    public var charset: String? {
        contentType?.split(separator: ";").dropFirst().lazy
            .map { $0.trimmingCharacters(in: .whitespaces) }
            .first { $0.lowercased().hasPrefix("charset=") }
            .map { String($0.dropFirst(8)).trimmingCharacters(in: CharacterSet(charactersIn: "\"'")).lowercased() }
    }

    /// Body as text: the declared charset, then UTF-8, then Latin-1, which
    /// decodes any byte sequence.
    public var text: String {
        if let charset, charset != "utf-8", charset != "utf8" {
            let encoding = CFStringConvertEncodingToNSStringEncoding(
                CFStringConvertIANACharSetNameToEncoding(charset as CFString))
            if encoding != UInt(kCFStringEncodingInvalidId),
               let decoded = String(data: body, encoding: String.Encoding(rawValue: encoding)) {
                return decoded
            }
        }
        return String(data: body, encoding: .utf8)
            ?? String(data: body, encoding: .isoLatin1) ?? ""
    }
}

public enum AppHTTPError: Error, Equatable, Sendable, CustomStringConvertible {
    case unsupportedScheme
    case privateAddress(String)
    case tooManyRedirects
    case insecureRedirect
    case timedOut
    case oversized(limit: Int)
    case cancelled
    /// curl failures before an HTTP request can be sent. Safe to try another checked address.
    case connectionFailed(code: Int32)
    case network(String)
    /// The web sandbox refused or could not carry the request. The text is
    /// written for the person and is shown as it is.
    case sandbox(String)

    public var description: String {
        switch self {
        case .unsupportedScheme: "Only http and https addresses can be read."
        case .privateAddress(let host):
            "\(host) is a local or private network address, which TUFF does not read."
        case .tooManyRedirects: "The address redirected too many times."
        case .insecureRedirect: "The address redirected from https to an insecure http address."
        case .timedOut: "The request timed out."
        case .oversized(let limit):
            "The response was larger than \(Self.sizeDescription(limit)), so reading stopped."
        case .cancelled: "Stopped."
        case .connectionFailed(let code):
            code == 35
                ? "The secure connection failed during the TLS handshake (code 35). The site or network may be refusing the connection. Try again on another network."
                : "The connection to the site could not be established (code \(code)). Check your connection and try again."
        case .network(let message): "The request failed: \(message)"
        case .sandbox(let message): message
        }
    }

    /// Whole mebibytes read as "2 MB", matching how the limits are documented;
    /// anything else falls back to kibibytes.
    static func sizeDescription(_ bytes: Int) -> String {
        let mebibyte = 1_024 * 1_024
        if bytes >= mebibyte, bytes % mebibyte == 0 { return "\(bytes / mebibyte) MB" }
        return "\(bytes / 1_024) KB"
    }
}

/// Performs one bounded request. Production uses a pinned-address transport;
/// tests substitute fixtures.
public protocol AppHTTPTransport: Sendable {
    func perform(_ request: AppHTTPRequest) async throws -> AppHTTPResponse
}

/// Which hosts the app's web reader may contact. Loopback, link-local,
/// private, carrier-grade NAT and unique-local addresses are refused, both
/// as literals and when a name resolves to one, so a page or a model cannot
/// point the reader at the user's router or another local service. The
/// production transport connects only to an address from this checked set.
public enum AppNetworkPolicy {
    public static func isPublicHost(_ host: String) -> Bool {
        let lowered = host.lowercased().trimmingCharacters(in: CharacterSet(charactersIn: "[]."))
        if lowered.isEmpty || lowered == "localhost" || lowered.hasSuffix(".localhost")
            || lowered.hasSuffix(".local") || lowered.hasSuffix(".internal")
            || lowered.hasSuffix(".home.arpa") || !lowered.contains(".") && !lowered.contains(":") {
            return false
        }
        if let address = IPAddress(lowered) { return address.isPublic }
        return true
    }

    /// Resolves `host` and refuses it if any address is not public.
    public static func resolvesToPublicAddresses(_ host: String) -> Bool {
        guard isPublicHost(host) else { return false }
        if IPAddress(host.lowercased().trimmingCharacters(in: CharacterSet(charactersIn: "[]"))) != nil {
            return true
        }
        var hints = addrinfo()
        hints.ai_socktype = SOCK_STREAM
        var list: UnsafeMutablePointer<addrinfo>?
        guard getaddrinfo(host, nil, &hints, &list) == 0, let first = list else {
            // An unresolvable name cannot be fetched anyway; let the request
            // report the failure.
            return true
        }
        defer { freeaddrinfo(list) }
        var cursor: UnsafeMutablePointer<addrinfo>? = first
        while let entry = cursor {
            if let address = IPAddress(sockaddr: entry.pointee.ai_addr), !address.isPublic {
                return false
            }
            cursor = entry.pointee.ai_next
        }
        return true
    }

    struct IPAddress {
        let bytes: [UInt8]

        init?(_ text: String) {
            var v4 = in_addr()
            var v6 = in6_addr()
            if inet_pton(AF_INET, text, &v4) == 1 {
                bytes = withUnsafeBytes(of: v4) { Array($0) }
            } else if inet_pton(AF_INET6, text, &v6) == 1 {
                bytes = withUnsafeBytes(of: v6) { Array($0) }
            } else {
                return nil
            }
        }

        init?(sockaddr pointer: UnsafeMutablePointer<sockaddr>?) {
            guard let pointer else { return nil }
            switch Int32(pointer.pointee.sa_family) {
            case AF_INET:
                bytes = pointer.withMemoryRebound(to: sockaddr_in.self, capacity: 1) {
                    withUnsafeBytes(of: $0.pointee.sin_addr) { Array($0) }
                }
            case AF_INET6:
                bytes = pointer.withMemoryRebound(to: sockaddr_in6.self, capacity: 1) {
                    withUnsafeBytes(of: $0.pointee.sin6_addr) { Array($0) }
                }
            default:
                return nil
            }
        }

        var isPublic: Bool {
            if bytes.count == 4 { return Self.isPublicV4(bytes) }
            guard bytes.count == 16 else { return false }
            if bytes.allSatisfy({ $0 == 0 }) { return false }                     // ::
            if bytes[0..<15].allSatisfy({ $0 == 0 }) && bytes[15] == 1 { return false } // ::1
            if bytes[0] & 0xfe == 0xfc { return false }                          // fc00::/7
            if bytes[0] == 0xfe && bytes[1] & 0xc0 == 0x80 { return false }      // fe80::/10
            if bytes[0] == 0xff { return false }                                  // multicast
            // IPv4-mapped and -compatible addresses carry an IPv4 address.
            if bytes[0..<10].allSatisfy({ $0 == 0 }),
               (bytes[10] == 0xff && bytes[11] == 0xff) || (bytes[10] == 0 && bytes[11] == 0) {
                return Self.isPublicV4(Array(bytes[12..<16]))
            }
            // Global unicast only. This also excludes site-local addresses
            // and translation prefixes that can embed a private IPv4 target.
            guard bytes[0] & 0xe0 == 0x20 else { return false }                 // 2000::/3
            if bytes[0] == 0x20 && bytes[1] == 0x01 {
                if bytes[2] == 0x0d && bytes[3] == 0xb8 { return false }         // documentation
                if bytes[2] == 0 && bytes[3] == 0 { return false }              // Teredo
                if bytes[2] == 0 && bytes[3] == 2 { return false }              // benchmarking
            }
            if bytes[0] == 0x20 && bytes[1] == 0x02 {                           // 6to4
                return Self.isPublicV4(Array(bytes[2..<6]))
            }
            return true
        }

        private static func isPublicV4(_ b: [UInt8]) -> Bool {
            switch (b[0], b[1]) {
            case (0, _), (10, _), (127, _): return false
            case (100, 64...127): return false        // carrier-grade NAT
            case (169, 254): return false             // link-local
            case (172, 16...31): return false
            case (192, 168): return false
            case (192, 0) where b[2] == 0: return false
            case (198, 18...19): return false         // benchmarking
            default: return b[0] < 224               // multicast and reserved
            }
        }
    }
}

/// URLSession without cookies, credentials or a cache, with a redirect limit
/// and a byte limit enforced while reading.
public final class URLSessionHTTPTransport: NSObject, AppHTTPTransport, @unchecked Sendable {
    public static let maximumRedirects = 5
    public static let userAgent = "TUFF/8.0 (macOS; local assistant; +https://github.com/rexmhall09/TUFF)"

    private let session: URLSession
    private let checksHosts: Bool

    /// `protocolClasses` lets tests answer requests without a network.
    public init(checksHosts: Bool = true, protocolClasses: [AnyClass]? = nil) {
        let configuration = URLSessionConfiguration.ephemeral
        if let protocolClasses { configuration.protocolClasses = protocolClasses }
        configuration.httpCookieStorage = nil
        configuration.httpShouldSetCookies = false
        configuration.urlCache = nil
        configuration.requestCachePolicy = .reloadIgnoringLocalCacheData
        configuration.urlCredentialStorage = nil
        configuration.httpMaximumConnectionsPerHost = 2
        session = URLSession(configuration: configuration)
        self.checksHosts = checksHosts
    }

    public func perform(_ request: AppHTTPRequest) async throws -> AppHTTPResponse {
        if checksHosts {
            #if os(macOS)
            return try await PinnedHTTPTransport().perform(request)
            #else
            throw AppHTTPError.network("Public-address pinning is not available on this platform.")
            #endif
        }
        try Self.checkScheme(request.url)
        if checksHosts { try await Self.checkHost(request.url) }
        var urlRequest = URLRequest(url: request.url, timeoutInterval: request.timeoutSeconds)
        urlRequest.httpMethod = request.method
        urlRequest.httpBody = request.body
        urlRequest.setValue(Self.userAgent, forHTTPHeaderField: "User-Agent")
        for (key, value) in request.headers { urlRequest.setValue(value, forHTTPHeaderField: key) }

        let delegate = RedirectGuard(checksHosts: checksHosts)
        let deadline = ContinuousClock.now + .seconds(request.timeoutSeconds)
        let prepared = urlRequest
        // Cancelling the calling task cancels the group, whose child ends the
        // URLSession task; a deadline child bounds the whole request.
        return try await withThrowingTaskGroup(of: AppHTTPResponse?.self) { group in
                group.addTask { [session] in
                    let (bytes, response) = try await session.bytes(for: prepared, delegate: delegate)
                    if let failure = delegate.failure { throw failure }
                    guard let http = response as? HTTPURLResponse else {
                        throw AppHTTPError.network("not an HTTP response")
                    }
                    if http.expectedContentLength > Int64(request.maximumBytes) {
                        throw AppHTTPError.oversized(limit: request.maximumBytes)
                    }
                    var body = Data()
                    body.reserveCapacity(min(request.maximumBytes, 256 * 1_024))
                    var chunk = [UInt8]()
                    chunk.reserveCapacity(16_384)
                    for try await byte in bytes {
                        chunk.append(byte)
                        if chunk.count == 16_384 {
                            body.append(contentsOf: chunk)
                            chunk.removeAll(keepingCapacity: true)
                            if body.count > request.maximumBytes {
                                throw AppHTTPError.oversized(limit: request.maximumBytes)
                            }
                            try Task.checkCancellation()
                        }
                    }
                    body.append(contentsOf: chunk)
                    if body.count > request.maximumBytes {
                        throw AppHTTPError.oversized(limit: request.maximumBytes)
                    }
                    return AppHTTPResponse(
                        finalURL: http.url ?? request.url, statusCode: http.statusCode,
                        contentType: http.value(forHTTPHeaderField: "Content-Type"), body: body)
                }
                group.addTask {
                    try await Task.sleep(until: deadline, clock: .continuous)
                    return nil
                }
                do {
                    guard let first = try await group.next(), let response = first else {
                        group.cancelAll()
                        throw AppHTTPError.timedOut
                    }
                    group.cancelAll()
                    return response
                } catch let error as AppHTTPError {
                    group.cancelAll()
                    throw error
                } catch is CancellationError {
                    group.cancelAll()
                    throw AppHTTPError.cancelled
                } catch let error as URLError {
                    group.cancelAll()
                    if let failure = delegate.failure { throw failure }
                    switch error.code {
                    case .timedOut: throw AppHTTPError.timedOut
                    case .cancelled: throw AppHTTPError.cancelled
                    default: throw AppHTTPError.network(error.localizedDescription)
                    }
                } catch {
                    group.cancelAll()
                    throw AppHTTPError.network("\(error)")
                }
            }
    }

    static func checkScheme(_ url: URL) throws {
        guard let scheme = url.scheme?.lowercased(), scheme == "http" || scheme == "https" else {
            throw AppHTTPError.unsupportedScheme
        }
    }

    static func checkHost(_ url: URL) async throws {
        guard let host = url.host else { throw AppHTTPError.unsupportedScheme }
        let allowed = await Task.detached(priority: .userInitiated) {
            AppNetworkPolicy.resolvesToPublicAddresses(host)
        }.value
        guard allowed else { throw AppHTTPError.privateAddress(host) }
    }

    /// Counts redirects and checks every hop against the same rules as the
    /// first request.
    private final class RedirectGuard: NSObject, URLSessionTaskDelegate, @unchecked Sendable {
        private let lock = NSLock()
        private var redirects = 0
        private var storedFailure: AppHTTPError?
        let checksHosts: Bool

        init(checksHosts: Bool) { self.checksHosts = checksHosts }

        var failure: AppHTTPError? { lock.withLock { storedFailure } }

        func urlSession(_ session: URLSession, task: URLSessionTask,
                        willPerformHTTPRedirection response: HTTPURLResponse,
                        newRequest request: URLRequest) async -> URLRequest? {
            let refusal: AppHTTPError? = lock.withLock {
                redirects += 1
                if redirects > URLSessionHTTPTransport.maximumRedirects { return .tooManyRedirects }
                return nil
            }
            if let refusal { return refuse(refusal) }
            guard let url = request.url, let scheme = url.scheme?.lowercased(),
                  scheme == "http" || scheme == "https" else { return refuse(.unsupportedScheme) }
            if response.url?.scheme?.lowercased() == "https", scheme == "http" {
                return refuse(.insecureRedirect)
            }
            if checksHosts, let host = url.host,
               !AppNetworkPolicy.resolvesToPublicAddresses(host) {
                return refuse(.privateAddress(host))
            }
            return request
        }

        private func refuse(_ error: AppHTTPError) -> URLRequest? {
            lock.withLock { storedFailure = error }
            return nil
        }
    }
}
