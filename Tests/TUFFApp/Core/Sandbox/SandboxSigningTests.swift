#if os(macOS)
import Foundation
import Testing
@testable import TUFFAppCore

// MARK: The shared vectors

private struct SigningVectors: Decodable {
    struct Request: Decodable {
        let name: String
        let method: String
        let path: String
        let nonce: String
        let time: String
        let body: String
        let bodySHA256: String
        let expectedAuth: String

        enum CodingKeys: String, CodingKey {
            case name, method, path, nonce, time
            case body = "body_utf8"
            case bodySHA256 = "body_sha256"
            case expectedAuth = "expected_auth"
        }
    }

    struct Response: Decodable {
        let name: String
        let method: String
        let path: String
        let nonce: String
        let status: Int
        let body: String
        let bodySHA256: String
        let expectedAuth: String

        enum CodingKeys: String, CodingKey {
            case name, method, path, nonce, status
            case body = "body_utf8"
            case bodySHA256 = "body_sha256"
            case expectedAuth = "expected_auth"
        }
    }

    let secret: String
    let request: [Request]
    let response: [Response]

    enum CodingKeys: String, CodingKey {
        case secret = "secret_utf8"
        case request, response
    }

    var key: Data { Data(secret.utf8) }
}

private struct SigningVectorsNotFound: Error, CustomStringConvertible {
    let description: String
}

/// The folder that holds `Package.swift`, found from this file's own path.
private func signingRepositoryRoot() throws -> URL {
    var directory = URL(fileURLWithPath: #filePath).deletingLastPathComponent()
    for _ in 0..<16 {
        let manifest = directory.appendingPathComponent("Package.swift")
        if FileManager.default.fileExists(atPath: manifest.path) { return directory }
        directory = directory.deletingLastPathComponent()
    }
    throw SigningVectorsNotFound(description: "no Package.swift above \(#filePath)")
}

private func loadSigningVectors() throws -> SigningVectors {
    let file = try signingRepositoryRoot().appendingPathComponent("Sandbox/test-vectors/signing.json")
    return try JSONDecoder().decode(SigningVectors.self, from: Data(contentsOf: file))
}

private let signingLowercaseHex = Set("0123456789abcdef")

private func signingIsLowercaseHex(_ text: String, length: Int) -> Bool {
    text.count == length && text.allSatisfy { signingLowercaseHex.contains($0) }
}

// MARK: SandboxSigning

@Suite struct SandboxSigningTests {
    @Test func theLabelsAreTheOnesTheServerUses() {
        #expect(SandboxSigning.requestLabel == "tuff-sandbox-req-v1")
        #expect(SandboxSigning.responseLabel == "tuff-sandbox-resp-v1")
    }

    @Test func theVectorsFileHasExamplesOfBothKinds() throws {
        let vectors = try loadSigningVectors()
        #expect(!vectors.secret.isEmpty)
        #expect(vectors.request.count >= 3)
        #expect(vectors.response.count >= 3)
    }

    @Test func requestAuthReproducesEveryVector() throws {
        let vectors = try loadSigningVectors()
        let key = vectors.key
        for vector in vectors.request {
            let body = Data(vector.body.utf8)
            #expect(SandboxSigning.sha256Hex(body) == vector.bodySHA256, "body hash of \(vector.name)")
            let auth = SandboxSigning.requestAuth(
                key: key, method: vector.method, path: vector.path, nonce: vector.nonce,
                time: vector.time, body: body)
            #expect(auth == vector.expectedAuth, "request vector \(vector.name)")
            #expect(signingIsLowercaseHex(auth, length: 64), "request vector \(vector.name)")
        }
    }

    @Test func responseAuthReproducesEveryVector() throws {
        let vectors = try loadSigningVectors()
        let key = vectors.key
        for vector in vectors.response {
            let body = Data(vector.body.utf8)
            #expect(SandboxSigning.sha256Hex(body) == vector.bodySHA256, "body hash of \(vector.name)")
            let auth = SandboxSigning.responseAuth(
                key: key, method: vector.method, path: vector.path, nonce: vector.nonce,
                status: vector.status, body: body)
            #expect(auth == vector.expectedAuth, "response vector \(vector.name)")
            #expect(signingIsLowercaseHex(auth, length: 64), "response vector \(vector.name)")
        }
    }

    @Test func everyResponseVectorIsAcceptedByIsValidResponse() throws {
        let vectors = try loadSigningVectors()
        let key = vectors.key
        for vector in vectors.response {
            let valid = SandboxSigning.isValidResponse(
                vector.expectedAuth, key: key, method: vector.method, path: vector.path,
                nonce: vector.nonce, status: vector.status, body: Data(vector.body.utf8))
            #expect(valid, "response vector \(vector.name)")
        }
    }

    @Test func aWrongKeyGivesAnotherSignature() throws {
        let vectors = try loadSigningVectors()
        let other = Data("some-other-secret".utf8)
        let first = try #require(vectors.request.first)
        let auth = SandboxSigning.requestAuth(
            key: other, method: first.method, path: first.path, nonce: first.nonce,
            time: first.time, body: Data(first.body.utf8))
        #expect(auth != first.expectedAuth)
    }

    @Test func changingAnyRequestFieldChangesTheSignature() throws {
        let vectors = try loadSigningVectors()
        let key = vectors.key
        let vector = try #require(vectors.request.first(where: { $0.name == "page" }))
        let body = Data(vector.body.utf8)
        func auth(method: String? = nil, path: String? = nil, nonce: String? = nil,
                  time: String? = nil, body changed: Data? = nil) -> String {
            SandboxSigning.requestAuth(
                key: key, method: method ?? vector.method, path: path ?? vector.path,
                nonce: nonce ?? vector.nonce, time: time ?? vector.time, body: changed ?? body)
        }
        #expect(auth() == vector.expectedAuth)
        #expect(auth(method: "GET") != vector.expectedAuth)
        #expect(auth(path: "/v1/http") != vector.expectedAuth)
        #expect(auth(nonce: String(repeating: "cd", count: 32)) != vector.expectedAuth)
        #expect(auth(time: "1786000043") != vector.expectedAuth)
        #expect(auth(body: Data(vector.body.utf8) + Data(" ".utf8)) != vector.expectedAuth)
        #expect(auth(body: Data()) != vector.expectedAuth)
    }

    @Test func aRequestSignatureIsNotAValidResponseSignature() throws {
        // Same method, path and nonce; the labels keep the two apart.
        let key = Data("tuff-sandbox-test-secret-0123456789abcdef".utf8)
        let nonce = String(repeating: "00", count: 32)
        let request = SandboxSigning.requestAuth(
            key: key, method: "GET", path: "/health", nonce: nonce, time: "200", body: Data())
        let valid = SandboxSigning.isValidResponse(
            request, key: key, method: "GET", path: "/health", nonce: nonce, status: 200, body: Data())
        #expect(!valid)
    }

    // MARK: isValidResponse

    private struct Sample {
        let key = Data("tuff-sandbox-test-secret-0123456789abcdef".utf8)
        let method = "GET"
        let path = "/health"
        let nonce = String(repeating: "ab", count: 32)
        let status = 200
        let body = Data(#"{"status": "ok", "profile": "web", "api": 1}"#.utf8)

        var auth: String {
            SandboxSigning.responseAuth(key: key, method: method, path: path, nonce: nonce,
                                        status: status, body: body)
        }

        func accepts(_ given: String) -> Bool {
            SandboxSigning.isValidResponse(given, key: key, method: method, path: path,
                                           nonce: nonce, status: status, body: body)
        }
    }

    @Test func isValidResponseAcceptsTheCorrectSignature() {
        let sample = Sample()
        #expect(signingIsLowercaseHex(sample.auth, length: 64))
        #expect(sample.accepts(sample.auth))
    }

    @Test func isValidResponseRejectsOneChangedCharacter() {
        let sample = Sample()
        let good = sample.auth
        for position in [0, 1, 31, 62, 63] {
            var characters = Array(good)
            characters[position] = characters[position] == "0" ? "1" : "0"
            #expect(!sample.accepts(String(characters)), "position \(position)")
        }
    }

    @Test func isValidResponseRejectsTheWrongLength() {
        let sample = Sample()
        let good = sample.auth
        #expect(!sample.accepts(String(good.dropLast())))
        #expect(!sample.accepts(String(good.dropLast(2))))
        #expect(!sample.accepts(String(good.dropFirst(2))))
        #expect(!sample.accepts(good + "0"))
        #expect(!sample.accepts(good + "00"))
        #expect(!sample.accepts(good + good))
        #expect(!sample.accepts(String(good.prefix(32))))
    }

    @Test func isValidResponseRejectsTextThatIsNotHex() {
        let sample = Sample()
        let good = sample.auth
        #expect(!sample.accepts("g" + String(good.dropFirst())))
        #expect(!sample.accepts(String(good.dropLast()) + "z"))
        #expect(!sample.accepts(" " + String(good.dropFirst())))
        #expect(!sample.accepts(good + "\n"))
        #expect(!sample.accepts(" " + good + " "))
        #expect(!sample.accepts("0x" + String(good.dropFirst(2))))
        #expect(!sample.accepts(String(repeating: "é", count: 32)))
        #expect(!sample.accepts(String(repeating: "-", count: 64)))
    }

    @Test func isValidResponseRejectsAnEmptySignature() {
        #expect(!Sample().accepts(""))
    }

    @Test func isValidResponseReadsUppercaseHexAsTheSameBytes() {
        // The comparison is on the decoded bytes, so letter case does not matter.
        let sample = Sample()
        #expect(sample.accepts(sample.auth.uppercased()))
    }

    @Test func isValidResponseRejectsASignatureOfAnotherAnswer() {
        let sample = Sample()
        let good = sample.auth
        let key = sample.key
        func accepts(method: String? = nil, path: String? = nil, nonce: String? = nil,
                     status: Int? = nil, body: Data? = nil, key other: Data? = nil) -> Bool {
            SandboxSigning.isValidResponse(
                good, key: other ?? key, method: method ?? sample.method, path: path ?? sample.path,
                nonce: nonce ?? sample.nonce, status: status ?? sample.status, body: body ?? sample.body)
        }
        #expect(accepts())
        #expect(!accepts(method: "POST"))
        #expect(!accepts(path: "/v1/page"))
        #expect(!accepts(nonce: String(repeating: "cd", count: 32)))
        #expect(!accepts(status: 500))
        #expect(!accepts(body: Data("{}".utf8)))
        #expect(!accepts(body: Data()))
        #expect(!accepts(key: Data("another key".utf8)))
    }

    // MARK: Nonces and hashes

    @Test func aNonceIs64LowercaseHexCharacters() throws {
        let nonce = try SandboxSigning.newNonce()
        #expect(signingIsLowercaseHex(nonce, length: 64))
    }

    @Test func everyNonceIsNew() throws {
        var seen = Set<String>()
        for _ in 0..<200 {
            seen.insert(try SandboxSigning.newNonce())
        }
        #expect(seen.count == 200)
    }

    @Test func sha256HexOfEmptyDataIsTheKnownValue() {
        #expect(SandboxSigning.sha256Hex(Data())
            == "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855")
    }

    @Test func sha256HexOfAbcIsTheKnownValue() {
        #expect(SandboxSigning.sha256Hex(Data("abc".utf8))
            == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad")
    }

    @Test func sha256HexIsLowercaseAndFullLength() {
        let digest = SandboxSigning.sha256Hex(Data(repeating: 0xff, count: 1_000))
        #expect(signingIsLowercaseHex(digest, length: 64))
    }
}

// MARK: SandboxHealthProbe.check

private let healthProbeKey = Data("tuff-sandbox-test-secret-0123456789abcdef".utf8)
private let healthProbeNonce = String(repeating: "ab", count: 32)
private let healthProbeUnsignedMessage = "the answer is not signed with this start's secret"

private func healthProbeBody(status: String = "ok", profile: String = "web", api: Int = 1,
                        blocked: Int = 3) -> Data {
    Data(#"{"status": "\#(status)", "profile": "\#(profile)", "api": \#(api), "blocked_addresses": \#(blocked)}"#.utf8)
}

private func healthProbeSignature(_ body: Data, status: Int = 200, key: Data = healthProbeKey,
                            nonce: String = healthProbeNonce) -> String {
    SandboxSigning.responseAuth(key: key, method: "GET", path: "/health", nonce: nonce,
                                status: status, body: body)
}

/// What `check` throws, or nil when it returns an answer.
private func healthProbeCheckError(status: Int = 200, signature: String?, body: Data,
                        nonce: String = healthProbeNonce, key: Data = healthProbeKey) -> SandboxError? {
    do {
        _ = try SandboxHealthProbe.check(status: status, signature: signature, body: body,
                                         nonce: nonce, key: key)
        return nil
    } catch let error as SandboxError {
        return error
    } catch {
        Issue.record("unexpected error type: \(error)")
        return nil
    }
}

@Suite struct SandboxHealthProbeCheckTests {
    @Test func aSignedAnswerOfAWebSandboxIsHealthy() throws {
        let body = healthProbeBody(blocked: 3)
        let health = try SandboxHealthProbe.check(
            status: 200, signature: healthProbeSignature(body), body: body, nonce: healthProbeNonce, key: healthProbeKey)
        #expect(health == SandboxHealth(profile: "web", api: 1, blockedAddresses: 3))
    }

    @Test func zeroBlockedAddressesIsReportedAsZero() throws {
        let body = healthProbeBody(blocked: 0)
        let health = try SandboxHealthProbe.check(
            status: 200, signature: healthProbeSignature(body), body: body, nonce: healthProbeNonce, key: healthProbeKey)
        #expect(health.blockedAddresses == 0)
    }

    @Test func extraFieldsAndOtherSpacingAreFine() throws {
        let body = Data(#"{"extra":[1,2],"blocked_addresses":7,"api":1,"profile":"web","status":"ok"}"#.utf8)
        let health = try SandboxHealthProbe.check(
            status: 200, signature: healthProbeSignature(body), body: body, nonce: healthProbeNonce, key: healthProbeKey)
        #expect(health == SandboxHealth(profile: "web", api: 1, blockedAddresses: 7))
    }

    @Test func theBodyLimitIs64KiB() {
        #expect(SandboxHealthProbe.maximumBytes == 65_536)
    }

    // MARK: The signature comes first

    @Test func anUnsignedAnswerIsNotHealthy() {
        let body = healthProbeBody()
        #expect(healthProbeCheckError(signature: nil, body: body) == SandboxError.notHealthy(healthProbeUnsignedMessage))
    }

    @Test func anEmptyOrGarbageSignatureIsNotHealthy() {
        let body = healthProbeBody()
        for signature in ["", "not hex", String(repeating: "0", count: 64), String(repeating: "g", count: 64)] {
            #expect(healthProbeCheckError(signature: signature, body: body) == SandboxError.notHealthy(healthProbeUnsignedMessage),
                    "signature \(signature)")
        }
    }

    @Test func aSignatureFromAnotherSecretIsNotHealthyEvenWithValidContent() {
        let body = healthProbeBody()
        let foreign = healthProbeSignature(body, key: Data("someone else's secret".utf8))
        #expect(healthProbeCheckError(signature: foreign, body: body) == SandboxError.notHealthy(healthProbeUnsignedMessage))
    }

    @Test func aSignatureForAnotherNonceIsNotHealthy() {
        let body = healthProbeBody()
        let replayed = healthProbeSignature(body, nonce: String(repeating: "cd", count: 32))
        #expect(healthProbeCheckError(signature: replayed, body: body) == SandboxError.notHealthy(healthProbeUnsignedMessage))
    }

    @Test func aSignatureForAnotherStatusIsNotHealthy() {
        let body = healthProbeBody()
        // Signed as a 500, delivered as a 200.
        #expect(healthProbeCheckError(status: 200, signature: healthProbeSignature(body, status: 500), body: body)
            == SandboxError.notHealthy(healthProbeUnsignedMessage))
    }

    @Test func aChangedBodyBreaksTheSignature() {
        let body = healthProbeBody(blocked: 3)
        let signature = healthProbeSignature(body)
        #expect(healthProbeCheckError(signature: signature, body: healthProbeBody(blocked: 4))
            == SandboxError.notHealthy(healthProbeUnsignedMessage))
    }

    @Test func theContentIsNotReadBeforeTheSignatureIsChecked() {
        let garbage = Data("<html>this is not the sandbox</html>".utf8)
        // Nothing about the content shows: no "unexpected content", no status.
        #expect(healthProbeCheckError(signature: nil, body: garbage) == SandboxError.notHealthy(healthProbeUnsignedMessage))
        #expect(healthProbeCheckError(signature: "00", body: garbage) == SandboxError.notHealthy(healthProbeUnsignedMessage))
        #expect(healthProbeCheckError(status: 500, signature: nil, body: healthProbeBody())
            == SandboxError.notHealthy(healthProbeUnsignedMessage))
        #expect(healthProbeCheckError(signature: nil, body: healthProbeBody(profile: "other", api: 9))
            == SandboxError.notHealthy(healthProbeUnsignedMessage))
    }

    @Test func aWrongSignatureOnValidJSONGivesTheSignatureMessage() {
        let body = healthProbeBody()
        let wrong = healthProbeSignature(Data("different".utf8))
        #expect(healthProbeCheckError(signature: wrong, body: body) == SandboxError.notHealthy(healthProbeUnsignedMessage))
    }

    // MARK: Status, then content

    @Test func aSignedGarbageBodyIsUnexpectedContent() {
        for text in ["<html>hello</html>", "", "not json", "[1, 2, 3]", "\"ok\"", "null", "42"] {
            let body = Data(text.utf8)
            #expect(healthProbeCheckError(signature: healthProbeSignature(body), body: body)
                == SandboxError.notHealthy("unexpected content"), "body \(text)")
        }
    }

    @Test func aSignedErrorStatusIsNotHealthy() {
        let body = healthProbeBody()
        for status in [500, 503, 404, 401, 204, 201, 302] {
            #expect(healthProbeCheckError(status: status, signature: healthProbeSignature(body, status: status), body: body)
                == SandboxError.notHealthy("status \(status)"), "status \(status)")
        }
    }

    @Test func theStatusIsCheckedBeforeTheContent() {
        let garbage = Data("oops".utf8)
        #expect(healthProbeCheckError(status: 500, signature: healthProbeSignature(garbage, status: 500), body: garbage)
            == SandboxError.notHealthy("status 500"))
    }

    @Test func aStatusFieldOtherThanOkIsUnexpectedContent() {
        for status in ["error", "OK", "", "starting"] {
            let body = healthProbeBody(status: status)
            #expect(healthProbeCheckError(signature: healthProbeSignature(body), body: body)
                == SandboxError.notHealthy("unexpected content"), "status \(status)")
        }
    }

    @Test func missingFieldsAreUnexpectedContent() {
        let bodies = [
            #"{"profile": "web", "api": 1, "blocked_addresses": 3}"#,
            #"{"status": "ok", "api": 1, "blocked_addresses": 3}"#,
            #"{"status": "ok", "profile": "web", "blocked_addresses": 3}"#,
            #"{"status": "ok", "profile": "web", "api": 1}"#,
            #"{}"#,
        ]
        for text in bodies {
            let body = Data(text.utf8)
            #expect(healthProbeCheckError(signature: healthProbeSignature(body), body: body)
                == SandboxError.notHealthy("unexpected content"), "body \(text)")
        }
    }

    @Test func fieldsOfTheWrongTypeAreUnexpectedContent() {
        let bodies = [
            #"{"status": "ok", "profile": 1, "api": 1, "blocked_addresses": 3}"#,
            #"{"status": "ok", "profile": "web", "api": "1", "blocked_addresses": 3}"#,
            #"{"status": "ok", "profile": "web", "api": 1, "blocked_addresses": "3"}"#,
            #"{"status": "ok", "profile": null, "api": 1, "blocked_addresses": 3}"#,
            #"{"status": true, "profile": "web", "api": 1, "blocked_addresses": 3}"#,
            #"{"status": "ok", "profile": ["web"], "api": 1, "blocked_addresses": 3}"#,
            #"{"status": "ok", "profile": "web", "api": true, "blocked_addresses": 3}"#,
            #"{"status": "ok", "profile": "web", "api": 1, "blocked_addresses": 3.5}"#,
        ]
        for text in bodies {
            let body = Data(text.utf8)
            #expect(healthProbeCheckError(signature: healthProbeSignature(body), body: body)
                == SandboxError.notHealthy("unexpected content"), "body \(text)")
        }
    }

    // MARK: Profile and API version

    @Test func anotherProfileIsNotAWebSandbox() {
        let body = healthProbeBody(profile: "api")
        #expect(healthProbeCheckError(signature: healthProbeSignature(body), body: body)
            == SandboxError.notHealthy("a different kind of sandbox (api, API 1)"))
    }

    @Test func anotherApiVersionIsNotThisVersion() {
        for api in [0, 2, 99] {
            let body = healthProbeBody(api: api)
            #expect(healthProbeCheckError(signature: healthProbeSignature(body), body: body)
                == SandboxError.notHealthy("a different kind of sandbox (web, API \(api))"), "api \(api)")
        }
    }

    @Test func aLongProfileNameIsCutInTheMessage() {
        let body = healthProbeBody(profile: String(repeating: "p", count: 30))
        let shown = String(repeating: "p", count: 20) + "…"
        #expect(healthProbeCheckError(signature: healthProbeSignature(body), body: body)
            == SandboxError.notHealthy("a different kind of sandbox (\(shown), API 1)"))
    }

    @Test func aProfileNameWithControlCharactersIsCleanedInTheMessage() {
        // JSON escapes keep the raw bytes printable; the decoded name has a line break.
        let body = Data(#"{"status": "ok", "profile": "we\nb2", "api": 1, "blocked_addresses": 0}"#.utf8)
        #expect(healthProbeCheckError(signature: healthProbeSignature(body), body: body)
            == SandboxError.notHealthy("a different kind of sandbox (we b2, API 1)"))
    }

    @Test func theProfileIsComparedBeforeTheApiVersion() {
        let body = healthProbeBody(profile: "db", api: 2)
        #expect(healthProbeCheckError(signature: healthProbeSignature(body), body: body)
            == SandboxError.notHealthy("a different kind of sandbox (db, API 2)"))
    }
}
#endif
