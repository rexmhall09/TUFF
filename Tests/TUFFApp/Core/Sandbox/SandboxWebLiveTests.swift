#if os(macOS)
import Foundation
import Testing
@testable import TUFFAppCore

/// Web requests through the real sandbox to the real internet. It only runs
/// when `TUFF_SANDBOX_LIVE_IMAGE` names an image already in the local store
/// (see `SandboxLiveTests`):
///
///     TUFF_SANDBOX_LIVE_IMAGE=tuff-sandbox:dev Scripts/test.sh --filter SandboxWebLiveTests
///
/// It prints one `WEB LIVE:` line per check with the time it took. Searching
/// DuckDuckGo can be refused as automated traffic; that is reported, not
/// failed, because it says nothing about the sandbox.
@MainActor
@Suite(.serialized) struct SandboxWebLiveTests {
    @Test(.enabled(if: SandboxLiveTests.image != nil))
    func realPagesThroughTheRealSandbox() async throws {
        let reference = try #require(SandboxLiveTests.image)
        let tool = ContainerTool()
        try await tool.verify()
        let inspected = try #require(await tool.inspectImage(reference), "image \(reference) is not in the local store")
        let digest = try #require(SandboxImageProvisioner.indexDigest(in: inspected))
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("tuff sandbox web live \(UUID().uuidString)", isDirectory: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        var dependencies = SandboxDependencies.live()
        dependencies.pin = SandboxImagePin(reference: reference, digest: digest, source: .local)
        dependencies.stateDirectory = directory
        let controller = SandboxController(dependencies: dependencies)
        await controller.start()
        try #require(controller.state == .ready, "\(controller.state)")

        let access = SandboxWebAccess(controller: controller)
        let reader = SandboxPageReader(access: access, timeoutSeconds: 20)

        func timed<T>(_ label: String, _ work: () async throws -> T) async rethrows -> T {
            let began = Date()
            defer { print("WEB LIVE: \(label) took \(String(format: "%.1f", Date().timeIntervalSince(began))) s") }
            return try await work()
        }

        // An HTML page.
        let page = try await timed("html page") { try await reader.read(URL(string: "https://example.org/")!) }
        print("WEB LIVE: html title \(page.title ?? "-"), \(page.text.count) characters")
        // The extractor moves the page's heading into the title.
        #expect(((page.title ?? "") + " " + page.text).contains("Example Domain"))

        // A PDF, read inside the sandbox.
        let pdf = try await timed("pdf") {
            try await reader.read(URL(string: "https://www.w3.org/WAI/ER/tests/xhtml/testfiles/resources/pdf/dummy.pdf")!)
        }
        print("WEB LIVE: pdf \(pdf.text.count) characters")
        #expect(!pdf.text.isEmpty)

        // A page that does not exist.
        do {
            _ = try await timed("404") { try await reader.read(URL(string: "https://example.org/this-page-does-not-exist-tuff")!) }
            Issue.record("a missing page was read")
        } catch AppWebPageError.http(let status) {
            print("WEB LIVE: 404 reported as HTTP \(status)")
        }

        // Local and private addresses are refused, and nothing is fetched.
        for address in ["http://127.0.0.1:9100/", "http://169.254.169.254/latest/meta-data/", "http://192.168.1.1/"] {
            do {
                _ = try await timed("refuse \(address)") { try await reader.read(URL(string: address)!) }
                Issue.record("\(address) was read")
            } catch AppWebPageError.transport(.privateAddress) {
                print("WEB LIVE: \(address) refused as a private address")
            } catch {
                print("WEB LIVE: \(address) refused with: \(error)")
            }
        }

        // A search through /v1/http.
        let provider = DuckDuckGoSearchProvider(transport: SandboxHTTPTransport(access: access), timeoutSeconds: 20)
        do {
            let response = try await timed("duckduckgo search") { try await provider.search("swift programming language", count: 5) }
            print("WEB LIVE: search returned \(response.results.count) results")
            #expect(!response.results.isEmpty)
        } catch let error as AppWebSearchError {
            print("WEB LIVE: search not completed by DuckDuckGo (inconclusive): \(error)")
        }

        // A stopped sandbox means nothing is fetched.
        await controller.stop()
        do {
            _ = try await reader.read(URL(string: "https://example.org/")!)
            Issue.record("a page was read with the sandbox stopped")
        } catch AppWebPageError.transport(.sandbox(let message)) {
            print("WEB LIVE: stopped sandbox: \(message)")
        }
        let leftover = try await tool.containerIDs(all: true)
        #expect(!leftover.contains { $0.hasPrefix(SandboxController.containerNamePrefix) })
    }
}
#endif
