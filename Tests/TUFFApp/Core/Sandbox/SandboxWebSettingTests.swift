#if os(macOS)
import Foundation
import Testing
import TUFFEngine
import TUFFModelCatalog
@testable import TUFFAppCore

private func encoded(_ text: String) -> String { Data(text.utf8).base64EncodedString() }

private let sandboxedMac = TUFFDeviceCapabilities(
    unifiedMemoryBytes: 32 * TUFFModelCatalog.oneGiB, macOSMajorVersion: 26, appleSiliconGeneration: 2)

@Suite struct SandboxSettingFileTests {
    @Test func theSettingIsOffByDefaultAndOldFilesLoadOff() throws {
        #expect(!MacAppSettings().runWebInSandbox)
        let data = try JSONEncoder().encode(MacAppSettings())
        var legacy = try #require(JSONSerialization.jsonObject(with: data) as? [String: Any])
        #expect(legacy["runWebInSandbox"] != nil)
        legacy.removeValue(forKey: "runWebInSandbox")
        let migrated = try JSONDecoder().decode(
            MacAppSettings.self, from: JSONSerialization.data(withJSONObject: legacy))
        #expect(!migrated.runWebInSandbox)
        #expect(migrated.version == MacAppSettings.currentVersion)
    }

    @Test func theSettingRoundTrips() throws {
        var settings = MacAppSettings()
        settings.runWebInSandbox = true
        let decoded = try JSONDecoder().decode(MacAppSettings.self, from: JSONEncoder().encode(settings))
        #expect(decoded.runWebInSandbox)
        #expect(decoded == settings)
    }
}

extension SandboxStubTests {
    @MainActor
    @Suite struct SandboxSettingModelTests {
        private func makeModel(capabilities: TUFFDeviceCapabilities = sandboxedMac,
                           sandbox: SandboxHarness = SandboxHarness.make(freePort: SandboxStubProtocol.port),
                           toolStore: AppToolStore? = nil,
                           client: any AppInferenceClient = ScriptedInferenceClient([]),
                           directory: URL? = nil) -> AppModel {
            let model = makeAppModel(modelDirectory: directory, client: client,
                                     settingsPersistenceEnabled: directory != nil,
                                     deviceCapabilities: capabilities, toolStore: toolStore)
            model.installSandbox(sandbox.controller) {
                SandboxClient(lease: $0, protocolClasses: [SandboxStubProtocol.self])
            }
            return model
        }

        private func ready(_ model: AppModel) {
            model.modelPathText = FileManager.default.temporaryDirectory.path
            model.loadState = .ready(modelDirectory: FileManager.default.temporaryDirectory, loadSeconds: 1)
        }

        private func waitUntilIdle(_ model: AppModel) async throws {
            for _ in 0..<500 where model.isRunning {
                try await Task.sleep(for: .milliseconds(10))
            }
            #expect(!model.isRunning)
        }

        @Test func supportFollowsTheMacOSVersion() {
            let sandbox = SandboxHarness.make()
            defer { sandbox.cleanUp() }
            #expect(!makeModel(capabilities: .qualifyingTestHost, sandbox: sandbox).isSandboxSupported)
            #expect(makeModel(sandbox: sandbox).isSandboxSupported)
            #expect(!makeModel(sandbox: sandbox).runWebInSandbox)
        }

        @Test func theSettingPersistsAcrossRelaunch() async throws {
            let root = FileManager.default.temporaryDirectory
                .appendingPathComponent("tuff-sandbox-setting-\(UUID().uuidString)", isDirectory: true)
            try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
            defer { try? FileManager.default.removeItem(at: root) }
            let directory = root.appendingPathComponent("gemma4.gturbo", isDirectory: true)
            let sandbox = SandboxHarness.make()
            defer { sandbox.cleanUp() }
            let first = makeModel(sandbox: sandbox, directory: directory)
            first.runWebInSandbox = true
            await first.finishSandboxWork()
            first.flushPendingSettings()
            let other = SandboxHarness.make()
            defer { other.cleanUp() }
            let relaunched = makeModel(sandbox: other, directory: directory)
            #expect(relaunched.runWebInSandbox)
        }

        @Test func turningItOnStartsTheSandboxAndOffStopsIt() async throws {
            let sandbox = SandboxHarness.make()
            defer { sandbox.cleanUp() }
            let model = makeModel(sandbox: sandbox)
            model.runWebInSandbox = true
            await model.finishSandboxWork()
            #expect(sandbox.controller.state == .ready)
            model.runWebInSandbox = false
            await model.finishSandboxWork()
            #expect(sandbox.controller.state == .off)
            #expect(sandbox.world.runningContainers.isEmpty)
        }

        @Test func flippingTheSwitchQuicklyEndsWithTheLastChoice() async throws {
            let sandbox = SandboxHarness.make()
            defer { sandbox.cleanUp() }
            let model = makeModel(sandbox: sandbox)
            model.runWebInSandbox = true
            model.runWebInSandbox = false
            await model.finishSandboxWork()
            #expect(sandbox.controller.state == .off)
            #expect(sandbox.world.runningContainers.isEmpty)
        }

        @Test func turningItOnOffAndOnAgainEndsWithTheSandboxRunning() async throws {
            let sandbox = SandboxHarness.make()
            defer { sandbox.cleanUp() }
            let model = makeModel(sandbox: sandbox)
            model.runWebInSandbox = true
            model.runWebInSandbox = false
            model.runWebInSandbox = true
            await model.finishSandboxWork()
            #expect(sandbox.controller.state == .ready)
            #expect(sandbox.world.runningContainers.count == 1)
        }

        @Test func quittingStopsTheSandbox() async throws {
            let sandbox = SandboxHarness.make()
            defer { sandbox.cleanUp() }
            let model = makeModel(sandbox: sandbox)
            model.runWebInSandbox = true
            await model.finishSandboxWork()
            model.stopSandboxWhenQuitting()
            #expect(sandbox.controller.state == .off)
            #expect(sandbox.world.runningContainers.isEmpty)
        }

        @Test func withTheSettingOffNothingTouchesTheSandbox() async throws {
            let sandbox = SandboxHarness.make(freePort: SandboxStubProtocol.port)
            defer { sandbox.cleanUp() }
            SandboxStubProtocol.install(key: Data()) { _ in SandboxStubProtocol.Reply(status: 500) }
            let direct = FixtureHTTPTransport { _ in .html(DuckDuckGoFixtures.ordinary) }
            let client = ScriptedInferenceClient([
                ToolLoopFixtures.callRound([ToolLoopFixtures.search("swift actors")]),
                ToolLoopFixtures.answer("Done [1]."),
            ])
            let model = makeModel(sandbox: sandbox, toolStore: .inMemory(transport: direct), client: client)
            ready(model)
            model.webSearchEnabled = true
            model.promptText = "What are Swift actors?"
            model.run()
            try await waitUntilIdle(model)
            #expect(direct.requests.count == 1)
            #expect(SandboxStubProtocol.seen.isEmpty)
            #expect(sandbox.world.count("run") == 0)
            #expect(model.conversation.first?.sources.isEmpty == false)
        }

        @Test func withTheSettingOnTheSearchGoesOnlyThroughTheSandbox() async throws {
            let web = try await SandboxWebHarness.ready { _ in
                SandboxStubProtocol.json([
                    "final_url": "https://html.duckduckgo.com/html/", "status": 200,
                    "content_type": "text/html; charset=utf-8",
                    "body_base64": encoded(DuckDuckGoFixtures.ordinary)])
            }
            defer { web.cleanUp() }
            let direct = FixtureHTTPTransport { _ in
                Issue.record("a request went to the network directly")
                return .html("")
            }
            let client = ScriptedInferenceClient([
                ToolLoopFixtures.callRound([ToolLoopFixtures.search("swift actors")]),
                ToolLoopFixtures.answer("Done [1]."),
            ])
            let model = makeModel(sandbox: web.sandbox, toolStore: .inMemory(transport: direct), client: client)
            ready(model)
            model.runWebInSandbox = true
            await model.finishSandboxWork()
            model.webSearchEnabled = true
            model.promptText = "What are Swift actors?"
            model.run()
            try await waitUntilIdle(model)
            #expect(direct.requests.isEmpty)
            #expect(SandboxStubProtocol.seen.map(\.path) == ["/v1/http"])
            let turn = try #require(model.conversation.first)
            #expect(turn.toolRounds.first?.results.first?.status == .succeeded)
            #expect(!turn.sources.isEmpty)
            #expect(web.sandbox.controller.requestsInFlight == 0)
        }

        @Test func withTheSandboxDownTheWebFailsAndNothingIsSentDirectly() async throws {
            let sandbox = SandboxHarness.make(isInstalled: false)
            defer { sandbox.cleanUp() }
            let direct = FixtureHTTPTransport { _ in
                Issue.record("a request went to the network directly")
                return .html("")
            }
            let client = ScriptedInferenceClient([
                ToolLoopFixtures.callRound([ToolLoopFixtures.search("swift actors")]),
                ToolLoopFixtures.answer("I could not search."),
            ])
            let model = makeModel(sandbox: sandbox, toolStore: .inMemory(transport: direct), client: client)
            ready(model)
            model.runWebInSandbox = true
            await model.finishSandboxWork()
            guard case .failed = sandbox.controller.state else {
                Issue.record("\(sandbox.controller.state)")
                return
            }
            model.webSearchEnabled = true
            model.promptText = "What are Swift actors?"
            model.run()
            try await waitUntilIdle(model)
            #expect(direct.requests.isEmpty)
            let result = try #require(model.conversation.first?.toolRounds.first?.results.first)
            #expect(result.status == .failed)
            #expect(result.summary.contains("sandbox"))
        }

        @Test func aToolboxWithoutASandboxFailsInsteadOfGoingDirect() async throws {
            let direct = FixtureHTTPTransport { _ in
                Issue.record("a request went to the network directly")
                return .html("")
            }
            let store = AppToolStore.inMemory(transport: direct)
            let toolbox = store.makeToolbox(provider: .duckDuckGo, webInSandbox: true)
            await #expect(throws: AppWebSearchError.transport(
                .duckDuckGo, .sandbox("The web sandbox is not available."))) {
                _ = try await toolbox.webSearch("swift", 5)
            }
            await #expect(throws: AppWebPageError.transport(.sandbox("The web sandbox is not available."))) {
                _ = try await toolbox.readWebpage(URL(string: "https://example.org/")!)
            }
            #expect(direct.requests.isEmpty)
        }
    }
}
#endif
