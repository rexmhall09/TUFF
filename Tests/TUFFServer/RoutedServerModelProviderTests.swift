import Foundation
import Testing
import TUFFModelCatalog
import TUFFEngine
import TUFFAppCore
@testable import TUFFServerCore

private actor RoutedHTTPBackend: ServerInferenceBackend {
    nonisolated let visionCapability = "missing"
    var started = false
    var hold = false
    init(hold: Bool = false) { self.hold = hold }
    func finish() { hold = false }
    func waitForRelease() async throws {
        started = true
        while hold { try await Task.sleep(for: .milliseconds(5)) }
    }
    func generate(_ request: ValidatedChatRequest,
                  onEvent: @escaping @Sendable (ServerInferenceEvent) -> Void) async throws -> ServerCompletion {
        started = true
        while hold { try await Task.sleep(for: .milliseconds(5)) }
        onEvent(.content("Paris"))
        return ServerCompletion(content: "Paris", toolCalls: [], finishReason: "stop",
            usage: .init(promptTokens: 2, completionTokens: 1, totalTokens: 3))
    }
}

@Suite(.serialized) struct RoutedServerModelProviderTests {
    @Test func servingRuntimeUsesTheResolvedAllocationSnapshot() throws {
        let root = try root(); defer { try? FileManager.default.removeItem(at: root) }
        let installed = provider(root).installed
        let resolved = AppAutomaticMemoryPlan(contextTokens: 2_048, expertCacheSlots: 10,
            estimatedWorkingSetBytes: 7 << 30, safeBudgetBytes: 8 << 30,
            denseResidencyPolicy: .bounded(residentLayers: []), prefillChunkTokens: 32,
            retainedConversationBytes: 0)
        let runtime = installed.runtimeConfiguration(for: TUFFModelCatalog.ornith15_397B, resolvedPlan: resolved)
        #expect(runtime.expertCacheSlots == 10)
        #expect(runtime.prefillChunkTokens == 32)
        #expect(runtime.prefillPolicy == .off)
    }
    @Test func servingPlansGrowOnlyWithinTheMemoryBudget() throws {
        let root = try root(); defer { try? FileManager.default.removeItem(at: root) }
        let installed = provider(root).installed
        for model in TUFFModelCatalog.all {
            let plan = installed.inferencePlan(for: model)
            #expect(plan.contextTokens >= model.runtimeDefaults.contextTokens)
            if plan.contextTokens != model.runtimeDefaults.contextTokens
                || plan.expertCacheSlots != model.runtimeDefaults.expertCacheSlots {
                #expect(model.estimatedInferenceWorkingSetBytes(
                    contextTokens: plan.contextTokens, expertCacheSlots: plan.expertCacheSlots,
                    prefillChunkTokens: model.recommendedPrefillChunkTokens(on: installed.device))
                    <= installed.device.safeAppMemoryBudgetBytes)
            }
            #expect(installed.runtimeConfiguration(for: model).expertCacheSlots == plan.expertCacheSlots)
            if model.family == .gptOss {
                #expect(installed.runtimeConfiguration(for: model).prefillPolicy == .chunked)
            }
        }
        let small = provider(root, memory: 1 << 30).installed
        for model in TUFFModelCatalog.all {
            #expect(small.inferencePlan(for: model).contextTokens == model.runtimeDefaults.contextTokens)
            #expect(small.inferencePlan(for: model).expertCacheSlots == model.runtimeDefaults.expertCacheSlots)
            if model.family == .gptOss {
                #expect(small.runtimeConfiguration(for: model).prefillPolicy == .chunked)
            }
        }
    }

    @Test func modelDiscoveryAdvertisesTheServingPlan() throws {
        let root = try root(); defer { try? FileManager.default.removeItem(at: root) }
        for model in TUFFModelCatalog.all { try install(model, root: root) }
        let p = provider(root)
        let encoded = try JSONEncoder().encode(p.modelList())
        let list = try JSONDecoder().decode(OpenAIModelList.self, from: encoded)
        for row in list.data {
            let model = try #require(ServerInstalledModels.descriptor(named: row.id))
            #expect(row.contextLength == p.installed.inferencePlan(for: model).contextTokens)
            #expect(row.maxOutputTokens == min(4_096, (row.contextLength ?? 0) / 4))
        }
    }
    private func root() throws -> URL {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent("TUFFRouterTests-\(UUID())")
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        return root
    }
    private func install(_ descriptor: TUFFModelDescriptor, root: URL) throws {
        let dir = root.appendingPathComponent(descriptor.installDirectoryName)
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        let c = ArchConfig.registeredArchitectures[ModelVariant(rawValue: descriptor.architecture.id.rawValue)!]!
        let arch: [String: Any] = ["variant": c.variant.rawValue, "family": c.family.rawValue,
            "hiddenSize": c.hiddenSize, "ffnIntermediate": c.intermediateSize,
            "moeIntermediateSize": c.moeIntermediateSize, "numHeads": c.numHeads,
            "numKVHeads": c.numKVHeads, "numFullKVHeads": c.numFullKVHeads,
            "headDim": c.headDim, "fullHeadDim": c.fullHeadDim, "vocabSize": c.vocabSize,
            "slidingWindow": c.slidingWindow, "finalLogitSoftcap": c.finalLogitSoftcap,
            "ropeTheta": c.ropeTheta, "fullRopeTheta": c.fullRopeTheta,
            "partialRotaryFactor": c.partialRotaryFactor, "numLayers": c.numLayers,
            "numExperts": c.numExperts, "topKExperts": c.topKExperts,
            "tieWordEmbeddings": c.tieWordEmbeddings, "attentionKEqV": c.attentionKEqV,
            "hiddenActivation": c.hiddenActivation, "fullAttentionLayerMask": c.fullAttentionLayerMask.map { Int($0) }]
        try JSONSerialization.data(withJSONObject: ["magic": "GTURBO", "versionMajor": 1,
            "versionMinor": 1, "flags": [:], "modelID": descriptor.source.manifestModelID,
            "arch": arch, "files": [:], "expertsPerLayer": c.numExperts,
            "numLayers": c.numLayers, "expertStride": 1]).write(to: dir.appendingPathComponent("manifest.json"))
        try Data("{}".utf8).write(to: dir.appendingPathComponent("verified-install.json"))
    }
    private func provider(_ root: URL, backend: RoutedHTTPBackend = RoutedHTTPBackend(),
                          memory: UInt64 = 16 << 30, registry: TUFFResidencyRegistry? = nil) -> RoutedServerModelProvider {
        RoutedServerModelProvider(installed: .init(modelsRoot: root, device: .init(
            unifiedMemoryBytes: memory, macOSMajorVersion: 26, appleSiliconGeneration: 2)),
            settings: .init(url: nil, fallback: .init(defaultModel: "gemma4-e2b")),
            residency: registry, loader: { _, _ in backend })
    }
    /// An imported checkpoint is offered under a name made from its own,
    /// planned as its template, and loaded under the template's identity.
    @Test func customCheckpointsAreServedByName() async throws {
        let root = try root(); defer { try? FileManager.default.removeItem(at: root) }
        try install(TUFFModelCatalog.gemma4_E2B, root: root)
        let id = "custom-" + UUID().uuidString.lowercased()
        let template = TUFFModelCatalog.gemma4_E2B
        let custom = TUFFModelDescriptor(id: template.id, selector: "x", apiModelID: "x",
            displayName: "x", shortName: "x", summary: "", family: template.family,
            architecture: template.architecture, installDirectoryName: "\(id).gturbo",
            source: TUFFModelSource(repoID: "local/\(id)", revision: "r", sourceIndexSHA256: "s",
                manifestModelID: "local/my-tune", approximateDownloadBytes: 0,
                installedBytes: 1, reserveBytes: 0),
            hardware: template.hardware, memory: template.memory,
            capabilities: template.capabilities, reasoningControl: template.reasoningControl)
        try install(custom, root: root)
        let registry: [String: Any] = ["schemaVersion": 1, "checkpoints": [[
            "id": id, "name": "My Gemma Tune!", "sourcePath": "/tmp/source.gturbo",
            "templateID": template.id.rawValue,
            "sourceIndexSHA256": String(repeating: "a", count: 64),
            "manifestSHA256": String(repeating: "b", count: 64), "installedBytes": 1]]]
        try JSONSerialization.data(withJSONObject: registry)
            .write(to: root.appendingPathComponent("custom-checkpoints.json"))

        let p = provider(root)
        #expect(p.modelList().data.map(\.id) == [template.apiModelID, "my-gemma-tune"])
        let routed = try p.route("my-gemma-tune")
        #expect(routed.id == "my-gemma-tune")
        let descriptor = try #require(p.installed.descriptor(named: "my-gemma-tune"))
        #expect(descriptor.displayName == "My Gemma Tune!")
        #expect(!descriptor.capabilities.contains(.imageInput))
        #expect(p.installed.engineReportedModelID(for: descriptor) == template.apiModelID)
        #expect(p.installed.engineReportedModelID(for: template) == template.apiModelID)
        #expect(p.installed.planningDescriptor(for: descriptor).isCustomCheckpoint)
        #expect(p.installed.directory(for: descriptor).lastPathComponent == "\(id).gturbo")
    }

    @Test func customCheckpointNamesAreStableAndNeverShadowCatalogModels() throws {
        func checkpoint(_ name: String) -> AppCustomCheckpoint {
            AppCustomCheckpoint(id: "custom-" + UUID().uuidString.lowercased(), name: name,
                sourcePath: "/x", templateID: .gemma4_E2B,
                sourceIndexSHA256: String(repeating: "a", count: 64),
                manifestSHA256: String(repeating: "b", count: 64), installedBytes: 1)
        }
        let list = [checkpoint("Tuned Model"), checkpoint("tuned model"), checkpoint("gemma4"),
                    checkpoint("!!!"), checkpoint("default")]
        let names = AppCustomCheckpointRegistry.apiNames(for: list)
        #expect(list.map { names[$0.id] } == ["tuned-model", "tuned-model-2", "gemma4-custom",
                                               "custom", "default-custom"])
    }

    @Test func mismatchedPackCannotMasqueradeAsCatalogModel() throws {
        let root = try root(); defer { try? FileManager.default.removeItem(at: root) }
        try install(TUFFModelCatalog.gemma4_E4B, root: root)
        try FileManager.default.createSymbolicLink(at: root.appendingPathComponent(TUFFModelCatalog.gemma4_E2B.installDirectoryName),
            withDestinationURL: root.appendingPathComponent(TUFFModelCatalog.gemma4_E4B.installDirectoryName))
        let p = provider(root)
        #expect(throws: ServerRequestError.self) { try p.route("default") }
        #expect(p.modelList().data.map(\.id) == [TUFFModelCatalog.gemma4_E4B.apiModelID])
    }
    @Test func aliasesModelsListUnavailableAndIneligible() throws {
        let root = try root(); defer { try? FileManager.default.removeItem(at: root) }
        try install(TUFFModelCatalog.gemma4_E2B, root: root)
        let p = provider(root)
        for name in ["default", "gemma4-e2b", "gemma-4-e2b-it"] {
            #expect(try p.route(name).id == TUFFModelCatalog.gemma4_E2B.apiModelID)
            #expect(try p.route(name).requestedID == name)
        }
        #expect(p.modelList().data.map(\.id) == [TUFFModelCatalog.gemma4_E2B.apiModelID])
        #expect(throws: ServerRequestError.self) { try p.route("gemma4") }
        #expect(throws: ServerRequestError.unknownModel) { try p.route("bogus") }
        #expect(throws: ServerRequestError.self) { try provider(root, memory: 1 << 30).route("default") }
    }
    @Test func qualifiedModelsRemainAvailableAndReserveRuntimeMemory() async throws {
        let root = try root(); defer { try? FileManager.default.removeItem(at: root) }
        for model in TUFFModelCatalog.all { try install(model, root: root) }
        let registry = TUFFResidencyRegistry(directory: root.appendingPathComponent("leases"))
        let p = provider(root, registry: registry)
        #expect(p.modelList().data.count == TUFFModelCatalog.all.filter { $0.qualification == .qualified }.count)
        let model = TUFFModelCatalog.gemma4_26B_A4B
        let companion = try VisionPackLocation.companionURL(forTextModel: p.installed.directory(for: model))
        try FileManager.default.createDirectory(at: companion, withIntermediateDirectories: true)
        #expect(p.installed.estimatedBytes(for: model) >= p.installed.device.safeAppMemoryBudgetBytes)
        let target = try p.route(model.apiModelID)
        try await p.run(target, onQueued: {}, prepare: { _ in () }, operation: { _, _ in
            let active = registry.activeRecords()
            #expect(active.count == 1)
            #expect(active[0].estimatedBytes == p.installed.estimatedBytes(for: model))
            #expect(throws: TUFFResidencyRegistry.MemoryBusy.self) {
                try registry.reserve(.init(owner: .app, modelID: "other", estimatedBytes: 1),
                    budgetBytes: p.installed.device.safeAppMemoryBudgetBytes)
            }
        })
        await p.shutdown()
    }
    @Test func legacyManifestsUseTheRuntimesExactArchitectureResolution() throws {
        let root = try root(); defer { try? FileManager.default.removeItem(at: root) }
        let p = provider(root)
        for model in [TUFFModelCatalog.gemma4_26B_A4B, TUFFModelCatalog.qwen36_35B_A3B,
                      TUFFModelCatalog.minimaxM27, TUFFModelCatalog.gemma4_E2B] {
            try install(model, root: root)
            let url = p.installed.directory(for: model).appendingPathComponent("manifest.json")
            var manifest = try JSONSerialization.jsonObject(with: Data(contentsOf: url)) as! [String: Any]
            var arch = manifest["arch"] as! [String: Any]
            arch.removeValue(forKey: "variant")
            if model.family == .gemma4 { arch.removeValue(forKey: "family") }
            manifest["arch"] = arch; manifest["versionMinor"] = 0
            try JSONSerialization.data(withJSONObject: manifest).write(to: url)
            if model.id == .gemma4_E2B {
                #expect(!p.installed.isInstalled(model))
            } else {
                #expect(try p.route(model.apiModelID).id == model.apiModelID)
            }
        }
    }
    @Test func memoryBusyFailsBeforeLoading() async throws {
        let root = try root(); defer { try? FileManager.default.removeItem(at: root) }
        try install(TUFFModelCatalog.gemma4_E2B, root: root)
        let registry = TUFFResidencyRegistry(directory: root.appendingPathComponent("leases"))
        let lease = try registry.acquire(.init(owner: .app, modelID: "other", estimatedBytes: 16 << 30))
        defer { lease.release() }
        let backend = RoutedHTTPBackend(), p = provider(root, backend: backend, registry: registry)
        let target = try p.route("default")
        await #expect(throws: ServerRequestError.self) {
            try await p.run(target, onQueued: {}, prepare: { _ in 1 }, operation: { _, _ in 2 })
        }
        #expect(await backend.started == false)
        await p.shutdown()
    }
    /// A client can tell a running server from a loaded model: `/health`
    /// names the resident model, or reports null, and says whether a request
    /// is running, without a control token.
    @Test func healthReportsTheLoadedModelAndRunningRequests() async throws {
        let root = try root(); defer { try? FileManager.default.removeItem(at: root) }
        try install(TUFFModelCatalog.gemma4_E2B, root: root)
        let backend = RoutedHTTPBackend(hold: true), p = provider(root, backend: backend)
        let idle = await p.health()
        #expect(idle.loadedModel == nil)
        #expect(idle.activeRequests == 0)
        let json = String(decoding: try JSONEncoder().encode(idle), as: UTF8.self)
        #expect(json.contains(#""loaded_model":null"#))

        let server = TUFFHTTPServer(provider: p)
        let channel = try await server.start(port: 0)
        let port = try #require(channel.localAddress?.port)
        let task = Task { try await p.run(p.route("default"), onQueued: {}, prepare: { _ in 1 }, operation: { _, _ in
            try await backend.waitForRelease()
            return 1
        }) }
        for _ in 0..<1000 { if await backend.started { break }; try await Task.sleep(for: .milliseconds(1)) }
        let data = try await URLSession.shared.data(
            from: URL(string: "http://127.0.0.1:\(port)/health")!).0
        let body = try #require(JSONSerialization.jsonObject(with: data) as? [String: Any])
        #expect(body["loaded_model"] as? String == TUFFModelCatalog.gemma4_E2B.apiModelID)
        #expect(body["active_requests"] as? Int == 1)
        #expect(body["status"] as? String == "ok")
        await backend.finish()
        _ = try await task.value
        try await channel.close()
        await p.shutdown()
    }

    @Test func httpJSONStreamingAndControlAuthorization() async throws {
        let root = try root(); defer { try? FileManager.default.removeItem(at: root) }
        try install(TUFFModelCatalog.gemma4_E2B, root: root)
        let p = provider(root)
        let control = ServerControl(token: String(repeating: "a", count: 64),
            status: { await ServerControl.status(provider: p, version: "7.0.0") },
            unloadIfIdle: { await p.scheduler.unloadIfIdle() })
        let server = TUFFHTTPServer(provider: p, control: control)
        let channel = try await server.start(port: 0)
        let port = try #require(channel.localAddress?.port)
        do {
            let base = "http://127.0.0.1:\(port)"
            let (models, _) = try await URLSession.shared.data(from: URL(string: base + "/v1/models")!)
            #expect(try JSONDecoder().decode(OpenAIModelList.self, from: models).data.count == 1)
            for streaming in [false, true] {
                var request = URLRequest(url: URL(string: base + "/v1/chat/completions")!)
                request.httpMethod = "POST"; request.setValue("application/json", forHTTPHeaderField: "Content-Type")
                request.httpBody = try JSONSerialization.data(withJSONObject: ["model": "default", "stream": streaming,
                    "messages": [["role": "user", "content": "Capital of France?"]]])
                let (data, response) = try await URLSession.shared.data(for: request)
                #expect((response as? HTTPURLResponse)?.statusCode == 200)
                if streaming {
                    let text = String(decoding: data, as: UTF8.self)
                    #expect(text.contains("chat.completion.chunk")); #expect(text.contains("data: [DONE]"))
                } else {
                    let object = try #require(try JSONSerialization.jsonObject(with: data) as? [String: Any])
                    #expect(object["object"] as? String == "chat.completion")
                    #expect(object["model"] as? String == TUFFModelCatalog.gemma4_E2B.apiModelID)
                    let choices = try #require(object["choices"] as? [[String: Any]])
                    #expect((choices.first?["message"] as? [String: Any])?["content"] as? String == "Paris")
                    #expect((object["usage"] as? [String: Any])?["total_tokens"] as? Int == 3)
                }
            }
            var unload = URLRequest(url: URL(string: base + ServerControl.unloadPath)!)
            unload.httpMethod = "POST"; unload.setValue("Bearer bad", forHTTPHeaderField: "Authorization")
            let (_, bad) = try await URLSession.shared.data(for: unload)
            #expect((bad as? HTTPURLResponse)?.statusCode == 401)
            unload.setValue("Bearer " + control.token, forHTTPHeaderField: "Authorization")
            let (_, good) = try await URLSession.shared.data(for: unload)
            #expect((good as? HTTPURLResponse)?.statusCode == 200)
            let (status, _) = try await URLSession.shared.data(from: URL(string: base + ServerControl.statusPath)!)
            #expect(try JSONDecoder().decode(ServerControlStatus.self, from: status).residentModel == nil)
        } catch { try? await server.shutdown(); throw error }
        try await server.shutdown()
    }
    @Test func unloadReturns409WhileGenerating() async throws {
        let root = try root(); defer { try? FileManager.default.removeItem(at: root) }
        try install(TUFFModelCatalog.gemma4_E2B, root: root)
        let backend = RoutedHTTPBackend(hold: true), p = provider(root, backend: backend)
        let control = ServerControl(token: String(repeating: "a", count: 64),
            status: { await ServerControl.status(provider: p, version: "7.0.0") },
            unloadIfIdle: { await p.scheduler.unloadIfIdle() })
        let server = TUFFHTTPServer(provider: p, control: control)
        let channel = try await server.start(port: 0)
        let port = try #require(channel.localAddress?.port)
        let task = Task { try await p.run(p.route("default"), onQueued: {}, prepare: { _ in 1 }, operation: { _, _ in
            try await backend.waitForRelease()
            return 1
        }) }
        for _ in 0..<1000 { if await backend.started { break }; try await Task.sleep(for: .milliseconds(1)) }
        #expect(await backend.started)
        var request = URLRequest(url: URL(string: "http://127.0.0.1:\(port)" + ServerControl.unloadPath)!)
        request.httpMethod = "POST"; request.setValue("Bearer " + control.token, forHTTPHeaderField: "Authorization")
        let (_, response) = try await URLSession.shared.data(for: request)
        #expect((response as? HTTPURLResponse)?.statusCode == 409)
        await backend.finish(); _ = try await task.value
        try await server.shutdown()
    }
    @Test func loadedSessionWithoutVisionRejectsImagesBeforeGeneration() async throws {
        let root = try root(); defer { try? FileManager.default.removeItem(at: root) }
        try install(TUFFModelCatalog.gemma4_E2B, root: root)
        let directory = root.appendingPathComponent(TUFFModelCatalog.gemma4_E2B.installDirectoryName)
        let vision = try VisionPackLocation.companionURL(forTextModel: directory)
        try FileManager.default.createDirectory(at: vision, withIntermediateDirectories: true)
        let backend = RoutedHTTPBackend(), p = provider(root, backend: backend)
        #expect(try p.route("default").visionCapability == "ready")
        let server = TUFFHTTPServer(provider: p)
        let channel = try await server.start(port: 0)
        let port = try #require(channel.localAddress?.port)
        do {
            var request = URLRequest(url: URL(string: "http://127.0.0.1:\(port)/v1/chat/completions")!)
            request.httpMethod = "POST"; request.setValue("application/json", forHTTPHeaderField: "Content-Type")
            request.httpBody = Data(#"{"model":"default","messages":[{"role":"user","content":[{"type":"text","text":"Describe it"},{"type":"image_url","image_url":{"url":"data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+aD1sAAAAASUVORK5CYII="}}]}]}"#.utf8)
            let (data, response) = try await URLSession.shared.data(for: request)
            #expect((response as? HTTPURLResponse)?.statusCode == 400)
            #expect(String(decoding: data, as: UTF8.self).contains("image_input_unavailable"))
            #expect(await backend.started == false)
        } catch { try? await server.shutdown(); throw error }
        try await server.shutdown()
    }
    @Test func argumentsValidateAndResolveOverrides() throws {
        let a = try RouterServerArguments.parse(["--all-models", "--port", "9999", "--default-model", "gemma4-e2b", "--unload-after", "immediately", "--queue-limit", "2"])
        #expect(a.resolvedSettings(saved: nil).port == 9999)
        #expect(a.resolvedSettings(saved: nil).unloadDelay == .immediately)
        #expect(try RouterServerArguments.parse(["--background"]).background)
        for input in [["--all-models", "--port", "0"], ["--background", "--queue-limit", "17"], ["--all-models", "--unload-after", "-1"], ["--all-models", "--default-model", "missing"]] {
            #expect(throws: ServerArgumentError.self) { try RouterServerArguments.parse(input) }
        }
    }

    @Test func installedModelsUseCatalogRuntimeDefaultsForThisMac() {
        let sixteen = TUFFDeviceCapabilities(unifiedMemoryBytes: 16 * TUFFModelCatalog.oneGiB,
                                             macOSMajorVersion: 26, appleSiliconGeneration: 2)
        let installed = ServerInstalledModels(modelsRoot: URL(fileURLWithPath: "/models"), device: sixteen)
        for descriptor in [TUFFModelCatalog.minimaxM27, TUFFModelCatalog.default] {
            let runtime = installed.runtimeConfiguration(for: descriptor)
            #expect(runtime.expertCacheSlots == descriptor.runtimeDefaults.expertCacheSlots)
            #expect(runtime.prefillChunkTokens == descriptor.recommendedPrefillChunkTokens(on: sixteen))
        }
    }

    @Test func routingNeedsNoModeFlag() throws {
        let plain = try RouterServerArguments.parse([])
        #expect(!plain.background)
        #expect(plain.resolvedSettings(saved: nil) == TUFFBackgroundServerSettings())
        // 7.0.0 scripts passed --all-models; it still parses and changes nothing.
        #expect(try RouterServerArguments.parse(["--all-models"]) == plain)
        #expect(!RouterServerArguments.usage.contains("--all-models"))
    }

    @Test(arguments: ["--model", "--max-context", "--expert-cache-slots", "--prefill-chunk-tokens", "--vision-pack"])
    func removedFixedModelFlagsExplainTheReplacement(flag: String) {
        #expect {
            try RouterServerArguments.parse([flag, "x"])
        } throws: { error in
            guard case ServerArgumentError.invalid(let message) = error else { return false }
            return message.hasPrefix("\(flag) was removed in TUFF 7.1")
        }
    }
}
