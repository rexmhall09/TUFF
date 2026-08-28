import Foundation

public struct AppPromptPreset: Identifiable, Decodable, Sendable, Equatable {
    public let id: String
    public let title: String
    public let prompt: String

    public init(id: String, title: String, prompt: String) {
        self.id = id
        self.title = title
        self.prompt = prompt
    }

    public static let all: [AppPromptPreset] = loadBundledPrompts()
    public static var primary: [AppPromptPreset] { Array(all.prefix(3)) }
    public static var secondary: [AppPromptPreset] { Array(all.dropFirst(3)) }

    private static let resourceBundle: Bundle = {
        if let resources = Bundle.main.resourceURL,
           let packaged = Bundle(
               url: resources.appendingPathComponent(
                   "TUFF_TUFFAppCore.bundle",
                   isDirectory: true
               )
           ) {
            return packaged
        }
        return Bundle.module
    }()

    private static func loadBundledPrompts() -> [AppPromptPreset] {
        guard let url = resourceBundle.url(forResource: "app-prompts", withExtension: "json"),
              let data = try? Data(contentsOf: url),
              let prompts = try? JSONDecoder().decode([AppPromptPreset].self, from: data),
              prompts.count == 7,
              Set(prompts.map(\.id)).count == prompts.count,
              prompts.allSatisfy({ !$0.id.isEmpty && !$0.title.isEmpty && !$0.prompt.isEmpty })
        else {
            return []
        }
        return prompts
    }
}
