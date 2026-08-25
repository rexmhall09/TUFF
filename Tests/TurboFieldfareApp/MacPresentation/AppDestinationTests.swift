import Testing
@testable import TurboFieldfareMacPresentation

@Suite struct AppDestinationTests {
    @Test func navigationOrderAndLabelsStayStable() {
        #expect(AppDestination.allCases == [.chat, .models, .server, .settings])
        #expect(AppDestination.allCases.map(\.title)
            == ["Chat", "Models", "Server", "Settings"])
    }

    @Test func everyDestinationHasAUniqueSymbolAndIdentifier() {
        let destinations = AppDestination.allCases

        #expect(Set(destinations.map(\.id)).count == destinations.count)
        #expect(Set(destinations.map(\.systemImage)).count == destinations.count)
    }
}
