import Testing
@testable import TUFFMacPresentation

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
        #expect(destinations.map(\.keyboardShortcut) == ["1", "2", "3", "4"])
        #expect(Set(destinations.map(\.keyboardShortcut)).count == destinations.count)
    }

    @Test func minimumWindowFitsTheSidebarAndChatDetail() {
        #expect(AppWindowLayout.minimumHorizontalAllowance
            <= AppWindowLayout.minimumWidth)
        #expect(AppWindowLayout.defaultWidth >= AppWindowLayout.minimumWidth)
        #expect(AppWindowLayout.defaultHeight >= AppWindowLayout.minimumHeight)
        #expect(AppWindowLayout.minimumWidth == 640)
        #expect(AppWindowLayout.minimumHeight == 440)
        #expect(AppWindowLayout.sidebarMinimumWidth
            <= AppWindowLayout.sidebarIdealWidth)
        #expect(AppWindowLayout.sidebarIdealWidth
            <= AppWindowLayout.sidebarMaximumWidth)
    }

    @Test func navigationStartsInChatAndSelectsEveryDestination() {
        var navigation = AppNavigationState()
        #expect(navigation.destination == .chat)

        for destination in AppDestination.allCases {
            navigation.select(destination)
            #expect(navigation.destination == destination)
        }
    }
}
