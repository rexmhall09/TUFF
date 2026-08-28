public struct AppNavigationState: Equatable, Sendable {
    public private(set) var destination: AppDestination

    public init(destination: AppDestination = .chat) {
        self.destination = destination
    }

    public mutating func select(_ destination: AppDestination) {
        self.destination = destination
    }
}
