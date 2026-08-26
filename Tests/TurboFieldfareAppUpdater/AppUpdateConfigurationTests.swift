import Foundation
import Testing
@testable import TurboFieldfareAppUpdater

@Suite struct AppUpdateConfigurationTests {
    private let publicKey = Data(repeating: 0xA5, count: 32).base64EncodedString()

    @Test func acceptsSignedHTTPSFeed() throws {
        let result = AppUpdateConfiguration.resolve(infoDictionary: [
            "SUFeedURL": "https://github.com/rexmhall09/TUFF/releases/latest/download/appcast.xml",
            "SUPublicEDKey": publicKey,
        ])
        let configuration = try result.get()
        #expect(configuration.feedURL.scheme == "https")
        #expect(configuration.publicKey == publicKey)
    }

    @Test func missingConfigurationFailsClosed() {
        #expect(AppUpdateConfiguration.resolve(infoDictionary: nil)
            == .failure(.missingConfiguration))
        #expect(AppUpdateConfiguration.resolve(infoDictionary: [:])
            == .failure(.missingConfiguration))
    }

    @Test func rejectsUnsignedOrInsecureConfiguration() {
        #expect(AppUpdateConfiguration.resolve(infoDictionary: [
            "SUFeedURL": "http://example.com/appcast.xml",
            "SUPublicEDKey": publicKey,
        ]) == .failure(.insecureOrInvalidFeedURL))
        #expect(AppUpdateConfiguration.resolve(infoDictionary: [
            "SUFeedURL": "https://example.com/appcast.xml",
            "SUPublicEDKey": "not-a-public-key",
        ]) == .failure(.invalidPublicKey))
    }
}
