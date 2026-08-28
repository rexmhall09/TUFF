import Testing
@testable import TUFFAppCore

@Suite struct DecodeServiceLaunchPolicyTests {
    @Test func kickstartTargetsGUIJobWithoutRestartingIt() {
        let arguments = DecodeServiceInferenceClient.kickstartArguments(
            uid: 501, label: "com.tuff.decode.test")

        #expect(arguments == [
            "kickstart", "gui/501/com.tuff.decode.test",
        ])
        #expect(!arguments.contains("-k"))
    }

    @Test func socketFailureRemainsThePrimaryDiagnostic() {
        let message = DecodeServiceInferenceClient.socketFailureMessage(
            socketError: "No such file or directory", kickstartError: nil)

        #expect(message ==
            "decode service socket did not become ready: No such file or directory")
    }

    @Test func socketFailureIncludesKickstartDiagnosticWhenAvailable() {
        let message = DecodeServiceInferenceClient.socketFailureMessage(
            socketError: "No such file or directory",
            kickstartError: "Operation not permitted")

        #expect(message ==
            "decode service socket did not become ready: No such file or directory; launchctl kickstart failed: Operation not permitted")
    }
}
