#if os(macOS)
import Foundation
import Testing
@testable import TUFFAppCore

// MARK: Helpers

private let protectionSampleBlocked = ["127.0.0.1", "192.168.1.20", "::1", "fe80::1"]

private func protectionGoodListing(_ blocked: [String] = protectionSampleBlocked) -> String {
    FakeContainerWorld.goodNftListing(blocked: blocked)
}

private func protectionProblem(_ listing: String, blocked: [String] = protectionSampleBlocked) -> String? {
    SandboxProtection.firewallProblem(listing, blocked: blocked)
}

/// `listing` with every `target` replaced; fails the test if there is none.
private func protectionReplacing(_ target: String, with replacement: String,
                                 in listing: String) throws -> String {
    try #require(listing.contains(target), "the listing has no \(target)")
    return listing.replacingOccurrences(of: target, with: replacement)
}

/// `listing` with `line` put in front of the first `anchor`.
private func protectionInserting(_ line: String, before anchor: String,
                                 in listing: String) throws -> String {
    let range = try #require(listing.range(of: anchor), "the listing has no \(anchor)")
    var copy = listing
    copy.insert(contentsOf: line, at: range.lowerBound)
    return copy
}

/// `listing` without any line that contains `needle`.
private func protectionRemovingLines(containing needle: String, from listing: String) -> String {
    listing.split(separator: "\n", omittingEmptySubsequences: false)
        .filter { !$0.contains(needle) }
        .joined(separator: "\n")
}

private func protectionPrivilegeMessage(_ shown: String) -> String {
    "the web server does not run as the unprivileged user without capabilities (found: \(shown))"
}

// MARK: privilegeProblem

@Suite struct SandboxPrivilegeProblemTests {
    @Test func nothingIsWrongForUser10001WithoutCapabilities() {
        #expect(SandboxProtection.privilegeProblem("10001 none", status: 0) == nil)
        #expect(SandboxProtection.privilegeProblem("10001 none\n", status: 0) == nil)
        #expect(SandboxProtection.privilegeProblem("  10001 none \r\n\n", status: 0) == nil)
    }

    @Test func aNonZeroStatusIsAProblemEvenWithTheRightOutput() {
        #expect(SandboxProtection.privilegeProblem("10001 none", status: 1)
            == protectionPrivilegeMessage("10001 none"))
        #expect(SandboxProtection.privilegeProblem("10001 none", status: 127)
            == protectionPrivilegeMessage("10001 none"))
        #expect(SandboxProtection.privilegeProblem("10001 none", status: -1) != nil)
    }

    @Test func rootIsAProblem() {
        #expect(SandboxProtection.privilegeProblem("0 none", status: 0)
            == protectionPrivilegeMessage("0 none"))
    }

    @Test func capabilitiesAreAProblem() {
        #expect(SandboxProtection.privilegeProblem("10001 CapEff", status: 0)
            == protectionPrivilegeMessage("10001 CapEff"))
        #expect(SandboxProtection.privilegeProblem("10001 CapPrm,CapEff,CapBnd", status: 0)
            == protectionPrivilegeMessage("10001 CapPrm,CapEff,CapBnd"))
        #expect(SandboxProtection.privilegeProblem("0 CapBnd", status: 0) != nil)
    }

    @Test func otherOutputIsAProblem() {
        #expect(SandboxProtection.privilegeProblem("10002 none", status: 0) != nil)
        #expect(SandboxProtection.privilegeProblem("10001 none extra", status: 0) != nil)
        #expect(SandboxProtection.privilegeProblem("10001none", status: 0) != nil)
        #expect(SandboxProtection.privilegeProblem("None", status: 0) != nil)
        #expect(SandboxProtection.privilegeProblem("Traceback (most recent call last):", status: 1) != nil)
    }

    @Test func noOutputIsReportedAsNothing() {
        #expect(SandboxProtection.privilegeProblem("", status: 0) == protectionPrivilegeMessage("nothing"))
        #expect(SandboxProtection.privilegeProblem("  \n\t\n", status: 0) == protectionPrivilegeMessage("nothing"))
        #expect(SandboxProtection.privilegeProblem("", status: 1) == protectionPrivilegeMessage("nothing"))
    }

    @Test func theFoundTextIsOneLineAndCut() {
        #expect(SandboxProtection.privilegeProblem("10001 none\nsecond line", status: 0)
            == protectionPrivilegeMessage("10001 none second line"))
        let long = String(repeating: "x", count: 200)
        #expect(SandboxProtection.privilegeProblem(long, status: 0)
            == protectionPrivilegeMessage(String(repeating: "x", count: 80) + "…"))
    }

    @Test func mixedOrRootGroupIdsAreAProblem() {
        #expect(SandboxProtection.privilegeProblem("mixed-ids none", status: 0) != nil)
    }

    @Test func theCheckScriptLooksAtEveryUserAndGroupId() {
        #expect(SandboxProtection.privilegeCheck.contains("status[\"Uid\"].split()"))
        #expect(SandboxProtection.privilegeCheck.contains("status[\"Gid\"].split()"))
    }

    @Test func theCheckScriptNamesTheServerAndPrintsUserAndCapabilities() {
        #expect(SandboxProtection.privilegeCheck.contains("/app/server.py"))
        #expect(SandboxProtection.privilegeCheck.contains("CapEff"))
        #expect(SandboxProtection.tableName == "tuff_sandbox_web")
    }
}

// MARK: firewallProblem

@Suite struct SandboxFirewallProblemTests {
    // MARK: The listing a real sandbox prints

    @Test func theGoodListingHasNoProblem() {
        #expect(protectionProblem(protectionGoodListing()) == nil)
    }

    @Test func theGoodListingForOtherAddressSetsHasNoProblem() {
        #expect(protectionProblem(protectionGoodListing([]), blocked: []) == nil)
        #expect(protectionProblem(protectionGoodListing(["10.1.2.3"]), blocked: ["10.1.2.3"]) == nil)
        #expect(protectionProblem(protectionGoodListing(["::1"]), blocked: ["::1"]) == nil)
        let mixed = ["127.0.0.1", "192.168.1.20", "203.0.113.77", "::1", "fe80::aede:48ff:fe33:3455",
                     "2a02:1234::5678"]
        #expect(protectionProblem(protectionGoodListing(mixed), blocked: mixed) == nil)
    }

    @Test func extraBlockedAddressesInTheListingAreFine() {
        let listing = protectionGoodListing(["127.0.0.1", "10.9.9.9", "::1", "fe80::2"])
        #expect(protectionProblem(listing, blocked: ["127.0.0.1", "::1"]) == nil)
    }

    @Test func aListingIndentedWithSpacesIsRead() {
        let listing = protectionGoodListing().replacingOccurrences(of: "\t", with: "    ")
        #expect(protectionProblem(listing) == nil)
    }

    @Test func theDnsAccessMayUseTheDomainNameForPort53() throws {
        let listing = try protectionReplacing("udp dport 53 accept", with: "udp dport domain accept",
                                              in: protectionGoodListing())
        #expect(protectionProblem(listing) == nil)
    }

    @Test func aMetaNfprotoPrefixOnTheInboundRulesIsAccepted() throws {
        let listing = try protectionReplacing(
            "\t\tip saddr 192.168.64.1 tcp dport 9100 accept",
            with: "\t\tmeta nfproto ipv4 ip saddr 192.168.64.1 tcp dport 9100 accept",
            in: protectionGoodListing())
        #expect(protectionProblem(listing) == nil)
    }

    // MARK: The table and its chains

    @Test func anotherTableIsNotTheSandboxsTable() {
        #expect(protectionProblem("", blocked: []) == "is not the sandbox's table")
        #expect(protectionProblem("table inet other {\n}\n", blocked: []) == "is not the sandbox's table")
    }

    @Test func aTableWithoutChainsHasNoOutputOrInputRules() throws {
        #expect(protectionProblem("table inet tuff_sandbox_web {\n}\n", blocked: [])
            == "has no output or input rules")
        let noInput = try protectionReplacing("chain input {", with: "chain inbound {",
                                              in: protectionGoodListing())
        #expect(protectionProblem(noInput) == "has no output or input rules")
        let noOutput = try protectionReplacing("chain output {", with: "chain outbound {",
                                               in: protectionGoodListing())
        #expect(protectionProblem(noOutput) == "has no output or input rules")
    }

    // MARK: Default policies

    @Test func anOutputPolicyThatIsNotDropIsFound() throws {
        let listing = try protectionReplacing(
            "hook output priority filter; policy drop;",
            with: "hook output priority filter; policy accept;", in: protectionGoodListing())
        #expect(protectionProblem(listing) == "does not drop outbound traffic by default")
    }

    @Test func anInputPolicyThatIsNotDropIsFound() throws {
        let listing = try protectionReplacing(
            "hook input priority filter; policy drop;",
            with: "hook input priority filter; policy accept;", in: protectionGoodListing())
        #expect(protectionProblem(listing) == "does not drop inbound traffic by default")
    }

    @Test func aMissingPolicyIsFound() throws {
        let output = try protectionReplacing(
            "hook output priority filter; policy drop;", with: "hook output priority filter;",
            in: protectionGoodListing())
        #expect(protectionProblem(output) == "does not drop outbound traffic by default")
        let input = try protectionReplacing(
            "hook input priority filter; policy drop;", with: "hook input priority filter;",
            in: protectionGoodListing())
        #expect(protectionProblem(input) == "does not drop inbound traffic by default")
    }

    // MARK: The blocked sets

    @Test func anAddressMissingFromBlocked4IsFound() {
        let listing = protectionGoodListing(["127.0.0.1", "::1", "fe80::1"])
        #expect(protectionProblem(listing, blocked: ["127.0.0.1", "192.168.1.20", "::1", "fe80::1"])
            == "does not block 192.168.1.20")
    }

    @Test func anAddressMissingFromBlocked6IsFound() {
        let listing = protectionGoodListing(["127.0.0.1", "192.168.1.20", "::1"])
        #expect(protectionProblem(listing, blocked: ["127.0.0.1", "192.168.1.20", "::1", "fe80::1"])
            == "does not block fe80::1")
    }

    @Test func anEmptyBlockedSetIsFoundWhenAddressesAreWanted() {
        let listing = protectionGoodListing([])
        #expect(protectionProblem(listing, blocked: ["127.0.0.1"]) == "does not block 127.0.0.1")
        #expect(protectionProblem(listing, blocked: ["::1"]) == "does not block ::1")
    }

    @Test func anIPv4AddressInTheIPv6SetDoesNotCount() throws {
        // The address is listed, but in the set of the wrong family.
        let listing = try protectionReplacing(
            "elements = { ::1, fe80::1 }", with: "elements = { ::1, fe80::1, 192.168.1.20 }",
            in: protectionGoodListing(["127.0.0.1", "::1", "fe80::1"]))
        #expect(protectionProblem(listing, blocked: ["127.0.0.1", "192.168.1.20", "::1", "fe80::1"])
            == "does not block 192.168.1.20")
    }

    @Test func aMissingBlockedSetIsFound() throws {
        let noFour = try protectionReplacing("set blocked4 {", with: "set renamed4 {", in: protectionGoodListing())
        #expect(protectionProblem(noFour) == "has no set of blocked addresses (blocked4)")
        let noSix = try protectionReplacing("set blocked6 {", with: "set renamed6 {", in: protectionGoodListing())
        #expect(protectionProblem(noSix) == "has no set of blocked addresses (blocked6)")
    }

    // MARK: Rules for the blocked and private ranges

    @Test func aMissingRuleForASetIsFound() {
        let listing = protectionGoodListing()
        #expect(protectionProblem(protectionRemovingLines(containing: "@private4", from: listing))
            == "has no rule for @private4")
        #expect(protectionProblem(protectionRemovingLines(containing: "@blocked6", from: listing))
            == "has no rule for @blocked6")
        #expect(protectionProblem(protectionRemovingLines(containing: "@blocked4", from: listing))
            == "has no rule for @blocked4")
    }

    @Test func aPrivateRangeThatIsAcceptedInsteadOfRejectedIsFound() throws {
        let listing = try protectionReplacing(
            "\t\tip daddr @private4 reject\n", with: "\t\tip daddr @private4 accept\n",
            in: protectionGoodListing())
        #expect(protectionProblem(listing) == "does not reject @private4 (ip daddr @private4 accept)")
    }

    @Test func aBlockedRangeThatIsNotRejectedIsFound() throws {
        let dropped = try protectionReplacing(
            "\t\tip daddr @blocked4 reject\n", with: "\t\tip daddr @blocked4 drop\n",
            in: protectionGoodListing())
        #expect(protectionProblem(dropped) == "does not reject @blocked4 (ip daddr @blocked4 drop)")
        let accepted = try protectionReplacing(
            "\t\tip6 daddr @blocked6 reject\n", with: "\t\tip6 daddr @blocked6 accept\n",
            in: protectionGoodListing())
        #expect(protectionProblem(accepted) == "does not reject @blocked6 (ip6 daddr @blocked6 accept)")
    }

    @Test func aRefusedIPv6RangeThatIsAcceptedIsFound() throws {
        let listing = try protectionReplacing(
            "\t\tip6 daddr @refused6 reject\n", with: "\t\tip6 daddr @refused6 accept\n",
            in: protectionGoodListing())
        #expect(protectionProblem(listing) == "does not reject @refused6 (ip6 daddr @refused6 accept)")
    }

    // MARK: Accepts in the output chain

    @Test func anOddAcceptInTheOutputChainIsFound() throws {
        let listing = try protectionInserting(
            "\t\ttcp dport 443 accept\n", before: "\t\tip daddr @blocked4 reject\n", in: protectionGoodListing())
        #expect(protectionProblem(listing) == "accepts more than expected (tcp dport 443 accept)")
    }

    @Test func variousOddOutputAcceptsAreFound() throws {
        let odd = [
            "ip daddr 10.0.0.5 accept",
            "oifname \"eth0\" accept",
            "ip daddr 8.8.8.8 tcp dport 443 accept",
            "ip6 daddr ::/0 accept",
            "ct state new accept",
            "meta l4proto udp accept",
        ]
        for rule in odd {
            let listing = try protectionInserting(
                "\t\t\(rule)\n", before: "\t\tip daddr @blocked4 reject\n", in: protectionGoodListing())
            #expect(protectionProblem(listing) == "accepts more than expected (\(rule))", "rule \(rule)")
        }
    }

    @Test func anOddAcceptAfterTheLastRuleIsFound() throws {
        let listing = try protectionInserting(
            "\t\ttcp dport 22 accept\n", before: "\t\tmeta l4proto tcp reject with tcp reset\n\t\treject\n",
            in: protectionGoodListing())
        #expect(protectionProblem(listing) == "accepts more than expected (tcp dport 22 accept)")
    }

    @Test func aJumpOrGotoInTheOutputChainIsFound() throws {
        for verb in ["jump", "goto"] {
            let listing = try protectionInserting(
                "\t\t\(verb) elsewhere\n", before: "\t\tip daddr @blocked4 reject\n", in: protectionGoodListing())
            #expect(protectionProblem(listing) == "sends traffic to other rules or to a program", "\(verb)")
        }
    }

    @Test func aQueueVerdictInTheOutputChainIsFound() throws {
        let listing = try protectionInserting(
            "\t\tqueue num 1\n", before: "\t\tip daddr @blocked4 reject\n", in: protectionGoodListing())
        #expect(protectionProblem(listing) == "sends traffic to other rules or to a program")
    }

    @Test func aSecondServerPortRuleIsFound() throws {
        let listing = try protectionInserting(
            "\t\tip saddr 10.9.9.9 tcp dport 9100 accept\n",
            before: "\t\tip saddr 192.168.64.1 tcp dport 9100 accept\n", in: protectionGoodListing())
        #expect(protectionProblem(listing) == "does not accept the server's port from the gateway exactly once")
    }

    @Test func aMissingServerPortRuleIsFound() throws {
        let listing = try protectionReplacing(
            "\t\tip saddr 192.168.64.1 tcp dport 9100 accept\n", with: "", in: protectionGoodListing())
        #expect(protectionProblem(listing) == "does not accept the server's port from the gateway exactly once")
    }

    @Test func aPrivateRangeMissingFromTheSetIsFound() throws {
        let listing = try protectionReplacing("192.168.0.0/16, ", with: "", in: protectionGoodListing())
        #expect(protectionProblem(listing) == "does not refuse the private range 192.168.0.0/16")
    }

    @Test func aJumpInTheInputChainIsFound() throws {
        let listing = try protectionInserting(
            "\t\tjump elsewhere\n", before: "\t\tip saddr 192.168.64.1 tcp dport 9100 accept\n",
            in: protectionGoodListing())
        #expect(protectionProblem(listing) == "sends inbound traffic to other rules or to a program")
    }

    @Test func aBareAcceptIsNotTheIPv4Accept() throws {
        let listing = try protectionReplacing(
            "\t\tmeta nfproto ipv4 accept\n", with: "\t\taccept\n", in: protectionGoodListing())
        #expect(protectionProblem(listing) == "accepts more than expected (accept)")
    }

    // MARK: Order of the generic accepts

    @Test func theGenericIPv4AcceptBeforeTheBlockedRulesIsFound() throws {
        let without = try protectionReplacing("\t\tmeta nfproto ipv4 accept\n", with: "", in: protectionGoodListing())
        let moved = try protectionInserting(
            "\t\tmeta nfproto ipv4 accept\n",
            before: "\t\tip daddr @blocked4 meta l4proto tcp reject with tcp reset\n", in: without)
        #expect(protectionProblem(moved)
            == "accepts IPv4 traffic before it rejects the blocked and private ranges")
    }

    @Test func theGenericIPv4AcceptBetweenBlockedAndPrivateRulesIsFound() throws {
        let without = try protectionReplacing("\t\tmeta nfproto ipv4 accept\n", with: "", in: protectionGoodListing())
        let moved = try protectionInserting(
            "\t\tmeta nfproto ipv4 accept\n",
            before: "\t\tip daddr @private4 meta l4proto tcp reject with tcp reset\n", in: without)
        #expect(protectionProblem(moved)
            == "accepts IPv4 traffic before it rejects the blocked and private ranges")
    }

    @Test func aBareAcceptBeforeThePrivateRulesIsFoundToo() throws {
        let without = try protectionReplacing("\t\tmeta nfproto ipv4 accept\n", with: "", in: protectionGoodListing())
        let moved = try protectionInserting(
            "\t\taccept\n", before: "\t\tip daddr @private4 reject\n", in: without)
        #expect(protectionProblem(moved) == "accepts more than expected (accept)")
    }

    @Test func theGlobalIPv6AcceptBeforeTheBlockedRulesIsFound() throws {
        let without = try protectionReplacing("\t\tip6 daddr 2000::/3 accept\n", with: "", in: protectionGoodListing())
        let moved = try protectionInserting(
            "\t\tip6 daddr 2000::/3 accept\n",
            before: "\t\tip6 daddr @blocked6 meta l4proto tcp reject with tcp reset\n", in: without)
        #expect(protectionProblem(moved)
            == "accepts IPv6 traffic before it rejects the blocked and refused ranges")
    }

    @Test func theGlobalIPv6AcceptBeforeTheRefusedRulesIsFound() throws {
        let without = try protectionReplacing("\t\tip6 daddr 2000::/3 accept\n", with: "", in: protectionGoodListing())
        let moved = try protectionInserting(
            "\t\tip6 daddr 2000::/3 accept\n",
            before: "\t\tip6 daddr @refused6 meta l4proto tcp reject with tcp reset\n", in: without)
        #expect(protectionProblem(moved)
            == "accepts IPv6 traffic before it rejects the blocked and refused ranges")
    }

    @Test func theGenericAcceptsMayBeLeftOutAltogether() throws {
        let without = try protectionReplacing("\t\tip6 daddr 2000::/3 accept\n", with: "", in: protectionGoodListing())
        #expect(protectionProblem(without) == nil)
    }

    // MARK: Inbound rules

    @Test func anExtraInboundAcceptIsFound() throws {
        let listing = try protectionReplacing(
            "\t\tip saddr 192.168.64.1 tcp dport 9100 accept\n",
            with: "\t\tip saddr 192.168.64.1 tcp dport 9100 accept\n\t\ttcp dport 22 accept\n",
            in: protectionGoodListing())
        #expect(protectionProblem(listing) == "accepts inbound traffic it should not (tcp dport 22 accept)")
    }

    @Test func variousOddInboundAcceptsAreFound() throws {
        let odd = [
            "accept",
            "ip saddr 192.168.64.1 tcp dport 22 accept",
            "ip saddr 192.168.64.1 udp dport 9100 accept",
            "ip saddr 0.0.0.0/0 tcp dport 9100 accept",
            "tcp dport 9100 accept",
            "ip6 saddr ::1 tcp dport 9100 accept",
            "ip saddr 192.168.64.1 tcp dport 9100 counter accept",
        ]
        for rule in odd {
            let listing = try protectionReplacing(
                "\t\tip saddr 192.168.64.1 tcp dport 9100 accept\n",
                with: "\t\tip saddr 192.168.64.1 tcp dport 9100 accept\n\t\t\(rule)\n",
                in: protectionGoodListing())
            #expect(protectionProblem(listing) == "accepts inbound traffic it should not (\(rule))", "rule \(rule)")
        }
    }

    // MARK: IPv6 unique-local and link-local addresses

    @Test(arguments: [
        "ip6 daddr fc00::/7 accept",
        "ip6 daddr fdcc::/16 accept",
        "ip6 daddr fd00::1 accept",
        "ip6 daddr fe80::/10 accept",
        "meta nfproto ipv6 ip6 daddr fc00::/7 accept",
    ])
    func anAcceptForUniqueLocalOrLinkLocalIPv6IsFound(rule: String) throws {
        let expected = "accepts more than expected (\(rule))"
        let before = try protectionInserting(
            "\t\t\(rule)\n", before: "\t\tip6 daddr 2000::/3 accept\n", in: protectionGoodListing())
        #expect(protectionProblem(before) == expected)
        let atTheEnd = try protectionInserting(
            "\t\t\(rule)\n", before: "\t\tmeta l4proto tcp reject with tcp reset\n\t\treject\n",
            in: protectionGoodListing())
        #expect(protectionProblem(atTheEnd) == expected)
        let atTheStart = try protectionInserting(
            "\t\t\(rule)\n", before: "\t\tip daddr @blocked4 reject\n", in: protectionGoodListing())
        #expect(protectionProblem(atTheStart) == expected)
    }

    @Test func theGoodListingAcceptsNoIPv6AddressesOutsideTheGlobalRange() throws {
        let listing = protectionGoodListing()
        let start = try #require(listing.range(of: "chain output {"))
        let end = try #require(listing.range(of: "chain input {"))
        let chain = String(listing[start.upperBound..<end.lowerBound])
        let lines = chain.split(whereSeparator: \.isNewline)
            .map { $0.trimmingCharacters(in: .whitespaces) }
            .filter { !$0.isEmpty && $0 != "}" }
        let accepts = lines.filter { $0.contains("accept") }
        #expect(accepts == [
            "ip daddr 192.168.64.1 udp dport 53 accept",
            "ip daddr 192.168.64.1 tcp dport 53 accept",
            "oifname \"lo\" accept",
            "ct state established,related accept",
            "meta nfproto ipv4 accept",
            "ip6 daddr 2000::/3 accept",
        ])
        // The one IPv6 accept is 2000::/3: the first three bits are 001.
        #expect(accepts.filter { $0.contains("ip6") } == ["ip6 daddr 2000::/3 accept"])
        func covered(byFirstByte byte: UInt8) -> Bool { byte & 0xe0 == 0x20 }
        #expect(covered(byFirstByte: 0x20))
        #expect(covered(byFirstByte: 0x2a))
        #expect(!covered(byFirstByte: 0xfc))   // fc00::/7 starts with fc or fd
        #expect(!covered(byFirstByte: 0xfd))
        #expect(!covered(byFirstByte: 0xfe))   // fe80::/10
        // Everything else that is not accepted ends in the two catch-all rejects.
        #expect(lines.suffix(2) == ["meta l4proto tcp reject with tcp reset", "reject"])
    }

    @Test func aBareIPv6AcceptIsNotMistakenForAnIPv4One() throws {
        // `meta nfproto ipv6 accept` would let all IPv6 traffic out, unique-local and
        // link-local addresses included, and skip the refused6 rules when it sits above
        // them. Only the exact `meta nfproto ipv4 accept` counts as the IPv4 accept.
        let listing = try protectionInserting(
            "\t\tmeta nfproto ipv6 accept\n",
            before: "\t\tmeta l4proto tcp reject with tcp reset\n\t\treject\n", in: protectionGoodListing())
        #expect(protectionProblem(listing) != nil)
    }
}

// MARK: verify

private struct ProtectionBrokenRunner: SandboxProcessRunning {
    struct Failure: LocalizedError {
        var errorDescription: String? { "the runner is broken" }
    }

    func run(executable: String, arguments: [String],
             timeout: TimeInterval) async throws -> SandboxProcessResult {
        throw Failure()
    }

    func runBlocking(executable: String, arguments: [String],
                     timeout: TimeInterval) -> SandboxProcessResult? {
        nil
    }
}

private let protectionContainerName = "tuff-sandbox-test0001"

private func protectionTool(_ runner: any SandboxProcessRunning) -> ContainerTool {
    ContainerTool(runner: runner, path: FakeContainerWorld.containerPath, isExecutable: { _ in true })
}

/// What `verify` throws, or nil when it passes.
private func protectionVerifyError(runner: any SandboxProcessRunning,
                                   blocked: [String] = protectionSampleBlocked,
                                   reported: Int? = nil) async -> SandboxError? {
    let health = SandboxHealth(profile: "web", api: 1, blockedAddresses: reported ?? blocked.count)
    do {
        try await SandboxProtection.verify(
            tool: protectionTool(runner), container: protectionContainerName,
            blocked: blocked, health: health)
        return nil
    } catch let error as SandboxError {
        return error
    } catch {
        Issue.record("unexpected error type: \(error)")
        return nil
    }
}

/// A fake world whose firewall shows `blocked`, as the sandbox would after start.
private func protectionWorld(blocked: [String] = protectionSampleBlocked) -> FakeContainerWorld {
    let world = FakeContainerWorld()
    world.nftListingOverride = FakeContainerWorld.goodNftListing(blocked: blocked)
    return world
}

@Suite struct SandboxProtectionVerifyTests {
    @Test func aProtectedSandboxPasses() async throws {
        let world = protectionWorld()
        let error = await protectionVerifyError(runner: world)
        #expect(error == nil)
        #expect(world.count("exec nft") == 1)
        #expect(world.count("exec python3") == 1)
    }

    @Test func theChecksRunInsideTheNamedContainer() async throws {
        let world = protectionWorld()
        _ = await protectionVerifyError(runner: world)
        #expect(world.containerCalls == [
            ["exec", protectionContainerName, "nft", "list", "table", "inet", "tuff_sandbox_web"],
            ["exec", protectionContainerName, "python3", "-c", SandboxProtection.privilegeCheck],
        ])
    }

    @Test func aSandboxWithNoBlockedAddressesPasses() async {
        let world = protectionWorld(blocked: [])
        #expect(await protectionVerifyError(runner: world, blocked: []) == nil)
    }

    @Test func theListingBuiltFromTheLastRunPasses() async throws {
        // Without an override, the fake prints the blocked addresses of the last `container run`.
        let world = FakeContainerWorld()
        let tool = protectionTool(world)
        try await tool.runDetached(arguments: [
            "--name", protectionContainerName,
            "--env", "SANDBOX_BLOCKED_ADDRESSES=127.0.0.1,192.168.1.20,::1",
        ])
        let blocked = ["127.0.0.1", "192.168.1.20", "::1"]
        #expect(world.lastRunBlockedAddresses == blocked)
        #expect(await protectionVerifyError(runner: world, blocked: blocked) == nil)
    }

    @Test func aBlockedCountThatDiffersIsRefusedBeforeAnythingRuns() async {
        let world = protectionWorld()
        let error = await protectionVerifyError(runner: world, reported: 3)
        #expect(error == SandboxError.protectionNotVerified("it blocks 3 addresses, but 4 were given"))
        #expect(world.calls.isEmpty)
        let more = await protectionVerifyError(runner: world, reported: 5)
        #expect(more == SandboxError.protectionNotVerified("it blocks 5 addresses, but 4 were given"))
        let none = await protectionVerifyError(runner: world, reported: 0)
        #expect(none == SandboxError.protectionNotVerified("it blocks 0 addresses, but 4 were given"))
        #expect(world.calls.isEmpty)
    }

    @Test func aBadPrivilegeAnswerIsRefused() async {
        let cases: [(output: String, shown: String)] = [
            ("0 none", "0 none"),
            ("10001 CapEff", "10001 CapEff"),
            ("", "nothing"),
            ("garbage", "garbage"),
        ]
        for item in cases {
            let world = protectionWorld()
            world.privilegeOutput = item.output
            let error = await protectionVerifyError(runner: world)
            #expect(error == SandboxError.protectionNotVerified(protectionPrivilegeMessage(item.shown)),
                    "output \(item.output)")
        }
    }

    @Test func aFailingPrivilegeCommandIsRefused() async {
        let world = protectionWorld()
        world.failures["exec python3"] = SandboxProcessResult(status: 1, output: "Traceback")
        let error = await protectionVerifyError(runner: world)
        #expect(error == SandboxError.protectionNotVerified(protectionPrivilegeMessage("Traceback")))
    }

    @Test func aBadFirewallListingIsRefused() async throws {
        let world = FakeContainerWorld()
        world.nftListingOverride = try protectionReplacing(
            "hook output priority filter; policy drop;",
            with: "hook output priority filter; policy accept;", in: protectionGoodListing())
        let error = await protectionVerifyError(runner: world)
        #expect(error == SandboxError.protectionNotVerified("the firewall does not drop outbound traffic by default"))
    }

    @Test func aMissingBlockedAddressInTheListingIsRefused() async {
        let world = protectionWorld(blocked: ["127.0.0.1", "::1", "fe80::1"])
        let error = await protectionVerifyError(runner: world)
        #expect(error == SandboxError.protectionNotVerified("the firewall does not block 192.168.1.20"))
    }

    @Test func aListingOfAnotherTableIsRefused() async {
        let world = FakeContainerWorld()
        world.nftListingOverride = "table inet filter {\n}\n"
        let error = await protectionVerifyError(runner: world)
        #expect(error == SandboxError.protectionNotVerified("the firewall is not the sandbox's table"))
    }

    @Test func theFirewallIsReportedBeforeThePrivileges() async throws {
        let world = FakeContainerWorld()
        world.nftListingOverride = try protectionReplacing(
            "hook input priority filter; policy drop;",
            with: "hook input priority filter; policy accept;", in: protectionGoodListing())
        world.privilegeOutput = "0 CapEff"
        let error = await protectionVerifyError(runner: world)
        #expect(error == SandboxError.protectionNotVerified("the firewall does not drop inbound traffic by default"))
        // Both commands ran before either answer was judged.
        #expect(world.count("exec nft") == 1)
        #expect(world.count("exec python3") == 1)
    }

    @Test func aFailingNftCommandMeansTheFirewallIsNotLoaded() async {
        let world = protectionWorld()
        world.failures["exec nft"] = SandboxProcessResult(status: 1, output: "Error: No such file or directory")
        let error = await protectionVerifyError(runner: world)
        #expect(error == SandboxError.protectionNotVerified("the firewall is not loaded"))
    }

    @Test func aTimedOutNftCommandMeansTheFirewallIsNotLoaded() async {
        let world = protectionWorld()
        world.failures["exec nft"] = SandboxProcessResult(
            status: 0, output: FakeContainerWorld.goodNftListing(blocked: protectionSampleBlocked), timedOut: true)
        let error = await protectionVerifyError(runner: world)
        #expect(error == SandboxError.protectionNotVerified("the firewall is not loaded"))
    }

    @Test func aRunnerThatThrowsMeansTheChecksCouldNotRun() async {
        let error = await protectionVerifyError(runner: ProtectionBrokenRunner())
        #expect(error == SandboxError.protectionNotVerified("the checks could not run (the runner is broken)"))
    }

    @Test func theErrorTextTellsThePersonWhatHappened() async {
        let world = protectionWorld()
        world.privilegeOutput = "0 none"
        let error = await protectionVerifyError(runner: world)
        let text = error?.description ?? ""
        #expect(text.hasPrefix("The sandbox's protection could not be verified, so it was not used: "))
        #expect(text.contains("found: 0 none"))
    }
}
#endif
