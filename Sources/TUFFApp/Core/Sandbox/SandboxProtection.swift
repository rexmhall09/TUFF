#if os(macOS)
import Foundation

/// Checks, from outside the VM, that the sandbox that answered is protected
/// the way `Sandbox/` builds it: the firewall is loaded with the Mac's
/// addresses blocked, and the web server runs as an unprivileged user with
/// no capabilities. The signed `/health` answer already shows that the server
/// is this version's sandbox; these checks show that what surrounds it still
/// holds. They run before the sandbox is used, and again on request.
enum SandboxProtection {
    static let tableName = "tuff_sandbox_web"

    /// Ranges the sandbox must never reach (the private4 set of
    /// `Sandbox/firewall-web.nft` holds more).
    static let privateRanges = ["10.0.0.0/8", "172.16.0.0/12", "192.168.0.0/16",
                                "169.254.0.0/16", "127.0.0.0/8", "100.64.0.0/10"]

    /// Prints the user id and any capability sets of the process running
    /// /app/server.py, as `Scripts/sandbox_selftest.sh` does.
    static let privilegeCheck = """
    import glob
    for path in glob.glob("/proc/[0-9]*/cmdline"):
        try:
            if b"/app/server.py" not in open(path, "rb").read().split(b"\\0"):
                continue
            status = dict(l.split(":", 1) for l in open(path[:-7] + "status") if ":" in l)
        except OSError:
            continue
        caps = [n for n in ("CapPrm", "CapEff", "CapBnd", "CapAmb") if int(status[n], 16)]
        uids = set(status["Uid"].split())
        if len(uids) != 1 or "0" in status["Gid"].split():
            print("mixed-ids", ",".join(caps) or "none")
        else:
            print(uids.pop(), ",".join(caps) or "none")
        break
    """

    /// Throws `protectionNotVerified` with the first thing that is wrong.
    /// `blocked` is the list the sandbox was started with, already
    /// normalized (see `HostAddresses.normalized`).
    static func verify(tool: ContainerTool, container: String, blocked: [String],
                       health: SandboxHealth) async throws {
        guard health.blockedAddresses == blocked.count else {
            throw SandboxError.protectionNotVerified(
                "it blocks \(health.blockedAddresses) addresses, but \(blocked.count) were given")
        }
        let listing: SandboxProcessResult
        let privileges: SandboxProcessResult
        do {
            listing = try await tool.exec(container, ["nft", "list", "table", "inet", tableName])
            privileges = try await tool.exec(container, ["python3", "-c", privilegeCheck])
        } catch {
            throw SandboxError.protectionNotVerified(
                "the checks could not run (\(SandboxText.oneLine(error.localizedDescription)))")
        }
        guard listing.status == 0, !listing.timedOut else {
            throw SandboxError.protectionNotVerified("the firewall is not loaded")
        }
        if let problem = firewallProblem(listing.output, blocked: blocked) {
            throw SandboxError.protectionNotVerified("the firewall \(problem)")
        }
        if let problem = privilegeProblem(privileges.output, status: privileges.status) {
            throw SandboxError.protectionNotVerified(problem)
        }
    }

    /// `nil` when the server runs as user 10001 with no capability set.
    static func privilegeProblem(_ output: String, status: Int32) -> String? {
        let found = output.trimmingCharacters(in: .whitespacesAndNewlines)
        if status == 0, found == "10001 none" { return nil }
        let shown = found.isEmpty ? "nothing" : SandboxText.oneLine(found, limit: 80)
        return "the web server does not run as the unprivileged user without capabilities (found: \(shown))"
    }

    /// Reads `nft list table inet tuff_sandbox_web` and says what is
    /// wrong, or nil when it has the shape `Sandbox/firewall-web.nft` gives
    /// it: both hooks drop by default, every given address is in the blocked
    /// set of its family, the blocked and private ranges are rejected before
    /// anything is accepted, and the only accepts are the ones the file
    /// writes.
    static func firewallProblem(_ listing: String, blocked: [String]) -> String? {
        guard listing.contains("table inet \(tableName)") else { return "is not the sandbox's table" }
        guard let output = block(named: "chain output", in: listing),
              let input = block(named: "chain input", in: listing) else {
            return "has no output or input rules"
        }
        let outputLines = lines(of: output)
        let inputLines = lines(of: input)
        guard outputLines.contains(where: { $0.contains("hook output") && $0.contains("policy drop") }) else {
            return "does not drop outbound traffic by default"
        }
        guard inputLines.contains(where: { $0.contains("hook input") && $0.contains("policy drop") }) else {
            return "does not drop inbound traffic by default"
        }
        for (name, family) in [("blocked4", "4"), ("blocked6", "6")] {
            let wanted = blocked.filter { $0.contains(":") == (family == "6") }
            guard let set = block(named: "set \(name)", in: listing) else {
                return "has no set of blocked addresses (\(name))"
            }
            let present = Set(addresses(in: set))
            if let missing = wanted.first(where: { !present.contains($0) }) {
                return "does not block \(missing)"
            }
        }
        if let set = block(named: "set private4", in: listing) {
            for range in privateRanges where !set.contains(range) {
                return "does not refuse the private range \(range)"
            }
        }
        return outputProblem(outputLines) ?? inputProblem(inputLines)
    }

    private static func outputProblem(_ lines: [String]) -> String? {
        if lines.contains(where: { $0.contains("jump ") || $0.contains("goto ") || $0.contains("queue") }) {
            return "sends traffic to other rules or to a program"
        }
        func indexes(of set: String) -> [Int] {
            lines.indices.filter { lines[$0].contains("@\(set)") }
        }
        for set in ["blocked4", "blocked6", "private4"] where indexes(of: set).isEmpty {
            return "has no rule for @\(set)"
        }
        for set in ["blocked4", "blocked6", "private4", "refused6"] {
            for index in indexes(of: set) {
                let line = lines[index]
                if line.contains("accept") || !line.contains("reject") {
                    return "does not reject @\(set) (\(SandboxText.oneLine(line, limit: 80)))"
                }
            }
        }
        var widestV4: Int?
        var widestV6: Int?
        for (index, line) in lines.enumerated() where line.contains("accept") {
            switch plainAccept(line) {
            case .loopback, .replies, .dns:
                continue
            case .anyIPv4:
                widestV4 = widestV4 ?? index
            case .globalIPv6:
                widestV6 = widestV6 ?? index
            case .other:
                return "accepts more than expected (\(SandboxText.oneLine(line, limit: 80)))"
            }
        }
        let earlier = ["blocked4", "blocked6", "private4"].flatMap { indexes(of: $0) }
        if let widestV4, earlier.contains(where: { $0 > widestV4 }) {
            return "accepts IPv4 traffic before it rejects the blocked and private ranges"
        }
        if let widestV6 {
            let later = indexes(of: "blocked6") + indexes(of: "refused6")
            if later.contains(where: { $0 > widestV6 }) {
                return "accepts IPv6 traffic before it rejects the blocked and refused ranges"
            }
        }
        return nil
    }

    private static func inputProblem(_ lines: [String]) -> String? {
        let serverRules = lines.filter { $0.wholeMatch(of: #/(?:meta nfproto ipv4 )?ip saddr [0-9.]+ tcp dport 9100 accept/#) != nil }
        if serverRules.count != 1 { return "does not accept the server's port from the gateway exactly once" }
        if lines.contains(where: { $0.contains("jump ") || $0.contains("goto ") || $0.contains("queue") }) {
            return "sends inbound traffic to other rules or to a program"
        }
        for line in lines where line.contains("accept") {
            let plain = line.replacingOccurrences(of: "meta nfproto ipv4 ", with: "")
            if plain == "iifname \"lo\" accept" || plain == "ct state established,related accept" {
                continue
            }
            if plain.wholeMatch(of: #/ip saddr [0-9.]+ tcp dport 9100 accept/#) != nil { continue }
            return "accepts inbound traffic it should not (\(SandboxText.oneLine(line, limit: 80)))"
        }
        return nil
    }

    private enum Accept { case loopback, replies, dns, anyIPv4, globalIPv6, other }

    /// What an `accept` line in the output chain is, in the forms
    /// `firewall-web.nft` and `entrypoint.sh` write (nft may leave out the
    /// `meta nfproto` part when it prints them back).
    private static func plainAccept(_ line: String) -> Accept {
        // Only the IPv4 form counts as "any address": a bare `accept` or
        // `meta nfproto ipv6 accept` would let IPv6 out too.
        if line == "meta nfproto ipv4 accept" { return .anyIPv4 }
        let plain = line.replacingOccurrences(of: "meta nfproto ipv4 ", with: "")
        switch plain {
        case "oifname \"lo\" accept": return .loopback
        case "ct state established,related accept": return .replies
        case "ip6 daddr 2000::/3 accept": return .globalIPv6
        default:
            if plain.wholeMatch(of: #/ip6? daddr [0-9a-fA-F.:]+ (?:udp|tcp) dport (?:53|domain) accept/#) != nil {
                return .dns
            }
            return .other
        }
    }

    /// The text of `set blocked4 {` or `chain output {` up to the start of
    /// the next set or chain.
    private static func block(named name: String, in listing: String) -> String? {
        guard let start = listing.range(of: "\(name) {") else { return nil }
        let rest = listing[start.upperBound...]
        let ends = ["\n\tset ", "\n\tchain ", "\n    set ", "\n    chain "]
            .compactMap { rest.range(of: $0)?.lowerBound }
        let end = ends.min() ?? rest.endIndex
        return String(rest[..<end])
    }

    private static func lines(of block: String) -> [String] {
        block.split(whereSeparator: \.isNewline)
            .map { $0.trimmingCharacters(in: .whitespaces) }
            .filter { !$0.isEmpty && $0 != "}" }
    }

    /// Every IP address in the elements of a set, in the form
    /// `HostAddresses.normalized` gives.
    private static func addresses(in set: String) -> [String] {
        guard let elements = set.range(of: "elements =") else { return [] }
        let tokens = set[elements.upperBound...].split { character in
            !(character.isHexDigit || character == "." || character == ":")
        }
        return tokens.compactMap { HostAddresses.canonical(String($0)) }
    }
}
#endif
