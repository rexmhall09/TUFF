#!/usr/bin/env python3
"""Tests for the files around server.py: the firewall, the default-gateway
helper, entrypoint.sh, the Containerfile and the requirements files.

Everything is plain file reading and shell runs with stand-ins for nft,
python3 and sleep, so it needs the standard library only. The tests that ask
the real nft to check a file are skipped when nft is missing or not allowed to
run here.

    cd Sandbox && python3 test_infra.py
"""
from __future__ import annotations

import ipaddress
import os
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import default_gateway  # noqa: E402
import server  # noqa: E402


def read(name):
    with open(os.path.join(HERE, name), encoding="utf-8") as handle:
        return handle.read()


def without_comments(text):
    return "\n".join(line.split("#", 1)[0].rstrip() for line in text.splitlines())


# --- firewall-web.nft -------------------------------------------------------

def nft_check(arguments, extra_text=None):
    """Runs `nft -c` on the firewall file (plus `extra_text`, appended). Returns
    the finished process, or skips the test when nft cannot check rules here."""
    if shutil.which("nft") is None:
        raise unittest.SkipTest("nft is not installed")
    path = os.path.join(HERE, "firewall-web.nft")
    cleanup = None
    if extra_text is not None:
        handle = tempfile.NamedTemporaryFile("w", suffix=".nft", delete=False)
        handle.write(read("firewall-web.nft") + "\n" + extra_text)
        handle.close()
        path = cleanup = handle.name
    try:
        result = subprocess.run(["nft", "-c", *arguments, "-f", path], capture_output=True,
                                text=True, timeout=60)
    finally:
        if cleanup:
            os.unlink(cleanup)
    return result


def nft_unavailable(stderr):
    return any(text in stderr for text in ("ermission", "not supported", "not permitted",
                                           "Protocol not supported"))


class FirewallFileTests(unittest.TestCase):
    def setUp(self):
        self.text = read("firewall-web.nft")
        self.code = without_comments(self.text)

    def chain(self, name):
        start = self.code.index(f"chain {name} {{")
        end = self.code.index("\n    }", start)
        return [line.strip() for line in self.code[start:end].splitlines()[1:] if line.strip()]

    def set_elements(self, name):
        match = re.search(r"set " + name + r" \{.*?elements = \{(.*?)\}", self.code, re.S)
        self.assertIsNotNone(match, name)
        return [item.strip() for item in match.group(1).replace("\n", " ").split(",")
                if item.strip()]

    def test_the_table_is_replaced_not_added_to(self):
        lines = [line.strip() for line in self.code.splitlines()]
        first = lines.index("table inet tuff_sandbox_web")
        delete = lines.index("delete table inet tuff_sandbox_web")
        define = lines.index("table inet tuff_sandbox_web {")
        self.assertLess(first, delete)
        self.assertLess(delete, define)
        self.assertEqual(self.code.count("table inet"), 3)
        self.assertEqual(self.code.count("tuff_sandbox_web"), 3)

    def test_inbound_drops_everything_new_except_the_servers_port(self):
        self.assertEqual(self.chain("input"), [
            "type filter hook input priority filter; policy drop;",
            'iifname "lo" accept',
            "ct state established,related accept",
            "ip saddr $gateway tcp dport 9100 accept",
        ])

    def test_server_port_is_9100(self):
        self.assertRegex(read("server.py"), r'PORT = int\(os\.environ\.get\("PORT", "9100"\)\)')
        if "PORT" not in os.environ:
            self.assertEqual(server.PORT, 9100)
        self.assertIn("EXPOSE 9100", read("Containerfile"))

    def test_port_9100_is_accepted_from_the_gateway_only_and_never_over_ipv6(self):
        lines = [line for line in self.chain("input") if "9100" in line]
        self.assertEqual(len(lines), 1)
        self.assertIn("ip saddr $gateway", lines[0])
        self.assertNotIn("ip6", "\n".join(self.chain("input")))
        # entrypoint.sh passes the gateway with -D; the file must not define it.
        self.assertNotIn("define gateway", self.code)
        self.assertEqual(self.code.count("$gateway"), 1)

    def test_each_hook_has_one_chain_and_9100_is_not_in_the_output_chain(self):
        self.assertEqual(self.code.count("hook input"), 1)
        self.assertEqual(self.code.count("hook output"), 1)
        self.assertNotIn("dport 9100", "\n".join(self.chain("output")))
        self.assertNotIn("forward", self.code)

    def test_outbound_policy_is_drop(self):
        self.assertIn("type filter hook output priority filter; policy drop;",
                      self.chain("output"))

    def test_the_output_chain_is_exactly_this(self):
        self.assertEqual(self.chain("output"), [
            "type filter hook output priority filter; policy drop;",
            'oifname "lo" accept',
            "ct state established,related accept",
            "meta nfproto ipv4 ip daddr @blocked4 meta l4proto tcp reject with tcp reset",
            "meta nfproto ipv4 ip daddr @blocked4 reject",
            "meta nfproto ipv6 ip6 daddr @blocked6 meta l4proto tcp reject with tcp reset",
            "meta nfproto ipv6 ip6 daddr @blocked6 reject",
            "meta nfproto ipv4 ip daddr @private4 meta l4proto tcp reject with tcp reset",
            "meta nfproto ipv4 ip daddr @private4 reject",
            "meta nfproto ipv4 accept",
            "meta nfproto ipv6 ip6 daddr @refused6 meta l4proto tcp reject with tcp reset",
            "meta nfproto ipv6 ip6 daddr @refused6 reject",
            "meta nfproto ipv6 ip6 daddr 2000::/3 accept",
            "meta l4proto tcp reject with tcp reset",
            "reject",
        ])

    def test_nothing_is_accepted_before_the_private_sets_are_refused(self):
        rules = self.chain("output")[1:]
        accepts = [i for i, rule in enumerate(rules) if rule.endswith("accept")]
        refuse4 = max(i for i, rule in enumerate(rules) if "@private4" in rule)
        refuse6 = max(i for i, rule in enumerate(rules) if "@refused6" in rule)
        general4 = rules.index("meta nfproto ipv4 accept")
        general6 = rules.index("meta nfproto ipv6 ip6 daddr 2000::/3 accept")
        self.assertLess(refuse4, general4)
        self.assertLess(refuse6, general6)
        # Only loopback and replies are accepted without a destination check.
        self.assertEqual([rules[i] for i in accepts[:2]],
                         ['oifname "lo" accept', "ct state established,related accept"])

    def test_the_blocked_sets_exist_and_start_empty(self):
        self.assertRegex(self.code, r"set blocked4 \{\s*type ipv4_addr\s*\}")
        self.assertRegex(self.code, r"set blocked6 \{\s*type ipv6_addr\s*\}")

    def test_blocked_address_rules_sit_after_the_established_accept_and_before_the_private_set(self):
        rules = self.chain("output")
        established = rules.index("ct state established,related accept")
        for family, ip in (("4", "ip"), ("6", "ip6")):
            nfproto = "ipv4" if family == "4" else "ipv6"
            reset = f"meta nfproto {nfproto} {ip} daddr @blocked{family} meta l4proto tcp reject with tcp reset"
            plain = f"meta nfproto {nfproto} {ip} daddr @blocked{family} reject"
            self.assertIn(reset, rules)
            self.assertIn(plain, rules)
            self.assertLess(established, rules.index(reset))
            self.assertLess(rules.index(reset), rules.index(plain), family)
        first_private = min(i for i, rule in enumerate(rules) if "@private4" in rule)
        first_refused6 = min(i for i, rule in enumerate(rules) if "@refused6" in rule)
        self.assertLess(rules.index("meta nfproto ipv4 ip daddr @blocked4 reject"), first_private)
        self.assertLess(rules.index("meta nfproto ipv6 ip6 daddr @blocked6 reject"), first_refused6)
        # Replies on connections the Mac opened are accepted before any of them.
        self.assertEqual(self.code.count("ct state established,related accept"), 2)

    def test_the_private_and_refused_sets_are_present(self):
        private4 = self.set_elements("private4")
        for network in ("10.0.0.0/8", "100.64.0.0/10", "127.0.0.0/8", "169.254.0.0/16",
                        "172.16.0.0/12", "192.168.0.0/16", "224.0.0.0/3", "0.0.0.0/8"):
            self.assertIn(network, private4)
        self.assertEqual(self.set_elements("refused6"), ["2001::/32", "2001:db8::/32", "2002::/16"])
        self.assertIn("type ipv4_addr", self.code)
        self.assertIn("type ipv6_addr", self.code)
        self.assertEqual(self.code.count("flags interval"), 2)

    def test_the_private_set_covers_what_the_server_refuses(self):
        networks = [ipaddress.ip_network(item) for item in self.set_elements("private4")]
        must_be_in = (
            "0.0.0.0", "0.1.2.3", "10.0.0.1", "10.255.255.255", "100.64.0.1", "100.127.255.254",
            "127.0.0.1", "127.255.255.255", "169.254.169.254", "172.16.0.1", "172.31.255.255",
            "192.0.0.1", "192.0.2.1", "192.88.99.1", "192.168.64.1", "192.168.0.1", "198.18.0.1",
            "198.19.255.255", "198.51.100.1", "203.0.113.1", "224.0.0.1", "239.255.255.255",
            "240.0.0.1", "255.255.255.255")
        must_not_be_in = (
            "1.1.1.1", "8.8.8.8", "93.184.216.34", "100.63.255.255", "100.128.0.0",
            "172.15.255.255", "172.32.0.0", "192.169.0.1", "198.17.255.255", "198.20.0.0",
            "223.255.255.255", "11.0.0.1", "9.255.255.255")
        for address in must_be_in:
            with self.subTest(address=address):
                self.assertTrue(any(ipaddress.ip_address(address) in n for n in networks))
                self.assertFalse(server.is_public_address(address))
        for address in must_not_be_in:
            with self.subTest(address=address):
                self.assertFalse(any(ipaddress.ip_address(address) in n for n in networks))
                self.assertTrue(server.is_public_address(address))

    def test_the_refused_ipv6_ranges_are_the_servers(self):
        listed = {ipaddress.ip_network(item) for item in self.set_elements("refused6")}
        self.assertEqual(listed, set(server.IPV6_REFUSED))
        self.assertIn(f"ip6 daddr {server.IPV6_UNICAST} accept", "\n".join(self.chain("output")))

    def test_the_file_loads_with_a_gateway_and_not_without(self):
        ok = nft_check(["-D", "gateway=192.168.64.1"])
        if nft_unavailable(ok.stderr):
            self.skipTest("nft cannot check rules here: " + ok.stderr.strip())
        self.assertEqual(ok.returncode, 0, ok.stderr)
        missing = nft_check([])
        self.assertNotEqual(missing.returncode, 0)
        self.assertIn("gateway", missing.stderr)

    def test_loaded_for_real_in_a_throwaway_network_namespace(self):
        if shutil.which("unshare") is None or shutil.which("nft") is None:
            self.skipTest("unshare or nft is not installed")
        path = os.path.join(HERE, "firewall-web.nft")
        script = (f"nft -D gateway=192.168.64.1 -f {shlex.quote(path)} && "
                  'nft add element inet tuff_sandbox_web blocked4 "{ 192.168.64.1 }" && '
                  'nft add element inet tuff_sandbox_web blocked6 "{ fe80::1 }" && '
                  "nft -nn list chain inet tuff_sandbox_web output && "
                  "nft -nn list set inet tuff_sandbox_web blocked4")
        result = subprocess.run(["unshare", "-n", "sh", "-c", script], capture_output=True,
                                text=True, timeout=60)
        if result.returncode != 0 and (nft_unavailable(result.stderr) or "unshare" in result.stderr):
            self.skipTest("cannot use a network namespace here: " + result.stderr.strip())
        self.assertEqual(result.returncode, 0, result.stderr)
        lines = [line.strip() for line in result.stdout.splitlines()]
        established = next(i for i, line in enumerate(lines) if line.startswith("ct state") and line.endswith("accept"))
        blocked4 = [i for i, line in enumerate(lines) if "@blocked4" in line and "reject" in line]
        blocked6 = [i for i, line in enumerate(lines) if "@blocked6" in line and "reject" in line]
        private4 = [i for i, line in enumerate(lines) if "@private4" in line]
        self.assertEqual(len(blocked4), 2, result.stdout)
        self.assertEqual(len(blocked6), 2, result.stdout)
        self.assertLess(established, min(blocked4 + blocked6))
        self.assertLess(max(blocked4 + blocked6), min(private4))
        self.assertIn("tcp reset", lines[blocked4[0]])
        self.assertIn("tcp reset", lines[blocked6[0]])
        self.assertIn("192.168.64.1", result.stdout.split("set blocked4")[1])


# --- default_gateway.py -----------------------------------------------------

SAMPLE_ROUTE = (
    "Iface\tDestination\tGateway \tFlags\tRefCnt\tUse\tMetric\tMask\t\tMTU\tWindow\tIRTT\n"
    "eth0\t00000000\t0140A8C0\t0003\t0\t0\t0\t00000000\t0\t0\t0\n"
    "eth0\t0040A8C0\t00000000\t0001\t0\t0\t0\t00FFFFFF\t0\t0\t0\n"
)


class DefaultGatewayTests(unittest.TestCase):
    """default_gateway.py feeds the firewall rule, so it must be strict."""

    def setUp(self):
        self.module = default_gateway

    def route(self, *rows):
        header = SAMPLE_ROUTE.splitlines()[0]
        return header + "\n" + "\n".join(rows) + "\n"

    def test_a_line_that_is_not_hex_is_skipped_whatever_the_header_says(self):
        text = "eth0 Destination Gateway Flags RefCnt Use Metric Mask MTU Window IRTT\n" \
            + SAMPLE_ROUTE.splitlines()[1] + "\n"
        self.assertEqual(self.module.default_gateway(text), "192.168.64.1")

    def test_reads_the_gateway_from_a_real_looking_route_table(self):
        self.assertEqual(self.module.default_gateway(SAMPLE_ROUTE), "192.168.64.1")

    def test_picks_the_default_route_with_the_lowest_metric(self):
        text = self.route(
            "eth1\t00000000\t0101A8C0\t0003\t0\t0\t200\t00000000\t0\t0\t0",
            "eth0\t00000000\t0140A8C0\t0003\t0\t0\t100\t00000000\t0\t0\t0")
        self.assertEqual(self.module.default_gateway(text), "192.168.64.1")

    def test_no_default_route_is_an_error(self):
        for text in ("", SAMPLE_ROUTE.splitlines()[0] + "\n",
                     self.route("eth0\t0040A8C0\t00000000\t0001\t0\t0\t0\t00FFFFFF\t0\t0\t0")):
            with self.assertRaises(self.module.GatewayError):
                self.module.default_gateway(text)

    def test_routes_that_are_not_usable_defaults_are_skipped(self):
        for row in (
            # on-link default (gateway 0), route down, no gateway flag, not a /0 mask
            "eth0\t00000000\t00000000\t0001\t0\t0\t0\t00000000\t0\t0\t0",
            "eth0\t00000000\t0140A8C0\t0002\t0\t0\t0\t00000000\t0\t0\t0",
            "eth0\t00000000\t0140A8C0\t0001\t0\t0\t0\t00000000\t0\t0\t0",
            "eth0\t00000000\t0140A8C0\t0003\t0\t0\t0\t000000FF\t0\t0\t0",
            "eth0\t00000000\t0140A8C0",
            "eth0\t00000000\tZZZZZZZZ\t0003\t0\t0\t0\t00000000\t0\t0\t0",
        ):
            with self.assertRaises(self.module.GatewayError, msg=row):
                self.module.default_gateway(self.route(row))

    def test_addresses_that_make_no_sense_as_a_gateway_are_refused(self):
        for hex_gateway in ("0100007F", "000000E0", "FFFFFFFF", "000000F0"):
            row = f"eth0\t00000000\t{hex_gateway}\t0003\t0\t0\t0\t00000000\t0\t0\t0"
            with self.assertRaises(self.module.GatewayError, msg=hex_gateway):
                self.module.default_gateway(self.route(row))

    def test_validate_accepts_only_plain_dotted_quads(self):
        self.assertEqual(self.module.validate("10.0.0.1"), "10.0.0.1")
        for bad in ("", "1.2.3", "1.2.3.4.5", "01.2.3.4", "1.2.3.256", " 1.2.3.4",
                    "1.2.3.4\n", "1.2.3.4; accept", "1.2.3.4/24", "::1", "1.2.3.4 ip saddr",
                    "$(id)", "0x7f.0.0.1", "1.2.3.٤"):
            with self.assertRaises(self.module.GatewayError, msg=repr(bad)):
                self.module.validate(bad)

    def test_a_better_metric_on_a_route_that_is_not_usable_does_not_win(self):
        text = self.route(
            "eth1\t00000000\t00000000\t0001\t0\t0\t1\t00000000\t0\t0\t0",
            "eth0\t00000000\t0140A8C0\t0003\t0\t0\t500\t00000000\t0\t0\t0")
        self.assertEqual(self.module.default_gateway(text), "192.168.64.1")

    def test_main_returns_1_with_a_message_and_prints_nothing_on_failure(self):
        import contextlib
        import io
        with tempfile.TemporaryDirectory() as directory:
            bad = os.path.join(directory, "bad")
            with open(bad, "w") as handle:
                handle.write("Iface\tDestination\n")
            out, err = io.StringIO(), io.StringIO()
            with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
                code = self.module.main(["default_gateway.py", bad])
            self.assertEqual((code, out.getvalue()), (1, ""))
            self.assertTrue(err.getvalue().strip())

    def test_command_line_prints_the_gateway_or_fails_with_a_message(self):
        script = os.path.join(HERE, "default_gateway.py")
        with tempfile.TemporaryDirectory() as directory:
            good, bad = (os.path.join(directory, n) for n in ("good", "bad"))
            with open(good, "w") as handle:
                handle.write(SAMPLE_ROUTE)
            with open(bad, "w") as handle:
                handle.write("Iface\tDestination\n")
            ok = subprocess.run([sys.executable, script, good], capture_output=True, text=True)
            self.assertEqual((ok.returncode, ok.stdout, ok.stderr), (0, "192.168.64.1\n", ""))
            for path in (bad, os.path.join(directory, "missing")):
                failed = subprocess.run([sys.executable, script, path], capture_output=True,
                                        text=True)
                self.assertEqual(failed.returncode, 1)
                self.assertEqual(failed.stdout, "")
                self.assertTrue(failed.stderr.strip())


# --- entrypoint.sh ----------------------------------------------------------

FIREWALL_CHAIN = "nft insert rule inet tuff_sandbox_web output"


def add_element(family, address):
    """The call entrypoint.sh makes for one blocked address (family 4 or 6)."""
    return f"nft add element inet tuff_sandbox_web blocked{family} {{ {address} }}"


def dns_rules(family, address):
    base = f"{FIREWALL_CHAIN} {family} daddr {address}"
    return [base + " udp dport 53 accept", base + " tcp dport 53 accept"]


class Run:
    def __init__(self, result, calls, server_env, directory_text):
        self.result = result
        self.returncode = result.returncode
        self.stderr = result.stderr
        self.calls = calls
        self.server_env = server_env
        self.script = directory_text

    @property
    def inserts(self):
        return [call for call in self.calls if call.startswith("nft insert")]

    @property
    def elements(self):
        return [call for call in self.calls if call.startswith("nft add element")]

    @property
    def server_started(self):
        return "python3 /app/server.py" in self.calls

    def index(self, call):
        return self.calls.index(call)


class EntrypointTests(unittest.TestCase):
    """Runs entrypoint.sh with stand-ins for nft, python3 and sleep.

    The python3 stand-in runs the real interpreter for `python3 -` and
    `python3 -c` (the address-parsing snippets inside the script), hands
    `python3 /app/default_gateway.py` to the helper under test, and only records
    every other call (in particular `python3 /app/server.py`)."""

    GATEWAY = "#!/bin/sh\necho 192.168.64.1\n"

    def run_entrypoint(self, helper_body=GATEWAY, nft_exit=0, nft_fail_on=None, blocked=None,
                       resolv=""):
        script = read("entrypoint.sh")
        self.assertIn("/etc/resolv.conf", script)
        self.assertIn("/app/firewall-web.nft", script)
        with tempfile.TemporaryDirectory() as directory:
            def write(name, body):
                path = os.path.join(directory, name)
                with open(path, "w") as handle:
                    handle.write(body)
                os.chmod(path, 0o755)
                return path

            log = os.path.join(directory, "calls")
            env_log = os.path.join(directory, "server-env")
            helper = write("helper", helper_body)
            write("python3", (
                "#!/bin/sh\n"
                'case "$1" in\n'
                f'  /app/default_gateway.py) shift; exec {shlex.quote(helper)} "$@" ;;\n'
                f'  -|-c) exec {shlex.quote(sys.executable)} "$@" ;;\n'
                "esac\n"
                f'echo "python3 $*" >> {shlex.quote(log)}\n'
                f'echo "${{SANDBOX_BLOCKED_ADDRESSES-<unset>}}" > {shlex.quote(env_log)}\n'))
            fail = ""
            if nft_fail_on:
                fail = f'case "$*" in *{nft_fail_on}*) exit 1 ;; esac\n'
            write("nft", f'#!/bin/sh\necho "nft $*" >> {shlex.quote(log)}\n{fail}exit {nft_exit}\n')
            write("sleep", "#!/bin/sh\nexit 0\n")
            resolv_path = write("resolv.conf", resolv)
            patched = script.replace("/etc/resolv.conf", resolv_path).replace(
                "/app/firewall-web.nft", "FIREWALL")
            entry = write("entrypoint.sh", patched)
            env = {"PATH": directory + ":/usr/bin:/bin"}
            if blocked is not None:
                env["SANDBOX_BLOCKED_ADDRESSES"] = blocked
            result = subprocess.run(["/bin/sh", entry], env=env, capture_output=True, text=True,
                                    timeout=60)
            calls, server_env = [], None
            if os.path.exists(log):
                with open(log) as handle:
                    calls = handle.read().splitlines()
            if os.path.exists(env_log):
                with open(env_log) as handle:
                    server_env = handle.read().rstrip("\n")
            return Run(result, calls, server_env, patched)

    # -- gateway and firewall ---------------------------------------------

    def test_passes_the_gateway_to_nft_and_logs_it(self):
        run = self.run_entrypoint()
        self.assertEqual(run.returncode, 0, run.stderr)
        self.assertIn("nft -D gateway=192.168.64.1 -f FIREWALL", run.calls)
        self.assertEqual(run.calls[0], "nft -D gateway=192.168.64.1 -f FIREWALL")
        self.assertEqual(run.calls[-1], "python3 /app/server.py")
        self.assertIn("192.168.64.1", run.stderr)
        self.assertEqual(run.inserts, [])

    def test_without_a_gateway_it_refuses_to_start_before_touching_the_firewall(self):
        run = self.run_entrypoint("#!/bin/sh\necho 'no default gateway found' >&2\nexit 1\n",
                                  blocked="10.0.0.1")
        self.assertEqual(run.returncode, 1)
        self.assertIn("refusing to start", run.stderr)
        self.assertIn("no default gateway found", run.stderr)
        self.assertEqual(run.calls, [])

    def test_a_gateway_that_does_not_look_like_ipv4_is_refused(self):
        for output in ("1.2.3.4; accept", "$(id)", "1.2.3.4 ", "..", "::1", ".1.2.3",
                       "0", "4294967295", "1.2.3", "1.2.3.4.5", "1234.1.1.1", "a.b.c.d"):
            run = self.run_entrypoint(f"#!/bin/sh\nprintf '%s\\n' '{output}'\n")
            self.assertEqual(run.returncode, 1, output)
            self.assertIn("is not an IPv4 address; refusing to start", run.stderr, output)
            self.assertEqual(run.calls, [], output)

    def test_a_firewall_that_fails_to_load_stops_the_start(self):
        run = self.run_entrypoint("#!/bin/sh\necho 10.0.0.1\n", nft_exit=1, blocked="10.0.0.5")
        self.assertEqual(run.returncode, 1)
        self.assertIn("could not load the sandbox firewall", run.stderr)
        self.assertFalse(run.server_started)
        self.assertEqual(run.inserts, [])

    def test_a_gateway_that_is_found_on_a_later_try_is_used(self):
        counter = "COUNT"
        body = (f'#!/bin/sh\nn=$(cat {counter} 2>/dev/null || echo 0)\n'
                f'echo $((n + 1)) > {counter}\n'
                'if [ "$n" -lt 3 ]; then echo "not yet" >&2; exit 1; fi\necho 192.168.64.1\n')
        with tempfile.TemporaryDirectory() as directory:
            body = body.replace("COUNT", os.path.join(directory, "count"))
            run = self.run_entrypoint(body)
        self.assertEqual(run.returncode, 0, run.stderr)
        self.assertIn("nft -D gateway=192.168.64.1 -f FIREWALL", run.calls)

    # -- SANDBOX_BLOCKED_ADDRESSES ----------------------------------------

    def test_each_blocked_address_is_added_to_the_set_of_its_family(self):
        run = self.run_entrypoint(blocked="192.168.1.5,fe80::1,2001:db8::5")
        self.assertEqual(run.returncode, 0, run.stderr)
        self.assertEqual(run.elements, [add_element(4, "192.168.1.5"), add_element(6, "fe80::1"),
                                        add_element(6, "2001:db8::5")])
        self.assertEqual(run.inserts, [])
        self.assertTrue(run.server_started)

    def test_whitespace_empty_parts_and_mapped_addresses(self):
        run = self.run_entrypoint(blocked=" 192.168.1.5 , ,, fe80::1,\t::ffff:10.0.0.9 ,  ")
        self.assertEqual(run.returncode, 0, run.stderr)
        self.assertEqual(run.elements, [add_element(4, "192.168.1.5"), add_element(6, "fe80::1"),
                                        add_element(4, "10.0.0.9")])

    def test_ipv6_addresses_are_written_in_their_short_form(self):
        run = self.run_entrypoint(blocked="FE80:0:0:0:0:0:0:1,2001:0DB8:0000:0000:0000:0000:0000:0005")
        self.assertEqual(run.elements, [add_element(6, "fe80::1"), add_element(6, "2001:db8::5")])

    def test_an_ipv4_mapped_address_in_any_spelling_becomes_the_ipv4_address(self):
        run = self.run_entrypoint(blocked="::ffff:192.0.2.7,::FFFF:c000:0207,0:0:0:0:0:ffff:c000:207")
        self.assertEqual(run.returncode, 0, run.stderr)
        self.assertEqual(run.elements, [add_element(4, "192.0.2.7")] * 3)

    def test_no_per_address_reject_rule_is_inserted_any_more(self):
        run = self.run_entrypoint(blocked="192.168.1.5,fe80::1")
        self.assertEqual([c for c in run.calls if "reject" in c], [])
        self.assertEqual(run.inserts, [])

    def test_nothing_is_added_when_the_variable_is_empty_or_unset(self):
        for blocked in (None, "", " ", ",", " , ,, "):
            with self.subTest(blocked=blocked):
                run = self.run_entrypoint(blocked=blocked)
                self.assertEqual(run.returncode, 0, run.stderr)
                self.assertEqual(run.elements, [])
                self.assertEqual(run.inserts, [])
                self.assertTrue(run.server_started)

    def test_the_blocked_addresses_are_added_before_the_dns_rules(self):
        run = self.run_entrypoint(
            blocked="192.168.1.5,fe80::1",
            resolv="# comment\nsearch example\nnameserver 192.168.64.1\nnameserver fd00::53\n"
                   "  # nameserver 1.1.1.1\noptions ndots:1\n")
        self.assertEqual(run.returncode, 0, run.stderr)
        blocked = [add_element(4, "192.168.1.5"), add_element(6, "fe80::1")]
        dns = dns_rules("ip", "192.168.64.1") + dns_rules("ip6", "fd00::53")
        self.assertEqual(run.calls, ["nft -D gateway=192.168.64.1 -f FIREWALL"] + blocked + dns
                         + ["python3 /app/server.py"])

    def test_dns_rules_are_added_without_a_blocklist_too(self):
        run = self.run_entrypoint(resolv="nameserver 192.168.64.1\n")
        self.assertEqual(run.returncode, 0, run.stderr)
        self.assertEqual(run.inserts, dns_rules("ip", "192.168.64.1"))
        self.assertEqual(run.elements, [])

    def test_no_nameserver_means_no_dns_rule(self):
        run = self.run_entrypoint(blocked="10.0.0.1", resolv="search example\n")
        self.assertEqual(run.elements, [add_element(4, "10.0.0.1")])
        self.assertEqual(run.inserts, [])
        self.assertTrue(run.server_started)

    def test_a_nameserver_that_is_not_an_address_stops_the_start(self):
        for line in ("nameserver not-an-address", "nameserver 1.2.3.4;accept", "nameserver $(id)"):
            with self.subTest(line=line):
                run = self.run_entrypoint(resolv=line + "\n")
                self.assertEqual(run.returncode, 1)
                self.assertIn("unreadable nameserver", run.stderr)
                self.assertIn("refusing to start", run.stderr)
                self.assertFalse(run.server_started)
                self.assertEqual(run.inserts, [])

    def test_a_malformed_list_stops_the_start_with_no_insert_and_no_server(self):
        for blocked in ("10.0.0.1,notanip", "notanip", "1.2.3.4/24", "999.1.1.1", "10.0.0.1;8.8.8.8",
                        "$(id)", "10.0.0.1 extra", "10.0.0.1 10.0.0.2", "localhost", "0x7f.0.0.1",
                        "fe80::1%eth0", "192.168.1.5,fe80::1%en0", "::1%1", "10.0.0.1,,%",
                        "010.0.0.1", "3232235777"):
            with self.subTest(blocked=blocked):
                run = self.run_entrypoint(blocked=blocked, resolv="nameserver 192.168.64.1\n")
                self.assertEqual(run.returncode, 1, run.stderr)
                self.assertIn("refusing to start", run.stderr)
                self.assertIn("SANDBOX_BLOCKED_ADDRESSES", run.stderr)
                self.assertEqual(run.inserts, [])
                self.assertEqual(run.elements, [])
                self.assertFalse(run.server_started)
                self.assertEqual(run.calls, ["nft -D gateway=192.168.64.1 -f FIREWALL"])

    def test_a_zone_id_is_named_in_the_message(self):
        run = self.run_entrypoint(blocked="fe80::1%en0")
        self.assertEqual(run.returncode, 1)
        self.assertIn("zone id", run.stderr)

    def test_a_failing_nft_call_stops_the_start(self):
        for fail_on in ("add element", "dport 53"):
            with self.subTest(fail_on=fail_on):
                run = self.run_entrypoint(blocked="10.0.0.1", resolv="nameserver 192.168.64.1\n",
                                          nft_fail_on=fail_on)
                self.assertNotEqual(run.returncode, 0)
                self.assertFalse(run.server_started)

    def test_the_server_still_sees_the_blocklist_it_parses_itself(self):
        run = self.run_entrypoint(blocked="10.0.0.1, 192.168.1.5")
        self.assertEqual(run.server_env, "10.0.0.1, 192.168.1.5")
        self.assertNotIn("unset SANDBOX_BLOCKED_ADDRESSES", run.script)

    def test_the_server_is_started_last_with_exec(self):
        script = read("entrypoint.sh")
        self.assertTrue(script.startswith("#!/bin/sh\n"))
        self.assertRegex(script, r"(?m)^set -eu$")
        self.assertEqual(script.rstrip().splitlines()[-1], "exec python3 /app/server.py")
        self.assertEqual(script.count("python3 /app/server.py"), 1)
        self.assertTrue(os.access(os.path.join(HERE, "entrypoint.sh"), os.X_OK))

    # -- the calls are valid nft ------------------------------------------

    def test_the_calls_are_valid_nft(self):
        run = self.run_entrypoint(
            blocked="192.168.1.5,fe80::1,::ffff:10.0.0.9",
            resolv="nameserver 192.168.64.1\nnameserver fd00::53\n")
        self.assertEqual(len(run.elements), 3)
        self.assertEqual(len(run.inserts), 4)
        text = "\n".join(line[len("nft "):] for line in run.calls[1:-1]) + "\n"
        result = nft_check(["-D", "gateway=192.168.64.1"], extra_text=text)
        if nft_unavailable(result.stderr):
            self.skipTest("nft cannot check rules here: " + result.stderr.strip())
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_the_calls_produce_the_right_chain_in_a_throwaway_network_namespace(self):
        if shutil.which("unshare") is None or shutil.which("nft") is None:
            self.skipTest("unshare or nft is not installed")
        run = self.run_entrypoint(blocked="192.168.64.1,10.9.9.9,fe80::1",
                                  resolv="nameserver 192.168.64.1\n")
        path = os.path.join(HERE, "firewall-web.nft")
        commands = [f"nft -D gateway=192.168.64.1 -f {shlex.quote(path)}"]
        for call in run.calls[1:-1]:
            commands.append(call.replace("{ ", '"{ ').replace(" }", ' }"'))
        commands.append("nft -nn list chain inet tuff_sandbox_web output")
        result = subprocess.run(["unshare", "-n", "sh", "-c", " && ".join(commands)],
                                capture_output=True, text=True, timeout=60)
        if result.returncode != 0 and (nft_unavailable(result.stderr) or "unshare" in result.stderr):
            self.skipTest("cannot use a network namespace here: " + result.stderr.strip())
        self.assertEqual(result.returncode, 0, result.stderr)
        lines = [line.strip() for line in result.stdout.splitlines()]
        dns = [i for i, line in enumerate(lines) if "dport 53 accept" in line]
        loopback = next(i for i, line in enumerate(lines) if line.startswith('oifname "lo"'))
        established = next(i for i, line in enumerate(lines) if line.startswith("ct state") and line.endswith("accept"))
        blocked = [i for i, line in enumerate(lines) if "@blocked" in line]
        self.assertEqual(len(dns), 2)
        self.assertEqual(len(blocked), 4)
        self.assertLess(max(dns), loopback)
        self.assertLess(established, min(blocked))


# --- Containerfile ----------------------------------------------------------

class ContainerfileTests(unittest.TestCase):
    def setUp(self):
        self.text = read("Containerfile")
        self.lines = [line for line in self.text.splitlines()
                      if line.strip() and not line.lstrip().startswith("#")]
        # Instructions with their continuation lines joined.
        self.instructions = []
        current = ""
        for line in self.lines:
            current += line.rstrip("\\").rstrip() + " " if line.endswith("\\") else line + "\n"
            if not line.endswith("\\"):
                self.instructions.append(current.strip())
                current = ""
        self.run = next(i for i in self.instructions if i.startswith("RUN "))

    def test_the_base_image_is_pinned_by_digest(self):
        froms = [i for i in self.instructions if i.startswith("FROM ")]
        self.assertEqual(len(froms), 1)
        self.assertRegex(froms[0], r"\AFROM \S+/python:3\.12-slim@sha256:[0-9a-f]{64}\Z")

    def test_requirements_are_installed_with_hashes_and_without_dependency_resolution(self):
        self.assertRegex(
            self.run,
            r"pip install --require-hashes --no-deps -r /app/requirements\.txt")
        self.assertNotRegex(self.run, r"pip install (?!--require-hashes)")
        copy = next(i for i in self.instructions if i.startswith("COPY requirements.txt"))
        self.assertTrue(copy.endswith("/app/requirements.txt"))
        self.assertLess(self.text.index(copy), self.text.index("pip install"))

    def test_apt_sources_are_rewritten_to_the_snapshot_and_the_build_fails_otherwise(self):
        run = self.run
        self.assertRegex(self.text, r"(?m)^ARG DEBIAN_SNAPSHOT=\d{8}T\d{6}Z$")
        self.assertIn("/etc/apt/sources.list.d/debian.sources", run)
        self.assertIn("/etc/apt/sources.list", run)
        sed = run.index("sed -i")
        check = run.index("grep -rE")
        update = run.index("apt-get update")
        install = run.index("apt-get install")
        pip = run.index("pip install")
        self.assertLess(sed, check)
        self.assertLess(check, update)
        self.assertLess(update, install)
        self.assertLess(install, pip)
        # The live archives are named in the check, and a hit stops the build.
        self.assertRegex(run, r"grep -rE '[^']*deb\\\.debian\\\.org[^']*'")
        self.assertIn("security", run[check:update])
        self.assertIn("exit 1", run[check:update])
        self.assertIn("set -eux", run)
        for needle in ("snapshot.debian.org/archive/debian/${DEBIAN_SNAPSHOT}",
                       "snapshot.debian.org/archive/debian-security/${DEBIAN_SNAPSHOT}"):
            self.assertIn(needle, run[sed:check])
        # debian-security is rewritten before debian, whose pattern is its prefix.
        self.assertLess(run.index("deb.debian.org/debian-security"),
                        run.index("deb.debian.org/debian#"))
        self.assertIn("security.debian.org", run[sed:check])
        self.assertIn("nftables", run)

    def test_the_copy_list_has_the_four_runtime_files_and_no_tests(self):
        sources = []
        for instruction in self.instructions:
            if instruction.startswith("COPY "):
                parts = [p for p in instruction.split()[1:] if not p.startswith("--")]
                sources += parts[:-1]
        for needed in ("server.py", "firewall-web.nft", "entrypoint.sh", "default_gateway.py"):
            self.assertIn(needed, sources)
        for source in sources:
            self.assertNotIn("test", source.lower(), source)
            self.assertNotIn(source, (".", "./", "*", "Sandbox", "Sandbox/"))
            self.assertFalse(source.endswith((".md", ".json")), source)
            self.assertNotEqual(source, "sandbox_client.py")
        self.assertEqual(sorted(set(sources)), sorted(sources))
        self.assertNotRegex(self.text, r"(?m)^(ADD|VOLUME)\b")
        self.assertEqual(sorted(sources), sorted(
            ["requirements.txt", "server.py", "firewall-web.nft", "entrypoint.sh",
             "default_gateway.py"]))

    def test_every_copied_file_exists(self):
        for name in ("requirements.txt", "server.py", "firewall-web.nft", "entrypoint.sh",
                     "default_gateway.py"):
            self.assertTrue(os.path.isfile(os.path.join(HERE, name)), name)

    def test_the_entrypoint_is_the_script_and_the_image_does_not_start_as_the_sandbox_user(self):
        self.assertRegex(self.text, r'(?m)^ENTRYPOINT \["/app/entrypoint\.sh"\]$')
        self.assertEqual(self.text.count("ENTRYPOINT"), 1)
        self.assertNotRegex(self.text, r"(?m)^(USER|CMD)\b")
        self.assertIn("COPY server.py firewall-web.nft entrypoint.sh default_gateway.py /app/",
                      self.text)
        self.assertTrue(os.access(os.path.join(HERE, "entrypoint.sh"), os.X_OK))

    def test_the_user_and_port_match_server_py(self):
        self.assertIn(f"--uid {server.SANDBOX_UID} ", self.run)
        self.assertEqual(server.SANDBOX_UID, 10001)
        self.assertIn("/usr/sbin/nologin", self.run)
        self.assertRegex(self.text, r"(?m)^EXPOSE 9100$")

    def test_image_hygiene(self):
        self.assertIn("PYTHONDONTWRITEBYTECODE=1", self.text)
        self.assertIn("PIP_NO_CACHE_DIR=1", self.text)
        self.assertIn("rm -rf /var/lib/apt/lists/*", self.run)


# --- requirements -----------------------------------------------------------

def logical_lines(text):
    """Requirement lines with their `\\` continuations joined; comments dropped."""
    result, current = [], ""
    for raw in text.splitlines():
        line = raw.strip()
        if not current and (not line or line.startswith("#")):
            continue
        if line.endswith("\\"):
            current += line[:-1].strip() + " "
        else:
            result.append((current + line).strip())
            current = ""
    return result


class RequirementsTests(unittest.TestCase):
    def test_every_pinned_requirement_has_a_hash(self):
        entries = logical_lines(read("requirements.txt"))
        self.assertGreater(len(entries), 10)
        names = set()
        for entry in entries:
            with self.subTest(entry=entry[:60]):
                match = re.match(r"\A([A-Za-z0-9][A-Za-z0-9._-]*)==([A-Za-z0-9.!+_-]+) (.*)\Z", entry)
                self.assertIsNotNone(match, entry[:80])
                name, version, rest = match.groups()
                hashes = rest.split()
                self.assertTrue(hashes)
                for item in hashes:
                    self.assertRegex(item, r"\A--hash=sha256:[0-9a-f]{64}\Z")
                self.assertEqual(len(set(hashes)), len(hashes))
                self.assertNotIn(name.lower(), names)
                names.add(name.lower())

    def test_nothing_in_requirements_txt_is_unpinned_or_comes_from_elsewhere(self):
        text = read("requirements.txt")
        for line in text.splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            self.assertFalse(stripped.startswith(("-e", "--index-url", "--extra-index-url",
                                                  "--find-links", "-f ", "-i ", "git+", "http")),
                             stripped[:60])
            self.assertNotIn("@", stripped.split("--hash")[0])
            self.assertNotRegex(stripped.split("--hash")[0], r"(>=|<=|~=|>|<|!=|===)")

    def test_requirements_in_pins_the_direct_dependencies_exactly(self):
        pins = {}
        for line in read("requirements.in").splitlines():
            line = line.split("#", 1)[0].strip()
            if line:
                match = re.fullmatch(r"([A-Za-z0-9._-]+)==([0-9][A-Za-z0-9.+!_-]*)", line)
                self.assertIsNotNone(match, line)
                pins[match.group(1).lower()] = match.group(2)
        self.assertIn("trafilatura", pins)
        self.assertIn("pypdf", pins)
        self.assertEqual(set(pins), {"trafilatura", "pypdf"})

    def test_requirements_txt_has_the_same_versions_as_requirements_in(self):
        locked = {}
        for entry in logical_lines(read("requirements.txt")):
            name, _, rest = entry.partition("==")
            locked[name.lower()] = rest.split()[0]
        for line in read("requirements.in").splitlines():
            line = line.split("#", 1)[0].strip()
            if line:
                name, _, version = line.partition("==")
                self.assertEqual(locked.get(name.lower()), version, name)

    def test_pypdf_is_listed_without_extras(self):
        # The extras would add crypto and image libraries to the image.
        for name in ("requirements.in", "requirements.txt"):
            self.assertNotRegex(read(name), r"(?m)^pypdf\[")


if __name__ == "__main__":
    unittest.main()
