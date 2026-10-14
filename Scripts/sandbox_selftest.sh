#!/usr/bin/env bash
# Builds, starts and checks the web sandbox (Sandbox/) with Apple `container`.
# For development and the Mac tests: the app itself starts the sandbox from the
# published image once the lifecycle code exists.
#
#   Scripts/sandbox_selftest.sh build      build the image tuff-sandbox:dev from Sandbox/
#   Scripts/sandbox_selftest.sh start      start a fresh sandbox on 127.0.0.1:<port>
#   Scripts/sandbox_selftest.sh stop       stop and remove it and its secret folder
#   Scripts/sandbox_selftest.sh selftest   check the running sandbox
#   Scripts/sandbox_selftest.sh all        build, start, selftest, stop
#
# Environment: TUFF_SANDBOX_PORT (default 9100), TUFF_SANDBOX_IMAGE (default
# tuff-sandbox:dev; set it to a pulled image to test that one, then skip
# `build`). The self-test needs python3, curl and the Mac's `ifconfig`.
#
# `start` makes a new random secret in a folder only you can use, mounts that
# folder read-only into the VM, and passes the Mac's own addresses so the VM
# can refuse them, the way the app does. Every request the self-test sends is
# signed, and every answer is checked, with Sandbox/sandbox_client.py.

set -euo pipefail
set -f

image="${TUFF_SANDBOX_IMAGE:-tuff-sandbox:dev}"
name="tuff-sandbox-web-selftest"
port="${TUFF_SANDBOX_PORT:-9100}"
script_directory="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
context="$script_directory/../Sandbox"
client="$context/sandbox_client.py"
# Where `start` puts the secret folder; `stop` removes it.
state="${TMPDIR:-/tmp}/tuff-sandbox-selftest"
secret_folder="$state/secret"
secret_file="$secret_folder/secret"
given_file="$state/blocked-addresses"
# A public address the self-test adds to the blocklist (see `start`).
test_address="198.41.0.4"

if ! [[ "$port" =~ ^[1-9][0-9]{1,4}$ ]] || (( 10#$port < 10 || 10#$port > 65535 )); then
  echo "TUFF_SANDBOX_PORT must be a port number from 10 to 65535, not: $port" >&2
  exit 2
fi
port=$((10#$port))

require_container() {
  if ! command -v container >/dev/null 2>&1; then
    echo "Apple container is not installed; see https://github.com/apple/container/releases" >&2
    exit 1
  fi
  # The kernel install prompts if it is not asked for explicitly.
  container system status >/dev/null 2>&1 || container system start --enable-kernel-install
}

# The Mac's own interface addresses, one per line, without zone ids. The VM is
# told to refuse every one of them (SANDBOX_BLOCKED_ADDRESSES).
mac_addresses() {
  ifconfig | awk '$1 == "inet" || $1 == "inet6" { sub(/%.*/, "", $2); print $2 }' | sort -u
}

new_secret() {
  # A fresh folder (0700) with the secret (0600): 32 random bytes as 64 hex
  # characters, made without the secret ever being in a command line.
  rm -rf -- "$state"
  mkdir -p -m 700 -- "$secret_folder"
  (umask 077 && LC_ALL=C od -An -N32 -tx1 /dev/urandom | tr -d ' \n' >"$secret_file")
  [[ "$(wc -c <"$secret_file" | tr -d ' ')" == 64 ]] || { echo "could not make a secret" >&2; exit 1; }
}

call() { # METHOD PATH [JSON]: prints "STATUS<newline>BODY"; exit 2 if the answer is not signed right
  python3 -I "$client" --port "$port" --secret-file "$secret_file" "$@"
}

json_field() { # FIELD: reads the printed answer (status line, then JSON body) on stdin
  python3 -I -c '
import json, sys
lines = sys.stdin.read().split("\n", 1)
try:
    body = json.loads(lines[1]) if len(lines) > 1 and lines[1].strip() else {}
except ValueError:  # not JSON: something other than the sandbox answered
    body = {}
if not isinstance(body, dict):
    body = {}
field = sys.argv[1]
print(body.get("error", {}).get("code", "") if field == "error" else body.get(field, ""))
' "$1"
}

in_vm() { container exec "$name" "$@"; }

case "${1:-}" in
  build)
    require_container
    builder_was_running=1
    if builder="$(container builder status --quiet 2>/dev/null)" && [[ -z "$builder" ]]; then
      builder_was_running=""
    fi
    build_status=0
    container build -t "$image" "$context" || build_status=$?
    [[ -z "$builder_was_running" ]] && container builder stop >/dev/null 2>&1 || true
    exit "$build_status"
    ;;
  start)
    if [[ $- == *x* ]]; then
      echo "not starting with xtrace on: the trace could print the sandbox secret" >&2
      exit 1
    fi
    require_container
    container stop "$name" >/dev/null 2>&1 || true
    container delete "$name" >/dev/null 2>&1 || true
    new_secret
    blocked="$(mac_addresses | paste -sd, -)"
    # Almost every address of a Mac is private, link-local or loopback, and the
    # VM refuses those anyway, so they do not show that the blocklist works. One
    # public address (a.root-servers.net, nothing TUFF talks to) is added here
    # only for the self-test, so the check can tell the list from the rest.
    blocked="$blocked,$test_address"
    # What the VM is given, kept for the self-test: the Mac gains addresses
    # after this point (the vmnet bridge exists only while a container runs).
    printf '%s\n' "$blocked" | tr ',' '\n' >"$given_file"
    # The entrypoint starts as root with these four capabilities only, to load
    # the firewall; the server then reads the secret, gives them all up and runs
    # as user 10001.
    container run --detach --rm --name "$name" \
      --read-only --tmpfs /tmp \
      --cap-drop ALL --cap-add NET_ADMIN --cap-add SETUID --cap-add SETGID --cap-add SETPCAP \
      --ulimit nproc=512 --cpus 2 --memory 1G \
      --mount "type=bind,source=$secret_folder,target=/run/tuff-sandbox,readonly" \
      --env "SANDBOX_BLOCKED_ADDRESSES=$blocked" \
      --publish "127.0.0.1:$port:9100" \
      "$image" >/dev/null
    for _ in $(seq 1 100); do
      if call GET /health >/dev/null 2>&1; then
        echo "web sandbox ready at http://127.0.0.1:$port (blocking: $blocked)"
        exit 0
      fi
      if out="$(container logs "$name" 2>/dev/null)" && grep -q "refusing to start" <<<"$out"; then
        echo "the sandbox refused to start:" >&2
        tail -n 5 <<<"$out" >&2
        exit 1
      fi
      sleep 0.3
    done
    echo "the sandbox did not answer; see: container logs $name" >&2
    exit 1
    ;;
  stop)
    require_container
    container stop "$name" >/dev/null 2>&1 || true
    container delete "$name" >/dev/null 2>&1 || true
    rm -rf -- "$state"
    if remaining="$(container list --all --quiet 2>/dev/null)" && grep -qxF "$name" <<<"$remaining"; then
      echo "could not confirm the sandbox was removed; see: container list --all" >&2
      exit 1
    fi
    echo "web sandbox stopped"
    ;;
  selftest)
    require_container
    [[ -f "$secret_file" ]] && call GET /health >/dev/null 2>&1 \
      || { echo "start the sandbox first: $0 start" >&2; exit 1; }
    failures=0
    pass() { echo "  ok    $1"; }
    fail() { echo "  FAIL  $1" >&2; failures=$((failures + 1)); }
    gateway="$(in_vm python3 /app/default_gateway.py)" \
      || { echo "  FAIL  could not read the VM's default gateway" >&2; exit 1; }
    # (macOS ships bash 3.2, which has no mapfile.)
    mac=()
    while IFS= read -r line; do [[ -n "$line" ]] && mac+=("$line"); done < <(mac_addresses)
    # The addresses the VM was started with. The Mac's list now can hold more
    # (the vmnet bridge exists only while the VM runs); those are private or
    # link-local, refused by the private-range rules, and tested below as URLs.
    given=()
    if [[ -f "$given_file" ]]; then
      while IFS= read -r line; do [[ -n "$line" ]] && given+=("$line"); done <"$given_file"
    else
      given=("${mac[@]}" "$test_address")
    fi
    late=""
    for address in "${mac[@]}"; do
      printf '%s\n' "${given[@]}" | grep -qFx -- "$address" || late+=" $address"
    done

    echo "Health:"
    health="$(call GET /health)" || fail "/health is not signed correctly"
    [[ "$health" == *'"profile": "web"'* ]] && pass "/health answers, signed" || fail "/health: $health"
    count="$(printf '%s' "$health" | json_field blocked_addresses)"
    expected="$(printf '%s\n' "${given[@]}" | python3 -I -c '
import ipaddress, sys
found = set()
for line in sys.stdin:
    if line.strip():
        ip = ipaddress.ip_address(line.strip())
        found.add(ip.ipv4_mapped if getattr(ip, "ipv4_mapped", None) else ip)
print(len(found))
')"
    [[ "$count" == "$expected" ]] \
      && pass "the sandbox blocks all $count addresses it was given (the Mac's and one test address)" \
      || fail "the sandbox reports $count blocked addresses, $expected were given"
    [[ -z "$late" ]] || echo "  note  addresses the Mac gained after the start (not on the blocklist, the bridge ones are private ranges, tested as URLs below):$late"

    echo "Page requests the sandbox must refuse:"
    blocked_urls=(
      "http://$gateway/" "http://$gateway:8080/" "http://127.0.0.1:8080/" "http://localhost/"
      "http://[::1]/" "http://10.0.0.1/" "http://172.16.0.1/" "http://192.168.1.1/"
      "http://169.254.169.254/latest/meta-data/" "http://100.100.100.200/" "http://0.0.0.0/"
      "http://2130706433/" "http://localtest.me/" "http://example.com:8080/"
      "file:///etc/passwd" "ftp://example.com/"
    )
    for address in "${mac[@]}"; do
      [[ "$address" == *:* ]] && blocked_urls+=("http://[$address]/") || blocked_urls+=("http://$address/")
    done
    # Public, so only the blocklist can be why it is refused.
    reply="$(call POST /v1/page "{\"url\": \"http://$test_address/\", \"timeout_seconds\": 20}" || true)"
    code="$(printf '%s' "$reply" | json_field error)"
    [[ "$code" == private_address ]] \
      && pass "the public test address $test_address on the blocklist is refused by the server" \
      || fail "the blocklisted public address $test_address was not refused by the server: ${reply:0:200}"
    for url in "${blocked_urls[@]}"; do
      reply="$(call POST /v1/page "{\"url\": \"$url\", \"timeout_seconds\": 20}" || true)"
      code="$(printf '%s' "$reply" | json_field error)"
      case "$code" in
        private_address|blocked_port|invalid_url|unsupported_scheme|dns_error) pass "$url refused ($code)" ;;
        *) fail "$url was not refused: ${reply:0:200}" ;;
      esac
    done

    echo "Search requests the sandbox must refuse:"
    refuse_http() { # LABEL EXPECTED_CODE JSON
      local reply code
      reply="$(call POST /v1/http "$3" || true)"
      code="$(printf '%s' "$reply" | json_field error)"
      [[ "$code" == "$2" ]] && pass "$1 refused ($code)" || fail "$1 was not refused ($2 expected): ${reply:0:200}"
    }
    refuse_http "a site that is not a search provider" forbidden_origin \
      '{"url": "https://example.com/", "method": "GET"}'
    refuse_http "http instead of https" forbidden_origin \
      '{"url": "http://html.duckduckgo.com/html/", "method": "GET"}'
    refuse_http "a Cookie header" invalid_header \
      '{"url": "https://html.duckduckgo.com/html/", "method": "GET", "headers": {"Cookie": "a=b"}}'
    refuse_http "a Brave key sent to Tavily" invalid_header \
      '{"url": "https://api.tavily.com/search", "method": "POST", "headers": {"X-Subscription-Token": "x"}}'
    refuse_http "a Tavily key sent to DuckDuckGo" invalid_header \
      '{"url": "https://html.duckduckgo.com/html/", "method": "GET", "headers": {"Authorization": "Bearer x"}}'

    echo "The VM's own protection:"
    if in_vm nft list table inet tuff_sandbox_web >/dev/null 2>&1; then
      pass "the firewall is loaded"
    else
      fail "the firewall is not loaded"
    fi
    input_rules="$(in_vm nft -nn list chain inet tuff_sandbox_web input 2>/dev/null || true)"
    if grep -q 'hook input.*policy drop' <<<"$input_rules" \
        && grep -qF "ip saddr $gateway tcp dport 9100 accept" <<<"$input_rules" \
        && [[ "$(grep -c 'dport 9100' <<<"$input_rules")" == 1 ]]; then
      pass "new inbound connections are dropped except to port 9100 from the gateway $gateway"
    else
      fail "the input chain is missing or not as expected: ${input_rules:0:200}"
    fi
    # The addresses are elements of the sets blocked4 and blocked6 (see
    # firewall-web.nft), which the output chain rejects.
    sets="$(in_vm nft -nn list sets 2>/dev/null || true)"
    missing=""
    for address in "${given[@]}"; do
      # Whole elements only: "10.0.0.1" must not match "10.0.0.12".
      tr -c '0-9a-fA-F:.' '\n' <<<"$sets" | grep -qFx -- "$address" || missing+=" $address"
    done
    [[ -z "$missing" ]] && pass "every blocked address is in the firewall's blocked sets" \
      || fail "not in the firewall's blocked sets:$missing"
    output_rules="$(in_vm nft -nn list chain inet tuff_sandbox_web output 2>/dev/null || true)"
    grep -q '@blocked4' <<<"$output_rules" && grep -q '@blocked6' <<<"$output_rules" \
      && pass "the output chain rejects both blocked sets" \
      || fail "the output chain does not use the blocked sets: ${output_rules:0:200}"
    # Only the firewall can make this fail with a refusal (a connection reset)
    # at once: without the rule the address is simply unanswered.
    refusal="$(container exec --user 10001:10001 "$name" python3 -c "
import socket, sys
try:
    socket.create_connection((sys.argv[1], 80), timeout=4).close()
    print('connected')
except ConnectionRefusedError:
    print('refused')
except OSError as error:
    print('other', type(error).__name__)
" "$test_address" 2>/dev/null || true)"
    [[ "$refusal" == refused ]] \
      && pass "a connection to the blocklisted public address $test_address is refused by the firewall" \
      || fail "a connection to $test_address: ${refusal:-no answer} (refused expected)"
    privileges="$(in_vm python3 -c '
import glob
for path in glob.glob("/proc/[0-9]*/cmdline"):
    try:
        if b"/app/server.py" not in open(path, "rb").read().split(b"\0"):
            continue
        status = dict(l.split(":", 1) for l in open(path[:-7] + "status") if ":" in l)
    except OSError:
        continue
    caps = [n for n in ("CapPrm", "CapEff", "CapBnd", "CapAmb") if int(status[n], 16)]
    print(status["Uid"].split()[0], ",".join(caps) or "none")
    break
' 2>/dev/null || true)"
    [[ "$privileges" == "10001 none" ]] \
      && pass "the server runs as user 10001 with no capabilities" \
      || fail "the server's user and capabilities are not as expected: ${privileges:-not found}"

    echo "Direct connections from inside the VM (as code that took the server over), which must fail:"
    targets=("$gateway:8080" "$gateway:22" "$gateway:443" "192.168.1.1:80" "10.0.0.1:80" "169.254.169.254:80")
    for address in "${mac[@]}"; do
      [[ "$address" == *:* ]] && targets+=("[$address]:80") || targets+=("$address:80")
    done
    for target in "${targets[@]}"; do
      if container exec --user 10001:10001 "$name" python3 -c "
import socket, sys
host, port = sys.argv[1].rsplit(':', 1)
try:
    socket.create_connection((host.strip('[]'), int(port)), timeout=3).close()
except OSError:
    sys.exit(1)
" "$target" >/dev/null 2>&1; then
        fail "the VM can open a connection to $target"
      else
        pass "the VM cannot connect to $target"
      fi
    done

    echo "Every TCP port on the Mac (its VM-side address), from inside the VM:"
    open_ports="$(container exec --user 10001:10001 "$name" python3 -c '
import socket, sys
from concurrent.futures import ThreadPoolExecutor
host = sys.argv[1]
def probe(port):
    try:
        socket.create_connection((host, port), timeout=1).close()
        return port
    except OSError:
        return None
with ThreadPoolExecutor(64) as pool:
    print(" ".join(str(p) for p in pool.map(probe, range(1, 65536)) if p))
' "$gateway" 2>/dev/null)" || open_ports="error"
    mac_dns="no"
    in_vm awk '$1 == "nameserver" { print $2 }' /etc/resolv.conf 2>/dev/null | grep -qx "$gateway" && mac_dns="yes"
    case "$open_ports" in
      "") pass "no port on the Mac is reachable" ;;
      53) [[ "$mac_dns" == "yes" ]] && pass "only the Mac's DNS port (53) is reachable, as configured" \
            || fail "the Mac's port 53 is reachable although the VM does not use the Mac for DNS" ;;
      error) fail "the port scan of the Mac did not run" ;;
      *) fail "the VM can reach these ports on the Mac: $open_ports" ;;
    esac

    echo "Requests to the sandbox API that must be refused:"
    plain() { curl -s -o /dev/null -w '%{http_code}' --max-time 10 "$@"; }
    s="$(plain "http://127.0.0.1:$port/health")"
    [[ "$s" == 401 ]] && pass "an unsigned request is refused" || fail "an unsigned request answered $s"
    s="$(plain -H 'Host: attacker.example' "http://127.0.0.1:$port/health")"
    [[ "$s" == 403 ]] && pass "a foreign Host header is refused" || fail "a foreign Host header answered $s"
    s="$(plain -H 'Content-Type: text/plain' --data '{"url":"https://example.com/"}' "http://127.0.0.1:$port/v1/page")"
    [[ "$s" == 401 || "$s" == 403 || "$s" == 415 ]] && pass "a browser-style simple POST is refused ($s)" \
      || fail "a simple POST answered $s"
    replay="$(python3 -I - "$client" "$port" "$secret_file" <<'PY'
import importlib.util, os, sys, time, http.client
spec = importlib.util.spec_from_file_location("client", sys.argv[1])
client = importlib.util.module_from_spec(spec); spec.loader.exec_module(client)
port, secret = int(sys.argv[2]), client.read_secret(sys.argv[3])
nonce, stamp = os.urandom(32).hex(), str(int(time.time()))
auth = client.request_auth(secret, "GET", "/health", nonce, stamp, b"")
codes = []
for _ in range(2):
    c = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
    c.request("GET", "/health", headers={"Host": f"127.0.0.1:{port}", "X-Sandbox-Nonce": nonce,
                                          "X-Sandbox-Time": stamp, "X-Sandbox-Auth": auth})
    codes.append(c.getresponse().status); c.close()
stale = str(int(time.time()) - 120)
c = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
n2 = os.urandom(32).hex()
c.request("GET", "/health", headers={"Host": f"127.0.0.1:{port}", "X-Sandbox-Nonce": n2,
          "X-Sandbox-Time": stale, "X-Sandbox-Auth": client.request_auth(secret, "GET", "/health", n2, stale, b"")})
codes.append(c.getresponse().status)
print(*codes)
PY
)"
    [[ "$replay" == "200 401 401" ]] \
      && pass "a repeated request and a request two minutes old are refused" \
      || fail "replay and stale checks answered: $replay (200 401 401 expected)"

    echo "Reading pages:"
    reply="$(call POST /v1/page '{"url": "https://example.com/"}' || true)"
    # The words are in the page body, not in its title: the title is read in
    # the server process, the body by the extraction child.
    body_text="$(printf '%s' "$reply" | json_field text)"
    if [[ "$body_text" == *"xample"* ]]; then
      pass "https://example.com/ read as text"
    else
      fail "https://example.com/ was not read as text (the extraction may not run); check the VM's internet access: ${reply:0:200}"
    fi
    reply="$(call POST /v1/page '{"url": "https://www.w3.org/WAI/ER/tests/xhtml/testfiles/resources/pdf/dummy.pdf"}' || true)"
    body_text="$(printf '%s' "$reply" | json_field text)"
    if [[ "$reply" == *"application/pdf"* && -n "$body_text" ]]; then
      pass "a PDF read as text inside the VM"
    else
      echo "  warn  the sample PDF could not be read (the site may have moved): ${reply:0:160}"
    fi
    reply="$(call POST /v1/page '{"url": "https://example.com/this-page-does-not-exist-tuff"}' || true)"
    [[ "$reply" == *'"status": 404'* ]] && pass "a 404 comes back as status 404 with no text" \
      || fail "a missing page: ${reply:0:200}"
    reply="$(call POST /v1/http '{"url": "https://html.duckduckgo.com/html/", "method": "POST", "headers": {"Content-Type": "application/x-www-form-urlencoded"}, "body_base64": "cT10dWZmKw=="}' || true)"
    [[ "$reply" == *'"status": '* && "$reply" == *body_base64* ]] \
      && pass "a DuckDuckGo search went through /v1/http" \
      || fail "the DuckDuckGo request: ${reply:0:200}"

    echo "A hostile PDF:"
    # Passed with -c, not on standard input: see the file for why.
    verdict="$(container exec --user 10001:10001 "$name" python3 -c "$(cat "$context/selftest_pdf_bomb.py")" 2>&1 || true)"
    [[ "$verdict" == ok* ]] && pass "$verdict" || fail "decompression bomb: ${verdict:0:200}"
    call GET /health >/dev/null 2>&1 && pass "the server still answers afterwards" \
      || fail "the server does not answer after the bomb"
    echo "Many idle connections must not stop the server:"
    flood="$(python3 -I - "$port" "$client" "$secret_file" <<'PY'
import importlib.util, socket, sys, time
port = int(sys.argv[1])
spec = importlib.util.spec_from_file_location("client", sys.argv[2])
client = importlib.util.module_from_spec(spec); spec.loader.exec_module(client)
secret = client.read_secret(sys.argv[3])
held = []
for _ in range(200):
    try:
        held.append(socket.create_connection(("127.0.0.1", port), timeout=2))
    except OSError:
        break
time.sleep(12)  # longer than the read deadline of the server (10 s); the sockets stay open
try:
    status, _, signed = client.call(port, secret, "GET", "/health", timeout=15)
    print("answered" if status == 200 and signed else f"status {status}")
except OSError:
    print("no answer")
PY
)"
    # The server serves at most 32 connections at once and cuts a connection
    # that has not finished its request after 10 s; the idle ones are still open
    # on this side when the request is made.
    [[ "$flood" == answered ]] \
      && pass "the server answers while 200 idle connections are open (after its 10 s read deadline)" \
      || fail "the server did not answer with 200 idle connections open: $flood"

    if [[ "$failures" -gt 0 ]]; then
      echo "$failures check(s) failed" >&2
      exit 1
    fi
    echo "all sandbox checks passed"
    ;;
  all)
    "$0" build
    "$0" start
    status=0
    "$0" selftest || status=$?
    "$0" stop || true
    exit "$status"
    ;;
  *)
    sed -n '2,12p' "$0" | sed 's/^# \{0,1\}//'
    exit 2
    ;;
esac
