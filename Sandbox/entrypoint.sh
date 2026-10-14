#!/bin/sh
# Loads the VM's firewall (outbound and inbound), then starts the server, which
# reads its secret, drops root and every capability before it serves anything.
# Refuses to start without the firewall.
#
# SANDBOX_PROFILE picks the firewall: web (the default) loads firewall-web.nft,
# adds the Mac's addresses to its blocked sets and lets the VM's name servers
# through; files loads firewall-files.nft, which allows no outbound connection
# at all, so neither of those two steps applies.
set -eu
profile="${SANDBOX_PROFILE-web}"
case "$profile" in
  web) firewall=/app/firewall-web.nft ;;
  files) firewall=/app/firewall-files.nft ;;
  *) echo "SANDBOX_PROFILE must be web or files, not '$profile'; refusing to start" >&2; exit 1 ;;
esac
# Inbound port 9100 is accepted from the VM's default gateway only, the address
# `container run --publish` connects from. Read it from the routing table
# (the network may take a moment to come up), check it again here, and refuse
# to start without it rather than open 9100 to every address.
gateway=""
for _ in 1 2 3 4 5 6 7 8 9 10; do
  gateway="$(python3 /app/default_gateway.py 2>/dev/null)" && break
  gateway=""
  sleep 0.5
done
if [ -z "$gateway" ]; then
  python3 /app/default_gateway.py >&2 || true
  echo "no usable default gateway; cannot restrict port 9100 to it, refusing to start" >&2
  exit 1
fi
# Exactly four dot-separated groups of one to three digits.
if [ "$(expr "x$gateway" : 'x[0-9]\{1,3\}\.[0-9]\{1,3\}\.[0-9]\{1,3\}\.[0-9]\{1,3\}$')" = 0 ]; then
  echo "default gateway '$gateway' is not an IPv4 address; refusing to start" >&2
  exit 1
fi
if ! nft -D "gateway=$gateway" -f "$firewall"; then
  echo "could not load the sandbox firewall; refusing to start" >&2
  exit 1
fi
echo "firewall: inbound port 9100 accepted from the default gateway $gateway only" >&2
# The Mac's own addresses (web profile only), passed in by the app as a
# comma-separated list. Each one is refused, in the firewall (the sets blocked4
# and blocked6) and again in server.py. A list that cannot be read is a mistake
# in the launcher: refuse to start. The server reads the same variable, so it
# is not unset here.
if [ "$profile" = web ] && [ -n "${SANDBOX_BLOCKED_ADDRESSES:-}" ]; then
  rules="$(python3 - "$SANDBOX_BLOCKED_ADDRESSES" <<'PY'
import ipaddress, sys
for part in sys.argv[1].split(","):
    part = part.strip()
    if not part:
        continue
    if "%" in part:
        sys.exit(f"an address with a zone id is not accepted: {part[:64]!r}")
    ip = ipaddress.ip_address(part)
    if isinstance(ip, ipaddress.IPv6Address) and ip.ipv4_mapped is not None:
        ip = ip.ipv4_mapped
    print(ip.version, ip)
PY
)" || { echo "SANDBOX_BLOCKED_ADDRESSES cannot be read; refusing to start" >&2; exit 1; }
  # Each address goes into the set of its family (see firewall-web.nft). The
  # here-document, not a pipe, keeps a failing nft inside this shell, so set -e
  # stops the start.
  while read -r family address; do
    [ -n "$address" ] || continue
    nft add element inet tuff_sandbox_web "blocked$family" "{ $address }"
  done <<RULES
$rules
RULES
fi
# The VM's DNS servers (web profile only; the files VM has no DNS). By default that is the Mac, which forwards to the same
# DNS the rest of the Mac uses; the firewall refuses the Mac, so its address
# gets an exception for DNS (port 53) only. `insert` puts these rules at the top
# of the chain, above the blocked-address rules.
nameservers=""
if [ "$profile" = web ]; then
  nameservers="$(awk '$1 == "nameserver" { print $2 }' /etc/resolv.conf)" || nameservers=""
fi
for server in $nameservers; do
  family="$(python3 -c 'import ipaddress, sys; print("ip6" if ipaddress.ip_address(sys.argv[1]).version == 6 else "ip")' "$server")" \
    || { echo "unreadable nameserver $server; refusing to start" >&2; exit 1; }
  nft insert rule inet tuff_sandbox_web output "$family" daddr "$server" udp dport 53 accept
  nft insert rule inet tuff_sandbox_web output "$family" daddr "$server" tcp dport 53 accept
done
exec python3 /app/server.py
