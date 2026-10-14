#!/usr/bin/env python3
"""Prints the VM's default IPv4 gateway, read from /proc/net/route.

entrypoint.sh puts the answer into the inbound firewall rule, so it is parsed
strictly and anything doubtful is an error: exit status 1 and a message on
stderr, nothing on stdout. The image has no iproute2, so this reads the kernel's
table directly.

Usage: default_gateway.py [route-file]    (default /proc/net/route)
"""
import ipaddress
import re
import socket
import struct
import sys

ROUTE_FILE = "/proc/net/route"
RTF_UP = 0x1
RTF_GATEWAY = 0x2
DOTTED_QUAD = re.compile(r"(?:0|[1-9][0-9]{0,2})(?:\.(?:0|[1-9][0-9]{0,2})){3}")


class GatewayError(Exception):
    pass


def default_gateway(text):
    """The gateway of the default route with the lowest metric, as a string.

    /proc/net/route columns: Iface Destination Gateway Flags RefCnt Use
    Metric Mask MTU Window IRTT. The address bytes are in network order; the
    kernel prints them as one native-endian u32 in hex, so "=L" below (not
    "<L") turns that number back into the bytes. A default route has
    destination and mask 0, is up, and has a gateway (an on-link default route
    has gateway 0 and is skipped). Raises GatewayError when there is none or
    the address is not an ordinary unicast IPv4 address.
    """
    best = None
    for line in text.splitlines():
        fields = line.split()
        if len(fields) < 8 or fields[0] == "Iface":
            continue
        try:
            destination, gateway, flags = int(fields[1], 16), int(fields[2], 16), int(fields[3], 16)
            metric, mask = int(fields[6]), int(fields[7], 16)
        except ValueError:
            continue
        if destination != 0 or mask != 0 or gateway == 0:
            continue
        if flags & (RTF_UP | RTF_GATEWAY) != (RTF_UP | RTF_GATEWAY):
            continue
        if best is None or metric < best[0]:
            best = (metric, gateway)
    if best is None:
        raise GatewayError("no default gateway found in the routing table")
    text_address = socket.inet_ntoa(struct.pack("=L", best[1]))
    return validate(text_address)


def validate(text_address):
    if not DOTTED_QUAD.fullmatch(text_address):
        raise GatewayError(f"default gateway is not a dotted-quad IPv4 address: {text_address!r}")
    try:
        address = ipaddress.IPv4Address(text_address)
    except ValueError:
        raise GatewayError(f"default gateway is not a valid IPv4 address: {text_address!r}")
    if (address.is_unspecified or address.is_loopback or address.is_multicast
            or address.is_reserved):
        raise GatewayError(f"default gateway is not a usable unicast address: {text_address}")
    return str(address)


def main(argv):
    path = argv[1] if len(argv) > 1 else ROUTE_FILE
    try:
        with open(path, encoding="ascii", errors="replace") as handle:
            text = handle.read()
        print(default_gateway(text))
    except (OSError, GatewayError) as error:
        sys.stderr.write(f"{error}\n")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
