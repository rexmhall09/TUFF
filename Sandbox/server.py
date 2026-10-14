#!/usr/bin/env python3
"""Web sandbox server for TUFF (the `web` profile).

This process runs inside an Apple `container` Linux VM and is published only
on the Mac's loopback address. It is the part of TUFF that touches the
internet when the sandbox setting is on: pages and search answers are
downloaded here, behind the VM's firewall, and pages and PDFs are turned into
text here, so the app only receives finished text.

Endpoints (JSON in, JSON out). Every request is signed, see "Signing" below.
  GET  /health
  POST /v1/page  {"url": str, "max_bytes": int?, "timeout_seconds": number?,
                  "max_characters": int?}
  POST /v1/http  {"url": str, "method": "GET"|"POST", "headers": object?,
                  "body_base64": str?, "max_bytes": int?, "timeout_seconds": number?}

/v1/page fetches a web page and answers {final_url, status, content_type,
title, text}. HTML is turned into text and a PDF is read with pypdf, each in a
child process with a time limit (and, for a PDF, a memory limit), so a hostile
page or PDF can only end its own child. Only the text leaves the VM. A page
that answers with an HTTP error status comes back with that status and no
text; a type that is neither a web page, plain text nor a PDF is refused.

/v1/http is the narrow door for search requests. It reaches only
https://html.duckduckgo.com, https://api.search.brave.com and
https://api.tavily.com (port 443), sends only the headers on the allow-list,
and answers {final_url, status, content_type, body_base64}. The API key
headers go to their own provider only: Authorization to api.tavily.com and
X-Subscription-Token to api.search.brave.com. No header or body is logged.

Both endpoints refuse anything but http(s) on ports 80 and 443 (443 only for
/v1/http), and refuse every address that is not globally routable, and every
address in SANDBOX_BLOCKED_ADDRESSES (the Mac's own addresses, which the app
passes in at start). The resolved address is the one connected to, so a DNS
answer cannot change between the check and the connection. Redirects are
followed by hand, checked the same way and under one time limit; they follow
the rules of the app's PinnedHTTPTransport: never from https to http, and a
request with a body or a key header is never sent on to another origin.

Those checks live in this process, so they would not bind code that took it
over. firewall-web.nft, loaded by entrypoint.sh before this server starts,
applies the same rule to the whole VM, with one exception for DNS to the VM's
name server, and the server then drops root and every Linux capability so it
cannot change the firewall.

Signing. A request carries
  X-Sandbox-Nonce: 64 hexadecimal characters, fresh for every request
  X-Sandbox-Time:  whole seconds since 1970, in decimal
  X-Sandbox-Auth:  hex(HMAC-SHA256(secret, "tuff-sandbox-req-v1\\n" + method
                   + "\\n" + path + "\\n" + nonce + "\\n" + time + "\\n"
                   + hex(SHA256(request body))))
and the server refuses a request whose values are missing or wrong, whose time
is more than 60 s from its own clock, or whose nonce it has already accepted
within that window. Every answer to a request with a well-formed nonce carries
  X-Sandbox-Auth: hex(HMAC-SHA256(secret, "tuff-sandbox-resp-v1\\n" + method
                   + "\\n" + path + "\\n" + nonce + "\\n" + status + "\\n"
                   + hex(SHA256(answer body))))
which the app checks before it parses anything. An answer that the server's
HTTP library sends before this code runs (a malformed request line, an
unsupported HTTP version, an overlong line) is not signed, so the app treats
any unsigned answer as an error. A signed 401 or 4xx is an error too: the
server signs error answers for any well-formed nonce, so a signature proves
who answered, not that the request succeeded. An answer to HEAD is signed over
the body a GET would carry, and has none.

The secret is the content of the file SANDBOX_SECRET_FILE without one trailing
newline, at least 32 bytes. The server reads it once, as root, before it drops
its privileges, and refuses to start without it. Reading it as root does not
hide it from the unprivileged user: a folder mounted into the VM is readable by
every user in it (see the mount test in the #38 issue), so a program that
takes over the server's extraction child could read it. It protects the app
against other programs on the Mac, not against code running in the VM. Sandbox/test-vectors/signing.json holds fixed examples
that both this code and the app's tests must reproduce.
"""

from __future__ import annotations

import base64
import codecs
import ctypes
import errno
import gzip
import hashlib
import hmac
import html
import http.client
import io
import ipaddress
import json
import logging
import multiprocessing
import os
import re
import resource
import signal
import socket
import ssl
import stat
import sys
import threading
import time
import unicodedata
import urllib.parse
import zlib
from dataclasses import dataclass
from html.parser import HTMLParser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Callable

try:
    import trafilatura
except ImportError:  # The extractor is optional so tests can run without it.
    trafilatura = None

try:
    import pypdf
    from pypdf import PdfReader
except ImportError:  # Without pypdf a PDF is refused; /health says "pdf": false.
    pypdf = PdfReader = None

PORT = int(os.environ.get("PORT", "9100"))
USER_AGENT = os.environ.get(
    "USER_AGENT", "Mozilla/5.0 (compatible; TUFF-Sandbox/1.0; +https://github.com/rexmhall09/TUFF)")
# Each wait for a connection or data. Some official sites take over 20 s to
# answer, so a request's own timeout_seconds (at most MAX_TIMEOUT) decides.
FETCH_TIMEOUT = float(os.environ.get("FETCH_TIMEOUT", "30"))
# The longest one request may take when its caller gives no deadline.
FETCH_DEADLINE = float(os.environ.get("FETCH_DEADLINE", "45"))
# Extraction runs in a separate process with its own limit, because a hostile
# page can keep the extractor busy for a long time and Python threads cannot be
# stopped. Only the first MAX_EXTRACT_CHARS of a page are read.
EXTRACT_TIMEOUT = float(os.environ.get("EXTRACT_TIMEOUT", "15"))
MAX_EXTRACT_CHARS = 2_000_000
# The fallback parser (html.parser) runs in a child as well, over fewer
# characters and with a shorter limit than the extractor.
FALLBACK_TIMEOUT = float(os.environ.get("FALLBACK_TIMEOUT", "10"))
MAX_FALLBACK_CHARS = 500_000
MAX_DOWNLOAD_BYTES = int(os.environ.get("MAX_DOWNLOAD_BYTES", str(5 * 1024 * 1024)))
# PDFs are read by pypdf in a child process (see read_pdf_bounded). A PDF may be
# as large as any download; one that is longer is refused, never cut. At most
# MAX_PDF_PAGES pages are read (the app's own limit), the child is killed after
# PDF_TIMEOUT seconds, and it may not grow beyond PDF_MEMORY_LIMIT bytes of
# address space (the VM has 1 GiB in all). One PDF is read at a time; a request
# waits up to PDF_BUSY_WAIT seconds for its turn. No OCR.
MAX_PDF_BYTES = MAX_DOWNLOAD_BYTES
MAX_PDF_PAGES = 40
PDF_TIMEOUT = float(os.environ.get("PDF_TIMEOUT", "15"))
PDF_MEMORY_LIMIT = 512 * 1024 * 1024
PDF_BUSY_WAIT = 20.0
# Per-stream limits for pypdf (its defaults allow 75 to 300 MB of work for one
# stream). A stream over them is not decoded and its page counts as failed.
PDF_STREAM_LIMIT = 20_000_000
# application/pdf is read as a PDF whatever the body (a body that is not one is
# "pdf_unreadable"). A response with a generic or missing type is read as a PDF
# only when its body starts with the PDF magic; no other type is sniffed.
PDF_TYPES = ("application/pdf", "application/x-pdf")
PDF_SNIFFED_TYPES = ("", "application/octet-stream", "binary/octet-stream")
PDF_MAGIC = b"%PDF-"
MAX_REDIRECTS = 5
MAX_REQUEST_BYTES = 64 * 1024
# Seconds a client may stay silent on an open connection (see Handler.timeout).
REQUEST_TIMEOUT = 30
# The whole request (line, headers and body) must arrive within this many
# seconds; the processing of an authenticated request is not limited by it.
READ_DEADLINE = 10.0
# Connections served at the same time. One more is closed without an answer, so
# idle or slow connections cannot use up the VM's process limit.
MAX_CONNECTIONS = 32
REDIRECT_STATUSES = (301, 302, 303, 307, 308)
ALLOWED_PORTS = {"http": 80, "https": 443}
# Request limits and defaults. The app sends its own current limits.
DEFAULT_MAX_BYTES = 2 * 1024 * 1024
DEFAULT_TIMEOUT = 15
MAX_TIMEOUT = 60
# The least time /v1/page leaves for turning a fetched page or PDF into text.
MIN_PROCESSING = 3.0
DEFAULT_MAX_CHARACTERS = 100_000
MAX_MESSAGE_CHARS = 200
# A success is refused when its final address, content type or content encoding
# is longer than this: the caller reserves a fixed room for them, and a site
# (or a redirect chain) must not be able to make them longer.
MAX_FINAL_URL_CHARS = 2048
MAX_CONTENT_TYPE_CHARS = 256
# The Accept value /v1/page sends (it is what the app's page reader sends).
PAGE_ACCEPT = "text/html,application/xhtml+xml,text/plain;q=0.9,application/pdf;q=0.8"
# /v1/http: the only origins it reaches, and which header goes where.
HTTP_ORIGINS = {
    "html.duckduckgo.com": "https://html.duckduckgo.com",
    "api.search.brave.com": "https://api.search.brave.com",
    "api.tavily.com": "https://api.tavily.com",
}
HTTP_METHODS = ("GET", "POST")
HTTP_HEADERS = {"accept": "Accept", "content-type": "Content-Type",
                "x-subscription-token": "X-Subscription-Token",
                "authorization": "Authorization"}
# A key header is sent to its own provider only.
HTTP_KEY_HEADERS = {"x-subscription-token": "api.search.brave.com",
                    "authorization": "api.tavily.com"}
MAX_HEADER_VALUE_CHARS = 2048
MAX_BODY_BYTES = 48 * 1024
LOOPBACK_HOSTS = {"127.0.0.1", "localhost", "[::1]", "::1"}
# Titles sit in the head; searching further only gives a hostile page more
# text to make the search slow.
TITLE_SEARCH_CHARS = 32 * 1024
SANDBOX_UID = 10001
# Names Python decodes as text. Anything else (idna, rot13, base64...) falls
# back to UTF-8 instead of failing or decoding into something odd.
TEXT_CHARSETS = {
    "ascii", "utf-8", "utf-16", "utf-16-le", "utf-16-be", "utf-32", "iso8859-1",
    "iso8859-2", "iso8859-3", "iso8859-4", "iso8859-5", "iso8859-6", "iso8859-7",
    "iso8859-8", "iso8859-9", "iso8859-10", "iso8859-13", "iso8859-14",
    "iso8859-15", "iso8859-16", "cp1250", "cp1251", "cp1252", "cp1253", "cp1254",
    "cp1255", "cp1256", "cp1257", "cp1258", "cp874", "koi8-r", "koi8-u",
    "mac-roman", "shift_jis", "cp932", "euc_jp", "iso2022_jp", "euc_kr", "cp949",
    "gb2312", "gbk", "gb18030", "big5", "big5hkscs",
}
TEXT_TYPES = ("text/html", "application/xhtml+xml", "text/plain")

# Addresses the sandbox must never connect to, even if public: the Mac's own
# interface addresses, passed in by the app as a comma-separated list. Set once
# by main() through load_blocked_addresses().
BLOCKED_ADDRESSES: frozenset = frozenset()
BLOCKED_ENV = "SANDBOX_BLOCKED_ADDRESSES"


class ToolError(Exception):
    """A request the tool refuses or cannot complete, reported to the caller."""

    def __init__(self, message: str, code: str, status: int = 422):
        super().__init__(message)
        self.message = message
        self.code = code
        self.status = status


# --- Address policy ---------------------------------------------------------

IPV6_UNICAST = ipaddress.ip_network("2000::/3")
# Inside 2000::/3 but carrying an IPv4 address a gateway could route to
# (6to4, Teredo), or reserved for documentation.
IPV6_REFUSED = [ipaddress.ip_network(n) for n in ("2002::/16", "2001::/32", "2001:db8::/32")]
# Refused by firewall-web.nft as well, though not every Python version calls it
# non-global: the 6to4 relay anycast range.
IPV4_REFUSED = [ipaddress.ip_network("192.88.99.0/24")]


def is_public_address(address: str) -> bool:
    try:
        ip = ipaddress.ip_address(address)
    except ValueError:
        return False
    if isinstance(ip, ipaddress.IPv6Address):
        if ip.ipv4_mapped is not None:
            ip = ip.ipv4_mapped
        # is_global alone accepts forms such as ::127.0.0.1 and the NAT64
        # prefix, which a gateway can turn into a private IPv4 address. Only
        # the global unicast range is allowed.
        elif ip not in IPV6_UNICAST or any(ip in net for net in IPV6_REFUSED):
            return False
    if ip in BLOCKED_ADDRESSES or (isinstance(ip, ipaddress.IPv4Address)
                                   and any(ip in net for net in IPV4_REFUSED)):
        return False
    return ip.is_global and not ip.is_multicast


def system_resolver(host: str, port: int) -> list[str]:
    try:
        infos = socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)
    except socket.gaierror as error:
        raise ToolError(f"could not resolve {host}: {error.strerror}", "dns_error", 502)
    except UnicodeError:
        # Empty or over-long labels make the idna codec fail inside getaddrinfo.
        raise ToolError("URL has an invalid host name", "invalid_url")
    return [info[4][0] for info in infos]


@dataclass(frozen=True)
class Target:
    scheme: str
    host: str
    port: int
    path: str
    address: str

    @property
    def url(self) -> str:
        default = ALLOWED_PORTS[self.scheme]
        netloc = self.host_header if self.port == default else f"{self.host_header}:{self.port}"
        return f"{self.scheme}://{netloc}{self.path}"

    @property
    def host_header(self) -> str:
        """The host as it is written in a URL or Host header; an IPv6 literal
        needs its brackets."""
        return f"[{self.host}]" if ":" in self.host else self.host


# Every ASCII character `quote` must leave alone, so that only non-ASCII
# characters change.
ASCII_URL_SAFE = "".join(chr(c) for c in range(0x21, 0x7f))

_NUMBER_LABEL = re.compile(r"[0-9]+|0x[0-9a-f]*")
_NUMBER_LIKE_LABEL = re.compile(r"[0-9]+|0x[0-9a-z]*")
_DOTTED_QUAD_PART = re.compile(r"0|[1-9][0-9]{0,2}")


def is_noncanonical_number(host: str) -> bool:
    """True for a host that C libraries read as an IPv4 address, or that is
    built like one, but is not a plain dotted quad.

    inet_aton and getaddrinfo accept 2130706433, 0x7f000001, 127.1 and
    0177.0.0.1, and they disagree: glibc reads a leading 0 as octal (127.0.0.1)
    where macOS reads it as decimal (177.0.0.1). The policy must not depend on
    that reading, so every such spelling is refused before any lookup. Only
    four decimal parts without leading zeros, each at most 255, are an
    address; a host whose last label is a number, or whose labels are all
    numbers, is never a name (no top-level domain is numeric). IPv6 literals
    and names with a letter in their last label are not affected.
    """
    if ":" in host:
        return False
    labels = host.split(".")
    if not (_NUMBER_LABEL.fullmatch(labels[-1])
            or all(_NUMBER_LIKE_LABEL.fullmatch(label) for label in labels)):
        return False
    if len(labels) == 4 and all(_DOTTED_QUAD_PART.fullmatch(label) for label in labels):
        try:
            ipaddress.IPv4Address(host)
        except ValueError:
            return True
        return False
    return True


def check_url(url: str, resolver: Callable[[str, int], list[str]]) -> Target:
    """Parses `url` and pins it to one public address, or refuses it."""
    if not isinstance(url, str) or len(url) > 4096:
        raise ToolError("url must be a string of at most 4096 characters", "invalid_url")
    try:
        parts = urllib.parse.urlsplit(url.strip())
    except ValueError:
        # Malformed addresses such as "http://[::1/" or a fullwidth "@".
        raise ToolError("URL is not a valid address", "invalid_url")
    scheme = parts.scheme.lower()
    if scheme not in ALLOWED_PORTS:
        raise ToolError("only http and https URLs can be fetched", "invalid_url")
    if parts.username or parts.password:
        raise ToolError("URLs with credentials are refused", "invalid_url")
    if (CONTROL_CHARACTERS.search(url) or has_format_character(url)
            or any(c.isspace() for c in url.strip())):
        raise ToolError("URL contains spaces or control characters", "invalid_url")
    host = (parts.hostname or "").rstrip(".").lower()
    if not host:
        raise ToolError("URL has no host", "invalid_url")
    if not host.isascii():
        # Internationalised names are sent as punycode, as browsers do.
        try:
            host = host.encode("idna").decode("ascii")
        except UnicodeError:
            raise ToolError("URL has an invalid internationalised host name", "invalid_url")
        # The ideographic, fullwidth and halfwidth full stops (U+3002, U+FF0E,
        # U+FF61) became ASCII dots in the step above, so the trailing one is
        # dropped here, before the number check and the lookup see the host.
        host = host.rstrip(".")
        if not host:
            raise ToolError("URL has no host", "invalid_url")
    try:
        port = parts.port or ALLOWED_PORTS[scheme]
    except ValueError:
        raise ToolError("URL has an invalid port", "invalid_url")
    if port != ALLOWED_PORTS[scheme]:
        raise ToolError(f"only port {ALLOWED_PORTS[scheme]} is allowed for {scheme}", "blocked_port")
    # Before any lookup: libc's reading of these spellings differs between
    # systems (see is_noncanonical_number), so none of them is trusted.
    if is_noncanonical_number(host):
        raise ToolError("URL has a number as host that is not a plain dotted IPv4 address",
                        "invalid_url")
    addresses = resolver(host, port)
    if not addresses:
        raise ToolError(f"could not resolve {host}", "dns_error", 502)
    # Every answer must be public: picking the one public answer out of a
    # mixed set would still let a rebinding name aim at the host next time.
    blocked = [a for a in addresses if not is_public_address(a)]
    if blocked:
        raise ToolError(f"{host} resolves to a non-public address", "blocked_address", 403)
    # http.client refuses non-ASCII request lines; send them as UTF-8 escapes,
    # as a browser does. Existing %XX escapes and ASCII are left as they are.
    path = urllib.parse.quote(parts.path or "/", safe=ASCII_URL_SAFE)
    if parts.query:
        path += "?" + urllib.parse.quote(parts.query, safe=ASCII_URL_SAFE)
    # IPv4 first: the VM may have no IPv6 route, and every answer is public.
    preferred = sorted(addresses, key=lambda a: ":" in a)[0]
    return Target(scheme, host, port, path, preferred)


# --- Transport --------------------------------------------------------------

@dataclass
class Response:
    status: int
    headers: dict[str, str]
    body: bytes
    truncated: bool = False


# Called with (method, target, headers, body) and the keyword arguments
# `deadline` and `max_bytes`.
Transport = Callable[..., Response]


class _SendOnlyGivenHeaders:
    """http.client adds `Accept-Encoding: identity` to a request that has no
    Accept-Encoding header. The transport sends the headers it is given and
    nothing else."""

    # The event pinned_transport's deadline timer sets; connect() looks at it.
    expired: threading.Event | None = None

    def putrequest(self, method, url, skip_host=False, skip_accept_encoding=False):
        return super().putrequest(method, url, skip_host, True)


class _PinnedHTTPConnection(_SendOnlyGivenHeaders, http.client.HTTPConnection):
    def __init__(self, target: Target, timeout: float):
        super().__init__(target.host, target.port, timeout=timeout)
        self._address = target.address

    def connect(self):
        self.sock = socket.create_connection((self._address, self.port), self.timeout)


class _PinnedHTTPSConnection(_SendOnlyGivenHeaders, http.client.HTTPSConnection):
    def __init__(self, target: Target, timeout: float):
        super().__init__(target.host, target.port, timeout=timeout,
                         context=ssl.create_default_context())
        self._address = target.address

    def connect(self):
        sock = socket.create_connection((self._address, self.port), self.timeout)
        # The socket is stored before the handshake, so the deadline timer can
        # shut it down while the handshake waits (wrap_socket would detach
        # `sock` and keep it out of the timer's reach until it returned).
        secure = self._context.wrap_socket(sock, server_hostname=self.host,
                                           do_handshake_on_connect=False)
        self.sock = secure
        if self.expired is not None and self.expired.is_set():
            raise OSError("the deadline passed before the handshake")
        try:
            secure.do_handshake()
        except AttributeError:
            # The timer's shutdown() ran between the check and the handshake.
            raise OSError("the deadline passed during the handshake") from None


def declared_length(headers: dict[str, str]) -> int | None:
    """The Content-Length a response announces, if it is a plain number."""
    value = headers.get("content-length", "").strip()
    return int(value) if value.isascii() and value.isdigit() else None


def pinned_transport(method: str, target: Target, headers: dict[str, str],
                     body: bytes | None, deadline: float | None = None,
                     max_bytes: int | None = None) -> Response:
    """`deadline` bounds the whole request in seconds. `max_bytes` caps the
    body that is read (at most MAX_DOWNLOAD_BYTES by default); a longer body,
    or a Content-Length above it, comes back as `truncated` without the rest
    being read."""
    cls = _PinnedHTTPSConnection if target.scheme == "https" else _PinnedHTTPConnection
    limit = FETCH_DEADLINE if deadline is None else deadline
    cap = MAX_DOWNLOAD_BYTES if max_bytes is None else min(max_bytes, MAX_DOWNLOAD_BYTES)
    # A wait longer than the deadline is cut by the timer; one second more
    # lets the timer win the race and report a timeout, not a failed connect.
    connection = cls(target, min(FETCH_TIMEOUT, limit + 1))
    expired = threading.Event()
    connection.expired = expired

    def expire():
        # Shutting the socket down wakes a read blocked in another thread.
        expired.set()
        if connection.sock is not None:
            try:
                connection.sock.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass

    timer = threading.Timer(limit, expire)
    timer.daemon = True
    timer.start()
    too_slow = ToolError(f"{target.host} took longer than {limit:g} s", "fetch_timeout", 504)
    try:
        # Connect first, so that a connection or handshake that ends after the
        # timer fired is noticed. The timer can only shut down a socket that
        # exists; once it does, it wakes any later wait.
        connection.connect()
        if expired.is_set():
            raise too_slow
        connection.request(method, target.path, body=body, headers=headers)
        response = connection.getresponse()
        received = {k.lower(): v for k, v in response.getheaders()}
        if sum(1 for k, _ in response.getheaders() if k.lower() == "location") > 1:
            raise ToolError(f"{target.host} sent ambiguous redirect addresses", "fetch_failed", 502)
        announced = declared_length(received)
        if max_bytes is not None:
            if announced is not None and announced > cap:
                return Response(response.status, received, b"", truncated=True)
            if response.status in REDIRECT_STATUSES and received.get("location", "").strip():
                # The caller follows the redirect; its body is not wanted.
                return Response(response.status, received, b"")
        chunks, size, ended = [], 0, False
        while size <= cap and not expired.is_set():
            chunk = response.read1(min(65536, cap + 1 - size))
            if not chunk:
                ended = True
                break
            chunks.append(chunk)
            size += len(chunk)
        if expired.is_set():
            raise too_slow
        if max_bytes is not None and ended and response.length:
            # The connection closed before the announced length arrived. A
            # short body must not look like the whole page.
            raise ToolError(f"{target.host} closed the connection before the whole body "
                            f"arrived ({size} of {announced} bytes)", "fetch_failed", 502)
        data = b"".join(chunks)
        return Response(
            status=response.status,
            headers=received,
            body=data[:cap],
            truncated=len(data) > cap)
    except (OSError, http.client.HTTPException, ValueError) as error:
        if expired.is_set():
            raise too_slow
        raise ToolError(f"request to {target.host} failed: {error}", "fetch_failed", 502)
    finally:
        timer.cancel()
        connection.close()


def decode_body(response: Response, limit: int | None = None) -> bytes:
    """The body, unpacked when the site sent it gzip or deflate anyway (the
    sandbox never asks for compression). A body that unpacks to more than
    `limit` bytes is refused (`oversized`), never cut."""
    limit = MAX_DOWNLOAD_BYTES if limit is None else limit
    encoding = response.headers.get("content-encoding", "identity").lower()
    if encoding in ("", "identity"):
        return response.body
    if encoding not in ("gzip", "deflate"):
        raise ToolError(f"unsupported content encoding {encoding}", "unsupported_content")
    wbits = 16 + zlib.MAX_WBITS if encoding == "gzip" else zlib.MAX_WBITS
    try:
        inflater = zlib.decompressobj(wbits)
        # Bounded so a small compressed bomb cannot fill the VM's memory: one
        # byte past the limit is enough to know the body is too long.
        data = inflater.decompress(response.body, limit + 1)
    except (zlib.error, gzip.BadGzipFile):
        raise ToolError("response body could not be decompressed", "unsupported_content", 502)
    if len(data) > limit:
        raise ToolError(f"the body is larger than {limit} bytes", "oversized", 502)
    return data


def charset_of(content_type: str) -> str:
    match = re.search(r"charset=\"?([\w.:-]+)", content_type, re.I)
    if match:
        try:
            name = codecs.lookup(match.group(1)).name
        except LookupError:
            return "utf-8"
        if name in TEXT_CHARSETS:
            return name
    return "utf-8"


META_CHARSET = re.compile(rb"""<meta[^>]{0,200}?charset\s*=\s*["']?([\w.:-]+)""", re.I)


def page_charset(content_type: str, body: bytes) -> str:
    """The HTTP header's charset, else the page's own <meta charset>, else UTF-8."""
    if "charset=" in content_type.lower():
        return charset_of(content_type)
    match = META_CHARSET.search(body[:4096])
    if match:
        return charset_of("charset=" + match.group(1).decode("ascii", "replace"))
    return "utf-8"


# C0 and C1 controls other than tab and newline, and the Unicode bidi
# overrides: a page title holding ESC sequences could otherwise rewrite the
# terminal the report is printed in. Also invisible characters: the soft
# hyphen, zero-width characters, fillers, the byte order mark, the Unicode
# tag block and the variation selectors other than U+FE0F, which can spell out
# instructions the model reads but a person reviewing the text cannot see.
# The emoji variation selector U+FE0F is ordinary text and stays.
CONTROL_CHARACTERS = re.compile(
    "[\x00-\x08\x0b-\x1f\x7f-\x9f\xad\u034f\u061c\u115f\u1160\u17b4\u17b5"
    "\u180b-\u180f\u200b-\u200f\u202a-\u202e\u2060-\u2069\u2800\u3164\ufe00-\ufe0e\ufeff"
    "\uffa0\U000e0000-\U000e007f\U000e0100-\U000e01ef]")


# Every Unicode format character (category Cf) is removed as well: the list
# above misses about fifty of them (U+206A-206F, U+FFF9-FFFB, U+070F, the
# Egyptian hieroglyph and shorthand format controls, ...). Built once at
# import (about 0.2 s); applying it to a 240 000 character page takes under
# a millisecond. Arabic number signs (U+0600-0605, U+06DD, U+08E2) are
# visible but rare and are dropped with the rest.
FORMAT_CHARACTERS = {
    cp: None for cp in range(sys.maxunicode + 1) if unicodedata.category(chr(cp)) == "Cf"}


def has_format_character(text: str) -> bool:
    return any(unicodedata.category(c) == "Cf" for c in text)


def clean_text(text: str) -> str:
    for separator in ("\r\n", "\r", "\u2028", "\u2029"):
        text = text.replace(separator, "\n")
    return CONTROL_CHARACTERS.sub("", text.translate(FORMAT_CHARACTERS))


def collapse_whitespace(text: str) -> str:
    """Runs of blanks inside a line become one space, lines are stripped and
    three or more line breaks become two. Every pattern here is a plain run
    (linear), so a line of millions of blanks costs one pass."""
    lines = (re.sub(r"[ \t\r\f\v]+", " ", line).strip() for line in text.split("\n"))
    return re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip()


# --- Extraction -------------------------------------------------------------

class _FallbackText(HTMLParser):
    """Plain-text fallback when trafilatura is missing or finds no article."""

    SKIP = {"script", "style", "noscript", "template", "svg", "head"}
    BLOCK = {"p", "div", "li", "br", "h1", "h2", "h3", "h4", "h5", "h6", "tr",
             "section", "article", "blockquote", "pre"}

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self.skipping = 0

    def handle_starttag(self, tag, attrs):
        if tag in self.SKIP:
            self.skipping += 1
        elif tag in self.BLOCK:
            self.parts.append("\n")

    def handle_endtag(self, tag):
        if tag in self.SKIP and self.skipping:
            self.skipping -= 1
        elif tag in self.BLOCK:
            self.parts.append("\n")

    def handle_data(self, data):
        if not self.skipping:
            self.parts.append(data)

    def text(self) -> str:
        return collapse_whitespace("".join(self.parts))


def clean_title(title: str) -> str:
    return clean_text(re.sub(r"\s+", " ", title)).strip()[:300]


TITLE_OPEN = re.compile(r"<title(?:\s[^<>]{0,500})?>", re.I)


def extract_title(markup: str) -> str:
    # A bounded, linear search: a lazy `<title>(.*?)</title>` over the whole
    # page takes hours on a page of unclosed <title> tags, and holds the
    # interpreter lock while it runs.
    head = markup[:TITLE_SEARCH_CHARS]
    match = TITLE_OPEN.search(head)
    if not match:
        return ""
    end = head.lower().find("</title>", match.end())
    if end < 0:
        return ""
    return clean_title(html.unescape(head[match.end():end]))


def extract_text(markup: str, url: str) -> str:
    if trafilatura is not None:
        text = trafilatura.extract(
            markup, url=url, include_comments=False, include_tables=True,
            favor_recall=True, deduplicate=True)
        if text and text.strip():
            return clean_text(text).strip()
    return fallback_text(markup)


def fallback_text(markup: str) -> str:
    parser = _FallbackText()
    parser.feed(markup)
    parser.close()
    return clean_text(parser.text())


def _extract_in_child(sender, extractor, markup: str, url: str) -> None:
    # Raw UTF-8 bytes, never a pickle: the child parses hostile pages, and
    # unpickling what it sends would let an exploited parser run code in the
    # server. An empty message means "failed" and the parent falls back.
    try:
        text = extractor(markup, url)
        sender.send_bytes(b"\x01" + text.encode("utf-8", "replace") if isinstance(text, str) else b"")
    except Exception:
        sender.send_bytes(b"")
    finally:
        sender.close()


def _fallback_in_child(markup: str, url: str) -> str:
    return fallback_text(markup)


def _child_message(target, args: tuple, timeout: float, limit: int) -> bytes | None:
    """The one message `target(sender, *args)` sends from a child process, or
    None when the child sends nothing, dies or is still busy after `timeout`
    seconds (it is killed then). Raw bytes only, never a pickle: the child
    parses hostile files, and unpickling what it sends would let an exploited
    parser run code in the server. recv_bytes never unpickles; `limit` bounds
    what is read."""
    context = multiprocessing.get_context("forkserver")
    receiver, sender = context.Pipe(duplex=False)
    child = context.Process(target=target, args=(sender, *args), daemon=True)
    child.start()
    sender.close()
    message = None
    # poll() returns once data starts to arrive, and recv_bytes() then waits
    # for the rest of the message; a child that stalls half way is killed here
    # (the read then ends with EOFError) instead of holding this thread.
    watchdog = threading.Timer(timeout + 1, child.kill)
    watchdog.daemon = True
    watchdog.start()
    try:
        if receiver.poll(timeout):
            message = receiver.recv_bytes(limit)
    except (EOFError, OSError):
        pass
    finally:
        watchdog.cancel()
        receiver.close()
        if child.is_alive():
            child.kill()
        child.join(5)
    return message


def _run_in_child(extractor, markup: str, url: str, timeout: float) -> str | None:
    """`extractor(markup, url)` run in a child process that is killed after
    `timeout` seconds. None when it fails, returns no text or runs out of time."""
    message = _child_message(_extract_in_child, (extractor, markup, url), timeout,
                             4 * MAX_EXTRACT_CHARS + 1)
    if message is not None and message.startswith(b"\x01"):
        return clean_text(message[1:].decode("utf-8", "replace"))
    return None


def extract_text_bounded(markup: str, url: str, timeout: float | None = None,
                         extractor: Callable[[str, str], str] = extract_text,
                         total: float | None = None) -> str:
    """Runs `extractor` in a child process and stops it after `timeout`
    seconds. A page the extractor cannot finish in time, or fails on, is read
    with the simple fallback parser instead. That is html.parser, which has had
    quadratic cases (CVE-2025-6069), so it runs the same way: in a child with
    its own limit and over fewer characters. A page it cannot finish either
    gives no text, never an unbounded wait in this process. With `total`, the
    fallback gets no more than what is left of that many seconds (at least one)."""
    started = time.monotonic()
    markup = markup[:MAX_EXTRACT_CHARS]
    text = _run_in_child(extractor, markup, url, EXTRACT_TIMEOUT if timeout is None else timeout)
    if text is not None:
        return text
    sys.stderr.write(f"extraction of {url} timed out or failed; used the fallback parser\n")
    fallback_limit = FALLBACK_TIMEOUT
    if total is not None:
        fallback_limit = min(FALLBACK_TIMEOUT, max(1.0, total - (time.monotonic() - started)))
    text = _run_in_child(_fallback_in_child, markup[:MAX_FALLBACK_CHARS], url, fallback_limit)
    if text is None:
        sys.stderr.write(f"the fallback parser could not read {url} either; no text\n")
        return ""
    return text


# --- PDF --------------------------------------------------------------------

class PdfRefused(Exception):
    """A PDF that is read no further, with the tool error code to report."""

    def __init__(self, code: str):
        super().__init__(code)
        self.code = code


def read_pdf(data: bytes) -> dict:
    """Runs in the child. The text of at most MAX_PDF_PAGES pages, each cleaned
    like HTML text, plus the document title and the page counts. Never
    decrypts: an encrypted PDF is refused, whatever its password."""
    if PdfReader is None:
        raise PdfRefused("pdf_unreadable")
    logging.getLogger("pypdf").setLevel(logging.CRITICAL)
    pypdf.overwrite_configuration(
        maximum_declared_stream_length=PDF_STREAM_LIMIT,
        array_based_stream_maximum_output_length=PDF_STREAM_LIMIT,
        jbig2_maximum_output_length=PDF_STREAM_LIMIT,
        lzw_maximum_output_length=PDF_STREAM_LIMIT,
        run_length_maximum_output_length=PDF_STREAM_LIMIT,
        zlib_maximum_output_length=PDF_STREAM_LIMIT,
        brotli_maximum_output_length=PDF_STREAM_LIMIT,
        image_maximum_buffer_size=PDF_STREAM_LIMIT,
        stream_decoding_work_maximum_length=4 * PDF_STREAM_LIMIT,
        jbig2dec_binary=None,  # never start an external program
    )
    reader = PdfReader(io.BytesIO(data), strict=False)
    if reader.is_encrypted:
        raise PdfRefused("pdf_encrypted")
    total = len(reader.pages)
    title = ""
    try:
        title = str(reader.metadata.title or "")[:1000] if reader.metadata else ""
    except Exception:
        pass
    parts: list[str] = []
    chars = pages_read = pages_failed = 0
    cut = False
    for number in range(min(total, MAX_PDF_PAGES)):
        pages_read += 1
        try:
            text = collapse_whitespace(clean_text(reader.pages[number].extract_text() or ""))
        except Exception:
            pages_failed += 1  # one odd page does not cost the rest
            continue
        if len(text) > MAX_EXTRACT_CHARS - chars:
            text, cut = text[:MAX_EXTRACT_CHARS - chars], True
        if text:
            parts.append(text)
            chars += len(text)
        if cut:
            break
    return {"title": title, "text": "\n\n".join(parts), "pages": total,
            "pages_read": pages_read, "pages_failed": pages_failed, "cut": cut}


def _pdf_in_child(sender, reader, data: bytes) -> None:
    # \x01 + JSON on success, \x02 + an error code on a refusal; anything else
    # (a crash, a kill, no message) is "unreadable". Parsed with json, never pickle.
    try:
        try:
            resource.setrlimit(resource.RLIMIT_AS, (PDF_MEMORY_LIMIT, PDF_MEMORY_LIMIT))
        except (ValueError, OSError):
            pass  # not every system allows it; the VM's own limits still apply
        try:
            result = reader(data)
            sender.send_bytes(b"\x01" + json.dumps(result, ensure_ascii=False).encode("utf-8", "replace"))
        except PdfRefused as refusal:
            sender.send_bytes(b"\x02" + refusal.code.encode("ascii"))
        except Exception:
            sender.send_bytes(b"\x02pdf_unreadable")
    finally:
        sender.close()


_PDF_SLOT = threading.BoundedSemaphore(1)


def read_pdf_bounded(data: bytes, timeout: float | None = None,
                     reader: Callable[[bytes], dict] = read_pdf,
                     busy_wait: float | None = None, budget: float | None = None) -> dict:
    """`reader(data)` in a child process that is killed after `timeout`
    seconds. Returns {"title", "text", "pages", "pages_read", "pages_failed",
    "cut"} with text and title cleaned again here (the child is not trusted),
    or raises ToolError: pdf_encrypted, or pdf_unreadable for a PDF that is
    malformed or could not be read in time, or pdf_busy (503) when another
    PDF has been read for longer than `busy_wait` (PDF_BUSY_WAIT) seconds.
    With `budget` (seconds), the wait for the slot and the reading together stay
    within it (the reading gets at least 1 s)."""
    limit = PDF_TIMEOUT if timeout is None else timeout
    started = time.monotonic()
    unreadable = ToolError("the PDF could not be read: it is damaged, or reading it took longer "
                           f"than {limit:g} s", "pdf_unreadable")
    # One PDF at a time: each child may use PDF_MEMORY_LIMIT, and the VM is small.
    if not _PDF_SLOT.acquire(timeout=PDF_BUSY_WAIT if busy_wait is None else busy_wait):
        raise ToolError("another PDF is being read; try again in a moment", "pdf_busy", 503)
    if budget is not None:
        limit = min(limit, max(1.0, budget - (time.monotonic() - started)))
    try:
        message = _child_message(_pdf_in_child, (reader, data), limit, 4 * MAX_EXTRACT_CHARS + 65536)
    finally:
        _PDF_SLOT.release()
    if message is None:
        raise unreadable
    if message.startswith(b"\x02"):
        code = message[1:].decode("ascii", "replace")
        if code == "pdf_encrypted":
            raise ToolError("the PDF is encrypted; encrypted PDFs are not read", code)
        raise unreadable
    try:
        # Any exception: a deeply nested answer raises RecursionError, not ValueError.
        result = json.loads(message[1:]) if message.startswith(b"\x01") else None
        title, text = result["title"], result["text"]
        counts = [result[key] for key in ("pages", "pages_read", "pages_failed")]
        if (not isinstance(title, str) or not isinstance(text, str) or type(result["cut"]) is not bool
                or not all(type(n) is int and n >= 0 for n in counts)):
            raise ValueError("unexpected types")
    except Exception:
        raise unreadable from None
    result["title"] = clean_title(title)
    result["text"] = collapse_whitespace(clean_text(text[:MAX_EXTRACT_CHARS]))
    return result


def last_path_segment(url: str) -> str:
    segment = urllib.parse.unquote(urllib.parse.urlsplit(url).path.rstrip("/").rsplit("/", 1)[-1])
    return clean_title(segment)


def pdf_text(url: str, body: bytes, max_characters: int,
             budget: float | None = None) -> tuple[str, str]:
    """(title, text) of a PDF's bytes. The text is cleaned like HTML text (see
    read_pdf_bounded) and carries a closing note when pages were left out. The
    title is the document's own, else the last part of the address. The result
    holds at most `max_characters` characters, the note included when it fits.
    With `budget` (seconds), the wait for the PDF slot and the reading together
    take no longer than that (the reading gets at least 1 s)."""
    if budget is None:
        result = read_pdf_bounded(body)
    else:
        result = read_pdf_bounded(body, timeout=min(PDF_TIMEOUT, budget),
                                  busy_wait=min(PDF_BUSY_WAIT, budget), budget=budget)
    notes = []
    if result["pages_read"] < result["pages"]:
        notes.append(f"only the first {result['pages_read']} of {result['pages']} pages were read")
    if result["pages_failed"]:
        notes.append(f"{result['pages_failed']} pages could not be read")
    if result["cut"]:
        notes.append(f"the text was cut after {MAX_EXTRACT_CHARS} characters")
    text = result["text"]
    if not text:
        if result["pages_failed"] and result["pages_failed"] >= result["pages_read"]:
            raise ToolError("the PDF's pages could not be read", "pdf_unreadable")
        raise ToolError("the PDF has no text that can be read (scanned pages are not read; "
                        "there is no OCR)", "pdf_unreadable")
    note = f"\n\n[PDF note: {'; '.join(notes)}.]" if notes else ""
    if len(text) + len(note) > max_characters:
        room = max_characters - len(note)
        text = text[:room] + note if room > 0 else text[:max_characters]
    else:
        text += note
    return result["title"] or last_path_segment(url), text




# --- Blocked addresses ------------------------------------------------------

def load_blocked_addresses(value: str) -> frozenset:
    """The addresses in a comma-separated list, as address objects. An
    IPv4-mapped IPv6 address counts as the IPv4 address it carries. Raises
    ValueError for anything that is not a plain IP address, so a mistake in
    the launcher stops the sandbox instead of leaving a hole in its blocklist."""
    found = set()
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        if "%" in part:
            raise ValueError(f"an address with a zone id is not accepted: {part[:64]!r}")
        try:
            ip = ipaddress.ip_address(part)
        except ValueError:
            raise ValueError(f"not an IP address: {part[:64]!r}") from None
        if isinstance(ip, ipaddress.IPv6Address) and ip.ipv4_mapped is not None:
            ip = ip.ipv4_mapped
        found.add(ip)
    if len(found) > 256:
        raise ValueError("more than 256 blocked addresses")
    return frozenset(found)


# --- Requests the tools make ------------------------------------------------

def url_scheme(url: str) -> str:
    try:
        return urllib.parse.urlsplit(url.strip()).scheme.lower()
    except ValueError:
        return ""


def origin_of(url: str) -> tuple[str, str, int] | None:
    """(scheme, host, port) of an address, or None when it cannot be read."""
    try:
        parts = urllib.parse.urlsplit(url.strip())
        scheme = parts.scheme.lower()
        host = (parts.hostname or "").rstrip(".").lower()
        port = parts.port or ALLOWED_PORTS.get(scheme, 0)
    except ValueError:
        return None
    return (scheme, host, port) if host else None


def allowed_http_origin(url: str) -> bool:
    """True for https on port 443 of one of the three search providers, with no
    credentials in the address."""
    origin = origin_of(url)
    if origin is None:
        return False
    try:
        parts = urllib.parse.urlsplit(url.strip())
    except ValueError:
        return False
    return (origin[0] == "https" and origin[2] == 443 and origin[1] in HTTP_ORIGINS
            and not parts.username and not parts.password)


def check_scheme(url: str) -> None:
    """A scheme other than http and https is `unsupported_scheme`; check_url
    reports the rest."""
    if isinstance(url, str) and url_scheme(url) not in ("", *ALLOWED_PORTS):
        raise ToolError("only http and https URLs can be fetched", "unsupported_scheme")


# check_url and the transport name two errors differently from what this API
# promises its caller.
API_CODES = {"blocked_address": "private_address", "fetch_timeout": "timed_out"}


def api_error(error: ToolError) -> ToolError:
    """An error as the API reports it: its own code names, and a message
    without control characters or line breaks, of limited length."""
    text = clean_text(error.message)
    text = "".join(" " if c.isspace() else c for c in text)
    text = "".join(c for c in text if unicodedata.category(c)[0] != "C").strip()
    return ToolError(text[:MAX_MESSAGE_CHARS], API_CODES.get(error.code, error.code), error.status)


def one_line(value: str, limit: int) -> str:
    """A header value of a site, cleaned. Longer than `limit` is refused: the
    caller reserves a fixed room for it."""
    value = clean_text(value)
    if len(value) > limit:
        raise ToolError(f"a header of the answer is longer than {limit} characters",
                        "fetch_failed", 502)
    return value


@dataclass
class WebTools:
    resolver: Callable[[str, int], list[str]] = system_resolver
    transport: Transport = pinned_transport
    clock: Callable[[], float] = time.monotonic

    # -- shared -------------------------------------------------------------

    def _checked_target(self, url, remaining: float, too_slow: ToolError) -> Target:
        """check_url, but given up on when the name lookup outlasts the
        deadline. The lookup cannot be stopped; its thread ends when the
        resolver gives up (some 10 to 30 s), and counts against the VM's
        process limit until then."""
        if remaining <= 0:
            raise too_slow
        outcome: dict = {}

        def run():
            try:
                check_scheme(url)
                outcome["target"] = check_url(url, self.resolver)
            except BaseException as error:
                outcome["error"] = error

        worker = threading.Thread(target=run, daemon=True)
        worker.start()
        worker.join(remaining)
        if worker.is_alive():
            raise too_slow
        if "error" in outcome:
            raise outcome["error"]
        return outcome["target"]

    def _exchange(self, method: str, url: str, headers: dict[str, str], body: bytes | None,
                  max_bytes: int, timeout_seconds: float,
                  search_origins_only: bool) -> tuple[str, Response, float]:
        """One request with its redirects. Returns the final address, the
        final response and the time left. One deadline covers the name lookups,
        connections and reads of every hop.

        Redirects follow PinnedHTTPTransport: never from https to http, and a
        request with a body or a header other than Accept is never sent on to
        another origin. With `search_origins_only` a redirect may not leave the
        three search sites either. A refused redirect is `insecure_redirect`."""
        expires = self.clock() + timeout_seconds
        too_slow = ToolError(f"the request took longer than {timeout_seconds:g} s", "timed_out", 504)
        current = url
        for _ in range(MAX_REDIRECTS + 1):
            if search_origins_only and not allowed_http_origin(current):
                raise ToolError("this address is not one of the search sites the sandbox may reach",
                                "forbidden_origin", 403)
            target = self._checked_target(current, expires - self.clock(), too_slow)
            remaining = expires - self.clock()
            if remaining <= 0:
                raise too_slow
            # No Accept-Encoding, no cookies, no language: only what the caller
            # asked for and the sandbox's own User-Agent.
            sent = {"Host": target.host_header, "User-Agent": USER_AGENT, **headers}
            try:
                response = self.transport(method, target, sent, body,
                                          deadline=remaining, max_bytes=max_bytes)
            except ToolError as error:
                raise too_slow if error.code == "fetch_timeout" else error
            location = response.headers.get("location", "").strip()
            if response.status in REDIRECT_STATUSES and location:
                destination = urllib.parse.urljoin(target.url, location)
                if target.scheme == "https" and url_scheme(destination) == "http":
                    raise ToolError("a redirect from https to http was refused",
                                    "insecure_redirect", 502)
                if search_origins_only and not allowed_http_origin(destination):
                    raise ToolError("a redirect away from the search sites was refused",
                                    "insecure_redirect", 502)
                if origin_of(target.url) != origin_of(destination) and (
                        body is not None
                        or any(name.lower() != "accept" for name in headers)):
                    raise ToolError("a request with a body or a key was sent on to another "
                                    "site; refused", "insecure_redirect", 502)
                if response.status == 303 or (response.status in (301, 302) and method == "POST"):
                    method, body = "GET", None
                    headers = {k: v for k, v in headers.items() if k.lower() != "content-type"}
                current = destination
                continue
            announced = declared_length(response.headers)
            if (response.truncated or len(response.body) > max_bytes
                    or (announced is not None and announced > max_bytes)):
                raise ToolError(f"the body is larger than {max_bytes} bytes", "oversized", 502)
            if len(target.url) > MAX_FINAL_URL_CHARS:
                raise ToolError(f"the final address is longer than {MAX_FINAL_URL_CHARS} "
                                "characters", "fetch_failed", 502)
            return target.url, response, expires - self.clock()
        raise ToolError("too many redirects", "too_many_redirects", 502)

    @staticmethod
    def _limits(max_bytes, timeout_seconds) -> None:
        if (not isinstance(max_bytes, int) or isinstance(max_bytes, bool)
                or not 1 <= max_bytes <= MAX_DOWNLOAD_BYTES):
            raise ToolError(f"max_bytes must be between 1 and {MAX_DOWNLOAD_BYTES}",
                            "invalid_argument")
        if (not isinstance(timeout_seconds, (int, float)) or isinstance(timeout_seconds, bool)
                or not 0 < timeout_seconds <= MAX_TIMEOUT):
            raise ToolError(f"timeout_seconds must be more than 0 and at most {MAX_TIMEOUT}",
                            "invalid_argument")

    # -- /v1/page -----------------------------------------------------------

    def page(self, url, max_bytes=DEFAULT_MAX_BYTES, timeout_seconds=DEFAULT_TIMEOUT,
             max_characters=DEFAULT_MAX_CHARACTERS) -> dict:
        try:
            return self._page(url, max_bytes, timeout_seconds, max_characters)
        except ToolError as error:
            raise api_error(error) from None

    def _page(self, url, max_bytes, timeout_seconds, max_characters) -> dict:
        self._limits(max_bytes, timeout_seconds)
        if (not isinstance(max_characters, int) or isinstance(max_characters, bool)
                or not 1 <= max_characters <= MAX_EXTRACT_CHARS):
            raise ToolError(f"max_characters must be between 1 and {MAX_EXTRACT_CHARS}",
                            "invalid_argument")
        final_url, response, left = self._exchange(
            "GET", url, {"Accept": PAGE_ACCEPT}, None, max_bytes, timeout_seconds, False)
        content_type = one_line(response.headers.get("content-type", ""), MAX_CONTENT_TYPE_CHARS)
        answer = {"final_url": final_url, "status": response.status,
                  "content_type": content_type, "title": "", "text": ""}
        if not 200 <= response.status < 300:
            return answer  # the app reports the status; an error page is not read
        kind = content_type.lower().split(";")[0].strip()
        body = decode_body(response, max_bytes)
        is_pdf = False
        if kind in PDF_TYPES:
            is_pdf = True
        elif kind in PDF_SNIFFED_TYPES:
            is_pdf = body.startswith(PDF_MAGIC)
            if not is_pdf and kind:
                raise ToolError(f"{final_url} is {kind}, not a web page", "unsupported_content")
        elif kind not in (*TEXT_TYPES, "text/markdown"):
            raise ToolError(f"{final_url} is {kind}, not a web page", "unsupported_content")
        if is_pdf:
            if PdfReader is None:
                raise ToolError("this sandbox cannot read PDFs", "unsupported_content")
            title, text = pdf_text(final_url, body, max_characters, max(MIN_PROCESSING, left))
        else:
            raw = body.decode(page_charset(content_type, body), errors="replace")
            # What is left of the deadline bounds the processing too: the
            # extractor, the fallback parser and PDF reading each get no more
            # than that, and at least MIN_PROCESSING seconds, so a slow fetch
            # does not leave a page with no time to be read. An answer can
            # therefore take a few seconds more than timeout_seconds.
            budget = max(MIN_PROCESSING, left)
            if kind in ("text/plain", "text/markdown"):
                # Runs of blanks are collapsed as in the fallback parser, so a
                # page of millions of spaces is not kept as such.
                title, text = "", collapse_whitespace(clean_text(raw))
            else:
                title = extract_title(raw)
                text = extract_text_bounded(raw, final_url, timeout=min(EXTRACT_TIMEOUT, budget),
                                           total=budget)
            text = text[:max_characters]
        answer["title"] = title
        answer["text"] = text
        return answer

    # -- /v1/http -----------------------------------------------------------

    def http(self, url, method, headers=None, body_base64=None, max_bytes=DEFAULT_MAX_BYTES,
             timeout_seconds=DEFAULT_TIMEOUT) -> dict:
        try:
            return self._http(url, method, headers, body_base64, max_bytes, timeout_seconds)
        except ToolError as error:
            raise api_error(error) from None

    def _http(self, url, method, headers, body_base64, max_bytes, timeout_seconds) -> dict:
        self._limits(max_bytes, timeout_seconds)
        if method not in HTTP_METHODS:
            raise ToolError("method must be GET or POST", "invalid_argument")
        if not isinstance(url, str) or not allowed_http_origin(url):
            raise ToolError("this address is not one of the search sites the sandbox may reach",
                            "forbidden_origin", 403)
        sent = self._allowed_headers(headers, url)
        body = None
        if body_base64 is not None:
            if not isinstance(body_base64, str) or len(body_base64) > 4 * MAX_BODY_BYTES // 3 + 4:
                raise ToolError(f"the body must be at most {MAX_BODY_BYTES} bytes", "invalid_argument")
            try:
                body = base64.b64decode(body_base64, validate=True)
            except ValueError:
                raise ToolError("body_base64 is not valid base64", "invalid_argument") from None
            if len(body) > MAX_BODY_BYTES:
                raise ToolError(f"the body must be at most {MAX_BODY_BYTES} bytes", "invalid_argument")
            if method == "GET" and body:
                raise ToolError("a GET request has no body", "invalid_argument")
            if method == "GET":
                body = None
        final_url, response, _ = self._exchange(
            method, url, sent, body, max_bytes, timeout_seconds, True)
        data = decode_body(response, max_bytes)
        return {"final_url": final_url, "status": response.status,
                "content_type": one_line(response.headers.get("content-type", ""),
                                         MAX_CONTENT_TYPE_CHARS),
                "body_base64": base64.b64encode(data).decode("ascii")}

    @staticmethod
    def _allowed_headers(headers, url: str) -> dict[str, str]:
        """The caller's headers, if every one is on the allow-list. Names are
        written in their canonical form. A key header goes to its provider
        only."""
        if headers is None:
            return {}
        if not isinstance(headers, dict):
            raise ToolError("headers must be an object", "invalid_argument")
        origin = origin_of(url)
        host = origin[1] if origin else ""
        sent: dict[str, str] = {}
        for name, value in headers.items():
            if not isinstance(name, str) or not isinstance(value, str):
                raise ToolError("headers must be text", "invalid_argument")
            lowered = name.lower()
            if lowered not in HTTP_HEADERS:
                raise ToolError(f"the header {name[:40]!r} is not allowed", "invalid_header")
            canonical = HTTP_HEADERS[lowered]
            if canonical in sent:
                raise ToolError(f"the header {canonical} is given twice", "invalid_header")
            if lowered in HTTP_KEY_HEADERS and HTTP_KEY_HEADERS[lowered] != host:
                raise ToolError(f"the header {canonical} is not sent to this site", "invalid_header")
            if (len(value) > MAX_HEADER_VALUE_CHARS or not value.isascii()
                    or not value.isprintable()):
                raise ToolError(f"the value of {canonical} is not plain printable text of at most "
                                f"{MAX_HEADER_VALUE_CHARS} characters", "invalid_header")
            sent[canonical] = value
        return sent


# --- Signing ----------------------------------------------------------------

REQUEST_LABEL = "tuff-sandbox-req-v1"
RESPONSE_LABEL = "tuff-sandbox-resp-v1"
NONCE_HEADER = "X-Sandbox-Nonce"
TIME_HEADER = "X-Sandbox-Time"
AUTH_HEADER = "X-Sandbox-Auth"
NONCE_PATTERN = re.compile(r"[0-9a-fA-F]{64}")
TIME_PATTERN = re.compile(r"[0-9]{1,12}")
AUTH_PATTERN = re.compile(r"[0-9a-fA-F]{64}")
TIME_WINDOW = 60
MIN_SECRET_BYTES = 32
MAX_SECRET_BYTES = 4096
DEFAULT_SECRET_FILE = "/run/tuff-sandbox/secret"
NONCE_CAPACITY = 100_000


def request_signature(secret: bytes, method: str, path: str, nonce: str, timestamp: str,
                      body: bytes) -> str:
    text = "\n".join((REQUEST_LABEL, method, path, nonce, timestamp, hashlib.sha256(body).hexdigest()))
    return hmac.new(secret, text.encode(), hashlib.sha256).hexdigest()


def response_signature(secret: bytes, method: str, path: str, nonce: str, status: int,
                       body: bytes) -> str:
    text = "\n".join((RESPONSE_LABEL, method, path, nonce, str(status), hashlib.sha256(body).hexdigest()))
    return hmac.new(secret, text.encode(), hashlib.sha256).hexdigest()


def load_secret(path: str) -> bytes:
    """The secret in a file, without one trailing newline. It exits when the
    file is missing, not a regular file, or too short or long, with a message
    that never holds the secret. Called once, as root, before privileges are
    dropped; the file is not read again."""
    try:
        descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    except OSError as error:
        sys.exit(f"cannot open the secret file: {error.strerror}")
    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            sys.exit("the secret file is not a regular file")
        with os.fdopen(descriptor, "rb") as handle:
            descriptor = -1
            # Two bytes more than the longest secret: one trailing newline is
            # stripped, and a second one then still makes it too long.
            value = handle.read(MAX_SECRET_BYTES + 2)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
    value = value.removesuffix(b"\n")
    if len(value) > MAX_SECRET_BYTES:
        sys.exit(f"the secret is longer than {MAX_SECRET_BYTES} bytes; refusing to start")
    if len(value) < MIN_SECRET_BYTES:
        sys.exit(f"the secret is shorter than {MIN_SECRET_BYTES} bytes; refusing to start")
    return value


class NonceStore:
    """Nonces accepted within the time window. Only requests with a correct
    signature are stored, so a caller without the secret cannot fill it."""

    def __init__(self, window: int = TIME_WINDOW, capacity: int = NONCE_CAPACITY):
        self.window = window
        self.capacity = capacity
        self.seen: dict[str, int] = {}
        self.lock = threading.Lock()
        self.pruned_at = 0.0

    def add(self, nonce: str, stamp: int, now: float) -> bool:
        """False when the nonce was accepted before and is still remembered."""
        key = nonce.lower()
        with self.lock:
            if now - self.pruned_at >= 1 or len(self.seen) >= self.capacity:
                self.seen = {k: e for k, e in self.seen.items() if e >= now}
                self.pruned_at = now
            if key in self.seen:
                return False
            if len(self.seen) >= self.capacity:
                raise ToolError("too many requests in the last minute", "busy", 503)
            # A request with this stamp passes the time check until stamp + window.
            self.seen[key] = stamp + self.window + 1
            return True


# --- HTTP API ---------------------------------------------------------------

def host_allowed(header: str | None) -> bool:
    """Only loopback Host names, so a web page in a Mac browser cannot reach
    this API through DNS rebinding."""
    if not header:
        return False
    host = header.strip().lower()
    if host.startswith("["):
        host = host[: host.find("]") + 1]
    elif ":" in host:
        host = host.rsplit(":", 1)[0]
    return host in LOOPBACK_HOSTS


PAGE_FIELDS = {"url", "max_bytes", "timeout_seconds", "max_characters"}
HTTP_FIELDS = {"url", "method", "headers", "body_base64", "max_bytes", "timeout_seconds"}


class Handler(BaseHTTPRequestHandler):
    server_version = "TUFFSandbox/1.0"
    # A client that connects and then says nothing (or stops in the middle of
    # a request) holds a thread; after this many seconds without data the
    # connection is closed. It limits waits on the client's socket only, not
    # the fetch a request runs.
    timeout = REQUEST_TIMEOUT
    tools: WebTools = WebTools()
    # Set by main(); tests set their own. Without a secret nothing is served.
    secret: bytes = b""
    nonces: NonceStore = NonceStore()
    now: Callable[[], float] = staticmethod(time.time)

    # The nonce of the request being answered, or None when its answer is not
    # signed, and the request line's method and path.
    _nonce: str | None = None
    _method: str = ""
    _path: str = ""

    def setup(self):
        super().setup()
        # One deadline for the whole request line, headers and body together:
        # Handler.timeout limits each wait, so a client sending a byte every
        # few seconds would hold its thread (and a process slot) for ever.
        self._read_timer = threading.Timer(READ_DEADLINE, self._read_expired)
        self._read_timer.daemon = True
        self._read_timer.start()

    def _read_expired(self):
        try:
            self.connection.shutdown(socket.SHUT_RD)
        except OSError:
            pass

    def _request_read(self):
        timer = getattr(self, "_read_timer", None)
        if timer is not None:
            timer.cancel()

    def finish(self):
        self._request_read()
        super().finish()

    def log_request(self, code="-", size="-"):
        # Method, path and status only: never headers, bodies or addresses.
        sys.stderr.write(f"{self.command} {self.path.split('?', 1)[0][:64]} {code}\n")

    def log_message(self, fmt, *args):
        sys.stderr.write(("%s\n" % (fmt % args))[:200])

    def _send(self, status: int, payload: dict):
        data = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        if self._nonce is not None and self.secret:
            self.send_header(AUTH_HEADER, response_signature(
                self.secret, self._method, self._path, self._nonce, status, data))
        self.end_headers()
        if self._method != "HEAD":
            self.wfile.write(data)

    def _error(self, error: ToolError):
        self._send(error.status, {"error": {"message": error.message, "code": error.code}})

    def _read_body(self, method: str) -> bytes:
        """Reads the body before any check, so the signature covers the bytes
        that were sent. A missing, unparsable or oversized Content-Length is an
        error; a client that stops sending (Handler.timeout) is answered with
        408 and the connection is closed."""
        if self.headers.get("Transfer-Encoding") is not None:
            self.close_connection = True
            raise ToolError("chunked requests are not accepted", "invalid_request", 400)
        raw = self.headers.get("Content-Length")
        if raw is None:
            if method == "POST":
                raise ToolError("Content-Length is required", "invalid_request", 411)
            return b""
        digits = raw.strip()
        if not (digits.isascii() and digits.isdigit()):
            length = -1
        elif len(digits.lstrip("0")) > 18:
            # Far above any limit; not parsed, so Python versions that cap the
            # size of int("9" * 5000) answer the same.
            length = MAX_REQUEST_BYTES + 1
        else:
            length = int(digits)
        if length < 0:
            raise ToolError("Content-Length is not a number", "invalid_request", 400)
        if length > MAX_REQUEST_BYTES:
            self.close_connection = True
            raise ToolError("request body is too large", "invalid_request", 413)
        try:
            data = self.rfile.read(length) if length else b""
        except (TimeoutError, socket.timeout):
            self.close_connection = True
            raise ToolError("timed out waiting for the request body", "request_timeout", 408)
        except OSError:
            self.close_connection = True
            raise ToolError("the request body could not be read", "invalid_request", 400)
        if len(data) < length:
            # The read deadline closed the reading side, or the client left.
            self.close_connection = True
            raise ToolError("timed out waiting for the request body", "request_timeout", 408)
        return data

    def _note_nonce(self) -> None:
        """Remembers a well-formed nonce so that every answer to this request
        is signed, error answers too. Signing an error reveals nothing: the
        answer's status and body are the server's own, only the nonce, method
        and path come from the caller."""
        values = self.headers.get_all(NONCE_HEADER) or []
        if len(values) == 1 and NONCE_PATTERN.fullmatch(values[0].strip()):
            self._nonce = values[0].strip()

    def _authenticate(self, method: str, body: bytes) -> None:
        if not self.secret:  # main() always sets one; nothing is served without it
            raise ToolError("the sandbox has no secret", "internal_error", 500)
        fail = ToolError("the request is not signed correctly", "unauthorized", 401)
        stamps = self.headers.get_all(TIME_HEADER) or []
        auths = self.headers.get_all(AUTH_HEADER) or []
        if (self._nonce is None or len(stamps) != 1 or len(auths) != 1
                or not TIME_PATTERN.fullmatch(stamps[0].strip())
                or not AUTH_PATTERN.fullmatch(auths[0].strip())):
            raise fail
        stamp_text = stamps[0].strip()
        now = self.now()
        if abs(now - int(stamp_text)) > TIME_WINDOW:
            raise ToolError("the request's time is too far from the sandbox's clock",
                            "stale_request", 401)
        expected = request_signature(self.secret, method, self._path, self._nonce, stamp_text, body)
        if not hmac.compare_digest(expected, auths[0].strip().lower()):
            raise fail
        if not self.nonces.add(self._nonce, int(stamp_text), now):
            raise ToolError("this request was already received", "replayed_request", 401)

    def _handle(self, method: str):
        self._nonce = None
        self._method = method
        self._path = self.path
        self._note_nonce()
        try:
            try:
                body = self._read_body(method)
            finally:
                self._request_read()
            if not host_allowed(self.headers.get("Host")):
                raise ToolError("forbidden host", "forbidden_host", 403)
            self._authenticate(method, body)
            if method == "GET":
                if self.path != "/health":
                    raise ToolError("not found", "not_found", 404)
                return self._send(200, {
                    "status": "ok", "profile": "web", "api": 1, "pdf": PdfReader is not None,
                    "extractor": "trafilatura" if trafilatura is not None else "fallback",
                    "blocked_addresses": len(BLOCKED_ADDRESSES)})
            if method != "POST":
                raise ToolError("method not allowed", "method_not_allowed", 405)
            if self.path not in ("/v1/page", "/v1/http"):
                raise ToolError("not found", "not_found", 404)
            # A browser cannot send application/json cross-origin without a
            # preflight, which this server never answers.
            media_type = (self.headers.get("Content-Type") or "").split(";")[0].strip().lower()
            if media_type != "application/json":
                raise ToolError("Content-Type must be application/json", "invalid_request", 415)
            if not body:
                raise ToolError("request body is missing", "invalid_request", 400)
            try:
                request = json.loads(body)
            except (ValueError, RecursionError):
                raise ToolError("request body is not JSON", "invalid_request", 400) from None
            if not isinstance(request, dict):
                raise ToolError("request body must be a JSON object", "invalid_request", 400)
            if self.path == "/v1/page":
                self._fields(request, PAGE_FIELDS)
                result = self.tools.page(
                    request.get("url"), request.get("max_bytes", DEFAULT_MAX_BYTES),
                    request.get("timeout_seconds", DEFAULT_TIMEOUT),
                    request.get("max_characters", DEFAULT_MAX_CHARACTERS))
            else:
                self._fields(request, HTTP_FIELDS)
                result = self.tools.http(
                    request.get("url"), request.get("method"), request.get("headers"),
                    request.get("body_base64"), request.get("max_bytes", DEFAULT_MAX_BYTES),
                    request.get("timeout_seconds", DEFAULT_TIMEOUT))
            self._send(200, result)
        except ToolError as error:
            self._error(error)
        except Exception as error:  # Never leak a traceback to the caller.
            sys.stderr.write(f"internal error: {type(error).__name__}\n")
            self._error(ToolError("internal error", "internal_error", 500))

    @staticmethod
    def _fields(request: dict, allowed: set) -> None:
        unknown = sorted(set(request) - allowed)
        if unknown:
            raise ToolError(f"unknown field {unknown[0][:40]!r}", "invalid_argument", 400)

    def do_GET(self):
        self._handle("GET")

    def do_POST(self):
        self._handle("POST")

    def do_PUT(self):
        self._handle("PUT")

    def do_DELETE(self):
        self._handle("DELETE")

    def do_PATCH(self):
        self._handle("PATCH")

    def do_HEAD(self):
        self._handle("HEAD")

    def do_OPTIONS(self):
        self._handle("OPTIONS")


class BoundedServer(ThreadingHTTPServer):
    """ThreadingHTTPServer that serves at most `max_connections` connections at
    once. Past that, a new connection is closed without an answer."""
    daemon_threads = True

    def __init__(self, address, handler, max_connections: int = MAX_CONNECTIONS):
        super().__init__(address, handler)
        self._slots = threading.BoundedSemaphore(max_connections)

    def process_request(self, request, client_address):
        if not self._slots.acquire(blocking=False):
            self.shutdown_request(request)
            return
        try:
            super().process_request(request, client_address)
        except BaseException:
            self._slots.release()
            raise

    def process_request_thread(self, request, client_address):
        try:
            super().process_request_thread(request, client_address)
        finally:
            self._slots.release()


def drop_privileges(uid: int = SANDBOX_UID) -> None:
    """Started as root so entrypoint.sh can load the firewall; from here on
    the server runs as an unprivileged user with no capabilities and cannot
    regain any, so it cannot change the firewall either."""
    if os.getuid() != 0:
        return
    libc = ctypes.CDLL(None, use_errno=True)
    pr_capbset_drop, pr_set_no_new_privs = 24, 38
    for capability in range(64):
        if libc.prctl(pr_capbset_drop, capability, 0, 0, 0) != 0:
            if ctypes.get_errno() == errno.EINVAL:
                break  # Past the last capability this kernel knows.
            raise OSError(ctypes.get_errno(), "could not drop a capability")
    os.setgroups([])
    os.setgid(uid)
    os.setuid(uid)
    if libc.prctl(pr_set_no_new_privs, 1, 0, 0, 0) != 0:
        raise OSError(ctypes.get_errno(), "could not set no_new_privs")
    with open("/proc/self/status") as status:
        fields = dict(line.split(":", 1) for line in status if ":" in line)
    for name in ("CapInh", "CapPrm", "CapEff", "CapBnd", "CapAmb"):
        if int(fields.get(name, "0").strip(), 16) != 0:
            raise OSError(errno.EPERM, f"{name} is not empty after dropping privileges")


def stop_on_sigterm(server) -> None:
    """The server is PID 1 in the VM and PID 1 ignores SIGTERM unless it has a
    handler, so `container stop` would wait out its grace period before
    killing it. Shut down cleanly instead. `shutdown()` waits for
    `serve_forever()`, which the signal interrupts in this very thread, so it
    runs in another one."""
    def handler(signum, frame):
        threading.Thread(target=server.shutdown, daemon=True).start()
    signal.signal(signal.SIGTERM, handler)


def main():
    global BLOCKED_ADDRESSES
    # The secret is read first, while this process is still root, and only
    # once. The blocklist is taken out of the environment as well.
    Handler.secret = load_secret(os.environ.get("SANDBOX_SECRET_FILE", DEFAULT_SECRET_FILE))
    try:
        BLOCKED_ADDRESSES = load_blocked_addresses(os.environ.pop(BLOCKED_ENV, ""))
    except ValueError as error:
        sys.exit(f"{BLOCKED_ENV}: {error}; refusing to start")
    drop_privileges()
    # 0.0.0.0, not 127.0.0.1: `container run --publish` forwards the Mac's
    # 127.0.0.1:<port> to this VM's network address, not to its loopback, so a
    # loopback-only server would never be reached. The VM's input firewall
    # (firewall-web.nft) accepts new connections on this port only, and only
    # from the VM's default gateway, which entrypoint.sh puts into the rule;
    # the Mac side is published on 127.0.0.1 only. Anything that can send from
    # the gateway address (processes on the Mac) can still connect, so the
    # Host check and the signature check stay.
    server = BoundedServer(("0.0.0.0", PORT), Handler)
    stop_on_sigterm(server)
    sys.stderr.write(f"TUFF web sandbox listening on :{PORT}\n")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    sys.exit(0)


if __name__ == "__main__":
    main()
