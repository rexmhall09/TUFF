#!/usr/bin/env python3
"""End-to-end tests for the signed HTTP API of the web sandbox (server.py).

A real ThreadingHTTPServer on 127.0.0.1 serves server.Handler with a fake
resolver and a fake transport, a controllable clock and a fresh NonceStore per
test. Requests are built here with an independent HMAC computation (not with
server.request_signature), and every answer is checked the same way. Runs with
the standard library only: nothing here needs trafilatura or pypdf.

    cd Sandbox && python3 test_signing.py
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import http.client
import io
import json
import os
import re
import signal
import socket
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from unittest import mock

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import sandbox_client  # noqa: E402
import server  # noqa: E402

with open(os.path.join(HERE, "test-vectors", "signing.json"), encoding="utf-8") as _handle:
    VECTORS = json.load(_handle)
VECTOR_SECRET = VECTORS["secret_utf8"].encode()

SECRET = b"unit-test-secret-0123456789abcdef-0123456789abcdef"
NONCE = "X-Sandbox-Nonce"
TIME = "X-Sandbox-Time"
AUTH = "X-Sandbox-Auth"
AUTO = object()
START = 1_800_000_000


def read_source(name):
    with open(os.path.join(HERE, name), encoding="utf-8") as handle:
        return handle.read()


# --- Independent signature computation --------------------------------------

def expected_request_auth(secret, method, path, nonce, stamp, body):
    text = ("tuff-sandbox-req-v1\n" + method + "\n" + path + "\n" + nonce + "\n" + stamp
            + "\n" + hashlib.sha256(body).hexdigest())
    return hmac.new(secret, text.encode("utf-8"), hashlib.sha256).hexdigest()


def expected_response_auth(secret, method, path, nonce, status, body):
    text = ("tuff-sandbox-resp-v1\n" + method + "\n" + path + "\n" + nonce + "\n" + str(status)
            + "\n" + hashlib.sha256(body).hexdigest())
    return hmac.new(secret, text.encode("utf-8"), hashlib.sha256).hexdigest()


# --- Header list edits ------------------------------------------------------

def drop(name):
    return lambda headers: [h for h in headers if h[0].lower() != name.lower()]


def add(name, value):
    return lambda headers: headers + [(name, value)]


def replace(name, value):
    return lambda headers: [(n, value if n.lower() == name.lower() else v) for n, v in headers]


def both(*edits):
    def run(headers):
        for edit in edits:
            headers = edit(headers)
        return headers
    return run


# --- Harness ----------------------------------------------------------------

class Clock:
    def __init__(self, value):
        self.value = float(value)

    def __call__(self):
        return self.value


class FakeWorld:
    """The resolver and transport behind WebTools: no network is touched."""

    def __init__(self):
        self.calls = []
        self.failure = None
        self.reply = server.Response(200, {"content-type": "text/plain; charset=utf-8"},
                                     b"hello world")

    def resolver(self, host, port):
        return ["93.184.216.34"]

    def transport(self, method, target, headers, body, deadline=None, max_bytes=None):
        self.calls.append((method, target, dict(headers), body))
        if self.failure is not None:
            raise self.failure
        return self.reply


class Answer:
    def __init__(self, response, body, method, path, nonce):
        self.status = response.status
        self.headers = {k.lower(): v for k, v in response.getheaders()}
        self.body = body
        self.method = method
        self.path = path
        self.nonce = nonce

    @property
    def signature(self):
        return self.headers.get("x-sandbox-auth")

    @property
    def json(self):
        return json.loads(self.body)

    @property
    def code(self):
        return self.json["error"]["code"]


class LiveServer(unittest.TestCase):
    """A loopback server per test, with the secret and clock under test control."""

    secret = SECRET
    server_timeout = None

    def setUp(self):
        self.clock = Clock(START)
        self.world = FakeWorld()
        self.log = io.StringIO()
        tools = server.WebTools(resolver=self.world.resolver, transport=self.world.transport)
        for patch in (
            mock.patch.object(sys, "stderr", self.log),
            mock.patch.object(server.Handler, "secret", self.secret),
            mock.patch.object(server.Handler, "tools", tools),
            mock.patch.object(server.Handler, "nonces", server.NonceStore()),
            mock.patch.object(server.Handler, "now", staticmethod(self.clock)),
        ):
            patch.start()
            self.addCleanup(patch.stop)
        self.httpd = server.ThreadingHTTPServer(("127.0.0.1", 0), server.Handler)
        self.httpd.daemon_threads = True
        self.port = self.httpd.server_address[1]
        self.thread = threading.Thread(target=self.httpd.serve_forever,
                                       kwargs={"poll_interval": 0.01}, daemon=True)
        self.thread.start()
        self.addCleanup(self._stop)

    def _stop(self):
        self.httpd.shutdown()
        self.httpd.server_close()
        self.thread.join(5)

    # -- requests ---------------------------------------------------------

    def headers_for(self, method, path, body=b"", *, nonce=None, stamp=None, secret=None,
                    content_type=AUTO, host=AUTO, length=AUTO, sign=None, auth=None):
        nonce = os.urandom(32).hex() if nonce is None else nonce
        stamp = str(int(self.clock.value)) if stamp is None else str(stamp)
        secret = self.secret if secret is None else secret
        signed = {"method": method, "path": path, "body": body, **(sign or {})}
        if auth is None:
            auth = expected_request_auth(secret, signed["method"], signed["path"], nonce, stamp,
                                         signed["body"])
        headers = []
        if host is AUTO:
            host = f"127.0.0.1:{self.port}"
        if host is not None:
            headers.append(("Host", host))
        headers += [(NONCE, nonce), (TIME, stamp), (AUTH, auth)]
        if content_type is AUTO:
            content_type = "application/json" if method == "POST" else None
        if content_type is not None:
            headers.append(("Content-Type", content_type))
        if length is AUTO:
            length = str(len(body)) if (method == "POST" or body) else None
        if length is not None:
            headers.append(("Content-Length", length))
        return nonce, headers

    def send_raw(self, method, path, headers, body=b"", nonce=None, timeout=5):
        connection = http.client.HTTPConnection("127.0.0.1", self.port, timeout=timeout)
        try:
            connection.putrequest(method, path, skip_host=True, skip_accept_encoding=True)
            for name, value in headers:
                connection.putheader(name, value)
            connection.endheaders(body or None)
            response = connection.getresponse()
            return Answer(response, response.read(), method, path, nonce)
        finally:
            connection.close()

    def exchange(self, method, path, body=b"", *, edit=None, timeout=5, **options):
        nonce, headers = self.headers_for(method, path, body, **options)
        if edit is not None:
            headers = edit(headers)
        return self.send_raw(method, path, headers, body, nonce=nonce, timeout=timeout)

    def post(self, path, payload, **options):
        return self.exchange("POST", path, json.dumps(payload).encode(), **options)

    # -- assertions -------------------------------------------------------

    def assertSigned(self, answer, body=None, nonce=None):
        expected = expected_response_auth(
            self.secret, answer.method, answer.path, answer.nonce if nonce is None else nonce,
            answer.status, answer.body if body is None else body)
        self.assertEqual(answer.signature, expected)

    def assertError(self, answer, status, code, signed=True):
        self.assertEqual((answer.status, answer.code), (status, code), answer.body)
        if signed:
            self.assertSigned(answer)
        else:
            self.assertIsNone(answer.signature)

    def page_payload(self):
        return {"url": "https://example.com/"}


# --- Test vectors -----------------------------------------------------------

class VectorTests(unittest.TestCase):
    def test_the_vector_file_has_the_expected_shape(self):
        self.assertGreaterEqual(len(VECTORS["request"]), 3)
        self.assertGreaterEqual(len(VECTORS["response"]), 3)
        self.assertGreaterEqual(len(VECTOR_SECRET), server.MIN_SECRET_BYTES)

    def test_request_vectors(self):
        for vector in VECTORS["request"]:
            with self.subTest(vector["name"]):
                body = vector["body_utf8"].encode("utf-8")
                self.assertEqual(hashlib.sha256(body).hexdigest(), vector["body_sha256"])
                args = (VECTOR_SECRET, vector["method"], vector["path"], vector["nonce"],
                        vector["time"], body)
                self.assertEqual(server.request_signature(*args), vector["expected_auth"])
                self.assertEqual(sandbox_client.request_auth(*args), vector["expected_auth"])
                self.assertEqual(expected_request_auth(*args), vector["expected_auth"])

    def test_response_vectors(self):
        for vector in VECTORS["response"]:
            with self.subTest(vector["name"]):
                body = vector["body_utf8"].encode("utf-8")
                self.assertEqual(hashlib.sha256(body).hexdigest(), vector["body_sha256"])
                args = (VECTOR_SECRET, vector["method"], vector["path"], vector["nonce"],
                        vector["status"], body)
                self.assertEqual(server.response_signature(*args), vector["expected_auth"])
                self.assertEqual(sandbox_client.response_auth(*args), vector["expected_auth"])
                self.assertEqual(expected_response_auth(*args), vector["expected_auth"])

    def test_every_field_of_a_vector_changes_the_signature(self):
        vector = VECTORS["request"][1]
        body = vector["body_utf8"].encode()
        base = [VECTOR_SECRET, vector["method"], vector["path"], vector["nonce"], vector["time"],
                body]
        for index, other in ((0, b"x" * 41), (1, "GET"), (2, "/v1/http"), (3, "00" * 32),
                             (4, "1786000043"), (5, body + b" ")):
            args = list(base)
            args[index] = other
            self.assertNotEqual(server.request_signature(*args), vector["expected_auth"], index)


class VectorServerTests(LiveServer):
    secret = VECTOR_SECRET

    def test_the_server_accepts_each_vector_request_exactly_as_written(self):
        # expected_auth comes straight from the JSON file: no code computes it.
        for vector in VECTORS["request"]:
            with self.subTest(vector["name"]):
                self.clock.value = int(vector["time"])
                answer = self.exchange(
                    vector["method"], vector["path"], vector["body_utf8"].encode("utf-8"),
                    nonce=vector["nonce"], stamp=vector["time"], auth=vector["expected_auth"])
                self.assertEqual(answer.status, 200, answer.body)
                self.assertSigned(answer)


# --- Valid requests ---------------------------------------------------------

class SignedRoundTripTests(LiveServer):
    def test_health(self):
        answer = self.exchange("GET", "/health")
        self.assertEqual(answer.status, 200)
        self.assertSigned(answer)
        data = answer.json
        self.assertEqual((data["status"], data["profile"], data["api"]), ("ok", "web", 1))
        self.assertEqual(data["blocked_addresses"], len(server.BLOCKED_ADDRESSES))
        self.assertIn("pdf", data)
        self.assertIn(data["extractor"], ("trafilatura", "fallback"))
        self.assertEqual(answer.headers["content-type"], "application/json")
        self.assertEqual(answer.headers["cache-control"], "no-store")
        self.assertEqual(int(answer.headers["content-length"]), len(answer.body))

    def test_page(self):
        answer = self.post("/v1/page", self.page_payload())
        self.assertEqual(answer.status, 200, answer.body)
        self.assertSigned(answer)
        data = answer.json
        self.assertEqual((data["final_url"], data["status"], data["text"]),
                         ("https://example.com/", 200, "hello world"))
        self.assertEqual(len(self.world.calls), 1)
        method, target, _, body = self.world.calls[0]
        self.assertEqual((method, target.address, body), ("GET", "93.184.216.34", None))

    def test_http(self):
        self.world.reply = server.Response(200, {"content-type": "application/json"},
                                           b'{"results": []}')
        payload = {"url": "https://api.tavily.com/search", "method": "POST",
                   "headers": {"Content-Type": "application/json"},
                   "body_base64": base64.b64encode(b'{"query": "x"}').decode()}
        answer = self.post("/v1/http", payload)
        self.assertEqual(answer.status, 200, answer.body)
        self.assertSigned(answer)
        data = answer.json
        self.assertEqual(base64.b64decode(data["body_base64"]), b'{"results": []}')
        self.assertEqual(self.world.calls[0][3], b'{"query": "x"}')

    def test_sandbox_client_call_verifies_the_answers(self):
        self.clock.value = time.time()
        status, body, signed = sandbox_client.call(self.port, SECRET, "GET", "/health")
        self.assertEqual((status, signed), (200, True))
        self.assertEqual(json.loads(body)["status"], "ok")
        status, body, signed = sandbox_client.call(self.port, SECRET, "POST", "/v1/page",
                                                   self.page_payload())
        self.assertEqual((status, signed), (200, True))
        self.assertEqual(json.loads(body)["text"], "hello world")

    def test_sandbox_client_call_with_the_wrong_secret_gets_a_signed_401(self):
        self.clock.value = time.time()
        status, body, signed = sandbox_client.call(self.port, b"w" * 40, "GET", "/health")
        # The answer is signed with the server's secret, so the client (which
        # holds another one) cannot verify it.
        self.assertEqual(status, 401)
        self.assertFalse(signed)
        self.assertEqual(json.loads(body)["error"]["code"], "unauthorized")

    def test_uppercase_hex_in_the_auth_header_is_accepted(self):
        nonce, headers = self.headers_for("GET", "/health")
        headers = [(n, v.upper() if n == AUTH else v) for n, v in headers]
        answer = self.send_raw("GET", "/health", headers, nonce=nonce)
        self.assertEqual(answer.status, 200)
        self.assertSigned(answer)

    def test_a_get_with_a_body_is_signed_over_that_body(self):
        answer = self.exchange("GET", "/health", b"abc")
        self.assertEqual(answer.status, 200)
        bad = self.exchange("GET", "/health", b"abc", sign={"body": b""})
        self.assertError(bad, 401, "unauthorized")


# --- Authentication ---------------------------------------------------------

class AuthenticationTests(LiveServer):
    def test_a_missing_duplicated_or_malformed_nonce_gets_an_unsigned_401(self):
        other = "ef" * 32
        cases = [("missing", drop(NONCE)), ("duplicated", add(NONCE, other))]
        for value in ("", "ab" * 31, "a" * 63, "a" * 65, "g" + "a" * 63, "ab" * 32 + "!",
                      "0x" + "a" * 62):
            cases.append((f"malformed {value!r}", replace(NONCE, value)))
        for name, edit in cases:
            with self.subTest(name):
                answer = self.exchange("GET", "/health", edit=edit)
                self.assertError(answer, 401, "unauthorized", signed=False)

    def test_a_missing_duplicated_or_malformed_time_gets_a_signed_401(self):
        cases = [("missing", drop(TIME)), ("duplicated", add(TIME, str(START)))]
        for value in ("", "abc", "-5", "1.5", "0x10", "1234567890123", "12 34"):
            cases.append((f"malformed {value!r}", replace(TIME, value)))
        for name, edit in cases:
            with self.subTest(name):
                answer = self.exchange("GET", "/health", edit=edit)
                self.assertError(answer, 401, "unauthorized")

    def test_a_missing_duplicated_or_malformed_auth_gets_a_signed_401(self):
        cases = [("missing", drop(AUTH)), ("duplicated", add(AUTH, "ab" * 32))]
        for value in ("", "ab" * 31, "a" * 63, "a" * 65, "z" * 64, "ab" * 32 + "!"):
            cases.append((f"malformed {value!r}", replace(AUTH, value)))
        for name, edit in cases:
            with self.subTest(name):
                answer = self.exchange("GET", "/health", edit=edit)
                self.assertError(answer, 401, "unauthorized")

    def test_the_duplicate_of_a_correct_header_is_still_refused(self):
        nonce, headers = self.headers_for("GET", "/health")
        auth = dict(headers)[AUTH]
        answer = self.send_raw("GET", "/health", headers + [(AUTH, auth)], nonce=nonce)
        self.assertError(answer, 401, "unauthorized")

    def test_a_request_without_any_signature_headers(self):
        nonce, headers = self.headers_for("GET", "/health")
        answer = self.send_raw("GET", "/health", headers[:1], nonce=nonce)
        self.assertError(answer, 401, "unauthorized", signed=False)

    def test_the_wrong_secret(self):
        answer = self.exchange("GET", "/health", secret=b"w" * 40)
        self.assertError(answer, 401, "unauthorized")

    def test_a_secret_that_differs_in_one_byte(self):
        answer = self.exchange("GET", "/health", secret=SECRET[:-1] + b"X")
        self.assertError(answer, 401, "unauthorized")

    def test_a_body_changed_after_signing(self):
        signed_body = json.dumps(self.page_payload()).encode()
        sent_body = json.dumps({"url": "https://example.org/"}).encode()
        answer = self.exchange("POST", "/v1/page", sent_body, sign={"body": signed_body})
        self.assertError(answer, 401, "unauthorized")
        self.assertEqual(self.world.calls, [])

    def test_a_method_changed_after_signing(self):
        answer = self.exchange("GET", "/health", sign={"method": "POST"})
        self.assertError(answer, 401, "unauthorized")
        answer = self.exchange("POST", "/v1/page", json.dumps(self.page_payload()).encode(),
                               sign={"method": "GET"})
        self.assertError(answer, 401, "unauthorized")

    def test_a_path_changed_after_signing(self):
        answer = self.exchange("GET", "/health", sign={"path": "/v1/page"})
        self.assertError(answer, 401, "unauthorized")
        body = json.dumps(self.page_payload()).encode()
        answer = self.exchange("POST", "/v1/page", body, sign={"path": "/v1/http"})
        self.assertError(answer, 401, "unauthorized")

    def test_a_query_string_is_part_of_the_signed_path(self):
        answer = self.exchange("GET", "/health?x=1", sign={"path": "/health"})
        self.assertError(answer, 401, "unauthorized")

    def test_a_nonce_or_time_changed_after_signing(self):
        nonce, headers = self.headers_for("GET", "/health")
        other = "12" * 32
        answer = self.send_raw("GET", "/health", replace(NONCE, other)(headers), nonce=other)
        self.assertError(answer, 401, "unauthorized")
        nonce, headers = self.headers_for("GET", "/health")
        answer = self.send_raw("GET", "/health", replace(TIME, str(START + 1))(headers),
                               nonce=nonce)
        self.assertError(answer, 401, "unauthorized")

    def test_time_61_seconds_off_in_either_direction_is_stale(self):
        for offset in (61, -61, 3600, -3600):
            with self.subTest(offset=offset):
                answer = self.exchange("GET", "/health", stamp=START + offset)
                self.assertError(answer, 401, "stale_request")

    def test_time_exactly_60_seconds_off_is_accepted_in_either_direction(self):
        for offset in (60, -60, 59, -59, 0):
            with self.subTest(offset=offset):
                answer = self.exchange("GET", "/health", stamp=START + offset)
                self.assertEqual(answer.status, 200)
                self.assertSigned(answer)

    def test_a_fraction_of_a_second_over_the_window_is_stale(self):
        self.clock.value = START + 0.5
        self.assertError(self.exchange("GET", "/health", stamp=START - 60), 401, "stale_request")
        self.assertEqual(self.exchange("GET", "/health", stamp=START - 59).status, 200)
        self.assertEqual(self.exchange("GET", "/health", stamp=START + 60).status, 200)
        self.assertError(self.exchange("GET", "/health", stamp=START + 61), 401, "stale_request")

    def test_the_largest_time_value_is_stale_not_an_error(self):
        answer = self.exchange("GET", "/health", stamp="999999999999")
        self.assertError(answer, 401, "stale_request")

    def test_a_stale_request_does_not_use_up_its_nonce(self):
        nonce = "5a" * 32
        self.assertError(self.exchange("GET", "/health", nonce=nonce, stamp=START - 100),
                         401, "stale_request")
        self.assertEqual(server.Handler.nonces.seen, {})
        self.assertEqual(self.exchange("GET", "/health", nonce=nonce).status, 200)

    def test_the_same_request_again_is_a_replay(self):
        nonce, headers = self.headers_for("GET", "/health")
        first = self.send_raw("GET", "/health", headers, nonce=nonce)
        second = self.send_raw("GET", "/health", headers, nonce=nonce)
        self.assertEqual(first.status, 200)
        self.assertError(second, 401, "replayed_request")

    def test_a_replay_is_refused_for_a_new_signature_over_the_same_nonce(self):
        nonce = "7c" * 32
        self.assertEqual(self.exchange("GET", "/health", nonce=nonce).status, 200)
        self.clock.value += 5
        answer = self.exchange("GET", "/health", nonce=nonce)
        self.assertError(answer, 401, "replayed_request")

    def test_a_nonce_in_the_other_letter_case_is_the_same_nonce(self):
        nonce = "ab" * 31 + "cd"
        self.assertEqual(self.exchange("GET", "/health", nonce=nonce).status, 200)
        answer = self.exchange("GET", "/health", nonce=nonce.upper())
        self.assertError(answer, 401, "replayed_request")

    def test_the_same_nonce_is_accepted_again_after_it_expires(self):
        nonce = "9e" * 32
        self.assertEqual(self.exchange("GET", "/health", nonce=nonce).status, 200)
        self.clock.value += 30
        self.assertError(self.exchange("GET", "/health", nonce=nonce), 401, "replayed_request")
        self.clock.value = START + 62
        answer = self.exchange("GET", "/health", nonce=nonce)
        self.assertEqual(answer.status, 200)
        self.assertSigned(answer)

    def test_the_same_request_after_the_window_is_stale(self):
        nonce, headers = self.headers_for("GET", "/health")
        self.assertEqual(self.send_raw("GET", "/health", headers, nonce=nonce).status, 200)
        self.clock.value += 120
        self.assertError(self.send_raw("GET", "/health", headers, nonce=nonce), 401,
                         "stale_request")

    def test_a_request_without_a_good_signature_does_not_store_its_nonce(self):
        nonce = "cd" * 32
        self.assertError(self.exchange("GET", "/health", nonce=nonce, secret=b"w" * 40),
                         401, "unauthorized")
        self.assertEqual(server.Handler.nonces.seen, {})
        answer = self.exchange("GET", "/health", nonce=nonce)
        self.assertEqual(answer.status, 200)
        self.assertEqual(list(server.Handler.nonces.seen), [nonce])

    def test_unauthenticated_requests_cannot_fill_the_nonce_store(self):
        self.set_nonces(capacity=2)
        for _ in range(5):
            self.assertError(self.exchange("GET", "/health", secret=b"w" * 40), 401,
                             "unauthorized")
        self.assertEqual(server.Handler.nonces.seen, {})
        self.assertEqual(self.exchange("GET", "/health").status, 200)

    def set_nonces(self, **options):
        server.Handler.nonces = server.NonceStore(**options)

    def test_a_full_nonce_store_answers_503_busy_and_prunes_when_time_has_passed(self):
        self.set_nonces(capacity=2)
        self.assertEqual(self.exchange("GET", "/health").status, 200)
        self.assertEqual(self.exchange("GET", "/health").status, 200)
        refused_nonce = "3b" * 32
        answer = self.exchange("GET", "/health", nonce=refused_nonce)
        self.assertError(answer, 503, "busy")
        self.assertNotIn(refused_nonce, server.Handler.nonces.seen)
        self.assertEqual(len(server.Handler.nonces.seen), 2)
        self.clock.value += 62
        again = self.exchange("GET", "/health", nonce=refused_nonce)
        self.assertEqual(again.status, 200)
        self.assertEqual(len(server.Handler.nonces.seen), 1)

    def test_a_replay_is_reported_as_such_even_when_the_store_is_full(self):
        self.set_nonces(capacity=1)
        nonce, headers = self.headers_for("GET", "/health")
        self.assertEqual(self.send_raw("GET", "/health", headers, nonce=nonce).status, 200)
        self.assertError(self.send_raw("GET", "/health", headers, nonce=nonce), 401,
                         "replayed_request")

    def test_an_empty_secret_serves_nothing(self):
        # Without a secret every authenticated route answers 500 internal_error,
        # unsigned (there is no key to sign with), even for a request "signed"
        # with the empty key.
        with mock.patch.object(server.Handler, "secret", b""):
            nonce, headers = self.headers_for("GET", "/health", secret=b"")
            answer = self.send_raw("GET", "/health", headers, nonce=nonce)
            self.assertError(answer, 500, "internal_error", signed=False)
            body = json.dumps(self.page_payload()).encode()
            answer = self.exchange("POST", "/v1/page", body, secret=b"")
            self.assertError(answer, 500, "internal_error", signed=False)
        self.assertEqual(self.world.calls, [])
        self.assertEqual(server.Handler.nonces.seen, {})


# --- NonceStore (unit) ------------------------------------------------------

class NonceStoreTests(unittest.TestCase):
    def test_add_remembers_and_refuses_a_second_time_ignoring_case(self):
        store = server.NonceStore()
        self.assertTrue(store.add("aa" * 32, 1000, 1000))
        self.assertFalse(store.add("aa" * 32, 1000, 1001))
        self.assertFalse(store.add("AA" * 32, 1000, 1001))
        self.assertTrue(store.add("bb" * 32, 1000, 1001))

    def test_an_entry_is_dropped_once_its_stamp_can_no_longer_pass_the_time_check(self):
        store = server.NonceStore()
        self.assertTrue(store.add("aa" * 32, 1000, 1000))
        # A request stamped 1000 passes the time check until 1060.
        self.assertFalse(store.add("aa" * 32, 1060, 1060))
        self.assertTrue(store.add("aa" * 32, 1062, 1062))

    def test_the_capacity_is_enforced_with_a_503_busy_tool_error(self):
        store = server.NonceStore(capacity=2)
        store.add("01" * 32, 1000, 1000)
        store.add("02" * 32, 1000, 1000)
        with self.assertRaises(server.ToolError) as caught:
            store.add("03" * 32, 1000, 1000)
        self.assertEqual((caught.exception.status, caught.exception.code), (503, "busy"))
        self.assertEqual(len(store.seen), 2)

    def test_old_entries_are_pruned_to_make_room(self):
        store = server.NonceStore(capacity=2)
        store.add("01" * 32, 1000, 1000)
        store.add("02" * 32, 1000, 1000)
        self.assertTrue(store.add("03" * 32, 1100, 1100))
        self.assertEqual(set(store.seen), {"03" * 32})

    def test_a_replay_of_a_remembered_nonce_is_false_not_an_error_when_full(self):
        store = server.NonceStore(capacity=1)
        store.add("01" * 32, 1000, 1000)
        self.assertFalse(store.add("01" * 32, 1000, 1000))

    def test_many_threads_adding_one_nonce_succeed_exactly_once(self):
        store = server.NonceStore()
        results = []

        def work():
            results.append(store.add("ee" * 32, 1000, 1000))

        threads = [threading.Thread(target=work) for _ in range(16)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        self.assertEqual(results.count(True), 1)
        self.assertEqual(results.count(False), 15)


# --- The answer's signature -------------------------------------------------

class ResponseBindingTests(LiveServer):
    def test_the_signature_binds_method_path_nonce_status_and_body(self):
        answer = self.exchange("GET", "/health")
        sent = answer.signature
        self.assertSigned(answer)
        base = [SECRET, "GET", "/health", answer.nonce, 200, answer.body]
        self.assertEqual(expected_response_auth(*base), sent)
        variants = {
            "secret": (0, b"y" * 40),
            "method": (1, "POST"),
            "path": (2, "/v1/page"),
            "nonce": (3, "00" * 32),
            "status": (4, 201),
            "body": (5, answer.body + b" "),
        }
        for name, (index, value) in variants.items():
            with self.subTest(name):
                args = list(base)
                args[index] = value
                self.assertNotEqual(expected_response_auth(*args), sent)
                if index:
                    self.assertNotEqual(server.response_signature(*args), sent)

    def test_the_same_request_with_another_nonce_gets_another_signature(self):
        a = self.exchange("GET", "/health")
        b = self.exchange("GET", "/health")
        self.assertEqual(a.body, b.body)
        self.assertNotEqual(a.signature, b.signature)

    def test_an_error_answer_is_not_valid_as_another_status(self):
        answer = self.exchange("GET", "/nope")
        self.assertError(answer, 404, "not_found")
        self.assertNotEqual(expected_response_auth(SECRET, "GET", "/nope", answer.nonce, 200,
                                                   answer.body), answer.signature)

    def test_the_answer_signature_is_lowercase_hex_of_64_characters(self):
        answer = self.exchange("GET", "/health")
        self.assertRegex(answer.signature, r"\A[0-9a-f]{64}\Z")

    def test_the_signed_nonce_is_the_one_as_sent_in_the_request(self):
        nonce = "AB" * 32
        answer = self.exchange("GET", "/health", nonce=nonce)
        self.assertSigned(answer, nonce=nonce)
        self.assertNotEqual(answer.signature, expected_response_auth(
            SECRET, "GET", "/health", nonce.lower(), 200, answer.body))


# --- Error answers ----------------------------------------------------------

class ErrorAnswerTests(LiveServer):
    def test_unknown_paths_are_404(self):
        self.assertError(self.exchange("GET", "/nope"), 404, "not_found")
        self.assertError(self.exchange("GET", "/v1/page"), 404, "not_found")
        self.assertError(self.exchange("GET", "/health/"), 404, "not_found")
        self.assertError(self.exchange("POST", "/v1/nope", b"{}"), 404, "not_found")
        self.assertError(self.exchange("POST", "/health", b"{}"), 404, "not_found")

    def test_a_query_string_makes_the_path_unknown(self):
        self.assertError(self.exchange("GET", "/health?x=1"), 404, "not_found")

    def test_forbidden_host_is_403_and_signed(self):
        self.assertError(self.exchange("GET", "/health", host="evil.com"), 403, "forbidden_host")

    def test_wrong_content_type_is_415(self):
        answer = self.post("/v1/page", self.page_payload(), content_type="text/plain")
        self.assertError(answer, 415, "invalid_request")

    def test_bad_json_is_400(self):
        for body in (b"{not json", b"", b"\xff\xfe", b'{"url": '):
            with self.subTest(body=body):
                answer = self.exchange("POST", "/v1/page", body)
                self.assertError(answer, 400, "invalid_request")

    def test_json_that_is_not_an_object_is_400(self):
        for body in (b"[]", b"[1]", b'"text"', b"5", b"null", b"true"):
            with self.subTest(body=body):
                self.assertError(self.exchange("POST", "/v1/page", body), 400, "invalid_request")

    def test_json_nested_beyond_the_parser_is_400_not_500(self):
        body = b"[" * 60000
        self.assertError(self.exchange("POST", "/v1/page", body), 400, "invalid_request")

    def test_a_declared_length_over_the_limit_is_413(self):
        answer = self.exchange("POST", "/v1/page", b"", length=str(server.MAX_REQUEST_BYTES + 1))
        self.assertError(answer, 413, "invalid_request")
        answer = self.exchange("POST", "/v1/page", b"", length="999999999999")
        self.assertError(answer, 413, "invalid_request")

    def test_a_body_of_exactly_the_limit_is_read(self):
        body = b"{" + b" " * (server.MAX_REQUEST_BYTES - 2) + b"}"
        self.assertEqual(len(body), server.MAX_REQUEST_BYTES)
        answer = self.exchange("POST", "/v1/page", body)
        self.assertNotEqual(answer.status, 413)
        self.assertSigned(answer)
        self.assertEqual(answer.code, "invalid_url")  # {} has no url

    def test_a_post_without_content_length_is_411(self):
        answer = self.exchange("POST", "/v1/page", b"", length=None)
        self.assertError(answer, 411, "invalid_request")

    def test_a_content_length_that_is_not_a_number_is_400(self):
        for value in ("abc", "-1", "+5", "1e3", "0x10", "1 2", ""):
            with self.subTest(value=value):
                answer = self.exchange("POST", "/v1/page", b"", length=value)
                self.assertError(answer, 400, "invalid_request")

    def test_a_content_length_with_very_many_digits_is_413_on_every_python(self):
        for value in ("9" * 19, "9" * 5000, "0" * 5000 + "1" * 19):
            with self.subTest(digits=len(value)):
                answer = self.exchange("POST", "/v1/page", b"", length=value)
                self.assertError(answer, 413, "invalid_request")

    def test_leading_zeros_do_not_make_a_content_length_too_large(self):
        answer = self.exchange("POST", "/v1/page", b"", length="0" * 30)
        self.assertNotEqual(answer.status, 413)

    def test_chunked_transfer_encoding_is_400_for_post_and_get(self):
        answer = self.exchange("POST", "/v1/page", b"", length=None,
                               edit=add("Transfer-Encoding", "chunked"))
        self.assertError(answer, 400, "invalid_request")
        answer = self.exchange("GET", "/health", edit=add("Transfer-Encoding", "chunked"))
        self.assertError(answer, 400, "invalid_request")
        answer = self.exchange("POST", "/v1/page", b"", edit=add("Transfer-Encoding", "identity"))
        self.assertError(answer, 400, "invalid_request")

    def test_an_unknown_field_is_400_invalid_argument(self):
        for path, payload in (
                ("/v1/page", {"url": "https://example.com/", "bogus": 1}),
                ("/v1/page", {"url": "https://example.com/", "method": "GET"}),
                ("/v1/http", {"url": "https://api.tavily.com/search", "method": "GET",
                              "extra": True}),
                ("/v1/http", {"url": "https://api.tavily.com/search", "method": "GET",
                              "max_characters": 5})):
            with self.subTest(path=path, payload=payload):
                self.assertError(self.post(path, payload), 400, "invalid_argument")
        self.assertEqual(self.world.calls, [])

    def test_a_tool_error_is_passed_through_with_its_status_and_code(self):
        answer = self.post("/v1/page", {"url": "https://example.com/", "max_bytes": 0})
        self.assertError(answer, 422, "invalid_argument")
        answer = self.post("/v1/page", {"url": 5})
        self.assertError(answer, 422, "invalid_url")
        answer = self.post("/v1/http", {"url": "https://example.org/", "method": "GET"})
        self.assertError(answer, 403, "forbidden_origin")

    def test_error_answers_are_json_objects_with_message_and_code_only(self):
        answer = self.exchange("GET", "/nope")
        self.assertEqual(set(answer.json), {"error"})
        self.assertEqual(set(answer.json["error"]), {"message", "code"})
        self.assertEqual(answer.headers["content-type"], "application/json")

    def test_errors_before_authentication_are_signed_only_with_a_good_nonce(self):
        good = self.exchange("POST", "/v1/page", b"", length="70000",
                             secret=b"w" * 40)
        self.assertError(good, 413, "invalid_request")
        bad = self.exchange("POST", "/v1/page", b"", length="70000", edit=replace(NONCE, "zz"))
        self.assertError(bad, 413, "invalid_request", signed=False)
        bad = self.exchange("GET", "/health", host="evil.com", edit=drop(NONCE))
        self.assertError(bad, 403, "forbidden_host", signed=False)


# --- Host header ------------------------------------------------------------

class HostTests(LiveServer):
    def test_loopback_names_are_accepted(self):
        for host in ("127.0.0.1", "127.0.0.1:1234", "127.0.0.1:80", "localhost", "LocalHost:80",
                     "LOCALHOST", "[::1]", "[::1]:9100", "[::1]:1"):
            with self.subTest(host=host):
                answer = self.exchange("GET", "/health", host=host)
                self.assertEqual(answer.status, 200)
                self.assertSigned(answer)

    def test_spoofed_hosts_are_403(self):
        for host in ("evil.com", "evil.com:80", "127.0.0.1.evil.com", "127.0.0.1.evil.com:9100",
                     "localhost.evil.com", "evil.localhost.com", "127.0.0.1@evil.com",
                     "evil.com#127.0.0.1", "evil.com:127.0.0.1", "", " ", "0.0.0.0",
                     "127.0.0.2", "[::2]", "[::1", "10.0.0.1", "::1x"):
            with self.subTest(host=host):
                answer = self.exchange("GET", "/health", host=host)
                self.assertError(answer, 403, "forbidden_host")

    def test_a_missing_host_header_is_403(self):
        self.assertError(self.exchange("GET", "/health", host=None), 403, "forbidden_host")

    def test_two_host_headers_do_not_help_a_spoofed_one(self):
        nonce, headers = self.headers_for("GET", "/health", host="evil.com")
        headers.append(("Host", "127.0.0.1"))
        answer = self.send_raw("GET", "/health", headers, nonce=nonce)
        # The first Host header is the one that counts.
        self.assertError(answer, 403, "forbidden_host")

    def test_the_host_is_checked_before_the_signature(self):
        answer = self.exchange("GET", "/health", host="evil.com", secret=b"w" * 40)
        self.assertError(answer, 403, "forbidden_host")

    def test_the_host_check_covers_post_as_well(self):
        answer = self.post("/v1/page", self.page_payload(), host="evil.com")
        self.assertError(answer, 403, "forbidden_host")
        self.assertEqual(self.world.calls, [])

    def test_a_rejected_host_does_not_use_up_the_nonce(self):
        nonce = "4d" * 32
        self.assertError(self.exchange("GET", "/health", host="evil.com", nonce=nonce), 403,
                         "forbidden_host")
        self.assertEqual(server.Handler.nonces.seen, {})
        self.assertEqual(self.exchange("GET", "/health", nonce=nonce).status, 200)

    def test_host_allowed_directly(self):
        for header, expected in ((None, False), ("", False), ("127.0.0.1", True),
                                 ("[::1]:5", True), ("localhost:5", True),
                                 ("127.0.0.1.evil.com", False), ("evil.com", False)):
            self.assertEqual(server.host_allowed(header), expected, header)


# --- Methods and content types ----------------------------------------------

class MethodAndContentTypeTests(LiveServer):
    def test_content_type_application_json_is_required_for_post(self):
        body = json.dumps(self.page_payload()).encode()
        for value in ("application/json", "application/json; charset=utf-8",
                      "application/json;charset=UTF-8", "Application/JSON",
                      "APPLICATION/JSON; charset=utf-8", "application/json ; charset=x"):
            with self.subTest(value=value):
                answer = self.exchange("POST", "/v1/page", body, content_type=value)
                self.assertEqual(answer.status, 200, answer.body)
                self.assertSigned(answer)
        for value in ("", "text/plain", "text/json", "application/jsonx", "application/json-patch+json",
                      "application/x-www-form-urlencoded", "multipart/form-data; boundary=x",
                      "json", ";application/json"):
            with self.subTest(value=value):
                answer = self.exchange("POST", "/v1/page", body, content_type=value)
                self.assertError(answer, 415, "invalid_request")
        answer = self.exchange("POST", "/v1/page", body, content_type=None)
        self.assertError(answer, 415, "invalid_request")
        self.assertEqual(len(self.world.calls), 6)

    def test_get_health_ignores_the_content_type(self):
        for value in (None, "text/plain", "application/json", "garbage", ""):
            with self.subTest(value=value):
                answer = self.exchange("GET", "/health", content_type=value)
                self.assertEqual(answer.status, 200)
                self.assertSigned(answer)

    def test_other_methods_are_405_after_authentication(self):
        expected = {"error": {"message": "method not allowed", "code": "method_not_allowed"}}
        for method in ("PUT", "DELETE", "PATCH", "OPTIONS"):
            for path in ("/health", "/v1/page", "/nope"):
                with self.subTest(method=method, path=path):
                    answer = self.exchange(method, path)
                    self.assertError(answer, 405, "method_not_allowed")
                    self.assertEqual(answer.json, expected)
                    self.assertEqual(answer.headers["content-type"], "application/json")

    def test_head_is_405_with_a_signature_over_the_body_it_does_not_send(self):
        answer = self.exchange("HEAD", "/health")
        self.assertEqual(answer.status, 405)
        self.assertEqual(answer.body, b"")
        body = json.dumps({"error": {"message": "method not allowed",
                                     "code": "method_not_allowed"}}).encode()
        self.assertEqual(int(answer.headers["content-length"]), len(body))
        self.assertSigned(answer, body=body)

    def test_other_methods_without_a_good_signature_are_401_not_405(self):
        for method in ("PUT", "DELETE", "PATCH", "OPTIONS", "HEAD"):
            with self.subTest(method=method):
                answer = self.exchange(method, "/health", secret=b"w" * 40)
                self.assertEqual(answer.status, 401)
                if method != "HEAD":
                    self.assertEqual(answer.code, "unauthorized")
                self.assertIsNotNone(answer.signature)

    def test_other_methods_with_a_body_are_still_405(self):
        answer = self.exchange("PUT", "/v1/page", b'{"url": "https://example.com/"}')
        self.assertError(answer, 405, "method_not_allowed")
        self.assertEqual(self.world.calls, [])

    def test_a_replayed_405_is_a_replay(self):
        nonce, headers = self.headers_for("PUT", "/health")
        self.assertEqual(self.send_raw("PUT", "/health", headers, nonce=nonce).status, 405)
        self.assertError(self.send_raw("PUT", "/health", headers, nonce=nonce), 401,
                         "replayed_request")


# --- Internal errors --------------------------------------------------------

class InternalErrorTests(LiveServer):
    def assertNoTraceback(self, answer):
        text = answer.body.decode()
        for fragment in ("Traceback", "RuntimeError", "boom", "File \"", ".py", "line "):
            self.assertNotIn(fragment, text)

    def test_an_exception_in_a_tool_is_a_signed_500_without_details(self):
        self.world.failure = RuntimeError("boom: /home/sandbox/server.py line 42")
        for path, payload in (
                ("/v1/page", self.page_payload()),
                ("/v1/http", {"url": "https://api.tavily.com/search", "method": "GET"})):
            with self.subTest(path=path):
                answer = self.post(path, payload)
                self.assertError(answer, 500, "internal_error")
                self.assertEqual(answer.json, {"error": {"message": "internal error",
                                                         "code": "internal_error"}})
                self.assertNoTraceback(answer)

    def test_the_log_names_the_exception_type_only(self):
        self.world.failure = RuntimeError("boom: secret detail")
        self.post("/v1/page", self.page_payload())
        text = self.log.getvalue()
        self.assertIn("internal error: RuntimeError", text)
        self.assertIn("POST /v1/page 500", text)
        self.assertNotIn("boom", text)
        self.assertNotIn("Traceback", text)

    def test_a_stub_tool_that_fails_in_other_ways(self):
        class Stub:
            def page(self, *args):
                raise self.error

            http = page

        for error in (ValueError("x"), KeyError("k"), ZeroDivisionError(), OSError("disk"),
                      RecursionError()):
            with self.subTest(error=type(error).__name__):
                stub = Stub()
                stub.error = error
                server.Handler.tools = stub
                answer = self.post("/v1/page", self.page_payload())
                self.assertError(answer, 500, "internal_error")
                self.assertNoTraceback(answer)

    def test_a_tool_error_from_the_tool_keeps_its_status_and_code(self):
        class Stub:
            def page(self, *args):
                raise server.ToolError("teapot here", "custom_code", 418)

            def http(self, *args):
                raise server.ToolError("too fast", "slow_down", 429)

        server.Handler.tools = Stub()
        self.assertError(self.post("/v1/page", self.page_payload()), 418, "custom_code")
        answer = self.post("/v1/http", {"url": "https://api.tavily.com/x", "method": "GET"})
        self.assertError(answer, 429, "slow_down")
        self.assertEqual(answer.json["error"]["message"], "too fast")

    def test_a_tool_error_raised_by_the_transport_is_passed_through(self):
        self.world.failure = server.ToolError("upstream said no", "fetch_failed", 502)
        answer = self.post("/v1/page", self.page_payload())
        self.assertError(answer, 502, "fetch_failed")
        self.assertEqual(answer.json["error"]["message"], "upstream said no")

    def test_the_server_keeps_serving_after_an_internal_error(self):
        self.world.failure = RuntimeError("boom")
        self.assertEqual(self.post("/v1/page", self.page_payload()).status, 500)
        self.world.failure = None
        self.assertEqual(self.post("/v1/page", self.page_payload()).status, 200)


# --- Slow clients -----------------------------------------------------------

class SlowClientTests(LiveServer):
    def talk(self, head, body):
        """Sends `head` and `body`, then nothing, and returns everything the
        server sends until it closes the connection."""
        sock = socket.create_connection(("127.0.0.1", self.port), timeout=5)
        try:
            sock.sendall(head + body)
            data = b""
            while True:
                chunk = sock.recv(65536)
                if not chunk:
                    return data
                data += chunk
        finally:
            sock.close()

    def request_head(self, announced):
        nonce, headers = self.headers_for("POST", "/v1/page", b"", length=str(announced))
        lines = "POST /v1/page HTTP/1.1\r\n" + "".join(f"{n}: {v}\r\n" for n, v in headers)
        return nonce, lines.encode("latin-1") + b"\r\n"

    def parse(self, data):
        head, _, body = data.partition(b"\r\n\r\n")
        lines = head.decode("latin-1").split("\r\n")
        status = int(lines[0].split()[1])
        headers = {k.lower(): v for k, _, v in (line.partition(": ") for line in lines[1:])}
        return status, headers, body

    def test_a_client_that_announces_a_body_and_stops_gets_408_and_is_closed(self):
        with mock.patch.object(server.Handler, "timeout", 0.3):
            nonce, head = self.request_head(50)
            started = time.monotonic()
            data = self.talk(head, b'{"ur')
        self.assertLess(time.monotonic() - started, 4)
        status, headers, body = self.parse(data)
        self.assertEqual(status, 408)
        self.assertEqual(json.loads(body)["error"]["code"], "request_timeout")
        self.assertEqual(headers["x-sandbox-auth"], expected_response_auth(
            SECRET, "POST", "/v1/page", nonce, 408, body))

    def test_a_client_that_sends_the_headers_and_no_body_at_all_gets_408(self):
        with mock.patch.object(server.Handler, "timeout", 0.3):
            nonce, head = self.request_head(10)
            data = self.talk(head, b"")
        status, headers, body = self.parse(data)
        self.assertEqual(status, 408)
        self.assertEqual(headers["x-sandbox-auth"], expected_response_auth(
            SECRET, "POST", "/v1/page", nonce, 408, body))

    def test_a_client_that_stops_in_the_middle_of_the_headers_is_just_closed(self):
        with mock.patch.object(server.Handler, "timeout", 0.3):
            data = self.talk(b"GET /hea", b"")
        self.assertEqual(data, b"")

    def test_a_client_that_connects_and_says_nothing_is_closed(self):
        with mock.patch.object(server.Handler, "timeout", 0.3):
            self.assertEqual(self.talk(b"", b""), b"")

    def test_a_body_shorter_than_content_length_is_408_as_soon_as_the_client_is_done(self):
        # The client half-closes after 4 of 50 bytes: no timeout is involved.
        nonce, head = self.request_head(50)
        sock = socket.create_connection(("127.0.0.1", self.port), timeout=5)
        try:
            sock.sendall(head + b'{"ur')
            sock.shutdown(socket.SHUT_WR)
            data = b""
            while True:
                chunk = sock.recv(65536)
                if not chunk:
                    break
                data += chunk
        finally:
            sock.close()
        status, headers, body = self.parse(data)
        self.assertEqual(status, 408)
        self.assertEqual(json.loads(body)["error"]["code"], "request_timeout")
        self.assertEqual(headers["x-sandbox-auth"], expected_response_auth(
            SECRET, "POST", "/v1/page", nonce, 408, body))
        self.assertEqual(server.Handler.nonces.seen, {})

    def test_the_timeout_clause_catches_both_names_of_the_exception(self):
        # Python before 3.10 raises socket.timeout, later ones TimeoutError.
        source = read_source("server.py")
        self.assertIn("except (TimeoutError, socket.timeout):", source)

    def test_the_default_timeout_is_the_module_constant(self):
        self.assertEqual(server.Handler.timeout, server.REQUEST_TIMEOUT)
        self.assertGreater(server.REQUEST_TIMEOUT, 0)

    def test_a_408_does_not_use_up_the_nonce(self):
        with mock.patch.object(server.Handler, "timeout", 0.3):
            nonce, head = self.request_head(10)
            self.talk(head, b"")
        self.assertEqual(server.Handler.nonces.seen, {})


# --- Whole-request read deadline and connection cap -------------------------

class ReadDeadlineTests(LiveServer):
    def setUp(self):
        super().setUp()
        self.expired = []
        original = server.Handler._read_expired
        expired = self.expired

        def recording(handler):
            expired.append(time.monotonic())
            original(handler)

        # Only the deadline may be the cause: each single wait is far longer.
        for patch in (mock.patch.object(server, "READ_DEADLINE", 0.5),
                      mock.patch.object(server.Handler, "timeout", 30),
                      mock.patch.object(server.Handler, "_read_expired", recording)):
            patch.start()
            self.addCleanup(patch.stop)

    def drip(self, data, pause=0.1, limit=4.0):
        """Sends `data` a byte at a time. Returns the seconds until the server
        closed or answered, and what it sent."""
        import select
        sock = socket.create_connection(("127.0.0.1", self.port), timeout=5)
        started = time.monotonic()
        received = b""
        try:
            for index in range(len(data)):
                if time.monotonic() - started > limit:
                    break
                try:
                    sock.send(data[index:index + 1])
                except OSError:
                    break
                if select.select([sock], [], [], pause)[0]:
                    break
            sock.settimeout(5)
            try:
                while True:
                    chunk = sock.recv(65536)
                    if not chunk:
                        break
                    received += chunk
            except OSError:
                pass
            return time.monotonic() - started, received
        finally:
            sock.close()

    def test_headers_sent_byte_by_byte_are_cut_off_at_the_deadline(self):
        nonce, headers = self.headers_for("GET", "/health")
        data = ("GET /health HTTP/1.1\r\n" + "".join(f"{n}: {v}\r\n" for n, v in headers)
                + "\r\n").encode()
        elapsed, received = self.drip(data)
        self.assertLess(elapsed, 3.0)
        self.assertGreaterEqual(elapsed, 0.4)
        self.assertEqual(len(self.expired), 1)
        # Nothing was served: the request never completed.
        self.assertNotIn(b" 200 ", received.split(b"\r\n", 1)[0])
        self.assertEqual(self.world.calls, [])

    def test_a_partial_request_line_followed_by_silence_is_cut_off(self):
        started = time.monotonic()
        sock = socket.create_connection(("127.0.0.1", self.port), timeout=5)
        try:
            sock.sendall(b"GET /hea")
            data = b""
            while True:
                chunk = sock.recv(65536)
                if not chunk:
                    break
                data += chunk
        finally:
            sock.close()
        self.assertLess(time.monotonic() - started, 3.0)
        self.assertEqual(len(self.expired), 1)
        self.assertNotIn(b" 200 ", data)

    def test_a_body_dripped_slowly_is_cut_off_with_408(self):
        body = json.dumps(self.page_payload()).encode()
        nonce, headers = self.headers_for("POST", "/v1/page", body)
        head = ("POST /v1/page HTTP/1.1\r\n" + "".join(f"{n}: {v}\r\n" for n, v in headers)
                + "\r\n").encode()
        sock = socket.create_connection(("127.0.0.1", self.port), timeout=5)
        started = time.monotonic()
        try:
            sock.sendall(head)
            import select
            for index in range(len(body)):
                try:
                    sock.send(body[index:index + 1])
                except OSError:
                    break
                if select.select([sock], [], [], 0.1)[0]:
                    break
            data = b""
            sock.settimeout(5)
            while True:
                chunk = sock.recv(65536)
                if not chunk:
                    break
                data += chunk
        finally:
            sock.close()
        self.assertLess(time.monotonic() - started, 3.0)
        status = int(data.split(b" ", 2)[1])
        self.assertEqual(status, 408)
        self.assertEqual(self.world.calls, [])

    def test_a_client_that_finishes_in_time_is_not_affected(self):
        answer = self.post("/v1/page", self.page_payload())
        self.assertEqual(answer.status, 200)
        self.assertSigned(answer)
        time.sleep(0.7)  # past the deadline: the timer must have been cancelled
        self.assertEqual(self.expired, [])

    def test_the_timer_does_not_fire_on_a_request_that_is_processed_for_longer(self):
        class Slow:
            def page(self, *args):
                time.sleep(1.0)
                return {"final_url": "https://example.com/", "status": 200,
                        "content_type": "text/plain", "title": "", "text": "late"}

        server.Handler.tools = Slow()
        answer = self.post("/v1/page", self.page_payload(), timeout=10)
        self.assertEqual(answer.status, 200)
        self.assertEqual(answer.json["text"], "late")
        self.assertSigned(answer)
        self.assertEqual(self.expired, [])


class ReadDeadlineSourceTests(unittest.TestCase):
    def test_the_deadline_is_ten_seconds_by_default(self):
        self.assertEqual(server.READ_DEADLINE, 10.0)
        self.assertLess(server.READ_DEADLINE, server.REQUEST_TIMEOUT)


class BoundedServerTests(unittest.TestCase):
    def setUp(self):
        self.entered = threading.Semaphore(0)
        self.release = threading.Event()
        entered, release = self.entered, self.release

        class Holding(server.BaseHTTPRequestHandler):
            """Keeps a connection open until released; says nothing."""
            def handle(self):
                entered.release()
                release.wait(10)
                self.wfile.write(b"served\n")

        self.addCleanup(self.release.set)
        self.httpd = server.BoundedServer(("127.0.0.1", 0), Holding, max_connections=2)
        self.port = self.httpd.server_address[1]
        thread = threading.Thread(target=self.httpd.serve_forever,
                                  kwargs={"poll_interval": 0.01}, daemon=True)
        thread.start()

        def stop():
            self.release.set()
            self.httpd.shutdown()
            self.httpd.server_close()
            thread.join(5)

        self.addCleanup(stop)
        self.sockets = []

    def connect(self):
        sock = socket.create_connection(("127.0.0.1", self.port), timeout=5)
        self.sockets.append(sock)
        self.addCleanup(sock.close)
        return sock

    def test_the_third_concurrent_connection_is_closed_without_an_answer(self):
        first, second = self.connect(), self.connect()
        self.assertTrue(self.entered.acquire(timeout=5))
        self.assertTrue(self.entered.acquire(timeout=5))
        third = self.connect()
        self.assertEqual(third.recv(100), b"")
        self.assertFalse(self.entered.acquire(timeout=0.2))  # never reached the handler
        self.release.set()
        self.assertEqual(first.recv(100), b"served\n")
        self.assertEqual(second.recv(100), b"served\n")

    def test_a_slot_is_freed_when_a_request_ends_and_the_next_connection_is_served(self):
        first, second = self.connect(), self.connect()
        self.assertTrue(self.entered.acquire(timeout=5))
        self.assertTrue(self.entered.acquire(timeout=5))
        self.release.set()
        self.assertEqual(first.recv(100), b"served\n")
        self.assertEqual(second.recv(100), b"served\n")
        deadline = time.monotonic() + 5
        while True:
            fourth = self.connect()
            if fourth.recv(100) == b"served\n":
                break
            self.assertLess(time.monotonic(), deadline, "the slots were not released")
            time.sleep(0.05)

    def test_one_ending_connection_makes_room_while_the_other_still_holds(self):
        release_first = threading.Event()
        order = []
        entered = self.entered

        class Selective(server.BaseHTTPRequestHandler):
            def handle(self):
                order.append(1)
                entered.release()
                if len(order) == 1:
                    release_first.wait(10)
                else:
                    self.release_event.wait(10)
                self.wfile.write(b"served\n")

        Selective.release_event = self.release
        self.httpd.RequestHandlerClass = Selective
        first = self.connect()
        self.connect()
        self.assertTrue(self.entered.acquire(timeout=5))
        self.assertTrue(self.entered.acquire(timeout=5))
        self.assertEqual(self.connect().recv(100), b"")
        release_first.set()
        self.assertEqual(first.recv(100), b"served\n")
        deadline = time.monotonic() + 5
        while True:
            again = self.connect()
            again.settimeout(0.3)
            try:
                if again.recv(100) == b"":
                    raise AssertionError("closed")
            except (TimeoutError, socket.timeout):
                break  # accepted and now waiting in the handler
            except AssertionError:
                self.assertLess(time.monotonic(), deadline, "no slot was freed")
                time.sleep(0.05)
        self.release.set()

    def test_main_builds_a_bounded_server_with_the_default_cap(self):
        self.assertTrue(issubclass(server.BoundedServer, server.ThreadingHTTPServer))
        self.assertEqual(server.MAX_CONNECTIONS, 32)
        self.assertRegex(read_source("server.py"),
                         r'server = BoundedServer\(\("0\.0\.0\.0", PORT\), Handler\)')
        self.assertEqual(server.BoundedServer.__init__.__defaults__, (server.MAX_CONNECTIONS,))
        self.assertTrue(server.BoundedServer.daemon_threads)



# --- Logging ----------------------------------------------------------------

class LoggingTests(LiveServer):
    API_KEY = "tvly-FAKE-API-KEY-8c1f0d5e9a7b4c2d6e3f"
    ODD_HEADER = "odd-header-value-that-must-not-be-logged"
    QUESTION = "a private question about my health"
    LINE = re.compile(r"\A(GET|POST|PUT|HEAD) /\S{0,64} [0-9]{3}\Z")

    def test_only_method_path_and_status_are_logged(self):
        self.world.reply = server.Response(200, {"content-type": "application/json"}, b"{}")
        body = json.dumps({
            "url": "https://api.tavily.com/search", "method": "POST",
            "headers": {"Authorization": "Bearer " + self.API_KEY,
                        "Content-Type": "application/json"},
            "body_base64": base64.b64encode(json.dumps({"query": self.QUESTION}).encode()).decode(),
        }).encode()
        nonce, headers = self.headers_for("POST", "/v1/http", body)
        headers += [("Authorization", "Bearer " + self.API_KEY), ("X-Custom", self.ODD_HEADER),
                    ("Cookie", "session=" + self.ODD_HEADER), ("User-Agent", self.ODD_HEADER)]
        answer = self.send_raw("POST", "/v1/http", headers, body, nonce=nonce)
        self.assertEqual(answer.status, 200, answer.body)
        # The key really went to the transport, so a clean log means something.
        self.assertEqual(self.world.calls[0][2]["Authorization"], "Bearer " + self.API_KEY)
        auth_value = dict(headers)[AUTH]

        # More traffic: refusals of every kind.
        self.exchange("GET", "/health")
        self.exchange("GET", "/health", secret=b"w" * 40)
        self.exchange("GET", "/health", host="evil.com")
        self.exchange("GET", "/nope?api_key=" + self.API_KEY)
        self.exchange("POST", "/v1/page", b"{}", content_type="text/plain")
        self.exchange("POST", "/v1/page", b"{not json " + self.API_KEY.encode())
        self.exchange("PUT", "/health", edit=add("X-Custom", self.ODD_HEADER))

        text = self.log.getvalue()
        lines = text.splitlines()
        self.assertGreaterEqual(len(lines), 8)
        for line in lines:
            self.assertRegex(line, self.LINE)
        self.assertEqual(lines[0], "POST /v1/http 200")
        self.assertIn("GET /nope 404", lines)
        for secret in (self.API_KEY, SECRET.decode(), auth_value, nonce, self.ODD_HEADER,
                       self.QUESTION, "Bearer", "tavily", "Authorization", "Cookie", "api_key",
                       "body_base64", "evil.com", "session="):
            self.assertNotIn(secret, text)
        self.assertNotRegex(text, r"[0-9a-fA-F]{32}")

    def test_the_path_is_cut_at_64_characters_and_loses_its_query(self):
        path = "/" + "a" * 300 + "?token=SECRETQUERY"
        self.exchange("GET", path)
        line = self.log.getvalue().strip()
        self.assertEqual(line, "GET /" + "a" * 63 + " 404")

    def test_a_rejected_request_logs_one_line(self):
        self.exchange("GET", "/health", secret=b"w" * 40)
        self.assertEqual(self.log.getvalue().splitlines(), ["GET /health 401"])


# --- load_secret ------------------------------------------------------------

class LoadSecretTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.directory = directory.name

    def write(self, data, name="secret"):
        path = os.path.join(self.directory, name)
        with open(path, "wb") as handle:
            handle.write(data)
        return path

    def assertRefused(self, path, secret=None):
        with self.assertRaises(SystemExit) as caught:
            server.load_secret(path)
        message = str(caught.exception.code)
        self.assertTrue(message)
        self.assertNotEqual(caught.exception.code, 0)
        if secret:
            self.assertNotIn(secret.decode("latin-1"), message)
        return message

    def test_exactly_32_bytes_is_accepted(self):
        value = b"0123456789abcdef" * 2
        self.assertEqual(server.load_secret(self.write(value)), value)

    def test_one_trailing_newline_is_removed(self):
        value = b"0123456789abcdef" * 2
        self.assertEqual(server.load_secret(self.write(value + b"\n")), value)

    def test_two_trailing_newlines_leave_one(self):
        value = b"0123456789abcdef" * 2
        self.assertEqual(server.load_secret(self.write(value + b"\n\n")), value + b"\n")

    def test_a_carriage_return_is_kept(self):
        value = b"0123456789abcdef" * 2
        self.assertEqual(server.load_secret(self.write(value + b"\r\n")), value + b"\r")

    def test_a_secret_of_binary_bytes_and_inner_newlines_is_kept_as_is(self):
        value = bytes(range(0, 64))
        self.assertEqual(server.load_secret(self.write(value)), value)

    def test_31_bytes_is_refused(self):
        value = b"0123456789abcdef" * 2
        for data in (value[:31], value[:31] + b"\n", b"x"):
            message = self.assertRefused(self.write(data), value[:31])
            self.assertIn("shorter", message)

    def test_32_bytes_followed_by_a_newline_is_32_bytes_not_33(self):
        value = b"s" * 31
        self.assertRefused(self.write(value + b"\n"), value)

    def test_empty_and_newline_only_files_are_refused(self):
        self.assertRefused(self.write(b""))
        self.assertRefused(self.write(b"\n"))

    def test_the_limit_of_4096_bytes(self):
        value = b"k" * server.MAX_SECRET_BYTES
        self.assertEqual(server.load_secret(self.write(value)), value)
        self.assertEqual(server.load_secret(self.write(value + b"\n")), value)
        for data in (value + b"k", value + b"k\n", value * 3):
            message = self.assertRefused(self.write(data), value)
            self.assertIn("longer", message)

    def test_4096_bytes_and_two_newlines_is_refused(self):
        value = b"k" * server.MAX_SECRET_BYTES
        message = self.assertRefused(self.write(value + b"\n\n"), value)
        self.assertIn("longer", message)

    def test_a_missing_file_is_refused(self):
        message = self.assertRefused(os.path.join(self.directory, "missing"))
        self.assertIn("cannot open", message)

    def test_a_directory_is_refused(self):
        message = self.assertRefused(self.directory)
        self.assertIn("not a regular file", message)

    @unittest.skipUnless(hasattr(os, "O_NOFOLLOW"), "needs O_NOFOLLOW")
    def test_a_symbolic_link_is_refused_even_to_a_good_secret(self):
        value = b"0123456789abcdef" * 2
        target = self.write(value)
        link = os.path.join(self.directory, "link")
        os.symlink(target, link)
        message = self.assertRefused(link, value)
        self.assertIn("cannot open", message)

    def test_a_device_that_is_not_a_regular_file_is_refused(self):
        if not os.path.exists("/dev/zero"):
            self.skipTest("no /dev/zero")
        message = self.assertRefused("/dev/zero")
        self.assertIn("not a regular file", message)

    def test_messages_never_contain_the_secret(self):
        value = b"TOPSECRETVALUE-" + b"x" * 16
        for data in (value[:10], value + b"y" * 5000):
            message = self.assertRefused(self.write(data), value)
            self.assertNotIn("TOPSECRET", message)

    def test_the_default_path_is_the_one_the_app_mounts(self):
        self.assertEqual(server.DEFAULT_SECRET_FILE, "/run/tuff-sandbox/secret")


# --- main() -----------------------------------------------------------------

CHILD = (
    "import sys, server\n"
    "class Announcing(server.BoundedServer):\n"
    "    def server_bind(self):\n"
    "        super().server_bind()\n"
    "        sys.stdout.write('PORT %d\\n' % self.server_address[1])\n"
    "        sys.stdout.flush()\n"
    "server.BoundedServer = Announcing\n"
    "server.drop_privileges = lambda: None\n"
    "server.PORT = 0\n"
    "server.main()\n"
)


def child_environment(**extra):
    env = {k: v for k, v in os.environ.items() if not k.startswith("SANDBOX_")}
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env.update(extra)
    return env


class MainProcessTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.secret = b"main-process-secret-0123456789abcdef-xyz"
        self.secret_file = os.path.join(directory.name, "secret")
        with open(self.secret_file, "wb") as handle:
            handle.write(self.secret + b"\n")
        self.directory = directory.name

    def spawn(self, **env):
        process = subprocess.Popen(
            [sys.executable, "-c", CHILD], cwd=HERE, env=child_environment(**env),
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        watchdog = threading.Timer(60, process.kill)
        watchdog.start()

        def cleanup():
            watchdog.cancel()
            if process.poll() is None:
                process.kill()
            process.wait()
            process.stdout.close()
            process.stderr.close()

        self.addCleanup(cleanup)
        return process

    def run_to_exit(self, **env):
        process = self.spawn(**env)
        stdout, stderr = process.communicate(timeout=60)
        return process.returncode, stdout, stderr

    def test_serves_reports_the_blocklist_and_exits_0_on_sigterm(self):
        blocked = "93.184.216.34, 192.168.1.5,fe80::1,::ffff:93.184.216.34,,"
        process = self.spawn(SANDBOX_SECRET_FILE=self.secret_file,
                             SANDBOX_BLOCKED_ADDRESSES=blocked)
        port = int(process.stdout.readline().split()[1])
        self.assertIn("listening", process.stderr.readline())
        status, body, signed = sandbox_client.call(port, self.secret, "GET", "/health", timeout=10)
        self.assertEqual((status, signed), (200, True))
        health = json.loads(body)
        self.assertEqual(health["blocked_addresses"], 3)  # the mapped one is the first
        self.assertEqual((health["status"], health["profile"]), ("ok", "web"))
        # The list reaches the address policy of the real process.
        status, body, signed = sandbox_client.call(
            port, self.secret, "POST", "/v1/page", {"url": "http://93.184.216.34/"}, timeout=10)
        self.assertEqual((status, signed), (403, True))
        self.assertEqual(json.loads(body)["error"]["code"], "private_address")
        # A wrong secret is refused by the real process too.
        status, _, _ = sandbox_client.call(port, b"w" * 40, "GET", "/health", timeout=10)
        self.assertEqual(status, 401)
        process.send_signal(signal.SIGTERM)
        self.assertEqual(process.wait(timeout=20), 0)

    def test_no_blocklist_means_zero_blocked_addresses(self):
        process = self.spawn(SANDBOX_SECRET_FILE=self.secret_file)
        port = int(process.stdout.readline().split()[1])
        status, body, signed = sandbox_client.call(port, self.secret, "GET", "/health", timeout=10)
        self.assertEqual((status, signed, json.loads(body)["blocked_addresses"]), (200, True, 0))
        process.send_signal(signal.SIGTERM)
        self.assertEqual(process.wait(timeout=20), 0)

    def test_a_bad_blocklist_exits_before_listening(self):
        code, stdout, stderr = self.run_to_exit(
            SANDBOX_SECRET_FILE=self.secret_file, SANDBOX_BLOCKED_ADDRESSES="10.0.0.1,notanip")
        self.assertNotEqual(code, 0)
        self.assertIn("SANDBOX_BLOCKED_ADDRESSES", stderr)
        self.assertIn("refusing to start", stderr)
        self.assertNotIn("listening", stderr)
        self.assertEqual(stdout, "")

    def test_a_missing_secret_file_exits_before_listening(self):
        code, stdout, stderr = self.run_to_exit(
            SANDBOX_SECRET_FILE=os.path.join(self.directory, "missing"),
            SANDBOX_BLOCKED_ADDRESSES="10.0.0.1")
        self.assertNotEqual(code, 0)
        self.assertIn("cannot open the secret file", stderr)
        self.assertNotIn("listening", stderr)
        self.assertEqual(stdout, "")

    def test_a_short_secret_exits_before_listening_without_printing_it(self):
        path = os.path.join(self.directory, "short")
        with open(path, "wb") as handle:
            handle.write(b"short-secret-value")
        code, stdout, stderr = self.run_to_exit(SANDBOX_SECRET_FILE=path)
        self.assertNotEqual(code, 0)
        self.assertIn("shorter", stderr)
        self.assertNotIn("short-secret-value", stderr + stdout)
        self.assertNotIn("listening", stderr)


class MainUnitTests(unittest.TestCase):
    """main() with a stand-in server, to see what it does and in which order."""

    def setUp(self):
        self.events = []
        events = self.events

        class FakeServer:
            def __init__(self, address, handler):
                events.append("bind")
                self.address = address
                self.handler = handler
                self.closed = False

            def serve_forever(self):
                events.append("serve")
                raise KeyboardInterrupt

            def server_close(self):
                events.append("close")

        self.FakeServer = FakeServer
        self.created = []
        self.paths = []

        def load_secret(path):
            events.append("secret")
            self.paths.append(path)
            return SECRET

        for patch in (
            mock.patch.object(server, "load_secret", load_secret),
            mock.patch.object(server, "drop_privileges", lambda: events.append("drop")),
            mock.patch.object(server, "stop_on_sigterm", lambda s: events.append("sigterm")),
            mock.patch.object(server, "BoundedServer", self.make_server),
            mock.patch.object(server, "BLOCKED_ADDRESSES", frozenset()),
            mock.patch.object(server.Handler, "secret", b""),
            mock.patch.dict(os.environ, {}),
            mock.patch.object(sys, "stderr", io.StringIO()),
        ):
            patch.start()
            self.addCleanup(patch.stop)
        os.environ.pop("SANDBOX_SECRET_FILE", None)
        os.environ.pop(server.BLOCKED_ENV, None)

    def make_server(self, address, handler):
        instance = self.FakeServer(address, handler)
        self.created.append(instance)
        return instance

    def run_main(self):
        with self.assertRaises(SystemExit) as caught:
            server.main()
        return caught.exception.code

    def test_main_removes_the_blocklist_from_the_environment_and_applies_it(self):
        os.environ[server.BLOCKED_ENV] = "10.1.2.3, ::ffff:10.1.2.3, 2001:db8::7"
        os.environ["SANDBOX_SECRET_FILE"] = "/somewhere/secret"
        self.assertEqual(self.run_main(), 0)
        self.assertNotIn(server.BLOCKED_ENV, os.environ)
        import ipaddress
        self.assertEqual(server.BLOCKED_ADDRESSES, frozenset({
            ipaddress.ip_address("10.1.2.3"), ipaddress.ip_address("2001:db8::7")}))
        self.assertEqual(server.Handler.secret, SECRET)
        self.assertEqual(self.paths, ["/somewhere/secret"])

    def test_the_secret_is_read_first_privileges_are_dropped_before_binding(self):
        os.environ[server.BLOCKED_ENV] = "10.1.2.3"
        self.assertEqual(self.run_main(), 0)
        self.assertEqual(self.events, ["secret", "drop", "bind", "sigterm", "serve", "close"])

    def test_without_a_secret_file_setting_the_app_mount_is_used(self):
        self.assertEqual(self.run_main(), 0)
        self.assertEqual(self.paths, ["/run/tuff-sandbox/secret"])

    def test_the_server_binds_all_addresses_on_the_configured_port(self):
        self.run_main()
        self.assertEqual(self.created[0].address, ("0.0.0.0", server.PORT))
        self.assertIs(self.created[0].handler, server.Handler)

    def test_it_prints_the_listening_line(self):
        self.run_main()
        self.assertIn("listening", sys.stderr.getvalue())

    def test_a_bad_blocklist_stops_before_privileges_are_dropped_or_anything_binds(self):
        for value in ("10.0.0.1,notanip", "fe80::1%eth0", "1.2.3.4/24", "10.0.0.1;8.8.8.8",
                      "localhost", ",".join(f"10.0.{i // 250}.{i % 250 + 1}" for i in range(300))):
            with self.subTest(value=value[:40]):
                del self.events[:]
                os.environ[server.BLOCKED_ENV] = value
                message = str(self.run_main())
                self.assertIn("SANDBOX_BLOCKED_ADDRESSES", message)
                self.assertIn("refusing to start", message)
                self.assertEqual(self.events, ["secret"])
                self.assertEqual(server.BLOCKED_ADDRESSES, frozenset())
                self.assertNotIn(server.BLOCKED_ENV, os.environ)

    def test_a_bad_secret_stops_before_the_blocklist_is_read(self):
        def refuse(path):
            raise SystemExit("the secret is shorter than 32 bytes; refusing to start")

        os.environ[server.BLOCKED_ENV] = "10.0.0.1"
        with mock.patch.object(server, "load_secret", refuse):
            self.assertIn("shorter", str(self.run_main()))
        self.assertEqual(self.events, [])
        self.assertEqual(os.environ[server.BLOCKED_ENV], "10.0.0.1")


class DropPrivilegesTests(unittest.TestCase):
    def test_it_returns_immediately_when_not_root(self):
        def fail(*args, **kwargs):
            raise AssertionError("must not be called")

        with mock.patch.object(server.os, "getuid", return_value=1000), \
                mock.patch.object(server.ctypes, "CDLL", fail), \
                mock.patch.object(server.os, "setuid", fail), \
                mock.patch.object(server.os, "setgid", fail), \
                mock.patch.object(server.os, "setgroups", fail):
            self.assertIsNone(server.drop_privileges())
            self.assertIsNone(server.drop_privileges(uid=12345))

    def test_the_sandbox_user_is_10001(self):
        self.assertEqual(server.SANDBOX_UID, 10001)


if __name__ == "__main__":
    unittest.main()
