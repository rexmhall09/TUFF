#!/usr/bin/env python3
"""Reference client for the web sandbox API, used by the self-test and the tests.

It signs a request the way the app does (see the docstring of server.py), sends
it to 127.0.0.1, and checks the signature of the answer before it prints
anything. The app's Swift client must produce the same bytes; both are checked
against Sandbox/test-vectors/signing.json.

Usage:
  sandbox_client.py --port 49152 --secret-file FILE GET /health
  sandbox_client.py --port 49152 --secret-file FILE POST /v1/page '{"url": "https://example.com/"}'

Prints the answer's status and body (JSON). Exits 0 for a verified answer with
status below 400, 1 for a verified error answer, 2 when the answer is not
signed correctly or the sandbox cannot be reached.
"""
from __future__ import annotations

import argparse
import hashlib
import hmac
import http.client
import json
import os
import sys
import time

REQUEST_LABEL = "tuff-sandbox-req-v1"
RESPONSE_LABEL = "tuff-sandbox-resp-v1"


def _mac(secret: bytes, *parts: str) -> str:
    return hmac.new(secret, "\n".join(parts).encode(), hashlib.sha256).hexdigest()


def request_auth(secret: bytes, method: str, path: str, nonce: str, timestamp: str,
                 body: bytes) -> str:
    return _mac(secret, REQUEST_LABEL, method, path, nonce, timestamp,
                hashlib.sha256(body).hexdigest())


def response_auth(secret: bytes, method: str, path: str, nonce: str, status: int,
                  body: bytes) -> str:
    return _mac(secret, RESPONSE_LABEL, method, path, nonce, str(status),
                hashlib.sha256(body).hexdigest())


def read_secret(path: str) -> bytes:
    with open(path, "rb") as handle:
        return handle.read().removesuffix(b"\n")


def call(port: int, secret: bytes, method: str, path: str, payload: object | None = None,
         timeout: float = 90.0, host: str = "127.0.0.1") -> tuple[int, bytes, bool]:
    """(status, body, signature_ok). `signature_ok` is True only when the answer
    carries the signature the secret, method, path, nonce, status and body
    call for."""
    body = b"" if payload is None else json.dumps(payload).encode()
    nonce = os.urandom(32).hex()
    timestamp = str(int(time.time()))
    headers = {
        "Host": f"127.0.0.1:{port}",
        "X-Sandbox-Nonce": nonce,
        "X-Sandbox-Time": timestamp,
        "X-Sandbox-Auth": request_auth(secret, method, path, nonce, timestamp, body),
    }
    if payload is not None:
        headers["Content-Type"] = "application/json"
    connection = http.client.HTTPConnection(host, port, timeout=timeout)
    try:
        connection.request(method, path, body=body if payload is not None else None,
                           headers=headers)
        response = connection.getresponse()
        answer = response.read()
        given = response.getheader("X-Sandbox-Auth") or ""
        expected = response_auth(secret, method, path, nonce, response.status, answer)
        return response.status, answer, hmac.compare_digest(given.lower(), expected)
    finally:
        connection.close()


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--secret-file", required=True)
    parser.add_argument("method", choices=["GET", "POST"])
    parser.add_argument("path")
    parser.add_argument("body", nargs="?", help="JSON for a POST")
    args = parser.parse_args(argv)
    payload = json.loads(args.body) if args.body else None
    try:
        status, answer, signed = call(args.port, read_secret(args.secret_file), args.method,
                                      args.path, payload)
    except OSError as error:
        sys.stderr.write(f"cannot reach the sandbox: {error}\n")
        return 2
    if not signed:
        sys.stderr.write("the answer is not signed correctly\n")
        return 2
    print(status)
    print(answer.decode("utf-8", "replace"))
    return 0 if status < 400 else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
