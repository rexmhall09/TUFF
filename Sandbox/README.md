# Web sandbox

The code that runs inside TUFF's web sandbox: a small Linux VM started with
Apple's [`container`](https://github.com/apple/container) tool. When the
sandbox setting is on (it is not part of the app yet), web search, page reads
and PDF reading go through this server instead of TUFF's own process. It is
the `web` profile; the folder search profile follows with its first caller.

This folder holds only the VM's side. It has no app change.

| File | What it is |
| --- | --- |
| `Containerfile` | The image: pinned base image, Debian packages from a dated snapshot, Python packages pinned by hash. |
| `entrypoint.sh` | Loads the firewall, refuses to start without it, then starts the server. |
| `firewall-web.nft` | Public addresses only; the Mac and the local network are refused; one inbound port, from the gateway only. |
| `default_gateway.py` | Reads the VM's default gateway for the inbound rule. |
| `server.py` | The API: `GET /health`, `POST /v1/page`, `POST /v1/http`, with signing both ways. Its docstring is the specification. |
| `sandbox_client.py` | A small client that signs a request and checks the answer. Used by the self-test and the tests. |
| `test-vectors/signing.json` | Fixed signature examples. The server, this client and (from the app change on) the Swift code must all reproduce them. |
| `requirements.in`, `requirements.txt` | Direct packages, and every package pinned with hashes. |
| `test_server.py`, `test_signing.py`, `test_infra.py` | Python tests. |
| `selftest_pdf_bomb.py` | Run inside the VM by the self-test: a decompression-bomb PDF must end only its child process. |

## Try it

Tests need nothing but `python3`. Tests that need `pypdf` or `trafilatura` skip
when they are missing; in a virtual environment,
`pip install trafilatura==2.3.0 pypdf==6.20.0` runs them all. (Do not use
`requirements.txt` for that on a Mac: its hashes are for the image's Linux
platform.)

```sh
python3 Sandbox/test_server.py
python3 Sandbox/test_signing.py
python3 Sandbox/test_infra.py
```

On a Mac with Apple `container` 1.5 or newer, `Scripts/sandbox_selftest.sh all`
builds the image, starts it, checks the firewall, the missing privileges, the
blocked Mac addresses and every endpoint, and stops it again. Set
`TUFF_SANDBOX_IMAGE` to check a pulled image instead of building one.

## What the server promises

- Every request is signed (`X-Sandbox-Nonce`, `X-Sandbox-Time`, `X-Sandbox-Auth`)
  with a secret that only the app and the VM know, and every answer is signed
  back, so the app can tell the sandbox from another program on the port. A
  request more than 60 seconds old, or one whose nonce was seen, is refused.
- `/v1/page` answers finished text only. HTML and PDFs are read in a child
  process with time (and, for a PDF, memory) limits.
- `/v1/http` reaches `html.duckduckgo.com`, `api.search.brave.com` and
  `api.tavily.com` over https, with a short header allow-list; a key header goes
  to its own provider only.
  The request is `{"url", "method": "GET" or "POST", "headers"?, "body_base64"?,
  "max_bytes"?, "timeout_seconds"?}`; a POST body goes in `body_base64` (base64
  of the bytes; the whole request, headers' values and URL included, may be at
  most 64 KiB, which leaves a little under 48 KiB of decoded body), and a field named `body` is refused as
  unknown. The answer is `{final_url, status, content_type, body_base64}`.
- Both refuse private, local and metadata addresses, and the Mac's own
  addresses (`SANDBOX_BLOCKED_ADDRESSES`), in the server and again in the
  VM's firewall.

## What it does not cover

DNS goes through the Mac, so the Mac sees which names are looked up. Search
keys pass through the VM inside the one request that needs them. The Mac's own
addresses are blocked one by one: other machines on the Mac's network, and
addresses the Mac gains after the VM started (for example temporary IPv6
addresses, or the address of the VM network's own bridge, which exists only while a
container runs), are not covered by that list (the bridge's are private ranges,
refused anyway); the VM's firewall refuses private
ranges, but a LAN host with a global IPv6 address is reachable if the VM has
IPv6 at all. The server serves 32 connections at once and cuts a request that
has not arrived in full after 10 seconds, so a local program cannot exhaust its
process limit; one that keeps opening connections can still occupy all 32 and
make the server refuse the app's requests for as long as it does that. The secret is readable by every user inside
the VM (see the note in `server.py`). A program
running as the same Mac user could read the secret, but it could read the user's
files directly anyway.
