# Sandbox

The code that runs inside TUFF's sandbox: a small Linux VM started with
Apple's [`container`](https://github.com/apple/container) tool. When the
sandbox setting is on (Settings > Search; see
[`docs/WEB_SANDBOX.md`](../docs/WEB_SANDBOX.md)), web search, page reads
and PDF reading go through this server instead of TUFF's own process. That is
the `web` profile. The same image, started a second time with
`SANDBOX_PROFILE=files`, is the `files` profile: a VM with no network that
reads the folders picked for local file search (see "The files profile" below).

This folder holds the VM's side. The app's side is in
`Sources/TUFFApp/Core/Sandbox/`.

| File | What it is |
| --- | --- |
| `Containerfile` | The image: pinned base image, Debian packages from a dated snapshot, Python packages pinned by hash. |
| `entrypoint.sh` | Loads the firewall of the profile (`SANDBOX_PROFILE`), refuses to start without it, then starts the server. |
| `firewall-web.nft` | Web profile: public addresses only; the Mac and the local network are refused; one inbound port, from the gateway only. |
| `firewall-files.nft` | Files profile: nothing outbound at all (loopback and replies only); the same single inbound port. |
| `default_gateway.py` | Reads the VM's default gateway for the inbound rule. |
| `server.py` | The API: `GET /health`, `POST /v1/page`, `POST /v1/http` (web) or `POST /v1/files/list`, `POST /v1/files/passages` (files), with signing both ways. Its docstring is the specification. |
| `sandbox_client.py` | A small client that signs a request and checks the answer. Used by the self-test and the tests. |
| `test-vectors/signing.json` | Fixed signature examples. The server, this client and the Swift code must all reproduce them. |
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
blocked Mac addresses and every endpoint, and stops it again; then it does the
same for the files profile (`files-start`, `files-selftest`, `files-stop`,
or `files-all` for that part alone): sample files mounted read-only, nothing
reachable from inside the VM, a symbolic link to `/etc/passwd` neither listed nor
read, a PDF bomb refused. Set `TUFF_SANDBOX_IMAGE` to check a pulled image
instead of building one.

To change the image (a new Python package, a newer base image) and ship it, see
[`docs/SANDBOX_IMAGE.md`](../docs/SANDBOX_IMAGE.md): rebuild, publish, pin the
new digest in the app.

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

## The files profile

`SANDBOX_PROFILE` is `web` (the default) or `files`; any other value, an empty
one included, stops the entrypoint and the server before they do anything. The
files VM has no network: `firewall-files.nft` drops everything outbound except
loopback and replies on the connection the app opened, the app starts it with
`--no-dns`, and `SANDBOX_BLOCKED_ADDRESSES` is not used (it is ignored if
given). The folders the user picked are mounted read-only at
`/files/<folder id>`, where the id is a lower-case UUID. The VM only reads: it
lists files and turns them into passages; TUFF's index stays on the Mac. Same
signing, request limits, connection cap and dropped privileges as the web
profile (privileges are dropped before any file is read).

`/health` answers `{"status", "profile": "files", "api": 1, "pdf",
"blocked_addresses": 0, "folders": [ids found under /files]}`. The two web
paths are unknown here, and the two paths below are unknown in the web profile.

- `POST /v1/files/list` takes `{"folder", "cursor"?, "limit"?}` (`limit` 1 to
  2000, default 1000) and answers `{"entries": [{"path", "size", "mtime"}],
  "next_cursor": path or null, "unreadable": int}`. `unreadable` is how many
  directories on this page could not be listed or opened (permission denied or
  any other error), each counted once; they are skipped, so a folder that is
  only partly readable lists what it can, and the count says it is not
  complete. A directory that the next page walks again is counted there, not on
  both. If the folder's own top level cannot be listed or opened for lack of
  permission, the answer is 403 `folder_unreadable` ("the folder cannot be
  read"); a folder that is not mounted stays 404 `not_found`. Only regular files are listed, depth first, by
  name at every level, files and directories interleaved. Hidden names,
  symbolic links, `.git`, `node_modules`, `build`, `dist` and the other build
  directories listed in `server.py`, anything deeper than 24 levels and files
  without an extension (other than `README*`) are left out. Paging keeps no
  state: `cursor` is the last path of the previous page, and the walk resumes
  after it without reading the directories that sort before it. The server sets
  no cap on the total; the app does. One page does a bounded amount of work: it
  ends after 1 MiB of JSON-encoded entries (paths with non-ASCII characters
  take up to 12 bytes a character), after 50,000 directory entries examined, or
  after 15 seconds, and carries on through `next_cursor`. So a page may hold
  fewer than `limit` entries, even none, with `next_cursor` set; the cursor is
  then the last entry examined, which need not be a listed file (it can be a
  hidden name, a skipped directory or a file without extension), and keep
  following it until it is null. A page always examines at least one new
  name, so its cursor moves on even when the time is already used up.
- `POST /v1/files/passages` takes `{"folder", "path"}` and answers `{"size",
  "mtime", "passages": [{"page", "line_start", "line_end", "text"}]}`. A `.pdf`
  is read page by page with pypdf in a child process (60 seconds, 512 MiB, one
  PDF at a time; at most 64 MiB and 500 pages); anything else is read as strict
  UTF-8 text of at most 2 MiB. Passages of about 1200 characters are cut on line
  boundaries the way the app does it (`AppLocalIndexer.passages`), a single
  longer line is cut into pieces, and a passage can be a little under 2400
  characters long (the cap on its text is 2500). Text passages carry exact
  1-based line numbers of the file (a CRLF ends one line), PDF passages carry
  the page number. CR, VT, FF and U+0085 in the text become a space and the other
  control characters except tab and newline are removed. A PDF with more than
  4,000,000 characters of text in all is refused (`oversized`). `size` and `mtime` are those of the open file and are checked
  again after reading; a file that changed meanwhile is `changed_while_reading`.
- Errors are `{"error": {"message", "code"}}`: `invalid_folder` and
  `invalid_path` (400), `folder_unreadable` (403), `not_found` (404), `not_a_file`, `oversized`,
  `too_many_passages`, `unreadable`, `empty` (422), `changed_while_reading` (409),
  `pdf_busy` (503). Messages never contain file contents.

A path is relative, `/` separated, 1 to 1024 characters, without empty, `.` or
`..` components or control characters. Every component is opened with
`O_NOFOLLOW` relative to the folder's descriptor, so a symbolic link anywhere on
the path, inside the folder or pointing out of it, is refused (`not_a_file`).
`/v1/files/passages` reads any regular file inside a mounted folder that it is
asked for, including ones the listing leaves out (hidden files, files in skipped
directories such as `node_modules`, files deeper than 24 levels, files without
an extension); the listing is a convenience of the app, not an access rule.
A hard link cannot be told from the file it points to: a hard link inside the
picked folder to a file elsewhere on the same disk is read like any other file.
The files VM cannot reach the network, write to the folders, or leave them
through links; it can read every regular file inside a folder it was given,
and the secret it signs with is readable inside it, as in the web profile.

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
