# Web sandbox

By default TUFF fetches search results and web pages itself and reads them
(HTML, PDF) in its own process. The web sandbox moves that work into a small
Linux virtual machine, so a hostile page or PDF meets the VM's limits and not
your Mac's files. It is **off by default**.

## Turn it on

**Settings > Search > Run web search and page reads in a sandbox.** The row is
shown on macOS 26 and newer. It needs Apple's
[`container`](https://github.com/apple/container/releases) tool (version 1.5 or
newer), installed at `/usr/local/bin/container`. If it is missing, the status
under the switch says so and links to Apple's releases page.

TUFF starts the sandbox when you turn the switch on and when the app opens with
it on. TUFF stops it after 10 minutes without web use; the next message that
uses the web starts it again, and a web request that finds it stopped waits
for that start. A sandbox that failed to start or stopped answering stays
failed, with the reason shown under the switch, and is tried again at your
next message. It stops when you turn the switch off and when TUFF quits. The
first start can download the image.

Until the maintainers publish the sandbox image, turning the switch on ends
with "The sandbox image has not been published yet". Developers can run a
locally built image (see below).

The switch applies from the next message. An answer that is already running
keeps using the sandbox; if you turn the switch off meanwhile, its remaining
web requests fail with a message instead of going out directly.

## What changes

With the switch on, every web search and every page the model reads goes
through the sandbox:

- Searches (DuckDuckGo, Brave, Tavily) are sent by the sandbox, which reaches
  only those three sites. TUFF still reads the result list.
- Pages and PDFs are fetched and turned into text inside the sandbox. TUFF
  receives only the text and the title, shortened to 200,000 characters, with
  control and invisible characters removed again.
- There is **no fallback**. If the sandbox is not ready, web search and page
  reads fail with a message and nothing is fetched directly. Turn the switch
  off to use the normal path.

Every request is signed with a secret made at each start, and an answer without
the matching signature is thrown away, so another program on the Mac cannot
pass for the sandbox. Local file search does not use the sandbox.

## What it protects, and what it does not

It protects against a page or PDF that tries to exploit the parser, to fill
memory or disk, or to reach other machines: the VM has a read-only filesystem,
runs without privileges, and its firewall refuses private ranges and this
Mac's own addresses (TUFF passes them in at each start).

It does not hide your searches or which sites you read: they still leave
through your connection. **DNS lookups go through your Mac**, so your Mac sees
which names are looked up. **API keys pass through the sandbox**: with Brave
or Tavily, the key goes from TUFF to the sandbox and from there only to that
provider; TUFF asks the sandbox to prove it is still the running one before it
sends a key. For every other request it does the same when the sandbox has not
given a signed answer for 5 seconds, so after a quiet time, or after the
sandbox has died and another program has taken its port, even a search query
is not sent before that check. Other machines on your network and addresses your Mac gains after
the start are not covered by the VM's block list. See
[`Sandbox/README.md`](../Sandbox/README.md) for the full list.

File access for the sandbox comes with a later change; this one has none.

## Developers

Before an image is published, run a locally built one: build it with
`Scripts/sandbox_selftest.sh build`, then start TUFF with
`TUFF_SANDBOX_DEV_IMAGE=tuff-sandbox:dev` and `TUFF_SANDBOX_DEV_DIGEST` set to
the image's index digest (`container image inspect tuff-sandbox:dev`, the
`configuration.descriptor.digest` field). TUFF checks that the local image has
exactly that digest and never pulls it.

How to publish a new image and change the digest the app pins:
[`SANDBOX_IMAGE.md`](SANDBOX_IMAGE.md).
