# Updating the sandbox image

For whoever maintains TUFF. The web and file sandboxes
([`WEB_SANDBOX.md`](WEB_SANDBOX.md)) run one Linux image built from
[`Sandbox/`](../Sandbox/README.md). TUFF does not trust a name or a tag: the app
carries the **digest** of the image it will run (`SandboxImagePin.release` in
`Sources/TUFFApp/Core/Sandbox/SandboxImage.swift`), pulls exactly that
content, and refuses an image whose digest differs. So updating the image is
always three steps in this order: build and test it, publish it and read its
digest, then change the digest in the app and release the app.

## What goes into the image

`Sandbox/Containerfile` copies `server.py`, `firewall-web.nft`,
`firewall-files.nft`, `entrypoint.sh` and `default_gateway.py`, and installs the
packages in `requirements.txt`. Everything is pinned, so the same files give the
same image:

| Pinned thing | Where | How to update |
| --- | --- | --- |
| Base image `python:3.12-slim` | the `FROM` line (by digest) | Take the digest Docker Hub shows for the tag now, in the same form as the current line. After the update check the Python patch level printed in the build log (see the comment in the Containerfile about CVE-2025-6069). |
| Debian packages (`nftables`) | `DEBIAN_SNAPSHOT` (a date on snapshot.debian.org) | Use a recent date and keep it together with the base image digest. |
| Python packages | `requirements.txt`, every package with hashes | Edit `requirements.in`, then run the `uv pip compile …` command written at the top of that file. It builds for Linux arm64; the file is not for a Mac's own Python. |
| The server and the firewalls | the files above | Edit them, with their tests (`Sandbox/test_*.py`). |
| The GitHub actions of the workflow | `.github/workflows/sandbox-image.yml`, each `uses:` line (by commit, the release in the comment) | Look up the commit the new release's tag points to and replace both the commit and the comment. The workflow can publish, so do not use a moving tag. |

Rebuild when any of those change. Changes to tests or to `Sandbox/README.md`
do not change the image. Rebuild also when a security advisory touches the base
image, `nftables`, `trafilatura`, `pypdf` or their dependencies; there is no
automatic update.

## 1. Build and test it locally (nothing is published)

On an Apple Silicon Mac with Apple `container` 1.5 or newer:

```sh
python3 Sandbox/test_server.py && python3 Sandbox/test_signing.py && python3 Sandbox/test_infra.py
Scripts/sandbox_selftest.sh all        # builds tuff-sandbox:dev, starts it, checks it, stops it
```

`all` checks both profiles: the firewalls, the missing privileges, every
endpoint, a read-only mount, nothing reachable from the file sandbox. To try
the image in the real app without publishing it, read its digest and start TUFF
with the two developer variables (also described in
[`WEB_SANDBOX.md`](WEB_SANDBOX.md#developers)):

```sh
D=$(container image inspect tuff-sandbox:dev | python3 -c 'import json,sys; print(json.load(sys.stdin)[0]["configuration"]["descriptor"]["digest"])')
env TUFF_SANDBOX_DEV_IMAGE=tuff-sandbox:dev TUFF_SANDBOX_DEV_DIGEST=$D /path/to/TUFF.app/Contents/MacOS/TUFF
```

TUFF then runs the local image, still checking that its digest is exactly `$D`,
and never pulls it. Turn on both switches under Settings > Search and try a
search, a page, a PDF and a folder. Also run the app's own tests
(`Scripts/test.sh --filter Sandbox`).

## 2. Publish it and read the digest

The workflow `.github/workflows/sandbox-image.yml` builds the image on a GitHub
arm64 runner and pushes it to the GitHub Container Registry as
`ghcr.io/<owner>/tuff-sandbox:<tag>` (owner in lower case). It never runs for a
pull request; only someone with write access starts it:

- GitHub > Actions > **Sandbox image** > Run workflow, with a tag such as `2`, or
- push a git tag named `sandbox-v2` (the tag becomes `2`).

When it finishes, the job summary shows **`ghcr.io/<owner>/tuff-sandbox@sha256:…`**.
That `sha256:…` is the digest to pin. It is the digest of one plain manifest
(the workflow turns provenance and SBOM attachments off on purpose, so the
digest names exactly the image TUFF pulls).

The first time, check the package's visibility: the package page on GitHub
(your profile or organization > Packages > `tuff-sandbox`) must be **Public**,
or no user's TUFF can pull it. New packages can start as Private; change it
under Package settings.

Then check the published image the way TUFF will use it:

```sh
IMAGE=ghcr.io/<owner>/tuff-sandbox@sha256:<digest from the summary>
container image pull $IMAGE
container image inspect $IMAGE | python3 -c 'import json,sys; print(json.load(sys.stdin)[0]["configuration"]["descriptor"]["digest"])'
TUFF_SANDBOX_IMAGE=$IMAGE Scripts/sandbox_selftest.sh all
```

The digest printed by `inspect` must equal the one from the workflow. TUFF
compares `configuration.descriptor.digest` of `container image inspect`
(`SandboxImageProvisioner.indexDigest(in:)`). That the two agree for an image
pulled from a registry was designed for but not yet seen when this was
written, because no image had been published; if they differ, stop and fix the
comparison before releasing.

## 3. Pin the digest in the app and release

In `Sources/TUFFApp/Core/Sandbox/SandboxImage.swift`, `SandboxImagePin.release`
holds the registry name (`ghcr.io/<owner>/tuff-sandbox`) and the digest. Put the
new digest there (the name's owner must be the account that published). Until a
real digest is there it is the placeholder (all zeros), and turning the setting
on says "The sandbox image has not been published yet".

Also update the tests that rely on the placeholder: in
`Tests/TUFFApp/Core/Sandbox/SandboxImageTests.swift`,
`theReleasePinIsStillAPlaceholder` and
`thePlaceholderPinMeansNoImageIsPublishedAndNothingRuns` (change them to use an
explicit placeholder pin, `SandboxImagePin.placeholderDigest`, so they keep
testing the "not published" path), and in `SandboxControllerTests.swift`
`failsWhenNoImageHasBeenPublished` (same change). The test that the release pin
is well formed stays. Then:

1. Build and run the unit tests (`Scripts/test.sh --filter Sandbox`).
2. On a Mac with no copy of the image (`container image delete` it, or use a
   fresh user), start the real app without the developer variables, turn the web
   sandbox on, and watch it download and start the image ("Downloading the
   sandbox image", then "Ready"). Ask a question that needs the web.
3. Release the app as usual.

The app that carries the new digest is the first one that runs the new image.
Older releases keep the digest they were built with.

## Keep old images

**Do not delete old tags or versions of the package in the registry while a
TUFF release that pins them is still in use.** Such an app would no longer be able
to download its sandbox image and the sandbox would stop working for it.
The digest is permanent: re-publishing the same tag makes a different digest
and does not change what an old app pulls.

## When it goes wrong

- *"The sandbox image is not the one this version of TUFF expects (wanted
  sha256:…, found sha256:…). It was not used."* The image in the local store
  is not the pinned one. With the developer variables, `TUFF_SANDBOX_DEV_DIGEST`
  is not the digest of `tuff-sandbox:dev` (it can differ by one character:
  recompute it with the command above, and again after every rebuild). With a
  released app, the local copy is wrong: `container image delete` it and let TUFF
  download it again.
- *"The sandbox image has not been published yet."* The pin in the app is still
  the placeholder.
- A build that fails at `apt-get update`: the snapshot date is not available
  yet, or too new; use an earlier one.
- A build that fails at `pip install --require-hashes`: `requirements.txt` was
  not generated for Linux arm64 with the command at the top of `requirements.in`.
