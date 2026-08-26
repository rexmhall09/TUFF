<p align="center">
  <img src="docs/assets/tuff-logo.png" alt="TUFF logo" width="220">
</p>

<h1 align="center">TUFF</h1>

<p align="center">
  <strong>Local language models on Apple Silicon, without treating memory as disposable.</strong><br>
  A native Swift and Metal runtime, model library, chat app, CLI, and loopback server.
</p>

<p align="center">
  <img alt="TUFF 2.0.0" src="https://img.shields.io/badge/TUFF-2.0.0-6F4DFF">
  <img alt="Swift 6.2 or later" src="https://img.shields.io/badge/Swift-6.2%2B-F05138?logo=swift&logoColor=white">
  <img alt="Metal 3.2 or later" src="https://img.shields.io/badge/Metal-3.2%2B-5E5CE6">
  <img alt="macOS 15 or later" src="https://img.shields.io/badge/macOS-15%2B-000000?logo=apple&logoColor=white">
  <a href="LICENSE"><img alt="Apache 2.0 license" src="https://img.shields.io/badge/License-Apache%202.0-2ea44f"></a>
</p>

<p align="center">
  <a href="#run-v2-locally">Run it</a> ·
  <a href="#the-five-model-library">Models</a> ·
  <a href="docs/OPENAI_SERVER.md">Server</a> ·
  <a href="docs/BENCHMARKS.md">Measurements</a> ·
  <a href="docs/SYSTEM_DESIGN.md">System design</a> ·
  <a href="CONTRIBUTING.md">Contribute</a>
</p>

TUFF runs five pinned instruction models through one architecture-aware runtime.
It does not wrap MLX or llama.cpp. The inference engine, Metal kernels,
streaming installer, application, CLI, and server are implemented in this
repository.

Large mixture-of-experts checkpoints stay on SSD. TUFF keeps their shared
weights and runtime state resident, fetches only the routed experts needed for
the current work, and reuses those experts through a bounded cache. That is how
the 61 GiB GPT-OSS 120B installation can produce coherent output on a 16 GB M2
Mac without swapping.

## v2 status

The `v2` branch is a local 2.0.0 candidate. The app, all five model paths,
hardware gating, persistent chat, app-hosted server, settings workspace, and
signed-update client are implemented. The local release package is ad-hoc
signed and not notarized.

The formal five-model benchmark campaign and GitHub release are intentionally
not published yet. Until that final release gate is requested and completed,
build or package v2 from this checkout.

## The five-model library

Every source is pinned to an exact revision and source-index fingerprint. The
installer writes directly into a bounded `.gturbo` directory instead of
staging a second full checkpoint.

| Model | Best fit | Installed text model | Minimum memory | Chat reasoning | Verified add-ons |
| --- | --- | ---: | ---: | --- | --- |
| Gemma 4 E4B IT | Small, dense, and quick to install | ~4.2 GB | 8 GB | Off / On | Text only in v2 |
| Gemma 4 26B-A4B IT | Balanced MoE default | ~14.3 GB | 8 GB | Off / On | Image input |
| Qwen3.6 35B-A3B | Coding, long context, and multimodal work | ~19.5 GB | 8 GB | Off / On, optional thought preservation | Image input |
| GPT-OSS 20B | Harmony reasoning and local tool workflows | ~13.8 GB | 16 GB | Low / Medium / High | None |
| GPT-OSS 120B | Largest local GPT-OSS checkpoint | ~65.4 GB | 16 GB | Low / Medium / High | None |

Gemma 4 E4B is text-only for this release. TUFF does not show unverified
capabilities as fake products. Gemma 4 26B and Qwen image input appear as
separate add-ons in the Models screen and require an M2 or newer Mac.

### Hardware eligibility

TUFF reads physical unified memory through `hw.memsize`. The shared model
registry combines that value with the selected context, KV layout, expert-cache
slots, and a system reserve. A model that is unsafe on the current Mac remains
visible, but Download and Load are disabled with a concrete reason. Disk-space
eligibility is checked separately.

The launch gates are based on real model runs, not parameter-count estimates.
GPT-OSS 120B is enabled on 16 GB Macs after a 4K-context M2 smoke reached EOS at
a 7,990,582,952-byte peak footprint with zero swaps. An unqualified future
checkpoint fails closed on every memory tier until it passes a real run.

## The Mac app

The v2 application uses a native four-destination `NavigationSplitView`:

- **Chat** keeps persistent named conversations, durable managed attachments,
  model binding, rename and delete actions, model-supported reasoning controls,
  and a collapsed Thinking section. Changing the model after a conversation
  has messages starts a new chat.
- **Models** puts five independently installed text models on the left and
  verified add-ons on the right. It supports download, resume, cancel, discard,
  repair, activation, and removal flows.
- **Server** starts the OpenAI-compatible HTTP layer inside the app. Chat and
  API requests share the same decode service and are serialized through one
  inference broker.
- **Settings** separates general behavior, per-model profiles, and advanced
  runtime controls. Context, temperature, sampling, expert cache, prefill,
  RDADVISE, and model-specific reasoning defaults live here.

The interface uses TUFF Violet (`#6F4DFF`), a transparent vector mark, and a
layered graphite and violet Mac icon. It supports keyboard navigation,
VoiceOver labels, reduced transparency, light and dark appearance, and compact
window layouts.

## Run v2 locally

Requirements:

- Apple Silicon Mac
- macOS 15 or newer
- Swift 6.2 or newer
- Metal 3.2 or newer
- Enough free disk for the selected model and a 1 GiB installation reserve

Build and run the clone-style app:

```bash
swift build -c release
.build/release/TUFF
```

Build the complete local 2.0.0 application, embedded Sparkle framework, archive,
and checksum:

```bash
Scripts/package_app.sh 2.0.0
open dist/TUFF.app
```

The package script creates:

- `dist/TUFF.app`
- `dist/TUFF-v2.0.0-macos-arm64.zip`
- `dist/TUFF-v2.0.0-macos-arm64.zip.sha256`

It also verifies arm64 executables, resources, signatures, the embedded update
framework, ZIP extraction, and the extracted app signature. The local package
is ad-hoc signed. It is not notarized and may require Control-clicking **Open**
or using **Open Anyway** in Privacy & Security.

When run from this checkout, the app uses these model locations:

| Selector | Local directory |
| --- | --- |
| `gemma4-e4b` | `scratch/gemma4-e4b.gturbo` |
| `gemma4` | `scratch/gemma4.gturbo` |
| `qwen36` | `scratch/qwen36.gturbo` |
| `gpt-oss-20b` | `scratch/gpt-oss-20b.gturbo` |
| `gpt-oss-120b` | `scratch/gpt-oss-120b.gturbo` |

Standalone packaged builds use Application Support. Existing compatible v1
Gemma and Qwen `.gturbo` installations remain readable and are not duplicated
or migrated.

## Signed updates

Packaged v2 builds use Sparkle to check TUFF's GitHub Releases appcast. The app
embeds an EdDSA public key and refuses an insecure feed, invalid key, unsigned
archive, or mismatched signature. General Settings exposes automatic checks,
optional automatic download and installation, and a manual **Check for Updates
Now** action.

Clone-style builds without an embedded signing key keep working, but their
updater stays disabled. A packaged build only offers a new binary after a
matching signed appcast has been published. No v2 appcast has been published
yet.

## Command-line interface

Install any pinned checkpoint with its stable selector:

```bash
swift run -c release TUFFRepack \
  --model gpt-oss-20b \
  --output scratch/gpt-oss-20b.gturbo
```

The selectors are `gemma4-e4b`, `gemma4`, `qwen36`, `gpt-oss-20b`, and
`gpt-oss-120b`. The aliases `e4b` and `gpt-oss` are also accepted. Interrupted
downloads retain verified ranges and can be continued with `--resume`. Use
`--discard-partial` to remove saved download state.

Run Harmony chat with GPT-OSS:

```bash
swift run -c release TUFFCLI \
  --model scratch/gpt-oss-20b.gturbo \
  --chat-prompt "Explain why bounded expert streaming matters." \
  --reasoning low \
  --max-new 256
```

Gemma and Qwen use `--thinking on|off`. GPT-OSS uses
`--reasoning low|medium|high`. All prompt dialects are rendered centrally, and
structured analysis is kept out of visible assistant output.

For multi-turn input, pass a JSON message array with `--messages-file`. Run
`swift run -c release TUFFCLI --help` for sampling, stop, context, cache,
prefill, RDADVISE, and image options.

### Image input

Gemma 4 26B and Qwen use an optional `<name>.vision.gturbo` companion beside
the text model. The app manages these in the right Models column. The CLI can
install one directly:

```bash
swift run -c release TUFFRepack \
  --vision-output scratch/gemma4.vision.gturbo \
  --text-model scratch/gemma4.gturbo
```

Then send one or more images:

```bash
swift run -c release TUFFCLI \
  --model scratch/gemma4.gturbo \
  --chat-prompt "What is in this image?" \
  --image photo.jpg
```

Image input is fail-closed. If the companion is missing, corrupt, the wrong
family, or unsupported by the current hardware, TUFF rejects the image instead
of silently answering a text-only prompt.

## Local OpenAI-compatible server

The safest way to run the server is through the app's Server screen after the
selected model is loaded. It binds only to `127.0.0.1`, shows health and queue
state, and cannot launch a second model process.

The standalone server remains available for headless use:

```bash
swift build -c release --product TUFFServer
.build/release/TUFFServer \
  --model scratch/gemma4.gturbo \
  --port 8080
```

Do not run the app, CLI, standalone server, or a model-using test at the same
time. The server has no authentication or TLS and must not be proxied, tunneled,
or exposed beyond loopback.

Supported endpoints:

- `GET /health`
- `GET /v1/models`
- `POST /v1/chat/completions`

Chat Completions supports JSON, streaming SSE, images for installed compatible
companions, function-tool declarations, prompt-prefix reuse, and model-aware
reasoning controls. The client remains responsible for authorizing and
executing every tool call. See the [server guide](docs/OPENAI_SERVER.md) for
the supported request surface and client examples.

## Architecture

TUFF keeps product behavior in one Foundation-only model registry shared by
the app, installer, CLI, and server. A model descriptor owns the pinned source,
fingerprint, install path, storage, hardware requirements, architecture,
capabilities, add-ons, memory profile, and runtime defaults.

The runtime separates checkpoint identity from architecture behavior:

- dense and mixture-of-experts feed-forward profiles
- Gemma, Qwen ChatML, and GPT-OSS Harmony prompt dialects
- MLX affine INT4 and GPT-OSS MXFP4 weight layouts
- model-specific attention, routing, RoPE, KV, and memory plans
- FP32 GPT-OSS residuals with FP16 projection, attention, expert, and KV paths
- bounded routed-expert reads and bounded chunked prefill scratch

The `.gturbo` major version remains v1. A backward-compatible minor extension
describes dense feed-forward and MXFP4 layouts. Older runtimes reject unfamiliar
feature flags cleanly, while current runtimes still read existing v1 Gemma and
Qwen installations.

[System design](docs/SYSTEM_DESIGN.md) covers file layout, memory ownership,
Metal kernels, prefill, expert streaming, prompt reuse, image companions, and
correctness invariants. The [experiment record](docs/experiments/EXPERIMENT_INVENTORY.md)
keeps the measured optimization history.

## Qualification and measurements

No model is enabled solely because its architecture compiles. Each family has
CPU-reference kernel tests, toy forward and prefill tests, repack validation,
tokenizer and template goldens, reference comparisons, and a coherent live
output check.

Current evidence includes:

| Model | Host evidence | Recorded peak footprint |
| --- | --- | ---: |
| Gemma 4 E4B | 16 GB M2, 8K qualification | 1.71 GiB |
| Gemma 4 26B-A4B | 8 GB M2 production runs | ~2.0 GiB |
| Qwen3.6 35B-A3B | 24 GB M5 production runs | ~1.45 GiB |
| GPT-OSS 20B | 16 GB M2, 4K CLI and server qualification | 5.11 GiB |
| GPT-OSS 120B | 16 GB M2, 4K coherent EOS smoke, zero swaps | 7.44 GiB |

These are hardware qualifications and previously recorded measurements, not a
completed v2 five-model benchmark matrix. The final fixed-prompt, three-process
median campaign is still deferred. Read [Benchmarks](docs/BENCHMARKS.md) for
exact commands, settings, output conditions, and protocol limitations. Use the
[community protocol](docs/COMMUNITY_BENCHMARKS.md) for comparable submissions.

## Develop and test

Run package tests only through the canonical serial runner:

```bash
Scripts/test.sh
```

Run the rest of the local release gate:

```bash
swift build -c release
ruby Scripts/check_markdown_links.rb
ruby Scripts/check_brand_assets.rb
ruby Scripts/check_app_version.rb
```

Real-model work requires macOS 15+, Swift 6.2+, enough free disk, acceptable
`memory_pressure -Q`, a complete verified model, and no competing model process.
Run only one app, CLI, server, or model-using test at a time. Never download a
second full checkpoint or duplicate an installed `.gturbo` model just to test.

Contributions of every kind are welcome. See [CONTRIBUTING.md](CONTRIBUTING.md)
for code, design, docs, model, test, benchmark, and AI-assisted contribution
guidelines.

## How I build TUFF

I use AI models heavily while developing TUFF. They help me explore designs,
write and revise code, build tests, review changes, and edit documentation. The
direction, tradeoffs, acceptance decisions, and responsibility for the project
are still mine. AI assistance is part of the process, not a substitute for
authorship or review.

AI-written code does not get a separate standard. It must be understood,
reviewed, tested, and attributed like any other contribution.

## License and model terms

TUFF source and documentation are licensed under the
[Apache License 2.0](LICENSE). Model weights are not included. The installer
downloads them from pinned source repositories, and those weights remain under
their original terms. See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

TUFF is an independent project and is not affiliated with, sponsored by, or
endorsed by Google, Alibaba, or OpenAI.

## Origin and credit

TUFF began as a fork of
[drumih/turbo-fieldfare](https://github.com/drumih/turbo-fieldfare) by Andrey
Mikhaylov. His project established the original Swift and Metal Gemma runtime,
bounded expert streaming, installer, and much of the foundation TUFF still
builds on.

TUFF is now evolving as its own project with a shared five-model platform,
dense and MXFP4 runtimes, multiple prompt dialects, persistent chat, model and
add-on management, an app-hosted server, and signed updates. The internal
`TurboFieldfare*` module names remain for source continuity and proper history.
