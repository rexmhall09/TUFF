<p align="center">
  <img src="Sources/TUFFApp/Mac/Resources/tuff-app-icon.png" alt="TUFF app icon" width="170">
</p>

<h1 align="center">TUFF</h1>

<p align="center">
  <strong>GPT-OSS 120B on a 16 GB Mac. Qwen3.6 35B-A3B down to an 8 GB Mac.</strong><br>
  Run large language models locally on Apple Silicon.<br>
  Native Swift, Metal, and an SSD-aware runtime that does not pretend memory is unlimited.
</p>

<p align="center">
  <a href="https://github.com/rexmhall09/TUFF/releases/latest"><strong>Download TUFF</strong></a>
  · <a href="https://rexmhall09.github.io/TUFF/">Website</a>
  · <a href="CONTRIBUTING.md">Contribute</a>
</p>

![TUFF running GPT-OSS 120B on a 16 GB Mac](docs/assets/tuff-v3-chat.png)

## What TUFF is

TUFF is a Mac app and inference engine for running a small set of models really
well on Apple Silicon. It includes the chat app, model downloader, Metal
runtime, command-line tools, and a local OpenAI-compatible server. It does not
wrap MLX or llama.cpp.

The project started with one unusual idea: large mixture-of-experts models do
not need every expert in memory at once. TUFF keeps the shared parts of a model
resident, reads the experts it needs from SSD, and reuses them through a bounded
cache. That is how I can run the 61 GiB GPT-OSS 120B checkpoint on my 16 GB M2
MacBook Air without swap.

Version 3 is a much bigger project than the original app. It has seven models,
persistent and continuous chats, Markdown and native LaTeX rendering on both
sides of the conversation, image and file attachments, per-model settings, a
shared local server, hardware eligibility checks, and signed binary updates.

## Why TUFF

Most local runners load a whole model into memory, so the largest model you can
run is the one that fits. TUFF keeps only the shared weights resident and streams
mixture-of-experts weights from SSD through a bounded cache, so model size is
governed by disk rather than by RAM.

That trade is not free. Streaming costs decode speed, and a model that fits
comfortably in memory will be faster in a runner built around that assumption.
TUFF is aimed at the case where the model does not fit at all.

| | TUFF | LM Studio | Ollama | Colibri | turbo-fieldfare |
| --- | :-: | :-: | :-: | :-: | :-: |
| Runs models bigger than your memory | ✅ | ❌ | ❌ | ✅ | ✅ |
| Built specifically for Apple Silicon | ✅ | ◐ | ◐ | ❌ | ✅ |
| Open source | ✅ | ◐ | ✅ | ✅ | ✅ |
| Desktop app rather than a CLI | ✅ | ✅ | ❌ | ❌ | ✅ |
| Local OpenAI-compatible server | ✅ | ✅ | ✅ | ❌ | ✅ |
| Image input | ✅ | ✅ | ✅ | ❌ | ◐ |
| Signed automatic updates | ✅ | ✅ | ✅ | ❌ | ❌ |
| Good on models that already fit in RAM | ◐ | ✅ | ✅ | ❌ | ◐ |
| Large model library | ◐ | ✅ | ✅ | ❌ | ❌ |
| Windows and Linux | ❌ | ✅ | ✅ | ✅ | ❌ |

✅ yes · ◐ partly · ❌ no

TUFF ships seven models rather than a library, but each one is pinned to a
revision, checksum-verified on install, and qualified on real hardware before it
appears. LM Studio and Ollama reach Apple Silicon through MLX and Metal backends;
TUFF and turbo-fieldfare are written for it and run nowhere else. LM Studio's
desktop app is proprietary, while its CLI and SDKs are MIT.

TUFF is a fork of [drumih/turbo-fieldfare](https://github.com/drumih/turbo-fieldfare),
which established the expert-streaming runtime for one Gemma checkpoint. v2 turned
that into a five-model platform with a rebuilt app, a second quantization format,
a dense architecture path, and a shared local server.

Pick something else if you want a large model library, Windows or Linux, or the
fastest possible decode for a model that already fits in your memory. TUFF is
for the case where the model does not fit at all, on a Mac, in an app.

## Download the app

TUFF requires an Apple Silicon Mac running macOS 15 or newer.

1. Download `TUFF-vX.Y.Z-macos-arm64.zip` from the
   [latest GitHub release](https://github.com/rexmhall09/TUFF/releases/latest).
2. Open the ZIP and move `TUFF.app` into Applications.
3. Open TUFF, choose Models, and download the model you want.

### About the security warning

TUFF is ad-hoc signed and is not notarized, because notarizing requires a paid
Apple Developer account. macOS will therefore warn you that it cannot verify the
developer, and may refuse to open the app on first launch.

To open it anyway, Control-click TUFF and choose **Open**, then confirm **Open**
again. You can also allow it from **System Settings > Privacy & Security** right
after the first attempt.

If you would rather not click through that warning, don't. Two better options:

- **Read the source.** Every line of this app is in this repository, including
  the packaging script that produces the exact release archive.
- **Build it yourself.** A build you produce locally is signed by your own
  machine and opens without any warning. See
  [Build it yourself](#build-it-yourself); it is one command once you have Xcode
  and Swift installed.

You can also verify that the archive you downloaded is the one published, by
comparing it against the `.sha256` file attached to the same release:

```bash
shasum -a 256 TUFF-vX.Y.Z-macos-arm64.zip
```

TUFF checks for signed updates automatically. Automatic download and
installation are on by default and can be changed in Settings. The updater only
accepts archives signed by TUFF's embedded EdDSA key.

## The app

The interface has four places:

- **Chat** keeps named conversations, restores them after a restart, and binds
  each conversation to its model. Return sends by default. If the selected
  model is installed but unloaded, sending a message loads it and continues
  automatically. The conversation is one continuous scroll: images, files and
  reasoning belong to the message they were sent with, and every answer is
  labelled with the model that produced it, so a chat that switched models
  mid-way still says which answer came from where. The model picker in the
  message box offers the models that are on this Mac; downloads live in Models.
- **Models** shows all eight checkpoints and their real disk and memory
  requirements. Image support lives inside the model card as a separate,
  optional download.
- **Server** runs an OpenAI-compatible endpoint on `127.0.0.1`. Chat and Server
  share one decode service, so TUFF never starts a second copy of the model.
- **Settings** keeps simple chat behavior separate from per-model context,
  sampling, cache, prefill, and memory controls. Per-model memory defaults to
  Auto; turning it off restores that model's saved manual choices.

Both sides of the conversation render headings, lists, links, quotes, code
blocks, inline code, emphasis, and LaTeX math such as `$a^2 + b^2 = c^2$`. Math
is typeset locally. Nothing is sent to a web renderer.

Text, Markdown, CSV, JSON and PDF attach as files rather than as pasted text.
The extracted text is what the model reads — every model reads text — but the
composer and the transcript show the file, with its name and an estimate of how
much of the context it uses. A file stays with its message, so a follow-up
question can still refer to it.

## Models

Every model source is pinned to an exact revision and fingerprint. Downloads go
straight into the final `.gturbo` installation, so TUFF does not stage a second
copy of a checkpoint first.

| Model | What I would use it for | Installed size | Minimum memory | Add-ons |
| --- | --- | ---: | ---: | --- |
| Gemma 4 E2B IT | The smallest model here | 2.64 GB | 8 GB | Image input |
| Gemma 4 E4B IT | A small and quick general model | 4.23 GB | 8 GB | Image input |
| Gemma 4 12B IT QAT | Dense quality without a mixture of experts | 10.98 GB | 16 GB | Image input |
| Gemma 4 26B-A4B IT | The balanced default | 14.29 GB | 8 GB | Image input |
| Qwen3.6 35B-A3B | Coding, long context, and images | 19.55 GB | 8 GB | Image input |
| GPT-OSS 20B | Reasoning and local tool workflows | 13.79 GB | 16 GB | None |
| GPT-OSS 120B | The largest and highest-quality option | 65.4 GB | 16 GB | None |
| MiniMax M2.7 4-bit | Always-thinking, file-backed 229B MoE | 128.71 GB | 16 GB | None |

Gemma and Qwen expose thinking on or off. GPT-OSS exposes low, medium, and high
reasoning. Qwen can also preserve thinking between turns from its advanced
profile. MiniMax M2.7 always reasons, so TUFF labels it as always on and keeps
its thought channel separate from the visible answer instead of presenting a
switch the checkpoint cannot honor.

TUFF reads physical unified memory with `hw.memsize` and plans against 75% of
it, leaving the rest to macOS and other applications. It then checks the
selected model, its context length, KV layout, and cache slots against that
budget. Models a Mac does not meet the requirements for stay visible but gray,
with Download and Load disabled and an explanation. Disk space is checked
separately.

Auto has three profiles, because the two things spare memory can buy are not
worth the same. Resident expert slots cut SSD reads during generation; context
tokens buy a longer conversation and make nothing faster.

| Profile | Context | Rest of the budget |
| --- | --- | --- |
| Speed | The checkpoint's qualified length | Resident experts |
| Balanced | Up to twice that length | Resident experts |
| Context | The longest that fits | Resident experts |

Balanced is the default for every model, including profiles saved by an
earlier build. The profiles differ in context and nothing else,
because context is the only thing measurement says is worth buying. On Gemma 4
26B-A4B, on a 16 GB Mac: 8K context with 16 cache slots gave 7.69 tok/s at
1.93 GB peak, 16K context with 16 slots gave 7.67 at 1.93 GB, and 8K context
with 128 slots gave 6.28 at 2.99 GB. Doubling the context is free; filling the
budget with expert slots costs throughput and a gigabyte of memory, because
each slot is its own Metal buffer and a command buffer referencing hundreds of
them pays for all of them.

So Auto keeps each checkpoint's qualified cache size, raising it only to the
sixteen slots chunked prefill needs. v4.0.0 and v4.0.1 spent spare memory on
slots instead, which is why this is worded as a correction. Raising the cache
by hand stays available for anyone who would rather trade throughput for fewer
SSD reads. Each profile is labelled in Settings with the context it would
actually resolve on this Mac. The manual context, cache, and prefill controls
stay visible but disabled until Auto is turned off, and turning it off restores
that model's saved choices.

### Running past the limits

**Bypass model restrictions**, in Settings, removes both gates: models this Mac
does not meet the requirements for become downloadable and loadable, and a
model whose manual context and cache exceed the budget will load. The
requirement is still shown, in orange, because bypassing is a decision to
proceed rather than a claim that the Mac is big enough. A model that does not
fit will swap heavily, and macOS may terminate TUFF while it loads or
generates. Nothing is damaged and the download stays on disk.

With restrictions in force, a model whose settings exceed the budget now says
so in the conversation, naming the settings responsible and how to proceed.
Previously it simply refused to load and the only explanation was in the
Settings memory section, which is not where someone trying to send a message is
looking.

The memory limits above come from real model runs. They are not guesses based
on parameter count. GPT-OSS 120B completed a coherent 4K-context run on my 16 GB
M2 with a 7.44 GiB peak process footprint and zero swap. This proves that the
configuration runs on that machine. It is not a claim that it will be fast for
every workload.

### Benchmarks

Every number below was measured on my own M2 MacBook Air (`Mac14,2`, 16 GB,
macOS 26.5.2, Swift 6.3.1) on AC power, using a fresh release CLI process per
run and `/usr/bin/time -l`. Prompt length, generated length, cache state, and
hardware all move these numbers, so a range across workloads is not run-to-run
variance.

One short question, `What is the capital of France?`, one process per model.
Decode rate excludes install, load, and prefill. Every model answered correctly:

Every rate is the **best of three runs**, because this fanless Mac shares its
GPU with the window server and the sweep ran while the machine was also driving
a desktop session. Repeated runs of one binary on one model spanned 19.1 to
37.9 tokens per second; the slow runs measured that interference, and the first
run of any model also faults its memory-mapped weights in from SSD. Every
individual run is recorded alongside the summary.

| Model | Decode | Prefill | Peak RSS |
| --- | ---: | ---: | ---: |
| Gemma 4 E2B IT | 43.96 tok/s | 0.56 s | 324 MiB |
| Gemma 4 E4B IT | 28.96 tok/s | 1.08 s | 324 MiB |
| Gemma 4 12B IT QAT | 6.88 tok/s | 14.55 s | 381 MiB |
| Gemma 4 26B-A4B IT | 7.69 tok/s | 11.03 s | 1,394 MiB |
| Qwen3.6 35B-A3B | 6.11 tok/s | 15.61 s | 1,417 MiB |
| GPT-OSS 20B | 2.36 tok/s | 19.03 s | 2,616 MiB |
| GPT-OSS 120B | 0.19 tok/s | 66.47 s | 2,430 MiB |
| MiniMax M2.7 4-bit | 0.33 tok/s | 71.37 s | 2,103 MiB |

These are conservative for that reason: the same build measured 40.30 tokens
per second on Gemma 4 E2B on an otherwise idle machine. Treat the ordering
across models as the signal and each absolute figure as a floor. Reproduce any
row with `Scripts/benchmark_simple.rb --repeat 3`.

MiniMax M2.7 needed 132 tokens to finish this answer, four past the 128-token
cap the sweep uses, because it always reasons and its thinking is not part of
the visible answer. Run to completion it answers correctly and stops at end of
turn, at 0.303 tok/s. Its row above is the capped run, measured the same way as
every other row.

The Gemma 4 12B QAT row was 0.04 tok/s in v4.0.1 and is 6.88 here. That was a
bug in v4.0.0 and v4.0.1, not a property of the model; v4.0.2 fixes it. See
below.

#### What v4.0.0 changed about decode

Decode on a dense model used to spend most of its time waiting rather than
computing. Three things caused that, and all three are fixed:

- **Attention.** Each query head ran in its own 256-thread threadgroup, so every
  cached position cost two threadgroup barriers, and the query heads sharing a
  key/value head each re-read that head's keys and values. Attention now gives
  one SIMD group to each query head, reduces a score with a single
  `simd_sum`, and has no barrier in its inner loop.
- **Command buffers.** A dense layer produces nothing the CPU reads, yet the
  runner still committed a command buffer per layer and blocked on it. On this
  Mac a round trip costs about 130 microseconds against roughly 500 microseconds
  of layer work, so 35 of them per token were most of the gap between GPU time
  and tokens per second. A dense token is now one command buffer, and where the sampler does
  not need a host pass it rides on that same buffer.
- **Weight mapping.** All of a model's resident weights were wrapped in one
  Metal buffer, up to gigabytes wide. Metal's per-dispatch cost for a bound
  buffer grows with that buffer's size, and decode issues hundreds of dispatches
  per token, so the size was being paid hundreds of times over. Weights are now
  mapped as many 64 MB regions, with any single larger tensor still getting a
  region of its own. Alone this took Gemma 4 E2B from 28.4 to 42.3 tokens per
  second in alternating builds.

  v4.0.0 and v4.0.1 applied that split to every model, which was wrong. A model
  whose weights do not fit in memory needs the opposite: one large mapping the
  kernel can page lazily, rather than many Metal buffers that are all forced
  resident together. Gemma 4 12B QAT — 10 GB of dense weights on a 16 GB Mac —
  collapsed from 6.97 to 0.034 tokens per second. **v4.0.2 applies the split
  only when a model's resident weights are at most a quarter of physical
  memory**, and maps everything larger as a single region.

Measured on Gemma 4 E2B with the harness above, alternating binaries on one
machine:

| Runner | Decode |
| --- | ---: |
| v3.0.3 | 17.94 tok/s |
| v4.0.0 | 40.30 tok/s |

Greedy output is byte-identical across the two runners for the same prompt and
seed, on both Gemma 4 E2B and Gemma 4 26B-A4B.

Mixture-of-experts decode is unchanged within run-to-run variance: it waits on
SSD reads for routed experts, not on the paths above. That is also why giving a
mixture-of-experts model a larger expert cache lowers its measured read wait
without raising its tokens per second, and why a dense model gains nothing at
all from extra memory. Memory buys residency; it does not buy bandwidth. Peak RSS understates what a
model costs, because the weights are memory-mapped and the kernel owns those
pages; the longer-workload rows below report footprint alongside RSS for that
reason.

Longer workloads on Gemma 4 E4B, at temperature `0.2`, Top-K `64`, Top-P `0.95`,
with a frozen prompt and seed. Both runs were coherent and stopped at end of
turn:

| Case | Context | Prompt / generated | Prefill | Decode | Peak RSS / footprint |
| --- | ---: | ---: | ---: | ---: | ---: |
| short-explanation | 4,096 | 57 / 440 | 3.08 s | 9.267 tok/s | 406.2 / 872.2 MiB |
| long-synthesis | 8,192 | 3,011 / 369 | 204.40 s | 6.820 tok/s | 582.3 / 1,748.5 MiB |

GPT-OSS 120B is the headline case: a verified 61 GiB install, run at 4,096-token
context with Low reasoning and 16 expert-cache slots, returning the requested
answer and stopping at EOS on a 16 GB machine with no swap. That establishes the
configuration runs on that hardware. It is not a claim that it is fast.

The 8 GB figures in the model table are a memory calculation, not a measurement.
They combine each model's measured peak footprint with TUFF's 2 GiB system
reserve. I have not run these models on an 8 GB Mac.

## Images

Every Gemma 4 model and Qwen3.6 use optional `.vision.gturbo` companion packs.
They are separate because a text-only user should not have to download image
weights. Install or remove the add-on from its model card.

A pack is bound to the exact text checkpoint it was built from, not just to the
model family. Installing one against a different checkpoint is refused at
download time rather than at first image.

Image input fails closed. If the companion is missing, corrupt, built for a
different model, or unsupported by the Mac, TUFF rejects the image. It never
silently drops an image and answers the remaining text.

Images stay in the conversation. A follow-up question is answered by a model
that can still see the picture, up to as many images as the context window
holds; older ones are dropped from the context, oldest first, exactly as older
turns are.

## Local server

The easiest way to start the server is from the Server screen in the app. The
supported endpoints are:

- `GET /health`
- `GET /v1/models`
- `POST /v1/chat/completions`

Chat Completions supports regular JSON, streaming SSE, model-aware reasoning,
function-tool declarations, prompt reuse, and images when the matching add-on
is installed. The client remains responsible for approving and executing every
tool call.

The server has no authentication or TLS. It binds only to `127.0.0.1`, and it
should not be proxied, tunneled, or exposed to another machine. Point any
OpenAI-compatible client at `http://127.0.0.1:<port>/v1` with any API key value;
`GET /v1/models` reports the identifier to send as `model`.

## Build it yourself

You need macOS 15+, Swift 6.2+, Metal 3.2+, and an Apple Silicon Mac.

```bash
git clone https://github.com/rexmhall09/TUFF.git
cd TUFF
swift build -c release
.build/release/TUFF
```

To build the complete app bundle, embedded updater, ZIP, and checksum:

```bash
Scripts/package_app.sh 3.0.2
open dist/TUFF.app
```

The clone-style executable looks for models under `scratch/`. The packaged app
keeps everything it owns in one place, `~/Library/Application Support/TUFF`,
with models in `Models/` and saved chats in `Chats/`. Models left in the older
`TurboFieldfare` directory by an earlier version are moved across on first
launch. Existing compatible v1 Gemma and Qwen installations remain readable.

### Command-line interface

The stable installer selectors are `gemma4-e2b`, `gemma4-e4b`,
`gemma4-12b-qat`, `gemma4`, `qwen36`, `gpt-oss-20b`, `gpt-oss-120b`, and
`minimax-m2.7`:

```bash
swift run -c release TUFFRepack \
  --model gpt-oss-20b \
  --output scratch/gpt-oss-20b.gturbo
```

Interrupted downloads keep their verified ranges. Continue with `--resume`, or
remove saved download state with `--discard-partial`.

Run a local chat from the CLI:

```bash
swift run -c release TUFFCLI \
  --model scratch/gpt-oss-20b.gturbo \
  --chat-prompt "Why does bounded expert streaming matter?" \
  --reasoning low \
  --max-new 256
```

The packaged app also includes a unified native command at
`TUFF.app/Contents/Resources/bin/tuff`. Add that directory to `PATH` to use the
short command everywhere:

```bash
export PATH="/Applications/TUFF.app/Contents/Resources/bin:$PATH"
tuff load minimax-m2.7
tuff prompt "Why does bounded expert streaming matter?"
tuff serve --port 8080
```

`tuff load` selects an already-installed model, opens the containing app, and
requests a real load. `prompt` and `serve` use the selected app model unless
`--model` names another catalog model or `.gturbo` path. Their context,
sampling, and expert-cache defaults come from the model catalog.

## How it works

One Foundation-only registry describes each checkpoint's source, fingerprint,
installation, hardware rules, architecture, add-ons, prompt dialect, and
defaults. The app, installer, CLI, and server all use that registry.

The runtime then separates the checkpoint from the architecture behavior:

- dense and mixture-of-experts feed-forward paths
- Gemma, Qwen ChatML, GPT-OSS Harmony, and MiniMax M2.7 prompts
- affine INT4 and GPT-OSS MXFP4 weights
- model-specific attention, routing, RoPE, KV, and memory plans
- bounded expert reads and bounded chunked-prefill scratch
- FP32 GPT-OSS residuals with FP16 projection, attention, expert, and KV paths

The `.gturbo` major version is still v1. A compatible minor extension describes
dense feed-forward and MXFP4 layouts. Older runtimes reject features they do not
understand instead of misreading the model.

The file format, memory ownership, Metal kernels, prefill, expert streaming,
prompt reuse, and image companions are documented in the source, which is the
copy that stays current.

## Tests

Run the serial package suite with:

```bash
Scripts/test.sh
```

Before a real model run, make sure there is enough disk space, memory pressure
is acceptable, the model installation is complete, and no other TUFF app, CLI,
server, decode service, test, or MLX process is using a model. Run one model
process at a time.

The test strategy includes CPU references for Metal kernels, toy-model forward
and prefill checks, format and repack verification, tokenizer and template
goldens, installer failure paths, persistent-chat tests, server contention,
update configuration, deterministic workspace renders, and real checkpoint
smokes. A green model-free suite does not prove that a checkpoint works.

## Contributing

I want all kinds of contributions that make TUFF better. Code, design, model
support, bug reports, accessibility, docs, tests, benchmark results, and small
quality-of-life fixes are all useful. You do not need to be a Swift or Metal
expert.

AI-assisted contributions are 100% welcome. Name the tool or model when you
know it, explain what it helped with, and review and test the result yourself.
You are still responsible for the code you submit.

Read [CONTRIBUTING.md](CONTRIBUTING.md) for the technical guardrails, test
commands, model qualification requirements, and what to include in a pull
request.

## AI and authorship

I use several AI models while building TUFF. They help me research,
write code, make tests, review changes, and edit documentation. I review the work, and take responsibility for the
project. AI assistance is part of how I build it. It does not make the project
less mine. Any other contributors are welcome to use AI, as long as they review code.

## License and credit

TUFF source and documentation use the [Apache License 2.0](LICENSE). Model
weights are not included and keep their original terms. Third-party package and
font details are in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

TUFF began as a fork of
[drumih/turbo-fieldfare](https://github.com/drumih/turbo-fieldfare) by Andrey
Mikhaylov. That project established the original Gemma runtime, bounded expert
streaming, and much of the foundation I still build on. TUFF is now evolving as
its own project, but I want the origin and credit to stay clear.
