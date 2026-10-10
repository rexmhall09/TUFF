# Local server

TUFF can serve your installed models to other apps, scripts and coding
agents through an OpenAI-compatible API. It only listens on `127.0.0.1` and
has no authentication, so keep it that way.

## Start it

**In the app:** turn on **Background API** on the Server screen and allow the
login item when macOS asks. It then runs whether TUFF is open or not. Pick a
default model, a port, and how long an idle model stays loaded.

![The Server screen](assets/tuff-server.png)

**In a terminal:**

```sh
tuff serve --default-model gemma4-e2b --unload-after 300 --port 8080
```

From a clone: `swift run -c release TUFFServer --models-root scratch`.

Point your client at `http://127.0.0.1:8080/v1`. `GET /v1/models` lists the
installed models and their context and output limits. A request for
`default` uses the default model; a model ID loads that one. Requests run one
at a time.

Chat and the API share one memory budget. If Chat has a model with image
support loaded, the API answers 503 instead of loading a second model.

## Endpoints

| Endpoint | |
| --- | --- |
| `GET /health` | Is it up |
| `GET /v1/models` | Installed models and limits |
| `POST /v1/chat/completions` | The main route, with everything below |
| `POST /v1/messages` | Messages format, text and function tools |
| `POST /v1/responses` | Responses format, text and function tools |

Chat Completions supports streaming, reasoning, function tools, images (with
an image pack installed) and prompt reuse. Your client runs tool calls; the
server never does.

Unknown fields return `unknown_parameter`, and unsupported values return
`unsupported_value`, so you find out instead of getting silently different
behavior. `chat_template_kwargs` accepts only `enable_thinking` and
`preserve_thinking`.

### Messages and Responses

These are smaller subsets, enough for clients that speak those formats.
Send the whole conversation every time; nothing is stored by ID. Reasoning is
left out of their output, so use Chat Completions if you want it.

They refuse things they can't do faithfully, rather than guessing: text after
a tool call, signed thinking, image blocks, server tools, cache-control,
forced tool choice, `stop_sequences`, stored responses and hosted tools,
among others. Gemma needs tool-only assistant turns, and GPT-OSS allows one
tool call per turn.

They are not drop-in replacements for the default Claude Code or Codex
setups, which send options these subsets refuse.

## Reasoning

Models that reason return it separately from the answer, as
`reasoning_content` (or `delta.reasoning_content` when streaming), with a
count in `usage.completion_tokens_details.reasoning_tokens`.

## Prompt reuse

The server keeps the last conversation's state, plus up to four others,
within a memory budget of at most 1 GB. Sending a conversation it has seen
continues from where it left off instead of reading everything again.
`usage.prompt_tokens_details.cached_tokens` says how many tokens were reused,
and `prompt_cache_key` helps it find the right conversation first. Under
memory pressure the extra states are dropped.

A new conversation that starts with the same system prompt and tools as an
earlier one (256 tokens or more) skips reading them again. The server also
saves that state to disk the first time it reads a system prompt, so it
survives restarts. The files live in `~/Library/Caches/TUFF/PrefixSnapshots`,
use at most 2 GB, and are only used for the same model, settings and TUFF
version.

`TUFF_CONVERSATION_CACHE_MB` lowers the memory budget (`0` keeps only the
current conversation), `TUFF_PREFIX_DISK_CACHE_MB` lowers the disk budget
(`0` turns it off), and `TFF_LOG_CACHE=1` logs each reuse decision.

## Timings

Every response includes `tuff_timings_seconds`: validation, queueing and
loading, prefill, decode, and time to the first visible output. The engine
spans overlap, so don't add them up.

## Agents

[OMP](OMP.md) has a tested setup. Big models can take minutes to read an
agent's instructions before they answer, so give your client a long timeout.
