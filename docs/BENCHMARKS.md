# Benchmarks

This page records TUFF measurements on an 8 GB M2 MacBook Air and a
24 GB M5 Pro. Each number belongs to the workload shown. Prompt length,
generated length, cache state, and hardware all change throughput, so ranges
across workloads are not run-to-run variation.

Each table states its workload and decoding settings. TUFF uses the
model installed by the [command-line instructions](../README.md#command-line-interface).
Decode rate excludes model installation, model loading, and prompt prefill.

## Results at a glance

| Host and runtime | Decode rate | Reported memory |
| --- | ---: | ---: |
| 16 GB M2, TUFF, Gemma 4 E4B | 6.82-9.27 tok/s | 0.85-1.71 GiB footprint |
| 8 GB M2, TUFF | 5.10-6.30 tok/s | ~1.9-2.1 GB footprint |
| 24 GB M5 Pro, TUFF | 31-35 tok/s | ~2.1 GB footprint |
| 24 GB M5 Pro, mlx-lm | 76.33-82.07 tok/s | 8.3-9.8 GB RSS; 14.7-15.3 GB GPU allocation |
| M5, TUFF, Qwen 3.6 35B-A3B | 18.8-23.1 tok/s | ~1.45 GB footprint |

## Gemma 4 E4B qualification

These rows ran on 2026-08-25 on an M2 MacBook Air (`Mac14,2`) with 16 GB of
memory, macOS 26.5.2, and Swift 6.3.1. The Mac was connected to power and Low
Power Mode was off for AC power. The installed model came from
`mlx-community/gemma-4-e4b-it-4bit` revision
`475b9088d29754a3379866cf5aeb6b41acd313c2`. The measured code was commit
`ff3fcf3db10b0e81c7b441a9e860a6e8e5ce6028`.

Both rows used a fresh release CLI process, temperature `0.2`, Top-K `64`,
Top-P `0.95`, the frozen community prompt and seed, and `/usr/bin/time -l`.
Each response was coherent and reported `stop=endOfTurn`. These are one-run
qualification measurements, not the three-run medians required by the v2
benchmark harness in the final release gate.

| Case | Context | Prompt / generated | Prefill | Decode | Peak RSS / footprint |
| --- | ---: | ---: | ---: | ---: | ---: |
| short-explanation | 4,096 | 57 / 440 | 3.08 s | 9.267 tok/s | 406.2 / 872.2 MiB |
| long-synthesis | 8,192 | 3,011 / 369 | 204.40 s | 6.820 tok/s | 582.3 / 1,748.5 MiB |

The 8,192-token configuration is the qualified default. The catalog records
the measured 1,833,438,160-byte peak footprint. Its 8 GB eligibility is a
conservative memory calculation using that measured peak and TUFF's 2 GiB
system reserve; this table is not an 8 GB device benchmark.

The short deterministic smoke also compared chunked prefill with the scalar
decode reference path. Both rendered 20 prompt tokens, generated the same eight
tokens (`The capital of France is Paris.`), and stopped at end of turn. Chunked
prefill reported 2.77 s and 14.402 tok/s decode; `--prefill off` reported 2.95 s
and 11.732 tok/s. This exact token agreement is the live reference check for
the installed checkpoint, while the model-free suite supplies the independent
CPU and toy-forward numerical references.

Native thinking was checked separately with both the CLI and loopback server.
The model reached a correct final answer and `stop=endOfTurn` after 113 tokens.
The server returned only `There are 5 balls in the box.` after the structured
decoder removed the thought channel.

Exact measured commands:

```bash
/usr/bin/time -l .build/release/TUFFCLI \
  --model scratch/gemma4-e4b.gturbo \
  --messages-file docs/benchmark-prompts/real-generation-v1/short-explanation.json \
  --thinking off --max-new 1024 --max-context 4096 \
  --temperature 0.2 --top-k 64 --top-p 0.95 --seed 20260721

/usr/bin/time -l .build/release/TUFFCLI \
  --model scratch/gemma4-e4b.gturbo \
  --messages-file docs/benchmark-prompts/real-generation-v1/long-synthesis.json \
  --thinking off --max-new 1024 --max-context 8192 \
  --temperature 0.2 --top-k 64 --top-p 0.95 --seed 20260723
```

## GPT-OSS qualification

GPT-OSS 20B ran on 2026-08-25 on an M2 MacBook Air (`Mac14,2`) with 16 GB of
memory, macOS 26.5.2, and Swift 6.3.1. The installed model came from
`openai/gpt-oss-20b` revision
`6cee5e81ee83917806bbde320786a8fb61efebee`; its source index and completed
13,791,724,503-byte `.gturbo` install both passed SHA-256 verification. The
measured code was commit `0a95916bcb9f5b4b1a7df17ddad3dcbbe9ccd6da`.

The first deterministic smoke used Low reasoning at 1,024 context and answered
`4` with no visible analysis or channel markers. It stopped at EOS after 16
generated tokens, prefilling 80 tokens in 16.50 seconds and decoding at 2.822
tok/s. Peak RSS was 1,912,078,336 bytes and peak footprint was 5,385,082,256
bytes.

The release server at commit `4fe67d4` passed `/health`, advertised
`gpt-oss-20b` from `/v1/models`, and returned only `4` for the same prompt sent
to `/v1/chat/completions` with `reasoning_effort: "low"`. The response reported
80 prompt tokens, 16 completion tokens, and `finish_reason: "stop"`; the server
then shut down cleanly.

The frozen short qualification case used a 4,096-token context, Low reasoning,
greedy decoding, and no experimental controls. Its 113-token Harmony prompt
prefilled in 20.23 seconds. The generated explanation was coherent but reached
the 128-token cap, so its 4.115 tok/s rate is a qualification measurement, not
a complete benchmark row. Peak RSS was 2,180,726,784 bytes and peak footprint
was 5,487,695,296 bytes. The catalog records that measured 4K footprint and
requires 16 GB unified memory; no 8 GB GPT-OSS run has been made.

Exact commands:

```bash
/usr/bin/time -l .build/release/TUFFCLI \
  --model scratch/gpt-oss-20b.gturbo \
  --chat-prompt 'What is 2+2? Reply with only the number.' \
  --reasoning low --max-new 32 --max-context 1024 \
  --temperature 0 --top-k 0 --top-p 1 \
  --expert-cache-slots 16 --prefill on \
  --prefill-chunk-tokens auto --rdadvise off

/usr/bin/time -l .build/release/TUFFCLI \
  --model scratch/gpt-oss-20b.gturbo \
  --messages-file docs/benchmark-prompts/real-generation-v1/short-explanation.json \
  --reasoning low --max-new 128 --max-context 4096 \
  --temperature 0 --top-k 0 --top-p 1 \
  --expert-cache-slots 16 --prefill on \
  --prefill-chunk-tokens auto --rdadvise off
```

GPT-OSS 120B completed a local qualification smoke on 2026-08-26 on the same
M2 MacBook Air with 16 GB of memory, macOS 26.5.2, and Swift 6.3.1. Its bounded
installer produced a verified 61 GiB `.gturbo` directory from revision
`b5c939de8f754692c1647ca79fbf85e8c1e70f8a`. The measured runtime was commit
`c2748ce`. A 4,096-token-context run with Low reasoning and 16 expert-cache
slots returned the requested final answer `Paris` and stopped at EOS. Peak
footprint was 7,990,582,952 bytes with zero swaps. This was a qualification
smoke, not a benchmark run; no three-run v2 benchmark result is claimed. The
catalog therefore enables 120B on 16 GB Macs and records the measured footprint
as its default-context safety baseline.

Exact qualification command:

```bash
/usr/bin/time -l .build/release/TUFFCLI \
  --model scratch/gpt-oss-120b.gturbo \
  --chat-prompt 'Reply with the single word Paris. Do not explain.' \
  --max-new 64 --max-context 4096 --reasoning low \
  --temperature 0 --top-k 0 --top-p 1 \
  --expert-cache-slots 16 --prefill on \
  --prefill-chunk-tokens auto --rdadvise bounded
```

## M2 measured decode

These rows ran on a `Mac14,15` M2 MacBook Air with 8 GB of memory. No
experiment, profiler, or trace mode was active.

| Prompt / generated tokens | Prefill | TTFT | Decode | Peak RSS / footprint |
| --- | ---: | ---: | ---: | ---: |
| 6 / 32 | 7,025 ms | 7,979 ms | 6.30 tok/s | 1,304 / 1,791 MiB |
| 121 / 64 | 7,934 ms | 8,862 ms | 5.10 tok/s | 1,528 / 1,776 MiB |
| 527 / 64 | 21,736 ms | 22,649 ms | 5.90 tok/s | 1,535 / 1,886 MiB |
| 1,017 / 128 | 36,729 ms | 37,656 ms | 5.38 tok/s | 1,455 / 1,971 MiB |

Each workload ran once in a fresh process. The file cache was warm but
uncontrolled, and every row produced the same token IDs as its validation
control. These four points show the production path running under the 8 GB
rule; they do not form a confidence interval or describe sustained long
generation.

### Where the short M2 row spent its time

A separate diagnostic pass on the six-token prompt divided a 162.8 ms decode
step into four broad parts:

| Work | ms/token |
| --- | ---: |
| Expert reads | 83.1 |
| Waiting in the command-buffer pipeline | 55.6 |
| Tied output head | 14.2 |
| Other runtime work | 9.9 |

The diagnostic instrumentation disabled the normal command-buffer pipeline
and reduced throughput to 4.23 tok/s. The breakdown explains where that run
spent time; it does not describe independent speedups or a performance bound.

## M5 measured decode

These rows ran on 2026-07-20 on a 24 GB M5 Pro (`Mac17,8`) with macOS 26.5.1,
Xcode 26.6, and Swift 6.3.3. No profiler or trace mode was active.

The benchmark uses chat-framed prompts and fixed, non-repeating natural
continuations. This keeps the generated text and expert-routing workload stable
without rewarding a model repetition loop. The complete production sampling
and decode path still runs for every token.

One warmup preceded three fresh-process measurements per workload. The table
reports medians; the file cache was warm but uncontrolled. A separate
free-generation smoke reached the end of each model turn without a repetition
loop.

| Prompt / generated tokens | Prefill / TTFT | Decode | Peak RSS / footprint |
| --- | ---: | ---: | ---: |
| 61 / 256 | 5,096 / 5,668 ms | 35.17 tok/s | 1,834 / 2,126 MiB |
| 430 / 256 | 6,762 / 7,325 ms | 34.72 tok/s | 1,851 / 2,142 MiB |
| 3,015 / 256 | 23,038 / 23,610 ms | 31.01 tok/s | 1,835 / 2,126 MiB |

## Qwen 3.6 35B-A3B measured decode

These rows ran on 2026-07-31 on an M5 with 24 GB of memory, macOS 26.5, and
Swift 6.2, against the experimental
[Qwen 3.6 35B-A3B](../docs/QWEN36_PERFORMANCE.md) path. They follow the
[community benchmark protocol](COMMUNITY_BENCHMARKS.md): the three frozen
`real-generation-v1` prompts with their fixed seeds, app sampling defaults
(temperature `0.2`, Top-K `64`, Top-P `0.95`), 4K context, 16 expert-cache
slots, one discarded warmup, then one measured run per case in a fresh
process. Every measured footer reported `stop=endOfTurn`.

| Case | Prompt / generated tokens | Prefill | Decode | Peak RSS / footprint |
| --- | --- | ---: | ---: | ---: |
| short-explanation | 62 / 493 | 7.74 s | 23.05 tok/s | 1,139 / 1,447 MiB |
| medium-review | 426 / 697 | 12.71 s | 21.20 tok/s | 1,142 / 1,448 MiB |
| long-synthesis | 2,940 / 700 | 59.16 s | 18.84 tok/s | 1,093 / 1,464 MiB |

Qwen 3.6 decodes slower than Gemma 4 on the same host while using about
0.7 GB less memory. The gap is expert-streaming I/O, not compute: Qwen's
18.1 GB expert pool does not fit the page cache, and its 16 slots cover 6.2%
of a layer's 256 experts against Gemma's 12.5% of 128. The
[performance notes](QWEN36_PERFORMANCE.md) break the token down phase by
phase.

### Under an 8 GB working set

The same host was constrained by pinning 16 GB resident in a separate process,
leaving about 8 GB for the OS, page cache, and the model. All three cases were
rerun unchanged, in fresh processes, with the same seeds:

| Case | Decode, 24 GB | Decode, ~8 GB | Footprint, 24 GB | Footprint, ~8 GB | Output |
| --- | ---: | ---: | ---: | ---: | --- |
| short-explanation | 23.05 tok/s | 22.95 tok/s | 1,447 MiB | 1,464 MiB | byte-identical |
| medium-review | 21.20 tok/s | 21.35 tok/s | 1,448 MiB | 1,448 MiB | byte-identical |
| long-synthesis | 18.84 tok/s | 18.62 tok/s | 1,464 MiB | 1,388 MiB | byte-identical |

Every case still reported `stop=endOfTurn`, and each generated file matched its
unconstrained counterpart byte for byte. Throughput and footprint are unchanged
within run-to-run noise, because the 18.1 GB expert pool does not fit the page
cache on either configuration — decode is already streaming from SSD, so
shrinking available memory does not change what the runtime reads.

This is emulated pressure on M5 hardware, not a measurement on a physical 8 GB
Mac; a real 8 GB machine has a slower SSD and GPU and should be expected to
decode more slowly, as the M2 rows above show for Gemma 4.

## Same-host MLX comparison

The same M5 Pro ran MLX 0.32.0 and mlx-lm 0.31.3 against the same checkpoint,
prompt-token IDs, and generated-token counts. MLX measured 82.07, 80.25, and
76.33 tok/s for the 121-, 527-, and 1,017-token prompts.

Treat this as throughput context, not a complete engine comparison:

- The engines ran in separate blocks rather than a balanced, interleaved order.
- Their first-token clocks started at different points, so TTFT is not comparable.
- Generated IDs matched for the shortest prompt but diverged for the two longer prompts.
- TUFF recorded a 1.89-2.09 GiB physical footprint. MLX reported
  14.66-15.31 GB of peak GPU allocation and 8.27-9.79 GB of peak process RSS.
  Those counters measure different things and should not be compared as a
  direct memory ratio.

The MLX process required the larger host and is not an 8 GB TUFF
deployment path.

## Reproduce and contribute a result

The [community benchmark guide](COMMUNITY_BENCHMARKS.md) uses short, medium,
and long chat-framed prompts with fixed seeds. It requires coherent output and
a normal end of turn, so a repetition loop cannot become a published speed
result. The public CLI's timing footer reports decode-only throughput without a
separate research harness.

Community runs generate their own output, while the reference table uses fixed
non-repeating continuations for token-for-token stability. Compare community
submissions only when their prompt and generated token counts match.

A current checkout may not reproduce a historical number after the runtime,
compiler, or operating system changes. Report the commit and all three rows
rather than presenting one run as a general hardware result.

Read [System design](SYSTEM_DESIGN.md) for the runtime and resource split,
[Experiments](OPTIMIZATION_JOURNEY.md) for the main wins and failures, and the
[measurement lessons](experiments/summaries/09-validation-and-measurement-lessons.md)
for the rules used to evaluate performance changes.
