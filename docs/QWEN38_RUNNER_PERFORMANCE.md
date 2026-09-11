# Qwen3.8 Flash Next runner performance

This follow-up targets 3 tokens/second on the local 16 GiB Mac. It preserves
weights, routed top-k, context capacity, sampling, and generated output.

## Retained changes

- Put cache-hit phase 1 and the shared expert in one GPU submission. Both
  depend on the same input and execute while missing experts are read.
- Reuse the routed argument buffer after the previous layer's router readback
  has established that its GPU consumers have completed.
- Pass an integer zero to the n-gram reader's read-ahead control. The old
  pointer argument was nonzero and enabled read-ahead instead. Apple's
  [fcntl documentation](https://developer.apple.com/library/archive/documentation/System/Conceptual/ManPages_iPhoneOS/man2/fcntl.2.html)
  specifies an integer value for `F_RDAHEAD`.
- Retry interrupted expert reads instead of treating `EINTR` as a model error.
- Keep active generation at user-initiated priority and prevent App Nap only
  while a completion is running, allowing normal system sleep.
- Overlap full SHA-256 verification of large expert files with one layer of
  lookahead. At most two checks run per model; all checks and descriptor-based
  file opening remain required before the corresponding weights are used.
- Reuse the app's current text conversation state when its token prefix matches.
  Ordinary non-thinking text follow-ups also use the server's native continuation
  encoder after checking the preceding messages and the generated response.
  Edits, changed settings, trimming, images, cancellation, and incompatible
  boundaries fall back to a fresh prefill. No second KV cache is allocated.

The existing filesystem caching policy and expert-cache size remain in place.
The shared-command GPU timing now includes cache-hit phase 1 when combined.
It should not be compared as isolated shared-expert kernel time against the
old runner's `gpu shared cbs` metric.

## Measurement protocol

Runs use fresh processes, serially, with no concurrent tests or builds.
The filesystem cache is not flushed, so repeat results and launch order matter.
All runs use temperature zero, thinking off, a 32K context limit, and 32
expert-cache slots unless explicitly identified otherwise. TPS excludes model
load and prefill.

The short prompt is `Explain why the sky is blue in a short paragraph.`,
with 48 generated tokens. The longer prompt is `Explain how a bicycle stays
balanced while moving, in about 150 words.`, with 128 generated tokens.

Raw logs, binaries, shader resources, source hashes, commands, and output hashes
are retained in `benchmark-results/qwen38-three-tps/`. `protocol.json` and
`exploration.json` identify the initial experiments; subsequent runs have
individual JSON records. The baseline is the existing 5.0.0 working-tree
runner, not an unmodified Git release.

## Experiments and decisions

| Candidate | Short prompt TPS | Longer prompt TPS | Decision |
| --- | ---: | ---: | --- |
| Previous 5.0.0 runner | 1.873, 2.346 | 2.351, 2.540 | Baseline |
| Bypass filesystem caching and combine GPU submissions | 2.482, 2.431 | 1.899, 2.439 | Drop cache bypass |
| Combine GPU submissions, normal filesystem caching | 2.552 | 2.575 | Retain submission change |

The first long-response pair ran baseline then cache bypass; the repeat ran
in reverse order. Normal filesystem caching won both long-response pairs.
The much slower initial baseline and cache-bypass long run also show why
these measurements do not support a broad percentage speedup claim.

Increasing the expert cache to 48 slots reduced short-prompt reads from 7,912
to 6,534 but slowed decode to 1.579 TPS. That configuration is not adopted.
The I/O-only bypass trial reached 2.464 TPS on the short prompt; its small
sample was insufficient to justify changing the default policy.

Every candidate produced byte-identical output for its prompt. At 32 slots,
short runs requested 24,370,479,104 expert bytes in 7,912 reads; longer runs
requested 68,478,828,544 bytes in 22,232 reads. These counts describe requested
expert data, not measured physical SSD traffic.

Artifacts named `TUFFCLI-final`, `final-*`, or `longer-final*` identify the
rejected cache-bypass candidate. Artifacts named `ship-*` identify an intermediate
packaged runner, including argument-buffer reuse but preceding preprocessing work. Earlier artifacts remain
intact so comparisons do not silently mix binaries.

## Correctness coverage

A new regression forces both cache hits and misses and compares the resulting
logits with cold and fully cached execution. The repeat also checks that a
fully cached pass performs no additional expert reads. Existing file-reading,
public API compatibility, Qwen, vision, prefill, and other-model tests remain
part of the canonical gate.

The subsequent `Scripts/test.sh --disable-sandbox` gate passed **1,601 tests in
258 suites** in 471.147 seconds, including continuation state, file descriptor
ownership, and asynchronous checksum failures. The public streamer initializer and the
original filesystem caching behavior are preserved.

## App preprocessing comparison

The packaged decode helpers were driven through their real IPC interface with
the same three-turn conversation: a 324-token bicycle note, a follow-up about
that note, and a third turn asking for the vehicle's name. Both used a 4K
context, 32 expert slots, 128-token prefill chunks, greedy sampling, and full
SHA-256 verification. The baseline ran first. Answers were identical.

| Turn | Previous helper | Helper with conversation reuse |
| --- | ---: | ---: |
| Initial long prompt | 184.93 s | 212.36 s |
| Follow-up | 159.17 s | 22.29 s |
| Third turn | 196.42 s | 24.72 s |

These are prefill times, excluding model loading and decode. The two follow-ups
improved by about 86% and 87%. The initial long prompt regressed in this pair;
this evidence does not establish an improvement for long cold prompts. There
was no cache flush or thermal normalization between the runs. Full IPC events,
responses, helper hashes, and the driving script are retained in
`benchmark-results/v5.0.0-validation/prefill-{before,after}.json` and
`benchmark_prefill.py`. A final guard additionally prevents an in-flight
completion from publishing a cache entry into a replaced or unloaded session.

A separate fresh-process pair used the 19-token capital-of-France chat prompt,
32K context, 32 expert slots, and greedy sampling. Short-prompt prefill fell
from **45.54 s to 29.52 s** (about 35%). Both produced the same nine-token answer;
decode was essentially unchanged at 0.601 and 0.599 tok/s. These short answers
do not establish sustained decode throughput. The records are
`benchmark-results/qwen38-three-tps/cold-preprocessing-{before,after}-1.json`.
The final cache lifecycle guard passed 17 focused tests in 4.630 seconds.

## Release app package

`Scripts/package_app.sh 5.0.0 dist/v5.0.0-release-public` completed successfully.
Both bundle version fields are 5.0.0. Code-signature verification and the ZIP
round-trip passed. Previous app bundles were preserved.

Packaged CLI SHA-256:
`ab185100ef5a5b60491b17766322977393bfce998892a87099bda44f65093c70`.

The final package includes the cache lifecycle guard and was used for the
[nine-model Paris and six-model photo checks](V5_MODEL_VALIDATION.md).
