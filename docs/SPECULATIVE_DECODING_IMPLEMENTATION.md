# Speculative decoding proof of concept

Status: staged implementation note, 2026-09-02

## What the current runtime exposes

`LogitProducer.produce(token:position:into:)` is a scalar decode operation. The
token is embedded, every target layer runs, the KV cursor advances once, and a
head result is produced for the next boundary. `RawCompletion` owns the visible
token, stop matcher, detokenizer, cancellation, and continuation bookkeeping.

`ChunkedPrefillRunner.prefillChunked` is a separate, bounded path. It already
does batched embedding, RMSNorm, attention, grouped routed-MoE work, and a
single final-row head. Its seed semantics must remain unchanged: callers get
only the logits or greedy token for the final prompt row.

The affine (`RealForwardRunner`) and GPT-OSS runners can therefore share most
of their layer work with speculative verification, but need an explicit
per-row target-head result. The Qwen3.6 path additionally updates a fixed-size
GDN recurrent state for every row; rewinding only `KVCacheManager.position`
would leave rejected rows in that state. Qwen remains disabled until a
recurrent-state transaction is added.

## Proposed API and ownership

Generation policy uses two independent protocols:

* `DraftTokenProducer` proposes a bounded candidate block from emitted history.
* `SpeculativeVerificationRunner` verifies the block and owns a transactional
  speculative tail.

The verifier returns the target greedy prediction at the boundary before every
candidate and after the final candidate. Thus an `N`-token proposal returns
`N + 1` target IDs. The runner may write all candidate KV rows into the normal
  bounded tail, but it must not expose them as committed state until the caller
  invokes `commitSpeculativePrefix(_:)`. Committing `j` restores the logical
  cursor to `start + j`; `j == 0` discards the entire transaction and
  `j == processedTokens` keeps it all. Physical rejected bytes can remain in a
  future tail slot because attention is bounded by the logical cursor and the
  next accepted/correction token overwrites the boundary.

The first policy is greedy only. Non-greedy configurations continue through
the ordinary sampler path even if a drafter is supplied. This avoids silently
introducing biased sampling while preserving all existing fused-head and
sampler fast paths.

## Production implementation sequence

1. Land pure acceptance logic, proposal/result types, and transaction tests.
2. Add the raw-completion greedy loop with exact token-by-token visible output
   handling. Stop/EOS/cancellation/max-token boundaries commit only the
   already-safe prefix and leave the boundary token uncommitted, matching the
   current loop.
3. Extend the affine prefill head with a bounded GPU argmax buffer containing
   one token per verification row. Reuse `executePrefillChunk` without changing
   public prefill seed behavior.
4. Add the same explicit verifier to GPT-OSS, using its existing grouped
   prefill expert path. The initial implementation may use bounded GPU logits
   for per-row head evaluation if no row-argmax kernel is available, but it must
   read only row argmax IDs back to Swift.
5. Add a prompt-lookup/n-gram drafter and a benchmark command. It requires no
   second model or tokenizer and is useful for measuring verifier economics
   before a trained drafter exists.

## Expected metrics and benchmark plan

Each run should record rounds, proposed/accepted/rejected tokens, acceptance
rate, verification wall time, draft time, fallback time, target tokens per
emitted token, command-buffer counts, and the existing target I/O counters.
The central comparison is scalar target work and SSD expert bytes per emitted
token, not only aggregate tok/s.

The matrix is baseline plus block sizes 2/4/6/8 on a resident control and an
SSD-streamed MoE model, with repetitive prose, normal prose, code, and long
context prompts. A verifier-only microbenchmark compares one block pass with
the same number of scalar target calls. Results are recorded with the current
commit and binary identity so resumed measurements cannot mix revisions.

## AngelSpec / DFly boundary

AngelSpec is being used as an algorithmic reference only; no AngelSpec source
or runtime dependency is copied into TUFF. Its public documentation describes
standard rejection-sampling verification and block-parallel DFlash/DFly
drafters. DFlash consumes selected target hidden states through dual-source KV;
DFly adds residual per-layer fusion and optional hidden-state correction. A
future TUFF drafter would need target hidden-state taps at selected layers,
Metal attention/projection kernels, a same-vocabulary head, and an optional
`.gturbo` companion manifest. Training and checkpoint conversion would remain
outside the Swift runtime.

AngelSpec is Apache-2.0 except for separately listed third-party components.
This POC does not copy code, so no third-party source attribution is required
in TUFF. If a later implementation adapts source rather than the algorithm,
retain the relevant Apache-2.0 notices and mark modified files as changed.
