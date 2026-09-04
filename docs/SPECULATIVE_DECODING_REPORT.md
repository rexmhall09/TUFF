# TUFF speculative decoding proof of concept

Date: 2026-09-03
Recommendation: **KEEP EXPERIMENTAL**

Latest code measurement checkout: `7e0ea2ca5dadc2ae07a7c7a2122b595677ee8c56`
Latest release CLI SHA-256: `ad7b153fc95c95e41f8eb50e3b7b803aa5de691ba32335261a334ecb12bdd5d9`
The latest matrix is preserved in
`benchmark-results/speculative-latest-m2-20260903/`. Earlier matrices are
kept in `benchmark-results/speculative-batched-gpt-m2-20260903/` and
`benchmark-results/speculative-final-gemma-m2-20260903/` for comparison; they
are separate fresh-process runs and must not be combined into one median.

## What changed

TUFF now has an opt-in greedy speculative path. `DraftTokenProducer` supplies a
bounded candidate block, `SpeculativeVerificationRunner` verifies it as an
explicit target transaction, and `RawCompletion` applies the accepted prefix
through the existing detokenizer, stop matcher, progress callbacks, history,
and cancellation path. The normal scalar path remains the default.

The runtime has two protection mechanisms for the SSD case. `auto` adapts its
block size from measured target wall time and routed-expert bytes, and a target
backend can expose its current greedy boundary token. When the first draft
token is already wrong, TUFF emits the correction with one scalar target step
and skips the doomed block verification entirely.

The first real drafter is a prompt-lookup/n-gram drafter. It searches a bounded
recent history for a repeated suffix and proposes the tokens that followed the
earlier occurrence. It has no second model, tokenizer conversion, or runtime
download requirement.

The GPT-OSS verifier now batches resident BF16 projection and RMSNorm rows and
the final-head argmax across a speculative block, while retaining grouped
routed-expert execution. The boundary argmax is cached between the fast reject
check and a successful verification. These changes target the actual TUFF
costs without changing the scalar path or allocating a vocabulary-sized result
per candidate.

The CLI controls are deliberately experimental and opt-in:

```text
--speculative off|greedy|auto
--speculative-draft-tokens 1...8
```

Speculation is enabled only for pure greedy generation. Non-greedy sampling,
multimodal prompts, and the Qwen3.6 Gated-DeltaNet path use ordinary decode.

## Verification and KV semantics

For an N-token proposal, verification processes all N candidate rows through
the existing bounded chunked layer machinery and returns N+1 GPU-argmax token
IDs. The first entry is the prediction at the current boundary; each following
entry is the prediction after one candidate. The policy accepts the longest
matching prefix, then emits the first target mismatch or the post-block bonus
token.

The affine runner uses a persistent eight-token candidate buffer, batched
prefill scratch, and one GPU argmax result per row. GPT-OSS uses its grouped
prefill expert path and a batched BF16 argmax-row kernel, with one token ID per
row written to a bounded shared buffer. No full block of CPU logits or
vocabulary-sized per-row scratch is retained.

`KVCacheManager.rewind(to:)` moves only the logical cursor. Verification may
write a bounded physical tail, but `commitSpeculativePrefix(j)` exposes only
the accepted j rows. Rejected physical bytes are outside the logical attention
range and are overwritten by the next committed boundary. Qwen is disabled
because its recurrent GDN state needs a matching transaction; rewinding KV
alone would be incorrect.

## Correctness status

Passed:

- all-accepted, first-rejected, and partial-prefix greedy acceptance
- correction and all-accepted bonus-token handling
- EOS, stop strings across candidate boundaries, max-token limits
- cancellation during drafting, verification, and accepted-token emission
- continuation and a subsequent user-turn boundary
- unsupported non-greedy fallback to scalar decode
- deterministic scripted greedy equivalence over many target sequences
- affine Gemma toy row-by-row argmax equivalence and rollback
- GPT-OSS toy row-by-row argmax equivalence and rollback
- Qwen verification is explicitly advertised as unsupported
- speculative-off behavior continues through the existing path

Focused suites passed during implementation: 29 raw-completion/speculative
tests plus the Gemma, GPT-OSS, BF16GEMV, and RMSNorm equivalence suites. The
canonical `Scripts/test.sh` gate passed against the latest checkpoint with
1,484 tests in 240 suites in 283.116 seconds.

## Measurements

Installed bundles were found in the TUFF app support directory rather than the
repository. The latest TPS matrix ran on a MacBook Air (M2, 16 GB), using one
warmup and three measured fresh-process runs for baseline, auto, and blocks
2/4/6/8, with `max-new 16`, `max-context 2048`, and the repetitive fixture.
GPT-OSS uses Harmony chat mode in the harness; Gemma uses raw completion mode.
The raw directory preserves every process, stdout/stderr, command, system
report, commit, and CLI hash. `benchmark-results/` is not tracked, so these
are paths on the machine that ran the matrix rather than links:

- latest resident Gemma plus streamed GPT-OSS:
  `benchmark-results/speculative-latest-m2-20260903/summary.md`
- earlier batched-kernel comparison:
  `benchmark-results/speculative-batched-gpt-m2-20260903/summary.md`
- earlier streamed-Gemma comparison:
  `benchmark-results/speculative-final-gemma-m2-20260903/summary.md`

The full four-prompt matrix remains available through
[Scripts/benchmark_speculative.rb](../Scripts/benchmark_speculative.rb). The
`--prompt NAME` filter was added so long-running streamed targets can be
measured in bounded, resumable slices. Every condition still records fresh
process repeats, medians, output-equivalence checks, commit hash, CLI SHA-256,
peak RSS, phase timings, acceptance, verification time, and verification
expert-I/O fields.

The available Metal toy fixtures provide a verifier-only measurement on this
MacBook Air (M2, 16 GB), Swift 6.3.3, macOS 26.6.2. These are not representative
large-model or SSD results. Each row is the median of four measurements after
the runner was initialized.

| Backend | Block | Scalar target ms | Block verify ms | Scalar/block speedup | Expert reads / bytes |
| --- | ---: | ---: | ---: | ---: | --- |
| dense Gemma toy | 2 | 1.655 | 1.215 | 1.36x | 0 / 0 |
| dense Gemma toy | 4 | 3.215 | 1.576 | 2.04x | 0 / 0 |
| dense Gemma toy | 6 | 6.122 | 1.880 | 3.26x | 0 / 0 |
| dense Gemma toy | 8 | 7.273 | 2.549 | 2.85x | 0 / 0 |
| GPT-OSS toy | 2 | 3.164 | 2.123 | 1.49x | 0 / 0, 8 cache hits |
| GPT-OSS toy | 4 | 5.958 | 1.947 | 3.06x | 0 / 0, 8 cache hits |
| GPT-OSS toy | 6 | 8.396 | 2.157 | 3.89x | 0 / 0, 8 cache hits |
| GPT-OSS toy | 8 | 11.441 | 2.209 | 5.18x | 0 / 0, 8 cache hits |

The toy result establishes that the production block path is not N scalar
`produce` calls and can be cheaper for these tiny shapes. It does not establish
the central SSD hypothesis: the toy expert cache is warm and the shapes are not
representative of a large model. The live matrices below provide end-to-end
prompt-lookup acceptance and total expert-traffic measurements.

The verifier reports expert cache-plan misses, estimated physical expert bytes,
hits, and misses per block. The benchmark also reports total decode-phase
expert reads and bytes per emitted token, which includes verification, scalar
fallback, correction, and bonus work. Verification command-buffer counts are
reported by the production runners; hardware GPU counters and physical SSD
counters remain unavailable.

### Latest-code live TPS result

The table below reports medians of three measured runs after one warmup. Decode
TPS excludes model load and prefill. All runs generated the capped 16 tokens,
and the benchmark compared speculative stdout with the baseline before saving
each condition. Expert traffic is the total decode-phase routed-expert traffic
per emitted token, including verification and scalar fallback/correction work.

| Model | Condition | Decode tok/s | vs baseline | Acceptance | Expert reads/token | Expert MiB/token | Verify ms | Boundary rejects/checks |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Gemma E4B resident | baseline | 26.732 | 1.000x | n/a | 0.00 | 0.00 | n/a | n/a |
| Gemma E4B resident | auto | 2.385 | 0.089x | 0.929 | 0.00 | 0.00 | 3,371.69 | 0/3 |
| Gemma E4B resident | block 2 | 5.255 | 0.197x | 0.909 | 0.00 | 0.00 | 1,928.13 | 1/6 |
| Gemma E4B resident | block 4 | 9.126 | 0.341x | 0.923 | 0.00 | 0.00 | 1,325.59 | 1/4 |
| Gemma E4B resident | block 6 | 7.391 | 0.276x | 0.929 | 0.00 | 0.00 | 1,842.41 | 0/3 |
| Gemma E4B resident | block 8 | 7.285 | 0.273x | 0.933 | 0.00 | 0.00 | 2,055.96 | 0/2 |
| GPT-OSS 20B streamed | baseline | 1.423 | 1.000x | n/a | 60.31 | 761.45 | n/a | n/a |
| GPT-OSS 20B streamed | auto | 1.424 | 1.001x | 0.000 | 60.31 | 761.45 | 0.00 | 1/1 |
| GPT-OSS 20B streamed | block 2 | 1.506 | 1.058x | 0.250 | 60.19 | 759.87 | 1,118.45 | 3/4 |
| GPT-OSS 20B streamed | block 4 | 1.436 | 1.009x | 0.273 | 61.38 | 774.86 | 3,043.63 | 2/3 |
| GPT-OSS 20B streamed | block 6 | 1.142 | 0.803x | 0.200 | 61.75 | 779.59 | 4,694.03 | 2/3 |
| GPT-OSS 20B streamed | block 8 | 1.278 | 0.898x | 0.167 | 62.50 | 789.06 | 4,347.50 | 2/3 |

These are separate fresh-process measurements, not a combined median across
checkouts. The resident control is visibly noisy across the two matrices, so
its block rows are useful as a control for the verifier path but not as a stable
speed claim. The latest streamed GPT-OSS rows are the comparison for the latest
code checkpoint.

The latest SSD result is mixed but informative. On streamed GPT-OSS, block 2
reached 1.058x baseline while reducing total traffic slightly from 761.45 to
759.87 MiB per emitted token. Block 4 was effectively flat at 1.009x but
increased traffic to 774.86 MiB/token; blocks 6/8 fell to 0.803--0.898x and
reached 779.59--789.06 MiB/token. Three of four block-2 proposals were
rejected by the boundary fast path, so this is not yet evidence of a robust
high-acceptance win. Auto mode avoids a full failed verification and stays at
baseline when the drafter has no accepted first token.

The earlier streamed-Gemma matrix increased total expert traffic from 216.41
to 224.82--225.42 MiB per emitted token and did not produce a repeatable TPS
gain. It remains useful as a second SSD-target control, but it is from an
earlier code checkpoint and is not combined numerically with the latest GPT
matrix.

## Known limitations

- Only greedy speculation is enabled. Distribution-preserving rejection
  sampling for temperature/top-k/top-p/repetition penalty is intentionally not
  implemented.
- Qwen3.6 is disabled until GDN state can be transactionally snapshot or
  rewound.
- Prompt lookup is intentionally cheap but will have poor acceptance on many
  prompts; it is a verifier exercise, not a trained drafter.
- The current block head is optimized for one argmax ID per row, not arbitrary
  target distributions.
- Expert-byte accounting is based on the existing cache plan and configured
  expert stride. It is an estimate of physical expert reads, not a hardware
  SSD-counter trace.
- The benchmark requires externally installed models and does not silently
  substitute toy fixtures for them.

## AngelSpec / DFly follow-up

AngelSpec is an algorithmic reference only. Its documentation describes target
prefill, parallel candidate verification, accepted-prefix handling, and
distribution-preserving rejection sampling; its DFlash/DFly material describes
block-parallel drafters that use target hidden features rather than a separate
full target forward. See the [speculative decoding overview](https://angelspec.readthedocs.io/en/latest/concepts/speculative_decoding.html),
[draft-model family](https://angelspec.readthedocs.io/en/latest/concepts/draft_model_family.html),
[DFlash](https://angelspec.readthedocs.io/en/latest/concepts/dflash.html), and
[DFly](https://angelspec.readthedocs.io/en/latest/concepts/dfly.html) docs.
No AngelSpec source was copied and TUFF has no AngelSpec runtime dependency.
The upstream repository is Apache-2.0 with separately listed third-party
components; any future source adaptation would need to preserve the relevant
notices. See the [upstream license](https://raw.githubusercontent.com/Tencent/AngelSpec/main/LICENSE).

For Qwen3.6 35B-A3B, the likely integration is an optional same-tokenizer,
feature-space drafter:

1. Add selected-layer hidden-state taps to `RealForwardRunner`. Today its
   prefill hidden buffer is scratch and is overwritten layer by layer; no
   target hidden-state API is retained. Qwen's GDN state would also need a
   transaction if Qwen is selected as the first target.
2. Export a small block-parallel module that consumes one or more selected
   target hidden states, applies residual/attention or low-rank projections,
   and predicts a bounded token block. The target's tokenizer and token IDs
   must be shared exactly.
3. Add Metal kernels for feature projections, block attention/KV assembly, and
   the drafter head. Keep the drafter optional and resident; never stream a
   second large model from SSD for the first experiment.
4. A practical sizing hypothesis is 4 layers at width 1,024 and gated width
   4,096. With Qwen's 248,320-token vocabulary, tied embeddings plus the
   transformer body are roughly 320M parameters, about 160 MB at 4-bit before
   metadata and scratch. Sharing the target embedding/head or using low-rank
   feature adapters could reduce new resident parameters to roughly 10–50M.
   These are design estimates, not measured AngelSpec or TUFF model sizes.
5. Train and distill the drafter outside TUFF, export quantized tensors and
   tokenizer identity into an optional `.gturbo` companion/manifest, then load
   it only when the target and tokenizer hashes agree. The manifest needs
   drafter version, target architecture, hidden taps, block size, quantization,
   and memory estimate.

The next experiment should run the remaining prompt fixtures on the installed
models and add a deliberately high-acceptance scripted/replay workload. That
will separate verifier economics from prompt-lookup quality. In parallel,
profile route overlap across candidate rows and make the streamed expert
planner expose enough reuse information to decide whether a block should be
attempted. If block verification remains slower even with high acceptance and
high route overlap, a trained DFly-style drafter is unlikely to fix the core
bottleneck.

## Recommendation

**KEEP EXPERIMENTAL —** block verification is real, KV-safe, and materially
cheaper than repeated scalar target calls in the toy microbenchmark. The latest
streamed GPT-OSS matrix has a narrow block-2 positive at 1.058x with slightly
lower expert traffic, but block 4 is flat and blocks 6/8 are slower with more
traffic; acceptance is only 0.167--0.273. The resident control is noisy and
does not show a resident-model benefit. Keep this as a first-class optimization
surface, but do not enable it by default or train a DFly drafter until a
high-acceptance, high-route-overlap workload demonstrates lower target work and
SSD bytes per emitted token.

## Files changed

- `Sources/TUFFEngine/Runtime/Generation/Speculative/`
- `Sources/TUFFEngine/Runtime/Generation/RawCompletion.swift`
- `Sources/TUFFEngine/Runtime/Generation/Sampler.swift`
- `Sources/TUFFEngine/Runtime/Inference/RealForwardRunner.swift`
- `Sources/TUFFEngine/Runtime/Inference/GPTOSSForwardRunner.swift`
- `Sources/TUFFEngine/Runtime/Inference/ModelForwardRunner.swift`
- `Sources/TUFFEngine/Runtime/Inference/ModelExpertIO.swift`
- `Sources/TUFFEngine/Runtime/KVCache/KVCacheManager.swift`
- `Sources/TUFFEngine/Kernels/Quant/BF16GEMV.swift`
- `Sources/TUFFEngine/Kernels/Primitives/RMSNorm.swift`
- `Sources/TUFFEngine/Kernels/Fusions/LMHeadChainInt4.swift`
- `Sources/TUFFEngine/Kernels/Sampling/Argmax.swift`
- `Sources/TUFFEngine/Metal/Primitives/rmsnorm.metal`
- `Sources/TUFFEngine/Metal/Quant/mxfp4.metal`
- `Sources/TUFFEngine/Metal/Sampling/logit.metal`
- `Sources/TUFFCLI/Args.swift`
- `Sources/TUFFCLI/Run.swift`
- `Tests/TUFFEngine/Core/Runtime/Generation/SpeculativeDecodingTests.swift`
- `Tests/TUFFEngine/Core/Runtime/Generation/RawCompletionLoopTests+Speculative.swift`
- `Tests/TUFFEngine/Core/Runtime/Generation/PromptLookupDraftTokenProducerTests.swift`
- `Tests/TUFFEngine/Core/Kernels/Quant/BF16GEMVTests.swift`
- `Tests/TUFFEngine/Core/Kernels/Primitives/RMSNormTests.swift`
- `Tests/TUFFEngine/Core/Runtime/DenseGemmaRunnerTests.swift`
- `Tests/TUFFEngine/Core/Runtime/GPTOSSRunnerTests.swift`
- `Tests/TUFFEngine/Core/Runtime/QwenRunnerTests.swift`
- `Tests/TUFFEngine/Core/CLI/CLIArgumentsTests.swift`
- `Scripts/benchmark_speculative.rb`
- `docs/benchmark-prompts/speculative-v1/`
- `docs/SPECULATIVE_DECODING_IMPLEMENTATION.md`
- `docs/SPECULATIVE_DECODING_REPORT.md`

## Future cleanup

- Add exact GPU timing and physical I/O counters if the Metal runtime exposes
  them without perturbing the hot path.
- Add transactional Qwen GDN state before enabling that target.
- Implement and test standard speculative rejection sampling before allowing
  any non-greedy configuration.
- Add the remaining real-model prompt slices and route-overlap telemetry; keep
  commit and CLI hashes alongside every run.
- Profile the batched final-head kernels and route-group command-buffer counts
  before attempting a larger block or a trained drafter.
