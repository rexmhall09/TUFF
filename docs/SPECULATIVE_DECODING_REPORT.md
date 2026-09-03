# TUFF speculative decoding proof of concept

Date: 2026-09-02
Recommendation: **KEEP EXPERIMENTAL**

Measurement checkout: `346d8c26e8d53e03c795fb2d83c85aae71afa3e2`
Release CLI SHA-256: `fe1fa4e86316b36bf69f2131ef4ebf376f1bc37b32761c26d20d0ca8ac1ef74d`

## What changed

TUFF now has an opt-in greedy speculative path. `DraftTokenProducer` supplies a
bounded candidate block, `SpeculativeVerificationRunner` verifies it as an
explicit target transaction, and `RawCompletion` applies the accepted prefix
through the existing detokenizer, stop matcher, progress callbacks, history,
and cancellation path. The normal scalar path remains the default.

The first real drafter is a prompt-lookup/n-gram drafter. It searches a bounded
recent history for a repeated suffix and proposes the tokens that followed the
earlier occurrence. It has no second model, tokenizer conversion, or runtime
download requirement.

The CLI controls are deliberately experimental and opt-in:

```text
--speculative off|greedy
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
prefill expert path and a persistent vocabulary-sized FP16 head scratch, with
one argmax result per row. No full block of CPU logits is retained.

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

Focused suites passed during implementation: 27 raw-completion tests, 14
Gemma/GPT-OSS runner tests, 24 CLI/drafter tests, and 9 generic speculative
contract tests. The canonical `Scripts/test.sh` gate then passed 1,473 tests
in 240 suites in 643.8 seconds.

## Measurements

The current checkout contains no installed `.gturbo` directories, so the
resident-control versus SSD-streamed-model end-to-end matrix was not run. The
reproducible harness is [Scripts/benchmark_speculative.rb](../Scripts/benchmark_speculative.rb)
and requires explicit `--resident-model PATH` and `--streamed-model PATH`
arguments. It runs baseline plus blocks 2/4/6/8 over repetitive, normal prose,
code-continuation, and long-context prompts, with fresh-process repeats,
medians, output-equivalence checks, commit hash, CLI SHA-256, peak RSS, phase
timings, acceptance, verification time, and verification expert-I/O fields.

The available Metal toy fixtures provide a verifier-only measurement on this
MacBook Air (M2, 16 GB), Swift 6.3.3, macOS 26.6.2. These are not representative
large-model or SSD results. Each row is the median of four measurements after
the runner was initialized.

| Backend | Block | Scalar target ms | Block verify ms | Scalar/block speedup | Expert reads / bytes |
| --- | ---: | ---: | ---: | ---: | --- |
| dense Gemma toy | 2 | 1.099 | 1.428 | 0.77x | 0 / 0 |
| dense Gemma toy | 4 | 10.051 | 1.241 | 8.10x | 0 / 0 |
| dense Gemma toy | 6 | 14.203 | 1.829 | 7.77x | 0 / 0 |
| dense Gemma toy | 8 | 22.020 | 2.202 | 10.00x | 0 / 0 |
| GPT-OSS toy | 2 | 6.201 | 1.621 | 3.83x | 0 / 0, 8 cache hits |
| GPT-OSS toy | 4 | 12.266 | 2.474 | 4.96x | 0 / 0, 8 cache hits |
| GPT-OSS toy | 6 | 19.498 | 4.941 | 3.95x | 0 / 0, 8 cache hits |
| GPT-OSS toy | 8 | 15.882 | 3.296 | 4.82x | 0 / 0, 8 cache hits |

The toy result establishes that the production block path is not N scalar
`produce` calls and can be cheaper for these tiny shapes. It does not establish
the central SSD hypothesis: the streamed fixture's expert cache was warm in
this microbenchmark, and no large-model physical expert-byte comparison is
available yet. The end-to-end prompt-lookup acceptance rate is also unmeasured
until a real model is installed.

The verifier reports expert cache-plan misses, estimated physical expert bytes,
hits, and misses per block when the model backend exposes them. Exact Metal
command-buffer and GPU-counter values remain unavailable in this POC and are
reported as unavailable by the benchmark rather than inferred.

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

The next experiment should install a qualified resident control and a streamed
GPT-OSS or Gemma MoE model, run the supplied matrix, and inspect verification
expert bytes per emitted token. If block verification is materially cheaper and
the prompt-lookup acceptance is low, that is the point to train a DFly-style
drafter. If verification does not reduce streamed target work, a trained
drafter is unlikely to fix the core bottleneck.

## Recommendation

**KEEP EXPERIMENTAL —** the architecture is correct and the toy block verifier
is promising, but the decisive SSD-streamed measurements and real acceptance
rates are not available in this checkout. Do not enable it by default or invest
in DFly training until the real-model benchmark reports target work and expert
bytes per emitted token.

## Files changed

- `Sources/TUFFEngine/Runtime/Generation/Speculative/`
- `Sources/TUFFEngine/Runtime/Generation/RawCompletion.swift`
- `Sources/TUFFEngine/Runtime/Generation/Sampler.swift`
- `Sources/TUFFEngine/Runtime/Inference/RealForwardRunner.swift`
- `Sources/TUFFEngine/Runtime/Inference/GPTOSSForwardRunner.swift`
- `Sources/TUFFEngine/Runtime/Inference/ModelForwardRunner.swift`
- `Sources/TUFFEngine/Runtime/KVCache/KVCacheManager.swift`
- `Sources/TUFFEngine/Kernels/Fusions/LMHeadChainInt4.swift`
- `Sources/TUFFEngine/Kernels/Sampling/Argmax.swift`
- `Sources/TUFFEngine/Metal/Sampling/logit.metal`
- `Sources/TUFFCLI/Args.swift`
- `Sources/TUFFCLI/Run.swift`
- `Tests/TUFFEngine/Core/Runtime/Generation/SpeculativeDecodingTests.swift`
- `Tests/TUFFEngine/Core/Runtime/Generation/RawCompletionLoopTests+Speculative.swift`
- `Tests/TUFFEngine/Core/Runtime/Generation/PromptLookupDraftTokenProducerTests.swift`
- `Tests/TUFFEngine/Core/Runtime/DenseGemmaRunnerTests.swift`
- `Tests/TUFFEngine/Core/Runtime/GPTOSSRunnerTests.swift`
- `Tests/TUFFEngine/Core/Runtime/QwenRunnerTests.swift`
- `Tests/TUFFEngine/Core/CLI/CLIArgumentsTests.swift`
- `Scripts/benchmark_speculative.rb`
- `docs/benchmark-prompts/speculative-v1/`
- `docs/SPECULATIVE_DECODING_IMPLEMENTATION.md`
- `docs/SPECULATIVE_DECODING_REPORT.md`

## Future cleanup

- Add exact command-buffer/GPU timing and physical I/O counters if the Metal
  runtime exposes them without perturbing the hot path.
- Add transactional Qwen GDN state before enabling that target.
- Implement and test standard speculative rejection sampling before allowing
  any non-greedy configuration.
- Add a real-model benchmark result artifact after model installation; keep
  commit and CLI hashes alongside every run.
- Revisit the block head to avoid serial per-row vocabulary GEMVs if profiling
  shows the head dominates verification.
