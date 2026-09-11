# TUFF 5.0.0: Qwen 3.8 runner update

The working tree already contained the Qwen 3.8 integration when this work
started. These changes preserve that integration and address its runtime,
vision, context controls, and Dock presentation.

## Changes

- Preserve Qwen's vision attention scale (`1/sqrt(72)`) when constructing the
  runtime. The previous `max(scale, 1)` changed that scale to one. This affects
  both Qwen families and requires an updated runner, not a new image pack.
- Hash the original image-pad token IDs in the n-gram PLE. The zero IDs used as
  embedding-gather placeholders are not the original model input.
- Preserve the multimodal RoPE offset when a text-only prefill follows an image.
- Retain up to 4,096 dequantized n-gram rows per runner. Repeated image-pad runs
  reuse rows instead of repeating three scattered reads per row.
- Specialize Qwen's hyper-connection, PLE, vocabulary-head, indexer, and fused
  QKV projections using their actual dimensions and quantization group size.
- Implement QSA compressed-block selection: pool raw index keys in groups of
  four, apply centered RMSNorm and MRoPE, score blocks using float32 products,
  select 512 complete blocks, and attend to their tokens plus the incomplete
  causal tail. Radix selection avoids a full sort and resolves cutoff ties by
  block index. The short-context path retains dense attention.
- Include index-key and position storage in automatic context memory estimates.
  Speed and Balanced target one quarter and one half of the affordable context;
  Context takes the largest window. Targets round down to menu choices and
  retain the qualified default as their floor. Expert-cache counts stay at
  their measured defaults, with the existing chunked-prefill minimum.
- Offer native model limits rather than a universal 64K cap: 128K for Gemma
  E2B/E4B and GPT-OSS; 200K for MiniMax M2.7; 256K for Gemma 12B/26B and Qwen.
- Keep the compiled system icon for packaged apps. Bare SwiftPM launches use
  an inset PNG fallback. The source artwork is unchanged by this correction.

The pinned [Qwen configuration](https://huggingface.co/mlx-community/Qwen3.8-Flash-Next-4bit/blob/07b5dc6c54600a359b87f1e53e7adf6351c72a2c/config.json)
specifies 262,144 positions. QSA follows the
[mlx-vlm implementation](https://github.com/Blaizzy/mlx-vlm/blob/main/mlx_vlm/models/qwen4_exp/language.py).

## Numerical evidence

The installed image pack was compared with mlx-vlm using exactly the same
BF16 patch pixels and weights. On the public
[cat photograph](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg),
630 projected features of width 2,560 gave:

| Runtime | Relative L2 error | Cosine similarity |
| --- | ---: | ---: |
| Before vision-scale correction | 0.93434 | 0.38904 |
| After vision-scale correction | 0.04028 | 0.99919 |

The remaining difference includes floating-point rounding in the two runtime
implementations. The regression test exercises the runtime's attention factory,
not just an isolated kernel with a manually supplied scale.

QSA tests compare query normalization, pooled keys, MRoPE, block selection, and
attention against independent CPU calculations. Selection is tested through
65,536 compressed blocks (the 256K range). Runner tests exercise sparse
prefill/decode agreement, incomplete blocks across chunk boundaries, reset,
image-token handoff, and text follow-ups after an image. Full-capacity selector
tests do not establish end-to-end generation quality at 256K on a 16 GB Mac.

## Reproducing vision parity

Use a Python environment with `numpy` and `mlx-vlm`, and save the pinned
checkpoint's `config.json` locally. These commands read an installed model;
they do not download or rewrite its weights.

```sh
TUFF_QWEN_VISION_PARITY_IMAGE=/absolute/path/image.jpeg \
TUFF_QWEN_VISION_PARITY_MODEL=/absolute/path/qwen38-flash-next.gturbo \
TUFF_QWEN_VISION_PARITY_OUTPUT=/tmp/qwen-vision-capture \
Scripts/test.sh --filter Qwen38VisionParityTests

python Scripts/check_qwen_vision_parity.py \
  /absolute/path/qwen38-flash-next.vision.gturbo \
  /tmp/qwen-vision-capture /absolute/path/config.json
```

The Python comparison fails if relative error reaches 0.08 or cosine similarity
falls to 0.995. The capture test is opt-in and is skipped in the ordinary suite.
Run the normal regression gate with `Scripts/test.sh`.

The real runner identified that photo as a Pallas's cat after the scale fix.
Before that correction it answered "zebra." Image timings overlap regression
work and are not used as throughput benchmark evidence.

## Build and regression gate

`Scripts/test.sh` passed 1,592 tests in 256 suites. The final parameterized
vision check then passed five tests, including runtime scale construction for
both Qwen families. `Scripts/package_app.sh 4.1.1 dist/qwen38-runner-preview`
completed with signature verification and ZIP round-trip validation. That initial local
preview used the old version number and was not installed or published. Its Info.plist contains both `CFBundleIconName` and
`CFBundleIconFile`, so the Dock uses the compiled icon.

A real 2,135-token prompt crossed the 2,048-token dense-attention range and
correctly recovered the passphrase `silver orchard`. The run used a 4K window,
256-token prefill chunks, 32 expert slots, temperature zero, and thinking off.
It completed normally with an end-of-turn token. This validates the real
checkpoint's sparse path beyond 2K; full 256K generation remains untested.

## Short-text timing evidence

Four serial fresh-process runs used the same 24-token prompt and generated
64 tokens at temperature zero, with a 2K window and 32 expert slots. No builds
or regression tests ran alongside these measurements. All four output hashes
matched. OS file-cache warmth was allowed, so launch order matters.

| Order | Runner | Prefill seconds | Decode tokens/second | Total seconds |
| --- | --- | ---: | ---: | ---: |
| 1 | Saved original binary | 44.34 | 1.879 | 82.79 |
| 2 | Updated binary | 42.10 | 2.291 | 74.19 |
| 3 | Saved original binary | 42.25 | 2.252 | 74.79 |
| 4 | Updated binary | 42.10 | 2.135 | 76.30 |

These results do **not** establish a consistent end-to-end throughput gain.
The updated range overlaps the original range, and the warm second pair
reverses the first pair's apparent improvement. The bounded row cache removes
repeated physical n-gram reads (covered directly by a regression test), and
projection specialization preserves numerical results, but a broad speedup
should not be claimed from this small sample.

The original is the release binary present when work began; its exact source
correspondence is not established. The working tree already had uncommitted
Qwen integration changes. Both binaries and raw commands/results are retained
locally, along with the benchmark script, under
`benchmark-results/qwen38-runner-update/`.

- Saved original CLI SHA-256: `0f92e8a1a8b11d0945fbe1e0739bce9628bff2814d17a8bdd60c1ee88e70171f`
- Updated CLI SHA-256: `b8507c081484dab13f241513c1b073201de4665d81423167b1b92f1977d4e1f3`
- Host: macOS 26.6.2, 16 GiB unified memory.

## Version 5.0.0 package

The update is named 5.0.0. The About-panel fallback, README, and version
regression expectation now agree. Seven focused About-panel tests passed.
`Scripts/package_app.sh 5.0.0 dist/v5.0.0` produced the local app and ZIP,
verified signatures, and validated a ZIP round-trip. Both bundle version keys
are 5.0.0. This package has not been installed or published.

## Follow-up: icon alignment and large-window generation

The Icon Composer layers used a vertical source center of 358.475 even though
the bird's complete path bounds, including its lower circular arc, are
128.19–895.81 (center 512). Both layers now transform around (512, 512).
The compiled icon was rendered and visually inspected; the bird remains at
the existing scale and is centered vertically.

Qwen4-Exp full-attention KV storage now starts at at most 2,048 tokens and
grows geometrically as the conversation advances. A 32K configuration starts
with 48 MiB of KV storage instead of 768 MiB, without lowering its maximum.
Growth uses an ordered GPU copy and preserves K and V independently.
Other model families retain their existing allocation policy. The new cache
regression covers pending GPU writes, non-power-of-two limits, stable reuse,
and reset.

### Follow-up timing evidence

The 48-token, temperature-zero sky prompt was run serially with 32 expert
slots. All runs produced identical text and the same 7,912 routed-expert
reads (24,370,479,104 requested bytes). On a 32K window, the first old-runner
run decoded at 0.738 tok/s, the growing-cache runner at 1.661 tok/s, and a
warmed repeat of the old runner at 1.560 tok/s. The separate 2K control
reached 1.906 tok/s. The much slower first run shows why its apparent 2.25x
improvement must not be advertised. The warmed 32K comparison is a modest
6.5% throughput difference in one pair, not a broad performance guarantee.

In the growing-cache run, expert I/O waits accounted for 12.35 seconds of
28.89 seconds of decode. GPU layer, routed, shared, and head work totalled
about 9.08 seconds. The remaining time includes submission, resource
management, and waits. The initial KV reduction of 720 MiB at 32K is a
deterministic benefit, independent of those timing fluctuations. Raw logs
and binary identities are in `benchmark-results/qwen38-generation-audit`.

The compiled 256px icon's violet artwork spans y=66…189: its vertical center
is 127.5, exactly the canvas center.

The updated real runner also passed the 2,135-token `silver orchard` retrieval
check with 256-token prefill chunks and a 4K configured window. That run grew
the KV allocation beyond 2K and ended normally with the correct passphrase.
This is a correctness check; its long-prompt timing is not an A/B benchmark.

The full follow-up gate passed **1,593 tests in 256 suites**. The final
package remains version 5.0.0.

## Further runner performance work

See [Qwen3.8 runner performance](QWEN38_RUNNER_PERFORMANCE.md) for the
subsequent 3 TPS attempt, retained changes, rejected experiments, and final
package verification.
