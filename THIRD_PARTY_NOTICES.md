# Third-party software and model terms

TUFF repository source is licensed under the
[Apache License 2.0](LICENSE). That license applies to TUFF's own source
and documentation. It does not relicense model weights or third-party
packages.

This file records the dependency review updated on 2026-08-26. It is an
attribution aid, not legal advice. Anyone distributing a compiled product must
also preserve the license and NOTICE material required by the exact dependency
versions included in that product.

## Model weights

Model weights are not included in this repository. The installer downloads a
pinned revision of each checkpoint and repacks it locally. Every entry below is
the exact revision TUFF installs; the license column records what that model
card states, and the model card remains authoritative.

| Model | Source repository | Pinned revision | License stated by the model card |
| --- | --- | --- | --- |
| Gemma 4 E4B IT | [`mlx-community/gemma-4-e4b-it-4bit`](https://huggingface.co/mlx-community/gemma-4-e4b-it-4bit) | `475b9088d29754a3379866cf5aeb6b41acd313c2` | Apache-2.0 quantization of Google's Gemma 4 E4B instruction checkpoint |
| Gemma 4 26B-A4B IT | [`mlx-community/gemma-4-26b-a4b-it-4bit`](https://huggingface.co/mlx-community/gemma-4-26b-a4b-it-4bit) | `0d77464eeb233a2da68ebf9d7dc4edaac7db956d` | Apache-2.0 quantization of Google's Gemma 4 26B-A4B instruction checkpoint |
| Qwen3.6 35B-A3B | [`mlx-community/Qwen3.6-35B-A3B-4bit`](https://huggingface.co/mlx-community/Qwen3.6-35B-A3B-4bit) | `38740b847e4cb78f352aba30aa41c76e08e6eb46` | Apache-2.0 quantization of Alibaba's Qwen3.6 35B-A3B checkpoint |
| GPT-OSS 20B | [`openai/gpt-oss-20b`](https://huggingface.co/openai/gpt-oss-20b) | `6cee5e81ee83917806bbde320786a8fb61efebee` | Apache-2.0 |
| GPT-OSS 120B | [`openai/gpt-oss-120b`](https://huggingface.co/openai/gpt-oss-120b) | `b5c939de8f754692c1647ca79fbf85e8c1e70f8a` | Apache-2.0 |
| MiniMax M2.7 | [`mlx-community/MiniMax-M2.7-4bit`](https://huggingface.co/mlx-community/MiniMax-M2.7-4bit) | `66d2e5cb7c5cda05251b4625c504af4b034df7ff` | Modified MIT terms linked by the model card |

Google publishes Gemma 4 under the
[Apache License 2.0](https://ai.google.dev/gemma/apache_2). The GPT-OSS
checkpoints are the official OpenAI releases and are installed in their native
MXFP4 form rather than requantized.

MiniMax M2.7 is governed by the upstream
[MiniMax model license](https://github.com/MiniMax-AI/MiniMax-M2.7/blob/main/LICENSE),
which the pinned model card labels `modified-mit`.

Downloaded weights remain a separate work governed by their source terms. Do
not redistribute weights as part of TUFF releases.

Optional image add-ons are companion packs built from the same Gemma 4 26B-A4B
and Qwen3.6 revisions listed above; they carry no separate terms.

## Swift package graph

The following table covers the complete graph reported by
`swift package show-dependencies` from the checked-in
[`Package.resolved`](Package.resolved). Exact revisions are recorded there.

| Package | Version | License in locked checkout |
| --- | --- | --- |
| [swift-transformers](https://github.com/huggingface/swift-transformers) | 1.3.3 | Apache-2.0 |
| [SwiftMath](https://github.com/mgriebling/SwiftMath) | 1.7.3 | MIT; bundled math fonts retain their GUST and SIL Open Font licenses |
| [Sparkle](https://github.com/sparkle-project/Sparkle) | 2.9.2 | BSD-3-Clause |
| [swift-jinja](https://github.com/huggingface/swift-jinja) | 2.3.6 | Apache-2.0 |
| [swift-huggingface](https://github.com/huggingface/swift-huggingface) | 0.9.0 | Apache-2.0 |
| [EventSource](https://github.com/mattt/EventSource) | 1.4.1 | MIT |
| [swift-nio](https://github.com/apple/swift-nio) | 2.101.3 | Apache-2.0; upstream NOTICE applies |
| [swift-atomics](https://github.com/apple/swift-atomics) | 1.3.0 | Apache-2.0 with Runtime Library Exception |
| [swift-collections](https://github.com/apple/swift-collections) | 1.5.1 | Apache-2.0 with Runtime Library Exception |
| [swift-system](https://github.com/apple/swift-system) | 1.6.4 | Apache-2.0 with Runtime Library Exception |
| [swift-crypto](https://github.com/apple/swift-crypto) | 4.5.0 | Apache-2.0; upstream NOTICE applies |
| [swift-asn1](https://github.com/apple/swift-asn1) | 1.7.0 | Apache-2.0; upstream NOTICE applies |
| [yyjson](https://github.com/ibireme/yyjson) | 0.12.0 | MIT |

No copyleft or custom non-commercial license was found in this resolved graph.
The dependency license files remain authoritative. For binary distribution,
collect their license and NOTICE files from the exact revisions in
`Package.resolved`; do not treat this summary as a substitute for that bundle.
