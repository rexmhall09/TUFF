# Third-party software and model terms

TUFF repository source is licensed under the
[Apache License 2.0](LICENSE). That license applies to TUFF's own source
and documentation. It does not relicense model weights or third-party
packages.

The dependency review below is dated 2026-08-26. The model source inventory
was synchronized with TUFF's catalog on 2026-10-06, including checks of the
pinned E2B, 12B QAT and Flash Next model cards. This inventory refresh is not a
new dependency or legal review. Anyone distributing a compiled product must
preserve the license and NOTICE material required by its exact dependencies.

## Model weights

Model weights are not included in this repository. The installer downloads a
pinned revision of each checkpoint and repacks it locally. Every entry below is
the exact revision TUFF installs. The license column distinguishes metadata in
the pinned quantization card from upstream publisher terms when that metadata
is absent or inconsistent. Source license files remain authoritative.

| Model | Source repository | Pinned revision | License stated by the model card |
| --- | --- | --- | --- |
| Gemma 4 E2B IT | [`mlx-community/gemma-4-e2b-it-4bit`](https://huggingface.co/mlx-community/gemma-4-e2b-it-4bit/tree/238767527555cb75a05732a84dff5d6ba0dd6809) | `238767527555cb75a05732a84dff5d6ba0dd6809` | Pinned card labels `gemma`; Google's current model card states Apache-2.0, as noted below |
| Gemma 4 E4B IT | [`mlx-community/gemma-4-e4b-it-4bit`](https://huggingface.co/mlx-community/gemma-4-e4b-it-4bit) | `475b9088d29754a3379866cf5aeb6b41acd313c2` | Apache-2.0 quantization of Google's Gemma 4 E4B instruction checkpoint |
| Gemma 4 12B IT QAT | [`mlx-community/gemma-4-12B-it-qat-4bit`](https://huggingface.co/mlx-community/gemma-4-12B-it-qat-4bit/tree/e70c6b3ba0979b3357dcd2f223ad8bde7787a6b6) | `e70c6b3ba0979b3357dcd2f223ad8bde7787a6b6` | No license field in the pinned card; see Google's Gemma 4 terms below |
| Gemma 4 26B-A4B IT | [`mlx-community/gemma-4-26b-a4b-it-4bit`](https://huggingface.co/mlx-community/gemma-4-26b-a4b-it-4bit) | `0d77464eeb233a2da68ebf9d7dc4edaac7db956d` | Apache-2.0 quantization of Google's Gemma 4 26B-A4B instruction checkpoint |
| Qwen3.6 35B-A3B | [`mlx-community/Qwen3.6-35B-A3B-4bit`](https://huggingface.co/mlx-community/Qwen3.6-35B-A3B-4bit) | `38740b847e4cb78f352aba30aa41c76e08e6eb46` | Apache-2.0 quantization of Alibaba's Qwen3.6 35B-A3B checkpoint |
| GPT-OSS 20B | [`openai/gpt-oss-20b`](https://huggingface.co/openai/gpt-oss-20b) | `6cee5e81ee83917806bbde320786a8fb61efebee` | Apache-2.0 |
| GPT-OSS 120B | [`openai/gpt-oss-120b`](https://huggingface.co/openai/gpt-oss-120b) | `b5c939de8f754692c1647ca79fbf85e8c1e70f8a` | Apache-2.0 |
| MiniMax M2.7 | [`mlx-community/MiniMax-M2.7-4bit`](https://huggingface.co/mlx-community/MiniMax-M2.7-4bit) | `66d2e5cb7c5cda05251b4625c504af4b034df7ff` | Modified MIT terms linked by the model card |
| Qwen3.8 Flash Next | [`mlx-community/Qwen3.8-Flash-Next-4bit`](https://huggingface.co/mlx-community/Qwen3.8-Flash-Next-4bit/tree/07b5dc6c54600a359b87f1e53e7adf6351c72a2c) | `07b5dc6c54600a359b87f1e53e7adf6351c72a2c` | No license field in the pinned card; the upstream publisher labels the model `qwen-community-1.0` |

Google publishes Gemma 4 under the
[Apache License 2.0](https://ai.google.dev/gemma/apache_2). The GPT-OSS
checkpoints are the official OpenAI releases and are installed in their native
MXFP4 form rather than requantized.

The pinned E2B quantization card's `gemma` tag differs from the Apache-2.0
license stated by Google's [E2B model card](https://huggingface.co/google/gemma-4-E2B-it).
The pinned 12B QAT quantization card has no license field or separate license
file; Google's [12B QAT model card](https://huggingface.co/google/gemma-4-12B-it-qat-q4_0-unquantized)
states Apache-2.0. These observations record the metadata difference rather
than resolving or replacing the source terms.

Flash Next's pinned quantization card also has no license field or separate
license file. Its linked [upstream model card](https://huggingface.co/Qwen/Qwen3.8-Flash-Next)
states `qwen-community-1.0` and supplies the
[Qwen Community License](https://huggingface.co/Qwen/Qwen3.8-Flash-Next/blob/main/LICENSE).
Do not infer Apache-2.0 terms from other Qwen checkpoints or from missing
quantization-card metadata.

MiniMax M2.7 is governed by the upstream
[MiniMax model license](https://github.com/MiniMax-AI/MiniMax-M2.7/blob/main/LICENSE),
which the pinned model card labels `modified-mit`.

Downloaded weights remain a separate work governed by their source terms. Do
not redistribute weights as part of TUFF releases.

The six optional image companions are repacked from the corresponding pinned
sources above: Gemma 4 E2B, E4B, 12B QAT and 26B-A4B, Qwen3.6 and Qwen3.8
Flash Next. Consult the same source model cards and license files for those
weights; packaging them separately does not replace their upstream terms.

## Swift package graph

The following table covers the complete graph reported by
`swift package show-dependencies` from the checked-in
[`Package.resolved`](Package.resolved). Exact revisions are recorded there,
except SwiftMath's, which is vendored and recorded in
[`Vendor/SwiftMath/README.md`](Vendor/SwiftMath/README.md).

| Package | Version | License in locked checkout |
| --- | --- | --- |
| [swift-transformers](https://github.com/huggingface/swift-transformers) | 1.3.3 | Apache-2.0 |
| [SwiftMath](https://github.com/mgriebling/SwiftMath) | 1.7.3, vendored with a resource-lookup patch | MIT; the bundled Latin Modern Math font retains its GUST Font License |
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

## Sandbox image

The web sandbox image built from [`Sandbox/`](Sandbox/) is a separate artifact,
not part of the app bundle. It contains the packages below, pinned by hash in
[`Sandbox/requirements.txt`](Sandbox/requirements.txt), on top of the pinned
`python:3.12-slim` base image (Debian and CPython, each under its own terms)
and Debian's `nftables` package (GPL-2.0-only, installed unmodified from a
dated snapshot). Licenses are those stated in each package's metadata at the
pinned version. `tld` offers MPL-1.1, GPL-2.0 or LGPL-2.1 terms and `regex`
adds the CNRI Python license; both and `certifi` (MPL-2.0) are used unmodified.
This list was not a legal review. Anyone redistributing the image must
preserve the license and NOTICE material of its exact contents.

| Package | Version | License |
| --- | --- | --- |
| [trafilatura](https://github.com/adbar/trafilatura) | 2.3.0 | Apache-2.0 |
| [pypdf](https://github.com/py-pdf/pypdf) | 6.20.0 | BSD-3-Clause |
| [courlan](https://github.com/adbar/courlan) | 1.4.0 | Apache-2.0 |
| [htmldate](https://github.com/adbar/htmldate) | 1.11.0 | Apache-2.0 |
| [jusText](https://github.com/miso-belica/jusText) | 3.0.2 | BSD-2-Clause |
| [lxml](https://github.com/lxml/lxml) and lxml_html_clean | 6.1.3, 0.4.5 | BSD-3-Clause |
| [dateparser](https://github.com/scrapinghub/dateparser) | 1.4.3 | BSD-3-Clause |
| [python-dateutil](https://github.com/dateutil/dateutil) | 2.9.0.post0 | BSD-3-Clause or Apache-2.0 |
| [Babel](https://github.com/python-babel/babel) | 2.18.0 | BSD-3-Clause |
| [tld](https://github.com/barseghyanartur/tld) | 0.13.2 | MPL-1.1, GPL-2.0-only or LGPL-2.1-or-later |
| [regex](https://github.com/mrabarnett/mrab-regex) | 2026.9.29 | Apache-2.0 and CNRI-Python |
| [certifi](https://github.com/certifi/python-certifi) | 2026.7.22 | MPL-2.0 |
| [charset-normalizer](https://github.com/jawah/charset_normalizer) | 3.5.2 | MIT |
| [urllib3](https://github.com/urllib3/urllib3) | 2.8.0 | MIT |
| [pytz](https://github.com/stub42/pytz), [six](https://github.com/benjaminp/six), [tzlocal](https://github.com/regebro/tzlocal) | 2026.4, 1.17.0, 5.4.4 | MIT |
