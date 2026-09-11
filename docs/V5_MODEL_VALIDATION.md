# TUFF 5.0.0 model validation

TUFF 5.0.0, measured 2026-09-10 on a 16 GB M2 MacBook Air. One fresh process per model, answering `What is the capital of France?` with a 4,096-token context, seed 20260721, and a 128-token output cap (256 for MiniMax). Decode speed excludes model loading and prefill; prefill includes the first-use weight checks. 9/9 runs named Paris. These short responses are smoke tests, not a sustained-throughput or model-quality comparison. Host load and filesystem caching can affect the timings.

| Model | Decode | Prefill | Peak RSS |
| --- | ---: | ---: | ---: |
| Gemma 4 E2B IT | 46.96 tok/s | 0.47 s | 324 MiB |
| Gemma 4 E4B IT | 27.71 tok/s | 0.80 s | 324 MiB |
| Gemma 4 12B IT QAT | 5.41 tok/s | 29.42 s | 385 MiB |
| Gemma 4 26B-A4B IT | 8.16 tok/s | 4.61 s | 1840 MiB |
| Qwen3.6 35B-A3B | 6.76 tok/s | 6.39 s | 1418 MiB |
| GPT-OSS 20B | 2.19 tok/s | 10.44 s | 2217 MiB |
| GPT-OSS 120B | 0.16 tok/s | 31.34 s | 1870 MiB |
| MiniMax M2.7 4-bit | 0.26 tok/s | 49.79 s | 2689 MiB |
| Qwen3.8 Flash Next 4-bit | 0.54 tok/s | 31.87 s | 1295 MiB |

## Photo smoke checks

The same packaged runner was tested with one local photo on all 6 image-compatible models. 6/6 responses passed the glasses-and-towel keyword smoke check, with responses also reviewed manually. E2B needed a 512-token rerun after the initial 128-token cap; the other photo runs used 128 tokens. This is a single-image check, not a general vision accuracy score. The photo and raw responses are kept out of the repository.

| Model | Status |
| --- | --- |
| Gemma 4 E2B IT | passed |
| Gemma 4 E4B IT | passed |
| Gemma 4 12B IT QAT | passed |
| Gemma 4 26B-A4B IT | passed |
| Qwen3.6 35B-A3B | passed |
| Qwen3.8 Flash Next 4-bit | passed |

## Reproduction

```sh
python3 Scripts/validate_release_models.py \
  --app dist/v5.0.0-release-public/TUFF.app \
  --model-root "$HOME/Library/Application Support/TUFF/Models" \
  --image /path/to/photo.jpeg \
  --output benchmark-results/release-validation
```

Packaged CLI SHA-256: `ab185100ef5a5b60491b17766322977393bfce998892a87099bda44f65093c70`.

Sources fingerprint: `5a8b676d592eabd3ffa7cd6badbae3dd4c34318de23e36d407a7524e89aa5e05`.

Raw evidence is retained locally under `benchmark-results/v5.0.0-validation/`. The fingerprint identifies the compiled source tree; the recorded benchmark base commit predates the release commit.
