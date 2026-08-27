# Contributing to TUFF

All contributions that make TUFF better are welcome.

That includes code, Metal kernels, app design, icons, accessibility, model
support, tests, bug reports, documentation, benchmark results, installation
feedback, and small quality-of-life fixes. You do not need to be a Swift or
Metal expert to help.

## Start where you can help

Useful contributions include:

- fixing a reproducible bug or confusing error
- improving Chat, Models, Server, or Settings behavior
- making the Mac app clearer, faster, or more accessible
- adding CPU references, toy fixtures, format tests, or failure cases
- qualifying a supported model on another Apple Silicon Mac
- proposing a model with a pinned source and realistic runtime path
- improving installation, server, architecture, or troubleshooting docs
- reviewing code, designs, benchmark evidence, or model behavior

Small fixes do not need an elaborate proposal. For larger architecture, format,
model, or interface work, open an issue or draft pull request early enough for
the direction to be discussed before the implementation becomes expensive.

## Technical guardrails

TUFF supports macOS 15, Swift 6.2, Metal 3.2, and Apple Silicon. Newer Metal
paths must remain optional and keep a tested Metal 3.2 fallback.

Preserve these invariants:

- Model installation and inference stay bounded in memory.
- The installer never stages a second full source checkpoint.
- Existing compatible `.gturbo` v1 installations remain readable.
- Unfamiliar format features fail clearly instead of being misread.
- Image input fails closed. An image is never accepted and silently ignored.
- The app, CLI, server, tests, and other model processes do not run
  concurrently against one machine during real-model work.
- The loopback server stays on `127.0.0.1`. It has no remote authentication or
  TLS and must not be exposed through a proxy or tunnel.
- No model is described as working until it passes the relevant real-model
  qualification.

Do not add an undocumented runtime switch, silently change a production
default, commit model weights, duplicate a local model, or purge someone else's
download state to make a test pass.

## Tests

Behavior changes should include a focused test in the same commit. Match the
test to the layer being changed:

- Metal kernels need an independent CPU reference and boundary coverage.
- Model families need toy forward, prefill, tokenizer, prompt, and format tests.
- Installer work needs resume, cancellation, discard, corruption, fingerprint,
  disk, RAM, and legacy-install coverage where applicable.
- Chat work needs persistence, attachment lifetime, restoration, model binding,
  context, and structured-output coverage.
- Server work needs queue, cancellation, contention, shutdown, and ingress
  coverage.
- UI work needs keyboard, VoiceOver, reduced-transparency, light and dark, and
  small-window review in proportion to the change.

Run package tests through the canonical serial runner:

```bash
Scripts/test.sh
```

Before requesting review, also run the checks that apply to your change:

```bash
swift build -c release
ruby Scripts/check_markdown_links.rb
ruby Scripts/check_brand_assets.rb
ruby Scripts/check_app_version.rb
```

If your change affects packaging, verify the archive rather than only the build
directory:

```bash
Scripts/package_app.sh 2.0.0
```

State exactly what you ran and what you did not run. A model-free green suite
does not prove that a real checkpoint works.

## Real-model changes

Before a model run, require:

- macOS 15 or newer
- Swift 6.2 or newer
- enough free disk
- acceptable `memory_pressure -Q`
- a complete verified `.gturbo` installation
- no TUFF app, CLI, server, decode service, model test, MLX process, or other
  local-model process already running

Run one model process at a time. Do not terminate someone else's process or
delete or reinstall their model to clear a preflight failure.

A new family or checkpoint needs evidence for the complete path:

1. Pin the repository, revision, source-index fingerprint, storage, and model
   identity in the shared registry.
2. Add architecture and format validation before runtime execution.
3. Compare primitives with independent CPU references.
4. Compare toy forward and prefill output with an independent implementation.
5. Add tokenizer and prompt-template goldens.
6. Verify repack output, resume, cancellation, and corruption handling.
7. Run the installed checkpoint and record coherent output, stop reason, and
   peak working set on qualifying hardware.
8. Set hardware and context gates from measured evidence.

If the available Mac cannot qualify the model, leave it unavailable and say
what real run is still required.

## Benchmark contributions

Run `Scripts/benchmark_simple.rb` for the launch-lineup numbers, or
`Scripts/benchmark_v2.rb` for the per-workload matrix, and report what the tool
prints. Include the commit, Mac model, unified memory, macOS, Swift version,
exact command, prompt and generated token counts, stop reason, prefill, time to
first token, decode rate, and memory. A repeating calibration prompt is not a
valid speed result, because repeated expert choices make decode artificially
fast.

Do not turn a smoke test, warmup, profiler run, contaminated session, or
model-free test into a performance claim. Review every captured file before
sharing it and remove personal paths or unrelated process details.

## Design and documentation

Design work is as welcome as runtime work. Keep TUFF's interface native, clean,
keyboard-usable, readable in light and dark appearance, and understandable
without knowing the runtime's internal vocabulary.

Documentation should describe behavior that exists and evidence that was
actually collected. Prefer direct language. Avoid generic marketing filler,
unverified compatibility claims, and benchmark conclusions broader than the
workload supports.

## AI-assisted contributions

AI-assisted contributions are 100% welcome.

In the pull request, include a short AI assistance note that:

- names the tool and model when known
- explains its role, such as exploration, implementation, tests, review, or
  documentation
- identifies any substantial generated or rewritten areas
- confirms that you reviewed and understood the submitted changes
- lists the tests and real-model checks you personally verified

You remain responsible for the contribution. Do not submit raw generated code
that you cannot explain, unreviewed model output, invented test results,
fabricated citations, leaked credentials, or private prompt content. Check
licenses and source terms before bringing generated assets, model files, or
third-party material into the repository.

A concise note is enough. The goal is honest attribution and accountable
review, not line-by-line labeling.

## Commits and pull requests

Keep commits focused and include their corresponding tests. A larger feature
can use several small commits that separately establish foundations, behavior,
interface work, and documentation.

A pull request should explain:

- what changed and why
- the important design or compatibility decisions
- tests and real-model checks run
- screenshots for visible app changes when possible
- remaining limitations or hardware that was not available
- AI assistance, if any

Preserve unrelated work in the branch and avoid committing generated build
products, downloaded checkpoints, personal benchmark data, or secrets.

By contributing, you agree that your contribution is licensed under the
repository's [Apache License 2.0](LICENSE).
