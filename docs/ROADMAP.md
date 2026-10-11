# Roadmap

What I'm working toward and where help would go furthest. It's a plan, not a
promise. If something here matters to you, say so in
[Discussions](https://github.com/rexmhall09/TUFF/discussions); it really does
change what I work on next.

## Known limits

- **Search is untested on GPT-OSS 120B and MiniMax M2.7.**
- **Small-block prefill is off by default** until it's shown to speed up real
  requests.

## Next

- **Benchmarks from lots of Macs.** Everything so far was measured on one
  16 GB M2. Results from 8 GB Macs, Pro and Max chips and M3 to M5 will tell
  me whether the memory floors and Auto settings are right.
- **Tested setups for more agents and editors**, beyond OMP.

## Maybe later

- **New models.** New MoE releases are a natural fit. Each one is a big job,
  but it splits into pieces.
- **Multi-token prediction** for models that ship prediction heads.
- **Safer page fetching**, using ideas from the sandboxed research tool in
  [Discussion #4](https://github.com/rexmhall09/TUFF/discussions/4).

## Where help counts most

1. **Run the benchmark** on your Mac and share it. Five minutes, no code.
2. **Pick up a [good first issue](https://github.com/rexmhall09/TUFF/issues?q=is%3Aissue%20is%3Aopen%20label%3A%22good%20first%20issue%22).**
3. **Try search** on GPT-OSS 120B or MiniMax if your Mac can run them.
4. **Polish the app.** Accessibility and small UI fixes are easy to review.
5. **Help with a model or faster prompts** if you like engine work.
   Open an issue first so we can plan it.
