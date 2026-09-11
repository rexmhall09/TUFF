#!/usr/bin/env python3
"""Capture golden activations for the qwen4_exp architecture.

TUFF's runner is hand-written Metal, and three of Qwen3.8 Flash Next's
mechanisms — four-stream hyper-connections, the hashed n-gram PLE table, and
Qwen Sparse Attention — have no counterpart anywhere else in the engine. A
synthetic fixture in the style of `QwenToySynthetic.swift` can show that the
plumbing holds together, but it cannot say whether the four streams are mixed
in the right order. That needs a reference.

This runs the checkpoint through mlx-vlm's `qwen4_exp` implementation and dumps
the activations at every boundary a TUFF kernel has to reproduce, as a flat
float32 blob plus a JSON index that Swift can read without a NumPy reader.

    python3 Scripts/capture_qwen4exp_golden.py --out golden/

Two modes:

`--mode modules` (the default) loads the weights of a single module out of the
checkpoint's safetensors, runs the reference implementation of that module over
recorded synthetic inputs, and captures the result. It costs a few megabytes,
so it runs anywhere, and it is what validates an individual kernel.

`--mode full` loads the whole model and captures a prefill pass plus a cached
decode step, because TUFF runs those through different code paths. This needs a
machine that can hold the checkpoint: on a 16 GB Mac it is killed during load
(SIGKILL, exit 137) well before the first token, since the reference has no
bounded expert cache and wires all 103 GiB. Bring `prepare_external_ple_model`
from the reference's `ple_storage` if you want the n-gram table out of the
parameter set; that still leaves 70 GiB of routed experts.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

DEFAULT_MODEL = "mlx-community/Qwen3.8-Flash-Next-4bit"
DEFAULT_REVISION = "07b5dc6c54600a359b87f1e53e7adf6351c72a2c"

# Short, fixed, and boring on purpose: the point is a reproducible token
# sequence long enough to exercise several layers of cache, not a good prompt.
DEFAULT_PROMPT = "The capital of France is Paris, and the capital of Japan is"

# Layers worth recording in full. 0 and 2 are gated DeltaNet, 1 carries the
# n-gram PLE module, 3 is the first full-attention layer, and 47 is the last
# layer before the final mixer.
DEFAULT_LAYERS = (0, 1, 2, 3, 47)


class Recorder:
    """Collects named float32 arrays in call order."""

    def __init__(self, layers):
        self.layers = set(layers)
        self.entries = []
        self.current_layer = None
        self.phase = "prefill"
        self._residual_calls = 0

    def want(self, layer):
        return layer is None or layer in self.layers

    def add(self, name, array):
        import mlx.core as mx

        value = mx.array(array).astype(mx.float32)
        mx.eval(value)
        self.entries.append((f"{self.phase}.{name}", value))

    def enter_layer(self, layer):
        self.current_layer = layer
        self._residual_calls = 0

    def next_residual_slot(self):
        slot = self._residual_calls
        self._residual_calls += 1
        return "attn" if slot == 0 else "mlp"


def install_hooks(module_ns, recorder):
    """Wrap the reference modules so each call records its tensors.

    Patching the classes rather than the instances keeps this working whether
    the model was built eagerly or lazily, and means a module added in a later
    mlx-vlm release simply goes unrecorded instead of raising.
    """
    originals = {}

    def patch(cls, name, wrapper_factory):
        if cls is None or not hasattr(cls, name):
            print(f"  note: {cls} has no {name}; skipping that hook")
            return
        original = getattr(cls, name)
        originals[(cls, name)] = original
        setattr(cls, name, wrapper_factory(original))

    decoder = getattr(module_ns, "Qwen4ExpDecoderLayer", None)
    residual = getattr(module_ns, "Qwen4ExpGatedResidual", None)
    ple = getattr(module_ns, "Qwen4ExpPLELayer", None)
    ngram = getattr(module_ns, "Qwen4ExpNGramEmbedding", None)

    layer_counter = {"next": 0}

    def decoder_wrapper(original):
        def wrapped(self, hidden_states, *args, **kwargs):
            index = getattr(self, "_tuff_layer_index", None)
            if index is None:
                index = layer_counter["next"]
                layer_counter["next"] += 1
                self._tuff_layer_index = index
            recorder.enter_layer(index)
            if recorder.want(index):
                recorder.add(f"layer{index:02d}.input", hidden_states)
            out = original(self, hidden_states, *args, **kwargs)
            if recorder.want(index):
                recorder.add(f"layer{index:02d}.output", out)
            recorder.current_layer = None
            return out

        return wrapped

    def residual_wrapper(original):
        def wrapped(self, hyper_input, *args, **kwargs):
            layer = recorder.current_layer
            out = original(self, hyper_input, *args, **kwargs)
            if recorder.want(layer):
                # The model-level mixer runs outside any decoder layer and
                # returns the mixed stream alone.
                tag = (
                    "mixer"
                    if layer is None
                    else f"layer{layer:02d}.{recorder.next_residual_slot()}_hc"
                )
                if isinstance(out, tuple):
                    mixed, hyper, injection = out
                    recorder.add(f"{tag}.mixed", mixed)
                    recorder.add(f"{tag}.injection_weights", injection)
                else:
                    recorder.add(f"{tag}.mixed", out)
            return out

        return wrapped

    def ple_wrapper(original):
        def wrapped(self, hidden_states, *args, **kwargs):
            layer = recorder.current_layer
            out = original(self, hidden_states, *args, **kwargs)
            if recorder.want(layer):
                recorder.add(f"layer{layer:02d}.ple.output", out)
            return out

        return wrapped

    def ngram_wrapper(original):
        def wrapped(self, input_ids, *args, **kwargs):
            out = original(self, input_ids, *args, **kwargs)
            recorder.add("ngram.output", out)
            return out

        return wrapped

    patch(decoder, "__call__", decoder_wrapper)
    patch(residual, "__call__", residual_wrapper)
    patch(ple, "__call__", ple_wrapper)
    patch(ngram, "__call__", ngram_wrapper)
    return originals


def restore_hooks(originals):
    for (cls, name), original in originals.items():
        setattr(cls, name, original)


def snapshot_directory(model, revision):
    """Resolve the local snapshot without downloading anything."""
    from huggingface_hub import snapshot_download

    return Path(
        snapshot_download(
            model, revision=revision, local_files_only=True,
        )
    )


def load_module_weights(snapshot, prefix):
    """Read just the tensors under `prefix`, from whichever shards hold them."""
    import mlx.core as mx

    index = json.loads((snapshot / "model.safetensors.index.json").read_text())
    wanted = {
        name: shard
        for name, shard in index["weight_map"].items()
        if name.startswith(prefix)
    }
    if not wanted:
        raise SystemExit(f"no tensors under {prefix!r}")

    weights = {}
    for shard in sorted(set(wanted.values())):
        loaded = mx.load(str(snapshot / shard))
        for name, source in wanted.items():
            if source == shard:
                weights[name[len(prefix):].lstrip(".")] = loaded[name]
    return weights


def capture_gated_residual(recorder, snapshot, layer, args):
    """Run the reference hyper-connection over recorded synthetic input.

    The four-stream residual is the mechanism with no counterpart anywhere else
    in the engine, and the arithmetic around it is easy to get subtly wrong —
    the stream count divides before each activation, the injection gate carries
    a factor of two, and the residual that survives is the unnormalized input.
    Running the reference module over inputs TUFF can replay pins all of that
    down against the checkpoint's own weights.
    """
    import mlx.core as mx
    import mlx.nn as nn
    from mlx_vlm.models.qwen4_exp import config as ref_config
    from mlx_vlm.models.qwen4_exp import language as ref_language

    raw = json.loads((snapshot / "config.json").read_text())
    text_config = ref_config.TextConfig.from_dict(raw["text_config"])
    quantization = raw.get("quantization", {})
    group_size = int(quantization.get("group_size", 32))
    bits = int(quantization.get("bits", 4))

    for slot in ("attn", "mlp"):
        prefix = (
            f"language_model.model.layers.{layer}.{slot}_hyper_connection"
        )
        weights = load_module_weights(snapshot, prefix)
        module = ref_language.Qwen4ExpGatedResidual(text_config)
        nn.quantize(module, group_size=group_size, bits=bits)
        module.load_weights(list(weights.items()))
        module.eval()

        streams = text_config.hc_count
        hidden = text_config.hidden_size
        key = mx.random.key(0xC0FFEE + layer)
        hyper = mx.random.normal((1, args.positions, streams * hidden), key=key)
        hyper = hyper.astype(mx.float16)

        tag = f"layer{layer:02d}.{slot}_hc"
        recorder.add(f"{tag}.input", hyper)
        mixed, passthrough, injection = module(hyper)
        recorder.add(f"{tag}.mixed", mixed)
        recorder.add(f"{tag}.injection_weights", injection)
        # The residual has to leave the block untouched; recording it makes a
        # runner that accidentally forwards the normalized stream fail here.
        recorder.add(f"{tag}.passthrough", passthrough)
        recorder.add(f"{tag}.hc_norm_weight", module.hc_norm.weight)

        # The steps between the projections, so each TUFF kernel can be checked
        # on its own against real inputs rather than only end to end. Without
        # these, a test would have to reproduce the quantized projections
        # before it could reach the arithmetic actually under test.
        normed = module.hc_norm(hyper)
        down = module.input_mix_weight_down(normed)
        lowrank = nn.silu(down / streams)
        up = module.input_mix_weight_up(lowrank)
        injection_raw = module.block_inject_weight(normed)
        for name, value in (
            ("normed", normed),
            ("mix_down", down),
            ("mix_lowrank", lowrank),
            ("mix_up", up),
            ("injection_raw", injection_raw),
        ):
            recorder.add(f"{tag}.{name}", value)

        print(f"  captured {tag}: mixed{tuple(mixed.shape)} "
              f"injection{tuple(injection.shape)}")


def write_bundle(entries, token_ids, metadata, out_dir):
    """Write one float32 blob plus a JSON index describing it."""
    out_dir.mkdir(parents=True, exist_ok=True)
    blob_path = out_dir / "golden.bin"
    index_path = out_dir / "golden.json"

    index = {
        "version": 1,
        "tokenIds": [int(t) for t in token_ids],
        "dtype": "float32",
        "tensors": [],
    }
    index.update(metadata)

    import numpy as np

    offset = 0
    with blob_path.open("wb") as blob:
        for name, value in entries:
            payload = np.array(value, dtype=np.float32, copy=False).ravel().tobytes()
            blob.write(payload)
            index["tensors"].append(
                {
                    "name": name,
                    "shape": [int(d) for d in value.shape],
                    "offset": offset,
                    "byteCount": len(payload),
                }
            )
            offset += len(payload)

    index_path.write_text(json.dumps(index, indent=2, sort_keys=True))
    return blob_path, index_path, offset


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--revision", default=DEFAULT_REVISION)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--out", default="golden-qwen4exp", type=Path)
    parser.add_argument(
        "--layers",
        default=",".join(str(layer) for layer in DEFAULT_LAYERS),
        help="comma-separated decoder layers to record in full",
    )
    parser.add_argument(
        "--mode",
        choices=("modules", "full"),
        default="modules",
        help="'modules' captures one module from its own weights (cheap); "
             "'full' loads the whole checkpoint and runs a forward pass",
    )
    parser.add_argument(
        "--positions",
        type=int,
        default=8,
        help="synthetic positions per module capture",
    )
    parser.add_argument(
        "--decode-steps",
        type=int,
        default=1,
        help="cached decode steps to capture after the prefill pass",
    )
    args = parser.parse_args()

    try:
        import mlx.core as mx
        from mlx_vlm import load
        from mlx_vlm.models import qwen4_exp
    except ImportError as error:
        print(f"error: {error}", file=sys.stderr)
        print("install with: pip install mlx-vlm", file=sys.stderr)
        return 1

    layers = tuple(int(v) for v in args.layers.split(",") if v.strip())

    if args.mode == "modules":
        snapshot = snapshot_directory(args.model, args.revision)
        print(f"module capture from {snapshot}")
        recorder = Recorder(layers)
        recorder.phase = "module"
        for layer in layers:
            capture_gated_residual(recorder, snapshot, layer, args)
        metadata = {
            "model": args.model,
            "revision": args.revision,
            "mode": args.mode,
            "recordedLayers": list(layers),
            "positions": args.positions,
        }
        blob, index, total = write_bundle(recorder.entries, [], metadata, args.out)
        print(f"wrote {len(recorder.entries)} tensors, {total/1e6:.2f} MB")
        print(f"  {blob}")
        print(f"  {index}")
        return 0

    print(f"loading {args.model} @ {args.revision[:8]}")
    # `lazy=True` leaves the parameters unevaluated and memory-mapped instead
    # of wiring all 103 GiB up front. On a machine smaller than the checkpoint
    # this is the difference between paging in what a token touches and not
    # loading at all.
    model, processor = load(args.model, revision=args.revision, lazy=True)

    tokenizer = getattr(processor, "tokenizer", processor)
    token_ids = tokenizer.encode(args.prompt)
    print(f"prompt is {len(token_ids)} tokens: {token_ids}")

    recorder = Recorder(layers)
    originals = install_hooks(qwen4_exp.language, recorder)
    try:
        prompt = mx.array([token_ids])

        # Prefill over the whole prompt. QSA keeps an auxiliary index-key cache
        # alongside the ordinary KV cache, so let the model build its own
        # rather than assuming the generic layout.
        recorder.phase = "prefill"
        language = model.language_model
        if hasattr(language, "make_cache"):
            cache = language.make_cache()
        else:
            from mlx_lm.models.cache import make_prompt_cache

            cache = make_prompt_cache(model)
        logits = language(prompt, cache=cache)
        if hasattr(logits, "logits"):
            logits = logits.logits
        mx.eval(logits)
        recorder.add("logits", logits)
        next_token = int(mx.argmax(logits[0, -1]).item())
        print(f"prefill argmax -> {next_token} {tokenizer.decode([next_token])!r}")

        # Cached decode, which TUFF runs through a different path entirely.
        for step in range(args.decode_steps):
            recorder.phase = f"decode{step}"
            step_logits = language(mx.array([[next_token]]), cache=cache)
            if hasattr(step_logits, "logits"):
                step_logits = step_logits.logits
            mx.eval(step_logits)
            recorder.add("logits", step_logits)
            next_token = int(mx.argmax(step_logits[0, -1]).item())
            print(f"decode{step} argmax -> {next_token} {tokenizer.decode([next_token])!r}")
    finally:
        restore_hooks(originals)

    metadata = {
        "model": args.model,
        "revision": args.revision,
        "prompt": args.prompt,
        "recordedLayers": list(layers),
        "decodeSteps": args.decode_steps,
    }
    blob, index, total = write_bundle(
        recorder.entries, token_ids, metadata, args.out
    )
    print(f"wrote {len(recorder.entries)} tensors, {total/1e6:.1f} MB")
    print(f"  {blob}")
    print(f"  {index}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
