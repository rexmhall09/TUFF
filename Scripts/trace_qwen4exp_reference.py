#!/usr/bin/env python3
"""Layer-by-layer reference trace for qwen4_exp, streaming weights from disk.

mlx-vlm's own loader wires all 103 GiB of Qwen3.8 Flash Next and is SIGKILLed
on a 16 GB Mac, which is why `capture_qwen4exp_golden.py --mode full` cannot
run here. This builds one decoder layer at a time, loads only that layer's
tensors, runs it, records the output and drops it — so the whole 48-layer
forward fits in a few gigabytes and takes about two minutes.

What that buys is a per-layer trace to bisect TUFF against. Three bugs came
out of it that no synthetic fixture could have caught, because each one
produced plausible output on uniform toy weights:

  * the gated-DeltaNet input projection dequantizing INT4 at group 64 on a
    group-32 checkpoint (`gdn_in_proj_gemv_simd` built with no function
    constants), and the same in the packed q/k/v GEMV;
  * the DeltaNet output gate applying silu where this architecture sets
    `output_gate_type: sigmoid`;
  * the full-attention layer reading `normed` — all four hyper-connection
    streams stacked — instead of the stream mixture, so it saw stream 0.

Usage:

    python3 Scripts/trace_qwen4exp_reference.py SNAPSHOT OUT.json 760,6511,314,9338,369

`SNAPSHOT` is the HF snapshot directory. The token ids are a prompt already
tokenized, so the Swift side can replay exactly the same sequence — feed them
to `RealForwardRunner.produce` and capture the residual through
`debugLayerSink`, which fires once before each layer and once after the last.
`tensors["layerNN.out"]` is `[1, positions, hc_count * hidden_size]`; take the
slice for the position under test.

Requires mlx-vlm in the environment (0.6.17 works; so does the newer source).
"""
import json, sys, gc
from pathlib import Path
import numpy as np
import mlx.core as mx
import mlx.nn as nn
from mlx_vlm.models.qwen4_exp import config as C, language as L
from mlx_vlm.models.cache import ArraysCache

snap = Path(sys.argv[1])
out_path = Path(sys.argv[2])
prompt_ids = [int(v) for v in sys.argv[3].split(",")]

raw = json.loads((snap / "config.json").read_text())
cfg = C.TextConfig.from_dict(raw["text_config"])
qcfg = raw.get("quantization", {})
BASE_GS, BASE_BITS = int(qcfg.get("group_size", 32)), int(qcfg.get("bits", 4))

index = json.loads((snap / "model.safetensors.index.json").read_text())["weight_map"]
_open = {}
def shard(name):
    if name not in _open:
        _open[name] = mx.load(str(snap / name))
    return _open[name]

def load_prefix(prefix):
    want = {k: v for k, v in index.items() if k.startswith(prefix)}
    return {k[len(prefix):].lstrip("."): shard(v)[k] for k, v in want.items()}

def quantize(module, predicate):
    nn.quantize(module, group_size=BASE_GS, bits=BASE_BITS,
                class_predicate=predicate)

def layer_predicate(path, module):
    # nn.quantize raises on any module it cannot quantize, so the predicate
    # has to exclude them itself — the PLE layer carries a Conv1d.
    if not hasattr(module, "to_quantized"):
        return False
    if path.endswith("mlp.gate") or path.endswith("shared_expert_gate"):
        return {"group_size": 64, "bits": 8, "mode": "affine"}
    return True

record = {}
def f32(value):
    return np.array(mx.array(value).astype(mx.float32), copy=False).ravel()

def note(tag, value):
    a = f32(value)
    record[tag] = {
        "maxAbs": float(np.abs(a).max()), "meanAbs": float(np.abs(a).mean()),
        "head": [float(v) for v in a[:8]],
    }
    print(f"{tag:34s} maxAbs={record[tag]['maxAbs']:10.4f} "
          f"meanAbs={record[tag]['meanAbs']:8.4f} {record[tag]['head'][:4]}",
          flush=True)

ids = mx.array([prompt_ids], dtype=mx.int64)

# --- embeddings -------------------------------------------------------------
embed = nn.QuantizedEmbedding(cfg.vocab_size, cfg.hidden_size,
                              group_size=BASE_GS, bits=BASE_BITS)
embed.load_weights(list(load_prefix("language_model.model.embed_tokens.").items()))
hidden = embed(ids)
mx.eval(hidden)
note("embed", hidden)
del embed
hidden = mx.tile(hidden, (1, 1, cfg.hc_count))

full_dump = {"embed": f32(hidden).tolist()}

# --- layers -----------------------------------------------------------------
model_shell = L.Qwen4ExpModel.__new__(L.Qwen4ExpModel)
caches = []
for i in range(cfg.num_hidden_layers):
    is_linear = cfg.layer_types[i] == "linear_attention"
    has_ple = (i + 1) in cfg.ple_layer_ids
    if is_linear:
        caches.append(ArraysCache(size=4 if has_ple else 2))
    else:
        caches.append(L.QSAKVCache())

fa_idx = next(i for i, t in enumerate(cfg.layer_types) if t != "linear_attention")
ssm_idx = next(i for i, t in enumerate(cfg.layer_types) if t == "linear_attention")
fa_mask = L._create_qwen3_5_attention_mask(hidden, caches[fa_idx])
ssm_mask = L._create_qwen3_5_ssm_mask(hidden, caches[ssm_idx])

for i in range(cfg.num_hidden_layers):
    layer = L.Qwen4ExpDecoderLayer(cfg, i)
    quantize(layer, layer_predicate)
    weights = load_prefix(f"language_model.model.layers.{i}.")
    layer.load_weights(list(weights.items()))
    layer.eval()
    is_linear = cfg.layer_types[i] == "linear_attention"
    hidden = layer(hidden, ids,
                   mask=ssm_mask if is_linear else fa_mask,
                   cache=caches[i], position_ids=None)
    mx.eval(hidden)
    note(f"layer{i:02d}.out", hidden)
    if True:
        full_dump[f"layer{i:02d}.out"] = f32(hidden).tolist()
    del layer, weights
    _open.clear()
    gc.collect()
    mx.clear_cache()

# --- mixer and head ---------------------------------------------------------
mixer = L.Qwen4ExpGatedResidual(cfg, use_combine=False)
quantize(mixer, layer_predicate)
mixer.load_weights(list(
    load_prefix("language_model.model.hyper_connection_mixer.").items()))
mixed = mixer(hidden)
mx.eval(mixed)
note("mixer.out", mixed)
full_dump["mixer.out"] = f32(mixed).tolist()
del mixer
_open.clear(); gc.collect(); mx.clear_cache()

head = nn.QuantizedLinear(cfg.hidden_size, cfg.vocab_size, bias=False,
                          group_size=BASE_GS, bits=BASE_BITS)
head.load_weights(list(load_prefix("language_model.lm_head.").items()))
logits = head(mixed)
mx.eval(logits)
note("logits", logits)

last = f32(logits[0, -1])
order = np.argsort(-last)[:10]
print("TOP:", [(int(t), float(last[t])) for t in order], flush=True)

out_path.write_text(json.dumps({
    "tokenIds": prompt_ids, "stats": record,
    "top": [[int(t), float(last[t])] for t in order],
    "tensors": full_dump,
}))
print("wrote", out_path, flush=True)
