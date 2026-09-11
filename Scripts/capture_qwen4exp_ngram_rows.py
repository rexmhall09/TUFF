"""Golden n-gram row ids straight from mlx-vlm's own hash.

Builds a Qwen4ExpNGramEmbedding without its 30 GiB table by bypassing
__init__, then feeds it the constants the checkpoint actually stores. The row
computation that runs is the reference's, not a transcription of it.
"""
import json, sys
from pathlib import Path
import mlx.core as mx
from mlx_vlm.models.qwen4_exp import language as L

snap = Path(sys.argv[1])
cfg = json.loads((snap / "config.json").read_text())["text_config"]

mult = [23703573157769, 20109073645365, 8052911324071]
offsets = [0, 20000003, 40000026, 60000059, 80000106, 100000165, 120000228,
           140000297, 160000374, 180000455, 200000548, 220000655, 240000802,
           260000955, 280001114, 300001275]
sizes = [20000003, 20000023, 20000033, 20000047, 20000059, 20000063, 20000069,
         20000077, 20000081, 20000093, 20000107, 20000147, 20000153, 20000159,
         20000161, 20000171]

m = L.Qwen4ExpNGramEmbedding.__new__(L.Qwen4ExpNGramEmbedding)
nn_base = type(m).__mro__[1]
nn_base.__init__(m)
m.ngram_size = cfg["ngram_size"]
m.context_len = m.ngram_size - 1
m.heads_per_ngram = cfg["heads_per_ngram"]
m.ngram_heads = m.context_len * m.heads_per_ngram
m.eos_token_id = cfg["eos_token_id"]
m.layer_multipliers = mx.array(mult, dtype=mx.int64)
m.ngram_heads_offsets = mx.array(offsets, dtype=mx.int64)
m.ngram_heads_vocab_sizes = mx.array(sizes, dtype=mx.int64)

captured = {}
def fake_embedding(ids):
    captured["rows"] = ids
    return mx.zeros((*ids.shape, 160))
m.ngram_embedding = fake_embedding

tokens = [101, 202, 303, 248044, 404, 505]
m(mx.array([tokens]), None)
rows = captured["rows"].tolist()[0]
print(json.dumps({
    "eosTokenId": m.eos_token_id, "ngramSize": m.ngram_size,
    "headsPerNgram": m.heads_per_ngram, "multipliers": mult,
    "headOffsets": offsets, "headVocabSizes": sizes,
    "tokens": tokens, "rows": rows,
}))
