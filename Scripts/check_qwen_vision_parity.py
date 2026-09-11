"""Compare a Qwen vision capture with mlx-vlm using installed pack weights.

The capture comes from Qwen38VisionParityTests; see docs/QWEN38_UPDATE_VALIDATION.md.
Requires numpy and mlx-vlm. No downloads or model mutation are performed.
"""
import argparse
import json
from pathlib import Path

import mlx.core as mx
import numpy as np
from mlx_vlm.models.qwen4_exp.config import VisionConfig
from mlx_vlm.models.qwen4_exp.vision import VisionModel


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pack", type=Path)
    parser.add_argument("capture", type=Path)
    parser.add_argument("checkpoint_config", type=Path)
    args = parser.parse_args()

    config = json.loads(args.checkpoint_config.read_text())["vision_config"]
    model = VisionModel(VisionConfig.from_dict(config))
    manifest = json.loads((args.pack / "manifest.json").read_text())
    weights = []
    for item in manifest["tensors"]:
        raw = np.fromfile(
            args.pack / item["file"], dtype=np.uint16,
            count=item["size"] // 2, offset=item["offset"],
        ).reshape(item["shape"])
        value = mx.array(raw).view(mx.bfloat16)
        weights.append((item["name"].removeprefix("vision_tower."), value))
    model.load_weights(weights)
    model.eval()

    shape = json.loads((args.capture / "shape.json").read_text())
    temporal = config["temporal_patch_size"]
    patch = config["patch_size"]
    channels = config["in_channels"]
    raw = np.fromfile(args.capture / "patches.bf16", dtype=np.uint16)
    raw = raw.reshape(-1, temporal, patch, patch, channels)
    # TUFF exports THWC patch vectors. The reference API accepts CTHW and
    # transposes them internally to the same Conv3D physical order.
    patches = raw.transpose(0, 4, 1, 2, 3).copy().reshape(
        -1, temporal * patch * patch * channels)
    pixels = mx.array(patches).view(mx.bfloat16)
    grid = mx.array([[1, shape["height"], shape["width"]]])
    features = model(pixels, grid)
    if isinstance(features, tuple):
        features = features[0]
    mx.eval(features)

    expected = np.asarray(features.astype(mx.float32)).reshape(-1)
    actual = np.fromfile(args.capture / "features.f16", dtype=np.float16).astype(np.float32)
    if actual.shape != expected.shape:
        raise SystemExit(f"Feature shape mismatch: {actual.shape} vs {expected.shape}")
    relative_error = float(np.linalg.norm(actual - expected) / np.linalg.norm(expected))
    cosine = float(np.dot(actual, expected) / (np.linalg.norm(actual) * np.linalg.norm(expected)))
    print(json.dumps({
        "shape": list(features.shape),
        "relative_error": relative_error,
        "cosine_similarity": cosine,
    }), flush=True)
    expected.tofile(args.capture / "reference.f32")
    if not (relative_error < 0.08 and cosine > 0.995):
        raise SystemExit("Vision features diverge from the reference")


if __name__ == "__main__":
    main()
