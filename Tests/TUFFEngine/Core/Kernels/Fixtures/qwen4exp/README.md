# qwen4_exp golden vectors

Captured from `mlx-community/Qwen3.8-Flash-Next-4bit` at revision
`07b5dc6c54600a359b87f1e53e7adf6351c72a2c`, by running mlx-vlm's reference
`Qwen4ExpGatedResidual` over recorded synthetic input using the checkpoint's
own layer-0 weights.

`golden.bin` is a flat float32 blob; `golden.json` names each tensor and gives
its shape, byte offset and length. The intermediate tensors — `normed`,
`mix_down`, `mix_lowrank`, `mix_up`, `injection_raw` — are there so each kernel
can be checked on its own without first reproducing the quantized projections
around it.

Regenerate with:

    python3 Scripts/capture_qwen4exp_golden.py \
        --mode modules --layers 0 --positions 2 \
        --out Tests/TUFFEngine/Core/Kernels/Fixtures/qwen4exp

That needs mlx-vlm and a local copy of the checkpoint, but only reads a few
megabytes of it, so it runs on any Mac.
