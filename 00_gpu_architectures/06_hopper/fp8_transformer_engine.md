# FP8 And Transformer Engine

Hopper adds FP8, and this is one of the big reasons H100 was aimed so hard at transformers.

## FP8 formats

Hopper talks about:

- E4M3
- E5M2

Very roughly:

- E4M3 = more mantissa, less range
- E5M2 = more range, less mantissa

So this is not just "use smaller float". Need scaling and format choice.

## Transformer Engine

Transformer Engine is the software/hardware path that helps choose FP8 vs higher precision in transformer layers.

The point:

> keep accuracy acceptable while getting FP8 speed and memory savings.

## What to study

- [ ] what tensors become FP8
- [ ] where scaling happens
- [ ] what accumulates in higher precision
- [ ] how frameworks expose this
- [ ] how TensorRT-LLM / Transformer Engine use it
- [ ] what cuBLASLt supports

Sources:

- https://developer.nvidia.com/blog/nvidia-hopper-architecture-in-depth/
- https://docs.nvidia.com/cuda/parallel-thread-execution/index.html

