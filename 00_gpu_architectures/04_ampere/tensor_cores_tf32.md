# Ampere Tensor Cores, TF32, BF16

Ampere is where Tensor Cores become easier to hit from normal deep learning code.

The key thing is TF32.

## TF32

TF32 is basically NVIDIA saying:

> lots of FP32 deep learning math does not need full FP32 mantissa, so route it through Tensor Cores.

So code can look FP32-ish, but the math path can use Tensor Cores.

This is why A100 was such a big jump for training workloads.

## BF16

BF16 keeps FP32-like exponent range with fewer mantissa bits.

Useful because:

- easier training stability than pure FP16 in many cases
- Tensor Core acceleration
- common in ML frameworks

## Structured sparsity

Ampere Tensor Cores can exploit fine-grained structured sparsity.

But important:

> sparsity speedup is not free. The model/weights must obey the structure and software must use the sparse path.

## Checklist

- [ ] understand TF32 vs FP32
- [ ] understand BF16 vs FP16
- [ ] understand structured sparsity at a high level
- [ ] check how PyTorch/cuBLAS enable TF32
- [ ] compare SGEMM with TF32-allowed vs strict FP32
- [ ] understand accuracy tradeoffs

Sources:

- https://developer.nvidia.com/blog/nvidia-ampere-architecture-in-depth/
- https://docs.nvidia.com/cuda/ampere-tuning-guide/index.html

