# Tensor Core Transition

This is the big conceptual jump.

CUDA cores are general. Tensor Cores are specialized matrix engines.

## Why Tensor Cores matter

Deep learning is mostly:

- matrix multiply
- convolution lowered to matmul-like work
- attention matmuls
- batched GEMMs

So NVIDIA adds hardware that screams at matrix multiply.

## Volta

- first Tensor Cores
- FP16 inputs, mixed accumulation path
- WMMA becomes the CUDA-facing way to use them manually

## Turing

- Tensor Cores move into RTX/inference world too
- INT8/INT4 inference story grows

## Later arc

- Ampere: TF32/BF16/sparsity
- Hopper: FP8
- Blackwell: FP4/NVFP4/microscaling

Checklist:

- [ ] understand MMA
- [ ] understand fragment/tile idea
- [ ] understand why layouts matter
- [ ] understand accumulation precision
- [ ] understand why libraries usually beat handwritten Tensor Core code

