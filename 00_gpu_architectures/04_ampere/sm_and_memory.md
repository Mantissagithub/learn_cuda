# Ampere SM And Memory Notes

The main thing I need to internalize: Ampere makes data staging more pipeline-ish.

Normal beginner CUDA:

1. load from global
2. store to shared
3. sync
4. compute

Ampere optimized CUDA:

1. async-copy tile from global to shared
2. compute on older tile
3. overlap these so Tensor Cores don't starve

## A100 path

- `sm_80`
- A100/A30
- TF32 Tensor Cores
- BF16
- structured sparsity
- 192 KB L1/shared per SM on A100
- 40 MB L2 on A100
- HBM2
- MIG

## RTX / GA10x path

- `sm_86`
- RTX 30, A10, A40, RTX A-series
- still Ampere, but not GA100
- good local learning target if you have RTX 30-ish hardware

## Checklist

- [ ] Write normal shared-memory tiled matmul.
- [ ] Find `cp.async` in CUTLASS.
- [ ] Compare sync-copy tiling vs async-copy tiling mentally.
- [ ] Check TF32 through cuBLAS.
- [ ] Learn what "MIG" partitions.
- [ ] Stop saying "Ampere" when I actually mean `sm_80` or `sm_86`.

## Sources

- https://developer.nvidia.com/blog/nvidia-ampere-architecture-in-depth/
- https://docs.nvidia.com/cuda/ampere-tuning-guide/index.html
- https://developer.nvidia.com/cuda/gpus

