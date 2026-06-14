# Ada Lovelace

Ada is the RTX 40 / L4 / L40 / RTX 6000 Ada generation.

For CUDA, the main thing is:

- public target is mostly `sm_89`
- this is not Hopper
- this is not `sm_90`
- it is more like the RTX/graphics + inference branch after Ampere

## What to remember

- better RT path
- Tensor Cores still matter
- L4/L40 make it relevant in servers too
- local dev machines often have Ada-ish GPUs
- but no Hopper-style TMA/cluster mental model as the main thing

## Checklist

- [ ] Check if my local GPU is `sm_89`.
- [ ] Compare Ada with Ampere GA10x, not with H100 first.
- [ ] Understand why RTX architecture docs talk a lot about graphics.
- [ ] Check what cuBLAS/cuDNN do differently on Ada.
- [ ] Read about L4 as an inference product.

## Files here

- [sm89_notes.md](sm89_notes.md): practical `sm_89` notes
- [rt_tensor_graphics_split.md](rt_tensor_graphics_split.md): RT vs Tensor vs CUDA compute

## Sources

- https://www.nvidia.com/en-us/geforce/ada-lovelace-architecture/
- https://developer.nvidia.com/cuda/gpus
- https://docs.nvidia.com/cuda/cuda-programming-guide/index.html
