# Turing

Turing is the RTX generation and `sm_75`.

It matters for CUDA because T4 is everywhere, and because it shows NVIDIA splitting graphics acceleration and AI acceleration.

## What to remember

- Tensor Cores continue after Volta
- RT Cores appear for ray tracing
- T4 becomes an important inference GPU
- RTX 20 appears on the consumer/workstation side

## CUDA angle

For raw CUDA, Turing is less dramatic than Hopper. But it matters because:

- Tensor Cores are now not just V100-only datacenter magic
- inference starts showing up everywhere
- mixed precision becomes normal

Checklist:

- [ ] know `sm_75`
- [ ] understand T4 as inference hardware
- [ ] understand Tensor Cores vs RT Cores
- [ ] understand why RTX docs talk less like CUDA docs

Sources:

- https://docs.nvidia.com/cuda/turing-tuning-guide/index.html
- https://developer.nvidia.com/cuda/gpus

