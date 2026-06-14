# Vera Rubin Platform Notes

This is the stuff I should not confuse with normal kernel-level CUDA.

Rubin public material is very platform-heavy. That means a lot of the interesting parts may show up through NCCL, TensorRT-LLM, cuBLAS, cuDNN, scheduling, NVLink, and system software before I ever write a raw CUDA kernel for it.

## Pieces

| Piece | Why I care |
| --- | --- |
| Rubin GPU | future accelerator target |
| Vera CPU | host/data movement/coherent platform side |
| NVLink 6 | scale-up fabric |
| NVLink-C2C | CPU-GPU coherent link story |
| ConnectX-9 | scale-out networking |
| BlueField-4 | DPU / infra control plane |
| RAS engine | keeping huge systems alive |

## Current caution

As of the public pages I checked, Rubin does not yet have the same CUDA target clarity as:

- Hopper: `sm_90`
- Blackwell data center: `sm_100` / `sm_103`
- RTX Blackwell: `sm_120`

So don't make up a Rubin `sm_XX`. Wait for CUDA docs.

## Checklist

- [ ] Watch compute capability table for Rubin entries.
- [ ] Watch PTX ISA release notes for Rubin targets.
- [ ] Watch CUTLASS for Rubin kernels.
- [ ] Watch NCCL/NVLink docs for topology changes.
- [ ] Watch Nsight Compute for new metrics.
- [ ] Separate rack-level claims from kernel-level facts.

## Sources

- https://www.nvidia.com/en-us/data-center/technologies/rubin/
- https://www.nvidia.com/en-us/data-center/vera-rubin-nvl72/
- https://developer.nvidia.com/cuda/gpus

