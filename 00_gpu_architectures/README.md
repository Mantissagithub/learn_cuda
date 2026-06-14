# GPU Architectures

This is the stuff I should probably understand before pretending CUDA is only about writing kernels.

CUDA code is just one layer. Under that, every architecture has different SMs, memory paths, tensor cores, async copy machinery, interconnects, and compiler targets. So this folder is for building that mental model first.

```mermaid
flowchart LR
  A[architecture] --> B[SM]
  B --> C[memory hierarchy]
  C --> D[tensor cores]
  D --> E[NVLink / system scale]
  E --> F[PTX + SASS targets]
  F --> G[better kernels]
```

## Reading order

| Order | Folder | Why it exists |
| --- | --- | --- |
| 1 | `00_architecture_map/` | big picture timeline |
| 2 | `09_compute_capability_ptx/` | what `sm_90`, `sm_100`, PTX, SASS actually mean |
| 3 | `01_tesla_fermi_kepler/` | early CUDA history, just enough context |
| 4 | `02_maxwell_pascal/` | perf/watt, HBM2, NVLink starting to matter |
| 5 | `03_volta_turing/` | tensor cores enter, independent thread scheduling |
| 6 | `04_ampere/` | TF32, sparsity, MIG, async copy |
| 7 | `05_ada_lovelace/` | RTX/Ada branch, mostly `sm_89` |
| 8 | `06_hopper/` | `sm_90`, TMA, clusters, DSM, FP8 |
| 9 | `07_blackwell/` | `sm_100`, `sm_103`, `sm_120`, FP4/NVFP4 |
| 10 | `08_rubin/` | newer platform-level direction after Blackwell |
| 11 | `10_deep_dive_checklists/` | what to inspect when going deep |

## How I want to read each architecture

- [ ] What compute capability is it?
- [ ] What `sm_XX` target does CUDA use for it?
- [ ] What changed inside the SM?
- [ ] Did Tensor Cores change?
- [ ] Did memory movement change?
- [ ] Did shared memory / L1 / L2 change?
- [ ] Did synchronization change?
- [ ] Did multi-GPU scale-up change?
- [ ] Does this change handwritten CUDA, or mostly library kernels?

## Sources I should keep open

- CUDA GPU compute capability table: https://developer.nvidia.com/cuda/gpus
- CUDA Programming Guide: https://docs.nvidia.com/cuda/cuda-programming-guide/index.html
- PTX ISA docs: https://docs.nvidia.com/cuda/parallel-thread-execution/index.html
- Ampere in-depth: https://developer.nvidia.com/blog/nvidia-ampere-architecture-in-depth/
- Hopper in-depth: https://developer.nvidia.com/blog/nvidia-hopper-architecture-in-depth/
- Blackwell architecture: https://www.nvidia.com/en-us/data-center/technologies/blackwell-architecture/
- Vera Rubin platform: https://www.nvidia.com/en-us/data-center/technologies/rubin/

