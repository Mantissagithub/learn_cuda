# Target Strings Quick Reference

This is the cheat sheet for `sm_XX`.

| Target | Architecture-ish meaning | Notes |
| --- | --- | --- |
| `sm_75` | Turing | T4 / RTX 20 |
| `sm_80` | Ampere GA100 | A100 / A30 |
| `sm_86` | Ampere GA10x | RTX 30 / A10 / A40 |
| `sm_87` | Orin | Jetson Orin |
| `sm_89` | Ada | RTX 40 / L4 / L40 |
| `sm_90` | Hopper | H100 / H200 / GH200 |
| `sm_90a` | Hopper-specific | not generic portable Hopper |
| `sm_100` | Blackwell datacenter | B200 / GB200 |
| `sm_103` | Blackwell Ultra | B300 / GB300 |
| `sm_110` | newer PTX family | public GPU table has Jetson T5000/T4000 at CC 11.0 |
| `sm_120` | RTX Blackwell | RTX 50 / RTX PRO Blackwell |
| `sm_121` | GB10 | DGX Spark / GB10 |

## My rule

Don't infer target strings from marketing names. Check:

1. `nvcc --help`
2. NVIDIA compute capability table
3. PTX ISA release notes
4. CUDA release notes

## About suffixes

- `a` = architecture-specific. Faster/newer features maybe, but less portable.
- `f` = family-specific target style from PTX notes.
- generic `sm_XX` first, suffix only when a feature needs it.

## Sources

- https://developer.nvidia.com/cuda/gpus
- https://docs.nvidia.com/cuda/parallel-thread-execution/index.html
- https://docs.nvidia.com/cuda/cuda-programming-guide/index.html

