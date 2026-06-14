# Compute Capability, PTX, And `sm_XX`

This is one of those things that looks like naming nonsense, but actually matters a lot.

## Terms

| Term | Meaning |
| --- | --- |
| compute capability | hardware feature version, like `8.0`, `9.0`, `10.0` |
| `compute_90` | virtual arch target, emits PTX for CC 9.0 |
| `sm_90` | real arch target, emits machine code for Hopper |
| PTX | NVIDIA virtual ISA |
| SASS | actual machine instructions the GPU runs |
| `sm_90a` | architecture-specific target, not generic portable Hopper |
| `sm_100f` | family-specific PTX target style |

So the flow is basically:

```mermaid
flowchart LR
  A[CUDA C++] --> B[PTX / virtual ISA]
  B --> C[SASS / real GPU ISA]
  C --> D[SM executes it]
```

## Current public mapping I care about

From NVIDIA's compute capability table:

| CC | Examples | Family |
| --- | --- | --- |
| 12.1 | GB10 / DGX Spark | Grace Blackwell small-system class |
| 12.0 | RTX PRO Blackwell, RTX 50 | RTX Blackwell |
| 11.0 | Jetson T5000/T4000 | newer Jetson public entries |
| 10.3 | B300/GB300 | Blackwell Ultra |
| 10.0 | B200/GB200 | Blackwell datacenter |
| 9.0 | H100/H200/GH200 | Hopper |
| 8.9 | L4/L40/RTX 40 | Ada |
| 8.7 | Jetson Orin | embedded Ampere-ish path |
| 8.6 | RTX 30/A10/A40 | Ampere GA10x |
| 8.0 | A100/A30 | Ampere GA100 |
| 7.5 | T4/RTX 20 | Turing |

Official table: https://developer.nvidia.com/cuda/gpus

## About `sm_90`, `sm_91`, `sm_100`, etc.

- `sm_90` = Hopper generic target.
- `sm_90a` = Hopper architecture-specific target. Use only when needed.
- `sm_91` = I did **not** find this in the current public NVIDIA compute capability table or PTX target list. So don't rely on it unless the local CUDA toolkit docs say so.
- `sm_100` = Blackwell datacenter, B200/GB200 side.
- `sm_103` = Blackwell Ultra, B300/GB300 side.
- `sm_110` = appears in PTX target notes; public table has Jetson T5000/T4000 at CC 11.0.
- `sm_120` = RTX Blackwell / RTX 50 style target.
- `sm_121` = GB10 / DGX Spark.

## Compile examples

```bash
nvcc kernel.cu -gencode arch=compute_90,code=sm_90
nvcc kernel.cu -gencode arch=compute_90,code=compute_90
nvcc kernel.cu -gencode arch=compute_100,code=sm_100
```

First one puts native Hopper machine code in the binary. Second one puts PTX fallback. Third one is Blackwell datacenter style if the toolkit supports it.

## Checklist

- [ ] Run `nvidia-smi --query-gpu=name,compute_cap --format=csv`.
- [ ] Compile a `.cu` file to PTX.
- [ ] Compile for the real local `sm_XX`.
- [ ] Use `cuobjdump --dump-sass`.
- [ ] Build a binary with SASS + PTX fallback.
- [ ] Check `nvcc --help` before assuming a target exists.
- [ ] Avoid `a` targets unless I know exactly why.

## Files here

- [target_strings.md](target_strings.md): quick `sm_XX` reference

## Sources

- https://developer.nvidia.com/cuda/gpus
- https://docs.nvidia.com/cuda/cuda-programming-guide/index.html#compute-capabilities
- https://docs.nvidia.com/cuda/parallel-thread-execution/index.html
