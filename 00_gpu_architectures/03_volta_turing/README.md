# Volta And Turing

This is where the current NVIDIA shape starts becoming obvious.

Volta introduces Tensor Cores. Turing takes that plus graphics/RT stuff and makes the RTX branch.

## Volta

- `sm_70`
- first Tensor Core generation
- V100 is the big datacenter GPU here
- independent thread scheduling shows up

The important CUDA lesson: old warp-synchronous assumptions can break. If threads need ordering, be explicit.

## Turing

- `sm_75`
- T4, RTX 20 era
- Tensor Cores continue
- RT Cores enter for ray tracing

The important CUDA lesson: same company, same CUDA ecosystem, but architecture goals can differ a lot between datacenter and graphics/inference parts.

## Checklist

- [ ] Understand what a Tensor Core actually does: matrix multiply-accumulate.
- [ ] Understand WMMA as a CUDA way to touch Tensor Cores.
- [ ] Understand independent thread scheduling.
- [ ] Compare V100 and T4 as very different products.
- [ ] Track how Tensor Core data types evolve after Volta.

## Files here

- [volta.md](volta.md): V100, `sm_70`, first Tensor Cores
- [turing.md](turing.md): T4/RTX, `sm_75`
- [tensor_core_transition.md](tensor_core_transition.md): why Tensor Cores changed the game

## Sources

- https://www.nvidia.com/en-us/data-center/volta-gpu-architecture/
- https://docs.nvidia.com/cuda/volta-tuning-guide/index.html
- https://docs.nvidia.com/cuda/turing-tuning-guide/index.html
- https://developer.nvidia.com/cuda/gpus
