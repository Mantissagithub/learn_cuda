# Tesla, Fermi, Kepler

This is old CUDA history, but it's useful because a lot of CUDA's weirdness makes sense from here.

I don't need to optimize for these GPUs now. I just need to know what ideas entered the model.

## Tesla

Early CUDA.

- grid / block / thread model starts here
- very explicit memory thinking
- not much comfort from caches compared to modern GPUs
- CUDA is basically "you manage parallelism and memory yourself"

## Fermi

This is where CUDA becomes more general-purpose.

- stronger cache hierarchy
- more usable for non-graphics compute
- compute capability around `sm_20` / `sm_21`

## Kepler

Kepler pushes throughput and scheduling.

- dynamic parallelism shows up
- Hyper-Q improves queue usage
- compute capability around `sm_30`, `sm_35`, `sm_37`

## Things to understand from this era

- [ ] Why blocks are the unit of scheduling.
- [ ] Why shared memory exists.
- [ ] Why old CUDA programmers cared so much about coalescing.
- [ ] Why caches did not remove the need for manual tiling.
- [ ] Why dynamic parallelism exists, but isn't usually the first tool.

## Files here

- [sm_evolution.md](sm_evolution.md): how early SM thinking evolved
- [memory_and_programming.md](memory_and_programming.md): why memory/coalescing/shared memory habits exist

## Sources

- https://developer.nvidia.com/cuda-legacy-gpus
- https://docs.nvidia.com/cuda/archive/
- https://docs.nvidia.com/cuda/archive/11.8.0/kepler-tuning-guide/index.html
