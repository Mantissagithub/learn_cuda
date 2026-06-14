# Hopper SM, Clusters, And TMA

The three Hopper ideas I should not handwave:

- clusters
- distributed shared memory
- TMA

## Thread-block clusters

Normal blocks are independent. A cluster is different because blocks in a cluster are guaranteed to be scheduled together.

That matters because then they can actually cooperate.

Checklist:

- [ ] Understand cluster vs grid.
- [ ] Understand why concurrent scheduling matters.
- [ ] Find the cluster launch API.
- [ ] Check cluster size limits.

## Distributed shared memory

This is basically shared memory across blocks in a cluster.

But I should not think "free global memory". It is still a very specific fast/local cooperation tool.

Checklist:

- [ ] Understand local shared vs remote shared.
- [ ] Understand when remote shared access is worth it.
- [ ] Understand cluster sync.
- [ ] Understand why this helps tiled algorithms.

## TMA

TMA moves tensor-shaped data from global to shared without wasting tons of ordinary CUDA threads on address arithmetic and copying.

Checklist:

- [ ] Understand TMA vs normal loads.
- [ ] Understand TMA vs Ampere `cp.async`.
- [ ] Read a CUTLASS TMA mainloop.
- [ ] Understand the barrier around completion.

## Sources

- https://developer.nvidia.com/blog/nvidia-hopper-architecture-in-depth/
- https://docs.nvidia.com/cuda/hopper-tuning-guide/index.html
- https://docs.nvidia.com/cuda/cuda-programming-guide/index.html
- https://docs.nvidia.com/cuda/parallel-thread-execution/index.html

