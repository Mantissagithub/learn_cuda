# MIG And Async Copy

Two Ampere ideas that are easy to mix up because one is system-level and one is kernel-level.

## MIG

MIG = Multi-Instance GPU.

On A100, the GPU can be partitioned into isolated GPU instances.

This is not just "run multiple processes". The partition gets its own slice of hardware resources.

Why it matters:

- cloud sharing
- predictable QoS
- better utilization
- isolation between users/jobs

Checklist:

- [ ] understand what MIG partitions
- [ ] understand why A100 server users care
- [ ] understand why MIG is not a kernel optimization
- [ ] know that not every Ampere GPU has MIG

## Async copy

Async copy is kernel-level.

Ampere gives `cp.async` so optimized kernels can move global memory into shared memory while other work continues.

Why it matters:

- better pipelining
- fewer threads wasted on copy bookkeeping
- useful for tiled GEMM/conv-like kernels

Checklist:

- [ ] find `cp.async` in generated PTX/SASS or CUTLASS
- [ ] understand wait groups at a high level
- [ ] compare with Hopper TMA later
- [ ] know when plain shared memory tiling is enough

