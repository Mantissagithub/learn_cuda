# Memory And Programming Model

The early CUDA memory model explains a lot of habits that still exist.

## Global memory

Global memory was slow relative to compute, so the old lesson was:

> don't keep going back to global memory if you can reuse data.

This is why tiling exists everywhere.

## Shared memory

Shared memory is programmer-managed cache-ish memory.

But it is not automatic cache:

- you decide what to load
- you decide layout
- you synchronize
- you avoid bank conflicts

## Coalescing

Early CUDA had stricter coalescing rules. Modern GPUs are more forgiving, but the core idea still matters:

> adjacent threads should usually touch adjacent memory.

Bad:

```cpp
x = a[threadIdx.x * stride];
```

Good:

```cpp
x = a[blockIdx.x * blockDim.x + threadIdx.x];
```

## Checklist

- [ ] understand why tiling exists
- [ ] understand why coalescing exists
- [ ] understand why shared memory is explicit
- [ ] understand why block independence is a design feature
- [ ] understand why synchronization is only cheap inside limited scopes

