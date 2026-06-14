# Kernels And Data Structures

This directory is for custom kernels, synchronization hazards, and data-layout experiments.

```mermaid
flowchart LR
  A[kernels] --> B[shared-memory matmul]
  A --> C[reductions]
  A --> D[transpose]
  A --> E[stencil]
  F[atomic_ops] --> G[race-free updates]
  H[csr] --> I[sparse matrix layout]
```

## Topics

| Directory | Focus |
| --- | --- |
| `kernels/` | Core kernel exercises and benchmarks |
| `atomic_ops/` | Atomic vs non-atomic behavior |
| `csr/` | Sparse matrix encoding/decoding |
