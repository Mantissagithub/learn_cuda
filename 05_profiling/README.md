# Profiling

This directory is for timing kernels and reading profiler output.

```mermaid
flowchart TD
  A[Compile with nvcc] --> B[Run executable]
  B --> C[nsys profile]
  C --> D[Timeline and API stats]
  D --> E[Nsight Compute report]
```

## Files

| Path | Focus |
| --- | --- |
| `benchmark.md` | Kernel benchmark notes |
| `profiling/` | NVTX, Nsight Systems, Nsight Compute reports |
