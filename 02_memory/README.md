# CUDA Memory

This directory is for memory spaces and access behavior.

```mermaid
flowchart TD
  A[Host memory] --> B[Global memory]
  B --> C[Constant memory]
  C --> D[Constant cache per SM]
  D --> E[Warp broadcast]
```

## Topics

| Directory | Focus |
| --- | --- |
| `constant_memory/` | Read-only cached memory for small shared coefficients |
