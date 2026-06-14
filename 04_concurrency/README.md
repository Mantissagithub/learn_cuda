# Concurrency

This directory is for overlapping transfers, kernels, and buffering.

```mermaid
flowchart LR
  A[Pinned memory] --> B[Streams]
  B --> C[Events]
  C --> D[Double buffering]
  D --> E[Overlap H2D, compute, D2H]
```

## Topics

| Directory | Focus |
| --- | --- |
| `streams/` | Stream ordering, events, callbacks, pinned memory |
| `double_buffering/` | Stream-based and shared-memory double buffering |
