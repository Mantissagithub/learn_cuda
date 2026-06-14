# Projects

This directory is for applied CUDA projects that are bigger than a single kernel note.

```mermaid
flowchart LR
  A[Input data] --> B[CUDA kernels]
  B --> C[CPU baseline]
  B --> D[GPU output]
  C --> E[Compare]
  D --> E
```

## Projects

| Directory | Focus |
| --- | --- |
| `edge_detection/` | GPU edge-detection project pointer |
