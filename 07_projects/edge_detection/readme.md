# Edge detection GPU

```mermaid
flowchart LR
  A[Input image] --> B[Copy to GPU]
  B --> C[Edge kernel]
  C --> D[Copy result back]
  D --> E[Output image]
```

So the codebase for this would be avalilable [here](https://github.com/Mantissagithub/edge_detection_gpu)
