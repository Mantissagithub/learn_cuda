# SM Evolution: Tesla To Kepler

This is the early SM story.

## Tesla

Tesla is the beginning of CUDA as a programming model.

What matters:

- lots of threads
- explicit global memory
- explicit shared memory
- blocks as scheduling units
- very little "the hardware will save me" feeling

The mental model is very manual:

```mermaid
flowchart LR
  A[thread] --> B[block]
  B --> C[SM]
  D[global memory] --> E[shared memory]
  E --> A
```

## Fermi

Fermi makes CUDA feel more general-purpose.

New-ish direction:

- better cache hierarchy
- stronger double-precision story than early GPUs
- more mature scheduling
- more serious C/C++ GPU compute target

## Kepler

Kepler is more throughput and better scheduling.

Things to remember:

- Hyper-Q improves work queue utilization
- Dynamic Parallelism appears
- SMX design pushes high throughput

## What I should learn

- [ ] why blocks are independent
- [ ] why shared memory is explicit
- [ ] why old CUDA tutorials obsess over coalescing
- [ ] why dynamic parallelism is not free
- [ ] why CUDA code has to expose enough parallelism

