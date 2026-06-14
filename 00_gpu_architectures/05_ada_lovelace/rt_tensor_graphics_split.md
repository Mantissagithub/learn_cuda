# RT, Tensor, Graphics Split

Ada docs talk a lot about RTX, DLSS, rendering, RT Cores, and graphics.

That is not irrelevant, but for this repo I need to translate it into CUDA terms.

## RT Cores

RT Cores are for ray tracing acceleration.

Not the main thing for CUDA kernels like:

- reductions
- matmul
- stencil
- CSR

## Tensor Cores

Tensor Cores are relevant to CUDA/library workloads.

They matter for:

- inference
- matmul
- convolution
- DLSS-style AI
- some mixed precision workloads

## CUDA lesson

Ada is useful for local CUDA learning, but don't assume datacenter-Hopper features exist.

Checklist:

- [ ] separate RT Core claims from CUDA compute claims
- [ ] identify Tensor Core data types supported through libraries
- [ ] compare Ada against Ampere GA10x first
- [ ] use NVIDIA docs for exact compute target

