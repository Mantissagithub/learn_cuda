# Blackwell SM Questions

This is not pretending I know every Blackwell internal detail.

This is the list of things I need to inspect as more docs/tooling become available.

## Questions

- [ ] what exactly changed in the Blackwell SM vs Hopper?
- [ ] what Tensor Core instructions are new?
- [ ] what parts are exposed in PTX?
- [ ] what parts only show through libraries?
- [ ] how does FP4/NVFP4 appear in PTX/SASS?
- [ ] what is the role of tensor memory / TMEM-like paths?
- [ ] how do CUTLASS Blackwell kernels structure mainloops?
- [ ] what occupancy constraints change?
- [ ] what Nsight Compute metrics changed?

## How to answer them

Use:

- PTX ISA release notes
- CUDA release notes
- CUTLASS source
- Nsight Compute
- `cuobjdump`
- NVIDIA architecture technical brief

## Why keep this file

Blackwell details are still more split across product/platform docs than Hopper's old technical blog. This file is the "don't handwave, go verify" list.

