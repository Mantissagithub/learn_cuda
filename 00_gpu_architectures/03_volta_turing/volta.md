# Volta

Volta is where Tensor Cores enter.

This is a huge break from thinking "CUDA cores do everything".

## V100 mental model

V100 is a datacenter GPU.

Important pieces:

- `sm_70`
- first Tensor Cores
- independent thread scheduling
- strong FP64
- HBM2

## Tensor Core shift

Before:

> optimize scalar/vector arithmetic on CUDA cores

After:

> map math into matrix multiply shapes that Tensor Cores can eat

This is why GEMM, conv, attention, and all the library paths become so important.

## Independent thread scheduling

Old warp-synchronous code could assume too much. Volta makes thread scheduling more flexible, which means I need explicit sync when correctness depends on it.

Checklist:

- [ ] understand Tensor Core MMA idea
- [ ] understand WMMA at a high level
- [ ] understand independent thread scheduling
- [ ] understand why old warp tricks need care
- [ ] compare V100 and later A100/H100 Tensor Core evolution

