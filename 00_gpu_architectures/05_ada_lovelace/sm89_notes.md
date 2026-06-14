# `sm_89` Notes

Ada's CUDA target to remember is `sm_89`.

Examples:

- RTX 4090/4080/4070/etc
- L4
- L40/L40S
- RTX 6000 Ada

## What this means for me

If I have an RTX 40 GPU, I should compile for `sm_89`, not `sm_90`.

Hopper blog posts don't directly apply just because both are "modern NVIDIA".

## Things to check locally

```bash
nvidia-smi --query-gpu=name,compute_cap --format=csv
```

Checklist:

- [ ] confirm local GPU compute capability
- [ ] compile a kernel with `-arch=sm_89` if applicable
- [ ] dump SASS and compare with Ampere
- [ ] benchmark basic kernels against Ampere notes carefully

Sources:

- https://developer.nvidia.com/cuda/gpus
- https://www.nvidia.com/en-us/geforce/ada-lovelace-architecture/

