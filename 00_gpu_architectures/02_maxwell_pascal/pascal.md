# Pascal

Pascal is much more important for the modern datacenter story.

The big thing:

> P100 brings HBM2 and NVLink into the serious accelerator picture.

## GP100 vs graphics Pascal

This is one of the recurring NVIDIA patterns:

- datacenter chip: strong FP64, HBM, NVLink
- graphics chip: gaming/pro graphics priorities

Same generation name does not mean same performance character.

## Pascal things to know

- HBM2 on P100
- NVLink
- stronger unified memory story
- high FP64 on GP100
- PCIe variants still exist

## Checklist

- [ ] understand why HBM2 changes bandwidth math
- [ ] understand PCIe vs NVLink
- [ ] understand why FP64 matters for HPC
- [ ] understand unified memory page migration at a high level
- [ ] compare P100 with consumer Pascal mentally

Sources:

- https://www.nvidia.com/en-us/data-center/pascal-gpu-architecture/
- https://docs.nvidia.com/cuda/pascal-tuning-guide/index.html

