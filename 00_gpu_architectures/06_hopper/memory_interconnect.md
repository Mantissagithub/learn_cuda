# Hopper Memory And Interconnect

Hopper is not just about the SM. It also has a strong memory and multi-GPU story.

## Memory

H100 SXM has HBM3 and very high bandwidth.

Why I care:

- matmul still needs feeding
- attention is memory sensitive
- HPC workloads often stream huge data
- L2/shared/global movement determines real performance

## L2 and shared

Hopper has big L2 and 256 KB combined L1/shared per SM.

But the really new programming thing is:

- TMA
- distributed shared memory
- clusters

## Interconnect

Hopper uses NVLink 4 and NVLink Switch systems.

The multi-GPU point:

> large models don't fit or train well on one GPU, so the interconnect becomes part of the architecture.

Checklist:

- [ ] understand HBM3 bandwidth role
- [ ] understand NVLink vs PCIe
- [ ] understand NVSwitch at a high level
- [ ] connect this to NCCL collectives
- [ ] connect this to model parallel training/inference

Sources:

- https://developer.nvidia.com/blog/nvidia-hopper-architecture-in-depth/

