# NVLink And HBM Context

This is the part of Pascal that keeps mattering later.

## HBM

HBM is about bandwidth.

Instead of only scaling compute, the GPU also needs to feed compute. If memory cannot feed the SMs, extra FLOPS are decorative.

So HBM matters for:

- dense linear algebra
- stencil codes
- bandwidth-bound HPC kernels
- big model training
- anything that streams huge tensors

## NVLink

PCIe is not enough for serious multi-GPU scaling.

NVLink matters because:

- GPU-GPU bandwidth is higher
- latency is better
- all-reduce / model parallel workloads care a lot
- later NVSwitch systems build on this idea

## The lesson

Modern GPUs are not just chips. They are memory + interconnect + software stacks.

Checklist:

- [ ] know PCIe vs NVLink difference
- [ ] know HBM vs GDDR at a high level
- [ ] understand why NCCL cares about topology
- [ ] connect NVLink to later Hopper/Blackwell/Rubin scale-up

