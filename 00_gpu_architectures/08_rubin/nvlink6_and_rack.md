# NVLink 6 And Rack Scale

Rubin's public story is rack scale.

The big thing from NVIDIA's page:

- NVLink 6
- 72 Rubin GPUs in NVL72 style systems
- 3.6 TB/s bandwidth per GPU through the fabric
- 260 TB/s rack connectivity

## Why this matters

For huge models:

- the GPU does math
- memory holds activations/weights/KV cache
- interconnect moves tensors between GPUs
- networking moves across racks

If communication is bad, the FLOPS don't matter.

## Checklist

- [ ] understand scale-up vs scale-out
- [ ] understand NVLink vs InfiniBand/Ethernet roles
- [ ] understand where NCCL fits
- [ ] track Rubin NVLink docs when more public
- [ ] separate product-page claims from CUDA kernel facts

Sources:

- https://www.nvidia.com/en-us/data-center/technologies/rubin/
- https://www.nvidia.com/en-us/data-center/vera-rubin-nvl72/

