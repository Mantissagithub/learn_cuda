# Blackwell NVLink And System Story

Blackwell is where NVIDIA's language becomes "AI factory" heavy.

Annoying marketing term, but the technical point is real:

> the rack/system is becoming the unit of compute, not just the GPU.

## NVLink 5

Blackwell uses fifth-gen NVLink in the datacenter platform story.

Why I care:

- model parallelism
- expert parallelism
- huge inference serving
- all-reduce / collective bandwidth
- lower cost per token depends on communication too

## GB200 / GB300 style systems

The GPU is one part. The platform includes:

- Grace CPU
- Blackwell GPU
- NVLink
- NVSwitch
- networking
- system software

## Checklist

- [ ] understand GPU-level vs rack-level claims
- [ ] understand NVLink generation differences
- [ ] connect NVLink to NCCL
- [ ] connect Blackwell systems to LLM inference
- [ ] avoid treating peak FLOPS as the whole story

Sources:

- https://www.nvidia.com/en-us/data-center/technologies/blackwell-architecture/
- https://docs.nvidia.com/data-center-gpu/line-card.pdf

