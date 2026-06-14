# Maxwell

Maxwell is mainly an efficiency architecture in my mental map.

Not the sexiest architecture to study now, but important because it shows NVIDIA caring hard about perf/watt and SM efficiency.

## What changed mentally

- better perf/watt
- more efficient SM design
- less brute-force feeling than Kepler
- practical graphics + compute balance

## What I should not overdo

I don't need to spend weeks here. Just understand:

- [ ] why perf/watt matters
- [ ] why SM organization changes can matter even if CUDA code looks same
- [ ] why occupancy is not the only performance metric
- [ ] why cache behavior matters

## Compute capability

Rough family:

- `sm_50`
- `sm_52`
- `sm_53`

Use the legacy NVIDIA table for exact devices.

Source: https://developer.nvidia.com/cuda-legacy-gpus

