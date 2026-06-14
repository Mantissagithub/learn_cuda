# Profiler Evidence

If I say a kernel is better, I need evidence.

## Nsight Compute things to inspect

- [ ] achieved occupancy
- [ ] theoretical occupancy
- [ ] register count
- [ ] shared memory usage
- [ ] memory throughput
- [ ] L1 hit rate
- [ ] L2 hit rate
- [ ] DRAM throughput
- [ ] warp stall reasons
- [ ] Tensor Core utilization
- [ ] shared memory bank conflicts
- [ ] branch efficiency / divergence

## What each tells me

| Metric | What it can indicate |
| --- | --- |
| low occupancy | register/shared/thread limit |
| high DRAM throughput | memory-bound kernel |
| low memory throughput + stalls | bad access pattern or latency |
| bank conflicts | shared memory layout problem |
| low Tensor Core utilization | math path not using Tensor Cores well |
| high branch stalls | divergence or control-flow issue |

## Workflow

1. Benchmark.
2. Profile.
3. Pick one bottleneck.
4. Change one thing.
5. Benchmark again.
6. Profile again.

No random "optimization" without a counter.

