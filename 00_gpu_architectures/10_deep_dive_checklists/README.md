# Deep-Dive Checklists

This is the "stop handwaving" checklist.

If I say I understand an architecture, I should be able to answer these with either docs, code, disassembly, or profiler evidence.

## Architecture checklist

- [ ] How many GPCs/TPCs/SMs does the full chip have?
- [ ] How many SMs does the shipped product expose?
- [ ] What is one SM made of?
- [ ] How many warp schedulers?
- [ ] What issue paths exist?
- [ ] What FP32/INT32/FP64 paths exist?
- [ ] What Tensor Core generation?
- [ ] What Tensor Core data types?
- [ ] How big is the register file?
- [ ] How much shared memory per SM?
- [ ] How does L1/shared memory split work?
- [ ] How big is L2?
- [ ] Is there L2 residency control or compression?
- [ ] What global memory tech: GDDR, HBM2, HBM3, HBM3E?
- [ ] What is the memory bandwidth?
- [ ] What async copy path exists: none, `cp.async`, TMA, newer tensor memory?
- [ ] What synchronization primitives are new?
- [ ] Can blocks cooperate across SMs?
- [ ] What NVLink generation?
- [ ] What PCIe generation?
- [ ] What MIG / virtualization / confidential computing support exists?
- [ ] What RAS features matter?

## Kernel checklist

- [ ] Is my kernel memory-bound?
- [ ] Is it compute-bound?
- [ ] Is it launch-bound?
- [ ] Are global loads coalesced?
- [ ] Is shared memory actually helping?
- [ ] Are there bank conflicts?
- [ ] Is register pressure killing occupancy?
- [ ] Can Tensor Cores be used?
- [ ] Can a library do this better?
- [ ] Can async copy hide memory movement?
- [ ] Is this optimization generic, family-specific, or architecture-specific?

## Evidence checklist

- [ ] PTX from `nvcc --ptx`
- [ ] SASS from `cuobjdump --dump-sass`
- [ ] Nsight Compute achieved occupancy
- [ ] memory throughput
- [ ] L2 hit rate
- [ ] shared memory bank conflict metrics
- [ ] warp stall reasons
- [ ] Tensor Core utilization
- [ ] DRAM throughput
- [ ] instruction mix

## Files here

- [profiler_evidence.md](profiler_evidence.md): Nsight evidence checklist
- [disassembly_ptx_sass.md](disassembly_ptx_sass.md): PTX/SASS proof workflow

## My rule

No vague "this should be faster". I need one of:

- a hardware reason
- a compiler reason
- a memory traffic reason
- a profiler counter
- a benchmark
