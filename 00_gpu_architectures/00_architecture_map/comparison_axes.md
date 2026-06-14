# Comparison Axes

This is the checklist I should use whenever I compare two NVIDIA GPU generations.

If I only say "new one has more FLOPS", that's useless. Need to ask what changed structurally.

## 1. SM shape

Questions:

- [ ] how many SMs in full chip?
- [ ] how many SMs in the actual shipped SKU?
- [ ] how many warp schedulers per SM?
- [ ] what can issue in parallel?
- [ ] FP32 path changed?
- [ ] INT32 path changed?
- [ ] FP64 path changed?
- [ ] Tensor Core generation changed?
- [ ] shared memory size changed?
- [ ] register pressure behavior changed?

Why this matters:

The SM is where kernels actually run. If I don't know what changed in the SM, I don't know whether a kernel should be tuned differently.

## 2. Memory hierarchy

Questions:

- [ ] global memory type: GDDR, HBM2, HBM3, HBM3E?
- [ ] memory bandwidth?
- [ ] L2 size?
- [ ] L2 bandwidth?
- [ ] L2 residency/compression controls?
- [ ] L1/shared memory split?
- [ ] shared memory bank behavior?
- [ ] async global-to-shared path?

Why this matters:

Most beginner CUDA kernels are memory-bound. So the architecture change often matters more through memory movement than through peak compute.

## 3. Tensor/matrix path

Questions:

- [ ] what Tensor Core generation?
- [ ] what MMA shapes?
- [ ] what data types?
- [ ] FP16?
- [ ] BF16?
- [ ] TF32?
- [ ] FP8?
- [ ] FP4/NVFP4?
- [ ] sparse Tensor Core path?
- [ ] does CUDA expose it directly, or mostly through libraries?

Why this matters:

Matmul/convolution/attention performance is usually won here, not with scalar CUDA cores.

## 4. Synchronization and async execution

Questions:

- [ ] only block-local sync?
- [ ] async copy?
- [ ] async barriers?
- [ ] thread-block clusters?
- [ ] distributed shared memory?
- [ ] TMA?
- [ ] CUDA Graphs relevant?

Why this matters:

Newer GPUs are increasingly about overlapping movement and math. If I don't understand async execution, I won't understand why modern CUTLASS kernels look so complicated.

## 5. Scale-up

Questions:

- [ ] PCIe generation?
- [ ] NVLink generation?
- [ ] NVSwitch?
- [ ] NVLink-C2C?
- [ ] rack-scale topology?
- [ ] NCCL implications?

Why this matters:

For LLMs and HPC, the GPU is not the whole machine. The topology can dominate.

