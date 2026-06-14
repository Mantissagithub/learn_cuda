# CUDA Topic Checklist

Current step: **0. GPU architecture deep dive**.

Checked items are things already present in this repo. The unchecked ones are the TODOs so I don't miss random important stuff later.

## 0. GPU Architecture Deep Dive

- [ ] Read the map in `00_gpu_architectures/`.
- [ ] Get architecture names vs compute capabilities straight.
- [ ] Understand `compute_XX` vs `sm_XX` vs PTX vs SASS.
- [ ] Understand why `sm_90` is Hopper.
- [ ] Understand why `sm_90a` is not just "better sm_90".
- [ ] Remember that I didn't find `sm_91` in the checked public NVIDIA docs.
- [ ] Understand Blackwell split: `sm_100`, `sm_103`, `sm_120`, `sm_121`.
- [ ] Read Tesla/Fermi/Kepler for historical context.
- [ ] Read Maxwell/Pascal for perf/watt, HBM2, NVLink context.
- [ ] Read Volta/Turing for Tensor Cores and independent thread scheduling.
- [ ] Read Ampere properly: TF32, BF16, sparsity, MIG, async copy.
- [ ] Read Ada as the `sm_89` RTX/server-inference branch.
- [ ] Read Hopper properly: TMA, clusters, DSM, FP8.
- [ ] Read Blackwell properly: FP4/NVFP4, Tensor Cores, NVLink 5, multi-die GPU.
- [ ] Track Rubin without inventing missing CUDA details.
- [ ] For each architecture, ask: does this change my kernel, or just the library/system stack?
- [ ] For each architecture, connect at least one claim to profiler/disassembly evidence.

## 1. GPU Mental Model

- [x] Understand why GPUs are throughput machines, not latency machines.
- [x] Know the CPU vs GPU split: orchestration on CPU, parallel work on GPU.
- [x] Understand the CUDA hierarchy: grid -> block -> warp -> thread.
- [x] Know what an SM is and why blocks are scheduled onto SMs.
- [x] Understand warps as groups of 32 threads.
- [x] Know that warp divergence hurts when threads in a warp branch differently.
- [x] Understand latency hiding: many resident warps let the SM swap work while memory is waiting.
- [x] Know the rough memory hierarchy: registers, shared memory, caches, global memory, host memory.
- [x] Know why high arithmetic intensity matters.
- [x] Be able to explain why small toy kernels often do not show real GPU speedups.

## 2. CUDA Toolchain And Local Setup

- [x] Compile a single `.cu` file with `nvcc`.
- [x] Run a CUDA binary locally.
- [x] Query basic GPU capability and device properties.
- [x] Know what compute capability means.
- [x] Understand host code vs device code in a `.cu` file.
- [x] Use `cudaGetDeviceProperties`.
- [x] Check errors after CUDA runtime calls.
- [x] Know the difference between compile-time errors, launch errors, and runtime synchronization errors.
- [ ] Add a small reusable CUDA error-checking macro/header.
- [ ] Add a simple build convention for examples, probably Makefiles or a tiny script.

## 3. First Kernels

- [x] Write and run vector addition.
- [x] Allocate device memory with `cudaMalloc`.
- [x] Copy host to device with `cudaMemcpy`.
- [x] Copy device to host with `cudaMemcpy`.
- [x] Launch a kernel with `<<<blocks, threads>>>`.
- [x] Use bounds checks inside kernels.
- [x] Compare CPU and GPU outputs for correctness.
- [x] Write a simple matrix multiplication kernel.
- [x] Understand why launch overhead matters for tiny workloads.
- [ ] Add a clean correctness-check helper for arrays/matrices.
- [ ] Add a tiny benchmark harness that avoids timing allocation by mistake.

## 4. Thread Indexing

- [x] Compute 1D global thread IDs.
- [x] Compute 2D global row/column indices.
- [x] Compute 3D global indices.
- [x] Flatten 2D and 3D indices into 1D offsets.
- [x] Work backward from a global ID to block/thread coordinates.
- [x] Understand `blockIdx`, `threadIdx`, `blockDim`, and `gridDim`.
- [x] Know why bounds checks are required when data size is not a multiple of block size.
- [ ] Add visual notes for row-major indexing.
- [ ] Add examples for grid-stride loops.
- [ ] Add examples for block-stride and warp-stride patterns.

## 5. Host/Device Memory Basics

- [x] Use `cudaMalloc` and `cudaFree`.
- [x] Use `cudaMemcpy` in both directions.
- [x] Understand host memory vs device global memory.
- [x] Know that device pointers cannot be dereferenced directly on the CPU.
- [x] Know that host pointers cannot be used directly by ordinary device kernels.
- [x] Understand allocation and copy costs as part of total runtime.
- [ ] Cover `cudaMemset`.
- [ ] Cover unified memory with `cudaMallocManaged`.
- [ ] Compare explicit copies vs unified memory on a small example.
- [ ] Cover page faults and prefetching for unified memory.
- [ ] Add notes on lifetime ownership and cleanup patterns.

## 6. Memory Coalescing

- [ ] Explain coalesced global memory access.
- [ ] Show contiguous thread access: thread `i` reads `a[i]`.
- [ ] Show strided access and why it wastes memory transactions.
- [ ] Compare row-major matrix access patterns.
- [ ] Compare column-major access on row-major data.
- [ ] Measure naive transpose vs coalesced/tiled transpose.
- [ ] Explain memory transaction size at a practical level.
- [ ] Explain alignment and why aligned contiguous loads help.
- [ ] Add a benchmark for contiguous, strided, and random access.
- [ ] Add Nsight Compute evidence for memory throughput differences.
- [ ] Connect this topic to cuBLAS column-major layout confusion.

## 7. Shared Memory

- [x] Use `__shared__` memory.
- [x] Understand shared memory as per-block memory.
- [x] Use `__syncthreads`.
- [x] Implement tiled matrix multiplication.
- [x] Understand why shared memory reduces global memory traffic.
- [x] Know that shared memory is limited and affects occupancy.
- [ ] Explain static vs dynamic shared memory.
- [ ] Cover shared memory bank conflicts.
- [ ] Add a bank-conflict microbenchmark.
- [ ] Cover padding to avoid bank conflicts.
- [ ] Cover shared memory layout choices for 2D tiles.

## 8. Reductions

- [x] Implement max reduction.
- [x] Compare naive/interleaved/sequential addressing variants.
- [x] Understand why reductions need synchronization inside a block.
- [x] Understand reducing global data in multiple stages.
- [x] Try first-add-during-load optimization.
- [x] Try last-warp unrolling.
- [ ] Add sum reduction.
- [ ] Add min reduction.
- [ ] Add block-level reduction helper.
- [ ] Add warp-shuffle reduction.
- [ ] Compare shared-memory reduction vs warp-shuffle reduction.
- [ ] Handle non-power-of-two input sizes cleanly.

## 9. Atomics

- [x] Compare atomic and non-atomic increments.
- [x] Understand lost updates from races.
- [x] Use `atomicAdd`.
- [x] Understand that atomics serialize conflicting updates.
- [x] Know that atomics are correctness tools, not automatic performance tools.
- [ ] Benchmark low-contention vs high-contention atomics.
- [ ] Cover atomic operations for integer and floating-point values.
- [ ] Cover block-local aggregation before global atomics.
- [ ] Cover histogram as a natural atomic example.
- [ ] Explain when prefix sums are better than atomics.

## 10. Constant Memory

- [x] Use `__constant__` memory.
- [x] Use `cudaMemcpyToSymbol`.
- [x] Understand the 64 KB constant memory limit.
- [x] Understand warp broadcast when all threads read the same address.
- [x] Understand why divergent constant reads serialize.
- [x] Implement a small constant-memory example.
- [x] Implement a 1D convolution coefficient example.
- [ ] Compare constant memory vs global memory for the same coefficients.
- [ ] Add a case where constant memory performs poorly.
- [ ] Cover good use cases: masks, coefficients, lookup tables, config.

## 11. Texture And Read-Only Cache Basics

- [ ] Explain texture memory historically vs modern read-only cache usage.
- [ ] Cover spatial locality for image-like access.
- [ ] Add a 2D image/read example.
- [ ] Compare ordinary global reads vs texture/read-only path where useful.
- [ ] Explain boundary handling modes at a high level.
- [ ] Connect this to image kernels like edge detection.

## 12. Streams And Events

- [x] Create CUDA streams.
- [x] Launch kernels into non-default streams.
- [x] Understand issue order inside a stream.
- [x] Understand possible overlap across streams.
- [x] Use CUDA events for timing.
- [x] Use events to reason about pipeline timing.
- [x] Add callback example.
- [ ] Cover default stream behavior carefully.
- [ ] Cover per-thread default stream vs legacy default stream.
- [ ] Add explicit synchronization examples: stream, event, device.
- [ ] Add failure cases where streams do not overlap.
- [ ] Use Nsight Systems to verify overlap.

## 13. Pinned Memory

- [x] Use `cudaMallocHost`.
- [x] Understand pageable vs pinned host memory.
- [x] Understand DMA at a practical level.
- [x] Benchmark pinned vs non-pinned examples.
- [x] Connect pinned memory to async copies and stream overlap.
- [ ] Cover `cudaHostAlloc` flags.
- [ ] Cover when pinned memory is harmful or excessive.
- [ ] Add a clean H2D/D2H bandwidth benchmark.
- [ ] Verify overlap requires pinned memory for async host transfers.

## 14. Double Buffering

- [x] Understand ping-pong buffers.
- [x] Implement stream-based double buffering.
- [x] Implement shared-memory-flavored double buffering.
- [x] Compare double-buffered and non-double-buffered timing.
- [x] Understand overlap: H2D, compute, D2H.
- [ ] Add a clearer chunk-size sweep.
- [ ] Add a timeline diagram from profiler output.
- [ ] Separate stream double buffering from shared-memory staging more explicitly.
- [ ] Cover when double buffering does not help.

## 15. Profiling

- [x] Add NVTX ranges.
- [x] Run Nsight Systems.
- [x] Save `.nsys-rep` and `.sqlite` reports.
- [x] Read CUDA API timing.
- [x] Read GPU kernel timing.
- [x] Read GPU memory copy timing.
- [x] Keep a benchmark notes file.
- [ ] Run Nsight Compute from CLI.
- [ ] Track achieved occupancy.
- [ ] Track memory throughput.
- [ ] Track SM utilization.
- [ ] Track warp stall reasons.
- [ ] Add a profiler checklist: what to inspect first, second, third.

## 16. Occupancy And Roofline Basics

- [ ] Explain occupancy as active warps / maximum active warps.
- [ ] Understand that higher occupancy is not always faster.
- [ ] Know limits: registers, shared memory, threads per block, blocks per SM.
- [ ] Use occupancy calculator or Nsight Compute occupancy section.
- [ ] Explain memory-bound vs compute-bound kernels.
- [ ] Explain arithmetic intensity.
- [ ] Build a tiny roofline-style note for one kernel.
- [ ] Connect occupancy to register pressure.
- [ ] Connect occupancy to shared memory usage.

## 17. Matrix Multiplication Progression

- [x] Implement naive matrix multiplication.
- [x] Implement shared-memory tiled matrix multiplication.
- [x] Benchmark CPU vs GPU matrix multiplication.
- [x] Understand row-major vs column-major mismatch.
- [x] Use cuBLAS SGEMM.
- [x] Use cuBLASLt.
- [x] Compare cuBLAS and CUTLASS examples.
- [ ] Add a clean naive vs tiled vs cuBLAS benchmark table.
- [ ] Add Tensor Core path notes.
- [ ] Cover WMMA with a minimal example.
- [ ] Cover accumulation precision: FP16 input, FP32 accumulate.
- [ ] Add correctness tolerance notes for floating point.

## 18. Sparse Data Structures

- [x] Explain CSR: `values`, `col_indices`, `row_ptr`.
- [x] Encode dense sparse-ish matrix into CSR.
- [x] Decode CSR back to dense.
- [x] Use row-based GPU parallelization.
- [x] Understand why naive atomics are awkward for sparse encoding.
- [x] Note prefix-sum memory pressure concerns.
- [ ] Implement GPU prefix sum or use CUB for row offsets.
- [ ] Add sparse matrix-vector multiply.
- [ ] Compare dense vs CSR memory usage.
- [ ] Compare dense matvec vs CSR matvec.
- [ ] Cover COO and ELL at a high level.

## 19. cuBLAS

- [x] Understand cuBLAS as dense linear algebra library.
- [x] Run SGEMM.
- [x] Run HGEMM.
- [x] Observe FP16 overflow issues.
- [x] Use cuBLASLt.
- [x] Use cuBLASXt.
- [x] Compare cuBLAS variants.
- [x] Document row-major vs column-major convention.
- [ ] Add a clean wrapper for row-major GEMM calls.
- [ ] Cover leading dimensions.
- [ ] Cover transpose flags correctly.
- [ ] Cover batched GEMM.
- [ ] Cover strided batched GEMM.
- [ ] Cover algorithm selection in cuBLASLt.

## 20. cuDNN

- [x] Understand cuDNN handle and descriptors.
- [x] Use tensor descriptors.
- [x] Use convolution descriptors.
- [x] Run cuDNN convolution.
- [x] Compare cuDNN convolution with naive kernel.
- [x] Run convolution algorithm selection.
- [x] Run activation examples: sigmoid and tanh.
- [x] Note NCHW vs NHWC.
- [x] Note graph API and fusion ideas.
- [ ] Add ReLU example.
- [ ] Add batchnorm or normalization example.
- [ ] Add layout conversion benchmark.
- [ ] Cover workspace sizing carefully.
- [ ] Cover cuDNN frontend API separately.

## 21. CUTLASS

- [x] Understand CUTLASS as templated GEMM building blocks.
- [x] Understand hierarchical GEMM: thread block, warp, thread tile.
- [x] Note global -> shared -> register data movement.
- [x] Cover WMMA at a conceptual level.
- [x] Compare CUTLASS and cuBLAS example timing.
- [ ] Build a minimal CUTLASS GEMM from scratch.
- [ ] Change tile shapes and observe performance.
- [ ] Explain epilogues.
- [ ] Explain layouts and strides in CUTLASS.
- [ ] Cover how CUTLASS relates to kernel fusion.

## 22. Cooperative Groups And Warp Primitives

- [ ] Explain why warp-level programming exists.
- [ ] Use `__shfl_down_sync`.
- [ ] Use `__ballot_sync`.
- [ ] Use `__activemask`.
- [ ] Implement warp-level reduction.
- [ ] Implement block-level reduction with warp primitives.
- [ ] Cover cooperative groups syntax.
- [ ] Compare cooperative groups readability vs raw warp intrinsics.

## 23. CUDA Graphs

- [ ] Explain launch overhead.
- [ ] Capture a sequence of operations into a graph.
- [ ] Instantiate and launch a CUDA graph.
- [ ] Compare repeated normal launches vs graph launches.
- [ ] Cover graph update basics.
- [ ] Show where graphs help in inference-style workloads.
- [ ] Show where graphs are overkill.

## 24. Multi-GPU Basics

- [ ] Query number of GPUs.
- [ ] Select device with `cudaSetDevice`.
- [ ] Allocate memory on multiple devices.
- [ ] Copy between devices if peer access is available.
- [ ] Check peer access support.
- [ ] Use cuBLASXt example as the library path.
- [ ] Explain when NCCL becomes relevant.
- [ ] Add a small multi-GPU note even if local machine has one GPU.

## 25. Kernel Fusion Patterns

- [ ] Explain HBM round trips as the main motivation.
- [ ] Fuse two elementwise ops manually.
- [ ] Compare separate kernels vs fused kernel.
- [ ] Track memory reads/writes before and after fusion.
- [ ] Cover fusion limits: registers, occupancy, code complexity.
- [ ] Connect fusion to FlashAttention-style tiling.
- [ ] Connect fusion to cuDNN graph/runtime fusion.
- [ ] Connect fusion to CUTLASS epilogues.

## 26. Inline PTX Basics

- [ ] Generate PTX from a `.cu` file.
- [ ] Read basic PTX structure.
- [ ] Identify loads, stores, arithmetic instructions.
- [ ] Add a tiny inline PTX instruction.
- [ ] Compare CUDA C expression vs inline PTX output.
- [ ] Understand why inline PTX should be rare.
- [ ] Cover constraints and register operands.
- [ ] Add a note on portability risks.

## 27. Cute DSL

- [ ] Explain what Cute DSL is trying to solve.
- [ ] Understand layout algebra basics.
- [ ] Understand shapes, strides, and coordinates.
- [ ] Express simple tensor layouts.
- [ ] Express tiled layouts.
- [ ] Map threads to data using layout concepts.
- [ ] Connect Cute DSL to CUTLASS.
- [ ] Build or study a tiny Cute example.
- [ ] Write notes translating one CUDA tiling example into Cute terms.

## 28. PTX And SASS Deep Dive

- [ ] Generate PTX with `nvcc`.
- [ ] Generate SASS/disassembly with `cuobjdump` or `nvdisasm`.
- [ ] Compare CUDA source, PTX, and SASS.
- [ ] Identify memory instructions.
- [ ] Identify arithmetic instructions.
- [ ] Identify predicate/branch instructions.
- [ ] Understand compiler optimization effects.
- [ ] Use profiler stall reasons to decide what assembly detail matters.
- [ ] Avoid overfitting source code to one generated instruction sequence.
- [ ] Write a final note connecting CUDA source, libraries, Cute DSL, PTX, and SASS.
