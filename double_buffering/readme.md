# Double Buffering

> Double buffering works by creating two separate memory buffers (typically in shared memory) for each input that needs to be loaded. While compute threads process data from buffer A, DMA (data movement) threads simultaneously load new data into buffer B. On the next iteration, the roles reverse: compute threads work on buffer B while DMA threads load into buffer A. This ping-pong pattern ensures that compute threads remain continuously busy rather than waiting for memory operations to complete

the above thing is the def, its basically, we keep two mem places to load data from the host device to gpu, so that while one is being used by the compute threads, the other is being loaded by the DMA threads. DMA -> direct memory access, so its a separate engine in the GPU which handles the mem transfer.

---
double buffering is most effective for <b>mem-bound kernels</b> that aren't limited by mem or register constraints. the technique excels at exploiting memory level parallelsim by having more dma threads issuing loads to the mem system.

---

### 1. stream based double buffering

Stream-based double buffering exploits independent streams to overlap data transfers with kernel computation. By dividing the workload into chunks and using multiple streams, data transfers for one chunk can occur simultaneously with kernel execution for another chunk. This approach is particularly effective when the data transfer time is comparable to or longer than the kernel execution time.

**Script:** [stream_based_db.cu](stream_based_db.cu)

**Execution Timeline:**

```
Time:    │  0  │  1  │  2  │  3  │  4  │  5  │
─────────┼─────┼─────┼─────┼─────┼─────┼─────┤
Stream 0 │ H2D │Comp │ D2H │ H2D │Comp │ ... │
         │ [0] │ [0] │ [0] │ [4] │ [4] │     │
─────────┼─────┼─────┼─────┼─────┼─────┼─────┤
Stream 1 │     │ H2D │Comp │ D2H │ H2D │ ... │
         │     │ [1] │ [1] │ [1] │ [5] │     │
─────────┼─────┼─────┼─────┼─────┼─────┼─────┤
Stream 2 │     │     │ H2D │Comp │ D2H │ ... │
         │     │     │ [2] │ [2] │ [2] │     │
─────────┼─────┼─────┼─────┼─────┼─────┼─────┤
Stream 3 │     │     │     │ H2D │Comp │ ... │
         │     │     │     │ [3] │ [3] │     │
```

---

### 2. shared mem based double buffering

Shared memory-based double buffering utilizes the GPU's fast shared memory to hold two buffers for data processing. While one buffer is being processed by the compute threads, the other buffer is being loaded with new data from global memory.

**Script:** [shared_mem_db.cu](shared_mem_db.cu)

**Execution Timeline:**

```
Time:    │  0     │  1     │  2     │  3     │  4     │  5     │  6     │  7     │
─────────┼────────┼────────┼────────┼────────┼────────┼────────┼────────┼────────┤
Stream 0 │ H2D[0] │Comp[0] │D2H[0]  │H2D[2]  │Comp[2] │D2H[2]  │H2D[4]  │...     │
─────────┼────────┼────────┼────────┼────────┼────────┼────────┼────────┼────────┤
Stream 1 │        │H2D[1]  │Comp[1] │D2H[1]  │H2D[3]  │Comp[3] │D2H[3]  │H2D[5]  │
─────────┼────────┼────────┼────────┼────────┼────────┼────────┼────────┼────────┤
```

```
Chunk Mapping:
chunk 0 → buffer 0 → stream 0
chunk 1 → buffer 1 → stream 1
chunk 2 → buffer 0 → stream 0
chunk 3 → buffer 1 → stream 1
chunk 4 → buffer 0 → stream 0
```

## Results

| Approach                   | Time (ms) |
|----------------------------|-----------|
| Stream Based Double Buffering | 0.173728 |
| Shared Memory Based Double Buffering | 0.164864 |
| No Double Buffering        | 0.234560 |

**Note**: These are tested on nvidia rtx 4060, results may vary on different hardware.

---
**Good read**: https://github.com/NVIDIA/cutlass/discussions/227

---