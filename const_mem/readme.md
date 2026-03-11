# Constant Memory

Basically this constant memory, is like the pinned memory from the cpu. But its a bit different, its a read-only mem, and also it is stored in global-mem (DRAM) on the GPU. Cached in a dedicated constant cache per SM (Streaming Multiprocessor).

| Component             | Size            |
| --------------------- | --------------- |
| Constant memory space | **64 KB total** |
| Constant cache per SM | **~8 KB**       |

Constant memory is optimized for the case where:
> All threads in a warp read the same address.

This is the pipeline basically, it behaves like a broadcast register for a warp.:

```
┌─────────────────────────┐
│   32 threads in warp    │
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│   all read coeffs[3]    │
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│ broadcast from constant │
│         cache           │
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│   1 memory transaction  │
└─────────────────────────┘
```

## Broadcasting Mechanisms

- case 1: all threads read the same address → 1 memory transaction (broadcast)
- case 2: threads read different addresses → 32 memory transactions (no broadcast)

### When to use constant memory?

- Model parameters
- Kernel coefficients
- Lookup tables
- Configuration constants
- Small read-only arrays

---
The two programs I've written
- [basic_cm.cu](basic_cm.cu): A simple example of using constant memory to store coefficients for a kernel.
- [1d_conv.cu](1d_conv.cu): A 1D convolution example that demonstrates the performance benefits of using constant memory for kernel coefficients.
