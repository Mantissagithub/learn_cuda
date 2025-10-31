# CUDA_API

This folder contains comparisons and implementations of key CUDA libraries for accelerated computing.

## Folders

### cuBLAS
Implementations and comparisons of CUDA Basic Linear Algebra Subroutines.

**Key files:**
- `compare.cu` - Performance comparison of cuBLAS operations
- `sgemm_hgemm.cu` - Single and half precision GEMM implementations
- `cublas_lt.cu` - CUBLASLT lightweight API
- `cublas_xt.cu` - CUBLASXT for multi-GPU operations

### cuDNN
Deep Learning acceleration using NVIDIA's cuDNN library.

**Key files:**
- `conv2d.cu` - 2D convolution operations
- `conv2d_algo_selector.cu` - Algorithm selection for convolutions
- `sigmoid.cu` - Sigmoid activation function
- `tanh.cu` - Tanh activation function

### CUTLASS
CUDA Templates for Linear Algebra Subroutines and Solvers - high-performance GEMM implementations.

**Key findings:**
- CUTLASS SGEMM: 26.2378 ms (100 runs)
- CUBLAS SGEMM: 28.1186 ms (100 runs)
- CUTLASS shows approximately 6.7% performance improvement over CUBLAS on SM 8.9

See `CUTLASS/readme.md` for detailed architectural documentation and tiling strategies.
