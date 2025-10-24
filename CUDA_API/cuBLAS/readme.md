# cuBLAS

So these all boil down to matrix multiplication and linear algebra operations. And this library is the state of the art in doing them. It stands for **_CUDA Basic Linear Algebra Subroutines_**.

And then there are many variants in this, lemme update it as I read tho:
- cuBLASLt
- cuBLASXt
- cuBLASDx

And last comes **CUTLASS**, which can be used to fuse kernels as in _flash-attention_ to get the max output. As we need to get store the result in **HBM** _(High-Bandwidth Memory)_ and get back the results from it, instead tile them store it in SRAM available in each SMs, and then retrieve and process more faster. So this was the idea of flash-attention, let's see how I'm gonna use it.

The matrix multiplication is a bit cracked over here, as CUDA takes **column-major**, as in when a matrix is given like this, it interprets like this, whereas C/C++ take it as row-major.

![row-major vs col-major](../../assets/row_m_col_m.png)

---

## SGEMM vs HGEMM

**Script:** [sgemm_hgemm.cu](sgemm_hgemm.cu)

### Compilation & Execution
```
nvcc sgemm_hgemm.cu -o sgemm_hgemm -lcublas && ./sgemm_hgemm
```

### Configuration
- **Matrix sizes:** M=8192, N=8192, K=8192
- **Estimated GPU memory usage:** 1.12 GB

### Results

#### Sample CUBLAS SGEMM Results (first 5×5 block):

| Col 0     | Col 1     | Col 2     | Col 3     | Col 4     |
|-----------|-----------|-----------|-----------|-----------|
| 202062.45 | 202923.62 | 201036.56 | 200185.86 | 200570.17 |
| 199339.61 | 199351.12 | 200286.06 | 198719.69 | 198125.89 |
| 200642.75 | 199398.78 | 200617.91 | 198644.61 | 199260.06 |
| 200016.61 | 199653.33 | 198613.52 | 198887.62 | 199084.45 |
| 203695.62 | 204948.42 | 203191.11 | 200763.20 | 203300.89 |

#### Sample CUBLAS HGEMM Results (first 5×5 block):

| Col 0 | Col 1 | Col 2 | Col 3 | Col 4 |
|-------|-------|-------|-------|-------|
| inf   | inf   | inf   | inf   | inf   |
| inf   | inf   | inf   | inf   | inf   |
| inf   | inf   | inf   | inf   | inf   |
| inf   | inf   | inf   | inf   | inf   |
| inf   | inf   | inf   | inf   | inf   |

#### Performance Comparison:

| Metric | Value |
|--------|-------|
| **CUBLAS SGEMM Time** | 0.191532 seconds (5740.61 GFLOPS) |
| **CUBLAS HGEMM Time** | 0.061726 seconds (17812.88 GFLOPS) |
| **HGEMM Speedup** | **3.10×** |

**Note:** Even though HGEMM is faster, the results are `inf`, so it's not usable in this case. This is caused by overflow.

**Additional Resource:** [modal-gpu-glossary](https://modal.com/gpu-glossary/host-software/cublas)

---

## cuBLASLt

This is the lightweight version of cuBLAS, designed for more flexibility and better performance on specific workloads. I need to discover the "specific workloads" tho?

### Terms to Know:
- **CUDA_R_32F** → 32-bit floating point number, and the "R" is just for real number description, not complex numbers
- **CUBLASLT_MATMUL_DESC_TRANSA** → Descriptor for transposing the first matrix in a matrix multiplication operation, specifically for matrix A

### Process Flow:

```
flowchart TD
  A[cudaMalloc] --> B[Memcpy]
  B --> C[createHandle]
  C --> D[create matrix descriptors]
  D --> E[create mat mul descriptors]
  E --> F[mat mul]
```

### Important Note:

From cuBLAS-Lt docs (search for "Dimensions m and k must be multiples of 4"):

> cuBLASLt requires matrix dimensions to be multiples of 4 (or sometimes higher powers of 2) due to hardware memory alignment requirements and optimization for Tensor Cores and vectorized operations.

Will research on this more and update later.

**Script:** [cublas_lt.cu](cublas_lt.cu)

### Execution Results:

#### Timing Breakdown:

| Operation | Time (seconds) |
|-----------|----------------|
| cudaMalloc d_a_fp32 | 1.913984 |
| cudaMalloc d_b_fp32 | 0.000007 |
| cudaMalloc d_c_fp32 | 0.000002 |
| cudaMemcpy d_a_fp32 | 0.000328 |
| cudaMemcpy d_b_fp32 | 0.000002 |
| cublasLtCreate | 0.001650 |
| cublasLtMatrixLayoutCreate (all 3) | 0.000291 |
| cublasLtMatmul (fp32) | 0.048033 |
| cudaMemcpy result (fp32) | 0.000042 |
| half conversion | 0.000001 |
| cudaMemcpy (fp16) | 0.000010 |
| cublasLtMatmulDescCreate (fp16) | 0.000000 |
| cublasLtMatmul (fp16) | 0.038677 |
| cudaMemcpy result (fp16) | 0.000019 |

#### CUBLASLt FP32 Result:

```
106.00  116.00  126.00  136.00
234.00  260.00  286.00  312.00
362.00  404.00  446.00  488.00
490.00  548.00  606.00  664.00
```

#### CUBLASLt FP16 Result:

```
inf  inf  inf  inf
inf  inf  inf  inf
0.00 0.00 0.00 0.00
0.00 0.00 0.00 0.00
```

---

## cuBLASXt

This is the multi-GPU version of cuBLAS, so that we can use multiple GPUs to do matrix multiplications and linear algebra operations.

**Script:** [cublas_xt.cu](cublas_xt.cu)

---

## Performance Comparison: cuBLAS Variants

**Script:** [compare.cu](compare.cu)

### Compilation & Execution
```
nvcc compare.cu -o compare -lcublas -lcublasLt && ./compare
```

### Configuration
- **Matrix sizes:** M=4096, N=4096, K=4096
- **Estimated GPU memory usage:** 0.56 GB

### Results by Library

#### ========== cuBLAS v2 SGEMM ==========

**Time:** 0.051940 seconds
**GFLOPS:** 2646.11

**Sample result (5×5):**
```
99904.68  101116.63  99355.57  99562.47  101291.06
101256.11 100536.03  100097.20 99764.00  100661.56
100927.05 101019.61  98372.89  99800.45  100138.67
99720.47  100719.33  99005.86  98904.87  100014.90
102931.36 103334.26  101111.21 101220.08 101405.79
```

#### ========== cuBLASLt ==========

**Time:** 0.024560 seconds
**GFLOPS:** 5596.10

**Sample result (5×5):**
```
99904.68  101116.63  99355.57  99562.47  101291.06
101256.11 100536.03  100097.20 99764.00  100661.56
100927.05 101019.61  98372.89  99800.45  100138.67
99720.47  100719.33  99005.86  98904.87  100014.90
102931.36 103334.26  101111.21 101220.08 101405.79
```

#### ========== cuBLASXt ==========

**Time:** 0.018074 seconds
**GFLOPS:** 7604.39

**Sample result (5×5):**
```
100854.20 99234.77  101435.17 101376.82 102448.23
101248.17 98484.77  99606.92  100681.07 101126.09
100171.94 98861.37  101653.14 100368.57 102016.46
100840.51 98954.64  101343.61 100794.39 101402.16
99642.48  97395.88  99657.77  98629.19  100659.97
```

### ========== Summary ==========

| Library    | Time (sec) | GFLOPS  |
|------------|------------|---------|
| cuBLAS v2  | 0.051940   | 2646.11 |
| cuBLASLt   | 0.024560   | 5596.10 |
| cuBLASXt   | 0.018074   | 7604.39 |

#### Relative Speedup:

| Comparison | Speedup |
|------------|---------|
| LT vs v2   | **2.11×** |
| XT vs v2   | **2.87×** |
| XT vs LT   | **1.36×** |

---
```