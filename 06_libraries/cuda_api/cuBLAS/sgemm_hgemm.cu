#include <bits/stdc++.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cuda_fp16.h>

// to run this code: nvcc sgemm_hgemm.cu -o sgemm_hgemm -lcublas && ./sgemm_hgemm

using namespace std;

#define M 8192
#define N 8192
#define K 8192

// For more aggressive sizing (≈4.8GB), you can try:
// #define M 16384
// #define N 16384
// #define K 16384

#define CUDA_CHECK(err) gpuAssert((err), __FILE__, __LINE__)
inline void gpuAssert(cudaError_t code, const char* file, int line, bool abort=true){
  if(code != cudaSuccess){
    fprintf(stderr, "GPUAssert: %s %s %d\n", cudaGetErrorString(code), file, line);
    if(abort) exit(code);
  }
}

#define CUBLAS_CHECK(err) cublasAssert((err), __FILE__, __LINE__)
inline void cublasAssert(cublasStatus_t code, const char* file, int line, bool abort=true){
  if(code != CUBLAS_STATUS_SUCCESS){
    fprintf(stderr, "CUBLASAssert: %d %s %d\n", code, file, line);
    if(abort) exit(code);
  }
}

double get_time(){
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

int main(){
  printf("Matrix sizes: M=%d, N=%d, K=%d\n", M, N, K);
  printf("Estimated GPU memory usage: %.2f GB\n",
         (M*K + K*N + M*N) * 6.0 / (1024*1024*1024));

  // Allocate on heap (too large for stack)
  float *a = new float[M*K];
  float *b = new float[K*N];
  float *c_cublas_s = new float[M*N];
  float *c_cublas_h = new float[M*N];

  // Initialize with random values
  for(int i=0; i<M*K; i++){
    a[i] = (float)(rand() % 100) / 10.0f;
  }
  for(int i=0; i<K*N; i++){
    b[i] = (float)(rand() % 100) / 10.0f;
  }

  cublasHandle_t handle;
  CUBLAS_CHECK(cublasCreate(&handle));

  float *d_a, *d_b, *d_c;
  CUDA_CHECK(cudaMalloc(&d_a, M*K*sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_b, K*N*sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_c, M*N*sizeof(float)));

  CUDA_CHECK(cudaMemcpy(d_a, a, M*K*sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_b, b, K*N*sizeof(float), cudaMemcpyHostToDevice));

  // row major A =
  // 1.0 2.0 3.0 4.0
  // 5.0 6.0 7.0 8.0

  // col major A =
  // 1.0 5.0
  // 2.0 6.0
  // 3.0 7.0
  // 4.0 8.0

  // memory layout (row)
  // 1.0 2.0 3.0 4.0 5.0 6.0 7.0 8.0

  // memory layout (col)
  // 1.0 5.0 2.0 6.0 3.0 7.0 4.0 8.0

  // copied the above explanation directly from this piece of code: [https://github.com/Infatoshi/cuda-course/blob/master/06_CUDA_APIs/01%20CUBLAS/01%20cuBLAS/01_Hgemm_Sgemm.cu](https://github.com/Infatoshi/cuda-course/blob/master/06_CUDA_APIs/01%20CUBLAS/01%20cuBLAS/01_Hgemm_Sgemm.cu)

  float alpha = 1.0f, beta = 0.0f;
  // the alpha and beta values over here is coz, C = alpha * A @ B + beta * C, this is how mat mul is done usually, the beta is like a noise, if we want to multiply and add values to the existing c after multiplying, as we can see this op in ml, even simpler version in every forward pass, y=w*x+b

  // now performing the sgemm (single precision general matrix multiply)
  double sgemm_start_time = get_time();
  CUBLAS_CHECK(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, d_b, N, d_a, K, &beta, d_c, N));
  CUDA_CHECK(cudaDeviceSynchronize()); // Ensure kernel completes for accurate timing
  double sgemm_end_time = get_time();

  // CUBLAS_OP_N -> no transpose, as we are already inverted the order of the matrix, and also gave the column major as the leading thing, this below explanation would suffice:
  // A is M x K (row-major), cuBLAS sees it as A^T (K x M, column-major),
  //   the leading dimension of A^T is K
  // B is K x N (row-major), cuBLAS sees it as B^T (N x K, column-major),
  //   the leading dimension of B^T is N
  // C is M x N (row-major), cuBLAS sees it as C^T (N x M, column-major),
  //   the leading dimension of C^T is N

  // note the swapped A and B, and the swapped M and N
  // cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
  //             N, M, K,
  //             alpha,
  //             B, N,  // leading dimension of B^T
  //             A, K,  // leading dimension of A^T
  //             beta,
  //             C, N); // leading dimension of C^T

  double sgemm_total_time = sgemm_end_time - sgemm_start_time;

  CUDA_CHECK(cudaMemcpy(c_cublas_s, d_c, M*N*sizeof(float), cudaMemcpyDeviceToHost));

  // half gemm, which is basically half precision gemm
  // for clarification single precision is basically 32 bits -> fp32, and half precision is 16 bits -> fp16

  half *d_a_h, *d_b_h, *d_c_h;
  CUDA_CHECK(cudaMalloc(&d_a_h, M*K*sizeof(half)));
  CUDA_CHECK(cudaMalloc(&d_b_h, K*N*sizeof(half)));
  CUDA_CHECK(cudaMalloc(&d_c_h, M*N*sizeof(half)));

  half *a_h = new half[M*K];
  half *b_h = new half[K*N];

  for(int i=0; i<M*K; i++){
    a_h[i] = __float2half(a[i]);
  }

  for(int i=0; i<K*N; i++){
    b_h[i] = __float2half(b[i]);
  }

  CUDA_CHECK(cudaMemcpy(d_a_h, a_h, M*K*sizeof(half), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_b_h, b_h, K*N*sizeof(half), cudaMemcpyHostToDevice));

  half alpha_h = __float2half(alpha), beta_h = __float2half(beta);
  double hgemm_start_time = get_time();
  CUBLAS_CHECK(cublasHgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha_h, d_b_h, N, d_a_h, K, &beta_h, d_c_h, N));
  CUDA_CHECK(cudaDeviceSynchronize());
  double hgemm_end_time = get_time();

  double hgemm_total_time = hgemm_end_time - hgemm_start_time;

  half *c_h = new half[M*N];
  CUDA_CHECK(cudaMemcpy(c_h, d_c_h, M*N*sizeof(half), cudaMemcpyDeviceToHost));
  for(int i=0; i<M*N; i++){
    c_cublas_h[i] = __half2float(c_h[i]);
  }

  printf("\nSample CUBLAS SGEMM Results (first 5x5 block):\n");
  for(int i=0; i<min(5,M); i++){
    for(int j=0; j<min(5,N); j++){
      printf("%.2f ", c_cublas_s[i*N+j]);
    }
    printf("\n");
  }

  printf("\nSample CUBLAS HGEMM Results (first 5x5 block):\n");
  for(int i=0; i<min(5,M); i++){
    for(int j=0; j<min(5,N); j++){
      printf("%.2f ", c_cublas_h[i*N+j]);
    }
    printf("\n");
  }

  printf("\nPerformance:\n");
  printf("CUBLAS SGEMM Time: %.6f seconds (%.2f GFLOPS)\n",
         sgemm_total_time, (2.0*M*N*K)/(sgemm_total_time*1e9));
  printf("CUBLAS HGEMM Time: %.6f seconds (%.2f GFLOPS)\n",
         hgemm_total_time, (2.0*M*N*K)/(hgemm_total_time*1e9));
  printf("HGEMM Speedup: %.2fx\n", sgemm_total_time/hgemm_total_time);

  // Cleanup
  delete[] a;
  delete[] b;
  delete[] c_cublas_s;
  delete[] c_cublas_h;
  delete[] a_h;
  delete[] b_h;
  delete[] c_h;

  CUDA_CHECK(cudaFree(d_a));
  CUDA_CHECK(cudaFree(d_b));
  CUDA_CHECK(cudaFree(d_c));
  CUDA_CHECK(cudaFree(d_a_h));
  CUDA_CHECK(cudaFree(d_b_h));
  CUDA_CHECK(cudaFree(d_c_h));
  CUBLAS_CHECK(cublasDestroy(handle));
  return 0;
}
