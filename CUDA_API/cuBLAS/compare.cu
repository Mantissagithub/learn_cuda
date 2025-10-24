#include <bits/stdc++.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cublasLt.h>
#include <cublasXt.h>
#include <cuda_fp16.h>
#include <ctime>

// to run this code: nvcc compare.cu -o compare -lcublas -lcublasLt && ./compare

using namespace std;

#define M 4096
#define N 4096
#define K 4096

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
  printf("matrix sizes: M=%d, N=%d, K=%d\n", M, N, K);
  printf("estimated gpu memory usage: %.2f GB\n",
         (M*K + K*N + M*N) * 3.0 * 4.0 / (1024*1024*1024));

  srand(time(0));

  float *a = new float[M*K];
  float *b = new float[K*N];
  float *c = new float[M*N];

  for(int i=0; i<M*K; i++){
    a[i] = (float)(rand() % 100) / 10.0f;
  }
  for(int i=0; i<K*N; i++){
    b[i] = (float)(rand() % 100) / 10.0f;
  }

  float *d_a, *d_b, *d_c;
  CUDA_CHECK(cudaMalloc(&d_a, M*K*sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_b, K*N*sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_c, M*N*sizeof(float)));

  CUDA_CHECK(cudaMemcpy(d_a, a, M*K*sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_b, b, K*N*sizeof(float), cudaMemcpyHostToDevice));

  float alpha = 1.0f, beta = 0.0f;

  printf("\n========== cublas v2 sgemm ==========\n");

  cublasHandle_t handle;
  CUBLAS_CHECK(cublasCreate(&handle));

  double v2_start = get_time();
  CUBLAS_CHECK(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, d_b, N, d_a, K, &beta, d_c, N));
  CUDA_CHECK(cudaDeviceSynchronize());
  double v2_end = get_time();
  double v2_time = v2_end - v2_start;

  CUDA_CHECK(cudaMemcpy(c, d_c, M*N*sizeof(float), cudaMemcpyDeviceToHost));

  printf("time: %.6f seconds\n", v2_time);
  printf("gflops: %.2f\n", (2.0*M*N*K)/(v2_time*1e9));
  printf("sample result (5x5): \n");
  for(int i=0; i<min(5,M); i++){
    for(int j=0; j<min(5,N); j++){
      printf("%.2f ", c[i*N+j]);
    }
    printf("\n");
  }

  printf("\n========== cublaslt ==========\n");

  cublasLtHandle_t handle_lt;
  CUBLAS_CHECK(cublasLtCreate(&handle_lt));

  cublasLtMatrixLayout_t mata_lt, matb_lt, matc_lt;
  CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&mata_lt, CUDA_R_32F, K, M, K));
  CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&matb_lt, CUDA_R_32F, N, K, N));
  CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&matc_lt, CUDA_R_32F, N, M, N));

  cublasLtMatmulDesc_t matmul_desc_lt;
  CUBLAS_CHECK(cublasLtMatmulDescCreate(&matmul_desc_lt, CUBLAS_COMPUTE_32F, CUDA_R_32F));

  cublasOperation_t transa = CUBLAS_OP_N;
  cublasOperation_t transb = CUBLAS_OP_N;

  CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(matmul_desc_lt, CUBLASLT_MATMUL_DESC_TRANSA, &transa, sizeof(transa)));
  CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(matmul_desc_lt, CUBLASLT_MATMUL_DESC_TRANSB, &transb, sizeof(transb)));

  double lt_start = get_time();
  CUBLAS_CHECK(cublasLtMatmul(handle_lt, matmul_desc_lt, &alpha, d_b, matb_lt, d_a, mata_lt, &beta, d_c, matc_lt, d_c, matc_lt, NULL, NULL, 0, 0));
  CUDA_CHECK(cudaDeviceSynchronize());
  double lt_end = get_time();
  double lt_time = lt_end - lt_start;

  CUDA_CHECK(cudaMemcpy(c, d_c, M*N*sizeof(float), cudaMemcpyDeviceToHost));

  printf("time: %.6f seconds\n", lt_time);
  printf("gflops: %.2f\n", (2.0*M*N*K)/(lt_time*1e9));
  printf("sample result (5x5): \n");
  for(int i=0; i<min(5,M); i++){
    for(int j=0; j<min(5,N); j++){
      printf("%.2f ", c[i*N+j]);
    }
    printf("\n");
  }

  printf("\n========== cublasxt ==========\n");

  cublasXtHandle_t handle_xt;
  CUBLAS_CHECK(cublasXtCreate(&handle_xt));

  int devices[1] = {0};
  CUBLAS_CHECK(cublasXtDeviceSelect(handle_xt, 1, devices));

  double xt_start = get_time();
  CUBLAS_CHECK(cublasXtSgemm(handle_xt, CUBLAS_OP_N, CUBLAS_OP_N,
                             M, N, K,
                             &alpha,
                             d_a, M,
                             d_b, K,
                             &beta,
                             d_c, M));
  CUDA_CHECK(cudaDeviceSynchronize());
  double xt_end = get_time();
  double xt_time = xt_end - xt_start;

  CUDA_CHECK(cudaMemcpy(c, d_c, M*N*sizeof(float), cudaMemcpyDeviceToHost));

  printf("time: %.6f seconds\n", xt_time);
  printf("gflops: %.2f\n", (2.0*M*N*K)/(xt_time*1e9));
  printf("sample result (5x5): \n");
  for(int i=0; i<min(5,M); i++){
    for(int j=0; j<min(5,N); j++){
      printf("%.2f ", c[i*N+j]);
    }
    printf("\n");
  }

  printf("\n========== summary ==========\n");
  printf("cublas v2: %.6f sec (%.2f gflops)\n", v2_time, (2.0*M*N*K)/(v2_time*1e9));
  printf("cublaslt:  %.6f sec (%.2f gflops)\n", lt_time, (2.0*M*N*K)/(lt_time*1e9));
  printf("cublasxt:  %.6f sec (%.2f gflops)\n", xt_time, (2.0*M*N*K)/(xt_time*1e9));

  printf("\nrelative speedup:\n");
  printf("lt vs v2: %.2fx\n", v2_time / lt_time);
  printf("xt vs v2: %.2fx\n", v2_time / xt_time);
  printf("xt vs lt: %.2fx\n", lt_time / xt_time);

  CUDA_CHECK(cudaFree(d_a));
  CUDA_CHECK(cudaFree(d_b));
  CUDA_CHECK(cudaFree(d_c));

  CUBLAS_CHECK(cublasDestroy(handle));
  CUBLAS_CHECK(cublasLtDestroy(handle_lt));
  CUBLAS_CHECK(cublasXtDestroy(handle_xt));

  CUBLAS_CHECK(cublasLtMatrixLayoutDestroy(mata_lt));
  CUBLAS_CHECK(cublasLtMatrixLayoutDestroy(matb_lt));
  CUBLAS_CHECK(cublasLtMatrixLayoutDestroy(matc_lt));
  CUBLAS_CHECK(cublasLtMatmulDescDestroy(matmul_desc_lt));

  delete[] a;
  delete[] b;
  delete[] c;

  return 0;
}
