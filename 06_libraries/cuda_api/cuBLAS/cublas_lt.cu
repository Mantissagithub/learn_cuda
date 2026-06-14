#include <bits/stdc++.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h> // this is secifically when we use "half" data type
#include <cublasLt.h>
#include <time.h>

// to run this code: nvcc cublas_lt.cu -o cublas_lt -L/usr/local/cuda/lib64 -lcublas -lcublasLt -lcudart && ./cublas_lt

using namespace std;

#define CUDA_CHECK(err) gpuAssert(err, __FILE__, __LINE__)
inline void gpuAssert(cudaError_t code, const char* file, int line, bool abort=true){
  if(code != cudaSuccess){
    fprintf(stderr, "GPUAssert: %s %s %d\n", cudaGetErrorString(code), file, line);
    if(abort) exit(code);
  }
}

#define CUBLAS_CHECK(err) cublasAssert(err, __FILE__, __LINE__)
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
  int M=4, K=4, N=4;

  float h_a[M*K] = {
      1.0f, 2.0f, 3.0f, 4.0f,
      5.0f, 6.0f, 7.0f, 8.0f,
      9.0f, 10.0f, 11.0f, 12.0f,
      13.0f, 14.0f, 15.0f, 16.0f
  };

  float h_b[K*N] = {
      1.0f, 2.0f, 3.0f, 4.0f,
      5.0f, 6.0f, 7.0f, 8.0f,
      9.0f, 10.0f, 11.0f, 12.0f,
      17.0f, 18.0f, 19.0f, 20.0f
  };

  // float h_c_cpu[  M*N] = {0};
  float h_c_gpu_fp32[M*N] = {0};
  float  h_c_gpu_fp16[M*N] = {0};

  double start, end;

  // first fp32
  float *d_a_fp32, *d_b_fp32, *d_c_fp32;
  start = get_time();
  CUDA_CHECK(cudaMalloc((void**)&d_a_fp32, M*K*sizeof(float)));
  end = get_time();
  printf("Time for cudaMalloc d_a_fp32: %f\n", end - start);
  start = get_time();
  CUDA_CHECK(cudaMalloc((void**)&d_b_fp32, K*N*sizeof(float)));
  end = get_time();
  printf("Time for cudaMalloc d_b_fp32: %f\n", end - start);
  start = get_time();
  CUDA_CHECK(cudaMalloc((void**)&d_c_fp32, M*N*sizeof(float)));
  end = get_time();
  printf("Time for cudaMalloc d_c_fp32: %f\n", end - start);

  start = get_time();
  CUDA_CHECK(cudaMemcpy(d_a_fp32, h_a, M*K*sizeof(float), cudaMemcpyHostToDevice));
  end = get_time();
  printf("Time for cudaMemcpy d_a_fp32: %f\n", end - start);
  start = get_time();
  CUDA_CHECK(cudaMemcpy(d_b_fp32, h_b, K*N*sizeof(float), cudaMemcpyHostToDevice));
  end = get_time();
  printf("Time for cudaMemcpy d_b_fp32: %f\n", end - start);

  start = get_time();
  cublasLtHandle_t handle;
  CUBLAS_CHECK(cublasLtCreate(&handle));
  end = get_time();
  printf("Time for cublasLtCreate: %f\n", end - start);

  // setting up matrix descriptors now
  start = get_time();
  cublasLtMatrixLayout_t mata_fp32, matb_fp32, matc_fp32;
  CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&mata_fp32, CUDA_R_32F, K, M, K));
  CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&matb_fp32, CUDA_R_32F, N, K, N));
  CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&matc_fp32, CUDA_R_32F, N, M, N));
  end = get_time();
  printf("Time for cublasLtMatrixLayoutCreate (all 3): %f\n", end - start);

  // now mat mul descriptors
  cublasLtMatmulDesc_t matmul_desc_fp32;
  CUBLAS_CHECK(cublasLtMatmulDescCreate(&matmul_desc_fp32, CUBLAS_COMPUTE_32F, CUDA_R_32F)); // name, compute type, scale type

  cublasOperation_t transa = CUBLAS_OP_N;
  cublasOperation_t transb = CUBLAS_OP_N;

  CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(matmul_desc_fp32, CUBLASLT_MATMUL_DESC_TRANSA, &transa, sizeof(transa)));
  CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(matmul_desc_fp32, CUBLASLT_MATMUL_DESC_TRANSB, &transb, sizeof(transb)));

  float alpha = 1.0f, beta = 0.0f;

  start = get_time();
  CUBLAS_CHECK(cublasLtMatmul(handle, matmul_desc_fp32, &alpha, d_b_fp32, matb_fp32, d_a_fp32, mata_fp32, &beta, d_c_fp32, matc_fp32, d_c_fp32, matc_fp32, NULL, NULL, 0, 0));
  end = get_time();
  printf("Time for cublasLtMatmul (fp32): %f\n", end - start);
  start = get_time();
  CUDA_CHECK(cudaMemcpy(h_c_gpu_fp32, d_c_fp32, M*N*sizeof(float), cudaMemcpyDeviceToHost));
  end = get_time();
  printf("Time for cudaMemcpy result (fp32): %f\n", end - start);

  // now fp16
  half h_a_h[M*K], h_b_h[K*N];
  start = get_time();
  for(int i=0; i<M*K; i++) h_a_h[i] = __float2half(h_a[i]);
  for(int i=0; i<K*N; i++) h_b_h[i] = __float2half(h_b[i]);
  end = get_time();
  printf("Time for half conversion: %f\n", end - start);

  half *d_a_h, *d_b_h, *d_c_h;
  CUDA_CHECK(cudaMalloc((void**)&d_a_h, M*K*sizeof(half)));
  CUDA_CHECK(cudaMalloc((void**)&d_b_h, K*N*sizeof(half)));
  CUDA_CHECK(cudaMalloc((void**)&d_c_h, M*N*sizeof(half)));
  start = get_time();
  CUDA_CHECK(cudaMemcpy(d_a_h, h_a_h, M*K*sizeof(half), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_b_h, h_b_h, K*N*sizeof(half), cudaMemcpyHostToDevice));
  end = get_time();
  printf("Time for cudaMemcpy (fp16): %f\n", end - start);

  cublasLtMatrixLayout_t mata_h, matb_h, matc_h;
  CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&mata_h, CUDA_R_16F, K, M, K));
  CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&matb_h, CUDA_R_16F, N, K, N));
  CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&matc_h, CUDA_R_16F, N, M, N));
  start = get_time();
  cublasLtMatmulDesc_t matmul_desc_h;
  CUBLAS_CHECK(cublasLtMatmulDescCreate(&matmul_desc_h, CUBLAS_COMPUTE_16F, CUDA_R_16F));
  end = get_time();
  printf("Time for cublasLtMatmulDescCreate (fp16): %f\n", end - start); // name, compute type, scale type

  CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(matmul_desc_h, CUBLASLT_MATMUL_DESC_TRANSA, &transa, sizeof(transa)));
  CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(matmul_desc_h, CUBLASLT_MATMUL_DESC_TRANSB, &transb, sizeof(transb)));

  half alpha_h = __float2half(alpha), beta_h = __float2half(beta);
  start = get_time();
  CUBLAS_CHECK(cublasLtMatmul(handle, matmul_desc_h, &alpha_h, d_b_h, matb_h, d_a_h, mata_h, &beta_h, d_c_h, matc_h, d_c_h, matc_h, NULL, NULL, 0, 0));
  end = get_time();
  printf("Time for cublasLtMatmul (fp16): %f\n", end - start);
  start = get_time();
  CUDA_CHECK(cudaMemcpy(h_c_gpu_fp16, d_c_h, M*N*sizeof(half), cudaMemcpyDeviceToHost));
  end = get_time();
  printf("Time for cudaMemcpy result (fp16): %f\n", end - start);

  // now printing results
  printf("CUBLASLt fp32 Result:\n");
  for(int i=0; i<M; i++){
    for(int j=0; j<N; j++){
      printf("%0.2f ", h_c_gpu_fp32[i*N + j]);
    }
    printf("\n");
  }

  printf("\nCUBLASLt fp16 Result:\n");
  for(int i=0; i<M; i++){
    for(int j=0; j<N; j++){
      printf("%0.2f ", __half2float(h_c_gpu_fp16[i*N + j]));
    }
    printf("\n");
  }

  // free all resources
  CUDA_CHECK(cudaFree(d_a_fp32));
  CUDA_CHECK(cudaFree(d_b_fp32));
  CUDA_CHECK(cudaFree(d_c_fp32));
  CUDA_CHECK(cudaFree(d_a_h));
  CUDA_CHECK(cudaFree(d_b_h));
  CUDA_CHECK(cudaFree(d_c_h));
  CUBLAS_CHECK(cublasLtDestroy(handle));
  CUBLAS_CHECK(cublasLtMatrixLayoutDestroy(mata_fp32));
  CUBLAS_CHECK(cublasLtMatrixLayoutDestroy(matb_fp32));
  CUBLAS_CHECK(cublasLtMatrixLayoutDestroy(matc_fp32));
  CUBLAS_CHECK(cublasLtMatrixLayoutDestroy(mata_h));
  CUBLAS_CHECK(cublasLtMatrixLayoutDestroy(matb_h));
  CUBLAS_CHECK(cublasLtMatrixLayoutDestroy(matc_h));
  CUBLAS_CHECK(cublasLtMatmulDescDestroy(matmul_desc_fp32));
  CUBLAS_CHECK(cublasLtMatmulDescDestroy(matmul_desc_h));

  return 0;
}