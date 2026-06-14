#include <bits/stdc++.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cublasXt.h>
#include <ctime>

// command to run: nvcc -o cublas_xt cublas_xt.cu -lcublas && ./cublas_xt

int M = 1024/4;
int N = 1024/4;
int K = 1024/4;

#define CHECK_CUBLAS(call) { cublasStatus_t err = call; if (err != CUBLAS_STATUS_SUCCESS) { std::cerr << "Error in " << #call << ", line " << __LINE__ << std::endl; exit(1); } }

int main(){

  srand(time(0));

  float *a_h = new float[M*K];
  float *b_h = new float[K*N];
  float *c_h = new float[M*N];
  float *c_gpu = new float[M*N];

  for(int i=0; i<M*K; i++) a_h[i] = static_cast<float>(rand()) / RAND_MAX;
  for(int i=0; i<K*N; i++) b_h[i] = static_cast<float>(rand()) / RAND_MAX;

  float alpha = 1.0f;
  float beta = 0.0f;

  for(int i=0;i<M;i++){
    for(int j=0;j<N;j++){
      c_h[i*N + j] = 0.0f;
      for(int k=0;k<K;k++){
        c_h[i*N + j] += a_h[i*K + k] * b_h[k*N + j];
      }
    }
  }

  cublasXtHandle_t handle;
  CHECK_CUBLAS( cublasXtCreate(&handle) );

  int devices[1] = {0};
  CHECK_CUBLAS( cublasXtDeviceSelect(handle, 1, devices) );

  CHECK_CUBLAS(cublasXtSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                             M, N, K,
                             &alpha,
                             a_h, M,
                             b_h, K,
                             &beta,
                             c_gpu, M));

  float max_diff = 0.0f;
  for(int i=0;i<M*N;i++){
    float diff = std::abs(c_h[i] - c_gpu[i]);
    if(diff > max_diff) max_diff = diff;
  }

  std::cout << "Max difference: " << max_diff << std::endl;

  delete[] a_h;
  delete[] b_h;
  delete[] c_h;
  delete[] c_gpu;

  return 0;
}