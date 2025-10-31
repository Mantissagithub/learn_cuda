#include <bits/stdc++.h>
#include <cublas_v2.h>
#include <cutlass/gemm/device/gemm.h>
#include <cutlass/util/host_tensor.h>
#include <cutlass/util/reference/device/gemm.h>
#include <cutlass/util/reference/host/gemm.h>
#include <cutlass/util/tensor_view_io.h>

using namespace std;

// command to run:
//  nvcc compare.cu -o compare \
  -I/home/$(whoami)/cutlass/include \
  -I/home/$(whoami)/cutlass/tools/util/include \
  -lcublas \
  -O3 \
  -std=c++17 \
  --expt-relaxed-constexpr \
  -arch=sm_89 \
  && ./compare

// results:
// CUBLAS SGEMM Time for 100 runs: 28.1186 ms
// Cutlass SGEMM Time for 100 runs: 26.2378 ms
// Results match!

#define CUDA_CHECK(err) gpuAssert((err), __FILE__, __LINE__)
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
    if (code != cudaSuccess)
    {
        fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

#define CUBLAS_CHECK(err) cublasAssert((err), __FILE__, __LINE__)
inline void cublasAssert(cublasStatus_t code, const char *file, int line, bool abort=true)
{
    if (code != CUBLAS_STATUS_SUCCESS)
    {
        fprintf(stderr,"CUBLASassert: %d %s %d\n", code, file, line);
        if (abort) exit(code);
    }
}

bool verify_results(const vector<float>& a, const vector<float>&b, int rows, int cols){
  float epsilon = 1e-2; // the tolerance epsilon till which the results can deviate
  for(int i=0; i<rows; i++){
    for(int j=0; j<cols; j++){
      float diff = fabs(a[i*cols + j] - b[i*cols + j]);
      if(diff > epsilon){
        cout<<"Mismatch at ("<<i<<","<<j<<"): "<<a[i*cols + j]<<" vs "<<b[i*cols + j]<<endl;
        return false;
      }
    }
  }
  return true;
}

int main(){
  int m=1024, n=1024, k=1024;
  size_t a_size = m*k*sizeof(float);
  size_t b_size = k*n*sizeof(float);
  size_t c_size = m*n*sizeof(float);

  vector<float> h_a(m*k);
  vector<float> h_b(k*n);
  vector<float> h_cublas(m*n);
  vector<float> h_cutlass(m*n);

  // std::mt19937 is a typedef for a Mersenne Twister pseudo-random number generator engine, specifically configured with a 32-bit word size and a state size of 624, as described in the 1998 paper by Matsumoto and Nishimura. It is widely used for generating high-quality uniform random numbers and is part of the C++ Standard Library.
  mt19937 rng(random_device{}());
  uniform_real_distribution<float> dist(0.0f, 1.0f);

  for(int i=0;i<m*k;i++) h_a[i] = dist(rng);
  for(int i=0;i<k*n;i++) h_b[i] = dist(rng);

  float *d_a, *d_b, *d_c;
  CUDA_CHECK(cudaMalloc(&d_a, a_size));
  CUDA_CHECK(cudaMalloc(&d_b, b_size));
  CUDA_CHECK(cudaMalloc(&d_c, c_size));

  CUDA_CHECK(cudaMemcpy(d_a, h_a.data(), a_size, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_b, h_b.data(), b_size, cudaMemcpyHostToDevice));

  cublasHandle_t handle;
  CUBLAS_CHECK(cublasCreate(&handle));
  float alpha = 1.0f;
  float beta = 0.0f;

  //warmup before profiling
  for(int i=0;i<10;i++){
    CUBLAS_CHECK(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &alpha, d_a, m, d_b, k, &beta, d_c, m));
  }
  CUDA_CHECK(cudaDeviceSynchronize());

  auto start = chrono::high_resolution_clock::now();
  for(int i=0;i<100;i++){
    CUBLAS_CHECK(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &alpha, d_a, m, d_b, k, &beta, d_c, m));
  }
  CUDA_CHECK(cudaDeviceSynchronize());
  auto end = chrono::high_resolution_clock::now();
  chrono::duration<double, std::milli> cublas_duration = end - start;
  cout<<"CUBLAS SGEMM Time for 100 runs: "<<cublas_duration.count()<<" ms"<<endl;
  CUDA_CHECK(cudaMemcpy(h_cublas.data(), d_c, c_size, cudaMemcpyDeviceToHost));

  using ColumnMajor = cutlass::layout::ColumnMajor;
  using Gemm = cutlass::gemm::device::Gemm<float, ColumnMajor, float, ColumnMajor, float, ColumnMajor>;

  Gemm gemm_op;
  Gemm::Arguments args({m, n, k}, {d_a, m}, {d_b, k}, {d_c, m}, {d_c, m}, {alpha, beta}); // so the params here are: M,N,K, A, B, C, D, alpha, beta

  //warmup runs
  for(int i=0;i<10;i++){
    cutlass::Status status = gemm_op(args);
    if(status != cutlass::Status::kSuccess){
      cout<<"Cutlass GEMM failed with error: "<<static_cast<int>(status)<<endl;
      return -1;
    }
  }

  CUDA_CHECK(cudaDeviceSynchronize());

  start = chrono::high_resolution_clock::now();
  for(int i=0;i<100;i++){
    cutlass::Status status = gemm_op(args);
    if(status != cutlass::Status::kSuccess){
      cout<<"Cutlass GEMM failed with error: "<<static_cast<int>(status)<<endl;
      return -1;
    }
  }
  CUDA_CHECK(cudaDeviceSynchronize());
  end = chrono::high_resolution_clock::now();
  chrono::duration<double, std::milli> cutlass_duration = end - start;
  cout<<"Cutlass SGEMM Time for 100 runs: "<<cutlass_duration.count()<<" ms"<<endl;
  CUDA_CHECK(cudaMemcpy(h_cutlass.data(), d_c, c_size, cudaMemcpyDeviceToHost));

  bool correct = verify_results(h_cublas, h_cutlass, m, n);
  if(correct){
    cout<<"Results match!"<<endl;
  } else {
    cout<<"Results do not match!"<<endl;
  }

  cublasDestroy(handle);
  CUDA_CHECK(cudaFree(d_a));
  CUDA_CHECK(cudaFree(d_b));
  CUDA_CHECK(cudaFree(d_c));

  return 0;
}


