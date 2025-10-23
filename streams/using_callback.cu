#include <bits/stdc++.h>
#include <cuda_runtime.h>
#include <time.h>

using namespace std;

#define CUDA_CHECK(err) gpuAssert((err), __FILE__, __LINE__)
inline void gpuAssert(cudaError_t code, const char* file, int line, bool abort=true){
  if(code != cudaSuccess){
    fprintf(stderr, "GPUAssert: %s %s %d\n", cudaGetErrorString(code), file, line);
    if(abort) exit(code);
  }
}

__global__ void kernel2(float* data, int n){
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if(idx < n){
    data[idx] *= 2.0f;
  }
}

__global__ void kernel1(float* data, int n){
  int idx = blockDim.x * blockIdx.x + threadIdx.x;
  if(idx < n){
    data[idx] += 1.0f;
  }
}

void CUDART_CB streamCallback(cudaStream_t stream, cudaError_t status, void *UserData){
  printf("Stream callback, operation completed\n");
}

int main(){
  int N = 10000;
  size_t size = N * sizeof(float);
  float *h_data, *d_data;
  cudaStream_t stream1, stream2;
  cudaEvent_t event;

  // cout<<event<<endl;

  CUDA_CHECK(cudaMallocHost(&h_data, size)); // pinned mem
  CUDA_CHECK(cudaMalloc(&d_data, size));

  for(int i=0;i<N;i++){
    h_data[i] = (float)i*2.0f;
  }

  int leastPriority, greaterPriority;
  CUDA_CHECK(cudaDeviceGetStreamPriorityRange(&leastPriority, &greaterPriority));
  CUDA_CHECK(cudaStreamCreateWithPriority(&stream1, cudaStreamNonBlocking, leastPriority));
  CUDA_CHECK(cudaStreamCreateWithPriority(&stream2, cudaStreamNonBlocking, greaterPriority));


  CUDA_CHECK(cudaEventCreate(&event));

  CUDA_CHECK(cudaMemcpyAsync(d_data, h_data, size, cudaMemcpyHostToDevice, stream1));
  kernel1<<<(N+255)/256, 256, 0, stream1>>>(d_data, N);

  CUDA_CHECK(cudaEventRecord(event, stream1));

  CUDA_CHECK(cudaStreamWaitEvent(stream2, event));

  kernel2<<<(N+255)/256, 256, 0, stream2>>>(d_data, N);

  CUDA_CHECK(cudaStreamAddCallback(stream2, streamCallback, NULL, 0));

  CUDA_CHECK(cudaMemcpyAsync(d_data, h_data, size, cudaMemcpyDeviceToHost, stream2));

  CUDA_CHECK(cudaStreamSynchronize(stream1));
  CUDA_CHECK(cudaStreamSynchronize(stream2));

  CUDA_CHECK(cudaFreeHost(h_data));
  CUDA_CHECK(cudaFree(d_data));
  CUDA_CHECK(cudaStreamDestroy(stream1));
  CUDA_CHECK(cudaStreamDestroy(stream2));
  CUDA_CHECK(cudaEventDestroy(event));
  return 0;
}

// result:
//  nvcc -o using_callback using_callback.cu
//  ./using_callback
// Stream callback, operation completed
//  