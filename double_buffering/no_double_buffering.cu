#include <bits/stdc++.h>
#include <cuda_runtime.h>

using namespace std;

#define N 1000

__global__ void simple_kernel(float* data, int size){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < size){
      data[idx] = sqrtf(data[idx]) * 2.0f;
    }
}

int main(){
  float *h_data;
  cudaMallocHost(&h_data, N*sizeof(float)); // pinned mem

  float *d_data;
  cudaMalloc(&d_data, N*sizeof(float));

  for(int i=0;i<N;i++){
    h_data[i] = (float)(i+1);
  }

  int blockSize = 256;
  int numBlocks = (N + blockSize - 1) / blockSize;

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  cudaEventRecord(start);

  cudaMemcpyAsync(d_data, h_data, N*sizeof(float), cudaMemcpyHostToDevice);

  simple_kernel<<<numBlocks, blockSize>>>(d_data, N);

  cudaMemcpyAsync(h_data, d_data, N*sizeof(float), cudaMemcpyDeviceToHost);

  cudaDeviceSynchronize();

  cudaEventRecord(stop);
  cudaEventSynchronize(stop);

  float ms = 0;
  cudaEventElapsedTime(&ms, start, stop);
  printf("Time taken: %f ms\n", ms);

  cudaFree(d_data);
  cudaFreeHost(h_data);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  return 0;
}