#include <bits/stdc++.h>
#include <cuda_runtime.h>

using namespace std;

#define N 1000
#define CHUNK_SIZE 100
#define NUM_STREAMS 4

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
  int numBlocks = (CHUNK_SIZE + blockSize - 1) / blockSize;

  cudaStream_t streams[NUM_STREAMS];
  for(int i=0;i<NUM_STREAMS;i++){
    cudaStreamCreate(&streams[i]);
  }

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  cudaEventRecord(start);

  int numChunks = (N+ CHUNK_SIZE - 1) / CHUNK_SIZE;

  for(int i=0;i<numChunks;i++){
    int stream_id = i % NUM_STREAMS;
    int offset = i * CHUNK_SIZE;
    int current_size = (i==CHUNK_SIZE-1) ? (N - offset) : CHUNK_SIZE;

    cudaMemcpyAsync(d_data, &h_data[offset], current_size*sizeof(float), cudaMemcpyHostToDevice, streams[stream_id]);

    int grid_size_adj = (current_size + blockSize - 1) / blockSize;
    simple_kernel<<<grid_size_adj, blockSize, 0, streams[stream_id]>>>(d_data, current_size);

    cudaMemcpyAsync(&h_data[offset], d_data, current_size*sizeof(float), cudaMemcpyDeviceToHost, streams[stream_id]);
  }

  for(int i=0;i<NUM_STREAMS;i++){
    cudaStreamSynchronize(streams[i]);
  }

  cudaEventRecord(stop);
  cudaEventSynchronize(stop);

  float ms = 0;
  cudaEventElapsedTime(&ms, start, stop);
  printf("Time taken: %f ms\n", ms);

  for(int i=0;i<NUM_STREAMS;i++){
    cudaStreamDestroy(streams[i]);
  }

  cudaFree(d_data);
  cudaFreeHost(h_data);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  return 0;
}