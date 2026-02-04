#include <bits/stdc++.h>
#include <cuda_runtime.h>

using namespace std;

#define N 1000
#define CHUNK_SIZE 100
#define NUM_STREAMS 2 //double buffering, seperate streams for seperate buffers

__global__ void simple_kernel(float* data, int size){
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size){
    data[idx] += 1.0f; //identical to the offset thing in chunksize
  }
}

int main(){
  int size = N*sizeof(float);
  float *h_data;
  cudaMallocHost(&h_data, size); // pinned mem

  float *d_data[2];
  cudaMalloc(&d_data[0], size);
  cudaMalloc(&d_data[1], size);

  for(int i=0;i<N;i++){
    h_data[i] = (float)i;
  }

  cudaStream_t streams[NUM_STREAMS];

  for(int i=0;i<NUM_STREAMS;i++){
    cudaStreamCreate(&streams[i]);
  }

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  cudaEventRecord(start);

  int numChunks = (N + CHUNK_SIZE - 1) / CHUNK_SIZE;

  for(int chunk=0;chunk<numChunks;chunk++){
    int chunkIdx = chunk % NUM_STREAMS;
    int offset = chunk * CHUNK_SIZE;
    int currentChunkSize = min(CHUNK_SIZE, N - offset);

    cudaMemcpyAsync(d_data[chunkIdx] + offset,
                h_data + offset,
                currentChunkSize * sizeof(float),
                cudaMemcpyHostToDevice,
                streams[chunkIdx]);

    int threadsPerBlock = 256;
    int blocksPerGrid = (currentChunkSize + threadsPerBlock - 1) / threadsPerBlock;
    simple_kernel<<<blocksPerGrid, threadsPerBlock, 0, streams[chunkIdx]>>>(
          d_data[chunkIdx] + offset,  // start at chunk offset
          currentChunkSize           // only process this chunk's size
      );

    cudaMemcpyAsync(h_data + offset,
                d_data[chunkIdx] + offset,
                currentChunkSize * sizeof(float),
                cudaMemcpyDeviceToHost,
                streams[chunkIdx]);
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

  return 0;
}