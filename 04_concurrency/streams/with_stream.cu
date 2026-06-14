#include <iostream>
#include <cuda_runtime.h>
#include <time.h>

#define NUM_STREAMS 4

__global__ void vectorAdd(float* a, float* b, float* c, int n){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx<n){
        c[idx] = a[idx] + b[idx];
    }
}

double get_time(){
    struct timespec tp;
    clock_gettime(CLOCK_MONOTONIC, &tp);
    return tp.tv_sec + tp.tv_nsec * 1e-9;
}

int main(){
    int size = 1024 * 1024;
    int bytes = size * sizeof(float);
    double starttime = get_time();

    cudaStream_t streamsp[NUM_STREAMS];
    for(int i=0; i<NUM_STREAMS; i++){
        cudaStreamCreate(&streamsp[i]);
    }

    float *h_a, *h_b, *h_c;
    cudaMallocHost((void**)&h_a, bytes);
    cudaMallocHost((void**)&h_b, bytes);
    cudaMallocHost((void**)&h_c, bytes);

    for(int i = 0; i < size; i++) {
        h_a[i] = i;
        h_b[i] = i * 2;
    }

    float *d_a[NUM_STREAMS], *d_b[NUM_STREAMS], *d_c[NUM_STREAMS];
    int streamSize = size / NUM_STREAMS;
    int streamBytes = streamSize * sizeof(float);

    for(int i=0;i<NUM_STREAMS;i++){
        cudaMalloc(&d_a[i], streamBytes);
        cudaMalloc(&d_b[i], streamBytes);
        cudaMalloc(&d_c[i], streamBytes);
    }

    for(int i=0;i<NUM_STREAMS;i++){
        int offset = i * streamSize;
        cudaMemcpyAsync(d_a[i], h_a + offset, streamBytes, cudaMemcpyHostToDevice, streamsp[i]);
        cudaMemcpyAsync(d_b[i], h_b + offset, streamBytes, cudaMemcpyHostToDevice, streamsp[i]);
        int blockSize = 256;
        int numBlocks = (streamSize + blockSize - 1) / blockSize;
        vectorAdd<<<numBlocks, blockSize, 0, streamsp[i]>>>(d_a[i], d_b[i], d_c[i], streamSize);
        cudaMemcpyAsync(h_c + offset, d_c[i], streamBytes, cudaMemcpyDeviceToHost, streamsp[i]);
    }

    // now comes the imp part, where we need to synchronize and wait for the streams to finish, all of them!!
    for(int i=0;i<NUM_STREAMS;i++){
        cudaStreamSynchronize(streamsp[i]);
    }

    for(int i=0;i<NUM_STREAMS;i++){
        cudaFree(d_a[i]);
        cudaFree(d_b[i]);
        cudaFree(d_c[i]);
        cudaStreamDestroy(streamsp[i]);
    }

    cudaFreeHost(h_a);
    cudaFreeHost(h_b);
    cudaFreeHost(h_c);

    double endtime = get_time();
    printf("Total time with streams: %.6f seconds\n", endtime - starttime);
    return 0;
}