#include <iostream>
#include <cuda_runtime.h>

#define NUM_STREAMS 4

__global__ void vectorAdd(float *a, float *b, float* c, int n){
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    if(idx < n){
        c[idx] = a[idx] + b[idx];
    }
}

int main(){
    int size = 1000000;
    int bytes = size * sizeof(float);

    cudaStream_t streams[NUM_STREAMS];
    for(int i=0;i<NUM_STREAMS;i++){
        cudaStreamCreate(&streams[i]);
    }

    cudaEvent_t startEvent, stopEvent;
    cudaEventCreate(&startEvent);
    cudaEventCreate(&stopEvent);

    cudaEvent_t memcpyh2d_events[NUM_STREAMS];
    cudaEvent_t kernel_events[NUM_STREAMS];
    cudaEvent_t memcpyd2h_events[NUM_STREAMS];

    for(int i=0;i<NUM_STREAMS;i++){
        cudaEventCreate(&memcpyh2d_events[i]);
        cudaEventCreate(&kernel_events[i]);
        cudaEventCreate(&memcpyd2h_events[i]);
    }

    float *h_a, *h_b, *h_c;
    cudaMallocHost((void**)&h_a, bytes);
    cudaMallocHost((void**)&h_b, bytes);
    cudaMallocHost((void**)&h_c, bytes);

    for(int i = 0; i < size; i++) {
        h_a[i] = i;
        h_b[i] = i * 2;
    }

    float *d_a[NUM_STREAMS], *d_b[NUM_STREAMS], *d_c[NUM_STREAMS];\
    int streamSize = size / NUM_STREAMS;
    int streamBytes = streamSize * sizeof(float);

    for(int i=0;i<NUM_STREAMS;i++){
        cudaMalloc(&d_a[i], bytes);
        cudaMalloc(&d_b[i], bytes);
        cudaMalloc(&d_c[i], bytes);
    }

    cudaEventRecord(startEvent);

    for(int i=0;i<NUM_STREAMS;i++){
        int offset = i * streamSize;

        cudaMemcpyAsync(d_a[i], h_a + offset, streamBytes, cudaMemcpyHostToDevice, streams[i]);
        cudaMemcpyAsync(d_b[i], h_b + offset, streamBytes, cudaMemcpyHostToDevice, streams[i]);
        cudaEventRecord(memcpyh2d_events[i], streams[i]);

        dim3 block(256);
        dim3 grid((streamSize + block.x - 1)/block.x);

        vectorAdd<<<grid, block, 0, streams[i]>>>(d_a[i], d_b[i], d_c[i], streamSize);
        cudaEventRecord(kernel_events[i], streams[i]);

        cudaMemcpyAsync(h_c + offset, d_c[i], streamBytes, cudaMemcpyDeviceToHost, streams[i]);
        cudaEventRecord(memcpyd2h_events[i], streams[i]);
    }


    cudaEventRecord(stopEvent);

    cudaEventSynchronize(stopEvent);

    float totalTime;
    cudaEventElapsedTime(&totalTime, startEvent, stopEvent);

    printf("Total time with events and streams: %f seconds\n", totalTime / 1000.0f);

    for(int i = 0; i < NUM_STREAMS; i++) {
        float h2dTime, kernelTime, d2hTime;
        
        cudaEventElapsedTime(&h2dTime, startEvent, memcpyh2d_events[i]);
         
        cudaEventElapsedTime(&kernelTime, startEvent, kernel_events[i]);
        
        cudaEventElapsedTime(&d2hTime, startEvent, memcpyd2h_events[i]);
        
        printf("Stream %d - H2D: %.3f ms, Kernel: %.3f ms, D2H: %.3f ms\n", i, h2dTime, kernelTime, d2hTime);
    }

    cudaEventDestroy(startEvent);
    cudaEventDestroy(stopEvent);

    for(int i=0;i<NUM_STREAMS;i++){
        cudaEventDestroy(memcpyh2d_events[i]);
        cudaEventDestroy(kernel_events[i]);
        cudaEventDestroy(memcpyd2h_events[i]);
    }

    for(int i=0;i<NUM_STREAMS;i++){
        cudaFree(d_a[i]); cudaFree(d_b[i]); cudaFree(d_c[i]);
        cudaStreamDestroy(streams[i]);
    }

    cudaFreeHost(h_a); cudaFreeHost(h_b); cudaFreeHost(h_c);

    return 0;
}