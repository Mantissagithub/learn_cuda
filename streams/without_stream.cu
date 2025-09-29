#include <iostream>
#include <cuda_runtime.h>
#include <time.h>

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
    double startime = get_time();

    float *h_a = (float*)malloc(bytes);
    float *h_b = (float*)malloc(bytes);
    float *h_c = (float*)malloc(bytes);
    
    for(int i = 0; i < size; i++) {
        h_a[i] = i;
        h_b[i] = i * 2;
    }
    
    float *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, bytes);
    cudaMalloc(&d_b, bytes);
    cudaMalloc(&d_c, bytes);

    cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice);
    
    dim3 block(256);
    dim3 grid((size + block.x - 1) / block.x);
    vectorAdd<<<grid, block>>>(d_a, d_b, d_c, size);
    
    cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost);
    
    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
    free(h_a); free(h_b); free(h_c);

    double endtime = get_time();
    printf("Time taken without streams: %.6f seconds\n", endtime - startime);

    return 0;
}
