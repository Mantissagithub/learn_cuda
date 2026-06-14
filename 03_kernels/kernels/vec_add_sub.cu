// 1. Element-wise Vector Operations (Easy)
// Implement kernels for:

// Vector subtraction: c[i] = a[i] - b[i]

// Vector element-wise multiplication: c[i] = a[i] * b[i]

// Vector mathematical functions: c[i] = sqrt(a[i] * a[i] + b[i] * b[i]) (magnitude)

// Compare performance with your vector addition kernel.

// Goal: Get comfortable with different mathematical operations and function calls in kernels.

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <time.h>

#define N 1000000
#define BLOCK_SIZE 256 // each block has 8 warps

void vector_sub_cpu(float *a, float *b, float *c, int n){
    for(int i=0;i<n;i++){
        c[i] = fabs(a[i] - b[i]);
    }
}

__global__ void vector_sub_gpu(float *a, float *b, float *c, int n){
    int index = threadIdx.x + blockIdx.x * blockDim.x;

    if(index < n){
        c[index] = fabs(a[index] - b[index]);
    }
}

void init_vector(float *vec, int n){
    for(int i=0;i<n;i++){
        vec[i] = (float)rand()/RAND_MAX;
    }
}

double get_time(){
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

int main(){
    float *h_a, *h_b, *h_c_cpu, *h_c_gpu;
    float *d_a, *d_b, *d_c;

    size_t size = N * sizeof(float);

    h_a = (float*)malloc(size);
    h_b = (float*)malloc(size);
    h_c_cpu = (float*)malloc(size);
    h_c_gpu = (float*)malloc(size);

    cudaMalloc(&d_a, size);
    cudaMalloc(&d_b, size);
    cudaMalloc(&d_c, size);

    srand(time(NULL));
    init_vector(h_a, N);
    init_vector(h_b, N);

    cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice);

    int num_blocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;

    printf("bencmarking cpu implementation...\n");
    float cpu_total_time = 0.0f;
    for(int i=0;i<20;i++){
        double start = get_time();
        vector_sub_cpu(h_a, h_b, h_c_cpu, N);
        double end = get_time();
        cpu_total_time += end - start;
    }
    float cpu_avg_time = cpu_total_time / 20.0;

    printf("benchmarking gpu implementation...\n");
    float gpu_total_time = 0.0f;
    for(int i=0;i<20;i++){
        double start = get_time();
        vector_sub_gpu<<<num_blocks, BLOCK_SIZE>>>(d_a, d_b, d_c, N);
        cudaDeviceSynchronize();
        double end = get_time();
        gpu_total_time += end - start;
    }
    float gpu_avg_time = gpu_total_time / 20.0;

    printf("Average CPU time: %f seconds\n", cpu_avg_time);
    printf("Average GPU time: %f seconds\n", gpu_avg_time);
    printf("Speedup: %f\n", cpu_avg_time / gpu_avg_time);

    free(h_a);
    free(h_b);
    free(h_c_cpu);
    free(h_c_gpu);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return 0;
}