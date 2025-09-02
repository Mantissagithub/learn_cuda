// 2. Matrix Addition with Shared Memory (Medium-Easy)
// Take your matrix multiplication code and modify it to do matrix addition, but this time:

// Use shared memory to load matrix tiles (like 32×32 blocks)

// Each thread loads one element into shared memory

// Perform addition on shared memory data

// Write results back to global memory

// Goal: Learn shared memory fundamentals without complex algorithms.

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <time.h>

#define N 1000000
#define BLOCK_SIZE 256

void matrix_add_cpu(float *a, float *b, float *c, int n){
    for(int i=0;i<n;i++){
        c[i] = a[i] + b[i];
    }
}

__global__ void matrix_add_gpu(float *a, float *b, float *c, int n){
    int index = threadIdx.x + blockIdx.x * blockDim.x;

    if(index<n){
        c[index] = a[index] + b[index];
    }
}

__global__ void matrix_add_gpu_shared(float *a, float *b, float *c, int n){
    __shared__ float shared_a[BLOCK_SIZE];
    __shared__ float shared_b[BLOCK_SIZE];

    int tid = threadIdx.x;
    int index = threadIdx.x + blockIdx.x * blockDim.x;

    if(index < n){
        shared_a[tid] = a[index];
        shared_b[tid] = b[index];
    }else{
        shared_a[tid] = 0.0f;
        shared_b[tid] = 0.0f;
    }


    __syncthreads();

    if(index < n){
        c[index] = shared_a[tid] + shared_b[tid];
    }
}

void init_vector(float *vec, int n){
    for(int i=0;i<n;i++){
        vec[i] = (float)rand() / RAND_MAX;
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

    int num_blocks = (N + BLOCK_SIZE -1) / BLOCK_SIZE;
    
    printf("benchmarking cpu implementation...\n");
    float cpu_total_time = 0.0f;
    for(int i=0;i<20;i++){
        double start = get_time();
        matrix_add_cpu(h_a, h_b, h_c_cpu, N);
        double end = get_time();
        cpu_total_time += end - start;
    }
    float cpu_avg_time = cpu_total_time / 20;
    // printf("CPU average time: %.6f seconds\n", cpu_avg_time);

    printf("benchmarking gpu implementation without shared mem....\n");
    float gpu_total_time = 0.0f;
    for(int i=0;i<20;i++){
        double start = get_time();
        matrix_add_gpu<<<num_blocks, BLOCK_SIZE>>>(d_a, d_b, d_c, N);
        cudaDeviceSynchronize();
        double end = get_time();
        gpu_total_time += end - start;
    }
    float gpu_avg_time = gpu_total_time / 20;
    // printf("GPU average time (no shared mem): %.6f seconds\n", gpu_avg_time);

    printf("benchmarking gpu implementation with shared mem....\n");
    float gpu_shared_total_time = 0.0f;
    for(int i=0;i<20;i++){
        cudaMemset(d_c, 0, size); 
        double start = get_time();
        matrix_add_gpu_shared<<<num_blocks, BLOCK_SIZE>>>(d_a, d_b, d_c, N);
        cudaDeviceSynchronize();
        double end = get_time();
        gpu_shared_total_time += end - start;
    }
    float gpu_shared_avg_time = gpu_shared_total_time / 20;
    // printf("GPU average time (with shared mem): %.6f seconds\n", gpu_shared_avg_time);

    printf("CPU average time: %.6f seconds\n", cpu_avg_time);
    printf("GPU average time (no shared mem): %.6f seconds\n", gpu_avg_time);
    printf("GPU average time (with shared mem): %.6f seconds\n", gpu_shared_avg_time);
    printf("Speedup (CPU to GPU no shared mem): %.2fx\n", cpu_avg_time / gpu_avg_time);
    printf("Speedup (CPU to GPU with shared mem): %.2fx\n", cpu_avg_time / gpu_shared_avg_time);
    printf("Speedup (GPU no shared mem to GPU with shared mem): %.2fx\n", gpu_avg_time / gpu_shared_avg_time);

    free(h_a);
    free(h_b);
    free(h_c_cpu);
    free(h_c_gpu);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return 0;
}

// results:
// ~/learn_cuda on main !1 ?1 ❯ ./practice/kernels/matrix_mul_shared_mem                                             at 16:40:22
// benchmarking cpu implementation...
// benchmarking gpu implementation without shared mem....
// benchmarking gpu implementation with shared mem....
// CPU average time: 0.001410 seconds
// GPU average time (no shared mem): 0.000026 seconds
// GPU average time (with shared mem): 0.000022 seconds
// Speedup (CPU to GPU no shared mem): 54.81x
// Speedup (CPU to GPU with shared mem): 62.87x
// Speedup (GPU no shared mem to GPU with shared mem): 1.15x
              