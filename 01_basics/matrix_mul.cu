// Why 32×32 for Matrix Multiplication?
// 1. Maximum Occupancy
// Most GPUs support up to 1,024 threads per block maximum

// 32×32 = 1,024 hits this limit perfectly

// This gives you the maximum possible threads per block

// 2. Shared Memory Efficiency
// For tiled matrix multiplication algorithms:

// Each block loads a 32×32 tile of matrix A and B into shared memory

// 32×32 = 1,024 float values = 4KB of shared memory per tile

// This fits well within typical shared memory limits (48KB-64KB per block)

// 3. Memory Coalescing
// 32 threads in a row access consecutive memory locations

// This aligns perfectly with warp size (32 threads)

// Each warp loads/stores a complete row of the tile efficiently

// 4. Computational Balance
// Research shows that 32×32 thread blocks are often optimal for matrix multiplication:

// Good balance between occupancy and resource usage

// Minimizes overhead from block management

// Maximizes data reuse in shared memory

// so gonna assign 1024 threads with diff blockdim configs

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <time.h>

#define N 256
#define M 512
#define K 256
#define BLOCK_SIZE 32

void matmul_cpu(float *a, float *b, float *c, int m, int n, int k){
    for(int i=0;i<m;i++){  // Changed from n to m
        for(int j=0;j<n;j++){   // Changed from m to n
            float sum = 0.0f;
            for(int l=0;l<k;l++){
                sum += a[i*k + l] * b[l*n+j];
            }
            c[i*n+j] = sum;
        }
    }
}

__global__ void matmul_gpu(float *a, float *b, float *c, int m, int n, int k){
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if(row < m && col < n){     
        float sum = 0.0f;
        for(int l=0;l<k;l++){
            sum += a[row*k + l] * b[l*n+col];
        }
        c[row*n+col] = sum;
    }
}

void _init_matrix(float *mat, int rows, int cols){  
    for(int i=0;i<rows;i++){
        for(int j=0;j<cols;j++){
            mat[i*cols+j] = (float)rand()/RAND_MAX;
        }
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

    int size_a = M * K * sizeof(float);
    int size_b = K * N * sizeof(float);
    int size_c = M * N * sizeof(float);

    h_a = (float*)malloc(size_a);
    h_b = (float*)malloc(size_b);
    h_c_cpu = (float*)malloc(size_c);
    h_c_gpu = (float*)malloc(size_c);

    srand(time(NULL));
    _init_matrix(h_a, M, K);
    _init_matrix(h_b, K, N);

    cudaMalloc(&d_a, size_a);
    cudaMalloc(&d_b, size_b);
    cudaMalloc(&d_c, size_c);

    cudaMemcpy(d_a, h_a, size_a, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, size_b, cudaMemcpyHostToDevice);

    dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridDim((N + BLOCK_SIZE - 1) / BLOCK_SIZE, (M + BLOCK_SIZE - 1) / BLOCK_SIZE);

    printf("Benchmarking cpu implementation...\n");
    float cpu_total_time = 0.0f;
    for(int i=0;i<20;i++){
        double start = get_time();
        matmul_cpu(h_a, h_b, h_c_cpu, M, N, K);
        double end = get_time();
        cpu_total_time += (end - start);
    }
    float cpu_avg_time = cpu_total_time / 20.0;
    // printf("Average CPU time: %f seconds\n", cpu_avg_time);

    printf("benchmarking gpu implementation...\n");
    float gpu_total_time = 0.0f;
    for(int i=0;i<20;i++){
        double start = get_time();
        matmul_gpu<<<gridDim, blockDim>>>(d_a, d_b, d_c, M, N, K);
        cudaDeviceSynchronize();
        double end = get_time();
        gpu_total_time += (end - start);
    }
    float gpu_avg_time = gpu_total_time / 20.0;

    printf("CPU average time: %f microseconds\n", (cpu_avg_time * 1e6f));
    printf("GPU average time: %f microseconds\n", (gpu_avg_time * 1e6f));
    printf("Speedup: %fx\n", cpu_avg_time / gpu_avg_time);

    free(h_a);
    free(h_b);
    free(h_c_cpu);
    free(h_c_gpu);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    return 0;
}