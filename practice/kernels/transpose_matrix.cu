// 4. Matrix Transpose (Medium-Hard)
// Implement matrix transpose two ways:

// Naive version: Direct output[col][row] = input[row][col]

// Optimized version: Use shared memory tiles to improve memory coalescing

// Measure and compare memory throughput. You'll see dramatic performance differences!

// Goal: Master memory coalescing and understand why memory access patterns matter.
// i read and understood seomthing from the official doc: https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html and has an image in the assets folder for my lookup : learn_cuda/assets/image copy.png

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <time.h>

#define MATRIX_SIZE 100
#define BLOCK_SIZE 16
#define TILE_SIZE 32

#define CUDA_CHECK(err) {gpuAssert((err), __FILE__, __LINE__);}
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true){
    if(code != cudaSuccess){
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if(abort) exit(code);
    }
}

void init_matrix(float *matrix, int size) {
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            matrix[i * size + j] = (float)(rand() % 100);
        }
    }
}

double get_time(){
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

void transpose_matrix_cpu(float *input, float *output, int size) {
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            output[j * size + i] = input[i * size + j];
        }
    }
}

__global__ void transpose_matrix_naive_gpu(float *a, float *b, int size){
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < size && col < size) {
        b[col * size + row] = a[row * size + col];
    }
}

__global__ void transpose_matrix_tiled_gpu(float *a, float *b, int size){
    __shared__ float tile[TILE_SIZE][TILE_SIZE];
    
    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;

    if (row < size && col < size) {
        tile[threadIdx.y][threadIdx.x] = a[row * size + col];
    }
    __syncthreads();

    row = blockIdx.x * TILE_SIZE + threadIdx.y;
    col = blockIdx.y * TILE_SIZE + threadIdx.x;

    if (row < size && col < size) {
        b[row * size + col] = tile[threadIdx.x][threadIdx.y];
    }
}

int main(){
    float *h_i, *h_o, *h_o_ref;
    float *d_i, *d_o;

    size_t size = MATRIX_SIZE * MATRIX_SIZE * sizeof(float);

    h_i = (float *)malloc(size);
    h_o = (float *)malloc(size);
    h_o_ref = (float *)malloc(size);
    CUDA_CHECK(cudaMalloc(&d_i, size));
    CUDA_CHECK(cudaMalloc(&d_o, size));

    init_matrix(h_i, MATRIX_SIZE);
    CUDA_CHECK(cudaMemcpy(d_i, h_i, size, cudaMemcpyHostToDevice));

    dim3 blockDimNaive(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridDimNaive((MATRIX_SIZE + BLOCK_SIZE - 1) / BLOCK_SIZE, 
                      (MATRIX_SIZE + BLOCK_SIZE - 1) / BLOCK_SIZE);

    printf("benchmarking cpu implementation...\n");
    double cpu_total_time = 0.0f;
    for(int i=0;i<20;i++){
        double start = get_time();
        transpose_matrix_cpu(h_i, h_o, MATRIX_SIZE);
        double end = get_time();
        cpu_total_time += end - start;
    }
    double cpu_avg_time = cpu_total_time / 20;

    printf("benchmarking naive gpu implementation....\n");
    double gpu_naive_total_time = 0.0f;
    for(int i=0;i<20;i++){
        double start = get_time();
        transpose_matrix_naive_gpu<<<gridDimNaive, blockDimNaive>>>(d_i, d_o, MATRIX_SIZE);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        double end = get_time();
        gpu_naive_total_time += end - start;
    }
    double gpu_naive_avg_time = gpu_naive_total_time / 20;

    dim3 blockDimTiled(TILE_SIZE, TILE_SIZE);
    dim3 gridDimTiled((MATRIX_SIZE + TILE_SIZE - 1) / TILE_SIZE, 
                      (MATRIX_SIZE + TILE_SIZE - 1) / TILE_SIZE);

    printf("benchmarking tiled gpu implementation....\n");
    double gpu_tiled_total_time = 0.0f;
    for(int i=0;i<20;i++){
        double start = get_time();
        transpose_matrix_tiled_gpu<<<gridDimTiled, blockDimTiled>>>(d_i, d_o, MATRIX_SIZE);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        double end = get_time();
        gpu_tiled_total_time += end - start;
    }
    double gpu_tiled_avg_time = gpu_tiled_total_time / 20;

    printf("CPU Average Time: %f seconds\n", cpu_avg_time);
    printf("GPU Naive Average Time: %f seconds\n", gpu_naive_avg_time);
    printf("Naive Speedup: %f\n", cpu_avg_time / gpu_naive_avg_time);
    printf("GPU Tiled Average Time: %f seconds\n", gpu_tiled_avg_time);
    printf("Tiled Speedup: %f\n", cpu_avg_time / gpu_tiled_avg_time);

    free(h_i);
    free(h_o);
    free(h_o_ref);
    cudaFree(d_i);
    cudaFree(d_o);

    return 0;
}