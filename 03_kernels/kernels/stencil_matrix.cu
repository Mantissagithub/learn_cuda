// 5. Simple 2D Stencil Operation (Hard)
// Implement a 2D stencil that replaces each element with the average of its 4 neighbors:

// text
// output[i][j] = (input[i-1][j] + input[i+1][j] + input[i][j-1] + input[i][j+1]) / 4
// Handle boundary conditions (use zero padding or clamp to edges)

// Use shared memory to reduce global memory accesses

// Each block loads a tile with "halo" regions (extra border elements)

// Goal: Learn 2D memory patterns and halo/ghost cell handling.

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <time.h>

#define N 100
#define BLOCK_SIZE 32

#define CUDA_CHECK(err) gpuAssert((err), __FILE__, __LINE__)
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true){
    if(code != cudaSuccess){
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if(abort) exit(code);
    }
}

void init_vector(float *matrix, int n){
    for(int i=0;i<n;i++){
        for(int j=0;j<n;j++){
            matrix[i*n+j] = (float)rand()/RAND_MAX;
        }
    }
}

double get_time(){
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

void stencil_matrix_cpu(float *input, float *o, int size){
    for(int i=0;i<size;i++){
        for(int j=0;j<size;j++){
            float up = (i > 0) ? input[(i-1)*size + j] : 0.0f;
            float down = (i < size-1) ? input[(i+1)*size + j] : 0.0f;
            float left = (j > 0) ? input[i*size + (j-1)] : 0.0f;
            float right = (j < size-1) ? input[i*size + (j+1)] : 0.0f;
            o[i*size + j] = (up + down + left + right) / 4.0f;
        }
    }
}

__global__ void stencil_matrix_gpu(float *g_idata, float *g_odata, int n){
    __shared__ float g_tile[BLOCK_SIZE+2][BLOCK_SIZE+2];
    int row = threadIdx.y + blockDim.y*blockIdx.y;
    int col = threadIdx.x + blockDim.x*blockIdx.x;

    int localRow = threadIdx.y + 1;
    int localCol = threadIdx.x + 1;
    //i was just dumb enough to load only these cells, but when you see in the outer picture we also need cells from the previous or before things too, as in like 
    // 1 2 3 4
    // 5 6 7 8
    // 9 10 11 12
    // let's say the tiles move like 4, when coming to the tile
    // 6 7 
    // 10 11
    // for 6 we need 2, 5 too, for the stencil calculation
    if(row < n && col < n){
        g_tile[localRow][localCol] = g_idata[row*n + col];
    }else{
        g_tile[localRow][localCol] = 0.0f;
    }

    // so this region is called halo it seems
    if(threadIdx.x == 0){
        g_tile[localRow][0] = (col > 0) ? g_idata[row*n + (col-1)] : 0.0f;
    }

    if(threadIdx.x == blockDim.x-1){
        g_tile[localRow][localCol+1] = (col < n-1) ? g_idata[row*n + (col+1)] : 0.0f;
    }

    if(threadIdx.y == 0){
        g_tile[0][localCol] = (row > 0) ? g_idata[(row-1)*n + col] : 0.0f;
    }

    if(threadIdx.y == blockDim.y-1){
        g_tile[localRow+1][localCol] = (row < n-1) ? g_idata[(row+1)*n + col] : 0.0f;
    }
    __syncthreads();

    if(row < n && col < n){
        float val = (g_tile[localRow-1][localCol] + g_tile[localRow+1][localCol] + g_tile[localRow][localCol-1] + g_tile[localRow][localCol+1]) / 4.0f;
        g_odata[row*n+col] = val;
    }
}

int main(){
    float *h_i, *h_o_cpu, *h_o_gpu;
    float *g_i, *g_o;

    size_t size = N*N*sizeof(float);
    h_i = (float*)malloc(size);
    h_o_cpu = (float*)malloc(size);
    h_o_gpu = (float*)malloc(size);

    CUDA_CHECK(cudaMalloc(&g_i, size));
    CUDA_CHECK(cudaMalloc(&g_o, size));

    srand(time(NULL));
    init_vector(h_i, N);

    CUDA_CHECK(cudaMemcpy(g_i, h_i, size, cudaMemcpyHostToDevice));

    dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridDim((N + BLOCK_SIZE - 1) / BLOCK_SIZE, (N + BLOCK_SIZE - 1) / BLOCK_SIZE);

    printf("benchmarking cpu implementation...\n");
    double cpu_total_time = 0.0f;
    for(int i=0;i<20;i++){
        double start = get_time();
        stencil_matrix_cpu(h_i, h_o_cpu, N);
        double end = get_time();
        cpu_total_time += end - start;
    }
    double cpu_avg_time = cpu_total_time / 20.0;

    printf("benchmarking gpu implementation....\n");
    double gpu_total_time = 0.0f;
    for(int i=0;i<20;i++){
        double start = get_time();
        stencil_matrix_gpu<<<gridDim, blockDim>>>(g_i, g_o, N);
        CUDA_CHECK(cudaDeviceSynchronize());
        double end = get_time();
        gpu_total_time += end - start;
    }
    double gpu_avg_time = gpu_total_time / 20.0;

    CUDA_CHECK(cudaMemcpy(h_o_gpu, g_o, size, cudaMemcpyDeviceToHost));
    bool correct = true;
    for(int i=0;i<N*N;i++){
        if(fabs(h_o_cpu[i] - h_o_gpu[i]) > 0.00001f){
            // printf("error at index %d: %f != %f\n", i, h_o_cpu[i], h_o_gpu[i]);
            correct = false;
        }
    }

    printf("GPU results are %s\n", correct ? "correct" : "incorrect");

    printf("CPU average time: %f seconds\n", cpu_avg_time);
    printf("GPU average time: %f seconds\n", gpu_avg_time);
    printf("Speedup: %f\n", cpu_avg_time / gpu_avg_time);

    free(h_i);
    free(h_o_cpu);
    free(h_o_gpu);
    CUDA_CHECK(cudaFree(g_i));
    CUDA_CHECK(cudaFree(g_o));

    return 0;
}
