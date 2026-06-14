#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <time.h>

#define N 1000000
#define BLOCK_SIZE 256

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            printf("CUDA error at %s:%d - %s\n", __FILE__, __LINE__, \
                   cudaGetErrorString(err)); \
            exit(1); \
        } \
    } while(0)

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
    double start_total = get_time();
    float *h_a, *h_b, *h_c_cpu, *h_c_gpu;
    float *d_a, *d_b, *d_c;

    size_t size = N * sizeof(float);

    int deviceCount;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
    if (deviceCount == 0) {
        printf("No CUDA devices found\n");
        return 1;
    }

    printf("Allocating pinned memory...\n");
    CUDA_CHECK(cudaMallocHost((void**)&h_a, size));
    CUDA_CHECK(cudaMallocHost((void**)&h_b, size));
    CUDA_CHECK(cudaMallocHost((void**)&h_c_cpu, size));
    CUDA_CHECK(cudaMallocHost((void**)&h_c_gpu, size));

    CUDA_CHECK(cudaMalloc(&d_a, size));
    CUDA_CHECK(cudaMalloc(&d_b, size));
    CUDA_CHECK(cudaMalloc(&d_c, size));

    srand(time(NULL));
    init_vector(h_a, N);
    init_vector(h_b, N);

    CUDA_CHECK(cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice));

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

    printf("benchmarking gpu implementation without shared mem....\n");
    float gpu_total_time = 0.0f;
    for(int i=0;i<20;i++){
        double start = get_time();
        matrix_add_gpu<<<num_blocks, BLOCK_SIZE>>>(d_a, d_b, d_c, N);
        CUDA_CHECK(cudaDeviceSynchronize());
        double end = get_time();
        gpu_total_time += end - start;
    }
    float gpu_avg_time = gpu_total_time / 20;

    printf("benchmarking gpu implementation with shared mem....\n");
    float gpu_shared_total_time = 0.0f;
    for(int i=0;i<20;i++){
        CUDA_CHECK(cudaMemset(d_c, 0, size)); 
        double start = get_time();
        matrix_add_gpu_shared<<<num_blocks, BLOCK_SIZE>>>(d_a, d_b, d_c, N);
        CUDA_CHECK(cudaDeviceSynchronize());
        double end = get_time();
        gpu_shared_total_time += end - start;
    }
    float gpu_shared_avg_time = gpu_shared_total_time / 20;

    printf("CPU average time: %.6f seconds\n", cpu_avg_time);
    printf("GPU average time (no shared mem): %.6f seconds\n", gpu_avg_time);
    printf("GPU average time (with shared mem): %.6f seconds\n", gpu_shared_avg_time);
    printf("Speedup (CPU to GPU no shared mem): %.2fx\n", cpu_avg_time / gpu_avg_time);
    printf("Speedup (CPU to GPU with shared mem): %.2fx\n", cpu_avg_time / gpu_shared_avg_time);
    printf("Speedup (GPU no shared mem to GPU with shared mem): %.2fx\n", gpu_avg_time / gpu_shared_avg_time);

    CUDA_CHECK(cudaFreeHost(h_a));
    CUDA_CHECK(cudaFreeHost(h_b));
    CUDA_CHECK(cudaFreeHost(h_c_cpu));
    CUDA_CHECK(cudaFreeHost(h_c_gpu));
    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_c));
    
    double end_total = get_time();
    printf("Total time with pinned mem: %.6f seconds\n", end_total - start_total);
    return 0;
}
