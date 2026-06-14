#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda_runtime.h>

#define N 10000000
#define BLOCK_SIZE 256 // each block has 8 warps so which is considerable

void vector_add_cpu(float *a, float *b, float *c, int n) {
    for (int i = 0; i < n; i++) {
        c[i] = a[i] + b[i];
    }
}

__global__ void vector_add_gpu(float *a, float *b, float *c, int n){
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if(index < n){
        c[index] = a[index] + b[index];
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

    srand(time(NULL));
    init_vector(h_a, N);
    init_vector(h_b, N);

    cudaMalloc(&d_a, size);
    cudaMalloc(&d_b, size);
    cudaMalloc(&d_c, size);


    //so dest, source, size, and then the particular method is the syntax
    cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice);

    int num_blocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE; // ceiling division
    // we do this inorder to define the grid and blocks and everything perfectly

    printf("Benchmarking cpu...\n");
    double cpu_total_time = 0.0;
    for(int i=0;i<20;i++){
        double start = get_time();
        vector_add_cpu(h_a, h_b, h_c_cpu, N);
        double end = get_time();
        cpu_total_time += (end - start);
    }
    double cpu_average_time = cpu_total_time / 20;
    printf("Average CPU time: %f seconds\n", cpu_average_time);

    printf("Benchmarking gpu...\n");
    double gpu_total_time = 0.0;
    for(int i=0;i<20;i++){
        double start = get_time();
        vector_add_gpu<<<num_blocks, BLOCK_SIZE>>>(d_a, d_b, d_c, N);
        cudaDeviceSynchronize();
        double end = get_time();
        gpu_total_time += (end - start);
    }
    double gpu_average_time = gpu_total_time / 20;
    printf("Average GPU time: %f seconds\n", gpu_average_time);

    printf("Speedup: %f\n", cpu_average_time / gpu_average_time);

    //checking the output of the gpu
    cudaMemcpy(h_c_gpu, d_c, size, cudaMemcpyDeviceToHost);  
    bool correct = true;
    for(int i=0;i<N;i++){
        if(fabs(h_c_cpu[i] - h_c_gpu[i]) > 1e-5){
            printf("Mismatch at index %d: CPU %f, GPU %f\n", i, h_c_cpu[i], h_c_gpu[i]);
            correct = false;
            break;
        }
    }

    printf("gpu results were %s\n", correct ? "correct" : "incorrect");

    free(h_a);
    free(h_b);
    free(h_c_cpu);
    free(h_c_gpu);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    return 0;
}