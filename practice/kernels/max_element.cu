#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <time.h>
// #include <climits.h>
#include <math.h>

#define N 4000000
#define BLOCK_SIZE 256

float maxi_gpu = -INFINITY;

float find_max_cpu(float *vec, int n){
    float maxi = -INFINITY;
    for(int i=0;i<n;i++){
        maxi = fmaxf(maxi, vec[i]);
    }
    return maxi;
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

// so to write the kernel, we can use shared mem, but whats the use of shared mem, nah, what else?
// so the question also gives me a hint to use multiple kernel launches
// and then also an array to store the results
// but no idea on how to do this, fuck, i'm dumb!!
// okay, lemme write the basic one first
// i'm happy that i was dumb, learnt a lot of things, haven't coded since 5 hours learning diff reductions,
// great great resource - [https://developer.download.nvidia.com/assets/cuda/files/reduction.pdf](https://developer.download.nvidia.com/assets/cuda/files/reduction.pdf)
// still there are reduction 6 and reduction 7, will look into them later

// __global__ void find_max_gpu(float *vec, int n){
//     __shared__ float sdata[BLOCK_SIZE];
//     int tid = threadIdx.x;
//     int index = blockIdx.x * blockDim.x + tid;
//     if (index < n) {
//         sdata[tid] = vec[index];
//     } else {
//         sdata[tid] = -INFINITY;
//     }
//     __syncthreads();
//     float maxi = -INFINITY;
//     for (int i = 0; i < BLOCK_SIZE; i++) {
//         maxi = fmaxf(maxi, sdata[i]);
//     }
//     maxi_gpu = fmaxf(maxi_gpu, maxi);
// }
// the shit kernel i wrote above

// reduction 1 - interleaved addressing approach
// so basically here i'm doing tree-based reduction where each thread compares with its neighbor
// the issue is divergent branching - half the threads become inactive at each step
// also lots of memory bank conflicts since consecutive threads access far apart memory locations
__global__ void find_max_interleaved_addressing(float *g_idata, float *g_odata){
    extern __shared__ float sdata[];
    int tid = threadIdx.x;
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    sdata[tid] = g_idata[index];  // load data from global to shared memory
    __syncthreads();

    // tree-based reduction with interleaved addressing
    // each step, stride doubles and half the threads participate
    for(unsigned int s=1;s<blockDim.x;s*=2){
        if(tid % (2*s) == 0){  // divergent branching here - bad for performance
            sdata[tid] = fmaxf(sdata[tid], sdata[tid+s]);
        }
        __syncthreads();
    }

    if(tid == 0) g_odata[blockIdx.x] = sdata[0];  // store block result
}

// reduction 2 - still interleaved but different indexing
// here i'm trying to reduce divergent branching by changing how i calculate indices
// but still not optimal because of the way i'm accessing memory
__global__ void find_max_interleaved_addressing_1(float *g_idata, float *g_odata){
    extern __shared__ float sdata1[];
    int tid = threadIdx.x;
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    sdata1[tid] = g_idata[index];
    __syncthreads();

    // different way of handling indices but still interleaved
    for(unsigned int s=1;s<blockDim.x;s*=2){
        int i = 2*s*tid;  // calculate index differently 
        if(i < blockDim.x){
            sdata1[i] = fmaxf(sdata1[i], sdata1[i+s]);  // still memory conflicts
        }
        __syncthreads();  // need this sync for correctness
    }

    if(tid == 0) g_odata[blockIdx.x] = sdata1[0];
}

// reduction 3 - sequential addressing (much better!)
// now i'm using consecutive threads to do the work instead of every other thread
// this eliminates divergent branching and reduces memory bank conflicts significantly  
__global__ void find_max_sequential_addressing(float *g_idata, float *g_odata){
    extern __shared__ float sdata2[];
    int tid = threadIdx.x;
    int index = blockDim.x*blockIdx.x + threadIdx.x;
    sdata2[tid] = g_idata[index];
    __syncthreads();

    // sequential addressing - threads 0,1,2... do the work instead of 0,2,4...
    // much better memory access pattern and no divergent branching
    for(unsigned int s=blockDim.x/2;s>0;s>>=1){
        if(tid < s){  // first s threads participate - no divergence!
            sdata2[tid] = fmaxf(sdata2[tid], sdata2[tid+s]);
        }
        __syncthreads();
    }
    if(tid == 0) g_odata[blockIdx.x] = sdata2[0];
}

// reduction 4 - first add during load
// here i'm being smart and doing the first reduction step during the initial load
// this means i can process 2x more data per thread block with same number of threads
__global__ void find_max_first_add_during_load(float *g_idata, float *g_odata){
    extern __shared__ float sdata3[];
    int tid = threadIdx.x;
    int index = blockIdx.x * (blockDim.x*2) + threadIdx.x;  // note the *2 here
    // immediately reduce two elements during load - doubles the work per block
    sdata3[tid] = fmaxf(g_idata[index], g_idata[index + blockDim.x]);
    __syncthreads();

    // same sequential addressing as before
    for(unsigned int s=blockDim.x/2;s>0;s>>=1){
        if(tid < s){
            sdata3[tid] = fmaxf(sdata3[tid], sdata3[tid+s]);
        }
        __syncthreads();
    }
    if(tid == 0) g_odata[blockIdx.x] = sdata3[0];
}

// warp-level reduction function
// within a warp, threads execute in lockstep so no __syncthreads() needed
// i can unroll the last 6 steps since warp size is 32
__device__ void warpReduce(volatile float *s, int tid){
    s[tid] = fmaxf(s[tid], s[tid+32]);  // no sync needed - warp executes together
    s[tid] = fmaxf(s[tid], s[tid+16]);
    s[tid] = fmaxf(s[tid], s[tid+8]);
    s[tid] = fmaxf(s[tid], s[tid+4]);
    s[tid] = fmaxf(s[tid], s[tid+2]);
    s[tid] = fmaxf(s[tid], s[tid+1]);
}

// reduction 5 - unroll last warp
// this is the smartest one - i handle the last warp specially since threads in a warp
// execute in perfect sync, so i can unroll those final steps without __syncthreads()
__global__ void find_max_unroll_last_warp(float *g_idata, float *g_odata){
    extern __shared__ float sdata4[];
    int tid = threadIdx.x;
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    sdata4[tid] = g_idata[index];
    __syncthreads();

    // normal reduction until we get to warp level (32 threads)
    for(unsigned int s=blockDim.x/2;s>32;s>>=1){
        if(tid < s){
            sdata4[tid] = fmaxf(sdata4[tid], sdata4[tid+s]);
        }
        __syncthreads();
    }

    // handle the last warp specially - no __syncthreads() needed
    if(tid < 32) warpReduce(sdata4, tid);
    if(tid == 0) g_odata[blockIdx.x] = sdata4[0];
}

// multi-kernel approach - first kernel reduces within blocks, second kernel reduces block results
float find_max_gpu_multi_kernel(float *d_a, int n, int choice){
    int num_blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    
    // special case for reduction 4 - it processes 2x data per block
    if(choice == 4) {
        num_blocks = (n + BLOCK_SIZE*2 - 1) / (BLOCK_SIZE*2);
    }
    
    float *d_block_results; // intermediary results from each block stored here
    cudaMalloc(&d_block_results, num_blocks * sizeof(float));

    // first kernel launch - reduce within each block
    switch(choice){
        case 1:
            find_max_interleaved_addressing<<<num_blocks, BLOCK_SIZE, BLOCK_SIZE*sizeof(float)>>>(d_a, d_block_results);
            break;
        case 2:
            find_max_interleaved_addressing_1<<<num_blocks, BLOCK_SIZE, BLOCK_SIZE*sizeof(float)>>>(d_a, d_block_results);
            break;
        case 3:
            find_max_sequential_addressing<<<num_blocks, BLOCK_SIZE, BLOCK_SIZE*sizeof(float)>>>(d_a, d_block_results);
            break;
        case 4:
            find_max_first_add_during_load<<<num_blocks, BLOCK_SIZE, BLOCK_SIZE*sizeof(float)>>>(d_a, d_block_results);
            break;
        case 5:
            find_max_unroll_last_warp<<<num_blocks, BLOCK_SIZE, BLOCK_SIZE*sizeof(float)>>>(d_a, d_block_results);
            break;
    }
    
    cudaDeviceSynchronize();

    // if we have more than one block, need another reduction
    while(num_blocks > 1){
        int new_num_blocks = (num_blocks + BLOCK_SIZE - 1) / BLOCK_SIZE;
        float *d_new_results;
        cudaMalloc(&d_new_results, new_num_blocks * sizeof(float));
        
        // use the best kernel for subsequent reductions
        find_max_unroll_last_warp<<<new_num_blocks, BLOCK_SIZE, BLOCK_SIZE*sizeof(float)>>>(d_block_results, d_new_results);
        cudaDeviceSynchronize();
        
        cudaFree(d_block_results);
        d_block_results = d_new_results;
        num_blocks = new_num_blocks;
    }

    // copy final result back
    float result;
    cudaMemcpy(&result, d_block_results, sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_block_results);
    
    return result;
}

int main(){
    float *h_a, *d_a;
    int size = N * sizeof(float);
    h_a = (float *)malloc(size);
    cudaMalloc(&d_a, size);

    init_vector(h_a, N);
    cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);

    printf("benchmarking cpu implementation...\n");
    float cpu_total_time = 0.0f;
    float max_cpu;
    for(int i=0;i<20;i++){
        double start = get_time();
        max_cpu = find_max_cpu(h_a, N);
        double end = get_time();
        cpu_total_time += end - start;
    }
    float cpu_avg_time = cpu_total_time / 20.0;
    printf("cpu average time: %f seconds\n", cpu_avg_time);

    // method names for output
    const char* method_names[] = {
        "none",
        "interleaved addressing", 
        "interleaved addressing 1",
        "sequential addressing",
        "first add during load", 
        "unroll last warp"
    };

    // automatically run all 5 gpu reduction methods
    for(int method = 1; method <= 5; method++){
        printf("\nrunning %s...\n", method_names[method]);
        float gpu_total_time = 0.0f;
        float max_gpu;
        
        for(int i=0;i<20;i++){
            double start = get_time();
            max_gpu = find_max_gpu_multi_kernel(d_a, N, method);
            double end = get_time();
            gpu_total_time += end - start;
        }
        
        float gpu_avg_time = gpu_total_time / 20.0;
        printf("gpu average time (%s): %f seconds\n", method_names[method], gpu_avg_time);
        printf("speedup: %.2fx\n", cpu_avg_time / gpu_avg_time);
        
        // verify correctness
        if(fabs(max_cpu - max_gpu) < 1e-5){
            printf("results match!\n");
        }else{
            printf("results dont match! cpu: %f gpu: %f\n", max_cpu, max_gpu);
        }
    }

    cudaFree(d_a);
    free(h_a);
    return 0;
}
