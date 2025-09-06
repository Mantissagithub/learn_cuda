// these atomic operations ensure atomicity as in like the db concept, the op is done only onece and if its not completely done, rollback all changes, and then the concept over heres is that when multiple threads are trying to update the same memory location, we want to ensure that the updates are done in a way that they do not interfere with each other, and the final result is consistent and correct. so atimic operations are used to ensure that when one thread is updating a memory location, no other thread can access that location until the update is complete, like using mutex locks in cpu programming.

// resource to read if not remembering atomics: https://github.com/Infatoshi/cuda-course/blob/master/05_Writing_your_First_Kernels/04%20Atomics/00_atomicAdd.cu

#include <stdio.h>
#include <cuda_runtime.h>
#include <time.h>

#define NUM_THREADS 1000
#define NUM_BLOCKS 1000

__global__ void nonAtomicAdd(int* counter){
    int old = *counter;
    int new_val = old + 1;
    *counter = new_val;
}

__global__ void atomicAdd(int* counter){
    int a = atomicAdd(counter, 1);
}

double get_time(){
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

int main(){
    int h_counterNonAtomic = 0;
    int h_counterAtomic = 0;
    
    int* d_counterNonAtomic;
    int* d_counterAtomic;

    cudaMalloc((void**)&d_counterNonAtomic, sizeof(int));
    cudaMalloc((void**)&d_counterAtomic, sizeof(int));

    cudaMemcpy(d_counterNonAtomic, &h_counterNonAtomic, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_counterAtomic, &h_counterAtomic, sizeof(int), cudaMemcpyHostToDevice);

    double start = get_time();
    nonAtomicAdd<<<NUM_BLOCKS, NUM_THREADS>>>(d_counterNonAtomic);
    double end = get_time();
    printf("Time taken by non-atomic add: %f seconds\n", end - start);
    cudaDeviceSynchronize();
    start = get_time();
    atomicAdd<<<NUM_BLOCKS, NUM_THREADS>>>(d_counterAtomic);
    end = get_time();
    printf("Time taken by atomic add: %f seconds\n", end - start);
    cudaDeviceSynchronize();
    cudaMemcpy(&h_counterNonAtomic, d_counterNonAtomic, sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(&h_counterAtomic, d_counterAtomic, sizeof(int), cudaMemcpyDeviceToHost);

    printf("non-atomic counter value: %d\n", h_counterNonAtomic);
    printf("atomic counter value: %d\n", h_counterAtomic);

    cudaFree(d_counterNonAtomic);
    cudaFree(d_counterAtomic);

    return 0;
}