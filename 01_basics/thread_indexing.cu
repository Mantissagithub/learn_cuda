#include <stdio.h>

using namespace std;

__global__ void thread_indexing() {
    // blockidx : x -> the exact appartment number on the floor
    //            y -> floor number in this building
    //            z -> building number in the city

    int block_id = blockIdx.x + blockIdx.y*gridDim.x + blockIdx.z*gridDim.y*gridDim.z; 
    int block_offset = 
    block_id* // times our apartment number
    blockDim.x*blockDim.y*blockDim.z; // total threads per block

    int thread_offset = threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y; // the same apartment logic for threads tho

    int id = block_offset + thread_offset;
    printf("block_id = %d, thread_id = %d, id = %d\n", block_id, thread_offset, id);

    return;
}

int main(int argc, char **argv){
    int b_x=2, b_y=3, b_z=4;
    int t_x=4, t_y=4, t_z=4;

    int blocks_per_grid = b_x * b_y * b_z;
    int threads_per_block = t_x * t_y * t_z;

    printf("blocks_per_grid = %d\n", blocks_per_grid);
    printf("threads_per_block = %d\n", threads_per_block);
    printf("total no. of threads = %d\n", blocks_per_grid * threads_per_block);

    dim3 blocksPerGrid(b_x, b_y, b_z);
    dim3 threadsPerBlock(t_x, t_y, t_z);

    thread_indexing<<<blocksPerGrid, threadsPerBlock>>>(); //the syntax for this will be like, i mean the convention -> kernelName<<<number_of_blocks, threads_per_block>>>(parameters);
    cudaDeviceSynchronize();
}