// this is a carbon copy of this implementation: https://github.com/Infatoshi/cuda-course/blob/master/06_CUDA_APIs/02%20CUDNN/01%20Conv2d_NCHW.cu
//, but understanding this took me a shot ton of 3 hours, coz im dumb bro, how good are these kernel engineers, i mean they can solve any leetcode queations easily, catching their knowledge would take me around a lifetime i think so!!

// command to run: nvcc -o conv2d conv2d.cu -lcudnn && ./conv2d

// results:
//  nvcc -o conv2d conv2d.cu -lcudnn && ./conv2d
// Image size: 4x4x1
// Kernel size: 3x3x1x1
// Batch size: 1
//   [SUCCESS] Algorithm 2: 0.010400 ms, Memory: 0 bytes
//   [SUCCESS] Algorithm 0: 0.016384 ms, Memory: 0 bytes
//   [SUCCESS] Algorithm 1: 0.016544 ms, Memory: 0 bytes
//   [SUCCESS] Algorithm 5: 0.062784 ms, Memory: 13056 bytes
//   [SUCCESS] Algorithm 7: 0.073728 ms, Memory: 4752 bytes
//   [SUCCESS] Algorithm 4: 0.094208 ms, Memory: 4656 bytes
//   [SUCCESS] Algorithm 6: 0.114528 ms, Memory: 17448 bytes
//   [FAILED]  Algorithm 3: status 3000
// Selected algorithm: 2
// cuDNN average time: 0.010318 ms
// Naive kernel average time: 0.003638 ms
// Max difference between cuDNN and naive kernel: 0.000000e+00

#include <cuda_runtime.h>
#include <cudnn.h>
#include <bits/stdc++.h>

#define CHECK_CUDA(call) { cudaError_t err = call; if (err != cudaSuccess) { printf("CUDA error: %s\n", cudaGetErrorString(err)); exit(1); } }
#define CHECK_CUDNN(call) { cudnnStatus_t err = call; if (err != CUDNN_STATUS_SUCCESS) { printf("cuDNN error: %s\n", cudnnGetErrorString(err)); exit(1); } }

__global__ void naiveConv2d(float* input, float* kernel, float* output, int width, int height, int inChannels, int outChannels, int kernelSize, int batchSize) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int outChannel = blockIdx.z % outChannels;
    int batchIdx = blockIdx.z / outChannels;

    if (x < width && y < height && outChannel < outChannels && batchIdx < batchSize) {
        float sum = 0.0f;
        int halfKernel = kernelSize / 2;
        for (int inChannel = 0; inChannel < inChannels; inChannel++) {
          // this is where the geniuseness starts coz, in leetcode grid questions i usually do like dirs={{0,1},{1,0},{0,-1},{-1,0}} and then loop over it to get the 4 directions, but here they have generalized it for any kernel size, as in any direction that i can go, its prak geniuseness bruhh
            for (int ky = -halfKernel; ky <= halfKernel; ky++) {
                for (int kx = -halfKernel; kx <= halfKernel; kx++) {
                    int ix = x + kx;
                    int iy = y + ky;
                    if (ix >= 0 && ix < width && iy >= 0 && iy < height) {
                      // and this index calculation got me stuck for a while toooo, took me around 1 hour to figure this out, damn im dumb broo
                        int inputIdx = ((batchIdx * inChannels + inChannel) * height + iy) * width + ix;
                        int kernelIdx = ((outChannel * inChannels + inChannel) * kernelSize + (ky + halfKernel)) * kernelSize + (kx + halfKernel);
                        sum += input[inputIdx] * kernel[kernelIdx];
                    }
                }
            }
        }
        int outputIdx = ((batchIdx * outChannels + outChannel) * height + y) * width + x;
        output[outputIdx] = sum;
    }
}

int main() {
    const int width = 4;
    const int height = 4;
    const int kernelSize = 3;
    const int inChannels = 1;
    const int outChannels = 1;
    const int batchSize = 1;
    const int inputSize = width * height * inChannels * batchSize;
    const int outputSize = width * height * outChannels * batchSize;
    const int kernelElements = kernelSize * kernelSize * inChannels * outChannels;

    std::cout << "Image size: " << width << "x" << height << "x" << inChannels << std::endl;
    std::cout << "Kernel size: " << kernelSize << "x" << kernelSize << "x" << inChannels << "x" << outChannels << std::endl;
    std::cout << "Batch size: " << batchSize << std::endl;

    float* h_input = (float*)malloc(inputSize * sizeof(float));
    float* h_kernel = (float*)malloc(kernelElements * sizeof(float));
    float* h_output_cudnn = (float*)malloc(outputSize * sizeof(float));
    float* h_output_naive = (float*)malloc(outputSize * sizeof(float));

    float input_values[] = {
        1, 2, 3, 4,
        5, 6, 7, 8,
        9, 10, 11, 12,
        13, 14, 15, 16,

    };

    float kernel_values[] = {
        1, 2, 3,
        4, 5, 6,
        7, 8, 9,
    };

    memcpy(h_input, input_values, inputSize * sizeof(float));
    memcpy(h_kernel, kernel_values, kernelElements * sizeof(float));

    float *d_input, *d_kernel, *d_output_cudnn, *d_output_naive;
    CHECK_CUDA(cudaMalloc(&d_input, inputSize * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_kernel, kernelElements * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_output_cudnn, outputSize * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_output_naive, outputSize * sizeof(float)));

    CHECK_CUDA(cudaMemcpy(d_input, h_input, inputSize * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_kernel, h_kernel, kernelElements * sizeof(float), cudaMemcpyHostToDevice));

    cudnnHandle_t cudnn;
    CHECK_CUDNN(cudnnCreate(&cudnn));

    cudnnTensorDescriptor_t inputDesc, outputDesc;
    cudnnFilterDescriptor_t kernelDesc;
    cudnnConvolutionDescriptor_t convDesc;

    CHECK_CUDNN(cudnnCreateTensorDescriptor(&inputDesc));
    CHECK_CUDNN(cudnnCreateTensorDescriptor(&outputDesc));
    CHECK_CUDNN(cudnnCreateFilterDescriptor(&kernelDesc));
    CHECK_CUDNN(cudnnCreateConvolutionDescriptor(&convDesc));

    CHECK_CUDNN(cudnnSetTensor4dDescriptor(inputDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, batchSize, inChannels, height, width));
    CHECK_CUDNN(cudnnSetTensor4dDescriptor(outputDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, batchSize, outChannels, height, width));
    CHECK_CUDNN(cudnnSetFilter4dDescriptor(kernelDesc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, outChannels, inChannels, kernelSize, kernelSize));
    CHECK_CUDNN(cudnnSetConvolution2dDescriptor(convDesc, kernelSize/2, kernelSize/2, 1, 1, 1, 1, CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));

    int requestedAlgoCount = CUDNN_CONVOLUTION_FWD_ALGO_COUNT;
    int returnedAlgoCount;
    cudnnConvolutionFwdAlgoPerf_t perfResults[CUDNN_CONVOLUTION_FWD_ALGO_COUNT];
    CHECK_CUDNN(cudnnFindConvolutionForwardAlgorithm(cudnn,
                                                    inputDesc,
                                                    kernelDesc,
                                                    convDesc,
                                                    outputDesc,
                                                    requestedAlgoCount,
                                                    &returnedAlgoCount,
                                                    perfResults));

    cudnnConvolutionFwdAlgo_t algo = perfResults[0].algo;
    float best_time = FLT_MAX;

    for(int i = 0; i < returnedAlgoCount; i++){
        if(perfResults[i].status == CUDNN_STATUS_SUCCESS){
            printf("  [SUCCESS] Algorithm %d: %.6f ms, Memory: %zu bytes\n",
                    perfResults[i].algo, perfResults[i].time, perfResults[i].memory);

            if(perfResults[i].time < best_time){
                algo = perfResults[i].algo;
                best_time = perfResults[i].time;
            }
        } else {
            printf("  [FAILED]  Algorithm %d: status %d\n",
                    perfResults[i].algo, perfResults[i].status);
        }
    }

    std::cout << "Selected algorithm: " << algo << std::endl;
    size_t workspaceSize;
    CHECK_CUDNN(cudnnGetConvolutionForwardWorkspaceSize(cudnn, inputDesc, kernelDesc, convDesc, outputDesc, algo, &workspaceSize));

    void* d_workspace;
    CHECK_CUDA(cudaMalloc(&d_workspace, workspaceSize));

    dim3 blockSize(16, 16);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y, outChannels * batchSize);

    const int warmupRuns = 5;
    const int benchmarkRuns = 20;
    float totalTime_cudnn = 0.0f;
    float totalTime_naive = 0.0f;

    float alpha = 1.0f, beta = 0.0f;

    for (int i = 0; i < warmupRuns; i++) {
        CHECK_CUDNN(cudnnConvolutionForward(cudnn, &alpha, inputDesc, d_input, kernelDesc, d_kernel, convDesc,
                                            algo, d_workspace, workspaceSize, &beta, outputDesc, d_output_cudnn));
        naiveConv2d<<<gridSize, blockSize>>>(d_input, d_kernel, d_output_naive, width, height, inChannels, outChannels, kernelSize, batchSize);
        CHECK_CUDA(cudaDeviceSynchronize());
    }

    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    for (int i = 0; i < benchmarkRuns; i++) {
        CHECK_CUDA(cudaEventRecord(start));
        CHECK_CUDNN(cudnnConvolutionForward(cudnn, &alpha, inputDesc, d_input, kernelDesc, d_kernel, convDesc,
                                            algo, d_workspace, workspaceSize, &beta, outputDesc, d_output_cudnn));
        CHECK_CUDA(cudaEventRecord(stop));
        CHECK_CUDA(cudaEventSynchronize(stop));

        float milliseconds = 0;
        CHECK_CUDA(cudaEventElapsedTime(&milliseconds, start, stop));
        totalTime_cudnn += milliseconds;

        CHECK_CUDA(cudaEventRecord(start));
        naiveConv2d<<<gridSize, blockSize>>>(d_input, d_kernel, d_output_naive, width, height, inChannels, outChannels, kernelSize, batchSize);
        CHECK_CUDA(cudaEventRecord(stop));
        CHECK_CUDA(cudaEventSynchronize(stop));

        CHECK_CUDA(cudaEventElapsedTime(&milliseconds, start, stop));
        totalTime_naive += milliseconds;
    }

    float avgTime_cudnn = totalTime_cudnn / benchmarkRuns;
    float avgTime_naive = totalTime_naive / benchmarkRuns;

    printf("cuDNN average time: %f ms\n", avgTime_cudnn);
    printf("Naive kernel average time: %f ms\n", avgTime_naive);

    CHECK_CUDA(cudaMemcpy(h_output_cudnn, d_output_cudnn, outputSize * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_output_naive, d_output_naive, outputSize * sizeof(float), cudaMemcpyDeviceToHost));

    float maxDiff = 0.0f;
    for (int i = 0; i < outputSize; i++) {
        float diff = fabs(h_output_cudnn[i] - h_output_naive[i]);
        if (diff > maxDiff) maxDiff = diff;
    }

    printf("Max difference between cuDNN and naive kernel: %e\n", maxDiff);

    // printf("\ncuDNN Output:\n");
    // for (int b = 0; b < batchSize; b++) {
    //     for (int c = 0; c < outChannels; c++) {
    //         printf("Channel %d:\n", c);
    //         for (int h = 0; h < height; h++) {
    //             for (int w = 0; w < width; w++) {
    //                 int idx = ((b * outChannels + c) * height + h) * width + w;
    //                 printf("%f ", h_output_cudnn[idx]);
    //             }
    //             printf("\n");
    //         }
    //         printf("\n");
    //     }
    // }

    // printf("\nNaive Kernel Output:\n");
    // for (int b = 0; b < batchSize; b++) {
    //     for (int c = 0; c < outChannels; c++) {
    //         printf("Channel %d:\n", c);
    //         for (int h = 0; h < height; h++) {
    //             for (int w = 0; w < width; w++) {
    //                 int idx = ((b * outChannels + c) * height + h) * width + w;
    //                 printf("%f ", h_output_naive[idx]);
    //             }
    //             printf("\n");
    //         }
    //         printf("\n");
    //     }
    // }

    CHECK_CUDNN(cudnnDestroyTensorDescriptor(inputDesc));
    CHECK_CUDNN(cudnnDestroyTensorDescriptor(outputDesc));
    CHECK_CUDNN(cudnnDestroyFilterDescriptor(kernelDesc));
    CHECK_CUDNN(cudnnDestroyConvolutionDescriptor(convDesc));
    CHECK_CUDNN(cudnnDestroy(cudnn));

    CHECK_CUDA(cudaFree(d_input));
    CHECK_CUDA(cudaFree(d_kernel));
    CHECK_CUDA(cudaFree(d_output_cudnn));
    CHECK_CUDA(cudaFree(d_output_naive));
    CHECK_CUDA(cudaFree(d_workspace));

    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));

    free(h_input);
    free(h_kernel);
    free(h_output_cudnn);
    free(h_output_naive);

    return 0;
}