// this is the kerenl forfinding the best algorithm for conv computation comaptible with my gpu, this was part of the bigger conv2d comparing program, but i have spent aorund 3 hours now understanding the bigger version, so will keep this seperate

#include <bits/stdc++.h>
#include <cudnn.h>
#include <cuda_runtime.h>

// command to run: nvcc -o conv2d_algo_selector conv2d_algo_selector.cu -lcudnn -diag-suppress=177 && ./conv2d_algo_selector

using namespace std;

#define CUDA_CHECK(err) gpuAssert((err), __FILE__, __LINE__)
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
    if (code != cudaSuccess)
    {
        fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString (code), file, line);
        if (abort) exit(code);
    }
}

#define CUDNN_CHECK(err) cudnnAssert((err), __FILE__, __LINE__)
inline void cudnnAssert(cudnnStatus_t code, const char *file, int line
, bool abort=true)
{
    if (code != CUDNN_STATUS_SUCCESS)
    {
        fprintf(stderr,"CUDNNassert: %s %s %d\n", cudnnGetErrorString (code), file, line);
        if (abort) exit(code);
    }
}

int main(){
  int batch_size = 1;
  int kernel_size = 3;
  int in_channels = 1;
  int out_channels = 1;
  int height = 32;
  int width = 32;

  int input_size = batch_size * in_channels * height * width;
  int output_size = batch_size * out_channels * height * width;

  int kernel_size_total = out_channels * in_channels * kernel_size * kernel_size;

  cudnnHandle_t handle;
  CUDNN_CHECK(cudnnCreate(&handle));

  cudnnTensorDescriptor_t input_desc, output_desc;
  CUDNN_CHECK(cudnnCreateTensorDescriptor(&input_desc));
  CUDNN_CHECK(cudnnCreateTensorDescriptor(&output_desc));

  cudnnFilterDescriptor_t kernel_desc;
  CUDNN_CHECK(cudnnCreateFilterDescriptor(&kernel_desc));

  cudnnConvolutionDescriptor_t conv_Desc;
  CUDNN_CHECK(cudnnCreateConvolutionDescriptor(&conv_Desc));

  CUDNN_CHECK(cudnnSetTensor4dDescriptor(input_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, batch_size, in_channels, height, width));
  CUDNN_CHECK(cudnnSetTensor4dDescriptor(output_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, batch_size, out_channels, height, width));
  CUDNN_CHECK(cudnnSetFilter4dDescriptor(kernel_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, out_channels, in_channels, kernel_size, kernel_size));
  CUDNN_CHECK(cudnnSetConvolution2dDescriptor(conv_Desc, kernel_size/2, kernel_size/2, 1, 1, 1, 1, CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT)); // convdesc, pad_height, pad_width, u, v, dilation_height, dilation_width, mode, computeType -> here diklation means spacing between kernel elements

  int requestedAlgoCount = CUDNN_CONVOLUTION_FWD_ALGO_COUNT;
  int returnedAlgoCount;

  cudnnConvolutionFwdAlgoPerf_t perfResults[CUDNN_CONVOLUTION_FWD_ALGO_COUNT];
  CUDNN_CHECK(cudnnGetConvolutionForwardAlgorithm_v7(handle, input_desc, kernel_desc, conv_Desc, output_desc, requestedAlgoCount, &returnedAlgoCount, perfResults));

  // cudnnConvolutionFwdAlgo_t best_algo = perfResults[0].algo;
  // float best_time = perfResults[0].time;

  // for(int i = 0; i < returnedAlgoCount; i++){
  //     if(perfResults[i].status == CUDNN_STATUS_SUCCESS){
  //         printf("Algorithm: %d, Time: %f ms, Memory: %zu bytes\n",
  //               perfResults[i].algo, perfResults[i].time, perfResults[i].memory);

  //         if(perfResults[i].time < best_time){
  //             best_algo = perfResults[i].algo;
  //             best_time = perfResults[i].time;
  //         }
  //     }
  // }
  // cout << "Best Algorithm: " << best_algo << " (Time: " << best_time << " ms)" << endl;

  // the results for this:
  // Algorithm: 1, Time: -1.000000 ms, Memory: 0 bytes
  // Algorithm: 0, Time: -1.000000 ms, Memory: 0 bytes
  // Algorithm: 2, Time: -1.000000 ms, Memory: 0 bytes
  // Algorithm: 6, Time: -1.000000 ms, Memory: 17448 bytes
  // Algorithm: 4, Time: -1.000000 ms, Memory: 67632 bytes
  // Algorithm: 5, Time: -1.000000 ms, Memory: 21760 bytes
  // Algorithm: 7, Time: -1.000000 ms, Memory: 295056 bytes
  // Best Algorithm: 1 (Time: -1 ms)

  // as they are heuritic seraches, lemme try with actual staimtion time time with cudnnFindConvolutionForwardAlgorithm

  printf("Testing cudnnFindConvolutionForwardAlgorithm (actual benchmarking)...\n");
  CUDNN_CHECK(cudnnFindConvolutionForwardAlgorithm(handle,
                                                    input_desc,
                                                    kernel_desc,
                                                    conv_Desc,
                                                    output_desc,
                                                    requestedAlgoCount,
                                                    &returnedAlgoCount,
                                                    perfResults));

  printf("Returned %d algorithms:\n\n", returnedAlgoCount);

  cudnnConvolutionFwdAlgo_t best_algo = perfResults[0].algo;
  float best_time = FLT_MAX;

  for(int i = 0; i < returnedAlgoCount; i++){
      if(perfResults[i].status == CUDNN_STATUS_SUCCESS){
          printf("  [SUCCESS] Algorithm %d: %.6f ms, Memory: %zu bytes\n",
                  perfResults[i].algo, perfResults[i].time, perfResults[i].memory);

          if(perfResults[i].time < best_time){
              best_algo = perfResults[i].algo;
              best_time = perfResults[i].time;
          }
      } else {
          printf("  [FAILED]  Algorithm %d: status %d\n",
                  perfResults[i].algo, perfResults[i].status);
      }
  }

  printf("\n best_algo: %d (time: %.6f ms)\n", best_algo, best_time);

  // mow the reults:
  //  nvcc -o conv2d_algo_selector conv2d_algo_selector.cu -lcudnn -diag-suppress=177 && ./conv2d_algo_selector
  // Testing cudnnFindConvolutionForwardAlgorithm (actual benchmarking)...
  // Returned 8 algorithms:

  //   [SUCCESS] Algorithm 2: 0.011264 ms, Memory: 0 bytes
  //   [SUCCESS] Algorithm 1: 0.014336 ms, Memory: 0 bytes
  //   [SUCCESS] Algorithm 0: 0.015360 ms, Memory: 0 bytes
  //   [SUCCESS] Algorithm 7: 0.066560 ms, Memory: 295056 bytes
  //   [SUCCESS] Algorithm 5: 0.081792 ms, Memory: 21760 bytes
  //   [SUCCESS] Algorithm 6: 0.111488 ms, Memory: 17448 bytes
  //   [SUCCESS] Algorithm 4: 0.141472 ms, Memory: 67632 bytes
  //   [FAILED]  Algorithm 3: status 3000

  // best_algo: 2 (time: 0.011264 ms)

  // freeing things
  CUDNN_CHECK(cudnnDestroyTensorDescriptor(input_desc));
  CUDNN_CHECK(cudnnDestroyTensorDescriptor(output_desc));
  CUDNN_CHECK(cudnnDestroyFilterDescriptor(kernel_desc));
  CUDNN_CHECK(cudnnDestroyConvolutionDescriptor(conv_Desc));
  CUDNN_CHECK(cudnnDestroy(handle));
  return 0;
}