#include <cuda_runtime.h>
#include <bits/stdc++.h>
#include <cudnn.h>

// command to run: nvcc -o tanh tanh.cu -lcudnn -lcublas && ./tanh

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
inline void cudnnAssert(cudnnStatus_t code, const char *file, int line, bool abort=true)
{
    if (code != CUDNN_STATUS_SUCCESS)
    {
        fprintf(stderr,"CUDNNassert: %s %s %d\n", cudnnGetErrorString (code), file, line);
        if (abort) exit(code);
    }
}

__global__ void naive_tanh_kernel(float* input, float* output, int size){
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if(idx < size){
    output[idx] = tanhf(input[idx]);
  }
}

float cpu_tanh(float x){return tanhf(x);}

void initialization(float* x, int size){
  for(int i=0;i<size;i++){
    x[i] = static_cast <float> (rand()) / static_cast <float> (RAND_MAX/10.0) - 5.0;
  }
}

int main(){
  // nchw format
  int batch_size = 32;
  int channels = 3;
  int height = 224;
  int widht = 224;

  int size = batch_size * channels * height * widht;

  float *h_input, *h_naive_output, *h_cudnn_output, *h_cpu_output;
  h_input = (float*)malloc(size * sizeof(float));
  h_naive_output = (float*)malloc(size * sizeof(float));
  h_cudnn_output = (float*)malloc(size * sizeof(float));
  h_cpu_output = (float*)malloc(size * sizeof(float));

  initialization(h_input, size);

  float *d_input, *d_output_naive, *d_output_cudnn;
  CUDA_CHECK(cudaMalloc((void**)&d_input, size * sizeof(float)));
  CUDA_CHECK(cudaMalloc((void**)&d_output_naive, size * sizeof(float)));
  CUDA_CHECK(cudaMalloc((void**)&d_output_cudnn, size * sizeof(float)));

  CUDA_CHECK(cudaMemcpy(d_input, h_input, size * sizeof(float), cudaMemcpyHostToDevice));

  cudaEvent_t start, stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));

  // so for benchmarking always do warmup runs, then actual runs
  int num_warmup = 10;
  int benchmark_iters = 100;

  float naive_times[benchmark_iters];
  float cudnn_times[benchmark_iters];

  dim3 block(256);
  dim3 grid((size + block.x - 1) / block.x);

  // warmup runs for naive
  for(int i=0;i<num_warmup;i++){
    naive_tanh_kernel<<<grid, block>>>(d_input, d_output_naive, size);
  }
  CUDA_CHECK(cudaDeviceSynchronize());

  // benchmark runs for naive
  for(int i=0;i<benchmark_iters;i++){
    CUDA_CHECK(cudaEventRecord(start));
    naive_tanh_kernel<<<grid, block>>>(d_input, d_output_naive, size);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaEventElapsedTime(&naive_times[i], start, stop));
  }

  cudnnHandle_t handle;
  CUDNN_CHECK(cudnnCreate(&handle));

  cudnnTensorDescriptor_t tensor_desc;
  CUDNN_CHECK(cudnnCreateTensorDescriptor(&tensor_desc));
  CUDNN_CHECK(cudnnSetTensor4dDescriptor(tensor_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, batch_size, channels, height, widht));

  cudnnActivationDescriptor_t activation_desc;
  CUDNN_CHECK(cudnnCreateActivationDescriptor(&activation_desc));
  CUDNN_CHECK(cudnnSetActivationDescriptor(activation_desc, CUDNN_ACTIVATION_TANH, CUDNN_PROPAGATE_NAN, 0.0));

  float alpha = 1.0f, beta = 0.0f;

  // warmup rund for cudnn kernels
  for(int i=0;i<num_warmup;i++){
    CUDNN_CHECK(cudnnActivationForward(handle, activation_desc, &alpha, tensor_desc, d_input, &beta, tensor_desc, d_output_cudnn));
  }
  CUDA_CHECK(cudaDeviceSynchronize());

  // benchmark runs for cudnn kernels
  for(int i=0;i<benchmark_iters;i++){
    CUDA_CHECK(cudaEventRecord(start));
    CUDNN_CHECK(cudnnActivationForward(handle, activation_desc, &alpha, tensor_desc, d_input, &beta, tensor_desc, d_output_cudnn));
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaEventElapsedTime(&cudnn_times[i], start, stop));
  }

  float avg_naive_time = 0.0f, avg_cudnn_time = 0.0f;
  for(int i=0;i<benchmark_iters;i++){
    avg_naive_time += naive_times[i];
    avg_cudnn_time += cudnn_times[i];
  }

  avg_naive_time /= benchmark_iters;
  avg_cudnn_time /= benchmark_iters;

  CUDA_CHECK(cudaMemcpy(h_naive_output, d_output_naive, size * sizeof(float), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(h_cudnn_output, d_output_cudnn, size * sizeof(float), cudaMemcpyDeviceToHost));

  for(int i=0;i<size;i++){
    h_cpu_output[i] = cpu_tanh(h_input[i]);
  }

  // verify correctness
  float max_error_naive = 0.0f;
  float max_error_cudnn = 0.0f;
  for(int i=0;i<size;i++){
    max_error_naive = max(max_error_naive, fabs(h_cpu_output[i] - h_naive_output[i]));
    max_error_cudnn = max(max_error_cudnn, fabs(h_cpu_output[i] - h_cudnn_output[i]));
  }

  printf("Average Naive Tanh Time: %f ms\n", avg_naive_time);
  printf("Average cuDNN Tanh Time: %f ms\n", avg_cudnn_time);
  printf("Max Error Naive Tanh: %f\n", max_error_naive);
  printf("Max Error cuDNN Tanh: %f\n", max_error_cudnn);

  // free resources
  free(h_input);
  free(h_naive_output);
  free(h_cudnn_output);
  free(h_cpu_output);
  CUDA_CHECK(cudaFree(d_input));
  CUDA_CHECK(cudaFree(d_output_naive));
  CUDA_CHECK(cudaFree(d_output_cudnn));
  CUDNN_CHECK(cudnnDestroyTensorDescriptor(tensor_desc));
  CUDNN_CHECK(cudnnDestroyActivationDescriptor(activation_desc));
  CUDNN_CHECK(cudnnDestroy(handle));
  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));
  return 0;
}

// results:
//  nvcc -o tanh tanh.cu -lcudnn -lcublas && ./tanh
// Average Naive Tanh Time: 0.154927 ms
// Average cuDNN Tanh Time: 0.155999 ms
// Max Error Naive Tanh: 0.000000
// Max Error cuDNN Tanh: 0.000000