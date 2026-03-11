#include <bits/stdc++.h>
#include <cuda_runtime.h>

using namespace std;

#define MASK_LEN 7

__constant__ float mask[MASK_LEN];

void init_mask(float *m, int len){
  for(int i=0;i<len;i++){
    m[i] = rand() % 100;
  }
}

__global__ void conv(float *data, float *out, int N){
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  int rad = MASK_LEN / 2;
  int start = idx - rad;
  float sum = 0.0f;
  for(int i=0;i<MASK_LEN;i++){
    int data_idx = start + i;
    if(data_idx >= 0 && data_idx < N){
      sum += data[data_idx] * mask[i];
    }
  }
  out[idx] = sum;
}

int main(){
  int N = 1000;
  float *h_data = new float[N];
  float *d_data;

  cudaMalloc(&d_data, N*sizeof(float));

  for(int i=0;i<N;i++){
    h_data[i] = (float)(i+1);
  }

  // just randomly initialize
  float *m = new float[MASK_LEN];
  init_mask(m, MASK_LEN);

  // now copy the mask to const mem mask
  cudaMemcpyToSymbol(mask, m, MASK_LEN*sizeof(float));

  int threads = 512;
  int blocks = (N + threads - 1) / threads;

  conv<<<blocks, threads>>>(h_data, d_data, N);
  cudaDeviceSynchronize();
  cudaMemcpy(h_data, d_data, N*sizeof(float), cudaMemcpyDeviceToHost);

  cout << "First 10 values of the output array: ";
  for(int i=0;i<10;i++){
    cout << h_data[i] << " ";
  }
  cout << endl;
  
  return 0;
}