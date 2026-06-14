#include <bits/stdc++.h>
#include <cuda_runtime.h>

// output:
//  nvcc -o basic_cm basic_cm.cu
//  ./basic_cm
// First 10 elements after kernel execution: 11 12 13 14 15 16 17 18 19 20

using namespace std;

__constant__ float x;

__global__ void kernel(float *data, int N){
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if(idx < N){
    data[idx] += x;
  }
}

int main(){
  int N = 1000;
  float *h_data = new float[N];
  float *d_data;

  cudaMalloc(&d_data, N*sizeof(float));

  for(int i=0;i<N;i++){
    h_data[i] = (float)(i+1);
  }

  float *h_x = new float[1];
  h_x[0] = 10.0f;

  cudaMemcpyToSymbol(x, h_x, sizeof(float));
  cudaMemcpy(d_data, h_data, N*sizeof(float), cudaMemcpyHostToDevice);

  int threads = 256;
  int blockSize = (N + 255) / 256;
  kernel<<<blockSize, threads>>>(d_data, N);
  cudaDeviceSynchronize();
  cudaMemcpy(h_data, d_data, N*sizeof(float), cudaMemcpyDeviceToHost);

  cout<<"First 10 elements after kernel execution: ";
  for(int i=0;i<10;i++){
    cout<<h_data[i]<<" ";
  }
  cout<<endl;

  return 0;
}

