# Benchmarking

## For matrix multiplication

file - [`matrix_multiplication`](./practice/kernels/matrix_mul_shared_mem.cu)

```
CPU average time: 71454.781250 microseconds
GPU average time: 1993.696655 microseconds
Speedup: 35.840347x
```
 
## For max element in an array, diff reductions

file - [`max_element`](./practice/kernels/max_element.cu)

```
max_element                                                                                                                         
benchmarking cpu implementation...
cpu average time: 0.008618 seconds

gpu average time (interleaved addressing): 0.002757 seconds
speedup: 3.13x
gpu average time (interleaved addressing 1): 0.000232 seconds
speedup: 37.16x.
gpu average time (sequential addressing): 0.000227 seconds
speedup: 37.90x
gpu average time (first add during load): 0.000161 seconds
speedup: 53.62x
gpu average time (unroll last warp): 0.000158 seconds
speedup: 54.38x
```
## For transpose of a matrix with naive and tiled as well

file - [`transpose`](./practice/kernels/transpose_matrix.cu)
```
CPU Average Time: 0.000015 seconds
GPU Naive Average Time: 0.000009 seconds
Naive Speedup: 1.611050
GPU Tiled Average Time: 0.000005 seconds
Tiled Speedup: 2.886452
```

## For stencil calcuation in a matrix

file - [`stencil_matrix`](./practice/kernels/stencil_matrix.cu)
```
CPU average time: 0.000043 seconds
GPU average time: 0.000010 seconds
Speedup: 4.475469
```