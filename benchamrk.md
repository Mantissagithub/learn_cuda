# Benchmarking

## For matrix multiplication

```
CPU average time: 71454.781250 microseconds
GPU average time: 1993.696655 microseconds
Speedup: 35.840347x
```
 
## For max element in an array, diff reductions

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
