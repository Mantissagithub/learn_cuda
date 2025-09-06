# Atomic Ops

So was astonished by results for this cuda kernel -> [`atomic_add.cu`](./atomic_add.cu)

Sample output:
```
Time taken by non-atomic add: 0.000094 seconds
Time taken by atomic add: 0.000008 seconds
non-atomic counter value: 87
atomic counter value: 1000000
```

Understood the concept more clearly as in like when using atomic_add, its happening serially so no <i>**corruption**</i> of data, and when the non atomic various values read diff things and the ouput way more diff. 