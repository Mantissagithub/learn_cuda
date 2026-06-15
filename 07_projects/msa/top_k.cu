// warp top-k over a block-score tensor.
//
// input idea:
//   scores[row, block_id] is already one score per sequence block.
//   for every row, we want the best K block_ids.
//
// why this can stay exp-free:
//   if later code would do softmax(score), top-k does not need exp(score).
//   exp is monotonic, so:
//     score_a < score_b  =>  exp(score_a) < exp(score_b)
//   example: scores [1.2, -0.5, 3.0] and exp scores [3.32, 0.60, 20.08]
//   have the same order. so the kernel compares raw scores directly.
//
// kernel split:
//   1. one warp owns one row.
//   2. each lane streams a strided slice of that row:
//        lane 0 -> block 0, 32, 64, ...
//        lane 1 -> block 1, 33, 65, ...
//   3. every lane keeps its own local top-k in registers.
//      vals[0] is kept as the current minimum/root, so a new score can be
//      rejected quickly if score <= vals[0].
//   4. lanes merge with __shfl_down_sync. the lower half receives candidates
//      from the upper half, inserts them into its local top-k, then the active
//      region halves again: 32 -> 16 -> 8 -> 4 -> 2 -> 1.
//   5. lane 0 writes the final K values and block indices for that row.

#include <bits/stdc++.h>
#include <cuda_runtime.h>
#include <float.h>

using namespace std;

#define CUDA_CHECK(err) gpuAssert((err), __FILE__, __LINE__)
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true) {
    if (code != cudaSuccess) {
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

// paper-style naming:
//   B_k = tokens per sequence block
//   B   = num_blocks = ceil(seq_len / B_k)
// so the actual top-k input here is [rows, B], not the full [rows, seq_len].

constexpr int K = 16;
constexpr int WARP_SIZE = 32;
constexpr int WARPS_PER_BLOCK = 4; // we can have 4 warps per block, so each block can process 4 rows of the input tensor
constexpr int THREADS_PER_BLOCK = WARP_SIZE * WARPS_PER_BLOCK; // 128
                                                              

// The local top-k is just two tiny register arrays:
//   vals = candidate scores
//   idxs = matching block ids
//
// We keep the weakest candidate at vals[0]. That makes insertion cheap:
//   incoming score <= vals[0]  -> ignore it
//   incoming score >  vals[0]  -> replace vals[0], then find the new minimum
//
// Small example with K=4 and local candidates [9, 6, 4, 7]:
//   after refresh_min: vals[0] = 4
//   incoming 3 gets rejected
//   incoming 8 replaces 4, then refresh_min makes vals[0] = 6
__device__ void refresh_min(float (&vals)[K], int (&idxs)[K]) {
  int min_pos = 0;
  float min_val = vals[0];

  #pragma unroll
  for(int i=1; i<K;i++) {
    if (vals[i] < min_val) {
      min_val = vals[i];
      min_pos = i;
    }
  }

  if(min_pos != 0) {
    // Put the current minimum at slot 0. The rest of the array is unsorted
    // until the optional final sort.
    float temp_val = vals[min_pos];
    int temp_idx = idxs[min_pos];
    vals[min_pos] = vals[0];
    idxs[min_pos] = idxs[0];
    vals[0] = temp_val;
    idxs[0] = temp_idx;
  }
}

__device__ void insert_topk_minroot(float v, int idx, float (&vals)[K], int (&idxs)[K]) {
  // vals[0] is the current weakest kept score.
  // If v cannot beat it, v cannot be in the top-k for this lane/merge group.
  if (v <= vals[0]) return;

  // Replace the current minimum with the new candidate.
  vals[0] = v;
  idxs[0] = idx;

  // Rebuild the min-root invariant for the next comparison.
  refresh_min(vals, idxs);
}

// After the warp merge, only lane 0 writes. Sorting is not needed for the
// selection itself; it just makes the printed/output row easier to read:
// largest score first, then next largest, etc.
__device__ void sort_desc(float (&vals)[K], int (&idxs)[K]) {
  #pragma unroll
  for(int i=0; i<K; i++){
    int best = i;

    #pragma unroll
    for(int j=i+1; j<K; j++) {
      if(vals[j] > vals[best]) {
        best = j;
      }
    }

    if (best != i) {
      float temp_val = vals[best];
      int temp_idx = idxs[best];
      vals[best] = vals[i];
      idxs[best] = idxs[i];
      vals[i] = temp_val;
      idxs[i] = temp_idx;
    }
  }
}

template<int SORT_OUTPUT>
__global__ void warp_topk_kernel(const float* __restrict__ scores, float* __restrict__ out_values, int* __restrict__ out_indices, int rows, int num_blocks) {
  int tid = threadIdx.x;
  int lane = tid & 31;
  int warp_id = tid >> 5;

  // blockIdx.x chooses a group of rows. warp_id chooses one row inside that
  // group. with 4 warps/block, block 0 handles rows 0..3, block 1 handles 4..7.
  int row = blockIdx.x * WARPS_PER_BLOCK + warp_id;
  if (row >= rows) return;

  const float* row_scores = scores + row * num_blocks;

  float vals[K];
  int idxs[K];

  #pragma unroll
  for(int i=0; i<K; i++) {
    vals[i] = -FLT_MAX; // initialize the top-k values to negative infinity
    idxs[i] = -1; // initialize the top-k indices to -1
  }

  // Each lane streams a 1/32 stride of the row.
  //
  // For num_blocks = 100:
  //   lane 0:  0, 32, 64, 96
  //   lane 1:  1, 33, 65, 97
  //
  //   lane 31: 31, 63, 95
  //
  // This gives coalesced reads at the start of each stride because lanes 0..31
  // touch adjacent block ids.

  for(int b=lane; b<num_blocks; b+=WARP_SIZE) {
    float score = row_scores[b]; // raw score; no exp needed for top-k order
    insert_topk_minroot(score, b, vals, idxs);
  }

  // Tree merge across lanes.
  //
  // At this point each lane has K candidates from its own strided slice.
  // Now lower lanes pull candidates from partner lanes:
  //   offset 16: lane 0 reads lane 16, lane 1 reads lane 17, ...
  //   offset  8: lane 0 reads lane  8, lane 1 reads lane  9, ...
  //   offset  1: lane 0 reads lane  1
  //
  // __shfl_down_sync(mask, x, offset) does not use shared memory. It moves a
  // register value from lane + offset to the current lane inside the warp.
  // So for offset=16:
  //   lane 0 gets lane 16's vals[j]
  //   lane 7 gets lane 23's vals[j]
  //   lane 16 would try to read lane 32, so we ignore it with lane < offset.
  //
  // mask says which lanes must participate in the warp collective. Here all 32
  // lanes are alive for the whole loop, and the if decides which lanes keep the
  // received candidate. It is not a shared-memory bank mask.
  unsigned mask = 0xffffffffu;
  
  for(int offset = 16; offset > 0; offset >>= 1) {
    #pragma unroll
    for(int j=0;j<K;j++){
      float other_v = __shfl_down_sync(mask, vals[j], offset);
      int other_i = __shfl_down_sync(mask, idxs[j], offset);

      if(lane < offset) {
        insert_topk_minroot(other_v, other_i, vals, idxs);
      }
    }
  }


  if(lane == 0){
    if constexpr(SORT_OUTPUT) {
      sort_desc(vals, idxs);
    }

    #pragma unroll
    for(int i=0; i<K; i++) {
      out_values[row * K + i] = vals[i];
      out_indices[row * K + i] = idxs[i];
    }
  }
}

void launch_warp_topk(const float* d_scores, float* d_out_values, int *d_out_indices, int rows, int num_blocks, bool sort_output){
  dim3 block(THREADS_PER_BLOCK);
  dim3 grid((rows + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK);

  if (sort_output) {
    warp_topk_kernel<true><<<grid, block>>>(d_scores, d_out_values, d_out_indices, rows, num_blocks);
  } else {
    warp_topk_kernel<false><<<grid, block>>>(d_scores, d_out_values, d_out_indices, rows, num_blocks);
  }
}

int main(int argc, char **argv) {
    // seq_len = N
    // B_k     = tokens per block
    // B       = num_blocks = ceil(N / B_k)
    //
    // top-k kernel input shape:
    //   [rows, B]
    //
    // top-k output:
    //   [rows, K]

    if (argc < 4) {
        cout << "usage: " << argv[0] << " <rows> <seq_len> <B_k>\n";
        return 0;
    }

    int rows = atoi(argv[1]);
    int seq_len = atoi(argv[2]);
    int B_k = atoi(argv[3]);

    int num_blocks = (seq_len + B_k - 1) / B_k;

    cout << "rows       = " << rows << "\n";
    cout << "seq_len N  = " << seq_len << "\n";
    cout << "B_k        = " << B_k << " tokens per block\n";
    cout << "blocks B   = " << num_blocks << "\n";
    cout << "top-k k    = " << K << "\n";

    CUDA_CHECK(cudaSetDevice(0));

    size_t input_size = (size_t)rows * num_blocks;
    size_t output_size = (size_t)rows * K;

    vector<float> h_scores(input_size);

    srand(0);
    for (size_t i = 0; i < input_size; ++i) {
        h_scores[i] = static_cast<float>(rand()) / RAND_MAX;
    }

    float *d_scores = nullptr;
    float *d_out_values = nullptr;
    int *d_out_indices = nullptr;

    CUDA_CHECK(cudaMalloc(&d_scores, input_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_out_values, output_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_out_indices, output_size * sizeof(int)));

    CUDA_CHECK(cudaMemcpy(
        d_scores,
        h_scores.data(),
        input_size * sizeof(float),
        cudaMemcpyHostToDevice
    ));

    bool sort_output = true;

    launch_warp_topk(
        d_scores,
        d_out_values,
        d_out_indices,
        rows,
        num_blocks,
        sort_output
    );

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    vector<float> h_out_values(output_size);
    vector<int> h_out_indices(output_size);

    CUDA_CHECK(cudaMemcpy(
        h_out_values.data(),
        d_out_values,
        output_size * sizeof(float),
        cudaMemcpyDeviceToHost
    ));

    CUDA_CHECK(cudaMemcpy(
        h_out_indices.data(),
        d_out_indices,
        output_size * sizeof(int),
        cudaMemcpyDeviceToHost
    ));

    cout << "\nfirst row top-k:\n";
    for (int i = 0; i < K; ++i) {
        cout << i
             << ": block_id=" << h_out_indices[i]
             << ", score=" << h_out_values[i]
             << "\n";
    }

    CUDA_CHECK(cudaFree(d_scores));
    CUDA_CHECK(cudaFree(d_out_values));
    CUDA_CHECK(cudaFree(d_out_indices));

    return 0;
}
