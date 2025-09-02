// 5. Simple 2D Stencil Operation (Hard)
// Implement a 2D stencil that replaces each element with the average of its 4 neighbors:

// text
// output[i][j] = (input[i-1][j] + input[i+1][j] + input[i][j-1] + input[i][j+1]) / 4
// Handle boundary conditions (use zero padding or clamp to edges)

// Use shared memory to reduce global memory accesses

// Each block loads a tile with "halo" regions (extra border elements)

// Goal: Learn 2D memory patterns and halo/ghost cell handling.