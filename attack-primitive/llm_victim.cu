#include <cuda.h>
#include <stdint.h>
#include <stdio.h>
#include <time.h>

// 4 memory regions, each 8MB, total 32MB
// Kernel cycles through them: region 0 for N rounds, region 1 for N rounds, etc.
#define TOTAL_BUF_SIZE  (32 * 1024 * 1024)
#define REGION_SIZE     (8 * 1024 * 1024)
#define NUM_REGIONS     4
#define REGION_ELEMS    (REGION_SIZE / sizeof(uint64_t))

// High-pressure: cycle through all 4 regions, many rounds each
// This kernel runs for several seconds
__global__ void
ffn_kernel(uint64_t *buf, int total_cycles)
{
  uint64_t idx = threadIdx.x + blockIdx.x * (uint64_t)blockDim.x;
  uint64_t stride = blockDim.x * (uint64_t)gridDim.x;

  for (int cycle = 0; cycle < total_cycles; ++cycle) {
    // Each cycle visits a different 8MB region
    int region = cycle % NUM_REGIONS;
    uint64_t *rbase = buf + region * REGION_ELEMS;

    for (uint64_t i = idx; i < REGION_ELEMS; i += stride) {
      rbase[i] += 1;
    }
  }
}

// Low-pressure: only access first 1MB, many rounds to last a few seconds
#define LOW_SIZE  (1 * 1024 * 1024)
#define LOW_ELEMS (LOW_SIZE / sizeof(uint64_t))

__global__ void
attn_kernel(uint64_t *buf, int total_rounds)
{
  uint64_t idx = threadIdx.x + blockIdx.x * (uint64_t)blockDim.x;
  uint64_t stride = blockDim.x * (uint64_t)gridDim.x;

  for (int round = 0; round < total_rounds; ++round) {
    for (uint64_t i = idx; i < LOW_ELEMS; i += stride) {
      buf[i] += 1;
    }
  }
}

int
main(int argc, char *argv[])
{
  int duration = 20;
  if (argc > 1)
    duration = atoi(argv[1]);

  uint64_t *d_buf;
  cudaMalloc(&d_buf, TOTAL_BUF_SIZE);
  cudaMemset(d_buf, 0, TOTAL_BUF_SIZE);
  cudaDeviceSynchronize();

  time_t start = time(NULL);

  while (time(NULL) - start < duration) {
    ffn_kernel<<<128, 256>>>(d_buf, 20000);
    cudaDeviceSynchronize();
    attn_kernel<<<32, 256>>>(d_buf, 100000);
    cudaDeviceSynchronize();
  }
  cudaFree(d_buf);
  return 0;
}
