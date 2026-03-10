#include <cuda.h>
#include <stdint.h>
#include <stdio.h>
#include <time.h>

#define BUF_SIZE    (64 * 1024 * 1024)
#define NUM_ROUNDS  50000

__global__ void
victim_kernel(uint64_t *buf, uint64_t num_elems)
{
  uint64_t idx = threadIdx.x + blockIdx.x * (uint64_t)blockDim.x;
  uint64_t stride = blockDim.x * (uint64_t)gridDim.x;

  for (int round = 0; round < NUM_ROUNDS; ++round) {
    for (uint64_t i = idx; i < num_elems; i += stride) {
      buf[i] += 1;
    }
  }
}

int
main(int argc, char *argv[])
{
  int duration = 15;  // seconds
  if (argc > 1)
    duration = atoi(argv[1]);

  uint64_t *d_buf;
  uint64_t num_elems = BUF_SIZE / sizeof(uint64_t);

  cudaMalloc(&d_buf, BUF_SIZE);
  cudaMemset(d_buf, 0, BUF_SIZE);
  cudaDeviceSynchronize();

  printf("[victim] starting, will run for ~%d seconds...\n", duration);

  time_t start = time(NULL);
  int launches = 0;
  while (time(NULL) - start < duration) {
    victim_kernel<<<128, 256>>>(d_buf, num_elems);
    cudaDeviceSynchronize();
    ++launches;
  }

  printf("[victim] done after %d kernel launches.\n", launches);
  cudaFree(d_buf);
  return 0;
}
