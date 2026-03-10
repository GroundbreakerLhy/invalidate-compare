#include <cuda.h>
#include <stdint.h>
#include <stdio.h>
#include <time.h>

#define BUF_SIZE    (4 * 1024 * 1024)
#define NUM_ROUNDS  50

__global__ void
disrupt_kernel(uint64_t *buf, uint64_t num_elems, uint64_t seed)
{
  uint64_t idx = threadIdx.x + blockIdx.x * (uint64_t)blockDim.x;
  // LCG PRNG per thread
  uint64_t state = seed + idx * 6364136223846793005ULL + 1;

  for (int round = 0; round < NUM_ROUNDS; ++round) {
    state = state * 6364136223846793005ULL + 1442695040888963407ULL;
    uint64_t addr = (state >> 16) % num_elems;
    buf[addr] += 1;
  }
}

int
main(int argc, char *argv[])
{
  int duration = 30;
  if (argc > 1)
    duration = atoi(argv[1]);

  uint64_t *d_buf;
  uint64_t num_elems = BUF_SIZE / sizeof(uint64_t);

  cudaMalloc(&d_buf, BUF_SIZE);
  cudaMemset(d_buf, 0, BUF_SIZE);
  cudaDeviceSynchronize();

  time_t start = time(NULL);
  int launches = 0;

  while (time(NULL) - start < duration) {
    disrupt_kernel<<<128, 256>>>(d_buf, num_elems, (uint64_t)launches);
    cudaDeviceSynchronize();
    ++launches;
  }
  cudaFree(d_buf);
  return 0;
}
