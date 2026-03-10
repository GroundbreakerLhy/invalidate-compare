#include <cuda.h>
#include <stdint.h>
#include <stdio.h>
#include <time.h>

#include "eviction-sets.h"

#define PAD_SIZE    0
#define CHUNK_SIZE  (128 * 1024 * 1024)
#define DATA_SIZE   (256L * 1024L)
#define META_SIZE   (EV_SET_NUM * DATA_SIZE)

#define EV_PER_BLK  64
#define CHUNK_ITERS 50      // samples per kernel launch
#define SWITCH_COST 1000
#define RUN_SECS    25      // total wall-clock duration

__global__ void
monitor(uint64_t *meta, uint64_t max_iters)
{
  uint64_t i;
  uint64_t *data;
  uint64_t st_addr;
  uint64_t ld_addr;
  uint64_t val = 10;
  uint64_t res[EV_SET_ST];

  uint8_t no;
  uint8_t *space;

  uint64_t start;
  uint64_t delta;
  uint64_t prev;

  int active;
  i = blockIdx.x * EV_PER_BLK + threadIdx.x;
  active = (i < EV_SET_NUM);

  if (active) {
    data = meta + i * (DATA_SIZE / sizeof(uint64_t));
    st_addr = data[0];
    ld_addr = data[1];
    space = (uint8_t *)(data + 1);
  } else {
    data = meta;
    st_addr = data[0];
    ld_addr = data[1];
    space = (uint8_t *)(data + 1);
  }

  asm volatile(
    ".reg .u64 stAddr0;"
    ".reg .u64 stAddr1;"
    ".reg .u64 stAddr2;"
    ".reg .u64 stAddr3;"
    ".reg .u64 stAddr4;"
    ".reg .u64 stAddr5;"
    ".reg .u64 stAddr6;"
    "mov.u64 stAddr0, %0;"
    "ld.u64 stAddr1, [stAddr0];"
    "ld.u64 stAddr2, [stAddr1];"
    "ld.u64 stAddr3, [stAddr2];"
    "ld.u64 stAddr4, [stAddr3];"
    "ld.u64 stAddr5, [stAddr4];"
    "ld.u64 stAddr6, [stAddr5];"
    :
    : "l" (st_addr)
  );

  asm volatile(
    ".reg .u64 ldAddrHead;"
    ".reg .u64 ldAddrFlag;"
    "mov.u64 ldAddrHead, %0;"
    "mov.u64 ldAddrFlag, %0;"
    :
    : "l" (ld_addr)
  );

  asm volatile(
    ".reg .u64 val;"
    ".reg .pred %p;"
    "mov.u64 val, %0;"
    :
    : "l" (val)
  );

  i = 0;
  #pragma unroll 1
  while (i < max_iters) {
    asm volatile(
      "l1:"
      "ld.u64.cg ldAddrHead, [ldAddrHead];"
      "setp.eq.u64 %p, ldAddrHead, ldAddrFlag;"
      "@!%p bra l1;"
    );

    asm volatile(
      "st.u64.wb [stAddr0+8], val;"
      "st.u64.wb [stAddr1+8], val;"
      "st.u64.wb [stAddr2+8], val;"
      "st.u64.wb [stAddr3+8], val;"
      "st.u64.wb [stAddr4+8], val;"
      "st.u64.wb [stAddr5+8], val;"
      "st.u64.wb [stAddr6+8], val;"
    );

    __syncthreads();
    // wait @ context switch
    prev = 0;
    start = clock64();
    do {
      delta = clock64() - start;
      if (delta - prev > SWITCH_COST)
        break;
      prev = delta;
    } while (1);

    // invalidate
    asm volatile(
      "discard.global.L2 [stAddr0], 128;"
      "discard.global.L2 [stAddr1], 128;"
      "discard.global.L2 [stAddr2], 128;"
      "discard.global.L2 [stAddr3], 128;"
      "discard.global.L2 [stAddr4], 128;"
      "discard.global.L2 [stAddr5], 128;"
      "discard.global.L2 [stAddr6], 128;"
    );

    asm volatile("ld.u64.cg %0, [stAddr0+8];" : "=l" (res[0]));
    asm volatile("ld.u64.cg %0, [stAddr1+8];" : "=l" (res[1]));
    asm volatile("ld.u64.cg %0, [stAddr2+8];" : "=l" (res[2]));
    asm volatile("ld.u64.cg %0, [stAddr3+8];" : "=l" (res[3]));
    asm volatile("ld.u64.cg %0, [stAddr4+8];" : "=l" (res[4]));
    asm volatile("ld.u64.cg %0, [stAddr5+8];" : "=l" (res[5]));
    asm volatile("ld.u64.cg %0, [stAddr6+8];" : "=l" (res[6]));

    if (active) {
      no = 0;
      for (uint8_t k = 0; k < EV_SET_ST; ++k) {
        if (res[k] == val) ++no;
      }
      space[i] = no;  // write sample (i incremented below, outside active check)
    }

    ++i;  // all threads (active and inactive) advance together
    ++val;
    asm volatile("mov.u64 val, %0;" : : "l" (val));
  }
}

/*******************************************************************************
 *
*******************************************************************************/
uint64_t
get_chunk_index(int i, int j)
{
  uint64_t idx = ev_sets[i][j];
  return idx * IDX_FACTOR;
}

uint64_t
get_meta_offset(int i)
{
  return i * (DATA_SIZE / sizeof(uint64_t));
}

int
main(int argc, char *argv[])
{
  cudaDeviceReset();

  uint64_t *pad;
  uint64_t *d_chunk;
  uint64_t *d_meta;
  uint64_t *h_chunk;
  uint64_t *h_meta_init;  // template: linked list setup, never modified
  uint64_t *h_meta;       // result buffer: overwritten after each chunk

  cudaMalloc(&pad, PAD_SIZE);
  cudaMalloc(&d_chunk, CHUNK_SIZE);
  cudaMalloc(&d_meta, META_SIZE);
  cudaDeviceSynchronize();

  h_chunk     = (uint64_t *)malloc(CHUNK_SIZE);
  h_meta_init = (uint64_t *)malloc(META_SIZE);
  h_meta      = (uint64_t *)malloc(META_SIZE);

  for (int i = 0; i < EV_SET_NUM; ++i) {
    for (int j = 0; j < EV_SET_ST; ++j) {
      int k = (j + 1) % EV_SET_ST;
      uint64_t x = get_chunk_index(i, j);
      uint64_t y = get_chunk_index(i, k);
      h_chunk[x] = (uint64_t)(d_chunk + y);
    }

    for (int j = 0; j < EV_SET_LD; ++j) {
      int k = (j + 1) % EV_SET_LD;
      uint64_t x = get_chunk_index(i, j + EV_SET_ST);
      uint64_t y = get_chunk_index(i, k + EV_SET_ST);
      h_chunk[x] = (uint64_t)(d_chunk + y);
    }

    uint64_t n = get_meta_offset(i);
    uint64_t u = get_chunk_index(i, 0);
    uint64_t v = get_chunk_index(i, EV_SET_ST);
    uint64_t w = get_chunk_index(i, 3);
    h_meta_init[n]     = (uint64_t)(d_chunk + u);
    h_meta_init[n + 1] = (uint64_t)(d_chunk + v);
    h_meta_init[n + 2] = (uint64_t)(d_chunk + w);
  }
  cudaMemcpy(d_chunk, h_chunk, CHUNK_SIZE, cudaMemcpyHostToDevice);
  cudaDeviceSynchronize();

  // output: stdout by default, or file if given
  FILE *output = stdout;
  if (argc > 1)
    output = fopen(argv[1], "w");
  setvbuf(output, NULL, _IONBF, 0);  // unbuffered: each write is immediate

  int blk_num = (EV_SET_NUM - 1) / EV_PER_BLK + 1;
  time_t t_start = time(NULL);

  while (time(NULL) - t_start < RUN_SECS) {
    // restore linked list setup in d_meta before each chunk
    cudaMemcpy(d_meta, h_meta_init, META_SIZE, cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();

    // run kernel for CHUNK_ITERS samples
    monitor<<<blk_num, EV_PER_BLK>>>(d_meta, CHUNK_ITERS);
    cudaDeviceSynchronize();

    // copy results
    cudaMemcpy(h_meta, d_meta, META_SIZE, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();

    // print one line per sample, all eviction sets space-separated
    for (int i = 0; i < CHUNK_ITERS; ++i) {
      for (int j = 0; j < EV_SET_NUM; ++j) {
        uint64_t n = get_meta_offset(j);
        uint8_t *space = (uint8_t *)(h_meta + n + 1);
        fprintf(output, "%d ", space[i]);
      }
      fprintf(output, "\n");
    }
  }

  if (output != stdout)
    fclose(output);
}
