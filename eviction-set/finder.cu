#include <cuda.h>
#include <unistd.h>

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <set>
#include <string>
#include <vector>
#include <map>

#define CACHE_SIZE  (64 * 1024 * 1024)
#define BLK_SIZE    128

#define TARGET_NUM  256

#define PAD_SIZE    (0)
#define CHUNK_SIZE  (192 * 1024 * 1024)

#define EVICTED     0x0000aaaa
#define NONEVICTED  0x0000bbbb

#define EV_SET_ELEM 16
#define EV_SET_ST   7
#define EV_SET_LD   9
#define OVERFLOW_TH 24

static const uint64_t START_POSITIONS[] = {0, 262144, 524288};
static const int NUM_RUNS = 3;

/******************************************************************************
 *
 *****************************************************************************/
__global__ void
check_eviction(uint64_t addr, uint64_t val, uint64_t *chain)
{
  uint64_t temp;

  asm volatile(
    ".reg .u64 addr_reg;"
    ".reg .u64 val_reg;"
    ".reg .u64 base_reg;"
    ".reg .u64 curr_reg;"
    ".reg .pred %p;"
    "mov.u64 addr_reg, %0;"
    "mov.u64 val_reg, %1;"
    "mov.u64 base_reg, %2;"
    "mov.u64 curr_reg, %2;"
    :
    : "l" (addr), "l" (val), "l" (chain)
  );

  asm volatile(
    "st.u64 [addr_reg], val_reg;"
  "L0:"
    "st.u64 [curr_reg + 8], curr_reg;"
    "ld.u64 curr_reg, [curr_reg];"
    "setp.eq.u64 %p, curr_reg, base_reg;"
    "@!%p bra L0;"
  );

  asm volatile(
    "discard.global.L2 [addr_reg], 128;"
    "ld.u64.cg %0, [addr_reg];"
    : "=l" (temp)
  );

  if (temp == val)
    chain[0] = EVICTED;
  else
    chain[0] = NONEVICTED;
}

/******************************************************************************
 * Run one finder pass with given START_POS, merge into accumulated results
 *****************************************************************************/
void
run_finder_pass(uint64_t *chunk, uint64_t *host, uint64_t start_pos,
                std::map<uint64_t, std::set<uint64_t>> &all_blocks)
{
  uint64_t blk_num = (2 * CACHE_SIZE) / BLK_SIZE;
  assert(start_pos < blk_num - 1);

  for (uint64_t target = 0; target < TARGET_NUM; ++target) {
    std::set<uint64_t> res_set;
    uint64_t target_idx = (target * BLK_SIZE) / sizeof(uint64_t);
    uint64_t addr = (uint64_t)(chunk + target_idx);
    uint64_t val = 0x00000000deadbeef;

    int64_t mid = 0;
    int64_t lower = start_pos;
    int64_t upper = start_pos + blk_num - 1;

    while (upper >= lower) {
      mid = (upper + lower + 1) / 2;

      std::vector<uint64_t> blk_vec(res_set.begin(), res_set.end());
      for (int64_t i = start_pos; i < mid; ++i) {
        if (res_set.count(i) != 0 || (uint64_t)i == target)
          continue;
        blk_vec.push_back(i);
      }
      std::sort(blk_vec.begin(), blk_vec.end());

      for (uint64_t i = 0; i < blk_vec.size(); ++i) {
        uint64_t j = (i + 1) % blk_vec.size();
        uint64_t x = (blk_vec[i] * BLK_SIZE) / sizeof(uint64_t);
        uint64_t y = (blk_vec[j] * BLK_SIZE) / sizeof(uint64_t);
        host[x] = (uint64_t)(chunk + y);
      }
      cudaMemcpy(chunk, host, CHUNK_SIZE, cudaMemcpyHostToDevice);
      cudaDeviceSynchronize();

      uint64_t first_idx = (blk_vec[0] * BLK_SIZE) / sizeof(uint64_t);
      uint64_t *chain = chunk + first_idx;

      check_eviction<<<1, 1>>>(addr, val++, chain);
      cudaDeviceSynchronize();

      cudaMemcpy(host, chunk, CHUNK_SIZE, cudaMemcpyDeviceToHost);
      cudaDeviceSynchronize();

      if (host[first_idx] == NONEVICTED) {
        if (lower == upper) {
          res_set.insert(mid);
          lower = start_pos;
          upper = mid - 1;
        } else {
          lower = mid;
        }
      } else {
        upper = mid - 1;
      }

      if (res_set.size() > OVERFLOW_TH) {
        res_set.clear();
        break;
      }
      sleep(1);
    }

    // merge into accumulated results
    for (auto blk : res_set)
      all_blocks[target].insert(blk);

    int total = all_blocks[target].size();
    int found = res_set.size();
    std::cout << "target " << target << ": +" << found
              << " (total " << total << ")" << std::endl;
  }
}

/******************************************************************************
 * Generate eviction-sets.h with greedy dedup
 *****************************************************************************/
void
generate_header(const std::map<uint64_t, std::set<uint64_t>> &all_blocks,
                const char *path)
{
  // collect targets with >= EV_SET_ELEM blocks
  std::vector<std::vector<uint64_t>> candidates;
  for (auto &kv : all_blocks) {
    if ((int)kv.second.size() >= EV_SET_ELEM) {
      std::vector<uint64_t> v(kv.second.begin(), kv.second.end());
      v.resize(EV_SET_ELEM);
      candidates.push_back(v);
    }
  }

  // greedy dedup: no block index shared across sets
  std::set<uint64_t> used;
  std::vector<std::vector<uint64_t>> final_sets;
  for (auto &s : candidates) {
    bool conflict = false;
    for (auto idx : s) {
      if (used.count(idx)) { conflict = true; break; }
    }
    if (conflict) continue;
    final_sets.push_back(s);
    for (auto idx : s) used.insert(idx);
  }

  std::cout << "valid sets: " << candidates.size()
            << ", after dedup: " << final_sets.size() << std::endl;

  std::ofstream f(path);
  f << "#ifndef _EVICTION_SETS_\n";
  f << "#define _EVICTION_SETS_\n\n";
  f << "#define EV_SET_NUM  " << final_sets.size() << "\n";
  f << "#define EV_SET_ELEM " << EV_SET_ELEM << "\n";
  f << "#define EV_SET_ST   " << EV_SET_ST << " \n";
  f << "#define EV_SET_LD   " << EV_SET_LD << " \n";
  f << "#define IDX_FACTOR  (128 / sizeof(uint64_t))\n\n";
  f << "const uint64_t ev_sets[][EV_SET_ELEM] = {\n";
  for (size_t i = 0; i < final_sets.size(); ++i) {
    f << "  {";
    for (int j = 0; j < EV_SET_ELEM; ++j) {
      if (j) f << ", ";
      f << final_sets[i][j];
    }
    f << (i < final_sets.size() - 1 ? "}," : "}") << " \n";
  }
  f << "};\n\n#endif\n";
  f.close();
}

/******************************************************************************
 *
 *****************************************************************************/
int
main(int argc, char *argv[])
{
  uint64_t *pad;
  uint64_t *chunk;
  uint64_t *host;

  cudaMalloc(&pad, PAD_SIZE);
  cudaMalloc(&chunk, CHUNK_SIZE);
  cudaDeviceSynchronize();
  host = new uint64_t[CHUNK_SIZE / sizeof(uint64_t)];

  if (argc == 1) {
    std::cout << "chunk VA: " << chunk << std::endl;
    std::cout << "dump GPU memory, then run: "
              << argv[0] << " <output.h>" << std::endl;
    std::getchar();
    return 1;
  }

  std::map<uint64_t, std::set<uint64_t>> all_blocks;

  for (int run = 0; run < NUM_RUNS; ++run) {
    uint64_t sp = START_POSITIONS[run];
    std::cout << "=== run " << run << " (START_POS=" << sp << ") ===" << std::endl;
    run_finder_pass(chunk, host, sp, all_blocks);

    // early exit if enough sets already
    int ready = 0;
    for (auto &kv : all_blocks)
      if ((int)kv.second.size() >= EV_SET_ELEM) ++ready;
    if (ready >= TARGET_NUM) {
      std::cout << "enough sets (" << ready << "), skipping remaining runs" << std::endl;
      break;
    }
  }

  generate_header(all_blocks, argv[1]);

  delete[] host;
  cudaFree(chunk);
  cudaFree(pad);
  return 0;
}
