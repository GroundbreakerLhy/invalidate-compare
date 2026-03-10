# invalidate-compare
This repository corresponds to our paper "**Invalidate+Compare: A Timer-Free GPU Cache Attack Primitive**".
- cache-policy: Contains experiments for reverse-engineering GPU cache properties.
- eviction-set: Contains the utility for finding eviction sets.
- attack-primitive: Provides an example of using our attack primitive.

### bibtex entry
```
@inproceedings{Zhang:2024:Security,
  author    = {Zhang, Zhenkai and Cai, Kunbei and Guo, Yanan and Yao, Fan and Gao, Xing},
  title     = {{Invalidate+Compare: A Timer-Free GPU Cache Attack Primitive}},
  year      = {2024},
  booktitle = {33rd USENIX Security Symposium (USENIX Security 24)},
}
```

## RTX 4080 Super Adaptation

Original code targets RTX 3080 (SM 8.0, 5MB L2). This fork adapts it to **RTX 4080 Super** (Ada Lovelace, SM 8.9, 64MB L2).

### Key Changes

**Architecture & Memory Parameters**
- SM architecture: `sm_80` → `sm_89` (all Makefiles)
- L2 cache size: 5MB → 64MB (`CACHE_SIZE = 64 * 1024 * 1024`)
- `PAD_SIZE`: removed (set to 0, GPU memory allocation is deterministic in clean environment)
- `CHUNK_SIZE`: finder 64MB → 192MB, attack-primitive 16MB → 128MB (scaled with L2 size)

**Eviction Set Finder (`eviction-set/finder.cu`)**
- Self-contained: multi-pass scanning with 3 `START_POS` values (0, 262144, 524288)
- Overflow threshold: 16 → 24 (handles larger L2 set associativity)
- Output: 64 non-overlapping eviction sets (EV_SET_NUM=64, EV_SET_ELEM=16)
- Usage: `./finder` to get chunk VA, then dump GPU memory, then `./finder ../attack-primitive/eviction-sets.h`

**Attack Primitive (`attack-primitive/primitive-example.cu`)**
- Comparison logic: consecutive match → total match (captures partial eviction)
- Execution model: single long kernel → chunked kernel launches (50 samples per chunk) with wall-clock timeout for real-time streaming output
- Inactive thread handling: threads beyond `EV_SET_NUM` use dummy data instead of early return (avoids `__syncthreads()` deadlock)

**LLM Attack Scenario (new files)**
- `llm_victim.cu`: simulates LLM inference with two alternating phases — FFN (32MB streaming scan, high cache pressure) and Attention (1MB dense access, low cache pressure)
- `disruptor.cu`: random memory access via LCG PRNG, competes for GPU context switch time slices
- `victim.cu`: basic victim with uniform 64MB access
- `run_llm_attack.sh`: orchestrates disruptor + monitor + LLM victim, outputs value distribution