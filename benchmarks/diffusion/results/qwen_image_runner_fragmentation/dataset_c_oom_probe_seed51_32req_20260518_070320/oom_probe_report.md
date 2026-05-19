# Qwen-Image Cache-DiT 外部碎片 OOM Probe 实验报告

## 实验目的

上一轮 Dataset C stress 实验已经在真实 Qwen-Image runner + 原始 contiguous Cache-DiT 路径下观察到严重外部碎片，但由于 A100 80GB 全局显存仍有较大余量，未触发真实 OOM。

本实验在上一轮基础上加入受控 memory pressure 和 contiguous allocation probe，用来验证：当 global free 被大 tensor 占位压低后，原始连续分配方式会因为碎片导致真实 CUDA OOM。

## 实验方法

实验仍然使用真实 runner-level E2E 路径：

- 模型：`/home/wangfuyin/.cache/modelscope/hub/models/Qwen/Qwen-Image`
- 执行路径：`DiffusionModelRunner.execute_stepwise()`
- cache backend：`cache_dit`
- paged cache：关闭，`enable_paged_cache=False`
- allocator 采样：`torch.cuda.memory_snapshot()`

新增 OOM probe 逻辑：

1. 先按 Dataset C 随机到达、多请求 resume、随机 early release 的方式制造碎片。
2. 等待所有请求都已到达，且 allocator snapshot 满足：
   - `frag_ratio >= 0.92`
   - `total_inactive_mib >= 3000`
   - `largest_inactive_mib <= 256`
3. 在该碎片状态下，申请一个大的 pressure tensor，把 CUDA global free 压到约 128 MiB。
4. 再尝试申请一个 3072 MiB 的连续 CUDA tensor。
5. 如果该连续分配失败，记录真实 `torch.cuda.OutOfMemoryError`。

这个 probe tensor 用来模拟“需要一个大连续 cache tensor / 高分辨率请求 cache block”的申请压力。它不是 HTTP serving 请求，但 CUDA tensor allocation、OOM、allocator 状态都是真实的。

## 运行命令

```bash
python -u benchmarks/diffusion/qwen_image_runner_fragmentation_e2e.py \
  --model /home/wangfuyin/.cache/modelscope/hub/models/Qwen/Qwen-Image \
  --num-requests 32 \
  --max-ticks 220 \
  --target-live-requests 20 \
  --min-live-requests 8 \
  --new-request-probability 0.75 \
  --release-probability 0.25 \
  --arrival-rate 1.0 \
  --seed 51 \
  --enable-oom-probe \
  --probe-min-frag-ratio 0.92 \
  --probe-min-total-inactive-mib 3000 \
  --probe-max-largest-inactive-mib 256 \
  --probe-target-global-free-mib 128 \
  --probe-allocation-mib 3072 \
  --output-dir benchmarks/diffusion/results/qwen_image_runner_fragmentation/dataset_c_oom_probe_seed51_32req_20260518_070320
```

## 请求分布

本次仍使用 Dataset C，`seed=51`，`num_requests=32`。

| profile | count |
|---|---:|
| `qwen_c_512_20` | 2 |
| `qwen_c_768_20` | 7 |
| `qwen_c_1024_25` | 12 |
| `qwen_c_1536_35` | 11 |

## OOM 前碎片状态

probe 触发前的关键 snapshot 是 tick 104 的 `run_cached_step`：

| metric | value |
|---|---:|
| action | `run_cached_step` |
| request | `qwen-c-026` |
| profile | `qwen_c_1024_25` |
| request progress | 3 / 25 |
| live requests | 11 |
| resident cache | 980.00 MiB |
| total inactive | 3182.76 MiB |
| largest inactive block | 216.00 MiB |
| fragmentation ratio | 0.9321 |
| reserved | 59224.00 MiB |
| allocated | 56041.24 MiB |
| inactive split | 1172.76 MiB |
| global free | 21296.75 MiB |

这说明在 pressure 前，allocator 内部已经有 3182.76 MiB inactive 空闲块，但最大连续 inactive block 只有 216 MiB。

## Pressure 与 OOM 事件

在 tick 104 的碎片状态下，实验申请了 pressure tensor：

| metric | value |
|---|---:|
| pressure tensor | 21168.75 MiB |
| global free before pressure | 21296.75 MiB |
| global free after pressure | 126.75 MiB |
| probe allocation | 3072.00 MiB |
| probe result | CUDA OOM |

真实 OOM 报错摘要：

```text
CUDA out of memory. Tried to allocate 3.00 GiB. GPU 0 has a total capacity of 79.14 GiB of which 2.09 GiB is free.
```

OOM 行的 allocator 状态：

| metric | value |
|---|---:|
| action | `oom_probe_oom` |
| total inactive | 1174.01 MiB |
| largest inactive block | 111.00 MiB |
| fragmentation ratio | 0.9055 |
| reserved | 78384.00 MiB |
| allocated | 77209.99 MiB |
| inactive split | 1174.01 MiB |
| global free | 2136.75 MiB |

注意：OOM 后 `global_free_mib` 变成 2136.75 MiB，是因为 PyTorch OOM retry 过程中释放了一部分可释放 cached blocks。但仍然无法满足 3 GiB 连续申请。

OOM 时有：

```text
global_free_mib + inactive_split_mib = 2136.75 + 1174.01 = 3310.76 MiB
```

这个值大于 3072 MiB 的 probe allocation，但其中 1174.01 MiB 是 split/trapped inactive memory，不能作为一个连续 3 GiB tensor 直接使用。因此 contiguous allocation 失败。这正是分页 cache 希望解决的核心问题：总的碎片化可用空间足够，但连续分配接口无法使用这些分散的 block。

## 总体结果

| metric | value |
|---|---:|
| `oom_observed` | `True` |
| `oom_probe_attempted` | `True` |
| `oom_probe_oom_observed` | `True` |
| `oom_probe_success` | `False` |
| `oom_probe_allocation_mib` | 3072.00 MiB |
| `oom_probe_pressure_allocated_mib` | 21168.75 MiB |
| `max_frag_ratio` | 0.9321 |
| `mean_frag_ratio` | 0.8736 |
| `max_frag_total_inactive_mib` | 3182.76 MiB |
| `max_frag_largest_inactive_mib` | 216.00 MiB |
| `max_resident_cache_mib` | 2571.69 MiB |

## 产物文件

| file | description |
|---|---|
| `trace.json` | Dataset C 随机请求 trace |
| `requests.csv` | 请求生命周期 |
| `timeline.csv` | 每个 runner event 和 OOM probe 的 allocator snapshot |
| `summary.json` | 汇总指标和 OOM 信息 |
| `charts/fragmentation_ratio.svg` | 碎片率曲线 |
| `charts/inactive_free_blocks.svg` | total inactive 与 largest inactive 对比 |
| `charts/resident_cache.svg` | resident cache 曲线 |

## 结论

这次实验在真实 Qwen-Image runner、真实 Cache-DiT、原始 contiguous CUDA tensor 分配路径上触发了真实 CUDA OOM。

OOM 发生前，allocator 内部已有 3182.76 MiB inactive 空闲块，但最大连续 inactive block 只有 216 MiB。加入 21168.75 MiB pressure tensor 后，3 GiB 连续 probe allocation 失败。OOM 后仍可观察到 `global_free_mib + inactive_split_mib = 3310.76 MiB > 3072 MiB`，说明问题不是总量绝对不足，而是可用空间被碎片化后无法服务连续大块申请。

这个结果比上一轮“高碎片率但未 OOM”的实验更强，可以作为 paged cache / paged attention 类分页 cache 设计的 motivation 证据。
