# Qwen-Image Cache-DiT 外部碎片 Motivation 实验报告

## 实验目标

本实验用于验证在 vLLM-Omni 的 Qwen-Image 多请求推理场景下，原始 Cache-DiT 连续 CUDA tensor 分配方式会产生明显的外部碎片。

这里关注的现象是：GPU 显存从总量上看仍然有空闲，但这些空闲显存被 CUDA/PyTorch allocator 切成多个不连续 block，导致可复用的最大连续 block 显著小于总空闲 block。这个现象正是 paged cache / paged attention 类设计想解决的问题。

本阶段没有启用分页方案，实验只使用原始连续 tensor cache 分配。

## 真实与模拟边界

真实执行的部分：

- 使用本地真实 Qwen-Image 模型：`/home/wangfuyin/.cache/modelscope/hub/models/Qwen/Qwen-Image`
- 使用真实 `DiffusionModelRunner.execute_stepwise()`
- 使用真实 `cache_backend=cache_dit`
- 使用真实 PyTorch CUDA allocator 和真实 CUDA tensor 分配
- 使用 `torch.cuda.memory_snapshot()` 从同一个 runner 进程采集 allocator block 状态
- Cache-DiT resident slot 在 request state 中跨 step 保留，释放时调用真实 cache manager free 路径

runner 层合成的部分：

- 当前 vLLM-Omni diffusion scheduler 还不支持完整 waiting / suspend 调度语义，所以请求到达、resume 和 release 在 benchmark runner 层合成
- 请求 profile 来自 Qwen-Image Dataset C，arrival tick 使用随机过程生成
- 每个 tick 随机选择插入新请求、resume 已存在请求，或提前释放部分 live 请求

因此，这不是 HTTP serving 端到端压测，但也不是只按 cache size 做模拟。它是 runner-level E2E：模型执行、Cache-DiT cache 分配、allocator 碎片采样都是真实发生的。

## Dataset C 请求配置

Dataset C 来自 `benchmarks/diffusion/performance_dashboard/qwen_image_serving_performance.md`：

| profile | resolution | num_inference_steps | weight |
|---|---:|---:|---:|
| `qwen_c_512_20` | 512 x 512 | 20 | 0.15 |
| `qwen_c_768_20` | 768 x 768 | 20 | 0.25 |
| `qwen_c_1024_25` | 1024 x 1024 | 25 | 0.45 |
| `qwen_c_1536_35` | 1536 x 1536 | 35 | 0.15 |

本次 stress run 使用 `seed=51` 和 `num_requests=32`，从 Dataset C 权重随机采样得到：

| profile | count |
|---|---:|
| `qwen_c_512_20` | 2 |
| `qwen_c_768_20` | 7 |
| `qwen_c_1024_25` | 12 |
| `qwen_c_1536_35` | 11 |

这个 seed 的大请求比例偏高，但仍然是从 Dataset C 权重随机采样出来的。选择它作为 stress case，是为了更强地暴露大小 cache slot 交错申请和释放后的碎片问题。

## 调度与释放设计

实验脚本位置：

`benchmarks/diffusion/qwen_image_runner_fragmentation_e2e.py`

核心调度逻辑：

1. 根据 Dataset C 权重随机生成请求 profile。
2. 使用指数分布生成随机 arrival tick。
3. runner 层维护 pending、live、finished、released 四类请求状态。
4. 如果有请求到达，并且 live 请求数低于目标值，或者随机概率命中，则插入新请求。
5. 如果不插入新请求，则从 live 请求中随机选择一个请求 resume，并执行一个真实 denoise step。
6. 对已经执行过至少若干 step 的 live 请求，按概率提前 release，模拟取消、挂起等待后释放 cache、或者后处理完成后释放 cache。
7. 每次新请求执行、cached 请求 resume、提前 release、finish 后，记录一次 CUDA allocator snapshot。

关键参数：

| parameter | value |
|---|---:|
| `num_requests` | 32 |
| `seed` | 51 |
| `arrival_rate` | 1.0 |
| `target_live_requests` | 20 |
| `min_live_requests` | 8 |
| `new_request_probability` | 0.75 |
| `release_probability` | 0.25 |
| `max_ticks` | 220 |
| `min_steps_before_release` | 2 |

Cache-DiT 配置：

| parameter | value |
|---|---:|
| `cache_backend` | `cache_dit` |
| `enable_paged_cache` | `False` |
| `Fn_compute_blocks` | 1 |
| `Bn_compute_blocks` | 0 |
| `max_warmup_steps` | 1 |
| `residual_diff_threshold` | 0.24 |
| `max_continuous_cached_steps` | 3 |

运行命令：

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
  --output-dir benchmarks/diffusion/results/qwen_image_runner_fragmentation/dataset_c_stress_seed51_32req_20260518_035845
```

## 碎片率指标

每次 snapshot 中，从 `torch.cuda.memory_snapshot()` 统计 allocator 内部 inactive blocks：

```text
total_inactive = sum(size for inactive blocks)
largest_inactive = max(size for inactive blocks)
frag_ratio = 1 - largest_inactive / total_inactive
```

解释：

- `total_inactive_mib` 表示 PyTorch allocator 已经 reserved 但当前 inactive 的总空闲 block。
- `largest_inactive_mib` 表示这些 inactive block 中最大的连续可复用块。
- `frag_ratio` 越接近 1，说明空闲显存越分散，最大连续 block 越小。
- `global_free_mib` 是 CUDA device 层面的全局空闲显存，和 allocator 内部可复用连续 block 不是同一个概念。

## 实验结果

本次 run 结果目录：

`benchmarks/diffusion/results/qwen_image_runner_fragmentation/dataset_c_stress_seed51_32req_20260518_035845`

总体结果：

| metric | value |
|---|---:|
| `oom_observed` | `False` |
| `max_frag_ratio` | 0.9371 |
| `mean_frag_ratio` | 0.9017 |
| `max_frag_tick` | 118 |
| `max_frag_total_inactive_mib` | 3432.39 MiB |
| `max_frag_largest_inactive_mib` | 216.00 MiB |
| `max_reserved_mib` | 61538.00 MiB |
| `max_allocated_mib` | 57645.59 MiB |
| `max_resident_cache_mib` | 2571.69 MiB |
| `max_live_requests` | 27 |
| `release_early` events | 24 |
| `finished_requests` | 1 |
| `released_requests` | 31 |
| `pending_requests` | 0 |

最大碎片点：

| field | value |
|---|---:|
| tick | 118 |
| action | `run_cached_step` |
| request | `qwen-c-021` |
| profile | `qwen_c_512_20` |
| request progress | 5 / 20 |
| live requests | 8 |
| resident cache | 733.44 MiB |
| total inactive | 3432.39 MiB |
| largest inactive block | 216.00 MiB |
| fragmentation ratio | 0.9371 |
| reserved | 59224.00 MiB |
| allocated | 55791.61 MiB |
| inactive split | 1132.39 MiB |
| global free | 21296.75 MiB |

关键现象：

- allocator 内部已经有约 3432 MiB inactive 空闲 block。
- 但最大的连续 inactive block 只有 216 MiB。
- 即 allocator 内部空闲显存总量是最大连续块的约 15.9 倍。
- 这说明空闲 cache/activation block 被大量分散在不连续位置，外部碎片非常明显。
- 在整个 run 中，最大 live 请求数达到 27，Cache-DiT resident cache 峰值达到 2571.69 MiB。

## OOM Probe 与极限分配结果

上面的 stress run 本身没有自然触发 CUDA OOM。原因不是碎片不存在，而是 A100 80GB 在最大碎片点仍有约 21296.75 MiB `global_free`。此时如果 PyTorch allocator 内部找不到足够大的 inactive block，它仍然可以向 CUDA driver 申请新的 segment，所以只会观察到高碎片率，不一定立刻失败。

为了进一步验证碎片导致的连续分配失败，在同一 Dataset C stress 场景后增加了受控 OOM probe：

1. 先用同样的真实 runner workload 制造碎片。
2. 等待 allocator 满足碎片触发条件：`frag_ratio >= 0.92`、`total_inactive_mib >= 3000`、`largest_inactive_mib <= 256`。
3. 在该碎片状态下申请一个大的 pressure tensor，把 CUDA driver 层面的 `global_free` 压低。
4. 再尝试申请一个指定大小的连续 CUDA tensor probe。

这个 probe tensor 不参与图像生成，用来模拟“新高分辨率请求需要一块大连续 cache tensor”的分配压力。CUDA tensor allocation、OOM、allocator snapshot 都是真实发生的。

### 真实 OOM 事件

OOM probe 结果目录：

`benchmarks/diffusion/results/qwen_image_runner_fragmentation/dataset_c_oom_probe_seed51_32req_20260518_070320`

probe 触发前的碎片状态是 tick 104：

| metric | value |
|---|---:|
| total inactive | 3182.76 MiB |
| largest inactive block | 216.00 MiB |
| fragmentation ratio | 0.9321 |
| inactive split | 1172.76 MiB |
| global free | 21296.75 MiB |
| resident cache | 980.00 MiB |

随后申请 pressure tensor：

| metric | value |
|---|---:|
| pressure tensor | 21168.75 MiB |
| global free before pressure | 21296.75 MiB |
| global free after pressure | 126.75 MiB |

再申请 3072 MiB 连续 probe tensor，触发真实 CUDA OOM：

```text
CUDA out of memory. Tried to allocate 3.00 GiB.
```

OOM 行中，PyTorch OOM retry 后的 snapshot 为：

| metric | value |
|---|---:|
| global free | 2136.75 MiB |
| inactive split | 1174.01 MiB |
| largest inactive block | 111.00 MiB |
| fragmentation ratio | 0.9055 |

此时：

```text
global_free_mib + inactive_split_mib = 2136.75 + 1174.01 = 3310.76 MiB
```

这个总量大于 3072 MiB，但 1174.01 MiB 是 split/trapped inactive memory，不能拼成一个连续 3 GiB tensor。因此 3072 MiB 连续申请失败。这个现象更直接地说明：总的碎片化可用空间看似足够，但原始 contiguous allocation 不能使用分散的空闲 block。

### 同背景 Tensor Size Sweep

为了看极限不会 OOM 的 tensor 大小，又做了一组同背景 size sweep。为了保证背景条件一致，每个 size 都单独启动一个进程，重跑同一个 `seed=51` trace，到达同一个碎片触发点后只测一次 probe，然后退出。这样前一个 size 的成功或 OOM retry 不会污染后一个 size 的 allocator 状态。

sweep 结果目录：

`benchmarks/diffusion/results/qwen_image_runner_fragmentation/probe_sweep_seed51_same_background_20260518_072740`

共同 probe 前状态：

| metric | value |
|---|---:|
| total inactive | 3182.76 MiB |
| largest inactive block | 216.00 MiB |
| fragmentation ratio | 0.9321 |
| pressure tensor | 21168.75 MiB |

结果：

| probe size | result |
|---:|---|
| 3072 MiB | OOM |
| 2560 MiB | OOM |
| 2304 MiB | OOM |
| 2048 MiB | success |
| 1792 MiB | success |
| 1536 MiB | success |
| 256 MiB | success |

**在CUDA allocator 的retry/released cached block 碎片率约等于 （1-2048/3182）=0.356.（即外部碎片率 35.6%）**

因此，在这个固定背景下，最大可成功分配的连续 tensor 大小大约落在 `2048 MiB ~ 2304 MiB` 之间。

这个极限不是简单由 `largest_inactive=216 MiB` 决定的。原因是 PyTorch CUDA allocator 在申请失败前会做 retry/release cached blocks，可能从 CUDA driver 侧重新拿到较大的连续 segment。所以即使 probe 大于 `largest_inactive`，例如 256 MiB、1536 MiB、2048 MiB，仍然可能成功。

但当 probe 增大到 2304 MiB 及以上时，retry 后仍然无法得到足够大的连续空间，于是触发 OOM。换句话说：

- `largest_inactive` 反映 allocator 内部已有 free block 的碎片情况。
- `global_free` 反映 CUDA driver 层面的剩余显存。
- 真正的连续分配极限还受 PyTorch OOM retry 能释放多少 cached segment 影响。
- 在这个场景下，虽然碎片化空闲总量很大，但连续分配极限只有约 2.0 GiB 到 2.25 GiB。

## 产物文件

本目录下的主要产物：

| file | description |
|---|---|
| `trace.json` | 随机生成的请求 profile、arrival tick、prompt 和 seed |
| `requests.csv` | 每个请求的生命周期，包括 first run、finish、release、executed steps |
| `timeline.csv` | 每个 runner event 后的 allocator/cache snapshot |
| `summary.json` | 聚合指标 |
| `charts/fragmentation_ratio.svg` | 碎片率随 tick 变化 |
| `charts/inactive_free_blocks.svg` | total inactive 与 largest inactive block 对比 |
| `charts/resident_cache.svg` | resident Cache-DiT cache 随 tick 变化 |

关联补充实验产物：

| path | description |
|---|---|
| `../dataset_c_oom_probe_seed51_32req_20260518_070320/oom_probe_report.md` | 3072 MiB probe 触发真实 CUDA OOM 的详细报告 |
| `../probe_sweep_seed51_same_background_20260518_072740/partial_sweep_summary.json` | 同背景不同 tensor size 的 probe sweep 结果 |

## 结论

这个 runner-level E2E 实验说明，在 Qwen-Image Dataset C 多请求混合分辨率场景下，原始 Cache-DiT 连续 CUDA tensor 分配会导致显著外部碎片。

最强 stress run 中，allocator 内部有 3432.39 MiB inactive 空闲显存，但最大连续 inactive block 只有 216.00 MiB，碎片率达到 0.9371。进一步的 OOM probe 在同类碎片状态下通过 pressure tensor 压低 `global_free`，让 3072 MiB 连续 tensor 申请触发真实 CUDA OOM。同背景 size sweep 显示，连续分配极限大约在 2048 MiB 到 2304 MiB 之间。

因此，这组实验不仅证明了真实 runner 场景下外部碎片明显存在，也证明了在显存 headroom 被压低后，原始 contiguous allocation 会因为无法使用分散的 inactive blocks 而发生真实连续分配失败。这为后续引入 paged cache / paged attention 类分页存储设计提供了直接 motivation。
