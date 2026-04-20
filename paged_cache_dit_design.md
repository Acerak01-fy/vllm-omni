# Paged Cache-DiT 方案可行性分析与详细设计

> 日期: 2026-04-11
> 项目: vLLM-Omni DiT Cache 显存碎片消除
> 目标: 利用 Paged-Attention 思想对 Fn/Bn buffer 分页存储，消除多 request 场景下的外部碎片；设计 Triton 算子实现 in-place 分页计算，避免 gather IO 开销。

---

## 一、现状问题分析

### 1.1 当前 Fn/Bn Buffer 分配模式

每个 request 在每个 transformer block 上持有独立的 cache buffer：

```
Per-request, per-block:
  Fn_residual:  [num_rows, seq_len, hidden_dim]  (e.g., [1, 256, 5120] = 2.5MB @ bf16)
  Bn_residual:  [num_rows, seq_len, hidden_dim]  (same)
  Bn_encoder:   [num_rows, txt_seq_len, hidden_dim] (e.g., [1, 77, 5120] = 0.75MB)

40 blocks × ~5.75 MB/block ≈ 230 MB/request (Wan2.2 512x512)
```

### 1.2 碎片产生机制

```
Time →
┌──────────────────────────────────────────────────────┐
│  [Req A: 512x512] [Req B: 256x256] [Req A: free]    │
│  ████████████████  ████████          ░░░░░░░░░░░░    │
│                                                       │
│  New Req C (768x768) needs 500MB contiguous           │
│  Total free = 600MB, but largest_free_block = 350MB   │
│  → OOM despite sufficient total memory!               │
└──────────────────────────────────────────────────────┘
```

不同 resolution 的 request 交替 allocate/free，造成 PyTorch CUDA allocator 的**外部碎片**。碎片率指标定义：

```
frag_ratio = 1 - (largest_free_block / total_free_memory)
  0.0 → 无碎片 (所有空闲内存连续)
  1.0 → 严重碎片 (空闲内存完全分散)
```

### 1.3 Cache-DiT 4阶段 Batch Forward 回顾

```
Stage 1: Fn blocks on FULL BATCH
  → 在所有 batch 元素上计算 forward blocks
  → 计算 fn_residual_full = hidden_states_new - original

Stage 2: Per-REQUEST can_cache DECISIONS
  → 逐 request 循环，使用各自的 context_manager
  → 调用 cm.can_cache(fn_residual[request_slice])
  → 划分为 compute_indices (需要 Mn) vs cache_indices (使用缓存)

Stage 3a: Mn blocks on COMPUTE GROUP (子集)
  → index_select() 提取需要计算的 request rows
  → 调用 call_Mn_blocks(compute_hs, ...)
  → 逐 request: 存储 Fn/Bn buffers 到 context_manager

Stage 3b: Cache GROUP applies CACHED RESIDUALS
  → 逐 request: 调用 cm.apply_cache() 使用存储的 Bn buffers
  → 残差加法: hs_cached = hs_current + cached_bn_residual
  → index_copy_() 写回 full batch

Stage 4: Bn blocks on FULL BATCH
  → 在组装后的 batch 上运行最终的 backward/post blocks
```

**关键洞察：** 每个 request 有**独立的 cache state**，但 **forward/backward blocks 在合并的 batch 上运行**以提高效率。逐 request 决策发生在 Stage 2。

### 1.4 Buffer Shape 示例 (Wan2.2)

```
hidden_dim = 5120
num_blocks = 40
latent_H = 16, latent_W = 16, frames = 1  (image)
  → seq_len = 16 × 16 = 256 tokens

Per-request per-block:
  Fn_residual:   [1, 256, 5120] = 2.5 MB (bf16)
  Fn_hidden:     [1, 256, 5120] = 2.5 MB
  Bn_residual:   [1, 256, 5120] = 2.5 MB
  Bn_hidden:     [1, 256, 5120] = 2.5 MB
  Bn_encoder:    [1, 77, 5120]  = 0.75 MB

Total per request ≈ 230 MB (40 blocks × ~5.75 MB)
```

---

## 二、方案可行性评估

### 2.1 Paged Fn/Bn Buffer Storage — 高度可行

| 维度 | 评估 |
|------|------|
| **理论基础** | vLLM AR 模型的 PagedAttention KV Cache 已充分验证此思路 |
| **适配性** | Fn/Bn buffer 是 persistent tensors（跨步骤驻留），生命周期明确，天然适合 pool 化管理 |
| **碎片消除** | 所有分配都是固定大小 page，完全消除外部碎片 |
| **复杂度** | 中等 — 需要 page table 管理但不涉及模型结构改动 |

**核心思路：**

- 预分配 GPU 上的 page pool：`[num_pages, page_size, hidden_dim]`
- 每个 Fn/Bn buffer 由若干 page 组成，通过 page_table 索引
- page_size 按 token 数分页（沿 seq_len 维度），例如 page_size=16 或 32 tokens

### 2.2 In-Place Paged Computation (Triton Kernel) — 可行，收益显著

**需要 Triton kernel 的三个操作：**

| 操作 | 当前实现 | Paged In-Place 方式 |
|------|---------|-------------------|
| `can_cache()` — L2/L1 相似度检查 | 对比两个 contiguous tensor | Triton: per-page partial reduction → cross-page aggregate |
| `apply_cache()` — 残差加法 | `hs += cached_bn_residual` | Triton: scatter-read from pages + elementwise add |
| `set_Bn/Fn_buffer()` — 写缓存 | 直接 slice assign | Triton: scatter-write contiguous data into pages |

**IO 开销对比：**

```
Gather 路径:  Read pages → Write contiguous copy → Compute → (可能再写回)
              IO: 2× buffer_size read + 2× buffer_size write

In-place 路径: Read pages + Read input → Compute → Write output
              IO: 1× buffer_size read + 1× input read + 1× output write

节省: ~40-50% 显存带宽
```

对于 Wan2.2 (hidden_dim=5120, 40 blocks)，每步每 request 的 cache 操作涉及 ~230MB 数据移动。in-place 可节省约 100MB 的无效 IO。

**Triton kernel 复杂度评估：** 这些都是 memory-bound 操作（elementwise add, norm），kernel 逻辑简单，主要是 page table 索引计算。难度约等于写一个 paged gather/scatter kernel。

### 2.3 关键限制与风险

| 风险 | 影响 | 缓解措施 |
|------|------|---------|
| Transformer forward 需要 contiguous batch tensor | Fn/Bn 分块计算不受影响（它们操作的是已有的 contiguous batch tensor，只有 cache 存储是分页的） | 只对 cache 存储分页，不改变模型计算路径 |
| Encoder Bn buffer shape 不同于 decoder | txt_seq_len ≠ latent_seq_len | 分别管理两个 page pool，或统一 page_size |
| Page pool 预分配占用固定显存 | 减少可用于计算的显存 | 按需配置 pool size，动态扩缩策略 |
| 多 transformer 架构 (Wan2.2 dual) | 每个 transformer 独立 cache | 共享 page pool 即可 |

---

## 三、详细设计方案

### 3.1 整体架构

```
┌─────────────────────────────────────────────────────────┐
│                    CacheManager                          │
│  activate() / deactivate() / free()                      │
│                                                          │
│  ┌──────────────────────────────────────────────────┐   │
│  │         PagedCachePool (NEW)                      │   │
│  │  ┌──────────────────────────────────┐            │   │
│  │  │ GPU Page Pool                     │            │   │
│  │  │ [num_pages, page_size, hidden_dim]│            │   │
│  │  │ dtype: bf16/fp16                  │            │   │
│  │  └──────────────────────────────────┘            │   │
│  │  ┌───────────────────┐                           │   │
│  │  │ Free List (CPU)   │  O(1) alloc/free          │   │
│  │  └───────────────────┘                           │   │
│  │  ┌───────────────────────────────────┐           │   │
│  │  │ Page Tables (per-request per-block)│           │   │
│  │  │ {req_id: {buffer_name: [page_ids]}}│          │   │
│  │  └───────────────────────────────────┘           │   │
│  └──────────────────────────────────────────────────┘   │
│                                                          │
│  ┌──────────────────────────────────────────────────┐   │
│  │         Triton Kernels (NEW)                      │   │
│  │  • paged_scatter_write                            │   │
│  │  • paged_residual_add                             │   │
│  │  • paged_l2_diff                                  │   │
│  └──────────────────────────────────────────────────┘   │
│                                                          │
│  ┌──────────────────────────────────────────────────┐   │
│  │    PagedCacheBackendSlot (extends CacheBackendSlot)│  │
│  │  payload: page_table_dict (not tensor refs)        │  │
│  └──────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────┘
```

### 3.2 模块一：PagedCachePool — 分页显存池

**文件位置：** `vllm_omni/diffusion/cache/paged_cache_pool.py`

```python
class PagedCachePool:
    """Pre-allocated GPU page pool for Fn/Bn cache buffers."""

    def __init__(self,
                 num_pages: int,           # 总页数 (e.g., 4096)
                 page_size: int,           # 每页 token 数 (e.g., 16)
                 hidden_dim: int,          # 隐藏维度 (e.g., 5120)
                 dtype: torch.dtype,       # bf16/fp16
                 device: torch.device):
        # 核心: 一次性预分配，消除所有后续碎片
        self.page_pool = torch.zeros(
            num_pages, page_size, hidden_dim,
            dtype=dtype, device=device
        )
        self.free_list = list(range(num_pages))  # Stack-based O(1) alloc
        self.num_pages = num_pages
        self.page_size = page_size

    def allocate(self, num_tokens: int) -> list[int]:
        """分配足够容纳 num_tokens 的页面，返回 page_id 列表"""
        num_needed = math.ceil(num_tokens / self.page_size)
        if num_needed > len(self.free_list):
            raise PagePoolExhaustedError(
                f"Need {num_needed} pages, only {len(self.free_list)} available"
            )
        page_ids = [self.free_list.pop() for _ in range(num_needed)]
        return page_ids

    def free(self, page_ids: list[int]):
        """归还页面到 free list"""
        self.free_list.extend(page_ids)

    def get_page_tensor(self, page_id: int) -> torch.Tensor:
        """获取单个 page 的 tensor view (zero-copy)"""
        return self.page_pool[page_id]
```

**page_size 选择策略：**

- 太大 → 内部碎片（最后一页浪费）
- 太小 → page table 过大，kernel overhead 增加
- 推荐：`page_size = 16 tokens`（对齐 GPU warp size）
- 内部碎片上限：`page_size × hidden_dim × dtype_size = 16 × 5120 × 2 = 160KB` per buffer

**显存预算计算 (Wan2.2 为例)：**

```
假设最大并发 8 requests, 512x512 resolution:
  seq_len = 256 tokens, pages_per_buffer = 256/16 = 16
  buffers per request = 40 blocks × 3 (Fn_res + Bn_res + Bn_enc) = 120
  pages per request ≈ 120 × 16 = 1920 pages
  8 requests × 1920 = 15,360 pages

Page pool size:
  15,360 × 16 × 5120 × 2 bytes = 2.4 GB

vs 当前动态分配: 8 × 230MB ≈ 1.8 GB (but with fragmentation!)

Overhead: ~30% 额外显存 for page pool slack
但: 碎片率从 30-60% 降至 0%
```

### 3.3 模块二：Triton Kernels

**文件位置：** `vllm_omni/diffusion/cache/kernels/paged_cache_ops.py`

#### Kernel 1: paged_scatter_write — 将 contiguous 数据写入分页

```python
@triton.jit
def paged_scatter_write_kernel(
    src_ptr,          # 源: contiguous [num_tokens, hidden_dim]
    page_pool_ptr,    # 目标: page_pool [num_pages, page_size, hidden_dim]
    page_table_ptr,   # page_ids [num_pages_needed]
    num_tokens: tl.constexpr,
    page_size: tl.constexpr,
    hidden_dim: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,  # 每个 program 处理的 elements
):
    pid = tl.program_id(0)
    # 计算当前 program 处理的 token 范围
    token_start = pid * BLOCK_SIZE

    for token_offset in range(BLOCK_SIZE):
        token_id = token_start + token_offset
        if token_id >= num_tokens:
            break

        # 查 page table: token_id → (page_id, offset_in_page)
        page_idx = token_id // page_size
        offset_in_page = token_id % page_size
        page_id = tl.load(page_table_ptr + page_idx)

        # 计算源和目标地址
        src_offset = token_id * hidden_dim
        dst_offset = (page_id * page_size + offset_in_page) * hidden_dim

        # 向量化复制 hidden_dim 维度
        for d in range(0, hidden_dim, 128):  # vectorize
            vals = tl.load(src_ptr + src_offset + d + tl.arange(0, 128))
            tl.store(page_pool_ptr + dst_offset + d + tl.arange(0, 128), vals)
```

#### Kernel 2: paged_residual_add — 从分页读取并加到 contiguous tensor

```python
@triton.jit
def paged_residual_add_kernel(
    output_ptr,       # in-place: contiguous [num_tokens, hidden_dim]
    page_pool_ptr,    # 源: page_pool (paged Bn_residual)
    page_table_ptr,   # page_ids
    num_tokens: tl.constexpr,
    page_size: tl.constexpr,
    hidden_dim: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    """output[i] += paged_cache[page_table[i // ps]][i % ps]"""
    pid = tl.program_id(0)
    token_id = pid
    if token_id >= num_tokens:
        return

    page_idx = token_id // page_size
    offset_in_page = token_id % page_size
    page_id = tl.load(page_table_ptr + page_idx)

    out_offset = token_id * hidden_dim
    cache_offset = (page_id * page_size + offset_in_page) * hidden_dim

    for d in range(0, hidden_dim, BLOCK_H):
        mask = (d + tl.arange(0, BLOCK_H)) < hidden_dim
        out_vals = tl.load(
            output_ptr + out_offset + d + tl.arange(0, BLOCK_H), mask=mask
        )
        cache_vals = tl.load(
            page_pool_ptr + cache_offset + d + tl.arange(0, BLOCK_H), mask=mask
        )
        tl.store(
            output_ptr + out_offset + d + tl.arange(0, BLOCK_H),
            out_vals + cache_vals,
            mask=mask,
        )
```

#### Kernel 3: paged_l2_diff — 分页 L2 norm 计算（相似度检查）

```python
@triton.jit
def paged_l2_diff_kernel(
    new_ptr,          # contiguous new Fn_residual [num_tokens, hidden_dim]
    page_pool_ptr,    # paged old Fn_residual
    page_table_ptr,   # page_ids
    output_ptr,       # partial sums [num_tokens] → later reduced
    num_tokens: tl.constexpr,
    page_size: tl.constexpr,
    hidden_dim: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    """Compute per-token ||new[i] - cached[i]||^2"""
    pid = tl.program_id(0)
    token_id = pid
    if token_id >= num_tokens:
        return

    page_idx = token_id // page_size
    offset_in_page = token_id % page_size
    page_id = tl.load(page_table_ptr + page_idx)

    new_offset = token_id * hidden_dim
    cache_offset = (page_id * page_size + offset_in_page) * hidden_dim

    acc = tl.zeros([], dtype=tl.float32)
    for d in range(0, hidden_dim, BLOCK_H):
        mask = (d + tl.arange(0, BLOCK_H)) < hidden_dim
        new_vals = tl.load(
            new_ptr + new_offset + d + tl.arange(0, BLOCK_H), mask=mask
        ).to(tl.float32)
        cache_vals = tl.load(
            page_pool_ptr + cache_offset + d + tl.arange(0, BLOCK_H), mask=mask
        ).to(tl.float32)
        diff = new_vals - cache_vals
        acc += tl.sum(diff * diff)

    tl.store(output_ptr + token_id, acc)
```

最终相似度 = `output.sum().sqrt() / (num_tokens * hidden_dim)` 在 Python 端完成（一次小的 reduce）。

### 3.4 模块三：PagedCacheContext — 替换现有 CachedContext

**修改位置：** `vllm_omni/diffusion/cache/cache_dit_batch.py`

```python
class PagedCacheContext:
    """替代 CachedContext，用 page table 代替 tensor 引用"""

    def __init__(self, pool: PagedCachePool):
        self.pool = pool
        self.page_tables: dict[str, list[int]] = {}  # buffer_name → page_ids
        self.shapes: dict[str, tuple] = {}  # buffer_name → logical shape

    def set_Fn_buffer(self, data: torch.Tensor, prefix: str):
        """将 contiguous data 写入分页存储"""
        num_tokens = data.shape[0] * data.shape[1]  # rows × seq_len
        key = f"{prefix}"

        # 释放旧页面（如果存在）
        if key in self.page_tables:
            self.pool.free(self.page_tables[key])

        # 分配新页面
        page_ids = self.pool.allocate(num_tokens)
        self.page_tables[key] = page_ids
        self.shapes[key] = data.shape

        # Triton scatter write (in-place, no contiguous copy)
        page_table_tensor = torch.tensor(
            page_ids, dtype=torch.int32, device=data.device
        )
        paged_scatter_write(
            data.reshape(-1, data.shape[-1]),
            self.pool.page_pool,
            page_table_tensor,
        )

    def set_Bn_buffer(self, data: torch.Tensor, prefix: str):
        """与 set_Fn_buffer 相同逻辑"""
        self.set_Fn_buffer(data, prefix)  # 复用相同的写入路径

    def apply_cache_inplace(self, hidden_states: torch.Tensor, prefix: str):
        """直接在 hidden_states 上加 cached Bn residual，无需 gather"""
        key = f"{prefix}"
        page_ids = self.page_tables[key]
        page_table_tensor = torch.tensor(
            page_ids, dtype=torch.int32, device=hidden_states.device
        )

        # Triton in-place add (避免 gather 的 IO 开销)
        paged_residual_add(
            hidden_states.reshape(-1, hidden_states.shape[-1]),
            self.pool.page_pool,
            page_table_tensor,
        )

    def can_cache(
        self, new_fn_data: torch.Tensor, prefix: str, threshold: float
    ) -> bool:
        """基于分页数据的相似度检查"""
        key = f"{prefix}"
        if key not in self.page_tables:
            return False

        page_ids = self.page_tables[key]
        page_table_tensor = torch.tensor(
            page_ids, dtype=torch.int32, device=new_fn_data.device
        )

        # Triton paged L2 diff
        diff = paged_l2_diff(
            new_fn_data.reshape(-1, new_fn_data.shape[-1]),
            self.pool.page_pool,
            page_table_tensor,
        )
        normalized_diff = diff.sum().sqrt() / new_fn_data.numel()
        return normalized_diff.item() < threshold

    def clear_buffers(self):
        """释放所有页面回 pool"""
        for page_ids in self.page_tables.values():
            self.pool.free(page_ids)
        self.page_tables.clear()
        self.shapes.clear()
```

### 3.5 模块四：PagedCacheDiTStateDriver 适配

**修改文件：** `vllm_omni/diffusion/cache/cache_dit_driver.py`

```python
class PagedCacheDiTStateDriver(CacheStateDriver):
    def __init__(self, backend, pipeline, pool: PagedCachePool):
        super().__init__(backend, pipeline)
        self.pool = pool

    def create_empty_slot(self) -> CacheBackendSlot:
        # payload 改为 PagedCacheContext 而非 raw tensor dicts
        contexts = tuple(
            {name: PagedCacheContext(self.pool) for name in context_names}
            for handle in self._handles
        )
        return CacheBackendSlot(
            backend_name="cache_dit_paged",
            payload=contexts,
            resident_bytes=0,  # 动态计算
        )

    def estimate_slot_bytes(self, slot) -> int:
        total = 0
        for handle_contexts in slot.payload:
            for ctx in handle_contexts.values():
                for page_ids in ctx.page_tables.values():
                    total += (
                        len(page_ids)
                        * self.pool.page_size
                        * self.pool.hidden_dim
                        * 2  # bf16
                    )
        return total

    def clear_slot(self, slot):
        for handle_contexts in slot.payload:
            for ctx in handle_contexts.values():
                ctx.clear_buffers()
```

### 3.6 模块五：Batch Forward 改造

**修改文件：** `vllm_omni/diffusion/cache/cache_dit_batch.py`

Stage 2 (can_cache) 和 Stage 3a/3b 的改造：

```python
# Stage 2: Per-request cache decision (改用 paged similarity)
for i, (start, end) in enumerate(request_ranges):
    cm = batch_contexts[i]
    fn_slice = fn_residual_full[start:end]  # 仍是 contiguous (来自 Stage 1)

    # 调用 paged can_cache — Triton kernel 直接对比 contiguous vs paged
    can_use = cm.can_cache(
        fn_slice,
        prefix=f"{cache_prefix}_Fn_residual",
        threshold=config.residual_diff_threshold,
    )

# Stage 3a: Mn compute group — set_Bn_buffer 改用 paged write
for i in compute_indices:
    cm = batch_contexts[i]
    # Triton scatter-write 到 page pool
    cm.set_Fn_buffer(
        fn_residual[local_start:local_end],
        prefix=f"{cache_prefix}_Fn_residual",
    )
    cm.set_Bn_buffer(
        mn_residual[local_start:local_end],
        prefix=f"{cache_prefix}_Bn_residual",
    )

# Stage 3b: Cache group — apply_cache 改用 paged in-place add
for i in cache_indices:
    cm = batch_contexts[i]
    # Triton in-place: hidden_states[start:end] += paged Bn_residual
    cm.apply_cache_inplace(
        hidden_states[start:end],
        prefix=f"{cache_prefix}_Bn_residual",
    )
```

**关键优势：** Stage 1 (Fn forward) 和 Stage 4 (Bn forward) **完全不变** — 它们操作的是 contiguous batch tensor。只有 cache 存储/读取路径改为分页。

---

## 四、工程量与时间线

### 4.1 任务分解

| 阶段 | 任务 | 预估天数 | 依赖 |
|------|------|---------|------|
| **P0: 基础设施** | | **12 天** | |
| P0.1 | PagedCachePool 实现 (alloc/free/pool管理) | 5 天 | 无 |
| P0.2 | Page table 数据结构 + PagedCacheContext | 5 天 | P0.1 |
| P0.3 | Pool 配置推导 (根据 max_batch, resolution 估算 num_pages) | 2 天 | P0.1 |
| **P1: Triton Kernels** | | **24 天** | |
| P1.1 | paged_scatter_write kernel + 单元测试 + 调试 | 7 天 | P0.1 |
| P1.2 | paged_residual_add kernel + 单元测试 + 调试 | 7 天 | P0.1 |
| P1.3 | paged_l2_diff kernel + 单元测试 + 调试 | 7 天 | P0.1 |
| P1.4 | Kernel benchmark (vs naive gather+compute) | 3 天 | P1.1-P1.3 |
| **P2: 系统集成** | | **18 天** | |
| P2.1 | PagedCacheDiTStateDriver 实现 | 7 天 | P0.2 |
| P2.2 | cache_dit_batch.py 改造 (Stage 2/3a/3b) | 7 天 | P2.1 |
| P2.3 | CacheManager batch mode 适配 | 4 天 | P2.1 |
| **P3: 特殊场景** | | **7 天** | |
| P3.1 | Encoder Bn buffer 分页 (不同 seq_len) | 2 天 | P2.2 |
| P3.2 | 多 transformer 架构支持 (Wan2.2 dual) | 2 天 | P2.2 |
| P3.3 | CFG companion request 兼容 | 2 天 | P2.2 |
| P3.4 | Page pool exhaustion 处理 (fallback/动态扩展) | 1 天 | P0.1 |
| **P4: 测试验证** | | **10 天** | |
| P4.1 | 数值正确性测试 (paged vs non-paged bit-exact) | 3 天 | P2 |
| P4.2 | 碎片率测试 (用现有 fragmentation.py 框架) | 2 天 | P2 |
| P4.3 | 性能 benchmark (throughput, latency, 显存) | 3 天 | P2 |
| P4.4 | 多 request 并发压力测试 | 2 天 | P2 |
| **Buffer: 意外缓冲** | 应对集成意外、Triton 边界 bug、设计返工等 | **7 天** | |
| **总计** | | **~78 日** | |

### 4.2 分期交付策略

> **核心原则：** 先用 gather 方式跑通端到端（解决碎片问题），再用 Triton kernel 替换（拿到性能收益）。Phase A 结束即有可交付成果，降低整体风险。

#### Phase A: Paged Storage + Gather Baseline（消除碎片）

> 目标：用 `page_pool[page_ids].reshape(...)` gather 方式实现分页存储，端到端跑通，证明碎片率 → 0%。
> 此阶段 **不需要 Triton kernel**，P2 的系统集成基于 PyTorch index_select/index_copy 实现。

```
Week 1-2: P0 基础设施 (12 天)
           ├─ P0.1: PagedCachePool 实现
           ├─ P0.2: PagedCacheContext (用 gather 读写)
           ├─ P0.3: Pool 配置推导
           └─ Milestone: pool alloc/free 单元测试通过 ✓

Week 3-5: P2 系统集成-Gather 版 (18 天)
           ├─ P2.1: PagedCacheDiTStateDriver
           ├─ P2.2: cache_dit_batch.py 改造 (gather 路径)
           ├─ P2.3: CacheManager 适配
           └─ Milestone: 单 request 端到端跑通 ✓

Week 6:   P3 特殊场景 (7 天)
           ├─ P3.1-P3.4: Encoder/多架构/CFG/exhaustion
           └─ Milestone: 多 request mixed-resolution batch 跑通 ✓

Week 7:   Phase A 验证 (5 天)
           ├─ 数值正确性 (gather-paged vs non-paged bit-exact)
           ├─ 碎片率对比测试
           └─ Milestone: 碎片率 → 0%, 正确性验证通过 ✓
```

**Phase A 交付物：** 碎片问题已解决，可合入主分支。性能可能因 gather IO 略有退化，但并发能力大幅提升。

---

#### Phase B: Triton In-Place 优化（提升性能）

> 目标：用 Triton kernel 替换 gather 路径，消除多余 IO 开销，拿到 ~40-50% 显存带宽节省。

```
Week 8-10:  P1 Triton Kernels (24 天)
            ├─ P1.1: paged_scatter_write + 单元测试 + 调试
            ├─ P1.2: paged_residual_add + 单元测试 + 调试
            ├─ P1.3: paged_l2_diff + 单元测试 + 调试
            └─ Milestone: 3 个 kernel 单元测试全部通过 ✓

Week 11:    替换集成 + Benchmark (5 天)
            ├─ 将 gather 路径替换为 Triton in-place 路径
            ├─ P1.4: Kernel benchmark (vs gather baseline)
            └─ Milestone: in-place 端到端跑通 ✓

Week 12:    Phase B 验证 (5 天)
            ├─ 数值正确性 (in-place vs gather bit-exact)
            ├─ 性能 benchmark (throughput, latency, 显存)
            ├─ 多 request 并发压力测试
            └─ Milestone: 性能无退化，IO 开销降低 ~40% ✓
```

**Phase B 交付物：** 性能优化完成，gather IO 开销消除。

---

#### 总体时间线

```
         Phase A (碎片消除)              Phase B (性能优化)
  ├─────────────────────────────┤├──────────────────────────┤
  Week 1  2  3  4  5  6  7      8  9  10  11  12  (+Buffer)
  P0──────┤                     P1─────────────┤
           P2─────────────┤                     替换+验证──┤
                           P3──┤
                              验证┤
  ──────────────────────────────────────────────────────────
  Phase A 交付: 碎片率→0%       Phase B 交付: IO 开销↓40%
  (可独立合入主分支)             (性能优化增量)
```

**含 Buffer 总工期: ~12-13 周（约 3 个月）**

---

## 五、预期收益

| 指标 | 当前 | Paged 方案 |
|------|------|-----------|
| 碎片率 (8 concurrent, mixed resolution) | 30-60%（理论开销） | <10% |
| 最大可服务并发数 (80GB GPU) | 4-6 requests | 8-12 requests |
| Cache 操作 IO 开销 | 2× (gather + compute) | 1× (in-place) |
| 显存利用率 | ~60-70% (含碎片) | ~85%+ |
| Page pool 固定开销 | 0 | ~10-15% pool slack |

---

## 六、关键设计决策建议

1. **page_size = 16 tokens** — 平衡碎片率和 kernel efficiency，对齐 GPU warp size
2. **Encoder/Decoder 共享 pool** — 虽然 seq_len 不同，但 hidden_dim 相同，统一管理更简单
3. **渐进式启用** — 通过环境变量 `EXPERIMENT_PAGED_CACHE` 控制，与现有 `EXPERIMENT_CACHEPOOL` 正交
4. **分期交付: Phase A gather baseline → Phase B Triton in-place** — Phase A 用 `page_pool[page_ids].reshape(...)` gather 实现，端到端跑通即可合入主分支解决碎片问题；Phase B 再用 Triton kernel 替换拿性能收益（详见 4.2 节）
5. **page_table 放 GPU** — 避免每步 CPU→GPU transfer，预分配 `torch.IntTensor` on device

---

## 七、核心代码关联文件索引

| 文件 | 用途 | 行数 |
|------|------|------|
| `vllm_omni/diffusion/cache/cache_dit_backend.py` | Main backend, enabler functions, custom adapters | ~1345 |
| `vllm_omni/diffusion/cache/cache_dit_driver.py` | Resident-state driver for stepwise serving | ~195 |
| `vllm_omni/diffusion/cache/cache_dit_batch.py` | Batched forward pass with per-request decisions | ~637 |
| `vllm_omni/diffusion/cache/cache_manager.py` | CacheManager & CacheStateDriver interface | ~233 |
| `vllm_omni/diffusion/cache/base.py` | CacheBackend abstract class | ~118 |
| `vllm_omni/diffusion/worker/utils.py` | CacheBackendSlot, DiffusionRequestState | ~160 |
| `vllm_omni/diffusion/sched/base_scheduler.py` | Request scheduling with cache slot lifecycle | — |
| `vllm_omni/diffusion/data.py` | DiffusionCacheConfig 配置参数 (Fn/Bn/thresholds) | — |
| `vllm_omni/diffusion/attention/layer.py` | Attention 层实现 | ~153 |
| `vllm_omni/diffusion/attention/backends/flash_attn.py` | FlashAttention backend | ~212 |
| `vllm_omni/diffusion/worker/diffusion_model_runner.py` | Model runner, batch execution | ~871 |
| `vllm_omni/diffusion/worker/input_batch.py` | InputBatch 多 request 聚合 | ~500+ |

---

## 八、分页可行性与 Omni 侧内存管理深度分析

> 本节基于对 vllm-omni 源码（cache_dit_backend.py, cache_dit_batch.py, cache_dit_driver.py, cache_manager.py）的详细审查，严谨论证两个核心问题：
> 1. 为什么 Fn/Bn buffer 能够分页存储？
> 2. 为什么内存管理权可以从 cache-dit 转移到 omni 侧？

### 8.1 Fn/Bn Buffer 能够分页的四个关键条件

#### 条件 1：Buffer 是"存储态"数据，非"计算态"数据

cache-dit 的 4 阶段 batch forward 中（`cache_dit_batch.py`），**模型计算始终在 contiguous batch tensor 上进行**：

- **Stage 1**（Fn blocks）和 **Stage 4**（Bn blocks）：对完整 batch 的 contiguous tensor 做 forward，完全不涉及 cache buffer
- **Stage 2**（can_cache 判断）：将 contiguous 的 `fn_residual_full[start:end]` 与已缓存数据**对比**
- **Stage 3a**（set buffer）：将 contiguous 的计算结果**写入** cache 存储
- **Stage 3b**（apply cache）：从 cache 存储**读出**并加到 contiguous 的 `hidden_states` 上

**关键洞察：Fn/Bn buffer 只在"存"和"取"的时刻被访问，不参与矩阵乘法、注意力等需要连续内存布局的核心计算。** 这意味着 buffer 的物理存储布局可以是任意的，只要读写接口正确即可。

对比 AR 模型的 KV cache：KV cache 同样是"存储态"数据（attention 计算时通过 page table 索引访问），vLLM 的 PagedAttention 已充分验证此模式可行。Fn/Bn buffer 的访问模式甚至比 KV cache 更简单——只有三种操作：

| 操作 | 访问模式 | 复杂度 |
|------|---------|--------|
| `set_Fn/Bn_buffer()` | scatter write（contiguous → paged） | 简单 |
| `apply_cache()` | gather read + elementwise add | 简单 |
| `can_cache()` | gather read + L2 norm reduction | 简单 |

#### 条件 2：Buffer 的 Shape 沿 seq_len 维度天然可分页

当前 buffer 的 shape 是 `[num_rows, seq_len, hidden_dim]`（例如 `[1, 256, 5120]`）。分页方案沿 **seq_len 维度** 切分：

```
原始: [1, 256, 5120]  (contiguous)
         ↓ 分页 (page_size=16)
分页后: 16 个 page, 每个 [16, 5120]
```

这能工作的原因是三种操作都是 **per-token independent** 的：

- `set_buffer`: 逐 token 写入，token 之间无依赖
- `apply_cache` 的残差加法 `hs[i] += cached[i]`：逐 token elementwise，无跨 token 依赖
- `can_cache` 的 L2 差异 `||new[i] - cached[i]||²`：逐 token 计算后 reduce，token 间可独立算 partial sum

**没有任何操作需要跨 token 的连续内存访问**，因此沿 seq_len 分页不会破坏计算语义。

#### 条件 3：Buffer 的生命周期清晰且可预测

从 `cache_dit_driver.py` 和 `cache_manager.py` 可以看到：

```
Request 到达 → activate() → create_empty_slot() [分配]
                          → install_slot()        [激活]
每个 denoise step:
  Stage 3a → set_Fn/Bn_buffer()                  [写入/更新]
  Stage 2  → can_cache()                          [读取比较]
  Stage 3b → apply_cache()                        [读取应用]
Request 完成 → free() → clear_slot()              [释放]
```

生命周期完全由 request 的开始和结束决定。没有 buffer 在 request 之间共享，没有复杂的引用关系。这意味着页面分配（allocate）和回收（free）的时机完全确定——这正是 page pool 最擅长的场景。

#### 条件 4：当前的 buffer clone 语义天然适配 scatter write

从源码看，`set_Fn_buffer` / `set_Bn_buffer` 内部做的是 `buffer.detach().clone()`——**深拷贝**。这意味着 cache-dit 已经在做一次完整的数据拷贝。将这次拷贝替换为 Triton 的 `paged_scatter_write`（将 contiguous 数据写入分页），**不会增加额外的数据移动开销**，只是改变了目标地址的布局。

### 8.2 内存管理权从 Cache-DiT 转移到 Omni 侧的可行性

#### 8.2.1 现状：Cache-DiT 的"隐式内存管理"

当前 cache-dit 的内存管理是 **隐式的、分散的**：

```python
# cache-dit 库内部（context.buffers dict）
def set_Fn_buffer(self, buffer, prefix):
    self._current_context.buffers[prefix] = buffer.detach().clone()
    # ↑ 隐式分配：PyTorch CUDA allocator 自动分配显存
    # 旧 tensor 被 Python GC 回收（时机不确定）
```

问题：
1. **每次 `set_buffer` 都触发 `cudaMalloc`**（通过 PyTorch caching allocator），这些分配的大小随 resolution 变化
2. **释放依赖 Python GC** —— 旧 buffer 被新 clone 覆盖后，旧 tensor 的释放时机不确定
3. **没有全局视角** —— cache-dit 不知道其他 request 的内存使用情况，无法做全局优化

这就是碎片的根源：不同 resolution 的 request 交替分配/释放不同大小的 tensor，导致 CUDA allocator 的地址空间碎片化。

#### 8.2.2 架构上已存在的管理分层

从代码看，omni 侧已经建立了一套完整的管理层：

```
CacheManager (omni 侧, cache_manager.py)
    │ activate() / deactivate() / free()
    │ estimate_slot_bytes()
    ↓
CacheDiTStateDriver (omni 侧, cache_dit_driver.py)
    │ create_empty_slot() / install_slot() / clear_slot()
    │ install_batch_slots() / deactivate_batch_slots()
    ↓
CachedContextManager (cache-dit 库)
    │ set_Fn_buffer() / apply_cache() / can_cache()
    ↓
context.buffers dict (cache-dit 库内部)
    → 实际 tensor 存储
```

**CacheManager 已经控制了 slot 的完整生命周期**（创建、激活、去激活、释放），并且通过 `CacheBackendSlot` 持有 payload 引用。唯一没控制的是**最底层的 tensor 存储** —— 这正是分页方案要替换的部分。

#### 8.2.3 转移路径：替换 context.buffers 为 PagedCacheContext

转移的关键是：**不改变上层接口，只替换底层存储**。

```
改造前:
  cm.set_Fn_buffer(data, prefix)
    → context.buffers[prefix] = data.detach().clone()  # cudaMalloc

改造后:
  cm.set_Fn_buffer(data, prefix)                       # 接口不变
    → PagedCacheContext.set_Fn_buffer(data, prefix)
      → pool.allocate(num_tokens)                      # page pool O(1) 分配
      → paged_scatter_write(data, pool, page_ids)      # 写入固定页
```

这可行的原因是 **cache-dit 的 CachedContextManager 的核心计算逻辑（can_cache 的阈值判断、cache/compute group 的划分、4 阶段调度）完全不依赖 buffer 的物理存储布局**。它只需要能够：
- 往 buffer 写数据
- 从 buffer 读数据并做加法
- 从 buffer 读数据并算 L2 距离

这三个操作都可以通过 page table + Triton kernel 实现，不需要修改 cache-dit 的决策逻辑。实现方式为 "Duck typing" —— `PagedCacheContext` 实现相同的方法签名（`set_Fn_buffer`、`set_Bn_buffer`、`apply_cache`、`can_cache`、`clear_buffers`）即可无缝替换原有 context 对象。

#### 8.2.4 Omni 侧管理的优势对比

| 维度 | cache-dit 自管理 | omni 侧统一管理 |
|------|-----------------|-----------------|
| 分配粒度 | 每次 clone 一个完整 tensor | 固定大小 page，O(1) 分配 |
| 碎片 | 不同 resolution 导致严重碎片 | 碎片率 → 0%（所有页等大） |
| 全局视角 | 无（每个 request 独立 clone） | CacheManager 可做 admission control |
| 预算控制 | 无法限制总用量 | page pool 大小固定，超出可 reject/evict |
| 复用 | 旧 tensor 等 GC 回收 | page 立即归还 free list，立即可复用 |

### 8.3 风险点与缓解分析

#### 风险 1：cache-dit 是外部库，能否修改其内部存储？

**评估：** 从 `cache_dit_driver.py` 看，omni 侧已经在做 monkey-patching（`cache_dit_backend.py` 中对 forward 的替换）和 context 切换（`_current_context` 赋值）。设计中的 `PagedCacheContext` 是**替换 context 对象**，不是修改 cache-dit 库源码。只要 `PagedCacheContext` 实现了相同方法签名，就可以无缝替换。

**缓解：** 走 "Duck typing" 路径。

#### 风险 2：page table 索引的 CPU→GPU 传输开销

每次 `set_buffer` / `apply_cache` 需要将 `page_ids` 列表转为 GPU tensor。如果每步每 block 都做，可能成为瓶颈。

**评估：** 第六节已给出方案 —— page_table 预分配在 GPU 上。page_ids 在 allocate 时确定后写入 GPU tensor，后续直接使用，无需反复传输。

**补充分析：** 同一 request 的 seq_len 在整个 denoise 过程中固定不变（由 resolution 决定），因此 page table 只需分配一次，后续所有 step 复用同一个 GPU page_table tensor。

#### 风险 3：Encoder Bn buffer 的 seq_len 不同

Encoder 的 `txt_seq_len`（如 77）与 decoder 的 `latent_seq_len`（如 256）不同，但设计建议共享 pool，因为 `hidden_dim` 相同。

**评估：** 可行。页面按 token 分页，不关心 token 来自 encoder 还是 decoder。77 tokens 在 page_size=16 时需要 5 页（最后一页用 13 个 token，浪费 3 个），内部碎片 = `3 × 5120 × 2 = 30KB`，完全可接受。

#### 风险 4：gather 阶段（Phase A）的性能退化

Phase A 用 `page_pool[page_ids].reshape(...)` 实现 gather，相比直接 `context.buffers[prefix]` 的连续读取，多了一次间接寻址。

**评估：** 这是有代价的。当前 `apply_cache` 是 `hs += cached_residual`，一次 elementwise add 即可。gather 方式变成：先 `page_pool[page_ids]` gather 到临时 contiguous tensor，再做 add。多了一次 buffer-size 的读写。但碎片消除的收益远大于此 —— 当前碎片导致 OOM 直接请求失败，而 gather 只是多几十微秒延迟。Phase B 的 Triton kernel 会消除此开销。

#### 风险 5：多 transformer 架构（如 Wan2.2 双 transformer）

**评估：** 从 `CacheDiTStateDriver._handles` 看，slot 的 payload 本身就是 per-transformer 的 tuple。共享 PagedCachePool 只需要每个 transformer 的 `PagedCacheContext` 引用同一个 pool 实例。

### 8.4 结论

**Cache-DiT 能够分页，且 omni 侧管理内存的方案是可行的。** 核心原因：

1. **计算与存储解耦：** Fn/Bn buffer 是纯"存储态"数据，所有对它的操作（write/read/add/L2）都是 per-token independent 的，不要求物理连续 —— 这是分页的技术基础。

2. **管理层已就位：** Omni 侧的 `CacheManager → CacheStateDriver → CacheBackendSlot` 已经控制了 slot 的完整生命周期，只差最底层的 tensor 存储没有统一管理 —— 用 `PagedCachePool` 替换 `context.buffers` dict 是自然的架构演进，不是重构。

3. **收益确定且无功能性风险：** 分页将碎片率从 30-60% 降至 ~0%，使最大并发数提升约 2 倍。4 阶段 forward 的 Stage 1/4（占计算量主体）完全不受影响，只有 Stage 2/3 的存取路径变化，且接口语义不变。

---

## 附录：Cache-DiT 计算流全景

### A. 完整 Denoising Step 数据流

```
Step t (denoise loop iteration):

Input:  latents [B_total, seq_len, d_hidden]   (all requests concatenated)
        encoder_hs [B_total, txt_seq_len, d_hidden]

[Stage 1] Fn blocks: FULL BATCH
  │  latents → Fn_forward(latents) → y_Fn
  │  fn_residual = y_Fn - latents
  │  (contiguous tensor, shared across all requests)
  ▼

[Stage 2] Per-request can_cache decision:
  │  for each request i:
  │    Compare fn_residual[i] vs cached_Fn_residual[i]
  │    → compute_group OR cache_group
  ▼

[Stage 3a] Mn blocks: COMPUTE GROUP only
  │  extract compute_rows from latents
  │  Mn_forward(compute_hs) → mn_out
  │  Store: Fn_residual → page pool (via Triton scatter_write)
  │  Store: Bn_residual → page pool (via Triton scatter_write)
  │  Write back to latents[compute_rows]
  │
[Stage 3b] Cache group: apply cached Bn
  │  for each cached request:
  │    latents[i] += paged_Bn_residual[i]  (via Triton residual_add)
  ▼

[Stage 4] Bn blocks: FULL BATCH
  │  latents → Bn_forward(latents) → output
  ▼

Output: updated latents [B_total, seq_len, d_hidden]
        → fed into next step's scheduler
```

### B. 显存生命周期对比

```
当前 (Dynamic Allocation):
  t=0: [Req A alloc 230MB] [Req B alloc 150MB] [Free: 620MB]
  t=5: [Req A free] [Req B keeps 150MB] [Free: 850MB, but fragmented]
  t=6: [Req C needs 400MB] → may OOM if largest_free < 400MB

Paged (Fixed Pool):
  Init: [Page Pool 2.4GB pre-allocated] [Free: 600MB for compute]
  t=0: [Req A uses 1920 pages] [Req B uses 960 pages] [Pool free: 1216 pages]
  t=5: [Req A returns 1920 pages] [Pool free: 3136 pages]
  t=6: [Req C takes 2560 pages] → always succeeds if pool has pages
  → NO fragmentation, deterministic allocation
```
