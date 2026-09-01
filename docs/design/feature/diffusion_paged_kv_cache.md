---
title: Scheduler-Managed Paged KV Cache for Diffusion DiT Stages
kind: feature
status: draft
owners:
  - "@Acerak01-fy"
  - "@zwhzzz0821"
primary_code_paths:
  - vllm_omni/diffusion/diffusion_kv/**
  - vllm_omni/diffusion/sched/**
  - vllm_omni/diffusion/worker/**
  - vllm_omni/diffusion/attention/**
related_code_paths:
  - vllm_omni/diffusion/models/hunyuan_image3/**
  - vllm_omni/diffusion/forward_context.py
  - vllm_omni/platforms/**
depends_on:
  - ../module/diffusion/diffusion_runtime.md
  - ../module/cache_management.md
validation_paths:
  - tests/diffusion/diffusion_kv/**
  - tests/diffusion/models/hunyuan_image3/**
  - tests/e2e/accuracy/test_hunyuan_image3.py
upstream_refs:
  - https://github.com/vllm-project/vllm-omni/issues/5244
  - https://github.com/vllm-project/vllm-omni/pull/5541
  - https://github.com/vllm-project/vllm-omni/pull/5550
  - https://github.com/vllm-project/vllm-omni/pull/6094
  - https://github.com/vllm-project/vllm-omni/pull/6102
  - https://github.com/vllm-project/vllm-omni/pull/6563
  - https://github.com/vllm-project/vllm-omni/pull/6658
last_reviewed: 2026-09-01
---

# Scheduler-Managed Paged KV Cache for Diffusion DiT Stages

This document defines the implementation contract for Scheduler-managed
key/value (KV) cache in a diffusion DiT stage. HunyuanImage-3.0 is the first
complete integration, using request-level execution on NVIDIA GPU and Ascend
NPU. The contract is not a general support claim for every model or platform.

## Scope and goals

Paging changes KV storage and ownership, not Hunyuan's attention semantics.
Prompt and reference-image tokens are stable across denoising steps, while
the timestep and generated-image span is rewritten. The paged path stores the
stable prefix in Worker-owned pages and updates only the changing span after
the first step.

The design guarantees:

- the Scheduler owns logical allocation and release;
- the Worker owns physical tensors, native BlockTables, slots, and metadata;
- all CFG rows are admitted atomically;
- the model supplies layout and attention spans but never allocates pages or
  activates the Worker runtime; and
- invalid or stale metadata fails explicitly instead of silently falling back
  to dense execution.

The current implementation does not cover Hunyuan paged step execution,
continuous batching, arbitrary public request batching
`supports_request_batch=False`, more than two CFG branches, imported
AR-to-DiT KV, independent Hunyuan KV contexts, cross-request prefix
publication, Ring attention, or AllGather-KV SP. The reserved
`paged_worker_local` mode is also not implemented.

## Architecture and ownership

~~~mermaid
flowchart LR
    A[Request] --> B[Model preprocessing]
    B -->|Prepared layout + KV requests| C[Scheduler]
    C -->|Allocation metadata| D[Executor / Worker RPC]
    D --> E[Worker KV backend]
    E -->|Pages + BlockTables + slots| F[Model runner]
    F -->|ForwardContext runtime| G[Common Attention]
    G --> H[CUDA paged attention]
    G --> I[Ascend FIA paged attention]
    C -. release logical blocks .-> C
    E -. clear physical rows .-> E
~~~

| Component | Owns | Does not own |
| --- | --- | --- |
| Model preprocessing | Token/image layout, positions, CFG count, `full_attn_spans` | Physical blocks or Worker state |
| Scheduler | Request lifecycle, logical sequences, admission, block release | Device tensors or native metadata |
| Executor/RPC | Transport of an immutable allocation snapshot | Allocation decisions |
| Worker/model runner | Physical KV tensors, rows, BlockTables, slots, active runtime | Scheduler block lifetime |
| Common attention layer | Parallel hooks and backend boundary | Request admission or page allocation |
| Platform backend | Native geometry, BlockTables, metadata, kernel selection | Model-specific token semantics |

The public request owns one logical sequence per Hunyuan CFG branch. These
sequences are allocated and released as one unit. `DiffusionKVMetadata`
is an allocation snapshot containing IDs, lengths, and block IDs; it contains no
device pointers, K/V tensors, or mutable Scheduler objects.

## Lifecycle

### Startup and cache sizing

Before admitting requests, the engine and Workers:

1. Resolve `diffusion_kv_mode` and prepare a maximum-shape profile request.
2. Register paged attention layers and report a native `KVCacheSpec` per group.
3. Profile the request to determine non-KV memory. This probe is marked
   `in_diffusion_kv_memory_profile` and is not a latency sample.
4. Build the native vLLM cache configuration and resolve block geometry.
5. Allocate rank-local physical pages and native BlockTables, then create the
   Scheduler's `DiffusionKVCacheManager` over the same geometry.

`kv_cache_memory_bytes` is an explicit physical KV-pool budget per Worker and
rank. It is not a request token count. Without it, sizing uses the normal
`gpu_memory_utilization` path:

~~~text
requested_memory = device_total_memory * gpu_memory_utilization
available_kv_memory = requested_memory - profiled_non_kv_memory
~~~

Once the pool exists, a sequence of length `seq_len` needs approximately
`ceil(seq_len / block_size)` blocks. Allocator `reserved` memory can exceed
live payload because it describes pool capacity and fragmentation.

Relevant configuration fields are:

| Field | Meaning |
| --- | --- |
| `diffusion_kv_mode` | `dense_legacy` (default) or `paged_scheduler` |
| `diffusion_kv_max_rows_per_request` | Maximum Worker rows, including CFG branches |
| `kv_cache_memory_bytes` | Optional explicit per-rank physical pool budget |
| `gpu_memory_utilization` | Automatic pool-sizing fraction when no byte budget is set |
| `max_model_len` | Per-sequence admission ceiling |
| `max_num_seqs` | Maximum public requests in one Scheduler wave |
| `max_num_batched_tokens` | Native attention token capacity for a prepared batch |

### Admission and release

The Scheduler facade validates IDs and lengths, computes the full reservation
for every CFG branch, and waits when the pool is temporarily full. When all
branches fit, it allocates them with `full_sequence_must_fit=True`; any
partial failure rolls back the whole request. One metadata snapshot is then
published with an allocation generation.

On the Worker, installing a snapshot validates the generation, group count,
block IDs, row capacity, and lengths; binds each logical identity to a free
row; stages BlockTables; and invalidates metadata if the tables changed.
Reinstalling the same generation is idempotent. A stale generation, duplicate
identity, invalid block, or mismatched group count is a hard error.

Scheduler logical blocks and Worker physical rows are released independently
on completion, cancellation, admission failure, errors, `close`, and
reinitialization. Dense requests never install paged metadata, and paged
requests reject legacy dense `past_key_values` payloads.

## Request execution and Hunyuan layout

The current contract is request-level execution (`step_execution=false`). The
runner installs the allocation snapshot, prepares row metadata, and executes
the normal pipeline forward. For each sequence it uses the following rows:

| Phase | `query_len` | `seq_len` | `kv_start_pos` |
| --- | ---: | ---: | ---: |
| First denoising/prefill | `seq_len` | `seq_len` | `0` |
| Later denoising steps | `target_len` | `prefix_len + target_len` | `prefix_len` |

The ordered sequence identity is unchanged between phases. The runner places
the active rows and `DiffusionPagedAttentionRuntime` in `ForwardContext`; the
denoising loop switches from prefill to denoise metadata once and reuses it
across all attention layers.

Hunyuan-specific preprocessing in `request_layout.py` derives:

~~~text
prefix_len = final generated-timestep scatter position
target_len = image tokens + timestep token + guidance token
seq_len    = final real position in the prepared row
~~~

`HunyuanImage3Pipeline` creates one KV sequence for each conditional or
unconditional CFG branch. CFG rows are internal branches of one request, not
unrelated public requests. `_forward_paged()` validates lengths and spans,
describes Q/K/V layout, clears the legacy dense prompt cache, and handles the
Hunyuan prompt/image split needed by strict Ulysses SP. It does not allocate
blocks or activate the Worker runtime.

Strict Ulysses SP is supported in request mode. The parallel strategy performs
its Q/K/V exchange, the paged adapter removes synthetic padding before native
metadata preparation, and zero placeholders are restored before the reverse
exchange. Ring and AllGather-KV SP are rejected because their metadata and
communication contracts are different.

## Attention execution

Dense and paged paths implement the same mixed causal/full attention. They
differ in storage and in how a backend realizes the mask.

### Dense path

The Hunyuan dense path materializes a 4-D mask and carries equivalent
`full_attn_spans` metadata. On NPU, the normal dense backend usually makes one
masked attention call per layer. On CUDA, the shared piecewise FA helper may
use one dense call per aligned segment. Thus "dense" describes the tensor
representation, not a universal number of kernel calls.

### Paged path

Paged execution passes `full_attn_spans` instead of a quadratic mask. Each row
is converted into aligned native segments:

| Segment | Native inputs | Causal flag |
| --- | --- | --- |
| Causal `[s, e)` | `Q[s:e]`, `K[:e]`, `V[:e]` | `true` |
| Full `[a, b)` | Query overlap, `K[:b]`, `V[:b]` | `false` |

The segment boundaries preserve bottom-right causal alignment. Native paged
calls read persistent Worker pages, and results are restored to the original
token order before `o_proj`, residual, and MLP layers consume them.

The K/V update is outside the segment loop:

~~~text
Q/K/V projection
  -> write the current K/V span once
  -> run all piecewise native attention segments
  -> restore output order
~~~

CUDA keeps the update in its native paged-attention contract. Ascend performs
an explicit normal-layout cache prewrite, then FIA segment calls read the
written pages without a K/V source. The obsolete logical-cache to PA_NZ and
back conversion is not part of this contract.

## Piecewise planning

`DiffusionPagedAttentionAdapter` resolves Worker rows, gathers BlockTables,
computes slots, builds platform metadata, and caches the segment plan for the
active forward.

| Row layout | Execution | Purpose |
| --- | --- | --- |
| Homogeneous query lengths and ranges | Keep `[B, T, ...]`, flatten each segment for the native call, then reshape/concatenate or write directly to the output buffer | Avoid the large indexed output scatter for homogeneous CFG rows |
| Heterogeneous lengths or offsets | `index_select` valid tokens, run native calls, then `index_copy_` into the original layout | General fallback preserving each row's token order |

The fast path applies only to rows inside one attention invocation. It does not
enable arbitrary public-request batching or change Scheduler allocation.

## Platform boundary

| Concern | NVIDIA GPU | Ascend NPU |
| --- | --- | --- |
| Block tables | Native vLLM `BlockTables` | `AscendBlockTables` from vLLM-Ascend |
| Paged kernel | Native FlashAttention/FA3 | Ascend `FusedInferAttentionScore` (FIA) |
| Cache update | Native paged writer/kernel | Explicit normal-layout prewrite |
| Attention metadata | vLLM GPU builder | Ascend builder with `ChunkedPrefill` state |
| SP specialization | Native strict Ulysses integration | Ascend paged strict-Ulysses path |
| Dense dependency | Platform-selected dense backend | MindIE-SD when available; SDPA only for the marked startup profile |

Hardware-specific imports and policy stay behind `OmniPlatform` hooks such as
`get_diffusion_kv_block_tables_cls()`,
`build_diffusion_kv_attn_metadata()`, and
`requires_diffusion_paged_kv_prewrite()`.

## Configuration and compatibility

`paged_scheduler` requires a native cache configuration,
`diffusion_kv_max_rows_per_request`, and a prepared memory-profile request.
The dense path remains the default `dense_legacy` path and keeps its existing
backend and compatibility cache.

| Combination | Status |
| --- | --- |
| Request execution, TP, no SP | Supported when the backend has paged support |
| Request execution, strict Ulysses SP | Supported by current GPU and NPU integrations |
| Request execution, two-branch CFG parallel | Supported with one allocated row per CFG rank |
| Hunyuan paged step execution | Rejected; use `dense_legacy` |
| Independent public request batching | Disabled (`supports_request_batch=False`) |
| Ring or AllGather-KV SP | Rejected in the paged path |
| Imported AR KV or independent Hunyuan contexts | Not implemented |
| Cross-request prefix publication | Disabled; pages are request-local |

## Invariants, errors, and validation

Implementations must preserve these invariants:

1. Each `(request_id, sequence_id/context_id)` maps to one active Worker row.
2. Every CFG branch fits before execution begins.
3. `kv_start_pos + query_len <= seq_len`, and installed pages cover `seq_len`.
4. Prefill and denoise retain identity; only lengths and write offset change.
5. Span and native metadata stay stable across layers in one forward.
6. Dense and paged ownership are exclusive, with no accidental fallback.
7. Scheduler blocks and Worker rows are both released on every terminal path.

Errors should be raised by the owner that can explain them: preprocessing for
malformed layout, Scheduler for capacity, Worker for row/table state, adapter
for shape or metadata mismatch, and platform for unavailable kernels or
parallel strategies. The only intentional pre-admission exception is the
explicitly marked memory profile.

Validation should cover mode and sizing, atomic Scheduler allocation, Worker
row/BlockTable contracts, adapter metadata and piecewise paths, Hunyuan layout,
and matched dense/paged E2E output. A run should record model and Omni commit,
platform/backend, TP/SP/CFGP topology, block geometry, KV budget, warmup
policy, request count, and output validity. Paged claims require native-path
evidence (CUDA paged FlashAttention or Ascend FIA/cache-writer events) and
must verify that no dense fallback was active.

## Implementation map

- Scheduler and requests: [`diffusion_kv/manager.py`](gh-file:vllm_omni/diffusion/diffusion_kv/manager.py), [`diffusion_kv/request.py`](gh-file:vllm_omni/diffusion/diffusion_kv/request.py), [`diffusion_kv/metadata.py`](gh-file:vllm_omni/diffusion/diffusion_kv/metadata.py)
- Initialization and Worker data plane: [`diffusion_kv/initialization.py`](gh-file:vllm_omni/diffusion/diffusion_kv/initialization.py), [`diffusion_kv/model_runner_backend.py`](gh-file:vllm_omni/diffusion/diffusion_kv/model_runner_backend.py)
- Adapter and runtime: [`diffusion_kv/paged_attention_adapter.py`](gh-file:vllm_omni/diffusion/diffusion_kv/paged_attention_adapter.py), [`worker/diffusion_model_runner.py`](gh-file:vllm_omni/diffusion/worker/diffusion_model_runner.py), [`forward_context.py`](gh-file:vllm_omni/diffusion/forward_context.py)
- Attention and piecewise planner: [`attention/layer.py`](gh-file:vllm_omni/diffusion/attention/layer.py), [`attention/backends/flash_attn.py`](gh-file:vllm_omni/diffusion/attention/backends/flash_attn.py), [`attention/backends/utils/piecewise_attn.py`](gh-file:vllm_omni/diffusion/attention/backends/utils/piecewise_attn.py)
- Hunyuan boundary: [`models/hunyuan_image3/request_layout.py`](gh-file:vllm_omni/diffusion/models/hunyuan_image3/request_layout.py), [`models/hunyuan_image3/pipeline_hunyuan_image3.py`](gh-file:vllm_omni/diffusion/models/hunyuan_image3/pipeline_hunyuan_image3.py), [`models/hunyuan_image3/hunyuan_image3_transformer.py`](gh-file:vllm_omni/diffusion/models/hunyuan_image3/hunyuan_image3_transformer.py)
- Platform hooks: [`platforms/interface.py`](gh-file:vllm_omni/platforms/interface.py), [`platforms/cuda/platform.py`](gh-file:vllm_omni/platforms/cuda/platform.py), [`platforms/npu/platform.py`](gh-file:vllm_omni/platforms/npu/platform.py)

## Related design work

The control-plane contracts were introduced in #5541 and #5550, native
Scheduler allocation was added in #6094, and the Worker data plane in #6102.
Hunyuan GPU integration and shared piecewise execution began in #6658 and
were consolidated with the Ascend implementation in #6563. Future DiT models
should reuse these contracts and add only model-specific layout, profiling,
and backend capability requirements.
