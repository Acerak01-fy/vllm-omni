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

This document defines the design contract for Scheduler-managed paged
key/value (KV) cache in a diffusion DiT stage. The first complete integration
is HunyuanImage-3.0 DiT, running in request-level execution on NVIDIA GPU and
Ascend NPU. The document describes ownership, lifecycle, metadata, and
attention execution; it is not a claim that every diffusion model or platform
supports the feature.

## Motivation

A diffusion transformer executes the same self-attention layers at every
denoising step. HunyuanImage-3.0 keeps the prompt and reference-image tokens
constant while rewriting the timestep and generated-image tokens. The legacy
dense path carries the complete K/V tensor through every step. Paged KV keeps
the stable prefix in Worker-owned pages and writes only the current span after
the first step.

Paging is a memory-management and execution feature, not a change to the
model's attention semantics. Hunyuan still has mixed causal/full attention:
text and image regions use the same rules as the dense implementation. The
difference is how the sequence is stored and how the mixed attention is
expressed to the native kernel.

## Goals and non-goals

| Goal | Contract |
| --- | --- |
| Scheduler ownership | Allocate and release logical KV blocks with the native vLLM KV cache manager. |
| Worker ownership | Allocate physical KV tensors, native BlockTables, slot mappings, and attention metadata on each rank. |
| Stable denoising reuse | Write the complete sequence once, then update only the timestep/image span. |
| Model isolation | Keep block allocation and runtime activation out of model code. The model supplies layout and attention spans. |
| Platform portability | Share the row and adapter contract while allowing CUDA and Ascend to provide different native metadata and kernels. |
| Failure safety | Reject invalid metadata and roll back partial CFG allocation instead of running a dense fallback. |

The current implementation does not provide:

- Hunyuan paged step execution or continuous batching;
- arbitrary Hunyuan request-level batching (`supports_request_batch` is false);
- Hunyuan CFG parallelism with more than its current two branches;
- imported AR-to-DiT KV, independently allocated Hunyuan KV contexts, or
  cross-request prefix publication;
- Ring attention or AllGather-KV sequence parallelism in the paged path; or
- the reserved `paged_worker_local` ownership mode.

Dense execution remains the default `dense_legacy` path and is not changed by
enabling this feature for another stage.

## Ownership model

The ownership boundary is deliberately split between the control plane and the
data plane:

| Component | Owns | Must not own |
| --- | --- | --- |
| Model preprocessing | Token/image layout, positions, CFG branch count, and `full_attn_spans` | Physical blocks or Worker runtime state |
| Scheduler | Public request lifecycle, logical `DiffusionKVRequest` objects, block admission, and release | Device tensors or native kernel metadata |
| Executor/RPC | Transport of the immutable allocation snapshot | Independent allocation decisions |
| Worker/model runner | Physical cache tensors, Worker rows, BlockTables, slot mappings, and active runtime | Scheduler block lifetime |
| Common `Attention` layer | Parallel hooks and the GPU/NPU backend boundary | Request admission or block-table construction |
| Platform backend | Native BlockTables, cache-writer contract, metadata builder, and kernel selection | Model-specific token semantics |

The same public request can own more than one logical sequence. Hunyuan uses
one sequence per CFG branch; those rows are allocated and released as one
atomic unit.

## Architecture

```mermaid
flowchart LR
    A[Request] --> B[Model preprocessing]
    B -->|HunyuanPreparedLayout + DiffusionKVRequest[]| C[Scheduler]
    C -->|DiffusionKVMetadata| D[Executor / Worker RPC]
    D --> E[Worker KV backend]
    E -->|physical pages + BlockTables / slots| F[Model runner]
    F -->|opaque ForwardContext runtime| G[Common Attention]
    G --> H[CUDA FA3 paged backend]
    G --> I[Ascend FIA paged backend]
    C -. release logical blocks .-> C
    E -. clear Worker rows .-> E
```

There are two important distinctions in this graph:

1. `DiffusionKVMetadata` is an allocation snapshot, not a cache tensor. It
   crosses the Scheduler-to-Worker boundary and contains physical block IDs
   selected by the Scheduler.
2. `DiffusionPagedAttentionRuntime` is created by the runner and exposed
   through `ForwardContext`. The model can trigger the current denoising phase
   through the normal forward-context API, but it does not create or activate
   the runtime itself.

## Startup and cache sizing

Paged KV is initialized before the Scheduler starts admitting real requests.
The startup sequence is:

1. `DiffusionEngine` resolves `diffusion_kv_mode` and prepares a maximum-shape
   profile request through the model's normal preprocessing hook.
2. Each Worker loads the model, registers marked paged attention layers, and
   reports a native `KVCacheSpec` for every cache group.
3. Each Worker profiles the prepared request to determine available KV memory.
   This forward runs before Scheduler pages exist and is explicitly marked as
   `in_diffusion_kv_memory_profile`.
4. The engine builds the native vLLM `VllmConfig`, calls
   `get_kv_cache_configs()`, creates the Scheduler cache configuration, and
   resolves the Scheduler and hash block sizes.
5. Rank-local `KVCacheConfig` objects are sent back to Workers. Workers create
   physical KV tensors and native BlockTables, then the Scheduler constructs a
   `DiffusionKVCacheManager` over the same native geometry.

The profile is a capacity probe, not a measured request. On NPU, SDPA may be
used only by this explicitly marked probe when the optional dense MindIE-SD
dependency is unavailable. A real paged request with a missing Worker adapter
fails instead of silently taking a dense path.

### Memory budget semantics

`kv_cache_memory_bytes` is an explicit per-Worker (and therefore per-rank) KV
pool budget. It sizes the already allocated physical page pool; it is not the
number of tokens in a request. When it is absent, the native sizing path uses
`gpu_memory_utilization` and subtracts profiled non-KV memory:

```text
requested_memory = device_total_memory * gpu_memory_utilization
available_kv_memory = requested_memory - profiled_non_kv_memory
```

Once a pool exists, a request consumes blocks according to its sequence
length, approximately:

```text
blocks_for_sequence = ceil(seq_len / block_size)
```

The Scheduler admits a request only when all of its CFG sequences fit. An
allocator or memory pool can reserve more memory than the live tensors use;
therefore allocator `reserved` values describe pool capacity and fragmentation,
not only active payload bytes.

The relevant configuration fields are:

| Field | Meaning |
| --- | --- |
| `diffusion_kv_mode` | `dense_legacy` (default) or `paged_scheduler`. |
| `diffusion_kv_max_rows_per_request` | Maximum Worker rows for one public request, including CFG sequences and future independent contexts. |
| `kv_cache_memory_bytes` | Optional explicit per-rank physical KV pool budget. |
| `gpu_memory_utilization` | Automatic pool-sizing fraction when no explicit byte budget is supplied. |
| `max_model_len` | Native per-sequence admission ceiling. Hunyuan can derive it from its model position limit when omitted. |
| `max_num_seqs` | Maximum number of public requests in a Scheduler wave. It is distinct from Hunyuan's internal CFG row count. |
| `max_num_batched_tokens` | Worker/native attention token capacity for one prepared batch. |

An illustrative stage-level configuration is:

```yaml
stages:
  - stage_id: 0
    diffusion_kv_mode: paged_scheduler
    diffusion_kv_max_rows_per_request: 2  # conditional + unconditional
    kv_cache_memory_bytes: 536870912      # 512 MiB per Worker
    max_num_seqs: 1
    max_model_len: 8192
    max_num_batched_tokens: 8192
    enforce_eager: true
```

This snippet shows the contract and is not a statement that every bundled
Hunyuan deploy file enables paging by default. The model and platform must also
resolve a paged-capable attention backend.

## Request admission and logical block lifecycle

### Preprocessing contract

The model preprocessor runs once, before Scheduler admission:

```text
OmniDiffusionRequest
  -> HunyuanPreparedLayout
  -> one DiffusionKVRequest per CFG branch
```

`HunyuanPreparedLayout` contains tokenized rows, image/RoPE information, and
the generated-image layout. It is reused by the Worker so token positions do
not get recomputed independently on different ranks.

For each Hunyuan branch, `DiffusionKVRequest` records:

| Field | Meaning |
| --- | --- |
| `sequence_id` | Stable branch identity within the public request. |
| `prefix_len` | End of the prompt/reference-image prefix that remains valid across denoising steps. |
| `target_len` | Timestep plus generated-image span rewritten at each step. |
| `seq_len` | Complete first-step allocation boundary. |
| `kv_contexts` | Independent contexts; Hunyuan currently supplies an empty tuple because prompt/image tokens share the primary sequence. |

The normal request payload does not carry mutable `DiffusionKVRequest` objects
to a Worker. `BaseScheduler` takes ownership of them and clears the request
field before constructing `NewRequestData`.

### Atomic admission

`DiffusionKVCacheManager` is a thin semantic facade over native vLLM
`KVCacheManager`:

1. Validate sequence IDs, lengths, model limits, and internal request IDs.
2. Compute the full-sequence reservation for every CFG branch.
3. Return `None` when the current pool is temporarily full, leaving the
   request waiting.
4. Allocate every branch with `full_sequence_must_fit=True` when capacity is
   available.
5. Roll back all allocations if any branch fails.
6. Publish one `DiffusionKVMetadata` snapshot with an allocation generation.

This prevents a conditional row from running while its unconditional partner
has no physical page. Prefix hashing and `cache_blocks` publication are
disabled in the current implementation, so pages are request-local rather
than a cross-request prefix cache.

### Release paths

The Scheduler releases logical blocks on normal completion, cancellation,
admission failure, error, `pop_request_state`, `close`, and reinitialization.
The engine separately asks Workers to clear the corresponding physical rows.
Neither side frees the other side's resources:

```text
terminal request
  -> Scheduler: DiffusionKVCacheManager.free_request()
  -> Worker: clear row contents and return Worker row slots
```

The same rule applies during partial startup failure: a created native manager
or Worker cache is closed before initialization propagates its exception.

## Scheduler-to-Worker metadata

`DiffusionKVMetadata` is the Scheduler allocation snapshot sent in
`NewRequestData` and through Executor RPC. It contains:

```text
DiffusionKVMetadata
  request_id
  allocation_generation
  sequences[]
    sequence_id
    prefix_len
    target_len
    seq_len
    block_ids per KV group
    context_ids
  contexts[] (currently unused by Hunyuan)
```

The payload carries logical lengths together with native block IDs so the
Worker can validate both views. It does not contain device pointers, K/V
tensors, or mutable Scheduler request objects.

## Worker data plane

### Physical initialization

`DiffusionKVModelRunnerBackend` registers every attention layer marked with a
`paged_kv_cache_role` and verifies that all layers in a cache group share one
native attention geometry. It then:

- obtains the platform BlockTables class;
- creates rank-local physical KV tensors through native vLLM cache builders;
- creates the platform-native BlockTables with the resolved block geometry;
- builds one `DiffusionPagedAttentionLayerAdapter` per layer; and
- creates a common `DiffusionPagedAttentionAdapter` for row resolution and
  metadata preparation.

On CUDA, the default BlockTables and metadata builders use vLLM's native
interfaces. On NPU, the platform supplies `AscendBlockTables`, Ascend block
geometry, and the Ascend metadata builder.

### Installing a snapshot

When a new request is scheduled, the Worker:

1. validates the request ID, allocation generation, group count, block IDs,
   row capacity, and token length;
2. resolves the public sequence/context identity to a free Worker row;
3. stages the block IDs in native BlockTables and applies the staged writes;
4. invalidates prepared metadata if BlockTables changed; and
5. stores the row binding for the lifetime of the request.

Repeated installation of the same generation is idempotent. A stale
generation, duplicate identity, invalid block, or mismatched group count is a
hard error. Dense execution must not install paged metadata, and paged mode
rejects legacy dense `past_key_values` payloads.

## Runner-owned request execution

The current feature uses request-level execution (`step_execution=false`). The
runner receives a complete request, installs its allocation snapshot, and
executes the normal `pipeline.forward()` call. It does not make the model aware
of Scheduler internals.

For each allocated sequence, the runner constructs two row descriptions:

| Phase | `query_len` | `seq_len` | `kv_start_pos` |
| --- | ---: | ---: | ---: |
| First denoising/prefill forward | `seq_len` | `seq_len` | `0` |
| Later denoising forward | `target_len` | `prefix_len + target_len` | `prefix_len` |

The rows keep the same ordered identity in both phases. The runner places the
metadata and a `DiffusionPagedAttentionRuntime` in `ForwardContext`. When the
Hunyuan denoising loop publishes its step index, the runtime swaps from the
prefill rows to the denoise rows once and reuses that phase for all attention
layers in the step.

The common attention layer observes the opaque Worker adapter only while the
forward is active. This keeps allocation, row binding, and activation in the
runner/Worker boundary and lets future DiT models reuse the same protocol.

## HunyuanImage-3.0 integration

### Model boundary

`request_layout.py` owns Hunyuan-specific preprocessing and derives the
Scheduler lengths from tokenizer positions:

```text
prefix_len = final generated-timestep scatter position
target_len = image tokens + timestep token + guidance token
seq_len    = final real position in the prepared row
```

`HunyuanImage3Pipeline` creates one KV sequence for each CFG branch. The
current pipeline deliberately declares `supports_request_batch = False`; a
public request is therefore normally run with `max_num_seqs=1`, while its
conditional and unconditional rows may still be processed together inside the
model's CFG batch.

`ImageKVCacheManager._forward_paged()` is intentionally a model boundary,
not a cache manager. It:

- validates the prepared query/sequence lengths and span rows;
- clears the legacy model-owned dense prompt cache;
- describes Hunyuan's Q/K/V layout and `full_attn_spans`; and
- for strict Ulysses, separates the joint prompt portion from the local image
  shard before invoking the common attention layer.

It does not allocate blocks, construct BlockTables, or activate the Worker
runtime.

### CFG parallelism

The current Hunyuan integration supports the two-branch conditional and
unconditional CFG layout. The Scheduler metadata contains one sequence per
branch, and the runner selects the branch belonging to the current CFG rank
before building rank-local rows. Positive and negative sequence identities
remain stable, so each rank writes and reads the correct physical page. This
is different from public request batching: CFG rows are internal branches of
one request, not unrelated requests merged by the Scheduler.

### Sequence parallelism

The paged path supports strict Ulysses SP in request mode. The parallel
strategy performs its normal Q/K/V exchange first. The common attention layer
removes synthetic SP padding before preparing paged metadata and restores zero
placeholders before the reverse exchange. Hunyuan supplies the prompt/image
split needed for its joint attention layout.

Ring attention and AllGather-KV SP are rejected explicitly because their
metadata and communication contracts do not match the current paged native
adapter.

## Attention execution

### Dense legacy path

The dense Hunyuan path materializes a 4-D mixed causal/full attention mask and
also carries the equivalent `full_attn_spans` metadata. Dense therefore
represents the same mixed attention pattern; it is not a different model mask.
The number and shape of native calls depend on the selected platform backend:

| Dense backend | Execution shape |
| --- | --- |
| CUDA `FLASH_ATTN` | The CUDA implementation prioritizes `full_attn_spans` and uses the shared piecewise FA helper, so one dense K/V call is made per aligned segment. No persistent page or Scheduler slot mapping is involved. |
| NPU `FLASH_ATTN` / MindIE-SD | Hunyuan passes the materialized mask to `attention_forward`; the normal Hunyuan dense path uses one masked call per layer. |
| Other dense backends | Follow their own mask/piecewise capability contract; they do not acquire Scheduler-managed pages. |

This distinction is important when comparing traces: "dense" describes the
tensor/cache representation, while piecewise execution describes how a
backend realizes Hunyuan's mixed mask.

### Paged path

The paged path passes `full_attn_spans` instead of requiring a quadratic mask.
The common piecewise planner converts each row into aligned segments:

| Segment | Native inputs | Causal flag |
| --- | --- | --- |
| Causal `[s, e)` | `Q[s:e]`, `K[:e]`, `V[:e]` | `true` |
| Full span `[a, b)` intersecting the query | `Q[overlap]`, `K[:b]`, `V[:b]` | `false` |

The segment boundaries preserve FlashAttention's bottom-right causal
alignment. The resulting outputs are restored to the original token order
before `o_proj`, residual, and MLP layers consume them.

Paged attention is consequently a sequence of native paged calls per layer. On
CUDA, Dense `FLASH_ATTN` can also be piecewise, whereas Dense NPU MindIE-SD
normally remains one masked call. In every case, the distinction is storage
and backend execution shape only; both paths implement the same Hunyuan
causal/full semantics.

### One K/V write per layer

The K/V update is deliberately outside the segment loop:

```text
current Q/K/V projection
  -> write the current K/V span to the Worker cache once
  -> run each piecewise attention segment against persistent pages
  -> restore output order
```

The CUDA/default backend keeps cache-update ownership in its native paged
attention contract. Ascend requires an explicit normal-layout cache prewrite;
its FIA segment calls read the just-written pages and pass no K/V source for
those reads. Thus a request with three segments performs one layer write and
three FIA calls, rather than three repeated writes. The obsolete logical-cache
to-PA_NZ-to-logical conversion is not part of the current design.

## Piecewise planning and fast paths

`DiffusionPagedAttentionAdapter` prepares common row metadata once per active
forward:

1. resolve logical identities to Worker row indices;
2. gather native BlockTables;
3. compute positions and slot mappings for the current write span;
4. build platform-native attention metadata; and
5. cache the piecewise plan and per-segment native metadata for all layers.

The planner has two output-layout paths:

### Homogeneous rows

When rows have the same local query length and segment ranges, the runner keeps
the row-major `[B, T, ...]` layout. Each segment is flattened only for the
native call, then reshaped and concatenated along the sequence dimension. On
NPU this avoids a large per-segment indexed `ScatterUpdate` for homogeneous CFG
rows.

### Heterogeneous rows

When local ranges differ, the planner packs valid tokens with `index_select`,
runs the native segment calls, and restores the original layout with
`index_copy_`. This fallback supports different row lengths and offsets while
preserving the contract seen by subsequent transformer layers. It is the
general path; the homogeneous optimization must not change its semantics.

The fast path concerns rows inside one attention invocation. It does not enable
arbitrary public-request batching, and it does not change Scheduler allocation.

## Platform boundary

| Concern | CUDA/GPU | Ascend/NPU |
| --- | --- | --- |
| Block tables | Native vLLM `BlockTables` | `AscendBlockTables` from vLLM-Ascend |
| Paged kernel | Native FlashAttention/FA3 contract | Ascend `FusedInferAttentionScore` (FIA) |
| Cache update | Owned by the native paged attention writer/kernel | Explicit normal-layout `do_kv_cache_update()` before segment calls |
| Attention metadata | vLLM GPU metadata builder | Ascend metadata builder with `ChunkedPrefill` state |
| SP specialization | Native strict Ulysses integration | Ascend paged backend specialization for strict Ulysses |
| Dense optional dependency | Platform-selected dense backend | MindIE-SD may serve dense attention; absence is tolerated only for the startup profile fallback |

The shared `OmniPlatform` interface owns capability hooks such as
`get_diffusion_kv_block_tables_cls()`,
`build_diffusion_kv_attn_metadata()`, and
`requires_diffusion_paged_kv_prewrite()`. Hardware-specific imports stay behind
these hooks, so the common adapter does not import vLLM-Ascend directly.

## Correctness invariants

The following invariants are required for a new model or backend integration:

1. **One identity, one row.** A `(request_id, sequence_id/context_id)` maps to
   exactly one Worker row during an active request.
2. **Allocation is complete before execution.** Every CFG branch fits before
   any branch is sent to a Worker.
3. **Write span is bounded.** `kv_start_pos + query_len <= seq_len` and the
   installed page count covers `seq_len`.
4. **Prefill and denoise identities agree.** Only lengths and the write offset
   change between phases.
5. **Attention metadata is stable within a forward.** Span layout and native
   segment metadata cannot change between layers.
6. **Dense and paged ownership are exclusive.** Dense requests do not install
   paged metadata; paged requests do not accept legacy dense KV payloads.
7. **No accidental fallback.** A missing adapter, unsupported backend, stale
   generation, or unsupported SP strategy raises an actionable error. The only
   pre-admission exception is the explicitly marked memory profile.
8. **Both owners clean up.** Scheduler logical blocks and Worker physical rows
   are released on every terminal path.

## Configuration and compatibility

The mode parser accepts `dense_legacy` and `paged_scheduler`. The
`paged_worker_local` enum value documents a migration topology but is rejected
until a separate implementation exists. `paged_scheduler` additionally
requires `diffusion_kv_max_rows_per_request`, a native cache configuration, and
a prepared memory-profile request.

For the current Hunyuan integration:

| Combination | Status |
| --- | --- |
| Request execution, no SP, TP | Supported when the selected backend has paged support. |
| Request execution, strict Ulysses SP | Supported by the current GPU and NPU integrations. |
| Request execution, two-branch CFG parallel | Supported when one allocated row exists per CFG rank. |
| Hunyuan step execution with paged KV | Rejected; use `dense_legacy` until a step-mode contract is added. |
| Hunyuan independent request batching | Not enabled (`supports_request_batch=False`). |
| Ring or AllGather-KV SP | Rejected in the paged path. |
| Imported AR KV or independent Hunyuan contexts | Not implemented. |
| Cross-request prefix cache publication | Disabled; pages are request-local. |

The dense path can continue to use its existing backend and model-owned
compatibility cache when `diffusion_kv_mode=dense_legacy`. Marking an attention
layer as paged-capable does not route unmarked dense layers through the native
KV pool.

## Validation strategy

Validation is split by ownership boundary:

| Area | Representative coverage |
| --- | --- |
| Mode and native sizing | `tests/diffusion/diffusion_kv/test_config.py`, `test_initialization.py` |
| Scheduler allocation | `test_manager.py`, `test_request.py`, `test_metadata.py`, and `test_diffusion_scheduler.py` |
| Worker rows and BlockTables | `test_block_tables.py`, `test_worker_contract.py`, and native GPU adapter tests |
| Adapter and piecewise metadata | `test_paged_attention_adapter.py`, `test_paged_attention_adapter_gpu.py`, and piecewise attention tests |
| Hunyuan layout/ownership | `tests/diffusion/models/hunyuan_image3/test_diffusion_kv_request.py` and `test_image_kv_cache_manager.py` |
| End-to-end correctness | Hunyuan E2E and pixel-accuracy tests with matched dense/paged inputs |

An end-to-end validation run should record the mode, model and Omni commit,
platform/backend, TP/SP/CFGP topology, block geometry, KV budget, warmup
policy, request count, and output validity. Paged claims require native-path
evidence (for example, CUDA paged FlashAttention or Ascend FIA/cache-writer
events) and must verify that no dense fallback was active. Profiling runs are
separate from latency samples because profiler instrumentation changes host and
device timing.

## Failure handling and observability

Errors are intentionally raised at the earliest owner that can explain them:

- preprocessing rejects a malformed Hunyuan layout or CFG count;
- the Scheduler reports an admission or capacity error;
- the Worker reports an invalid block table, row binding, or generation;
- the adapter reports a shape, dtype, span, or native metadata mismatch; and
- the platform reports an unavailable kernel or unsupported parallel strategy.

Request errors still pass through the normal diffusion output stream. Cleanup
is performed independently of output delivery, so a cancelled consumer cannot
leave Scheduler blocks or Worker rows active.

Useful debug records include the allocation generation, logical lengths,
resolved Worker row indices, native block IDs, active denoising phase, selected
backend, and whether the NPU prewrite contract was used. These records should
remain request-scoped and must not expose device pointers or prompt contents.

## Related implementation

- Scheduler facade: [`diffusion_kv/manager.py`](gh-file:vllm_omni/diffusion/diffusion_kv/manager.py)
- Request and allocation DTOs: [`diffusion_kv/request.py`](gh-file:vllm_omni/diffusion/diffusion_kv/request.py), [`diffusion_kv/metadata.py`](gh-file:vllm_omni/diffusion/diffusion_kv/metadata.py)
- Native cache initialization: [`diffusion_kv/initialization.py`](gh-file:vllm_omni/diffusion/diffusion_kv/initialization.py), [`vllm_config.py`](gh-file:vllm_omni/diffusion/vllm_config.py)
- Worker data plane: [`diffusion_kv/model_runner_backend.py`](gh-file:vllm_omni/diffusion/diffusion_kv/model_runner_backend.py)
- Generic adapter and runtime: [`diffusion_kv/paged_attention_adapter.py`](gh-file:vllm_omni/diffusion/diffusion_kv/paged_attention_adapter.py)
- Runner activation: [`diffusion_model_runner.py`](gh-file:vllm_omni/diffusion/worker/diffusion_model_runner.py), [`forward_context.py`](gh-file:vllm_omni/diffusion/forward_context.py)
- Common attention boundary: [`attention/layer.py`](gh-file:vllm_omni/diffusion/attention/layer.py), [`attention/backends/flash_attn.py`](gh-file:vllm_omni/diffusion/attention/backends/flash_attn.py)
- Piecewise planner: [`attention/backends/utils/piecewise_attn.py`](gh-file:vllm_omni/diffusion/attention/backends/utils/piecewise_attn.py)
- Hunyuan layout and model boundary: [`models/hunyuan_image3/request_layout.py`](gh-file:vllm_omni/diffusion/models/hunyuan_image3/request_layout.py), [`pipeline_hunyuan_image3.py`](gh-file:vllm_omni/diffusion/models/hunyuan_image3/pipeline_hunyuan_image3.py), [`hunyuan_image3_transformer.py`](gh-file:vllm_omni/diffusion/models/hunyuan_image3/hunyuan_image3_transformer.py)
- Platform hooks: [`platforms/interface.py`](gh-file:vllm_omni/platforms/interface.py), [`platforms/cuda/platform.py`](gh-file:vllm_omni/platforms/cuda/platform.py), [`platforms/npu/platform.py`](gh-file:vllm_omni/platforms/npu/platform.py)

## Related design work

The control-plane contracts were introduced in PR #5541 and PR #5550, native
Scheduler allocation was added in PR #6094, and the Worker data plane was added
in PR #6102. Hunyuan GPU integration and the shared piecewise execution work
began in PR #6658 and were consolidated with the Ascend implementation in PR #6563.
Future models should reuse these contracts and add only their model-specific
layout, capacity profile, and backend capability requirements.
