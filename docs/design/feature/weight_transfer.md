# Weight Transfer Engine

## Table of Contents

1. [Overview](#overview)
2. [Motivation](#motivation)
3. [Architecture](#architecture)
4. [Backends](#backends)
5. [Orchestrator Semantics](#orchestrator-semantics)
6. [HTTP API](#http-api)
7. [Payload Contract](#payload-contract)
8. [Measured Results](#measured-results)
9. [Configuration](#configuration)
10. [Testing](#testing)
11. [Related Files](#related-files)
12. [Future Work](#future-work)

## Overview

Weight transfer enables dynamic model weight updates on a live serving engine, without restart and without touching GPU memory layout. It is the mechanism RL training loops need: every training iteration produces new policy weights that must be applied to the rollout engine before the next generation round.

This feature ports the upstream vLLM weight transfer architecture — as of the vLLM 0.29 line that vLLM-Omni 0.28+/main tracks — onto vLLM-Omni's multi-stage orchestration. The four-phase protocol, payload contracts, and worker engine construction match upstream exactly; the adaptation is in how the orchestrator fans the protocol out to heterogeneous stages (AR vs diffusion), and one omni-specific backend (`omni_ipc`) for diffusers-style pipelines.

**Core benefit**: an RL trainer can update rollout weights in place (measured: applied weights match the shipped tensors bit-exactly, restored weights reproduce the original output bit-exactly), replacing verl-omni's current workarounds (ZMQ-sidecar weight pushes and checkpoint-restart cycles).

**Opt-in**: disabled unless `weight_transfer_config` is set in the deploy config (or via the `--weight-transfer-config` CLI flag inherited from upstream `EngineArgs`).

## Motivation

RLHF / RL post-training runs a tight loop: *generate rollouts → compute advantages → update policy weights → repeat*. Restarting the inference engine per iteration costs a full model reload and GPU re-initialization; before this feature, vLLM-Omni had no native path at all, so verl-omni resorted to process-level hacks (a ZMQ colocate sidecar that pokes weights into the engine, and `VLLMOmniHijack` monkey-patching) — duplicated logic, fragile, and invisible to the engine's own state (no KV invalidation hooks, no version tracking).

Upstream vLLM standardized this problem away with `vllm/distributed/weight_transfer/`: a four-phase protocol, worker-side engine classes per backend (`ipc`, `nccl`, `sparse_nccl`), trainer-side engines that build the payloads, and gated HTTP routes for the trainer to drive a remote rollout engine. Aligning with it — rather than inventing an omni protocol — keeps payload compatibility with the vLLM ecosystem (verl et al.) and lets upstream improvements flow downstream unchanged.

## Architecture

### Four-Phase Protocol

The lifecycle matches upstream method-for-method:

1. **Init** — `init_weight_transfer_engine(init_info)`: the trainer ships backend handshake params (e.g. `{"packed": false}` for IPC); the worker initializes its engine's transfer channel.
2. **Start** — `start_weight_update()`: open an update session; the worker rejects a second start before finish.
3. **Update** — `update_weights(update_info)` *(repeatable)*: ship weight chunks; each chunk carries backend-specific payload (names, dtypes, shapes, IPC handles / NCCL metadata).
4. **Finish** — `finish_weight_update()`: close the session; the worker finalizes and resets per-stage state. Sessions are reusable.

### End-to-End Topology

```text
┌──────────────────────────────────────────────────────────────────────┐
│ Trainer process (verl / RL loop)                                     │
│                                                                      │
│   WeightTransferTrainerFactory.trainer_init(                         │
│       IPCTrainerInitInfo(rank=0, packed=False),                      │
│       client=HTTPVLLMWeightSyncClient("http://rollout:8000"),        │
│       source=ModuleSource(training_model))                           │
│        │                                                             │
│        │ send_weights(): init → start → update* → finish             │
│        ▼                                                             │
└────── HTTP ──────────────────────────────────────────────────────────┘
                 │  (weight-transfer router, mounted only when enabled)
                 ▼
┌──────────────────────────────────────────────────────────────────────┐
│ AsyncOmni orchestrator (entrypoints/async_omni.py)                   │
│                                                                      │
│   _weight_transfer_stage_ids() ── select stages where the feature    │
│        │                        is configured (positional ids)       │
│        ▼                                                             │
│   collective_rpc(method, args, stage_ids=[...])                      │
│        │ strict result check: any {"error": ...} ⇒ RuntimeError      │
└────────┼─────────────────────────────────────────────────────────────┘
         │  per selected stage
         ▼
┌──────────────────────────────────────────────────────────────────────┐
│ Stage workers                                                        │
│                                                                      │
│  AR stage (vllm.v1.worker.gpu_worker)      Diffusion stage           │
│   └ lifecycle inherited as-is              └ lifecycle implemented    │
│     WeightTransferEngineFactory              + state machine         │
│       .create_engine(config,                + engine created AFTER   │
│        vllm_config, device, model)             model load, upstream  │
│                                                factory signature     │
│        │                                             │               │
│        ▼                                             ▼               │
│  upstream engine (layerwise reload)     omni_ipc engine (in place)   │
│  └ ipc / nccl / sparse_nccl             └ vllm/diffusion/worker/     │
│                                           weight_transfer_engine.py   │
└──────────────────────────────────────────────────────────────────────┘
```

### Configuration Propagation

`weight_transfer_config` is a pipeline-wide deploy field with stage-level override; every hop below was newly plumbed:

```text
deploy YAML / --weight-transfer-config (inherited upstream EngineArgs flag)
  │
  ▼ DeployConfig.weight_transfer_config            (stage_config.py)
  │   pipeline-wide ⇒ every stage's engine args; stage entry overrides
  ▼ _DiffusionConfigProjection.weight_transfer_config
  │   field_validator accepts dict *or* the typed upstream config
  │   (EngineArgs.__post_init__ coerces dicts upstream of us)          (omni_config.py)
  ▼ OmniDiffusionConfig.weight_transfer_config     (diffusion/data.py)
  ▼ configure_diffusion_vllm_config() maps it to the typed
    VllmConfig.weight_transfer_config              (diffusion/vllm_config.py)
  ▼ worker creates the engine after load_model()   (diffusion_worker.py)
```

AR stages resolve the field through the upstream `EngineArgs`/`VllmConfig` intersection (inherited, no omni plumbing needed for the native backends).

### Layerwise Reload vs In-Place Application

Upstream `ipc`/`nccl` wrap every update round in the **layerwise-reload protocol**: at `start_weight_update` each parameter is moved to meta device (its kernel-format storage saved), and as chunks stream in, per-parameter `weight_loader` wiring buffers them until a layer is complete; the layer is then re-materialized, the full weight-processing pipeline (quantization repack, fusion) re-runs, and the result is copied back into the original storage. This is what allows updating *quantized* weights without changing memory layout — but it requires models to be built through vLLM's reload-aware construction (`record_metadata_for_reloading` at build time). Omni's diffusers-style pipelines are not wired into that protocol: their layers carry no recorded metadata, so the meta-ization step deletes parameters and never restores them.

`omni_ipc` takes the other valid shape upstream itself ships — **apply in place, no layerwise reload** (exactly the design point of upstream's `sparse_nccl`): received tensors are rebuilt from CUDA IPC handles and applied through the pipeline's own `load_weights` onto the live parameters. For unquantized bf16 pipelines this is semantically equivalent and cheaper. The class inherits the entire IPC payload contract (handshake, handles, packed mode); only the two lifecycle hooks differ.

## Backends

Registered in upstream's factory (inherited) plus one omni variant:

| Backend | Application | Protocol | Use case |
| ------- | ----------- | -------- | -------- |
| `ipc` | CUDA IPC handles | layerwise reload | AR stages, single node |
| `nccl` | NCCL broadcast | layerwise reload | multi-GPU trainers |
| `sparse_nccl` | sparse patches | in place | updating a parameter subset |
| `omni_ipc` | CUDA IPC handles | in place via `load_weights` | diffusion stages (diffusers-style pipelines) |

`omni_ipc` is registered through the factory's runtime extension point (`register_engine`) when the diffusion worker boots; the trainer side is unchanged — payloads are the standard IPC ones, so the upstream trainer engine and HTTP client drive it unmodified.

## Orchestrator Semantics

Two rules the multi-stage fan-out adds on top of upstream's single-engine semantics:

1. **Selective routing** — lifecycle RPCs go only to stages where weight transfer is configured. Stage types expose the resolved config in different places (AR: `VllmConfig.weight_transfer_config`; diffusion inline: `stage_client.od_config`; diffusion subprocess: `stage_client.proc_manager.od_config`); `_weight_transfer_stage_ids()` probes all three. A stage without the feature never sees the methods (its worker would reject them); calling with *no* stage configured is a hard error, not a silent no-op.

2. **Strict error propagation** — `StagePool.collective_rpc` reports per-stage failures as `{"supported": False, "error": ...}` result dicts instead of raising. The lifecycle wrapper treats any such result as a failure and raises, so a 200 from the HTTP layer means *every* enabled stage succeeded. Workers reset their per-stage session state on failure, leaving survivors idle and consistent for a full-protocol retry.

## HTTP API

Four endpoints mirroring upstream's RLHF dev routes (`vllm/entrypoints/serve/dev/rlhf/api_router.py`), mounted **only when the feature is enabled**, with a security warning in the server log (upstream gates its variant behind `VLLM_SERVER_DEV_MODE`):

```text
POST /init_weight_transfer_engine   {"init_info": <backend handshake>}
POST /start_weight_update
POST /update_weights                {"update_info": <backend payload>}
POST /finish_weight_update
```

Missing fields ⇒ 400; worker-side failures (invalid payload fields, protocol violations such as update-before-start) surface as errors, never silent success.

## Payload Contract

Payloads are backend-specific and built by the **trainer-side** engines in upstream vLLM — callers do not hand-craft them:

```python
from vllm.distributed.weight_transfer.base import ModuleSource
from vllm.distributed.weight_transfer.clients import HTTPVLLMWeightSyncClient
from vllm.distributed.weight_transfer.factory import WeightTransferTrainerFactory
from vllm.distributed.weight_transfer.ipc_engine import IPCTrainerInitInfo

client = HTTPVLLMWeightSyncClient(base_url="http://rollout:8000")
engine = WeightTransferTrainerFactory.trainer_init(
    IPCTrainerInitInfo(rank=0, packed=False),   # rank 0 is the sender
    client=client,
    source=ModuleSource(training_model),
)
engine.send_weights()   # full round trip: init → start → update* → finish
```

Wire shapes (IPC): init `{"packed": bool}`; update `names / dtype_names / shapes / ipc_handles` (CUDA IPC handle dicts keyed by GPU UUID, pickled+base64 over HTTP). The backend is chosen **server-side** via `weight_transfer_config`; it is not an init payload field.

## Measured Results

Remote L4, `tiny-random/Qwen-Image`, single stage, `omni_ipc` backend:

| Check | Result |
| ----- | ------ |
| Live parameter vs shipped tensor (max abs diff) | 0.000000 — bit-exact application |
| Perturbed VAE-decoder weight → generated image (mean abs diff) | 0.028 — output changes |
| Re-shipped pristine weight → generated image (mean abs diff) | 0.000000 — bit-exact restore |
| Full suite | 25 unit + 4 HTTP e2e + 1 data-plane e2e, all passing |

Note on the perturbation target: this tiny untrained model is insensitive to transformer-weight perturbation (a fully random attention projection moves the output by only ~0.0002 — the untrained DiT's contribution is drowned by the VAE decode), so the e2e perturbs a VAE-decoder weight, which directly produces the pixels.

## Configuration

```yaml
# deploy override — pipeline-wide
weight_transfer_config:
  backend: omni_ipc   # diffusion pipeline; use ipc/nccl for AR stages

# or stage-scoped: only stage 1 gets the feature (stage 0 never sees RPCs)
weight_transfer_config:
  backend: ipc
stages:
  - stage_id: 1
    weight_transfer_config:
      backend: omni_ipc
```

```bash
# CLI (inherited from upstream EngineArgs)
vllm-omni serve MODEL --omni ... --weight-transfer-config '{"backend": "omni_ipc"}'
```

## Testing

- **Unit** (`tests/test_weight_transfer_omni.py`, 25 tests): config propagation through the deploy merge; AR inheritance pinned (lifecycle methods must come from upstream, not be copied); diffusion worker state machine (update-before-start, double-start, failure resets, session reuse); orchestrator semantics (error results raise, routing skips unconfigured stages, no-configured-stage is a hard error).
- **Data-plane e2e** (`tests/e2e/features/rlhf_test/test_weight_transfer_changes_output.py`): in-process `AsyncOmni`; payloads built by a helper process with the same upstream primitives the trainer engine uses (`reduce_tensor` + `IPCWeightTransferUpdateInfo`) — a process must not open its own exported IPC handles, so the exporter runs in a spawned helper that stays alive until finish (IPC handle args do not keep source storages alive). Asserts bit-exact application, output change under perturbation, and bit-exact restoration.
- **HTTP control-plane e2e** (`tests/e2e/features/rlhf_test/test_weight_transfer_http_api.py`): server with `--weight-transfer-config`; trainer handshake reaches the worker; invalid payloads, missing fields and update-before-start are rejected.

## Related Files

| File | Role |
| ---- | ---- |
| `vllm_omni/config/stage_config.py` | `weight_transfer_config` deploy field; pipeline-wide propagation |
| `vllm_omni/config/omni_config.py` | diffusion projection field + validator |
| `vllm_omni/diffusion/data.py` | `OmniDiffusionConfig.weight_transfer_config` |
| `vllm_omni/diffusion/vllm_config.py` | typed mapping into `VllmConfig` |
| `vllm_omni/diffusion/worker/diffusion_worker.py` | four-phase lifecycle, state machine, engine creation post-load |
| `vllm_omni/diffusion/worker/weight_transfer_engine.py` | `omni_ipc` backend + factory registration |
| `vllm_omni/entrypoints/async_omni.py` | orchestrator lifecycle API, selective routing, strict errors |
| `vllm_omni/entrypoints/serve/weight_transfer_api.py` | HTTP router (gated) |
| `vllm_omni/entrypoints/openai/api_server.py` | gated router mounting |
| `docs/user_guide/weight_transfer.md` | user-facing guide |

## Future Work

1. **Layerwise-reload adaptation for diffusion pipelines** — required only when updating *quantized* diffusion weights (fp8 DiT et al.). Three prerequisites: call `record_metadata_for_reloading` in the omni pipeline loader after construction (before any post-load processing mutates parameter layout); compose the pipeline's `AutoWeightsLoader` path with the wrapped per-parameter `weight_loader` (including tied-embedding name alignment); guarantee chunk boundaries keep layers complete (a layer materializes only when all its parameters arrive). Can coexist with `omni_ipc` as another opt-in backend.
2. **AR-stage structured-config plumbing** — route `weight_transfer_config` through the structured AR stage projection so LLM-only pipelines opt in via that path too.
3. **Multi-node** — exercise the `nccl` backends with distributed trainers.
4. **Sparse updates** — LoRA/adapter-shaped updates via `sparse_nccl`.
5. **Async updates** — overlap weight application with inference (double buffering).
