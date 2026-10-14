# Weight Transfer for RL Training

Weight transfer enables dynamic model weight updates during inference without restarting the engine. This is essential for Reinforcement Learning from Human Feedback (RLHF) and other RL-based training workflows that require frequent weight updates.

## Overview

Traditional approaches restart the inference engine for each weight update, causing significant overhead from model reloading and GPU initialization. vLLM-Omni's weight transfer feature enables in-place weight updates through a four-phase protocol aligned with upstream vLLM, across the pipeline's AR and diffusion stages.

## Configuration

Weight transfer is **opt-in**. Enable it with the pipeline-wide `weight_transfer_config` deploy field (or the `--weight-transfer-config` CLI flag, inherited from upstream vLLM `EngineArgs`):

```yaml
# deploy override
weight_transfer_config:
  backend: ipc

# For a dense diffusion stage, keep the apply policy stage-local.
stages:
  - stage_id: 0
    weight_transfer_apply_strategy: direct-load
```

Restrict it to specific stages with a stage-level override; lifecycle RPCs are then routed only to the stages where the feature is enabled:

```yaml
weight_transfer_config:
  backend: ipc

stages:
  - stage_id: 0
    # Stage 0 is not configured; it will not receive weight-transfer RPCs
  - stage_id: 1
    weight_transfer_config:
      backend: ipc
```

### Supported Transports and Apply Strategies

The worker-side backends registered by upstream vLLM are inherited unchanged, plus one omni-specific variant:

| Transport / strategy | Description | Use Case |
| ------- | ----------- | -------- |
| `ipc` | CUDA IPC handles (zero-copy, same node), layerwise-reload protocol | Single-node vLLM-native (AR) stages |
| `nccl` | NCCL broadcast from the trainer, layerwise-reload protocol | Multi-GPU trainers |
| `sparse_nccl` | Sparse tensor transfer over NCCL, applied in place | Updating a subset of parameters |
| `ipc` + `weight_transfer_apply_strategy: direct-load` | Standard IPC payloads, weights applied in place via the pipeline's `load_weights` | Dense, unquantized diffusion stages |

The `direct-load` strategy mirrors upstream's `sparse_nccl` design point. It is
implemented only for dense, unquantized diffusion weights. Use the default
`layerwise` strategy for quantized modules.

## Usage

### Trainer-side engine (recommended)

Payloads are backend-specific and are built by the trainer-side engines in upstream vLLM (`vllm.distributed.weight_transfer`). Do not hand-craft them:

```python
from vllm.distributed.weight_transfer.base import ModuleSource
from vllm.distributed.weight_transfer.clients import HTTPVLLMWeightSyncClient
from vllm.distributed.weight_transfer.factory import WeightTransferTrainerFactory
from vllm.distributed.weight_transfer.ipc_engine import IPCTrainerInitInfo

client = HTTPVLLMWeightSyncClient(base_url="http://localhost:8000")
trainer_engine = WeightTransferTrainerFactory.trainer_init(
    IPCTrainerInitInfo(rank=0, packed=False),  # rank 0 is the sender
    client=client,
    source=ModuleSource(training_model),
)

# Each call drives the full round trip on the serving side:
# start_weight_update -> update_weights -> finish_weight_update
trainer_engine.send_weights()
```

This is the same path verl-omni-style trainers use: `trainer_init` performs the HTTP init handshake (e.g. `{"init_info": {"packed": false}}` for IPC), and `send_weights` ships one payload per parameter chunk with CUDA IPC handles.

### HTTP API (what the server exposes)

The four endpoints mirror upstream vLLM's RLHF dev routes and are **mounted only when weight transfer is enabled** (with a security warning in the server log):

```text
POST /init_weight_transfer_engine   {"init_info": <backend handshake>}
POST /start_weight_update
POST /update_weights                {"update_info": <backend payload>}
POST /finish_weight_update
```

The `init_info` / `update_info` bodies are opaque to the HTTP layer; their schema is defined by the chosen backend (see `IPCWeightTransferInitInfo` / `IPCWeightTransferUpdateInfo` in upstream vLLM for the IPC shapes). Use the trainer-side engines above to produce them.

### Python API (orchestrator level)

`AsyncOmni` exposes the same four phases for in-process use (tests, colocated rollouts):

```python
engine = AsyncOmni(
    model="path/to/model",
    weight_transfer_config={"backend": "ipc"},
    weight_transfer_apply_strategy="direct-load",
)
await engine.init_weight_transfer_engine({"packed": False})  # IPC handshake
await engine.start_weight_update()
await engine.update_weights(update_info)   # backend payload, as above
await engine.finish_weight_update()
```

## Error Semantics

- Lifecycle calls fan out to every stage with weight transfer enabled and **fail if any stage reports an error** — a failed HTTP call means the round failed, it never silently succeeds.
- Each worker enforces the protocol state machine: `update_weights` before `start_weight_update`, or a second `start_weight_update` without `finish_weight_update`, are rejected.
- On failure, workers reset their per-stage update state; retry the full four-phase round.

## Limitations

- Upstream's layerwise-reload protocol (used by `ipc`/`nccl` to rebuild quantized kernel-format storages in place) requires models built through vLLM's native reload-aware construction. Dense diffusion stages can use `ipc` with `weight_transfer_apply_strategy: direct-load`, which applies received weights through the pipeline's `load_weights`.
- Multi-node trainers should use the `nccl` backend; `ipc` requires trainer and rollout workers to share GPUs on one node.

## See Also

- [Design RFC](../design/feature/weight_transfer.md) - Architecture and design rationale
- [Pipeline Configuration](../configuration/stage_configs.md) - Deploy configuration reference
- Upstream vLLM `vllm/distributed/weight_transfer/` - engine and payload reference
