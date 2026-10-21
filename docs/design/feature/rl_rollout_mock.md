# HunyuanImage3 RL rollout interface (mock)

This document defines the first rollout-side contract for HunyuanImage3 AR and
diffusion RL. It is a CPU mock for validating the boundary before a real UniRL
trainer is connected. It does not claim that a real checkpoint, CUDA IPC
handle, FSDP process, GRPO implementation, or FlowGRPO implementation has run.

## Scope

AR and diffusion are two independent recipes. A round updates one mode and
publishes one complete snapshot of that mode's trainable parameters. The mock
does not combine AR and diffusion losses in one optimizer step.

```text
prompt
  -> mock AR or diffusion rollout
  -> token/logprob or image/trajectory sample
  -> mock reward + advantage + loss
  -> trainer updates its local full parameter snapshot
  -> init -> start(version, manifest) -> update(chunk) x N -> finish
  -> rollout publishes version
  -> next rollout uses the new version
```

The four weight-transfer calls keep the same lifecycle names as `AsyncOmni`:

```python
await adapter.initialize({"backend": "ipc", "packed": False})
await adapter.publish(
    WeightPublication(
        version=version,
        parameter_names=names,
        chunks=(WeightUpdateChunk(version=version, parameter_names=names, payload=update_info),),
    )
)
```

The complete CPU mock can be driven without a server:

```python
engine = MockDiffusionRolloutEngine(num_inference_steps=4)
trainer = MockHunyuanImage3Trainer(engine)
result = await trainer.run_round("a red kite", seed=7)
assert result.published_version == 1
assert result.rollout.weight_version == 0
assert engine.current_version == 1
```

`update_weights` may be called repeatedly. The mock buffers chunks and makes a
version visible to generation only after `finish_weight_update` has received
every name declared by `start_weight_update`.

The `version` and manifest are trainer-side publication bookkeeping. The
production `RLWeightTransferAdapter` keeps that bookkeeping and translates only
the chunks into the existing Omni `init/start/update/finish` calls. The payload
itself remains an upstream IPC, HTTP, or other WTE payload.

## Rollout sample contracts

An AR sample contains:

- prompt text and `prompt_token_ids`;
- `response_token_ids` and aligned `response_token_logprobs`;
- `response_start` and `response_end` offsets into the concatenated `token_ids`;
- the `weight_version` used for generation.

A diffusion sample contains:

- a placeholder `image` tensor;
- the complete `latent_trajectory`;
- one `sigma` and `timestep` per denoising step;
- `sde_step_indices` identifying the steps retained for a FlowGRPO-like loss;
- conditioning metadata and a trajectory log probability;
- the `weight_version` used for generation.

These fields are model-output data. The mock reward and loss are deliberately
simple placeholders. A real trainer owns reward models, normalization,
backward, gradient accumulation, optimizer steps, and checkpoint export.

## Parameter ownership

The mock registry uses HunyuanImage3-shaped names to exercise the same routing
decision as the real model:

| Mode | Published by the mock full-parameter recipe | Frozen by default |
| ---- | ------------------------------------------- | ----------------- |
| AR | `model.*`, `lm_head.*` | auxiliary vision parameters |
| Diffusion | `model.*`, `patch_embed.*`, `time_embed.*`, `final_layer.*` | `vae.*`, `vision_model.*` |

The frozen VAE and vision tensors remain in the rollout registry but are not in
the update manifest. This makes accidental auxiliary-component updates visible
in tests.

## Lifecycle checks

The mock rejects:

- `start` or `update` before `init`;
- a second `start` while an update is active;
- non-monotonic publication versions;
- unknown, frozen, duplicate, wrong-shape, wrong-dtype, or non-CPU tensors;
- `finish` before all manifest names arrive;
- generation observing a new version before `finish`.

The checks are intentionally stronger than a dictionary assignment. They give
the future UniRL adapter a concrete manifest and publication-version contract,
while leaving transport-specific details (RPC, HTTP, or IPC payloads) outside
the model sample schema.

## Implementation and validation

The stable contracts and adapter are in `vllm_omni.rl_rollout`. The CPU mock
implementation is in `tests/rl_rollout/mock_hunyuan_image3.py`. The focused
tests are:

```bash
pytest -q -o addopts='' tests/rl_rollout/test_hunyuan_image3_mock.py
```

The `-o addopts=''` form is only needed in an environment where the optional
pytest-xdist plugin is unavailable; normal repository test runners keep their
configured `--dist=loadgroup` option.
