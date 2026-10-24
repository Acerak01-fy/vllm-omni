# HunyuanImage3 RL rollout interface (mock)

This document defines the rollout publication contract shared by the
HunyuanImage3 CPU mock and the QwenImage trainer integration. HunyuanImage3
remains a mock; its tests do not establish real AR or diffusion RL support.

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
itself remains an upstream WTE payload, carried by RPC or HTTP control calls.

## QwenImage streaming publication

```text
UniRL FlowGRPO + FSDP transformer
  -> canonical export names: transformer.*
  -> vLLM native packed IPC sender
  -> UniRL RLWeightSyncClient
  -> QwenImageRLAdapter -> RLWeightTransferAdapter
  -> Omni WTE control calls (RPC or HTTP)
  -> upstream IPCWeightTransferEngine -> layerwise reload
  -> the same running QwenImage pipeline
```

`QwenImageRLAdapter` validates the declared parameter names using
`QwenImageParameterPolicy`. The default recipe updates the full DiT
transformer and rejects VAE/text-encoder names before starting reload. A new
recipe can supply an explicit policy for other trainable components; its
trainer must provide the corresponding export and loss implementation.

The trainer supplies canonical names and native WTE payloads:

```python
await adapter.initialize({"packed": True})
await adapter.start_publication(version=1, parameter_names=names)
for chunk in trainer_encoded_chunks:
    await adapter.update_chunk(chunk)
await adapter.finish_publication()
```

`update_chunk` returns only after the underlying client acknowledges reception.
The native sender can then reuse its packed buffer. The adapter retains names
and version state, not tensor data or IPC handles. `publish(WeightPublication)`
uses the same lifecycle for already encoded chunks with independent storage.
There is no parameter-dictionary-to-IPC conversion in the model adapter.

Duplicate, unknown, or wrong-version chunks and incomplete streams fail the
publication. A transport or finish failure leaves `current_version` unchanged
and blocks reuse of the adapter. The trainer must quiesce generation during
updates and close/recreate a failed rollout runtime: the adapter cannot roll
back an in-progress layerwise reload. Its version is an acknowledged client
version; `AsyncOmni` does not itself stamp that version on generated samples.
UniRL retains its sample-version bookkeeping and distributed failure consensus.

The UniRL recipe configures `sync.native_client_adapter` with
`unirl.rollout.engine.vllm_omni.rl_weight_sync.build_rl_weight_sync_client` and an
`adapter_factory` resolving `vllm_omni.rl_rollout.QwenImageRLAdapter` through
`hydra.utils.get_class`. The factory creates `RLWeightSyncClient` once the
native sender supplies the manifest and current version. This bridge runs
in the synchronous training worker; it drives the async adapter in a private
loop and forwards each native payload unchanged to the existing sync client.
FSDP gathering, packed buffers, CUDA handles, and import release remain in
UniRL/vLLM. The worker's four WTE methods need no additional RL payload format.

To extend the publication boundary for another model, define its canonical
parameter policy, validate it before `start_publication`, and configure the
trainer bridge with that adapter. Separately validate the pipeline's loader
and reload metadata as described in [weight transfer](weight_transfer.md).
The model adapter alone does not make a pipeline reloadable.

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
