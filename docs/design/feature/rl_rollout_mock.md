# RL rollout model integration

Weight updates use vLLM's native WTE lifecycle. Model-specific integration
helpers live in `vllm_omni/rl_rollout/models/`; they do not own transport clients,
publication versions, or a second weight-update protocol.

## Native weight-transfer path

```text
trainer optimizer
  -> canonical weight source
  -> native WTE sender
  -> control client (RPC or HTTP)
  -> Omni worker
  -> upstream IPCWeightTransferEngine / layerwise reload
  -> the running pipeline
```

An async Omni client exposes the existing four calls:

```python
await engine.init_weight_transfer_engine(init_info)
await engine.start_weight_update()
for update_info in trainer_encoded_chunks:
    await engine.update_weights(update_info)
await engine.finish_weight_update()
```

The payload is built by the trainer's WTE sender. A CUDA IPC handle references
live storage: the sender keeps the source alive and waits for each update to
complete before overwriting a reusable packed buffer. The control client must
propagate worker failures. Generation must be quiescent throughout the update;
partial layerwise loading cannot be rolled back by a publication wrapper.

The trainer owns parameter selection, canonical export, manifest verification,
rank consensus, and publication/sample versions. Omni validates the worker
lifecycle and delegates reception and reload to vLLM. `AsyncOmni` accepts a
finish version for protocol compatibility but does not stamp it on samples.

## Model-specific helpers

```text
vllm_omni/rl_rollout/models/
  qwen_image.py        # full-transformer recipe parameter selection
  hunyuan_image3.py    # AR / diffusion parameter scopes used by the CPU mock
```

These modules contain stateless, lazy selectors. They preserve the source
values and storage rather than collecting full tensors. They are optional
helpers for callers building a weight source, not mandatory client wrappers.
The common worker contains no model-name dispatch for these selectors.

### QwenImage

`select_transformer_parameters` selects canonical `transformer.*` names from
an iterable of named parameters. The GPU reload regression uses it to create a
transformer-only source and checks that auxiliary weights remain unchanged.
A different recipe can choose a different source without changing WTE.

UniRL's full-parameter recipe builds its training backend around the transformer
and applies `name_remap: {"*": "transformer.*"}` during export. Its native sender
uses `VLLMNativeWeightSyncClient` directly. UniRL checks source metadata and
rank agreement and advances versions only after successful completion. LoRA
is disabled; VAE and text encoder weights are not exported.

Generation still uses Omni outputs and UniRL's `Sample`, `Part`, and
`LatentSegment` conversion. Capturing conditioning and denoising trajectories
remains in the trainer's QwenImage pipeline integration.

### HunyuanImage3

`select_stage_parameters` uses canonical `model.*` backbone names and the
selected recipe's heads. The `transformer.*` alias must not be exported as a
second copy of the same backbone.

| Recipe | Parameter scope | Auxiliary components excluded |
| ------ | --------------- | ----------------------------- |
| AR | `model.*`, `lm_head.*` | VAE, vision encoder/aligner |
| Diffusion | `model.*`, image projection/output and timestep/guidance embeddings | VAE, vision encoder/aligner |

The diffusion scope includes `patch_embed`, `final_layer`, `time_embed`,
`time_embed_2`, `timestep_emb`, and the optional `guidance_emb` and
`timestep_r_emb` components. Selection alone does not make HI3 reloadable.
Real HI3 IPC, model-loader compatibility, shared-backbone stage coordination,
and cache invalidation remain unvalidated.

## Adding another model

1. Use its existing pipeline `load_weights` implementation and validate the
   upstream reload prerequisites described in [weight transfer](weight_transfer.md).
2. Keep necessary RL-specific naming or model-state helpers in
   `models/<model_name>.py`. Bind real model-specific hooks at the model or
   pipeline that uses them; a compatible model needs no extra hook class.
3. Keep parameter gathering, wire encoding, and training decisions in the
   training framework. Reuse the native WTE client lifecycle.
4. For a multi-stage model, explicitly validate target routing, shared-weight
   replicas, request quiescence, and caches. Parameter prefix selection does
   not implement those runtime guarantees.
5. Verify real parameter application and subsequent generation; a CPU mock
   establishes only the behavior it actually simulates.

## CPU mock

`tests/rl_rollout/mock_hunyuan_image3.py` models independent AR and diffusion
recipes with CPU tensors and deterministic placeholder reward/loss updates.
It uses native lifecycle signatures with a test-only mapping payload. Updates
apply immediately; generation is blocked from start until a successful finish.
The mock's fixed manifest and version checks belong to the test harness and do
not claim equivalent bookkeeping in Omni's worker. Its sample/result dataclasses
live in `tests/rl_rollout/mock_types.py` rather than the production API.

```python
engine = MockDiffusionRolloutEngine(num_inference_steps=4)
trainer = MockHunyuanImage3Trainer(engine)
result = await trainer.run_round("a red kite", seed=7)
assert result.rollout.weight_version == 0
assert result.published_version == 1
```

The mock exercises parameter scopes, repeated publication, invalid order,
incomplete snapshots, frozen parameters, and tensor shape/dtype validation.
It does not simulate two stages sharing physical storage or real KV caches.

```bash
pytest -q -o addopts='' tests/rl_rollout
```

QwenImage's real IPC regression is
`tests/e2e/features/rlhf_test/test_weight_transfer_changes_output.py`. Full
FlowGRPO/PickScore acceptance additionally runs the UniRL two-GPU full-IPC
recipe through RPC and HTTP control. Record the exact revisions and artifacts;
these integration runs are distinct from the CPU mock.
