# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""E2E test verifying QwenImage transformer weight transfer.

Drives the *real* upstream IPC payload path: tensors are exported as CUDA IPC
handles by a helper process (a process must not open its own exported
handles, and the uniproc worker lives in this process) — mirroring
``IPCTrainerWeightTransferEngine._send_unpacked`` — shipped through the
``AsyncOmni`` four-phase lifecycle, and applied by the omni diffusion engine
(standard ``ipc`` transport and upstream layerwise reload, with weights routed
through the pipeline's ``load_weights``).

Flow:
1. Generate an image with the initial weights.
2. Ship a perturbed transformer snapshot through native IPC and verify that
   the live DiT parameter changed.
3. Restore the original transformer snapshot and verify generation remains
   valid.
"""

from __future__ import annotations

import uuid
from contextlib import ExitStack
from dataclasses import asdict
from functools import lru_cache

import numpy as np
import pytest
import torch
from torch.multiprocessing.reductions import reduce_tensor
from transformers import AutoTokenizer
from vllm.distributed.weight_transfer.ipc_engine import IPCWeightTransferUpdateInfo

from tests.helpers.mark import hardware_test
from vllm_omni.entrypoints.async_omni import AsyncOmni
from vllm_omni.inputs.data import OmniDiffusionSamplingParams
from vllm_omni.outputs import OmniRequestOutput
from vllm_omni.rl_rollout.models.qwen_image import select_transformer_parameters

MODEL = "tiny-random/Qwen-Image"
TOKENIZER_MODEL = "Qwen/Qwen2-1.5B-Instruct"
# Same handshake payload IPCTrainerWeightTransferEngine.trainer_init ships
# (ipc_engine.py): the only worker-side wire param is ``packed``.
IPC_INIT_INFO = {"packed": False}


def normalize_token_ids(tokenized_output) -> list[int]:
    """Normalize tokenizer outputs into a flat list[int]."""
    token_ids = tokenized_output
    if isinstance(tokenized_output, dict):
        if "input_ids" in tokenized_output:
            token_ids = tokenized_output["input_ids"]
    elif hasattr(tokenized_output, "input_ids"):
        token_ids = tokenized_output.input_ids

    if hasattr(token_ids, "tolist"):
        token_ids = token_ids.tolist()

    if isinstance(token_ids, tuple):
        token_ids = list(token_ids)

    if isinstance(token_ids, list) and len(token_ids) == 1 and isinstance(token_ids[0], (list, tuple)):
        token_ids = list(token_ids[0])

    if not isinstance(token_ids, list):
        raise TypeError(f"token_ids must be list-like, got {type(token_ids).__name__}")

    normalized_ids = []
    for token_id in token_ids:
        if hasattr(token_id, "item"):
            token_id = token_id.item()
        normalized_ids.append(int(token_id))
    return normalized_ids


@lru_cache(maxsize=1)
def _tokenize_prompt(text: str) -> list[int]:
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_MODEL, trust_remote_code=True)
    messages = [{"role": "user", "content": text}]
    token_ids = normalize_token_ids(tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=False))
    assert token_ids, "Tokenized prompt is empty"
    return token_ids


def _sampling_params(seed: int = 42) -> OmniDiffusionSamplingParams:
    return OmniDiffusionSamplingParams(
        num_inference_steps=2,
        guidance_scale=0.0,
        height=256,
        width=256,
        seed=seed,
    )


async def _generate_once(
    engine: AsyncOmni,
    prompt: str,
    *,
    request_id: str,
    sampling_params: OmniDiffusionSamplingParams,
) -> OmniRequestOutput:
    prompt_ids = _tokenize_prompt(prompt)
    prompt_dict = {"prompt_ids": prompt_ids}

    last_output = None
    async for output in engine.generate(
        prompt=prompt_dict,
        request_id=request_id,
        sampling_params_list=[sampling_params],
        output_modalities=["image"],
    ):
        last_output = output

    assert last_output is not None
    assert isinstance(last_output, OmniRequestOutput)
    assert last_output.images, "Expected at least one generated image"
    return last_output


def _image_to_array(output: OmniRequestOutput) -> np.ndarray:
    """Extract first image as normalized numpy array."""
    image = output.images[0]
    arr = np.asarray(image, dtype=np.float32) / 255.0
    assert arr.ndim == 3 and arr.shape[2] == 3
    return arr


def _get_diffusion_pipeline(engine: AsyncOmni) -> torch.nn.Module:
    """Reach the live diffusion pipeline of the (single-stage) model."""
    for stage_client in engine.engine.stage_clients:
        executor = getattr(getattr(stage_client, "_engine", None), "executor", None)
        worker = getattr(executor, "driver_worker", None)
        pipeline = getattr(getattr(worker, "model_runner", None), "pipeline", None)
        if pipeline is not None:
            return pipeline
    pytest.fail("Could not reach the diffusion pipeline through stage clients")


def _ipc_handle_exporter(request_q, response_q) -> None:
    """Helper-process target: export CUDA IPC handles for the given tensors.

    CUDA forbids a process from opening IPC handles it exported itself, and
    the uniproc diffusion worker lives in the test process — so the handles
    must come from a *different* process, exactly like a real trainer rank.
    Receives CPU tensor copies, materializes them on the local GPU, exports
    one handle per parameter, ships the payload back, then stays alive
    (holding the source storages) until released.
    """
    tensors = request_q.get()
    device_index = torch.accelerator.current_device_index()
    gpu_uuid = str(torch.cuda.get_device_properties(device_index).uuid)

    names: list[str] = []
    dtype_names: list[str] = []
    shapes: list[list[int]] = []
    ipc_handles: list[dict[str, tuple]] = []

    for name, cpu_tensor in tensors:
        weight = cpu_tensor.to("cuda").contiguous()
        names.append(name)
        dtype_names.append(str(weight.dtype).split(".")[-1])
        shapes.append(list(weight.shape))
        _, ipc_args = reduce_tensor(weight)
        ipc_handles.append({gpu_uuid: ipc_args})

    update_info = IPCWeightTransferUpdateInfo(
        names=names,
        dtype_names=dtype_names,
        shapes=shapes,
        ipc_handles=ipc_handles,
    )
    response_q.put(asdict(update_info))
    # Block until the caller releases us; exiting would free the IPC storages.
    request_q.get()


def _spawn_ipc_exporter(source_tensors: list[tuple[str, torch.Tensor]]):
    """Start the exporter helper process and return (payload, process)."""
    ctx = torch.multiprocessing.get_context("spawn")
    request_q = ctx.SimpleQueue()
    response_q = ctx.SimpleQueue()
    proc = ctx.Process(target=_ipc_handle_exporter, args=(request_q, response_q), daemon=True)
    proc.start()
    request_q.put([(name, tensor.detach().cpu().clone()) for name, tensor in source_tensors])
    payload = response_q.get()
    return payload, proc


async def _transfer_weights(engine: AsyncOmni, source_tensors: list[tuple[str, torch.Tensor]]) -> None:
    """Transfer a snapshot directly through the native Omni WTE lifecycle.

    The payload is built by a helper process that mirrors
    ``IPCTrainerWeightTransferEngine._send_unpacked`` (single trainer rank);
    it stays alive until ``finish_weight_update`` completes because CUDA IPC
    handle args do not keep the source storages alive.
    """
    payload, exporter = _spawn_ipc_exporter(source_tensors)
    try:
        await engine.start_weight_update()
        await engine.update_weights(payload)
        await engine.finish_weight_update()
    finally:
        del payload
        exporter.terminate()
        exporter.join(timeout=10)
        torch.cuda.ipc_collect()


@pytest.mark.core_model
@pytest.mark.diffusion
@hardware_test(res={"cuda": "L4"}, num_cards=1)
@pytest.mark.asyncio
async def test_qwen_image_transformer_weight_transfer():
    """Verify that native IPC reloads the QwenImage DiT transformer snapshot."""
    with ExitStack() as after:
        engine = AsyncOmni(
            model=MODEL,
            enforce_eager=True,
            max_num_seqs=1,
            weight_transfer_config={"backend": "ipc"},
        )
        after.callback(engine.shutdown)

        prompt = "a beautiful sunset over the ocean with vibrant orange clouds"
        sampling_params = _sampling_params(seed=42)

        # Step 1: Generate image1 with original weights
        output1 = await _generate_once(
            engine, prompt, request_id=f"original_{uuid.uuid4().hex[:8]}", sampling_params=sampling_params
        )
        image1 = _image_to_array(output1)
        assert np.isfinite(image1).all()

        pipeline = _get_diffusion_pipeline(engine)
        selected = tuple(select_transformer_parameters(pipeline.named_parameters()))
        selected_names = {name for name, _ in selected}
        frozen_tensors = {
            name: param.detach().cpu().clone()
            for name, param in pipeline.named_parameters()
            if name not in selected_names
        }

        # The QwenImage full-parameter RL boundary is the DiT transformer.
        # VAE and text encoders remain inference components and are not
        # included in the trainer publication.
        pristine_tensors: list[tuple[str, torch.Tensor]] = [(name, param.detach().clone()) for name, param in selected]
        assert pristine_tensors, "No transformer.* parameters found in the pipeline"

        # A deterministic nonzero perturbation also works for constant parameters.
        def _sensitivity(name: str) -> int:
            if "scale_shift_table" in name:
                return 0
            if "attn" in name or "norm" in name:
                return 1
            return 2

        candidates = [(name, tensor) for name, tensor in pristine_tensors if tensor.numel() > 10]
        target_name, _ = min(candidates, key=lambda item: _sensitivity(item[0]))
        print(f"[test] perturbing {target_name}")

        perturbed_tensors = [
            (
                name,
                tensor + 0.125 if name == target_name else tensor,
            )
            for name, tensor in pristine_tensors
        ]

        # Step 2: init handshake (same payload the trainer engine ships)
        await engine.init_weight_transfer_engine(dict(IPC_INIT_INFO))

        # Step 3: Ship perturbed transformer weights.
        await _transfer_weights(engine, perturbed_tensors)

        # Direct check that the shipped tensors actually landed in the live
        # parameters (separates "transfer delivered" from "output changed").
        live_params = dict(pipeline.named_parameters())
        live = live_params[target_name].detach().float().cpu()
        expected = dict(perturbed_tensors)[target_name].float().cpu()
        applied_diff = (live - expected).abs().max().item()
        pristine = dict(pristine_tensors)[target_name].float().cpu()
        changed_diff = (live - pristine).abs().max().item()
        print(f"[test] live-vs-shipped max diff: {applied_diff:.6f}; live-vs-pristine max diff: {changed_diff:.6f}")
        assert applied_diff == 0, "Live weight differs from the sent tensor"
        assert changed_diff > 0, "The perturbed weight did not reach the live model at all."

        output2 = await _generate_once(
            engine, prompt, request_id=f"perturbed_{uuid.uuid4().hex[:8]}", sampling_params=sampling_params
        )
        image2 = _image_to_array(output2)

        # Tiny random checkpoints can produce nearly identical pixels after a
        # transformer-only update. The live parameter check above is the
        # authoritative reload assertion; generation verifies the pipeline
        # remains usable after the update.

        # Step 4: Ship the pristine snapshot again and generate once more.
        restored_tensors = [(name, tensor.clone()) for name, tensor in pristine_tensors]
        await _transfer_weights(engine, restored_tensors)
        live_params = dict(pipeline.named_parameters())
        for name, tensor in restored_tensors:
            torch.testing.assert_close(live_params[name], tensor, rtol=0, atol=0)
        for name, tensor in frozen_tensors.items():
            torch.testing.assert_close(live_params[name].detach().cpu(), tensor, rtol=0, atol=0)

        output3 = await _generate_once(
            engine, prompt, request_id=f"restored_{uuid.uuid4().hex[:8]}", sampling_params=sampling_params
        )
        image3 = _image_to_array(output3)

        assert np.isfinite(image2).all() and np.isfinite(image3).all()

        print("\nQwenImage transformer weight transfer validation passed:")
        print(f"  - updated parameter: {target_name}")
        print(f"  - live-vs-pristine max diff: {changed_diff:.6f}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
