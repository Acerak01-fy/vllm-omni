# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Contract tests for the QwenImage full-parameter rollout adapter."""

from __future__ import annotations

import pytest

from vllm_omni.rl_rollout import QwenImageParameterPolicy, QwenImageRLAdapter, WeightPublication, WeightUpdateChunk

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


class _FakeClient:
    def __init__(self) -> None:
        self.calls: list[tuple[str, object]] = []

    async def init_weight_transfer_engine(self, init_info):
        self.calls.append(("init", init_info))

    async def start_weight_update(self) -> None:
        self.calls.append(("start", None))

    async def update_weights(self, update_info) -> None:
        self.calls.append(("update", update_info))

    async def finish_weight_update(self, weight_version=None) -> None:
        self.calls.append(("finish", weight_version))


@pytest.mark.asyncio
async def test_qwen_image_streams_transformer_chunks_and_finishes_version() -> None:
    client = _FakeClient()
    adapter = QwenImageRLAdapter(client)
    parameters = {
        "transformer.blocks.0.attn.weight": "attn-0",
        "transformer.blocks.0.mlp.weight": "mlp-0",
        "transformer.blocks.1.attn.weight": "attn-1",
        "vae.decoder.conv_in.weight": "vae",
        "text_encoder.layers.0.weight": "text",
    }
    names = adapter.trainable_parameter_names(parameters)
    assert len(names) == 3 and all(name.startswith("transformer.") for name in names)
    await adapter.initialize({"packed": True})
    await adapter.start_publication(version=1, parameter_names=names)
    # The sender reuses a buffer only after update_chunk acknowledges it.
    payload = {"names": names[:2], "ipc_handles": "buffer-handle"}
    await adapter.update_chunk(WeightUpdateChunk(version=1, parameter_names=names[:2], payload=payload))
    assert client.calls[-1] == ("update", payload)
    assert adapter.current_version == 0
    payload = {"names": names[2:], "ipc_handles": "buffer-handle"}
    await adapter.update_chunk(WeightUpdateChunk(version=1, parameter_names=names[2:], payload=payload))
    await adapter.finish_publication()
    assert adapter.current_version == 1
    assert [name for name, _ in client.calls] == ["init", "start", "update", "update", "finish"]
    assert client.calls[-1] == ("finish", "1")


def test_qwen_image_policy_can_be_explicitly_extended_for_a_recipe() -> None:
    policy = QwenImageParameterPolicy(
        trainable_prefixes=("transformer.", "vae."),
        frozen_prefixes=("text_encoder.",),
    )
    parameters = {
        "transformer.block.weight": 1,
        "vae.decoder.weight": 2,
        "text_encoder.layer.weight": 3,
    }

    assert policy.select_names(parameters) == ("transformer.block.weight", "vae.decoder.weight")


@pytest.mark.asyncio
@pytest.mark.parametrize("names", [("vae.decoder.weight",), ("transformer.weight", "vae.decoder.weight")])
async def test_qwen_image_rejects_auxiliary_weights_before_start(names) -> None:
    client = _FakeClient()
    adapter = QwenImageRLAdapter(client)
    await adapter.initialize({"packed": True})
    with pytest.raises(ValueError):
        await adapter.start_publication(version=1, parameter_names=names)
    assert [name for name, _ in client.calls] == ["init"]


@pytest.mark.asyncio
async def test_qwen_image_adapter_accepts_opaque_native_payload() -> None:
    client = _FakeClient()
    adapter = QwenImageRLAdapter(client)
    await adapter.initialize({"packed": True})

    publication = WeightPublication(
        version=1,
        parameter_names=("transformer.block.weight",),
        chunks=(
            WeightUpdateChunk(
                version=1,
                parameter_names=("transformer.block.weight",),
                payload={"ipc_handles_pickled": "opaque"},
            ),
        ),
    )
    await adapter.publish(publication)

    assert client.calls[-2:] == [
        ("update", {"ipc_handles_pickled": "opaque"}),
        ("finish", "1"),
    ]
